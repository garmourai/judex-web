#!/usr/bin/env python3
"""
Simple intrinsic calibration tool (chessboard).
Reads calibration config (checkerboard_path, is_fisheye) from per-folder config.json.
"""
import argparse
import json
import os
import yaml
import cv2
import numpy as np
import pickle
from datetime import datetime



def run_chessboard_calibration(images_dir, out_dir, is_fisheye=False):
    # Hardcoded repo defaults: 6x8 chessboard, 25mm square, min 3 images
    Ch_Dim = (6, 8)
    Sq_size = 25
    MIN_IMAGES = 3

    # Prepare object points
    obj_3D = np.zeros((Ch_Dim[0] * Ch_Dim[1], 3), np.float32)
    idx = 0
    for i in range(Ch_Dim[0]):
        for j in range(Ch_Dim[1]):
            obj_3D[idx][0] = i * Sq_size
            obj_3D[idx][1] = j * Sq_size
            idx += 1

    Ch_Dim_cv = (Ch_Dim[1], Ch_Dim[0])  # OpenCV expects (width, height) i.e., (cols, rows)

    images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        raise SystemExit('No images found in images dir')

    objpoints = []
    imgpoints = []
    reference_shape = None
    corner_dir = os.path.join(out_dir, 'corner_images')
    os.makedirs(corner_dir, exist_ok=True)

    for im_name in images:
        im_path = os.path.join(images_dir, im_name)
        im = cv2.imread(im_path)
        if im is None:
            continue
        if reference_shape is None:
            reference_shape = im.shape[:2][::-1]
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, Ch_Dim_cv, None)
        vis = im.copy()
        if ret:
            cv2.drawChessboardCorners(vis, Ch_Dim_cv, corners, ret)
            objpoints.append(obj_3D)
            corners2 = cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
        cv2.imwrite(os.path.join(corner_dir, im_name), vis)

    if len(objpoints) < MIN_IMAGES:
        raise SystemExit(f'Not enough chessboard detections ({len(objpoints)} found, need {MIN_IMAGES})')

    if is_fisheye:
        K = np.zeros((3,3))
        D = np.zeros((4,1))
        objp = [np.asarray(pts, dtype=np.float32).reshape(-1,1,3) for pts in objpoints]
        imgp = [np.asarray(pts, dtype=np.float32).reshape(-1,1,2) for pts in imgpoints]
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.fisheye.calibrate(objp, imgp, reference_shape, K, D, None, None, flags, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
        calib_type = 'fisheye'
    else:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, reference_shape, None, None)
        calib_type = 'pinhole'

    # Save results
    os.makedirs(out_dir, exist_ok=True)
    cam_obj = {
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.tolist(),
        'image_size': list(reference_shape),
        'reprojection_error': float(ret),
        'calibration_type': calib_type,
        'rvecs': [r.tolist() for r in rvecs],
        'tvecs': [t.tolist() for t in tvecs],
        'date': datetime.now().isoformat()
    }
    yaml_path = os.path.join(out_dir, 'camera_object.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(cam_obj, f)
    pkl_path = os.path.join(out_dir, 'camera_object.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(cam_obj, f)
    print(f"Saved camera object to {yaml_path} and {pkl_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Intrinsic calibration using chessboard images')
    parser.add_argument('--config-dir', required=True, help='Directory with config.json (will use checkerboard_path and is_fisheye from it)')
    parser.add_argument('--out', required=True, help='Output directory for calibration results')

    args = parser.parse_args()
    
    images_dir = None
    is_fisheye = False
    cfg_path = os.path.join(args.config_dir, 'config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
        if 'checkerboard_path' in cfg:
            images_dir = cfg['checkerboard_path']
            print(f"Using checkerboard_path from config: {images_dir}")
        if 'is_fisheye' in cfg:
            is_fisheye = cfg['is_fisheye']
            print(f"Using is_fisheye from config: {is_fisheye}")
    
    if not images_dir:
        raise SystemExit(f'checkerboard_path not found in {cfg_path}')
    run_chessboard_calibration(images_dir, args.out, is_fisheye)
