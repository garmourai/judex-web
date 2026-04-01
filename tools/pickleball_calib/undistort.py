#!/usr/bin/env python3
"""
Undistort images using intrinsic calibration (camera matrix and distortion coefficients).

Loads camera_object.yaml produced by intrinsic.py and applies undistortion to input images.
Supports both pinhole and fisheye models.
"""
import argparse
import os
import yaml
import cv2
import numpy as np


def load_camera_object(calib_yaml_path):
    """Load camera calibration from YAML file"""
    with open(calib_yaml_path, 'r') as f:
        cam_obj = yaml.safe_load(f)
    
    camera_matrix = np.array(cam_obj['camera_matrix'], dtype=np.float32)
    dist_coeffs = np.array(cam_obj['dist_coeffs'], dtype=np.float32).flatten()
    calib_type = cam_obj.get('calibration_type', 'pinhole')
    image_size = tuple(cam_obj.get('image_size', [1440, 1080]))  # (width, height)
    
    return camera_matrix, dist_coeffs, calib_type, image_size


def undistort_image(image_path, camera_matrix, dist_coeffs, calib_type='pinhole', output_path=None):
    """
    Undistort an image using the provided camera calibration.
    
    Args:
        image_path: Path to input image
        camera_matrix: Camera intrinsic matrix (3x3)
        dist_coeffs: Distortion coefficients
        calib_type: 'pinhole' or 'fisheye'
        output_path: Path to save undistorted image (optional)
        
    Returns:
        Undistorted image array
    """
    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit(f'Could not read image: {image_path}')
    
    h, w = img.shape[:2]
    
    if calib_type == 'fisheye':
        # Fisheye undistortion
        new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            camera_matrix, dist_coeffs, (w, h), np.eye(3), balance=1.0
        )
        undistorted = cv2.fisheye.undistortImage(
            img,
            camera_matrix,
            dist_coeffs,
            Knew=new_camera_matrix
        )
    else:
        # Standard (pinhole) undistortion - use balance=0 to keep all pixels
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1.0, (w, h)
        )
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, undistorted)
        print(f"Saved undistorted image to: {output_path}")
    
    return undistorted


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Undistort images using intrinsic calibration')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--calib-yaml', required=True, help='Path to camera_object.yaml from intrinsic calibration')
    parser.add_argument('--output', default=None, help='Path to save undistorted image (optional)')
    
    args = parser.parse_args()
    
    print(f"Loading calibration from: {args.calib_yaml}")
    camera_matrix, dist_coeffs, calib_type, image_size = load_camera_object(args.calib_yaml)
    print(f"Calibration type: {calib_type}")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coefficients: {dist_coeffs}")
    
    print(f"\nUndistorting image: {args.image}")
    undistorted = undistort_image(args.image, camera_matrix, dist_coeffs, calib_type, args.output)
    print(f"Undistorted image shape: {undistorted.shape}")
