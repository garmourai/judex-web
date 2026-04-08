#!/usr/bin/env python3
"""
Fit image->ground (x,y) homography from the four court corners only (p1, p3, p10, p12).
Report residuals on other ground points (especially midpoints p2, p11).

Uses undistorted pixels (same as notebook: undistortPoints + new_camera_matrix).

Interpretation:
- Corner residuals ~0 by construction.
- Nonzero residuals on p2/p11 mean those clicks are not consistent with the same
  planar homography as the four corners (click error, or non-planar court, or bad corners).
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import yaml


def undistort_points(pts: np.ndarray, K: np.ndarray, d: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    newK, _ = cv2.getOptimalNewCameraMatrix(K, d, (w, h), 1.0, (w, h))
    p = pts.reshape(-1, 1, 2).astype(np.float32)
    u = cv2.undistortPoints(p, K, d, P=newK)
    return u.reshape(-1, 2), newK


def world_from_image_H(H: np.ndarray, uv: np.ndarray) -> tuple[float, float]:
    x = np.array([uv[0], uv[1], 1.0], dtype=np.float64)
    w = H @ x
    return float(w[0] / w[2]), float(w[1] / w[2])


def img_from_world_Hinv(H: np.ndarray, xy: np.ndarray) -> tuple[float, float]:
    Hi = np.linalg.inv(H)
    X = np.array([xy[0], xy[1], 1.0], dtype=np.float64)
    v = Hi @ X
    return float(v[0] / v[2]), float(v[1] / v[2])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--court-info", required=True)
    p.add_argument("--camera-yaml", required=True)
    p.add_argument("--width", type=int, default=1440)
    p.add_argument("--height", type=int, default=1080)
    args = p.parse_args()

    with open(args.court_info, "r", encoding="utf-8") as f:
        court = json.load(f)
    with open(args.camera_yaml, "r", encoding="utf-8") as f:
        cam = yaml.safe_load(f)

    K = np.array(cam["camera_matrix"], dtype=np.float64)
    d = np.array(cam["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    img = np.array(court["image_points"], dtype=np.float64)
    world = np.array(court["world_points"], dtype=np.float64)
    names = court["point_names"]

    udist, _ = undistort_points(img, K, d, args.width, args.height)

    # Four ground corners: p1, p3, p10, p12 -> indices 0, 2, 3, 5
    idx_c = [0, 2, 3, 5]
    uv_c = udist[idx_c].astype(np.float32)
    xy_c = world[idx_c, :2].astype(np.float32)
    H, _ = cv2.findHomography(uv_c, xy_c, method=0)

    print("Homography fit: 4 corners only (undistorted pixels -> world x,y meters)\n")
    for i, ii in enumerate(idx_c):
        pred = world_from_image_H(H, udist[ii])
        true = xy_c[i]
        e = np.linalg.norm(np.array(pred) - true)
        print(f"  {names[ii]} corner residual: {e:.3e} m")

    print("\nGround points (world error vs ideal court plane through corners):")
    for i in range(6):
        pred = world_from_image_H(H, udist[i])
        true = world[i, :2]
        e = np.linalg.norm(np.array(pred) - true)
        tag = ""
        if i == 1:
            tag = "  [mid first baseline]"
        if i == 4:
            tag = "  [mid far baseline]"
        print(f"  {names[i]}: {e * 100:.2f} cm{tag}")

    print("\nGround points (undistorted image error: H^-1(world) vs click):")
    for i in range(6):
        pred = np.array(img_from_world_Hinv(H, world[i, :2]))
        obs = udist[i]
        e = np.linalg.norm(pred - obs)
        tag = ""
        if i in (1, 4):
            tag = "  [mid]"
        print(f"  {names[i]}: {e:.3f} px{tag}")


if __name__ == "__main__":
    main()
