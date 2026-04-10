#!/usr/bin/env python3
"""
Triangulation validation using source/sink court points.

Requires extrinsic_pose_undistorted.json from notebook Step 4 with
solve_space == "distorted" (solvePnP using original K + dist on raw clicks).

Uses camera_object.yaml for K, d; extrinsic JSON for rvec, tvec only.
Linearizes with undistortPoints(..., P=newK) and P = newK [R|t] (notebook Step 2).

Writes optional visual checks (unless --no-plot):
  <source|sink>/triangulation_undistorted_points_overlay.png

If extrinsic_pose_undistorted.json from the notebook includes extrinsic_holdout_indices,
prints an extra block with 3D triangulation vs court world for those holdout points only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def require_distorted_extrinsic(extr: dict, path: Path) -> None:
    if extr.get("solve_space") != "distorted":
        raise ValueError(
            f"{path}: expected 'solve_space': 'distorted' (re-run notebook Step 4). "
            f"Got {extr.get('solve_space')!r}."
        )


def _image_wh_from_yaml(cam_yaml: dict) -> tuple[int, int]:
    """Return (width, height) from camera_object.yaml image_size."""
    sz = cam_yaml.get("image_size")
    if sz is None or len(sz) < 2:
        raise ValueError("camera_object.yaml must contain image_size: [width, height]")
    return int(sz[0]), int(sz[1])


def optimal_new_camera_matrix(K: np.ndarray, d: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    """Same convention as calibration_testing.ipynb Step 2 (alpha=1.0)."""
    w, h = wh
    newK, _ = cv2.getOptimalNewCameraMatrix(K, d, (w, h), 1.0, (w, h))
    return newK


def _intrinsics_linearize_P(cam_yaml: dict, extr: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """K, d, newK, and P = newK[R|t] for pinhole + distorted extrinsic."""
    K = np.array(cam_yaml["camera_matrix"], dtype=np.float64)
    d = np.array(cam_yaml["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    wh = _image_wh_from_yaml(cam_yaml)
    newK = optimal_new_camera_matrix(K, d, wh)
    rvec = np.array(extr["rvec"], dtype=np.float64).reshape(3, 1)
    tvec = np.array(extr["tvec"], dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    P = newK @ np.hstack((R, tvec))
    return K, d, newK, P


def projection_matrix_from_extrinsic(cam_yaml: dict, extr: dict) -> np.ndarray:
    """3x4 P = newK[R|t], same convention as undistorted_uv_and_projection_matrix."""
    _, _, _, P = _intrinsics_linearize_P(cam_yaml, extr)
    return P


def undistorted_uv_and_projection_matrix(
    cam_yaml: dict, extr: dict, pts_distorted: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Distorted pixels (N,2) -> undistorted newK pixels (N,2) and 3x4 P for triangulatePoints.
    Matches notebook Step 2 newK + undistortPoints(..., P=newK).
    """
    K, d, newK, P = _intrinsics_linearize_P(cam_yaml, extr)
    pts = pts_distorted.reshape(-1, 1, 2).astype(np.float64)
    x = cv2.undistortPoints(pts, K, d, P=newK).reshape(-1, 2)
    return x, P


def linearize_pixels_and_projection(
    cam_yaml: dict, extr: dict, pts_distorted: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Undistorted pixel coords (N,2) and 3x4 P = newK[R|t] for triangulatePoints."""
    return undistorted_uv_and_projection_matrix(cam_yaml, extr, pts_distorted)


def triangulate_points(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    x1_t = x1.T.astype(np.float64)
    x2_t = x2.T.astype(np.float64)
    X_h = cv2.triangulatePoints(P1, P2, x1_t, x2_t)
    X = (X_h[:3, :] / X_h[3:, :]).T
    return X


def print_holdout_triangulation_3d(
    src_ext: dict,
    src_names: list[str],
    shared: list[str],
    X_ref: np.ndarray,
    X_tri: np.ndarray,
    errs: np.ndarray,
) -> None:
    """
    If extrinsic_pose_undistorted.json contains extrinsic_holdout_indices (from notebook Step 4),
    print 3D triangulated position vs court world for those points only.
    """
    holdout = src_ext.get("extrinsic_holdout_indices")
    if not holdout:
        return
    print("")
    print("=== Holdout points (excluded from per-camera extrinsic solve) — 3D triangulation ===")
    print(
        "Stereo triangulation uses the same image clicks; each camera pose was fit without these points."
    )
    print("")
    name_to_shared_row = {n: j for j, n in enumerate(shared)}
    for hi in holdout:
        if hi < 0 or hi >= len(src_names):
            print(f"  index {hi}: out of range; skip")
            continue
        name = src_names[hi]
        if name not in name_to_shared_row:
            print(f"  {name} (index {hi}): not shared with sink; skip 3D")
            continue
        j = name_to_shared_row[name]
        xr, yr, zr = X_ref[j]
        xt, yt, zt = X_tri[j]
        dx, dy, dz = xt - xr, yt - yr, zt - zr
        print(f"  {name} (court index {hi})")
        print(f"    expected world (m)  x={xr:10.4f}  y={yr:10.4f}  z={zr:10.4f}")
        print(f"    triangulated (m)    x={xt:10.4f}  y={yt:10.4f}  z={zt:10.4f}")
        print(f"    delta                 x={dx:+10.4f}  y={dy:+10.4f}  z={dz:+10.4f}  |err|={errs[j]:.4f} m")
        print("")


def _find_camera_image(cam_dir: Path) -> Path | None:
    """e.g. calibration_testing/source -> source.png"""
    stem = cam_dir.name.lower()
    for name in (f"{stem}.png", f"{stem}.jpg", "image.png"):
        p = cam_dir / name
        if p.is_file():
            return p
    return None


def save_undistorted_points_overlay(
    cam_dir: Path,
    cam_yaml: dict,
    pts_undistorted: np.ndarray,
    point_names: list[str],
) -> Path | None:
    """
    Draw linearized points on the same undistorted frame as the notebook:
    getOptimalNewCameraMatrix + undistort(..., newK), or reuse *_undistorted_new.png
    when present and shape matches yaml image_size.
    """
    img_path = _find_camera_image(cam_dir)
    if img_path is None:
        print(f"Warning: no {cam_dir.name}.png/.jpg in {cam_dir}; skip overlay plot.")
        return None

    K = np.array(cam_yaml["camera_matrix"], dtype=np.float64)
    d = np.array(cam_yaml["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    wh = _image_wh_from_yaml(cam_yaml)
    newK = optimal_new_camera_matrix(K, d, wh)

    stem = cam_dir.name.lower()
    precomputed = cam_dir / f"{stem}_undistorted_new.png"
    used_precomputed_bg = False
    undist: np.ndarray | None = None
    if precomputed.is_file():
        cand = cv2.imread(str(precomputed))
        if cand is not None and cand.shape[1] == wh[0] and cand.shape[0] == wh[1]:
            undist = cand
            used_precomputed_bg = True

    if undist is None:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: could not read {img_path}; skip overlay plot.")
            return None
        if img.shape[1] != wh[0] or img.shape[0] != wh[1]:
            print(
                f"Warning: {img_path} size {(img.shape[1], img.shape[0])} != "
                f"yaml image_size {wh}; skip overlay plot."
            )
            return None
        undist = cv2.undistort(img, K, d, None, newK)

    vis = undist.copy()
    n = min(len(pts_undistorted), len(point_names))
    for i in range(n):
        u, v = int(round(float(pts_undistorted[i, 0]))), int(round(float(pts_undistorted[i, 1])))
        cv2.circle(vis, (u, v), 8, (0, 255, 0), -1)
        cv2.circle(vis, (u, v), 9, (0, 0, 0), 1)
        cv2.putText(
            vis,
            str(point_names[i]),
            (u + 8, v - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
    bg_note = f"bg={precomputed.name}" if used_precomputed_bg else "bg=undistort(img, newK)"
    cv2.putText(
        vis,
        f"pts=undistortPoints(..., P=newK); {bg_note}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
    )
    out_path = cam_dir / "triangulation_undistorted_points_overlay.png"
    cv2.imwrite(str(out_path), vis)
    print(f"Saved undistorted-point overlay: {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", default="calibration_testing/source")
    p.add_argument("--sink-dir", default="calibration_testing/sink")
    p.add_argument("--no-plot", action="store_true", help="Skip writing overlay PNGs")
    args = p.parse_args()

    source_dir = Path(args.source_dir)
    sink_dir = Path(args.sink_dir)

    src_ext_path = source_dir / "extrinsic_pose_undistorted.json"
    snk_ext_path = sink_dir / "extrinsic_pose_undistorted.json"

    src_court = load_json(source_dir / "court_info.json")
    snk_court = load_json(sink_dir / "court_info.json")
    src_cam = load_yaml(source_dir / "camera_object.yaml")
    snk_cam = load_yaml(sink_dir / "camera_object.yaml")
    src_ext = load_json(src_ext_path)
    snk_ext = load_json(snk_ext_path)

    require_distorted_extrinsic(src_ext, src_ext_path)
    require_distorted_extrinsic(snk_ext, snk_ext_path)

    src_names = src_court["point_names"]
    snk_names = snk_court["point_names"]

    shared = [name for name in src_names if name in snk_names]
    if not shared:
        raise RuntimeError("No shared point names found between source and sink.")

    src_idx = {n: i for i, n in enumerate(src_names)}
    snk_idx = {n: i for i, n in enumerate(snk_names)}

    src_img_dist = np.array(src_court["image_points"], dtype=np.float64)
    snk_img_dist = np.array(snk_court["image_points"], dtype=np.float64)
    src_world = np.array(src_court["world_points"], dtype=np.float64)

    x1_all, P_src = linearize_pixels_and_projection(src_cam, src_ext, src_img_dist)
    x2_all, P_snk = linearize_pixels_and_projection(snk_cam, snk_ext, snk_img_dist)

    if not args.no_plot:
        save_undistorted_points_overlay(source_dir, src_cam, x1_all, src_names)
        save_undistorted_points_overlay(sink_dir, snk_cam, x2_all, snk_names)

    x1 = np.array([x1_all[src_idx[n]] for n in shared], dtype=np.float64)
    x2 = np.array([x2_all[snk_idx[n]] for n in shared], dtype=np.float64)
    X_ref = np.array([src_world[src_idx[n]] for n in shared], dtype=np.float64)

    X_tri = triangulate_points(P_src, P_snk, x1, x2)
    errs = np.linalg.norm(X_tri - X_ref, axis=1)

    print("=== Triangulation Validation ===")
    print(f"Source dir: {source_dir}")
    print(f"Sink dir  : {sink_dir}")
    print(f"Shared points used: {len(shared)}")
    print("")
    print("Extrinsic reprojection (from extrinsic_pose_undistorted.json, distorted px):")
    print(f"  source rmse_px: {src_ext.get('rmse_px', 'n/a')}")
    print(f"  sink   rmse_px: {snk_ext.get('rmse_px', 'n/a')}")
    print("")

    print_holdout_triangulation_3d(src_ext, src_names, shared, X_ref, X_tri, errs)

    print("Per-point: expected (world) vs triangulated (meters)")
    print("")
    for i, name in enumerate(shared):
        xr, yr, zr = X_ref[i]
        xt, yt, zt = X_tri[i]
        dx, dy, dz = xt - xr, yt - yr, zt - zr
        print(f"  {name}")
        print(f"    expected  x={xr:10.4f}  y={yr:10.4f}  z={zr:10.4f}")
        print(f"    reality   x={xt:10.4f}  y={yt:10.4f}  z={zt:10.4f}")
        print(f"    delta     x={dx:+10.4f}  y={dy:+10.4f}  z={dz:+10.4f}  |err|={errs[i]:.4f} m")
        print("")

    print("")
    print("Overall error:")
    print(f"  mean  = {np.mean(errs):.4f} m")
    print(f"  rmse  = {np.sqrt(np.mean(errs**2)):.4f} m")
    print(f"  max   = {np.max(errs):.4f} m")
    print(f"  min   = {np.min(errs):.4f} m")


if __name__ == "__main__":
    main()
