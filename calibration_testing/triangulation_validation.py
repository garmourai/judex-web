#!/usr/bin/env python3
"""
Triangulation validation using source/sink court points.

Default mode (no flags): loads ``court_info.json`` from both directories, **pairs rows by
``point_names``** (same name → same physical corner), takes **distorted** ``image_points`` from
each camera, linearizes to undistorted pixels + ``P = newK[R|t]``, triangulates, and compares
to **world_points** from the **source** file (must match sink world for those names).

Requires extrinsic_pose_undistorted.json from notebook Step 4 with
solve_space == "distorted" (solvePnP using original K + dist on raw clicks).

Uses camera_object.yaml for K, d; extrinsic JSON for rvec, tvec only.
Linearizes with undistortPoints(..., P=newK) and P = newK [R|t] (notebook Step 2).

Writes optional visual checks (unless --no-plot):
  <source|sink>/triangulation_undistorted_points_overlay.png

If extrinsic_pose_undistorted.json from the notebook includes extrinsic_holdout_indices,
prints an extra block with 3D triangulation vs court world for those holdout points only.

Use --reproject-world-points to validate stereo triangulation using world_points.txt (e.g. p1–p16):
for each 3D row, project with source and sink extrinsics to distorted pixels, linearize to
undistorted newK pixels (same as notebook), triangulate with P = newK[R|t], and compare to
the reference world coordinates (should be ~0 m error up to float noise).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
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


def default_triangulation_errors_txt_path(source_dir: Path, sink_dir: Path) -> Path:
    """``calibration_testing/triangulation_errors.txt`` when source and sink share a parent folder."""
    if source_dir.resolve().parent == sink_dir.resolve().parent:
        return source_dir.parent / "triangulation_errors.txt"
    return source_dir / "triangulation_errors.txt"


def write_triangulation_errors_txt(
    path: Path,
    *,
    mode_label: str,
    source_dir: Path,
    sink_dir: Path,
    names: list[str],
    X_ref: np.ndarray,
    X_tri: np.ndarray,
    errs: np.ndarray,
    extra_lines: list[str] | None = None,
) -> None:
    """Write per-point 3D errors and deltas to a text file."""
    lines: list[str] = [
        "# Triangulation 3D error: |X_triangulated - X_reference| (meters)",
        f"# UTC time: {datetime.now(timezone.utc).isoformat()}",
        f"# mode: {mode_label}",
        f"# source_dir: {source_dir.resolve()}",
        f"# sink_dir:   {sink_dir.resolve()}",
    ]
    if extra_lines:
        lines.extend("# " + x if not x.startswith("#") else x for x in extra_lines)
    lines.append("")
    lines.append("name\t|err|_m\tdx_m\tdy_m\tdz_m\tref_x\tref_y\tref_z\ttri_x\ttri_y\ttri_z")
    for i, name in enumerate(names):
        xr, yr, zr = float(X_ref[i, 0]), float(X_ref[i, 1]), float(X_ref[i, 2])
        xt, yt, zt = float(X_tri[i, 0]), float(X_tri[i, 1]), float(X_tri[i, 2])
        dx, dy, dz = xt - xr, yt - yr, zt - zr
        e = float(errs[i])
        lines.append(
            f"{name}\t{e:.8f}\t{dx:+.8f}\t{dy:+.8f}\t{dz:+.8f}\t"
            f"{xr:.8f}\t{yr:.8f}\t{zr:.8f}\t{xt:.8f}\t{yt:.8f}\t{zt:.8f}"
        )
    lines.append("")
    lines.append(f"mean_m\t{float(np.mean(errs)):.8f}")
    lines.append(f"rmse_m\t{float(np.sqrt(np.mean(errs**2))):.8f}")
    lines.append(f"max_m\t{float(np.max(errs)):.8f}")
    lines.append(f"min_m\t{float(np.min(errs)):.8f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved triangulation errors: {path}")


def print_per_point_error_table(names: list[str], errs: np.ndarray, *, scientific: bool = False) -> None:
    """One row per point: 3D distance |X_tri - X_ref| in meters."""
    w = max(8, max((len(n) for n in names), default=8))
    print("Per-point 3D error (|triangulated − reference|, meters):")
    print(f"{'name':<{w}}  {'|err|':>14}")
    print("-" * (w + 2 + 14))
    for name, e in zip(names, errs):
        es = f"{e:14.6e}" if scientific else f"{e:14.4f}"
        print(f"{name:<{w}}  {es}")
    print("")


def parse_world_points_txt(path: Path) -> tuple[np.ndarray, list[str]]:
    """Same format as calibration_testing.ipynb Step 4: optional name then x y z."""
    rows: list[list[float]] = []
    names: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.split("#")[0].strip()
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                continue
            try:
                float(parts[0])
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                name = f"p{len(rows) + 1}"
            except ValueError:
                if len(parts) < 4:
                    continue
                name = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            rows.append([x, y, z])
            names.append(name)
    if not rows:
        raise ValueError(f"No points parsed from {path}")
    return np.asarray(rows, dtype=np.float64), names


def distorted_reprojection_world(
    world_xyz: np.ndarray,
    cam_yaml: dict,
    extr: dict,
) -> np.ndarray:
    """3D world points -> distorted image pixels using K, d, rvec, tvec."""
    K = np.array(cam_yaml["camera_matrix"], dtype=np.float64)
    d = np.array(cam_yaml["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    rvec = np.array(extr["rvec"], dtype=np.float64).reshape(3, 1)
    tvec = np.array(extr["tvec"], dtype=np.float64).reshape(3, 1)
    proj, _ = cv2.projectPoints(
        world_xyz.reshape(-1, 1, 3).astype(np.float64),
        rvec,
        tvec,
        K,
        d,
    )
    return proj.reshape(-1, 2)


def _validate_court_json_lengths(court: dict, path: Path) -> list[str]:
    names = court["point_names"]
    n_img = len(court["image_points"])
    n_w = len(court["world_points"])
    if len(names) != n_img or len(names) != n_w:
        raise ValueError(
            f"{path}: length mismatch — point_names={len(names)}, image_points={n_img}, "
            f"world_points={n_w}"
        )
    if len(names) != len(set(names)):
        dup = [n for n in names if names.count(n) > 1]
        raise ValueError(f"{path}: duplicate point_names: {sorted(set(dup))}")
    return names


def _aligned_world_reference(
    src_court: dict,
    snk_court: dict,
    shared: list[str],
    source_path: Path,
    sink_path: Path,
    atol: float = 1e-4,
) -> np.ndarray:
    """One 3D row per shared name; source and sink world_points must agree within ``atol`` (m)."""
    src_names = src_court["point_names"]
    snk_names = snk_court["point_names"]
    src_idx = {n: i for i, n in enumerate(src_names)}
    snk_idx = {n: i for i, n in enumerate(snk_names)}
    src_w = np.array(src_court["world_points"], dtype=np.float64)
    snk_w = np.array(snk_court["world_points"], dtype=np.float64)
    rows: list[np.ndarray] = []
    for n in shared:
        a = src_w[src_idx[n]]
        b = snk_w[snk_idx[n]]
        if not np.allclose(a, b, rtol=0.0, atol=atol):
            raise ValueError(
                f"world_points disagree for '{n}' between {source_path} and {sink_path}:\n"
                f"  source: {a.tolist()}\n"
                f"  sink:   {b.tolist()}\n"
                f"Fix court_info so shared points use the same 3D coordinates."
            )
        rows.append(a)
    return np.asarray(rows, dtype=np.float64)


def run_court_triangulation_validation(
    source_dir: Path,
    sink_dir: Path,
    src_cam: dict,
    snk_cam: dict,
    src_ext: dict,
    snk_ext: dict,
    no_plot: bool,
    errors_txt: Path | None = None,
) -> None:
    """Triangulate using paired ``court_info.json`` clicks (matched by ``point_names``)."""
    src_path = source_dir / "court_info.json"
    snk_path = sink_dir / "court_info.json"
    src_court = load_json(src_path)
    snk_court = load_json(snk_path)

    src_names = _validate_court_json_lengths(src_court, src_path)
    snk_names = _validate_court_json_lengths(snk_court, snk_path)

    src_set = set(src_names)
    snk_set = set(snk_names)
    only_src = sorted(src_set - snk_set)
    only_snk = sorted(snk_set - src_set)
    shared = [n for n in src_names if n in snk_set]
    if not shared:
        raise RuntimeError(
            f"No overlapping point_names between {src_path.name} and {snk_path.name}. "
            f"Only in source: {only_src}. Only in sink: {only_snk}."
        )

    src_idx = {n: i for i, n in enumerate(src_names)}
    snk_idx = {n: i for i, n in enumerate(snk_names)}

    src_img_dist = np.array(src_court["image_points"], dtype=np.float64)
    snk_img_dist = np.array(snk_court["image_points"], dtype=np.float64)

    X_ref = _aligned_world_reference(src_court, snk_court, shared, src_path, snk_path)

    x1_all, P_src = linearize_pixels_and_projection(src_cam, src_ext, src_img_dist)
    x2_all, P_snk = linearize_pixels_and_projection(snk_cam, snk_ext, snk_img_dist)

    if not no_plot:
        save_undistorted_points_overlay(source_dir, src_cam, x1_all, src_names)
        save_undistorted_points_overlay(sink_dir, snk_cam, x2_all, snk_names)

    x1 = np.array([x1_all[src_idx[n]] for n in shared], dtype=np.float64)
    x2 = np.array([x2_all[snk_idx[n]] for n in shared], dtype=np.float64)

    X_tri = triangulate_points(P_src, P_snk, x1, x2)
    errs = np.linalg.norm(X_tri - X_ref, axis=1)

    print("=== Triangulation Validation (court_info.json, paired by point_names) ===")
    print(f"Source: {source_dir / 'court_info.json'}")
    print(f"Sink:   {sink_dir / 'court_info.json'}")
    print(f"Shared points: {len(shared)} (triangulated pairs)")
    if only_src or only_snk:
        print(f"  Skipped (not in both): only source {only_src or '[]'}; only sink {only_snk or '[]'}")
    print("")
    print("Pairing: same name → source image_points[i] + sink image_points[j] → stereo pair")
    print("")
    print(f"{'name':<8} {'src_idx':>7} {'snk_idx':>7}  world xyz (m, from source; checked vs sink)")
    for i, n in enumerate(shared):
        si, sj = src_idx[n], snk_idx[n]
        wx, wy, wz = X_ref[i]
        print(f"{n:<8} {si:7d} {sj:7d}  ({wx:.4f}, {wy:.4f}, {wz:.4f})")
    print("")
    print("Extrinsic reprojection (per camera, distorted px, from extrinsic JSON):")
    print(f"  source rmse_px: {src_ext.get('rmse_px', 'n/a')}")
    print(f"  sink   rmse_px: {snk_ext.get('rmse_px', 'n/a')}")
    print("")

    print_holdout_triangulation_3d(src_ext, src_names, shared, X_ref, X_tri, errs)

    print("Per-point: expected world (court_info) vs triangulated (meters)")
    print("")
    for i, name in enumerate(shared):
        xr, yr, zr = X_ref[i]
        xt, yt, zt = X_tri[i]
        dx, dy, dz = xt - xr, yt - yr, zt - zr
        print(f"  {name}")
        print(f"    expected  x={xr:10.4f}  y={yr:10.4f}  z={zr:10.4f}")
        print(f"    triang    x={xt:10.4f}  y={yt:10.4f}  z={zt:10.4f}")
        print(f"    delta     x={dx:+10.4f}  y={dy:+10.4f}  z={dz:+10.4f}  |err|={errs[i]:.4f} m")
        print("")

    print_per_point_error_table(shared, errs, scientific=False)

    print("Overall 3D error vs court world_points:")
    print(f"  mean  = {np.mean(errs):.4f} m")
    print(f"  rmse  = {np.sqrt(np.mean(errs**2)):.4f} m")
    print(f"  max   = {np.max(errs):.4f} m")
    print(f"  min   = {np.min(errs):.4f} m")

    extra = [
        f"court source: {src_path}",
        f"court sink:   {snk_path}",
        f"shared point count: {len(shared)}",
        f"source extrinsic rmse_px (distorted): {src_ext.get('rmse_px', 'n/a')}",
        f"sink extrinsic rmse_px (distorted): {snk_ext.get('rmse_px', 'n/a')}",
    ]
    if only_src or only_snk:
        extra.append(f"only in source (skipped for stereo): {only_src}")
        extra.append(f"only in sink (skipped for stereo): {only_snk}")
    write_triangulation_errors_txt(
        errors_txt or default_triangulation_errors_txt_path(source_dir, sink_dir),
        mode_label="court_info.json (paired by point_names)",
        source_dir=source_dir,
        sink_dir=sink_dir,
        names=shared,
        X_ref=X_ref,
        X_tri=X_tri,
        errs=errs,
        extra_lines=extra,
    )


def run_reproject_world_points_triangulation(
    source_dir: Path,
    sink_dir: Path,
    src_cam: dict,
    snk_cam: dict,
    src_ext: dict,
    snk_ext: dict,
    no_plot: bool,
    errors_txt: Path | None = None,
) -> None:
    wp_path = source_dir / "world_points.txt"
    if not wp_path.is_file():
        raise FileNotFoundError(f"Missing {wp_path} (create it or copy from sink)")

    X_ref, names = parse_world_points_txt(wp_path)
    snk_wp = sink_dir / "world_points.txt"
    if snk_wp.is_file() and snk_wp.read_text(encoding="utf-8") != wp_path.read_text(encoding="utf-8"):
        print(f"Warning: {sink_dir}/world_points.txt differs from source; using source file for 3D reference.")

    uv_src_d = distorted_reprojection_world(X_ref, src_cam, src_ext)
    uv_snk_d = distorted_reprojection_world(X_ref, snk_cam, snk_ext)

    x1, P_src = undistorted_uv_and_projection_matrix(src_cam, src_ext, uv_src_d)
    x2, P_snk = undistorted_uv_and_projection_matrix(snk_cam, snk_ext, uv_snk_d)

    X_tri = triangulate_points(P_src, P_snk, x1, x2)
    errs = np.linalg.norm(X_tri - X_ref, axis=1)

    print("")
    print("=== Triangulation: reprojected world_points.txt (source & sink) ===")
    print(f"World file: {wp_path}")
    print(f"Points: {len(names)}  ({', '.join(names[:5])}{'...' if len(names) > 5 else ''})")
    print(
        "Pipeline: world 3D -> projectPoints(K,d) per camera -> undistortPoints(..., P=newK) -> "
        "triangulatePoints(P_src, P_snk) vs reference world from file."
    )
    print("")

    if not no_plot:
        save_undistorted_points_overlay(source_dir, src_cam, x1, names)
        save_undistorted_points_overlay(sink_dir, snk_cam, x2, names)

    print("Per-point: reference world vs triangulated (meters)")
    print("")
    for i, name in enumerate(names):
        xr, yr, zr = X_ref[i]
        xt, yt, zt = X_tri[i]
        dx, dy, dz = xt - xr, yt - yr, zt - zr
        print(f"  {name}")
        print(f"    reference x={xr:10.4f}  y={yr:10.4f}  z={zr:10.4f}")
        print(f"    triang    x={xt:10.4f}  y={yt:10.4f}  z={zt:10.4f}")
        print(f"    delta     x={dx:+10.4f}  y={dy:+10.4f}  z={dz:+10.4f}  |err|={errs[i]:.6e} m")
        print("")

    print_per_point_error_table(names, errs, scientific=True)

    print("Overall 3D error (should be ~0 for synthetic reprojection):")
    print(f"  mean  = {np.mean(errs):.6e} m")
    print(f"  rmse  = {np.sqrt(np.mean(errs**2)):.6e} m")
    print(f"  max   = {np.max(errs):.6e} m")
    print("")

    write_triangulation_errors_txt(
        errors_txt or default_triangulation_errors_txt_path(source_dir, sink_dir),
        mode_label="world_points.txt (synthetic reprojection pipeline check)",
        source_dir=source_dir,
        sink_dir=sink_dir,
        names=names,
        X_ref=X_ref,
        X_tri=X_tri,
        errs=errs,
        extra_lines=[f"world_points reference: {wp_path}"],
    )


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
    p.add_argument(
        "--reproject-world-points",
        action="store_true",
        help=(
            "Use world_points.txt in source dir (e.g. p1–p16): for each 3D point, reproject in "
            "both cameras, linearize to undistorted pixels, triangulate, compare to file (pipeline check)."
        ),
    )
    p.add_argument(
        "--errors-txt",
        type=Path,
        default=None,
        help="Write triangulation errors TSV (default: <parent of source>/triangulation_errors.txt)",
    )
    args = p.parse_args()

    source_dir = Path(args.source_dir)
    sink_dir = Path(args.sink_dir)

    src_ext_path = source_dir / "extrinsic_pose_undistorted.json"
    snk_ext_path = sink_dir / "extrinsic_pose_undistorted.json"

    src_cam = load_yaml(source_dir / "camera_object.yaml")
    snk_cam = load_yaml(sink_dir / "camera_object.yaml")
    src_ext = load_json(src_ext_path)
    snk_ext = load_json(snk_ext_path)

    require_distorted_extrinsic(src_ext, src_ext_path)
    require_distorted_extrinsic(snk_ext, snk_ext_path)

    if args.reproject_world_points:
        run_reproject_world_points_triangulation(
            source_dir,
            sink_dir,
            src_cam,
            snk_cam,
            src_ext,
            snk_ext,
            args.no_plot,
            errors_txt=args.errors_txt,
        )
        return

    run_court_triangulation_validation(
        source_dir,
        sink_dir,
        src_cam,
        snk_cam,
        src_ext,
        snk_ext,
        args.no_plot,
        errors_txt=args.errors_txt,
    )


if __name__ == "__main__":
    main()
