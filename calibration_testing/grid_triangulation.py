#!/usr/bin/env python3
"""
Export / validate 10x10 world grid intersections (z=0) for triangulation.

Export: project synthetic world points with projectPoints(K,d), map to undistorted
pixels with undistortPoints(..., P=newK), same as calibration_testing.ipynb overlay.

Validate: triangulate stored undistorted_uv with P = newK[R|t] vs world_points in JSON.

Row order for world_points and undistorted_uv: outer i = x index, inner j = y index,
k = i * n + j (i,j in 0..n-1).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

_CAL = Path(__file__).resolve().parent
if str(_CAL) not in sys.path:
    sys.path.insert(0, str(_CAL))

from triangulation_validation import (
    load_json,
    load_yaml,
    projection_matrix_from_extrinsic,
    require_distorted_extrinsic,
    triangulate_points,
    undistorted_uv_and_projection_matrix,
)


def world_bounds_from_court(world_pts: np.ndarray) -> tuple[float, float, float, float]:
    wp = np.asarray(world_pts, dtype=np.float64)
    xmin, xmax = float(wp[:, 0].min()), float(wp[:, 0].max())
    ymin, ymax = float(wp[:, 1].min()), float(wp[:, 1].max())
    return xmin, xmax, ymin, ymax


def world_grid_intersections(n: int, xmin: float, xmax: float, ymin: float, ymax: float) -> np.ndarray:
    """
    n*n points on z=0. Order: for i in 0..n-1 (x index), for j in 0..n-1 (y index):
    (xs[i], ys[j], 0). Flat index k = i * n + j.
    """
    xs = np.linspace(xmin, xmax, n, dtype=np.float64)
    ys = np.linspace(ymin, ymax, n, dtype=np.float64)
    out = np.zeros((n * n, 3), dtype=np.float64)
    k = 0
    for i in range(n):
        for j in range(n):
            out[k, 0] = xs[i]
            out[k, 1] = ys[j]
            out[k, 2] = 0.0
            k += 1
    return out


def world_points_from_source_court(court_path: Path, n: int) -> tuple[np.ndarray, dict[str, float]]:
    court = load_json(court_path)
    wp = np.asarray(court["world_points"], dtype=np.float64)
    xmin, xmax, ymin, ymax = world_bounds_from_court(wp)
    bounds = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "z": 0.0}
    world = world_grid_intersections(n, xmin, xmax, ymin, ymax)
    return world, bounds


def synthetic_distorted_uv(
    world_xyz: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    proj, _ = cv2.projectPoints(
        world_xyz.reshape(-1, 1, 3).astype(np.float64),
        rvec,
        tvec,
        K,
        d,
    )
    return proj.reshape(-1, 2)


def cmd_export(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir)
    sink_dir = Path(args.sink_dir)
    out_path = Path(args.output)
    n = int(args.grid_n)

    court_src = source_dir / "court_info.json"
    src_ext_path = source_dir / "extrinsic_pose_undistorted.json"
    snk_ext_path = sink_dir / "extrinsic_pose_undistorted.json"

    world, bounds = world_points_from_source_court(court_src, n)

    src_cam = load_yaml(source_dir / "camera_object.yaml")
    snk_cam = load_yaml(sink_dir / "camera_object.yaml")
    src_ext = load_json(src_ext_path)
    snk_ext = load_json(snk_ext_path)

    require_distorted_extrinsic(src_ext, src_ext_path)
    require_distorted_extrinsic(snk_ext, snk_ext_path)

    Ks = np.array(src_cam["camera_matrix"], dtype=np.float64)
    ds = np.array(src_cam["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    rs = np.array(src_ext["rvec"], dtype=np.float64).reshape(3, 1)
    ts = np.array(src_ext["tvec"], dtype=np.float64).reshape(3, 1)
    dist_s = synthetic_distorted_uv(world, rs, ts, Ks, ds)
    uv_src, _ = undistorted_uv_and_projection_matrix(src_cam, src_ext, dist_s)

    Kn = np.array(snk_cam["camera_matrix"], dtype=np.float64)
    dn = np.array(snk_cam["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    rn = np.array(snk_ext["rvec"], dtype=np.float64).reshape(3, 1)
    tn = np.array(snk_ext["tvec"], dtype=np.float64).reshape(3, 1)
    dist_n = synthetic_distorted_uv(world, rn, tn, Kn, dn)
    uv_snk, _ = undistorted_uv_and_projection_matrix(snk_cam, snk_ext, dist_n)

    payload = {
        "schema": "grid_triangulation_input_v1",
        "grid_n": n,
        "world_row_order": "k = i * n + j; i = x index, j = y index (linspace bounds)",
        "world_bounds": bounds,
        "world_points": world.tolist(),
        "source": {
            "undistorted_uv": uv_src.tolist(),
            "dir": str(source_dir),
        },
        "sink": {
            "undistorted_uv": uv_snk.tolist(),
            "dir": str(sink_dir),
        },
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source_dir": str(source_dir.resolve()),
            "sink_dir": str(sink_dir.resolve()),
            "court_bounds_from": str(court_src.resolve()),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path} ({n * n} points)")


def cmd_validate(args: argparse.Namespace) -> None:
    path = Path(args.input)
    source_dir = Path(args.source_dir)
    sink_dir = Path(args.sink_dir)

    data = load_json(path)
    if data.get("schema") != "grid_triangulation_input_v1":
        print(f"Warning: unexpected schema {data.get('schema')!r}")

    n = int(data["grid_n"])
    world = np.asarray(data["world_points"], dtype=np.float64)
    uv_src = np.asarray(data["source"]["undistorted_uv"], dtype=np.float64)
    uv_snk = np.asarray(data["sink"]["undistorted_uv"], dtype=np.float64)

    assert world.shape[0] == n * n
    assert uv_src.shape == (n * n, 2) and uv_snk.shape == (n * n, 2)

    src_cam = load_yaml(source_dir / "camera_object.yaml")
    snk_cam = load_yaml(sink_dir / "camera_object.yaml")
    src_ext = load_json(source_dir / "extrinsic_pose_undistorted.json")
    snk_ext = load_json(sink_dir / "extrinsic_pose_undistorted.json")

    require_distorted_extrinsic(src_ext, source_dir / "extrinsic_pose_undistorted.json")
    require_distorted_extrinsic(snk_ext, sink_dir / "extrinsic_pose_undistorted.json")

    P_src = projection_matrix_from_extrinsic(src_cam, src_ext)
    P_snk = projection_matrix_from_extrinsic(snk_cam, snk_ext)

    X_tri = triangulate_points(P_src, P_snk, uv_src, uv_snk)
    X_ref = world
    d = X_tri - X_ref
    errs = np.linalg.norm(d, axis=1)
    err_xy = np.linalg.norm(d[:, :2], axis=1)

    print("=== Grid triangulation validation ===")
    print(f"Input: {path}")
    print(f"Points: {len(world)}  (grid_n={n})")
    print("")
    print("Overall |err|_3d (m):")
    print(f"  mean  = {np.mean(errs):.6f}")
    print(f"  rmse  = {np.sqrt(np.mean(errs**2)):.6f}")
    print(f"  max   = {np.max(errs):.6f}")
    print(f"  min   = {np.min(errs):.6f}")
    print("")
    print("Planar xy only (m):")
    print(f"  mean  = {np.mean(err_xy):.6f}")
    print(f"  rmse  = {np.sqrt(np.mean(err_xy**2)):.6f}")
    print(f"  max   = {np.max(err_xy):.6f}")

    if args.csv:
        csv_path = Path(args.csv)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                ["i", "j", "k", "x_ref", "y_ref", "z_ref", "x_tri", "y_tri", "z_tri", "err3d_m", "err_xy_m"]
            )
            for k in range(n * n):
                i, j = divmod(k, n)
                w.writerow(
                    [
                        i,
                        j,
                        k,
                        X_ref[k, 0],
                        X_ref[k, 1],
                        X_ref[k, 2],
                        X_tri[k, 0],
                        X_tri[k, 1],
                        X_tri[k, 2],
                        errs[k],
                        err_xy[k],
                    ]
                )
        print(f"Wrote {csv_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Grid world triangulation export / validate")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("export", help="Build grid, project to undistorted UV, write JSON")
    pe.add_argument("--source-dir", default="calibration_testing/source")
    pe.add_argument("--sink-dir", default="calibration_testing/sink")
    pe.add_argument("--output", default="calibration_testing/grid_triangulation_input.json")
    pe.add_argument("--grid-n", type=int, default=10, dest="grid_n")
    pe.set_defaults(func=cmd_export)

    pv = sub.add_parser("validate", help="Triangulate from JSON + yaml + extrinsics")
    pv.add_argument("--input", default="calibration_testing/grid_triangulation_input.json")
    pv.add_argument("--source-dir", default="calibration_testing/source")
    pv.add_argument("--sink-dir", default="calibration_testing/sink")
    pv.add_argument("--csv", default="", help="Optional path to write per-point CSV")
    pv.set_defaults(func=cmd_validate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
