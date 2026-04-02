#!/usr/bin/env python3
"""
Generate interactive Plotly 3D trajectory HTML(s) from merged trajectory_filter_stats.

Chunks by sync-frame ID (default: at most 1000 frames per HTML), matching
run_shot_segmentation_and_visualization.py overlay chunking.

Depends on old_Badminton_pipeline_3d-realtime (DetectionTree + Plotly HTML).

Python deps (install if missing): ``pip install plotly scipy matplotlib numpy``;
use SciPy/Matplotlib builds compatible with your NumPy (e.g. SciPy 1.11+ for NumPy 2).

Usage:
    python cv_code/run_3d_trajectory_html.py \\
      --trajectory-output /mnt/data/cv_output/trajectory_output \\
      --output-dir /mnt/data/cv_output/trajectory_3d_html
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "cv_code"))

from shot_segmentation_and_landing.filter_stats_merger import merge_filter_stats

_OLD_BADMINTON = _REPO_ROOT / "old_Badminton_pipeline_3d-realtime"
sys.path.insert(0, str(_OLD_BADMINTON))

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

from shot_detection_and_trajectory_analysis.trajectory_tree import DetectionTree
from shot_detection_and_trajectory_analysis.visualisation import (
    create_3d_trajectory_visualization,
    save_trajectories,
)

DEFAULT_TRAJECTORY_OUTPUT = "/mnt/data/cv_output/trajectory_output"
DEFAULT_OUTPUT_DIR = "/mnt/data/cv_output/trajectory_3d_html"
DEFAULT_WORLD_POINTS = str(_REPO_ROOT / "tools" / "pickleball_calib" / "worldpickleball.txt")


def _chunk_ranges(start_frame: int, end_frame: int, block_size: int) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    cur = start_frame
    while cur <= end_frame:
        nxt = min(cur + block_size - 1, end_frame)
        ranges.append((cur, nxt))
        cur = nxt + 1
    return ranges


def _sync_frame_bounds(
    trajectory_output_dir: str,
    start_frame: Optional[int],
    end_frame: Optional[int],
) -> Optional[Tuple[int, int]]:
    merged_rows, _ = merge_filter_stats(trajectory_output_dir, verbose=False)
    if not merged_rows:
        return None
    frames = [int(r["frame_number_after_sync"]) for r in merged_rows]
    lo_all, hi_all = min(frames), max(frames)
    lo = start_frame if start_frame is not None else lo_all
    hi = end_frame if end_frame is not None else hi_all
    lo = max(lo, lo_all)
    hi = min(hi, hi_all)
    if lo > hi:
        return None
    return lo, hi


def _prepare_plotly_world_file(world_points_path: Optional[str]) -> Optional[str]:
    if not world_points_path or not os.path.exists(world_points_path):
        return None
    try:
        raw = np.loadtxt(world_points_path, comments="#")
    except Exception:
        return None
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < 3:
        return None
    xyz = raw[:, -3:].astype(float)
    fd, path = tempfile.mkstemp(suffix="_court_xyz.txt", text=True)
    os.close(fd)
    np.savetxt(path, xyz, fmt="%.9f")
    return path


def load_world_xyz_for_bounds(world_points_path: Optional[str]) -> Optional[np.ndarray]:
    if not world_points_path or not os.path.exists(world_points_path):
        return None
    try:
        raw = np.loadtxt(world_points_path, comments="#")
    except Exception:
        return None
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < 3:
        return None
    return raw[:, -3:].astype(float)


def point_in_bounds(point: np.ndarray, min_bound: np.ndarray, max_bound: np.ndarray) -> bool:
    return bool(np.all(point >= min_bound) and np.all(point <= max_bound))


def _rows_to_frame_dict(merged_rows: List[Dict[str, Any]]) -> Dict[int, List[Tuple[float, float, float]]]:
    out: Dict[int, List[Tuple[float, float, float]]] = {}
    for row in merged_rows:
        f = int(row["frame_number_after_sync"])
        pt = (float(row["picked_x"]), float(row["picked_y"]), float(row["picked_z"]))
        if pt == (0.0, 0.0, 0.0):
            continue
        out.setdefault(f, []).append(pt)
    return out


def build_trajectories_from_merged_rows(
    merged_rows: List[Dict[str, Any]],
    world_xyz: Optional[np.ndarray],
) -> Tuple[List[Any], Dict[int, List[Tuple[int, Tuple[float, float, float]]]], int, int]:
    data_dict = _rows_to_frame_dict(merged_rows)
    if not data_dict:
        raise ValueError("No valid 3D points in merged rows for this chunk")

    frame_numbers = sorted(data_dict.keys())
    start_frame = min(frame_numbers)
    end_frame = max(frame_numbers)

    if world_xyz is not None:
        min_bound = world_xyz.min(axis=0)
        max_bound = world_xyz.max(axis=0)
    else:
        min_bound = np.array([-10.0, -10.0, 0.0])
        max_bound = np.array([20.0, 20.0, 10.0])

    class DummyVideo:
        def get(self, prop):
            if _CV2_AVAILABLE:
                if prop == cv2.CAP_PROP_FRAME_WIDTH:
                    return 1920
                if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                    return 1080
            else:
                if prop == 3:
                    return 1920
                if prop == 4:
                    return 1080
            return 0

    tree = DetectionTree(start_frame, end_frame, DummyVideo())
    stop_after = 10

    for frame in range(start_frame, end_frame + 1):
        if frame not in data_dict:
            continue
        coords_all = data_dict[frame]
        coords = [
            pt
            for pt in coords_all
            if point_in_bounds(np.array(pt, dtype=float), min_bound, max_bound)
        ]
        stop_after = tree.add_detections(coords, frame, stop_after)

    stored_trajectories = tree.get_stored_trajectories()
    detections_per_frame = save_trajectories(stored_trajectories)
    return stored_trajectories, detections_per_frame, start_frame, end_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plotly 3D trajectory HTML from merged trajectory_filter_stats (chunked)."
    )
    parser.add_argument("--trajectory-output", default=DEFAULT_TRAJECTORY_OUTPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--world-points", default=DEFAULT_WORLD_POINTS)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--single-html", action="store_true")
    parser.add_argument("--trail-length", type=int, default=10)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.start_frame is not None and args.end_frame is not None and args.start_frame > args.end_frame:
        raise SystemExit("--start-frame must be <= --end-frame")

    os.makedirs(args.output_dir, exist_ok=True)

    merged_rows, _ = merge_filter_stats(
        args.trajectory_output,
        verbose=args.verbose_merge,
    )
    if not merged_rows:
        raise SystemExit("No merged trajectory rows; check --trajectory-output")

    bounds = _sync_frame_bounds(args.trajectory_output, args.start_frame, args.end_frame)
    if bounds is None:
        raise SystemExit("No valid sync-frame bounds after clipping.")

    lo, hi = bounds
    wp = args.world_points if os.path.exists(args.world_points) else None
    world_xyz = load_world_xyz_for_bounds(wp)
    plotly_world_path = _prepare_plotly_world_file(wp)

    if args.world_points and not os.path.exists(args.world_points) and not args.quiet:
        print(f"[3d-html] world points not found: {args.world_points}, using default axes")

    use_chunks = not args.single_html and args.chunk_size > 0
    if use_chunks:
        chunks = _chunk_ranges(lo, hi, args.chunk_size)
        if not args.quiet:
            print(
                f"[3d-html] sync range [{lo}, {hi}] -> {len(chunks)} HTML chunk(s) "
                f"(<= {args.chunk_size} sync-frame IDs each)"
            )
    else:
        chunks = [(lo, hi)]

    base = os.path.join(args.output_dir, "trajectory_3d")

    try:
        for chunk_lo, chunk_hi in chunks:
            rows_chunk = [
                r
                for r in merged_rows
                if chunk_lo <= int(r["frame_number_after_sync"]) <= chunk_hi
            ]
            if len(chunks) == 1:
                out_html = os.path.join(args.output_dir, "trajectory_3d.html")
            else:
                out_html = f"{base}_sync_{chunk_lo}_to_{chunk_hi}.html"

            try:
                stored, det_per_frame, sf, ef = build_trajectories_from_merged_rows(rows_chunk, world_xyz)
            except ValueError as e:
                if not args.quiet:
                    print(f"[3d-html] skip [{chunk_lo}, {chunk_hi}]: {e}")
                continue

            if not stored:
                if not args.quiet:
                    print(f"[3d-html] skip [{chunk_lo}, {chunk_hi}]: no trajectories")
                continue

            title = f"3D trajectory (sync {sf}–{ef}, chunk {chunk_lo}–{chunk_hi})"
            create_3d_trajectory_visualization(
                det_per_frame,
                stored,
                None,
                title=title,
                output_file=out_html,
                world_points_file=plotly_world_path,
                trail_length=args.trail_length,
            )
            if not args.quiet:
                print(f"[3d-html] wrote: {out_html}")
    finally:
        if plotly_world_path and os.path.exists(plotly_world_path):
            try:
                os.unlink(plotly_world_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
