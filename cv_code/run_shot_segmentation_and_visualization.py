#!/usr/bin/env python3
"""
Convenience runner for shot segmentation + optional event visualization.

Default output directory:
    /mnt/data/cv_output/trajectory_and_visualization

Usage examples:

1) JSON only
    python cv_code/run_shot_segmentation_and_visualization.py

2) JSON + overlay video
    python cv_code/run_shot_segmentation_and_visualization.py \
      --video-path /path/to/source_video.mp4 \
      --camera-pkl-path <latest under tools/pickleball_calib>/source/camera_object.pkl \\
      --triplet-csv-path /mnt/data/mar30_test/segments_1541/sync/hls_sync_1541_triple.csv
"""

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

# Keep imports compatible with current repo layout/camera pickle unpickling.
sys.path.insert(0, ".")
sys.path.insert(0, "cv_code")

from shot_segmentation_and_landing.filter_stats_merger import merge_filter_stats
from shot_segmentation_and_landing.postprocess_pipeline import run_coarse_rally_pipeline
from shot_segmentation_and_landing.visualize_events import create_event_overlay_video


DEFAULT_TRAJECTORY_OUTPUT = "/mnt/data/cv_output/trajectory_output"
DEFAULT_OUTPUT_DIR = "/mnt/data/cv_output/trajectory_and_visualization"
DEFAULT_CALIB_BASE = "/home/ubuntu/test_work/judex-web/tools/pickleball_calib/calibration_1512"
DEFAULT_CAMERA_PKL = os.path.join(DEFAULT_CALIB_BASE, "source", "camera_object.pkl")
DEFAULT_VIDEO_PATH = "/mnt/data/mar30_test/ts_segments_source/1541/playlist.m3u8"
DEFAULT_TRIPLET_CSV = "/mnt/data/mar30_test/segments_1541/sync/hls_sync_1541_triple.csv"


def _chunk_ranges(start_frame: int, end_frame: int, block_size: int) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    cur = start_frame
    while cur <= end_frame:
        nxt = min(cur + block_size - 1, end_frame)
        ranges.append((cur, nxt))
        cur = nxt + 1
    return ranges


def _sync_frame_bounds_for_visualization(
    trajectory_output_dir: str,
    start_frame: Optional[int],
    end_frame: Optional[int],
) -> Optional[Tuple[int, int]]:
    """
    Inclusive sync-frame range [lo, hi] for overlay chunking.
    Uses merged trajectory rows; optional start/end clip that range.
    Returns None if no trajectory rows exist.
    """
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run postprocess shot segmentation and optional overlay visualization."
    )
    parser.add_argument("--trajectory-output", default=DEFAULT_TRAJECTORY_OUTPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-frame-gap", type=int, default=150)
    parser.add_argument("--net-y", type=float, default=6.7)
    parser.add_argument("--start-frame", type=int, default=None, help="Optional inclusive start sync frame")
    parser.add_argument("--end-frame", type=int, default=None, help="Optional inclusive end sync frame")
    parser.add_argument(
        "--visualization-block-size",
        type=int,
        default=1000,
        help=(
            "Split overlay into multiple MP4s, each covering at most this many sync-frame IDs "
            "(default 1000). Set to 0 for one file. Bounds come from trajectory data unless "
            "--start-frame/--end-frame clip them."
        ),
    )
    parser.add_argument(
        "--single-overlay",
        action="store_true",
        help="Write one rally_shot_bounce_overlay.mp4 instead of chunked files.",
    )
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument(
        "--video-path",
        default=DEFAULT_VIDEO_PATH,
        help=(
            "Source camera stream/video for overlay. Defaults to source playlist from run_cv_pipeline.py. "
            "Set empty to skip visualization."
        ),
    )
    parser.add_argument(
        "--camera-pkl-path",
        default=DEFAULT_CAMERA_PKL,
        help="Camera calibration pickle used for 3D->2D reprojection.",
    )
    parser.add_argument(
        "--triplet-csv-path",
        default=DEFAULT_TRIPLET_CSV,
        help=(
            "Triplet CSV path. Source_Index sequence is used as sync-frame reference "
            "for overlay rendering."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    json_path = os.path.join(args.output_dir, "rally_shot_bounce.json")
    overlay_path = os.path.join(args.output_dir, "rally_shot_bounce_overlay.mp4")

    payload = run_coarse_rally_pipeline(
        trajectory_output_dir=args.trajectory_output,
        max_frame_gap=args.max_frame_gap,
        net_y=args.net_y,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        verbose=not args.quiet,
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if not args.quiet:
        print(f"[runner] wrote JSON: {json_path}")

    if not args.video_path:
        raise ValueError("Visualization is enabled by default; --video-path is required.")

    if args.start_frame is not None and args.end_frame is not None and args.start_frame > args.end_frame:
        raise ValueError("--start-frame must be <= --end-frame")

    triplet_csv_path = args.triplet_csv_path if os.path.exists(args.triplet_csv_path) else ""
    if not triplet_csv_path and not args.quiet:
        print("[runner] triplet csv not found; falling back to identity sync->video frame mapping")

    # Chunked visualization: default split into ~visualization_block_size sync-frame spans
    # (easy to download). Optional --single-overlay for one file.
    use_chunks = (
        not args.single_overlay
        and args.visualization_block_size > 0
    )
    bounds = _sync_frame_bounds_for_visualization(
        args.trajectory_output, args.start_frame, args.end_frame
    )

    if use_chunks and bounds is not None:
        lo, hi = bounds
        chunks = _chunk_ranges(lo, hi, args.visualization_block_size)
        if not args.quiet:
            print(
                f"[runner] overlay sync range [{lo}, {hi}] -> {len(chunks)} chunk(s) "
                f"(<= {args.visualization_block_size} sync-frame IDs each)"
            )

        base_no_ext, ext = os.path.splitext(overlay_path)
        ext = ext or ".mp4"
        for chunk_start, chunk_end in chunks:
            chunk_out = f"{base_no_ext}_{chunk_start}_to_{chunk_end}{ext}"
            create_event_overlay_video(
                trajectory_output_dir=args.trajectory_output,
                postprocess_json_path=json_path,
                video_path=args.video_path,
                camera_pkl_path=args.camera_pkl_path,
                output_video_path=chunk_out,
                triplet_csv_path=(triplet_csv_path or None),
                start_sync_frame=chunk_start,
                end_sync_frame=chunk_end,
                verbose=not args.quiet,
            )
            if not args.quiet:
                print(f"[runner] wrote overlay chunk: {chunk_out}")
    else:
        if use_chunks and bounds is None and not args.quiet:
            print("[runner] no merged trajectory rows; writing single overlay without chunk bounds")
        create_event_overlay_video(
            trajectory_output_dir=args.trajectory_output,
            postprocess_json_path=json_path,
            video_path=args.video_path,
            camera_pkl_path=args.camera_pkl_path,
            output_video_path=overlay_path,
            triplet_csv_path=(triplet_csv_path or None),
            start_sync_frame=args.start_frame,
            end_sync_frame=args.end_frame,
            verbose=not args.quiet,
        )
        if not args.quiet:
            print(f"[runner] wrote overlay: {overlay_path}")


if __name__ == "__main__":
    main()

