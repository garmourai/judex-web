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
      --camera-pkl-path cv_code/calib_data/.../camera_object.pkl \
      --triplet-csv-path /mnt/data/mar30_test/segments_1541/sync/hls_sync_1541_triple.csv
"""

import argparse
import json
import os
import sys
from typing import List, Tuple

# Keep imports compatible with current repo layout/camera pickle unpickling.
sys.path.insert(0, ".")
sys.path.insert(0, "cv_code")

from shot_segmentation_and_landing.postprocess_pipeline import run_coarse_rally_pipeline
from shot_segmentation_and_landing.visualize_events import create_event_overlay_video


DEFAULT_TRAJECTORY_OUTPUT = "/mnt/data/cv_output/trajectory_output"
DEFAULT_OUTPUT_DIR = "/mnt/data/cv_output/trajectory_and_visualization"
DEFAULT_CAMERA_PKL = "cv_code/calib_data/1232_court2_1232/d8-3a-dd-ef-e9-03/camera_object.pkl"
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
        help="Maximum sync-frame span per overlay video chunk when start/end are provided.",
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

    triplet_csv_path = args.triplet_csv_path if os.path.exists(args.triplet_csv_path) else ""
    if not triplet_csv_path and not args.quiet:
        print("[runner] triplet csv not found; falling back to identity sync->video frame mapping")

    # Chunked visualization: when explicit range is provided, split into
    # max-N-frame blocks (default 1000) and render one video per chunk.
    if args.start_frame is not None and args.end_frame is not None:
        if args.start_frame > args.end_frame:
            raise ValueError("--start-frame must be <= --end-frame")
        if args.visualization_block_size <= 0:
            raise ValueError("--visualization-block-size must be > 0")

        chunks = _chunk_ranges(args.start_frame, args.end_frame, args.visualization_block_size)
        if not args.quiet:
            print(f"[runner] rendering {len(chunks)} visualization chunk(s)")

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

