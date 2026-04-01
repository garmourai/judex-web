"""
End-to-end: merge trajectory_filter_stats → coarse rallies → net crossings → shot points + bounce.

Run from repo root::

    PYTHONPATH=cv_code python -m shot_segmentation_and_landing.postprocess_pipeline \\
        --trajectory-output /mnt/data/cv_output/trajectory_output \\
        --output /mnt/data/cv_output/rally_shot_bounce.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .coarse_rally import (
    merged_rows_to_best_per_frame,
    merged_rows_to_segment_data,
    segment_coarse_rallies_by_frame_gap,
    subset_best_per_frame,
)
from .filter_stats_merger import merge_filter_stats
from .net_crossing_detector import NET_Y, NetCrossingDetector
from .shot_point_detection import detect_shot_points
from .visualize_events import create_event_overlay_video


def landing_point_for_shot(
    shot: Dict[str, Any],
    best_per_frame: Dict[int, Tuple[int, Tuple[float, float, float]]],
) -> Optional[Dict[str, float]]:
    """If bounce_frame is set, return x,y,z at that frame (landing proxy)."""
    bf = shot.get("bounce_frame")
    if bf is None or bf not in best_per_frame:
        return None
    _, (x, y, z) = best_per_frame[bf]
    return {"frame": int(bf), "x": float(x), "y": float(y), "z": float(z)}


def run_coarse_rally_pipeline(
    trajectory_output_dir: str,
    max_frame_gap: int = 150,
    net_y: float = NET_Y,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    1. Merge all segment trajectory_filter_stats.csv files.
    2. Coarse rally ranges by frame gap.
    3. Per coarse rally: net crossings (fresh NetCrossingDetector), shot points + bounce.
    """
    merged_rows, duplicates = merge_filter_stats(trajectory_output_dir, verbose=verbose)
    if start_frame is not None or end_frame is not None:
        lo = start_frame if start_frame is not None else -10**18
        hi = end_frame if end_frame is not None else 10**18
        merged_rows = [
            r for r in merged_rows if lo <= int(r["frame_number_after_sync"]) <= hi
        ]
    if not merged_rows:
        return {
            "trajectory_output_dir": trajectory_output_dir,
            "coarse_rallies": [],
            "merged_frame_count": 0,
            "duplicate_rows_replaced": len(duplicates),
        }

    best_global = merged_rows_to_best_per_frame(merged_rows)
    sorted_frames = sorted(best_global.keys())
    coarse_ranges = segment_coarse_rallies_by_frame_gap(sorted_frames, max_frame_gap)

    if verbose:
        print(f"[postprocess] merged frames: {len(sorted_frames)}, coarse rallies: {len(coarse_ranges)}")

    results: List[Dict[str, Any]] = []

    for rally_idx, (f0, f1) in enumerate(coarse_ranges):
        rows_here = [r for r in merged_rows if f0 <= r["frame_number_after_sync"] <= f1]
        if not rows_here:
            continue

        bpf = subset_best_per_frame(best_global, f0, f1)
        segment_data = merged_rows_to_segment_data(rows_here)

        detector = NetCrossingDetector(net_y=net_y, verbose=False)
        new_crossings: List[int] = []
        for _ in detector.process_segment(segment_data, segment_name=f"coarse_{rally_idx}_{f0}_{f1}"):
            pass
        new_crossings = detector.get_all_crossings()

        shot_points = detect_shot_points(bpf, new_crossings, net_y=net_y)

        enriched_shots: List[Dict[str, Any]] = []
        for sp in shot_points:
            d = dict(sp)
            d["landing"] = landing_point_for_shot(sp, bpf)
            enriched_shots.append(d)

        results.append(
            {
                "coarse_rally_index": rally_idx,
                "frame_start": f0,
                "frame_end": f1,
                "net_crossings": new_crossings,
                "shots": enriched_shots,
            }
        )

        if verbose:
            print(
                f"  [coarse {rally_idx}] frames {f0}-{f1}: "
                f"net_crossings={len(new_crossings)}, shots={len(shot_points)}"
            )

    return {
        "trajectory_output_dir": trajectory_output_dir,
        "max_frame_gap": max_frame_gap,
        "net_y": net_y,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "merged_frame_count": len(sorted_frames),
        "duplicate_rows_replaced": len(duplicates),
        "coarse_rallies": results,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Coarse rally → net → shot → bounce (landing proxy)")
    p.add_argument(
        "--trajectory-output",
        default=os.environ.get("TRAJECTORY_OUTPUT_DIR", "/mnt/data/cv_output/trajectory_output"),
        help="Directory containing <start>_to_<end>/trajectory_filter_stats.csv",
    )
    p.add_argument("--output", "-o", default="", help="Write JSON here (default: <unique_parent>/rally_shot_bounce.json)")
    p.add_argument("--max-frame-gap", type=int, default=150, help="Split coarse rally when gap between frames exceeds this")
    p.add_argument("--net-y", type=float, default=NET_Y)
    p.add_argument("--start-frame", type=int, default=None, help="Optional inclusive start sync frame")
    p.add_argument("--end-frame", type=int, default=None, help="Optional inclusive end sync frame")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--visualize", action="store_true", help="Create event overlay video")
    p.add_argument("--video-path", default="", help="Source camera video path for overlay")
    p.add_argument("--camera-pkl-path", default="", help="Camera calibration pickle path for reprojection")
    p.add_argument(
        "--triplet-csv-path",
        default="",
        help="Triplet CSV path; Source_Index sequence drives sync-frame mapping for overlay",
    )
    p.add_argument(
        "--visualization-output",
        default="",
        help="Overlay output video path (default: <output_json_dir>/rally_shot_bounce_overlay.mp4)",
    )
    args = p.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.trajectory_output))
    default_out = os.path.join(out_dir, "rally_shot_bounce.json")
    out_path = args.output or default_out

    payload = run_coarse_rally_pipeline(
        args.trajectory_output,
        max_frame_gap=args.max_frame_gap,
        net_y=args.net_y,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        verbose=not args.quiet,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if not args.quiet:
        print(f"[postprocess] wrote {out_path}")

    if args.visualize:
        if not args.video_path or not args.camera_pkl_path:
            raise ValueError("--visualize requires --video-path and --camera-pkl-path")
        viz_out = args.visualization_output or os.path.join(
            os.path.dirname(out_path), "rally_shot_bounce_overlay.mp4"
        )
        create_event_overlay_video(
            trajectory_output_dir=args.trajectory_output,
            postprocess_json_path=out_path,
            video_path=args.video_path,
            camera_pkl_path=args.camera_pkl_path,
            output_video_path=viz_out,
            triplet_csv_path=(args.triplet_csv_path or None),
            start_sync_frame=args.start_frame,
            end_sync_frame=args.end_frame,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
