"""
Shot segmentation and landing (bounce) — minimal vendored pipeline for Judex.

Pipeline: merge ``trajectory_filter_stats.csv`` → coarse rallies (frame gap) →
:class:`NetCrossingDetector` → :func:`detect_shot_points` (includes bounce frame).

Run::

    PYTHONPATH=cv_code python -m shot_segmentation_and_landing.postprocess_pipeline \\
        --trajectory-output /path/to/trajectory_output
"""

from .net_crossing_detector import (
    NET_Y,
    NET_HEIGHT,
    MIN_FRAMES_BETWEEN_CROSSINGS,
    CrossingResult,
    NetCrossingDetector,
)
from .filter_stats_merger import (
    get_segment_folders_sorted,
    read_filter_stats_csv,
    merge_filter_stats,
    write_merged_csv,
    write_duplicates_report,
)
from .shot_point_detection import (
    detect_shot_points,
    detect_bounce_point,
    calculate_shot_speed,
)
from .coarse_rally import (
    segment_coarse_rallies_by_frame_gap,
    subset_best_per_frame,
    merged_rows_to_best_per_frame,
    merged_rows_to_segment_data,
)
from .calib_paths import (
    default_sink_camera_pkl_path,
    default_source_camera_pkl_path,
    get_latest_calib_data_dir,
    pickleball_calib_dir,
)
from .postprocess_pipeline import run_coarse_rally_pipeline, landing_point_for_shot
from .visualize_events import create_event_overlay_video

__all__ = [
    "default_sink_camera_pkl_path",
    "default_source_camera_pkl_path",
    "get_latest_calib_data_dir",
    "pickleball_calib_dir",
    "NET_Y",
    "NET_HEIGHT",
    "MIN_FRAMES_BETWEEN_CROSSINGS",
    "CrossingResult",
    "NetCrossingDetector",
    "get_segment_folders_sorted",
    "read_filter_stats_csv",
    "merge_filter_stats",
    "write_merged_csv",
    "write_duplicates_report",
    "detect_shot_points",
    "detect_bounce_point",
    "calculate_shot_speed",
    "segment_coarse_rallies_by_frame_gap",
    "subset_best_per_frame",
    "merged_rows_to_best_per_frame",
    "merged_rows_to_segment_data",
    "run_coarse_rally_pipeline",
    "landing_point_for_shot",
    "create_event_overlay_video",
]
