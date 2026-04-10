#!/usr/bin/env python3
# Prefer: ./run_cv_pipeline.sh  (conda env myenv — see script) or: conda activate myenv && python run_cv_pipeline.py
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "cv_code")  # needed to unpickle camera_object.pkl (camera.Camera class)

from cv_code.pipeline_config import PipelineConfig
from cv_code.triplet_pipeline_runner import run_triplet_pipeline
from cv_code.time_profiler import TimeProfiler

CALIB_BASE = "/home/ubuntu/test_work/judex-web/calibration_testing"

# Triplet sync CSV (single file: Source_Index, per-camera frame indices, etc.)
TRIPLET_CSV_PATH = "/mnt/data/mar30_test/sync_reports/segments_1547/sync/hls_sync_1547_triple.csv"

# Optional: limit triplet CSV rows by Source_Index (same as dist_tracker Frame / global sync id).
# Inclusive range; use None for no bound on that side. If no row falls in range, inference writes
# no frames and correlation sees empty dist_tracker CSVs — widen the range or use None, None.
TRIPLET_SOURCE_INDEX_MIN = 0
TRIPLET_SOURCE_INDEX_MAX = 50000

# Shared output root (CSV dirs, profiler, TrackNet overlay MP4s under tracknet_overlay/<source|sink>/)
UNIQUE_OUTPUT_DIR = "/mnt/data/cv_output"
TRACKNET_VISUALIZATION_DIR = f"{UNIQUE_OUTPUT_DIR}/tracknet_overlay"

# Inference thread: TrackNet bbox overlay MP4s per batch (cv_code/inference/inference.py)
enable_tracknet_batch_overlay_videos = True

# Correlation thread: optional MP4 outputs (see cv_code/correlation/correlation_worker.py)
# — trajectory_videos: 3D points reprojected onto camera-1 frames (under unique_output_dir/<cam1>/visualization_videos/)
# — stitched_videos: side-by-side both cameras with correlation labels (under unique_output_dir/stitched_correlation_videos/)
enable_correlation_trajectory_videos = True
enable_correlation_stitched_videos = True

config = PipelineConfig(
    camera_1_id="source",
    camera_2_id="sink",
    # TrackNet — all explicit (no PipelineConfig defaults for these)
    tracknet_heatmap_threshold=0.2,
    tracknet_visualization_fps=30.0,
    tracknet_batch_size=4,
    tracknet_seq_len=8,
    tracknet_visualization_dir=TRACKNET_VISUALIZATION_DIR,
    tracknet_file="cv_code/inference/weights/TrackNet_best_16.engine",
    camera_1_output_dir=f"{UNIQUE_OUTPUT_DIR}/source",
    camera_2_output_dir=f"{UNIQUE_OUTPUT_DIR}/sink",
    unique_output_dir=UNIQUE_OUTPUT_DIR,
    camera_1_object_path=f"{CALIB_BASE}/source/camera_object.pkl",
    camera_2_object_path=f"{CALIB_BASE}/sink/camera_object.pkl",
    triplet_csv_path=TRIPLET_CSV_PATH,
    source_segments_dir="/mnt/data/mar30_test/sync_reports/ts_segments_source/1547/",
    sink_segments_dir="/mnt/data/mar30_test/sync_reports/ts_segments_sink/1547/",
    triplet_source_index_min=TRIPLET_SOURCE_INDEX_MIN,
    triplet_source_index_max=TRIPLET_SOURCE_INDEX_MAX,
    enable_tracknet_visualization=enable_tracknet_batch_overlay_videos,
    enable_visualization=enable_correlation_trajectory_videos,
    enable_stitched_visualization=enable_correlation_stitched_videos,
)

profiler = TimeProfiler(filepath=f"{UNIQUE_OUTPUT_DIR}/time_profiling_results.txt")

# HLS dirs: playlist.m3u8 + seg_*.ts per camera. Triplet CSV: TRIPLET_CSV_PATH above.
run_triplet_pipeline(
    config=config,
    profiler=profiler,
)
