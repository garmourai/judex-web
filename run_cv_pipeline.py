#!/usr/bin/env python3
# Uses conda env `myenv` by default (re-execs via conda run). Opt out: JUDEX_ALLOW_NON_MYENV=1
# Or: ./run_cv_pipeline.sh  |  conda activate myenv && python run_cv_pipeline.py
import sys
import os
import shutil


def _ensure_conda_myenv() -> None:
    """Re-run this script under `conda run -n myenv` when not already in myenv."""
    if os.environ.get("JUDEX_MYENV_CHILD") == "1":
        return
    if os.environ.get("JUDEX_ALLOW_NON_MYENV") == "1":
        return
    if os.environ.get("CONDA_DEFAULT_ENV") == "myenv":
        return
    conda = shutil.which("conda")
    if conda is None:
        print(
            "[run_cv_pipeline] conda not on PATH; continuing. "
            "If imports fail, use: ./run_cv_pipeline.sh",
            flush=True,
        )
        return
    env = os.environ.copy()
    env["JUDEX_MYENV_CHILD"] = "1"
    script = os.path.abspath(__file__)
    argv = [
        conda,
        "run",
        "--no-capture-output",
        "-n",
        "myenv",
        "python",
        script,
        *sys.argv[1:],
    ]
    print("[run_cv_pipeline] Switching to conda env myenv …", flush=True)
    os.execvpe(conda, argv, env)


_ensure_conda_myenv()

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
TRIPLET_SOURCE_INDEX_MAX = 100000

# Shared output root (CSV dirs, profiler, TrackNet overlay MP4s under tracknet_overlay/<source|sink>/)
UNIQUE_OUTPUT_DIR = "/mnt/data/cv_output"
TRACKNET_VISUALIZATION_DIR = f"{UNIQUE_OUTPUT_DIR}/tracknet_overlay"

# Inference thread: TrackNet bbox overlay MP4s per batch (cv_code/inference/inference.py)
enable_tracknet_batch_overlay_videos = False

# Correlation thread: optional MP4 outputs (see cv_code/correlation/correlation_worker.py)
# — trajectory_videos: 3D points reprojected onto camera-1 frames (under unique_output_dir/<cam1>/visualization_videos/)
# — stitched_videos: side-by-side both cameras with correlation labels (under unique_output_dir/stitched_correlation_videos/)
enable_correlation_trajectory_videos = False
enable_correlation_stitched_videos = False


def _reset_output_dir(path: str) -> None:
    """Start each run from a clean output directory."""
    if not path:
        raise ValueError("UNIQUE_OUTPUT_DIR must be set")
    norm = os.path.abspath(path)
    if norm in ("/", ""):
        raise ValueError(f"Refusing to delete unsafe output path: {path!r}")
    if os.path.exists(norm):
        print(f"[run_cv_pipeline] Removing existing output dir: {norm}", flush=True)
        shutil.rmtree(norm)
    os.makedirs(norm, exist_ok=True)
    print(f"[run_cv_pipeline] Created fresh output dir: {norm}", flush=True)


_reset_output_dir(UNIQUE_OUTPUT_DIR)

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
    correlation_triplet_csv_path=TRIPLET_CSV_PATH,
    correlation_source_segments_dir="/mnt/data/mar30_test/sync_reports/ts_segments_source/1547/",
    correlation_sink_segments_dir="/mnt/data/mar30_test/sync_reports/ts_segments_sink/1547/",
)

profiler = TimeProfiler(filepath=f"{UNIQUE_OUTPUT_DIR}/time_profiling_results.txt")

# HLS dirs: playlist.m3u8 + seg_*.ts per camera. Triplet CSV: TRIPLET_CSV_PATH above.
run_triplet_pipeline(
    config=config,
    profiler=profiler,
)
