#!/usr/bin/env python3
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "cv_code")  # needed to unpickle camera_object.pkl (camera.Camera class)

from cv_code.pipeline_config import PipelineConfig
from cv_code.triplet_pipeline_runner import run_triplet_pipeline
from cv_code.time_profiler import TimeProfiler

# Optional: limit triplet CSV rows by Source_Index (same as dist_tracker Frame / global sync id).
# Inclusive range; use None for no bound on that side. If no row falls in range, inference writes
# no frames and correlation sees empty dist_tracker CSVs — widen the range or use None, None.
TRIPLET_SOURCE_INDEX_MIN = 200
TRIPLET_SOURCE_INDEX_MAX = 50000

config = PipelineConfig(
    camera_1_id="source",
    camera_2_id="sink",
    tracknet_file="cv_code/inference/weights/TrackNet_best_16.engine",
    camera_1_output_dir="/mnt/data/cv_output/source",
    camera_2_output_dir="/mnt/data/cv_output/sink",
    unique_output_dir="/mnt/data/cv_output",
    camera_1_object_path="cv_code/calib_data/1232_court2_1232/d8-3a-dd-ef-e9-03/camera_object.pkl",
    camera_2_object_path="cv_code/calib_data/1232_court2_1232/2c-cf-67-16-73-9a/camera_object.pkl",
    triplet_source_index_min=TRIPLET_SOURCE_INDEX_MIN,
    triplet_source_index_max=TRIPLET_SOURCE_INDEX_MAX,
)

profiler = TimeProfiler(filepath="/mnt/data/cv_output/time_profiling_results.txt")

# Correlation debug videos (see cv_code/correlation/correlation_worker.py)
enable_visualization = True
enable_stitched_visualization = True

run_triplet_pipeline(
    triplet_csv_path="/mnt/data/mar30_test/segments_1541/sync/hls_sync_1541_triple.csv",
    source_segments_dir="/mnt/data/mar30_test/ts_segments_source/1541/",
    source_segment_csvs_dir="/mnt/data/mar30_test/segments_1541/source/",
    sink_segments_dir="/mnt/data/mar30_test/ts_segments_sink/1541/",
    sink_segment_csvs_dir="/mnt/data/mar30_test/segments_1541/sink/",
    config=config,
    enable_visualization=enable_visualization,
    enable_stitched_visualization=enable_stitched_visualization,
    profiler=profiler,
)
