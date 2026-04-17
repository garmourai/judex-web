#!/usr/bin/env python3
"""
Path/config defaults for bounce_clips_from_hls.py.
"""

DATA_ROOT = "/mnt/data"
CV_OUTPUT = f"{DATA_ROOT}/cv_output"
SYNC_REPORTS = f"{DATA_ROOT}/mar30_test/sync_reports"
SEGMENT_SYNC_DIR = f"{SYNC_REPORTS}/segments_1547/sync"

DEFAULT_BOUNCE_CSV = f"{CV_OUTPUT}/correlation/bounce_events.csv"
DEFAULT_CAMERA = "both"
DEFAULT_SEGMENTS_DIR_SOURCE = f"{SYNC_REPORTS}/ts_segments_source/1547"
DEFAULT_SEGMENTS_DIR_SINK = f"{SYNC_REPORTS}/ts_segments_sink/1547"
# Same layout as triplet pipeline (unique_output_dir/reader/source|sink/hls_segment_frame_index.csv)
DEFAULT_READER_MANIFEST_SOURCE = f"{CV_OUTPUT}/reader/source"
DEFAULT_READER_MANIFEST_SINK = f"{CV_OUTPUT}/reader/sink"
DEFAULT_TRIPLET_CSV = f"{SEGMENT_SYNC_DIR}/hls_sync_1547_triple.csv"
DEFAULT_OUTPUT_DIR = f"{CV_OUTPUT}/bounce_clips"
