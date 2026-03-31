"""
Judex Pipeline Module

Provides the triplet-CSV-based realtime inference pipeline that reads
pre-synchronized frame indices from triplet.csv and video from HLS .ts
segment files.
"""

from .m3u8_reader import M3U8SegmentReader

from .triplet_csv_reader import (
    TripletRow,
    OriginalFrameBuffer,
    TripletCSVReaderWorker,
)

from .triplet_pipeline_runner import (
    TripletPipeline,
    run_triplet_pipeline,
)

__all__ = [
    "M3U8SegmentReader",
    "TripletRow",
    "OriginalFrameBuffer",
    "TripletCSVReaderWorker",
    "TripletPipeline",
    "run_triplet_pipeline",
]
