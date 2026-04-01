"""
Pipeline Configuration for Realtime Badminton Pipeline

This module defines the configuration dataclass that holds all
paths and settings needed for the realtime processing pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class PipelineConfig:
    """
    Configuration for the realtime processing pipeline.
    
    This class holds all paths, IDs, and settings needed to run
    the realtime inference pipeline after initial setup is complete.
    """
    
    # Camera IDs
    camera_1_id: str
    camera_2_id: str
    camera_3_id: Optional[str] = None
    camera_4_id: Optional[str] = None
    
    # Video paths
    camera_1_video_path: str = ""
    camera_2_video_path: str = ""
    
    # Output directories
    camera_1_output_dir: str = ""
    camera_2_output_dir: str = ""
    unique_output_dir: str = ""
    
    # Calibration paths
    camera_1_calibration_folder: str = ""
    camera_2_calibration_folder: str = ""
    camera_1_object_path: str = ""
    camera_2_object_path: str = ""
    
    # Metadata paths
    camera_1_json_path: str = ""
    camera_2_json_path: str = ""
    sync_info_output_path: str = ""
    # Sensor timestamp offset between camera streams at start_frame (nanoseconds).
    # Defined as: cam1_sensor_ts(start_frame) - cam2_sensor_ts(start_frame)
    start_diff_val: Optional[int] = None
    # Sensor timestamps of the first frame (start_frame) for each camera (nanoseconds). Set when metadata is available.
    camera_1_start_timestamp_ns: Optional[int] = None
    camera_2_start_timestamp_ns: Optional[int] = None

    # Model files
    tracknet_file: str = ""
    yolo_file: str = ""
    
    # Processing settings
    match_type: str = "singles"
    start_end_frames: List[List[int]] = field(default_factory=lambda: [[0, 10000]])

    # Visualization settings
    # When False, the correlation/triangulation thread will NOT create visualization videos.
    enable_visualization: bool = False
    # When true, creates a stitched correlation video showing both camera views side-by-side
    # with correlation points labeled (A, B, C...). Requires enable_visualization=true and both staging buffers.
    enable_stitched_visualization: bool = False
    
    # Video info (populated after setup)
    camera_1_total_frames: int = 0
    camera_2_total_frames: int = 0
    # Original-resolution frame size for camera 1 (optional; else taken from OriginalFrameBuffer after first frame)
    camera_1_original_frame_width: Optional[int] = None
    camera_1_original_frame_height: Optional[int] = None

    # Triplet CSV debug slice: only process rows whose Source_Index is in [min, max] inclusive.
    # Matches global Frame / dist_tracker.csv Frame column. None = no limit on that side.
    triplet_source_index_min: Optional[int] = None
    triplet_source_index_max: Optional[int] = None

    @property
    def start_frame(self) -> int:
        """Get the starting frame from start_end_frames."""
        if self.start_end_frames and len(self.start_end_frames) > 0:
            return self.start_end_frames[0][0]
        return 0
    
    @property
    def end_frame(self) -> int:
        """Get the ending frame from start_end_frames."""
        if self.start_end_frames and len(self.start_end_frames) > 0:
            return self.start_end_frames[0][1]
        return 10000
    
    def __repr__(self) -> str:
        return (
            f"PipelineConfig(\n"
            f"  camera_1_id='{self.camera_1_id}',\n"
            f"  camera_2_id='{self.camera_2_id}',\n"
            f"  camera_1_video='{self.camera_1_video_path}',\n"
            f"  camera_2_video='{self.camera_2_video_path}',\n"
            f"  frames=[{self.start_frame}, {self.end_frame}]\n"
            f")"
        )


