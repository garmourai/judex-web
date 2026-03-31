"""
File and directory management utilities.
"""

import os


class FileManager:
    """Class for managing directories and file paths."""
    
    def __init__(self, config):
        """
        Initialize file manager with configuration.
        
        Args:
            config: CorrelationConfig object
        """
        self.config = config
    
    def create_output_directory(self, output_dir):
        """
        Create output directory if it doesn't exist.
        
        Args:
            output_dir: Path to output directory
        """
        os.makedirs(output_dir, exist_ok=True)
    
    def create_segment_directory(self, base_output_dir, cam1_start_frame, cam1_end_frame):
        """
        Create segment-specific output directory.
        
        Args:
            base_output_dir: Base output directory
            cam1_start_frame: Start frame for camera 1
            cam1_end_frame: End frame for camera 1
            
        Returns:
            Path to segment output directory
        """
        segment_output_path = os.path.join(base_output_dir, f"{cam1_start_frame}_to_{cam1_end_frame}")
        os.makedirs(segment_output_path, exist_ok=True)
        return segment_output_path
    
    def get_tracker_csv_path(self, segment_output_path):
        """
        Get path for tracker CSV file.
        
        Args:
            segment_output_path: Segment output directory path
            
        Returns:
            Path to tracker CSV file
        """
        return os.path.join(segment_output_path, self.config.TRACKER_CSV_FILE)
    
    def get_correlation_video_path(self, segment_output_path):
        """
        Get path for correlation video file.
        
        Args:
            segment_output_path: Segment output directory path
            
        Returns:
            Path to correlation video file
        """
        return os.path.join(segment_output_path, self.config.CORRELATION_VIDEO_FILE)
    
    def get_cost_matrix_path(self, segment_output_path):
        """
        Get path for cost matrix file.
        
        Args:
            segment_output_path: Segment output directory path
            
        Returns:
            Path to cost matrix file
        """
        return os.path.join(segment_output_path, self.config.COST_MATRIX_FILE)
    
    def get_epipolar_values_path(self, segment_output_path):
        """
        Get path for epipolar values file.
        
        Args:
            segment_output_path: Segment output directory path
            
        Returns:
            Path to epipolar values file
        """
        return os.path.join(segment_output_path, self.config.EPIPOLAR_VALUES_FILE)
    
    def get_reprojection_values_path(self, segment_output_path):
        """
        Get path for reprojection values file.
        
        Args:
            segment_output_path: Segment output directory path
            
        Returns:
            Path to reprojection values file
        """
        return os.path.join(segment_output_path, self.config.REPROJ_VALUES_FILE)

    def get_temporal_values_path(self, segment_output_path):
        """
        Get path for temporal values file.
        
        Args:
            segment_output_path: Segment output directory path
            
        Returns:
            Path to temporal values file
        """
        return os.path.join(segment_output_path, self.config.TEMPORAL_VALUES_FILE)
