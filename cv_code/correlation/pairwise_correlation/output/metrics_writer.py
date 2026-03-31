"""
Metrics writing utilities for cost matrices and analysis data.
"""

import os


class MetricsWriter:
    """Class for writing cost matrices and other metrics to files."""
    
    def __init__(self, config):
        """
        Initialize metrics writer with configuration.
        
        Args:
            config: CorrelationConfig object
        """
        self.config = config
    
    def write_cost_matrix(self, output_path, frame_num, cost_matrix):
        """
        Write cost matrix to file.
        
        Args:
            output_path: Path to the output file
            frame_num: Frame number
            cost_matrix: Cost matrix to write
        """
        with open(output_path, 'a') as f:
            f.write(f"Frame {frame_num}\n")
            for i in range(cost_matrix.shape[0]):
                for j in range(cost_matrix.shape[1]):
                    f.write(f"src={i}, sink={j}, cost={cost_matrix[i, j]:.2f}\n")
            f.write("\n")
    
    def write_epipolar_values(self, output_path, frame_num, epipolar_matrix):
        """
        Write epipolar values to file.
        
        Args:
            output_path: Path to the output file
            frame_num: Frame number
            epipolar_matrix: Epipolar matrix to write
        """
        with open(output_path, 'a') as f:
            f.write(f"Frame {frame_num}\n")
            for i in range(epipolar_matrix.shape[0]):
                for j in range(epipolar_matrix.shape[1]):
                    f.write(f"src={i}, sink={j}, cost={epipolar_matrix[i, j]:.2f}\n")
            f.write("\n")
    
    def write_reprojection_values(self, output_path, frame_num, rho_matrix):
        """
        Write reprojection values to file.
        
        Args:
            output_path: Path to the output file
            frame_num: Frame number
            rho_matrix: Reprojection matrix to write
        """
        with open(output_path, 'a') as f:
            f.write(f"Frame {frame_num}\n")
            for i in range(rho_matrix.shape[0]):
                for j in range(rho_matrix.shape[1]):
                    f.write(f"src={i}, sink={j}, cost={rho_matrix[i, j]:.2f}\n")
            f.write("\n")
    
    def clear_existing_files(self, segment_output_path):
        """
        Clear existing metric files if they exist.
        
        Args:
            segment_output_path: Path to segment output directory
        """
        cost_file = os.path.join(segment_output_path, self.config.COST_MATRIX_FILE)
        epi_polar_values_file = os.path.join(segment_output_path, self.config.EPIPOLAR_VALUES_FILE)
        reproj_values_file = os.path.join(segment_output_path, self.config.REPROJ_VALUES_FILE)
        temporal_values_file = os.path.join(segment_output_path, self.config.TEMPORAL_VALUES_FILE)
        
        for file_path in [cost_file, epi_polar_values_file, reproj_values_file, temporal_values_file]:
            if os.path.exists(file_path):
                os.remove(file_path)

    def write_temporal_values(self, output_path, frame_num, temporal_matrix):
        """
        Write temporal values to file.
        
        Args:
            output_path: Path to the output file
            frame_num: Frame number
            temporal_matrix: Temporal cost matrix to write
        """
        with open(output_path, 'a') as f:
            f.write(f"Frame {frame_num}\n")
            for i in range(temporal_matrix.shape[0]):
                for j in range(temporal_matrix.shape[1]):
                    f.write(f"src={i}, sink={j}, cost={temporal_matrix[i, j]:.2f}\n")
            f.write("\n")
