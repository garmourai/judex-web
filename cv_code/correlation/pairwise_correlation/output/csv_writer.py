"""
CSV writing utilities for tracking results.
"""

import csv


class CSVWriter:
    """Class for writing tracking results to CSV files."""
    
    def __init__(self, config):
        """
        Initialize CSV writer with configuration.
        
        Args:
            config: CorrelationConfig object
        """
        self.config = config
    
    def create_tracker_csv(self, csv_path):
        """
        Create a new tracker CSV file with headers.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            csv.writer object
        """
        csvfile = open(csv_path, 'w', newline='')
        writer = csv.writer(csvfile)
        # Added Point_Coordinates_Camera1 and Point_Coordinates_Camera2 to track which points formed each tracker
        # Format: [(x1, y1), (x2, y2), ...] for each camera
        writer.writerow(['Frame', 'Visibility', 'X', 'Y', 'Z', 'Original_Frame', 
                        'Point_Coordinates_Camera1', 'Point_Coordinates_Camera2'])
        return writer, csvfile
    
    def write_frame_data(self, writer, frame_num, xs, ys, zs, original_frame=None, 
                        point_coords_cam1=None, point_coords_cam2=None):
        """
        Write frame data to CSV.
        
        Args:
            writer: CSV writer object
            frame_num: Frame number
            xs: List of X coordinates
            ys: List of Y coordinates
            zs: List of Z coordinates
            original_frame: Original frame number (optional)
            point_coords_cam1: List of (x, y) tuples from camera 1 that formed each tracker (optional)
            point_coords_cam2: List of (x, y) tuples from camera 2 that formed each tracker (optional)
        """
        if original_frame is None:
            original_frame = frame_num
            
        if xs:
            visibility = 1
            x_str = '[' + ', '.join(f"{x:.4f}" for x in xs) + ' ]'
            y_str = '[' + ', '.join(f"{y:.4f}" for y in ys) + ' ]'
            z_str = '[' + ', '.join(f"{z:.4f}" for z in zs) + ' ]'
        else:
            visibility = 0
            x_str = y_str = z_str = '[0]'
        
        # Format point coordinates as [(x1, y1), (x2, y2), ...]
        if point_coords_cam1 is not None and len(point_coords_cam1) > 0:
            cam1_coords_str = '[' + ', '.join(f"({x:.4f}, {y:.4f})" for x, y in point_coords_cam1) + ']'
        else:
            cam1_coords_str = '[]'
            
        if point_coords_cam2 is not None and len(point_coords_cam2) > 0:
            cam2_coords_str = '[' + ', '.join(f"({x:.4f}, {y:.4f})" for x, y in point_coords_cam2) + ']'
        else:
            cam2_coords_str = '[]'

        writer.writerow([frame_num, visibility, x_str, y_str, z_str, original_frame, 
                        cam1_coords_str, cam2_coords_str])
