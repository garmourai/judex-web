"""
Data loading and validation utilities.
"""

import os
import pickle
import pandas as pd
from .coordinate_processor import process_coordinates


class Camera:
    """Camera calibration object."""
    
    def __init__(
        self,
        camera_matrix,
        rotation_matrix=None,
        translation_vectors=None,
        calibration_rotation_vectors=None,
        calibration_translation_vectors=None,
        projection_matrix=None,
        distortion_coefficients=None,
        calibration_type=None,
        final_dimensions=None,
        new_scaled_camera_matrix=None,
        image_size=None,
        scale_factor=None,
        threshold=None,
        new_camera_matrix=None
    ):
        self.camera_matrix = camera_matrix
        self.rotation_matrix = rotation_matrix
        self.translation_vectors = translation_vectors
        self.calibration_rotation_vectors = calibration_rotation_vectors
        self.calibration_translation_vectors = calibration_translation_vectors
        self.projection_matrix = projection_matrix
        self.distortion_coefficients = distortion_coefficients
        self.calibration_type = calibration_type
        self.final_dimensions = final_dimensions
        self.new_scaled_camera_matrix = new_scaled_camera_matrix
        self.image_size = image_size
        self.scale_factor = scale_factor
        self.threshold = threshold
        self.new_camera_matrix = new_camera_matrix


class DataLoader:
    """Class for loading camera calibration and tracking data."""
    
    def load_camera_objects(self, source_cam_path, sink_cam_path):
        """
        Load camera calibration objects from pickle files.
        
        Args:
            source_cam_path: Path to source camera calibration file
            sink_cam_path: Path to sink camera calibration file
            
        Returns:
            Tuple of (source_cam, sink_cam) Camera objects
        """
        if not os.path.exists(source_cam_path):
            raise FileNotFoundError(f"Source camera calibration file not found: {source_cam_path}")
        if not os.path.exists(sink_cam_path):
            raise FileNotFoundError(f"Sink camera calibration file not found: {sink_cam_path}")
            
        with open(source_cam_path, 'rb') as f:
            source_cam_data = pickle.load(f)

        with open(sink_cam_path, 'rb') as f:
            sink_cam_data = pickle.load(f)

        source_cam = Camera(
            camera_matrix=source_cam_data.camera_matrix,
            rotation_matrix=source_cam_data.rotation_matrix,
            translation_vectors=source_cam_data.translation_vectors,
            calibration_rotation_vectors=source_cam_data.calibration_rotation_vectors,
            calibration_translation_vectors=source_cam_data.calibration_translation_vectors,
            projection_matrix=source_cam_data.projection_matrix,
            distortion_coefficients=source_cam_data.distortion_coefficients,
            calibration_type=source_cam_data.calibration_type,
            final_dimensions=source_cam_data.final_dimensions,
            new_scaled_camera_matrix=source_cam_data.new_scaled_camera_matrix,
            new_camera_matrix=source_cam_data.new_camera_matrix,
        )

        sink_cam = Camera(
            camera_matrix=sink_cam_data.camera_matrix,
            rotation_matrix=sink_cam_data.rotation_matrix,
            translation_vectors=sink_cam_data.translation_vectors,
            calibration_rotation_vectors=sink_cam_data.calibration_rotation_vectors,
            calibration_translation_vectors=sink_cam_data.calibration_translation_vectors,
            projection_matrix=sink_cam_data.projection_matrix,
            distortion_coefficients=sink_cam_data.distortion_coefficients,
            calibration_type=sink_cam_data.calibration_type,
            final_dimensions=sink_cam_data.final_dimensions,
            new_scaled_camera_matrix=sink_cam_data.new_scaled_camera_matrix,
            new_camera_matrix=sink_cam_data.new_camera_matrix,
        )

        return source_cam, sink_cam
    
    def load_tracking_data(self, csv_path, frame_range=None):
        """
        Load tracking data from CSV file and process coordinates.
        
        Args:
            csv_path: Path to tracking CSV file
            frame_range: Optional tuple (min_frame, max_frame) to only load frames in this range
            
        Returns:
            Dictionary mapping frame numbers to coordinate lists
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Tracking CSV file not found: {csv_path}")
        
        # CSV format: Frame,X,Y,Visibility (preferred). Legacy 5-column: Frame,extra,X,Y,Visibility
        # where X and Y can be lists like "[300, 500, 0]"
        # The CSV is written without quotes, so pandas will fail to parse due to commas in lists
        # We need to manually parse the CSV to handle lists properly
        # Optimization: If frame_range is provided, we can skip frames outside the range early
        data = []
        min_frame = None
        max_frame = None
        if frame_range:
            min_frame, max_frame = frame_range
        
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            if len(lines) == 0:
                return {}
            
            # Skip header
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                
                # Early optimization: Extract frame number first (before full parsing)
                # Frame is the first field before the first comma (when not inside brackets)
                # Quick check: find first comma that's not inside brackets
                first_comma_pos = -1
                bracket_depth = 0
                for i, char in enumerate(line):
                    if char == '[':
                        bracket_depth += 1
                    elif char == ']':
                        bracket_depth -= 1
                    elif char == ',' and bracket_depth == 0:
                        first_comma_pos = i
                        break
                
                # If we have a frame range, check frame number before full parsing
                if frame_range and first_comma_pos > 0:
                    try:
                        frame = int(line[:first_comma_pos].strip())
                        # Skip if frame is outside range
                        if frame < min_frame or frame > max_frame:
                            continue
                    except ValueError:
                        # If we can't parse frame number, continue with full parsing
                        pass
                
                # Parse line: Frame,X,Y,Visibility (optional fifth field)
                # X and Y can be lists like "[300, 500, 0]" or single numbers
                # Strategy: Track bracket depth to properly split fields
                fields = []
                current_field = ""
                bracket_depth = 0
                
                for char in line:
                    if char == '[':
                        bracket_depth += 1
                        current_field += char
                    elif char == ']':
                        bracket_depth -= 1
                        current_field += char
                    elif char == ',' and bracket_depth == 0:
                        # This comma is a field separator
                        fields.append(current_field.strip())
                        current_field = ""
                    else:
                        current_field += char
                
                # Add the last field
                if current_field:
                    fields.append(current_field.strip())
                
                if len(fields) == 4:
                    try:
                        frame_id = int(fields[0])
                        if frame_range and (frame_id < min_frame or frame_id > max_frame):
                            continue
                        x_str = fields[1]
                        y_str = fields[2]
                        vis = int(fields[3])
                        data.append({
                            'Frame': frame_id,
                            'X': x_str,
                            'Y': y_str,
                            'Visibility': vis
                        })
                    except (ValueError, IndexError):
                        continue
                elif len(fields) == 5:
                    try:
                        frame_id = int(fields[0])
                        if frame_range and (frame_id < min_frame or frame_id > max_frame):
                            continue
                        x_str = fields[2]
                        y_str = fields[3]
                        vis = int(fields[4])
                        data.append({
                            'Frame': frame_id,
                            'X': x_str,
                            'Y': y_str,
                            'Visibility': vis
                        })
                    except (ValueError, IndexError):
                        continue
        
        # Convert to DataFrame for compatibility with existing code
        if len(data) == 0:
            return {}
        
        df = pd.DataFrame(data)
        coords = {row['Frame']: process_coordinates(row) for _, row in df.iterrows()}
        return coords
