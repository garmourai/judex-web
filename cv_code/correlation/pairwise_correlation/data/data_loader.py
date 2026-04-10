"""
Data loading and validation utilities.
"""

import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from .coordinate_processor import process_coordinates


def _coerce_pickled_camera_payload(payload, pkl_path: str):
    """
    Normalize calibration pickles from ``calibration_testing`` (dict with ``camera_matrix``,
    ``dist_coeffs``, ``image_size``, …) into an object with the attributes expected by
    :class:`Camera`.

    If ``extrinsic_pose_undistorted.json`` sits next to the ``.pkl`` (same directory), load
    ``rvec``/``tvec`` (``solve_space == "distorted"``) and set ``rotation_matrix``,
    ``translation_vectors``, and ``projection_matrix = newK @ [R|t]`` using the same
    ``newK`` convention as the calibration notebook (alpha=1.0).

    Legacy pickles (e.g. ``cv_code.camera.Camera`` instances) are returned unchanged.
    """
    if not isinstance(payload, dict):
        return payload

    d = payload
    p = Path(pkl_path)
    K = np.asarray(d["camera_matrix"], dtype=np.float64)
    dist = np.asarray(d["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    wh = (int(d["image_size"][0]), int(d["image_size"][1]))
    ctype = d.get("calibration_type") or "pinhole"

    if ctype == "fisheye":
        D = dist.astype(np.float64).reshape(4, 1)
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, wh, np.eye(3), balance=1.0
        )
        new_scaled = newK
    else:
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, wh, 1.0, wh)
        new_scaled = None

    R = None
    tvec = None
    P = None
    ext_path = p.resolve().parent / "extrinsic_pose_undistorted.json"
    if ext_path.is_file():
        extr = json.loads(ext_path.read_text(encoding="utf-8"))
        if extr.get("solve_space") == "distorted":
            rvec = np.asarray(extr["rvec"], dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(extr["tvec"], dtype=np.float64).reshape(3, 1)
            R, _ = cv2.Rodrigues(rvec)
            P = newK @ np.hstack((R, tvec))

    ns = SimpleNamespace()
    ns.camera_matrix = K
    ns.distortion_coefficients = dist
    ns.new_camera_matrix = newK
    ns.new_scaled_camera_matrix = new_scaled
    ns.calibration_type = ctype
    ns.image_size = wh
    ns.rotation_matrix = R
    ns.translation_vectors = tvec
    ns.projection_matrix = P
    ns.calibration_rotation_vectors = d.get("rvecs")
    ns.calibration_translation_vectors = d.get("tvecs")
    ns.final_dimensions = None
    return ns


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

        source_cam_data = _coerce_pickled_camera_payload(source_cam_data, source_cam_path)
        sink_cam_data = _coerce_pickled_camera_payload(sink_cam_data, sink_cam_path)

        for label, cam_data, path in (
            ("source", source_cam_data, source_cam_path),
            ("sink", sink_cam_data, sink_cam_path),
        ):
            if getattr(cam_data, "projection_matrix", None) is None:
                raise ValueError(
                    f"{label} camera: missing projection_matrix after loading {path}. "
                    "For calibration_testing dict PKLs, place extrinsic_pose_undistorted.json "
                    "(notebook Step 4, solve_space: distorted) in the same folder as the .pkl."
                )

        def _to_camera(cam_data):
            return Camera(
                camera_matrix=cam_data.camera_matrix,
                rotation_matrix=getattr(cam_data, "rotation_matrix", None),
                translation_vectors=getattr(cam_data, "translation_vectors", None),
                calibration_rotation_vectors=getattr(cam_data, "calibration_rotation_vectors", None),
                calibration_translation_vectors=getattr(
                    cam_data, "calibration_translation_vectors", None
                ),
                projection_matrix=getattr(cam_data, "projection_matrix", None),
                distortion_coefficients=getattr(cam_data, "distortion_coefficients", None),
                calibration_type=getattr(cam_data, "calibration_type", None),
                final_dimensions=getattr(cam_data, "final_dimensions", None),
                new_scaled_camera_matrix=getattr(cam_data, "new_scaled_camera_matrix", None),
                new_camera_matrix=getattr(cam_data, "new_camera_matrix", None),
                image_size=getattr(cam_data, "image_size", None),
            )

        source_cam = _to_camera(source_cam_data)
        sink_cam = _to_camera(sink_cam_data)

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
