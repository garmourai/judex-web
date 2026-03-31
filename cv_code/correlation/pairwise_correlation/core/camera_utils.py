"""
Camera calibration and distortion utilities.
"""

import cv2
import numpy as np


def undistort_points(cam, pts):
    """
    Undistort 2D points using the calibration parameters stored in the Camera object.
    
    Parameters:
    - cam: Camera object with .camera_matrix, .distortion_coefficients, and .new_camera_matrix or .new_scaled_camera_matrix.
    - pts: Numpy array of shape (N, 1, 2) or (N, 2) — distorted pixel coordinates.
    
    Returns:
    - undistorted_pts: Numpy array of shape (N, 2) — undistorted pixel coordinates.
    """
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim == 2:
        pts = pts.reshape(-1, 1, 2)  # Convert to shape (N, 1, 2) for OpenCV

    if cam.calibration_type == "fisheye":
        undistorted = cv2.fisheye.undistortPoints(
            pts,
            cam.camera_matrix,
            cam.distortion_coefficients,
            P=cam.new_scaled_camera_matrix
        )[:, 0, :]
    else:
        undistorted = cv2.undistortPoints(
            pts,
            cam.camera_matrix,
            cam.distortion_coefficients,
            P=cam.new_camera_matrix
        )[:, 0, :]

    return undistorted


def undistort_point(point, cam, pixel_output=True):
    """
    Undistort a single 2D point using camera parameters.

    Parameters:
    - point (tuple): (x, y) pixel coordinates in the distorted image.
    - cam (camera.Camera): Camera object with calibration parameters.
    - pixel_output (bool): If True, output will be in pixel coordinates of undistorted image.
                           If False, returns normalized undistorted coordinates.

    Returns:
    - tuple: Undistorted point in pixel coordinates or normalized coordinates.
    """

    # Convert to (N, 1, 2) format
    pts = np.array([[point]], dtype=np.float32)

    if cam.calibration_type == "fisheye":
        undistorted = cv2.fisheye.undistortPoints(
            pts,
            cam.camera_matrix,
            cam.distortion_coefficients,
            P=cam.new_scaled_camera_matrix if pixel_output else None
        )
    else:
        undistorted = cv2.undistortPoints(
            pts,
            cam.camera_matrix,
            cam.distortion_coefficients,
            P=cam.new_camera_matrix if pixel_output else None
        )

    undistorted = undistorted[0, 0]  # shape (2,)

    if pixel_output:
        # Already returned in pixel space if P is provided
        return tuple(undistorted)
    else:
        # Return normalized image coordinates
        return tuple(undistorted)
