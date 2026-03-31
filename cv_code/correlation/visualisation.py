"""
Realtime-specific visualisation helpers.

Provides only what the main_realtime flow needs, separated from
shot_detection_and_trajectory_analysis.visualisation.
"""

import numpy as np


def reproject_point(cam, point3d):
    """
    Reprojects a 3D point into 2D pixel coordinates using the camera's projection matrix.

    Parameters:
    - cam: Camera object containing `projection_matrix`.
    - point3d: (x, y, z) tuple or array — 3D point in world coordinates.

    Returns:
    - (u, v): 2D pixel coordinates in the image.
    """
    point3d = np.asarray(point3d, dtype=float).reshape(3, 1)
    X_homog = np.vstack((point3d, [1]))  # (4, 1)

    P = np.asarray(cam.projection_matrix, dtype=float)  # shape (3, 4)
    x_proj = P @ X_homog  # shape (3, 1)

    # Normalize to get pixel coordinates
    x_proj /= x_proj[2, 0]
    u, v = x_proj[0, 0], x_proj[1, 0]

    return (u, v)
