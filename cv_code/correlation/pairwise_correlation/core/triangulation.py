"""
Triangulation algorithms for 3D point reconstruction.
"""

import numpy as np


def triangulate_dlt(camera1_coords, camera1_obj, camera2_coords, camera2_obj):
    """
    This function takes in two arrays of points from 2 different cameras and finds the original points
    : param camera1_coords: The (u,v) coordinates of the points in camera 1
    : type: numpy array of shape (m, 2)
    : param camera1: Projection Matrix of camera 1
    : type: camera object (see camera.py)
    : param camera2_coords: The (u,v) coordinates of the points in camera 2
    : type: numpy array of shape (m, 2)
    : param camera2: Projection Matrix of camera 2
    : return: A numpy array of shape (m, 3) of the (x,y,z) coordinates of the original points
    """
    camera1_P = camera1_obj.projection_matrix
    camera2_P = camera2_obj.projection_matrix
    pt2d_CxPx2 = np.array([camera1_coords, camera2_coords])  # shape (m, 2, 2)
    P_Cx3x4 = np.array([camera1_P, camera2_P])  # shape (2, 3, 4)
    Nc, Np, _ = pt2d_CxPx2.shape

    # P0 - xP2
    x = P_Cx3x4[:, 0, :][:, None, :] - np.einsum('ij,ik->ijk', pt2d_CxPx2[:, :, 0], P_Cx3x4[:, 2, :])
    # P1 - yP2
    y = P_Cx3x4[:, 1, :][:, None, :] - np.einsum('ij,ik->ijk', pt2d_CxPx2[:, :, 1], P_Cx3x4[:, 2, :])

    Ab = np.concatenate([x, y])
    Ab = np.swapaxes(Ab, 0, 1)
    assert Ab.shape == (Np, Nc * 2, 4)

    A = Ab[:, :, :3]
    b = -Ab[:, :, 3]
    AtA = np.linalg.pinv(A)

    X = np.einsum('ijk,ik->ij', AtA, b)
    return X
