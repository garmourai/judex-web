# This file has the definition of the Camera class

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle

class Camera:
    def __init__(self,name):
        self.name = name
        self.camera_matrix = None
        self.distortion_coefficients = None
        self.projection_matrix = None
        self.calibration_rotation_vectors = None
        self.calibration_translation_vectors = None
        self.new_camera_matrix = None
        self.rotation_matrix = None
        self.translation_vectors = None
        self.calibration_type = None
        self.final_dimensions = None
        self.new_scaled_camera_matrix = None
        self.image_size = None
        self.scale_factor = None
        self.threshold = None

    def     calibrate(self,pixel_coordinates, world_coordinates, imgpath, camera_obj_path):
        """
        Calibrate the camera using pixel and world coordinates
        :param pixel_coordinates: Pixel coordinates of the calibration points in the format [[x1,y1],[x2,y2],...]
        :type: List of lists
        :param world_coordinates: World coordinates of the calibration points in the format [[x1,y1,z1],[x2,y2,z2],...]
        :type: List of lists
        :param imgpath: Path to the image used for calibration
        :type: String
        :return: None
        """
        image = cv2.imread(imgpath)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # Calibrate the camera
        ret, self.camera_matrix, self.distortion_coefficients, self.calibration_rotation_vectors, self.calibration_translation_vectors = cv2.calibrateCamera(world_coordinates,pixel_coordinates, (image.shape[1], image.shape[0]), None, None)
        print(f"RMS Reprojection Error: {ret}")
        
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion_coefficients, (image.shape[1], image.shape[0]), 0.2, (image.shape[1], image.shape[0]))
        imgpoints=[]
        mean_error = 0
        # Project through the camera matrix to get the pixel coordinates of the calibration points
        for i in range(len(world_coordinates)):
            imgpoints2, _ = cv2.projectPoints(world_coordinates[i], self.calibration_rotation_vectors[i], self.calibration_translation_vectors[i], self.camera_matrix, self.distortion_coefficients)
            for point in imgpoints2:
                imgpoints.append(point)

            imgpoints2 = imgpoints2.squeeze()  # Remove unnecessary dimension
            imgpoints2 = np.array(imgpoints2)   # Convert to numpy array

            # Ensure pixel_coordinates[i] is a numpy array of shape (N, 2)
            pixel_coords = np.array(pixel_coordinates[i])

            # Calculate error
            error = np.linalg.norm(pixel_coords - imgpoints2) / len(imgpoints2)
            mean_error += error
        # print(mean_error / len(world_coordinates))
        imgpoints = np.array(imgpoints,dtype=np.float32)
        # Save the camera object as a pickle file
        pickle.dump(self, open(camera_obj_path, 'wb'))
        # utils.plot_img_world_points(image,imgpoints[:,0,:],None,None,"Projected Points in Zhang's Calib")
    
    def append_ones_column(self, x):
        """
        Append a column of ones to the input matrix `x`.
        
        Args:
        - x (numpy.array): Input matrix of shape (n, m).
        
        Returns:
        - numpy.array: Output matrix with an additional column of ones, shape (n, m+1).
        """
        n = x.shape[0]
        ones_column = np.ones((n, 1))
        return np.concatenate((x, ones_column), axis=1)

    def compute_projection_matrix(self, image_points, world_points,imgpath, camera_obj_path):
        """
        Compute the camera projection matrix using the Direct Linear Transformation (DLT) algorithm.
        
        Args:
        - image_points (numpy.array): Array of image coordinates of shape (n, 2).
        - world_points (numpy.array): Array of corresponding world coordinates of shape (n, 3).
        
        Returns:
        - numpy.array: Projection matrix P of shape (3, 4).
        """
        # Append ones column to image_points and world_points
        image_points_hom = self.append_ones_column(image_points)
        world_points_hom = self.append_ones_column(world_points)

        # Initialize the matrix A
        n = world_points_hom.shape[0]
        A = np.zeros((2 * n, 12))

        for i in range(n):
            xt, yt, zt = world_points_hom[i, 0], world_points_hom[i, 1], world_points_hom[i, 2]
            u, v = image_points_hom[i, 0], image_points_hom[i, 1]

            A[2*i] = [-xt, -yt, -zt, -1, 0, 0, 0, 0, u*xt, u*yt, u*zt, u]
            A[2*i+1] = [0, 0, 0, 0, -xt, -yt, -zt, -1, v*xt, v*yt, v*zt, v]

        # Perform Singular Value Decomposition (SVD) on A
        _, _, V = np.linalg.svd(A)

        # Extract the last row of V and normalize to get the projection matrix P
        P = V[-1, :] / V[-1, -1]

        # Reshape P to (3, 4) matrix
        P = P.reshape(3, 4)
        self.projection_matrix = P
        pickle.dump(self, open(camera_obj_path, 'wb'))
        return P
    
    def get_reprojected_points(self, P, X):
        """
        Compute reprojected 2D image points from 3D world points using the projection matrix P.
        
        Args:
        - P (numpy.array): Projection matrix of shape (3, 4).
        - X (numpy.array): Array of 3D world coordinates of shape (n, 3).
        
        Returns:
        - numpy.array: Reprojected 2D image points of shape (n, 2).
        """
        X_hom = self.append_ones_column(X)
        
        # Project 3D points to 2D using projection matrix P
        points_homogeneous = P @ X_hom.T
        points_homogeneous = points_homogeneous / points_homogeneous[2]
        points = points_homogeneous.T[:, 0:2]
        
        return points

    def get_reprojection_error(self, image_points, world_points, P):
        """
        Compute reprojection error between actual image points and reprojected points using projection matrix P.
        
        Args:
        - image_points (numpy.array): Array of actual image coordinates of shape (n, 2).
        - world_points (numpy.array): Array of corresponding world coordinates of shape (n, 3).
        - P (numpy.array): Projection matrix of shape (3, 4).
        
        Returns:
        - float: Mean squared reprojection error.
        """
        # Get reprojected points using the projection matrix P
        reprojected_points = self.get_reprojected_points(P, world_points)
        
        # Compute mean squared error (MSE) between reprojected points and actual image points
        diff = image_points[:, 0:2] - reprojected_points
        N = len(image_points)
        reprojection_error = (1 / N) * np.sum(np.linalg.norm(diff, axis=1) ** 2)
        
        return reprojection_error
    
    def print_calibration_data(self):
        print(f"Camera: {self.name}")
        
        if self.camera_matrix is not None:
            print("Camera Matrix:\n", self.camera_matrix)
        else:
            print("Camera Matrix: Not Set")

        if self.distortion_coefficients is not None:
            print("Distortion Coefficients:\n", self.distortion_coefficients)
        else:
            print("Distortion Coefficients: Not Set")

        if self.new_camera_matrix is not None:
            print("New Camera Matrix:\n", self.new_camera_matrix)
        else:
            print("New Camera Matrix: Not Set")