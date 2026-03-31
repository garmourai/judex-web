"""
Frame segment processing utilities.
"""

import cv2
import numpy as np
from ..core.triangulation import triangulate_dlt
from ..core.matching import match_shuttles
from ..core.camera_utils import undistort_points, undistort_point


class SegmentProcessor:
    """Class for processing individual frame segments."""
    
    def __init__(self, config):
        """
        Initialize segment processor with configuration.
        
        Args:
            config: CorrelationConfig object
        """
        self.config = config
        # Maintain previous-frame state for temporal consistency in engine path
        self._prev_undist_pts1 = None
        self._prev_undist_pts2 = None
        self._prev_frame_num = None
    
    def split_frame_segments(self, frame_segments):
        """
        Split frame segments into smaller chunks based on max segment length.
        
        Args:
            frame_segments: List of [start_frame, end_frame] pairs
            
        Returns:
            List of split frame segments
        """
        split_frame_segments = []
        
        for segment in frame_segments:
            start_frame = segment[0]
            end_frame = segment[1]

            while start_frame <= end_frame:
                split_end = min(start_frame + self.config.MAX_SEGMENT_LENGTH - 1, end_frame)
                split_frame_segments.append([start_frame, split_end])
                start_frame = split_end + 1

        return split_frame_segments
    
    def adjust_frame_segments_to_video_length(self, frame_segments, total_frames):
        """
        Adjust frame segments to ensure they don't exceed video length.
        
        Args:
            frame_segments: List of [start_frame, end_frame] pairs
            total_frames: Total number of frames in video
            
        Returns:
            List of adjusted frame segments
        """
        updated_frame_segments = []
        for segment in frame_segments:
            start_frame = segment[0]
            end_frame = min(segment[1], total_frames - 1)
            updated_frame_segments.append([start_frame, end_frame])
        
        return updated_frame_segments
    
    def process_frame_matching(self, frame_num, pts1, pts2, camera_1_cam, camera_2_cam, alpha, beta):
        """
        Process matching for a single frame.
        
        Args:
            frame_num: Frame number
            pts1: Points from camera 1
            pts2: Points from camera 2
            camera_1_cam: Camera 1 object
            camera_2_cam: Camera 2 object
            alpha: Alpha parameter for matching
            beta: Beta parameter for matching
            
        Returns:
            Tuple of (undist_pts1, undist_pts2, matches, cost_matrix, rho_matrix, epipolar_matrix, temporal_matrix)
        """
        undist_pts1 = [undistort_point((x, y), camera_1_cam, pixel_output=True) for (x, y) in pts1]
        undist_pts2 = [undistort_point((x, y), camera_2_cam, pixel_output=True) for (x, y) in pts2]

        # Compute frame gap if previous state exists
        if self._prev_frame_num is None:
            prev_gap = 1
        else:
            prev_gap = max(1, frame_num - self._prev_frame_num)

        # Use previous undistorted points for temporal consistency
        matches, cost_matrix, rho_matrix, epipolar_matrix, temporal_matrix = match_shuttles(
            camera_1_cam, camera_2_cam,
            undist_pts1, undist_pts2,
            alpha, beta, gamma=0.5,
            cost_threshold=None,
            prev_pts1=self._prev_undist_pts1,
            prev_pts2=self._prev_undist_pts2,
            prev_matches=getattr(self, "_prev_matches", None),
            prev_frame_gap=prev_gap,
            max_frame_gap_for_temporal=5,
            frame_num = frame_num
        )

        # Update previous state for next call
        self._prev_undist_pts1 = undist_pts1
        self._prev_undist_pts2 = undist_pts2
        self._prev_frame_num = frame_num
        self._prev_matches = matches
        
        # Return temporal matrix for logging and analysis
        return undist_pts1, undist_pts2, matches, cost_matrix, rho_matrix, epipolar_matrix, temporal_matrix
    
    def triangulate_matches(self, filtered_matches, pts1, pts2, camera_1_cam, camera_2_cam):
        """
        Triangulate matched points to get 3D coordinates.
        
        Args:
            filtered_matches: List of filtered (i, j) match pairs
            pts1: Points from camera 1
            pts2: Points from camera 2
            camera_1_cam: Camera 1 object
            camera_2_cam: Camera 2 object
            
        Returns:
            Lists of X, Y, Z coordinates
        """
        xs, ys, zs = [], [], []
        
        for i, j in filtered_matches:
            p1 = np.array([[pts1[i]]], dtype=np.float32)
            p2 = np.array([[pts2[j]]], dtype=np.float32)

            u1 = undistort_points(camera_1_cam, p1)
            u2 = undistort_points(camera_2_cam, p2)

            traj = triangulate_dlt(u1, camera_1_cam, u2, camera_2_cam)

            xs.append(traj[0, 0])
            ys.append(traj[0, 1])
            zs.append(traj[0, 2])
        
        return xs, ys, zs
