"""
Realtime-specific pairwise correlation module.

This module provides a realtime-optimized version of pairwise correlation
that removes frame_sync_info_path dependency since frames are already synced
on a shared global frame id in the realtime flow.
"""

import os
import cv2
import numpy as np
from .pairwise_correlation.data.data_loader import DataLoader
from .pairwise_correlation.processing.segment_processor import SegmentProcessor
from .pairwise_correlation.processing.cost_analyzer import CostAnalyzer
from .pairwise_correlation.output.file_manager import FileManager
from .pairwise_correlation.output.csv_writer import CSVWriter
from .pairwise_correlation.output.metrics_writer import MetricsWriter
from .pairwise_correlation.config.settings import CorrelationConfig


class RealtimeCorrelationEngine:
    """Main engine for performing pairwise correlation in realtime flow."""
    
    def __init__(self, config):
        """
        Initialize correlation engine with configuration.
        
        Args:
            config: CorrelationConfig object
        """
        self.config = config
        self.data_loader = DataLoader()
        self.segment_processor = SegmentProcessor(config)
        self.cost_analyzer = CostAnalyzer(config)
        self.file_manager = FileManager(config)
        self.csv_writer = CSVWriter(config)
        self.metrics_writer = MetricsWriter(config)
    
    def do_pairwise_correlation(self, camera_1_cam_path, camera_2_cam_path,
                               camera_1_id, camera_2_id,
                               camera_1_video_path, camera_2_video_path,
                               output_dir, frame_segments,
                               camera_1_tracker_csv, camera_2_tracker_csv,
                               create_video=False):
        """
        Perform pairwise shuttle matching between two camera views and generate visualization.

        Both cameras must use the same global frame id column in their dist_tracker CSVs.

        Args:
            camera_1/2_cam_path: Path to camera calibration pickle files
            camera_1/2_id: Identifier for cameras
            camera_1/2_video_path: Path to input videos
            output_dir: Directory to save results
            frame_segments: List of frame ranges to process [(start1, end1), (start2, end2)]
            camera_1_tracker_csv / camera_2_tracker_csv: Paths to each camera's dist_tracker CSV
            create_video: Flag to enable video creation
        """
        alpha = self.config.DEFAULT_ALPHA
        beta = self.config.DEFAULT_BETA

        perform_correlation_output = os.path.join(output_dir, "perform_correlation_output")

        # Create directories
        self.file_manager.create_output_directory(perform_correlation_output)

        # Load camera objects
        camera_1_cam, camera_2_cam = self.data_loader.load_camera_objects(camera_1_cam_path, camera_2_cam_path)
        
        # Determine frame range for optimization (only load frames we'll actually process)
        # Calculate min and max frame across all segments
        if frame_segments:
            min_frame = min([seg[0] for seg in frame_segments])
            max_frame = max([seg[1] for seg in frame_segments])
            frame_range = (min_frame, max_frame)
        else:
            frame_range = None
        
        # Load tracking data (optimized: only load frames in the range we'll process)
        coords1 = self.data_loader.load_tracking_data(camera_1_tracker_csv, frame_range=frame_range)
        coords2 = self.data_loader.load_tracking_data(camera_2_tracker_csv, frame_range=frame_range)
        
        print(f"Processing {len(frame_segments)} frame segments")
        if frame_range:
            print(f"Loaded tracking data for frames {frame_range[0]} to {frame_range[1]} (optimized loading)")

        cap1 = None
        cap2 = None

        # Split frame segments
        frame_segments = self.segment_processor.split_frame_segments(frame_segments)

        print("frame_segments: ", frame_segments)

        for segment in frame_segments:
            self._process_segment(
                segment,
                coords1,
                coords2,
                camera_1_cam,
                camera_2_cam,
                camera_1_id,
                camera_2_id,
                camera_1_video_path,
                camera_2_video_path,
                perform_correlation_output,
                cap1,
                cap2,
                alpha,
                beta,
            )

        # Cleanup resources
        if cap1 is not None:
            cap1.release()
        if cap2 is not None:
            cap2.release()
    
    def _process_segment(self, segment, coords1, coords2, camera_1_cam, camera_2_cam,
                        camera_1_id, camera_2_id, camera_1_video_path, camera_2_video_path,
                        perform_correlation_output,
                        cap1, cap2, alpha, beta):
        """Process a single frame segment."""
        all_match_costs = []
        every_match_costs = []
        frame_cost_matrices = {}
        other_info = {}
        cam1_start_frame, cam1_end_frame = segment
        
        # Create segment output directory
        segment_output_path = self.file_manager.create_segment_directory(
            perform_correlation_output, cam1_start_frame, cam1_end_frame
        )

        # Clear existing metric files
        self.metrics_writer.clear_existing_files(segment_output_path)

        # In realtime flow, both cameras use the same global frame id as the key.
        camera_2_frame_map = {
            src: src for src in range(cam1_start_frame, cam1_end_frame + 1)
        }

        # First pass: collect cost data
        self._first_pass_processing(camera_2_frame_map, coords1, coords2, camera_1_cam, camera_2_cam,
                                  alpha, beta, all_match_costs, every_match_costs,
                                  frame_cost_matrices, other_info)

        # Calculate global threshold
        global_cost_threshold = self.cost_analyzer.calculate_global_cost_threshold(all_match_costs)
        global_cost_threshold = float('inf')  # Override as in original code

        # Second pass: full processing
        self._second_pass_processing(frame_cost_matrices, other_info, camera_2_frame_map,
                                   coords1, coords2, camera_1_cam, camera_2_cam,
                                   camera_1_id, camera_2_id, segment_output_path,
                                   cap1, cap2, global_cost_threshold)
    
    def _first_pass_processing(self, camera_2_frame_map, coords1, coords2, camera_1_cam, camera_2_cam,
                              alpha, beta, all_match_costs, every_match_costs,
                              frame_cost_matrices, other_info):
        """First pass: collect cost data for threshold calculation."""
        for frame_num in sorted(camera_2_frame_map.keys()):
            pts1 = coords1.get(frame_num, [])
            frame_num_cam2 = camera_2_frame_map[frame_num]
            pts2 = coords2.get(frame_num_cam2, [])

            _, _, matches, cost_matrix, rho_matrix, epipolar_matrix, temporal_matrix = self.segment_processor.process_frame_matching(
                frame_num, pts1, pts2, camera_1_cam, camera_2_cam, alpha, beta
            )

            if matches:
                min_cost = self.cost_analyzer.find_min_match_cost(matches, cost_matrix)
                if min_cost is not None:
                    all_match_costs.append(min_cost)
                
                match_costs = self.cost_analyzer.collect_match_costs(matches, cost_matrix)
                every_match_costs.extend(match_costs)

            # Save per-frame data for 2nd pass
            frame_cost_matrices[frame_num] = (pts1, pts2, cost_matrix, matches)
            other_info[frame_num] = (rho_matrix, epipolar_matrix, temporal_matrix)
        
    
    def _second_pass_processing(self, frame_cost_matrices, other_info, camera_2_frame_map,
                              coords1, coords2, camera_1_cam, camera_2_cam,
                              camera_1_id, camera_2_id, segment_output_path,
                              cap1, cap2, global_cost_threshold):
        """Second pass: apply global threshold and do full processing."""
        # Setup CSV writer
        csv_path = self.file_manager.get_tracker_csv_path(segment_output_path)
        writer, csvfile = self.csv_writer.create_tracker_csv(csv_path)

        try:
            frame_count = 0
            total_frames_to_process = len(frame_cost_matrices.keys())
            print(f"Processing {total_frames_to_process} frames for correlation...")
            
            for frame_num in sorted(frame_cost_matrices.keys()):
                frame_count += 1
                if frame_count % 100 == 0:  # Print progress every 100 frames
                    print(f"Processed {frame_count}/{total_frames_to_process} frames...")
                    
                pts1, pts2, cost_matrix, matches = frame_cost_matrices[frame_num]
                rho_matrix, epipolar_matrix, temporal_matrix = other_info[frame_num]

                # Filter matches using global threshold
                filtered_matches = self.cost_analyzer.filter_matches_by_threshold(
                    matches, cost_matrix, global_cost_threshold
                )

                # Write metrics
                self._write_frame_metrics(segment_output_path, frame_num, cost_matrix, 
                                        epipolar_matrix, rho_matrix, temporal_matrix)

                # Triangulate matches
                xs, ys, zs = self.segment_processor.triangulate_matches(
                    filtered_matches, pts1, pts2, camera_1_cam, camera_2_cam
                )

                # Extract point coordinates from filtered_matches
                # filtered_matches is a list of (i, j) tuples where:
                # i = index into pts1 (camera 1)
                # j = index into pts2 (camera 2)
                # Extract actual (x, y) coordinates from pts1 and pts2
                point_coords_cam1 = [pts1[match[0]] for match in filtered_matches] if filtered_matches else []
                point_coords_cam2 = [pts2[match[1]] for match in filtered_matches] if filtered_matches else []

                # Write CSV data with point coordinates
                self.csv_writer.write_frame_data(writer, frame_num, xs, ys, zs, frame_num,
                                                point_coords_cam1, point_coords_cam2)
            
            print(f"Completed processing {frame_count} frames for correlation.")

        finally:
            csvfile.close()
    
    def _write_frame_metrics(self, segment_output_path, frame_num, cost_matrix, epipolar_matrix, rho_matrix, temporal_matrix=None):
        """Write frame metrics to files."""
        cost_file = self.file_manager.get_cost_matrix_path(segment_output_path)
        epi_polar_values_file = self.file_manager.get_epipolar_values_path(segment_output_path)
        reproj_values_file = self.file_manager.get_reprojection_values_path(segment_output_path)
        temporal_values_file = self.file_manager.get_temporal_values_path(segment_output_path)
        
        self.metrics_writer.write_cost_matrix(cost_file, frame_num, cost_matrix)
        self.metrics_writer.write_epipolar_values(epi_polar_values_file, frame_num, epipolar_matrix)
        self.metrics_writer.write_reprojection_values(reproj_values_file, frame_num, rho_matrix)
        if temporal_matrix is not None:
            self.metrics_writer.write_temporal_values(temporal_values_file, frame_num, temporal_matrix)


def do_pairwise_correlation_realtime(camera_1_cam_path, camera_2_cam_path,
                                     camera_1_id, camera_2_id,
                                     camera_1_video_path, camera_2_video_path,
                                     output_dir, frame_segments,
                                     camera_1_tracker_csv, camera_2_tracker_csv,
                                     create_video=False):
    """
    Perform pairwise shuttle matching between two camera views in realtime flow.

    Uses explicit dist_tracker CSV paths per camera (shared global frame id column).
    """
    config = CorrelationConfig()
    engine = RealtimeCorrelationEngine(config)
    engine.do_pairwise_correlation(
        camera_1_cam_path, camera_2_cam_path,
        camera_1_id, camera_2_id,
        camera_1_video_path, camera_2_video_path,
        output_dir, frame_segments,
        camera_1_tracker_csv, camera_2_tracker_csv,
        create_video=create_video,
    )
