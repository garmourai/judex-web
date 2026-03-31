"""
Realtime Inference Module for Badminton Pipeline

This module handles TrackNet inference for realtime processing
with parallel frame reading and inference.
"""

import os
import time
import threading
import random
import numpy as np
import torch
from typing import Optional, Dict, List, Tuple
from torch.utils.data import DataLoader
import cv2
import pandas as pd
import shutil
import pickle
import json
import re

# Import CUDA for thread-local initialization
import pycuda.driver as cuda
import pycuda.autoinit

from .trt_utils import load_engine, allocate_buffers, do_inference
from .shared import Shuttlecock_Trajectory_Dataset, predict_location, predict_multi_location
from .utils_general import WIDTH, HEIGHT, to_img

from .frame_reader import TracknetBuffer, FrameData
from collections import deque

# Import trajectory realtime module for trajectory creation after triangulation (realtime package copy)
from ..trajectory.trajectory_realtime import create_trajectories_realtime, TrajectoryHandoffContext
# Import trajectory merging functions (realtime package copy)
from ..trajectory.merge_traj_realtime import merge_trajectories, merge_overlapping_trajectories



class RealtimeInference:
    """
    Handles TrackNet inference for realtime processing.
    
    This class manages:
    - TensorRT engine loading
    - Batch inference processing
    - CSV file writing per camera
    """
    
    def __init__(
        self,
        tracknet_file: str,
        camera_1_id: str,
        camera_2_id: str,
        camera_1_output_dir: str,
        camera_2_output_dir: str,
        batch_size: int = 4,
        seq_len: int = 8,
        profiler=None
    ):
        self.tracknet_file = tracknet_file
        self.camera_1_id = camera_1_id
        self.camera_2_id = camera_2_id
        self.camera_1_output_dir = camera_1_output_dir
        self.camera_2_output_dir = camera_2_output_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.profiler = profiler
        
        # Create output directories
        os.makedirs(camera_1_output_dir, exist_ok=True)
        os.makedirs(camera_2_output_dir, exist_ok=True)
        
        # CSV file paths
        self.csv_1_path = os.path.join(camera_1_output_dir, 'dist_tracker.csv')
        self.csv_2_path = os.path.join(camera_2_output_dir, 'dist_tracker.csv')
        
        # Initialize CSV files
        self._init_csv_files()
        
        # TensorRT engine
        self.engine = None
        self.context = None
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.stream = None
        
        # Frame counters for CSV
        self.frame_counter_1 = 0
        self.frame_counter_2 = 0
        
        # Track which cameras have had their median calculated
        self._median_calculated_for_cameras = set()  # Set of camera_ids for which median has been calculated

        # Profiling batch counters (per camera) for TimeProfiler "Batch" column
        # This makes batch numbers increment instead of staying 0.
        self._profiling_batch_by_camera: Dict[str, int] = {}
        
        self._is_initialized = False
    
    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        headers = "Frame,X,Y,Visibility\n"
        with open(self.csv_1_path, 'w') as f:
            f.write(headers)
        with open(self.csv_2_path, 'w') as f:
            f.write(headers)
    
    def initialize(self) -> bool:
        """Initialize TensorRT engine and buffers."""
        try:
            print("[InferenceWorker] Loading TensorRT engine...")
            self.engine, self.context = load_engine(self.tracknet_file)
            
            print("[InferenceWorker] Allocating buffers...")
            self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(
                self.engine, self.batch_size
            )
            
            self._is_initialized = True
            print("[InferenceWorker] ✅ Inference engine initialized")
            return True
        except Exception as e:
            print(f"[InferenceWorker] ❌ Failed to initialize inference engine: {e}")
            return False
    
    def _frames_to_numpy_array(self, frames: List[FrameData]) -> np.ndarray:
        """
        Convert list of FrameData to numpy array for dataset.
        
        Args:
            frames: List of FrameData objects
            
        Returns:
            numpy array of shape (N, H, W, 3) in RGB format
        """
        # Keep frames as BGR, then convert to RGB using numpy slicing (matches non-realtime flow)
        frame_list = [frame_data.frame for frame_data in frames]  # Keep as BGR
        frame_arr = np.array(frame_list)[:, :, :, ::-1]  # Convert to RGB using slicing
        return frame_arr
    
    def _process_batch(
        self,
        frames: List[FrameData],
        camera_mapping: List[str],
        camera_id: Optional[str] = None
    ) -> Dict[str, List]:
        """
        Process a batch of frames through TrackNet (all frames together).
        
        Args:
            frames: List of FrameData objects (96 frames total, mixed from both cameras)
            camera_mapping: List of camera IDs corresponding to each frame in frames
            camera_id: Optional camera ID for debugging (to save first batch input)
            
        Returns:
            Dictionary with predictions: {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
        """
        if not self._is_initialized:
            raise RuntimeError("Inference engine not initialized")
        
        if len(frames) == 0:
            return {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
        
        start_time_setup = time.time()
        if len(camera_mapping) != len(frames):
            raise ValueError(f"Camera mapping length ({len(camera_mapping)}) doesn't match frames length ({len(frames)})")
        
        # Get original image dimensions for coordinate scaling
        # Frames are already resized to 288x512 in buffer, so we use stored original dimensions
        if frames[0].original_width is not None and frames[0].original_height is not None:
            original_w = frames[0].original_width
            original_h = frames[0].original_height
        else:
            # Fallback: assume frames are already at TrackNet size (no scaling needed)
            original_w, original_h = WIDTH, HEIGHT
        
        w_scaler = original_w / WIDTH
        h_scaler = original_h / HEIGHT
        img_scaler = (w_scaler, h_scaler)
        
        # Convert frames to numpy array
        start_time_frames_to_numpy = time.time()
        frame_arr = self._frames_to_numpy_array(frames)
        time_taken_frames_to_numpy = time.time() - start_time_frames_to_numpy

        # Save frame_arr[:30] for debugging first batch of target camera (DISABLED)
        # if camera_id == "2c-cf-67-16-73-9a" and camera_id not in self._saved_first_batch_input:
        #     debug_dir = os.path.join(self.camera_1_output_dir, '..', 'debug_inputs', camera_id)
        #     os.makedirs(debug_dir, exist_ok=True)
        #     np.savez_compressed(
        #         os.path.join(debug_dir, 'frame_arr_first_30_realtime.npz'),
        #         frame_arr_first_30=frame_arr[:30]
        #     )
        #     print(f"[InferenceWorker]    💾 Saved frame_arr[:30] to {debug_dir}/frame_arr_first_30_realtime.npz")

        # Dataset uses in-memory frame_arr only; root_dir is unused but required by ctor
        time_taken_setup = time.time() - start_time_setup
        # Create dataset (timed separately)
        dataset_start = time.time()
        
        # Enable saving processed median for first batch of target camera
        # Set class variables BEFORE creating dataset (median calculated in __init__)
        is_first_batch_for_camera = camera_id not in self._median_calculated_for_cameras
        
        if is_first_batch_for_camera:
            print(f"[InferenceWorker]    🔄 First batch for {camera_id} - will calculate median from first 30 frames")
            
            # Enable saving processed median for debug camera (DISABLED)
            # if camera_id == "2c-cf-67-16-73-9a":
            #     Shuttlecock_Trajectory_Dataset._save_processed_median = True
            #     debug_dir = os.path.join(self.camera_1_output_dir, '..', 'debug_inputs', camera_id)
            #     Shuttlecock_Trajectory_Dataset._processed_median_debug_dir = debug_dir
            #     print(f"[InferenceWorker]    🔍 DEBUG: Set _save_processed_median=True, debug_dir={debug_dir}")
            # else:
            Shuttlecock_Trajectory_Dataset._save_processed_median = False
            Shuttlecock_Trajectory_Dataset._processed_median_debug_dir = None
        else:
            # Median already calculated for this camera, will be reused by dataset
            Shuttlecock_Trajectory_Dataset._save_processed_median = False
            Shuttlecock_Trajectory_Dataset._processed_median_debug_dir = None
        
        # Create dataset with camera_id for per-camera median storage
        dataset = Shuttlecock_Trajectory_Dataset(
            root_dir=self.camera_1_output_dir,
            seq_len=self.seq_len,
            sliding_step=self.seq_len,  # Non-overlapping
            data_mode='heatmap',
            bg_mode='concat',
            frame_arr=frame_arr,
            padding=True,
            camera_id=camera_id  # Pass camera_id for per-camera median storage
        )
        dataset_duration = time.time() - dataset_start
        
        # Mark that median has been calculated for this camera (if it was the first batch)
        # Note: The dataset class now handles this internally, but we track it here too for consistency
        if is_first_batch_for_camera:
            self._median_calculated_for_cameras.add(camera_id)

        # Create DataLoader (timed separately)
        dataloader_start = time.time()
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Use 0 for thread safety
            drop_last=False
        )
        dataloader_duration = time.time() - dataloader_start
        
        # Process batches
        pred_dict = {
            'Frame': [],
            'X': [],
            'Y': [],
            'Visibility': [],
            'BatchPosition': [],
        }
        
        # Timing for inference and post-processing
        total_inference_time = 0.0
        total_postprocess_time = 0.0
        
        for_loop_start = time.time()
        for step, (indices, x) in enumerate(data_loader):
            if step == 0:
                for_loop_duration = time.time() - for_loop_start
            x = x.float()
            temp_shape = x.shape
            
            # Save first batch input for camera 2c-cf-67-16-73-9a for debugging (DISABLED)
            # if camera_id == "2c-cf-67-16-73-9a" and step == 0 and camera_id not in self._saved_first_batch_input:
            #     self._save_first_batch_input(
            #         input_tensor=x,
            #         indices=indices,
            #         temp_shape=temp_shape,
            #         camera_id=camera_id,
            #         original_frame_indices=original_frame_indices,
            #         frame_arr_shape=frame_arr.shape
            #     )
            #     self._saved_first_batch_input.add(camera_id)
            
            # Prepare input for TensorRT
            start_time_reshape = time.time()
            x_flat = x.reshape(-1)
            self.inputs[0]['host'] = x_flat
            time_taken_reshape = time.time() - start_time_reshape

            # Run inference (timed separately)
            inference_start = time.time()
            y_pred = do_inference(
                self.engine,
                self.context,
                self.bindings,
                self.inputs,
                self.outputs,
                self.stream,
                temp_shape
            )
            inference_duration = time.time() - inference_start
            total_inference_time += inference_duration
            
            # Process predictions (timed separately)
            postprocess_start = time.time()
            y_pred_np = (y_pred > 0.25).detach().cpu().numpy()
            
            # Convert to image format (N, L, H, W)
            batch_size, seq_len = y_pred_np.shape[0], y_pred_np.shape[1]
            y_pred_img = y_pred_np.reshape(batch_size, seq_len, HEIGHT, WIDTH)
            
            # Extract coordinates from heatmaps
            # indices shape: (batch_size, seq_len, 2) where indices[n][f] = (video_idx, frame_pos_in_batch)
            indices_np = indices.numpy() if torch.is_tensor(indices) else indices
            
            for n in range(batch_size):
                # Check if we have valid indices for this batch item
                if n >= len(indices_np):
                    if step == 0:
                        print(f"[InferenceWorker]    ⚠️  Warning: batch item {n} >= len(indices_np) ({len(indices_np)}), skipping")
                    continue
                
                # Get actual sequence length for this batch item
                # indices_np[n] should have shape (seq_len, 2), but might be smaller if dataset has issues
                actual_seq_len = min(seq_len, len(indices_np[n]))
                
                # Also check the shape of indices_np[n] - it should be (seq_len, 2)
                if len(indices_np[n].shape) != 2 or indices_np[n].shape[1] != 2:
                    if step == 0:
                        print(f"[InferenceWorker]    ⚠️  Warning: indices_np[{n}] has unexpected shape {indices_np[n].shape}, expected (seq_len, 2), skipping batch item")
                    continue
                
                for f in range(actual_seq_len):
                    # Check bounds before accessing
                    if f >= len(indices_np[n]):
                        if step == 0 and n == 0:
                            print(f"[InferenceWorker]    ⚠️  Warning: frame {f} >= len(indices_np[{n}]) ({len(indices_np[n])}), skipping")
                        continue
                    
                    # Get frame position in batch from dataset indices: indices[n][f][1] is the position in frame_arr
                    if len(indices_np[n][f]) < 2:
                        if step == 0 and n == 0 and f == 0:
                            print(f"[InferenceWorker]    ⚠️  Warning: indices_np[{n}][{f}] has length {len(indices_np[n][f])}, expected 2, skipping")
                        continue
                    
                    frame_pos_in_batch = int(indices_np[n][f][1])
                    
                    if frame_pos_in_batch < len(frames):
                        frame_id_i = frames[frame_pos_in_batch].frame_id
                    else:
                        if step == 0 and n == 0 and f == 0:
                            print(f"[InferenceWorker]    ⚠️  Warning: frame_pos_in_batch {frame_pos_in_batch} >= {len(frames)}, skipping")
                        continue
                    
                    # Check if we have a valid heatmap for this frame
                    if f >= y_pred_img.shape[1]:
                        if step == 0 and n == 0:
                            print(f"[InferenceWorker]    ⚠️  Warning: frame {f} >= y_pred_img.shape[1] ({y_pred_img.shape[1]}), skipping")
                        continue
                    
                    heatmap = y_pred_img[n][f]
                    
                    # Convert boolean heatmap to uint8 format (0-255) for OpenCV
                    # Boolean True/False -> 255/0 for findContours
                    heatmap_uint8 = (heatmap.astype(np.float32) * 255).astype(np.uint8)
                    
                    # Multi-shuttle detection: get all detections (up to 3)
                    all_bbox_pred = predict_multi_location(heatmap_uint8)
                    all_cx = []
                    all_cy = []
                    
                    for bbox in all_bbox_pred:
                        # Match non-realtime: round center first, then scale and round again
                        cx_var = int(bbox[0] + bbox[2] / 2)
                        cy_var = int(bbox[1] + bbox[3] / 2)
                        cx_var = int(cx_var * img_scaler[0])
                        cy_var = int(cy_var * img_scaler[1])
                        all_cx.append(cx_var)
                        all_cy.append(cy_var)
                    
                    # Visibility: 1 if any detection found, 0 if all are (0,0)
                    vis_pred = 0
                    if len(all_cx) > 0:
                        # Check if any detection is non-zero
                        for cx, cy in zip(all_cx, all_cy):
                            if cx != 0 or cy != 0:
                                vis_pred = 1
                                break
                    
                    # Store multi-detection results (X and Y as lists, matching shuttle_tracking_main.py format)
                    # Also store batch position for camera mapping
                    pred_dict['Frame'].append(frame_id_i)
                    pred_dict['X'].append(all_cx)  # List of X coordinates
                    pred_dict['Y'].append(all_cy)  # List of Y coordinates
                    pred_dict['Visibility'].append(vis_pred)
                    pred_dict['BatchPosition'].append(frame_pos_in_batch)  # Store position for camera mapping
            
            postprocess_duration = time.time() - postprocess_start
            total_postprocess_time += postprocess_duration
            
            loop_end_time = time.time()
        
        # Total inference and post-processing times (not separately profiled)
        
        print(f"[InferenceWorker]    ✅ All batches processed, total predictions: {len(pred_dict['Frame'])}")
        
        return pred_dict
    
    def _save_first_batch_input(
        self,
        input_tensor: torch.Tensor,
        indices: torch.Tensor,
        temp_shape: tuple,
        camera_id: str,
        original_frame_indices: List[int],
        frame_arr_shape: tuple
    ):
        """
        Save the first batch input for a specific camera for debugging/comparison.
        
        Args:
            input_tensor: The input tensor x (before flattening) with shape temp_shape
            indices: The indices from DataLoader
            temp_shape: Shape of the input tensor
            camera_id: Camera ID
            original_frame_indices: List of original frame indices from frames
            frame_arr_shape: Shape of the frame_arr that was used to create the dataset
        """
        try:
            # Create debug directory
            debug_dir = os.path.join(self.camera_1_output_dir, '..', 'debug_inputs', camera_id)
            os.makedirs(debug_dir, exist_ok=True)
            
            # Convert tensors to numpy
            input_np = input_tensor.detach().cpu().numpy()
            indices_np = indices.numpy() if torch.is_tensor(indices) else indices
            
            # Prepare metadata
            metadata = {
                'camera_id': camera_id,
                'batch_number': 0,  # First batch (step 0)
                'timestamp': time.time(),
                'tensor_shape': temp_shape,
                'tensor_dtype': str(input_np.dtype),
                'original_frame_indices': original_frame_indices,
                'frame_arr_shape': frame_arr_shape,
                'num_frames': len(original_frame_indices),
                'indices_shape': indices_np.shape if hasattr(indices_np, 'shape') else str(type(indices_np))
            }
            
            # Save as .npz file (compressed numpy archive)
            save_path = os.path.join(debug_dir, 'first_batch_input.npz')
            np.savez_compressed(
                save_path,
                input_tensor=input_np,
                indices=indices_np,
                **{k: v for k, v in metadata.items() if isinstance(v, (int, float, str, tuple, list))}
            )
            
            # Save metadata as JSON for easier reading
            import json
            metadata_path = os.path.join(debug_dir, 'first_batch_metadata.json')
            # Convert numpy types to native Python types for JSON
            json_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, (int, float, str)):
                    json_metadata[k] = v
                elif isinstance(v, tuple):
                    json_metadata[k] = list(v)
                elif isinstance(v, list):
                    json_metadata[k] = v
                else:
                    json_metadata[k] = str(v)
            
            with open(metadata_path, 'w') as f:
                json.dump(json_metadata, f, indent=2)
            
            print(f"[InferenceWorker] 💾 Saved first batch input for camera {camera_id}")
            print(f"[InferenceWorker]    Input tensor shape: {temp_shape}")
            print(f"[InferenceWorker]    Saved to: {save_path}")
            print(f"[InferenceWorker]    Metadata saved to: {metadata_path}")
            
        except Exception as e:
            print(f"[InferenceWorker] ⚠️  Warning: Failed to save first batch input: {e}")
            import traceback
            traceback.print_exc()
    
    def _write_csv_append(self, csv_path: str, predictions: Dict[str, List], start_frame: int):
        """
        Append predictions to CSV file.
        Supports multi-shuttle detection where X and Y are lists.
        
        CSV format (per row): Frame,X,Y,Visibility
        """
        with open(csv_path, 'a') as f:
            num_preds = len(predictions.get('Frame', []))
            for i in range(num_preds):
                frame_num = predictions['Frame'][i]
                x = predictions['X'][i]
                y = predictions['Y'][i]
                vis = predictions['Visibility'][i]
                
                if isinstance(x, list):
                    x_str = str(x)
                else:
                    x_str = str(x)
                
                if isinstance(y, list):
                    y_str = str(y)
                else:
                    y_str = str(y)
                
                f.write(f"{frame_num},{x_str},{y_str},{vis}\n")
    
    def process_camera_batch(
        self,
        buffer: TracknetBuffer,
        camera_id: str,
        num_frames: int = 96,
        staging_buffer_1=None,
        staging_buffer_2=None
    ) -> bool:
        """
        Process 96 frames from a single camera.
        
        Args:
            buffer: TracknetBuffer containing frames
            camera_id: Camera ID to process frames from
            num_frames: Number of frames to process (default: 96)
            staging_buffer_1: Optional StagingBuffer for camera 1
            staging_buffer_2: Optional StagingBuffer for camera 2
            
        Returns:
            True if processing successful
        """
        process_start_time = time.time()
        profile_batch = self._profiling_batch_by_camera.get(camera_id, 0)
        current_size = len(buffer)
        
        # Adjust num_frames if we have fewer frames available
        if current_size < num_frames:
            num_frames = current_size
            print(f"[InferenceWorker]    ⚠️  Only {num_frames} frames available, processing with fewer frames")
        
        if num_frames == 0:
            print("[InferenceWorker]    ⚠️  No frames to process")
            return False
        
        # Get batch of frames (without removing yet)
        frames_batch = buffer.get_frames_batch(num_frames)
        
        if len(frames_batch) < num_frames:
            print(f"[InferenceWorker]    ⚠️  Could not get {num_frames} frames, got {len(frames_batch)}")
            return False
        
        # Verify all frames are from the expected camera
        for i, frame_data in enumerate(frames_batch):
            if frame_data.camera_id != camera_id:
                print(f"[InferenceWorker]    ⚠️  Warning: Frame {i} is from camera {frame_data.camera_id}, expected {camera_id}")
                # Still proceed, but log the mismatch
        
        # Create camera mapping (all same camera_id since we're processing one camera's batch)
        camera_mapping = [camera_id] * len(frames_batch)
        
        try:
            # Process frames through TrackNet
            print(f"[InferenceWorker] Processing {len(frames_batch)} frames from {camera_id}...")
            print(f"[InferenceWorker]    Starting inference processing...")
            
            # Process batch and record processing time
            inference_start = time.time()
            all_predictions = self._process_batch(frames_batch, camera_mapping, camera_id=camera_id)
            inference_duration = time.time() - inference_start
            if self.profiler:
                profiler_key = f"inference_process_96_frames_{camera_id}"
                self.profiler.record(profiler_key, inference_duration, write_immediately=True, batch=profile_batch)
            
            print(f"[InferenceWorker]    ✅ Inference processing completed, got {len(all_predictions['Frame'])} predictions")
            
            # All predictions are from the same camera, so write directly to that camera's CSV
            predictions = {
                'Frame': all_predictions['Frame'],
                'X': all_predictions['X'],
                'Y': all_predictions['Y'],
                'Visibility': all_predictions['Visibility'],
            }
            
            # Determine which CSV to write to
            if camera_id == self.camera_1_id:
                csv_path = self.csv_1_path
                frame_counter = self.frame_counter_1
                self.frame_counter_1 += len(predictions['Frame'])
            elif camera_id == self.camera_2_id:
                csv_path = self.csv_2_path
                frame_counter = self.frame_counter_2
                self.frame_counter_2 += len(predictions['Frame'])
            else:
                print(f"[InferenceWorker]    ❌ Unknown camera_id: {camera_id}")
                return False
            
            # Write to CSV
            print(f"[InferenceWorker]    Writing {len(predictions['Frame'])} predictions to {camera_id} CSV...")
            self._write_csv_append(csv_path, predictions, frame_counter)
            print(f"[InferenceWorker] ✅ Wrote {len(predictions['Frame'])} predictions to {camera_id} CSV")
            
            # Remove processed frames from TracknetBuffer (original-res pairs stay in OriginalFrameBuffer until correlation clear_batch)
            buffer_size_before = len(buffer)
            try:
                removed = buffer.remove_frames(
                    num_frames,
                    staging_buffer_1=staging_buffer_1,
                    staging_buffer_2=staging_buffer_2,
                    camera_1_id=self.camera_1_id,
                    camera_2_id=self.camera_2_id,
                    clear_staging_buffers=False,
                )
            except Exception as e:
                print(f"[InferenceWorker] ❌ remove_frames: {e}")
                import traceback
                traceback.print_exc()
                raise
            buffer_size_after = len(buffer)
            if removed != num_frames:
                print(f"[InferenceWorker] ⚠️ remove_frames: got {removed}, expected {num_frames} (before→after {buffer_size_before}→{buffer_size_after})")
            process_duration = time.time() - process_start_time
            if self.profiler:
                profiler_key = f"whole_inference_process_time_{camera_id}"
                self.profiler.record(profiler_key, process_duration, write_immediately=True, batch=profile_batch)

            # Advance per-camera profiling batch counter on success
            self._profiling_batch_by_camera[camera_id] = profile_batch + 1
            return True
            
        except Exception as e:
            print(f"[InferenceWorker] ❌ Error processing camera batch: {e}")
            import traceback
            traceback.print_exc()
            return False


def get_trajectory_color(traj_id: int) -> Tuple[int, int, int]:
    """
    Generate unique color for trajectory ID using golden ratio for good distribution.
    
    Args:
        traj_id: Trajectory identifier
        
    Returns:
        BGR color tuple for OpenCV
    """
    import colorsys
    # Use golden ratio for good color distribution
    hue = (traj_id * 0.618033988749895) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
    # Convert to BGR for OpenCV (0-255 range)
    return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))


