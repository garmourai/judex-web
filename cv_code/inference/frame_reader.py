"""
Frame Reader Module for Realtime Badminton Pipeline

This module provides classes for reading and buffering video frames
from multiple cameras in a synchronized manner.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from collections import deque
import pickle
from pathlib import Path
import threading


@dataclass
class FrameData:
    """Represents a single frame with its metadata."""
    frame: np.ndarray
    camera_stream_index: int  # Per-camera consecutive index (e.g. source vs sink row index)
    timestamp_ms: float
    camera_id: str
    frame_id: int  # Global sync id (matches CSV Frame column and correlation keys)
    original_width: Optional[int] = None  # Original frame width before resizing
    original_height: Optional[int] = None  # Original frame height before resizing
    sensor_timestamp: Optional[int] = None  # Sensor timestamp in nanoseconds from metadata JSON
    is_placeholder: bool = False          # True if this entry is a synthetic empty frame
    original_frame_number: Optional[int] = None  # Original video frame number (before any drops)


class TracknetBuffer:
    """
    A buffer to store frames for TrackNet inference processing.
    
    This buffer stores frames from cameras and maintains
    a mapping to track which frames belong to which camera.
    
    Attributes:
        max_size: Maximum number of frames to buffer (default: 576)
        frames: Deque storing FrameData objects from all cameras
        camera_map: List mapping buffer index to camera_id
    """
    
    def __init__(self, max_size: int = 576):
        self.max_size = max_size
        self.frames: deque[FrameData] = deque(maxlen=max_size)
        self.camera_map: List[str] = []  # Maps buffer index to camera_id
        self._frame_count = 0
        self._is_paused = False
        # Thread safety
        self._lock = threading.Lock()
        self._not_full = threading.Condition(self._lock)  # Notify when buffer has space
        # Per-camera batch queues: each entry is (num_frames_in_batch, optional_metadata_dict)
        # TrackNet coordinator enqueues one entry per logical batch once ALL frames are pushed.
        self._batch_queues: Dict[str, deque[Tuple[int, Optional[Dict]]]] = {}
        self._batch_available = threading.Condition(self._lock)
    
    def add_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_ms: float,
        camera_id: str,
        original_width: Optional[int] = None,
        original_height: Optional[int] = None,
        sensor_timestamp: Optional[int] = None,
        tracknet_index: Optional[int] = None,
        is_placeholder: bool = False,
    ) -> bool:
        """
        Add a frame to the unified buffer (thread-safe).
        
        Args:
            frame: The video frame as numpy array (should already be resized to TrackNet size 288x512)
            frame_index: Per-camera stream index (consecutive after drops for that camera)
            timestamp_ms: Timestamp in milliseconds
            camera_id: Identifier for the camera this frame came from
            original_width: Original frame width before resizing (for coordinate scaling)
            original_height: Original frame height before resizing (for coordinate scaling)
            
        Returns:
            True if frame was added, False if buffer is full
        """
        with self._lock:
            # Check if full (we already have the lock, so check directly)
            if len(self.frames) >= self.max_size:
                self._is_paused = True
                return False
            
            fid = tracknet_index if tracknet_index is not None else frame_index
            frame_data = FrameData(
                frame=frame,
                camera_stream_index=frame_index,
                timestamp_ms=timestamp_ms,
                camera_id=camera_id,
                frame_id=fid,
                original_width=original_width,
                original_height=original_height,
                sensor_timestamp=sensor_timestamp,
                is_placeholder=is_placeholder,
            )
            self.frames.append(frame_data)
            self.camera_map.append(camera_id)
            self._frame_count += 1
            self._is_paused = False
            
            return True

    def register_camera(self, camera_id: str) -> None:
        """
        Ensure internal batch queue exists for a camera.
        Safe to call multiple times.
        """
        with self._lock:
            if camera_id not in self._batch_queues:
                self._batch_queues[camera_id] = deque()
    
    def add_batch_info(self, camera_id: str, num_frames: int, metadata: Optional[Dict] = None) -> None:
        """
        Record that a logical batch for `camera_id` with `num_frames` frames
        has been completely pushed into this buffer.

        The inference worker will consume these batch descriptors to know exactly
        how many frames to read for the next batch, instead of assuming a fixed size.
        """
        if num_frames <= 0:
            return
        with self._lock:
            if camera_id not in self._batch_queues:
                self._batch_queues[camera_id] = deque()
            self._batch_queues[camera_id].append((num_frames, metadata or {}))
            # Notify any inference thread waiting for a batch for this camera.
            self._batch_available.notify_all()

    def wait_for_next_batch(
        self,
        camera_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[Tuple[int, Optional[Dict]]]:
        """
        Block until a batch descriptor for `camera_id` is available or timeout.

        Returns:
            (num_frames, metadata) if a batch is available, otherwise None on timeout.
        """
        with self._batch_available:
            if camera_id not in self._batch_queues:
                self._batch_queues[camera_id] = deque()
            queue = self._batch_queues[camera_id]
            if not queue:
                if not self._batch_available.wait(timeout=timeout):
                    return None
            if not queue:
                return None
            return queue.popleft()
    
    def wait_for_space(self, timeout: Optional[float] = None) -> bool:
        """
        Wait until buffer has space (not full) (thread-safe).
        
        Args:
            timeout: Maximum time to wait in seconds (None = wait indefinitely)
            
        Returns:
            True if space available, False if timeout
        """
        with self._not_full:
            # Check directly using len(self.frames) since we already have the lock
            # Don't call is_full() as it would try to acquire the lock again (deadlock!)
            while len(self.frames) >= self.max_size:
                if not self._not_full.wait(timeout=timeout):
                    return False
            return True
    
    def get_camera_for_index(self, index: int) -> Optional[str]:
        """Get the camera_id for a frame at a specific buffer index (thread-safe)."""
        with self._lock:
            if 0 <= index < len(self.camera_map):
                return self.camera_map[index]
            return None
    
    def is_paused(self) -> bool:
        """Check if buffer is paused (full)."""
        return self._is_paused
    
    def remove_frames(self, num_frames: int, staging_buffer_1=None, staging_buffer_2=None, camera_1_id: Optional[str] = None, camera_2_id: Optional[str] = None, clear_staging_buffers: bool = False) -> int:
        """
        Remove frames from the front of the buffer (thread-safe).
        Optionally removes corresponding frames from staging buffers if provided and clear_staging_buffers=True.
        
        Args:
            num_frames: Number of frames to remove
            staging_buffer_1: Optional StagingBuffer for camera 1
            staging_buffer_2: Optional StagingBuffer for camera 2
            camera_1_id: Camera ID for staging_buffer_1
            camera_2_id: Camera ID for staging_buffer_2
            clear_staging_buffers: If True, also remove frames from staging buffers (default: False)
            
        Returns:
            Number of frames actually removed
        """
        with self._lock:
            removed = 0
            initial_size = len(self.frames)
            n = min(num_frames, initial_size)
            camera_1_count = 0
            camera_2_count = 0
            first_fd: Optional[FrameData] = None
            last_fd: Optional[FrameData] = None

            for _ in range(n):
                if len(self.frames) == 0:
                    break
                if self.camera_map:
                    camera_id = self.camera_map[0]
                    if camera_1_id and camera_id == camera_1_id:
                        camera_1_count += 1
                    elif camera_2_id and camera_id == camera_2_id:
                        camera_2_count += 1

                fd = self.frames.popleft()
                if self.camera_map:
                    self.camera_map.pop(0)
                removed += 1
                self._frame_count -= 1
                if first_fd is None:
                    first_fd = fd
                last_fd = fd

            cam1_label = camera_1_id or "cam1"
            cam2_label = camera_2_id or "cam2"
            if removed > 0 and first_fd is not None and last_fd is not None:
                f0, f1 = first_fd.frame_id, last_fd.frame_id
                s0, s1 = first_fd.camera_stream_index, last_fd.camera_stream_index
                print(
                    f"[remove_frames] TracknetBuffer removed {removed} "
                    f"(size {initial_size}→{len(self.frames)}; {cam1_label}={camera_1_count} {cam2_label}={camera_2_count}; "
                    f"frame_id={f0}..{f1}; stream_idx={s0}..{s1})"
                )
            elif removed == 0:
                print(f"[remove_frames] TracknetBuffer removed 0 (requested {num_frames}, size was {initial_size})")

            if clear_staging_buffers:
                if staging_buffer_1 and camera_1_count > 0:
                    staging_buffer_1.remove_frames(camera_1_count)
                if staging_buffer_2 and camera_2_count > 0:
                    staging_buffer_2.remove_frames(camera_2_count)

            if len(self.frames) < self.max_size:
                self._is_paused = False
            self._not_full.notify_all()

            return removed
    
    def get_frames_batch(self, num_frames: int) -> List[FrameData]:
        """
        Get a batch of frames from the front of the buffer without removing them (thread-safe).
        
        Args:
            num_frames: Number of frames to get
            
        Returns:
            List of FrameData objects
        """
        with self._lock:
            batch = []
            for i in range(min(num_frames, len(self.frames))):
                batch.append(self.frames[i])
            return batch
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about frames in the buffer by camera (thread-safe)."""
        with self._lock:
            stats = {}
            for camera_id in self.camera_map:
                stats[camera_id] = stats.get(camera_id, 0) + 1
            return stats
    
    def __len__(self) -> int:
        """Get current buffer size (thread-safe)."""
        with self._lock:
            return len(self.frames)

    def clear(self) -> None:
        """Remove all frames and batch descriptors (e.g. force-stop / pipeline teardown)."""
        with self._lock:
            self.frames.clear()
            self.camera_map.clear()
            self._frame_count = 0
            self._is_paused = False
            for q in self._batch_queues.values():
                q.clear()
            self._not_full.notify_all()
            self._batch_available.notify_all()

    def __repr__(self) -> str:
        stats = self.get_statistics()
        stats_str = ", ".join([f"{cam}: {count}" for cam, count in stats.items()])
        return f"TracknetBuffer(frames={len(self)}/{self.max_size}, paused={self._is_paused}, [{stats_str}])"


class VideoReader:
    """
    Reads frames from a video file.
    
    Attributes:
        video_path: Path to the video file
        camera_id: Identifier for the camera
        cap: OpenCV VideoCapture object
    """
    
    def __init__(self, video_path: str, camera_id: str):
        self.video_path = video_path
        self.camera_id = camera_id
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_frame_index = 0
        self._total_frames = 0
        self._fps = 0.0
        self._is_opened = False
    
    def open(self) -> bool:
        """Open the video file for reading."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"❌ Error: Could not open video file {self.video_path}")
            return False
        
        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._is_opened = True
        print(f"✅ Opened video: {self.camera_id} ({self._total_frames:,} frames, {self._fps:.2f} fps)")
        return True
    
    def read_frame(self) -> Optional[Tuple[np.ndarray, int, float]]:
        """
        Read the next frame from the video.
        
        Returns:
            Tuple of (frame, frame_index, timestamp_ms) or None if no more frames
        """
        if not self._is_opened or self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        frame_index = self.current_frame_index
        timestamp_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        self.current_frame_index += 1
        
        return frame, frame_index, timestamp_ms
    
    def seek_to_frame(self, frame_index: int) -> bool:
        """Seek to a specific frame in the video."""
        if not self._is_opened or self.cap is None:
            return False
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self.current_frame_index = frame_index
        return True
    
    def close(self) -> None:
        """Release the video capture object."""
        if self.cap is not None:
            self.cap.release()
            self._is_opened = False
            print(f"🔒 Closed video: {self.camera_id}")
    
    @property
    def total_frames(self) -> int:
        return self._total_frames
    
    @property
    def fps(self) -> float:
        return self._fps
    
    @property
    def is_opened(self) -> bool:
        return self._is_opened
    
    def __repr__(self) -> str:
        return f"VideoReader(camera_id='{self.camera_id}', frame={self.current_frame_index}/{self._total_frames})"


class DualCameraFrameReader:
    """
    Thin wrapper that holds two VideoReader instances for dual-camera video input.

    Used by the pipeline to provide camera 1 and camera 2 video streams and IDs
    to the camera reader worker threads. Frames are read via reader_1/reader_2
    and pushed to staging buffers by the workers.

    Attributes:
        camera_1_id: First camera identifier
        camera_2_id: Second camera identifier
        reader_1: VideoReader for camera 1
        reader_2: VideoReader for camera 2
    """

    def __init__(
        self,
        camera_1_video_path: str,
        camera_2_video_path: str,
        camera_1_id: str,
        camera_2_id: str,
    ):
        self.camera_1_id = camera_1_id
        self.camera_2_id = camera_2_id
        self.reader_1 = VideoReader(camera_1_video_path, camera_1_id)
        self.reader_2 = VideoReader(camera_2_video_path, camera_2_id)
        self._is_initialized = False

    def initialize(self) -> bool:
        """
        Open both video readers.

        Returns:
            True if both videos were opened successfully.
        """
        print(f"\n{'='*60}")
        print("Initializing Dual Camera Frame Reader")
        print(f"{'='*60}")
        success_1 = self.reader_1.open()
        success_2 = self.reader_2.open()
        if success_1 and success_2:
            self._is_initialized = True
            print(f"✅ Camera 1: {self.camera_1_id}, Camera 2: {self.camera_2_id}")
            return True
        return False

    def get_status(self) -> Dict:
        """Return current status (initialized, reader positions)."""
        return {
            'initialized': self._is_initialized,
            'camera_1': {
                'id': self.camera_1_id,
                'current_frame': self.reader_1.current_frame_index,
                'total_frames': self.reader_1.total_frames,
            },
            'camera_2': {
                'id': self.camera_2_id,
                'current_frame': self.reader_2.current_frame_index,
                'total_frames': self.reader_2.total_frames,
            },
        }

    def close(self) -> None:
        """Close both video readers."""
        self.reader_1.close()
        self.reader_2.close()
        self._is_initialized = False

    def __repr__(self) -> str:
        return (
            f"DualCameraFrameReader(camera_1={self.camera_1_id}, "
            f"camera_2={self.camera_2_id}, initialized={self._is_initialized})"
        )


