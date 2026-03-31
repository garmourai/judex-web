"""
Triplet Pipeline Runner for Judex

Orchestrates the three-thread pipeline:
  1. TripletCSVReaderWorker  — reads triplet.csv + .ts segments, fills buffers
  2. InferenceWorker          — runs TensorRT on TracknetBuffers (unchanged)
  3. CorrelationWorker        — triangulates + draws viz from OriginalFrameBuffer

All components are wired here; InferenceWorker and CorrelationWorker are
imported unchanged from the realtime module.
"""

import os
import signal
import threading
import time
import datetime
from typing import Optional

from .inference.frame_reader import TracknetBuffer
from .pipeline_config import PipelineConfig
from .inference.inference import RealtimeInference
from .inference.inference_worker import inference_worker
from .correlation.correlation_worker import correlation_worker_wrapper

from .triplet_csv_reader import OriginalFrameBuffer, TripletCSVReaderWorker


class TripletPipeline:
    """
    Runs the triplet-CSV-based inference pipeline.

    Inputs:
        triplet_csv_path         — path to hls_sync_*_triple.csv (incremental)
        source_segments_dir      — directory containing seg_XXXXX.ts for source cam
        source_segment_csvs_dir  — directory containing segment_XXXXX.csv for source cam
        sink_segments_dir        — same for sink cam
        sink_segment_csvs_dir    — same for sink cam
        config                   — PipelineConfig (provides camera IDs, calibration,
                                   output dirs, model paths)
    """

    def __init__(
        self,
        triplet_csv_path: str,
        source_segments_dir: str,
        source_segment_csvs_dir: str,
        sink_segments_dir: str,
        sink_segment_csvs_dir: str,
        config: PipelineConfig,
        chunk_size: int = 96,
        pacing_seconds: float = 3.2,
        enable_visualization: bool = False,
        enable_stitched_visualization: bool = False,
        profiler=None,
    ):
        self._triplet_csv_path = triplet_csv_path
        self._source_segments_dir = source_segments_dir
        self._source_segment_csvs_dir = source_segment_csvs_dir
        self._sink_segments_dir = sink_segments_dir
        self._sink_segment_csvs_dir = sink_segment_csvs_dir
        self.config = config
        self._chunk_size = chunk_size
        self._pacing_seconds = pacing_seconds
        self._enable_visualization = enable_visualization
        self._enable_stitched_visualization = enable_stitched_visualization
        self._profiler = profiler

        # Shared buffers
        self._tracknet_buffer_1: Optional[TracknetBuffer] = None
        self._tracknet_buffer_2: Optional[TracknetBuffer] = None
        self._original_buffer: Optional[OriginalFrameBuffer] = None

        # Threading events
        self._stop_event = threading.Event()
        self._tracknet_coordinator_done_event = threading.Event()
        self._inference_done_event = threading.Event()

        # Threads
        self._reader_thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None
        self._correlation_thread: Optional[threading.Thread] = None

        # Inference engine
        self._inference: Optional[RealtimeInference] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> bool:
        """
        Start all threads, wait for completion, return True on success.
        Registers a SIGINT handler so Ctrl+C triggers a clean shutdown.
        """
        print(f"[TripletPipeline] {'='*60}")
        print(f"[TripletPipeline] Starting Triplet Pipeline")
        print(f"[TripletPipeline]   Source cam : {self.config.camera_1_id}")
        print(f"[TripletPipeline]   Sink cam   : {self.config.camera_2_id}")
        print(f"[TripletPipeline]   Triplet CSV: {self._triplet_csv_path}")
        print(f"[TripletPipeline]   Chunk size : {self._chunk_size}")
        print(f"[TripletPipeline]   Pacing     : {self._pacing_seconds}s")
        print(f"[TripletPipeline] {'='*60}")

        pipeline_start = time.time()

        # Register SIGINT handler for clean shutdown
        original_sigint = signal.getsignal(signal.SIGINT)
        def _handle_sigint(sig, frame):
            print("\n[TripletPipeline] Ctrl+C received — stopping all threads...")
            self._stop_event.set()
            # Unblock any threads waiting on these events
            self._tracknet_coordinator_done_event.set()
            self._inference_done_event.set()
        signal.signal(signal.SIGINT, _handle_sigint)

        try:
            # Create buffers
            self._tracknet_buffer_1 = TracknetBuffer(max_size=196)
            self._tracknet_buffer_2 = TracknetBuffer(max_size=196)
            self._original_buffer = OriginalFrameBuffer(max_size=196)

            # Create inference engine (TensorRT init happens inside worker thread)
            self._inference = RealtimeInference(
                tracknet_file=self.config.tracknet_file,
                camera_1_id=self.config.camera_1_id,
                camera_2_id=self.config.camera_2_id,
                camera_1_output_dir=self.config.camera_1_output_dir,
                camera_2_output_dir=self.config.camera_2_output_dir,
                batch_size=4,
                seq_len=8,
                profiler=None,
            )

            # Reset events
            self._stop_event.clear()
            self._tracknet_coordinator_done_event.clear()
            self._inference_done_event.clear()

            # Start threads
            self._start_threads()

            # Wait for all threads to exit
            print(f"[TripletPipeline] Waiting for all threads to complete...", flush=True)
            for thread in (
                self._reader_thread,
                self._inference_thread,
                self._correlation_thread,
            ):
                if thread:
                    thread.join()

            print("[TripletPipeline] All worker threads joined.", flush=True)
            if self._profiler:
                fp = getattr(self._profiler, "_filepath", None) or "profiler output"
                print(f"[TripletPipeline] Writing profiler summary ({fp})...", flush=True)
                self._profiler.write_to_file()
                print("[TripletPipeline] Profiler summary written.", flush=True)

            total_duration = time.time() - pipeline_start
            print(f"[TripletPipeline] {'='*60}", flush=True)
            print(f"[TripletPipeline] Pipeline complete — "
                  f"{str(datetime.timedelta(seconds=int(total_duration)))}", flush=True)
            print(f"[TripletPipeline] {'='*60}", flush=True)
            return True

        finally:
            signal.signal(signal.SIGINT, original_sigint)

    # ------------------------------------------------------------------
    # Thread setup
    # ------------------------------------------------------------------

    def _start_threads(self) -> None:
        """Create and start the three worker threads."""

        # 1. TripletCSVReaderWorker
        reader_worker = TripletCSVReaderWorker(
            triplet_csv_path=self._triplet_csv_path,
            source_segments_dir=self._source_segments_dir,
            source_segment_csvs_dir=self._source_segment_csvs_dir,
            sink_segments_dir=self._sink_segments_dir,
            sink_segment_csvs_dir=self._sink_segment_csvs_dir,
            camera_1_id=self.config.camera_1_id,
            camera_2_id=self.config.camera_2_id,
            tracknet_buffer_1=self._tracknet_buffer_1,
            tracknet_buffer_2=self._tracknet_buffer_2,
            original_buffer=self._original_buffer,
            stop_event=self._stop_event,
            tracknet_coordinator_done_event=self._tracknet_coordinator_done_event,
            camera_1_output_dir=self.config.camera_1_output_dir,
            camera_2_output_dir=self.config.camera_2_output_dir,
            chunk_size=self._chunk_size,
            pacing_seconds=self._pacing_seconds,
            profiler=self._profiler,
        )
        self._reader_thread = threading.Thread(
            target=reader_worker.run,
            daemon=False,
            name="TripletCSVReader",
        )

        # 2. InferenceWorker (unchanged from realtime/)
        self._inference_thread = threading.Thread(
            target=inference_worker,
            args=(
                self._tracknet_buffer_1,
                self._tracknet_buffer_2,
                self._inference,
                self._stop_event,
                self._profiler,  # profiler
                self._original_buffer,   # staging_buffer_1 (passed through to process_camera_batch)
                None,                    # staging_buffer_2
                self._inference_done_event,
                self._tracknet_coordinator_done_event,
            ),
            daemon=False,
            name="InferenceWorker",
        )

        # 3. CorrelationWorker
        if self._inference is not None:
            self._correlation_thread = threading.Thread(
                target=correlation_worker_wrapper,
                args=(
                    self._inference.csv_1_path,
                    self._inference.csv_2_path,
                    self.config.camera_1_id,
                    self.config.camera_2_id,
                    self.config.camera_1_object_path,
                    self.config.camera_2_object_path,
                    self.config.camera_1_video_path,
                    self.config.camera_2_video_path,
                    self.config.unique_output_dir,
                    None,   # frame_sync_info_path (unused in realtime flow)
                    self._stop_event,
                    self._profiler,   # profiler
                    0.05,   # check_interval
                    self._original_buffer,   # staging_buffer_1 → OriginalFrameBuffer
                    self._original_buffer,   # staging_buffer_2 → same buffer (both cams stored here)
                    self._enable_visualization,
                    self._enable_stitched_visualization,
                    self._inference_done_event,
                ),
                kwargs={
                    "chunk_size": self._chunk_size,
                    "original_frame_width": self.config.camera_1_original_frame_width,
                    "original_frame_height": self.config.camera_1_original_frame_height,
                },
                daemon=False,
                name="CorrelationWorker",
            )
            # Note: threading.Thread accepts kwargs= as a separate parameter for keyword args

        # Start
        self._reader_thread.start()
        self._inference_thread.start()
        if self._correlation_thread:
            self._correlation_thread.start()

        print(f"[TripletPipeline] All threads started:")
        print(f"[TripletPipeline]   TripletCSVReader  — reads CSV + segments, fills buffers")
        print(f"[TripletPipeline]   InferenceWorker   — TensorRT inference on TracknetBuffers")
        print(f"[TripletPipeline]   CorrelationWorker — triangulation + visualization")


def run_triplet_pipeline(
    triplet_csv_path: str,
    source_segments_dir: str,
    source_segment_csvs_dir: str,
    sink_segments_dir: str,
    sink_segment_csvs_dir: str,
    config: PipelineConfig,
    chunk_size: int = 96,
    pacing_seconds: float = 3.2,
    enable_visualization: bool = False,
    enable_stitched_visualization: bool = False,
    profiler=None,
) -> bool:
    """Convenience wrapper: create and run a TripletPipeline."""
    pipeline = TripletPipeline(
        triplet_csv_path=triplet_csv_path,
        source_segments_dir=source_segments_dir,
        source_segment_csvs_dir=source_segment_csvs_dir,
        sink_segments_dir=sink_segments_dir,
        sink_segment_csvs_dir=sink_segment_csvs_dir,
        config=config,
        chunk_size=chunk_size,
        pacing_seconds=pacing_seconds,
        enable_visualization=enable_visualization,
        enable_stitched_visualization=enable_stitched_visualization,
        profiler=profiler,
    )
    return pipeline.run()
