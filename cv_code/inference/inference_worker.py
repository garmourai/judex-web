import time
import threading
from typing import Optional

import pycuda.driver as cuda

from .frame_reader import TracknetBuffer
from .inference import RealtimeInference


def inference_worker(
    tracknet_buffer_1: TracknetBuffer,
    tracknet_buffer_2: TracknetBuffer,
    inference: RealtimeInference,
    stop_event: threading.Event,
    profiler=None,
    staging_buffer_1=None,
    staging_buffer_2=None,
    inference_done_event: Optional[threading.Event] = None,
    tracknet_coordinator_done_event: Optional[threading.Event] = None,
):
    """
    Worker thread for running inference.
    
    Args:
        buffer: TracknetBuffer to read frames from
        inference: RealtimeInference instance
        stop_event: Event to signal thread to stop
    """
    print("[InferenceWorker] 🚀 Inference worker thread started")
    
    # Initialize CUDA in this thread (CUDA contexts are thread-local)
    cuda_context = None
    print("[InferenceWorker] 🔧 Initializing CUDA context in inference thread...")
    cuda_init_start = time.time()
    # Initialize CUDA driver if not already initialized
    cuda.init()
    # Get the default device (usually 0)
    device = cuda.Device(0)
    # Create a CUDA context for this thread
    cuda_context = device.make_context()
    cuda_init_duration = time.time() - cuda_init_start
    if profiler:
        profiler.record("inference_cuda_context_creation", cuda_init_duration, write_immediately=True, batch=0)
    print(f"[InferenceWorker] ✅ CUDA context created in inference thread (took {cuda_init_duration:.4f}s)")
    
    # Initialize TensorRT engine in this thread (now CUDA context is active)
    print("[InferenceWorker] 🔧 Initializing TensorRT engine in inference thread...")
    tensorrt_init_start = time.time()
    if not inference.initialize():
        print("[InferenceWorker] ❌ Failed to initialize TensorRT engine in inference thread")
        if cuda_context:
            cuda_context.pop()  # Clean up context
        if inference_done_event is not None:
            inference_done_event.set()
        return
    tensorrt_init_duration = time.time() - tensorrt_init_start
    if profiler:
        profiler.record("inference_tensorrt_engine_loading", tensorrt_init_duration, write_immediately=True, batch=0)
    print(f"[InferenceWorker] ✅ TensorRT engine initialized (took {tensorrt_init_duration:.4f}s)")
    
    wait_count = 0
    INITIAL_WAIT_TIMEOUT = 30.0  # Wait up to 30 seconds for initial frames
    initial_wait_start = time.time()
    has_processed_any_frames = False  # Track if we've processed at least one batch
    
    # Track which camera we're expecting next (alternate between camera 1 and camera 2)
    expected_camera = inference.camera_1_id
    
    def _profile_inference_sleep(duration_s: float, reason: str, expected_cam: str):
        """Sleep without profiling (sleep time is now captured in wait_for_batch_time)."""
        time.sleep(duration_s)
    
    while not stop_event.is_set():
        # Select the correct tracknet buffer based on expected camera
        if expected_camera == inference.camera_1_id:
            current_buffer = tracknet_buffer_1
            current_buffer_size = len(tracknet_buffer_1)
        else:
            current_buffer = tracknet_buffer_2
            current_buffer_size = len(tracknet_buffer_2)
        
        # Retrieve the next logical batch descriptor for this camera. This tells us
        # exactly how many frames belong to the next batch, rather than assuming 96.
        should_proceed = False
        frames_available = 0
        batch_meta = None

        # Start timing for complete batch inference (from waiting to processing completion)
        complete_batch_start_time = time.time()
        
        # Start timing for waiting/polling for next batch
        wait_for_batch_start_time = time.time()
        
        # Poll for next batch every 0.1 seconds instead of a single long timeout
        # This allows us to check exit conditions more frequently
        batch_info = None
        POLL_INTERVAL = 0.1  # Check every 0.1 seconds
        while batch_info is None and not stop_event.is_set():
            # Try to get batch with short timeout
            batch_info = current_buffer.wait_for_next_batch(
                expected_camera,
                timeout=POLL_INTERVAL,
            )
            
            if batch_info is not None:
                # Batch is available, break out of polling loop
                break
            
            # No batch yet - check if we should exit while waiting
            # Check if tracknet coordinator is done and buffers are empty
            if tracknet_coordinator_done_event is not None and tracknet_coordinator_done_event.is_set():
                if len(tracknet_buffer_1) == 0 and len(tracknet_buffer_2) == 0:
                    # Coordinator done and buffers empty - exit
                    break
            
            # Continue polling (will sleep POLL_INTERVAL in wait_for_next_batch)
        
        if batch_info is not None:
            frames_in_batch, batch_meta = batch_info
            # Ensure the frames are actually present in the buffer (coordinator promises this).
            frames_available = min(frames_in_batch, len(current_buffer))
            should_proceed = frames_available > 0
        else:
            # No batch available (either timeout or exit condition)
            should_proceed = False
            frames_available = 0

        if should_proceed and frames_available > 0:
            # End timing for waiting/polling for batch (only record when we proceed)
            wait_for_batch_duration = time.time() - wait_for_batch_start_time

            # Print and profile buffer sizes at the start of every batch
            other_buffer = (
                tracknet_buffer_2 if expected_camera == inference.camera_1_id
                else tracknet_buffer_1
            )
            cur_size = len(current_buffer)
            other_size = len(other_buffer)
            orig_size = len(staging_buffer_1) if staging_buffer_1 is not None else -1
            print(
                f"[InferenceWorker] 📥 Buffer status before {expected_camera} batch: "
                f"{cur_size}/{current_buffer.max_size} (current tb), "
                f"{other_size}/{other_buffer.max_size} (other tb), "
                f"orig={orig_size}"
            )
            if profiler:
                _pre_batch_num = getattr(inference, "_profiling_batch_by_camera", {}).get(expected_camera, 0)
                profiler.record(
                    f"inference_tb_current_size_{expected_camera}",
                    float(cur_size),
                    write_immediately=True,
                    batch=_pre_batch_num,
                    thread="InferenceWorker",
                    metadata=f"max={current_buffer.max_size}",
                )
                profiler.record(
                    f"inference_orig_size_{expected_camera}",
                    float(orig_size),
                    write_immediately=True,
                    batch=_pre_batch_num,
                    thread="InferenceWorker",
                )

            print(
                f"[InferenceWorker] ✅ Processing batch of {frames_available} frames from {expected_camera} "
                f"(buffer: {len(current_buffer)}/{current_buffer.max_size}, "
                f"meta={batch_meta})"
            )
            
            # Process camera batch (timing is done inside process_camera_batch)
            success = inference.process_camera_batch(
                current_buffer,
                camera_id=expected_camera,
                num_frames=frames_available,
                staging_buffer_1=staging_buffer_1,
                staging_buffer_2=staging_buffer_2
            )
            
            if success:
                # End timing for complete batch inference (from wait start to processing completion)
                complete_batch_duration = time.time() - complete_batch_start_time
                
                # Profile timing metrics
                if profiler:
                    # _profiling_batch_by_camera is incremented inside process_camera_batch,
                    # so subtract 1 to get the 0-indexed batch number matching the reader
                    batch_num = getattr(inference, "_profiling_batch_by_camera", {}).get(expected_camera, 1) - 1
                    cam = expected_camera  # e.g. "source" or "sink"

                    # Record time spent waiting/polling for batch (per camera)
                    profiler.record(
                        f"inference_wait_for_batch_{cam}",
                        wait_for_batch_duration,
                        write_immediately=True,
                        batch=batch_num,
                        thread="InferenceWorker",
                    )

                    # Record active processing time only (complete - wait)
                    active_duration = complete_batch_duration - wait_for_batch_duration
                    profiler.record(
                        f"inference_active_process_{cam}",
                        active_duration,
                        write_immediately=True,
                        batch=batch_num,
                        thread="InferenceWorker",
                    )

                    # Record complete batch time (wait + process) per camera
                    profiler.record(
                        f"inference_complete_batch_{cam}",
                        complete_batch_duration,
                        write_immediately=True,
                        batch=batch_num,
                        thread="InferenceWorker",
                    )

                    # Profile number of frames processed in this batch
                    profiler.record(
                        f"inference_frames_{cam}",
                        frames_available,
                        write_immediately=True,
                        batch=batch_num,
                        thread="InferenceWorker",
                    )
                
                # Mark that we've processed at least one batch
                has_processed_any_frames = True
                
                # Print buffer status after processing
                final_size = len(current_buffer)
                print(f"[InferenceWorker] 📊 Buffer status after {expected_camera} batch: {final_size}/{current_buffer.max_size} frames remaining")
                wait_count = 0  # Reset counter on success
                
                # Switch to the other camera for next iteration
                if expected_camera == inference.camera_1_id:
                    expected_camera = inference.camera_2_id
                else:
                    expected_camera = inference.camera_1_id
                
                # Mark end of successful batch (used for gap measurement)
                last_success_end_time = time.time()
            else:
                print("[InferenceWorker] ⚠️  Camera batch processing failed, retrying...")
                _profile_inference_sleep(0.1, reason="process_camera_batch_failed", expected_cam=expected_camera)  # Small delay if processing failed
        else:
            wait_count += 1
            # Check if we should continue (video might be ending)
            # Select the correct buffer to check size
            if expected_camera == inference.camera_1_id:
                current_buffer_size = len(tracknet_buffer_1)
            else:
                current_buffer_size = len(tracknet_buffer_2)
            time_since_start = time.time() - initial_wait_start
            
            if current_buffer_size == 0:
                # If we've never processed any frames and it's been less than INITIAL_WAIT_TIMEOUT, keep waiting
                if not has_processed_any_frames and time_since_start < INITIAL_WAIT_TIMEOUT:
                    if wait_count % 20 == 0:  # Print every 2 seconds
                        print(f"[InferenceWorker] ⏳ Inference waiting for initial frames... (waited {time_since_start:.1f}s, will wait up to {INITIAL_WAIT_TIMEOUT}s)")
                    _profile_inference_sleep(0.1, reason="waiting_initial_frames", expected_cam=expected_camera)
                    continue
                # If we've processed frames before and buffer is empty, check if tracknet coordinator is done
                elif has_processed_any_frames:
                    # Primary exit: Check if tracknet coordinator is done AND both buffers are empty
                    if tracknet_coordinator_done_event is not None and tracknet_coordinator_done_event.is_set():
                        if len(tracknet_buffer_1) == 0 and len(tracknet_buffer_2) == 0:
                            # TrackNet coordinator done and both buffers empty - all frames processed, safe to exit
                            print("[InferenceWorker] ✅ No more frames to process (tracknet coordinator done and both buffers empty)")
                            break
                        else:
                            # TrackNet coordinator done but buffers still have frames - continue processing
                            continue
                    else:
                        # Buffer empty but tracknet coordinator still active - wait for more frames
                        _profile_inference_sleep(0.1, reason="buffer_empty_waiting_for_coordinator", expected_cam=expected_camera)
                        continue
                else:
                    # Never processed frames and timeout reached
                    print(f"[InferenceWorker] ⚠️  No frames received after {INITIAL_WAIT_TIMEOUT}s, exiting inference worker")
                    break
            else:
                # Buffer has frames, just wait
                _profile_inference_sleep(0.1, reason="buffer_has_frames_but_not_ready", expected_cam=expected_camera)
    
    # Signal completion so correlation can exit cleanly
    if inference_done_event is not None:
        inference_done_event.set()

    # Clean up CUDA context before exiting
    if cuda_context:
        cuda_context.pop()
        print("[InferenceWorker] 🧹 CUDA context cleaned up")
    
    print("[InferenceWorker] 🛑 Inference worker thread stopped")

