"""
Time Profiler for Realtime Pipeline

This module provides thread-safe time profiling for tracking
performance metrics across multiple threads.
"""

import time
import threading
import os
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime


class TimeProfiler:
    """
    Thread-safe time profiler for tracking operation durations.
    """
    
    def __init__(self, filepath: Optional[str] = None):
        self._lock = threading.Lock()
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._start_times: Dict[str, float] = {}
        self._counts: Dict[str, int] = defaultdict(int)
        self._total_times: Dict[str, float] = defaultdict(float)
        self._filepath = filepath
        self._file_initialized = False
        self._batch_counter = 0
    
    def start(self, operation: str):
        """Start timing an operation."""
        with self._lock:
            key = f"start_{operation}_{self._counts[operation]}"
            self._start_times[operation] = time.time()
    
    def end(self, operation: str):
        """End timing an operation and record the duration."""
        with self._lock:
            if operation in self._start_times:
                duration = time.time() - self._start_times[operation]
                self._timings[operation].append(duration)
                self._total_times[operation] += duration
                self._counts[operation] += 1
                del self._start_times[operation]
    
    def record(
        self,
        operation: str,
        duration: float,
        write_immediately: bool = False,
        metadata: Optional[str] = None,
        batch: Optional[int] = None,
        thread: Optional[str] = None,
    ):
        """Record a duration directly (for operations timed externally).
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
            write_immediately: Whether to write to file immediately
            metadata: Optional metadata string to include in the log entry
            batch: Optional batch number to log (caller-provided). If None, uses current internal batch counter.
            thread: Optional thread name to log. If None, uses current thread name.
        """
        with self._lock:
            self._timings[operation].append(duration)
            self._total_times[operation] += duration
            self._counts[operation] += 1
        
        # Write immediately if requested and filepath is set (outside lock to avoid deadlock)
        if write_immediately and self._filepath:
            self._write_batch_entry(operation, duration, metadata=metadata, batch=batch, thread=thread)
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        with self._lock:
            if operation not in self._timings or len(self._timings[operation]) == 0:
                return {
                    'count': 0,
                    'total': 0.0,
                    'mean': 0.0,
                    'min': 0.0,
                    'max': 0.0
                }
            
            timings = self._timings[operation]
            return {
                'count': len(timings),
                'total': sum(timings),
                'mean': sum(timings) / len(timings),
                'min': min(timings),
                'max': max(timings)
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        with self._lock:
            return {
                op: self._stats_for_operation_unlocked(list(self._timings[op]))
                for op in self._timings.keys()
            }
    
    def _ensure_file_initialized(self):
        """Initialize the profiling file with headers if not already done."""
        if self._file_initialized or not self._filepath:
            return
        
        try:
            # Create directory if it doesn't exist
            dir_path = os.path.dirname(self._filepath)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            with open(self._filepath, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("REALTIME PIPELINE TIME PROFILING RESULTS\n")
                f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                f.write("BATCH-BY-BATCH TIMINGS\n")
                f.write("(Correlation batches also get a CORRELATION SEGMENT BREAKDOWN block with summed parts.)\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Batch':<8} {'Thread':<18} {'Operation':<35} {'Duration (s)':<15} {'Timestamp':<20}\n")
                f.write("-" * 80 + "\n")
                f.flush()
                os.fsync(f.fileno())
            
            self._file_initialized = True
            print(f"✅ Profiling file initialized: {self._filepath}")
        except Exception as e:
            print(f"❌ Failed to initialize profiling file {self._filepath}: {e}")
            import traceback
            traceback.print_exc()
    
    def _write_batch_entry(
        self,
        operation: str,
        duration: float,
        metadata: Optional[str] = None,
        batch: Optional[int] = None,
        thread: Optional[str] = None,
    ):
        """Write a single batch entry to the file.
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
            metadata: Optional metadata string to append to the entry
            batch: Optional batch number override
            thread: Optional thread name override
        """
        if not self._filepath:
            return
        
        try:
            # Initialize file if needed (this acquires lock internally)
            self._ensure_file_initialized()
            
            # Resolve batch/thread (need lock for batch counter)
            if batch is None:
                with self._lock:
                    batch_num = self._batch_counter
            else:
                batch_num = batch

            thread_name = thread or threading.current_thread().name
            
            # Write to file (outside lock)
            with open(self._filepath, 'a') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if metadata:
                    f.write(f"{batch_num:<8} {thread_name:<18} {operation:<35} {duration:<15.4f} {timestamp:<20} [{metadata}]\n")
                else:
                    f.write(f"{batch_num:<8} {thread_name:<18} {operation:<35} {duration:<15.4f} {timestamp:<20}\n")
                f.flush()  # Ensure immediate write
                try:
                    os.fsync(f.fileno())  # Force write to disk
                except:
                    pass  # Some file systems don't support fsync
        except Exception as e:
            print(f"⚠️  Warning: Failed to write profiling entry to {self._filepath}: {e}")
            import traceback
            traceback.print_exc()

    def write_correlation_segment_breakdown(
        self,
        batch: int,
        seg_meta: str,
        t_wait_ready: float,
        t_pairwise: float,
        t_trajectory: float,
        t_viz_overlay: float,
        t_viz_stitched: float,
        t_accounted: float,
        t_overhead: float,
        t_wall: float,
        t_net_crossings: float = 0.0,
    ) -> None:
        """
        Append a readable block to the profiling file matching console segment timing.
        wait = idle before this segment; pairwise + trajectory + viz + viz_stitched
        + net_crossings = accounted; + overhead ≈ wall.
        """
        if not self._filepath:
            return
        try:
            self._ensure_file_initialized()
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            chk = t_accounted + t_overhead
            lines = [
                "\n",
                "=" * 80 + "\n",
                f"CORRELATION SEGMENT BREAKDOWN  batch={batch}  {ts}\n",
                f"  {seg_meta}\n",
                "-" * 80 + "\n",
                f"  wait before segment      (correlation_wait_for_ready_s)      {t_wait_ready:10.4f} s\n",
                f"  pairwise+triangulation   (correlation_segment_pairwise_s)    {t_pairwise:10.4f} s\n",
                f"  trajectory               (correlation_segment_trajectory_s)    {t_trajectory:10.4f} s\n",
                f"  viz cam1 overlay         (correlation_segment_viz_overlay_s)   {t_viz_overlay:10.4f} s\n",
                f"  viz stitched             (correlation_segment_viz_stitched_s)  {t_viz_stitched:10.4f} s\n",
                f"  net crossings (live)     (correlation_net_crossings_live_s)    {t_net_crossings:10.4f} s\n",
                "-" * 80 + "\n",
                f"  accounted sum            (correlation_segment_accounted_s)     {t_accounted:10.4f} s\n",
                f"  overhead                 (correlation_segment_overhead_s)      {t_overhead:10.4f} s\n",
                f"  wall (full segment)      (correlation_complete_segment_time) {t_wall:10.4f} s\n",
                f"  check (accounted+overhead)                                      {chk:10.4f} s\n",
                "=" * 80 + "\n",
            ]
            with open(self._filepath, "a") as f:
                f.writelines(lines)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
        except Exception as e:
            print(f"⚠️  Warning: Failed to write correlation segment breakdown: {e}")
    
    def record_batch_complete(self, batch_type: str = "inference"):
        """Record that a batch is complete and increment batch counter."""
        with self._lock:
            self._batch_counter += 1
            batch_num = self._batch_counter
        
        # Write outside lock to avoid deadlock
        if self._filepath:
            try:
                self._ensure_file_initialized()
                with open(self._filepath, 'a') as f:
                    f.write(f"\n--- Batch {batch_num} ({batch_type}) completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk
            except Exception as e:
                print(f"⚠️  Warning: Failed to write batch completion: {e}")
    
    def _stats_for_operation_unlocked(self, timings: List[float]) -> Dict[str, float]:
        if not timings:
            return {
                "count": 0,
                "total": 0.0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        return {
            "count": len(timings),
            "total": sum(timings),
            "mean": sum(timings) / len(timings),
            "min": min(timings),
            "max": max(timings),
        }

    def write_to_file(self, filepath: Optional[str] = None):
        """Append a compact final summary to the profiling file (fast; avoids huge DETAILED dumps)."""
        target_file = filepath or self._filepath
        if not target_file:
            return

        detailed = os.environ.get("JUDEX_PROFILER_DETAILED", "").strip() in ("1", "true", "yes")

        # Snapshot under one lock acquisition (do not call get_all_stats — it would re-acquire the same Lock).
        with self._lock:
            file_initialized = self._file_initialized
            timings_snapshot = {k: list(v) for k, v in self._timings.items()}
            all_stats = {
                op: self._stats_for_operation_unlocked(v)
                for op, v in self._timings.items()
            }

        mode = 'a' if file_initialized else 'w'
        with open(target_file, mode) as f:
            if not file_initialized:
                f.write("=" * 80 + "\n")
                f.write("REALTIME PIPELINE TIME PROFILING RESULTS\n")
                f.write("=" * 80 + "\n\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("FINAL SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write("STATISTICS BY OPERATION\n")
            f.write("-" * 80 + "\n")
            for operation, stats in sorted(all_stats.items()):
                if stats['count'] > 0:
                    f.write(f"{operation}:\n")
                    f.write(f"  Count: {stats['count']}\n")
                    f.write(f"  Total Time: {stats['total']:.4f} seconds\n")
                    f.write(f"  Mean Time: {stats['mean']:.4f} seconds\n")
                    f.write(f"  Min Time: {stats['min']:.4f} seconds\n")
                    f.write(f"  Max Time: {stats['max']:.4f} seconds\n")
                    f.write("\n")

            corr_ops = [
                "correlation_wait_for_ready_s",
                "correlation_segment_pairwise_s",
                "correlation_segment_trajectory_s",
                "correlation_segment_viz_overlay_s",
                "correlation_segment_viz_stitched_s",
                "correlation_segment_accounted_s",
                "correlation_segment_overhead_s",
                "correlation_complete_segment_time",
            ]
            f.write("CORRELATION SEGMENT TIMING (totals across all correlation batches)\n")
            f.write("-" * 80 + "\n")
            f.write(
                "correlation_wait_for_ready_s = idle (poll/sleep) after previous segment until this one starts. "
                "accounted = pairwise+trajectory+viz_overlay+viz_stitched; wall = segment work only.\n"
            )
            for op in corr_ops:
                st = all_stats.get(op)
                if st and st["count"] > 0:
                    f.write(
                        f"  {op:<40} count={st['count']:<6} "
                        f"total={st['total']:.4f}s  mean={st['mean']:.4f}s\n"
                    )
            f.write("-" * 80 + "\n\n")

            f.write("=" * 80 + "\n")
            f.write(f"Profiling completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")

            if detailed:
                f.write("\n" + "=" * 80 + "\n")
                f.write("DETAILED TIMINGS (JUDEX_PROFILER_DETAILED=1)\n")
                f.write("=" * 80 + "\n\n")
                for operation, timings in sorted(timings_snapshot.items()):
                    if len(timings) > 0:
                        f.write(f"{operation}:\n")
                        f.write(f"  Total operations: {len(timings)}\n")
                        f.write(f"  Individual timings (seconds):\n")
                        for i, timing in enumerate(timings, 1):
                            f.write(f"    Operation {i}: {timing:.4f}\n")
                        f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("END OF PROFILING RESULTS\n")
            f.write("=" * 80 + "\n")

