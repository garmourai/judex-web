"""
Net Crossing Detector Module

Detects net crossings from filter stats data using the same logic as rally_segmentation.py.
Designed for incremental processing as segments become available.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


# Badminton net Y coordinate (center of court)
NET_Y = 6.7

# Net height in meters (pickleball center height: 34 inches)
NET_HEIGHT = 0.8636

# Minimum frame gap required between two accepted net crossings
MIN_FRAMES_BETWEEN_CROSSINGS = 15

# Maximum frame gap to consider consecutive (skip pair if gap > this)
MAX_FRAME_GAP = 10

# Validation parameters
MIN_HEIGHT_RATIO = 0.8  # Z must be >= 80% of net height
Y_PROXIMITY_THRESHOLD = 2.0  # At least one point within 2m of net
MIN_DISTANCE_RATIO = 0.3  # Crossing distance >= avg * 0.3
MAX_DISTANCE_RATIO = 3.5  # Crossing distance <= avg * 3.5
LOOKBACK_FRAMES = 5  # Frames to look back for distance averaging
MIN_DISTANCES_FOR_AVG = 3  # Minimum distances needed for averaging


@dataclass
class Detection:
    """Simple detection data structure."""
    frame: int
    x: float
    y: float
    z: float


@dataclass
class CrossingResult:
    """Result of a crossing check."""
    frame: int
    is_valid: bool
    rejection_reason: Optional[str] = None
    details: Dict = field(default_factory=dict)


class NetCrossingDetector:
    """
    Detects net crossings incrementally as new segment data arrives.
    
    Maintains state across segments to properly handle:
    - Lookback for distance validation
    - Minimum gap between crossings
    - Crossings that span segment boundaries
    """
    
    def __init__(
        self,
        net_y: float = NET_Y,
        net_height: float = NET_HEIGHT,
        min_frames_between_crossings: int = MIN_FRAMES_BETWEEN_CROSSINGS,
        max_frame_gap: int = MAX_FRAME_GAP,
        min_height_ratio: float = MIN_HEIGHT_RATIO,
        y_proximity_threshold: float = Y_PROXIMITY_THRESHOLD,
        min_distance_ratio: float = MIN_DISTANCE_RATIO,
        max_distance_ratio: float = MAX_DISTANCE_RATIO,
        lookback_frames: int = LOOKBACK_FRAMES,
        verbose: bool = True,
    ):
        self.net_y = net_y
        self.net_height = net_height
        self.min_frames_between_crossings = min_frames_between_crossings
        self.max_frame_gap = max_frame_gap
        self.min_height_ratio = min_height_ratio
        self.y_proximity_threshold = y_proximity_threshold
        self.min_distance_ratio = min_distance_ratio
        self.max_distance_ratio = max_distance_ratio
        self.lookback_frames = lookback_frames
        self.verbose = verbose
        
        # State
        self.last_processed_frame: int = -1
        self.detection_history: List[Detection] = []  # Recent detections for lookback
        self.last_crossing_frame: Optional[int] = None
        self.crossings: List[int] = []
        self.invalid_crossings: List[CrossingResult] = []
        
        # Keep more history than lookback for safety
        self._history_size = lookback_frames + 10
    
    def process_segment(
        self, 
        segment_data: List[Dict],
        segment_name: str = ""
    ) -> List[int]:
        """
        Process new segment data and detect net crossings.
        
        Args:
            segment_data: List of dicts with frame_number_after_sync, picked_x, picked_y, picked_z
            segment_name: Name of segment for logging
            
        Returns:
            List of new crossing frames found in this segment
        """
        if not segment_data:
            return []
        
        # Filter to only new frames (> last_processed_frame)
        new_data = [
            d for d in segment_data 
            if d['frame_number_after_sync'] > self.last_processed_frame
        ]
        
        if not new_data:
            if self.verbose:
                print(f"    ⏭️  No new frames to process (all <= {self.last_processed_frame})")
            return []
        
        # Sort by frame number
        new_data.sort(key=lambda d: d['frame_number_after_sync'])
        
        new_crossings = []
        
        for data in new_data:
            det = Detection(
                frame=data['frame_number_after_sync'],
                x=data['picked_x'],
                y=data['picked_y'],
                z=data['picked_z'],
            )
            
            # Check for crossing with previous detection
            if self.detection_history:
                prev_det = self.detection_history[-1]
                
                # Check frame gap
                frame_gap = det.frame - prev_det.frame
                if frame_gap <= self.max_frame_gap:
                    result = self._check_and_validate_crossing(prev_det, det)
                    
                    if result.is_valid:
                        # Check minimum gap from last accepted crossing
                        if self.last_crossing_frame is not None:
                            gap_from_last = det.frame - self.last_crossing_frame
                            if gap_from_last < self.min_frames_between_crossings:
                                result.is_valid = False
                                result.rejection_reason = f"Too close to last crossing ({gap_from_last} < {self.min_frames_between_crossings} frames)"
                                self.invalid_crossings.append(result)
                                if self.verbose:
                                    print(f"      ⏭️  Frame {det.frame}: {result.rejection_reason}")
                            else:
                                # Accept crossing
                                self.crossings.append(det.frame)
                                self.last_crossing_frame = det.frame
                                new_crossings.append(det.frame)
                                if self.verbose:
                                    print(f"      ✅ Net crossing at frame {det.frame}")
                        else:
                            # First crossing - accept
                            self.crossings.append(det.frame)
                            self.last_crossing_frame = det.frame
                            new_crossings.append(det.frame)
                            if self.verbose:
                                print(f"      ✅ Net crossing at frame {det.frame}")
                    elif result.rejection_reason:
                        # Was a crossing candidate but failed validation
                        self.invalid_crossings.append(result)
                        if self.verbose:
                            print(f"      ❌ Frame {det.frame}: {result.rejection_reason}")
            
            # Add to history
            self.detection_history.append(det)
            
            # Trim history
            if len(self.detection_history) > self._history_size:
                self.detection_history = self.detection_history[-self._history_size:]
            
            # Update last processed frame
            self.last_processed_frame = det.frame
        
        if self.verbose and segment_name:
            if new_crossings:
                print(f"    🎯 Segment {segment_name}: {len(new_crossings)} new crossing(s) at {new_crossings}")
            else:
                print(f"    🎯 Segment {segment_name}: no new crossings")
        
        return new_crossings
    
    def _check_and_validate_crossing(
        self, 
        det1: Detection, 
        det2: Detection
    ) -> CrossingResult:
        """
        Check if a crossing occurred and validate it.
        
        Returns CrossingResult with is_valid=True only if:
        1. Y crossed net_y between det1 and det2
        2. All validation rules pass
        """
        result = CrossingResult(frame=det2.frame, is_valid=False)
        
        # Check if Y crossed net_y
        crossed = (
            (det1.y < self.net_y and det2.y > self.net_y) or 
            (det1.y > self.net_y and det2.y < self.net_y)
        )
        
        if not crossed:
            # Not a crossing - no rejection reason (normal case)
            return result
        
        # It's a crossing candidate - now validate
        result.details['det1_y'] = det1.y
        result.details['det2_y'] = det2.y
        result.details['net_y'] = self.net_y
        
        # Rule 1: Height check (Z >= 80% of net height)
        min_z = self.net_height * self.min_height_ratio
        if det2.z < min_z:
            result.rejection_reason = f"Height too low (z={det2.z:.3f} < {min_z:.3f})"
            result.details['z'] = det2.z
            result.details['min_z'] = min_z
            return result
        
        # Rule 2: Y-proximity check (at least one point within threshold of net)
        y_dist1 = abs(det1.y - self.net_y)
        y_dist2 = abs(det2.y - self.net_y)
        min_y_dist = min(y_dist1, y_dist2)
        
        if y_dist1 > self.y_proximity_threshold and y_dist2 > self.y_proximity_threshold:
            result.rejection_reason = f"Too far from net (|y1-net|={y_dist1:.3f}, |y2-net|={y_dist2:.3f}, both > {self.y_proximity_threshold:.3f})"
            result.details['y_dist1'] = y_dist1
            result.details['y_dist2'] = y_dist2
            return result
        
        # Rule 3: Distance consistency check
        if len(self.detection_history) >= self.lookback_frames:
            # Calculate step distances over lookback window
            distances = []
            history = self.detection_history[-self.lookback_frames:]
            
            for i in range(len(history) - 1):
                d1, d2 = history[i], history[i + 1]
                dist = np.sqrt(
                    (d2.x - d1.x)**2 + 
                    (d2.y - d1.y)**2 + 
                    (d2.z - d1.z)**2
                )
                distances.append(dist)
            
            if distances:
                # Calculate crossing distance
                crossing_dist = np.sqrt(
                    (det2.x - det1.x)**2 + 
                    (det2.y - det1.y)**2 + 
                    (det2.z - det1.z)**2
                )
                
                # Establish thresholds
                if len(distances) >= MIN_DISTANCES_FOR_AVG:
                    avg_dist = np.mean(distances)
                    min_valid = avg_dist * self.min_distance_ratio
                    max_valid = avg_dist * self.max_distance_ratio
                else:
                    min_obs = float(np.min(distances))
                    max_obs = float(np.max(distances))
                    min_valid = min_obs * self.min_distance_ratio
                    max_valid = max_obs * self.max_distance_ratio
                
                if not (min_valid <= crossing_dist <= max_valid):
                    result.rejection_reason = f"Distance inconsistent (d={crossing_dist:.3f} not in [{min_valid:.3f}, {max_valid:.3f}])"
                    result.details['crossing_dist'] = crossing_dist
                    result.details['min_valid'] = min_valid
                    result.details['max_valid'] = max_valid
                    return result
        
        # All checks passed
        result.is_valid = True
        return result
    
    def get_all_crossings(self) -> List[int]:
        """Return all detected crossings."""
        return self.crossings.copy()
    
    def get_invalid_crossings(self) -> List[CrossingResult]:
        """Return all invalid crossing attempts for debugging."""
        return self.invalid_crossings.copy()
    
    def get_summary(self) -> Dict:
        """Return summary of detection state."""
        return {
            'total_crossings': len(self.crossings),
            'total_invalid': len(self.invalid_crossings),
            'last_processed_frame': self.last_processed_frame,
            'last_crossing_frame': self.last_crossing_frame,
            'history_size': len(self.detection_history),
        }
