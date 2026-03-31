from .constants import *
import numpy as np

class Detection:
    def __init__(self, x, y, z, frame_number, in_motion=False, can_pick=1):
        self.x = x
        self.y = y
        self.z = z
        self.frame_number = frame_number
        self.in_motion = in_motion
        self.can_pick = can_pick
        
class Trajectory:
    def __init__(self):
        self.detections = {}
        self.has_motion = False
        self.filled_frames = {}
        self.net_hit_frames = {}

    def check_motion(self, detection):
        """Check if detection indicates motion."""
        if len(self.detections) > 0:
            last_key = list(self.detections.keys())[-1]
            last_detection = self.detections[last_key]

            distance = sum(
                (a - b) ** 2 for a, b in zip(detection, (last_detection.x, last_detection.y, last_detection.z))
            ) ** 0.5

            if MOTION_THRESHOLDS[0] < distance < MOTION_THRESHOLDS[1]:
                self.has_motion = True
                return True
        return False
    
    def add_detection(self, detection, frame_number):
        """Add a detection to the trajectory."""
        in_motion = False
        if len(self.detections) > 0:
            last_detection = self.detections[list(self.detections.keys())[-1]]
            if frame_number - last_detection.frame_number == 1:
                in_motion = self.check_motion(detection)
            else:
                in_motion = last_detection.in_motion
        
        detection_obj = Detection(detection[0], detection[1], detection[2], frame_number, in_motion)
        self.detections[frame_number] = detection_obj
    
class DetectionQueue:
    def __init__(self, frame_width: int, frame_height: int):
        self.detections = []
        self.push_to_trajectory = False
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.has_overlap = False
        self.trajectory_index = None

    def push(self, detection, trajectory):
        """Add detection to queue and validate."""
        self.detections.append(detection)
        self.push_to_trajectory = self.validate(trajectory)

    def clear(self, trajectory):
        """Clear queue by adding detections to trajectory."""
        for detection in self.detections:
            if (self.has_overlap and 
                len(list(trajectory.detections.keys())) != 0 and 
                detection[1] <= sorted(list(trajectory.detections.keys()))[-1]):
                continue
            trajectory.add_detection(detection[0], detection[1])

    def validate(self, trajectory):
        """Validate if queue should be pushed to trajectory."""
        if len(self.detections) == 0:
            return False
        return True
