"""
Trajectory Handoff Context Module

This module defines data structures for passing trajectory state between
real-time segments to enable trajectory continuity across segment boundaries.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import json
import numpy as np


@dataclass
class DetectionState:
    """Serializable state of a Detection object."""
    x: float
    y: float
    z: float
    frame_number: int
    in_motion: bool = False
    can_pick: int = 1
    
    def to_dict(self) -> Dict:
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'frame_number': self.frame_number,
            'in_motion': self.in_motion,
            'can_pick': self.can_pick
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DetectionState':
        return cls(
            x=data['x'],
            y=data['y'],
            z=data['z'],
            frame_number=data['frame_number'],
            in_motion=data.get('in_motion', False),
            can_pick=data.get('can_pick', 1)
        )


@dataclass
class TrajectoryState:
    """Serializable state of a Trajectory object."""
    detections: Dict[int, DetectionState] = field(default_factory=dict)
    has_motion: bool = False
    filled_frames: Dict[int, int] = field(default_factory=dict)
    net_hit_frames: Dict[int, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'detections': {k: v.to_dict() for k, v in self.detections.items()},
            'has_motion': self.has_motion,
            'filled_frames': self.filled_frames,
            'net_hit_frames': self.net_hit_frames
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TrajectoryState':
        detections = {
            int(k): DetectionState.from_dict(v) 
            for k, v in data.get('detections', {}).items()
        }
        return cls(
            detections=detections,
            has_motion=data.get('has_motion', False),
            filled_frames={int(k): v for k, v in data.get('filled_frames', {}).items()},
            net_hit_frames={int(k): v for k, v in data.get('net_hit_frames', {}).items()}
        )


@dataclass
class QueueState:
    """Serializable state of a DetectionQueue object."""
    detections: List[Tuple[Tuple[float, float, float], int]] = field(default_factory=list)
    push_to_trajectory: bool = False
    trajectory_index: Optional[int] = None
    has_overlap: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'detections': [
                {'coords': list(coords), 'frame': frame}
                for coords, frame in self.detections
            ],
            'push_to_trajectory': self.push_to_trajectory,
            'trajectory_index': self.trajectory_index,
            'has_overlap': self.has_overlap
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'QueueState':
        detections = [
            (tuple(d['coords']), d['frame'])
            for d in data.get('detections', [])
        ]
        return cls(
            detections=detections,
            push_to_trajectory=data.get('push_to_trajectory', False),
            trajectory_index=data.get('trajectory_index'),
            has_overlap=data.get('has_overlap', False)
        )


@dataclass
class TrajectoryHandoffContext:
    """
    Context to pass between segments for trajectory continuity.
    
    This contains all the state needed to continue building trajectories
    from one segment to the next without losing active trajectory information.
    
    Attributes:
        queues: List of active queue states (trajectory hypotheses being built)
        active_trajectories: Dictionary of trajectories that haven't been stored yet
        trajectory_count: Counter for trajectory IDs (for continuity)
        latest_frame_number: Last frame processed (for frame gap calculations)
        segment_end_frame: The frame number where this context was captured
    """
    queues: List[QueueState] = field(default_factory=list)
    active_trajectories: Dict[int, TrajectoryState] = field(default_factory=dict)
    trajectory_count: int = 0
    latest_frame_number: int = -1
    segment_end_frame: int = 0
    
    def to_dict(self) -> Dict:
        """Serialize context to dictionary."""
        return {
            'queues': [q.to_dict() for q in self.queues],
            'active_trajectories': {
                str(k): v.to_dict() for k, v in self.active_trajectories.items()
            },
            'trajectory_count': self.trajectory_count,
            'latest_frame_number': self.latest_frame_number,
            'segment_end_frame': self.segment_end_frame
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TrajectoryHandoffContext':
        """Deserialize context from dictionary."""
        queues = [QueueState.from_dict(q) for q in data.get('queues', [])]
        active_trajectories = {
            int(k): TrajectoryState.from_dict(v)
            for k, v in data.get('active_trajectories', {}).items()
        }
        return cls(
            queues=queues,
            active_trajectories=active_trajectories,
            trajectory_count=data.get('trajectory_count', 0),
            latest_frame_number=data.get('latest_frame_number', -1),
            segment_end_frame=data.get('segment_end_frame', 0)
        )
    
    def to_json(self) -> str:
        """Serialize context to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TrajectoryHandoffContext':
        """Deserialize context from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def save_to_file(self, filepath: str):
        """Save context to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'TrajectoryHandoffContext':
        """Load context from a JSON file."""
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))
    
    def is_empty(self) -> bool:
        """Check if context has no active state."""
        return len(self.queues) == 0 and len(self.active_trajectories) == 0
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the context."""
        return (
            f"TrajectoryHandoffContext("
            f"queues={len(self.queues)}, "
            f"active_trajectories={len(self.active_trajectories)}, "
            f"trajectory_count={self.trajectory_count}, "
            f"latest_frame={self.latest_frame_number}, "
            f"segment_end={self.segment_end_frame})"
        )
