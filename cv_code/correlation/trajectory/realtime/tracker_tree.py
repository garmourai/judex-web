"""
Realtime-specific trajectory tree module.
"""
import numpy as np
from .models import *
from .handoff_context import (
    TrajectoryHandoffContext,
    TrajectoryState,
    DetectionState,
    QueueState
)
from .tracking_params import *


class DetectionTree:
    def __init__(self, start_frame, end_frame, frame_width: int, frame_height: int):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.trajectories = {}
        self.stored_trajectories = []
        self.filled_frames = {}
        self.net_hit_frames = {}
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.trajectory_count = 0
        self.queues = []
        self.latest_frame_number = -1

    def check_null(self, coordinates):
        return coordinates[0] == 0 and coordinates[1] == 0 and coordinates[2] == 0

    def min_distance_per_frame(self, detections, coordinates, frame_number):
        min_distance_per_frame = 1e6
        distance = 0
        cur_coordinates = coordinates
        for idx in range(len(detections) - 1, -1, -1):
            prev_coordinates = detections[idx][0]
            distance += sum((p - c) ** 2 for p, c in zip(prev_coordinates, cur_coordinates)) ** 0.5
            frame_gap = frame_number - detections[idx][1]
            if frame_gap == 0:
                continue
            if frame_gap >= FRAME_GAP_THRESHOLD or distance / frame_gap > DISTANCE_PER_FRAME_THRESHOLD:
                break
            if distance / frame_gap < min_distance_per_frame:
                min_distance_per_frame = distance / frame_gap
            cur_coordinates = prev_coordinates
        return min_distance_per_frame

    def euclidean_distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def get_latest_frame_in_queue(self, queue):
        return max(det[1] for det in queue.detections)

    def perpendicular_distance_to_ray(self, p1, p2, p3, point):
        a = np.array(p1)
        b = np.array(p3)
        point = np.array(point)
        direction = b - a
        if np.linalg.norm(direction) == 0:
            return None
        projection_length = np.dot(point - a, direction) / np.linalg.norm(direction)
        if projection_length < 0:
            return None
        projection = a + (projection_length / np.linalg.norm(direction)) * direction
        return np.linalg.norm(point - projection)

    def mean_distance_among_last_three(self, queue):
        if len(queue.detections) < 2:
            return 0
        range_len = min(3, len(queue.detections))
        pts = [np.array(queue.detections[-i - 1][0]) for i in range(range_len)]
        if range_len == 2:
            return np.linalg.norm(pts[0] - pts[1])
        d1 = np.linalg.norm(pts[0] - pts[1])
        d2 = np.linalg.norm(pts[1] - pts[2])
        return (d1 + d2) / 2

    def enqueue_detections(self, points, frame_number):
        assigned = set()
        sorted_queues = sorted(self.queues, key=lambda q: len(q.detections), reverse=True)
        for i, coordinates in enumerate(points):
            candidate_queues = []
            for queue in sorted_queues:
                if len(queue.detections) < 3:
                    continue
                last_coords = queue.detections[-1][0]
                dist = self.euclidean_distance(last_coords, coordinates)
                if dist < VERY_CLOSE_THRESHOLD:
                    candidate_queues.append((queue, dist, self.get_latest_frame_in_queue(queue)))
                elif dist < CLOSE_THRESHOLD:
                    detections = [detection[0] for detection in queue.detections]
                    if len(detections) >= 3:
                        p1, p2, p3 = detections[-3:]
                    elif len(detections) == 2:
                        p1, p2 = detections[-2:]
                        p3 = p2
                    else:
                        continue
                    perp_dist = self.perpendicular_distance_to_ray(p1, p2, p3, coordinates)
                    if perp_dist is not None and perp_dist < PERPENDICULAR_THRESHOLD:
                        candidate_queues.append((queue, perp_dist, self.get_latest_frame_in_queue(queue)))
                elif dist < FAR_THRESHOLD:
                    mean_dist = self.mean_distance_among_last_three(queue)
                    if self.euclidean_distance(last_coords, coordinates) < FAR_DISTANCE_MULTIPLIER * mean_dist:
                        candidate_queues.append((queue, dist, self.get_latest_frame_in_queue(queue)))

            if candidate_queues:
                selected = max(candidate_queues, key=lambda x: x[2])[0]
                if selected.trajectory_index is None:
                    selected.push((coordinates, frame_number), None)
                else:
                    selected.push((coordinates, frame_number), self.trajectories[selected.trajectory_index])
                assigned.add(i)
                continue

            min_dist = 1e6
            selected_queues = []
            for queue in sorted_queues:
                if len(queue.detections) >= 3:
                    continue
                last_frame = queue.detections[-1][1] if queue.detections else None
                if last_frame is None or frame_number != last_frame + 1:
                    continue
                dist_per_frame = self.min_distance_per_frame(queue.detections, coordinates, frame_number)
                if dist_per_frame < min_dist:
                    min_dist = dist_per_frame
                selected_queues.append((queue, dist_per_frame))

            close_queues = [q for q, d in selected_queues if abs(d - min_dist) <= 0.5]
            selected_queue = max(
                close_queues, key=lambda q: (len(q.detections), max(d[1] for d in q.detections))
            ) if close_queues else None

            if selected_queue and min_dist < 1e6:
                if selected_queue.trajectory_index is None:
                    selected_queue.push((coordinates, frame_number), None)
                else:
                    selected_queue.push((coordinates, frame_number), self.trajectories[selected_queue.trajectory_index])
                assigned.add(i)

        for i, coordinates in enumerate(points):
            if i not in assigned:
                queue = DetectionQueue(self.frame_width, self.frame_height)
                queue.push((coordinates, frame_number), None)
                self.queues.append(queue)
                # print(
                #     f"[CorrelationWorker]   [Tree] New queue created at frame={frame_number} "
                #     f"point=({coordinates[0]:.4f},{coordinates[1]:.4f},{coordinates[2]:.4f})"
                # )

    def process_queues(self, frame_number):
        if len(self.queues) == 0:
            return
        queue_idxs_to_remove = []
        for queue_idx, queue in enumerate(self.queues):
            latest_frame_number = queue.detections[-1][1]
            if frame_number - latest_frame_number > FRAME_GAP_THRESHOLD:
                if queue.push_to_trajectory:
                    if queue.trajectory_index is None:
                        queue.trajectory_index = self.trajectory_count
                        self.trajectories[queue.trajectory_index] = Trajectory()
                        # print(
                        #     f"[CorrelationWorker]   [Tree] Created trajectory id={queue.trajectory_index} "
                        #     f"from queue_len={len(queue.detections)} at frame={frame_number}"
                        # )
                        self.trajectory_count += 1
                    if self.trajectories[queue.trajectory_index] is None:
                        self.trajectories[queue.trajectory_index] = Trajectory()
                    queue.clear(self.trajectories[queue.trajectory_index])
                    if len(self.trajectories[queue.trajectory_index].detections) == 0:
                        self.trajectories[queue.trajectory_index] = None
                queue_idxs_to_remove.append(queue_idx)
        self.queues = [queue for idx, queue in enumerate(self.queues) if idx not in queue_idxs_to_remove]

    def prune_trajectory(self, frame_number):
        for key, value in self.trajectories.items():
            var_trajectory = value
            if var_trajectory is not None:
                var_trajectory_latest_frame = max(var_trajectory.detections.keys())
                if frame_number - var_trajectory_latest_frame > FRAME_DIFF_THRESHOLD or frame_number == self.end_frame:
                    for i in list(var_trajectory.filled_frames.keys()):
                        self.filled_frames[i] = 1
                    for i in list(var_trajectory.net_hit_frames.keys()):
                        self.net_hit_frames[i] = 1
                    first_f = min(var_trajectory.detections.keys())
                    last_f = max(var_trajectory.detections.keys())
                    # print(
                    #     f"[CorrelationWorker]   [Tree] Stored trajectory id={key} "
                    #     f"frames={first_f}->{last_f} points={len(var_trajectory.detections)} "
                    #     f"reason={'end_frame' if frame_number == self.end_frame else 'frame_diff'}"
                    # )
                    self.stored_trajectories.append(var_trajectory)
                    self.trajectories[key] = None

    def add_detections(self, points, frame_number):
        if len(points) != 0 and not (len(points) == 1 and self.check_null(points[0])):
            self.enqueue_detections(points, frame_number)
            self.latest_frame_number = frame_number
        self.process_queues(frame_number)
        self.prune_trajectory(frame_number)

    def get_stored_trajectories(self):
        return self.stored_trajectories

    def get_handoff_context(self, frame_gap_threshold: int = 10) -> TrajectoryHandoffContext:
        context = TrajectoryHandoffContext(
            trajectory_count=self.trajectory_count,
            latest_frame_number=self.latest_frame_number,
            segment_end_frame=self.end_frame
        )
        for queue in self.queues:
            queue_state = QueueState(
                detections=list(queue.detections),
                push_to_trajectory=queue.push_to_trajectory,
                trajectory_index=queue.trajectory_index,
                has_overlap=queue.has_overlap
            )
            context.queues.append(queue_state)
        for traj_id, traj in self.trajectories.items():
            if traj is None or len(traj.detections) == 0:
                continue
            latest_frame = max(traj.detections.keys())
            frame_gap = self.latest_frame_number - latest_frame
            if frame_gap <= frame_gap_threshold:
                detection_states = {}
                for frame_num, det in traj.detections.items():
                    detection_states[frame_num] = DetectionState(
                        x=det.x, y=det.y, z=det.z, frame_number=det.frame_number, in_motion=det.in_motion, can_pick=det.can_pick
                    )
                traj_state = TrajectoryState(
                    detections=detection_states,
                    has_motion=traj.has_motion,
                    filled_frames=dict(traj.filled_frames),
                    net_hit_frames=dict(traj.net_hit_frames)
                )
                context.active_trajectories[traj_id] = traj_state
        return context

    def restore_from_context(self, context: TrajectoryHandoffContext):
        if context is None or context.is_empty():
            return
        self.trajectory_count = context.trajectory_count
        self.latest_frame_number = context.latest_frame_number
        for traj_id, traj_state in context.active_trajectories.items():
            traj = Trajectory()
            traj.has_motion = traj_state.has_motion
            traj.filled_frames = dict(traj_state.filled_frames)
            traj.net_hit_frames = dict(traj_state.net_hit_frames)
            for frame_num, det_state in traj_state.detections.items():
                det = Detection(
                    x=det_state.x, y=det_state.y, z=det_state.z, frame_number=det_state.frame_number, in_motion=det_state.in_motion, can_pick=det_state.can_pick
                )
                traj.detections[frame_num] = det
            self.trajectories[traj_id] = traj
        for queue_state in context.queues:
            queue = DetectionQueue(self.frame_width, self.frame_height)
            queue.detections = list(queue_state.detections)
            queue.push_to_trajectory = queue_state.push_to_trajectory
            queue.trajectory_index = queue_state.trajectory_index
            queue.has_overlap = queue_state.has_overlap
            self.queues.append(queue)

    def finalize_for_handoff(self, current_frame: int, frame_gap_threshold: int = 10):
        newly_stored = []
        self.process_queues(current_frame)
        for key, traj in list(self.trajectories.items()):
            if traj is None:
                continue
            if len(traj.detections) == 0:
                self.trajectories[key] = None
                continue
            latest_frame = max(traj.detections.keys())
            frame_gap = current_frame - latest_frame
            if frame_gap > frame_gap_threshold:
                for i in list(traj.filled_frames.keys()):
                    self.filled_frames[i] = 1
                for i in list(traj.net_hit_frames.keys()):
                    self.net_hit_frames[i] = 1
                first_f = min(traj.detections.keys())
                last_f = max(traj.detections.keys())
                # print(
                #     f"[CorrelationWorker]   [Tree] Finalize-handoff store trajectory id={key} "
                #     f"frames={first_f}->{last_f} points={len(traj.detections)} frame_gap={frame_gap}"
                # )
                self.stored_trajectories.append(traj)
                newly_stored.append(traj)
                self.trajectories[key] = None
        return newly_stored
