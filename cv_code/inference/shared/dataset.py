"""
Realtime copy of TrackNet dataset for frame_arr inference from shuttle_tracking.shared.dataset.
Only supports: frame_arr + data_mode='heatmap' + bg_mode='concat' + camera_id (realtime path).
"""

import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset

from ..utils_general import HEIGHT, WIDTH, SIGMA


class Shuttlecock_Trajectory_Dataset(Dataset):
    """Minimal dataset for realtime TrackNet inference from a frame array."""

    _cached_medians = {}
    _median_calculated_for_cameras = set()
    _save_processed_median = False
    _processed_median_debug_dir = None
    _cached_median = None
    _median_calculated = False

    def __init__(
        self,
        root_dir,
        split="train",
        seq_len=8,
        sliding_step=1,
        data_mode="heatmap",
        bg_mode="concat",
        frame_alpha=-1,
        rally_dir=None,
        frame_arr=None,
        pred_dict=None,
        padding=False,
        debug=False,
        camera_id=None,
        HEIGHT=HEIGHT,
        WIDTH=WIDTH,
        SIGMA=SIGMA,
    ):
        assert frame_arr is not None and data_mode == "heatmap"
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.mag = 1
        self.sigma = SIGMA
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.sliding_step = sliding_step
        self.data_mode = data_mode
        self.bg_mode = bg_mode
        self.frame_alpha = frame_alpha
        self.frame_arr = frame_arr
        self.pred_dict = pred_dict
        self.padding = padding and sliding_step == seq_len

        self.data_dict, self.img_config = self._gen_input_from_frame_arr()

        if bg_mode:
            self.camera_id = camera_id
            if camera_id is not None:
                is_first = camera_id not in Shuttlecock_Trajectory_Dataset._median_calculated_for_cameras
                if is_first:
                    median = np.median(self.frame_arr[:30], 0)
                    if self.bg_mode == "concat":
                        median = Image.fromarray(median.astype("uint8"))
                        median = np.array(median.resize(size=(self.WIDTH, self.HEIGHT)))
                        median_final = np.moveaxis(median, -1, 0)
                        if Shuttlecock_Trajectory_Dataset._save_processed_median:
                            debug_dir = Shuttlecock_Trajectory_Dataset._processed_median_debug_dir
                            if debug_dir:
                                os.makedirs(debug_dir, exist_ok=True)
                                np.savez_compressed(
                                    os.path.join(debug_dir, "processed_median_final.npz"),
                                    processed_median=median_final,
                                )
                                Shuttlecock_Trajectory_Dataset._save_processed_median = False
                        Shuttlecock_Trajectory_Dataset._cached_medians[camera_id] = median_final
                        Shuttlecock_Trajectory_Dataset._median_calculated_for_cameras.add(camera_id)
                        print(f"   ✅ Calculated and cached median for camera {camera_id}")
                    else:
                        Shuttlecock_Trajectory_Dataset._cached_medians[camera_id] = median
                        Shuttlecock_Trajectory_Dataset._median_calculated_for_cameras.add(camera_id)
                else:
                    if camera_id not in Shuttlecock_Trajectory_Dataset._cached_medians:
                        median = np.median(self.frame_arr[:30], 0)
                        if self.bg_mode == "concat":
                            median = Image.fromarray(median.astype("uint8"))
                            median = np.array(median.resize(size=(self.WIDTH, self.HEIGHT)))
                            median_final = np.moveaxis(median, -1, 0)
                            Shuttlecock_Trajectory_Dataset._cached_medians[camera_id] = median_final
                        else:
                            Shuttlecock_Trajectory_Dataset._cached_medians[camera_id] = median
                        Shuttlecock_Trajectory_Dataset._median_calculated_for_cameras.add(camera_id)
                self.median = Shuttlecock_Trajectory_Dataset._cached_medians.get(camera_id)
                if self.median is None:
                    raise ValueError(f"Failed to get median for camera {camera_id}")
            else:
                if not Shuttlecock_Trajectory_Dataset._median_calculated:
                    median = np.median(self.frame_arr[:30], 0)
                    if self.bg_mode == "concat":
                        median = Image.fromarray(median.astype("uint8"))
                        median = np.array(median.resize(size=(self.WIDTH, self.HEIGHT)))
                        median_final = np.moveaxis(median, -1, 0)
                        Shuttlecock_Trajectory_Dataset._cached_median = median_final
                    else:
                        Shuttlecock_Trajectory_Dataset._cached_median = median
                    Shuttlecock_Trajectory_Dataset._median_calculated = True
                self.median = Shuttlecock_Trajectory_Dataset._cached_median

    def _gen_input_from_frame_arr(self):
        h, w, _ = self.frame_arr[0].shape
        h_scaler, w_scaler = h / self.HEIGHT, w / self.WIDTH
        id_arr = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
        last_idx = -1
        for i in range(0, len(self.frame_arr), self.sliding_step):
            tmp_idx = []
            for f in range(self.seq_len):
                if i + f < len(self.frame_arr):
                    tmp_idx.append((0, i + f))
                    last_idx = i + f
                else:
                    if self.padding:
                        tmp_idx.append((0, last_idx))
                    else:
                        break
            if len(tmp_idx) == self.seq_len:
                id_arr = np.concatenate((id_arr, [tmp_idx]), axis=0)
        return dict(id=id_arr), dict(img_scaler=(w_scaler, h_scaler), img_shape=(w, h))

    def __len__(self):
        return len(self.data_dict["id"])

    def __getitem__(self, idx):
        data_idx = self.data_dict["id"][idx]
        imgs = self.frame_arr[data_idx[:, 1], ...]
        median_img = self.median
        num_channels = 3
        frames = np.zeros(
            (self.seq_len * num_channels + median_img.shape[0], self.HEIGHT, self.WIDTH),
            dtype=np.float32,
        )
        offset = median_img.shape[0]
        for i in range(self.seq_len):
            if imgs[i].shape[0] != self.HEIGHT or imgs[i].shape[1] != self.WIDTH:
                img = cv2.resize(imgs[i], (self.WIDTH, self.HEIGHT))
                img = np.moveaxis(np.array(img), -1, 0)
            else:
                img = np.moveaxis(np.array(imgs[i]), -1, 0)
            frames[offset + i * num_channels : offset + (i + 1) * num_channels] = img
        frames[: median_img.shape[0]] = median_img
        frames /= 255.0
        return data_idx, frames
