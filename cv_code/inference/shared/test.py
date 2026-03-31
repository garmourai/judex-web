"""
Realtime copy of heatmap prediction helpers from shuttle_tracking.shared.test.
Only predict_location and predict_multi_location for main_realtime flow.
"""

import cv2
import numpy as np


def predict_location(heatmap):
    """Get coordinates from the heatmap (single max-area bbox).

    Args:
        heatmap (numpy.ndarray): A single heatmap with shape (H, W)

    Returns:
        x, y, w, h (Tuple[int, int, int, int]): bounding box of max area.
    """
    if np.amax(heatmap) == 0:
        return 0, 0, 0, 0
    (cnts, _) = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in cnts]
    max_area_idx = 0
    max_area = rects[0][2] * rects[0][3]
    for i in range(1, len(rects)):
        area = rects[i][2] * rects[i][3]
        if area > max_area:
            max_area_idx = i
            max_area = area
    x, y, w, h = rects[max_area_idx]
    return x, y, w, h


def predict_multi_location(heatmap):
    """Get all coordinates from the heatmap (up to 3 bboxes by area).

    Args:
        heatmap (numpy.ndarray): A single heatmap with shape (H, W)

    Returns:
        List of (x, y, w, h) bounding boxes, sorted by area descending.
    """
    if np.amax(heatmap) == 0:
        return [[0, 0, 0, 0]]
    (cnts, _) = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in cnts]
    all_pred = []
    for i in range(len(rects)):
        area = rects[i][2] * rects[i][3]
        all_pred.append((area, rects[i]))
    all_pred.sort(reverse=True)
    final_pred = []
    num_pred = 3
    for i in range(len(all_pred)):
        if i > num_pred:
            break
        final_pred.append(all_pred[i][1])
    return final_pred
