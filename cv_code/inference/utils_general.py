"""
Realtime copy of only what main_realtime flow needs from shuttle_tracking.utils.general.
Constants and helpers for TrackNet input size and image conversion.
"""

import numpy as np

# TrackNet input dimensions (must match model)
HEIGHT = 288
WIDTH = 512
SIGMA = 2.5


def to_img(image: np.ndarray) -> np.ndarray:
    """Convert normalized image [0, 1] to uint8 [0, 255]."""
    image = image * 255
    image = image.astype("uint8")
    return image
