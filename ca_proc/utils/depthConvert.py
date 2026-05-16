import torch
from PIL import Image
import numpy as np


def depthImageToDepthMap(depth_img: np.ndarray, d_min: float, d_max: float) -> np.ndarray:
    """Convert a grayscale depth image (uint8) to a metric depth map (float).

    Parameters
    ----------
    depth_img : np.ndarray  shape (H, W), dtype uint8, values in [0, 255]
    d_min     : float  minimum valid depth in metres
    d_max     : float  maximum valid depth in metres

    Returns
    -------
    depth_map : np.ndarray  shape (H, W), dtype float32, values in [d_min, d_max]
    """
    assert depth_img.dtype == np.uint8, "depth_img must be uint8"
    assert d_max > d_min, "d_max must be greater than d_min"

    depth_normalized = depth_img.astype(np.float32) / 255.0
    depth_map = depth_normalized * (d_max - d_min) + d_min
    return depth_map


def depthMapToDepthImage(depth_map: np.ndarray, d_min: float, d_max: float) -> np.ndarray:
    """Convert a metric depth map (float) to a grayscale depth image (uint8).

    Parameters
    ----------
    depth_map : np.ndarray  shape (H, W), dtype float
    d_min     : float  minimum valid depth in metres
    d_max     : float  maximum valid depth in metres

    Returns
    -------
    depth_img : np.ndarray  shape (H, W), dtype uint8, values in [0, 255]
    """
    assert depth_map.ndim == 2, "depth_map must be a 2-D array (H, W)"
    assert np.issubdtype(depth_map.dtype, np.floating), "depth_map must be float"
    assert d_max > d_min, "d_max must be greater than d_min"

    depth_clamped    = np.clip(depth_map, d_min, d_max)
    depth_normalized = (depth_clamped - d_min) / (d_max - d_min)
    depth_img        = (depth_normalized * 255).astype(np.uint8)
    return depth_img
