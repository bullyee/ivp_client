from typing import List, Tuple, Any
import numpy as np

class KeypointData:
    def __init__(self, pixel: Tuple[int, int], descriptor: np.ndarray, world_coord: np.ndarray):
        self.pixel: Tuple[int, int] = pixel  # (u, v) in image
        self.descriptor: np.ndarray = descriptor  # ORB descriptor (32 values)
        self.world_coord: np.ndarray = world_coord  # (X, Y, Z) in world coordinates

    def __repr__(self) -> str:
        return f"Keypoint(pixel={self.pixel}, world={self.world_coord})"

class ImageData:
    def __init__(self, keypointData: List[KeypointData], R: np.ndarray, T: np.ndarray):
        self.keypointData: List[KeypointData] = keypointData
        self.R: np.ndarray = R  # Rotation matrix (3x3)
        self.T: np.ndarray = T  # Translation vector (3x1)

class DbData:
    def __init__(self, keypointData: List[KeypointData], histogram: np.ndarray):
        self.keypointData: List[KeypointData] = keypointData
        self.histogram: np.ndarray = histogram  # An image's BoVW histogram
