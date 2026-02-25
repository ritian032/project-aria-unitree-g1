#!/usr/bin/env python3
"""
AriaStereoImageSource

基于 Project Aria Python SDK 的简单双目图像源。
只负责从 StreamingClient 回调中缓存最新的 SLAM 左右目图像。
"""

import aria.sdk as aria
import numpy as np
from projectaria_tools.core.sensor_data import ImageDataRecord


class AriaStereoImageSource:
    """
    作为 StreamingClientObserver，只负责存最新的 SLAM1/SLAM2 图像。
    在主循环中通过 get_stereo_pair() 获取。
    """

    def __init__(self) -> None:
        self.images: dict[aria.CameraId, np.ndarray] = {}

    def on_image_received(self, image: np.ndarray, record: ImageDataRecord) -> None:
        # 参考 visualizer.py，把图像旋转到和相机一致的朝向
        if record.camera_id != aria.CameraId.EyeTrack:
            image = np.rot90(image, -1)
        else:
            image = np.rot90(image, 2)

        self.images[record.camera_id] = image

    def get_stereo_pair(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        返回 (left, right) = (Slam1, Slam2) 图像。
        如果某一侧还没有数据则返回 None。
        """
        left = self.images.get(aria.CameraId.Slam1)
        right = self.images.get(aria.CameraId.Slam2)
        return left, right

