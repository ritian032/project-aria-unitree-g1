#!/usr/bin/env python3
"""
Inspire 机械手控制工具：捏合→开合映射、手指伸展、DDS 发布。
供 webcam_teleop_g1 和 aria_stereo_teleop_g1 共用。
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _safe_norm2(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    d = float(np.linalg.norm(a - b))
    return d if d > eps else eps


def calc_open_ratio_from_landmarks(
    landmarks: np.ndarray,
    pinch_min: float = 0.12,
    pinch_max: float = 0.65,
) -> float:
    """
    根据 MediaPipe landmarks 计算 Inspire 开合比例。
    landmarks: shape (21, 2) 或 (21, 3)，取前两列作为 xy
    捏合时映射到 0（握紧），便于抓取。
    """
    pts = np.asarray(landmarks, dtype=np.float32)
    if pts.ndim >= 2 and pts.shape[1] >= 2:
        pts = pts[:, :2]
    else:
        return 1.0

    wrist = pts[0]
    middle_mcp = pts[9]
    hand_size = _safe_norm2(wrist, middle_mcp)

    thumb_tip = pts[4]
    index_tip = pts[8]
    pinch = float(np.linalg.norm(thumb_tip - index_tip)) / hand_size

    open_ratio = (pinch - pinch_min) / (pinch_max - pinch_min)
    open_ratio = _clip01(open_ratio)
    if open_ratio < 0.25:
        open_ratio = 0.0
    return open_ratio


def calc_finger_opens_from_landmarks(landmarks: np.ndarray) -> list[float]:
    """
    根据各指尖-掌指关节的相对长度，估计每根手指的伸展程度（0=握紧, 1=张开）。
    返回 [thumb, index, middle, ring, pinky]
    """
    pts = np.asarray(landmarks, dtype=np.float32)
    if pts.ndim >= 2 and pts.shape[1] >= 2:
        pts = pts[:, :2]
    else:
        return [1.0] * 5

    wrist = pts[0]
    middle_mcp = pts[9]
    hand_size = _safe_norm2(wrist, middle_mcp)

    def _norm_len(a_idx: int, b_idx: int) -> float:
        return float(np.linalg.norm(pts[a_idx] - pts[b_idx])) / hand_size

    def _to_ratio(v: float, v_min: float = 0.10, v_max: float = 0.45) -> float:
        return _clip01((v - v_min) / (v_max - v_min))

    thumb = _to_ratio(_norm_len(1, 4))
    index = _to_ratio(_norm_len(5, 8))
    middle = _to_ratio(_norm_len(9, 12))
    ring = _to_ratio(_norm_len(13, 16))
    pinky = _to_ratio(_norm_len(17, 20))
    return [thumb, index, middle, ring, pinky]


def send_inspire_open_ratios(
    publisher,
    open_right: Optional[float],
    open_left: Optional[float],
    fingers_right: Optional[list[float]],
    fingers_left: Optional[list[float]],
    inspire_kp: float = 2.0,
    inspire_kd: float = 0.10,
) -> None:
    """
    发布 Inspire 手掌电机命令到 rt/inspire/cmd。
    MotorCmds_.cmds 顺序：cmds[0:6] 右手，cmds[6:12] 左手。
    q=1 更张开，q=0 更握紧。
    """
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

    msg = MotorCmds_()
    msg.cmds = []
    for _ in range(12):
        c = unitree_go_msg_dds__MotorCmd_()
        c.q = 1.0
        c.dq = 0.0
        c.tau = 0.0
        c.kp = inspire_kp
        c.kd = inspire_kd
        msg.cmds.append(c)

    if open_right is not None or fingers_right is not None:
        g = _clip01(open_right) if open_right is not None else 1.0
        if fingers_right is not None and len(fingers_right) == 5:
            thumb, index, middle, ring, pinky = [g * _clip01(v) for v in fingers_right]
        else:
            thumb = index = middle = ring = pinky = g
        msg.cmds[0].q = pinky
        msg.cmds[1].q = ring
        msg.cmds[2].q = middle
        msg.cmds[3].q = index
        msg.cmds[4].q = thumb
        msg.cmds[5].q = thumb

    if open_left is not None or fingers_left is not None:
        g = _clip01(open_left) if open_left is not None else 1.0
        if fingers_left is not None and len(fingers_left) == 5:
            thumb, index, middle, ring, pinky = [g * _clip01(v) for v in fingers_left]
        else:
            thumb = index = middle = ring = pinky = g
        base = 6
        msg.cmds[base + 0].q = pinky
        msg.cmds[base + 1].q = ring
        msg.cmds[base + 2].q = middle
        msg.cmds[base + 3].q = index
        msg.cmds[base + 4].q = thumb
        msg.cmds[base + 5].q = thumb

    publisher.Write(msg)
