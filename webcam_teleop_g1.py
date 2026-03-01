#!/usr/bin/env python3
"""
webcam_teleop_g1.py

用单目电脑摄像头 + MediaPipe Hands 控制：
1) G1 双臂（相对位移控制，按 r 锁定/重置参考姿态）
2) Inspire 双手掌（捏合程度控制开合，左右手分别控制）

运行前置：
- 仿真先启动，并开启 Inspire DDS（sim_main.py 里加 --enable_inspire_dds）
- 本脚本再启动
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "未找到 mediapipe。请先在你的运行环境里安装 mediapipe（注意不要把 numpy 升到 2.x）。"
    ) from e

# 双臂 IK 使用本地封装（不依赖 cwd）
_aria_g1_dir = os.path.dirname(os.path.abspath(__file__))
if _aria_g1_dir not in sys.path:
    sys.path.insert(0, _aria_g1_dir)
from g1_arm_ik_standalone import get_g1_arm_ik

# 手臂控制器仍从 xr_teleoperate 引入
XR_TELEOP_ROOT = os.environ.get("XR_TELEOP_ROOT", "/home/ritian/xr_teleoperate")
if XR_TELEOP_ROOT not in sys.path:
    sys.path.append(XR_TELEOP_ROOT)
from teleop.robot_control.robot_arm import G1_29_ArmController

# Unitree SDK（仍用于 Inspire 手）
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_


@dataclass
class TeleopConfig:
    cam_id: int = 0
    cam_width: int = 1280
    cam_height: int = 720
    cam_fps: int = 30

    # 机器人两只手的“舒适悬停点”(Home Pose)，单位：米（相对骨盆）
    # Z=0.08 很低，贴近桌面；手向下可再降
    robot_home_l: Tuple[float, float, float] = (0.35, 0.25, 0.08)   # X前, Y左, Z上
    robot_home_r: Tuple[float, float, float] = (0.35, -0.25, 0.08)

    # 手在图像里移动 -> 机器人末端 (X前, Y左, Z上)
    # du=水平, dv=垂直(下为正), ds=手大小变化
    xy_scale_m_per_norm: float = 1.5        # du -> dy 左右
    z_scale_m_per_norm: float = 4.0         # dv -> dz 上下（手向下=手臂降低，够桌子）
    x_depth_scale_m_per_norm: float = 0.5   # ds -> dx 前后（融合掌宽+腕中指，略提高权重）
    depth_smooth_alpha: float = 0.75       # 深度 EMA 平滑系数，越大越平滑（减少手姿突变）
    max_relative_dist_m: float = 0.60      # 单次最大相对位移（米）
    # MediaPipe 置信度：低于此值视为不可靠，不更新该手的目标
    min_hand_confidence: float = 0.5

    # 捏合比值（pinch/hand_size）映射到 open_ratio；调小 pinch_min 便于捏合时手能握紧抓物
    pinch_min: float = 0.12
    pinch_max: float = 0.65

    # Inspire 手掌电机参数（归一化 q：1=更张开，0=更握紧）
    inspire_kp: float = 2.0
    inspire_kd: float = 0.10
    # 手腕滚转的放大系数（人手翻转角 → 机器人腕关节角）
    wrist_roll_gain: float = 1.0


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _safe_norm2(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    d = float(np.linalg.norm(a - b))
    return d if d > eps else eps


def _calc_depth_proxy(pts: np.ndarray) -> float:
    """
    单目深度代理：融合手腕-中指根、掌宽，减少单一指标受手姿影响。
    手离镜头近 -> 值大；远 -> 值小。
    """
    wrist = pts[0]
    middle_mcp = pts[9]
    index_mcp = pts[5]
    pinky_mcp = pts[17]
    d1 = _safe_norm2(wrist, middle_mcp)
    d2 = _safe_norm2(index_mcp, pinky_mcp)
    return 0.6 * d1 + 0.4 * d2  # 掌宽相对稳定，减少手指弯曲影响


def _calc_open_ratio_from_landmarks(landmarks_xy: np.ndarray, cfg: TeleopConfig) -> float:
    """
    landmarks_xy: shape (21,2) in normalized image coords [0..1]
    捏合时尽量映射到 0（握紧），便于在 sim 里抓取物体。
    """
    wrist = landmarks_xy[0]
    middle_mcp = landmarks_xy[9]
    hand_size = _safe_norm2(wrist, middle_mcp)

    thumb_tip = landmarks_xy[4]
    index_tip = landmarks_xy[8]
    pinch = float(np.linalg.norm(thumb_tip - index_tip)) / hand_size

    open_ratio = (pinch - cfg.pinch_min) / (cfg.pinch_max - cfg.pinch_min)
    open_ratio = _clip01(open_ratio)
    # 抓取时：稍微捏合就发“完全握紧”，提高抓起来成功率
    if open_ratio < 0.25:
        open_ratio = 0.0
    return open_ratio


def _calc_wrist_roll_from_landmarks(landmarks_xy: np.ndarray) -> float:
    """根据食指掌指关节和小指掌指关节的连线，估计手腕绕前臂轴的旋转角（2D 简化版）"""
    index_mcp = landmarks_xy[5]
    pinky_mcp = landmarks_xy[17]
    v = pinky_mcp - index_mcp  # 从食指根指向小指根
    # atan2(y, x): 在图像坐标中，手掌翻转会让这条连线的角度发生明显变化
    angle = float(np.arctan2(v[1], v[0]))
    return angle


def _calc_finger_opens_from_landmarks(landmarks_xy: np.ndarray) -> list[float]:
    """根据各指尖-掌指关节的相对长度，估计每根手指的伸展程度（0=握紧, 1=张开）"""
    wrist = landmarks_xy[0]
    middle_mcp = landmarks_xy[9]
    hand_size = _safe_norm2(wrist, middle_mcp)

    def _norm_len(a_idx: int, b_idx: int) -> float:
        return float(np.linalg.norm(landmarks_xy[a_idx] - landmarks_xy[b_idx])) / hand_size

    # 经验范围：完全弯曲时相对长度 ~0.10，完全伸直 ~0.45，做一个线性映射
    def _to_ratio(v: float, v_min: float = 0.10, v_max: float = 0.45) -> float:
        return _clip01((v - v_min) / (v_max - v_min))

    # MediaPipe 手指数：thumb(1,4), index(5,8), middle(9,12), ring(13,16), pinky(17,20)
    thumb = _to_ratio(_norm_len(1, 4))
    index = _to_ratio(_norm_len(5, 8))
    middle = _to_ratio(_norm_len(9, 12))
    ring = _to_ratio(_norm_len(13, 16))
    pinky = _to_ratio(_norm_len(17, 20))

    # 返回顺序：thumb, index, middle, ring, pinky
    return [thumb, index, middle, ring, pinky]


def _send_inspire_open_ratios(
    publisher: ChannelPublisher,
    open_right: Optional[float],
    open_left: Optional[float],
    fingers_right: Optional[list[float]],
    fingers_left: Optional[list[float]],
    cfg: TeleopConfig,
) -> None:
    """
    MotorCmds_.cmds 顺序按 IsaacLab 的 Inspire 模型约定：
    - cmds[0:6]  : 右手 (R_pinky..R_thumb_yaw)
    - cmds[6:12] : 左手 (L_pinky..L_thumb_yaw)

    q 取值是归一化控制量，InspireDDS 中的映射为：
    - q=1 -> 关节角更靠近 min（更“张开”）
    - q=0 -> 关节角更靠近 max（更“握紧”）
    """
    msg = MotorCmds_()
    msg.cmds = []
    for _ in range(12):
        c = unitree_go_msg_dds__MotorCmd_()
        c.q = 1.0
        c.dq = 0.0
        c.tau = 0.0
        c.kp = cfg.inspire_kp
        c.kd = cfg.inspire_kd
        msg.cmds.append(c)

    # 右手：cmds[0:6] = [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
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
        msg.cmds[5].q = thumb  # yaw 跟着一起动

    # 左手：cmds[6:12] = [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
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


def _clamp_vec(v: np.ndarray, max_len: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= max_len:
        return v
    return v * (max_len / (n + 1e-9))


def main() -> None:
    cfg = TeleopConfig()

    print("[1/3] 初始化 DDS 通信...")
    # 仿真里 DDS 使用的是 channel 1，这里保持一致，并让底层自动选择网卡
    ChannelFactoryInitialize(1)

    inspire_publisher = ChannelPublisher("rt/inspire/cmd", MotorCmds_)
    inspire_publisher.Init()

    print("[1/3] 初始化 G1 手臂 IK 与控制器...")
    arm_ik = get_g1_arm_ik()
    arm_ctrl = G1_29_ArmController(motion_mode=False, simulation_mode=True)

    running_flag = [True]

    print("[2/3] 打开电脑摄像头...")
    cap = cv2.VideoCapture(cfg.cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: id={cfg.cam_id}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_height)
    cap.set(cv2.CAP_PROP_FPS, cfg.cam_fps)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
    )

    teleop_active = False
    ref: Dict[str, Tuple[float, float, float]] = {}   # "Left"/"Right" -> (u,v,hand_size)
    ref_roll: Dict[str, float] = {}                  # "Left"/"Right" -> 基准手腕翻转角

    home_l = np.array(cfg.robot_home_l, dtype=np.float32)
    home_r = np.array(cfg.robot_home_r, dtype=np.float32)

    print("\n" + "=" * 60)
    print("系统就绪！操作：")
    print("- r: 锁定/重置参考姿态（进入相对控制）")
    print("- q: 退出")
    print("=" * 60 + "\n")

    last_open_l: Optional[float] = None
    last_open_r: Optional[float] = None
    last_fingers_l: Optional[list[float]] = None
    last_fingers_r: Optional[list[float]] = None
    # 手离开画面时保持上一帧位姿，避免突然跳回 home
    last_left_tf: Optional[np.ndarray] = None
    last_right_tf: Optional[np.ndarray] = None
    # 深度 EMA 平滑状态
    last_ds_l: Optional[float] = None
    last_ds_r: Optional[float] = None

    try:
        while running_flag[0]:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.005)
                continue

            frame = cv2.flip(frame, 1)  # 镜像更符合“看着镜子操控”的直觉
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            # "Left"/"Right" -> (pts, confidence)，仅当置信度 >= min_hand_confidence 时采纳
            detected: Dict[str, Tuple[np.ndarray, float]] = {}

            if res.multi_hand_landmarks and res.multi_handedness:
                for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    label = hd.classification[0].label
                    score = float(hd.classification[0].score)
                    if score >= cfg.min_hand_confidence:
                        pts = np.array([[p.x, p.y] for p in lm.landmark], dtype=np.float32)
                        detected[label] = (pts, score)
                    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            if teleop_active:
                # 先置空，避免 IK 调用时变量未定义（仅本帧有对应手时才赋值）
                hand3d_l: Optional[list] = None
                hand3d_r: Optional[list] = None
                roll_l: Optional[float] = None
                roll_r: Optional[float] = None
                left_tf: Optional[np.ndarray] = None
                right_tf: Optional[np.ndarray] = None
                # 人手 Left -> 机器人左臂；人手 Right -> 机器人右臂
                if "Left" in detected:
                    pts, _ = detected["Left"]
                    wrist = pts[0]
                    middle_mcp = pts[9]
                    depth_proxy = _calc_depth_proxy(pts)
                    if "Left" not in ref:
                        ref["Left"] = (float(wrist[0]), float(wrist[1]), float(depth_proxy))
                    ru, rv, rs = ref["Left"]
                    du = float(wrist[0]) - ru
                    dv = float(wrist[1]) - rv
                    ds_raw = float(depth_proxy) - rs
                    ds = ds_raw if last_ds_l is None else (cfg.depth_smooth_alpha * last_ds_l + (1 - cfg.depth_smooth_alpha) * ds_raw)
                    last_ds_l = ds

                    # 图像垂直(dv)->Z，水平(du)->Y；dy=-du 使“手右移->臂右移”（Y正=左，手右则Y减）
                    dx = ds * cfg.x_depth_scale_m_per_norm
                    dy = -du * cfg.xy_scale_m_per_norm
                    dz = -dv * cfg.z_scale_m_per_norm   # 手向下 -> 手臂降低
                    hand3d_l = [dx, dy, dz]
                    # 构造左手末端在机器人坐标系下的目标位姿（仅使用平移，旋转用单位阵）
                    left_tf = np.eye(4, dtype=np.float32)
                    left_tf[:3, 3] = home_l + np.array([dx, dy, dz], dtype=np.float32)

                    last_open_l = _calc_open_ratio_from_landmarks(pts, cfg)
                    last_fingers_l = _calc_finger_opens_from_landmarks(pts)
                    # 计算左手腕翻转角，并做相对参考（按 r 键锁定基准）
                    roll_abs = _calc_wrist_roll_from_landmarks(pts)
                    if "Left" not in ref_roll:
                        ref_roll["Left"] = roll_abs
                    roll_l = roll_abs - ref_roll["Left"]

                if "Right" in detected:
                    pts, _ = detected["Right"]
                    wrist = pts[0]
                    middle_mcp = pts[9]
                    depth_proxy = _calc_depth_proxy(pts)
                    if "Right" not in ref:
                        ref["Right"] = (float(wrist[0]), float(wrist[1]), float(depth_proxy))
                    ru, rv, rs = ref["Right"]
                    du = float(wrist[0]) - ru
                    dv = float(wrist[1]) - rv
                    ds_raw = float(depth_proxy) - rs
                    ds = ds_raw if last_ds_r is None else (cfg.depth_smooth_alpha * last_ds_r + (1 - cfg.depth_smooth_alpha) * ds_raw)
                    last_ds_r = ds

                    dx = ds * cfg.x_depth_scale_m_per_norm
                    dy = -du * cfg.xy_scale_m_per_norm   # 手右->臂右
                    dz = -dv * cfg.z_scale_m_per_norm   # 手向下 -> 手臂降低
                    hand3d_r = [dx, dy, dz]
                    right_tf = np.eye(4, dtype=np.float32)
                    right_tf[:3, 3] = home_r + np.array([dx, dy, dz], dtype=np.float32)

                    last_open_r = _calc_open_ratio_from_landmarks(pts, cfg)
                    last_fingers_r = _calc_finger_opens_from_landmarks(pts)
                    roll_abs = _calc_wrist_roll_from_landmarks(pts)
                    if "Right" not in ref_roll:
                        ref_roll["Right"] = roll_abs
                    roll_r = roll_abs - ref_roll["Right"]

                # 手离开画面时保持上一帧位姿，避免突然跳回 home；首次无检测时用 home
                if left_tf is None:
                    left_tf = last_left_tf.copy() if last_left_tf is not None else np.eye(4, dtype=np.float32)
                    if last_left_tf is None:
                        left_tf[:3, 3] = home_l
                if right_tf is None:
                    right_tf = last_right_tf.copy() if last_right_tf is not None else np.eye(4, dtype=np.float32)
                    if last_right_tf is None:
                        right_tf[:3, 3] = home_r

                # 将手腕滚转角叠加到末端姿态的旋转矩阵中（绕局部 X 轴旋转）
                def _apply_roll(tf: np.ndarray, roll: Optional[float]) -> np.ndarray:
                    if roll is None:
                        return tf
                    angle = float(np.clip(roll * cfg.wrist_roll_gain, -1.5, 1.5))
                    c, s = np.cos(angle), np.sin(angle)
                    R_x = np.array([[1, 0, 0],
                                    [0, c,-s],
                                    [0, s, c]], dtype=np.float32)
                    tf = tf.copy()
                    tf[:3, :3] = R_x @ tf[:3, :3]
                    return tf

                left_tf = _apply_roll(left_tf, roll_l)
                right_tf = _apply_roll(right_tf, roll_r)
                # 有检测时更新“上一帧位姿”，供手离开时保持
                if "Left" in detected:
                    last_left_tf = left_tf.copy()
                if "Right" in detected:
                    last_right_tf = right_tf.copy()

                # 使用 G1_29_ArmIK 求解双臂 IK，并通过 G1_29_ArmController 控制真实关节
                try:
                    current_q = arm_ctrl.get_current_dual_arm_q()
                    current_dq = arm_ctrl.get_current_dual_arm_dq()
                    sol_q, sol_tauff = arm_ik.solve_ik(left_tf, right_tf, current_q, current_dq)
                    arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
                except Exception as e:
                    print(f"[IK] G1_29_ArmIK solve error: {e}")

                # 发送 Inspire 命令（两只手分别）
                _send_inspire_open_ratios(
                    inspire_publisher,
                    open_right=last_open_r,
                    open_left=last_open_l,
                    fingers_right=last_fingers_r,
                    fingers_left=last_fingers_l,
                    cfg=cfg,
                )

            # UI overlay
            status = "ACTIVE" if teleop_active else "IDLE"
            cv2.putText(
                frame,
                f"Teleop: {status}   Hands: {len(detected)}   (r lock/reset, q quit)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if teleop_active else (255, 255, 0),
                2,
            )
            if last_open_l is not None:
                cv2.putText(
                    frame,
                    f"Inspire Left open: {last_open_l:.2f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
            if last_open_r is not None:
                cv2.putText(
                    frame,
                    f"Inspire Right open: {last_open_r:.2f}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("Webcam Teleop (G1 arms + Inspire hands)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                # 进入/重置相对控制：清空参考点，下次检测到手就重新设参考
                teleop_active = True
                ref.clear()
                ref_roll.clear()
                last_ds_l = last_ds_r = None  # 重置深度平滑
                print("[teleop] 已锁定/重置参考姿态（相对控制）")
            if key == ord("h"):
                # 通过官方控制器将双臂回到 home 姿态
                try:
                    arm_ctrl.ctrl_dual_arm_go_home()
                except Exception as e:
                    print(f"[home] 调用 ctrl_dual_arm_go_home 出错: {e}")
                teleop_active = False
                ref.clear()
                ref_roll.clear()
                last_ds_l = last_ds_r = None
                # 同步 last_tf，避免再次按 r 后无手时跳回旧位姿
                last_left_tf = np.eye(4, dtype=np.float32)
                last_left_tf[:3, 3] = home_l
                last_right_tf = np.eye(4, dtype=np.float32)
                last_right_tf[:3, 3] = home_r
                print("[teleop] 已重置机器人到仿真初始姿态（按 r 可再次锁定手势）")

    finally:
        running_flag[0] = False
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()

