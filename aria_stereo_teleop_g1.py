#!/usr/bin/env python3
"""
aria_stereo_teleop_g1.py

Aria 眼镜双目遥操作 G1 双臂：
1. 双目立体显示 (左眼+右眼拼接)
2. 双臂独立控制 (自动区分左右手)
3. 相对位移控制 (按 'r' 键锁定零点)
4. G1_29_ArmIK 精确 IK + 手离开保持、MediaPipe 置信度过滤
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np
import aria.sdk as aria

# Unitree SDK（G1_29_ArmController 自管 DDS）
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

# 双臂 IK 使用本地封装
_aria_g1_dir = os.path.dirname(os.path.abspath(__file__))
if _aria_g1_dir not in sys.path:
    sys.path.insert(0, _aria_g1_dir)
from g1_arm_ik_standalone import get_g1_arm_ik

# 手臂控制器仍从 xr_teleoperate 引入
XR_TELEOP_ROOT = os.environ.get("XR_TELEOP_ROOT", "/home/ritian/xr_teleoperate")
if XR_TELEOP_ROOT not in sys.path:
    sys.path.append(XR_TELEOP_ROOT)
from teleop.robot_control.robot_arm import G1_29_ArmController

# Custom Modules
from stereo_teleop_arm import HandTracker, StereoHandTracker
from aria_stereo_source import AriaStereoImageSource

# ================== 核心配置 ==================
SCALE_FACTOR = 1.5
MAX_RELATIVE_DIST = 0.5
MIN_HAND_CONFIDENCE = 0.5

# 坐标变换: Aria (Z前, X右, Y下) -> Robot (X前, Y左, Z上)
ROTATION_MATRIX = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0]
])
ROBOT_HOME_L = np.array([0.35,  0.25, 0.3])
ROBOT_HOME_R = np.array([0.35, -0.25, 0.3])
# =============================================


def transform_point(point_aria):
    return np.dot(ROTATION_MATRIX, np.array(point_aria))


def clamp_vector(vec, max_len):
    length = np.linalg.norm(vec)
    if length > max_len:
        return vec * (max_len / length)
    return vec


def main() -> None:
    parser = argparse.ArgumentParser(description="Aria 眼镜 G1 双臂遥操作")
    parser.add_argument("--sim", action="store_true", help="仿真模式（Isaac Sim），DDS channel 1")
    args = parser.parse_args()

    # ========= 1. 初始化 DDS 与 G1 手臂 =========
    print("[1/3] 初始化 DDS 与 G1 手臂 IK...")
    if args.sim:
        ChannelFactoryInitialize(1)
    else:
        ChannelFactoryInitialize(0, "lo")

    arm_ik = get_g1_arm_ik()
    arm_ctrl = G1_29_ArmController(motion_mode=False, simulation_mode=args.sim)

    running_flag = [True]

    # ========= 2. 初始化 Aria 眼镜 =========
    print("[2/3] 初始化 Aria 视频流...")
    aria.set_log_level(aria.Level.Info)
    client = aria.StreamingClient()
    config = client.subscription_config
    config.subscriber_data_type = aria.StreamingDataType.Slam
    config.message_queue_size[aria.StreamingDataType.Slam] = 1
    config.security_options.use_ephemeral_certs = True
    client.subscription_config = config

    source = AriaStereoImageSource()
    client.set_streaming_client_observer(source)
    client.subscribe()

    stereo_tracker = StereoHandTracker(
        stereo_camera=None,
        hand_tracker=HandTracker(min_hand_confidence=MIN_HAND_CONFIDENCE),
        baseline=0.12,
        focal_length=400.0,
    )

    # ========= 3. 控制循环状态 =========
    teleop_active = False
    ref_hand_L = None
    ref_hand_R = None
    last_left_target = None
    last_right_target = None

    print("\n" + "=" * 60)
    print("🚀 系统就绪！操作指南：")
    print("  r: 激活/重置参考姿态")
    print("  h: 双臂回到 home 位姿")
    print("  q: 退出")
    print("=" * 60 + "\n")

    try:
        while running_flag[0]:
            left_frame, right_frame = source.get_stereo_pair()
            if left_frame is None or right_frame is None:
                time.sleep(0.005)
                continue

            hands_data, l_vis, r_vis = stereo_tracker.track_hands_3d_from_frames(left_frame, right_frame)

            curr_hand_L = None
            curr_hand_R = None
            if hands_data and len(hands_data) > 0:
                sorted_hands = sorted(hands_data, key=lambda p: p[0])
                if len(sorted_hands) == 1:
                    if sorted_hands[0][0] < 0:
                        curr_hand_L = np.array(sorted_hands[0])
                    else:
                        curr_hand_R = np.array(sorted_hands[0])
                elif len(sorted_hands) >= 2:
                    curr_hand_L = np.array(sorted_hands[0])
                    curr_hand_R = np.array(sorted_hands[1])

            # --- 键盘交互 ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                if not teleop_active:
                    if curr_hand_L is not None or curr_hand_R is not None:
                        ref_hand_L = curr_hand_L
                        ref_hand_R = curr_hand_R
                        teleop_active = True
                        print("✅ [LOCKED] 控制已激活！")
                    else:
                        print("❌ [FAIL] 没检测到手，无法激活。")
                else:
                    teleop_active = False
                    print("⏸️ [RESET] 控制已暂停。")
            elif key == ord("h"):
                try:
                    arm_ctrl.ctrl_dual_arm_go_home()
                    last_left_target = ROBOT_HOME_L.copy()
                    last_right_target = ROBOT_HOME_R.copy()
                    print("[home] 双臂已回到 home")
                except Exception as e:
                    print(f"[home] 错误: {e}")

            # --- 核心控制：G1_29_ArmIK + 手离开保持 ---
            if teleop_active:
                target_L = None
                target_R = None

                if curr_hand_L is not None and ref_hand_L is not None:
                    delta = transform_point(curr_hand_L - ref_hand_L)
                    delta = clamp_vector(delta * SCALE_FACTOR, MAX_RELATIVE_DIST)
                    target_L = (ROBOT_HOME_L + delta).astype(np.float32)
                    last_left_target = target_L.copy()
                else:
                    target_L = np.array(last_left_target, dtype=np.float32) if last_left_target is not None else np.array(ROBOT_HOME_L, dtype=np.float32)

                if curr_hand_R is not None and ref_hand_R is not None:
                    delta = transform_point(curr_hand_R - ref_hand_R)
                    delta = clamp_vector(delta * SCALE_FACTOR, MAX_RELATIVE_DIST)
                    target_R = (ROBOT_HOME_R + delta).astype(np.float32)
                    last_right_target = target_R.copy()
                else:
                    target_R = np.array(last_right_target, dtype=np.float32) if last_right_target is not None else np.array(ROBOT_HOME_R, dtype=np.float32)

                left_tf = np.eye(4, dtype=np.float32)
                left_tf[:3, 3] = target_L
                right_tf = np.eye(4, dtype=np.float32)
                right_tf[:3, 3] = target_R

                try:
                    current_q = arm_ctrl.get_current_dual_arm_q()
                    current_dq = arm_ctrl.get_current_dual_arm_dq()
                    sol_q, sol_tauff = arm_ik.solve_ik(left_tf, right_tf, current_q, current_dq)
                    arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
                except Exception as e:
                    print(f"[IK] 求解错误: {e}")

            # --- 双目可视化 ---
            if l_vis is not None and r_vis is not None:
                color = (0, 255, 0) if teleop_active else (0, 0, 255)
                text = "ACTIVE" if teleop_active else "STANDBY (Press 'r')"
                cv2.putText(l_vis, f"L-Cam: {text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(r_vis, "R-Cam", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                combined_img = np.hstack((l_vis, r_vis))
                cv2.imshow("Aria Stereo Teleop (Left / Right)", combined_img)

    finally:
        running_flag[0] = False
        client.unsubscribe()
        cv2.destroyAllWindows()
        print("程序已退出。")


if __name__ == "__main__":
    main()
