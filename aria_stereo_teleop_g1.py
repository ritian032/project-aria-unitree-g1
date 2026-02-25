#!/usr/bin/env python3
"""
aria_teleop_final.py

功能完全体：
1. 双目立体显示 (左眼+右眼拼接)
2. 双臂独立控制 (自动区分左右手)
3. 相对位移控制 (按 'r' 键锁定零点)
4. 坐标系自动转换 (Aria -> Robot)
"""

import time
import cv2
import numpy as np
import aria.sdk as aria
from threading import Thread

# Unitree SDK
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

# Custom Modules
from stereo_teleop_arm import (
    ArmController,
    HandTracker,
    StereoHandTracker,
    LowStateHandler,
    LowCmdWrite,
)
from aria_stereo_source import AriaStereoImageSource

# ================== 核心配置 ==================
# 灵敏度：手移动 1cm，机器人移动 1.5cm
SCALE_FACTOR = 1.5           

# 安全限制：最大相对移动距离 (米)，防止动作过大
MAX_RELATIVE_DIST = 0.5      

# 坐标变换矩阵: Aria (Z前, X右, Y下) -> Robot (X前, Y左, Z上)
# 修正了之前的映射，确保手往前伸，机器人也往前伸
ROTATION_MATRIX = np.array([
    [ 0,  0,  1],  # Robot X (前) = Aria Z (前)
    [-1,  0,  0],  # Robot Y (左) = -Aria X (右)
    [ 0, -1,  0]   # Robot Z (上) = -Aria Y (下)
])
# =============================================

def transform_point(point_aria):
    """ 将 Aria 坐标点转换到 Robot 坐标系 """
    return np.dot(ROTATION_MATRIX, np.array(point_aria))

def clamp_vector(vec, max_len):
    """ 限制向量长度，防止瞬间飞出 """
    length = np.linalg.norm(vec)
    if length > max_len:
        return vec * (max_len / length)
    return vec

def main() -> None:
    # ========= 1. 初始化机器人通信 =========
    print("[1/3] 初始化 DDS 通信...")
    ChannelFactoryInitialize(0, "lo")
    arm_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
    arm_publisher.Init()

    # ⚠️ 关闭镜像模式，因为我们要分别控制两只手
    controller = ArmController(mirror_mode=False)

    state_sub = ChannelSubscriber("rt/lowstate", LowState_)
    state_sub.Init(lambda msg: LowStateHandler(controller, msg), 10)

    print("[1/3] 等待机器人心跳...")
    while not controller.first_update:
        time.sleep(0.1)

    # 启动控制线程
    crc = CRC()
    running_flag = [True]
    Thread(target=LowCmdWrite, args=(controller, arm_publisher, crc, running_flag), daemon=True).start()

    # 定义机器人两只手的“舒适悬停点” (Home Pose)
    # 当你没操作时，手会停在这里 (单位: 米，相对于骨盆)
    ROBOT_HOME_L = np.array([0.35,  0.25, 0.3]) # 左手: 前0.35, 左0.25, 上0.3
    ROBOT_HOME_R = np.array([0.35, -0.25, 0.3]) # 右手: Y是负的

    # ========= 2. 初始化 Aria 眼镜 =========
    print("[2/3] 初始化 Aria 视频流...")
    aria.set_log_level(aria.Level.Info)
    client = aria.StreamingClient()
    
    # 配置 SLAM 相机订阅
    config = client.subscription_config
    config.subscriber_data_type = aria.StreamingDataType.Slam
    config.message_queue_size[aria.StreamingDataType.Slam] = 1
    config.security_options.use_ephemeral_certs = True
    client.subscription_config = config

    source = AriaStereoImageSource()
    client.set_streaming_client_observer(source)
    client.subscribe()

    # 初始化视觉算法
    stereo_tracker = StereoHandTracker(
        stereo_camera=None, 
        hand_tracker=HandTracker(), 
        baseline=0.12,      # Aria 基线约 12cm
        focal_length=400.0  # 稍微调小焦距以增大视场容错
    )

    # ========= 3. 控制循环状态 =========
    teleop_active = False
    ref_hand_L = None  # 左手锁定时的原点
    ref_hand_R = None  # 右手锁定时的原点

    print("\n" + "="*60)
    print("🚀 系统就绪！操作指南：")
    print("1. 站在 Aria 视野内，举起双手。")
    print("2. 调整到一个舒服的姿势。")
    print("3. 按键盘 'r' 键 -> 激活控制 (Lock)。")
    print("4. 再次按 'r' 键 -> 重置原点 (Reset)。")
    print("5. 按 'q' 键 -> 退出。")
    print("="*60 + "\n")

    try:
        while running_flag[0]:
            left_frame, right_frame = source.get_stereo_pair()
            if left_frame is None or right_frame is None:
                time.sleep(0.005)
                continue

            # 视觉处理：获取 3D 手部坐标
            # 注意：这里的 r_vis 必须接收，用于双目显示
            hands_data, l_vis, r_vis = stereo_tracker.track_hands_3d_from_frames(left_frame, right_frame)
            
            # --- 自动区分左右手 ---
            curr_hand_L = None
            curr_hand_R = None

            if hands_data and len(hands_data) > 0:
                # 按 X 轴排序 (画面左边是左手，右边是右手)
                sorted_hands = sorted(hands_data, key=lambda p: p[0]) 
                
                if len(sorted_hands) == 1:
                    # 只有一只手：判断是在左边还是右边
                    if sorted_hands[0][0] < 0: curr_hand_L = np.array(sorted_hands[0])
                    else:                      curr_hand_R = np.array(sorted_hands[0])
                elif len(sorted_hands) >= 2:
                    # 有两只手：左边给左手，右边给右手
                    curr_hand_L = np.array(sorted_hands[0])
                    curr_hand_R = np.array(sorted_hands[1])

            # --- 键盘交互 ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # 切换激活状态
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
                    print("⏸️ [RESET] 控制已暂停，回到待机。")

            # --- 核心控制算法 ---
            targets_to_send = [None, None] # [Target_L, Target_R]

            if teleop_active:
                # 1. 计算左臂目标
                if curr_hand_L is not None and ref_hand_L is not None:
                    delta = transform_point(curr_hand_L - ref_hand_L)
                    delta = clamp_vector(delta * SCALE_FACTOR, MAX_RELATIVE_DIST)
                    targets_to_send[0] = ROBOT_HOME_L + delta
                else:
                    # 如果手跟丢了，保持在最后已知位置或Home，这里选择保持Home防止乱动
                    targets_to_send[0] = ROBOT_HOME_L

                # 2. 计算右臂目标
                if curr_hand_R is not None and ref_hand_R is not None:
                    delta = transform_point(curr_hand_R - ref_hand_R)
                    delta = clamp_vector(delta * SCALE_FACTOR, MAX_RELATIVE_DIST)
                    targets_to_send[1] = ROBOT_HOME_R + delta
                else:
                    targets_to_send[1] = ROBOT_HOME_R
                
                # 3. 发送指令
                # 确保你的 ArmController 有 update_targets_absolute 方法
                if hasattr(controller, 'update_targets_absolute'):
                    controller.update_targets_absolute(targets_to_send)
                else:
                    print("❌ 错误：请更新 stereo_teleop_arm.py 添加 update_targets_absolute 方法！")

            # --- 双目可视化 ---
            if l_vis is not None and r_vis is not None:
                # 状态显示
                color = (0, 255, 0) if teleop_active else (0, 0, 255)
                text = "ACTIVE" if teleop_active else "STANDBY (Press 'r')"
                
                # 在左图写字
                cv2.putText(l_vis, f"L-Cam: {text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                # 在右图写字
                cv2.putText(r_vis, "R-Cam", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # 横向拼接图像
                combined_img = np.hstack((l_vis, r_vis))
                cv2.imshow("Aria Stereo Teleop (Left / Right)", combined_img)

    finally:
        running_flag[0] = False
        client.unsubscribe()
        cv2.destroyAllWindows()
        print("程序已退出。")

if __name__ == "__main__":
    main()