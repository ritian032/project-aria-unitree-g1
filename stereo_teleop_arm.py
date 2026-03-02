#!/usr/bin/env python3
"""
基于双目摄像头的手臂遥操作系统
使用双目摄像头进行手部跟踪，控制 G1 机器人手臂
"""

import time
import cv2
import numpy as np
import threading
from typing import Optional, Tuple
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

kPi = 3.141592654
kPi_2 = 1.57079632

class G1JointIndex:
    """G1 关节索引定义"""
    # Left leg
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleRoll = 5
    
    # Right leg
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleRoll = 11
    
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14
    
    # Left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21
    
    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28
    
    kNotUsedJoint = 29


class StereoCamera:
    """双目摄像头类"""
    
    def __init__(self, left_camera_id: int = 0, right_camera_id: int = 1):
        """
        初始化双目摄像头
        
        Args:
            left_camera_id: 左摄像头设备ID
            right_camera_id: 右摄像头设备ID
        """
        self.left_camera_id = left_camera_id
        self.right_camera_id = right_camera_id
        self.left_cap = None
        self.right_cap = None
        self.stereo_calibrated = False
        
        # 立体视觉参数（需要标定）
        self.baseline = 0.12  # 基线距离（米），需要根据实际硬件标定
        self.focal_length = 640.0  # 焦距（像素），需要标定
        
    def init(self) -> bool:
        """初始化摄像头"""
        try:
            self.left_cap = cv2.VideoCapture(self.left_camera_id)
            self.right_cap = cv2.VideoCapture(self.right_camera_id)
            
            if not self.left_cap.isOpened() or not self.right_cap.isOpened():
                print(f"错误：无法打开摄像头 {self.left_camera_id} 或 {self.right_camera_id}")
                return False
            
            # 设置摄像头参数
            self.left_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.left_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.right_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.right_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            print(f"双目摄像头初始化成功：左={self.left_camera_id}, 右={self.right_camera_id}")
            return True
        except Exception as e:
            print(f"摄像头初始化失败：{e}")
            return False
    
    def read_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """读取左右图像"""
        ret_left, frame_left = self.left_cap.read()
        ret_right, frame_right = self.right_cap.read()
        
        if ret_left and ret_right:
            return frame_left, frame_right
        return None, None
    
    def release(self):
        """释放摄像头资源"""
        if self.left_cap:
            self.left_cap.release()
        if self.right_cap:
            self.right_cap.release()


class HandTracker:
    """手部跟踪器（使用 MediaPipe 或 OpenCV）"""
    
    def __init__(self, min_hand_confidence: float = 0.5):
        """初始化手部跟踪器

        Args:
            min_hand_confidence: MediaPipe 置信度阈值，低于此值的手部检测将被过滤
        """
        self.min_hand_confidence = min_hand_confidence
        # 优先尝试使用 MediaPipe，不同版本的 API 稍有差异，这里做兼容处理
        try:
            import mediapipe as mp
            # 大部分版本都支持 mp.solutions
            mp_solutions = mp.solutions
            mp_hands = mp_solutions.hands
            mp_drawing = mp_solutions.drawing_utils

            self.use_mediapipe = True
            self.mp_hands = mp_hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.3,  # 降低检测阈值，提高检测率
                min_tracking_confidence=0.3,   # 降低跟踪阈值，提高稳定性
            )
            self.mp_drawing = mp_drawing
            print("使用 MediaPipe 进行手部跟踪")
        except ImportError as e:
            # 调试输出具体 ImportError，方便排查版本/依赖问题
            self.use_mediapipe = False
            print("MediaPipe 导入失败，回退到 OpenCV 占位实现。错误信息：", repr(e))
            print("建议在当前环境重新安装或更换 mediapipe 版本，例如：")
            print("  pip install 'mediapipe==0.10.14'")
    
    def detect_hands(self, image: np.ndarray) -> list:
        """
        检测手部关键点
        
        Returns:
            list: 手部关键点列表，每个元素是 (x, y, z) 坐标
        """
        if self.use_mediapipe:
            return self._detect_mediapipe(image)
        else:
            return self._detect_opencv(image)
    
    def _detect_mediapipe(self, image: np.ndarray) -> list:
        """使用 MediaPipe 检测手部
        
        Returns:
            list: 每个元素是 (landmarks, handedness, score) 的元组，仅包含置信度 >= min_hand_confidence 的检测
            - landmarks: 21个关键点的列表 [[x, y, z], ...]
            - handedness: 'Left' 或 'Right'（用户的左手或右手）
            - score: MediaPipe 置信度 [0..1]
        """
        # MediaPipe Hands expects RGB 3-channel input
        if image is None:
            return []
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        hands_data = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                score = float(handedness.classification[0].score)
                if score < self.min_hand_confidence:
                    continue
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                hands_data.append((landmarks, label, score))
        return hands_data
    
    def _detect_opencv(self, image: np.ndarray) -> list:
        """使用 OpenCV 简单检测（占位符，需要实现）"""
        # TODO: 实现基于 OpenCV 的手部检测
        return []


class StereoHandTracker:
    """双目手部跟踪器"""
    
    def __init__(
        self,
        stereo_camera: Optional[StereoCamera],
        hand_tracker: HandTracker,
        baseline: float = 0.12,
        focal_length: float = 640.0,
    ):
        """
        初始化双目手部跟踪器
        
        Args:
            stereo_camera: 双目摄像头对象
            hand_tracker: 手部跟踪器对象
            baseline: 双目基线（米），当 stereo_camera 为 None 时使用
            focal_length: 焦距（像素），当 stereo_camera 为 None 时使用
        """
        self.stereo_camera = stereo_camera
        self.hand_tracker = hand_tracker
        # 如果传入了 StereoCamera，则优先使用其中的标定参数
        if self.stereo_camera is not None:
            self.baseline = getattr(self.stereo_camera, "baseline", baseline)
            self.focal_length = getattr(self.stereo_camera, "focal_length", focal_length)
        else:
            self.baseline = baseline
            self.focal_length = focal_length
    
    def _track_hands_3d_from_frames(
        self, left_frame: np.ndarray, right_frame: np.ndarray
    ) -> Tuple[Optional[list], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        使用给定的左右图像跟踪手部3D位置。
        供内部和外部（例如 Aria 双目源）复用。
        """
        # 兼容灰度图（Aria SLAM 图像通常是单通道）
        if left_frame is not None and left_frame.ndim == 2:
            left_frame = cv2.cvtColor(left_frame, cv2.COLOR_GRAY2BGR)
        if right_frame is not None and right_frame.ndim == 2:
            right_frame = cv2.cvtColor(right_frame, cv2.COLOR_GRAY2BGR)

        # 图像增强：提高检测率（对灰度图特别有效）
        if left_frame is not None:
            # 对比度增强
            left_frame = cv2.convertScaleAbs(left_frame, alpha=1.5, beta=10)
        if right_frame is not None:
            right_frame = cv2.convertScaleAbs(right_frame, alpha=1.5, beta=10)

        # 在左右图像中检测手部
        left_hands_raw = self.hand_tracker.detect_hands(left_frame)
        right_hands_raw = self.hand_tracker.detect_hands(right_frame)
        
        # 调试：显示检测到的数量（用于终端和图像窗口）
        debug_info = f"左图: {len(left_hands_raw)} 只手, 右图: {len(right_hands_raw)} 只手"
        
        # 解析检测结果：新格式 (landmarks, handedness, score) 或 (landmarks, handedness)
        def parse_hand_data(hand_data):
            """解析手部数据，兼容新旧格式"""
            if isinstance(hand_data, tuple) and len(hand_data) >= 2:
                return hand_data[0], hand_data[1]  # (landmarks, handedness)
            else:
                return hand_data, None  # 旧格式，只有landmarks
        
        left_hands = [parse_hand_data(h) for h in left_hands_raw]
        right_hands = [parse_hand_data(h) for h in right_hands_raw]
        
        # 立体匹配：使用 handedness 信息来正确配对左右手
        # 目标：用户的左手 → hands_3d[0]，用户的右手 → hands_3d[1]
        hands_3d: list[list[float]] = []
        hands_handedness: list[str] = []  # 记录每只手的 handedness
        hands_landmarks: list[list] = []  # 每只手对应的 MediaPipe landmarks，用于 Inspire 捏合

        if len(left_hands) > 0 and len(right_hands) > 0:
            h, w = left_frame.shape[:2]

            def wrist_xy(landmarks: list) -> tuple[float, float]:
                wrist = landmarks[0]  # [x, y, z] normalized
                return wrist[0] * w, wrist[1] * h

            # 配对策略：优先使用 handedness，如果没有则按 x 坐标排序
            # 1. 如果有 handedness，按 handedness 配对
            # 2. 如果没有 handedness，按 x 坐标排序配对
            
            left_with_info = []
            right_with_info = []
            
            for i, (landmarks, handedness) in enumerate(left_hands):
                if len(landmarks) > 0:
                    x, y = wrist_xy(landmarks)
                    left_with_info.append((i, landmarks, handedness, x, y))
            
            for i, (landmarks, handedness) in enumerate(right_hands):
                if len(landmarks) > 0:
                    x, y = wrist_xy(landmarks)
                    right_with_info.append((i, landmarks, handedness, x, y))
            
            # 智能配对：优先使用 handedness，否则按 x 坐标最接近配对
            left_list = sorted(left_with_info, key=lambda t: t[3])   # by x
            right_list = sorted(right_with_info, key=lambda t: t[3])  # by x
            
            # 配对策略：对于左图中的每只手，在右图中找最佳匹配
            used_right_indices = set()
            
            for left_item in left_list:
                _, left_landmarks, left_hand, left_x, left_y = left_item
                
                # 在右图中找最佳匹配
                best_right_idx = None
                best_disparity = float('inf')
                
                for right_idx, right_item in enumerate(right_list):
                    if right_idx in used_right_indices:
                        continue
                    
                    _, right_landmarks, right_hand, right_x, right_y = right_item
                    
                    # 优先匹配 handedness 相同的手
                    if left_hand is not None and right_hand is not None:
                        if left_hand == right_hand:
                            disparity = abs(left_x - right_x)
                            if disparity < best_disparity:
                                best_disparity = disparity
                                best_right_idx = right_idx
                    
                    # 如果没有 handedness 匹配，按 x 坐标最接近
                    if best_right_idx is None:
                        disparity = abs(left_x - right_x)
                        if disparity < best_disparity:
                            best_disparity = disparity
                            best_right_idx = right_idx
                
                # 如果找到匹配且视差合理（放宽视差要求）
                if best_right_idx is not None and best_disparity >= 0.05:  # 从 0.1 降到 0.05
                    _, right_landmarks, right_hand, right_x, right_y = right_list[best_right_idx]
                    used_right_indices.add(best_right_idx)
                    
                    depth = (self.baseline * self.focal_length) / best_disparity
                    depth = np.clip(depth, 0.2, 3.0)
                    x = (left_x - w / 2) * depth / self.focal_length
                    y = (left_y - h / 2) * depth / self.focal_length
                    z = depth
                    
                    hands_3d.append([float(x), float(y), float(z)])
                    # 使用左图的 handedness（更可靠）
                    hands_handedness.append(left_hand or 'Unknown')
                    # 保存左图 landmarks 供 Inspire 捏合计算
                    hands_landmarks.append(left_landmarks)
            
            # 确保顺序：用户的左手 hands_3d[0]，用户的右手 hands_3d[1]
            if len(hands_3d) == 2 and len(hands_handedness) == 2:
                if hands_handedness[0] == 'Right' and hands_handedness[1] == 'Left':
                    hands_3d[0], hands_3d[1] = hands_3d[1], hands_3d[0]
                    hands_handedness[0], hands_handedness[1] = hands_handedness[1], hands_handedness[0]
                    hands_landmarks[0], hands_landmarks[1] = hands_landmarks[1], hands_landmarks[0]
        
        # 在图像上绘制手部关键点（使用之前检测的结果）
        if self.hand_tracker.use_mediapipe:
            # 重新处理图像用于绘制（detect_hands 已经处理过了，这里只是为了绘制）
            left_frame_rgb = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
            right_frame_rgb = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)
            
            left_results = self.hand_tracker.hands.process(left_frame_rgb)
            right_results = self.hand_tracker.hands.process(right_frame_rgb)
            
            if left_results.multi_hand_landmarks:
                for hand_landmarks in left_results.multi_hand_landmarks:
                    self.hand_tracker.mp_drawing.draw_landmarks(
                        left_frame, hand_landmarks, self.hand_tracker.mp_hands.HAND_CONNECTIONS
                    )
            
            if right_results.multi_hand_landmarks:
                for hand_landmarks in right_results.multi_hand_landmarks:
                    self.hand_tracker.mp_drawing.draw_landmarks(
                        right_frame, hand_landmarks, self.hand_tracker.mp_hands.HAND_CONNECTIONS
                    )
        
        # 在图像上添加调试信息
        if left_frame is not None:
            cv2.putText(left_frame, debug_info, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(left_frame, f"匹配到 {len(hands_3d)} 只手", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 终端调试：当检测到多只手或数量不一致时打印
        if len(left_hands_raw) != len(right_hands_raw) or len(left_hands_raw) >= 2:
            print(f"[调试] {debug_info} -> 匹配到 {len(hands_3d)} 只手的3D位置")
        
        # 返回 hands_3d、hands_landmarks（与 hands_3d 一一对应，用于 Inspire 捏合）、可视化帧
        # 注意：hands_3d[0] 应该是用户的左手，hands_3d[1] 应该是用户的右手
        
        return hands_3d, hands_landmarks, left_frame, right_frame

    def track_hands_3d(self) -> Tuple[Optional[list], list, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        使用内部 StereoCamera 采集图像并跟踪手部3D位置。
        返回 (hands_3d, hands_landmarks, left_frame, right_frame)。
        """
        left_frame, right_frame = self.stereo_camera.read_frames()
        if left_frame is None or right_frame is None:
            return None, [], None, None
        return self._track_hands_3d_from_frames(left_frame, right_frame)

    def track_hands_3d_from_frames(
        self, left_frame: np.ndarray, right_frame: np.ndarray
    ) -> Tuple[Optional[list], list, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        从外部传入的左右图像跟踪手部3D位置。
        供 Aria SLAM 图像等外部双目源调用。
        返回 (hands_3d, hands_landmarks, left_frame, right_frame)。
        """
        if left_frame is None or right_frame is None:
            return None, [], None, None
        return self._track_hands_3d_from_frames(left_frame, right_frame)


class ArmController:
    """手臂控制器（基于手部3D位置）"""
    
    def __init__(self, mirror_mode=False):
        """初始化手臂控制器
        
        Args:
            mirror_mode: 如果为 True，当只有一只手时，镜像控制两只臂
        """
        self.target_positions = [0.0] * 30
        self.kp = 60.0
        self.kd = 1.5
        # 针对 G1 关节做一个大致的角度范围限制（弧度），防止动作过大/反关节
        # 数值是经验值，主要目的是约束而不是精确复现官方极限
        self.joint_limits = {
            # 左臂
            G1JointIndex.LeftShoulderPitch: (-1.6, 1.6),
            G1JointIndex.LeftShoulderRoll: (-1.2, 1.2),
            G1JointIndex.LeftShoulderYaw: (-1.8, 1.8),
            G1JointIndex.LeftElbow: (0.0, 2.6),
            G1JointIndex.LeftWristRoll: (-1.8, 1.8),
            G1JointIndex.LeftWristPitch: (-1.5, 1.5),
            G1JointIndex.LeftWristYaw: (-1.8, 1.8),
            # 右臂
            G1JointIndex.RightShoulderPitch: (-1.6, 1.6),
            G1JointIndex.RightShoulderRoll: (-1.2, 1.2),
            G1JointIndex.RightShoulderYaw: (-1.8, 1.8),
            G1JointIndex.RightElbow: (0.0, 2.6),
            G1JointIndex.RightWristRoll: (-1.8, 1.8),
            G1JointIndex.RightWristPitch: (-1.5, 1.5),
            G1JointIndex.RightWristYaw: (-1.8, 1.8),
        }
        # 关节变化的最大步长（rad/控制周期），用于简单限速；过大易抖，过小机械臂响应慢
        # 为了让手臂更容易跟上目标，这里适当加大
        self.max_joint_step = 0.25
        self.mirror_mode = mirror_mode  # 镜像模式：一只手控制两只臂
        
        # 手臂关节索引
        self.left_arm_joints = [
            G1JointIndex.LeftShoulderPitch,
            G1JointIndex.LeftShoulderRoll,
            G1JointIndex.LeftShoulderYaw,
            G1JointIndex.LeftElbow,
            G1JointIndex.LeftWristRoll,
            G1JointIndex.LeftWristPitch,
            G1JointIndex.LeftWristYaw,
        ]
        
        self.right_arm_joints = [
            G1JointIndex.RightShoulderPitch,
            G1JointIndex.RightShoulderRoll,
            G1JointIndex.RightShoulderYaw,
            G1JointIndex.RightElbow,
            G1JointIndex.RightWristRoll,
            G1JointIndex.RightWristPitch,
            G1JointIndex.RightWristYaw,
        ]
        
        # 机器人手臂参数（需要根据实际机器人调整）
        self.arm_base_height = 0.5  # 手臂基座高度（米）
        self.arm_reach = 0.6  # 手臂最大伸展长度（米）
        
        self.low_state = None
        self.first_update = False
        # 记录仿真启动时的初始关节姿态，用于一键回到“初始站姿”
        self.initial_positions = None

    def update_targets_absolute(self, targets_list):
        """
        接收绝对坐标并解算 IK。
        targets_list: [target_pos_left (3,), target_pos_right (3,)]
        如果某个元素为 None，则忽略该手臂。
        """
        # 处理左臂 (targets_list[0])
        if len(targets_list) >= 1 and targets_list[0] is not None:
            pos_L = targets_list[0]
            # 这里调用你的左臂 IK 解算器
            # 假设你有 self.ik_solver_left 和 self.current_q_left
            try:
                q_L = self.ik_solver_left.solve(pos_L, self.current_q_left)
                self.set_left_arm_joint_targets(q_L)
            except Exception as e:
                print(f"Left IK Error: {e}")

        # 处理右臂 (targets_list[1])
        if len(targets_list) >= 2 and targets_list[1] is not None:
            pos_R = targets_list[1]
            # 这里调用你的右臂 IK 解算器
            try:
                q_R = self.ik_solver_right.solve(pos_R, self.current_q_right)
                self.set_right_arm_joint_targets(q_R)
            except Exception as e:
                print(f"Right IK Error: {e}")
    
    def update_from_low_state(self, low_state):
        """更新机器人状态"""
        if not self.first_update:
            self.low_state = low_state
            self.initial_positions = []
            for i in range(30):
                if i < len(low_state.motor_state):
                    q_i = low_state.motor_state[i].q
                    self.target_positions[i] = q_i
                    self.initial_positions.append(q_i)
                else:
                    self.target_positions[i] = 0.0
                    self.initial_positions.append(0.0)
            self.first_update = True
            print("已读取机器人当前状态")
        else:
            self.low_state = low_state
    
    def reset_to_initial(self):
        """将目标关节位置重置为仿真启动时的初始姿态"""
        if self.initial_positions is None:
            print("尚未读取到初始姿态，无法重置")
            return
        for i in range(min(len(self.target_positions), len(self.initial_positions))):
            self.target_positions[i] = float(self.initial_positions[i])
        print("已将机器人目标关节重置为初始姿态")
    
    def compute_ik_from_hand_position(self, hand_3d_pos: list, is_left_arm: bool = True) -> list:
        """
        根据“手的相对位移”直接映射到关节角（线性近似 IK）
        
        说明：
            - hand_3d_pos 被视为相对位移 [dx, dy, dz]，数值越大关节转动越多
            - 为避免缩放过大失控，先将 dx, dy, dz 归一化到 [-1, 1] 再映射到关节角
        """
        if hand_3d_pos is None or len(hand_3d_pos) < 3:
            return [0.0] * 7

        dx_raw, dy_raw, dz_raw = hand_3d_pos

        # 使用双曲正切把输入压缩到 [-1, 1]，避免缩放系数过大导致数值爆炸，
        # 同时保证小位移时仍然近似线性
        dx = float(np.tanh(dx_raw))
        dy = float(np.tanh(dy_raw))
        dz = float(np.tanh(dz_raw))

        # 关节角线性映射系数（rad/单位归一化位移），尽量接近关节可用范围
        k_yaw = 1.8    # 左右转肩
        k_roll = 1.6   # 侧倾肩
        k_pitch = 2.0  # 前后抬肩
        k_elbow = 1.5  # 弯肘变化

        # 直观映射：
        # dx：水平左右 → 肩 yaw
        shoulder_yaw = dx * k_yaw
        # dy：图像中向上/向下 → 肩 roll/pitch（这里简单用 pitch 控制抬起/放下）
        shoulder_roll = dy * k_roll
        shoulder_pitch = dz * k_pitch

        # 肘关节：以一个基础弯曲角为中心，前后位移决定弯曲/伸直
        base_elbow = 1.5  # 中等弯曲
        elbow = base_elbow - dz * k_elbow

        # 手腕角度由上层（比如翻转手腕逻辑）单独控制，这里保持 0
        wrist_roll = 0.0
        wrist_pitch = 0.0
        wrist_yaw = 0.0

        return [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw]
    
    def update_targets_from_hands(self, hands_3d: list):
        """根据检测到的手部位置更新目标

        设计为：用户的左手控制机器人左臂，用户的右手控制机器人右臂：
        - hands_3d[0] -> 用户的左手 -> 机器人左臂
        - hands_3d[1] -> 用户的右手 -> 机器人右臂

        说明：
        - 如果只检测到一只手，则只更新对应的手臂
        - 代码已使用 MediaPipe 的 handedness 信息来确保正确配对
        """
        if not hands_3d:
            return

        # 第一个手（应该是用户的左手）→ 机器人左臂
        if len(hands_3d) >= 1:
            left_joints = self.compute_ik_from_hand_position(hands_3d[0], is_left_arm=True)
            for i, joint_idx in enumerate(self.left_arm_joints):
                if i < len(left_joints):
                    self.target_positions[joint_idx] = left_joints[i]
            
            # 镜像模式：如果只有一只手，镜像控制右臂
            if self.mirror_mode and len(hands_3d) == 1:
                # 镜像：x 取反，y 和 z 保持不变
                mirrored_pos = [-hands_3d[0][0], hands_3d[0][1], hands_3d[0][2]]
                right_joints = self.compute_ik_from_hand_position(mirrored_pos, is_left_arm=False)
                for i, joint_idx in enumerate(self.right_arm_joints):
                    if i < len(right_joints):
                        self.target_positions[joint_idx] = right_joints[i]

        # 第二个手（应该是用户的右手）→ 机器人右臂
        if len(hands_3d) >= 2:
            right_joints = self.compute_ik_from_hand_position(hands_3d[1], is_left_arm=False)
            for i, joint_idx in enumerate(self.right_arm_joints):
                if i < len(right_joints):
                    self.target_positions[joint_idx] = right_joints[i]
    
    def get_target_positions(self):
        """获取目标位置"""
        return self.target_positions.copy()


def LowCmdWrite(controller: ArmController, publisher, crc, running_flag):
    """控制循环：发布 LowCmd"""
    while running_flag[0]:
        if not controller.first_update:
            time.sleep(0.1)
            continue
        
        # 创建 LowCmd
        low_cmd = unitree_hg_msg_dds__LowCmd_()
        low_cmd.mode_pr = 0
        low_cmd.mode_machine = 0
        
        target_positions = controller.get_target_positions()
        
        # 定义腿部关节（固定）
        leg_joints = [
            G1JointIndex.LeftHipPitch, G1JointIndex.LeftHipRoll, G1JointIndex.LeftHipYaw,
            G1JointIndex.LeftKnee, G1JointIndex.LeftAnklePitch, G1JointIndex.LeftAnkleRoll,
            G1JointIndex.RightHipPitch, G1JointIndex.RightHipRoll, G1JointIndex.RightHipYaw,
            G1JointIndex.RightKnee, G1JointIndex.RightAnklePitch, G1JointIndex.RightAnkleRoll,
            G1JointIndex.WaistYaw, G1JointIndex.WaistRoll, G1JointIndex.WaistPitch,
        ]
        
        all_arm_joints = controller.left_arm_joints + controller.right_arm_joints
        
        # 设置所有关节命令
        for i in range(30):
            if controller.low_state and i < len(controller.low_state.motor_state):
                if i in leg_joints:
                    low_cmd.motor_cmd[i].q = controller.low_state.motor_state[i].q
                    low_cmd.motor_cmd[i].kp = controller.kp * 2.0
                    low_cmd.motor_cmd[i].kd = controller.kd * 2.0
                elif i in all_arm_joints:
                    # 关节角目标（期望值）
                    target_q = float(target_positions[i])
                    # 当前实际关节角
                    current_q = controller.low_state.motor_state[i].q
                    # 简单限速：每个控制周期最大角度变化 self.max_joint_step
                    max_step = getattr(controller, "max_joint_step", None)
                    if max_step is not None:
                        delta = np.clip(target_q - current_q, -max_step, max_step)
                        cmd_q = current_q + float(delta)
                    else:
                        cmd_q = target_q
                    # 关节角度裁剪到安全范围内（如果配置了的话）
                    limits = getattr(controller, "joint_limits", {}).get(i)
                    if limits is not None:
                        q_min, q_max = limits
                        cmd_q = float(np.clip(cmd_q, q_min, q_max))
                    low_cmd.motor_cmd[i].q = cmd_q
                    low_cmd.motor_cmd[i].kp = controller.kp
                    low_cmd.motor_cmd[i].kd = controller.kd
                elif i == G1JointIndex.kNotUsedJoint:
                    low_cmd.motor_cmd[i].q = 1.0
                    low_cmd.motor_cmd[i].kp = 0.0
                    low_cmd.motor_cmd[i].kd = 0.0
                else:
                    low_cmd.motor_cmd[i].q = controller.low_state.motor_state[i].q
                    low_cmd.motor_cmd[i].kp = controller.kp
                    low_cmd.motor_cmd[i].kd = controller.kd
            else:
                low_cmd.motor_cmd[i].q = float(target_positions[i]) if i < len(target_positions) else 0.0
                low_cmd.motor_cmd[i].kp = controller.kp
                low_cmd.motor_cmd[i].kd = controller.kd
            
            low_cmd.motor_cmd[i].dq = 0.0
            low_cmd.motor_cmd[i].tau = 0.0
        
        low_cmd.crc = crc.Crc(low_cmd)
        publisher.Write(low_cmd)
        time.sleep(0.02)  # 50Hz


def LowStateHandler(controller: ArmController, msg: LowState_):
    """LowState 回调函数"""
    controller.update_from_low_state(msg)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="基于双目摄像头的手臂遥操作")
    parser.add_argument("--left_camera", type=int, default=0, help="左摄像头设备ID")
    parser.add_argument("--right_camera", type=int, default=1, help="右摄像头设备ID")
    parser.add_argument("--baseline", type=float, default=0.12, help="双目基线距离（米）")
    parser.add_argument("--focal_length", type=float, default=640.0, help="焦距（像素）")
    args = parser.parse_args()
    
    print("=" * 70)
    print("基于双目摄像头的手臂遥操作系统")
    print("=" * 70)
    
    # 初始化 DDS
    print("初始化 DDS 通信...")
    ChannelFactoryInitialize(1)
    
    arm_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
    arm_publisher.Init()
    
    controller = ArmController()
    
    def state_handler(msg):
        LowStateHandler(controller, msg)
    
    state_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    state_subscriber.Init(state_handler, 10)
    
    print("DDS 通信初始化完成")
    
    # 初始化双目摄像头
    print("初始化双目摄像头...")
    stereo_camera = StereoCamera(args.left_camera, args.right_camera)
    stereo_camera.baseline = args.baseline
    stereo_camera.focal_length = args.focal_length
    
    if not stereo_camera.init():
        print("摄像头初始化失败，退出程序")
        exit(1)
    
    # 初始化手部跟踪器
    print("初始化手部跟踪器...")
    hand_tracker = HandTracker()
    stereo_hand_tracker = StereoHandTracker(stereo_camera, hand_tracker)
    
    print("等待接收机器人状态...")
    while not controller.first_update:
        time.sleep(0.1)
    
    print("=" * 70)
    print("系统已就绪，开始遥操作")
    print("按 'q' 键退出程序")
    print("=" * 70)
    
    # 启动控制线程
    running_flag = [True]
    crc = CRC()
    
    control_thread = threading.Thread(
        target=LowCmdWrite,
        args=(controller, arm_publisher, crc, running_flag),
        daemon=True
    )
    control_thread.start()
    
    try:
        while running_flag[0]:
            # 跟踪手部
            hands_3d, left_frame, right_frame = stereo_hand_tracker.track_hands_3d()
            
            hands_3d, hands_landmarks, left_frame, right_frame = stereo_hand_tracker.track_hands_3d()

            if hands_3d is not None and left_frame is not None and right_frame is not None:
                # 更新控制器目标
                controller.update_targets_from_hands(hands_3d)
                
                # 显示图像
                cv2.imshow("Left Camera", left_frame)
                cv2.imshow("Right Camera", right_frame)
                
                # 显示3D位置信息
                if len(hands_3d) > 0:
                    for i, hand_pos in enumerate(hands_3d):
                        print(f"手部 {i+1} 3D位置: x={hand_pos[0]:.3f}m, y={hand_pos[1]:.3f}m, z={hand_pos[2]:.3f}m")
            
            # 检查退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("退出程序...")
                running_flag[0] = False
                break
            
            time.sleep(0.033)  # ~30Hz
    
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        running_flag[0] = False
    except Exception as e:
        print(f"\n程序错误: {e}")
        import traceback
        traceback.print_exc()
        running_flag[0] = False
    finally:
        cv2.destroyAllWindows()
        stereo_camera.release()
        print("程序结束")
