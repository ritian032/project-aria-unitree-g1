#!/usr/bin/env python3
"""
G1 双臂 IK 独立封装：单独使用 xr_teleoperate 的 G1_29_ArmIK，不依赖摄像头或 DDS。

依赖：xr_teleoperate 及其环境（pinocchio, casadi 等）。可复用 unitree_sim 或 xr_teleoperate 的 conda 环境。

用法 1 - 作为模块导入：

    from g1_arm_ik_standalone import get_g1_arm_ik

    ik = get_g1_arm_ik()
    left_tf = np.eye(4);  left_tf[:3, 3] = [0.35, 0.25, 0.22]   # X前, Y左, Z上
    right_tf = np.eye(4); right_tf[:3, 3] = [0.35, -0.25, 0.22]
    current_q = np.zeros(14)   # 可选，当前 14 个臂关节角
    current_dq = np.zeros(14)   # 可选
    sol_q, sol_tauff = ik.solve_ik(left_tf, right_tf, current_q, current_dq)

用法 2 - 命令行测试：

    python g1_arm_ik_standalone.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

# 默认 xr_teleoperate 路径，可通过环境变量 XR_TELEOP_ROOT 覆盖
XR_TELEOP_ROOT = os.environ.get("XR_TELEOP_ROOT", "/home/ritian/xr_teleoperate")
TELEOP_DIR = os.path.join(XR_TELEOP_ROOT, "teleop")


def get_g1_arm_ik(visualization: bool = False):
    """
    创建 G1_29_ArmIK 实例，自动处理工作目录与路径。
    返回的实例可直接调用 .solve_ik(left_tf, right_tf, current_q=None, current_dq=None)。

    Args:
        visualization: 是否开启 Meshcat 可视化（默认 False）

    Returns:
        G1_29_ArmIK 实例
    """
    if not os.path.isdir(TELEOP_DIR):
        raise FileNotFoundError(
            f"xr_teleoperate/teleop 目录不存在: {TELEOP_DIR}，"
            "请设置环境变量 XR_TELEOP_ROOT 或安装 xr_teleoperate"
        )
    if XR_TELEOP_ROOT not in sys.path:
        sys.path.insert(0, XR_TELEOP_ROOT)

    from teleop.robot_control.robot_arm_ik import G1_29_ArmIK

    old_cwd = os.getcwd()
    try:
        os.chdir(TELEOP_DIR)
        ik = G1_29_ArmIK(Unit_Test=False, Visualization=visualization)
    finally:
        os.chdir(old_cwd)
    return ik


def solve_dual_arm_ik(
    left_tf: np.ndarray,
    right_tf: np.ndarray,
    current_q: np.ndarray | None = None,
    current_dq: np.ndarray | None = None,
    ik_instance=None,
):
    """
    双臂 IK 求解：给定左右末端 4x4 位姿，返回 14 维关节角与前馈力矩。

    Args:
        left_tf: 左臂末端 4x4 齐次变换（机器人坐标系）
        right_tf: 右臂末端 4x4 齐次变换
        current_q: 当前 14 个臂关节角，None 则用内部初值
        current_dq: 当前 14 个臂关节角速度，None 则用 0
        ik_instance: 已创建的 G1_29_ArmIK；None 则内部创建一次（有状态，多次调用会复用）

    Returns:
        (sol_q, sol_tauff): 各为 shape (14,) 的 numpy 数组
    """
    if ik_instance is None:
        ik_instance = get_g1_arm_ik()
    left_tf = np.asarray(left_tf, dtype=np.float64)
    right_tf = np.asarray(right_tf, dtype=np.float64)
    if current_q is not None:
        current_q = np.asarray(current_q, dtype=np.float64).ravel()
        if current_q.size != 14:
            raise ValueError("current_q 长度须为 14")
    if current_dq is not None:
        current_dq = np.asarray(current_dq, dtype=np.float64).ravel()
        if current_dq.size != 14:
            raise ValueError("current_dq 长度须为 14")
    return ik_instance.solve_ik(left_tf, right_tf, current_q, current_dq)


if __name__ == "__main__":
    print("G1 双臂 IK 独立测试（home 位姿）")
    ik = get_g1_arm_ik()
    home_l = np.array([0.35, 0.25, 0.22])
    home_r = np.array([0.35, -0.25, 0.22])
    left_tf = np.eye(4, dtype=np.float64)
    left_tf[:3, 3] = home_l
    right_tf = np.eye(4, dtype=np.float64)
    right_tf[:3, 3] = home_r
    current_q = np.zeros(14)
    current_dq = np.zeros(14)
    sol_q, sol_tauff = ik.solve_ik(left_tf, right_tf, current_q, current_dq)
    print("sol_q (14 个臂关节角):", sol_q)
    print("sol_tauff (14 维前馈力矩):", sol_tauff)
    print("OK")
