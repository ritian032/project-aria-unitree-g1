# Aria Stereo → Unitree G1 Bimanual Teleoperation (Isaac Sim)

This repository provides a **Project Aria SLAM stereo–driven** teleoperation module to control the **Unitree G1 humanoid dual arms** in **Isaac Sim / Isaac Lab**.

本仓库提供一套基于 **Project Aria SLAM 双目灰度图** 的遥操作模块，用于在 **Isaac Sim / Isaac Lab** 仿真中控制 **Unitree G1 双臂**。

---

## 1. Highlights / 特点

- **Egocentric stereo input** from Project Aria (SLAM left/right images)  
  使用 Aria 的第一人称 SLAM 左右目灰度图作为双目输入
- **MediaPipe Hands** for 2D keypoints + **simple stereo triangulation** for 3D wrist position  
  MediaPipe 手部关键点检测 + 基于视差的简单双目三角测量得到 3D 手位置
- **Lightweight IK** mapping from 3D hand to G1 arm joints  
  使用简化几何 IK 将手部 3D 位置映射为 G1 上肢关节角
- **DDS-based control** compatible with Unitree SDK2 Python  
  基于 `unitree_sdk2_python` + CycloneDDS，通过 `rt/lowcmd` / `rt/lowstate` 实现控制与反馈
- **Mirror mode**: if only one hand is detected, control both arms symmetrically  
  镜像模式：只有一只手时也能对称驱动双臂（便于演示与容错）

---

## 2. Repository Structure / 仓库结构

Core files / 核心文件：

- `aria_stereo_teleop_g1.py`  
  Main entry. Initializes DDS, subscribes Aria images, runs tracking+IK, publishes `LowCmd`.  
  主程序：初始化 DDS，订阅 Aria 图像，运行追踪+IK，发布 LowCmd。

- `aria_stereo_source.py`  
  Aria StreamingClient wrapper. Provides SLAM stereo pairs.  
  Aria 图像源封装：输出 SLAM 左右目帧。

- `stereo_teleop_arm.py`  
  HandTracker (MediaPipe), StereoHandTracker (3D via disparity), ArmController (IK + joint targets).  
  手部检测 + 双目 3D + IK + 关节目标更新的核心实现。

---

## 3. Dependencies / 依赖

This repo is meant to be used **on top of** the following projects:

- Unitree simulation repo: `unitree_sim_isaaclab`  
- Unitree SDK2 Python: `unitree_sdk2_python`  
- NVIDIA Isaac Sim 5.0 and Isaac Lab v2.2  
- Project Aria Tools / SDK (`aria.sdk`)

本仓库是“上层模块”，依赖以下工程：

- `unitree_sim_isaaclab`（G1 仿真与任务）
- `unitree_sdk2_python`（DDS 控制与消息）
- Isaac Sim 5.0 + Isaac Lab v2.2
- Project Aria SDK（`aria.sdk`）

### Python packages (this module)
We recommend pinning:

bash
pip install --no-deps "mediapipe==0.10.14"
pip install "numpy==1.26.0" "opencv-python<5"


> Note: Isaac Sim requires NumPy 1.x (e.g., 1.26.0). Avoid NumPy 2.x.

---

## 4. Setup / 安装

### 4.1 Install Isaac Sim + Isaac Lab
Follow the official `unitree_sim_isaaclab` installation guide (Isaac Sim 5.0 + Isaac Lab v2.2).

按照 `unitree_sim_isaaclab` 项目文档安装 Isaac Sim 5.0 + Isaac Lab v2.2。

### 4.2 Install CycloneDDS and Unitree SDK2 Python
Build CycloneDDS, set environment variables, then install `unitree_sdk2_python`.

编译 CycloneDDS，设置环境变量，然后安装 `unitree_sdk2_python`。

Example:

bash
export CYCLONEDDS_HOME="$HOME/cyclonedds/install"
export CMAKE_PREFIX_PATH="$HOME/cyclonedds/install:$CMAKE_PREFIX_PATH"


### 4.3 Install Project Aria SDK
Install `aria.sdk` following the official Project Aria Tools documentation.

按官方文档安装 `aria.sdk`。

---

## 5. Run / 运行

You need **two terminals**.





### Terminal 1: Start simulation / 启动仿真

conda activate unitree_sim_env
export CYCLONEDDS_HOME="$HOME/cyclonedds/install"
export CMAKE_PREFIX_PATH="$HOME/cyclonedds/install:$CMAKE_PREFIX_PATH"

cd ~/unitree_sim_isaaclab
python sim_main.py --device cpu --enable_cameras \
  --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint \
  --enable_dex3_dds \
  --robot_type g129

Wait until Isaac Sim window loads and G1 appears.


Terminal 2: Start Aria teleoperation / 启动 Aria 遥操
Make sure Aria streaming is started on the device, then:

conda activate unitree_sim_envexport CYCLONEDDS_HOME="$HOME/cyclonedds/install"export CMAKE_PREFIX_PATH="$HOME/cyclonedds/install:$CMAKE_PREFIX_PATH"cd <this_repo>python aria_stereo_teleop_g1.py

Expected behavior / 预期现象：
Two windows: Aria Left SLAM, Aria Right SLAM
Terminal prints hands_3d and arm targets
G1 arms move in simulation
Exit: press q in image window or Ctrl+C.

6. Notes / 注意事项
6.1 Two-hand control is not always stable
Because SLAM grayscale images can be low-contrast or blurred, MediaPipe may detect only one hand in one of the stereo views, leading to only one 3D hand point.
This repo provides an optional mirror mode to still demonstrate bimanual motions with one detected hand.
由于 SLAM 灰度图质量受限，左右图经常出现“只检测到一侧有手”的情况，导致只能得到 1 个 3D 手点。
因此提供了 镜像模式：只有一只手时也能驱动双臂对称动作。
6.2 NumPy compatibility
Isaac Sim requires NumPy 1.x. If you see ABI errors, pin numpy==1.26.0.


Troubleshooting

## 9. Troubleshooting / 问题排查与复现说明

This section summarizes the most common issues we encountered when reproducing the system (NumPy conflicts, MediaPipe version, DDS config, Aria detection), and how to fix them.

本节总结了在复现过程中容易遇到的几个问题（NumPy 冲突、MediaPipe 版本、DDS 配置、Aria 检测等）以及对应的解决办法，方便你或使用者快速排查。

---

### 9.1 NumPy / Isaac Sim ABI conflict

**Symptom / 现象：**

- Running `sim_main.py` or importing `pinocchio`, `scipy` in the Isaac Sim environment gives errors like:

A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
...
ImportError: numpy._core.multiarray failed to import
# project-aria-unitree-g1
