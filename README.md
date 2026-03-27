# Dual-Arm Panda Teleoperation

A ROS 2 Humble workspace for real-time teleoperation of dual Franka Panda arms using camera-based hand tracking. The operator controls both end-effectors and grippers using hand gestures captured from a standard RGB camera.

---
## Demo



## Overview

| Feature | Details |
|---|---|
| Robot | Dual 7-DOF Franka Panda arms with 2-finger grippers |
| Visualization | RViz2 with MoveIt 2 motion planning |
| Tracking | MediaPipe Hands (21 landmarks per hand) |
| Control | 2D end-effector translation + 1D wrist rotation + gripper open/close |
| IK Solver | KDL via MoveIt 2 `/compute_ik` service |
| Noise handling | Exponential Moving Average (EMA) smoothing, dead-band filtering |

---

## Repository Structure

```
sereact_ws/src/
├── dual_arm_teleop/          # Teleoperation node (hand tracking -> IK -> joint trajectory)
│   ├── dual_arm_teleop/
│   │   ├── teleop_node.py    # Main ROS 2 node
│   │   └── hand_tracker.py   # MediaPipe wrapper with EMA smoothing
│   ├── config/
│   │   └── teleop_params.yaml
│   └── launch/
│       └── teleop.launch.py
└── moveit_resources/         # Dual-arm Panda MoveIt config (third-party, see Citation)
    └── dual_arm_panda_moveit_config/
```

---

## System Requirements

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10+
- A computer with an RGB camera

---

## Dependencies

### ROS 2 Humble

Install following this [tutorial](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)

### ROS 2 packages

Install via `apt`:

```bash
sudo apt install -y \
  ros-humble-moveit \
  ros-humble-moveit-ros-planning-interface \
  ros-humble-moveit-simple-controller-manager \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-joint-state-broadcaster \
  ros-humble-joint-trajectory-controller \
  ros-humble-xacro \
  ros-humble-kdl-parser \
  ros-humble-tf2-ros \
  ros-humble-tf2-geometry-msgs \
  ros-humble-rviz2
```

### Python packages

```bash
pip install mediapipe opencv-python numpy
```

---

## Environment Setup

### 1. Clone / set up the workspace

```bash
mkdir -p ~/sereact_ws/src
cd ~/sereact_ws/src
# Place or clone this repository here
```

### 2. Build

```bash
cd ~/sereact_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```
---

## Running

Two terminals are required.

### Terminal 1 — Robot visualization and MoveIt

```bash
cd ~/sereact_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch dual_arm_panda_moveit_config demo.launch.py
```

This starts:
- `robot_state_publisher` (URDF + TF)
- `ros2_control_node` with mock hardware
- `joint_state_broadcaster`, `left_arm_controller`, `right_arm_controller`, `left_gripper_controller`, `right_gripper_controller`
- `move_group` (MoveIt 2, provides `/compute_ik`)
- RViz2

### Terminal 2 — Teleoperation node

```bash
cd ~/sereact_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch dual_arm_teleop teleop.launch.py
```

A camera window opens showing the hand tracking window.

---

## Teleoperation Controls

| Gesture | Action |
|---|---|
| **Point index finger up** (hold ~0.7 s) | Activate arm control |
| **Close fist** (hold ~0.7 s) | Deactivate arm control |
| **Move hand left / right** | End-effector moves left / right (world Y axis) |
| **Move hand up / down** | End-effector moves up / down (world Z axis) |
| **Tilt wrist** (left/right) | Rotate end-effector around world X axis |
| **Open / close thumb-index pinch** | Open / close gripper |
| **Q** (in camera window) | Quit |

The camera view displays each arm's activation status (green = ACTIVE, red = INACTIVE) and a gesture progress bar while a gesture is being held.

---

## Third-Party Code

### MoveIt Resources

The MoveIt configuration for the dual Panda arm setup is based on the `dual_arm_panda_moveit_config` package from the **MoveIt Resources** repository:

```
MoveIt Resources (ros2 branch)
https://github.com/moveit/moveit_resources
License: BSD 3-Clause
Maintainers: MoveIt contributors
```

### MediaPipe

Hand tracking uses **MediaPipe Hands** by Google:

```
MediaPipe
https://github.com/google-ai-edge/mediapipe
License: Apache 2.0
```
