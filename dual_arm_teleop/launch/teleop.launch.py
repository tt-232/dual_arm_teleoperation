"""
Teleop launch file.

Starts the hand-tracking teleop node.
Arm control uses /compute_ik service directly (no MoveIt Servo needed).

Run AFTER the main demo is up:
  ros2 launch dual_arm_panda_moveit_config demo.launch.py
  ros2 launch dual_arm_teleop teleop.launch.py
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    teleop_share = get_package_share_directory('dual_arm_teleop')
    teleop_params = os.path.join(teleop_share, 'config', 'teleop_params.yaml')

    teleop_node = Node(
        package='dual_arm_teleop',
        executable='teleop_node',
        name='dual_arm_teleop',
        output='screen',
        parameters=[teleop_params],
    )

    return LaunchDescription([teleop_node])
