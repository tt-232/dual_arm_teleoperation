import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():

    moveit_config = (
        MoveItConfigsBuilder("dual_arm_panda")
        .robot_description(file_path="config/panda.urdf.xacro")
        .robot_description_semantic(file_path="config/panda.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    # Start the actual move_group node/action server
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    # RViz
    rviz_config = os.path.join(
        get_package_share_directory("dual_arm_panda_moveit_config"),
        "launch/moveit.rviz",
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
        ],
    )

    # Publish TF
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description],
    )

    # ros2_control using mock hardware (FakeSystem)
    ros2_controllers_path = os.path.join(
        get_package_share_directory("dual_arm_panda_moveit_config"),
        "config/",
        "ros2_controllers.yaml",
    )
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[ros2_controllers_path],
        remappings=[
            ("/controller_manager/robot_description", "/robot_description"),
        ],
        output="both",
    )

    # Spawn controllers — use list form for cmd to avoid shell quoting issues
    def make_spawner(controller_name):
        return TimerAction(
            period=2.0,
            actions=[
                ExecuteProcess(
                    cmd=["ros2", "run", "controller_manager", "spawner", controller_name],
                    output="screen",
                )
            ],
        )

    # Delay move_group so controller_manager is ready before Ros2ControlManager connects
    move_group_node_delayed = TimerAction(
        period=5.0,
        actions=[move_group_node],
    )

    return LaunchDescription(
        [
            robot_state_publisher,
            ros2_control_node,
            move_group_node_delayed,
            rviz_node,
            make_spawner("joint_state_broadcaster"),
            make_spawner("left_arm_controller"),
            make_spawner("right_arm_controller"),
            make_spawner("left_gripper_controller"),
            make_spawner("right_gripper_controller"),
        ]
    )
