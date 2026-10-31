"""Launch the whole robot on the RPi in the operation configuration: sequencer and deliberator.

The micro-ROS agent and the sequencer run in SCHED_FIFO 50 (ADR 039); the sequencer waits for
/goal_pose (mission by plan). slam_toolbox maps continuously and publishes /map and map -> odom;
the planner plans over that map, with the obstacle layer over /scan (ADRs 040 and 041). Not used
by the verifications, which use the test configuration (F7 D1); it is the source of the rqt_graph
of V-09.

Usage: ros2 launch robot_bringup robot.launch.py

Criticality: T2, product (robot-docs codigo.md).
"""

from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

PACKAGE_SHARE = Path(get_package_share_directory('robot_bringup'))
LIDAR_LAUNCH = str(PACKAGE_SHARE / 'launch' / 'lidar.launch.py')
# The test configuration first, then the operation overrides: the last file wins.
DELIBERATOR_PARAMETERS = [
    str(PACKAGE_SHARE / 'config' / 'deliberator_test.yaml'),
    str(PACKAGE_SHARE / 'config' / 'deliberator_operation.yaml'),
]
SLAM_SHARE = Path(get_package_share_directory('slam_toolbox'))
SLAM_PARAMETERS = [
    str(SLAM_SHARE / 'config' / 'mapper_params_online_async.yaml'),
    str(PACKAGE_SHARE / 'config' / 'slam.yaml'),
]
# In activation order: the costmap of the planner waits for the map -> odom of the SLAM.
MANAGED_NODES = ['slam_toolbox', 'planner_server']

AGENT_EXECUTABLE = str(
    Path(get_package_prefix('micro_ros_agent')) / 'lib' / 'micro_ros_agent' / 'micro_ros_agent'
)
AGENT_ARGUMENTS = ['serial', '--dev', '/dev/ttyAMA0', '-b', '921600']  # UART of ADR 043
REALTIME_PREFIX = ['chrt', '-f', '50']  # ADR 039; needs rtprio 50 in limits.d on the RPi


def build_micro_ros_agent() -> ExecuteProcess:
    """Start the agent as a plain process: it is not a ROS node and takes no --ros-args."""
    return ExecuteProcess(cmd=[*REALTIME_PREFIX, AGENT_EXECUTABLE, *AGENT_ARGUMENTS])


def build_sequencer() -> Node:
    """Without the mission parameter, the sequencer waits for /goal_pose (arquitetura.md)."""
    return Node(
        package='robot_sequencer',
        executable='sequencer',
        name='sequencer',
        prefix=' '.join(REALTIME_PREFIX),
    )


def build_lidar() -> IncludeLaunchDescription:
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(LIDAR_LAUNCH))


def build_slam_toolbox() -> Node:
    return Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        parameters=SLAM_PARAMETERS,
    )


def build_odom_to_base_link() -> Node:
    return Node(
        package='odom_to_tf_ros2',
        executable='odom_to_tf',
        name='odom_to_tf',
        parameters=DELIBERATOR_PARAMETERS,
    )


def build_planner_server() -> Node:
    return Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        parameters=DELIBERATOR_PARAMETERS,
    )


def build_lifecycle_manager() -> Node:
    return Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager',
        parameters=[{'autostart': True, 'node_names': MANAGED_NODES}],
    )


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            build_micro_ros_agent(),
            build_sequencer(),
            build_lidar(),
            build_slam_toolbox(),
            build_odom_to_base_link(),
            build_planner_server(),
            build_lifecycle_manager(),
        ]
    )
