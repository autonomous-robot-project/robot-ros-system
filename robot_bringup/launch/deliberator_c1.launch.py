"""Launch the deliberator of V-08 C1: the test configuration of V-04 plus the SLAM as load.

Includes deliberator_test.launch.py unchanged: the planner keeps planning over M, with map -> odom
the static identity (ADR 035). Adds the LiDAR (lidar.launch.py) and slam_toolbox, which processes
/scan into its own frame and topic, consumed by nobody, and publishes no transform (ADR 040).
The costmap has no obstacle layer, so /scan is only load (test configuration, robot-docs
arquitetura.md).

Usage: ros2 launch robot_bringup deliberator_c1.launch.py

Criticality: T2, product (robot-docs codigo.md).
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

PACKAGE_SHARE = Path(get_package_share_directory('robot_bringup'))
DELIBERATOR_TEST_LAUNCH = str(PACKAGE_SHARE / 'launch' / 'deliberator_test.launch.py')
LIDAR_LAUNCH = str(PACKAGE_SHARE / 'launch' / 'lidar.launch.py')
SLAM_PARAMETERS = str(PACKAGE_SHARE / 'config' / 'slam.yaml')
SLAM_LOAD_PARAMETERS = str(PACKAGE_SHARE / 'config' / 'slam_load.yaml')
SLAM_SHARE = Path(get_package_share_directory('slam_toolbox'))
SLAM_DEFAULT_PARAMETERS = str(SLAM_SHARE / 'config' / 'mapper_params_online_async.yaml')


def build_included(launch_file: str) -> IncludeLaunchDescription:
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(launch_file))


def build_slam_toolbox() -> Node:
    """Load the package defaults, then slam.yaml, then slam_load.yaml: the last one wins."""
    return Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        parameters=[SLAM_DEFAULT_PARAMETERS, SLAM_PARAMETERS, SLAM_LOAD_PARAMETERS],
    )


def build_slam_lifecycle_manager() -> Node:
    """Activate slam_toolbox, apart from the manager of deliberator_test.launch.py."""
    return Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_slam',
        parameters=[{'autostart': True, 'node_names': ['slam_toolbox']}],
    )


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            build_included(DELIBERATOR_TEST_LAUNCH),
            build_included(LIDAR_LAUNCH),
            build_slam_toolbox(),
            build_slam_lifecycle_manager(),
        ]
    )
