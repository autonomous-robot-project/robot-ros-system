"""Launch the deliberator in the test configuration of V-04: the planner over the map M.

One process per node (robot-docs arquitetura.md). The map_server serves M, the planner_server
answers ComputePathToPose with Theta* (ADRs 014 and 034), and the lifecycle_manager activates
both. map -> odom is the identity, static (ADR 035), and odom_to_tf_ros2 turns the controller's
odometry into odom -> base_link (ADR 026). Not launching this file is turning the deliberator
off (RF-05).

Usage: ros2 launch robot_bringup deliberator_test.launch.py

Criticality: T2, product (robot-docs codigo.md).
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

PACKAGE_SHARE = Path(get_package_share_directory('robot_bringup'))
PARAMETERS = str(PACKAGE_SHARE / 'config' / 'deliberator_test.yaml')
MAP_M = str(PACKAGE_SHARE / 'maps' / 'm.yaml')
MANAGED_NODES = ['map_server', 'planner_server']


def build_map_to_odom() -> Node:
    """Identity and static: the test map M has its origin at the odometry origin (ADR 035)."""
    return Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom',
        arguments=['--frame-id', 'map', '--child-frame-id', 'odom'],
    )


def build_odom_to_base_link() -> Node:
    return Node(
        package='odom_to_tf_ros2',
        executable='odom_to_tf',
        name='odom_to_tf',
        parameters=[PARAMETERS],
    )


def build_map_server() -> Node:
    return Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        parameters=[{'yaml_filename': MAP_M}],
    )


def build_planner_server() -> Node:
    return Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        parameters=[PARAMETERS],
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
            build_map_to_odom(),
            build_odom_to_base_link(),
            build_map_server(),
            build_planner_server(),
            build_lifecycle_manager(),
        ]
    )
