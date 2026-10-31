"""Launch the deliberator of V-08 C1: the test configuration of V-04 plus the SLAM as load.

Includes deliberator_test.launch.py unchanged: the planner keeps planning over M, with map -> odom
the static identity (ADR 035). Adds the LD19 driver, base_link -> laser and slam_toolbox, which
processes /scan into its own frame and topic, consumed by nobody, and publishes no transform
(ADR 040, F8 D1). The costmap has no obstacle layer, so /scan is only load (F7 D1).

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
SLAM_LOAD_PARAMETERS = str(PACKAGE_SHARE / 'config' / 'slam_load.yaml')
SLAM_SHARE = Path(get_package_share_directory('slam_toolbox'))
SLAM_DEFAULT_PARAMETERS = str(SLAM_SHARE / 'config' / 'mapper_params_online_async.yaml')

LASER_FRAME = 'laser'
# Pose of the LD19 optical centre in base_link: a medir (F8 I0, #25), with a ruler from the
# centre of the wheel axle. It does not change the load of C1.
LASER_X = 0.0  # m, x_l
LASER_Y = 0.0  # m, y_l
LASER_Z = 0.0  # m, z_l
LASER_YAW = 0.0  # rad, psi_l
LD19_PORT = '/dev/ttyUSB0'  # a confirmar (F8 I0): LD19 on a USB adapter
LD19_BAUD_RATE = 230400  # baud, the LD19 serial rate (driver ld19.launch.py)


def build_deliberator_test() -> IncludeLaunchDescription:
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(DELIBERATOR_TEST_LAUNCH))


def build_ld19_driver() -> Node:
    return Node(
        package='ldlidar_stl_ros2',
        executable='ldlidar_stl_ros2_node',
        name='ld19',
        parameters=[
            {
                'product_name': 'LDLiDAR_LD19',
                'topic_name': 'scan',
                'frame_id': LASER_FRAME,
                'port_name': LD19_PORT,
                'port_baudrate': LD19_BAUD_RATE,
                'laser_scan_dir': True,  # counterclockwise, as REP 103
                'enable_angle_crop_func': False,
            }
        ],
    )


def build_base_link_to_laser() -> Node:
    return Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_link_to_laser',
        arguments=[
            '--x',
            str(LASER_X),
            '--y',
            str(LASER_Y),
            '--z',
            str(LASER_Z),
            '--yaw',
            str(LASER_YAW),
            '--frame-id',
            'base_link',
            '--child-frame-id',
            LASER_FRAME,
        ],
    )


def build_slam_toolbox() -> Node:
    """Load the package defaults first, then the overrides of slam_load.yaml: the last wins."""
    return Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        parameters=[SLAM_DEFAULT_PARAMETERS, SLAM_LOAD_PARAMETERS],
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
            build_deliberator_test(),
            build_ld19_driver(),
            build_base_link_to_laser(),
            build_slam_toolbox(),
            build_slam_lifecycle_manager(),
        ]
    )
