"""Launch the LD19 driver and base_link -> laser: the LiDAR of every configuration.

Included by deliberator_c1.launch.py (V-08 C1, where /scan is only the load of the SLAM) and by
robot.launch.py (operation), so the pose of the laser in base_link has a single source.

Usage: included; alone, ros2 launch robot_bringup lidar.launch.py

Criticality: T2, product (robot-docs codigo.md).
"""

from launch import LaunchDescription
from launch_ros.actions import Node

LASER_FRAME = 'laser'
# Pose of the LD19 optical centre in base_link: a medir (F8 I0, #25), with a ruler from the
# centre of the wheel axle. It does not change the load of C1.
LASER_X = 0.0  # m, x_l
LASER_Y = 0.0  # m, y_l
LASER_Z = 0.0  # m, z_l
LASER_YAW = 0.0  # rad, psi_l
LD19_PORT = '/dev/ttyUSB0'  # CP2102 USB adapter of the LD19 (nota 0019)
LD19_BAUD_RATE = 230400  # baud, the LD19 serial rate (driver ld19.launch.py)


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


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([build_ld19_driver(), build_base_link_to_laser()])
