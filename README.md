# robot-ros-system

Sequencer and deliberator of the robot (ROS 2 Jazzy, RPi 5). Architecture and decisions: `robot-docs`.

| Package | Content |
| --- | --- |
| `robot_verification` | Pass/fail criterion of each bench verification, evaluated on a `ros2 bag` (regression) |

Build in a colcon workspace together with `robot-interfaces`: `colcon build --symlink-install`.
