# robot-ros-system

Sequencer and deliberator of the robot (ROS 2 Jazzy, RPi 5). Architecture and decisions: `robot-docs`.

| Package | Content |
| --- | --- |
| `robot_verification` | Link check, link load measurement, and the pass/fail criterion of each bench verification on a `ros2 bag` (regression) |
| `robot_bringup` | Launch files and configuration: the deliberator in the test configuration (map M, Theta*, static map -> odom, `odom_to_tf_ros2`), and in V-08 C1 with the LD19 and `slam_toolbox` as load |

Build in a colcon workspace together with `robot-interfaces`: `colcon build --symlink-install`.

Lint and format: Ruff 0.17.2 (`pipx install ruff==0.17.2`), configured in `ruff.toml`. Run `ruff format` and `ruff check` before committing; `colcon test` checks both.

## RPi environment

- ROS packages are held during the bench phase (`apt-mark hold 'ros-jazzy-*'`), so results come from a fixed stack. A ROS package installed later must be held too.
- Everything built from source lives in one workspace, `~/robot_ws`, including the micro-ROS agent (`micro-ROS-Agent` and `micro_ros_msgs`, branch `jazzy`).
- `odom_to_tf_ros2` is not in the Jazzy apt repository: it is built from source in `~/robot_ws`, pinned at tag `1.1.1` (`git clone --branch 1.1.1 --depth 1 https://github.com/gstavrinos/odom_to_tf_ros2.git`).
- `ldlidar_stl_ros2` (LD19 driver) is not in the Jazzy apt repository: it is built from source in `~/robot_ws`, pinned at tag `v3.0.3` (`git clone --branch v3.0.3 --depth 1 https://github.com/ldrobotSensorTeam/ldlidar_stl_ros2.git`). The tag does not compile with GCC 13 (Ubuntu 24.04), which no longer includes `pthread.h` indirectly; patch one line before building: `sed -i 's|^//#include <pthread.h>|#include <pthread.h>|' ldlidar_driver/include/logger/log_module.h`.
- After any ROS update (unhold, upgrade): `colcon build` the whole workspace, then re-hold. A binary built against older ROS libraries crashes at run time (robot-docs, nota 0005).
- At the start of each bench session and after any update: `ros2 run robot_verification check_link`.
