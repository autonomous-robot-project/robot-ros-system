# robot-ros-system

Sequencer and deliberator of the robot (ROS 2 Jazzy, RPi 5). Architecture and decisions: `robot-docs`.

| Package | Content |
| --- | --- |
| `robot_verification` | Link check, and the pass/fail criterion of each bench verification on a `ros2 bag` (regression) |

Build in a colcon workspace together with `robot-interfaces`: `colcon build --symlink-install`.

Lint and format: Ruff 0.17.2 (`pipx install ruff==0.17.2`), configured in `ruff.toml`. Run `ruff format` and `ruff check` before committing; `colcon test` checks both.

## RPi environment

- ROS packages are held during the bench phase (`apt-mark hold 'ros-jazzy-*'`), so results come from a fixed stack. A ROS package installed later must be held too.
- Everything built from source lives in one workspace, `~/robot_ws`, including the micro-ROS agent (`micro-ROS-Agent` and `micro_ros_msgs`, branch `jazzy`).
- After any ROS update (unhold, upgrade): `colcon build` the whole workspace, then re-hold. A binary built against older ROS libraries crashes at run time (robot-docs, nota 0005).
- At the start of each bench session and after any update: `ros2 run robot_verification check_link`.
