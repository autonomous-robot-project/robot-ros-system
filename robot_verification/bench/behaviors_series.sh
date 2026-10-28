#!/bin/bash
set -m
source /opt/ros/jazzy/setup.bash; source ~/robot_ws/install/setup.bash
BAG=$1
AGENT=$(ros2 pkg prefix micro_ros_agent)/lib/micro_ros_agent/micro_ros_agent
$AGENT serial --dev /dev/ttyAMA0 -b 921600 >/dev/null 2>&1 & A=$!
sleep 3
ros2 bag record -s mcap -o $BAG /controller/command /controller/state /controller/instrumentation /controller/odometry >/dev/null 2>&1 & B=$!
sleep 2
python3 ~/fw/behaviors_series.py
kill -INT $B; wait $B
kill $A; wait
echo "done: turn the motor switch off"
