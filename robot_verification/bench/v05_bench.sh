#!/bin/bash
# Bench check of the V-05 meter chain (not evidence): agent, bag of the controller topics, command at 20 Hz.
set -m
source /opt/ros/jazzy/setup.bash; source ~/robot_ws/install/setup.bash; source ~/spike_build/install/setup.bash
BAG=$1
AGENT=$(ros2 pkg prefix micro_ros_agent)/lib/micro_ros_agent/micro_ros_agent
$AGENT serial --dev /dev/ttyAMA0 -b 921600 >/dev/null 2>&1 & A=$!
sleep 3
ros2 bag record -s mcap -o $BAG /controller/command /controller/state /controller/instrumentation >/dev/null 2>&1 & B=$!
sleep 2
ros2 topic pub -r 20 --qos-reliability best_effort --qos-depth 1 /controller/command robot_interfaces/msg/Command "{behavior: 0, linear: 0.2}" >/dev/null 2>&1 & P=$!
sleep 20
ros2 run robot_verification measure_link_load
sleep 300
kill -INT $B; wait $B
kill $P $A; wait
ros2 run robot_verification check_v05 $BAG
