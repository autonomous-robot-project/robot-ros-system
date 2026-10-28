#!/bin/bash
# Bench check of TURN_TO and GO_TO (passo 7 da #32, not evidence): S2 -> S3 of V-01 with the wheels
# suspended. Records command, State, Instrumentation and Odometry; commands at 20 Hz.
set -m
source /opt/ros/jazzy/setup.bash; source ~/robot_ws/install/setup.bash; source ~/spike_build/install/setup.bash
BAG=$1
AGENT=$(ros2 pkg prefix micro_ros_agent)/lib/micro_ros_agent/micro_ros_agent
$AGENT serial --dev /dev/ttyAMA0 -b 921600 >/dev/null 2>&1 & A=$!
sleep 3
ros2 bag record -s mcap -o $BAG /controller/command /controller/state /controller/instrumentation /controller/odometry >/dev/null 2>&1 & B=$!
sleep 2
publish() {  # publish(seconds, command)
    timeout -s INT $1 ros2 topic pub -r 20 --qos-reliability best_effort --qos-depth 1 /controller/command robot_interfaces/msg/Command "$2" >/dev/null 2>&1
}
echo "TWIST(0, 0)";       publish 3 "{behavior: 0}"
echo "TURN_TO(pi/2)";     publish 5 "{behavior: 1, theta: 1.5707963}"
echo "GO_TO(0, 0.5)";     publish 6 "{behavior: 2, x: 0.0, y: 0.5}"
echo "TWIST(0, 0)";       publish 3 "{behavior: 0}"
kill -INT $B; wait $B
kill $A; wait
echo "done: turn the motor switch off"
