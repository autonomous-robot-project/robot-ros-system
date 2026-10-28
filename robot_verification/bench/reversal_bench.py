"""Bench check of a wheel reversal (passo 7 da #32, not evidence): forward, reverse, stop, from one
publisher at 20 Hz. Reproduces the lost direction of ADR 019. Usage: reversal_bench.py (records nothing;
run with ros2 bag record alongside)."""
import time
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from robot_interfaces.msg import Command
SCHEDULE = [(0.15, 3.0), (-0.15, 2.0), (0.0, 3.0)]  # (linear m/s, seconds)
rclpy.init()
node = rclpy.create_node("reversal_bench")
pub = node.create_publisher(Command, "/controller/command", QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
time.sleep(1.0)
for linear, seconds in SCHEDULE:
    print(f"TWIST({linear}, 0)", flush=True)
    end = time.time() + seconds
    while time.time() < end:
        pub.publish(Command(behavior=Command.TWIST, linear=linear))
        time.sleep(0.05)
rclpy.shutdown()
