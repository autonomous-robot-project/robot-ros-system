"""Bench series of reversals, TURN_TO and GO_TO on the current firmware (passo 7 da #32, not evidence).
One publisher at 20 Hz; targets relative to the odometry, as the sequencer computes them."""
import math, time
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from robot_interfaces.msg import Command

QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
rclpy.init(); node = rclpy.create_node("behaviors_series")
pub = node.create_publisher(Command, "/controller/command", QOS)
latest = {}
node.create_subscription(Odometry, "/controller/odometry", lambda m: latest.__setitem__("o", m), QOS)

def pose():
    m = latest["o"]; q = m.pose.pose.orientation
    return m.pose.pose.position.x, m.pose.pose.position.y, 2 * math.atan2(q.z, q.w), m.twist.twist
def hold(command, seconds):
    end = time.time() + seconds
    while time.time() < end:
        pub.publish(command); rclpy.spin_once(node, timeout_sec=0.0); time.sleep(0.05)
def until_settled(command, timeout):
    start = time.time(); moved = False; still_since = None
    while time.time() - start < timeout:
        pub.publish(command); rclpy.spin_once(node, timeout_sec=0.0); time.sleep(0.05)
        tw = pose()[3]; moving = abs(tw.linear.x) > 0 or abs(tw.angular.z) > 0
        moved |= moving
        if moved and not moving:
            still_since = still_since or time.time()
            if time.time() - still_since > 0.7: return time.time() - start
        else: still_since = None
    return None
def twist(v): return Command(behavior=Command.TWIST, linear=v)
def mark(text): print(f"{time.time():.3f} {text}", flush=True)

hold(twist(0.0), 1.5)
for k in range(10):
    mark(f"reversal {k+1}"); hold(twist(0.15), 2.0); hold(twist(-0.15), 2.0); hold(twist(0.0), 1.0)
for delta in (90, -90, 45, -45, 170, -170, 30):
    x, y, th, _ = pose(); target = th + math.radians(delta)
    t = until_settled(Command(behavior=Command.TURN_TO, theta=target), 6.0)
    x, y, th, _ = pose(); err = math.degrees(math.remainder(target - th, 2 * math.pi))
    mark(f"TURN_TO {delta:+d} deg: settled {t if t is None else round(t, 2)} s, final error {err:+.1f} deg")
    hold(twist(0.0), 1.0)
def go(distance, lateral, align):
    x, y, th, _ = pose()
    tx = x + distance * math.cos(th) - lateral * math.sin(th); ty = y + distance * math.sin(th) + lateral * math.cos(th)
    if align:
        until_settled(Command(behavior=Command.TURN_TO, theta=math.atan2(ty - y, tx - x)), 6.0); hold(twist(0.0), 0.5)
    t = until_settled(Command(behavior=Command.GO_TO, x=tx, y=ty), 15.0)
    x, y, th, _ = pose()
    mark(f"GO_TO {distance} m ahead, {lateral} m left, aligned {align}: settled {t if t is None else round(t, 2)} s, final distance {math.hypot(tx - x, ty - y) * 1000:.0f} mm")
    hold(twist(0.0), 1.0)
go(1.0, 0.0, True); go(1.0, 0.3, False); go(2.0, 0.0, True)
hold(twist(0.0), 1.0)
rclpy.shutdown()
