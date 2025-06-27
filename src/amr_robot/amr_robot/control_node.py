import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class ControlNode(Node):
    def __init__(self):
        super().__init__("control_node")
        self.subscription = self.create_subscription(
            Twist, "cmd_vel", self.listener_callback, 10
        )

    def listener_callback(self, msg: Twist):
        self.get_logger().info(
            f"Received cmd_vel: {msg}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
