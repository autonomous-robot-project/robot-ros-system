import rclpy
from geometry_msgs.msg import Twist
from rclpy.logging import get_logger
from rclpy.node import Node
from std_msgs.msg import String


class ControlNode(Node):
    def __init__(self):
        super().__init__("control_node")
        self.subscription = self.create_subscription(
            Twist, "cmd_vel", self.listener_callback, 10
        )
        self.publisher_ = self.create_publisher(String, "send_to_micro", 10)

        self.car_width = 0.2

    def listener_callback(self, msg: Twist):
        self.get_logger().info(f"Received cmd_vel: {msg}")
        data = f"{self.get_left_and_right_motor_speeds(msg)}"
        answer = String(data=data)
        self.publisher_.publish(answer)

    def get_left_and_right_motor_speeds(self, cmd_vel: Twist):
        left_motor_speed = cmd_vel.linear.x - (self.car_width / 2) * cmd_vel.angular.z
        right_motor_speed = cmd_vel.linear.x + (self.car_width / 2) * cmd_vel.angular.z
        return left_motor_speed, right_motor_speed


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    get_logger("control_node").info("Control node started")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
