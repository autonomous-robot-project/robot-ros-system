import time
import rclpy
import serial
from rclpy.node import Node
from std_msgs.msg import String


class UartCommNode(Node):
    def __init__(self):
        super().__init__("uart_node")
        self.publisher_ = self.create_publisher(String, "micro_raw_data", 10)
        self.subscriber_ = self.create_subscription(
            String, "send_to_micro", self.send_to_micro, 10
        )

        self.handshake_timeout = 5

        try:
            # Configure the serial port
            self.serial_port = serial.Serial(
                port="/dev/ttyUSB0",
                baudrate=115200,
                timeout=1,
            )
            self.serial_port.dtr = False
            self.serial_port.rts = False
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()

            self.get_logger().info("Waiting for microcontroller to be ready...")
            self.handshake()

            self.get_logger().info(
                "Connected to microcontroller. Ready to receive and send data."
            )
        except serial.SerialException as e:
            self.get_logger().error(f"Could not open serial port: {e}")
            rclpy.shutdown()
            return

        # Create a timer to read and publish data periodically
        self.timer = self.create_timer(0.02, self.read_from_micro)  # 10 Hz

    def handshake(self):
        self.serial_port.write("INIT\n".encode())

        start = time.time()
        while True:
            line = self.serial_port.readline().decode().strip()
            if line == "READY":
                break
            if time.time() - start > self.handshake_timeout:
                raise TimeoutError("Microcontroller did not respond in time")

    def read_from_micro(self):
        if self.serial_port.in_waiting > 0:
            try:
                line = self.serial_port.readline().decode("utf-8").strip()
                if line:
                    msg = String()
                    msg.data = line
                    self.publisher_.publish(msg)
                    self.get_logger().info(f'Publishing: "{msg.data}"')
            except Exception as e:
                self.get_logger().error(f"Error reading from serial port: {e}")

    def send_to_micro(self, msg: String):
        try:
            data = msg.data + "\n"
            self.serial_port.write(data.encode("utf-8"))
        except Exception as e:
            self.get_logger().error(f"Error sending data to serial port: {e}")


def main(args=None):
    rclpy.init(args=args)
    uart_node = UartCommNode()

    if rclpy.ok():
        try:
            rclpy.spin(uart_node)
        except KeyboardInterrupt:
            pass
        finally:
            # Cleanly close the serial port and shutdown
            uart_node.serial_port.close()
            uart_node.destroy_node()
            rclpy.try_shutdown()
