"""Synthetic messages and streams shared by the unit tests of the verification scripts."""

import math
import struct
from types import SimpleNamespace

from robot_verification.recording import GO_TO, TURN_TO, TWIST, Sample


def to_float32(value):
    """Round to float32, as the float32 fields of the messages arrive from the bag."""
    return struct.unpack('f', struct.pack('f', value))[0]


def command(linear=0.2, angular=0.0):
    return SimpleNamespace(
        behavior=TWIST,
        linear=to_float32(linear),
        angular=to_float32(angular),
        theta=0.0,
        x=0.0,
        y=0.0,
    )


def turn_to(theta):
    return SimpleNamespace(
        behavior=TURN_TO, linear=0.0, angular=0.0, theta=to_float32(theta), x=0.0, y=0.0
    )


def go_to(x, y):
    return SimpleNamespace(
        behavior=GO_TO, linear=0.0, angular=0.0, theta=0.0, x=to_float32(x), y=to_float32(y)
    )


def state(echo, faults=0, reflexes=0):
    return SimpleNamespace(faults=faults, reflexes=reflexes, echo=echo)


def odometry(linear=0.0, x=0.0, y=0.0, theta=0.0):
    """Return an Odometry with a planar pose and the linear speed of its twist."""
    orientation = SimpleNamespace(z=math.sin(theta / 2), w=math.cos(theta / 2))
    return SimpleNamespace(
        pose=SimpleNamespace(
            pose=SimpleNamespace(position=SimpleNamespace(x=x, y=y), orientation=orientation)
        ),
        twist=SimpleNamespace(twist=SimpleNamespace(linear=SimpleNamespace(x=linear))),
    )


def instrumentation(
    expiries=0,
    expiry_delay_s=0.0,
    left_wheel_speed=0.0,
    right_wheel_speed=0.0,
    max_execution_time_us=0,
    max_response_time_us=0,
    deadline_misses=0,
    test_event_index=0,
    test_event_response_us=0,
):
    return SimpleNamespace(
        expiries=expiries,
        expiry_delay_us=round(expiry_delay_s * 1e6),
        max_execution_time_us=max_execution_time_us,
        max_response_time_us=max_response_time_us,
        deadline_misses=deadline_misses,
        test_event_index=test_event_index,
        test_event_response_us=test_event_response_us,
        left_wheel_speed=left_wheel_speed,
        right_wheel_speed=right_wheel_speed,
    )


def periodic(start_s, end_s, period_s, message_at):
    """Return samples every period_s from start_s to end_s, the message built for each time."""
    count = round((end_s - start_s) / period_s)
    times = [start_s + k * period_s for k in range(count)]
    return [Sample(t, message_at(t)) for t in times]
