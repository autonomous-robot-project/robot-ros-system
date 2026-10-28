"""
Analysis of the TURN_TO, GO_TO and reversal bench runs (robot-docs nota 0014), from the bags.

Criticality: T1, evidence (robot-docs codigo.md).

What: for each constant command, the pose at its end; the runaways (a wheel read above 0.3 m/s,
more than v_max + omega_max b/2 allows in these runs) and the reading of both wheels during each
reverse phase; the FAULT_COGNIZANT intervals; the deadlines. With --series, the readings every
0.2 s.
Why: the lost direction of ADR 019 at T_vel = 0.51 s, and its absence at 0.3 s.

Usage: python3 analyze_behaviors.py <bag directory> [--series]. Run with the ROS environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys

from nav_msgs.msg import Odometry
from robot_interfaces.msg import Command, Instrumentation, State
from robot_verification.recording import Stream, read_bag

TOPICS = {
    '/controller/command': Command,
    '/controller/state': State,
    '/controller/instrumentation': Instrumentation,
    '/controller/odometry': Odometry,
}
RUNAWAY_SPEED = 0.3  # m/s, above any legitimate wheel speed of these runs
REVERSE = -0.15  # m/s, the reverse phase of reversal_bench and behaviors_series
SETTLED_AFTER_S = 0.8  # from the reverse command to both wheels expected negative
EPISODE_GAP_S = 1.0
SERIES_STEP_S = 0.2
FAULT_COGNIZANT = 1
BEHAVIOR_NAMES = ('TWIST', 'TURN_TO', 'GO_TO')


@dataclass(frozen=True)
class Phase:
    """A constant command, from its first to its last message."""

    command: Command
    start_s: float
    end_s: float


def get_heading(odometry: Odometry) -> float:
    orientation = odometry.pose.pose.orientation
    return 2.0 * math.atan2(orientation.z, orientation.w)


def is_same_command(a: Command, b: Command) -> bool:
    return (a.behavior, a.linear, a.angular, a.theta, a.x, a.y) == (
        b.behavior,
        b.linear,
        b.angular,
        b.theta,
        b.x,
        b.y,
    )


def split_into_phases(commands: Stream[Command]) -> list[Phase]:
    phases: list[Phase] = []
    for sample in commands:
        if phases and is_same_command(phases[-1].command, sample.message):
            phases[-1] = Phase(phases[-1].command, phases[-1].start_s, sample.time_s)
        else:
            phases.append(Phase(sample.message, sample.time_s, sample.time_s))
    return phases


def find_runaways(instrumentation: Stream[Instrumentation]) -> list[tuple[float, float, str]]:
    """Return each runaway episode: start, end and wheel."""
    episodes: list[tuple[float, float, str]] = []
    for sample in instrumentation:
        for wheel, speed in (
            ('L', sample.message.left_wheel_speed),
            ('R', sample.message.right_wheel_speed),
        ):
            if abs(speed) <= RUNAWAY_SPEED:
                continue
            last = episodes[-1] if episodes else None
            if last and last[2] == wheel and sample.time_s - last[1] < EPISODE_GAP_S:
                episodes[-1] = (last[0], sample.time_s, wheel)
            else:
                episodes.append((sample.time_s, sample.time_s, wheel))
    return episodes


def find_fault_intervals(states: Stream[State]) -> list[tuple[float, float]]:
    """Return each FAULT_COGNIZANT interval: start and duration."""
    intervals: list[tuple[float, float]] = []
    start: float | None = None
    for sample in states:
        active = bool(sample.message.faults & FAULT_COGNIZANT)
        if active and start is None:
            start = sample.time_s
        if not active and start is not None:
            intervals.append((start, sample.time_s - start))
            start = None
    return intervals


def report(streams: dict[str, Stream], series: bool) -> list[str]:
    commands, states = streams['/controller/command'], streams['/controller/state']
    instrumentation, odometry = (
        streams['/controller/instrumentation'],
        streams['/controller/odometry'],
    )
    origin_s = commands[0].time_s
    lines: list[str] = []
    phases = split_into_phases(commands)
    for phase in phases:
        end = odometry.at(phase.end_s).message
        position = end.pose.pose.position
        lines.append(
            f'{BEHAVIOR_NAMES[phase.command.behavior]:7s} linear {phase.command.linear:+.2f} '
            f'{phase.start_s - origin_s:6.1f}-{phase.end_s - origin_s:6.1f} s: end pose '
            f'x {position.x:+.3f} y {position.y:+.3f} '
            f'theta {math.degrees(get_heading(end)):+.1f} deg'
        )
    for phase in phases:
        if phase.command.behavior != 0 or not math.isclose(
            phase.command.linear, REVERSE, rel_tol=1e-6
        ):
            continue
        readings = instrumentation.between(phase.start_s + SETTLED_AFTER_S, phase.end_s + 0.05)
        non_negative = [
            (
                round(s.time_s - phase.start_s, 1),
                round(s.message.left_wheel_speed, 3),
                round(s.message.right_wheel_speed, 3),
            )
            for s in readings
            if s.message.left_wheel_speed >= 0 or s.message.right_wheel_speed >= 0
        ]
        lines.append(
            f'reverse phase at {phase.start_s - origin_s:6.1f} s: '
            f'non-negative readings {non_negative}'
        )
    runaways = [
        (round(a - origin_s, 1), round(b - a, 1), wheel)
        for a, b, wheel in find_runaways(instrumentation)
    ]
    lines.append(f'runaway episodes (start, duration, wheel): {runaways}')
    faults = [(round(a - origin_s, 1), round(d, 2)) for a, d in find_fault_intervals(states)]
    lines.append(f'FAULT_COGNIZANT intervals (start, duration): {faults}')
    first, last = instrumentation[0].message, instrumentation[-1].message
    lines.append(
        f'C_max {last.max_execution_time_us} us, '
        f'R_max {last.max_response_time_us} us (since boot), '
        f'deadline misses in the recording {last.deadline_misses - first.deadline_misses}'
    )
    if series:
        lines += format_series(streams, origin_s)
    return lines


def format_series(streams: dict[str, Stream], origin_s: float) -> list[str]:
    states, instrumentation = streams['/controller/state'], streams['/controller/instrumentation']
    odometry = streams['/controller/odometry']
    end_s = streams['/controller/command'][-1].time_s
    lines: list[str] = []
    t = origin_s
    while t <= end_s:
        state, wheels, pose = states.at(t), instrumentation.at(t), odometry.at(t)
        if state and wheels and pose:
            lines.append(
                f'{t - origin_s:5.1f} echo {state.message.echo.behavior} '
                f'faults {state.message.faults} | '
                f'theta {math.degrees(get_heading(pose.message)):+7.1f} | '
                f'wheels L {wheels.message.left_wheel_speed:+.3f} '
                f'R {wheels.message.right_wheel_speed:+.3f}'
            )
        t += SERIES_STEP_S
    return lines


def main() -> None:
    if len(sys.argv) not in (2, 3):
        print('usage: analyze_behaviors.py <bag directory> [--series]')
        sys.exit(2)
    streams = read_bag(sys.argv[1], TOPICS)
    print('\n'.join(report(streams, series='--series' in sys.argv)))


if __name__ == '__main__':
    main()
