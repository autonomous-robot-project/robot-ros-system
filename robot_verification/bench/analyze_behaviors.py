"""
Analysis of the TURN_TO, GO_TO and reversal bench runs (robot-docs nota 0014), from the bags.

Criticality: T1, evidence (robot-docs codigo.md).

What: the pose at the end of each constant command; the readings of both wheels in each reverse
phase; the runaways; the FAULT_COGNIZANT intervals; the deadlines. With --series, the readings
every SERIES_STEP_S.
How: the bag through ControllerRecording, one class for the measurements and one for the report.
Why: the lost direction of ADR 019 at T_vel = 0.51 s, and its absence at 0.3 s.

Usage: python3 analyze_behaviors.py <bag directory> [--series]. Run with the ROS environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from typing import TYPE_CHECKING

from robot_verification.measure_closed_loop import Wheel, get_wheel_speed
from robot_verification.recording import ControllerRecording, Sample, Stream, is_close_float32

if TYPE_CHECKING:
    from nav_msgs.msg import Odometry
    from robot_interfaces.msg import Command, Instrumentation, State

# A wheel read faster than this ran away: above v_max + omega_max b/2 = 0.4 m/s with margin for the
# ramp, and above the 0.15 m/s of the reversals.
RUNAWAY_SPEED: float = 0.3  # m/s
EPISODE_GAP_S: float = 1.0  # readings farther apart are separate episodes
# The reverse phase of reversal_bench and behaviors_series; both wheels are expected negative from
# SETTLED_AFTER_S after its first command to one Instrumentation period after its last.
REVERSE: float = -0.15  # m/s
SETTLED_AFTER_S: float = 0.8
INSTRUMENTATION_PERIOD_S: float = 0.05  # half of the 10 Hz period, to take the last reading
SERIES_STEP_S: float = 0.2

# Copies of the codes of Command.msg and State.msg (ROS only for the type checker): keep equal.
TWIST: int = 0  # Command.TWIST
FAULT_COGNIZANT: int = 1  # State.FAULT_COGNIZANT
BEHAVIOR_NAMES: tuple[str, ...] = ('TWIST', 'TURN_TO', 'GO_TO')  # by Command.behavior
WHEEL_LABELS: dict[Wheel, str] = {Wheel.LEFT: 'L', Wheel.RIGHT: 'R'}


@dataclass(frozen=True)
class Phase:
    """A constant command, from its first to its last message."""

    command: Command
    start_s: float
    end_s: float


@dataclass(frozen=True)
class RunawayEpisode:
    """Consecutive readings of one wheel above RUNAWAY_SPEED."""

    wheel: Wheel
    start_s: float
    end_s: float


@dataclass(frozen=True)
class FaultInterval:
    start_s: float
    duration_s: float


def get_heading(odometry: Odometry) -> float:
    # Yaw of a planar quaternion (0, 0, sin(theta/2), cos(theta/2)).
    orientation = odometry.pose.pose.orientation
    return 2.0 * math.atan2(orientation.z, orientation.w)


def get_command_values(command: Command) -> tuple[int, float, float, float, float, float]:
    return (command.behavior, command.linear, command.angular, command.theta, command.x, command.y)


def is_reverse_phase(phase: Phase) -> bool:
    command = phase.command
    return command.behavior == TWIST and is_close_float32(command.linear, REVERSE)


def is_fault_active(state: Sample[State]) -> bool:
    return bool(state.message.faults & FAULT_COGNIZANT)


def split_into_phases(commands: Stream[Command]) -> list[Phase]:
    phases: list[Phase] = []
    for sample in commands:
        continues_phase = bool(phases) and get_command_values(
            phases[-1].command
        ) == get_command_values(sample.message)
        if continues_phase:
            phases[-1] = Phase(phases[-1].command, phases[-1].start_s, sample.time_s)
        else:
            phases.append(Phase(sample.message, sample.time_s, sample.time_s))
    return phases


def find_runaways_of(
    instrumentation: Stream[Instrumentation], wheel: Wheel
) -> list[RunawayEpisode]:
    episodes: list[RunawayEpisode] = []
    for sample in instrumentation:
        if abs(get_wheel_speed(sample, wheel)) <= RUNAWAY_SPEED:
            continue
        if episodes and sample.time_s - episodes[-1].end_s < EPISODE_GAP_S:
            episodes[-1] = RunawayEpisode(wheel, episodes[-1].start_s, sample.time_s)
        else:
            episodes.append(RunawayEpisode(wheel, sample.time_s, sample.time_s))
    return episodes


def find_fault_intervals_in(states: Stream[State]) -> list[FaultInterval]:
    """Return each FAULT_COGNIZANT interval; one still active at the end lasts until then."""
    intervals: list[FaultInterval] = []
    start_s: float | None = None
    for state in states:
        if is_fault_active(state) and start_s is None:
            start_s = state.time_s
        if not is_fault_active(state) and start_s is not None:
            intervals.append(FaultInterval(start_s, state.time_s - start_s))
            start_s = None
    if start_s is not None:
        intervals.append(FaultInterval(start_s, states[-1].time_s - start_s))
    return intervals


def format_readings(sample: Sample[Instrumentation], since_s: float) -> tuple[float, float, float]:
    """Return (time since since_s, left speed, right speed), rounded for the report."""
    return (
        round(sample.time_s - since_s, 1),
        round(get_wheel_speed(sample, Wheel.LEFT), 3),
        round(get_wheel_speed(sample, Wheel.RIGHT), 3),
    )


class BehaviorsRun:
    """The quantities measured on the recording of a bench run."""

    def __init__(self, recording: ControllerRecording):
        self.recording: ControllerRecording = recording
        self.phases: list[Phase] = split_into_phases(recording.commands)
        self.origin_s: float = recording.commands[0].time_s

    def get_end_pose(self, phase: Phase) -> Odometry:
        return self.recording.odometry.at(phase.end_s).message

    def select_reverse_phases(self) -> list[Phase]:
        return [phase for phase in self.phases if is_reverse_phase(phase)]

    def find_non_negative_readings(self, phase: Phase) -> list[Sample[Instrumentation]]:
        """Return the readings of the reverse phase in which a wheel is not negative."""
        readings = self.recording.instrumentation.between(
            phase.start_s + SETTLED_AFTER_S, phase.end_s + INSTRUMENTATION_PERIOD_S
        )
        return [s for s in readings if any(get_wheel_speed(s, w) >= 0 for w in Wheel)]

    def find_runaways(self) -> list[RunawayEpisode]:
        episodes = [e for w in Wheel for e in find_runaways_of(self.recording.instrumentation, w)]
        return sorted(episodes, key=lambda episode: episode.start_s)

    def find_fault_intervals(self) -> list[FaultInterval]:
        return find_fault_intervals_in(self.recording.states)

    def measure_deadline_misses(self) -> int:
        first, last = self.recording.instrumentation[0], self.recording.instrumentation[-1]
        return last.message.deadline_misses - first.message.deadline_misses

    def get_last_instrumentation(self) -> Instrumentation:
        return self.recording.instrumentation[-1].message


class BehaviorsReport:
    """The report of a bench run: phases, reverse phases, runaways, faults and deadlines."""

    def __init__(self, run: BehaviorsRun, series: bool):
        self.run: BehaviorsRun = run
        self.series: bool = series

    def __str__(self) -> str:
        lines = (
            self._format_phase_lines()
            + self._format_reverse_lines()
            + [
                self._format_runaway_line(),
                self._format_fault_line(),
                self._format_deadline_line(),
            ]
        )
        if self.series:
            lines += self._format_series_lines()
        return '\n'.join(lines)

    def _format_phase_lines(self) -> list[str]:
        return [self._format_phase_line(phase) for phase in self.run.phases]

    def _format_phase_line(self, phase: Phase) -> str:
        end = self.run.get_end_pose(phase)
        position = end.pose.pose.position
        return (
            f'{BEHAVIOR_NAMES[phase.command.behavior]:7s} linear {phase.command.linear:+.2f} '
            f'{self._since_origin(phase.start_s):6.1f}-{self._since_origin(phase.end_s):6.1f} s: '
            f'end pose x {position.x:+.3f} y {position.y:+.3f} '
            f'theta {math.degrees(get_heading(end)):+.1f} deg'
        )

    def _format_reverse_lines(self) -> list[str]:
        return [self._format_reverse_line(phase) for phase in self.run.select_reverse_phases()]

    def _format_reverse_line(self, phase: Phase) -> str:
        readings = [
            format_readings(s, since_s=phase.start_s)
            for s in self.run.find_non_negative_readings(phase)
        ]
        return (
            f'reverse phase at {self._since_origin(phase.start_s):6.1f} s: '
            f'non-negative readings {readings}'
        )

    def _format_runaway_line(self) -> str:
        runaways = [
            (
                self._round_since_origin(e.start_s),
                round(e.end_s - e.start_s, 1),
                WHEEL_LABELS[e.wheel],
            )
            for e in self.run.find_runaways()
        ]
        return f'runaway episodes (start, duration, wheel): {runaways}'

    def _format_fault_line(self) -> str:
        faults = [
            (self._round_since_origin(f.start_s), round(f.duration_s, 2))
            for f in self.run.find_fault_intervals()
        ]
        return f'FAULT_COGNIZANT intervals (start, duration): {faults}'

    def _format_deadline_line(self) -> str:
        last = self.run.get_last_instrumentation()
        return (
            f'C_max {last.max_execution_time_us} us, '
            f'R_max {last.max_response_time_us} us (since boot), '
            f'deadline misses in the recording {self.run.measure_deadline_misses()}'
        )

    def _format_series_lines(self) -> list[str]:
        duration_s = self._since_origin(self.run.recording.commands[-1].time_s)
        instants = [
            self.run.origin_s + k * SERIES_STEP_S
            for k in range(int(duration_s / SERIES_STEP_S) + 1)
        ]
        lines = [self._format_series_line(t) for t in instants]
        return [line for line in lines if line is not None]

    def _format_series_line(self, t: float) -> str | None:
        """Return the readings at t, or None before the first of each topic."""
        recording = self.run.recording
        state, wheels = recording.states.at(t), recording.instrumentation.at(t)
        pose = recording.odometry.at(t)
        if state is None or wheels is None or pose is None:
            return None
        return (
            f'{self._since_origin(t):5.1f} echo {state.message.echo.behavior} '
            f'faults {state.message.faults} | '
            f'theta {math.degrees(get_heading(pose.message)):+7.1f} | '
            f'wheels L {get_wheel_speed(wheels, Wheel.LEFT):+.3f} '
            f'R {get_wheel_speed(wheels, Wheel.RIGHT):+.3f}'
        )

    def _round_since_origin(self, time_s: float) -> float:
        return round(self._since_origin(time_s), 1)

    def _since_origin(self, time_s: float) -> float:
        return time_s - self.run.origin_s


def main() -> None:
    if len(sys.argv) not in (2, 3):
        print('usage: python3 analyze_behaviors.py <bag directory> [--series]')
        sys.exit(2)
    run = BehaviorsRun(ControllerRecording.from_bag(sys.argv[1]))
    print(BehaviorsReport(run, series='--series' in sys.argv))


if __name__ == '__main__':
    main()
