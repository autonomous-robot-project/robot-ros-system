"""
Closed loop of the wheels on the motor, judged on the ros2 bag recorded by run_closed_loop.

Criticality: T1, evidence (robot-docs codigo.md).

What: after a stop, each wheel speed reads zero in time; each low speed is held without a limit
cycle; no false FAULT_COGNIZANT.
How: splits the commands into steps and checks the Instrumentation and the State in each of them.
Why: confirms on the motor v_min and T_vel, parameters of V-02 (robot-docs nota 0011).

Usage: measure_closed_loop <bag directory>. Exit code: 0 pass, 1 fail, 2 invalid run.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import itertools
import math
import statistics
import sys
from typing import TYPE_CHECKING

from robot_verification.recording import (
    ControllerRecording,
    Sample,
    Stream,
    Verdict,
    is_close_float32,
)

if TYPE_CHECKING:
    from robot_interfaces.msg import Command, Instrumentation, State

# Schedule of the run, (linear speed in m/s, duration in s); run_closed_loop publishes it.
STOP_FROM: float = 0.2
LOW_SPEEDS: tuple[float, ...] = (0.07, 0.10, 0.15)
LOW_SPEED_S: float = 8.0
SCHEDULE: list[tuple[float, float]] = [
    (0.0, 2.0),
    (STOP_FROM, 3.0),
    (0.0, 3.0),
    *((v, LOW_SPEED_S) for v in LOW_SPEEDS),
    (0.0, 3.0),
]

# Constants of the firmware (control_config.h) and of nota 0010.
A_DEC: float = 0.5  # m/s²
V_MIN: float = 0.07  # m/s
T_VEL_S: float = 0.51
BRAKE_S: float = 0.05
INSTRUMENTATION_PERIOD_S: float = 0.1  # 10 Hz (ADR 025)

# Stop: the reference ramps down at A_DEC and the PI brakes below v_min, the brake stops the wheel,
# and the speed reads zero T_vel after the last edge. The limit adds two Instrumentation periods.
STOP_EXPECTED_S: float = (STOP_FROM - V_MIN) / A_DEC + BRAKE_S + T_VEL_S
STOP_LIMIT_S: float = STOP_EXPECTED_S + 2 * INSTRUMENTATION_PERIOD_S

# Low speeds, over the last STEADY_WINDOW_S of the step: never zero, spread below the limit.
STEADY_WINDOW_S: float = 5.0
MIN_STEADY_SAMPLES: int = 40  # of the 50 that the Instrumentation gives in the window
SPREAD_LIMIT: float = 0.20  # standard deviation / mean

FAULT_COGNIZANT: int = 1  # State.FAULT_COGNIZANT


class Wheel(Enum):
    LEFT = 'left'
    RIGHT = 'right'


@dataclass(frozen=True)
class Step:
    """A constant command of the schedule, from its first message to the first of the next."""

    linear: float
    start_s: float
    end_s: float


@dataclass(frozen=True)
class StepSpeeds:
    """The speeds of one wheel over the steady window of a step."""

    speeds: tuple[float, ...]

    @property
    def spread(self) -> float:
        """Return the standard deviation over the mean."""
        return statistics.pstdev(self.speeds) / self.mean if self.mean > 0 else math.inf

    @property
    def mean(self) -> float:
        return statistics.fmean(self.speeds)

    @property
    def zero_samples(self) -> int:
        return self.speeds.count(0.0)


def get_wheel_speed(sample: Sample[Instrumentation], wheel: Wheel) -> float:
    instrumentation = sample.message
    if wheel is Wheel.LEFT:
        return instrumentation.left_wheel_speed
    return instrumentation.right_wheel_speed


def has_cognizant_failure(state: Sample[State]) -> bool:
    return bool(state.message.faults & FAULT_COGNIZANT)


def is_stop_transition(before: Step, after: Step) -> bool:
    return is_close_float32(before.linear, STOP_FROM) and after.linear == 0.0


def is_low_speed_step(step: Step) -> bool:
    return any(is_close_float32(step.linear, v) for v in LOW_SPEEDS)


def find_when_speed_stays_zero(
    samples: list[Sample[Instrumentation]], wheel: Wheel
) -> Sample[Instrumentation] | None:
    """
    Return the sample from which the wheel speed is zero until the last sample, or None.

    The firmware publishes a literal 0.0 when no edge came for T_vel, so the comparison is exact.
    """
    first_zero: Sample[Instrumentation] | None = None
    for sample in reversed(samples):
        if get_wheel_speed(sample, wheel) != 0.0:
            break
        first_zero = sample
    return first_zero


def find_step_starts(commands: Stream[Command]) -> list[Sample[Command]]:
    """Return the commands whose linear speed differs from the one before."""
    starts: list[Sample[Command]] = []
    for command in commands:
        if not starts or command.message.linear != starts[-1].message.linear:
            starts.append(command)
    return starts


def split_into_steps(commands: Stream[Command]) -> list[Step]:
    """Split the commands into steps of constant speed; the last ends at the last command."""
    if not commands:
        return []
    starts = find_step_starts(commands)
    ends_s = [start.time_s for start in starts[1:]] + [commands[-1].time_s]
    bounds = zip(starts, ends_s, strict=True)
    return [Step(start.message.linear, start.time_s, end_s) for start, end_s in bounds]


class ClosedLoopRun:
    """The quantities measured on the recording of a closed-loop run."""

    def __init__(self, recording: ControllerRecording):
        self.recording: ControllerRecording = recording
        self.steps: list[Step] = split_into_steps(recording.commands)

    def measure_zero_after_s(self, wheel: Wheel) -> float | None:
        """Return the time from the stop command to the speed staying zero, or None."""
        stop = self.find_stop_step()
        stays_zero = find_when_speed_stays_zero(self.get_instrumentation_in(stop), wheel)
        return None if stays_zero is None else stays_zero.time_s - stop.start_s

    def find_stop_step(self) -> Step:
        pairs = itertools.pairwise(self.steps)
        return next(after for before, after in pairs if is_stop_transition(before, after))

    def get_step_speeds(self, step: Step, wheel: Wheel) -> StepSpeeds:
        window = self.get_steady_window(step)
        return StepSpeeds(tuple(get_wheel_speed(sample, wheel) for sample in window))

    def select_low_speed_steps(self) -> list[Step]:
        return [step for step in self.steps if is_low_speed_step(step)]

    def find_false_alarms(self) -> list[Sample[State]]:
        states = self.recording.states.between(self.steps[0].start_s, math.inf)
        return [state for state in states if has_cognizant_failure(state)]

    def get_last_echoed_linear_in(self, step: Step) -> float | None:
        """Return the linear speed echoed by the last state of the step, or None."""
        states = self.recording.states.between(step.start_s, step.end_s)
        return states[-1].message.echo.linear if states else None

    def get_instrumentation_in(self, step: Step) -> list[Sample[Instrumentation]]:
        return self.recording.instrumentation.between(step.start_s, step.end_s)

    def get_steady_window(self, step: Step) -> list[Sample[Instrumentation]]:
        return self.recording.instrumentation.between(step.end_s - STEADY_WINDOW_S, step.end_s)


class ClosedLoopVerification:
    """
    The criteria and the validity conditions of the closed-loop run, and its verdict.

    Valid run: the schedule produced its steps in order, the echo showed each commanded speed (the
    controller was applying it), and the Instrumentation was recorded in each step and window.
    """

    def __init__(self, run: ClosedLoopRun):
        self.run: ClosedLoopRun = run

    # --- Verdict ---

    @property
    def verdict(self) -> Verdict:
        if not self.run_is_valid:
            return Verdict.INVALID
        passed = (
            self.did_every_wheel_stop_in_time()
            and self.is_every_low_speed_held()
            and self.was_free_of_false_alarms()
        )
        return Verdict.PASS if passed else Verdict.FAIL

    @property
    def run_is_valid(self) -> bool:
        return (
            self.was_schedule_produced()
            and self.was_every_command_echoed()
            and self.was_instrumentation_recorded_in_every_step()
            and self.was_every_steady_window_recorded()
        )

    # --- Criteria ---

    def did_every_wheel_stop_in_time(self) -> bool:
        return all(self.did_stop_in_time(wheel) for wheel in Wheel)

    def did_stop_in_time(self, wheel: Wheel) -> bool:
        zero_after_s = self.run.measure_zero_after_s(wheel)
        return zero_after_s is not None and zero_after_s <= STOP_LIMIT_S

    def is_every_low_speed_held(self) -> bool:
        pairs = itertools.product(self.run.select_low_speed_steps(), Wheel)
        return all(self.is_speed_held(step, wheel) for step, wheel in pairs)

    def is_speed_held(self, step: Step, wheel: Wheel) -> bool:
        """Check that the loop holds the speed of the step: never zero, spread below the limit."""
        speeds = self.run.get_step_speeds(step, wheel)
        return speeds.zero_samples == 0 and speeds.spread < SPREAD_LIMIT

    def was_free_of_false_alarms(self) -> bool:
        return not self.run.find_false_alarms()

    # --- Validity of the run ---

    def was_schedule_produced(self) -> bool:
        commanded = [step.linear for step in self.run.steps]
        scheduled = [linear for linear, _ in SCHEDULE]
        return len(commanded) == len(scheduled) and all(
            map(is_close_float32, commanded, scheduled)
        )

    def was_every_command_echoed(self) -> bool:
        return all(self.was_command_echoed(step) for step in self.run.steps)

    def was_command_echoed(self, step: Step) -> bool:
        echoed = self.run.get_last_echoed_linear_in(step)
        return echoed is not None and is_close_float32(echoed, step.linear)

    def was_instrumentation_recorded_in_every_step(self) -> bool:
        return all(self.run.get_instrumentation_in(step) for step in self.run.steps)

    def was_every_steady_window_recorded(self) -> bool:
        windows = (self.run.get_steady_window(step) for step in self.run.select_low_speed_steps())
        return all(len(window) >= MIN_STEADY_SAMPLES for window in windows)


class ClosedLoopReport:
    """The report of a closed-loop run: the measured quantities and the verdict of each check."""

    def __init__(self, verification: ClosedLoopVerification):
        self.verification: ClosedLoopVerification = verification
        self.run: ClosedLoopRun = verification.run

    def __str__(self) -> str:
        lines = [self._format_steps_line()]
        if self.verification.was_schedule_produced():
            lines += (
                self._format_stop_lines()
                + self._format_low_speed_lines()
                + self._format_run_lines()
            )
        lines.append(f'closed loop: {self.verification.verdict.name}')
        return '\n'.join(lines)

    def _format_steps_line(self) -> str:
        return f'closed loop: {len(self.run.steps)} steps ({len(SCHEDULE)} scheduled)'

    def _format_stop_lines(self) -> list[str]:
        return [self._format_stop_line(wheel) for wheel in Wheel]

    def _format_stop_line(self, wheel: Wheel) -> str:
        zero_after_s = self.run.measure_zero_after_s(wheel)
        shown = 'never' if zero_after_s is None else f'{zero_after_s:.2f} s'
        return (
            f'  stop, {wheel.value}: zero after {shown} (expected {STOP_EXPECTED_S:.2f} s, '
            f'limit {STOP_LIMIT_S:.2f} s), passed: {self.verification.did_stop_in_time(wheel)}'
        )

    def _format_low_speed_lines(self) -> list[str]:
        pairs = itertools.product(self.run.select_low_speed_steps(), Wheel)
        return [self._format_low_speed_line(step, wheel) for step, wheel in pairs]

    def _format_low_speed_line(self, step: Step, wheel: Wheel) -> str:
        speeds = self.run.get_step_speeds(step, wheel)
        return (
            f'  {step.linear:.2f} m/s, {wheel.value}: mean {speeds.mean:.3f} m/s, '
            f'spread {speeds.spread:.0%}, zero samples {speeds.zero_samples}, '
            f'passed: {self.verification.is_speed_held(step, wheel)}'
        )

    def _format_run_lines(self) -> list[str]:
        return [
            f'  FAULT_COGNIZANT states: {len(self.run.find_false_alarms())}',
            f'  echo of every step: {self.verification.was_every_command_echoed()}',
        ]


def main() -> None:
    if len(sys.argv) != 2:
        print('usage: measure_closed_loop <bag directory>')
        sys.exit(Verdict.INVALID.value)
    run = ClosedLoopRun(ControllerRecording.from_bag(sys.argv[1]))
    verification = ClosedLoopVerification(run)
    print(ClosedLoopReport(verification))
    sys.exit(verification.verdict.value)


if __name__ == '__main__':
    main()
