"""
V-03: reactive reflex (RF-03), judged on the ros2 bag recorded by run_v03.

Criticality: T1, evidence (robot-docs codigo.md).

What: with the screen in front of the VL53L0X the measured speed falls to zero while the command
stays at v_r, with no fault code active in the State; after the screen is removed the measured
speed comes back to v_r with no new command.
How: run_v03 publishes TWIST(v_r, 0) and marks each placement and removal of the screen
(/verification/v03_marker, "placed" and "removed") at the end of a countdown. The speed is judged
in windows bounded by these markers, not by the reflex bit of the State, which would judge the
controller by its own detection (robot-docs nota 0015, D8). An operator error only fails a run,
never passes it.
Why: V-03 (robot-docs verifications/V-03).

Criterion (V-03): the linear speed of the Odometry is zero from each placement + T_settle until the
removal; it is within v_r +- e_r from each removal + T_resume until the next placement, or the end;
no fault code is active from the first placement to the end. The reflex bit is reported, not
judged.

Validity of the run: three placements and removals, in order; the echo of the State shows
TWIST(v_r, 0) from before the first placement to the end, so no stop comes from an expired command;
the speed is within v_r +- e_r in the second before the first placement, so the robot was moving
(before the others, the criterion of the resume judges it); the State is recorded without holes;
and the Odometry is recorded in every judged window.

Usage: check_v03 <bag directory>. Exit code: 0 pass, 1 fail, 2 invalid run.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import sys
from typing import TYPE_CHECKING

from robot_verification.recording import (
    ControllerRecording,
    Sample,
    Stream,
    Verdict,
    is_close_float32,
    read_bag,
)

if TYPE_CHECKING:
    from nav_msgs.msg import Odometry
    from robot_interfaces.msg import State

# Parameters of V-03.
V_R: float = 0.2  # m/s, commanded in the trial
T_O_S: float = 2.0  # time the screen is held
EXPECTED_SCREENS: int = 3

# Parameters of the controller (robot-docs nota 0011, nota 0014, nota 0015).
T_C_S: float = 0.010
T_RANGING_S: float = 0.033  # one ranging of the VL53L0X
A_DEC: float = 0.5  # m/s²
A_ACCEL: float = 0.25  # m/s²
V_MIN: float = 0.07  # m/s, below it the PI brakes
BRAKE_S: float = 0.05  # from v_min to the wheel stopped
T_VEL_S: float = 0.3  # no edge for T_vel reads as speed zero
WHEEL_SETTLING_S: float = 0.5  # t_s of design

# Conditions of the run.
OPERATOR_REACTION_S: float = 0.5  # from an announcement to the screen in place, on a countdown
LINK_DELAY_S: float = 0.05  # from the controller to the bag
ODOMETRY_PERIOD_S: float = 0.05  # 20 Hz (ADR 027)
SENSOR_LATENCY_S: float = T_RANGING_S + T_C_S  # the next ranging, then the next cycle
REPORT_DELAY_S: float = LINK_DELAY_S + ODOMETRY_PERIOD_S

# T_settle: from "placed" to the measured speed zero in the bag. The reference ramps down to v_min,
# the PI brakes, and the speed reads zero T_vel after the last edge (robot-docs nota 0011).
STOP_S: float = (V_R - V_MIN) / A_DEC + BRAKE_S + T_VEL_S
T_SETTLE_S: float = OPERATOR_REACTION_S + SENSOR_LATENCY_S + STOP_S + REPORT_DELAY_S

# T_resume: from "removed" to the measured speed back at v_r in the bag: the ramp up, then the
# settling of the wheel loop.
RAMP_UP_S: float = V_R / A_ACCEL
T_RESUME_S: float = (
    OPERATOR_REACTION_S + SENSOR_LATENCY_S + RAMP_UP_S + WHEEL_SETTLING_S + REPORT_DELAY_S
)

# e_r: band around v_r. The noise per wheel was 2 sigma = 0.009 to 0.015 m/s and the overshoot
# from rest to 0.2 m/s up to 0.029 m/s (robot-docs nota 0011); the Odometry averages both wheels.
E_R: float = 0.03  # m/s
STEADY_BEFORE_S: float = 1.0  # window before each placement where the robot must be moving
MAX_STATE_GAP_S: float = 0.1  # longer is a hole in the recording

# Codes of State.msg, checked against the generated message when a bag is read.
REFLEX_FRONT_OBSTACLE: int = 1
TWIST: int = 0  # Command.TWIST

MARKER_TOPIC: str = '/verification/v03_marker'
PLACED, REMOVED = 'placed', 'removed'


@dataclass(frozen=True)
class ScreenWindow:
    """A placement of the screen as announced by run_v03: from "placed" to "removed"."""

    placed_s: float
    removed_s: float


def get_linear_speed(sample: Sample[Odometry]) -> float:
    return sample.message.twist.twist.linear.x


def is_at_trial_speed(speed: float) -> bool:
    return abs(speed - V_R) <= E_R


def has_active_fault(state: Sample[State]) -> bool:
    return state.message.faults != 0


def has_front_obstacle_reflex(state: Sample[State]) -> bool:
    return bool(state.message.reflexes & REFLEX_FRONT_OBSTACLE)


def echoes_trial_command(state: Sample[State]) -> bool:
    echo = state.message.echo
    return echo.behavior == TWIST and is_close_float32(echo.linear, V_R) and echo.angular == 0.0


def get_marker_times(markers: Stream[str], name: str) -> list[float]:
    return [marker.time_s for marker in markers if marker.message == name]


def build_screen_windows(markers: Stream[str]) -> list[ScreenWindow]:
    """Pair each "placed" with the "removed" that follows it, in order."""
    placed, removed = get_marker_times(markers, PLACED), get_marker_times(markers, REMOVED)
    return [ScreenWindow(p, r) for p, r in zip(placed, removed, strict=False)]


def find_holes(states: list[Sample[State]]) -> list[tuple[Sample[State], Sample[State]]]:
    """Return each pair of consecutive states farther apart than MAX_STATE_GAP_S."""
    return [
        (before, after)
        for before, after in itertools.pairwise(states)
        if after.time_s - before.time_s > MAX_STATE_GAP_S
    ]


class ScreenRun:
    """The quantities measured on the recording of a V-03 run, window by window."""

    def __init__(self, recording: ControllerRecording, markers: Stream[str]):
        self.recording: ControllerRecording = recording
        self.markers: Stream[str] = markers
        self.windows: list[ScreenWindow] = build_screen_windows(markers)

    @classmethod
    def from_bag(cls, path: str) -> ScreenRun:
        # ROS imports here, so the criterion can be tested without ROS.
        from robot_interfaces.msg import Command, State
        from std_msgs.msg import String

        assert State.REFLEX_FRONT_OBSTACLE == REFLEX_FRONT_OBSTACLE
        assert Command.TWIST == TWIST
        streams = read_bag(path, {MARKER_TOPIC: String})
        markers = Stream(Sample(m.time_s, m.message.data) for m in streams[MARKER_TOPIC])
        return cls(ControllerRecording.from_bag(path), markers)

    @property
    def end_s(self) -> float:
        """Return the time of the last recorded state, or -inf if none."""
        states = self.recording.states
        return states[-1].time_s if states else -math.inf

    def get_speeds_while_screened(self, window: ScreenWindow) -> list[float]:
        return self._get_speeds_between(window.placed_s + T_SETTLE_S, window.removed_s)

    def get_speeds_after_removal(self, index: int) -> list[float]:
        start_s = self.windows[index].removed_s + T_RESUME_S
        is_last = index + 1 == len(self.windows)
        end_s = self.end_s if is_last else self.windows[index + 1].placed_s
        return self._get_speeds_between(start_s, end_s)

    def get_speeds_before_placement(self, window: ScreenWindow) -> list[float]:
        return self._get_speeds_between(window.placed_s - STEADY_BEFORE_S, window.placed_s)

    def find_faulty_states(self) -> list[Sample[State]]:
        """Return the states with a fault code, from the first placement to the end."""
        states = self.recording.states.between(self.windows[0].placed_s, math.inf)
        return [state for state in states if has_active_fault(state)]

    def find_states_without_trial_echo(self) -> list[Sample[State]]:
        """Return the states with another echo, from the second before the first placement."""
        start_s = self.windows[0].placed_s - STEADY_BEFORE_S
        states = self.recording.states.between(start_s, math.inf)
        return [state for state in states if not echoes_trial_command(state)]

    def find_state_holes(self) -> list[tuple[Sample[State], Sample[State]]]:
        return find_holes(list(self.recording.states))

    def count_reflex_states(self, window: ScreenWindow) -> int:
        """Return how many states of the window report the front obstacle reflex (report only)."""
        states = self.recording.states.between(window.placed_s, window.removed_s)
        return sum(1 for state in states if has_front_obstacle_reflex(state))

    def _get_speeds_between(self, start_s: float, end_s: float) -> list[float]:
        return [get_linear_speed(s) for s in self.recording.odometry.between(start_s, end_s)]


class V03Verification:
    """The criterion and the validity conditions of V-03, and its verdict."""

    def __init__(self, run: ScreenRun):
        self.run: ScreenRun = run

    # --- Verdict ---

    @property
    def verdict(self) -> Verdict:
        if not self.run_is_valid:
            return Verdict.INVALID
        passed = (
            all(self.did_stop_while_screened(window) for window in self.run.windows)
            and all(self.did_resume_after(index) for index in range(len(self.run.windows)))
            and self.was_free_of_faults()
        )
        return Verdict.PASS if passed else Verdict.FAIL

    @property
    def run_is_valid(self) -> bool:
        return (
            self.were_screens_marked()
            and self.was_command_echoed_throughout()
            and self.was_moving_before_first_placement()
            and self.was_state_recorded_without_holes()
            and self.was_odometry_recorded_in_every_window()
        )

    # --- Criterion ---

    def did_stop_while_screened(self, window: ScreenWindow) -> bool:
        return all(speed == 0.0 for speed in self.run.get_speeds_while_screened(window))

    def did_resume_after(self, index: int) -> bool:
        return all(is_at_trial_speed(speed) for speed in self.run.get_speeds_after_removal(index))

    def was_free_of_faults(self) -> bool:
        return not self.run.find_faulty_states()

    # --- Validity of the run ---

    def were_screens_marked(self) -> bool:
        """Check for three placements and removals, alternating, starting with a placement."""
        placed = get_marker_times(self.run.markers, PLACED)
        removed = get_marker_times(self.run.markers, REMOVED)
        alternating = sorted(placed + removed) == [
            t for window in self.run.windows for t in (window.placed_s, window.removed_s)
        ]
        return len(placed) == len(removed) == EXPECTED_SCREENS and alternating

    def was_command_echoed_throughout(self) -> bool:
        return not self.run.find_states_without_trial_echo()

    def was_moving_before_first_placement(self) -> bool:
        speeds = self.run.get_speeds_before_placement(self.run.windows[0])
        return bool(speeds) and all(is_at_trial_speed(speed) for speed in speeds)

    def was_state_recorded_without_holes(self) -> bool:
        return bool(self.run.recording.states) and not self.run.find_state_holes()

    def was_odometry_recorded_in_every_window(self) -> bool:
        screened = (self.run.get_speeds_while_screened(w) for w in self.run.windows)
        resumed = (self.run.get_speeds_after_removal(i) for i in range(len(self.run.windows)))
        return all(itertools.chain(screened, resumed))


class V03Report:
    """The report of a V-03 run: the checks of each window and the verdict."""

    def __init__(self, verification: V03Verification):
        self.verification: V03Verification = verification
        self.run: ScreenRun = verification.run

    def __str__(self) -> str:
        lines = [self._format_header_line(), self._format_tolerance_line()]
        if self.verification.were_screens_marked():
            lines += [self._format_moving_line()]
            lines += self._format_window_lines() + self._format_run_lines()
        lines.append(f'V-03: {self.verification.verdict.name}')
        return '\n'.join(lines)

    def _format_header_line(self) -> str:
        return f'V-03: {len(self.run.windows)} screens ({EXPECTED_SCREENS} expected)'

    def _format_tolerance_line(self) -> str:
        return (
            f'  zero from {T_SETTLE_S:.2f} s after each placement; v_r +- {E_R} m/s from '
            f'{T_RESUME_S:.2f} s after each removal'
        )

    def _format_moving_line(self) -> str:
        moving = self.verification.was_moving_before_first_placement()
        return f'  at v_r before the first placement: {moving}'

    def _format_window_lines(self) -> list[str]:
        return [self._format_window_line(index) for index in range(len(self.run.windows))]

    def _format_window_line(self, index: int) -> str:
        window = self.run.windows[index]
        return (
            f'  screen {index + 1}: stopped: {self.verification.did_stop_while_screened(window)}, '
            f'resumed: {self.verification.did_resume_after(index)}, '
            f'reflex states: {self.run.count_reflex_states(window)}'
        )

    def _format_run_lines(self) -> list[str]:
        verification = self.verification
        return [
            f'  states with a fault code: {len(self.run.find_faulty_states())}',
            f'  echo of TWIST(v_r, 0) throughout: {verification.was_command_echoed_throughout()}',
            f'  state recorded without holes: {verification.was_state_recorded_without_holes()}',
        ]


def main() -> None:
    if len(sys.argv) != 2:
        print('usage: check_v03 <bag directory>')
        sys.exit(Verdict.INVALID.value)
    verification = V03Verification(ScreenRun.from_bag(sys.argv[1]))
    print(V03Report(verification))
    sys.exit(verification.verdict.value)


if __name__ == '__main__':
    main()
