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
judged. Zero is an exact 0.0: the firmware publishes a literal zero after T_vel with no edge, and
the Odometry averages the two wheels.

Validity of the run: three placements and removals, in order; the State is recorded from a second
before the first placement, without holes; its echo shows TWIST(v_r, 0) from then to the end, so no
stop comes from an expired command; the speed is within v_r +- e_r in the second before the first
placement, so the robot was moving (before the others, the criterion of the resume judges it); and
every judged window holds at least MIN_WINDOW_SAMPLES of the Odometry, which also catches a
recording cut short.

Usage: check_v03 <bag directory>. Exit code: 0 pass, 1 fail, 2 invalid run.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from typing import TYPE_CHECKING

from robot_verification.recording import (
    ControllerRecording,
    Sample,
    Stream,
    Verdict,
    find_holes,
    is_close_float32,
    read_markers,
)

if TYPE_CHECKING:
    from nav_msgs.msg import Odometry
    from robot_interfaces.msg import State

# Parameters of V-03.
V_R: float = 0.2  # m/s, commanded in the trial
T_O_S: float = 2.0  # time the screen is held; run_v03 counts it down
EXPECTED_SCREENS: int = 3

# Parameters of the controller (robot-docs nota 0011, nota 0014, nota 0015).
T_C_S: float = 0.010
T_RANGING_S: float = 0.033  # one ranging of the VL53L0X
A_DEC: float = 0.5  # m/s²
A_ACEL: float = 0.25  # m/s²
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
RAMP_UP_S: float = V_R / A_ACEL
T_RESUME_S: float = (
    OPERATOR_REACTION_S + SENSOR_LATENCY_S + RAMP_UP_S + WHEEL_SETTLING_S + REPORT_DELAY_S
)

# e_r: band around v_r. The noise per wheel was 2 sigma = 0.009 to 0.015 m/s and the overshoot
# from rest to 0.2 m/s up to 0.029 m/s (robot-docs nota 0011); the Odometry averages both wheels.
E_R: float = 0.03  # m/s
STEADY_BEFORE_S: float = 1.0  # window before each placement where the robot must be moving
MAX_STATE_GAP_S: float = 0.1  # longer is a hole in the recording
# Fewest Odometry samples in a judged window: 0.5 s at 20 Hz, two thirds of the shortest window
# (zero while screened, T_o - T_settle = 0.75 s).
MIN_WINDOW_SAMPLES: int = 10

# Codes of State.msg, checked against the generated message when a bag is read.
REFLEX_FRONT_OBSTACLE: int = 1
TWIST: int = 0  # Command.TWIST

MARKER_TOPIC: str = '/verification/v03_marker'
PLACED: str = 'placed'
REMOVED: str = 'removed'


@dataclass(frozen=True)
class ScreenWindow:
    """
    A placement of the screen as announced by run_v03: from "placed" to "removed".

    next_placed_s bounds the resume after the removal; None for the last screen, whose resume runs
    to the end of the recording.
    """

    placed_s: float
    removed_s: float
    next_placed_s: float | None


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
    """
    Pair the k-th "placed" with the k-th "removed", in order.

    An unpaired marker is dropped here; were_screens_marked then rejects the run, because the
    markers no longer match the windows.
    """
    placed, removed = get_marker_times(markers, PLACED), get_marker_times(markers, REMOVED)
    next_placed: list[float | None] = [*placed[1:], None]
    pairs = zip(placed, removed, next_placed, strict=False)
    return [ScreenWindow(p, r, n) for p, r, n in pairs]


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

        assert State.REFLEX_FRONT_OBSTACLE == REFLEX_FRONT_OBSTACLE
        assert Command.TWIST == TWIST
        return cls(ControllerRecording.from_bag(path), read_markers(path, MARKER_TOPIC))

    def get_window_marker_times(self) -> list[float]:
        """Return the times of the windows in order: placed, removed, placed, removed..."""
        return [t for window in self.windows for t in (window.placed_s, window.removed_s)]

    def get_judged_window_speeds(self) -> list[list[float]]:
        """Return the speeds of every judged window, the one before the first placement first."""
        before = [self.get_speeds_before_placement(self.windows[0])]
        screened = [self.get_speeds_while_screened(window) for window in self.windows]
        resumed = [self.get_speeds_after_removal(window) for window in self.windows]
        return before + screened + resumed

    def get_speeds_while_screened(self, window: ScreenWindow) -> list[float]:
        return self._get_speeds_between(window.placed_s + T_SETTLE_S, window.removed_s)

    def get_speeds_after_removal(self, window: ScreenWindow) -> list[float]:
        end_s = self.end_s if window.next_placed_s is None else window.next_placed_s
        return self._get_speeds_between(window.removed_s + T_RESUME_S, end_s)

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
        return find_holes(self.recording.states, MAX_STATE_GAP_S)

    def count_reflex_states(self, window: ScreenWindow) -> int:
        """Return how many states of the window report the front obstacle reflex (report only)."""
        states = self.recording.states.between(window.placed_s, window.removed_s)
        return sum(1 for state in states if has_front_obstacle_reflex(state))

    @property
    def start_s(self) -> float:
        """Return the time of the first recorded state, or inf if none."""
        states = self.recording.states
        return states[0].time_s if states else math.inf

    @property
    def end_s(self) -> float:
        """Return the time of the last recorded state, or -inf if none."""
        states = self.recording.states
        return states[-1].time_s if states else -math.inf

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
            and all(self.did_resume_after(window) for window in self.run.windows)
            and self.was_free_of_faults()
        )
        return Verdict.PASS if passed else Verdict.FAIL

    @property
    def run_is_valid(self) -> bool:
        # were_screens_marked comes first: the other conditions read the first window.
        return (
            self.were_screens_marked()
            and self.was_state_recorded_from_steady_start()
            and self.was_state_recorded_without_holes()
            and self.was_command_echoed_throughout()
            and self.was_odometry_recorded_in_every_window()
            and self.was_moving_before_first_placement()
        )

    # --- Criterion ---

    def did_stop_while_screened(self, window: ScreenWindow) -> bool:
        return all(speed == 0.0 for speed in self.run.get_speeds_while_screened(window))

    def did_resume_after(self, window: ScreenWindow) -> bool:
        return all(is_at_trial_speed(speed) for speed in self.run.get_speeds_after_removal(window))

    def was_free_of_faults(self) -> bool:
        return not self.run.find_faulty_states()

    # --- Validity of the run ---

    def were_screens_marked(self) -> bool:
        """Check for three placements and removals, alternating, starting with a placement."""
        placed = get_marker_times(self.run.markers, PLACED)
        removed = get_marker_times(self.run.markers, REMOVED)
        alternating = sorted(placed + removed) == self.run.get_window_marker_times()
        return len(placed) == len(removed) == EXPECTED_SCREENS and alternating

    def was_state_recorded_from_steady_start(self) -> bool:
        return self.run.start_s <= self.run.windows[0].placed_s - STEADY_BEFORE_S

    def was_state_recorded_without_holes(self) -> bool:
        return bool(self.run.recording.states) and not self.run.find_state_holes()

    def was_command_echoed_throughout(self) -> bool:
        return not self.run.find_states_without_trial_echo()

    def was_odometry_recorded_in_every_window(self) -> bool:
        windows = self.run.get_judged_window_speeds()
        return all(len(speeds) >= MIN_WINDOW_SAMPLES for speeds in windows)

    def was_moving_before_first_placement(self) -> bool:
        speeds = self.run.get_speeds_before_placement(self.run.windows[0])
        return all(is_at_trial_speed(speed) for speed in speeds)


class V03Report:
    """The report of a V-03 run: the checks of each window and the verdict."""

    def __init__(self, verification: V03Verification):
        self.verification: V03Verification = verification
        self.run: ScreenRun = verification.run

    def __str__(self) -> str:
        lines = [self._format_header_line(), self._format_tolerance_line()]
        if self.verification.were_screens_marked():
            lines += self._format_validity_lines()
            lines += self._format_window_lines()
            lines.append(self._format_fault_line())
        lines.append(f'V-03: {self.verification.verdict.name}')
        return '\n'.join(lines)

    def _format_header_line(self) -> str:
        return f'V-03: {len(self.run.windows)} screens ({EXPECTED_SCREENS} expected)'

    def _format_tolerance_line(self) -> str:
        return (
            f'  zero from {T_SETTLE_S:.2f} s after each placement; v_r +- {E_R} m/s from '
            f'{T_RESUME_S:.2f} s after each removal'
        )

    def _format_validity_lines(self) -> list[str]:
        verification = self.verification
        conditions = [
            ('state from a second before', verification.was_state_recorded_from_steady_start()),
            ('state without holes', verification.was_state_recorded_without_holes()),
            ('echo of TWIST(v_r, 0) throughout', verification.was_command_echoed_throughout()),
            ('odometry in every window', verification.was_odometry_recorded_in_every_window()),
            (
                'at v_r before the first placement',
                verification.was_moving_before_first_placement(),
            ),
        ]
        return [f'  {name}: {held}' for name, held in conditions]

    def _format_window_lines(self) -> list[str]:
        windows = enumerate(self.run.windows, start=1)
        return [self._format_window_line(number, window) for number, window in windows]

    def _format_window_line(self, number: int, window: ScreenWindow) -> str:
        return (
            f'  screen {number}: stopped: {self.verification.did_stop_while_screened(window)}, '
            f'resumed: {self.verification.did_resume_after(window)}, '
            f'reflex states: {self.run.count_reflex_states(window)}'
        )

    def _format_fault_line(self) -> str:
        return f'  states with a fault code: {len(self.run.find_faulty_states())}'


def main() -> None:
    if len(sys.argv) != 2:
        print('usage: check_v03 <bag directory>')
        sys.exit(Verdict.INVALID.value)
    verification = V03Verification(ScreenRun.from_bag(sys.argv[1]))
    print(V03Report(verification))
    sys.exit(verification.verdict.value)


if __name__ == '__main__':
    main()
