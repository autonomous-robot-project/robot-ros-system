"""Unit tests of the closed-loop measurement on synthetic command, state and instrumentation."""

from dataclasses import dataclass

from robot_verification.measure_closed_loop import (
    FAULT_COGNIZANT,
    SCHEDULE,
    STOP_EXPECTED_S,
    STOP_LIMIT_S,
    ClosedLoopReport,
    ClosedLoopRun,
    ClosedLoopVerification,
)
from robot_verification.recording import ControllerRecording, Verdict
from synthetic import command, instrumentation, periodic, state

COMMAND_PERIOD_S = 0.05  # 20 Hz
STATE_PERIOD_S = 0.01  # 100 Hz
INSTRUMENTATION_PERIOD_S = 0.1  # 10 Hz
RIPPLE = 0.05  # relative, alternating sign: a spread of 5 %


@dataclass
class SyntheticClosedLoopRun:
    """
    A closed-loop run as run_closed_loop would record it.

    The controller behaves as configured: each wheel follows the commanded speed with RIPPLE,
    and after the stop it reads zero from stop_zero_s on.
    """

    stop_zero_s: float = STOP_EXPECTED_S
    limit_cycle_at: float | None = None  # commanded speed at which the wheel stops and restarts
    false_alarm: bool = False
    command_applied: bool = True
    skipped_steps: int = 0  # steps of the schedule left out at the end
    instrumentation_hole_s: tuple[float, float] | None = None  # no Instrumentation in between

    def build_schedule(self):
        steps, start_s = [], 0.0
        for linear, duration_s in SCHEDULE[: len(SCHEDULE) - self.skipped_steps]:
            steps.append((linear, start_s, start_s + duration_s))
            start_s += duration_s
        return steps

    def get_end_s(self):
        return self.build_schedule()[-1][2]

    def build_recording(self):
        return ControllerRecording(
            periodic(
                0.0, self.get_end_s(), COMMAND_PERIOD_S, lambda t: command(self._get_linear_at(t))
            ),
            periodic(0.0, self.get_end_s(), STATE_PERIOD_S, self._build_state_at),
            self._build_instrumentation(),
        )

    def _build_instrumentation(self):
        samples = periodic(
            0.0, self.get_end_s(), INSTRUMENTATION_PERIOD_S, self._build_instrumentation_at
        )
        if self.instrumentation_hole_s is None:
            return samples
        start_s, end_s = self.instrumentation_hole_s
        return [sample for sample in samples if not start_s <= sample.time_s < end_s]

    def _find_step_at(self, t):
        steps = self.build_schedule()
        index = next(i for i, (_, start_s, end_s) in enumerate(steps) if start_s <= t < end_s)
        return steps, index

    def _get_linear_at(self, t):
        steps, index = self._find_step_at(t)
        return steps[index][0]

    def _get_speed_at(self, t):
        steps, index = self._find_step_at(t)
        linear, start_s, _ = steps[index]
        if linear == 0.0:
            previous = steps[index - 1][0] if index > 0 else 0.0
            return previous if t - start_s < self.stop_zero_s else 0.0
        if linear == self.limit_cycle_at and round(t / INSTRUMENTATION_PERIOD_S) % 4 == 0:
            return 0.0
        sign = 1 if round(t / INSTRUMENTATION_PERIOD_S) % 2 else -1
        return linear * (1 + sign * RIPPLE)

    def _build_state_at(self, t):
        echo = command(self._get_linear_at(t) if self.command_applied else 0.0)
        faulted = self.false_alarm and 5.0 <= t < 5.5
        return state(echo, FAULT_COGNIZANT if faulted else 0)

    def _build_instrumentation_at(self, t):
        speed = self._get_speed_at(t)
        return instrumentation(left_wheel_speed=speed, right_wheel_speed=speed)


def judge(run):
    return ClosedLoopVerification(ClosedLoopRun(run.build_recording())).verdict


def test_pass_when_the_loop_holds_and_stops_as_expected():
    assert judge(SyntheticClosedLoopRun()) == Verdict.PASS


def test_pass_with_the_latest_stop_allowed():
    assert judge(SyntheticClosedLoopRun(stop_zero_s=STOP_LIMIT_S - 0.05)) == Verdict.PASS


def test_fail_when_the_speed_reads_zero_too_late():
    assert judge(SyntheticClosedLoopRun(stop_zero_s=STOP_LIMIT_S + 0.2)) == Verdict.FAIL


def test_fail_when_the_speed_never_reads_zero():
    assert judge(SyntheticClosedLoopRun(stop_zero_s=10.0)) == Verdict.FAIL


def test_fail_on_a_limit_cycle_at_the_lowest_speed():
    assert judge(SyntheticClosedLoopRun(limit_cycle_at=0.07)) == Verdict.FAIL


def test_fail_on_a_false_alarm():
    assert judge(SyntheticClosedLoopRun(false_alarm=True)) == Verdict.FAIL


def test_invalid_when_the_controller_ignores_the_command():
    assert judge(SyntheticClosedLoopRun(command_applied=False)) == Verdict.INVALID


def test_invalid_when_the_schedule_is_incomplete():
    assert judge(SyntheticClosedLoopRun(skipped_steps=1)) == Verdict.INVALID


def test_report_has_one_line_per_low_speed_and_wheel():
    report = str(
        ClosedLoopReport(
            ClosedLoopVerification(ClosedLoopRun(SyntheticClosedLoopRun().build_recording()))
        )
    )
    assert report.count('m/s, left:') == 3 and report.count('m/s, right:') == 3


def test_invalid_when_a_steady_window_is_short_of_samples():
    # The 0.07 m/s step spans 8 to 16 s; its steady window is the last 5 s.
    assert judge(SyntheticClosedLoopRun(instrumentation_hole_s=(12.0, 15.0))) == Verdict.INVALID
