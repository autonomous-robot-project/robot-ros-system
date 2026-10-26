"""Unit tests of the V-02 criterion on synthetic state, instrumentation and marker streams."""

from dataclasses import dataclass

from robot_verification.check_v02 import (
    COAST_DOWN_S,
    CUT,
    DETECTION_LATENCY_S,
    EXPECTED_CUTS,
    FAULT_COGNIZANT,
    FAULT_WATCHDOG_RESTART,
    INJECT,
    OPERATOR_REACTION_S,
    RESTORE,
    V02,
    V_R,
    Recording,
)
from robot_verification.recording import Sample, Verdict
from synthetic import command, instrumentation, periodic, state

STATE_PERIOD_S = 0.01  # 100 Hz
INSTRUMENTATION_PERIOD_S = 0.1  # 10 Hz
START_S = 3.0  # steady state before the first cut
CUT_S = 3.0  # V-02 procedure
BETWEEN_CUTS_S = 5.0
RESTART_S = 3.0  # no state while the controller restarts and the session comes back
AFTER_RESTART_S = 10.0


@dataclass
class SyntheticV02Run:
    """
    A V-02 run as run_v02 would record it, from a controller that behaves as configured.

    The operator switches the supply off operator_reaction_s after each "cut" and back on
    operator_reaction_s after each "restore"; once off, the wheel takes coast_down_s to slow down.
    """

    cuts: int = EXPECTED_CUTS
    operator_reaction_s: float = 0.2
    coast_down_s: float = 0.3
    detection_s: float = DETECTION_LATENCY_S  # from the slowed-down wheel to the failure code
    clears_after_s: float | None = 0.3  # from the supply back on to the code off; None: never
    false_failure_at_start: bool = False
    wheel_stops: bool = True
    command_applied: bool = True
    watchdog_code_after_restart: bool = True

    def cut_times(self):
        return [START_S + k * (CUT_S + BETWEEN_CUTS_S) for k in range(self.cuts)]

    def injection_s(self):
        return START_S + self.cuts * (CUT_S + BETWEEN_CUTS_S)

    def recording(self):
        injection = self.injection_s()
        restart_end = injection + RESTART_S
        states = periodic(0.0, injection, STATE_PERIOD_S, self._state_at) + periodic(
            restart_end, restart_end + AFTER_RESTART_S, STATE_PERIOD_S, self._state_after_restart
        )
        samples = periodic(0.0, injection, INSTRUMENTATION_PERIOD_S, self._instrumentation_at)
        return Recording(states, samples, self._markers())

    def _markers(self):
        markers = []
        for cut in self.cut_times():
            markers += [Sample(cut, CUT), Sample(cut + CUT_S, RESTORE)]
        return markers + [Sample(self.injection_s(), INJECT)]

    def _supply_off(self, cut):
        return cut + self.operator_reaction_s

    def _supply_on(self, cut):
        return cut + CUT_S + self.operator_reaction_s

    def _wheel_stopped(self, t):
        return any(
            self._supply_off(cut) + self.coast_down_s <= t < self._supply_on(cut)
            for cut in self.cut_times()
        )

    def _failure_active(self, t):
        if self.false_failure_at_start and 1.0 <= t < 1.5:
            return True
        for cut in self.cut_times():
            detected = self._supply_off(cut) + self.coast_down_s + self.detection_s
            never_clears = self.clears_after_s is None
            cleared = (
                self.injection_s() if never_clears else self._supply_on(cut) + self.clears_after_s
            )
            if detected <= t < cleared:
                return True
        return False

    def _state_at(self, t):
        echo = command(V_R) if self.command_applied else command(0.0)
        return state(echo, FAULT_COGNIZANT if self._failure_active(t) else 0)

    def _state_after_restart(self, _t):
        faults = FAULT_WATCHDOG_RESTART if self.watchdog_code_after_restart else 0
        return state(command(V_R), faults)

    def _instrumentation_at(self, t):
        stopped = self.wheel_stops and self._wheel_stopped(t)
        return instrumentation(left_wheel_speed=0.0 if stopped else V_R)


def verdict_of(run):
    return V02(run.recording()).verdict


def test_pass_when_the_failure_follows_each_cut():
    assert verdict_of(SyntheticV02Run()) == Verdict.PASS


def test_pass_with_the_slowest_operator_and_wheel_allowed():
    slowest = SyntheticV02Run(operator_reaction_s=OPERATOR_REACTION_S, coast_down_s=COAST_DOWN_S)
    assert verdict_of(slowest) == Verdict.PASS


def test_fail_when_detection_is_late_beyond_the_tolerances():
    # The tolerances for the operator and the wheel also absorb a late detection: only a delay
    # beyond them fails.
    late = SyntheticV02Run(detection_s=DETECTION_LATENCY_S + OPERATOR_REACTION_S + COAST_DOWN_S)
    assert verdict_of(late) == Verdict.FAIL


def test_fail_when_the_failure_does_not_clear():
    assert verdict_of(SyntheticV02Run(clears_after_s=None)) == Verdict.FAIL


def test_fail_when_the_start_is_read_as_a_failure():
    assert verdict_of(SyntheticV02Run(false_failure_at_start=True)) == Verdict.FAIL


def test_fail_without_the_watchdog_code_after_the_restart():
    assert verdict_of(SyntheticV02Run(watchdog_code_after_restart=False)) == Verdict.FAIL


def test_invalid_when_the_wheel_did_not_stop():
    assert verdict_of(SyntheticV02Run(wheel_stops=False)) == Verdict.INVALID


def test_invalid_when_the_controller_ignores_the_command():
    assert verdict_of(SyntheticV02Run(command_applied=False)) == Verdict.INVALID


def test_invalid_with_wrong_number_of_cuts():
    assert verdict_of(SyntheticV02Run(cuts=2)) == Verdict.INVALID
