"""Unit tests of the V-03 criterion on synthetic state, odometry and marker streams."""

from dataclasses import dataclass

from robot_verification.check_v03 import (
    E_R,
    EXPECTED_SCREENS,
    PLACED,
    REFLEX_FRONT_OBSTACLE,
    REMOVED,
    T_O_S,
    T_RESUME_S,
    T_SETTLE_S,
    V_R,
    ScreenRun,
    V03Report,
    V03Verification,
)
from robot_verification.recording import ControllerRecording, Sample, Stream, Verdict
from synthetic import command, odometry, periodic, state

STATE_PERIOD_S = 0.01  # 100 Hz
ODOMETRY_PERIOD_S = 0.05  # 20 Hz
START_S = 3.0  # steady state before the first placement
BETWEEN_SCREENS_S = 4.0  # from a removal to the next placement
AFTER_LAST_S = 4.0
FAULT_FRONT_SENSOR = 8


@dataclass
class SyntheticV03Run:
    """
    A V-03 run as run_v03 would record it, from a controller that behaves as configured.

    The speed drops to zero stop_after_s after each "placed" and is back at v_r resume_after_s
    after each "removed"; None means never.
    """

    screens: int = EXPECTED_SCREENS
    stop_after_s: float | None = 1.0
    resume_after_s: float | None = 1.5
    speed_offset: float = 0.0  # added to v_r while moving
    fault_while_screened: bool = False
    command_applied: bool = True
    hole_at_s: float | None = None  # no state for 0.5 s from this time
    after_last_s: float = AFTER_LAST_S  # recording after the last removal
    states_from_s: float = 0.0  # first state recorded
    odometry_recorded: bool = True
    removal_first: bool = False  # operator error: each "removed" comes before its "placed"

    def placement_times(self) -> list[float]:
        return [START_S + k * (T_O_S + BETWEEN_SCREENS_S) for k in range(self.screens)]

    def end_s(self) -> float:
        last_removed_s = START_S + self.screens * (T_O_S + BETWEEN_SCREENS_S) - BETWEEN_SCREENS_S
        return last_removed_s + self.after_last_s

    def run(self) -> ScreenRun:
        states = [
            s
            for s in periodic(self.states_from_s, self.end_s(), STATE_PERIOD_S, self._state_at)
            if self._recorded(s)
        ]
        odometries = (
            periodic(0.0, self.end_s(), ODOMETRY_PERIOD_S, lambda t: odometry(self._speed_at(t)))
            if self.odometry_recorded
            else []
        )
        recording = ControllerRecording(
            commands=[], states=states, instrumentation=[], odometry=odometries
        )
        return ScreenRun(recording, Stream(self._markers()))

    def _markers(self) -> list[Sample[str]]:
        markers: list[Sample[str]] = []
        first, second = (REMOVED, PLACED) if self.removal_first else (PLACED, REMOVED)
        for placed_s in self.placement_times():
            markers += [Sample(placed_s, first), Sample(placed_s + T_O_S, second)]
        return markers

    def _is_screened(self, time_s: float) -> bool:
        return any(p <= time_s < p + T_O_S for p in self.placement_times())

    def _is_stopped(self, time_s: float) -> bool:
        for placed_s in self.placement_times():
            stopped_s = None if self.stop_after_s is None else placed_s + self.stop_after_s
            resumed_s = (
                None if self.resume_after_s is None else placed_s + T_O_S + self.resume_after_s
            )
            after_stop = stopped_s is not None and time_s >= stopped_s
            before_resume = resumed_s is None or time_s < resumed_s
            next_placed_s = placed_s + T_O_S + BETWEEN_SCREENS_S
            if after_stop and before_resume and time_s < next_placed_s:
                return True
        return False

    def _speed_at(self, time_s: float) -> float:
        return 0.0 if self._is_stopped(time_s) else V_R + self.speed_offset

    def _state_at(self, time_s: float):
        echo = command(V_R) if self.command_applied else command(0.0)
        screened = self._is_screened(time_s)
        faults = FAULT_FRONT_SENSOR if self.fault_while_screened and screened else 0
        reflexes = REFLEX_FRONT_OBSTACLE if screened else 0
        return state(echo, faults=faults, reflexes=reflexes)

    def _recorded(self, sample: Sample) -> bool:
        return self.hole_at_s is None or not self.hole_at_s <= sample.time_s < self.hole_at_s + 0.5


def get_verdict(run: SyntheticV03Run) -> Verdict:
    return V03Verification(run.run()).verdict


def test_nominal_run_passes():
    assert get_verdict(SyntheticV03Run()) is Verdict.PASS


def test_stop_and_resume_just_within_the_tolerances_pass():
    run = SyntheticV03Run(stop_after_s=T_SETTLE_S - 0.06, resume_after_s=T_RESUME_S - 0.06)
    assert get_verdict(run) is Verdict.PASS


def test_report_shows_every_screen_and_the_verdict():
    report = str(V03Report(V03Verification(SyntheticV03Run().run())))
    assert report.count('stopped: True') == EXPECTED_SCREENS
    assert report.endswith('V-03: PASS')


def test_never_stopping_fails():
    assert get_verdict(SyntheticV03Run(stop_after_s=None)) is Verdict.FAIL


def test_stopping_late_fails():
    assert get_verdict(SyntheticV03Run(stop_after_s=T_SETTLE_S + 0.1)) is Verdict.FAIL


def test_never_resuming_fails():
    assert get_verdict(SyntheticV03Run(resume_after_s=None)) is Verdict.FAIL


def test_resuming_late_fails():
    assert get_verdict(SyntheticV03Run(resume_after_s=T_RESUME_S + 0.1)) is Verdict.FAIL


def test_fault_code_while_screened_fails():
    assert get_verdict(SyntheticV03Run(fault_while_screened=True)) is Verdict.FAIL


def test_speed_off_v_r_before_the_first_placement_is_invalid():
    assert get_verdict(SyntheticV03Run(speed_offset=E_R + 0.01)) is Verdict.INVALID


def test_missing_screen_is_invalid():
    assert get_verdict(SyntheticV03Run(screens=EXPECTED_SCREENS - 1)) is Verdict.INVALID


def test_expired_command_is_invalid():
    assert get_verdict(SyntheticV03Run(command_applied=False)) is Verdict.INVALID


def test_hole_in_the_state_is_invalid():
    assert get_verdict(SyntheticV03Run(hole_at_s=START_S + 1.0)) is Verdict.INVALID


def test_recording_cut_short_after_the_last_removal_is_invalid():
    run = SyntheticV03Run(after_last_s=T_RESUME_S + 0.2)
    assert get_verdict(run) is Verdict.INVALID


def test_state_starting_after_the_steady_window_is_invalid():
    assert get_verdict(SyntheticV03Run(states_from_s=START_S - 0.5)) is Verdict.INVALID


def test_recording_without_odometry_is_invalid():
    assert get_verdict(SyntheticV03Run(odometry_recorded=False)) is Verdict.INVALID


def test_removal_before_placement_is_invalid():
    assert get_verdict(SyntheticV03Run(removal_first=True)) is Verdict.INVALID


def test_report_of_a_run_without_markers_shows_the_invalid_verdict():
    report = str(V03Report(V03Verification(SyntheticV03Run(screens=0).run())))
    assert report.endswith('V-03: INVALID')
