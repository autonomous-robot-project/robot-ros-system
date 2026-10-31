"""Unit tests of the V-01 criterion on synthetic executions."""

from dataclasses import dataclass, field
import math

from robot_verification.check_v01 import (
    BAG,
    FAILURE,
    NODES_AFTER,
    NODES_BEFORE,
    NOMINAL,
    V_1,
    Execution,
    V01Report,
    V01Verification,
    find_missing_paths,
)
from robot_verification.recording import ControllerRecording, Sample, Verdict
from synthetic import command, go_to, periodic, state, turn_to

STATE_PERIOD_S = 0.01  # 100 Hz
COMMAND_PERIOD_S = 0.05  # republished at 20 Hz
ECHO_DELAY_S = 0.01
HOLE_S = 0.5
FAULT_COGNIZANT = 1
FAULT_DEADLINE_MISSED = 2
LISTED_NODES = [
    '/controller',
    '/sequencer',
    '/rosbag2_recorder',
    '/run_v01',
    '/_ros2cli_daemon_0_1a',
]

S1 = command(V_1, 0.0)
S2 = turn_to(math.pi / 2)
S3 = go_to(1.0, 0.5)
S4 = command(0.0, 0.0)


@dataclass
class SyntheticV01Execution:
    """
    An execution as run_v01 would record it.

    The sequencer sends each command of `changes` at its time and republishes the one in force at
    20 Hz from idle_from_s (TWIST(0, 0) before the first change); the controller echoes a command
    ECHO_DELAY_S after it is sent, except the change at index `unechoed`.
    """

    changes: list[tuple[float, object]]
    end_s: float
    idle_from_s: float = 0.0
    fault_from_s: float | None = None  # FAULT_COGNIZANT active from this time
    fault_until_s: float = math.inf
    other_faults: int = 0  # active through the whole execution
    unechoed: int | None = None
    hole_at_s: float | None = None  # no State for HOLE_S from this time
    nodes_before: list[str] = field(default_factory=lambda: list(LISTED_NODES))
    nodes_after: list[str] = field(default_factory=lambda: list(LISTED_NODES))

    def command_at(self, time_s: float) -> object:
        sent = [c for t, c in self.changes if t <= time_s]
        return sent[-1] if sent else S4

    def echo_at(self, time_s: float) -> object:
        echoed = [
            c
            for index, (t, c) in enumerate(self.changes)
            if t + ECHO_DELAY_S <= time_s and index != self.unechoed
        ]
        return echoed[-1] if echoed else S4

    def faults_at(self, time_s: float) -> int:
        fault_on = (
            self.fault_from_s is not None and self.fault_from_s <= time_s < self.fault_until_s
        )
        return self.other_faults | (FAULT_COGNIZANT if fault_on else 0)

    def is_in_hole(self, time_s: float) -> bool:
        return self.hole_at_s is not None and self.hole_at_s <= time_s < self.hole_at_s + HOLE_S

    def build(self) -> Execution:
        republished = periodic(self.idle_from_s, self.end_s, COMMAND_PERIOD_S, self.command_at)
        sent = [Sample(t, c) for t, c in self.changes]
        commands = sorted(republished + sent, key=lambda sample: sample.time_s)
        states = [
            sample
            for sample in periodic(
                0.0,
                self.end_s,
                STATE_PERIOD_S,
                lambda t: state(self.echo_at(t), faults=self.faults_at(t)),
            )
            if not self.is_in_hole(sample.time_s)
        ]
        recording = ControllerRecording(commands, states, [])
        return Execution(recording, self.nodes_before, self.nodes_after)


def nominal(**changes) -> SyntheticV01Execution:
    run = SyntheticV01Execution(changes=[(1.0, S1), (6.5, S2), (8.0, S3), (11.5, S4)], end_s=16.5)
    for name, value in changes.items():
        setattr(run, name, value)
    return run


def failure(**changes) -> SyntheticV01Execution:
    run = SyntheticV01Execution(
        changes=[(1.0, S1), (3.012, S4)], end_s=8.0, fault_from_s=3.0, fault_until_s=3.3
    )
    for name, value in changes.items():
        setattr(run, name, value)
    return run


def verify(nominal_run: SyntheticV01Execution, failure_run: SyntheticV01Execution):
    return V01Verification(nominal_run.build(), failure_run.build())


def test_a_correct_run_passes():
    verification = verify(nominal(), failure())

    assert verification.is_rf01_met()
    assert verification.is_rf04_met()
    assert verification.is_rf05_met()
    assert verification.verdict is Verdict.PASS


def test_the_idle_twist_zero_before_the_mission_is_not_a_stage():
    assert verify(nominal(idle_from_s=0.0), failure(idle_from_s=0.0)).verdict is Verdict.PASS
    assert verify(nominal(idle_from_s=1.0), failure(idle_from_s=1.0)).verdict is Verdict.PASS


def test_a_change_never_echoed_fails_rf01():
    verification = verify(nominal(unechoed=2), failure())

    assert not verification.is_rf01_met()
    assert verification.verdict is Verdict.FAIL


def test_a_nominal_sequence_without_go_to_fails_rf04():
    skipped = nominal(changes=[(1.0, S1), (6.5, S2), (8.0, S4)])

    assert verify(skipped, failure()).verdict is Verdict.FAIL


def test_a_turn_after_the_fault_fails_rf04():
    turned = failure(changes=[(1.0, S1), (3.012, S2)])

    assert not verify(nominal(), turned).is_rf04_met()


def test_a_stop_before_the_fault_fails_rf04():
    # S1 -> S4, but the fault turns on only after the stop: the stop did not follow it.
    early_stop = failure(changes=[(1.0, S1), (2.5, S4)], fault_from_s=3.0)

    verification = verify(nominal(), early_stop)

    assert not verification.did_stop_follow_the_fault()
    assert verification.verdict is Verdict.FAIL


def test_a_fault_in_the_nominal_execution_fails_rf04():
    aborted = nominal(changes=[(1.0, S1), (3.012, S4)], fault_from_s=3.0, fault_until_s=3.3)

    assert verify(aborted, failure()).verdict is Verdict.FAIL


def test_a_deliberator_node_fails_rf05():
    with_planner = nominal(nodes_after=[*LISTED_NODES, '/planner_server'])

    verification = verify(with_planner, failure())

    assert not verification.is_rf05_met()
    assert verification.verdict is Verdict.FAIL


def test_a_list_without_the_sequencer_fails_rf05():
    # An empty list (discovery failed) proves nothing.
    assert verify(nominal(), failure(nodes_before=[])).verdict is Verdict.FAIL


def test_a_hole_in_the_state_invalidates_the_run():
    assert verify(nominal(hole_at_s=7.0), failure()).verdict is Verdict.INVALID


def test_a_failure_execution_without_a_fault_is_invalid():
    # The wheel was not locked: the sequence cannot be judged.
    assert verify(nominal(), failure(fault_from_s=None)).verdict is Verdict.INVALID


def test_a_fault_already_active_before_the_run_is_invalid():
    # Active from the first State: the bag has no activation to follow.
    assert verify(nominal(), failure(fault_from_s=0.0)).verdict is Verdict.INVALID


def test_another_code_that_stays_active_is_not_an_activation():
    # A code active from the first State never turns on in the bag; the wheel fault still does.
    assert verify(nominal(), failure(other_faults=FAULT_DEADLINE_MISSED)).verdict is Verdict.PASS


def test_the_report_names_each_stage_and_ends_with_the_verdict():
    report = str(V01Report(verify(nominal(), failure())))

    assert 'S1  TWIST(0.20 m/s, 0.00 rad/s)' in report
    assert 'S3  GO_TO(1.000, 0.500 m)' in report
    assert 'activation in S1 -> next change in 12 ms' in report
    assert report.endswith('V-01: PASS')


def test_find_missing_paths_lists_what_run_v01_did_not_save(tmp_path):
    for execution in (NOMINAL, FAILURE):
        (tmp_path / execution / BAG).mkdir(parents=True)
        (tmp_path / execution / NODES_BEFORE).write_text('/sequencer\n')
    (tmp_path / NOMINAL / NODES_AFTER).write_text('/sequencer\n')

    assert find_missing_paths(tmp_path) == [tmp_path / FAILURE / NODES_AFTER]
