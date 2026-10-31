"""
V-01: selection of behaviors (RF-01, RF-04, RF-05), judged on the two bags recorded by run_v01.

Criticality: T1, evidence (robot-docs codigo.md).

What: the sequencer selects the primitive behaviors of the controller from the result of the
previous ones, without the deliberator.
How: run_v01 records one bag per execution (nominal, then with a wheel locked by hand during S1)
and saves `ros2 node list` before and after each one. Each S_k is recognized by the behavior of a
change of the recorded command, never by data of the sequencer, which is under test: S1 =
TWIST(v₁, 0), S2 = TURN_TO, S3 = GO_TO, S4 = TWIST(0, 0). The parameters and the pose at each
change are reported, not judged. Times are reception times in the bag, on the clock of the RPi.
Why: V-01 (robot-docs verifications/V-01.md, nota 0016).

Criterion (V-01):
- RF-01: in each transition, the behavior and the parameters reported by the controller equal the
  ones sent: after each change of the command, a State echoes it before the next change.
- RF-04: the sequence is S1 -> S2 -> S3 -> S4 in the nominal execution and S1 -> S4 in the one with
  the fault, where the change to S4 follows the activation of a fault code: a code turns active
  while S1 is in force, and the next change of the command is S4.
- RF-05: both sequences run without the deliberator: before and after each execution, the node
  list has the sequencer and no node outside the allowed ones.

Validity of the run: the State is recorded without holes in both executions, and the one with the
fault has the activation of a fault code (the wheel was locked). The bags and the four node lists
must exist (checked before reading).

Usage: check_v01 <run directory>, with nominal/ and falha/ inside, each with the bag and the node
lists. Exit code: 0 pass, 1 fail, 2 invalid run.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import itertools
import math
from pathlib import Path
import sys
from typing import TYPE_CHECKING

from robot_verification.recording import (
    GO_TO,
    TURN_TO,
    TWIST,
    ControllerRecording,
    Sample,
    Verdict,
    describe_command,
    find_changes,
    find_holes,
    get_command_key,
    get_heading,
    is_close_float32,
)

if TYPE_CHECKING:
    from robot_interfaces.msg import Command, State

# Parameters of V-01.
V_1: float = 0.2  # m/s, v₁ of S1

MAX_STATE_GAP_S: float = 0.1  # longer is a hole in the recording (State at 100 Hz)

# The run directory: one subdirectory per execution, each with the bag and the node lists.
NOMINAL: str = 'nominal'
FAILURE: str = 'falha'
BAG: str = 'bag'
NODES_BEFORE: str = 'nodes_antes.txt'
NODES_AFTER: str = 'nodes_depois.txt'

# Nodes allowed besides the sequencer: the controller (micro-ROS), run_v01, the bag recorder, and
# the ros2 command line, whose daemon and probes are named _ros2cli_*.
SEQUENCER_NODE: str = '/sequencer'
ALLOWED_NODES: frozenset[str] = frozenset(
    {SEQUENCER_NODE, '/controller', '/run_v01', '/rosbag2_recorder'}
)
ALLOWED_NODE_PREFIX: str = '/_ros2cli'


class Stage(Enum):
    S1 = auto()  # TWIST(v₁, 0)
    S2 = auto()  # TURN_TO
    S3 = auto()  # GO_TO
    S4 = auto()  # TWIST(0, 0)


NOMINAL_SEQUENCE: list[Stage | None] = [Stage.S1, Stage.S2, Stage.S3, Stage.S4]
FAILURE_SEQUENCE: list[Stage | None] = [Stage.S1, Stage.S4]


def get_stage(command: Command) -> Stage | None:
    """Return the S_k of V-01 the command belongs to, or None for any other command."""
    if command.behavior == TURN_TO:
        return Stage.S2
    if command.behavior == GO_TO:
        return Stage.S3
    if command.behavior == TWIST and command.angular == 0.0:
        if is_close_float32(command.linear, V_1):
            return Stage.S1
        if command.linear == 0.0:
            return Stage.S4
    return None


def is_echo_of(echo: Command, command: Command) -> bool:
    sent, echoed = get_command_key(command), get_command_key(echo)
    return sent[0] == echoed[0] and all(
        is_close_float32(e, s) for e, s in zip(echoed[1:], sent[1:], strict=True)
    )


def is_allowed_node(name: str) -> bool:
    return name in ALLOWED_NODES or name.startswith(ALLOWED_NODE_PREFIX)


def find_missing_paths(run_directory: Path) -> list[Path]:
    paths = [
        run_directory / execution / name
        for execution in (NOMINAL, FAILURE)
        for name in (BAG, NODES_BEFORE, NODES_AFTER)
    ]
    return [path for path in paths if not path.exists()]


def read_node_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


@dataclass(frozen=True)
class Execution:
    """The recording of one execution of V-01 and the nodes listed before and after it."""

    recording: ControllerRecording
    nodes_before: list[str]
    nodes_after: list[str]

    @classmethod
    def from_directory(cls, path: Path) -> Execution:
        # ROS imports here, so the criterion can be tested without ROS.
        from robot_interfaces.msg import Command

        assert (Command.TWIST, Command.TURN_TO, Command.GO_TO) == (TWIST, TURN_TO, GO_TO)
        return cls(
            ControllerRecording.from_bag(str(path / BAG)),
            read_node_list(path / NODES_BEFORE),
            read_node_list(path / NODES_AFTER),
        )

    def get_command_changes(self) -> list[Sample[Command]]:
        """Return the changes of the command from the first one that is not TWIST(0, 0)."""
        # The sequencer sends TWIST(0, 0) while it waits for the first Odometry: not a stage.
        changes = find_changes(self.recording.commands, get_command_key)
        return list(itertools.dropwhile(lambda c: get_stage(c.message) is Stage.S4, changes))

    def get_stages(self) -> list[Stage | None]:
        return [get_stage(change.message) for change in self.get_command_changes()]

    def find_echo(self, change_index: int) -> Sample[State] | None:
        """Return the first State that echoes the change before the next one, or None."""
        changes = self.get_command_changes()
        start_s = changes[change_index].time_s
        end_s = changes[change_index + 1].time_s if change_index + 1 < len(changes) else math.inf
        states = self.recording.states.between(start_s, end_s)
        command = changes[change_index].message
        return next((s for s in states if is_echo_of(s.message.echo, command)), None)

    def find_fault_activations(self) -> list[Sample[State]]:
        """Return each State with a fault code that the State before it did not have."""
        return [
            after
            for before, after in itertools.pairwise(self.recording.states)
            if after.message.faults & ~before.message.faults
        ]

    def find_activations_during_first_stage(self) -> list[Sample[State]]:
        """Return the fault activations after the first change and before the second one."""
        changes = self.get_command_changes()
        if len(changes) < 2:
            return []
        start_s, end_s = changes[0].time_s, changes[1].time_s
        return [s for s in self.find_fault_activations() if start_s < s.time_s < end_s]

    def find_state_holes(self) -> list[tuple[Sample[State], Sample[State]]]:
        return find_holes(self.recording.states, MAX_STATE_GAP_S)

    def find_disallowed_nodes(self) -> list[str]:
        return [n for n in self.nodes_before + self.nodes_after if not is_allowed_node(n)]

    def was_sequencer_listed(self) -> bool:
        return SEQUENCER_NODE in self.nodes_before and SEQUENCER_NODE in self.nodes_after


class V01Verification:
    """The criterion and the validity conditions of V-01, and its verdict."""

    def __init__(self, nominal: Execution, failure: Execution):
        self.nominal: Execution = nominal
        self.failure: Execution = failure

    # --- Verdict ---

    @property
    def verdict(self) -> Verdict:
        if not self.run_is_valid:
            return Verdict.INVALID
        passed = self.is_rf01_met() and self.is_rf04_met() and self.is_rf05_met()
        return Verdict.PASS if passed else Verdict.FAIL

    @property
    def run_is_valid(self) -> bool:
        return self.were_states_recorded_without_holes() and self.was_a_fault_activated()

    # --- Criterion ---

    def is_rf01_met(self) -> bool:
        return self.was_every_change_echoed(self.nominal) and self.was_every_change_echoed(
            self.failure
        )

    def is_rf04_met(self) -> bool:
        return (
            self.nominal.get_stages() == NOMINAL_SEQUENCE
            and self.failure.get_stages() == FAILURE_SEQUENCE
            and self.did_stop_follow_the_fault()
        )

    def is_rf05_met(self) -> bool:
        return all(
            execution.was_sequencer_listed() and not execution.find_disallowed_nodes()
            for execution in (self.nominal, self.failure)
        )

    def was_every_change_echoed(self, execution: Execution) -> bool:
        changes = execution.get_command_changes()
        return all(execution.find_echo(index) is not None for index in range(len(changes)))

    def did_stop_follow_the_fault(self) -> bool:
        """With S1 -> S4 in the failure execution, S1 was in force when a fault code turned on."""
        return bool(self.failure.find_activations_during_first_stage())

    # --- Validity of the run ---

    def were_states_recorded_without_holes(self) -> bool:
        return all(
            bool(execution.recording.states) and not execution.find_state_holes()
            for execution in (self.nominal, self.failure)
        )

    def was_a_fault_activated(self) -> bool:
        return bool(self.failure.find_fault_activations())


class V01Report:
    """The report of a V-01 run: each change of the command, the checks and the verdict."""

    def __init__(self, verification: V01Verification):
        self.verification: V01Verification = verification

    def __str__(self) -> str:
        verification = self.verification
        lines = [
            *self._format_execution_lines(NOMINAL, verification.nominal),
            *self._format_execution_lines(FAILURE, verification.failure),
            *self._format_activation_lines(),
            f'  state without holes: {verification.were_states_recorded_without_holes()}',
            f'  fault activated in falha: {verification.was_a_fault_activated()}',
            f'  RF-01: {verification.is_rf01_met()}',
            f'  RF-04: {verification.is_rf04_met()}',
            f'  RF-05: {verification.is_rf05_met()}',
            f'V-01: {verification.verdict.name}',
        ]
        return '\n'.join(lines)

    def _format_execution_lines(self, name: str, execution: Execution) -> list[str]:
        changes = execution.get_command_changes()
        origin_s = changes[0].time_s if changes else 0.0
        lines = [f'V-01 {name}: {len(changes)} changes of the command']
        for index, change in enumerate(changes):
            lines.append(self._format_change_line(execution, index, change, origin_s))
        lines.append(f'  nodes before: {" ".join(execution.nodes_before)}')
        lines.append(f'  nodes after: {" ".join(execution.nodes_after)}')
        return lines

    def _format_change_line(
        self, execution: Execution, index: int, change: Sample[Command], origin_s: float
    ) -> str:
        stage = get_stage(change.message)
        echo = execution.find_echo(index)
        echo_text = f'echo +{(echo.time_s - change.time_s) * 1e3:.0f} ms' if echo else 'no echo'
        odometry = execution.recording.odometry.at(change.time_s)
        pose_text = 'no odometry'
        if odometry is not None:
            position = odometry.message.pose.pose.position
            pose_text = (
                f'pose ({position.x:.3f}, {position.y:.3f} m, '
                f'{get_heading(odometry.message):.3f} rad)'
            )
        return (
            f'  {change.time_s - origin_s:7.3f} s  {stage.name if stage else "?"}  '
            f'{describe_command(change.message)}  {echo_text}  {pose_text}'
        )

    def _format_activation_lines(self) -> list[str]:
        """Report the time from the activation in S1 to the next change, as information."""
        failure = self.verification.failure
        activations = failure.find_activations_during_first_stage()
        changes = failure.get_command_changes()
        if not activations or len(changes) < 2:
            return ['  falha: no fault activation while S1 was in force']
        delay_ms = (changes[1].time_s - activations[0].time_s) * 1e3
        return [f'  falha: activation in S1 -> next change in {delay_ms:.0f} ms (information)']


def main() -> None:
    if len(sys.argv) != 2:
        print('usage: check_v01 <run directory>')
        sys.exit(Verdict.INVALID.value)
    run_directory = Path(sys.argv[1])
    missing = find_missing_paths(run_directory)
    if missing:
        print('\n'.join(f'missing: {path}' for path in missing))
        print(f'V-01: {Verdict.INVALID.name}')
        sys.exit(Verdict.INVALID.value)
    verification = V01Verification(
        Execution.from_directory(run_directory / NOMINAL),
        Execution.from_directory(run_directory / FAILURE),
    )
    print(V01Report(verification))
    sys.exit(verification.verdict.value)


if __name__ == '__main__':
    main()
