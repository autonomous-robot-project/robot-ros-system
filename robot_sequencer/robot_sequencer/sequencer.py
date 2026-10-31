"""
Engine of the sequencer: runs a mission step by step and reacts to the codes of the controller.

Criticality: T2, product (robot-docs codigo.md).

The node calls on_odometry (20 Hz) and on_state (100 Hz) in one thread, publishes right away the
command a call returns, and republishes `command` on its timer (ADR 033).
- A step begins with the pose of the latest Odometry: its behavior is resolved to a command, and
  the codes seen before are forgotten, so a code already active counts as activated (ADR 037).
- The termination of the step is tested where its input changes: a pose on each Odometry, a code
  on each State (ADR 033). When it is met, the next step begins; after the last one the mission
  succeeds with TWIST(0, 0).
- A code activated in the State gets the reaction of the step, else of the mission, else abort
  (ADR 032). Abort ends the mission with TWIST(0, 0); recover runs its steps and then enters the
  current step again, with a new pose; ignore leaves the step running. With several codes
  activated at once, abort comes first, then recover.
The mission begins on the first Odometry, since a step needs the pose; a State before it is
ignored. Domain without ROS (ADR 047).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from robot_sequencer.geometry import Pose
from robot_sequencer.mission import (
    STOP,
    Abort,
    Cleared,
    Code,
    Command,
    Mission,
    OdometryReading,
    Recover,
    Step,
)


class MissionStatus(Enum):
    WAITING = auto()  # no Odometry yet
    RUNNING = auto()
    SUCCEEDED = auto()
    ABORTED = auto()


@dataclass(frozen=True)
class RunningStep:
    index: int  # of the mission step; a recovery step keeps the index of the step it returns to
    step: Step
    start: Pose
    command: Command
    is_recovery: bool

    def is_done_by_reading(self, reading: OdometryReading) -> bool:
        termination = self.step.termination
        if isinstance(termination, Cleared):
            return False
        return termination.is_met(self.start, self.command, reading)

    def is_done_by_codes(self, active_codes: frozenset[Code]) -> bool:
        termination = self.step.termination
        return isinstance(termination, Cleared) and termination.code not in active_codes


class Sequencer:
    def __init__(self, mission: Mission):
        self._mission = mission
        self._status = MissionStatus.WAITING
        self._running: RunningStep | None = None
        self._recovery: list[Step] = []  # steps of a recover still to run
        self._seen_codes: frozenset[Code] = frozenset()
        self._pose: Pose | None = None

    @property
    def status(self) -> MissionStatus:
        return self._status

    @property
    def command(self) -> Command:
        """Return the command in force, for the timer that republishes it."""
        return self._running.command if self._running is not None else STOP

    def on_odometry(self, reading: OdometryReading) -> Command | None:
        """Return the new command if the reading begins the mission or ends the step."""
        self._pose = reading.pose
        if self._status is MissionStatus.WAITING:
            return self._enter_step(0)
        if self._running is not None and self._running.is_done_by_reading(reading):
            return self._end_step(self._running)
        return None

    def on_state(self, active_codes: frozenset[Code]) -> Command | None:
        """Return the new command if a code was activated or cleared and the step changes."""
        if self._running is None:
            return None
        activated = active_codes - self._seen_codes
        self._seen_codes = active_codes
        reaction = self._react_to(self._running, activated)
        if reaction is not None:
            return reaction
        if self._running.is_done_by_codes(active_codes):
            return self._end_step(self._running)
        return None

    def _react_to(self, running: RunningStep, activated: frozenset[Code]) -> Command | None:
        reactions = [self._mission.get_reaction(running.step, code) for code in activated]
        if any(isinstance(reaction, Abort) for reaction in reactions):
            return self._finish(MissionStatus.ABORTED)
        recovers = [reaction for reaction in reactions if isinstance(reaction, Recover)]
        if recovers:
            self._recovery = list(recovers[0].steps)
            return self._enter_step(running.index)
        return None

    def _end_step(self, running: RunningStep) -> Command:
        """Enter what follows: after a recovery step, the step it returns to; else the next."""
        return self._enter_step(running.index if running.is_recovery else running.index + 1)

    def _enter_step(self, index: int) -> Command:
        """Enter the next recovery step if any is left, else the mission step at index."""
        assert self._pose is not None  # a step only begins after the first Odometry
        if self._recovery:
            step, is_recovery = self._recovery.pop(0), True
        elif index == len(self._mission.steps):
            return self._finish(MissionStatus.SUCCEEDED)
        else:
            step, is_recovery = self._mission.steps[index], False
        command = step.behavior.resolve(self._pose)
        self._running = RunningStep(index, step, self._pose, command, is_recovery)
        self._seen_codes = frozenset()
        self._status = MissionStatus.RUNNING
        return command

    def _finish(self, status: MissionStatus) -> Command:
        self._running = None
        self._recovery = []
        self._status = status
        return STOP
