"""
Engine of the sequencer: runs a mission step by step and reacts to the codes of the controller.

Criticality: T2, product (robot-docs codigo.md).

The node calls on_odometry (20 Hz) and on_state (100 Hz) in one thread, publishes right away the
command a call returns, and republishes `command` on its timer (ADR 033).
- A step begins with the pose of the latest Odometry: its behavior is resolved to a command, and
  the codes seen before are forgotten, so a code already active counts as activated (ADR 037).
- The termination of the step is tested on each Odometry (ADR 033). When it is met, the next step
  begins; after the last one the mission succeeds with TWIST(0, 0).
- A code activated in the State aborts the mission with TWIST(0, 0): abort is the default
  reaction (ADR 032), and the missions of V-01 declare no other.
The mission begins on the first Odometry, since a step needs the pose; a State before it is
ignored. Domain without ROS (ADR 047).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from robot_sequencer.geometry import Pose
from robot_sequencer.mission import STOP, Code, Command, Mission, OdometryReading, Step


class MissionStatus(Enum):
    WAITING = auto()  # no Odometry yet
    RUNNING = auto()
    SUCCEEDED = auto()
    ABORTED = auto()


@dataclass(frozen=True)
class RunningStep:
    index: int
    step: Step
    start: Pose
    command: Command

    def is_done(self, reading: OdometryReading) -> bool:
        return self.step.termination.is_met(self.start, self.command, reading)


class Sequencer:
    def __init__(self, mission: Mission):
        self._mission = mission
        self._status = MissionStatus.WAITING
        self._running: RunningStep | None = None
        self._seen_codes: frozenset[Code] = frozenset()

    @property
    def status(self) -> MissionStatus:
        return self._status

    @property
    def command(self) -> Command:
        """Return the command in force, for the timer that republishes it."""
        return self._running.command if self._running is not None else STOP

    def on_odometry(self, reading: OdometryReading) -> Command | None:
        """Return the new command if the reading begins the mission or ends the step."""
        if self._status is MissionStatus.WAITING:
            return self._begin_step(0, reading.pose)
        if self._running is not None and self._running.is_done(reading):
            return self._begin_step(self._running.index + 1, reading.pose)
        return None

    def on_state(self, active_codes: frozenset[Code]) -> Command | None:
        """Return TWIST(0, 0) if a code was activated since the previous State of the step."""
        if self._running is None:
            return None
        activated = active_codes - self._seen_codes
        self._seen_codes = active_codes
        if activated:
            return self._finish(MissionStatus.ABORTED)
        return None

    def _begin_step(self, index: int, pose: Pose) -> Command:
        if index == len(self._mission.steps):
            return self._finish(MissionStatus.SUCCEEDED)
        step = self._mission.steps[index]
        self._running = RunningStep(index, step, pose, step.behavior.resolve(pose))
        self._seen_codes = frozenset()
        self._status = MissionStatus.RUNNING
        return self._running.command

    def _finish(self, status: MissionStatus) -> Command:
        self._running = None
        self._status = status
        return STOP
