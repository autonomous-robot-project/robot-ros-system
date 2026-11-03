"""
Engine of the sequencer: runs a mission step by step and reacts to the codes of the controller.

Criticality: T2, product (robot-docs codigo.md).

The node calls on_odometry (20 Hz), on_state (100 Hz), on_heartbeat and on_timer (20 Hz) in one
thread, publishes right away the command a call returns, and republishes `command` on its timer
(ADR 033).
- A step begins with the pose of the latest Odometry: its behavior is resolved to a command, and
  the codes seen before are forgotten, so a code already active counts as activated (ADR 037).
- The termination of the step is tested where its input changes: a pose on each Odometry, a code
  on each State (ADR 033). When it is met, the next step begins; after the last one the mission
  succeeds with TWIST(0, 0).
- A code activated in the State gets the reaction of the step, else of the mission, else abort
  (ADR 032). Abort ends the mission with TWIST(0, 0); recover runs its steps and then enters the
  current step again, with a new pose; ignore leaves the step running. With several codes
  activated at once, abort comes first, then recover.
- A TURN_TO above θ_max turns θ_max toward the target and, once there, enters the step again;
  a TURN_TO or GO_TO that stays at rest out of its tolerance for T_stall after moving raises
  NO_PROGRESS, which gets its reaction like a code (ADR 036).
- The sequencer watches the link and the notebook (ADR 037) with three conditions that join the
  codes of the State: COMMAND_NOT_ECHOED (the echo differs from the command for more than T_eco),
  LINK_LOST (no State for more than T_LINK, counted from the first State) and HEARTBEAT_LOST (no
  heartbeat for more than T_HB, or none yet). The State checks the echo; the timer checks the
  ages, since no callback runs when the messages stop. When the State comes back after LINK_LOST,
  the codes seen before are forgotten, as at the entry of a step.
A step whose target is a vertex of the plan has no target without map -> odom: the mission
aborts, the only reaction to TF_MISSING (ADR 035).
A mission of a verification begins on the first Odometry, since a step needs the pose; a State
before it is ignored. With no mission (the mission by plan), the sequencer is idle until a goal:
it holds TWIST(0, 0) while the plan is computed, then begins the mission of the plan from the
latest pose. A new goal preempts the mission. A failed plan has no mission to look a reaction up
in: PLAN_FAILED aborts, the default (ADR 034). Domain without ROS (ADR 047).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from robot_sequencer.geometry import Pose
from robot_sequencer.mission import (
    STOP,
    Abort,
    Cleared,
    Command,
    Condition,
    HeadingReached,
    MapToOdom,
    MissingTransformError,
    Mission,
    OdometryReading,
    PositionReached,
    Recover,
    StateReading,
    Step,
    get_no_map_to_odom,
    limit_turn,
)

# Thresholds of the monitors (ADR 037), times in s.
COMMAND_EXPIRY_S: float = 0.2  # T_cmd: the controller expires a command after it (RNF-02)
LINK_TIMEOUT_S: float = COMMAND_EXPIRY_S  # T_LINK = T_cmd: the controller has stopped by then
# T_cmd + 2 periods of republication; 34 times the worst latency command -> echo, 8.8 ms
# (robot-docs nota 0006).
ECHO_TIMEOUT_S: float = 0.3  # T_eco
HEARTBEAT_TIMEOUT_S: float = 0.5  # T_HB

# T_stall (ADR 036): rest out of tolerance that raises NO_PROGRESS, counted in Odometry readings.
# It covers the ramp from rest to v_min (0.28 s), T_vel (0.3 s) and the dead zone of the motors,
# where one edge and then T_vel at rest gave a false NO_PROGRESS (robot-docs nota 0018); the same
# 1 s as T_f of V-02.
STALL_TIME_S: float = 1.0
ODOMETRY_PERIOD_S: float = 0.05  # 20 Hz (ADR 025)
STALL_READINGS: int = round(STALL_TIME_S / ODOMETRY_PERIOD_S)


class MissionStatus(Enum):
    IDLE = auto()  # no mission: the mission by plan waits for a goal
    PLANNING = auto()  # a goal waits for its plan
    WAITING = auto()  # no Odometry yet
    RUNNING = auto()
    SUCCEEDED = auto()
    ABORTED = auto()


@dataclass
class RunningStep:
    index: int  # of the mission step; a recovery step keeps the index of the step it returns to
    step: Step
    start: Pose
    command: Command
    is_recovery: bool
    is_partial_turn: bool  # a TURN_TO cut to θ_max: the step is entered again at its end
    has_moved: bool = False  # in some Odometry of the step
    rest_readings: int = 0  # consecutive readings at rest since the last motion

    def note_motion(self, reading: OdometryReading) -> None:
        if reading.is_at_rest:
            self.rest_readings += 1
        else:
            self.has_moved, self.rest_readings = True, 0

    def has_stalled(self) -> bool:
        """Return whether the robot has just completed T_stall at rest, out of the tolerance."""
        has_tolerance = isinstance(self.step.termination, HeadingReached | PositionReached)
        return has_tolerance and self.has_moved and self.rest_readings == STALL_READINGS

    def is_done_by_reading(self, reading: OdometryReading) -> bool:
        termination = self.step.termination
        if isinstance(termination, Cleared):
            return False
        return termination.is_met(self.start, self.command, reading)

    def is_done_by_conditions(self, active_conditions: frozenset[Condition]) -> bool:
        termination = self.step.termination
        return isinstance(termination, Cleared) and termination.condition not in active_conditions


class Sequencer:
    def __init__(self, mission: Mission | None, get_map_to_odom: MapToOdom = get_no_map_to_odom):
        """
        Run the mission, or wait for one with None.

        get_map_to_odom returns the latest pose of odom in map, if any.
        """
        self._mission = mission if mission is not None else Mission([])
        self._get_map_to_odom = get_map_to_odom
        self._status = MissionStatus.IDLE if mission is None else MissionStatus.WAITING
        self._running: RunningStep | None = None
        self._recovery: list[Step] = []  # steps of a recover still to run
        self._seen_conditions: frozenset[Condition] = frozenset()
        self._pose: Pose | None = None
        self._controller_conditions: frozenset[Condition] = frozenset()  # of the latest State
        self._last_state_s: float | None = None
        self._last_heartbeat_s: float | None = None
        self._echo_differs_since_s: float | None = None
        self._reaction_conditions: frozenset[Condition] = frozenset()  # behind the latest reaction

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
        running = self._running
        if running is None:
            return None
        if running.is_done_by_reading(reading):
            return self._end_step(running)
        running.note_motion(reading)
        if running.has_stalled():
            return self._react_to(running, frozenset({Condition.NO_PROGRESS}))
        return None

    def on_state(self, state: StateReading, now: float) -> Command | None:
        """Return the new command if a code was activated or cleared and the step changes."""
        if self._is_link_lost(now):
            self._seen_conditions = frozenset()  # the link is back (ADR 037)
        self._last_state_s = now
        self._controller_conditions = state.conditions
        self._track_echo(state.echo, now)
        return self._update_conditions(now)

    def pop_reaction_conditions(self) -> frozenset[Condition]:
        """Return the codes behind the latest abort or recover, once, for the log (#51)."""
        conditions, self._reaction_conditions = self._reaction_conditions, frozenset()
        return conditions

    def wait_for_plan(self) -> Command:
        """Preempt the mission for a new goal: TWIST(0, 0) until its plan arrives."""
        return self._finish(MissionStatus.PLANNING)

    def start(self, mission: Mission) -> Command | None:
        """Begin the mission of a plan from the latest pose, else on the first Odometry."""
        self._mission = mission
        if self._pose is None:
            self._status = MissionStatus.WAITING
            return None
        return self._enter_step(0)

    def on_plan_failed(self) -> Command:
        return self._finish(MissionStatus.ABORTED)  # the only reaction to PLAN_FAILED

    def on_heartbeat(self, now: float) -> None:
        self._last_heartbeat_s = now

    def on_timer(self, now: float) -> Command | None:
        """Return the new command if the age of the State or of the heartbeat activates a code."""
        return self._update_conditions(now)

    def _update_conditions(self, now: float) -> Command | None:
        if self._running is None:
            return None
        active_conditions = self._controller_conditions | self._get_own_conditions(now)
        activated = active_conditions - self._seen_conditions
        self._seen_conditions = active_conditions
        reaction = self._react_to(self._running, activated)
        if reaction is not None:
            return reaction
        if self._running.is_done_by_conditions(active_conditions):
            return self._end_step(self._running)
        return None

    def _get_own_conditions(self, now: float) -> frozenset[Condition]:
        conditions: set[Condition] = set()
        if self._echo_differs_since_s is not None:
            if now - self._echo_differs_since_s > ECHO_TIMEOUT_S:
                conditions.add(Condition.COMMAND_NOT_ECHOED)
        if self._is_link_lost(now):
            conditions.add(Condition.LINK_LOST)
        if self._last_heartbeat_s is None or now - self._last_heartbeat_s > HEARTBEAT_TIMEOUT_S:
            conditions.add(Condition.HEARTBEAT_LOST)
        return frozenset(conditions)

    def _is_link_lost(self, now: float) -> bool:
        return self._last_state_s is not None and now - self._last_state_s > LINK_TIMEOUT_S

    def _track_echo(self, echo: Command, now: float) -> None:
        """Note when the echo began to differ from the command in force."""
        if self.command.is_echoed_by(echo):
            self._echo_differs_since_s = None
        elif self._echo_differs_since_s is None:
            self._echo_differs_since_s = now

    def _react_to(self, running: RunningStep, activated: frozenset[Condition]) -> Command | None:
        reactions = {
            condition: self._mission.get_reaction(running.step, condition)
            for condition in activated
        }
        aborts = frozenset(
            condition for condition, reaction in reactions.items() if isinstance(reaction, Abort)
        )
        if aborts:
            self._reaction_conditions = aborts
            return self._finish(MissionStatus.ABORTED)
        recovers = [reaction for reaction in reactions.values() if isinstance(reaction, Recover)]
        if recovers:
            self._reaction_conditions = frozenset(
                condition
                for condition, reaction in reactions.items()
                if isinstance(reaction, Recover)
            )
            self._recovery = list(recovers[0].steps)
            return self._enter_step(running.index)
        return None

    def _end_step(self, running: RunningStep) -> Command:
        """Enter the same step after a partial turn, else what follows a recovery or the step."""
        if running.is_partial_turn:
            return self._begin_step(running.step, running.index, running.is_recovery)
        return self._enter_step(running.index if running.is_recovery else running.index + 1)

    def _enter_step(self, index: int) -> Command:
        """Enter the next recovery step if any is left, else the mission step at index."""
        if self._recovery:
            step, is_recovery = self._recovery.pop(0), True
        elif index == len(self._mission.steps):
            return self._finish(MissionStatus.SUCCEEDED)
        else:
            step, is_recovery = self._mission.steps[index], False
        return self._begin_step(step, index, is_recovery)

    def _begin_step(self, step: Step, index: int, is_recovery: bool) -> Command:
        """Resolve the step from the latest pose and forget the codes seen before (ADR 037)."""
        assert self._pose is not None  # a step only begins after the first Odometry
        try:
            target_command = step.behavior.resolve(self._pose, self._get_map_to_odom)
        except MissingTransformError:
            return self._finish(MissionStatus.ABORTED)  # the only reaction to TF_MISSING
        command = limit_turn(target_command, self._pose)
        self._running = RunningStep(
            index,
            step,
            self._pose,
            command,
            is_recovery,
            is_partial_turn=command != target_command,
        )
        self._seen_conditions = frozenset()
        self._status = MissionStatus.RUNNING
        return command

    def _finish(self, status: MissionStatus) -> Command:
        self._running = None
        self._recovery = []
        self._status = status
        return STOP
