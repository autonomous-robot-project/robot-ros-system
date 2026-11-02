"""
Node of the sequencer: translates the topics of the controller into calls to the Sequencer.

Criticality: T2, product (robot-docs codigo.md).

One node, one thread (ADRs 031 and 033): the State (100 Hz) and the Odometry (20 Hz) of the
controller and the heartbeat of the notebook go to the Sequencer, a command it returns is
published in the same callback, and a timer checks the ages of the State and of the heartbeat
(ADR 037) and republishes the command in force at f_cmd so it does not expire (RNF-02). The times
come from the steady clock, so an adjustment of the system clock does not age a message. With no
mission running the command is TWIST(0, 0). The mission comes from the parameter `mission`
(arquitetura.md). The node holds no logic of its own: the domain is in sequencer.py and
mission.py (ADR 047).

With `mission` empty, the mission by plan (arquitetura.md): a goal on /goal_pose, in map, is sent
to the planner_server as ComputePathToPose, asynchronously, so nothing blocks the executor; only
the answer to the latest goal counts, and a new goal cancels the pending one (ADR 034). The
vertices of the Path become the mission. The map -> odom comes from a tf2 buffer in this executor,
looked up with timeout zero at the entry of each step (ADR 035).

With `mission:='v08'`, the load of C1 of V-08 (F8 D2): plans from p₀ to p_g of V-04, asked
asynchronously again at each answer, so the planner is always busy while the sequencer reacts to
the events; each plan is discarded, and the planner_server publishes it on `plan`. After an error
or with no planner_server (C2), the next request waits for the timer, so the sequencer at FIFO 50
never loops on failures; the errors are counted, not a reason to abort.

Usage: chrt -f 50 <prefix>/lib/robot_sequencer/sequencer --ros-args -p "mission:='v01'"
(quoted, so an empty mission is read as text),
(the executable itself, not `ros2 run`, which does not forward SIGTERM; priority of ADR 039).
"""

from collections.abc import Callable
import math

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point as PointMessage
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from robot_interfaces.msg import Command as CommandMessage
from robot_interfaces.msg import State
from std_msgs.msg import Empty
from tf2_ros import Buffer, TransformException, TransformListener

from robot_sequencer.geometry import Point, Pose
from robot_sequencer.mission import Behavior, Code, Command, OdometryReading, StateReading
from robot_sequencer.missions import MISSIONS, build_plan_mission
from robot_sequencer.plan import extract_vertices
from robot_sequencer.sequencer import Sequencer

COMMAND_TOPIC = '/controller/command'
STATE_TOPIC = '/controller/state'
ODOMETRY_TOPIC = '/controller/odometry'
HEARTBEAT_TOPIC = '/heartbeat'  # std_msgs/Empty at 10 Hz from the notebook (arquitetura.md)
GOAL_TOPIC = '/goal_pose'  # geometry_msgs/PoseStamped in map, from RViz or run_v04
PLANNER_ACTION = 'compute_path_to_pose'  # of the planner_server of Nav2
MAP_FRAME = 'map'
ODOM_FRAME = 'odom'
MISSION_BY_PLAN = ''  # the empty `mission` waits for /goal_pose
GOAL_QUEUE_DEPTH = 10  # the default QoS of RViz and of run_v04: reliable
MISSION_WITH_PLANNING_LOAD = 'v08'  # C1 of V-08 (F8 D2)
LOAD_START = (0.0, 0.0)  # m in map, p₀ of V-04
LOAD_GOAL = (2.0, 2.0)  # m in map, p_g of V-04
LOAD_LOG_PERIOD_S: float = 10.0

# QoS of every topic of the link (ADR 025), the same as robot_verification/link.py.
LINK_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

COMMAND_PERIOD_S: float = 0.05  # 1 / f_cmd, f_cmd = 20 Hz (ADR 033)
NS_PER_S: float = 1e9

FAULT_CODES: dict[int, Code] = {
    State.FAULT_COGNIZANT: Code.FAULT_COGNIZANT,
    State.FAULT_DEADLINE_MISSED: Code.FAULT_DEADLINE_MISSED,
    State.FAULT_WATCHDOG_RESTART: Code.FAULT_WATCHDOG_RESTART,
    State.FAULT_FRONT_SENSOR: Code.FAULT_FRONT_SENSOR,
    State.FAULT_TEST: Code.FAULT_TEST,
}
REFLEX_CODES: dict[int, Code] = {
    State.REFLEX_FRONT_OBSTACLE: Code.REFLEX_FRONT_OBSTACLE,
}
MASK_BITS = 32  # uint32 masks of State.msg


def get_active_codes(state: State) -> frozenset[Code]:
    return get_codes_of_mask(state.faults, FAULT_CODES) | get_codes_of_mask(
        state.reflexes, REFLEX_CODES
    )


def get_codes_of_mask(mask: int, codes: dict[int, Code]) -> frozenset[Code]:
    """Return the code of each set bit; a bit not in `codes` is Code.UNKNOWN."""
    bits = (1 << position for position in range(MASK_BITS))
    return frozenset(codes.get(bit, Code.UNKNOWN) for bit in bits if mask & bit)


def get_command(message: CommandMessage) -> Command:
    return Command(
        Behavior(message.behavior),
        linear=message.linear,
        angular=message.angular,
        theta=message.theta,
        x=message.x,
        y=message.y,
    )


def get_state_reading(state: State) -> StateReading:
    return StateReading(get_active_codes(state), echo=get_command(state.echo))


def get_heading(orientation: Quaternion) -> float:
    # Planar pose: the quaternion is a rotation about z only.
    return 2.0 * math.atan2(orientation.z, orientation.w)


def get_point(position: PointMessage) -> Point:
    return Point(position.x, position.y)


def get_odometry_reading(odometry: Odometry) -> OdometryReading:
    pose = odometry.pose.pose
    return OdometryReading(
        Pose(get_point(pose.position), get_heading(pose.orientation)),
        linear=odometry.twist.twist.linear.x,
        angular=odometry.twist.twist.angular.z,
    )


def build_command_message(command: Command) -> CommandMessage:
    return CommandMessage(
        behavior=command.behavior.value,
        linear=command.linear,
        angular=command.angular,
        theta=command.theta,
        x=command.x,
        y=command.y,
    )


def build_map_pose(x: float, y: float) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = MAP_FRAME
    pose.pose.position.x = x
    pose.pose.position.y = y
    return pose


def get_vertices(path: Path, goal: PoseStamped) -> list[Point]:
    points = [get_point(stamped.pose.position) for stamped in path.poses]
    return extract_vertices(points, get_point(goal.pose.position))


class PlanClient:
    """
    Asks the planner_server for a plan, asynchronously (ADRs 014 and 033).

    Only the answer to the latest goal counts: an older one is discarded by the number of its
    request, and a new goal cancels the pending one (ADR 034).
    """

    def __init__(
        self,
        node: Node,
        on_path: Callable[[Path, PoseStamped], None],
        on_failure: Callable[[str], None],
    ):
        self._client = ActionClient(node, ComputePathToPose, PLANNER_ACTION)
        self._on_path = on_path
        self._on_failure = on_failure
        self._request = 0  # number of the latest goal
        self._pending: ClientGoalHandle | None = None

    def request(self, goal: PoseStamped, start: PoseStamped | None = None) -> None:
        """Ask for a plan to the goal, from the start if given, else from the pose of the robot."""
        self._cancel_pending()
        self._request += 1
        if goal.header.frame_id != MAP_FRAME:
            self._on_failure(f'goal in {goal.header.frame_id!r}, not in {MAP_FRAME!r}')
        elif not self._client.server_is_ready():
            self._on_failure('no planner_server')
        else:
            request = self._request
            action_goal = ComputePathToPose.Goal(goal=goal)
            if start is not None:
                action_goal.start, action_goal.use_start = start, True
            future = self._client.send_goal_async(action_goal)
            future.add_done_callback(lambda done: self._on_accepted(done.result(), request, goal))

    def _on_accepted(self, handle: ClientGoalHandle, request: int, goal: PoseStamped) -> None:
        if request != self._request:
            handle.cancel_goal_async()  # a newer goal came while this one was sent
        elif not handle.accepted:
            self._on_failure('goal rejected by the planner_server')
        else:
            self._pending = handle
            future = handle.get_result_async()
            future.add_done_callback(lambda done: self._on_result(done.result(), request, goal))

    def _on_result(
        self,
        response: ComputePathToPose.Impl.GetResultService.Response,
        request: int,
        goal: PoseStamped,
    ) -> None:
        if request != self._request:
            return  # the answer to an older goal (ADR 034)
        self._pending = None
        result = response.result
        if response.status != GoalStatus.STATUS_SUCCEEDED or not result.path.poses:
            self._on_failure(f'status {response.status}, error {result.error_code}')
        else:
            self._on_path(result.path, goal)

    def _cancel_pending(self) -> None:
        if self._pending is not None:
            self._pending.cancel_goal_async()
            self._pending = None


class PlanningLoad:
    """The planner always busy from p₀ to p_g of V-04, each plan discarded (C1 of V-08, F8 D2)."""

    def __init__(self, node: Node):
        self._logger = node.get_logger()
        self._plans = PlanClient(node, self._on_path, self._on_failure)
        self._start = build_map_pose(*LOAD_START)
        self._goal = build_map_pose(*LOAD_GOAL)
        self._is_waiting = False  # for the answer to a request
        self._paths = 0
        self._failures = 0

    def keep_busy(self) -> None:
        """Ask again after an error or with no planner_server; called by the timer."""
        if not self._is_waiting:
            self._request()

    def _request(self) -> None:
        self._is_waiting = True
        self._plans.request(self._goal, self._start)

    def _on_path(self, path: Path, goal: PoseStamped) -> None:
        self._paths += 1
        self._log_count()
        self._request()

    def _on_failure(self, reason: str) -> None:
        self._failures += 1
        self._is_waiting = False
        self._log_count(reason)

    def _log_count(self, last_failure: str = '') -> None:
        self._logger.info(
            f'planning load: {self._paths} plans, {self._failures} failures {last_failure}',
            throttle_duration_sec=LOAD_LOG_PERIOD_S,
        )


class SequencerNode(Node):
    def __init__(self):
        super().__init__('sequencer')
        mission_name = self.declare_parameter('mission', MISSION_BY_PLAN).value
        if mission_name != MISSION_BY_PLAN and mission_name not in MISSIONS:
            known = [MISSION_BY_PLAN, *sorted(MISSIONS)]
            raise ValueError(f'mission {mission_name!r} unknown; use one of {known}')
        mission = MISSIONS[mission_name]() if mission_name in MISSIONS else None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)  # no thread (nota 0004)
        self._sequencer = Sequencer(mission, self._get_map_to_odom)
        self._commands = self.create_publisher(CommandMessage, COMMAND_TOPIC, LINK_QOS)
        self.create_subscription(State, STATE_TOPIC, self._on_state, LINK_QOS)
        self.create_subscription(Odometry, ODOMETRY_TOPIC, self._on_odometry, LINK_QOS)
        self.create_subscription(Empty, HEARTBEAT_TOPIC, self._on_heartbeat, LINK_QOS)
        self._steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(COMMAND_PERIOD_S, self._on_timer)
        self._load = PlanningLoad(self) if mission_name == MISSION_WITH_PLANNING_LOAD else None
        if mission is None:
            self._plans = PlanClient(self, self._on_path, self._on_plan_failure)
            self.create_subscription(PoseStamped, GOAL_TOPIC, self._on_goal, GOAL_QUEUE_DEPTH)
            self.get_logger().info(f'mission by plan: waiting for {GOAL_TOPIC}')
        else:
            self.get_logger().info(f'mission {mission_name}: waiting for the first Odometry')

    def _on_state(self, state: State) -> None:
        self._publish_if_new(self._sequencer.on_state(get_state_reading(state), self._get_now_s()))

    def _on_odometry(self, odometry: Odometry) -> None:
        self._publish_if_new(self._sequencer.on_odometry(get_odometry_reading(odometry)))

    def _on_goal(self, goal: PoseStamped) -> None:
        position = goal.pose.position
        self.get_logger().info(f'goal ({position.x:.3f}, {position.y:.3f}): planning')
        self._publish_if_new(self._sequencer.wait_for_plan())
        self._plans.request(goal)

    def _on_path(self, path: Path, goal: PoseStamped) -> None:
        vertices = get_vertices(path, goal)
        self.get_logger().info(f'plan of {len(path.poses)} poses, vertices {vertices}')
        self._publish_if_new(self._sequencer.start(build_plan_mission(vertices)))

    def _on_plan_failure(self, reason: str) -> None:
        self.get_logger().warning(f'PLAN_FAILED: {reason}')
        self._publish_if_new(self._sequencer.on_plan_failed())

    def _get_map_to_odom(self) -> Pose | None:
        """Return the latest pose of odom in map, looked up with timeout zero (ADR 035)."""
        try:
            stamped = self._tf_buffer.lookup_transform(
                MAP_FRAME, ODOM_FRAME, Time(), timeout=Duration()
            )
        except TransformException as error:
            self.get_logger().warning(f'TF_MISSING: {error}')
            return None
        transform = stamped.transform
        return Pose(get_point(transform.translation), get_heading(transform.rotation))

    def _on_heartbeat(self, _: Empty) -> None:
        self._sequencer.on_heartbeat(self._get_now_s())

    def _on_timer(self) -> None:
        """Publish the command in force, the new one if an age activated a code; keep the load."""
        self._log_if_new(self._sequencer.on_timer(self._get_now_s()))
        self._commands.publish(build_command_message(self._sequencer.command))
        if self._load is not None:
            self._load.keep_busy()

    def _publish_if_new(self, command: Command | None) -> None:
        if command is None:
            return
        self._commands.publish(build_command_message(command))
        self._log_if_new(command)

    def _log_if_new(self, command: Command | None) -> None:
        if command is not None:
            self.get_logger().info(f'{self._sequencer.status.name}: {command}')

    def _get_now_s(self) -> float:
        return self._steady_clock.now().nanoseconds / NS_PER_S


def main() -> None:
    rclpy.init()
    try:
        rclpy.spin(SequencerNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
