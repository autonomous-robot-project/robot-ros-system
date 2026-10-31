"""
Charts of a V-01 run: what the sequencer commanded, what the controller echoed and did.

Criticality: T2, product (robot-docs codigo.md).

One image against time, with a vertical line at each change of the command: the behavior
commanded and echoed, the speeds and the heading of the Odometry (with the TURN_TO target), and
the masks of the State. One image of the path in odom, with each GO_TO target. It also prints
each change of the command, the echo and the masks, with the time from the first message. For
checking by eye (robot-docs nota 0016): the verdict of V-01 comes from its check_v01.

Usage: plot_v01 <bag directory> <output prefix>.
Writes <prefix>_tempo.png and <prefix>_trajetoria.png.
"""

from __future__ import annotations

from collections.abc import Callable
import math
from pathlib import Path
import sys
from typing import TYPE_CHECKING

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from robot_verification.plot_recording import MUTED, SERIES
from robot_verification.recording import (
    BEHAVIOR_NAMES,
    GO_TO,
    TURN_TO,
    TWIST,
    ControllerRecording,
    Stream,
    describe_command,
    find_changes,
    get_heading,
)

if TYPE_CHECKING:
    from nav_msgs.msg import Odometry

SECOND_SERIES = '#eb6834'  # slot 2 of the categorical palette, drawn dashed
MIN_LABEL_GAP_S = 0.3  # s, closer changes of the command get a line but no label


def find_origin_s(recording: ControllerRecording) -> float:
    streams = (recording.commands, recording.states, recording.odometry)
    return min(stream[0].time_s for stream in streams if stream)


def print_changes(recording: ControllerRecording, origin_s: float) -> None:
    lines = [
        (sample.time_s, f'comando {describe_command(sample.message)}')
        for sample in find_changes(recording.commands, describe_command)
    ]
    lines += [
        (sample.time_s, f'eco     {describe_command(sample.message.echo)}')
        for sample in find_changes(recording.states, lambda s: describe_command(s.echo))
    ]
    lines += [
        (
            sample.time_s,
            f'códigos faults={sample.message.faults} reflexes={sample.message.reflexes}',
        )
        for sample in find_changes(recording.states, lambda s: (s.faults, s.reflexes))
    ]
    for time_s, text in sorted(lines):
        print(f'{time_s - origin_s:8.3f} s  {text}')


def plot_timeline(recording: ControllerRecording, origin_s: float, run: str, path: str) -> None:
    figure, axes = plt.subplots(5, 1, figsize=(7, 9), sharex=True)
    figure.suptitle(run, x=0.02, ha='left', fontsize=9)
    behavior, linear, angular, heading, masks = axes
    plot_behavior(behavior, recording, origin_s)
    plot_linear_speed(linear, recording, origin_s)
    plot_odometry(angular, recording.odometry, origin_s, lambda o: o.twist.twist.angular.z)
    angular.set_title('ω da Odometria', loc='left')
    angular.set_ylabel('rad/s')
    plot_heading(heading, recording, origin_s)
    plot_masks(masks, recording, origin_s)
    mark_command_changes(axes, recording, origin_s)
    masks.set_xlabel('tempo desde a primeira mensagem (s)')
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_behavior(axis: Axes, recording: ControllerRecording, origin_s: float) -> None:
    axis.step(
        [s.time_s - origin_s for s in recording.commands],
        [s.message.behavior for s in recording.commands],
        where='post',
        color=SERIES,
        linewidth=1.5,
        label='comandado',
    )
    axis.step(
        [s.time_s - origin_s for s in recording.states],
        [s.message.echo.behavior for s in recording.states],
        where='post',
        color=SECOND_SERIES,
        linewidth=1.5,
        linestyle='--',
        label='eco no Estado',
    )
    axis.set_yticks(range(len(BEHAVIOR_NAMES)), BEHAVIOR_NAMES)
    axis.set_title('Comportamento', loc='left')
    axis.legend(loc='upper right', frameon=False)


def plot_odometry(
    axis: Axes,
    odometry: Stream[Odometry],
    origin_s: float,
    read: Callable[[Odometry], float],
    label: str | None = None,
) -> None:
    axis.plot(
        [s.time_s - origin_s for s in odometry],
        [read(s.message) for s in odometry],
        color=SERIES,
        linewidth=1.0,
        label=label,
    )


def plot_linear_speed(axis: Axes, recording: ControllerRecording, origin_s: float) -> None:
    plot_odometry(
        axis, recording.odometry, origin_s, lambda o: o.twist.twist.linear.x, 'da Odometria'
    )
    # The commanded v exists only in a TWIST: the line breaks during TURN_TO and GO_TO.
    axis.step(
        [s.time_s - origin_s for s in recording.commands],
        [
            s.message.linear if s.message.behavior == TWIST else math.nan
            for s in recording.commands
        ],
        where='post',
        color=SECOND_SERIES,
        linewidth=1.5,
        linestyle='--',
        label='comandado (TWIST)',
    )
    axis.set_title('v', loc='left')
    axis.set_ylabel('m/s')
    axis.legend(loc='upper right', frameon=False)


def plot_heading(axis: Axes, recording: ControllerRecording, origin_s: float) -> None:
    plot_odometry(axis, recording.odometry, origin_s, get_heading, 'da Odometria')
    axis.set_title('Rumo θ', loc='left')
    axis.set_ylabel('rad')
    turns = [s for s in recording.commands if s.message.behavior == TURN_TO]
    if not turns:
        return  # one series: the title names it
    axis.plot(
        [s.time_s - origin_s for s in turns],
        [s.message.theta for s in turns],
        color=SECOND_SERIES,
        linewidth=1.5,
        linestyle='--',
        label='alvo do TURN_TO',
    )
    axis.legend(loc='lower right', frameon=False)


def plot_masks(axis: Axes, recording: ControllerRecording, origin_s: float) -> None:
    times = [s.time_s - origin_s for s in recording.states]
    axis.step(
        times,
        [s.message.faults for s in recording.states],
        where='post',
        color=SERIES,
        linewidth=1.5,
        label='falhas',
    )
    axis.step(
        times,
        [s.message.reflexes for s in recording.states],
        where='post',
        color=SECOND_SERIES,
        linewidth=1.5,
        linestyle='--',
        label='reflexos',
    )
    axis.set_title('Estado: máscaras', loc='left')
    axis.set_ylabel('máscara')
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_ylim(bottom=-0.5, top=max(1.0, axis.get_ylim()[1]))
    axis.legend(loc='upper right', frameon=False)


def mark_command_changes(axes: Axes, recording: ControllerRecording, origin_s: float) -> None:
    changes = find_changes(recording.commands, describe_command)
    for axis in axes:
        for sample in changes:
            axis.axvline(sample.time_s - origin_s, color=MUTED, linewidth=0.6, alpha=0.6)
    top = axes[0]
    labeled_s = -math.inf
    for sample in changes:
        if sample.time_s - labeled_s < MIN_LABEL_GAP_S:
            continue  # the line stays; the label would overlap the one before
        labeled_s = sample.time_s
        top.annotate(
            describe_command(sample.message),
            (sample.time_s - origin_s, 1.0),
            xycoords=('data', 'axes fraction'),
            rotation=90,
            va='top',
            ha='right',
            fontsize=6,
            color=MUTED,
        )


def plot_path(recording: ControllerRecording, run: str, path: str) -> None:
    figure, axis = plt.subplots(figsize=(5, 5))
    figure.suptitle(run, x=0.02, ha='left', fontsize=9)
    odometry = recording.odometry
    axis.plot(
        [s.message.pose.pose.position.x for s in odometry],
        [s.message.pose.pose.position.y for s in odometry],
        color=SERIES,
        linewidth=1.5,
        label='posição da Odometria',
    )
    goals = find_changes(
        Stream(s for s in recording.commands if s.message.behavior == GO_TO),
        lambda c: (c.x, c.y),
    )
    axis.plot(
        [s.message.x for s in goals],
        [s.message.y for s in goals],
        linestyle='none',
        marker='x',
        markersize=9,
        color=SECOND_SERIES,
        label='alvo do GO_TO',
    )
    axis.set_aspect('equal', adjustable='datalim')
    axis.set_title('Trajetória em odom (rodas suspensas)', loc='left')
    axis.set_xlabel('x (m)')
    axis.set_ylabel('y (m)')
    axis.legend(loc='upper left', frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    if len(sys.argv) != 3:
        print('usage: plot_v01 <bag directory> <output prefix>')
        sys.exit(2)
    bag, prefix = sys.argv[1], sys.argv[2]
    recording = ControllerRecording.from_bag(bag)
    run = f'Execução: {Path(bag).name}'
    origin_s = find_origin_s(recording)
    print_changes(recording, origin_s)
    plot_timeline(recording, origin_s, run, f'{prefix}_tempo.png')
    plot_path(recording, run, f'{prefix}_trajetoria.png')


if __name__ == '__main__':
    main()
