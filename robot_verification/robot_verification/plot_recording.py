"""
Charts of every field recorded from the controller topics, one image per field.

Criticality: T2, product (robot-docs codigo.md).

Plots each field of the command, the State and the Instrumentation against the time from the
first message of the recording, so the recorded data can be checked by eye (robot-docs nota 0013).
The numbers of a verification come from its check_vNN, not from these charts.

Usage: plot_recording <bag directory> <output prefix>. Writes <prefix>_<field>.png for each field.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import sys
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from robot_verification.recording import ControllerRecording, Stream

if TYPE_CHECKING:
    from robot_interfaces.msg import Command, Instrumentation, State

SERIES, MUTED, GRID, SURFACE = '#2a78d6', '#52514e', '#e4e3df', '#fcfcfb'

plt.rcParams.update(
    {
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'axes.edgecolor': GRID,
        'axes.labelcolor': MUTED,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'axes.grid': True,
        'grid.color': GRID,
        'grid.linewidth': 0.8,
        'font.size': 8,
        'axes.spines.top': False,
        'axes.spines.right': False,
    }
)


@dataclass(frozen=True)
class Field[MessageT]:
    """A recorded field: the name of its image, its title, its unit and how to read it."""

    name: str
    title: str
    unit: str
    read: Callable[[MessageT], float]


COMMAND_FIELDS: tuple[Field[Command], ...] = (
    Field('comando_behavior', 'Comando: comportamento', 'código', lambda c: c.behavior),
    Field('comando_linear', 'Comando: v', 'm/s', lambda c: c.linear),
    Field('comando_angular', 'Comando: ω', 'rad/s', lambda c: c.angular),
    Field('comando_theta', 'Comando: θ (TURN_TO)', 'rad', lambda c: c.theta),
    Field('comando_x', 'Comando: x (GO_TO)', 'm', lambda c: c.x),
    Field('comando_y', 'Comando: y (GO_TO)', 'm', lambda c: c.y),
)

STATE_FIELDS: tuple[Field[State], ...] = (
    Field('estado_faults', 'Estado: máscara de falhas', 'máscara', lambda s: s.faults),
    Field('estado_reflexes', 'Estado: máscara de reflexos', 'máscara', lambda s: s.reflexes),
    Field(
        'estado_eco_behavior', 'Estado: eco do comportamento', 'código', lambda s: s.echo.behavior
    ),
    Field('estado_eco_linear', 'Estado: eco de v', 'm/s', lambda s: s.echo.linear),
    Field('estado_eco_angular', 'Estado: eco de ω', 'rad/s', lambda s: s.echo.angular),
    Field('estado_eco_theta', 'Estado: eco de θ', 'rad', lambda s: s.echo.theta),
    Field('estado_eco_x', 'Estado: eco de x', 'm', lambda s: s.echo.x),
    Field('estado_eco_y', 'Estado: eco de y', 'm', lambda s: s.echo.y),
)

INSTRUMENTATION_FIELDS: tuple[Field[Instrumentation], ...] = (
    Field(
        'instr_expiries',
        'Instrumentação: vencimentos desde o boot',
        'contagem',
        lambda i: i.expiries,
    ),
    Field(
        'instr_expiry_delay',
        'Instrumentação: atraso do último vencimento',
        'µs',
        lambda i: i.expiry_delay_us,
    ),
    Field(
        'instr_c_max',
        'Instrumentação: C_max desde o boot',
        'µs',
        lambda i: i.max_execution_time_us,
    ),
    Field(
        'instr_r_max',
        'Instrumentação: R_max desde o boot',
        'µs',
        lambda i: i.max_response_time_us,
    ),
    Field(
        'instr_deadline_misses',
        'Instrumentação: prazos perdidos desde o boot',
        'contagem',
        lambda i: i.deadline_misses,
    ),
    Field(
        'instr_left_wheel_speed',
        'Instrumentação: velocidade da roda esquerda',
        'm/s',
        lambda i: i.left_wheel_speed,
    ),
    Field(
        'instr_right_wheel_speed',
        'Instrumentação: velocidade da roda direita',
        'm/s',
        lambda i: i.right_wheel_speed,
    ),
)


def find_origin_s(recording: ControllerRecording) -> float:
    """Return the time of the first message of the recording, on any topic."""
    streams = (recording.commands, recording.states, recording.instrumentation)
    return min(stream[0].time_s for stream in streams if stream)


def plot_field[MessageT](
    stream: Stream[MessageT], field: Field[MessageT], origin_s: float, prefix: str
) -> None:
    figure, axis = plt.subplots(figsize=(7, 2.6))
    axis.plot(
        [sample.time_s - origin_s for sample in stream],
        [field.read(sample.message) for sample in stream],
        color=SERIES,
        linewidth=1.0,
    )
    axis.set_title(f'{field.title} ({len(stream)} mensagens)', loc='left')
    axis.set_xlabel('tempo desde a primeira mensagem (s)')
    axis.set_ylabel(field.unit)
    figure.tight_layout()
    figure.savefig(f'{prefix}_{field.name}.png', dpi=150)
    plt.close(figure)


def plot_recording(recording: ControllerRecording, prefix: str) -> None:
    origin_s = find_origin_s(recording)
    for field in COMMAND_FIELDS:
        plot_field(recording.commands, field, origin_s, prefix)
    for field in STATE_FIELDS:
        plot_field(recording.states, field, origin_s, prefix)
    for field in INSTRUMENTATION_FIELDS:
        plot_field(recording.instrumentation, field, origin_s, prefix)


def main() -> None:
    if len(sys.argv) != 3:
        print('usage: plot_recording <bag directory> <output prefix>')
        sys.exit(2)
    plot_recording(ControllerRecording.from_bag(sys.argv[1]), sys.argv[2])


if __name__ == '__main__':
    main()
