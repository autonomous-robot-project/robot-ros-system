"""
Figures of the final round of the verifications, for the monograph (robot-docs nota 0020).

Criticality: T2, product (robot-docs codigo.md).

One figure per verification, each showing what its criterion judges, read from the bags by the
same classes as the checks: the response time Δ of V-07 under C0, C1 and C2 of V-08 against
T_seq; the expiry delay of each interruption of V-06 against T_cmd; the wheel speeds and the
cognizant fault around the power cuts of V-02; the stages sent and echoed in both executions of
V-01, and in a second figure a zoom on the echo of each change against the State period; the
speed and the front reflex around the screens of V-03; the plan over M and the path of the
odometry in V-04. The numbers and the verdicts come from the checks; the figures only show them.

Usage: plot_final_round <evidencias directory>. Writes fig_*.pdf and fig_*.png into the
2026-11-02_final folders of each verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import TYPE_CHECKING

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from robot_verification.check_v01 import Execution
from robot_verification.check_v02 import FAULT_COGNIZANT, FAULT_WATCHDOG_RESTART, V_R, Recording
from robot_verification.check_v03 import REFLEX_FRONT_OBSTACLE, ScreenRun, get_linear_speed
from robot_verification.check_v04 import P_0, P_G, PlanRun
from robot_verification.check_v06 import MAX_EXPIRY_DELAY_S, T_CMD_S, V06
from robot_verification.check_v07 import T_SEQ_US, ResponseRun
from robot_verification.recording import (
    BEHAVIOR_NAMES,
    ControllerRecording,
    Sample,
)

if TYPE_CHECKING:
    from robot_interfaces.msg import Command

FINAL = '2026-11-02_final'

# Palette of the dataviz reference: categorical slots in fixed order, status colors only for
# states, recessive ink and grid (robot_verification/plot_recording.py uses the same).
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']
STATUS_CRITICAL, STATUS_WARNING = '#d03b3b', '#fab219'
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
SHADE = '#e9e8e4'  # windows of the procedure: power cut, screen placed
WALL = '#3d3d3a'  # occupied cells of M (V-04)

WIDTH_IN = 6.3  # text width of an A4 page with 2.5 cm margins
MAP_M = Path(__file__).resolve().parents[2] / 'robot_bringup' / 'maps'
FORBIDDEN_RADIUS_M = 0.331  # r + m + res of the costmap of the trials (ADR 042)


@dataclass
class Interval:
    start_s: float
    end_s: float


def format_decimal(value: float, _position: int | None = None) -> str:
    """Tick label with a decimal comma, as in the monograph."""
    text = f'{value:g}'
    return text.replace('.', ',')


def style_axes(axes: Axes) -> None:
    axes.set_facecolor(SURFACE)
    axes.grid(color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ('top', 'right'):
        axes.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        axes.spines[side].set_color(MUTED)
    axes.tick_params(colors=MUTED, labelcolor=INK)
    axes.xaxis.set_major_formatter(FuncFormatter(format_decimal))
    axes.yaxis.set_major_formatter(FuncFormatter(format_decimal))


def save(
    figure: Figure, folder: Path, name: str, layout_rect: tuple[float, ...] = (0, 0, 1, 1)
) -> None:
    figure.tight_layout(rect=layout_rect)
    for suffix in ('pdf', 'png'):
        figure.savefig(folder / f'{name}.{suffix}', dpi=200, facecolor='white')
    plt.close(figure)
    print(f'{folder / name}.pdf')


def find_intervals(samples: list[Sample], is_active) -> list[Interval]:
    """Return the intervals where is_active holds, between consecutive samples."""
    intervals: list[Interval] = []
    start: float | None = None
    for sample in samples:
        if is_active(sample) and start is None:
            start = sample.time_s
        elif not is_active(sample) and start is not None:
            intervals.append(Interval(start, sample.time_s))
            start = None
    if start is not None:
        intervals.append(Interval(start, samples[-1].time_s))
    return intervals


def shade_intervals(axes: Axes, intervals: list[Interval], origin_s: float, label: str) -> None:
    for index, interval in enumerate(intervals):
        axes.axvspan(
            interval.start_s - origin_s,
            interval.end_s - origin_s,
            color=SHADE,
            linewidth=0,
            label=label if index == 0 else None,
        )


def build_time_figure() -> tuple[Figure, Axes, Axes]:
    """Return a plot against time over a thin panel for the codes of the State."""
    figure, (axes, codes) = plt.subplots(
        2, 1, figsize=(WIDTH_IN, 3.4), sharex=True, gridspec_kw={'height_ratios': [5, 1.2]}
    )
    style_axes(axes)
    style_axes(codes)
    codes.grid(False)
    return figure, axes, codes


def draw_code_rows(
    codes: Axes, rows: list[tuple[str, list[Interval], str]], origin_s: float
) -> None:
    """One row per code of the State, filled where the code is active."""
    for row, (_, intervals, color) in enumerate(rows):
        spans = [
            (i.start_s - origin_s, i.end_s - origin_s - (i.start_s - origin_s)) for i in intervals
        ]
        codes.broken_barh(spans, (row - 0.35, 0.7), color=color, linewidth=0)
    for row, (label, _, _) in enumerate(rows):
        codes.text(
            0.005,
            row,
            label,
            transform=codes.get_yaxis_transform(),
            color=INK,
            fontsize=8,
            va='center',
        )
    codes.set_yticks([])
    codes.set_ylim(-0.6, len(rows) - 0.4)
    codes.spines['left'].set_visible(False)


def is_stop(command) -> bool:
    return command.behavior == 0 and command.linear == 0.0 and command.angular == 0.0


@dataclass
class Segment:
    """A stretch of time in which one command of the mission was in force, sent or echoed."""

    start_s: float
    end_s: float
    command: Command
    label: str


def get_stage_label(number: int, command: Command) -> str:
    return 'S4' if is_stop(command) else f'S{number}'  # S4 ends every mission (V-01)


def get_sent_segments(execution: Execution) -> list[Segment]:
    """Return each command of the mission from its change to the next one, labeled S1, S2..."""
    changes = execution.get_command_changes()
    ends = [change.time_s for change in changes[1:]] + [
        execution.recording.states[len(execution.recording.states) - 1].time_s
    ]
    return [
        Segment(change.time_s, end_s, change.message, get_stage_label(number, change.message))
        for number, (change, end_s) in enumerate(zip(changes, ends, strict=True), start=1)
    ]


def get_echoed_segments(execution: Execution) -> list[Segment]:
    """Return each command from the first State that echoes it to the next echo.

    The echo matches the behavior and every parameter, with the same test as check_v01.
    """
    sent = get_sent_segments(execution)
    echoes = [execution.find_echo(index) for index in range(len(sent))]
    starts = [echo.time_s for echo in echoes if echo is not None]
    ends = starts[1:] + [execution.recording.states[len(execution.recording.states) - 1].time_s]
    echoed = [segment for segment, echo in zip(sent, echoes, strict=True) if echo is not None]
    return [
        Segment(start_s, end_s, segment.command, segment.label)
        for start_s, end_s, segment in zip(starts, ends, echoed, strict=True)
    ]


def get_behavior_color(command: Command) -> str:
    return SHADE if is_stop(command) else SERIES[command.behavior]


def draw_segments(axes: Axes, segments: list[Segment], origin_s: float, row: float) -> None:
    """Draw each segment on the row, colored by its behavior and labeled with its stage."""
    for segment in segments:
        start = segment.start_s - origin_s
        axes.broken_barh(
            [(start, segment.end_s - segment.start_s)],
            (row - 0.3, 0.6),
            color=get_behavior_color(segment.command),
            linewidth=0,
        )
        axes.text(start + 0.1, row, segment.label, color=INK, va='center', fontsize=8)


def draw_fault_row(axes: Axes, execution: Execution, origin_s: float, row: float) -> None:
    states = list(execution.recording.states)
    faults = find_intervals(states, lambda s: bool(s.message.faults & FAULT_COGNIZANT))
    spans = [(f.start_s - origin_s, f.end_s - f.start_s) for f in faults]
    axes.broken_barh(spans, (row - 0.3, 0.6), color=STATUS_CRITICAL, linewidth=0)


def draw_full_scale(axes: Axes, execution: Execution, origin_s: float, title: str) -> None:
    """Draw the stages sent, the stages echoed in the State and the cognizant fault."""
    style_axes(axes)
    axes.grid(axis='y', visible=False)
    draw_segments(axes, get_sent_segments(execution), origin_s, row=2)
    draw_segments(axes, get_echoed_segments(execution), origin_s, row=1)
    draw_fault_row(axes, execution, origin_s, row=0)
    axes.set_yticks([0, 1, 2], ['FAULT_COGNIZANT', 'relatado no State', 'comando'], fontsize=8)
    axes.tick_params(axis='y', length=0)
    axes.set_ylim(-0.5, 2.5)
    axes.set_xlim(-0.2, 13)
    axes.set_xlabel('Tempo desde o início do S1 (s)', fontsize=8)
    axes.set_title(title, loc='left', fontsize=9, color=INK)


ECHO_WINDOW_MS = (-15.0, 25.0)  # around each change: about two State periods (100 Hz) each side


def draw_echo_zoom(
    axes: Axes, execution: Execution, index: int, segment: Segment, row: int
) -> None:
    """Draw, around one change, the State samples, the first echo and any fault activation."""
    start_s = segment.start_s + ECHO_WINDOW_MS[0] / 1e3
    end_s = segment.start_s + ECHO_WINDOW_MS[1] / 1e3
    states = execution.recording.states.between(start_s, end_s)
    sample_ms = [(state.time_s - segment.start_s) * 1e3 for state in states]
    axes.plot(
        sample_ms,
        [row] * len(sample_ms),
        '|',
        color=MUTED,
        markersize=8,
    )
    echo = execution.find_echo(index)
    if echo is not None:
        axes.plot((echo.time_s - segment.start_s) * 1e3, row, 'o', markersize=6, color=INK)
    activations = [a for a in execution.find_fault_activations() if start_s <= a.time_s <= end_s]
    for activation in activations:
        axes.plot(
            (activation.time_s - segment.start_s) * 1e3,
            row,
            'x',
            markersize=7,
            color=STATUS_CRITICAL,
        )


def draw_zoom(axes: Axes, execution: Execution) -> None:
    """Draw one row per change of the command, in milliseconds around the change."""
    style_axes(axes)
    segments = get_sent_segments(execution)
    for index, segment in enumerate(segments):
        draw_echo_zoom(axes, execution, index, segment, row=index)
    axes.axvline(0, color=INK, linewidth=1, linestyle='--')
    axes.set_yticks(range(len(segments)), [segment.label for segment in segments], fontsize=8)
    axes.set_ylim(len(segments) - 0.5, -0.5)
    axes.set_xlim(*ECHO_WINDOW_MS)
    axes.set_xlabel('Tempo desde a troca do comando (ms)', fontsize=8)
    axes.set_ylabel('Troca para', fontsize=8)


EXECUTIONS = (('nominal', 'Execução nominal'), ('falha', 'Execução com a roda travada no S1'))


def build_stage_legend_handles() -> list:
    """Return the behaviors and the fault span of the figure of the stages."""
    patch = plt.matplotlib.patches.Patch
    behaviors = zip([*SERIES, SHADE], [*BEHAVIOR_NAMES, 'TWIST(0, 0)'], strict=True)
    return [
        *(patch(color=color, label=label) for color, label in behaviors),
        patch(color=STATUS_CRITICAL, label='FAULT_COGNIZANT ativo'),
    ]


def build_zoom_legend_handles() -> list:
    """Return the markers of the figure of the echo zoom."""
    line = plt.matplotlib.lines.Line2D
    return [
        line([], [], color=INK, linewidth=1, linestyle='--', label='troca do comando'),
        line(
            [], [], marker='|', linestyle='', color=MUTED, markersize=8, label='amostra do State'
        ),
        line(
            [],
            [],
            marker='o',
            linestyle='',
            color=INK,
            markersize=6,
            label='primeiro State com o eco',
        ),
        line(
            [],
            [],
            marker='x',
            linestyle='',
            color=STATUS_CRITICAL,
            markersize=7,
            label='ativação do FAULT_COGNIZANT',
        ),
    ]


def plot_stages(folder: Path, executions: list[tuple[Execution, str]]) -> None:
    """V-01: the stages sent and echoed in both executions, at full scale."""
    figure, panels = plt.subplots(2, 1, figsize=(WIDTH_IN, 4.6), sharex=True)
    for axes, (execution, title) in zip(panels, executions, strict=True):
        draw_full_scale(axes, execution, get_sent_segments(execution)[0].start_s, title)
    panels[0].set_xlabel('')  # shared time axis, labeled below
    figure.legend(
        handles=build_stage_legend_handles(), loc='lower center', ncol=5, frameon=False, fontsize=7
    )
    save(figure, folder, 'fig_v01_selecao', layout_rect=(0, 0.06, 1, 1))  # legend below


def plot_echo_zooms(folder: Path, executions: list[tuple[Execution, str]]) -> None:
    """V-01: a zoom on the echo of each change of the command, in both executions."""
    rows = [len(execution.get_command_changes()) for execution, _ in executions]
    figure, panels = plt.subplots(
        2,
        1,
        figsize=(WIDTH_IN, 1.6 + 0.45 * sum(rows)),
        sharex=True,
        gridspec_kw={'height_ratios': rows},
    )
    for axes, (execution, title) in zip(panels, executions, strict=True):
        draw_zoom(axes, execution)
        axes.set_title(title, loc='left', fontsize=9, color=INK)
    panels[0].set_xlabel('')  # shared time axis, labeled below
    figure.legend(
        handles=build_zoom_legend_handles(), loc='lower center', ncol=2, frameon=False, fontsize=7
    )
    save(figure, folder, 'fig_v01_eco', layout_rect=(0, 0.1, 1, 1))  # legend below


def plot_behavior_selection(evidence: Path) -> None:
    """V-01: one figure of the stages and one of the echo zoom, each with both executions."""
    folder = evidence / 'V-01' / FINAL
    executions = [(Execution.from_directory(folder / name), title) for name, title in EXECUTIONS]
    plot_stages(folder, executions)
    plot_echo_zooms(folder, executions)


def plot_response_times(evidence: Path) -> None:
    """V-07 and V-08: cumulative distribution of Δ in C0, C1 and C2, against T_seq."""
    runs = {
        'C0 (V-07)': evidence / 'V-07' / FINAL / 'v07',
        'C1 (deliberador ativo)': evidence / 'V-08' / f'{FINAL}_C1' / 'v08_c1',
        'C2 (CPU saturada)': evidence / 'V-08' / f'{FINAL}_C2' / 'v08_c2',
    }
    figure, axes = plt.subplots(figsize=(WIDTH_IN, 3.0))
    style_axes(axes)
    for color, (label, bag) in zip(SERIES, runs.items(), strict=True):
        responses_ms = np.sort(np.array(ResponseRun.from_bag(str(bag)).get_responses_us())) / 1e3
        fraction = np.arange(1, len(responses_ms) + 1) / len(responses_ms)
        maximum = format_decimal(round(responses_ms[-1], 1))
        axes.step(
            responses_ms,
            fraction,
            where='post',
            color=color,
            linewidth=2,
            label=f'{label}: n = {len(responses_ms)}, máx. {maximum} ms',
        )
    t_seq_ms = T_SEQ_US / 1e3
    axes.axvline(t_seq_ms, color=STATUS_CRITICAL, linewidth=1.2)
    axes.text(t_seq_ms - 1, 0.05, r'$T_{seq}$ = 100 ms', color=INK, ha='right', fontsize=9)
    axes.set_xlim(0, t_seq_ms * 1.08)
    axes.set_ylim(0, 1.02)
    axes.set_xlabel('Δ: ativação do código no controlador → chegada do TWIST(0, 0) (ms)')
    axes.set_ylabel('Fração dos eventos')
    axes.legend(loc='center right', frameon=False, fontsize=8)
    save(figure, evidence / 'V-07' / FINAL, 'fig_v07_v08_resposta')


def plot_expiry_delays(evidence: Path) -> None:
    """V-06: delay of the expiry measured on the ESP32 in each interruption, against T_cmd."""
    folder = evidence / 'V-06' / FINAL
    verification = V06(ControllerRecording.from_bag(str(folder / 'v06')))
    delays_ms = [verification.expiry_delay_s(gap) * 1e3 for gap in verification.gaps]
    figure, axes = plt.subplots(figsize=(WIDTH_IN, 2.8))
    style_axes(axes)
    numbers = range(1, len(delays_ms) + 1)
    axes.plot(numbers, delays_ms, 'o', color=SERIES[0], markersize=6)
    references = (
        (T_CMD_S, r'$T_{cmd}$ = 200 ms', MUTED),
        (MAX_EXPIRY_DELAY_S, r'$T_{cmd} + T_c$ = 210 ms (limite)', STATUS_CRITICAL),
    )
    for value, label, color in references:
        axes.axhline(value * 1e3, color=color, linewidth=1.2)
        axes.text(
            len(delays_ms) + 0.5,
            value * 1e3 + 0.4,
            label,
            color=INK,
            va='bottom',
            ha='right',
            fontsize=9,
        )
    axes.set_xlim(0, len(delays_ms) + 1)
    axes.set_xticks([1, 5, 10, 15, 20, 25, 30])
    axes.set_ylim(T_CMD_S * 1e3 - 4, MAX_EXPIRY_DELAY_S * 1e3 + 3)
    axes.set_xlabel('Interrupção do publicador')
    axes.set_ylabel('Atraso do vencimento (ms)')
    save(figure, folder, 'fig_v06_vencimento')


def plot_power_cuts(evidence: Path) -> None:
    """V-02: wheel speeds and the cognizant fault around the three power cuts."""
    folder = evidence / 'V-02' / FINAL
    recording = Recording.from_bag(str(folder / 'v02'))
    samples = list(recording.instrumentation)
    origin_s = samples[0].time_s
    times = [sample.time_s - origin_s for sample in samples]
    figure, axes, codes = build_time_figure()
    cuts = [Interval(window.cut_s, window.restore_s) for window in recording.cut_windows()]
    shade_intervals(axes, cuts, origin_s, 'chave dos motores desligada')
    wheels = (('roda esquerda', 'left_wheel_speed'), ('roda direita', 'right_wheel_speed'))
    highest = 0.0
    for color, (label, field) in zip(SERIES, wheels, strict=False):
        speeds = [getattr(sample.message, field) for sample in samples]
        highest = max(highest, *speeds)
        axes.plot(times, speeds, color=color, linewidth=1.2, label=label)
    axes.axhline(V_R, color=MUTED, linewidth=1)
    axes.text(times[-1], V_R, ' $v_r$', color=INK, va='center', fontsize=9)
    injection_s = recording.injection_s()
    if injection_s is not None:
        for panel in (axes, codes):
            panel.axvline(injection_s - origin_s, color=MUTED, linewidth=1)
        axes.text(
            injection_s - origin_s, highest, ' GPIO 11 aterrado', color=INK, fontsize=8, va='top'
        )
    states = list(recording.states)
    draw_code_rows(
        codes,
        [
            (
                'FAULT_WATCHDOG_RESTART',
                find_intervals(states, lambda s: bool(s.message.faults & FAULT_WATCHDOG_RESTART)),
                STATUS_CRITICAL,
            ),
            (
                'FAULT_COGNIZANT',
                find_intervals(states, lambda s: bool(s.message.faults & FAULT_COGNIZANT)),
                STATUS_CRITICAL,
            ),
        ],
        origin_s,
    )
    axes.set_ylim(-0.02, highest * 1.08)
    axes.set_ylabel('Velocidade medida (m/s)')
    codes.set_xlabel('Tempo (s)')
    axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=False, fontsize=8)
    save(figure, folder, 'fig_v02_falha_cognizante')


def plot_screens(evidence: Path) -> None:
    """V-03: forward speed and the front reflex around the three screens."""
    folder = evidence / 'V-03' / FINAL
    run = ScreenRun.from_bag(str(folder / 'v03'))
    odometry = list(run.recording.odometry)
    origin_s = odometry[0].time_s
    figure, axes, codes = build_time_figure()
    screens = [Interval(window.placed_s, window.removed_s) for window in run.windows]
    shade_intervals(axes, screens, origin_s, 'anteparo, entre os anúncios de colocação e retirada')
    speeds = [get_linear_speed(sample) for sample in odometry]
    axes.plot(
        [sample.time_s - origin_s for sample in odometry],
        speeds,
        color=SERIES[0],
        linewidth=1.2,
        label='velocidade linear (Odometry)',
    )
    axes.axhline(V_R, color=MUTED, linewidth=1, label='$v_\\mathrm{r}$ = 0,2 m/s')
    states = list(run.recording.states)
    draw_code_rows(
        codes,
        [
            (
                'REFLEX_FRONT_OBSTACLE',
                find_intervals(states, lambda s: bool(s.message.reflexes & REFLEX_FRONT_OBSTACLE)),
                STATUS_WARNING,
            )
        ],
        origin_s,
    )
    axes.set_ylim(-0.02, max(speeds) * 1.08)
    axes.set_ylabel('Velocidade (m/s)')
    codes.set_xlabel('Tempo (s)')
    axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.32), ncol=2, frameon=False, fontsize=8)
    save(figure, folder, 'fig_v03_reflexo')


def read_map_m() -> tuple[np.ndarray, float, tuple[float, float]]:
    """Return the occupancy of M (True = occupied, row 0 at the top), its resolution and origin."""
    lines = (MAP_M / 'm.yaml').read_text().splitlines()
    fields = dict(line.split(':', 1) for line in lines if ':' in line)
    resolution = float(fields['resolution'])
    origin = [float(value) for value in fields['origin'].strip(' []').split(',')[:2]]
    image = plt.imread(MAP_M / 'm.pgm')
    return image < 0.5 * image.max(), resolution, (origin[0], origin[1])


def get_forbidden_zone(occupied: np.ndarray, resolution: float) -> np.ndarray:
    """Cells within r + m + res of an occupied cell: where the plan may not go (ADR 042)."""
    from scipy.ndimage import distance_transform_edt

    return distance_transform_edt(~occupied) * resolution <= FORBIDDEN_RADIUS_M


def plot_map(evidence: Path) -> None:
    """V-04: the map M given to the deliberator, with p_0 and p_g, for the procedure."""
    folder = evidence / 'V-04' / FINAL
    figure, axes = plt.subplots(figsize=(WIDTH_IN * 0.75, WIDTH_IN * 0.75))
    draw_map_m(axes)
    axes.plot(*zip(P_0, P_G, strict=True), 'o', color=INK, markersize=5)
    annotate_ends(axes)
    add_map_legend(axes)
    save(figure, folder, 'fig_v04_mapa')


def plot_plan(evidence: Path) -> None:
    """V-04: the plan over M, its vertices and the path of the odometry."""
    folder = evidence / 'V-04' / FINAL
    run = PlanRun.from_bag(str(folder / 'v04'))
    figure, axes = plt.subplots(figsize=(WIDTH_IN * 0.75, WIDTH_IN * 0.75))
    draw_map_m(axes)
    path = run.plans[0].message.poses
    axes.plot(
        [pose.pose.position.x for pose in path],
        [pose.pose.position.y for pose in path],
        color=MUTED,
        linewidth=1,
        label='plano do Theta* (Path)',
    )
    odometry = list(run.recording.odometry)
    axes.plot(
        [sample.message.pose.pose.position.x for sample in odometry],
        [sample.message.pose.pose.position.y for sample in odometry],
        color=SERIES[0],
        linewidth=2,
        label='trajetória da odometria',
    )
    vertices = run.expected_vertices
    axes.plot(
        [x for x, _ in vertices],
        [y for _, y in vertices],
        'o',
        color=SERIES[1],
        markersize=7,
        markeredgecolor=SURFACE,
        label='vértices (alvos dos GO_TO)',
    )
    annotate_ends(axes)
    add_map_legend(axes)
    save(figure, folder, 'fig_v04_plano')


def draw_map_m(axes: Axes) -> None:
    """Draw M: the wall and the inflation zone of the costmap, on axes in the map frame."""
    occupied, resolution, origin = read_map_m()
    rows, columns = occupied.shape
    style_axes(axes)
    axes.grid(False)
    zone = np.where(occupied, 2, np.where(get_forbidden_zone(occupied, resolution), 1, 0))
    colors = plt.matplotlib.colors.ListedColormap([SURFACE, SHADE, WALL])
    # One vector cell per map cell: an imshow goes into the PDF as a 2-bit indexed image, which
    # poppler draws skewed. Row 0 of the image is the top of M, so the rows are flipped.
    axes.pcolormesh(
        origin[0] + np.arange(columns + 1) * resolution,
        origin[1] + np.arange(rows + 1) * resolution,
        np.flipud(zone),
        cmap=colors,
        vmin=0,
        vmax=2,
        edgecolors='face',  # no seams between cells
    )
    axes.set_aspect('equal')
    axes.set_xlabel('x em map (m)')
    axes.set_ylabel('y em map (m)')


def annotate_ends(axes: Axes) -> None:
    axes.annotate('$p_0$', P_0, xytext=(-14, -12), textcoords='offset points', color=INK)
    axes.annotate('$p_g$', P_G, xytext=(6, 6), textcoords='offset points', color=INK)


def add_map_legend(axes: Axes) -> None:
    """Add the legend below the axes, so M stays in view: the lines drawn, then the map areas."""
    map_areas = [
        Patch(color=WALL, label='parede de $M$'),
        Patch(color=SHADE, label='zona de inflação do mapa de custos'),
    ]
    lines, _ = axes.get_legend_handles_labels()
    axes.legend(
        handles=[*lines, *map_areas],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncols=2,
        frameon=True,
        facecolor=SURFACE,
        edgecolor=GRID,
        fontsize=8,
    )


def main() -> None:
    if len(sys.argv) != 2:
        print('usage: plot_final_round <evidencias directory>')
        sys.exit(2)
    evidence = Path(sys.argv[1])
    plot_behavior_selection(evidence)
    plot_response_times(evidence)
    plot_expiry_delays(evidence)
    plot_power_cuts(evidence)
    plot_screens(evidence)
    plot_map(evidence)
    plot_plan(evidence)


if __name__ == '__main__':
    main()
