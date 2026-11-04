"""Generate the synthetic map M of V-04 as a map_server image and metadata (m.pgm, m.yaml).

M is 3 x 3 m at 0.05 m resolution, with one wall that forces the plan to turn. A cell is
occupied when its centre lies inside the wall rectangle. The script writes next to itself, and
both outputs are versioned, so the deliberator always serves the same M (robot-docs nota 0003).

Usage: python3 make_map_m.py

V-04 parameter M: synthetic, 3 x 3 m, resolution 0.05 m, with a 0.1 x 2.0 m wall (x from 0.5 to
0.6 m, y from -0.5 to 1.5 m) that forces a turn.

Criticality: T2, product (robot-docs codigo.md).
"""

from pathlib import Path

MAP_SIZE = 3.0  # m, side of the square map (V-04, M)
MAP_RESOLUTION = 0.05  # m per cell, res (V-04; ADR 041)
# Map frame position of the lower-left corner: p0 = (0, 0) lies 0.5 m inside the map
MAP_ORIGIN = (-0.5, -0.5)  # m
WALL_X = (0.5, 0.6)  # m (V-04, M)
WALL_Y = (-0.5, 1.5)  # m (V-04, M)

FREE = 254
OCCUPIED = 0
IMAGE_NAME = 'm.pgm'
METADATA_NAME = 'm.yaml'


def is_inside_wall(x: float, y: float) -> bool:
    return WALL_X[0] <= x <= WALL_X[1] and WALL_Y[0] <= y <= WALL_Y[1]


def get_cell_centre(row: int, col: int, cells: int) -> tuple[float, float]:
    """Map frame centre of a cell; PGM row 0 is the top of the map."""
    x = MAP_ORIGIN[0] + (col + 0.5) * MAP_RESOLUTION
    y = MAP_ORIGIN[1] + (cells - 1 - row + 0.5) * MAP_RESOLUTION
    return x, y


def build_image(cells: int) -> bytes:
    pixels = bytearray()
    for row in range(cells):
        for col in range(cells):
            pixels.append(OCCUPIED if is_inside_wall(*get_cell_centre(row, col, cells)) else FREE)
    return bytes(pixels)


def build_pgm(cells: int, image: bytes) -> bytes:
    return f'P5\n{cells} {cells}\n255\n'.encode() + image


def build_metadata() -> str:
    return (
        f'image: {IMAGE_NAME}\n'
        'mode: trinary\n'
        f'resolution: {MAP_RESOLUTION}\n'
        f'origin: [{MAP_ORIGIN[0]}, {MAP_ORIGIN[1]}, 0.0]\n'
        'negate: 0\n'
        'occupied_thresh: 0.65\n'
        'free_thresh: 0.25\n'
    )


def main() -> None:
    cells = round(MAP_SIZE / MAP_RESOLUTION)
    image = build_image(cells)
    here = Path(__file__).parent
    (here / IMAGE_NAME).write_bytes(build_pgm(cells, image))
    (here / METADATA_NAME).write_text(build_metadata())
    print(f'{cells} x {cells} cells, {image.count(OCCUPIED)} occupied')


if __name__ == '__main__':
    main()
