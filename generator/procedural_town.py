"""Generate a deterministic, offline street-and-building plan for a town."""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

from .osm import OSMFeature


TOWN_PROFILES = {
    "custom": {},
    "small_town": {},
    "compact_city": {
        "block_size_m": 120, "building_density": 0.9, "park_chance": 0.05,
        "commercial_chance": 0.3, "civic_chance": 0.1,
        "industrial_chance": 0.02, "river_chance": 0.15,
        "street_pattern": "grid", "road_irregularity": 0.0,
    },
    "suburb": {
        "block_size_m": 230, "building_density": 0.56, "park_chance": 0.12,
        "commercial_chance": 0.08, "civic_chance": 0.035,
        "industrial_chance": 0.01, "river_chance": 0.12,
        "street_pattern": "organic", "road_irregularity": 0.12,
    },
    "rural_village": {
        "block_size_m": 300, "building_density": 0.42, "park_chance": 0.08,
        "commercial_chance": 0.06, "civic_chance": 0.03,
        "industrial_chance": 0.025, "river_chance": 0.32,
        "street_pattern": "organic", "road_irregularity": 0.3,
    },
    "industrial": {
        "block_size_m": 240, "building_density": 0.56, "park_chance": 0.06,
        "commercial_chance": 0.1, "civic_chance": 0.04,
        "industrial_chance": 0.22, "river_chance": 0.1,
        "street_pattern": "grid", "road_irregularity": 0.05,
    },
    "riverside": {
        "block_size_m": 180, "building_density": 0.62, "park_chance": 0.11,
        "commercial_chance": 0.14, "civic_chance": 0.06,
        "industrial_chance": 0.02, "water_chance": 0.08,
        "river_chance": 1.0, "street_pattern": "organic",
        "road_irregularity": 0.2,
    },
}


@dataclass(frozen=True)
class TownParameters:
    block_size_m: float = 180.0
    building_density: float = 0.68
    park_chance: float = 0.10
    commercial_chance: float = 0.16
    civic_chance: float = 0.06
    industrial_chance: float = 0.04
    water_chance: float = 0.04
    river_chance: float = 0.2
    town_type: str = "small_town"
    street_pattern: str = "organic"
    road_irregularity: float = 0.18
    max_building_side: int = 40
    seed: int = 1

    @classmethod
    def from_dict(cls, data: dict | None, default_seed: int = 1) -> "TownParameters":
        if not isinstance(data, dict):
            raise ValueError("Procedural town parameters must be an object.")
        town_type = data.get("town_type", "small_town")
        if not isinstance(town_type, str) or town_type not in TOWN_PROFILES:
            raise ValueError("town_type must be a supported town character.")
        limits = {
            "block_size_m": (60.0, 500.0),
            "building_density": (0.15, 1.0),
            "park_chance": (0.0, 0.6),
            "commercial_chance": (0.0, 0.8),
            "civic_chance": (0.0, 0.3),
            "industrial_chance": (0.0, 0.3),
            "water_chance": (0.0, 0.4),
            "river_chance": (0.0, 1.0),
            "road_irregularity": (0.0, 0.35),
            "max_building_side": (12, 200),
            "seed": (0, 2 ** 31 - 1),
        }
        values = {**TOWN_PROFILES[town_type], "seed": default_seed,
                  "town_type": town_type}
        for name, (low, high) in limits.items():
            if name not in data:
                continue
            raw = data[name]
            integer = name in {"seed", "max_building_side"}
            try:
                value = int(raw) if integer else float(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be a number.") from exc
            if integer and isinstance(raw, float) and not raw.is_integer():
                raise ValueError(f"{name} must be a whole number.")
            if (not integer and not math.isfinite(value)) \
                    or not low <= value <= high:
                raise ValueError(f"{name} must be between {low:g} and {high:g}.")
            values[name] = value
        street_pattern = data.get(
            "street_pattern", TOWN_PROFILES[town_type].get("street_pattern", "organic"))
        if not isinstance(street_pattern, str) \
                or street_pattern not in {"organic", "grid"}:
            raise ValueError("street_pattern must be 'organic' or 'grid'.")
        values["street_pattern"] = street_pattern
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)


def _road_positions(extent: int, spacing: int, rng: random.Random,
                    variation: float) -> list[int]:
    positions = [0]
    while positions[-1] + spacing < extent - 1:
        step = max(16, round(spacing * rng.uniform(1.0 - variation,
                                                  1.0 + variation)))
        next_position = positions[-1] + step
        if next_position >= extent - 1:
            break
        positions.append(next_position)
    if positions[-1] != extent - 1:
        positions.append(extent - 1)
    return positions


def _coordinate(proj, x: float, y: float) -> tuple[float, float]:
    lat, lon = proj.to_latlon(x, y)
    return float(lat), float(lon)


def _way(proj, feature_id: int, tags: dict,
         points: list[tuple[float, float]]) -> OSMFeature:
    geometry = [_coordinate(proj, x, y) for x, y in points]
    return OSMFeature(feature_id, "way", tags, geometry)


def _segment_intersects(a, b, c, d) -> bool:
    def orientation(p, q, r):
        return ((q[1] - p[1]) * (r[0] - q[0])
                - (q[0] - p[0]) * (r[1] - q[1]))

    def on_segment(p, q, r):
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
                and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    o1, o2 = orientation(a, b, c), orientation(a, b, d)
    o3, o4 = orientation(c, d, a), orientation(c, d, b)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    return ((o1 == 0 and on_segment(a, c, b))
            or (o2 == 0 and on_segment(a, d, b))
            or (o3 == 0 and on_segment(c, a, d))
            or (o4 == 0 and on_segment(c, b, d)))


def _crosses_water(points, water_line) -> bool:
    return any(_segment_intersects(a, b, c, d)
               for a, b in zip(points, points[1:])
               for c, d in zip(water_line, water_line[1:]))


def _line_near_rect(line, left: float, top: float,
                    right: float, bottom: float, margin: float) -> bool:
    left, top = left - margin, top - margin
    right, bottom = right + margin, bottom + margin
    corners = ((left, top), (right, top), (right, bottom),
               (left, bottom), (left, top))
    for start, end in zip(line, line[1:]):
        if (left <= start[0] <= right and top <= start[1] <= bottom
                or left <= end[0] <= right and top <= end[1] <= bottom):
            return True
        if any(_segment_intersects(start, end, a, b)
               for a, b in zip(corners, corners[1:])):
            return True
    return False


def _riverbank(line, width: float) -> list[tuple[float, float]]:
    left_bank, right_bank = [], []
    for index, point in enumerate(line):
        before = line[max(0, index - 1)]
        after = line[min(len(line) - 1, index + 1)]
        dx, dy = after[0] - before[0], after[1] - before[1]
        length = max(1.0, math.hypot(dx, dy))
        nx, ny = -dy / length * width, dx / length * width
        left_bank.append((point[0] + nx, point[1] + ny))
        right_bank.append((point[0] - nx, point[1] - ny))
    return left_bank + list(reversed(right_bank)) + [left_bank[0]]


def _river_path(width: int, height: int, horizontal: bool,
                rng: random.Random, irregularity: float) -> list[tuple[float, float]]:
    extent = width if horizontal else height
    cross_extent = height if horizontal else width
    center = rng.uniform(cross_extent * 0.3, cross_extent * 0.7)
    amplitude = cross_extent * (0.06 + irregularity * 0.2)
    phase = rng.uniform(0, math.tau)
    waves = rng.choice((1, 1, 2))
    points = []
    for index in range(17):
        along = (extent - 1) * index / 16
        across = center + amplitude * math.sin(
            math.tau * waves * index / 16 + phase)
        points.append((along, across) if horizontal else (across, along))
    return points


def _pond(left: int, top: int, right: int, bottom: int,
          rng: random.Random) -> list[tuple[float, float]]:
    cx = rng.uniform(left + (right - left) * 0.4,
                     right - (right - left) * 0.4)
    cy = rng.uniform(top + (bottom - top) * 0.4,
                     bottom - (bottom - top) * 0.4)
    rx = (right - left) * rng.uniform(0.15, 0.25)
    ry = (bottom - top) * rng.uniform(0.15, 0.25)
    phase = rng.uniform(0, math.tau)
    points = []
    for index in range(13):
        angle = math.tau * index / 12
        shore = rng.uniform(0.82, 1.12)
        points.append((cx + math.cos(angle + phase) * rx * shore,
                       cy + math.sin(angle + phase) * ry * shore))
    points[-1] = points[0]
    return points


def generate(proj, parameters: TownParameters, templates=None,
             max_features: int = 100_000, should_stop=None,
             max_building_side: int | None = None) -> list[OSMFeature]:
    """Build a connected street network, green/water features and town lots."""
    import knoxstop

    rng = random.Random(parameters.seed)
    metres_per_tile = proj.meters_per_tile
    spacing = max(16, round(parameters.block_size_m / metres_per_tile))
    building_side_limit = min(
        parameters.max_building_side,
        (max_building_side if max_building_side is not None
         else parameters.max_building_side))
    organic = parameters.street_pattern == "organic"
    bend = parameters.road_irregularity if organic else 0.0
    variation = bend * 0.8
    xs = _road_positions(proj.width, spacing, rng, variation)
    ys = _road_positions(proj.height, spacing, rng, variation)
    nodes = []
    for row, y in enumerate(ys):
        node_row = []
        for col, x in enumerate(xs):
            edge_x = col in (0, len(xs) - 1)
            edge_y = row in (0, len(ys) - 1)
            jitter = spacing * bend * 0.42
            node_row.append((
                x if edge_x else x + rng.uniform(-jitter, jitter),
                y if edge_y else y + rng.uniform(-jitter, jitter)))
        nodes.append(node_row)

    features: list[OSMFeature] = []
    feature_id = -1

    def add(tags: dict, points: list[tuple[float, float]]) -> None:
        nonlocal feature_id
        features.append(_way(proj, feature_id, tags, points))
        feature_id -= 1
        if len(features) > max_features:
            raise ValueError(
                f"This selection would create more than {max_features:,} "
                "procedural features. Draw a smaller area or increase block size.")

    river = None
    if rng.random() < parameters.river_chance:
        river = _river_path(proj.width, proj.height, rng.choice((True, False)),
                            rng, bend)

    def road_tags(index: int, name: str, points) -> dict:
        if index % 6 == 0:
            highway, width, lanes = "primary", "12", "2"
        elif index % 3 == 0:
            highway, width, lanes = "secondary", "8", "2"
        elif index % 2 == 0:
            highway, width, lanes = "tertiary", "6", "2"
        else:
            highway, width, lanes = "residential", "5", "1"
        tags = {"highway": highway, "name": name, "width": f"{width} m",
                "lanes": lanes, "surface": "asphalt"}
        if river and _crosses_water(points, river):
            tags.update({"bridge": "yes", "layer": "1"})
        return tags

    local_names = ("Oak Street", "Maple Avenue", "Pine Lane", "Cedar Street",
                   "School Road", "River Road", "Meadow Lane", "Elm Street",
                   "Parkway", "Church Street", "Mill Road", "Market Street")

    def road_name(index: int, horizontal: bool) -> str:
        axis = "east-west" if horizontal else "north-south"
        if index % 6 == 0:
            return f"County Route {axis} {index // 6 + 1}"
        if index % 3 == 0:
            return f"Market Road {axis} {index // 3}"
        return local_names[(index * 5 + int(horizontal) * 7) % len(local_names)]

    for col in range(len(xs)):
        for row in range(len(ys) - 1):
            start, end = nodes[row][col], nodes[row + 1][col]
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            offset = rng.uniform(-spacing * bend * 0.18,
                                 spacing * bend * 0.18)
            middle = ((start[0] + end[0]) / 2 - dy / max(length, 1) * offset,
                      (start[1] + end[1]) / 2 + dx / max(length, 1) * offset)
            points = [start, middle, end] if bend else [start, end]
            add(road_tags(col, road_name(col, False), points), points)
    for row in range(len(ys)):
        for col in range(len(xs) - 1):
            start, end = nodes[row][col], nodes[row][col + 1]
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            offset = rng.uniform(-spacing * bend * 0.18,
                                 spacing * bend * 0.18)
            middle = ((start[0] + end[0]) / 2 - dy / max(length, 1) * offset,
                      (start[1] + end[1]) / 2 + dx / max(length, 1) * offset)
            points = [start, middle, end] if bend else [start, end]
            add(road_tags(row, road_name(row, True), points), points)

    if river:
        bank_width = max(3.0, 12.0 / metres_per_tile)
        add({"leisure": "park", "name": "Riverbank Greenway"},
            _riverbank(river, bank_width))
        add({"waterway": "river", "name": "Town River"}, river)

    for row in range(len(ys) - 1):
        if row % 8 == 0:
            knoxstop.check(should_stop, "the procedural town")
        for col in range(len(xs) - 1):
            corners = (nodes[row][col], nodes[row][col + 1],
                       nodes[row + 1][col], nodes[row + 1][col + 1])
            left = math.ceil(min(point[0] for point in corners) + 10 + spacing * bend * 0.08)
            right = math.floor(max(point[0] for point in corners) - 10 - spacing * bend * 0.08)
            top = math.ceil(min(point[1] for point in corners) + 10 + spacing * bend * 0.08)
            bottom = math.floor(max(point[1] for point in corners) - 10 - spacing * bend * 0.08)
            if right - left < 12 or bottom - top < 12:
                continue
            river_in_block = bool(river and _line_near_rect(
                river, left, top, right, bottom, 12.0 / metres_per_tile))
            if rng.random() < parameters.park_chance:
                add({"leisure": "park", "name": "Town Park"},
                    [(left, top), (right, top), (right, bottom),
                     (left, bottom), (left, top)])
                path_y = rng.uniform(top + 3, bottom - 3)
                add({"highway": "footway", "surface": "gravel",
                     "name": "Park Path"},
                    [(left + 3, path_y), ((left + right) / 2, path_y + 2),
                     (right - 3, path_y)])
                continue
            if not river_in_block and rng.random() < parameters.water_chance:
                pond = _pond(left, top, right, bottom, rng)
                add({"natural": "water", "water": "pond",
                     "name": "Town Pond"}, pond)
                continue

            centrality = 1.0 - min(1.0, math.dist(
                (col + 0.5, row + 0.5),
                ((len(xs) - 1) / 2, (len(ys) - 1) / 2))
                / max(1.0, math.hypot(len(xs), len(ys)) / 2))
            civic_block = rng.random() < parameters.civic_chance * (
                0.5 + centrality)
            industrial_block = (not civic_block and
                                rng.random() < parameters.industrial_chance *
                                (1.2 - centrality))
            commercial_block = rng.random() < min(
                0.95, parameters.commercial_chance * (0.45 + centrality))
            y = top
            while y < bottom - 5:
                x = left
                row_height = 0
                while x < right - 5:
                    if civic_block:
                        family, kind = "civic", "civic"
                    elif industrial_block:
                        family, kind = "industrial", "industrial"
                    elif commercial_block and (x == left or rng.random() < 0.45):
                        family, kind = "commercial", rng.choice(
                            ("shop", "restaurant"))
                    elif centrality > 0.45 and rng.random() < 0.12:
                        family, kind = "residential", "apartment"
                    else:
                        family, kind = "residential", "house"
                    width, height = _building_size(
                        family, right - x, bottom - y, rng, templates,
                        max_side=building_side_limit)
                    if width < 3 or height < 3:
                        break
                    row_height = max(row_height, height)
                    if rng.random() <= parameters.building_density:
                        if river and _line_near_rect(
                                river, x, y, x + width, y + height,
                                18.0 / metres_per_tile):
                            x += width + 2
                            continue
                        if kind == "shop":
                            tags = {"building": "retail", "shop": "convenience",
                                    "building:levels": str(rng.choice((1, 1, 2)))}
                        elif kind == "restaurant":
                            tags = {"building": "restaurant", "amenity": "restaurant",
                                    "building:levels": str(rng.choice((1, 1, 2)))}
                        elif kind == "office":
                            tags = {"building": "office", "office": "company",
                                    "building:levels": str(rng.choice((2, 3, 4)))}
                        elif kind == "industrial":
                            tags = {"building": "industrial",
                                    "industrial": rng.choice(("factory", "warehouse")),
                                    "building:levels": str(rng.choice((1, 1, 2)))}
                        elif kind == "civic":
                            amenity = rng.choice(("school", "place_of_worship",
                                                  "clinic", "fire_station"))
                            tags = {"building": "public", "amenity": amenity,
                                    "building:levels": str(rng.choice((1, 1, 2)))}
                        elif kind == "apartment":
                            tags = {"building": "apartments",
                                    "building:levels": str(rng.choice((3, 4, 5)))}
                        else:
                            tags = {"building": "house",
                                    "building:levels": str(rng.choice((1, 1, 2))),
                                    "building:material": rng.choice(
                                        ("brick", "wood", "concrete"))}
                        add(tags, [(x, y), (x + width, y), (x + width, y + height),
                                   (x, y + height), (x, y)])
                    gap = max(2, round(2 + (1.0 - parameters.building_density) * 10))
                    x += width + gap
                if row_height == 0:
                    break
                y += row_height + 4
    return features


def _building_size(family: str, available_width: int, available_height: int,
                   rng: random.Random, templates=None,
                   max_side: int = 40) -> tuple[int, int]:
    if templates is not None:
        sizes = [(width, height)
                 for width, height in templates.dimensions(
                     family, min(max_side, available_width - 2),
                     min(max_side, available_height - 2))
                 if width >= 3 and height >= 3]
        if sizes:
            return rng.choice(sizes)
    if family == "commercial":
        sizes = ((12, 12), (14, 16), (16, 14), (18, 18), (20, 16))
    else:
        sizes = ((10, 12), (12, 14), (14, 16), (16, 14), (18, 16))
    fitting = [(w, h) for w, h in sizes
               if w <= available_width - 2 and h <= available_height - 2]
    return rng.choice(fitting) if fitting else (0, 0)
