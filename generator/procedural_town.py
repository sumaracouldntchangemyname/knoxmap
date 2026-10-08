"""Generate a deterministic, offline street-and-building plan for a town."""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

from shapely.geometry import LineString, MultiPoint, Point, Polygon, box
from shapely.ops import linemerge, unary_union, voronoi_diagram

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
    low, high = cross_extent * 0.18, cross_extent * 0.82
    points = [(0.0, rng.uniform(low, high)),
              (float(extent - 1), rng.uniform(low, high))]
    for _ in range(5):
        refined = [points[0]]
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            midpoint = (x0 + x1) / 2
            displacement = rng.uniform(-1, 1) * cross_extent * (
                0.10 + irregularity * 0.28)
            across = max(low, min(high, (y0 + y1) / 2 + displacement))
            refined.append((midpoint, across))
            refined.append((x1, y1))
        points = refined
    return points if horizontal else [(y, x) for x, y in points]


def _organic_blocks(width: int, height: int, spacing: int,
                    rng: random.Random, variation: float):
    """Make a connected, curving block fabric from spaced town-center points."""
    bounds = box(0, 0, width - 1, height - 1)
    # Bridson sampling avoids the residual rows and columns of a jittered
    # lattice. The padded domain lets boundary blocks have ordinary neighbors.
    radius = spacing * 0.9
    cell_size = radius / math.sqrt(2)
    x0, y0 = -spacing, -spacing
    grid_w = math.ceil((width + spacing * 2) / cell_size)
    grid_h = math.ceil((height + spacing * 2) / cell_size)
    grid = [[-1] * grid_w for _ in range(grid_h)]
    sites: list[tuple[float, float]] = []
    active: list[int] = []

    def add_site(point):
        index = len(sites)
        sites.append(point)
        active.append(index)
        gx, gy = int((point[0] - x0) / cell_size), int((point[1] - y0) / cell_size)
        grid[gy][gx] = index

    add_site((rng.uniform(x0, width + spacing),
              rng.uniform(y0, height + spacing)))
    while active and len(sites) < 20_000:
        active_index = rng.randrange(len(active))
        base = sites[active[active_index]]
        found = False
        for _ in range(30):
            angle = rng.random() * math.tau
            distance = radius * math.sqrt(rng.uniform(1.0, 4.0))
            point = (base[0] + math.cos(angle) * distance,
                     base[1] + math.sin(angle) * distance)
            if not (x0 <= point[0] < width + spacing
                    and y0 <= point[1] < height + spacing):
                continue
            gx, gy = int((point[0] - x0) / cell_size), int((point[1] - y0) / cell_size)
            valid = True
            for cy in range(max(0, gy - 2), min(grid_h, gy + 3)):
                for cx in range(max(0, gx - 2), min(grid_w, gx + 3)):
                    neighbor = grid[cy][cx]
                    if neighbor >= 0 and math.dist(point, sites[neighbor]) < radius:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                add_site(point)
                found = True
                break
        if not found:
            active.pop(active_index)

    cells = voronoi_diagram(MultiPoint(sites), envelope=bounds).geoms
    blocks = []
    for cell in cells:
        clipped = cell.intersection(bounds)
        if clipped.geom_type == "Polygon" and clipped.area >= spacing * spacing * 0.12:
            blocks.append(clipped)
    blocks = _curve_blocks(blocks, width, height, spacing, rng, variation)
    roads = linemerge(unary_union([block.boundary for block in blocks]))
    lines = list(getattr(roads, "geoms", [roads]))
    return blocks, [line for line in lines if line.geom_type == "LineString"
                    and line.length >= 8
                    and not (line.bounds[2] < 2 or line.bounds[0] > width - 3
                             or line.bounds[3] < 2 or line.bounds[1] > height - 3)]


def _curve_blocks(blocks: list[Polygon], width: int, height: int,
                  spacing: int, rng: random.Random,
                  variation: float) -> list[Polygon]:
    """Curve shared street edges identically on both sides of each block."""
    amplitude = spacing * min(0.04, variation * 0.12)
    if amplitude < 0.15:
        return blocks
    cache: dict[tuple[tuple[float, float], tuple[float, float]],
                tuple[tuple[float, float], ...]] = {}

    def curve_edge(start, end):
        a = (round(start[0], 6), round(start[1], 6))
        b = (round(end[0], 6), round(end[1], 6))
        boundary_edge = (
            abs(a[0] - b[0]) < 1e-5 and
            (abs(a[0]) < 1e-5 or abs(a[0] - (width - 1)) < 1e-5)
        ) or (
            abs(a[1] - b[1]) < 1e-5 and
            (abs(a[1]) < 1e-5 or abs(a[1] - (height - 1)) < 1e-5)
        )
        length = math.dist(a, b)
        if boundary_edge or length < spacing * 0.42:
            return [a, b]
        key = tuple(sorted((a, b)))
        points = cache.get(key)
        if points is None:
            first, last = key
            dx, dy = last[0] - first[0], last[1] - first[1]
            segment_length = math.hypot(dx, dy)
            nx, ny = -dy / segment_length, dx / segment_length
            phase = rng.random() * math.tau
            count = max(3, math.ceil(segment_length / (spacing / 5)))
            curved = []
            for index in range(count + 1):
                t = index / count
                offset = amplitude * math.sin(math.pi * t) * math.sin(
                    math.tau * t + phase)
                curved.append((first[0] + dx * t + nx * offset,
                               first[1] + dy * t + ny * offset))
            points = tuple(curved)
            cache[key] = points
        return list(points if (a, b) == key else reversed(points))

    curved_blocks = []
    for block in blocks:
        ring = list(block.exterior.coords)
        points = []
        for start, end in zip(ring, ring[1:]):
            edge = curve_edge(start, end)
            points.extend(edge[:-1])
        points.append(points[0])
        curved = Polygon(points)
        if curved.is_valid and not curved.is_empty:
            curved_blocks.append(curved)
        else:
            curved_blocks.append(block)
    return curved_blocks


def _route_names(lines: list[LineString], classes: list[str]) -> list[str]:
    """Keep names continuous through junctions along the straightest route."""
    local_names = ("Oak Street", "Maple Avenue", "Pine Lane", "Cedar Street",
                   "School Road", "River Road", "Meadow Lane", "Elm Street",
                   "Parkway", "Church Street", "Mill Road", "Market Street")
    arterial_names = ("Main Street", "Market Avenue", "Center Street",
                      "Broadway", "Mill Road", "Parkway", "State Street")
    ends: dict[tuple[float, float], list[tuple[int, float, float]]] = {}
    for index, line in enumerate(lines):
        coords = list(line.coords)
        for start, next_point in ((coords[0], coords[1]),
                                  (coords[-1], coords[-2])):
            dx, dy = next_point[0] - start[0], next_point[1] - start[1]
            length = max(1e-9, math.hypot(dx, dy))
            key = (round(start[0], 5), round(start[1], 5))
            ends.setdefault(key, []).append((index, dx / length, dy / length))

    continuations = [set() for _ in lines]
    for incident in ends.values():
        pairs = []
        for left in range(len(incident)):
            for right in range(left + 1, len(incident)):
                a, ax, ay = incident[left]
                b, bx, by = incident[right]
                pairs.append((ax * bx + ay * by, a, b))
        used = set()
        for alignment, a, b in sorted(pairs):
            if alignment > -0.35 or a in used or b in used:
                continue
            used.update((a, b))
            continuations[a].add(b)
            continuations[b].add(a)

    components = []
    unseen = set(range(len(lines)))
    while unseen:
        first = min(unseen)
        stack, component = [first], []
        unseen.remove(first)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in continuations[current] & unseen:
                unseen.remove(neighbor)
                stack.append(neighbor)
        components.append(component)
    components.sort(key=lambda group: (
        round(min(lines[i].centroid.y for i in group), 3),
        round(min(lines[i].centroid.x for i in group), 3)))

    names = [""] * len(lines)
    counters = {"primary": 0, "secondary": 0, "local": 0}
    rank = {"primary": 3, "secondary": 2, "tertiary": 1, "residential": 0}
    for component in components:
        hierarchy = max((classes[i] for i in component), key=rank.get)
        if hierarchy == "primary":
            number = counters["primary"] + 1
            name = f"County Road {number}"
            counters["primary"] += 1
        elif hierarchy == "secondary":
            name = arterial_names[counters["secondary"] % len(arterial_names)]
            counters["secondary"] += 1
        else:
            name = local_names[counters["local"] % len(local_names)]
            counters["local"] += 1
        for index in component:
            names[index] = name
    return names


def _frontage_lots(block: Polygon, rng: random.Random,
                   min_frontage: float, max_frontage: float,
                   depth: float, road_setback: float):
    """Place non-overlapping parcels along the block's street-facing edges."""
    frontage_geometry = block.simplify(
        max(1.0, min_frontage * 0.25), preserve_topology=True)
    if frontage_geometry.geom_type != "Polygon":
        frontage_geometry = block
    exterior = list(frontage_geometry.exterior.coords)
    for start, end in zip(exterior, exterior[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        edge_length = math.hypot(dx, dy)
        if edge_length < min_frontage + 8:
            continue
        tx, ty = dx / edge_length, dy / edge_length
        nx, ny = -ty, tx
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        if not block.buffer(-1).covers(
                Point(midpoint[0] + nx * 2, midpoint[1] + ny * 2)):
            nx, ny = -nx, -ny
        cursor = rng.uniform(3.0, 7.0)
        stop = edge_length - rng.uniform(3.0, 7.0)
        while cursor + min_frontage <= stop:
            frontage = min(rng.uniform(min_frontage, max_frontage),
                           stop - cursor)
            if frontage < min_frontage:
                break
            along = cursor + frontage / 2
            center_x = start[0] + tx * along + nx * (
                road_setback + depth / 2)
            center_y = start[1] + ty * along + ny * (
                road_setback + depth / 2)
            half_front, half_depth = frontage / 2, depth / 2
            polygon = Polygon([
                (center_x - tx * half_front - nx * half_depth,
                 center_y - ty * half_front - ny * half_depth),
                (center_x + tx * half_front - nx * half_depth,
                 center_y + ty * half_front - ny * half_depth),
                (center_x + tx * half_front + nx * half_depth,
                 center_y + ty * half_front + ny * half_depth),
                (center_x - tx * half_front + nx * half_depth,
                 center_y - ty * half_front + ny * half_depth),
            ])
            if block.buffer(-road_setback).covers(polygon):
                yield polygon, (tx, ty, nx, ny)
            cursor += frontage + rng.uniform(2.0, 7.0)


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


def _projected_shape(proj, shape_data: dict | None):
    """Project a checked GeoJSON outline into the town's tile coordinates."""
    if not shape_data:
        return None
    polygons = shape_data["coordinates"]
    if shape_data["type"] == "Polygon":
        polygons = [polygons]
    projected = []
    for polygon in polygons:
        rings = [[proj.to_px(lat, lon) for lon, lat in ring]
                 for ring in polygon]
        if rings and len(rings[0]) >= 4:
            projected.append(Polygon(rings[0], rings[1:]))
    return unary_union(projected) if projected else None


def generate(proj, parameters: TownParameters, templates=None,
             max_features: int = 100_000, should_stop=None,
             max_building_side: int | None = None,
             selection_shape: dict | None = None) -> list[OSMFeature]:
    """Build a connected street network, green/water features and town lots."""
    import knoxstop

    rng = random.Random(parameters.seed)
    metres_per_tile = proj.meters_per_tile
    build_area = _projected_shape(proj, selection_shape)
    spacing = max(16, round(parameters.block_size_m / metres_per_tile))
    building_side_limit = min(
        parameters.max_building_side,
        (max_building_side if max_building_side is not None
         else parameters.max_building_side))
    organic = parameters.street_pattern == "organic"
    bend = parameters.road_irregularity if organic else 0.0
    variation = bend * 0.8
    nodes = []
    if organic:
        blocks, organic_roads = _organic_blocks(
            proj.width, proj.height, spacing, rng, variation)
    else:
        grid_variation = (0.12 if parameters.town_type == "compact_city"
                          else 0.08 if parameters.town_type == "industrial"
                          else 0.06)
        xs = _road_positions(proj.width, spacing, rng, grid_variation)
        ys = _road_positions(proj.height, spacing, rng, grid_variation)
        for row, y in enumerate(ys):
            node_row = []
            for x in xs:
                node_row.append((x, y))
            nodes.append(node_row)
        blocks, organic_roads = [], []

    features: list[OSMFeature] = []
    feature_id = -1

    def add(tags: dict, points: list[tuple[float, float]]) -> None:
        nonlocal feature_id
        if build_area is not None and "building" in tags:
            footprint = Polygon(points)
            if not build_area.covers(footprint):
                return
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

    def road_tags(index: int, name: str, points,
                  highway: str | None = None) -> dict:
        if highway is None:
            if index % 6 == 0:
                highway = "primary"
            elif index % 3 == 0:
                highway = "secondary"
            elif index % 2 == 0:
                highway = "tertiary"
            else:
                highway = "residential"
        width, lanes = {
            "primary": ("12", "2"), "secondary": ("8", "2"),
            "tertiary": ("6", "2"), "residential": ("5", "1"),
        }[highway]
        tags = {"highway": highway, "name": name, "width": f"{width} m",
                "lanes": lanes, "surface": "asphalt"}
        if river and _crosses_water(points, river):
            tags.update({"bridge": "yes", "layer": "1"})
        return tags

    major_roads = []
    if organic:
        ordered_roads = sorted(organic_roads, key=lambda line: (
            -line.length, round(line.centroid.y, 3), round(line.centroid.x, 3)))
        road_count = len(ordered_roads)
        road_classes = []
        for index in range(road_count):
            rank = index / max(1, road_count)
            road_classes.append(
                "primary" if rank < 0.06 else
                "secondary" if rank < 0.2 else
                "tertiary" if rank < 0.48 else "residential")
        route_names = _route_names(ordered_roads, road_classes)
        for index, road in enumerate(ordered_roads):
            points = list(road.coords)
            highway = road_classes[index]
            if highway in {"primary", "secondary"}:
                major_roads.append(road)
            add(road_tags(index, route_names[index], points, highway), points)
    else:
        road_specs = []
        for col in range(len(xs)):
            for row in range(len(ys) - 1):
                points = [nodes[row][col], nodes[row + 1][col]]
                road_specs.append((col, points))
        for row in range(len(ys)):
            for col in range(len(xs) - 1):
                points = [nodes[row][col], nodes[row][col + 1]]
                road_specs.append((row, points))
        road_lines = [LineString(points) for _index, points in road_specs]
        road_classes = [
            ("primary" if index % 6 == 0 else
             "secondary" if index % 3 == 0 else
             "tertiary" if index % 2 == 0 else "residential")
            for index, _points in road_specs]
        route_names = _route_names(road_lines, road_classes)
        for i, (index, points) in enumerate(road_specs):
            highway = road_classes[i]
            if highway in {"primary", "secondary"}:
                major_roads.append(road_lines[i])
            tags = road_tags(index, route_names[i], points, highway)
            add(tags, points)

    if river:
        bank_width = max(3.0, 12.0 / metres_per_tile)
        add({"leisure": "park", "name": "Riverbank Greenway"},
            _riverbank(river, bank_width))
        add({"waterway": "river", "name": "Town River"}, river)

    if organic:
        block_plans = [(block, 0, index) for index, block in enumerate(
            sorted(blocks, key=lambda shape: (shape.centroid.y,
                                               shape.centroid.x)))]
    else:
        block_plans = []
        for row in range(len(ys) - 1):
            for col in range(len(xs) - 1):
                corners = (nodes[row][col], nodes[row][col + 1],
                           nodes[row + 1][col + 1], nodes[row + 1][col])
                block_plans.append((Polygon(corners), row, col))

    industrial_side = rng.choice(("east", "west"))
    town_center = (proj.width * rng.uniform(0.44, 0.56),
                   proj.height * rng.uniform(0.44, 0.56))
    centers = [town_center]
    if parameters.town_type in {"compact_city", "suburb"}:
        spread = min(proj.width, proj.height) * (
            0.22 if parameters.town_type == "compact_city" else 0.3)
        centers.extend([
            (town_center[0] - spread, town_center[1] - spread * 0.52),
            (town_center[0] + spread, town_center[1] + spread * 0.48),
        ])
        if parameters.town_type == "compact_city":
            centers.append((town_center[0] - spread * 0.12,
                            town_center[1] + spread))
    center_radius = max(spacing, math.hypot(proj.width, proj.height) * 0.52)
    major_geometries = [LineString(road.coords) for road in major_roads]
    for block_index, (block, row, col) in enumerate(block_plans):
        if block_index % 8 == 0:
            knoxstop.check(should_stop, "the procedural town")
        setback = max(3.0, min(spacing * 0.08, 7.0 / metres_per_tile))
        inner = block.buffer(-setback)
        if inner.is_empty or inner.geom_type != "Polygon":
            continue
        left, top, right, bottom = inner.bounds
        left, top = math.ceil(left), math.ceil(top)
        right, bottom = math.floor(right), math.floor(bottom)
        center_distance = min(
            math.dist((block.centroid.x, block.centroid.y), center)
            for center in centers)
        centrality = max(0.0, 1.0 - center_distance / center_radius)
        block_center = Point(block.centroid.x, block.centroid.y)
        corridor_distance = min(
            (line.distance(block_center) for line in major_geometries),
            default=float("inf"))
        near_main_street = corridor_distance <= spacing * 0.58
        if right - left < 12 or bottom - top < 12:
            continue
        river_in_block = bool(river and _line_near_rect(
            river, left, top, right, bottom, 12.0 / metres_per_tile))
        if rng.random() < parameters.park_chance:
            add({"leisure": "park", "name": "Town Park"},
                list(inner.exterior.coords))
            path_y = (top + bottom) / 2
            path = LineString([(left + 3, path_y), (right - 3, path_y)])
            if inner.covers(path):
                add({"highway": "footway", "surface": "gravel",
                     "name": "Park Path"}, list(path.coords))
            continue
        if not river_in_block and rng.random() < parameters.water_chance:
            pond = _pond(left, top, right, bottom, rng)
            if inner.covers(Polygon(pond)):
                add({"natural": "water", "water": "pond",
                     "name": "Town Pond"}, pond)
                continue
        center_x, center_y = block.centroid.x, block.centroid.y
        edge_zone = center_x > proj.width * 0.78 if industrial_side == "east" \
            else center_x < proj.width * 0.22
        industrial_block = (edge_zone and rng.random() <
                            parameters.industrial_chance * 3.5)
        commercial_block = (
            (centrality > 0.5 or near_main_street)
            and rng.random() < min(
                0.9, parameters.commercial_chance
                * (1.25 + 1.0 * centrality
                   + (0.45 if near_main_street else 0.0))))
        civic_block = (not industrial_block and not commercial_block
                       and centrality > 0.44
                       and rng.random() < parameters.civic_chance
                       * (1.8 + centrality))
        if commercial_block:
            add({"landuse": "commercial", "name": "Town Centre"},
                list(inner.exterior.coords))
        elif industrial_block:
            add({"landuse": "industrial", "name": "Industrial Estate"},
                list(inner.exterior.coords))
        elif not civic_block:
            add({"landuse": "residential", "name": "Residential District"},
                list(inner.exterior.coords))

        if industrial_block:
            family, kind = "industrial", "industrial"
            min_frontage, max_frontage = 38, 64
            occupancy = min(0.94, 0.55 + parameters.building_density * 0.35)
        elif commercial_block:
            family, kind = "commercial", "shop"
            min_frontage, max_frontage = 18, 32
            occupancy = min(0.96, 0.62 + parameters.building_density * 0.34)
        elif civic_block:
            family, kind = "civic", "civic"
            min_frontage, max_frontage = 28, 48
            occupancy = min(0.9, 0.5 + parameters.building_density * 0.3)
        else:
            family, kind = "residential", "house"
            frontage_by_type = {
                "compact_city": (12, 20),
                "suburb": (18, 30),
                "rural_village": (25, 42),
                "industrial": (24, 38),
                "riverside": (20, 34),
            }
            min_frontage, max_frontage = frontage_by_type.get(
                parameters.town_type, (17, 29))
            density_floor = {
                "compact_city": 0.88, "suburb": 0.68,
                "rural_village": 0.58,
            }.get(parameters.town_type, 0.72)
            occupancy = min(
                0.98, 0.68 + parameters.building_density * 0.38)
            occupancy *= density_floor + (1.0 - density_floor) * centrality

        min_frontage /= metres_per_tile
        max_frontage /= metres_per_tile
        lot_depth = min(
            max(30.0 / metres_per_tile, (bottom - top) * rng.uniform(0.27, 0.36)),
            (bottom - top) * 0.44)
        road_setback = ((1.0 if commercial_block else
                         rng.uniform(1.0, 3.0)) / metres_per_tile)
        occupied: list[Polygon] = []
        house_number = rng.randrange(100, 900, 2)
        for lot, orientation in _frontage_lots(
                inner, rng, min_frontage, max_frontage,
                lot_depth, road_setback):
            if rng.random() > occupancy or civic_block and occupied:
                continue
            tx, ty, nx, ny = orientation
            kind_for_lot = kind
            family_for_lot = family
            if civic_block:
                kind_for_lot = rng.choices(
                    ("school", "church", "medical", "civic"),
                    weights=(0.38, 0.3, 0.16, 0.16))[0]
            elif commercial_block:
                kind_for_lot = rng.choice(("shop", "restaurant"))
            elif (centrality > 0.48 and rng.random() <
                  (0.22 if parameters.town_type == "compact_city"
                   else 0.12 if parameters.town_type == "small_town"
                   else 0.07)):
                kind_for_lot = "apartment"
                family_for_lot = "residential"

            frontage = lot.exterior.length / 2 - lot_depth
            max_width = min(building_side_limit, frontage - 4.0 / metres_per_tile)
            max_depth = min(building_side_limit, lot_depth - 5.0 / metres_per_tile)
            width, depth = _building_size(
                family_for_lot, int(max_width), int(max_depth), rng, templates,
                max_side=building_side_limit, metres_per_tile=metres_per_tile)
            if width < 3 or depth < 3:
                continue
            if commercial_block:
                front_setback = rng.uniform(0.8, 2.5) / metres_per_tile
            elif parameters.town_type == "compact_city":
                front_setback = rng.uniform(3.5, 7.0) / metres_per_tile
            elif parameters.town_type == "rural_village":
                front_setback = rng.uniform(6.0, 11.0) / metres_per_tile
            else:
                front_setback = rng.uniform(5.0, 9.0) / metres_per_tile
            side_offset = rng.uniform(-0.14, 0.14) * max(
                0.0, frontage - width)
            center_x = lot.centroid.x + tx * side_offset - nx * (
                lot_depth / 2 - front_setback - depth / 2)
            center_y = lot.centroid.y + ty * side_offset - ny * (
                lot_depth / 2 - front_setback - depth / 2)
            footprint = Polygon([
                (center_x - tx * width / 2 - nx * depth / 2,
                 center_y - ty * width / 2 - ny * depth / 2),
                (center_x + tx * width / 2 - nx * depth / 2,
                 center_y + ty * width / 2 - ny * depth / 2),
                (center_x + tx * width / 2 + nx * depth / 2,
                 center_y + ty * width / 2 + ny * depth / 2),
                (center_x - tx * width / 2 + nx * depth / 2,
                 center_y - ty * width / 2 + ny * depth / 2),
            ])
            if (not lot.covers(footprint)
                    or any(footprint.intersects(other) for other in occupied)
                    or (river and _line_near_rect(
                        river, *footprint.bounds, 18.0 / metres_per_tile))):
                continue
            occupied.append(footprint)
            if kind_for_lot in ("shop", "restaurant"):
                tags = {
                    "building": "retail" if kind_for_lot == "shop" else "restaurant",
                    "shop": "convenience" if kind_for_lot == "shop" else None,
                    "amenity": "restaurant" if kind_for_lot == "restaurant" else None,
                    "building:levels": str(rng.choice((1, 1, 2))),
                }
                tags = {key: value for key, value in tags.items()
                        if value is not None}
            elif kind_for_lot == "industrial":
                tags = {"building": "industrial",
                        "industrial": rng.choice(("factory", "warehouse")),
                        "building:levels": "1"}
            elif kind_for_lot in ("school", "church", "medical", "civic"):
                amenity = {"school": "school", "church": "place_of_worship",
                           "medical": "clinic", "civic": "fire_station"}[kind_for_lot]
                tags = {"building": "public", "amenity": amenity,
                        "building:levels": str(rng.choice((1, 1, 2)))}
            elif kind_for_lot == "apartment":
                tags = {"building": "apartments",
                        "building:levels": str(rng.choice((3, 4, 5)))}
            else:
                tags = {"building": "house",
                        "building:levels": str(rng.choice((1, 1, 2))),
                        "building:material": rng.choice(
                            ("brick", "wood", "concrete")),
                        "addr:housenumber": str(house_number)}
                house_number += rng.choice((2, 4, 6))
            add(tags, list(footprint.exterior.coords))
    return features


def _building_size(family: str, available_width: int, available_height: int,
                   rng: random.Random, templates=None,
                   max_side: int = 40,
                   metres_per_tile: float = 1.0) -> tuple[int, int]:
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
    elif family == "industrial":
        sizes = ((22, 30), (28, 36), (34, 24), (40, 32))
    elif family == "civic":
        sizes = ((18, 22), (24, 28), (30, 24), (32, 36))
    else:
        sizes = ((9, 11), (10, 13), (12, 14), (14, 16), (16, 13), (17, 18))
    sizes = tuple((max(3, round(width / metres_per_tile)),
                   max(3, round(height / metres_per_tile)))
                  for width, height in sizes)
    fitting = [(w, h) for w, h in sizes
               if w <= available_width - 2 and h <= available_height - 2]
    return rng.choice(fitting) if fitting else (0, 0)
