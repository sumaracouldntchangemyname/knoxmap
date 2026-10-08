import random
import tempfile
import unittest
from pathlib import Path

from shapely.geometry import Polygon

from generator import osm
from generator.procedural_town import (TownParameters, _building_size,
                                       _line_near_rect, generate)
from knoxbuild.footprint import place
import numpy as np


class GridProjection:
    width = 720
    height = 720
    meters_per_tile = 1.0

    @staticmethod
    def to_latlon(x, y):
        return y, x

    @staticmethod
    def to_px(lat, lon):
        return lon, lat


class ProceduralTownTests(unittest.TestCase):
    def test_seed_makes_the_same_features(self):
        params = TownParameters.from_dict({"seed": 42})
        first = generate(GridProjection(), params)
        second = generate(GridProjection(), params)
        self.assertEqual(
            [(feature.tags, feature.geometry) for feature in first],
            [(feature.tags, feature.geometry) for feature in second])

    def test_town_types_select_distinct_realistic_layouts(self):
        city = TownParameters.from_dict({"town_type": "compact_city"})
        village = TownParameters.from_dict({"town_type": "rural_village"})
        self.assertEqual(city.street_pattern, "grid")
        self.assertGreater(city.building_density, village.building_density)
        self.assertGreater(village.block_size_m, city.block_size_m)
        self.assertEqual(
            TownParameters.from_dict(city.to_dict()), city)

    def test_features_include_roads_buildings_and_a_park(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 12, "park_chance": 0.6,
                "commercial_chance": 0.8, "building_density": 1.0,
            }))
        categories = [osm.classify(f.tags, f.geometry[0] == f.geometry[-1])
                      for f in features]
        self.assertIn("road_major", categories)
        self.assertIn("road_minor", categories)
        self.assertIn("building", categories)
        self.assertIn("park", categories)

    def test_network_has_road_hierarchy_and_crossing_bridges(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 23, "street_pattern": "grid", "river_chance": 1,
                "water_chance": 0, "park_chance": 0,
            }))
        highways = {feature.tags.get("highway") for feature in features}
        self.assertTrue({"primary", "secondary", "tertiary", "residential"}
                        <= highways)
        rivers = [feature for feature in features
                  if feature.tags.get("waterway") == "river"]
        self.assertEqual(len(rivers), 1)
        self.assertTrue(any(feature.tags.get("bridge") == "yes"
                            for feature in features))
        self.assertEqual(osm.classify(rivers[0].tags), "water")

    def test_organic_streets_form_irregular_blocks_not_a_jittered_grid(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 2026, "river_chance": 0, "park_chance": 0,
                "water_chance": 0,
            }))
        roads = [f for f in features if f.tags.get("highway") in {
            "primary", "secondary", "tertiary", "residential"}]
        angled = sum(
            abs(b[1] - a[1]) > 1 and abs(b[0] - a[0]) > 1
            for road in roads for a, b in zip(road.geometry, road.geometry[1:]))
        self.assertGreater(angled, len(roads) // 3)
        classes = {road.tags["highway"] for road in roads}
        self.assertGreaterEqual(len(classes), 3)
        self.assertTrue(any(len(road.geometry) > 3 for road in roads))

    def test_generated_roads_share_intersections_and_zones_have_distinct_uses(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 82, "river_chance": 0, "park_chance": 0,
                "water_chance": 0, "commercial_chance": 0.5,
                "civic_chance": 0.15, "industrial_chance": 0.2,
            }))
        roads = [feature for feature in features
                 if feature.tags.get("highway") in {
                     "primary", "secondary", "tertiary", "residential"}]
        endpoints = {}
        for road in roads:
            for point in (road.geometry[0], road.geometry[-1]):
                key = (round(point[0], 7), round(point[1], 7))
                endpoints[key] = endpoints.get(key, 0) + 1
        self.assertGreater(sum(count >= 2 for count in endpoints.values()),
                           len(roads) // 4)
        landuses = {feature.tags.get("landuse") for feature in features}
        self.assertTrue({"residential", "commercial", "industrial"} <= landuses)

    def test_named_streets_continue_through_intersections(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 47, "town_type": "compact_city",
                "street_pattern": "grid", "river_chance": 0,
                "park_chance": 0, "water_chance": 0,
            }))
        roads = [feature for feature in features
                 if feature.tags.get("highway") in {
                     "primary", "secondary", "tertiary", "residential"}]
        named = {}
        for road in roads:
            named.setdefault(road.tags["name"], []).append(road)
        routes = [ways for ways in named.values() if len(ways) > 1]
        self.assertTrue(routes)
        self.assertTrue(any(
            len({(round(point[0], 7), round(point[1], 7))
                 for way in ways for point in (way.geometry[0], way.geometry[-1])})
            < 2 * len(ways)
            for ways in routes))

    def test_generated_road_names_survive_the_feature_cache_for_map_building(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 74, "river_chance": 0, "park_chance": 0,
                "water_chance": 0,
            }))
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "procedural_osm.json.gz"
            osm.save_cache(str(cache), (0, 0, 720, 720), features)
            cached = osm.load_cache(str(cache), (0, 0, 720, 720))

        self.assertIsNotNone(cached)
        named_roads = [f for f in cached
                       if f.tags.get("highway") in {
                           "primary", "secondary", "tertiary", "residential"}
                       and f.tags.get("name")]
        self.assertTrue(named_roads)
        self.assertEqual(
            [(f.tags, f.geometry) for f in features],
            [(f.tags, f.geometry) for f in cached])

    def test_grid_blocks_have_spacing_variation_instead_of_one_repeated_pitch(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 48, "town_type": "compact_city",
                "street_pattern": "grid", "river_chance": 0,
                "park_chance": 0, "water_chance": 0,
            }))
        roads = [feature for feature in features
                 if feature.tags.get("highway") in {
                     "primary", "secondary", "tertiary", "residential"}]
        x_positions = sorted({round(lon, 4) for road in roads
                              for lat, lon in road.geometry
                              if max(p[0] for p in road.geometry)
                              - min(p[0] for p in road.geometry) < 1e-5})
        gaps = {round(b - a, 1) for a, b in zip(x_positions, x_positions[1:])}
        self.assertGreaterEqual(len(gaps), 3)

    def test_compact_city_has_multiple_centers_and_more_apartments(self):
        params = {
            "seed": 93, "river_chance": 0, "park_chance": 0,
            "water_chance": 0, "commercial_chance": 0,
            "civic_chance": 0,
        }
        compact = generate(
            GridProjection(),
            TownParameters.from_dict({**params, "town_type": "compact_city"}))
        village = generate(
            GridProjection(),
            TownParameters.from_dict({**params, "town_type": "rural_village"}))
        compact_flats = sum(f.tags.get("building") == "apartments"
                            for f in compact)
        village_flats = sum(f.tags.get("building") == "apartments"
                            for f in village)
        self.assertGreater(compact_flats, village_flats)

    def test_irregular_selection_does_not_cut_through_generated_buildings(self):
        outline = {
            "type": "Polygon",
            "coordinates": [[(100, 100), (620, 130), (540, 620),
                            (250, 560), (100, 100)]],
        }
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 54, "river_chance": 0, "park_chance": 0,
                "water_chance": 0,
            }),
            selection_shape=outline)
        boundary = Polygon(outline["coordinates"][0])
        buildings = [feature for feature in features
                     if "building" in feature.tags]
        self.assertTrue(buildings)
        for building in buildings:
            footprint = Polygon([(lon, lat) for lat, lon in building.geometry])
            self.assertTrue(boundary.covers(footprint))

    def test_river_has_a_greenway_and_buildings_respect_its_setback(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 31, "river_chance": 1, "water_chance": 0,
                "park_chance": 0, "building_density": 1,
            }))
        river = next(f for f in features if f.tags.get("waterway") == "river")
        river_xy = [(lon, lat) for lat, lon in river.geometry]
        self.assertTrue(any(f.tags.get("name") == "Riverbank Greenway"
                            for f in features))
        buildings = [f for f in features if "building" in f.tags]
        self.assertTrue(buildings)
        for building in buildings:
            points = [(lon, lat) for lat, lon in building.geometry]
            self.assertFalse(_line_near_rect(
                river_xy, min(x for x, _ in points), min(y for _, y in points),
                max(x for x, _ in points), max(y for _, y in points), 18))

    def test_homes_are_spread_along_frontages_with_yards_and_no_overlap(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 25, "river_chance": 0, "park_chance": 0,
                "water_chance": 0,
            }))
        homes = [Polygon([(lon, lat) for lat, lon in feature.geometry])
                 for feature in features
                 if feature.tags.get("building") == "house"]
        self.assertGreaterEqual(len(homes), 40)
        for index, home in enumerate(homes):
            for other in homes[index + 1:]:
                self.assertLessEqual(home.intersection(other).area, 0.01)
        self.assertGreater(sum(home.area for home in homes), 1_000)
        self.assertLess(sum(home.area for home in homes), 80_000)
        nearest = sorted(
            min(home.distance(other) for j, other in enumerate(homes)
                if index != j)
            for index, home in enumerate(homes))
        self.assertGreater(nearest[len(nearest) // 2], 6)

    def test_house_footprints_keep_real_world_size_at_coarser_scale(self):
        width, depth = _building_size(
            "residential", 10, 12, random.Random(4),
            metres_per_tile=2)
        self.assertGreaterEqual(width * 2, 9)
        self.assertLessEqual(width * 2, 17)
        self.assertGreaterEqual(depth * 2, 11)
        self.assertLessEqual(depth * 2, 18)

    def test_ponds_and_organic_roads_follow_parameters(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({
                "seed": 24, "street_pattern": "organic",
                "road_irregularity": 0.3, "river_chance": 0,
                "water_chance": 0.4, "park_chance": 0,
            }))
        self.assertTrue(any(feature.tags.get("natural") == "water"
                            for feature in features))
        self.assertTrue(any(feature.tags.get("highway") == "primary"
                            and len(feature.geometry) > 2
                            for feature in features))

    def test_rectangular_building_lot_keeps_its_declared_dimensions(self):
        features = generate(
            GridProjection(),
            TownParameters.from_dict({"seed": 4, "park_chance": 0.0}))
        building = next(f for f in features if "building" in f.tags)
        points = [(lon, lat) for lat, lon in building.geometry]
        placed, reason = place(points, np.zeros((720, 720), dtype=bool),
                               alignment="real")
        self.assertIsNotNone(placed, reason)
        area = Polygon(points).area
        self.assertGreater(placed.tiles, area * 0.8)
        self.assertLess(placed.tiles, area * 1.2)

    def test_invalid_parameters_are_rejected(self):
        for data in ({"block_size_m": 0}, {"park_chance": 0.7},
                     {"building_density": float("nan")}, {"seed": -1},
                     {"street_pattern": "radial"},
                     {"town_type": "unknown"},
                     {"road_irregularity": 0.5},
                     {"max_building_side": 201},
                     {"max_building_side": 12.5}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                TownParameters.from_dict(data)


if __name__ == "__main__":
    unittest.main()
