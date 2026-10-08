import unittest
from types import SimpleNamespace

from generator import osm
from generator.procedural_town import (TownParameters, _line_near_rect,
                                       generate)
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
        self.assertEqual(
            (placed.width, placed.height),
            (round(max(p[0] for p in points) - min(p[0] for p in points)),
             round(max(p[1] for p in points) - min(p[1] for p in points))))

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
