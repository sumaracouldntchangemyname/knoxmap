import unittest
from unittest import mock
from xml.etree import ElementTree

from knoxbuild import catalog
from knoxbuild import layout
from knoxbuild.tbx import render_tbx


class WindowWallStyles(unittest.TestCase):
    REQUIRED = {f"{side}Window{index}"
                for side in ("West", "North")
                for index in range(1, 20)}

    def test_every_interior_style_has_all_window_cutouts(self):
        for entry in catalog.INTERIOR_WALLS:
            with self.subTest(wall=entry["tiles"].get("West")):
                self.assertTrue(self.REQUIRED <= entry["tiles"].keys())

    def test_export_completes_special_style_wall_cutouts(self):
        room = layout.Room(0, 0, 3, 3, kind="hall")
        plan = layout.Plan(4, 4, rooms=[room], grid=[[1] * 4 for _ in range(4)],
                           windows=[(1, 0, "N")])
        style = catalog.SPECIAL_STYLES["school"]

        root = ElementTree.fromstring(render_tbx(plan, "window-style-test", style))

        entries = [entry for entry in root.findall("tile_entry")
                   if entry.get("category") in ("exterior_walls", "interior_walls")]
        self.assertTrue(entries)
        for entry in entries:
            tile_names = {tile.get("enum") for tile in entry.findall("tile")}
            with self.subTest(category=entry.get("category")):
                self.assertTrue(self.REQUIRED <= tile_names)

    def test_occupied_public_rooms_get_a_window_when_facade_bays_miss(self):
        for kind in ("medicaloffice", "gym", "aesthetic"):
            with self.subTest(kind=kind):
                width = height = 12
                room = layout.Room(0, 0, width - 1, height - 1, kind=kind)
                plan = layout.Plan(width, height, rooms=[room],
                                   grid=[[1] * width for _ in range(height)])
                building = layout.Building(width, height, storeys=[plan])
                with mock.patch.object(layout, "_bays", return_value=[]):
                    layout._place_windows(building, "medical")
                self.assertTrue(plan.windows)

    def test_small_house_facades_get_a_front_window_when_one_fits(self):
        for seed in (37, 45, 53, 65, 72):
            with self.subTest(seed=seed):
                building = layout.build_building(12 + seed % 7, 10 + seed % 5,
                                                 seed=seed, kind="house")
                storey = building.storeys[0]
                front = layout._front_side(building, "house")
                front_edges = {
                    edge
                    for room_id in range(1, len(storey.rooms) + 1)
                    for side, wall in layout._outside_runs(storey, room_id)
                    if side in front
                    for edge in wall
                }
                self.assertTrue(any(edge in front_edges for edge in storey.windows))


if __name__ == "__main__":
    unittest.main()
