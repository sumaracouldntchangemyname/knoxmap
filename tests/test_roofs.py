import unittest
from xml.etree import ElementTree

from knoxbuild import catalog, layout, tbx


class RoofLayouts(unittest.TestCase):
    def test_three_tile_house_roof_uses_supported_pitched_tiles(self):
        pieces = tbx._roof_pieces(layout.roof_rects([[1] * 3 for _ in range(3)]),
                                  peaked=True, roof30=True)

        self.assertTrue(any(kind.startswith("Peak30") for kind, *_rest in pieces))

    def test_small_styled_house_exports_a_pitched_roof_object(self):
        building = layout.build_building(3, 3, seed=1, kind="house")
        root = ElementTree.fromstring(
            tbx.render_tbx(building, "small-house-roof", catalog.HOUSE_STYLES[0]))

        roofs = root.findall('.//object[@type="roof"]')
        self.assertTrue(any(obj.get("RoofType", "").startswith("Peak") for obj in roofs))

    def test_roof_rectangles_cover_an_irregular_footprint_once(self):
        grid = [[1] * 8 + [0] * 4 for _ in range(4)]
        grid.extend([[1] * 12 for _ in range(8)])
        rects = layout.roof_rects(grid)

        for peaked, roof30 in ((False, False), (True, False), (True, True)):
            with self.subTest(peaked=peaked, roof30=roof30):
                covered = [[0] * 12 for _ in range(12)]
                for _kind, _depth, _caps, (x0, y0, width, height) in tbx._roof_pieces(
                        rects, peaked, roof30):
                    for y in range(y0, y0 + height):
                        for x in range(x0, x0 + width):
                            covered[y][x] += 1
                self.assertEqual(covered, grid)

    def test_flat_roof_equipment_varies_by_building_without_overlapping(self):
        grid = [[1] * 10 for _ in range(10)]
        first = tbx._rooftop(grid, 10, 10, "house-a")
        second = tbx._rooftop(grid, 10, 10, "house-b")

        self.assertNotEqual(first, second)
        for placement in (first, second):
            for index, (_role, x, y, _orientation) in enumerate(placement):
                for _other, other_x, other_y, _other_orientation in placement[index + 1:]:
                    self.assertTrue(abs(x - other_x) > 1 or abs(y - other_y) > 1)


if __name__ == "__main__":
    unittest.main()