import unittest

import numpy as np

from generator import pz_colors as C
from knoxbuild.yards import _pave_patio, _yard_reservation


class YardPlacement(unittest.TestCase):
    def test_backyard_reservation_leaves_a_gap_between_fences(self):
        reserved = _yard_reservation(lambda u, v: (u, v), 4, 5)

        self.assertIn((4, 2), reserved)
        self.assertNotIn((5, 2), reserved)

    def test_patio_paving_preserves_tiles_claimed_by_neighbor_paths(self):
        ground = np.full((3, 4, 3), C.MEDIUM_GRASS, dtype=np.uint8)
        vegetation = np.full((3, 4, 3), C.TREES, dtype=np.uint8)
        crossable = np.ones((3, 4), dtype=bool)
        claimed = np.zeros((3, 4), dtype=bool)
        claimed[1, 1] = True

        paved = _pave_patio(ground, vegetation, crossable, claimed,
                            [(1, 1), (2, 1), (-1, 1)])

        self.assertEqual(paved, [(2, 1)])
        self.assertEqual(tuple(ground[1, 1]), C.MEDIUM_GRASS)
        self.assertEqual(tuple(ground[1, 2]), C.PAVING_STONE)
        self.assertEqual(tuple(vegetation[1, 1]), C.TREES)
        self.assertEqual(tuple(vegetation[1, 2]), C.VEG_NOTHING)


if __name__ == "__main__":
    unittest.main()