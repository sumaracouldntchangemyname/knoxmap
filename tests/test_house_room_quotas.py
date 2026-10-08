import unittest
from collections import Counter

from knoxbuild import layout
from knoxbuild.layout import build_building


class HouseRoomQuotas(unittest.TestCase):
    def test_cooking_rooms_prefer_adjacency_to_dining_rooms(self):
        for kitchen, dining in (("kitchen", "diningroom"),
                                ("restaurantkitchen", "restaurantdining")):
            with self.subTest(kitchen=kitchen, dining=dining):
                self.assertEqual(layout._door_cost(kitchen, dining), 1.5)

    def test_large_house_has_a_bounded_whole_building_room_count(self):
        for seed in range(3):
            with self.subTest(seed=seed):
                building = build_building(28, 24, levels=3, seed=seed, kind="house")
                counts = Counter(room.kind for room in building.rooms)
                bedrooms = counts["bedroom"] + counts["kidsbedroom"]
                self.assertLessEqual(bedrooms, 5)
                self.assertLessEqual(counts["bathroom"], 2)
                self.assertGreater(bedrooms, 0)

    def test_single_storey_house_has_at_most_two_bedrooms_and_one_bath(self):
        building = build_building(14, 12, seed=19, kind="house")
        counts = Counter(room.kind for room in building.rooms)

        self.assertLessEqual(counts["bedroom"] + counts["kidsbedroom"], 2)
        self.assertLessEqual(counts["bathroom"], 1)
        self.assertEqual(counts["bathroom"], 1)

    def test_upper_bathroom_stacks_over_lower_floor_plumbing(self):
        for seed in (4, 7, 19):
            with self.subTest(seed=seed):
                building = build_building(28, 24, levels=2, seed=seed, kind="house")
                lower, upper = building.storeys
                for storey in building.storeys:
                    adjacency = layout._neighbours(storey)
                    for index, room in enumerate(storey.rooms, 1):
                        if room.kind != "bathroom":
                            continue
                        neighbors = adjacency[index]
                        self.assertTrue(any(
                            storey.rooms[neighbor - 1].kind in {"bedroom", "kidsbedroom"}
                            for neighbor in neighbors))
                bathrooms = [room for room in upper.rooms if room.kind == "bathroom"]
                wet_rooms = [room for room in lower.rooms
                             if room.kind in {"bathroom", "kitchen", "laundry"}]
                self.assertEqual(len(bathrooms), 1)
                self.assertTrue(any(
                    max(room.x0, wet.x0) <= min(room.x1, wet.x1)
                    and max(room.y0, wet.y0) <= min(room.y1, wet.y1)
                    for room in bathrooms for wet in wet_rooms))


if __name__ == "__main__":
    unittest.main()
