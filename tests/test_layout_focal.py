import unittest
import random

from knoxbuild import layout


def center_furniture(room_kind, building_kind):
    width = height = 16
    room = layout.Room(0, 0, width - 1, height - 1, kind=room_kind)
    plan = layout.Plan(width, height, rooms=[room],
                       grid=[[1] * width for _ in range(height)], kind=building_kind)
    placed = layout._furnish_middle(plan, 1, room, set(), set())
    if not placed:
        raise AssertionError(f"no center group for {building_kind}/{room_kind}")
    return [role for role, *_ in plan.furniture]


class FocalRooms(unittest.TestCase):
    def test_lounge_seating_faces_the_tv(self):
        for room_kind, building_kind in (("livingroom", "house"),
                                         ("motelroom", "apartment")):
            with self.subTest(room_kind=room_kind):
                room = layout.Room(0, 0, 15, 15, kind=room_kind)
                plan = layout.Plan(16, 16, rooms=[room],
                                   grid=[[1] * 16 for _ in range(16)],
                                   kind=building_kind)
                layout._furnish(plan, random.Random(3))
                tv_x, tv_y = next((x, y) for role, x, y, _facing in plan.furniture
                                  if role == "tv")
                armchairs = [(x, y) for role, x, y, _facing in plan.furniture
                             if role == "armchair"]
                if room_kind == "livingroom":
                    self.assertTrue(any(x == tv_x and y < tv_y for x, y in armchairs))
                else:
                    self.assertTrue(any(x == tv_x and y > tv_y for x, y in armchairs))
                for role, x, y, facing in plan.furniture:
                    if role not in {"sofa", "armchair"}:
                        continue
                    cells = layout._cells_for(role, x, y, facing)
                    seat_x = sum(cx for cx, _cy in cells) / len(cells)
                    seat_y = sum(cy for _cx, cy in cells) / len(cells)
                    front = {"N": (0, 1), "S": (0, -1),
                             "W": (1, 0), "E": (-1, 0)}[facing]
                    self.assertGreater((tv_x - seat_x) * front[0]
                                       + (tv_y - seat_y) * front[1], 0)

    def test_motel_room_centres_bed_and_tv(self):
        roles = center_furniture("motelroom", "apartment")
        self.assertIn("double_bed", roles)
        self.assertIn("tv", roles)

    def test_church_has_altar_table_and_congregation(self):
        roles = center_furniture("church", "church")
        self.assertEqual(roles.count("table"), 1)
        self.assertGreaterEqual(roles.count("chair"), 5)

    def test_civic_lobby_centres_reception_counter(self):
        for building_kind in layout.CIVIC_KINDS | {"apartment"}:
            with self.subTest(building_kind=building_kind):
                roles = center_furniture("lobby", building_kind)
                self.assertEqual(roles.count("shop_counter"), 1)
                self.assertEqual(roles.count("chair"), 3)

    def test_bedrooms_center_bed_and_wardrobe(self):
        roles = center_furniture("bedroom", "house")
        self.assertIn("double_bed", roles)
        self.assertIn("wardrobe", roles)

    def test_medical_rooms_center_beds_and_seating(self):
        for room_kind in ("medical", "clinic", "medicaloffice", "dentist"):
            with self.subTest(room_kind=room_kind):
                roles = center_furniture(room_kind, "medical")
                self.assertIn("bed", roles)
                self.assertIn("sidetable", roles)
                self.assertIn("chair", roles)

    def test_warehouse_and_workshop_center_storage_groups(self):
        for room_kind in ("warehouse", "workshop"):
            with self.subTest(room_kind=room_kind):
                roles = center_furniture(room_kind, "industrial")
                self.assertIn("metal_rack", roles)
                self.assertIn("crate", roles)

    def test_hotel_reception_is_on_ground_floor(self):
        ground = layout.build_plan(32, 24, seed=19, kind="apartment", hotel=True)
        lobbies = [i for i, room in enumerate(ground.rooms, start=1)
                   if room.kind == "lobby"]
        self.assertEqual(len(lobbies), 1)
        self.assertTrue(any(role == "shop_counter"
                            and layout._room_at(ground, x, y) == lobbies[0]
                            for role, x, y, _facing in ground.furniture))

        upper = layout.build_plan(32, 24, seed=19, kind="apartment", hotel=True,
                                  ground=False)
        self.assertFalse(any(room.kind == "lobby" for room in upper.rooms))

    def test_classrooms_have_whiteboard_facing_seats(self):
        plan = layout.build_plan(32, 24, seed=7, kind="school")
        checked = 0
        for room_id, room in enumerate(plan.rooms, start=1):
            if room.kind != "classroom":
                continue
            room_items = [(role, x, y, facing)
                          for role, x, y, facing in plan.furniture
                          if layout._room_at(plan, x, y) == room_id]
            boards = [(x, y, facing) for role, x, y, facing in room_items
                      if role == "whiteboard"]
            self.assertEqual(len(boards), 1)
            board_facing = boards[0][2]
            seat_facing = {"N": "S", "W": "E"}[board_facing]
            self.assertTrue(any(role == "chair" and facing == seat_facing
                                for role, _x, _y, facing in room_items))
            checked += 1
        self.assertGreater(checked, 0)

    def test_mapped_main_entrance_uses_its_wall(self):
        width = height = 12
        room = layout.Room(0, 0, width - 1, height - 1, kind="lobby")
        plan = layout.Plan(width, height, rooms=[room],
                           grid=[[1] * width for _ in range(height)], kind="medical")
        layout._exterior_door(
            plan, random.Random(1),
            entrances=[(6.5, 12.0, {"entrance": "main"})])
        self.assertEqual(plan.doors, [(6, 12, "N")])

    def test_service_and_ground_main_nodes_create_separate_doors(self):
        width = height = 12
        room = layout.Room(0, 0, width - 1, height - 1, kind="hall")
        plan = layout.Plan(width, height, rooms=[room],
                           grid=[[1] * width for _ in range(height)], kind="hospital")
        layout._exterior_door(
            plan, random.Random(1), entrances=[
                (6.5, 12.0, {"entrance": "main", "level": "0"}),
                (7.5, 12.0, {"entrance": "yes", "level": "0"}),
                (0.0, 6.5, {"entrance": "service", "level": "0;1"}),
                (6.5, 0.0, {"entrance": "exit", "level": "1"}),
            ])
        self.assertEqual(set(plan.doors), {(6, 12, "N"), (7, 12, "N"),
                                           (0, 6, "W")})

    def test_tagged_garage_gets_a_wide_door_on_its_street_wall(self):
        storey = layout.build_building(12, 8, kind="shed", street="S",
                                       garage_door=True, seed=31).storeys[0]
        street_runs = [wall for room_id in range(1, len(storey.rooms) + 1)
                       for side, wall in layout._outside_runs(storey, room_id)
                       if side == "S"]

        self.assertTrue(any(sum(edge in storey.doors for edge in wall) >= 3
                            for wall in street_runs))


if __name__ == "__main__":
    unittest.main()