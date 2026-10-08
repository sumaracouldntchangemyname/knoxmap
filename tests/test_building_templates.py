import random
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import knoxpaths
from generator.procedural_town import _building_size
from knoxbuild.build import _make_one
from knoxbuild.settings import Settings
from knoxbuild.templates import (BuildingTemplate, TemplateCatalog,
                                 _family_for_path, configured_pool_path,
                                 copy_template, load_catalog)


def make_lot(path: Path, width: int, height: int, furniture: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = ('<building version="2" width="%d" height="%d">'
                '<room/><floor/><floor/></building>'
                % (width, height) if furniture else
                '<building version="2" width="%d" height="%d"><floor/>'
                '</building>' % (width, height))
    path.write_text(contents, encoding="utf-8")


class BuildingTemplateTests(unittest.TestCase):
    def test_pool_is_discovered_in_local_workshop_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            pool = base / "assets" / "workshop" / "Building Pool V3"
            make_lot(pool / "Residential" / "house.tbx", 8, 8)
            with patch.object(knoxpaths, "BASE_DIR", base), \
                    patch.object(knoxpaths, "_steam_libraries", return_value=[]), \
                    patch.object(knoxpaths, "load_config", return_value={}):
                self.assertEqual(Path(configured_pool_path()), pool.resolve())

    def test_pool_is_discovered_beside_steam_workshop_mods(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory)
            pool = (library / "steamapps" / "workshop" / "content" / "108600"
                    / "Building Pool V3")
            make_lot(pool / "Residential" / "house.tbx", 8, 8)
            with patch.object(knoxpaths, "BASE_DIR", library / "KnoxMap"), \
                    patch.object(knoxpaths, "_steam_libraries", return_value=[library]), \
                    patch.object(knoxpaths, "load_config", return_value={}):
                self.assertEqual(Path(configured_pool_path()), pool.resolve())

    def test_catalog_selects_only_matching_compatible_lots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wanted = root / "Buildings Catalogue Curated" / "Muldraugh" / \
                "Residential" / "lot_house.tbx"
            other_size = root / "Buildings Catalogue Curated" / "Muldraugh" / \
                "Residential" / "lot_large.tbx"
            unrelated = root / "Buildings Catalogue Curated" / "Muldraugh" / \
                "Industrial" / "lot_factory.tbx"
            make_lot(wanted, 12, 14, furniture=True)
            make_lot(other_size, 20, 20)
            make_lot(unrelated, 12, 14)

            catalog = load_catalog(root)
            selected = catalog.choose("house", 12, 14, 5)
            self.assertIsNotNone(selected)
            self.assertEqual(Path(selected.path), wanted.resolve())
            self.assertEqual((selected.width, selected.height), (12, 14))
            self.assertEqual(selected.levels, 2)
            self.assertIsNone(catalog.choose("house", 12, 12, 5))
            self.assertIsNone(catalog.choose("house", 14, 12, 5))
            self.assertEqual(catalog.choose("industrial", 12, 14, 5).family,
                             "industrial")

    def test_copy_preserves_template_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "Building Pool V3" / "Residential" / "house.tbx"
            make_lot(source, 10, 10, furniture=True)
            catalog = load_catalog(source.parents[1])
            template = catalog.choose("house", 10, 10, 1)
            output = Path(directory) / "map" / "buildings" / "house.tbx"
            output.parent.mkdir(parents=True)
            copy_template(template, str(output))
            self.assertEqual(output.read_bytes(), source.read_bytes())
            self.assertEqual(source.read_bytes(), output.read_bytes())

    def test_dimensions_are_available_for_procedural_layout(self):
        catalog = TemplateCatalog(Path("."), [
            BuildingTemplate("", 12, 14, "residential", 1, 2, 3, ""),
            BuildingTemplate("", 40, 40, "residential", 1, 2, 3, ""),
        ])
        self.assertEqual(catalog.dimensions("residential", 20, 20), [(12, 14)])
        self.assertEqual(
            _building_size("residential", 30, 30, random.Random(1), catalog),
            (12, 14))

    def test_workshop_categories_are_all_kept_usable(self):
        folders = {
            "Apartments": "residential",
            "Automotive": "industrial",
            "Business": "commercial",
            "City Buildings": "civic",
            "Construction": "industrial",
            "Education": "civic",
            "Entertainment": "civic",
            "Fire": "civic",
            "Hospital": "civic",
            "Merchandise": "commercial",
            "Military": "industrial",
            "Misc Industry": "industrial",
            "Outdoor City Spaces": "civic",
            "Police": "civic",
            "Recreational": "civic",
            "Religon": "civic",
            "Residential": "residential",
            "Resturant": "commercial",
            "Special": "civic",
            "Warehouse": "industrial",
        }
        for folder, family in folders.items():
            with self.subTest(folder=folder):
                self.assertEqual(_family_for_path(Path(folder) / "lot.tbx"),
                                 family)
        self.assertEqual(_family_for_path(
            Path("Buildings Catalogue Curated/Muldraugh/House.tbx")),
            "residential")
        self.assertEqual(_family_for_path(Path("Unexpected/lot.tbx")), "civic")

    def test_build_worker_uses_a_matching_tbx_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Building Pool V3"
            source = root / "Residential" / "house.tbx"
            make_lot(source, 10, 10, furniture=True)
            template = load_catalog(root).choose("house", 10, 10, 2)
            destination = Path(directory) / "map" / "buildings" / "lot.tbx"
            destination.parent.mkdir(parents=True)
            job = (10, 10, 1, False, 1, "house", None, Settings(), None,
                   "house", str(destination), None, False, [], False, [],
                   None, [], False, {}, template)
            result = _make_one(job)
            self.assertEqual(result, (2, 1, 0, None, []))
            self.assertEqual(destination.read_bytes(), source.read_bytes())


if __name__ == "__main__":
    unittest.main()
