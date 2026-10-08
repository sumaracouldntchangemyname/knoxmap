"""Discover and select exact-footprint BuildingEd lots from Building Pool V3.

Workshop lots are copied as-is: they are only candidates when their declared
dimensions match the footprint KnoxMap is placing. This avoids silently
cropping or stretching a finished building.
"""
from __future__ import annotations

import hashlib
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import knoxpaths


WORKSHOP_ID = "2790726238"
POOL_NAME = "Building Pool V3"


@dataclass(frozen=True)
class BuildingTemplate:
    path: str
    width: int
    height: int
    family: str
    levels: int
    rooms: int
    furniture: int
    digest: str


class TemplateCatalog:
    def __init__(self, root: Path, templates: list[BuildingTemplate],
                 invalid: int = 0):
        self.root = root
        self.templates = templates
        self.invalid = invalid

    def for_family(self, family: str) -> list[BuildingTemplate]:
        return [template for template in self.templates
                if template.family == family]

    def dimensions(self, family: str, max_width: int,
                   max_height: int) -> list[tuple[int, int]]:
        return sorted({(t.width, t.height) for t in self.for_family(family)
                       if t.width <= max_width and t.height <= max_height})

    def choose(self, kind: str | None, width: int, height: int,
               seed: int) -> BuildingTemplate | None:
        family = family_for_kind(kind)
        if family is None:
            return None
        candidates = [t for t in self.templates
                      if t.family == family and
                      (t.width, t.height) == (width, height)]
        if not candidates:
            return None
        candidates.sort(key=lambda t: t.path.casefold())
        return random.Random(seed).choice(candidates)


def family_for_kind(kind: str | None) -> str | None:
    kind = (kind or "house").lower()
    if kind in {"house", "apartment"}:
        return "residential"
    if kind in {"shop", "restaurant", "mall", "hotel"}:
        return "commercial"
    if kind in {"industrial", "barn", "garage", "military"}:
        return "industrial"
    if kind in {"civic", "school", "church", "medical", "police",
                "fire", "library", "castle", "stadium"}:
        return "civic"
    return None


def _pool_root(path: str | Path) -> Path | None:
    root = Path(path).expanduser()
    if not root.is_dir():
        return None
    if any(root.rglob("*.tbx")):
        return root.resolve()
    return None


def configured_pool_path() -> str:
    configured = knoxpaths.load_config().get("building_pool_path")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    candidates = []
    for library in knoxpaths._steam_libraries():
        workshop_content = (library / "steamapps" / "workshop" / "content"
                            / "108600")
        candidates.append(workshop_content / POOL_NAME)
        candidates.append(library / "steamapps" / "workshop" / "content"
                          / "108600" / WORKSHOP_ID / "mods" / POOL_NAME)
    candidates.append(knoxpaths.BASE_DIR / "assets" / "workshop" / POOL_NAME)
    candidates.append(knoxpaths.BASE_DIR / POOL_NAME)
    for candidate in candidates:
        if _pool_root(candidate):
            return str(candidate)
    return ""


def load_catalog(path: str | Path | None = None) -> TemplateCatalog:
    selected = str(path).strip() if path is not None else configured_pool_path()
    root = _pool_root(selected) if selected else None
    if root is None:
        raise FileNotFoundError(
            "Building Pool V3 was not found. Set its folder under "
            "Settings > Building Pool V3.")

    templates = []
    invalid = 0
    for file in sorted(root.rglob("*.tbx"), key=lambda p: str(p).casefold()):
        try:
            content = file.read_bytes()
            element = ET.fromstring(content)
            if element.tag != "building":
                raise ValueError("root element is not <building>")
            width, height = int(element.attrib["width"]), int(element.attrib["height"])
            if not (1 <= width <= 300 and 1 <= height <= 300):
                raise ValueError("lot dimensions are outside 1..300")
            family = _family_for_path(file.relative_to(root))
            if family is None:
                continue
            floors = element.findall("floor")
            levels = max(1, len(floors))
            rooms = len(element.findall("room"))
            furniture = sum(
                1 for floor in floors for obj in floor.findall(".//object")
                if obj.get("type") == "furniture")
            templates.append(BuildingTemplate(
                str(file.resolve()), width, height, family, levels, rooms,
                furniture, hashlib.sha1(content).hexdigest()))
        except (OSError, ET.ParseError, KeyError, ValueError):
            invalid += 1
    if not templates:
        raise ValueError(f"No usable Building Pool V3 lots found in {root}.")
    return TemplateCatalog(root, templates, invalid)


def _family_for_path(relative: Path) -> str:
    parts = [part.casefold() for part in relative.parts]
    stem = relative.stem.casefold()
    words = " ".join((*parts, stem))
    directory_families = {
        "apartments": "residential", "residential": "residential",
        "automotive": "industrial", "construction": "industrial",
        "misc industry": "industrial", "military": "industrial",
        "warehouse": "industrial", "business": "commercial",
        "merchandise": "commercial", "resturant": "commercial",
        "education": "civic", "entertainment": "civic", "fire": "civic",
        "hospital": "civic", "outdoor city spaces": "civic",
        "police": "civic", "recreational": "civic", "religon": "civic",
        "special": "civic", "city buildings": "civic",
    }
    if parts[0] in directory_families:
        return directory_families[parts[0]]
    if any(word in words for word in
            ("industrial", "industry", "automotive", "factory", "warehouse",
             "garage", "barn", "construction", "military", "mechanic",
             "workshop")):
        return "industrial"
    if any(word in words for word in
            ("retail", "commercial", "restaurant", "resturant",
             "entertainment", "hospitality", "business", "shop", "store",
             "mall", "diner", "merchandise")):
        return "commercial"
    if any(word in words for word in
            ("community services", "medical", "education", "school",
            "police", "fire", "hospital", "religon", "religion", "church",
            "city buildings", "recreational", "outdoor city spaces",
             "special", "office", "civic")):
        return "civic"
    if any(word in words for word in
            ("residential", "apartment", "house", "home", "cabin", "trailer",
             "townhouse", "farmhouse", "suburb")):
        return "residential"
    # The curated catalogue is predominantly homes, but its filenames often
    # carry a more specific type handled above. Every remaining valid BuildingEd
    # lot is still usable as a town building rather than silently discarded.
    if parts and parts[0] == "buildings catalogue curated":
        return "residential"
    return "civic"


def copy_template(template: BuildingTemplate, destination: str) -> None:
    shutil.copyfile(template.path, destination)
