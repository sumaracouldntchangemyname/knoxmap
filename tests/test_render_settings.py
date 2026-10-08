"""RENDER_SETTINGS has to stay level with what generate() really reads.

The page runs only the steps a changed setting needs: one of these and the map
is drawn again, anything else and the buildings are laid out afresh on the
ground that is already there. If a knob is added to the renderer and not to
this list, changing it would appear to do nothing at all - the map would never
be redrawn - which is the kind of bug nobody reports because it looks like the
setting is just weak.
"""
import inspect
import re
import unittest

import app
from knoxbuild.settings import Settings


# Read by generate() and deliberately not in the list: the building step
# rebuilds the spawn map, while the pool setting is reapplied to existing
# footprints during that step (and informs lot sizes only for new procedural
# layouts).
REDRAWN_BY_THE_BUILD = {"spawn_density", "use_building_pool"}


def settings_read_by(source) -> set:
    """The setting names an object's source reads off `settings`.

    Narrowed to the real fields, or `settings.to_dict()` counts as a knob.
    """
    if not isinstance(source, str):
        source = inspect.getsource(source)
    return {n for n in re.findall(r"settings\.(\w+)", source)
            if n in Settings().to_dict()}


class RenderSettings(unittest.TestCase):
    def test_every_one_of_them_is_a_real_setting(self):
        known = set(Settings().to_dict())
        self.assertEqual([k for k in app.RENDER_SETTINGS if k not in known], [])

    def test_no_duplicates(self):
        self.assertEqual(len(app.RENDER_SETTINGS), len(set(app.RENDER_SETTINGS)))

    def test_generate_reads_all_of_them(self):
        read = settings_read_by(app.generate)
        self.assertEqual([k for k in app.RENDER_SETTINGS if k not in read], [],
                         "listed as needing a redraw but generate() never reads it")

    def test_nothing_generate_reads_is_left_out(self):
        read = settings_read_by(app.generate)
        missed = read - set(app.RENDER_SETTINGS) - REDRAWN_BY_THE_BUILD
        self.assertEqual(sorted(missed), [],
                         "generate() reads these, so changing one has to redraw "
                         "the map - add them to RENDER_SETTINGS, or to "
                         "REDRAWN_BY_THE_BUILD with the reason")

    def test_the_exception_is_still_read_where_it_says(self):
        # If the build stops redrawing the spawn map, the excuse above is gone.
        from knoxbuild import population
        self.assertIn("spawn_density",
                      settings_read_by(inspect.getsource(population)))


if __name__ == "__main__":
    unittest.main()
