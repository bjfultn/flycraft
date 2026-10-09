# Does the minimap show enemies the screen misses? (2026-10-08)

The question: should the minimap feed the fly's eye, so it sees enemies that are off the
screen? Only if enemies ever are off the screen. [minimap_probe.py](minimap_probe.py) counts,
every step, the enemy units in pysc2's `raw_units` (all of them) and in `feature_units` (only
those on the screen). It also reads the camera rectangle off the minimap and counts minimap
enemy pixels outside it. DefeatRoaches (marines) and DefeatMarines (zerglings), seed 1, two
episodes each under two pilots: `wander` moves the squad to random screen points, `focus`
attacks the lowest enemy pixel.

Result ([minimap_probe.jsonl](minimap_probe.jsonl)): no enemy was ever off the screen, in
1,046 steps over 8 episodes. The camera never moved (minimap x 21 to 44, y 27 to 50 at 64 px),
and every enemy unit counted by `raw_units` was on the screen in every step.

The pathable area on the minimap is larger than the camera rectangle (x 11 to 53, y 20 to 54),
but nothing gets there: enemies spawn in view, and the body only orders moves to points on the
screen. So on these maps the minimap adds nothing to what the eye already sees. It will matter
on a full-size map, where the screen shows a small part of the map.
