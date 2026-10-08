import numpy as np
import pytest

from flycraft.game import render
from flycraft.game.render import ENEMY, NEUTRAL, SELF, Blob
from flycraft.game.tasks import TASKS, nearest

BEACON, ROACHES = TASKS["MoveToBeacon"], TASKS["DefeatRoaches"]


def _layer(*squares):
  layer = np.zeros((84, 84), np.int32)
  for x0, y0, size, value in squares:
    layer[y0:y0 + size, x0:x0 + size] = value
  return layer


def test_the_tasks():
  assert (BEACON.target, BEACON.order, BEACON.split, BEACON.cone_deg) == (NEUTRAL, "move", False, 0)
  assert (ROACHES.target, ROACHES.order, ROACHES.split, ROACHES.cone_deg) == (
    ENEMY, "attack", True, 30.0)


def test_a_beacon_is_one_target_even_split_in_two():
  layer = _layer((40, 40, 5, NEUTRAL))
  layer[40:45, 42] = SELF  # a marine cuts it in two
  assert len(render.blobs(layer, NEUTRAL)) == 2
  assert BEACON.targets(layer) == [render.find(layer, NEUTRAL)]
  assert BEACON.targets(_layer((40, 40, 4, SELF))) == []


def test_each_roach_is_a_target():
  layer = _layer((10, 10, 3, ENEMY), (50, 20, 3, ENEMY), (30, 30, 3, NEUTRAL))
  assert ROACHES.targets(layer) == [Blob((11.0, 11.0), 9), Blob((51.0, 21.0), 9)]


def test_the_scripted_agents_aim_like_pysc2s():
  layer = _layer((40, 40, 4, NEUTRAL), (10, 60, 3, ENEMY), (50, 61, 3, ENEMY))
  assert BEACON.aim(layer) == (41.5, 41.5)
  assert ROACHES.aim(layer) == (50.0, 63.0)  # the lowest row, its first pixel
  assert BEACON.aim(np.zeros((84, 84), np.int32)) is None
  assert ROACHES.aim(np.zeros((84, 84), np.int32)) is None


@pytest.mark.parametrize("xy, expected", [((12.0, 12.0), 0), ((60.0, 30.0), 1), (None, 0)])
def test_nearest(xy, expected):
  found = [Blob((10.0, 10.0), 9), Blob((50.0, 20.0), 9)]
  assert nearest(xy, found) == found[expected]
  assert nearest(xy, []) is None


# The squad strikes the roach the fly faces (spec 15: the object nearest the center of the
# frontal visual field). Squad at (40, 40); heading 0 is screen right, positive is clockwise.

def _strike(task, layer, heading):
  return task.strike(layer, task.targets(layer), (40.0, 40.0), heading)


def test_the_strike_is_the_roach_nearest_the_heading_not_the_nearest_roach():
  layer = _layer((70, 39, 3, ENEMY), (44, 49, 3, ENEMY))  # far ahead; near, 60 deg right
  assert _strike(ROACHES, layer, 0.0) == (71.0, 40.0)
  assert _strike(ROACHES, layer, 50.0) == (45.0, 50.0)


@pytest.mark.parametrize("heading, struck", [(29.0, True), (-29.0, True), (31.0, False),
                                             (-31.0, False)])
def test_only_a_roach_inside_the_cone_is_struck(heading, struck):
  layer = _layer((59, 39, 3, ENEMY))  # dead ahead of heading 0
  assert _strike(ROACHES, layer, heading) == ((60.0, 40.0) if struck else None)


def test_the_strike_lands_on_the_roach_even_when_its_middle_does_not():
  layer = _layer()
  layer[30:42, 60:62] = ENEMY  # an L: two touching roaches make one patch, middle (64, 38)
  layer[40:42, 62:74] = ENEMY
  assert ROACHES.targets(layer) == [Blob((64.0, 38.0), 48)]
  assert layer[38, 64] != ENEMY
  assert _strike(ROACHES, layer, 0.0) == (64.0, 40.0)  # the patch's pixel nearest its middle


def test_nothing_to_strike():
  assert _strike(ROACHES, _layer((40, 40, 2, SELF)), 0.0) is None
  layer = _layer((59, 39, 3, NEUTRAL))
  assert _strike(BEACON, layer, 0.0) is None  # the beacon is walked to, never struck
  assert ROACHES.strike(_layer((59, 39, 3, ENEMY)), [Blob((60.0, 40.0), 9)], None, 0.0) is None
