import math

import numpy as np
import pytest

from flycraft import eye
from flycraft.game import render
from flycraft.game.render import ENEMY, NEUTRAL, SELF, Blob


def _layer(*squares):
  """An 84x84 player_relative layer with filled squares (x0, y0, size, value)."""
  layer = np.zeros((84, 84), np.int32)
  for x0, y0, size, value in squares:
    layer[y0:y0 + size, x0:x0 + size] = value
  return layer


def test_find_reads_x_from_columns_and_y_from_rows():
  layer = _layer((10, 50, 3, SELF), (60, 20, 4, NEUTRAL))
  marine, beacon = render.find(layer, SELF), render.find(layer, NEUTRAL)
  assert marine == Blob((11.0, 51.0), 9)
  assert beacon == Blob((61.5, 21.5), 16)
  assert beacon.radius_px == pytest.approx(math.sqrt(16 / math.pi))


def test_find_returns_none_when_absent():
  assert render.find(_layer((10, 10, 3, SELF)), NEUTRAL) is None


@pytest.mark.parametrize("target, heading, az", [
  ((50.0, 40.0), 0.0, 0.0),  # dead ahead, heading screen-right
  ((40.0, 50.0), 0.0, 90.0),  # screen-down is clockwise of screen-right: the fly's right
  ((40.0, 30.0), 0.0, -90.0),
  ((30.0, 40.0), 0.0, 180.0),  # straight behind wraps to +180, never -180
  ((40.0, 30.0), -90.0, 0.0),  # heading screen-up
  ((50.0, 40.0), -90.0, 90.0),
  ((50.0, 50.0), 170.0, -125.0),  # bearing 45, heading 170: wraps across the seam
])
def test_egocentric_azimuth(target, heading, az):
  got_az, d = render.egocentric((40.0, 40.0), target, heading)
  assert got_az == pytest.approx(az)
  assert d == pytest.approx(math.dist((40.0, 40.0), target))


def test_angular_radius_looms_and_clips():
  assert render.angular_radius(3.0, 3.0) == pytest.approx(45.0)
  assert render.angular_radius(3.0, 30.0) == pytest.approx(math.degrees(math.atan(0.1)))
  assert render.angular_radius(3.0, 300.0) == render.MIN_RADIUS_DEG
  assert render.angular_radius(3.0, 1.0) == render.MAX_RADIUS_DEG
  assert render.angular_radius(3.0, 0.0) == render.MAX_RADIUS_DEG  # standing on it


def test_render_eye_draws_the_beacon_disk():
  beacon = Blob((40.0, 50.0), 28)  # 10 px below a marine heading screen-right: az +90
  img = render.render_eye((40.0, 40.0), beacon, 0.0, "dark_on_bright")
  r = render.angular_radius(beacon.radius_px, 10.0)
  np.testing.assert_array_equal(img, eye.disk(90.0, r, "dark_on_bright"))
  assert img.dtype == np.uint8 and img.shape == (eye.EYE_ROWS, eye.EYE_COLS)


def test_render_eye_is_blank_without_a_beacon_or_marine():
  for marine, beacon in [((40.0, 40.0), None), (None, Blob((1.0, 1.0), 9))]:
    img = render.render_eye(marine, beacon, 0.0, "bright_on_dark")
    np.testing.assert_array_equal(img, eye.blank("bright_on_dark"))


def test_closer_beacon_looks_bigger():
  far = render.render_eye((10.0, 40.0), Blob((70.0, 40.0), 28), 0.0, "dark_on_bright")
  near = render.render_eye((60.0, 40.0), Blob((70.0, 40.0), 28), 0.0, "dark_on_bright")
  assert (near < 80).sum() > (far < 80).sum()


def test_blobs_label_each_patch_in_scan_order():
  layer = _layer((60, 10, 3, ENEMY), (10, 30, 2, ENEMY), (40, 40, 3, SELF), (70, 70, 1, ENEMY))
  found = render.blobs(layer, ENEMY)
  assert found == [Blob((61.0, 11.0), 9), Blob((10.5, 30.5), 4), Blob((70.0, 70.0), 1)]
  assert render.blobs(layer, NEUTRAL) == []


def test_blobs_join_units_that_touch_even_at_a_corner():
  layer = _layer((10, 10, 2, ENEMY), (12, 12, 2, ENEMY), (30, 10, 2, ENEMY), (33, 10, 2, ENEMY))
  assert [b.n_px for b in render.blobs(layer, ENEMY)] == [8, 4, 4]


def test_blobs_reach_the_screen_edges():
  layer = _layer((0, 0, 2, ENEMY), (82, 82, 2, ENEMY))
  assert render.blobs(layer, ENEMY) == [Blob((0.5, 0.5), 4), Blob((82.5, 82.5), 4)]


def test_render_eye_draws_a_disk_per_target():
  left, right = Blob((40.0, 30.0), 28), Blob((40.0, 50.0), 12)  # heading screen-right
  img = render.render_eye((40.0, 40.0), [left, right], 0.0, "dark_on_bright")
  r_left, r_right = (render.angular_radius(b.radius_px, 10.0) for b in (left, right))
  np.testing.assert_array_equal(
    img, eye.disks([(-90.0, r_left), (90.0, r_right)], "dark_on_bright"))
  one = render.render_eye((40.0, 40.0), [right], 0.0, "dark_on_bright")
  np.testing.assert_array_equal(one, render.render_eye((40.0, 40.0), right, 0.0,
                                                       "dark_on_bright"))
  np.testing.assert_array_equal(render.render_eye((40.0, 40.0), [], 0.0, "dark_on_bright"),
                                eye.blank("dark_on_bright"))


# The flying fly's view (docs/m6/README.md, "The flying fly")

def _up(d, height=render.MAP_HEIGHT_PX):
  return math.degrees(math.atan2(height, d))


def test_map_eye_puts_a_target_at_its_azimuth_and_its_distance_up_from_the_horizon():
  squad = (40.0, 40.0)
  ahead, left = Blob((50.0, 40.0), 9), Blob((40.0, 20.0), 9)  # heading east: 10 px ahead, 20 left
  img = render.map_eye(squad, [ahead, left], 0.0, "dark_on_bright")
  assert _up(10.0) == pytest.approx(45.0)
  np.testing.assert_array_equal(img, eye.disks(
    [(0.0, render.MAP_DOT_DEG, 45.0), (-90.0, render.MAP_DOT_DEG, _up(20.0))], "dark_on_bright"))


def test_map_eye_turns_with_the_fly():
  squad = (40.0, 40.0)
  east, north = Blob((50.0, 40.0), 9), Blob((40.0, 20.0), 9)
  img = render.map_eye(squad, [east, north], -90.0, "dark_on_bright")  # facing north
  np.testing.assert_array_equal(img, eye.disks(
    [(90.0, render.MAP_DOT_DEG, 45.0), (0.0, render.MAP_DOT_DEG, _up(20.0))], "dark_on_bright"))


def test_map_eye_puts_a_far_target_near_the_horizon_and_a_higher_fly_sees_it_higher():
  far = Blob((80.0, 40.0), 9)
  img = render.map_eye((0.0, 40.0), [far], 0.0, "bright_on_dark")
  np.testing.assert_array_equal(img, eye.disk(0.0, render.MAP_DOT_DEG, "bright_on_dark",
                                              _up(80.0)))
  assert _up(80.0) < 8.0
  high = render.map_eye((0.0, 40.0), [far], 0.0, "bright_on_dark", height_px=40.0)
  np.testing.assert_array_equal(high, eye.disk(0.0, render.MAP_DOT_DEG, "bright_on_dark",
                                               _up(80.0, 40.0)))


def test_map_eye_draws_a_close_patch_at_the_size_it_looks():
  big = Blob((42.0, 40.0), 200)  # radius 8 px, 2 px ahead
  img = render.map_eye((40.0, 40.0), [big], 0.0, "bright_on_dark")
  r = render.angular_radius(big.radius_px, math.hypot(2.0, render.MAP_HEIGHT_PX))
  assert r > render.MAP_DOT_DEG
  np.testing.assert_array_equal(img, eye.disks([(0.0, r, _up(2.0))], "bright_on_dark"))


def test_map_eye_is_blank_without_the_squad_or_a_target():
  t = Blob((50.0, 50.0), 9)
  for squad, targets in (((40.0, 40.0), []), (None, [t])):
    np.testing.assert_array_equal(render.map_eye(squad, targets, 0.0, "dark_on_bright"),
                                  eye.blank("dark_on_bright"))


# The compass fly's view (docs/m6/README.md, "The compass fly")

def test_compass_eye_puts_an_eastward_target_at_positive_azimuth_and_zero_elevation():
  squad = (40.0, 40.0)
  east = Blob((50.0, 40.0), 9)  # 10 px east, level with the squad
  img = render.compass_eye(squad, [east], "dark_on_bright")
  d = render.COMPASS_DEG_PER_PX
  np.testing.assert_array_equal(
    img, eye.disks([(10.0 * d, render.MAP_DOT_DEG, 0.0)], "dark_on_bright"))


def test_compass_eye_puts_a_northward_target_at_positive_elevation_and_zero_azimuth():
  squad = (40.0, 40.0)
  north = Blob((40.0, 20.0), 9)  # smaller y: north of the squad
  img = render.compass_eye(squad, [north], "dark_on_bright")
  d = render.COMPASS_DEG_PER_PX
  np.testing.assert_array_equal(
    img, eye.disks([(0.0, render.MAP_DOT_DEG, 20.0 * d)], "dark_on_bright"))


def test_compass_eye_puts_a_southward_target_at_negative_elevation():
  squad = (40.0, 40.0)
  south = Blob((40.0, 60.0), 9)  # larger y: south of the squad
  img = render.compass_eye(squad, [south], "dark_on_bright")
  d = render.COMPASS_DEG_PER_PX
  np.testing.assert_array_equal(
    img, eye.disks([(0.0, render.MAP_DOT_DEG, -20.0 * d)], "dark_on_bright"))


def test_compass_eye_draws_a_close_patch_at_the_size_it_looks():
  big = Blob((42.0, 40.0), 2000)  # radius about 25 px, 2 px east
  img = render.compass_eye((40.0, 40.0), [big], "bright_on_dark")
  d = render.COMPASS_DEG_PER_PX
  r = big.radius_px * d
  assert r > render.MAP_DOT_DEG
  np.testing.assert_array_equal(img, eye.disks([(2.0 * d, r, 0.0)], "bright_on_dark"))


def test_compass_eye_is_blank_without_the_squad_or_a_target():
  t = Blob((50.0, 50.0), 9)
  for squad, targets in (((40.0, 40.0), []), (None, [t])):
    np.testing.assert_array_equal(render.compass_eye(squad, targets, "dark_on_bright"),
                                  eye.blank("dark_on_bright"))
