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
