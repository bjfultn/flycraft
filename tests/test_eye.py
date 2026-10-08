import numpy as np
import pytest

from flycraft.eye import (
  EYE_COLS,
  EYE_ROWS,
  POLARITY,
  az_to_col,
  azimuth,
  blank,
  disk,
  disks,
  el_to_row,
  grey,
  pixel_az,
  pixel_el,
  wrap_deg,
)


def test_wrap_deg_range():
  assert wrap_deg(180.0) == 180.0
  assert wrap_deg(-180.0) == 180.0
  assert wrap_deg(190.0) == -170.0
  assert wrap_deg(-190.0) == 170.0
  assert wrap_deg(540.0) == 180.0
  np.testing.assert_allclose(wrap_deg([0.0, 359.0, -359.0]), [0.0, -1.0, 1.0])


def test_azimuth_positive_is_right_in_screen_frame():
  # Screen frame: x right, y down, angles from atan2(dy, dx). Marine faces screen-up
  # (heading -90). A beacon to its screen-right has bearing 0, so azimuth +90.
  assert azimuth(0.0, -90.0) == 90.0
  assert azimuth(180.0, -90.0) == -90.0
  assert azimuth(-90.0, -90.0) == 0.0


def test_pixel_centres_and_inverse():
  assert pixel_az(0) == -177.5 and pixel_az(EYE_COLS - 1) == 177.5
  assert pixel_el(0) == 72.5 and pixel_el(EYE_ROWS - 1) == -72.5
  np.testing.assert_allclose(az_to_col(pixel_az(np.arange(EYE_COLS))), np.arange(EYE_COLS))
  np.testing.assert_allclose(el_to_row(pixel_el(np.arange(EYE_ROWS))), np.arange(EYE_ROWS))


def test_grey_and_blank():
  g = grey()
  assert g.shape == (EYE_ROWS, EYE_COLS) and g.dtype == np.uint8 and (g == 128).all()
  assert (blank("dark_on_bright") == 160).all()
  assert (blank("bright_on_dark") == 95).all()


def test_polarities_are_complements():
  (bg1, ob1), (bg2, ob2) = POLARITY["dark_on_bright"], POLARITY["bright_on_dark"]
  assert (bg2, ob2) == (255 - bg1, 255 - ob1)


def test_disk_lands_in_the_right_half_for_positive_azimuth():
  img = disk(90.0, 10.0, "dark_on_bright")
  assert img.dtype == np.uint8 and img.shape == (EYE_ROWS, EYE_COLS)
  dark = np.argwhere(img < 160)
  assert (pixel_az(dark[:, 1]) > 0).all()
  # Pixel (15, 53) spans az 85..90, el 0..-5: fully inside the disk.
  assert img[15, 53] == 0


def test_disk_area_matches_solid_angle():
  # A 20 degree disk at the equator covers about pi*20^2 square degrees = 50 pixels.
  img = disk(0.0, 20.0, "dark_on_bright")
  covered = (160 - img.astype(float)).sum() / 160
  assert covered == pytest.approx(np.pi * 20**2 / 25, rel=0.05)


def test_disk_wraps_across_the_seam():
  img = disk(180.0, 10.0, "dark_on_bright")
  assert img[:, 0].min() < 160 and img[:, -1].min() < 160
  np.testing.assert_array_equal(img[:, :2], img[:, -2:][:, ::-1])


def test_bright_on_dark_disk():
  img = disk(0.0, 10.0, "bright_on_dark")
  assert img.max() == 255 and img.min() == 95


@pytest.mark.parametrize("polarity", sorted(POLARITY))
def test_disks_of_one_spot_is_disk_and_of_none_is_blank(polarity):
  np.testing.assert_array_equal(disks([(37.5, 12.0)], polarity), disk(37.5, 12.0, polarity))
  np.testing.assert_array_equal(disks([], polarity), blank(polarity))


def test_overlapping_disks_take_the_larger_cover():
  both = disks([(-20.0, 15.0), (10.0, 15.0)], "dark_on_bright")
  left, right = disk(-20.0, 15.0, "dark_on_bright"), disk(10.0, 15.0, "dark_on_bright")
  np.testing.assert_array_equal(both, np.minimum(left, right))  # dark object: smaller is more
  assert (both < 160).sum() > (left < 160).sum()
