"""What the fly sees (spec section 7.2): the eye image, rendered from the screen layers.

Positions are (x, y) screen pixels, x right and y down; the player_relative layer is indexed
[y, x]. The beacon is drawn as a disk at its egocentric azimuth with angular radius
atan(r / d), so an approaching beacon looms. Nothing here knows about neurons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from flycraft import eye

SELF = 1  # player_relative values (pysc2 features.PlayerRelative)
NEUTRAL = 3
MIN_RADIUS_DEG = 2.0
MAX_RADIUS_DEG = 60.0


@dataclass(frozen=True)
class Blob:
  """A unit's footprint on the screen layer: centroid (x, y) and pixel count."""
  xy: tuple[float, float]
  n_px: int

  @property
  def radius_px(self) -> float:
    return math.sqrt(self.n_px / math.pi)


def find(layer: np.ndarray, value: int) -> Blob | None:
  ys, xs = np.nonzero(np.asarray(layer) == value)
  if xs.size == 0:
    return None
  return Blob((float(xs.mean()), float(ys.mean())), int(xs.size))


def egocentric(marine_xy, target_xy, heading_deg: float) -> tuple[float, float]:
  """(azimuth, distance) of target as the marine sees it: azimuth > 0 is the fly's right."""
  dx, dy = target_xy[0] - marine_xy[0], target_xy[1] - marine_xy[1]
  bearing = math.degrees(math.atan2(dy, dx))
  return float(eye.azimuth(bearing, heading_deg)), math.hypot(dx, dy)


def angular_radius(r_px: float, d_px: float) -> float:
  if d_px <= 0.0:
    return MAX_RADIUS_DEG
  return float(np.clip(math.degrees(math.atan(r_px / d_px)), MIN_RADIUS_DEG, MAX_RADIUS_DEG))


def render_eye(marine_xy, beacon: Blob | None, heading_deg: float, polarity: str) -> np.ndarray:
  """The eye image; the background alone when the marine or the beacon is not on screen."""
  if marine_xy is None or beacon is None:
    return eye.blank(polarity)
  az, d = egocentric(marine_xy, beacon.xy, heading_deg)
  return eye.disk(az, angular_radius(beacon.radius_px, d), polarity)
