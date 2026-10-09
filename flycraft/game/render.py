"""What the fly sees (spec section 7.2): the eye image, rendered from the screen layers.

Positions are (x, y) screen pixels, x right and y down; the player_relative layer is indexed
[y, x]. The walking fly (render_eye) sees each target (the beacon, or each roach) as a disk at
its egocentric azimuth with angular radius atan(r / d), so an approaching target looms. The
flying fly (map_eye) looks down on the map instead, so a target's distance is how far it is
from the horizon. The compass fly (compass_eye) also looks down on the map, but north up and
independent of heading: east and north are a target's screen-pixel offset from the squad,
scaled straight into azimuth and elevation. Nothing here knows about neurons.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from flycraft import eye

SELF = 1  # player_relative values (pysc2 features.PlayerRelative)
NEUTRAL = 3
ENEMY = 4
MIN_RADIUS_DEG = 2.0
MAX_RADIUS_DEG = 60.0
MAP_DOT_DEG = 10.0  # a target's least radius in the map view: the calibration's disk
MAP_HEIGHT_PX = 10.0  # the flying fly's height over the map: a target this far off is 45 deg up
COMPASS_DEG_PER_PX = 60.0 / 84  # a screen width, 84 px, spans 60 degrees (as in fly1)


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


def blobs(layer: np.ndarray, value: int) -> list[Blob]:
  """Each 8-connected patch of value as its own Blob, in scan order of its first pixel (top
  row first). Units that touch on the screen make one patch."""
  hit = np.asarray(layer) == value
  h, w = hit.shape
  mask, seen = hit.tolist(), [[False] * w for _ in range(h)]
  out = []
  for y0, x0 in zip(*np.nonzero(hit), strict=True):
    y0, x0 = int(y0), int(x0)
    if seen[y0][x0]:
      continue
    seen[y0][x0] = True
    stack, xs, ys = [(y0, x0)], [], []
    while stack:
      y, x = stack.pop()
      xs.append(x)
      ys.append(y)
      for ny in range(max(y - 1, 0), min(y + 2, h)):
        for nx in range(max(x - 1, 0), min(x + 2, w)):
          if mask[ny][nx] and not seen[ny][nx]:
            seen[ny][nx] = True
            stack.append((ny, nx))
    out.append(Blob((sum(xs) / len(xs), sum(ys) / len(ys)), len(xs)))
  return out


def nearest_px(layer: np.ndarray, value: int, xy) -> tuple[float, float] | None:
  """The pixel of value nearest xy (the first in scan order on a tie), or None. An order aimed
  there lands on a unit even when xy, a patch's middle, falls between units."""
  ys, xs = np.nonzero(np.asarray(layer) == value)
  if xs.size == 0:
    return None
  i = int(np.argmin((xs - xy[0]) ** 2 + (ys - xy[1]) ** 2))
  return float(xs[i]), float(ys[i])


def egocentric(marine_xy, target_xy, heading_deg: float) -> tuple[float, float]:
  """(azimuth, distance) of target as the marine sees it: azimuth > 0 is the fly's right."""
  dx, dy = target_xy[0] - marine_xy[0], target_xy[1] - marine_xy[1]
  bearing = math.degrees(math.atan2(dy, dx))
  return float(eye.azimuth(bearing, heading_deg)), math.hypot(dx, dy)


def angular_radius(r_px: float, d_px: float) -> float:
  if d_px <= 0.0:
    return MAX_RADIUS_DEG
  return float(np.clip(math.degrees(math.atan(r_px / d_px)), MIN_RADIUS_DEG, MAX_RADIUS_DEG))


def render_eye(marine_xy, targets: Blob | Sequence[Blob] | None, heading_deg: float,
               polarity: str) -> np.ndarray:
  """The eye image with a disk per target; the background alone when the marine or every
  target is off screen."""
  if isinstance(targets, Blob):
    targets = [targets]
  if marine_xy is None or not targets:
    return eye.blank(polarity)
  spots = []
  for t in targets:
    az, d = egocentric(marine_xy, t.xy, heading_deg)
    spots.append((az, angular_radius(t.radius_px, d)))
  return eye.disks(spots, polarity)


def map_eye(marine_xy, targets: Sequence[Blob], heading_deg: float, polarity: str,
            height_px: float = MAP_HEIGHT_PX) -> np.ndarray:
  """The flying fly's eye (M6, docs/m6/README.md, "The flying fly"): the map as a fly height_px
  over it sees it. Each target is at its azimuth from the heading, as the walking fly sees it,
  and atan(height_px / d) from the horizon, so a far target is near the horizon and a close one
  high up. That is the ground's view drawn above the horizon instead of below it, since this
  brain hardly answers anything below it (docs/m6/README.md, "All the way round, and up and
  down"). Each target is a disk at least MAP_DOT_DEG in radius, bigger when close enough to
  look it. The background alone when the squad or every target is off screen."""
  if marine_xy is None or not targets:
    return eye.blank(polarity)
  spots = []
  for t in targets:
    az, d = egocentric(marine_xy, t.xy, heading_deg)
    r = max(MAP_DOT_DEG, angular_radius(t.radius_px, math.hypot(d, height_px)))
    spots.append((az, r, math.degrees(math.atan2(height_px, d))))
  return eye.disks(spots, polarity)


def compass_eye(marine_xy, targets: Sequence[Blob], polarity: str) -> np.ndarray:
  """The compass fly's eye (M6, docs/m6/README.md, "The compass fly"): north up, the squad at
  the centre, independent of heading. Each target is at its screen-pixel offset (dx, dy) from
  the squad, east and north scaled straight into azimuth and elevation by COMPASS_DEG_PER_PX
  (a screen width, 84 px, spans 60 degrees). Each target is a disk at least MAP_DOT_DEG in
  radius, bigger when close enough to look it. The background alone when the squad or every
  target is off screen."""
  if marine_xy is None or not targets:
    return eye.blank(polarity)
  spots = []
  for t in targets:
    dx, dy = t.xy[0] - marine_xy[0], t.xy[1] - marine_xy[1]
    az, el = dx * COMPASS_DEG_PER_PX, -dy * COMPASS_DEG_PER_PX
    r = max(MAP_DOT_DEG, t.radius_px * COMPASS_DEG_PER_PX)
    spots.append((az, r, el))
  return eye.disks(spots, polarity)
