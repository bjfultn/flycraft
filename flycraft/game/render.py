"""What the fly sees (spec section 7.2): the eye image, rendered from the screen layers.

Positions are (x, y) screen pixels, x right and y down; the player_relative layer is indexed
[y, x]. Each target (the beacon, or each roach) is drawn as a disk at its egocentric azimuth
with angular radius atan(r / d), so an approaching target looms. Nothing here knows about
neurons.
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
