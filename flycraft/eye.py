"""Eye-image geometry and rendering, shared by the game client and the brain.

The eye image is uint8, EYE_ROWS x EYE_COLS, 5 degree bins. Columns run azimuth -180 to
+180 left to right (0 is the heading, positive is the fly's right). Rows run elevation +75
to -75 top to bottom. Pixel (r, c) is centred at az = -177.5 + 5c, el = 72.5 - 5r.
"""

from __future__ import annotations

import numpy as np

EYE_ROWS = 30
EYE_COLS = 72
BIN_DEG = 5.0
AZ0 = -180.0  # left edge of column 0
EL0 = 75.0  # top edge of row 0

# (background, object) grey levels per polarity. bright_on_dark is 255 minus the default.
POLARITY = {"dark_on_bright": (160, 0), "bright_on_dark": (95, 255)}

_SUB = 4  # supersamples per pixel per axis


def wrap_deg(a):
  """Wrap degrees into (-180, 180]."""
  w = np.mod(np.asarray(a, dtype=np.float64) + 180.0, 360.0) - 180.0
  w = np.where(w == -180.0, 180.0, w)
  return float(w) if np.ndim(w) == 0 else w


def azimuth(bearing_deg, heading_deg):
  """Egocentric azimuth of an object: positive means it is to the fly's right."""
  return wrap_deg(np.asarray(bearing_deg) - np.asarray(heading_deg))


def pixel_az(c):
  return AZ0 + BIN_DEG * (np.asarray(c) + 0.5)


def pixel_el(r):
  return EL0 - BIN_DEG * (np.asarray(r) + 0.5)


def az_to_col(az):
  """Fractional column index whose centre is at azimuth az (no wrap applied)."""
  return (np.asarray(az, dtype=np.float64) - AZ0) / BIN_DEG - 0.5


def el_to_row(el):
  """Fractional row index whose centre is at elevation el."""
  return (EL0 - np.asarray(el, dtype=np.float64)) / BIN_DEG - 0.5


def grey(level: int = 128) -> np.ndarray:
  return np.full((EYE_ROWS, EYE_COLS), level, dtype=np.uint8)


def blank(polarity: str) -> np.ndarray:
  """The scene with nothing in view: the polarity's background everywhere."""
  return grey(POLARITY[polarity][0])


def _angle_deg(az1, el1, az2, el2):
  a1, e1, a2, e2 = (np.radians(x) for x in (az1, el1, az2, el2))
  cos_d = np.sin(e1) * np.sin(e2) + np.cos(e1) * np.cos(e2) * np.cos(a1 - a2)
  return np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0)))


def _cover(az_deg: float, radius_deg: float, el_deg: float = 0.0) -> np.ndarray:
  """The fraction of each pixel's 4x4 supersamples inside the disk, (EYE_ROWS, EYE_COLS)."""
  off = (np.arange(_SUB) + 0.5) / _SUB * BIN_DEG
  az = (AZ0 + BIN_DEG * np.arange(EYE_COLS))[:, None] + off[None, :]  # (cols, sub)
  el = (EL0 - BIN_DEG * np.arange(EYE_ROWS))[:, None] - off[None, :]  # (rows, sub)
  d = _angle_deg(az[None, None, :, :], el[:, :, None, None], az_deg, el_deg)
  return (d <= radius_deg).mean(axis=(1, 3))


def _paint(cover: np.ndarray, polarity: str) -> np.ndarray:
  bg, obj = POLARITY[polarity]
  return np.rint(bg + (obj - bg) * cover).astype(np.uint8)


def disk(az_deg: float, radius_deg: float, polarity: str, el_deg: float = 0.0) -> np.ndarray:
  """Background with one disk of angular radius radius_deg centred at (az_deg, el_deg).

  Membership is by great-circle distance, so the disk wraps across the +-180 seam. Edge
  pixels take the covered fraction of 4x4 supersamples.
  """
  return _paint(_cover(az_deg, radius_deg, el_deg), polarity)


def disks(spots, polarity: str) -> np.ndarray:
  """Background with a disk per spot: (az_deg, radius_deg) on the horizon, or (az_deg,
  radius_deg, el_deg). Where disks overlap a pixel takes the largest cover, so one spot draws
  exactly disk() and none draws blank()."""
  cover = np.zeros((EYE_ROWS, EYE_COLS))
  for spot in spots:
    cover = np.maximum(cover, _cover(*spot))
  return _paint(cover, polarity)
