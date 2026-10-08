"""Photoreceptor viewing directions and eye-image drive (spec section 7.3).

Columns. A photoreceptor has no hex column of its own, so it takes the synapse-weighted
majority column of its columnar partners: L1-L3 for R1-R6, any hex-assigned non-lamina
partner for R7/R8. The winning partner's side is the eye.

Map. Lamina somata are retinotopic: soma y tracks elevation. Lamina soma z is noisy, but Mi1
soma z fits hex well and runs opposite to lamina z (the first optic chiasm), so the
anterior-posterior axis is minus Mi1's z slopes. Each eye is normalized by the 2nd-98th
percentiles over its own lamina columns: azimuth -15 to +165 and elevation +75 to -60, with
the left eye mirrored. If the fits fail their checks, `auto` falls back to 6 azimuth bands per
eye ranked along the lamina z axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flycraft import eye as eyemod
from flycraft.config import EyeConfig, RetinaConfig
from flycraft.data.build import POS_SOMA
from flycraft.data.connectome import Connectome

R16_PATTERN = r"^R1-R6$"
R78_PATTERN = r"^R[78]"
LAMINA_PATTERN = r"^L[1-5]$"
VPN_PATTERN = r"^LC10a$"
R16_PARTNERS = ("L1", "L2", "L3")
AZ_RANGE = (-15.0, 165.0)
EL_RANGE = (-60.0, 75.0)
N_BANDS = 6
MIN_Y_R2 = 0.9
MIN_MI1_Z_R2 = 0.5
HOP_MIN_SYN = 5


class RetinaMapError(ValueError):
  pass


def _fit(h: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, float]:
  """Least squares v ~ [1, h1, h2]; returns (coef, R^2)."""
  a = np.c_[np.ones(len(h)), h]
  coef = np.linalg.lstsq(a, v, rcond=None)[0]
  ss = ((v - v.mean()) ** 2).sum()
  r2 = 1.0 - ((v - a @ coef) ** 2).sum() / ss if ss > 0 else 0.0
  return coef, float(r2)


def _unit(az, el):
  a, e = np.radians(az), np.radians(el)
  return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], axis=-1)


def _angles(u):
  u = u / np.linalg.norm(u, axis=-1, keepdims=True)
  az = np.degrees(np.arctan2(u[..., 1], u[..., 0]))
  return az, np.degrees(np.arcsin(np.clip(u[..., 2], -1.0, 1.0)))


@dataclass
class EyeMap:
  """Hex -> (azimuth, elevation) for one eye, before mirroring."""

  mode: str  # "affine" | "bands"
  z_coef: np.ndarray  # A-P key = z_coef @ (h1, h2)
  y_coef: np.ndarray  # elevation key = y_coef @ (h1, h2)
  z_lo: float
  z_hi: float
  y_lo: float
  y_hi: float
  band_edges: np.ndarray

  def direction(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    zk = h @ self.z_coef
    if self.mode == "bands":
      band = np.searchsorted(self.band_edges, zk, side="right")
      return 30.0 * band.astype(np.float64), np.zeros(len(h))
    az = AZ_RANGE[0] + (AZ_RANGE[1] - AZ_RANGE[0]) * (zk - self.z_lo) / (self.z_hi - self.z_lo)
    yk = h @ self.y_coef
    el = EL_RANGE[1] - (EL_RANGE[1] - EL_RANGE[0]) * (yk - self.y_lo) / (self.y_hi - self.y_lo)
    return np.clip(az, *AZ_RANGE), np.clip(el, *EL_RANGE)


@dataclass
class RetinaGeometry:
  pr_idx: np.ndarray  # int64 (P,) photoreceptor neuron indices
  pr_side: np.ndarray  # "L" | "R" | "?" (unassigned)
  pr_az: np.ndarray  # float64 (P,), NaN where unassigned
  pr_el: np.ndarray
  pr_hex: np.ndarray  # float64 (P, 2): the column each photoreceptor took, NaN where unassigned
  lamina_idx: np.ndarray  # L1-L5, for tonic drive
  vpn_idx: np.ndarray  # LC10a with a receptive field
  vpn_side: np.ndarray
  vpn_az: np.ndarray
  vpn_el: np.ndarray
  map_mode: str
  info: dict = field(default_factory=dict)

  @property
  def assigned(self) -> np.ndarray:
    return self.pr_side != "?"

  @classmethod
  def build(cls, conn: Connectome, map_mode: str = "auto") -> RetinaGeometry:
    has_hex = np.isfinite(conn.hex).all(axis=1) & np.isin(conn.side, ("L", "R"))
    hexi = np.where(has_hex[:, None], np.nan_to_num(conn.hex), 0).round().astype(np.int64)
    maps, info = _eye_maps(conn, has_hex, map_mode)
    mode = maps["R"].mode

    def direction(idx):
      az = np.full(idx.size, np.nan)
      el = np.full(idx.size, np.nan)
      for s, sign in (("R", 1.0), ("L", -1.0)):
        m = conn.side[idx] == s
        a, e = maps[s].direction(hexi[idx[m]].astype(np.float64))
        az[m], el[m] = sign * a, e
      return az, el

    r16 = conn.select(R16_PATTERN)
    r78 = conn.select(R78_PATTERN)
    lamina = np.zeros(conn.n, bool)
    lamina[conn.select(LAMINA_PATTERN)] = True
    r16_ok = np.isin(conn.type, R16_PARTNERS) & has_hex
    r78_ok = ~lamina & has_hex
    pr_idx = np.concatenate([r16, r78]).astype(np.int64)
    col = np.full(pr_idx.size, -1, dtype=np.int64)  # winning partner neuron per photoreceptor
    col[: r16.size] = _majority_partner(conn, r16, r16_ok, hexi)
    col[r16.size:] = _majority_partner(conn, r78, r78_ok, hexi)
    ok = col >= 0
    side = np.full(pr_idx.size, "?", dtype=conn.side.dtype)
    side[ok] = conn.side[col[ok]]
    az = np.full(pr_idx.size, np.nan)
    el = np.full(pr_idx.size, np.nan)
    az[ok], el[ok] = direction(col[ok])
    pr_hex = np.full((pr_idx.size, 2), np.nan)
    pr_hex[ok] = hexi[col[ok]]
    vpn_idx, vpn_az, vpn_el = _vpn_fields(conn, has_hex, direction)
    info.update({
      "n_photoreceptors": int(pr_idx.size),
      "n_unassigned": int((~ok).sum()),
      "n_r16": {s: int(((conn.side[col[: r16.size][ok[: r16.size]]]) == s).sum()) for s in "LR"},
      "n_r78": {s: int(((conn.side[col[r16.size:][ok[r16.size:]]]) == s).sum()) for s in "LR"},
      "n_vpn": int(vpn_idx.size),
      "n_vpn_without_rf": int(conn.select(VPN_PATTERN).size - vpn_idx.size),
    })
    return cls(pr_idx, side, az, el, pr_hex, np.flatnonzero(lamina), vpn_idx, conn.side[vpn_idx],
               vpn_az, vpn_el, mode, info)


def _majority_partner(conn, src, partner_ok, hexi) -> np.ndarray:
  """For each src neuron, one partner from its synapse-weighted majority (side, h1, h2) column.

  Returns -1 where src has no qualifying partner. Ties go to the smallest column key.
  """
  pos = np.full(conn.n, -1, dtype=np.int64)
  pos[src] = np.arange(src.size)
  m = (pos[conn.pre] >= 0) & partner_ok[conn.post]
  p, q, w = pos[conn.pre[m]], conn.post[m].astype(np.int64), conn.syn[m].astype(np.int64)
  out = np.full(src.size, -1, dtype=np.int64)
  if p.size == 0:
    return out
  side_l = (conn.side[q] == "L").astype(np.int64)  # 0 = R, 1 = L; R sorts before L on ties
  # Group by the (src, side, h1, h2) tuple directly instead of packing it into one int: a
  # packed scalar key can alias two genuinely different columns (e.g. hex (1, -1000) and hex
  # (0, 0) used to pack to the same key), merging their votes and picking whichever was first.
  cols = np.stack([p, side_l, hexi[q, 0], hexi[q, 1]], axis=1)
  uniq, first, inv = np.unique(cols, axis=0, return_index=True, return_inverse=True)
  votes = np.bincount(inv, weights=w)
  up = uniq[:, 0]
  # uniq's rows are already sorted ascending by (src, side, h1, h2), so for a fixed src its row
  # index is already ascending in (side, h1, h2) -- exactly the old tie-break, without aliasing.
  order = np.lexsort((np.arange(uniq.shape[0]), -votes, up))  # by src, then votes, then key
  head = np.ones(order.size, bool)
  head[1:] = up[order][1:] != up[order][:-1]
  win = order[head]
  out[up[win]] = q[first[win]]
  return out


def _wsum(rows, vecs, w, n):
  """Weighted sum of vecs (E, 3) into n rows."""
  return np.stack([np.bincount(rows, weights=vecs[:, k] * w, minlength=n) for k in range(3)],
                  axis=1)


def _vpn_fields(conn, has_hex, direction):
  """LC10a receptive-field centres: weighted mean direction of hex inputs, direct or one hop.

  A hex input counts at any weight. A non-hex input counts when it reaches the LC10a with at
  least HOP_MIN_SYN synapses and itself gets that many from hex neurons; its direction is the
  weighted mean of those hex inputs.
  """
  vpn = conn.select(VPN_PATTERN)
  hex_dir = np.zeros((conn.n, 3))
  hx = np.flatnonzero(has_hex)
  hex_dir[hx] = _unit(*direction(hx))
  w = conn.syn.astype(np.float64)
  strong = conn.syn >= HOP_MIN_SYN
  m = strong & has_hex[conn.pre] & ~has_hex[conn.post]
  hop = _wsum(conn.post[m], hex_dir[conn.pre[m]], w[m], conn.n)
  norm = np.linalg.norm(hop, axis=1)
  usable = has_hex | (norm > 0)
  src_dir = hex_dir.copy()
  src_dir[norm > 0] = hop[norm > 0] / norm[norm > 0, None]
  pos = np.full(conn.n, -1, dtype=np.int64)
  pos[vpn] = np.arange(vpn.size)
  m = (pos[conn.post] >= 0) & usable[conn.pre] & (has_hex[conn.pre] | strong)
  acc = _wsum(pos[conn.post[m]], src_dir[conn.pre[m]], w[m], vpn.size)
  ok = np.linalg.norm(acc, axis=1) > 0
  az, el = _angles(acc[ok])
  return vpn[ok].astype(np.int64), az, el


def _eye_maps(conn, has_hex, map_mode):
  soma = conn.pos_source == POS_SOMA
  info = {"fits": {}, "map_reason": ""}
  fits = {}
  def pts(type_, s):
    i = np.flatnonzero((conn.type == type_) & (conn.side == s) & has_hex & soma)
    return conn.hex[i].astype(np.float64), conn.pos[i].astype(np.float64)

  for s in ("R", "L"):
    h_l1, p_l1 = pts("L1", s)
    h_mi, p_mi = pts("Mi1", s)
    if len(h_l1) < 6 or len(h_mi) < 6:
      fits[s] = None
      info["fits"][s] = {"n_l1": len(h_l1), "n_mi1": len(h_mi)}
      continue
    ly, ly_r2 = _fit(h_l1, p_l1[:, 1])
    lz, lz_r2 = _fit(h_l1, p_l1[:, 2])
    mz, mz_r2 = _fit(h_mi, p_mi[:, 2])
    fits[s] = (ly, lz, mz)
    info["fits"][s] = {"n_l1": len(h_l1), "n_mi1": len(h_mi), "l1_y": [*ly.round(3)],
                       "l1_y_r2": round(ly_r2, 3), "l1_z": [*lz.round(3)],
                       "l1_z_r2": round(lz_r2, 3), "mi1_z": [*mz.round(3)],
                       "mi1_z_r2": round(mz_r2, 3)}
  problems = []
  for s in ("R", "L"):
    f = info["fits"][s]
    if fits[s] is None:
      problems.append(f"{s}: too few L1/Mi1 somata with hex")
      continue
    ly, lz, mz = fits[s]
    if f["l1_y_r2"] < MIN_Y_R2:
      problems.append(f"{s}: L1 y R^2 {f['l1_y_r2']} < {MIN_Y_R2}")
    if f["mi1_z_r2"] < MIN_MI1_Z_R2:
      problems.append(f"{s}: Mi1 z R^2 {f['mi1_z_r2']} < {MIN_MI1_Z_R2}")
    if not (np.sign(lz[1:]) == -np.sign(mz[1:])).all() or not np.all(mz[1:]):
      problems.append(f"{s}: L1 z and Mi1 z slopes are not opposite on both hex axes")
  mode = map_mode
  if map_mode == "auto":
    mode = "bands" if problems else "affine"
  elif map_mode == "affine" and problems:
    raise RetinaMapError("retina.map=affine but the hex fits fail: " + "; ".join(problems))
  if mode == "bands" and any(fits[s] is None for s in fits):
    raise RetinaMapError("no retina map possible: " + "; ".join(problems))
  info["map_reason"] = "; ".join(problems) if problems else "fits pass"
  maps = {}
  for s in ("R", "L"):
    ly, lz, mz = fits[s]
    cols = conn.hex[(conn.type == "L1") & (conn.side == s) & has_hex].astype(np.float64)
    z_coef = -mz[1:] if mode == "affine" else lz[1:]
    zk, yk = cols @ z_coef, cols @ ly[1:]
    edges = np.quantile(zk, np.arange(1, N_BANDS) / N_BANDS)
    maps[s] = EyeMap(mode, z_coef, ly[1:], *np.percentile(zk, [2, 98]), *np.percentile(yk, [2, 98]),
                     edges)
  info["map"] = mode
  return maps, info


class Retina:
  """Turns an eye image into Poisson rates for the input neurons."""

  def __init__(self, geom: RetinaGeometry, cfg: RetinaConfig, eye_cfg: EyeConfig):
    self.geom, self.cfg = geom, cfg
    self.bg, self.obj = eyemod.POLARITY[eye_cfg.polarity]
    ok = geom.assigned
    a = geom.pr_az[ok]
    e = geom.pr_el[ok]
    col = eyemod.az_to_col(a)
    row = np.clip(eyemod.el_to_row(e), 0, eyemod.EYE_ROWS - 1)
    c0 = np.floor(col).astype(np.int64)
    r0 = np.floor(row).astype(np.int64)
    self._fc, self._fr = col - c0, row - r0
    self._c0, self._c1 = c0 % eyemod.EYE_COLS, (c0 + 1) % eyemod.EYE_COLS
    self._r0, self._r1 = r0, np.minimum(r0 + 1, eyemod.EYE_ROWS - 1)
    self._ok = ok
    self.scale = np.ones(geom.pr_idx.size)
    if cfg.eye_normalize:
      counts = {s: int((geom.pr_side == s).sum()) for s in ("L", "R")}
      mean = (counts["L"] + counts["R"]) / 2
      for s in ("L", "R"):
        if counts[s]:
          self.scale[geom.pr_side == s] = mean / counts[s]
    idx = [geom.pr_idx]
    if cfg.lamina_tonic_hz > 0:
      idx.append(geom.lamina_idx)
    self.vpn_on = cfg.mode == "photoreceptor+vpn"
    if self.vpn_on:
      idx.append(geom.vpn_idx)
      paz = eyemod.pixel_az(np.arange(eyemod.EYE_COLS))[None, :]
      pel = eyemod.pixel_el(np.arange(eyemod.EYE_ROWS))[:, None]
      d = eyemod._angle_deg(paz[None], pel[None], geom.vpn_az[:, None, None],
                            geom.vpn_el[:, None, None])
      mask = (d <= cfg.vpn_rf_radius_deg).reshape(len(geom.vpn_idx), -1).astype(np.float64)
      nearest = d.reshape(len(geom.vpn_idx), -1).argmin(axis=1)
      empty = mask.sum(axis=1) == 0
      mask[empty, nearest[empty]] = 1.0
      self._vpn_mask = mask / mask.sum(axis=1, keepdims=True)
    self.idx = np.concatenate(idx).astype(np.int64)

  @property
  def rfc_zero_idx(self) -> np.ndarray:
    """Poisson-driven neurons, which get no refractory period (spec section 5)."""
    return self.idx

  def sample(self, img: np.ndarray) -> np.ndarray:
    """Bilinear luminance (0..1) at each assigned photoreceptor's direction."""
    f = img.astype(np.float64) / 255.0
    top = f[self._r0, self._c0] * (1 - self._fc) + f[self._r0, self._c1] * self._fc
    bot = f[self._r1, self._c0] * (1 - self._fc) + f[self._r1, self._c1] * self._fc
    return top * (1 - self._fr) + bot * self._fr

  def rates(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(neuron indices, Poisson rates in Hz) for this eye image."""
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.shape != (
        eyemod.EYE_ROWS, eyemod.EYE_COLS):
      shape = getattr(img, "shape", None)
      dtype = getattr(img, "dtype", type(img).__name__)
      raise ValueError(f"eye image must be uint8 {eyemod.EYE_ROWS}x{eyemod.EYE_COLS}, "
                       f"got {dtype} {shape}")
    lum = np.full(self.geom.pr_idx.size, img.mean() / 255.0)
    lum[self._ok] = self.sample(img)
    out = [self.cfg.r_max_hz * lum * self.scale]
    if self.cfg.lamina_tonic_hz > 0:
      out.append(np.full(self.geom.lamina_idx.size, self.cfg.lamina_tonic_hz))
    if self.vpn_on:
      contrast = np.abs(img.astype(np.float64) - self.bg) / abs(self.obj - self.bg)
      out.append(self.cfg.vpn_rate_hz * (self._vpn_mask @ contrast.ravel()))
    return self.idx, np.concatenate(out)
