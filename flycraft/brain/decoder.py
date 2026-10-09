"""Descending-neuron decoder (spec section 7.4).

turn = rate(turn_pos) - rate(turn_neg) - b_turn, and fwd = rate(fwd) - b_fwd, where a group's
rate is the mean of its cells' exponentially filtered spike trains. Positive turn is a
clockwise (rightward) turn on screen (section 7.6).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from flycraft.config import DecoderConfig
from flycraft.data.connectome import Connectome

MDN_PATTERN = r"^MDN$"


class DecoderError(ValueError):
  pass


@dataclass(frozen=True)
class Groups:
  turn_pos: np.ndarray
  turn_neg: np.ndarray
  fwd: np.ndarray
  pitch_pos: np.ndarray
  pitch_neg: np.ndarray
  aotu_pos: np.ndarray  # empty unless decoder.aotu019_term
  aotu_neg: np.ndarray
  mdn: np.ndarray  # logged only

  @property
  def watch(self) -> np.ndarray:
    """Sorted unique neuron indices the decoder needs spikes for."""
    return np.unique(np.concatenate([self.turn_pos, self.turn_neg, self.fwd, self.pitch_pos,
                                     self.pitch_neg, self.aotu_pos, self.aotu_neg,
                                     self.mdn])).astype(np.int64)


def resolve_groups(conn: Connectome, cfg: DecoderConfig) -> Groups:
  """Neuron indices for each decoder group; a (type, side) with no cells fails loudly."""
  missing = []

  def pick(pairs):
    out = []
    for type_, side in pairs:
      i = np.flatnonzero((conn.type == type_) & (conn.side == side))
      if i.size == 0:
        missing.append(f"{type_} ({side})")
      out.append(i)
    return np.concatenate(out).astype(np.int64) if out else np.zeros(0, np.int64)

  turn_pos, turn_neg, fwd = pick(cfg.turn_pos), pick(cfg.turn_neg), pick(cfg.fwd)
  pitch_pos, pitch_neg = pick(cfg.pitch_pos), pick(cfg.pitch_neg)
  aotu = (("AOTU019", "R"),), (("AOTU019", "L"),)
  aotu_pos, aotu_neg = (pick(a) for a in aotu) if cfg.aotu019_term else (pick(()), pick(()))
  if missing:
    raise DecoderError("decoder cell types not in the connectome: " + ", ".join(missing))
  return Groups(turn_pos, turn_neg, fwd, pitch_pos, pitch_neg, aotu_pos, aotu_neg,
               conn.select(MDN_PATTERN))


class RateFilter:
  """Exponentially filtered spike trains in Hz: r <- r * exp(-dt/tau) + spikes / tau."""

  def __init__(self, k: int, tau_ms: float, dt_ms: float):
    self.a = math.exp(-dt_ms / tau_ms)
    self.hz_per_spike = 1000.0 / tau_ms
    self.r = np.zeros(k)

  def reset(self) -> None:
    self.r[:] = 0.0

  def update(self, spikes: np.ndarray) -> np.ndarray:
    """Advance over spikes, shape (steps, k), oldest step first; return the rates."""
    s = np.asarray(spikes, dtype=np.float64)
    decay = self.a ** np.arange(len(s) - 1, -1, -1)
    self.r = self.r * self.a ** len(s) + self.hz_per_spike * (decay @ s)
    return self.r.copy()


@dataclass(frozen=True)
class Calibration:
  b_turn: float = 0.0
  b_fwd: float = 0.0
  k_turn: float = 0.0  # deg/s of turn per Hz
  k_fwd: float = 0.0  # speed per Hz
  turn_silent: bool = True
  fwd_silent: bool = True
  p95_turn_hz: float = 0.0
  p95_fwd_hz: float = 0.0

  def to_dict(self) -> dict:
    return asdict(self)


def calibrate(blank: np.ndarray, stim: np.ndarray, cfg: DecoderConfig) -> Calibration:
  """Biases from blank-scene samples, gains from stimulus samples.

  blank and stim are (n, 2) arrays of raw (turn, fwd) decoder outputs, before bias.
  """
  blank, stim = np.asarray(blank, np.float64), np.asarray(stim, np.float64)
  b_turn, b_fwd = blank.mean(axis=0)
  p_turn = float(np.percentile(np.abs(stim[:, 0] - b_turn), 95))
  p_fwd = float(np.percentile(stim[:, 1] - b_fwd, 95))
  turn_silent = not p_turn >= cfg.gain_floor_hz
  fwd_silent = not p_fwd >= cfg.gain_floor_hz
  return Calibration(
    b_turn=float(b_turn), b_fwd=float(b_fwd),
    k_turn=0.0 if turn_silent else cfg.max_turn_deg_s / p_turn,
    k_fwd=0.0 if fwd_silent else (1.0 - cfg.s0) / p_fwd,
    turn_silent=turn_silent, fwd_silent=fwd_silent, p95_turn_hz=p_turn, p95_fwd_hz=p_fwd)


class Decoder:
  def __init__(self, groups: Groups, cfg: DecoderConfig, calib: Calibration | None = None,
               dt_ms: float = 0.1):
    self.groups, self.cfg = groups, cfg
    self.calib = Calibration() if calib is None else calib
    self.watch = groups.watch
    self.filter = RateFilter(self.watch.size, cfg.tau_ms, dt_ms)
    self._pos = {name: np.searchsorted(self.watch, getattr(groups, name))
                 for name in ("turn_pos", "turn_neg", "fwd", "pitch_pos", "pitch_neg",
                              "aotu_pos", "aotu_neg", "mdn")}
    self.t_game_s = cfg.decision_frames / cfg.game_fps

  def reset(self) -> None:
    self.filter.reset()

  def observe(self, spikes: np.ndarray) -> None:
    """spikes: (steps, len(watch)) bool, the watch neurons' spikes since the last call."""
    self.filter.update(spikes)

  def _mean(self, name: str) -> float:
    p = self._pos[name]
    return float(self.filter.r[p].mean()) if p.size else 0.0

  def raw(self) -> tuple[float, float]:
    """(turn, fwd) in Hz before bias subtraction."""
    turn = self._mean("turn_pos") - self._mean("turn_neg")
    if self.cfg.aotu019_term:
      turn += self._mean("aotu_pos") - self._mean("aotu_neg")
    return turn, self._mean("fwd")

  def signals(self) -> tuple[float, float]:
    turn, fwd = self.raw()
    return turn - self.calib.b_turn, fwd - self.calib.b_fwd

  def command(self) -> tuple[float, float]:
    """(dtheta in degrees for one decision, speed in 0..1)."""
    turn, fwd = self.signals()
    lim = self.cfg.max_turn_deg_s * self.t_game_s
    dtheta = float(np.clip(self.calib.k_turn * turn * self.t_game_s, -lim, lim))
    speed = float(np.clip(self.cfg.s0 + self.calib.k_fwd * fwd, 0.0, 1.0))
    return dtheta, speed

  def pitch(self) -> tuple[float, float]:
    """(pitch in [-1, 1], raw Hz before bias and scale). Positive is up, which is north in the
    compass eye (M6, docs/m6/README.md, "The compass fly")."""
    raw = self._mean("pitch_pos") - self._mean("pitch_neg")
    pitch = float(np.clip((raw - self.cfg.pitch_bias_hz) / self.cfg.pitch_scale_hz, -1.0, 1.0))
    return pitch, raw

  def rates(self) -> dict[str, float]:
    """Group mean rates, for logging."""
    return {name: self._mean(name) for name in self._pos}
