"""The real brain as a Controller (spec sections 5, 7, 9 and 11): the connectome LIF network,
calibrated by M1, in the game loop.

load_calibration reads M1's calibration.json (the rung that passed G1, w_scale, decoder gains)
and refuses one made for other wiring or another config. Fly runs each decision's brain time in
10 ms slices, always split the same way, so a view watching never changes the dynamics. After
each slice it hands the spike counts to the view, if one is connected.

A brain that runs away raises Runaway, which the server turns into abort "runaway" (spec section
9): the population rate over the last 1 s of brain time above 50 Hz, Poisson-driven inputs left
out as in G0, or a non-finite membrane state.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flycraft import eye as eyemod
from flycraft.brain.brain import Brain, BrainError
from flycraft.brain.calibrate import RUNGS, rung_config
from flycraft.brain.control import Command, Controller, Identity, Runaway
from flycraft.brain.decoder import Calibration
from flycraft.brain.lif import steps
from flycraft.brain.retina import RetinaGeometry
from flycraft.brain.wiring import Wiring
from flycraft.config import Config, to_dict
from flycraft.data.connectome import Connectome

FRAME_MS = 10.0  # brain time per view frame (spec section 11)
RUNAWAY_HZ = 50.0
RUNAWAY_WINDOW_MS = 1000.0
# Config keys a calibration does not depend on: where the data lives, which device runs the
# sim, how M1 searched (the result is what counts), and the compass fly's pitch readout, which
# is set from the grid sweep (docs/m6/README.md, "The compass fly"), not by M1.
UNCHECKED = ("connectome.data_dir", "sim.device", "calibration.", "decoder.pitch_")


class CalibrationError(ValueError):
  pass


@dataclass(frozen=True)
class FlyCalibration:
  rung: str
  w_scale: float
  decoder: Calibration


def _flat(d: dict, prefix: str = "") -> dict:
  out = {}
  for k, v in d.items():
    if isinstance(v, dict):
      out.update(_flat(v, f"{prefix}{k}."))
    else:
      out[f"{prefix}{k}"] = v
  return out


def config_diff(recorded: dict, cfg: Config) -> list[str]:
  """Checked keys whose recorded value differs from cfg's, sorted."""
  now = _flat(json.loads(json.dumps(to_dict(cfg))))
  then = _flat(recorded)
  return sorted(k for k in set(now) | set(then)
                if not k.startswith(UNCHECKED) and now.get(k) != then.get(k))


def _load_decoder_calibration(data: dict) -> Calibration:
  """A decoder_calibration dict as every Calibration field, present and finite (or bool)."""
  fields = dataclasses.fields(Calibration)
  missing = [f.name for f in fields if f.name not in data]
  if missing:
    raise CalibrationError(f"decoder_calibration is missing {', '.join(missing)}")
  for f in fields:
    v = data[f.name]
    if isinstance(f.default, bool):
      if not isinstance(v, bool):
        raise CalibrationError(f"decoder_calibration.{f.name} must be bool, got {v!r}")
    elif not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
      raise CalibrationError(f"decoder_calibration.{f.name} must be a finite number, got {v!r}")
  return Calibration(**data)


def load_calibration(path: str | Path, cfg: Config, wiring: Wiring) -> FlyCalibration:
  path = Path(path)
  try:
    rec = json.loads(path.read_text())
  except FileNotFoundError as e:
    raise CalibrationError(f"no calibration at {path}: run `flycraft m1-report` for this "
                           "config, or pass --calibration") from e
  except (OSError, ValueError) as e:
    raise CalibrationError(f"cannot read {path}: {e}") from e
  try:
    rung, fp, label = rec["ladder"]["rung"], rec["wiring"]["fingerprint"], rec["wiring"]["label"]
    if rung is None:
      raise CalibrationError(f"{path}: no rung passed G1, so there is no calibrated brain to run")
    if rung not in RUNGS:
      raise CalibrationError(f"{path}: unknown rung {rung!r}")
    if fp != wiring.fingerprint:
      raise CalibrationError(
        f"{path} is for {label} ({fp[:12]}), but this config's wiring is {wiring.label} "
        f"({wiring.fingerprint[:12]}): pass that wiring's --calibration")
    diff = config_diff(rec["config"], cfg)
    if diff:
      raise CalibrationError(f"{path} was made with a different config: {', '.join(diff)}")
    dec = _load_decoder_calibration(rec["decoder_calibration"])
    try:
      w_scale = float(rec["w_scale"])
    except (TypeError, ValueError) as e:
      raise CalibrationError(f"{path}: w_scale must be a number, got {rec['w_scale']!r}") from e
    if not math.isfinite(w_scale) or w_scale <= 0:
      raise CalibrationError(f"{path}: w_scale must be finite and > 0, got {w_scale!r}")
    return FlyCalibration(rung, w_scale, dec)
  except (KeyError, TypeError, AttributeError) as e:
    raise CalibrationError(f"{path}: not an M1 calibration record ({e!r})") from e


def brain_id(cfg: Config) -> str:
  c = cfg.connectome
  return "fly-real" if c.wiring == "real" else f"fly-scrambled-{c.scramble_seed}"


class Fly(Controller):
  """view, if given, needs has_clients and publish(info, counts, eye) (flycraft.brain.view)."""

  def __init__(self, conn: Connectome, wiring: Wiring, geom: RetinaGeometry, cfg: Config,
               calib: FlyCalibration, view=None):
    self.calib = calib
    self.cfg = rung_config(cfg, calib.rung)
    self.brain = Brain(conn, wiring, geom, self.cfg, w_scale=calib.w_scale, calib=calib.decoder)
    self.view = view
    self.fingerprint = wiring.fingerprint
    dec = self.cfg.decoder
    self.identity = Identity(brain_id(cfg), wiring.label, cfg.connectome.scramble_seed, False,
                             self.cfg.eye.polarity, dec.decision_frames)
    dt = self.cfg.sim.dt_ms
    total = steps(dec.decision_ms, dt)
    per = max(steps(FRAME_MS, dt), 1)
    full, rem = divmod(total, per)
    step_counts = [per] * full + ([rem] if rem else [])
    self.slices = [n * dt for n in step_counts]
    self._keep = np.ones(conn.n, bool)  # the population the runaway rule watches
    self._keep[self.brain.retina.idx] = False
    self._n_keep = max(int(self._keep.sum()), 1)
    self.brain.run(eyemod.grey(), FRAME_MS)  # warm up the device before the first timed decision
    self.start_episode(-1, self.cfg.sim.seed, "warmup")

  def start_episode(self, episode: int, seed: int, phase: str) -> None:
    """Fresh state, and the episode seed drives the brain's RNG (spec section 9)."""
    self.brain.reset()
    self.brain.gen.manual_seed(seed)
    self.episode, self.step_no, self.score, self.t_ms = episode, 0, 0.0, 0.0
    self._window: deque[tuple[int, float]] = deque()
    self._window_spikes, self._window_ms = 0, 0.0

  def end_episode(self, score: float, steps: int, aborted: str | None) -> None:
    """Drop the view's last frame (spec section 11): a page opening before the next episode
    starts should see nothing stale."""
    if self.view is not None:
      self.view.clear()

  def _pop_rate(self, counts: np.ndarray, ms: float) -> float:
    """This slice's population rate; raises Runaway if the last 1 s ran too hot."""
    spikes = int(counts[self._keep].sum())
    self._window.append((spikes, ms))
    self._window_spikes += spikes
    self._window_ms += ms
    while self._window_ms - self._window[0][1] >= RUNAWAY_WINDOW_MS - 1e-9:
      s, m = self._window.popleft()
      self._window_spikes -= s
      self._window_ms -= m
    if self._window_ms >= RUNAWAY_WINDOW_MS - 1e-9:
      hz = self._window_spikes / self._n_keep / (self._window_ms / 1000.0)
      if hz > RUNAWAY_HZ:
        raise Runaway(f"population at {hz:.1f} Hz over the last "
                      f"{self._window_ms:g} ms of brain time")
    return spikes / self._n_keep / (ms / 1000.0)

  def step(self, eye: np.ndarray, reward: float) -> Command:
    self.score += reward
    dec = self.brain.decoder
    for k, ms in enumerate(self.slices):
      try:
        self.brain.run(eye, ms)
      except BrainError as e:
        raise Runaway(str(e)) from e
      counts, ms_run = self.brain.take_counts()
      self.t_ms += ms_run
      pop = self._pop_rate(counts, ms_run)
      view = self.view
      if view is not None and view.has_clients:
        turn, fwd = dec.signals()
        dtheta, speed = dec.command()
        pitch, pitch_raw = dec.pitch()
        view.publish({
          "t": self.t_ms, "episode": self.episode, "step": self.step_no, "score": self.score,
          "reward": reward if k == 0 else 0.0, "turn": turn, "fwd": fwd, "dtheta": dtheta,
          "speed": speed, "pitch": pitch, "rates": dec.rates(), "pop_hz": pop}, counts, eye)
    self.step_no += 1
    dtheta, speed = dec.command()
    turn, fwd = dec.signals()
    pitch, pitch_raw = dec.pitch()
    return Command(dtheta, speed, turn, fwd, pitch, pitch_raw)
