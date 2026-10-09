"""Gate G0 (weight scale), gate G1 and its ladder, and decoder calibration (spec 5, 7.3, 7.4).

Every brain, real or scrambled, goes through the same procedure; nothing is hand-tuned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from flycraft import eye
from flycraft.brain.brain import Brain
from flycraft.brain.decoder import Calibration, calibrate
from flycraft.config import Config, with_overrides

RUNGS = {
  "default": {},
  "flip_polarity": {"eye.polarity": "bright_on_dark"},
  "tonic_20": {"retina.lamina_tonic_hz": 20.0},
  "tonic_50": {"retina.lamina_tonic_hz": 50.0},
  "vpn": {"retina.mode": "photoreceptor+vpn"},
}
RUNAWAY_HZ = 100.0


class G0Failure(RuntimeError):
  pass


def rung_config(cfg: Config, rung: str) -> Config:
  return with_overrides(cfg, RUNGS[rung])


@dataclass(frozen=True)
class G0Probe:
  scale: float
  mean_hz: float
  frac_over_100hz: float
  runaway: bool


@dataclass(frozen=True)
class G0Result:
  scale: float
  probes: tuple[G0Probe, ...]

  def to_dict(self) -> dict:
    return {"scale": self.scale, "probes": [p.__dict__ for p in self.probes]}


def probe_g0(brain: Brain) -> G0Probe:
  """2 s of uniform grey at the brain's current scale. Poisson-driven inputs are left out:
  the retina sets their rates, w_scale does not."""
  c = brain.cfg.calibration
  brain.reset()
  brain.run(eye.grey(), c.g0_blank_ms)
  counts, ms = brain.take_counts()
  keep = np.ones(counts.size, bool)
  keep[brain.retina.idx] = False
  rate = counts[keep] / (ms / 1000.0)
  mean, frac = float(rate.mean()), float((rate > RUNAWAY_HZ).mean())
  return G0Probe(brain.w_scale, mean, frac,
                 frac > c.g0_max_frac_100hz or mean > c.g0_max_mean_hz)


def run_g0(brain: Brain, log=print) -> G0Result:
  """The largest w_scale that does not run away, so every brain sits at its own edge: from 1,
  halve while it runs away or double while it does not (at most g0_max_halvings steps either
  way), then bisect the last step to g0_precision."""
  c = brain.cfg.calibration
  steps = c.g0_max_halvings
  probes: list[G0Probe] = []

  def quiet(scale: float) -> bool:
    brain.set_w_scale(scale)
    p = probe_g0(brain)
    probes.append(p)
    log(f"G0 w_scale {scale:.4g}: mean {p.mean_hz:.2f} Hz, "
        f"{100 * p.frac_over_100hz:.3f}% over {RUNAWAY_HZ:g} Hz"
        f"{' (runaway)' if p.runaway else ''}")
    return not p.runaway

  if quiet(1.0):
    lo, hi = 1.0, None
    for _ in range(steps):
      if not quiet(lo * 2):
        hi = lo * 2
        break
      lo *= 2
    if hi is None:
      raise G0Failure(f"brain never runs away up to w_scale {lo:.4g} after {steps} doublings")
  else:
    hi, lo = 1.0, None
    for _ in range(steps):
      if quiet(hi / 2):
        lo = hi / 2
        break
      hi /= 2
    if lo is None:
      raise G0Failure(f"brain still runs away at w_scale {hi:.4g} after {steps} halvings")
  while (hi - lo) / hi > c.g0_precision:
    mid = (lo + hi) / 2
    if quiet(mid):
      lo = mid
    else:
      hi = mid
  brain.set_w_scale(lo)
  return G0Result(lo, tuple(probes))


def z_score(a, b) -> float:
  """Welch z of mean(a) - mean(b). Zero spread gives 0 for no difference, else +-inf."""
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  d = float(a.mean() - b.mean())
  se = math.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
  if se == 0:
    return 0.0 if d == 0 else math.copysign(math.inf, d)
  return d / se


@dataclass
class G1Result:
  rung: str
  trials: dict[float, list[dict[str, float]]]  # azimuth -> per-trial rates (Hz)
  z_lc10a: float  # (R - L at the rightmost spot) minus (R - L at the leftmost), in SEs
  z_dn: float
  passed: bool
  blank: np.ndarray = field(repr=False)  # decoder raw (turn, fwd) samples, blank scene
  stim: np.ndarray = field(repr=False)  # the same over every stimulus trial

  def means(self) -> dict[float, dict[str, float]]:
    return {az: {k: float(np.mean([t[k] for t in ts])) for k in ts[0]}
            for az, ts in self.trials.items()}

  def to_dict(self) -> dict:
    return {"rung": self.rung, "z_lc10a": _finite(self.z_lc10a), "z_dn": _finite(self.z_dn),
            "passed": self.passed, "means_hz": {str(az): m for az, m in self.means().items()},
            "n_blank_samples": int(len(self.blank)), "n_stim_samples": int(len(self.stim))}


def _finite(x: float):
  if math.isnan(x):
    return "nan"
  return x if math.isfinite(x) else ("inf" if x > 0 else "-inf")


def _run_trial(brain: Brain, img: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float]]]:
  """One trial: every neuron's rate (Hz) after the warmup, and decoder samples every
  sample_every_ms after it."""
  c = brain.cfg.calibration
  brain.reset()
  samples, t = [], 0.0
  n = round(c.g1_trial_ms / c.sample_every_ms)
  for _ in range(n):
    brain.run(img, c.sample_every_ms)
    t += c.sample_every_ms
    if t <= c.sample_warmup_ms + 1e-9:
      brain.take_counts()  # rates count only the settled part of the trial
      continue
    samples.append(brain.decoder.raw())
  counts, ms = brain.take_counts()
  return counts / (ms / 1000.0), samples


def _trial(brain: Brain, img: np.ndarray) -> tuple[dict[str, float], list[tuple[float, float]]]:
  """One trial: the G1 groups' rates after the warmup, and the decoder samples."""
  rate, samples = _run_trial(brain, img)
  g, geom = brain.decoder.groups, brain.geom
  rates = {
    "lc10a_L": float(rate[geom.vpn_idx[geom.vpn_side == "L"]].mean()),
    "lc10a_R": float(rate[geom.vpn_idx[geom.vpn_side == "R"]].mean()),
    "dn_L": float(rate[g.turn_neg].mean()),
    "dn_R": float(rate[g.turn_pos].mean()),
  }
  return rates, samples


def measure_g1(brain: Brain, rung: str = "default", log=print) -> G1Result:
  """The G1 stimulus set: a spot at each azimuth, g1_repeats times, plus as many blanks."""
  cfg = brain.cfg
  c, pol = cfg.calibration, cfg.eye.polarity
  trials: dict[float, list[dict[str, float]]] = {}
  stim, blank = [], []
  for _ in range(c.g1_repeats):
    blank += _trial(brain, eye.blank(pol))[1]
  for az in c.g1_azimuths_deg:
    img = eye.disk(az, c.g1_radius_deg, pol)
    trials[az] = []
    for _ in range(c.g1_repeats):
      rates, samples = _trial(brain, img)
      trials[az].append(rates)
      stim += samples
  hi, lo = max(c.g1_azimuths_deg), min(c.g1_azimuths_deg)

  def lr(key: str, az: float) -> list[float]:
    return [t[f"{key}_R"] - t[f"{key}_L"] for t in trials[az]]

  z_lc = z_score(lr("lc10a", hi), lr("lc10a", lo))
  z_dn = z_score(lr("dn", hi), lr("dn", lo))
  passed = abs(z_lc) > c.g1_n_se or abs(z_dn) > c.g1_n_se
  log(f"G1 {rung}: z LC10a {z_lc:+.2f}, z DNa02 {z_dn:+.2f} -> {'pass' if passed else 'fail'}")
  return G1Result(rung, trials, z_lc, z_dn, passed, np.asarray(blank, np.float64).reshape(-1, 2),
                  np.asarray(stim, np.float64).reshape(-1, 2))


@dataclass
class LadderResult:
  rung: str | None  # the first rung that passed, or None
  g0: G0Result | None
  g1: G1Result | None
  tried: list[dict]

  @property
  def passed(self) -> bool:
    return self.rung is not None


def run_ladder(brain: Brain, log=print) -> LadderResult:
  """Try each rung in order with its own G0, and keep the first one whose G1 passes.

  A rung whose G0 fails is recorded as failed and the ladder moves on. On success the brain
  is left configured for that rung at its G0 scale. On failure it is restored to the
  configuration it came in with.
  """
  base, base_w_scale = brain.cfg, brain.w_scale
  tried = []
  for rung in base.calibration.ladder:
    log(f"ladder rung {rung}")
    brain.reconfigure(rung_config(base, rung))
    try:
      g0 = run_g0(brain, log)
    except G0Failure as e:
      log(f"G0 {rung}: {e}")
      tried.append({"rung": rung, "g0_scale": None, "z_lc10a": None, "z_dn": None,
                    "passed": False, "g0_failure": str(e)})
      continue
    g1 = measure_g1(brain, rung, log)
    tried.append({"rung": rung, "g0_scale": g0.scale, "z_lc10a": _finite(g1.z_lc10a),
                  "z_dn": _finite(g1.z_dn), "passed": g1.passed, "g0_failure": None})
    if g1.passed:
      return LadderResult(rung, g0, g1, tried)
  brain.reconfigure(base)
  brain.set_w_scale(base_w_scale)
  return LadderResult(None, None, None, tried)


def calibrate_decoder(brain: Brain, g1: G1Result) -> Calibration:
  """Biases from the blank trials, gains from the stimulus trials (spec 7.4)."""
  calib = calibrate(g1.blank, g1.stim, brain.cfg.decoder)
  brain.set_calibration(calib)
  return calib


def g1_stimuli(cfg: Config) -> list[tuple[str, np.ndarray]]:
  """The G1 stimulus set as (condition, eye image): the blank, then a spot at each azimuth."""
  c, pol = cfg.calibration, cfg.eye.polarity
  return [("blank", eye.blank(pol))] + [
    (f"{az:g}", eye.disk(az, c.g1_radius_deg, pol)) for az in c.g1_azimuths_deg]


@dataclass
class Responses:
  raw: dict[str, np.ndarray]  # condition -> (g1_repeats, 2): per-trial mean decoder (turn, fwd)
  pop_hz: dict[str, dict[str, float]]  # condition -> population -> mean rate (Hz)


def measure_responses(brain: Brain, pops: dict[str, np.ndarray] | None = None) -> Responses:
  """The G1 stimulus set, g1_repeats trials per condition, read out as the decoder's mean turn
  and forward signals per trial, plus the mean rate of each named population."""
  c = brain.cfg.calibration
  pops = pops or {}
  raw: dict[str, np.ndarray] = {}
  pop_hz: dict[str, dict[str, float]] = {}
  for cond, img in g1_stimuli(brain.cfg):
    per, sums = [], {k: 0.0 for k in pops}
    for _ in range(c.g1_repeats):
      rate, samples = _run_trial(brain, img)
      per.append(np.asarray(samples, np.float64).reshape(-1, 2).mean(axis=0))
      for k, idx in pops.items():
        sums[k] += float(rate[idx].mean()) if len(idx) else 0.0
    raw[cond] = np.asarray(per)
    pop_hz[cond] = {k: v / c.g1_repeats for k, v in sums.items()}
  return Responses(raw, pop_hz)


def contrast_z(a_hi, a_lo, b_hi, b_lo) -> float:
  """z of (mean(a_hi) - mean(a_lo)) - (mean(b_hi) - mean(b_lo)), four independent samples.
  Zero spread gives 0 for no difference, else +-inf."""
  groups = [np.asarray(x, np.float64) for x in (a_hi, a_lo, b_hi, b_lo)]
  m = [float(g.mean()) for g in groups]
  d = (m[0] - m[1]) - (m[2] - m[3])
  se = math.sqrt(sum(g.var(ddof=1) / g.size for g in groups))
  if se == 0:
    return 0.0 if d == 0 else math.copysign(math.inf, d)
  return d / se


@dataclass
class G2Result:
  factor: float
  n_edges: int
  z_turn: float  # change in the steering contrast (turn at the rightmost spot minus leftmost)
  z_fwd: float  # change in the forward signal over every G1 trial
  by_condition: dict[str, dict[str, float]]  # condition -> {"turn", "fwd"} z; not gating
  passed: bool
  before: Responses = field(repr=False)
  after: Responses = field(repr=False)

  def to_dict(self) -> dict:
    def means(r: Responses) -> dict:
      return {cond: {"turn": float(a[:, 0].mean()), "fwd": float(a[:, 1].mean()),
                     "pop_hz": r.pop_hz[cond]} for cond, a in r.raw.items()}
    return {"factor": self.factor, "n_edges": self.n_edges, "passed": self.passed,
            "z_turn": _finite(self.z_turn), "z_fwd": _finite(self.z_fwd),
            "by_condition": {cond: {k: _finite(v) for k, v in d.items()}
                             for cond, d in self.by_condition.items()},
            "before": means(self.before), "after": means(self.after)}


def run_g2(brain: Brain, edges: np.ndarray, pops: dict[str, np.ndarray] | None = None,
           log=print) -> G2Result:
  """Gate G2 (spec section 8): does scaling these plastic edges by g2_factor move steering?

  Runs the G1 stimulus set before and after the scaling. Two measures gate it, each a z of
  after minus before: the turn response (per-trial mean turn at the rightmost spot minus the
  leftmost, the contrast G1 reads) and the forward response (per-trial mean forward over every
  G1 trial, blanks included). Passes if either |z| exceeds g1_n_se. Per-condition z's are
  reported but do not gate: eight tests at 3 SE would pass by chance about 6% of the time.
  The edges get their weights back afterwards, even if a run fails.
  """
  c = brain.cfg.calibration
  edges = np.asarray(edges)
  before = measure_responses(brain, pops)
  w0 = brain.edge_weights(edges)
  brain.set_edge_weights(edges, w0 * c.g2_factor)
  try:
    after = measure_responses(brain, pops)
  finally:
    brain.set_edge_weights(edges, w0)
  hi, lo = f"{max(c.g1_azimuths_deg):g}", f"{min(c.g1_azimuths_deg):g}"
  z_turn = contrast_z(after.raw[hi][:, 0], after.raw[lo][:, 0],
                      before.raw[hi][:, 0], before.raw[lo][:, 0])
  z_fwd = z_score(np.concatenate([a[:, 1] for a in after.raw.values()]),
                  np.concatenate([a[:, 1] for a in before.raw.values()]))
  by_condition = {cond: {"turn": z_score(after.raw[cond][:, 0], before.raw[cond][:, 0]),
                         "fwd": z_score(after.raw[cond][:, 1], before.raw[cond][:, 1])}
                  for cond in before.raw}
  passed = abs(z_turn) > c.g1_n_se or abs(z_fwd) > c.g1_n_se
  for cond, d in by_condition.items():
    log(f"G2 {cond}: z turn {d['turn']:+.2f}, z fwd {d['fwd']:+.2f}")
  log(f"G2: {edges.size:,} edges at x{c.g2_factor:g}: z turn response {z_turn:+.2f}, "
      f"z forward {z_fwd:+.2f} -> {'pass' if passed else 'fail'}")
  return G2Result(c.g2_factor, int(edges.size), z_turn, z_fwd, by_condition, passed, before,
                  after)
