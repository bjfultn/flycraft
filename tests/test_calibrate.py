import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from flycraft.brain import calibrate as calibrate_mod
from flycraft.brain.brain import Brain
from flycraft.brain.calibrate import (
  RUNGS,
  G0Failure,
  G0Result,
  G1Result,
  calibrate_decoder,
  run_g0,
  run_ladder,
  rung_config,
  z_score,
)
from flycraft.brain.retina import RetinaGeometry
from flycraft.brain.wiring import load_wiring
from flycraft.config import LADDER_RUNGS, with_overrides

QUIET = lambda s: None  # noqa: E731


class FakeBrain:
  """Runs away above w_scale `limit`, by mean rate or by a fast tail. Neuron 0 is a Poisson
  input firing far above any limit, which G0 must ignore."""

  def __init__(self, cfg, limit, how="mean"):
    self.cfg, self.limit, self.how, self.w_scale = cfg, limit, how, 1.0
    self.retina = SimpleNamespace(idx=np.array([0]))

  def set_w_scale(self, s):
    self.w_scale = s

  def reset(self):
    pass

  def run(self, img, ms):
    self.ms = ms

  def take_counts(self):
    rate = np.full(1000, 5.0)
    if self.w_scale > self.limit:
      if self.how == "mean":
        rate[:] = 25.0
      else:
        rate[1:11] = 150.0  # 1% of neurons over 100 Hz, mean still low
    rate[0] = 1e5
    return rate * self.ms / 1000, self.ms


def test_g0_doubles_then_bisects_when_quiet_at_full_scale(synth_cfg):
  b = FakeBrain(synth_cfg, limit=5.0)
  r = run_g0(b, QUIET)
  assert [p.scale for p in r.probes[:4]] == [1.0, 2.0, 4.0, 8.0]
  assert 5.0 * (1 - synth_cfg.calibration.g0_precision) < r.scale <= 5.0
  assert b.w_scale == r.scale


def test_g0_keeps_the_last_quiet_doubling_when_its_edge_is_close(synth_cfg):
  r = run_g0(FakeBrain(synth_cfg, limit=2.0), QUIET)
  assert [p.runaway for p in r.probes[:3]] == [False, False, True]
  assert 2.0 * (1 - synth_cfg.calibration.g0_precision) < r.scale <= 2.0


def test_g0_gives_up_when_no_scale_runs_away(synth_cfg):
  b = FakeBrain(synth_cfg, limit=math.inf)
  with pytest.raises(G0Failure, match="never runs away up to w_scale 256 after 8 doublings"):
    run_g0(b, QUIET)


@pytest.mark.parametrize("how", ["mean", "tail"])
def test_g0_halves_then_bisects_to_the_precision(synth_cfg, how):
  b = FakeBrain(synth_cfg, limit=0.3, how=how)
  r = run_g0(b, QUIET)
  assert [p.scale for p in r.probes[:3]] == [1.0, 0.5, 0.25]
  assert 0.3 * (1 - synth_cfg.calibration.g0_precision) < r.scale <= 0.3
  assert b.w_scale == r.scale


def test_g0_gives_up_after_the_halvings(synth_cfg):
  with pytest.raises(G0Failure, match="after 8 halvings"):
    run_g0(FakeBrain(synth_cfg, limit=0.0), QUIET)


def test_finite_reports_nan_as_nan_not_minus_inf():
  assert calibrate_mod._finite(math.nan) == "nan"
  assert calibrate_mod._finite(math.inf) == "inf"
  assert calibrate_mod._finite(-math.inf) == "-inf"
  assert calibrate_mod._finite(2.5) == 2.5


def test_z_score():
  assert z_score([1, 2, 3], [0, 0, 1]) == pytest.approx(2.5)
  assert z_score([1, 2, 3], [1, 2, 3]) == 0.0
  assert z_score([0, 0], [0, 0]) == 0.0
  assert z_score([2, 2], [1, 1]) == math.inf
  assert z_score([1, 1], [2, 2]) == -math.inf


def test_rungs_cover_the_ladder(synth_cfg):
  assert set(RUNGS) == set(LADDER_RUNGS)
  assert rung_config(synth_cfg, "flip_polarity").eye.polarity == "bright_on_dark"
  assert rung_config(synth_cfg, "tonic_50").retina.lamina_tonic_hz == 50.0
  assert rung_config(synth_cfg, "vpn").retina.mode == "photoreceptor+vpn"
  assert rung_config(synth_cfg, "default") == synth_cfg


SHORT = {"calibration.g0_blank_ms": 200.0, "calibration.g1_trial_ms": 200.0,
         "calibration.g1_repeats": 3, "calibration.sample_every_ms": 50.0,
         "calibration.sample_warmup_ms": 50.0,
         "calibration.g1_azimuths_deg": [-25.0, 0.0, 25.0]}  # synth LC10a sit at +-25


@pytest.fixture(scope="module")
def synth_brain(conn, synth_cfg):
  def make(changes):
    cfg = with_overrides(synth_cfg, {**SHORT, **changes})
    return Brain(conn, load_wiring(cfg, conn, log=QUIET), RetinaGeometry.build(conn), cfg)
  return make


def test_ladder_climbs_to_the_first_passing_rung(synth_brain):
  # Synth photoreceptors only inhibit silent lamina, so only VPN injection carries the spot.
  b = synth_brain({"calibration.ladder": ["default", "vpn"]})
  r = run_ladder(b, QUIET)
  assert r.passed and r.rung == "vpn"
  assert [t["passed"] for t in r.tried] == [False, True]
  # silent at every scale, so G0 finds no edge
  assert r.tried[0]["g0_failure"] == "brain never runs away up to w_scale 256 after 8 doublings"
  assert b.cfg.retina.mode == "photoreceptor+vpn" and b.w_scale == r.g0.scale > 1.0
  assert r.g1.z_lc10a > 3
  m = r.g1.means()
  assert m[25.0]["lc10a_R"] > m[25.0]["lc10a_L"] == 0.0
  assert m[-25.0]["lc10a_L"] > m[-25.0]["lc10a_R"] == 0.0
  assert r.g1.blank.shape == (3 * 3, 2) and r.g1.stim.shape == (3 * 3 * 3, 2)
  json.dumps(r.g1.to_dict())
  calib = calibrate_decoder(b, r.g1)
  assert b.decoder.calib is calib
  assert not calib.turn_silent and calib.k_turn > 0  # at its edge the synth DNa02 fire too
  assert np.isfinite(b.decide(np.full((30, 72), 160, np.uint8))[0])


def test_failed_ladder_restores_the_config(synth_brain):
  b = synth_brain({"calibration.ladder": ["default"], "calibration.g1_repeats": 2,
                   "calibration.g1_trial_ms": 100.0})
  before, before_w_scale = b.cfg, b.w_scale
  r = run_ladder(b, QUIET)
  assert not r.passed and r.rung is None and len(r.tried) == 1
  assert b.cfg == before
  assert b.w_scale == before_w_scale  # not left at the failed rung's G0 scale


def test_failed_rung_restores_w_scale_even_after_g0_changes_it(synth_brain, monkeypatch):
  """A rung can pass G0 (changing w_scale) and then fail G1; the ladder must still restore
  w_scale, not just cfg, when it gives up."""
  b = synth_brain({"calibration.ladder": ["default"]})
  before_w_scale = b.w_scale

  def fake_g0(brain, log):
    brain.set_w_scale(0.3)  # simulate G0 bisecting down from the brain's current scale
    return G0Result(0.3, ())

  def fake_g1(brain, rung, log):
    return G1Result(rung, {}, 0.0, 0.0, False, np.zeros((0, 2)), np.zeros((0, 2)))

  monkeypatch.setattr(calibrate_mod, "run_g0", fake_g0)
  monkeypatch.setattr(calibrate_mod, "measure_g1", fake_g1)
  r = run_ladder(b, QUIET)
  assert not r.passed
  assert b.w_scale == before_w_scale


def test_g0_failure_on_a_rung_moves_to_the_next(synth_brain, monkeypatch):
  b = synth_brain({"calibration.ladder": ["default"]})
  before = b.cfg

  def runaway(brain, log):
    raise G0Failure("runaway at every scale after 8 halvings")

  monkeypatch.setattr(calibrate_mod, "run_g0", runaway)
  r = run_ladder(b, QUIET)
  assert not r.passed and r.g0 is None and r.g1 is None
  assert r.tried == [{"rung": "default", "g0_scale": None, "z_lc10a": None, "z_dn": None,
                      "passed": False, "g0_failure": "runaway at every scale after 8 halvings"}]
  assert b.cfg == before
