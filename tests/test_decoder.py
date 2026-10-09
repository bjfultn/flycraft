import dataclasses
import math

import numpy as np
import pytest

from flycraft.brain.decoder import (
  Calibration,
  Decoder,
  DecoderError,
  RateFilter,
  calibrate,
  resolve_groups,
)
from flycraft.config import DecoderConfig

CFG = DecoderConfig()
T_GAME = 8 / 22.4


def test_default_groups_resolve(conn):
  g = resolve_groups(conn, CFG)
  assert list(conn.type[g.turn_pos]) == ["DNa02"] and list(conn.side[g.turn_pos]) == ["R"]
  assert list(conn.type[g.turn_neg]) == ["DNa02"] and list(conn.side[g.turn_neg]) == ["L"]
  assert sorted(conn.side[g.fwd]) == ["L", "R"]
  assert list(conn.type[g.pitch_pos]) == ["DNbe007", "DNbe007"]
  assert list(conn.side[g.pitch_pos]) == ["L", "R"]
  assert list(conn.type[g.pitch_neg]) == ["DNge043", "DNge043"]
  assert list(conn.side[g.pitch_neg]) == ["L", "R"]
  assert g.aotu_pos.size == g.aotu_neg.size == 0
  assert g.mdn.size == 4
  assert g.watch.size == 12 and (np.diff(g.watch) > 0).all()


def test_missing_types_fail_loudly_and_name_them(conn):
  cfg = dataclasses.replace(CFG, turn_pos=(("DNx99", "R"),), fwd=(("DNp09", "M"),))
  with pytest.raises(DecoderError, match=r"DNx99 \(R\), DNp09 \(M\)"):
    resolve_groups(conn, cfg)


def test_aotu019_term(conn):
  g = resolve_groups(conn, dataclasses.replace(CFG, aotu019_term=True))
  assert list(conn.side[g.aotu_pos]) == ["R"] and list(conn.side[g.aotu_neg]) == ["L"]


def test_rate_filter_steady_state_and_decay():
  f = RateFilter(1, tau_ms=150.0, dt_ms=0.1)
  spikes = np.zeros((40_000, 1), bool)
  spikes[::200] = True  # 50 Hz; the last spike is 199 steps before the end
  r = f.update(spikes)[0]
  a = math.exp(-0.1 / 150.0)
  assert r == pytest.approx(1000.0 / 150.0 * a**199 / (1 - a**200))  # sawtooth trough
  assert 45.0 < r < 50.0  # the sawtooth averages 50 Hz
  r2 = f.update(np.zeros((500, 1), bool))[0]
  assert r2 == pytest.approx(r * math.exp(-50.0 / 150.0))


def test_rate_filter_chunking_does_not_matter():
  rng = np.random.default_rng(0)
  spikes = rng.random((3000, 3)) < 0.01
  a, b = RateFilter(3, 150.0, 0.1), RateFilter(3, 150.0, 0.1)
  a.update(spikes)
  for chunk in np.split(spikes, 6):
    b.update(chunk)
  np.testing.assert_allclose(a.r, b.r)


def test_calibration_on_synthetic_rates():
  rng = np.random.default_rng(1)
  blank = np.c_[rng.normal(2.0, 0.1, 200), rng.normal(10.0, 0.1, 200)]
  stim = np.c_[rng.uniform(-8, 12, 600), rng.uniform(10, 30, 600)]
  c = calibrate(blank, stim, CFG)
  assert c.b_turn == pytest.approx(2.0, abs=0.05) and c.b_fwd == pytest.approx(10.0, abs=0.05)
  assert c.p95_turn_hz == pytest.approx(np.percentile(np.abs(stim[:, 0] - c.b_turn), 95))
  assert c.k_turn * c.p95_turn_hz == pytest.approx(180.0)
  assert c.k_fwd * c.p95_fwd_hz == pytest.approx(0.5)
  assert not c.turn_silent and not c.fwd_silent


@pytest.mark.parametrize("blank, stim", [
  (np.zeros((50, 2)), np.zeros((50, 2))),
  (np.full((50, 2), 3.0), np.full((50, 2), 3.5)),
])
def test_gain_floor_gives_zero_gain_and_finite_commands(conn, blank, stim):
  c = calibrate(blank, stim, CFG)
  assert c.k_turn == 0.0 and c.k_fwd == 0.0
  assert c.turn_silent and c.fwd_silent
  d = Decoder(resolve_groups(conn, CFG), CFG, c)
  d.observe(np.ones((500, d.watch.size), bool))
  dtheta, speed = d.command()
  assert dtheta == 0.0 and speed == 0.5


def _spikes(d, hz):
  """Regular spike trains for 1 s: hz maps neuron index -> rate."""
  s = np.zeros((10_000, d.watch.size), bool)
  for i, rate in hz.items():
    if rate:
      s[:: int(10_000 / rate), np.searchsorted(d.watch, i)] = True
  return s


def test_right_dna02_turns_right_and_command_clips(conn):
  g = resolve_groups(conn, CFG)
  calib = Calibration(k_turn=180.0 / 10.0, k_fwd=0.5 / 20.0, turn_silent=False,
                      fwd_silent=False)
  d = Decoder(g, CFG, calib)
  d.observe(_spikes(d, {g.turn_pos[0]: 25, g.turn_neg[0]: 20}))
  turn, _ = d.signals()
  assert turn == pytest.approx(5.0, abs=0.6)
  dtheta, speed = d.command()
  assert dtheta == pytest.approx(calib.k_turn * turn * T_GAME)
  assert dtheta > 0 and speed == 0.5
  d.reset()
  d.observe(_spikes(d, {g.turn_neg[0]: 100, g.fwd[0]: 100, g.fwd[1]: 100}))
  dtheta, speed = d.command()
  assert dtheta == pytest.approx(-180.0 * T_GAME)
  assert speed == 1.0


def test_pitch_is_the_bias_subtracted_scaled_group_difference(conn):
  g = resolve_groups(conn, CFG)
  d = Decoder(g, CFG)
  d.observe(_spikes(d, {g.pitch_pos[0]: 25, g.pitch_pos[1]: 25, g.pitch_neg[0]: 20,
                       g.pitch_neg[1]: 20}))
  pitch, raw = d.pitch()
  assert raw == pytest.approx(5.0, abs=0.6)
  assert pitch == pytest.approx((raw - CFG.pitch_bias_hz) / CFG.pitch_scale_hz)


def test_pitch_clips_at_plus_and_minus_one(conn):
  g = resolve_groups(conn, CFG)
  d = Decoder(g, CFG)
  d.observe(_spikes(d, {g.pitch_pos[0]: 100, g.pitch_pos[1]: 100}))
  pitch, _ = d.pitch()
  assert pitch == 1.0
  d.reset()
  d.observe(_spikes(d, {g.pitch_neg[0]: 100, g.pitch_neg[1]: 100}))
  pitch, _ = d.pitch()
  assert pitch == -1.0
