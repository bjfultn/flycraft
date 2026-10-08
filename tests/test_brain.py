import math

import numpy as np
import pytest
import torch

from flycraft import eye
from flycraft.brain.brain import Brain, BrainError, resolve_device
from flycraft.brain.retina import RetinaGeometry
from flycraft.brain.wiring import load_wiring
from flycraft.config import with_overrides


@pytest.fixture(scope="module")
def geom(conn):
  return RetinaGeometry.build(conn)


@pytest.fixture(scope="module")
def wiring(conn, synth_cfg):
  return load_wiring(synth_cfg, conn, log=lambda s: None)


@pytest.fixture(scope="module")
def make(conn, wiring, geom, synth_cfg):
  def _make(changes=None):
    cfg = with_overrides(synth_cfg, changes) if changes else synth_cfg
    return Brain(conn, wiring, geom, cfg)
  return _make


def test_same_seed_gives_the_same_spikes(make):
  img = eye.disk(30, 10, "dark_on_bright")
  runs = []
  for seed in (0, 0, 1):
    b = make({"sim.seed": seed})
    b.run(img, 100)
    runs.append(b.take_counts()[0])
  assert runs[0].sum() > 0
  np.testing.assert_array_equal(runs[0], runs[1])
  assert not np.array_equal(runs[0], runs[2])


def test_photoreceptors_fire_at_the_retina_rate(make, geom):
  b = make({"retina.eye_normalize": False})
  b.run(eye.grey(128), 1000)
  counts, ms = b.take_counts()
  assert ms == pytest.approx(1000)
  rate = counts[geom.pr_idx].mean() / (ms / 1000)
  assert rate == pytest.approx(150 * 128 / 255, rel=0.03)


@pytest.mark.parametrize("az, side", [(25, "R"), (-25, "L")])
def test_vpn_spot_drives_only_the_same_side_lc10a(make, geom, az, side):
  b = make({"retina.mode": "photoreceptor+vpn"})
  b.run(eye.disk(az, 10, "dark_on_bright", el_deg=8.5), 300)
  rate = b.take_counts()[0] / 0.3
  other = "L" if side == "R" else "R"
  assert rate[geom.vpn_idx[geom.vpn_side == side]].mean() > 5
  assert rate[geom.vpn_idx[geom.vpn_side == other]].sum() == 0


def test_take_counts_restarts_the_counters(make):
  b = make()
  b.run(eye.grey(), 50)
  counts, ms = b.take_counts()
  assert counts.sum() > 0 and ms == pytest.approx(50)
  counts, ms = b.take_counts()
  assert counts.sum() == 0 and ms == 0
  b.run(eye.grey(), 20)
  b.reset()
  assert b.take_counts()[0].sum() == 0


def test_decide_gives_a_finite_command(make):
  b = make()
  dtheta, speed = b.decide(eye.grey())
  assert math.isfinite(dtheta) and 0.0 <= speed <= 1.0
  assert (dtheta, speed) == (0.0, 0.5)  # uncalibrated decoder: zero gain, speed s0


def test_non_finite_state_raises(make):
  b = make()
  b.lif.v[0] = float("nan")
  with pytest.raises(BrainError, match="non-finite"):
    b.run(eye.grey(), 1)


def test_reconfigure_swaps_the_inputs_and_keeps_the_network(make, geom, synth_cfg):
  b = make()
  assert not np.isin(geom.vpn_idx, b.retina.idx).any()
  b.reconfigure(with_overrides(synth_cfg, {"retina.mode": "photoreceptor+vpn"}))
  assert np.isin(geom.vpn_idx, b.retina.idx).all()
  assert (b.lif.rfc[torch.as_tensor(geom.vpn_idx)] == 0).all()
  with pytest.raises(ValueError, match="sim or connectome"):
    b.reconfigure(with_overrides(synth_cfg, {"sim.seed": 5}))


def test_device_resolution():
  assert resolve_device("cpu") == "cpu"
  assert resolve_device("auto") == ("cuda" if torch.cuda.is_available() else "cpu")
  if not torch.cuda.is_available():
    with pytest.raises(BrainError, match="no CUDA device"):
      resolve_device("cuda")
