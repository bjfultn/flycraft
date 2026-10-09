import math
from pathlib import Path

import numpy as np
import pytest
import torch

from flycraft.brain.lif import LIF, LIFParams, steps
from flycraft.brain.spikes import match_spikes

FIXTURE = Path(__file__).parent / "fixtures" / "golden_lif.npz"
P = LIFParams()


def one_edge(w_mV, dtype=torch.float64, mode="event", rfc_zero=None):
  return LIF(2, [0], [1], [w_mV], P, rfc_zero=rfc_zero, dtype=dtype, mode=mode)


def kick(*idx):
  return torch.tensor(idx, dtype=torch.int64)


def run(lif, n_steps, kicks=None):
  kicks = kicks or {}
  t, i = [], []
  for n in range(n_steps):
    spk = lif.step(kicks.get(n))
    for k in torch.nonzero(spk).squeeze(1).tolist():
      t.append(n)
      i.append(k)
  return np.array(t, dtype=np.int64), np.array(i, dtype=np.int64)


def test_steps_and_constants():
  assert steps(1.8, 0.1) == 18 and steps(2.2, 0.1) == 22 and steps(400.0, 0.1) == 4000
  assert P.kick_mV == pytest.approx(68.75)


def test_kick_spikes_next_step_and_delivers_after_delay():
  lif = one_edge(1.0)
  spk_steps = []
  g1 = []
  for n in range(25):
    spk = lif.step(kick(0) if n == 0 else None)
    if spk[0]:
      spk_steps.append(n)
    g1.append(float(lif.g[1]))
  assert spk_steps == [1]
  assert g1[18] == 0.0 and g1[19] == 1.0  # spike at step 1 arrives at step 1 + 18


def test_subthreshold_response_is_analytic():
  w = 2.0  # mV into g; peak response stays far below threshold
  lif = one_edge(w)
  v = []
  for n in range(400):
    lif.step(kick(0) if n == 0 else None)
    v.append(float(lif.v[1]))
  tm, tau, dt = P.t_mbr, P.tau_syn, P.dt_ms
  for n in range(20, 400):
    k = n - 19  # g jumps at step 19; v integrates from step 20
    expect = P.v0 + w * tau / (tm - tau) * (math.exp(-k * dt / tm) - math.exp(-k * dt / tau))
    assert v[n] == pytest.approx(expect, abs=1e-9)
  assert v[19] == P.v0


def test_threshold_is_strict():
  lif = LIF(2, [0], [1], [0.0], P, dtype=torch.float64)
  lif.v[0] = -44.9
  lif.v[1] = -45.0
  lif.Pvv, lif.Pvg = 1.0, 0.0  # freeze the decay to test the comparison alone
  spk = lif.step()
  assert bool(spk[0]) and not bool(spk[1])  # v > v_th, not >=


def test_refractory_drops_input_and_spaces_spikes():
  lif = one_edge(0.0)
  every = {n: kick(0) for n in range(100)}
  t, i = run(lif, 100, every)
  assert t[i == 0].tolist() == [1, 24, 47, 70, 93]


def test_zero_refractory_spikes_every_other_step():
  lif = one_edge(0.0, rfc_zero=[0])
  every = {n: kick(0) for n in range(12)}
  t, i = run(lif, 12, every)
  assert t[i == 0].tolist() == [1, 3, 5, 7, 9, 11]


def test_weight_scale():
  lif = one_edge(3.0)
  lif.set_weight_scale(0.5)
  for n in range(20):
    lif.step(kick(0) if n == 0 else None)
  assert float(lif.g[1]) == pytest.approx(1.5)


def test_reset_state():
  lif = one_edge(3.0)
  run(lif, 30, {0: kick(0)})
  lif.reset_state()
  assert lif.t == 0 and (lif.v == P.v0).all() and (lif.g == 0).all() and not lif.ring.any()


def test_slot_of_edge_addresses_csr_values():
  pre, post = np.array([2, 0, 1, 0]), np.array([0, 1, 2, 2])
  w = np.array([1.0, 2.0, 3.0, 4.0])
  lif = LIF(3, pre, post, w, P, dtype=torch.float64)
  np.testing.assert_array_equal(lif.val[lif.slot_of_edge].numpy(), w)
  np.testing.assert_array_equal(lif.col[lif.slot_of_edge].numpy(), post)


def fixture_run(dtype, mode):
  fx = np.load(FIXTURE)
  lif = LIF(50, fx["pre"], fx["post"], fx["w_mV"], P, rfc_zero=fx["rfc_zero"], dtype=dtype,
            mode=mode)
  kicks = {}
  for s, k in zip(fx["kick_steps"].tolist(), fx["kick_targets"].tolist(), strict=True):
    kicks.setdefault(s, []).append(k)
  kicks = {s: torch.tensor(v, dtype=torch.int64) for s, v in kicks.items()}
  t, i = run(lif, int(fx["n_steps"]), kicks)
  return fx, t, i


@pytest.mark.parametrize("mode", ["event", "spmv"])
def test_golden_fixture_float64_is_exact(mode):
  fx, t, i = fixture_run(torch.float64, mode)
  ref = sorted(zip(fx["ref_t"].tolist(), fx["ref_i"].tolist(), strict=True))
  assert sorted(zip(t.tolist(), i.tolist(), strict=True)) == ref
  assert len(ref) >= 300


def test_golden_fixture_float32_within_one_step():
  fx, t, i = fixture_run(torch.float32, "event")
  m = match_spikes(fx["ref_t"], fx["ref_i"], t, i, 50, tol=1)
  assert m.frac_matched >= 0.99
  assert m.frac_extra <= 0.01


def test_event_and_spmv_agree_on_a_random_network():
  rng = np.random.default_rng(3)
  n = 200
  mask = rng.random((n, n)) < 0.05
  np.fill_diagonal(mask, False)
  pre, post = np.nonzero(mask)
  w = np.where(rng.random(n) < 0.7, 1, -1)[pre] * rng.integers(1, 40, pre.size) * P.w_syn
  kicks = {s: torch.tensor(rng.choice(n, 5, replace=False)) for s in range(0, 2000, 3)}
  out = [run(LIF(n, pre, post, w, P, dtype=torch.float64, mode=m), 2000, kicks)
         for m in ("event", "spmv")]
  np.testing.assert_array_equal(out[0][0], out[1][0])
  np.testing.assert_array_equal(out[0][1], out[1][1])
  assert out[0][0].size > 100


def test_match_spikes():
  m = match_spikes([5, 10, 20], [0, 0, 1], [6, 10, 22, 30], [0, 0, 1, 1], 2, tol=1)
  assert (m.matched, m.n_ref, m.n_out) == (2, 3, 4)
  assert m.frac_matched == pytest.approx(2 / 3)
  assert m.frac_extra == pytest.approx(2 / 3)


@pytest.mark.gpu
def test_gpu_matches_cpu_float64():
  if not torch.cuda.is_available():
    pytest.skip("no CUDA device")
  fx, t_cpu, i_cpu = fixture_run(torch.float64, "event")
  lif = LIF(50, fx["pre"], fx["post"], fx["w_mV"], P, rfc_zero=fx["rfc_zero"],
            dtype=torch.float64, device="cuda")
  kicks = {}
  for s, k in zip(fx["kick_steps"].tolist(), fx["kick_targets"].tolist(), strict=True):
    kicks.setdefault(s, []).append(k)
  kicks = {s: torch.tensor(v, dtype=torch.int64, device="cuda") for s, v in kicks.items()}
  t, i = run(lif, int(fx["n_steps"]), kicks)
  np.testing.assert_array_equal(t, t_cpu)
  np.testing.assert_array_equal(i, i_cpu)


@pytest.mark.parametrize("mode", ["event", "spmv"])
def test_set_edge_weights_matches_a_network_built_with_them(mode):
  rng = np.random.default_rng(4)
  n = 200
  mask = rng.random((n, n)) < 0.05
  np.fill_diagonal(mask, False)
  pre, post = np.nonzero(mask)
  w = np.where(rng.random(n) < 0.7, 1, -1)[pre] * rng.integers(1, 40, pre.size) * P.w_syn
  edges = rng.choice(pre.size, pre.size // 4, replace=False)
  w2 = w.copy()
  w2[edges] *= 0.5
  kicks = {s: torch.tensor(rng.choice(n, 5, replace=False)) for s in range(0, 2000, 3)}
  edited = LIF(n, pre, post, w, P, dtype=torch.float64, mode=mode)
  edited.set_weight_scale(1.5)
  np.testing.assert_array_equal(edited.edge_weights(edges).numpy(), w[edges])
  edited.set_edge_weights(edges, w[edges] * 0.5)
  np.testing.assert_array_equal(edited.edge_weights(edges).numpy(), w2[edges])
  built = LIF(n, pre, post, w2, P, dtype=torch.float64, mode=mode)
  built.set_weight_scale(1.5)
  a, b = run(edited, 2000, kicks), run(built, 2000, kicks)
  np.testing.assert_array_equal(a[0], b[0])
  np.testing.assert_array_equal(a[1], b[1])
  assert a[0].size > 100
  edited.set_weight_scale(1.0)  # a later global scale keeps the edit
  np.testing.assert_array_equal(edited.val[edited.slot_of_edge].numpy(), w2)
