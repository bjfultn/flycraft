"""Brian2 reference run for the LIF golden fixture.

Builds a 50-neuron network with Shiu et al. 2024's equations and constants,
drives 5 of its neurons with a fixed Bernoulli kick train, runs it in Brian2
with numpy codegen, and saves the spikes to tests/fixtures/golden_lif.npz.
CI never runs this; it only reads the committed fixture.

Run from the repo root:

  docker run --rm -v "$PWD":/w -w /w python:3.11-slim \
    sh -c "pip install -q brian2==2.9.0 'numpy<2' && python tools/make_golden.py"
"""

from pathlib import Path

import brian2 as b2
import numpy as np
from brian2 import ms, mV

N = 50
P_CONNECT = 0.2
N_KICKED = 5
KICK_RATE_HZ = 150.0
DT_MS = 0.1
W_SYN = 0.275
F_POI = 250.0
N_STEPS = 4000  # 400 ms
MIN_SPIKES = 300
MIN_SPIKING = 20
MAX_MEAN_HZ = 80.0  # non-kicked cells; keeps the network below saturation
SYN_RANGE = (5, 60)
OUT = Path("tests/fixtures/golden_lif.npz")


def make_network(seed):
  rng = np.random.default_rng(seed)
  mask = rng.random((N, N)) < P_CONNECT
  np.fill_diagonal(mask, False)
  pre, post = np.nonzero(mask)
  syn = rng.integers(SYN_RANGE[0], SYN_RANGE[1] + 1, size=pre.size)
  sign = np.where(rng.random(N) < 0.7, 1, -1)
  w_mV = sign[pre] * syn * W_SYN
  kicked = np.sort(rng.choice(N, N_KICKED, replace=False))
  hits = rng.random((N_STEPS, N_KICKED)) < KICK_RATE_HZ * DT_MS / 1000.0
  kick_steps, kick_col = np.nonzero(hits)
  return pre, post, w_mV, kicked, kick_steps, kick_col


def run(seed):
  pre, post, w_mV, kicked, kick_steps, kick_col = make_network(seed)
  b2.start_scope()
  b2.prefs.codegen.target = "numpy"
  b2.defaultclock.dt = DT_MS * ms
  eqs = """
  dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
  dg/dt = -g / tau              : volt (unless refractory)
  rfc                           : second
  """
  ns = dict(v_0=-52 * mV, v_rst=-52 * mV, v_th=-45 * mV, t_mbr=20 * ms, tau=5 * ms,
            kick=F_POI * W_SYN * mV)
  neu = b2.NeuronGroup(N, eqs, threshold="v > v_th", reset="v = v_rst; g = 0 * mV",
                       refractory="rfc", method="linear", namespace=ns)
  neu.v = -52 * mV
  neu.g = 0 * mV
  neu.rfc = 2.2 * ms
  neu.rfc[kicked] = 0 * ms  # Shiu: Poisson targets have no refractory period
  syn = b2.Synapses(neu, neu, "w : volt", on_pre="g += w", delay=1.8 * ms, namespace=ns)
  syn.connect(i=pre, j=post)
  syn.w = w_mV * mV
  gen = b2.SpikeGeneratorGroup(N_KICKED, kick_col, kick_steps * DT_MS * ms)
  kick = b2.Synapses(gen, neu, on_pre="v_post += kick", namespace=ns)
  kick.connect(i=np.arange(N_KICKED), j=kicked)
  mon = b2.SpikeMonitor(neu)
  net = b2.Network(neu, syn, gen, kick, mon)
  net.run(N_STEPS * DT_MS * ms)
  ref_t = np.round(np.asarray(mon.t / ms) / DT_MS).astype(np.int64)
  ref_i = np.asarray(mon.i).astype(np.int64)
  return dict(pre=pre.astype(np.int32), post=post.astype(np.int32), w_mV=w_mV.astype(np.float64),
              kicked=kicked.astype(np.int64), rfc_zero=kicked.astype(np.int64),
              kick_steps=kick_steps.astype(np.int64),
              kick_targets=kicked[kick_col].astype(np.int64),
              n_steps=np.int64(N_STEPS), ref_t=ref_t, ref_i=ref_i,
              brian2_version=np.array(b2.__version__), seed=np.int64(seed))


def main():
  for seed in range(100):
    fx = run(seed)
    n_spk = fx["ref_t"].size
    n_active = np.unique(fx["ref_i"]).size
    counts = np.bincount(fx["ref_i"], minlength=N)
    free = np.setdiff1d(np.arange(N), fx["kicked"])
    mean_hz = counts[free].mean() / (N_STEPS * DT_MS / 1000.0)
    print(f"seed {seed}: {n_spk} spikes, {n_active} spiking neurons, "
          f"non-kicked mean {mean_hz:.1f} Hz", flush=True)
    if n_spk >= MIN_SPIKES and n_active >= MIN_SPIKING and mean_hz <= MAX_MEAN_HZ:
      OUT.parent.mkdir(parents=True, exist_ok=True)
      np.savez(OUT, **fx)
      print(f"wrote {OUT}")
      return
  raise SystemExit("no seed met the spike criteria")


if __name__ == "__main__":
  main()
