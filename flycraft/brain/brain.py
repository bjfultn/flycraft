"""The brain: the LIF network, driven by the retina and read by the decoder (spec sections 5, 7).

Each run() holds one eye image fixed. Poisson input events for the whole run are drawn up front
in chunks, so the GPU syncs once per chunk instead of once per step.
"""

from __future__ import annotations

import numpy as np
import torch

from flycraft.brain.decoder import Calibration, Decoder, resolve_groups
from flycraft.brain.lif import LIF, LIFParams, steps
from flycraft.brain.retina import Retina, RetinaGeometry
from flycraft.brain.wiring import Wiring
from flycraft.config import Config
from flycraft.data.connectome import Connectome

CHUNK_STEPS = 1000  # Poisson draws per batch: 1000 x 4,400 inputs is about 18 MB of uniforms


class BrainError(RuntimeError):
  pass


def resolve_device(name: str) -> str:
  if name == "auto":
    return "cuda" if torch.cuda.is_available() else "cpu"
  if name == "cuda" and not torch.cuda.is_available():
    raise BrainError("sim.device is cuda but torch sees no CUDA device")
  return name


class Brain:
  def __init__(self, conn: Connectome, wiring: Wiring, geom: RetinaGeometry, cfg: Config,
               w_scale: float = 1.0, calib: Calibration | None = None):
    sim = cfg.sim
    self.conn, self.geom, self.cfg = conn, geom, cfg
    self.dt = sim.dt_ms
    self.device = resolve_device(sim.device)
    dtype = torch.float64 if sim.dtype == "float64" else torch.float32
    params = LIFParams(dt_ms=sim.dt_ms, f_poi=sim.f_poi)
    weight = conn.sign[wiring.pre].astype(np.float64) * wiring.syn * params.w_syn
    self.retina = Retina(geom, cfg.retina, cfg.eye)
    self.lif = LIF(conn.n, wiring.pre, wiring.post, weight, params,
                   rfc_zero=self.retina.rfc_zero_idx, device=self.device, dtype=dtype,
                   mode=sim.mode)
    self.gen = torch.Generator(device=self.device)
    self.gen.manual_seed(sim.seed)
    self._counts = torch.zeros(conn.n, dtype=torch.int64, device=self.device)
    self._count_ms = 0.0
    self._set_decoder(calib)
    self.set_w_scale(w_scale)

  def _set_decoder(self, calib: Calibration | None) -> None:
    self.decoder = Decoder(resolve_groups(self.conn, self.cfg.decoder), self.cfg.decoder, calib,
                           self.dt)
    self._watch = torch.as_tensor(self.decoder.watch, device=self.device)

  def set_w_scale(self, scale: float) -> None:
    self.w_scale = float(scale)
    self.lif.set_weight_scale(self.w_scale)

  def edge_weights(self, edges: np.ndarray) -> np.ndarray:
    """Unscaled weights (mV) of these wiring edges."""
    return self.lif.edge_weights(edges).cpu().numpy().astype(np.float64)

  def set_edge_weights(self, edges: np.ndarray, weight_mV: np.ndarray) -> None:
    """Overwrite these wiring edges' unscaled weights; w_scale still applies on top."""
    self.lif.set_edge_weights(edges, weight_mV)

  def set_calibration(self, calib: Calibration) -> None:
    self.decoder.calib = calib

  def reconfigure(self, cfg: Config) -> None:
    """Adopt cfg's eye, retina, decoder and calibration settings. The network stays as built."""
    if cfg.sim != self.cfg.sim or cfg.connectome != self.cfg.connectome:
      raise ValueError("reconfigure cannot change the sim or connectome sections")
    self.cfg = cfg
    self.retina = Retina(self.geom, cfg.retina, cfg.eye)
    self.lif.set_rfc_zero(self.retina.rfc_zero_idx)
    self._set_decoder(None)

  def reset(self) -> None:
    """Fresh membrane, synapse and decoder state; the spike counters restart too."""
    self.lif.reset_state()
    self.decoder.reset()
    self.take_counts()

  def take_counts(self) -> tuple[np.ndarray, float]:
    """(spikes per neuron, brain ms) since the last call or reset."""
    counts, ms = self._counts.cpu().numpy().copy(), self._count_ms  # cpu() aliases on CPU
    self._counts.zero_()
    self._count_ms = 0.0
    return counts, ms

  def run(self, img: np.ndarray, ms: float) -> None:
    """Advance ms of brain time with the eye fixed on img."""
    idx, rate = self.retina.rates(img)
    n = steps(ms, self.dt)
    target = torch.as_tensor(idx, device=self.device)
    prob = torch.as_tensor(np.clip(rate * self.dt / 1000.0, 0.0, 1.0), dtype=torch.float32,
                           device=self.device)
    watch = torch.zeros((n, self._watch.numel()), dtype=torch.bool, device=self.device)
    for c0 in range(0, n, CHUNK_STEPS):
      c = min(CHUNK_STEPS, n - c0)
      hit = torch.rand((c, idx.size), generator=self.gen, device=self.device) < prob
      step_of, col = hit.nonzero(as_tuple=True)
      kicks = target[col]
      bounds = [0, *torch.bincount(step_of, minlength=c).cumsum(0).tolist()]
      for i in range(c):
        spk = self.lif.step(kicks[bounds[i]:bounds[i + 1]])
        self._counts += spk
        watch[c0 + i] = spk[self._watch]
    if not (torch.isfinite(self.lif.v).all() and torch.isfinite(self.lif.g).all()):
      raise BrainError(f"non-finite membrane state at step {self.lif.t}")
    self._count_ms += n * self.dt
    self.decoder.observe(watch.cpu().numpy())

  def decide(self, img: np.ndarray) -> tuple[float, float]:
    """One decision: decision_ms of brain time on img, then (dtheta_deg, speed)."""
    self.run(img, self.cfg.decoder.decision_ms)
    return self.decoder.command()
