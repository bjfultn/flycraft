"""Shiu et al. 2024 leaky integrate-and-fire network, matching Brian2 2.9.0 step for step.

    dv/dt = (v0 - v + g) / t_mbr   (unless refractory)
    dg/dt = -g / tau_syn           (unless refractory)

Per step n, in this order (tests/fixtures/golden_lif.npz is the arbiter):
  1. nr = neurons out of refractory: (n - last_spike) >= rfc_steps
  2. where nr: exact linear update of v and g over one dt
  3. spk = (v > v_th) & nr; last_spike[spk] = n
  4. acc = nr & ~spk: only these accept input this step. Brian2's `(unless refractory)`
     puts a conditional write on v and g, so input to a refractory or spiking cell is dropped.
  5. g += weights of spikes emitted delay_steps ago; v += kick_mV per Poisson input event
  6. spikers: v = v_rst, g = 0
  7. remember spk in the delay ring
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class LIFParams:
  dt_ms: float = 0.1
  v0: float = -52.0
  v_rst: float = -52.0
  v_th: float = -45.0
  t_mbr: float = 20.0
  tau_syn: float = 5.0
  t_rfc: float = 2.2
  t_dly: float = 1.8
  w_syn: float = 0.275
  f_poi: float = 250.0

  @property
  def kick_mV(self) -> float:
    """Voltage step per Poisson input event: F_POI * W_SYN, added to v."""
    return self.f_poi * self.w_syn


def steps(ms: float, dt_ms: float) -> int:
  """Brian2's conversion of a duration to whole steps."""
  return int(math.floor(ms / dt_ms + 1e-3))


class LIF:
  def __init__(self, n, pre, post, weight_mV, params: LIFParams | None = None, rfc_zero=None,
               device="cpu", dtype=torch.float32, mode="event"):
    params = params or LIFParams()
    if mode not in ("event", "spmv"):
      raise ValueError(f"mode must be event or spmv, got {mode!r}")
    self.n, self.p, self.mode = int(n), params, mode
    self.dev, self.dtype = torch.device(device), dtype
    pre = torch.as_tensor(np.asarray(pre), dtype=torch.int64)
    post = torch.as_tensor(np.asarray(post), dtype=torch.int64)
    w = torch.as_tensor(np.asarray(weight_mV, dtype=np.float64), dtype=dtype)
    order = torch.argsort(pre, stable=True)
    self.slot_of_edge = torch.empty_like(order)  # edge i lives at CSR slot slot_of_edge[i]
    self.slot_of_edge[order] = torch.arange(order.numel())
    self._pre, self._post = pre.to(self.dev), post.to(self.dev)
    self.col = post[order].to(self.dev)
    self.base = w[order].to(self.dev)
    counts = torch.bincount(pre, minlength=self.n)
    self.rowptr = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)]).to(self.dev)
    self.delay = steps(params.t_dly, params.dt_ms)
    self.rfc_steps = steps(params.t_rfc, params.dt_ms)
    dt = params.dt_ms
    self.Pvv = math.exp(-dt / params.t_mbr)
    self.Pgg = math.exp(-dt / params.tau_syn)
    self.Pvg = params.tau_syn / (params.t_mbr - params.tau_syn) * (self.Pvv - self.Pgg)
    self.rfc = torch.full((self.n,), self.rfc_steps, dtype=torch.int64, device=self.dev)
    if rfc_zero is not None:
      self.set_rfc_zero(rfc_zero)
    self.set_weight_scale(1.0)
    self.reset_state()

  def set_weight_scale(self, scale: float) -> None:
    self.scale = float(scale)
    self.val = self.base * self.scale
    if self.mode == "spmv":
      w = self.val[self.slot_of_edge.to(self.dev)]
      idx = torch.stack([self._post, self._pre])
      coo = torch.sparse_coo_tensor(idx, w, (self.n, self.n), check_invariants=False).coalesce()
      with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
        self.W = coo.to_sparse_csr()

  def edge_weights(self, edges) -> torch.Tensor:
    """Unscaled weights (mV) of these edges, indexed as the edges given to the constructor."""
    return self.base[self.slot_of_edge[torch.as_tensor(np.asarray(edges))].to(self.dev)]

  def set_edge_weights(self, edges, weight_mV) -> None:
    """Overwrite these edges' unscaled weights in place; the global scale still applies."""
    slots = self.slot_of_edge[torch.as_tensor(np.asarray(edges))].to(self.dev)
    w = torch.as_tensor(weight_mV, dtype=self.dtype, device=self.dev)
    self.base[slots] = w
    if self.mode == "spmv":
      self.set_weight_scale(self.scale)  # rebuilds the matrix
    else:
      self.val[slots] = self.base[slots] * self.scale

  def set_rfc_zero(self, idx) -> None:
    """Neurons in idx get no refractory period (Shiu: Poisson input targets)."""
    self.rfc.fill_(self.rfc_steps)
    idx = torch.as_tensor(np.asarray(idx), dtype=torch.int64, device=self.dev)
    self.rfc[idx] = 0

  def reset_state(self) -> None:
    self.v = torch.full((self.n,), self.p.v0, dtype=self.dtype, device=self.dev)
    self.g = torch.zeros(self.n, dtype=self.dtype, device=self.dev)
    self.last = torch.full((self.n,), -(10**9), dtype=torch.int64, device=self.dev)
    self.ring = torch.zeros((self.delay, self.n), dtype=torch.bool, device=self.dev)
    self.t = 0

  def _deliver(self, d: torch.Tensor) -> torch.Tensor:
    if self.mode == "spmv":
      return (self.W @ d.to(self.dtype).unsqueeze(1)).squeeze(1)
    inc = torch.zeros(self.n, dtype=self.dtype, device=self.dev)
    src = torch.nonzero(d).squeeze(1)
    if src.numel() == 0:
      return inc
    starts = self.rowptr[src]
    counts = self.rowptr[src + 1] - starts
    total = int(counts.sum())
    if total == 0:
      return inc
    offs = torch.repeat_interleave(starts - counts.cumsum(0) + counts, counts, output_size=total)
    eid = offs + torch.arange(total, device=self.dev)
    return inc.index_add_(0, self.col[eid], self.val[eid])

  def step(self, kick_idx: torch.Tensor | None = None) -> torch.Tensor:
    """Advance one dt. kick_idx: neuron indices receiving one Poisson event each (repeats add).

    Returns the bool spike vector for this step.
    """
    p, n = self.p, self.t
    nr = (n - self.last) >= self.rfc
    v = torch.where(nr, p.v0 + (self.v - p.v0) * self.Pvv + self.g * self.Pvg, self.v)
    g = torch.where(nr, self.g * self.Pgg, self.g)
    spk = (v > p.v_th) & nr
    self.last[spk] = n
    acc = nr & ~spk
    slot = n % self.delay
    zero = torch.zeros((), dtype=self.dtype, device=self.dev)
    g = g + torch.where(acc, self._deliver(self.ring[slot]), zero)
    if kick_idx is not None and kick_idx.numel():
      kv = torch.zeros(self.n, dtype=self.dtype, device=self.dev)
      kv.index_add_(0, kick_idx, torch.full(kick_idx.shape, p.kick_mV, dtype=self.dtype,
                                            device=self.dev))
      v = v + torch.where(acc, kv, zero)
    self.v = torch.where(spk, torch.full((), p.v_rst, dtype=self.dtype, device=self.dev), v)
    self.g = torch.where(spk, zero, g)
    self.ring[slot] = spk
    self.t += 1
    return spk
