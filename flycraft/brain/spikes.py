"""Compare two spike rasters neuron by neuron."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpikeMatch:
  matched: int
  n_ref: int
  n_out: int

  @property
  def frac_matched(self) -> float:
    return self.matched / self.n_ref if self.n_ref else 1.0

  @property
  def frac_extra(self) -> float:
    return (self.n_out - self.matched) / self.n_ref if self.n_ref else float(self.n_out > 0)


def match_spikes(ref_t, ref_i, out_t, out_i, n: int, tol: int = 1) -> SpikeMatch:
  """Greedy one-to-one match of spikes per neuron within +-tol steps."""
  ref_t, ref_i = np.asarray(ref_t), np.asarray(ref_i)
  out_t, out_i = np.asarray(out_t), np.asarray(out_i)
  matched = 0
  for k in range(n):
    a = np.sort(ref_t[ref_i == k])
    b = np.sort(out_t[out_i == k])
    i = j = 0
    while i < a.size and j < b.size:
      if abs(int(a[i]) - int(b[j])) <= tol:
        matched += 1
        i += 1
        j += 1
      elif a[i] < b[j]:
        i += 1
      else:
        j += 1
  return SpikeMatch(matched, int(ref_t.size), int(out_t.size))
