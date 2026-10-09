"""The dopamine map (spec section 8): which dopamine neurons teach which MBONs.

M[d, m] is the synapse count from DAN d to MBON m in the REAL connectome, normalized per MBON,
so every twin is taught through the same map. An MBON sits in a PAM compartment when most of
its DAN input comes from PAM neurons; those are the MBONs a reward depresses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flycraft.brain.wiring import MBON_PATTERN, Wiring
from flycraft.data.connectome import Connectome

PAM_PATTERN = r"^PAM"
PPL1_PATTERN = r"^PPL10[1-8]"


@dataclass(frozen=True, eq=False)
class DopamineMap:
  dans: np.ndarray  # neuron index (D,): PAM first, then PPL101-108
  pam: np.ndarray  # bool (D,)
  mbons: np.ndarray  # neuron index (B,)
  m: np.ndarray  # float64 (D, B): DAN -> MBON synapses; each column sums to 1, or 0 if no input

  @property
  def pam_frac(self) -> np.ndarray:
    """Per MBON, the fraction of its DAN input that comes from PAM (0 with no DAN input)."""
    return self.m[self.pam].sum(axis=0)

  def pam_mbons(self) -> np.ndarray:
    """Neuron indices of the MBONs in a PAM compartment: more than half their DAN input is
    PAM."""
    return self.mbons[self.pam_frac > 0.5]


def dopamine_map(conn: Connectome) -> DopamineMap:
  """Built from conn's own edges; pass the real connectome, never a twin's wiring."""
  pam, ppl1 = conn.select(PAM_PATTERN), conn.select(PPL1_PATTERN)
  dans = np.concatenate([pam, ppl1])
  mbons = conn.select(MBON_PATTERN)
  row = np.full(conn.n, -1, np.int64)
  row[dans] = np.arange(dans.size)
  col = np.full(conn.n, -1, np.int64)
  col[mbons] = np.arange(mbons.size)
  r, c = row[conn.pre], col[conn.post]
  hit = (r >= 0) & (c >= 0)
  m = np.zeros((dans.size, mbons.size))
  np.add.at(m, (r[hit], c[hit]), conn.syn[hit])
  total = m.sum(axis=0)
  m = np.divide(m, total, out=np.zeros_like(m), where=total > 0)
  is_pam = np.zeros(dans.size, bool)
  is_pam[:pam.size] = True
  return DopamineMap(dans, is_pam, mbons, m)


def compartment_edges(wiring: Wiring, mbons: np.ndarray, n: int) -> np.ndarray:
  """Indices of the plastic (KC -> MBON) edges onto these MBONs."""
  onto = np.zeros(n, bool)
  onto[mbons] = True
  return np.flatnonzero(wiring.exempt & onto[wiring.post])
