"""A tiny synthetic MaleCNS (about 800 neurons) in the real feather schema.

Geometry: each eye has 25 columns (hex 1..5 x 1..5). Lamina (L1-L5) and Mi1 somata sit
at affine functions of hex. Mi1 z runs opposite to L1 z on both hex axes, like the real
optic chiasm. The right eye has twice as many R1-R6 per column as the left, like the real
under-reconstructed left eye. A LC10a -> AOTU019 -> DNa02 chain per side gives the brain
tests a working steering circuit, and KC -> MBON edges form the exempt set.
"""

from __future__ import annotations

import base64
from pathlib import Path

import google_crc32c
import numpy as np
import pandas as pd

from flycraft.data.fetch import ANNOTATIONS, NEUROTRANSMITTERS, WEIGHTS

HEX = [(h1, h2) for h1 in range(1, 6) for h2 in range(1, 6)]
R16_PER_COL = {"R": 4, "L": 2}


def _crc(path: Path) -> str:
  return base64.b64encode(google_crc32c.Checksum(path.read_bytes()).digest()).decode()


class _Builder:
  def __init__(self, seed):
    self.rng = np.random.default_rng(seed)
    self.rows, self.nts, self.edges = [], {}, {}
    self.next_id = 10_000

  def add(self, type_, superclass, side=None, soma_um=None, hex_=None, nt="acetylcholine",
          predicted=None, root_side=None, cls=None):
    bid = self.next_id
    self.next_id += 7
    self.rows.append({
      "bodyId": bid, "type": type_, "superclass": superclass, "class": cls,
      "somaSide": side, "rootSide": root_side if root_side is not None else side,
      "somaLocation": None if soma_um is None else [int(round(v / 0.008)) for v in soma_um],
      "assignedOlHex1": np.nan if hex_ is None else float(hex_[0]),
      "assignedOlHex2": np.nan if hex_ is None else float(hex_[1]),
    })
    self.nts[bid] = (nt, predicted if predicted is not None else nt)
    return bid

  def edge(self, a, b, syn):
    self.edges[(a, b)] = self.edges.get((a, b), 0) + int(syn)


def make_tiny_connectome(out_dir: Path, seed: int = 0, mi1_noise_um: float = 0.0) -> dict:
  """Write the three feathers into out_dir; return {"pins": {name: crc32c}}."""
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  b = _Builder(seed)
  rng = b.rng
  cells = {}
  for s, sx in (("R", 1.0), ("L", -1.0)):
    lam = {}
    mi1 = {}
    for h1, h2 in HEX:
      y = 100 + 5 * h1 + 5 * h2
      for k, (ty, nt) in enumerate((("L1", "glutamate"), ("L2", "acetylcholine"),
                                    ("L3", "acetylcholine"), ("L4", "acetylcholine"),
                                    ("L5", "gaba"))):
        z = 200 + 3 * h1 - 3 * h2 + k
        lam[(ty, h1, h2)] = b.add(ty, "ol_intrinsic", s, (sx * (300 + 2 * h1), y + k, z),
                                  (h1, h2), nt)
      z = 300 - 8 * h1 + 8 * h2 + rng.normal(0, mi1_noise_um)
      mi1[(h1, h2)] = b.add("Mi1", "ol_intrinsic", s, (sx * 250, 120 + 5 * h1 + 5 * h2, z),
                            (h1, h2))
      b.edge(lam[("L1", h1, h2)], mi1[(h1, h2)], 25)
      b.edge(lam[("L2", h1, h2)], mi1[(h1, h2)], 10)
    dm = b.add("Dm9", "ol_intrinsic", s, (sx * 240, 140, 300), None)
    # Photoreceptors: no soma, no hex; side only from rootSide.
    for h1, h2 in HEX:
      for _ in range(R16_PER_COL[s]):
        r = b.add("R1-R6", "sensory", None, None, None, "histamine", root_side=s)
        b.edge(r, lam[("L1", h1, h2)], 40)
        b.edge(r, lam[("L2", h1, h2)], 30)
        b.edge(r, lam[("L3", h1, h2)], 20)
        if h1 < 5:
          b.edge(r, lam[("L1", h1 + 1, h2)], 8)  # minority vote for a neighbour column
      for ty in ("R7p", "R8y"):
        r = b.add(ty, "sensory", None, None, None, "histamine", root_side=s)
        b.edge(r, mi1[(h1, h2)], 20)
        b.edge(r, lam[("L3", 6 - h1, 6 - h2)], 30)  # decoy: lamina partners never vote
        b.edge(r, dm, 15)  # partner without hex
    lost = b.add("R1-R6", "sensory", None, None, None, "histamine", root_side=s)
    b.edge(lost, dm, 10)  # no hex-assigned partner: unassigned
    # Steering chain.
    aotu = b.add("AOTU019", "cb_intrinsic", s, (sx * 120, 80, 150), None)
    dna02 = b.add("DNa02", "descending", s, (sx * 60, 200, 200), None)
    dnp09 = b.add("DNp09", "descending", s, (sx * 70, 210, 200), None)
    b.edge(aotu, dna02, 60)
    for q, (lo1, lo2) in enumerate(((1, 1), (1, 3), (3, 1), (3, 3))):
      lc = b.add("LC10a", "visual_projection", s, (sx * 200, 90 + q, 160), None)
      for h1 in range(lo1, lo1 + 3):
        for h2 in range(lo2, lo2 + 3):
          b.edge(mi1[(h1, h2)], lc, 10)
      b.edge(lc, aotu, 40)
      b.edge(lc, dnp09, 10)
    for k in range(2):
      b.add("MDN", "descending", s, (sx * 50, 220 + k, 210), None)
    cells[s] = {"aotu": aotu, "dna02": dna02}
  # Mushroom body: KC -> MBON is the exempt set.
  kcs = [b.add("KCg-m" if k % 2 else "KCab-c", "cb_intrinsic", "RL"[k % 2],
               (10.0 * k, 50, 100), None) for k in range(40)]
  mbons = [b.add(f"MBON{k:02d}", "cb_intrinsic", "R", (300, 60 + k, 120), None, "glutamate")
           for k in range(1, 7)]
  for kc in kcs:
    for m in rng.choice(len(mbons), 2, replace=False):
      b.edge(kc, mbons[m], int(rng.integers(3, 9)))
    b.edge(kc, kcs[int(rng.integers(len(kcs)))], 2)
  b.edge(mbons[0], kcs[0], 4)  # MBON -> KC is not exempt
  for k in range(10):
    pam = b.add(f"PAM{k + 1:02d}_a", "cb_intrinsic", "R", (310, 70, 130 + k), None, "dopamine")
    b.edge(pam, mbons[k % len(mbons)], 5)
  for k in range(1, 9):
    ppl = b.add(f"PPL10{k}", "cb_intrinsic", "L", (320, 75, 140 + k), None, "dopamine")
    b.edge(ppl, mbons[k % len(mbons)], 5)
  b.add("PPL201", "cb_intrinsic", "L", (330, 75, 150), None, "dopamine")  # not a PPL1
  # Fillers with random wiring and every transmitter case.
  nt_cases = [("acetylcholine", None), ("gaba", None), ("glutamate", None), ("dopamine", None),
              ("unclear", "gaba"), ("unclear", "unclear"), ("GABA", None)]
  fillers = []
  for k in range(150):
    nt, pred = nt_cases[k % len(nt_cases)]
    soma = None if k % 5 == 0 else tuple(rng.uniform(0, 400, 3))
    side = ["L", "R", None, "M"][k % 4]
    fillers.append(b.add("" if k % 3 == 0 else f"F{k % 11}",
                         "vnc_intrinsic" if k % 2 else "cb_intrinsic",
                         side, soma, None, nt, pred))
  everyone = [r["bodyId"] for r in b.rows]
  for f in fillers:
    for t in rng.choice(len(everyone), 8, replace=False):
      if everyone[t] != f:
        b.edge(f, everyone[t], int(rng.integers(1, 31)))
  b.edge(fillers[1], fillers[1], 3)  # a real self-loop
  for t in ("MDN", "DNp09", "DNa02"):
    for r in b.rows:
      if r["type"] == t:
        b.edge(fillers[2], r["bodyId"], 5)
  o1 = b.add("", "orphan", None, None, None, "acetylcholine")
  o2 = b.add("", "orphan", None, None, None, "acetylcholine")
  b.edge(o1, o2, 7)  # no partner with a soma: class-centroid position
  for k in range(5):
    b.add(f"Unused{k}", "cb_intrinsic", "R", (1, 2, 3), None)  # annotated but no edges
  ann = pd.DataFrame(b.rows)
  ann = pd.concat([ann, ann.iloc[[3]]], ignore_index=True)  # one duplicated annotation row
  nt_rows = [{"body": bid, "consensus_nt": c, "predicted_nt": p}
             for bid, (c, p) in b.nts.items() if bid != fillers[3]]  # one body missing
  nt_rows.append({"body": 1, "consensus_nt": "gaba", "predicted_nt": "gaba"})  # not in set
  w = pd.DataFrame([{"body_pre": a, "body_post": c, "weight": s}
                    for (a, c), s in b.edges.items()])
  w = w.sample(frac=1.0, random_state=seed).reset_index(drop=True)
  ann.to_feather(out_dir / ANNOTATIONS)
  pd.DataFrame(nt_rows).to_feather(out_dir / NEUROTRANSMITTERS)
  w.to_feather(out_dir / WEIGHTS)
  pins = {name: _crc(out_dir / name) for name in (ANNOTATIONS, NEUROTRANSMITTERS, WEIGHTS)}
  return {"pins": pins, "missing_nt_body": fillers[3]}
