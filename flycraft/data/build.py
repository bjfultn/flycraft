"""Build the compact connectome cache from the three MaleCNS feathers.

The neuron set is every body in the traced-only edge list (VNC included). Neurons are
indexed 0..N-1 in ascending bodyId order. The cache is one .npz holding the neuron table,
int32 edges and a JSON manifest.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from flycraft.data.fetch import ANNOTATIONS, NEUROTRANSMITTERS, WEIGHTS, file_crc32c

BUILD_VERSION = 1
VOXEL_UM = 0.008  # MaleCNS voxels are 8 nm
JITTER_UM = 5.0
MLX_N_NEURONS = 166_700
MLX_N_EDGES = 24_469_412
MLX_NTS = ("acetylcholine", "gaba", "glutamate")

POS_SOMA, POS_PARTNERS, POS_CENTROID = 0, 1, 2


def _str_col(s: pd.Series) -> np.ndarray:
  return s.fillna("").astype(str).to_numpy(dtype=str)


def resolve_nt(consensus: pd.Series, predicted: pd.Series) -> np.ndarray:
  """consensus_nt if it is a real call, else predicted_nt if real, else 'unknown'."""
  c = consensus.str.lower()
  p = predicted.str.lower()
  c_ok = c.notna() & (c != "unclear")
  p_ok = p.notna() & (p != "unclear")
  return _str_col(c.where(c_ok, p.where(p_ok, "unknown")))


def sign_from_nt(nt: np.ndarray, inhibitory_nts) -> np.ndarray:
  inhibitory = np.isin(np.char.lower(nt.astype(str)), [x.lower() for x in inhibitory_nts])
  return np.where(inhibitory, -1, 1).astype(np.int8)


def resolve_side(soma_side: pd.Series, root_side: pd.Series) -> np.ndarray:
  soma = soma_side.where(soma_side.isin(["L", "R", "M"]))
  root = root_side.where(root_side.isin(["L", "R"]))
  return _str_col(soma.fillna(root).fillna("?"))


def _positions(soma_xyz, has_soma, pre, post, syn, superclass, seed=0):
  """Soma position, else synapse-weighted mean partner soma, else class centroid + jitter."""
  n = len(has_soma)
  pos = np.where(has_soma[:, None], soma_xyz, 0.0)
  source = np.where(has_soma, POS_SOMA, -1).astype(np.int8)
  w = syn.astype(np.float64)
  num = np.zeros((n, 3))
  den = np.zeros(n)
  for a, b in ((pre, post), (post, pre)):  # a's position from partner b
    wb = w * has_soma[b]
    den += np.bincount(a, weights=wb, minlength=n)
    for k in range(3):
      num[:, k] += np.bincount(a, weights=wb * pos[b, k], minlength=n)
  fill = ~has_soma & (den > 0)
  pos[fill] = num[fill] / den[fill, None]
  source[fill] = POS_PARTNERS
  rest = np.flatnonzero(source < 0)
  if rest.size:
    rng = np.random.default_rng(seed)
    global_c = pos[has_soma].mean(axis=0) if has_soma.any() else np.zeros(3)
    for cls in np.unique(superclass[rest]):
      members = rest[superclass[rest] == cls]
      ref = has_soma & (superclass == cls)
      centre = pos[ref].mean(axis=0) if ref.any() else global_c
      pos[members] = centre + rng.normal(0.0, JITTER_UM, size=(members.size, 3))
    source[rest] = POS_CENTROID
  return pos.astype(np.float32), source


def build_connectome(src_dir: Path, out_path: Path, inhibitory_nts, log=print) -> dict:
  src_dir = Path(src_dir).expanduser()
  log("reading edges")
  w = pd.read_feather(src_dir / WEIGHTS, columns=["body_pre", "body_post", "weight"])
  body_pre = w["body_pre"].to_numpy(np.int64)
  body_post = w["body_post"].to_numpy(np.int64)
  syn = w["weight"].to_numpy().astype(np.int32)
  del w
  ids = np.unique(np.concatenate([body_pre, body_post]))
  pre = np.searchsorted(ids, body_pre).astype(np.int32)
  post = np.searchsorted(ids, body_post).astype(np.int32)
  del body_pre, body_post
  n = ids.size

  log("reading annotations")
  cols = ["bodyId", "type", "superclass", "class", "somaSide", "rootSide", "somaLocation",
          "assignedOlHex1", "assignedOlHex2"]
  a_all = pd.read_feather(src_dir / ANNOTATIONS, columns=cols)
  annotated_with_superclass = int(a_all["superclass"].notna().sum())
  a = a_all.drop_duplicates("bodyId").set_index("bodyId").reindex(ids)
  del a_all
  soma = a["somaLocation"]
  has_soma = soma.notna().to_numpy()
  soma_xyz = np.zeros((n, 3))
  if has_soma.any():
    soma_xyz[has_soma] = np.stack(soma[has_soma].to_numpy()).astype(np.float64) * VOXEL_UM
  hex_ = a[["assignedOlHex1", "assignedOlHex2"]].to_numpy(dtype=np.float32)
  superclass = _str_col(a["superclass"])

  log("reading neurotransmitters")
  t = pd.read_feather(src_dir / NEUROTRANSMITTERS, columns=["body", "consensus_nt", "predicted_nt"])
  t = t.drop_duplicates("body").set_index("body").reindex(ids)
  nt = resolve_nt(t["consensus_nt"], t["predicted_nt"])
  mlx_style_edges = int(np.isin(t["consensus_nt"].str.lower().to_numpy(), MLX_NTS)[pre].sum())
  del t

  pos, pos_source = _positions(soma_xyz, has_soma, pre, post, syn, superclass)
  sign = sign_from_nt(nt, inhibitory_nts)

  manifest = {
    "dataset": "malecns-v1.0",
    "build_version": BUILD_VERSION,
    "sources": {name: file_crc32c(src_dir / name) for name in (ANNOTATIONS, NEUROTRANSMITTERS,
                                                                WEIGHTS)},
    "n_neurons": int(n),
    "n_edges": int(pre.size),
    "n_synapses": int(syn.sum(dtype=np.int64)),
    "n_self_loops": int((pre == post).sum()),
    "nt_counts": dict(Counter(nt.tolist()).most_common()),
    "build_inhibitory_nts": list(inhibitory_nts),
    "pos_source_counts": {"soma": int((pos_source == POS_SOMA).sum()),
                          "partners": int((pos_source == POS_PARTNERS).sum()),
                          "centroid": int((pos_source == POS_CENTROID).sum())},
    "mlx": {
      "n_neurons": MLX_N_NEURONS,
      "n_edges": MLX_N_EDGES,
      "annotated_with_superclass": annotated_with_superclass,
      "edges_pre_consensus_ach_gaba_glu": mlx_style_edges,
    },
  }
  out_path = Path(out_path).expanduser()
  out_path.parent.mkdir(parents=True, exist_ok=True)
  tmp = out_path.with_name(out_path.name + ".tmp.npz")
  np.savez(
    tmp,
    body_id=ids, type=_str_col(a["type"]), superclass=superclass,
    cell_class=_str_col(a["class"]), side=resolve_side(a["somaSide"], a["rootSide"]),
    pos=pos, pos_source=pos_source, hex=hex_, nt=nt, sign=sign,
    pre=pre, post=post, syn=syn, manifest=np.array(json.dumps(manifest)),
  )
  tmp.replace(out_path)
  log(f"built {out_path}: {n} neurons, {pre.size} edges "
      f"(drosophila-brain-mlx: {MLX_N_NEURONS}, {MLX_N_EDGES})")
  return manifest
