"""Real and scrambled wiring (spec section 6).

A scrambled twin keeps every edge's presynaptic neuron, synapse count and sign, and permutes
postsynaptic endpoints within the excitatory and inhibitory classes. KC -> MBON edges (the
plastic set) are exempt and stay real. Self-loops and duplicate pairs are repaired by swapping
targets inside the same class. Of what survives the repair passes: non-exempt pairs that still
duplicate another non-exempt edge are merged by summing; self-loops and clashes with an exempt
edge cannot be merged (there is nothing non-exempt to fold them into) and are dropped instead.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flycraft.config import Config
from flycraft.data.connectome import Connectome

WIRING_VERSION = 2
KC_PATTERN = r"^KC"
MBON_PATTERN = r"^MBON\d"


class ScrambleError(RuntimeError):
  pass


@dataclass(frozen=True, eq=False)
class Wiring:
  pre: np.ndarray  # int32 (E,)
  post: np.ndarray  # int32 (E,)
  syn: np.ndarray  # int32 (E,)
  exempt: np.ndarray  # bool (E,): the plastic KC -> MBON edges, identical in every twin
  fingerprint: str  # sha256 of pre, post, syn
  label: str  # "REAL WIRING" or "SCRAMBLED #<seed>"
  stats: dict


def fingerprint(pre, post, syn) -> str:
  h = hashlib.sha256()
  for a in (pre, post, syn):
    h.update(np.ascontiguousarray(a, dtype="<i4").tobytes())
  return h.hexdigest()


def exempt_mask(conn: Connectome, pre, post) -> np.ndarray:
  kc = np.zeros(conn.n, bool)
  kc[conn.select(KC_PATTERN)] = True
  mbon = np.zeros(conn.n, bool)
  mbon[conn.select(MBON_PATTERN)] = True
  return kc[pre] & mbon[post]


def _find_bad(pre, post, exempt, n) -> np.ndarray:
  """Non-exempt edges that are self-loops or repeat an earlier (pre, post) pair.

  Exempt edges sort first within a pair, so a clash with an exempt edge marks the other one.
  """
  key = pre.astype(np.int64) * n + post
  order = np.lexsort((~exempt, key))
  k = key[order]
  dup = np.zeros(k.size, bool)
  dup[1:] = k[1:] == k[:-1]
  bad = np.zeros(k.size, bool)
  bad[order[dup]] = True
  bad |= pre == post
  return bad & ~exempt


def _merge_duplicates(pre, post, syn, exempt, n):
  """Sum duplicate non-exempt pairs into their first edge. Returns (keep mask, syn)."""
  idx = np.flatnonzero(~exempt)
  key = pre[idx].astype(np.int64) * n + post[idx]
  uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
  keep = np.ones(pre.size, bool)
  if uniq.size == idx.size:
    return keep, syn
  syn = syn.copy()
  syn[idx[first]] = np.bincount(inv, weights=syn[idx]).astype(syn.dtype)
  keep[idx] = False
  keep[idx[first]] = True
  return keep, syn


def scramble(pre, post, syn, sign, exempt, seed: int, n: int, max_passes: int = 20,
             max_leftover_frac: float = 1e-3):
  """Degree- and sign-preserving target shuffle.

  Returns (pre, post, syn, exempt, stats). Leftover non-exempt duplicate pairs are merged by
  summing; leftover self-loops and leftover clashes with an exempt edge have nothing to merge
  into, so they are dropped instead (counted in stats as "dropped").
  """
  rng = np.random.default_rng(seed)
  pre, syn = np.asarray(pre), np.asarray(syn)
  exempt = np.asarray(exempt, bool)
  new = np.array(post, copy=True)
  esign = np.asarray(sign)[pre]
  classes = []
  for s in (1, -1):
    idx = np.flatnonzero(~exempt & (esign == s))
    new[idx] = new[idx][rng.permutation(idx.size)]
    classes.append(idx)
  passes = 0
  bad = _find_bad(pre, new, exempt, n)
  while bad.any() and passes < max_passes:
    passes += 1
    for idx in classes:
      b = idx[bad[idx]]
      pool = idx[~bad[idx]]
      k = min(b.size, pool.size)
      if k == 0:
        continue
      if k < b.size:
        b = rng.permutation(b)[:k]
      part = rng.choice(pool, size=k, replace=False)
      new[b], new[part] = new[part], new[b]
    bad = _find_bad(pre, new, exempt, n)
  leftover = int(bad.sum())
  self_loops = int((~exempt & (pre == new)).sum())
  if leftover > max_leftover_frac * pre.size:
    raise ScrambleError(
      f"scramble seed {seed}: {leftover} self-loops or duplicate edges left after {passes} "
      f"passes ({leftover / pre.size:.3%} of edges, limit {max_leftover_frac:.3%})")
  keep, syn = _merge_duplicates(pre, new, syn, exempt, n)
  merged = int((~keep).sum())
  # A leftover self-loop, or a leftover clash with an exempt edge, has no non-exempt duplicate
  # to be summed into (merging never touches exempt edges), so it would otherwise survive
  # untouched. Drop those instead of leaving them in the wiring (Minor 11).
  kept = np.flatnonzero(keep)
  still_self_loop = pre[kept] == new[kept]
  exempt_keys = (pre[exempt].astype(np.int64) * n + post[exempt]) if exempt.any() \
      else np.empty(0, np.int64)
  kept_keys = pre[kept].astype(np.int64) * n + new[kept]
  clashes_exempt = np.isin(kept_keys, exempt_keys)
  drop = ~exempt[kept] & (still_self_loop | clashes_exempt)
  dropped = int(drop.sum())
  keep[kept[drop]] = False
  stats = {"seed": seed, "passes": passes, "leftover": leftover, "self_loops": self_loops,
           "merged": merged, "dropped": dropped}
  return pre[keep], new[keep], syn[keep], exempt[keep], stats


def wiring_cache_path(cfg: Config) -> Path:
  c = cfg.connectome
  return Path(c.data_dir).expanduser() / f"{c.dataset}.scrambled-{c.scramble_seed}.npz"


def _source_key(conn: Connectome, exempt, seed: int) -> str:
  """Everything a twin depends on: the real edges, the signs, the exempt set and the seed."""
  h = hashlib.sha256(f"wiring v{WIRING_VERSION} seed {seed}".encode())
  for a in (conn.pre, conn.post, conn.syn, conn.sign, exempt):
    h.update(np.ascontiguousarray(a).tobytes())
  return h.hexdigest()


def _read_cache(path: Path, key: str, label: str) -> tuple[Wiring | None, str]:
  with np.load(path) as z:
    if str(z["source_key"]) != key:
      return None, "built from different edges, signs or seed"
    w = Wiring(z["pre"], z["post"], z["syn"], z["exempt"], str(z["fingerprint"]), label,
               json.loads(str(z["stats"])))
  if fingerprint(w.pre, w.post, w.syn) != w.fingerprint:
    return None, "edge arrays do not match their fingerprint"
  return w, ""


def load_wiring(cfg: Config, conn: Connectome, log=print) -> Wiring:
  exempt = exempt_mask(conn, conn.pre, conn.post)
  c = cfg.connectome
  if c.wiring == "real":
    return Wiring(conn.pre, conn.post, conn.syn, exempt, fingerprint(conn.pre, conn.post, conn.syn),
                  "REAL WIRING", {"seed": None})
  seed = c.scramble_seed
  label = f"SCRAMBLED #{seed}"
  path = wiring_cache_path(cfg)
  key = _source_key(conn, exempt, seed)
  if path.exists():
    w, reason = _read_cache(path, key, label)
    if w is not None:
      return w
    log(f"{path} is stale ({reason}); rebuilding")
  pre, post, syn, ex, stats = scramble(conn.pre, conn.post, conn.syn, conn.sign, exempt, seed,
                                       conn.n)
  fp = fingerprint(pre, post, syn)
  tmp = path.with_name(path.stem + ".tmp.npz")
  np.savez(tmp, pre=pre, post=post, syn=syn, exempt=ex, fingerprint=np.array(fp),
           source_key=np.array(key), stats=np.array(json.dumps(stats)))
  tmp.replace(path)
  log(f"scrambled wiring seed {seed}: {stats['passes']} repair passes, "
      f"{stats['leftover']} leftover, {stats['merged']} merged; cached at {path}")
  return Wiring(pre, post, syn, ex, fp, label, stats)
