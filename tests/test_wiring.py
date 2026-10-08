import dataclasses

import numpy as np
import pytest

from flycraft.brain.wiring import (
  ScrambleError,
  exempt_mask,
  fingerprint,
  load_wiring,
  scramble,
  wiring_cache_path,
)
from flycraft.config import load_config
from tests.conftest import CONFIGS


def scrambled_cfg(data_dir, seed=1):
  return load_config(CONFIGS / "scrambled.yaml",
                     [f"connectome.data_dir={data_dir}", f"connectome.scramble_seed={seed}"])


def do_scramble(conn, seed=1):
  ex = exempt_mask(conn, conn.pre, conn.post)
  return ex, scramble(conn.pre, conn.post, conn.syn, conn.sign, ex, seed, conn.n)


def per_neuron(conn, idx, edges):
  return np.bincount(idx[edges], minlength=conn.n)


def test_real_wiring_is_the_connectome(conn, synth_cfg):
  w = load_wiring(synth_cfg, conn)
  assert w.label == "REAL WIRING" and w.stats["seed"] is None
  assert w.pre is conn.pre and w.post is conn.post and w.syn is conn.syn
  assert w.fingerprint == fingerprint(conn.pre, conn.post, conn.syn)


def test_exempt_is_kc_to_numbered_mbon(conn):
  ex = exempt_mask(conn, conn.pre, conn.post)
  pre_t, post_t = conn.type[conn.pre], conn.type[conn.post]
  assert ex.sum() == 80  # 40 KCs x 2 MBONs in the synthetic brain
  assert all(p.startswith("KC") and q.startswith("MBON") for p, q in zip(pre_t[ex], post_t[ex],
                                                                         strict=True))
  assert not ex[(np.char.startswith(pre_t.astype(str), "KC"))
                & (np.char.startswith(post_t.astype(str), "KC"))].any()


def test_scramble_preserves_degrees_signs_and_weights(conn):
  ex, (pre, post, syn, ex2, stats) = do_scramble(conn)
  assert stats["merged"] == 0 and stats["leftover"] == 0
  np.testing.assert_array_equal(pre, conn.pre)
  exc_old, exc_new = conn.sign[conn.pre] > 0, conn.sign[pre] > 0
  np.testing.assert_array_equal(per_neuron(conn, conn.post, exc_old),
                                per_neuron(conn, post, exc_new))
  np.testing.assert_array_equal(per_neuron(conn, conn.post, ~exc_old),
                                per_neuron(conn, post, ~exc_new))
  for k in np.unique(conn.pre):
    np.testing.assert_array_equal(np.sort(conn.syn[conn.pre == k]), np.sort(syn[pre == k]))
  assert (post != conn.post).mean() > 0.5


def test_scramble_leaves_exempt_edges_alone(conn):
  ex, (pre, post, syn, ex2, _) = do_scramble(conn)
  np.testing.assert_array_equal(ex2, ex)
  np.testing.assert_array_equal(post[ex], conn.post[ex])
  np.testing.assert_array_equal(syn[ex], conn.syn[ex])


def test_scramble_removes_self_loops_and_duplicates(conn):
  assert (conn.pre == conn.post).any()  # the synthetic brain has a real self-loop
  _, (pre, post, _, _, stats) = do_scramble(conn)
  assert not (pre == post).any()
  key = pre.astype(np.int64) * conn.n + post
  assert np.unique(key).size == key.size
  assert stats["self_loops"] == 0


def test_scramble_is_seeded(conn):
  a = do_scramble(conn, 1)[1]
  b = do_scramble(conn, 1)[1]
  c = do_scramble(conn, 2)[1]
  np.testing.assert_array_equal(a[1], b[1])
  assert (a[1] != c[1]).any()


def test_unrepairable_class_fails_loudly():
  pre, post = np.array([0, 1], np.int32), np.array([1, 1], np.int32)
  sign = np.ones(2, np.int8)
  with pytest.raises(ScrambleError, match="1 self-loops or duplicate edges left"):
    scramble(pre, post, np.array([3, 4], np.int32), sign, np.zeros(2, bool), 0, 2)


def test_leftover_duplicates_merge_by_summing():
  # every target is neuron 2, so neuron 0's two edges can only ever duplicate each other
  pre, post = np.array([0, 0, 1], np.int32), np.array([2, 2, 2], np.int32)
  syn = np.array([3, 4, 5], np.int32)
  p, q, s, ex, stats = scramble(pre, post, syn, np.ones(3, np.int8), np.zeros(3, bool), 0, 3,
                                max_leftover_frac=0.5)
  assert stats == {"seed": 0, "passes": 20, "leftover": 1, "self_loops": 0, "merged": 1,
                   "dropped": 0}
  assert sorted(zip(p.tolist(), q.tolist(), s.tolist(), strict=True)) == [(0, 2, 7), (1, 2, 5)]


def test_leftover_self_loop_is_dropped_not_kept():
  # Minor 11: a single unrepairable edge (no partner to swap with) is a self-loop that must
  # be dropped, not merged (merging only sums exact-duplicate-key pairs; a lone self-loop has
  # a unique key and would otherwise survive untouched).
  p, q, s, ex, stats = scramble([0], [0], [5], [1], [False], 0, 1, max_leftover_frac=1.0)
  assert not (np.asarray(p) == np.asarray(q)).any()
  assert p.size == 0
  assert stats["dropped"] == 1 and stats["leftover"] == 1 and stats["self_loops"] == 1


def test_leftover_clash_with_an_exempt_edge_is_dropped():
  # A non-exempt edge that lands on the same (pre, post) pair as an exempt edge collides with
  # a connection it must never duplicate; it cannot be merged into the exempt edge (merging
  # only ever touches non-exempt edges), so it must be dropped.
  pre = np.array([0, 0], np.int32)
  post = np.array([1, 1], np.int32)  # edge 0 is exempt and reserves (0, 1)
  syn = np.array([3, 4], np.int32)
  exempt = np.array([True, False])
  p, q, s, ex, stats = scramble(pre, post, syn, np.ones(2, np.int8), exempt, 0, 2,
                                max_leftover_frac=1.0)
  assert stats["dropped"] == 1
  assert sorted(zip(p.tolist(), q.tolist(), ex.tolist(), strict=True)) == [(0, 1, True)]


def test_fingerprint_sees_every_array():
  pre, post, syn = np.arange(5), np.arange(5)[::-1], np.ones(5)
  fp = fingerprint(pre, post, syn)
  assert fp == fingerprint(pre.astype(np.int64), post, syn.astype(np.int32)) and len(fp) == 64
  syn2 = syn.copy()
  syn2[3] = 2
  assert fingerprint(pre, post, syn2) != fp
  assert fingerprint(post, pre, syn) != fp


def test_scrambled_wiring_is_cached(tmp_path, conn):
  cfg = scrambled_cfg(tmp_path)
  logs = []
  w1 = load_wiring(cfg, conn, log=logs.append)
  assert w1.label == "SCRAMBLED #1" and wiring_cache_path(cfg).exists()
  assert "cached at" in logs[-1]
  logs.clear()
  w2 = load_wiring(cfg, conn, log=logs.append)
  assert logs == []
  assert w2.fingerprint == w1.fingerprint == fingerprint(w2.pre, w2.post, w2.syn)
  np.testing.assert_array_equal(w2.exempt, w1.exempt)
  w3 = load_wiring(scrambled_cfg(tmp_path, 2), conn, log=logs.append)
  assert w3.fingerprint != w1.fingerprint


def test_cache_built_from_other_signs_is_rebuilt(tmp_path, conn):
  cfg = scrambled_cfg(tmp_path)
  w1 = load_wiring(cfg, conn, log=lambda s: None)
  sign = np.where(conn.type == "Mi1", -1, conn.sign).astype(np.int8)
  flipped = dataclasses.replace(conn, sign=sign)
  logs = []
  w2 = load_wiring(cfg, flipped, log=logs.append)
  assert "stale (built from different edges, signs or seed)" in logs[0]
  assert w2.fingerprint != w1.fingerprint


def test_tampered_cache_is_rebuilt(tmp_path, conn):
  cfg = scrambled_cfg(tmp_path)
  w1 = load_wiring(cfg, conn, log=lambda s: None)
  path = wiring_cache_path(cfg)
  with np.load(path) as z:
    d = {k: z[k] for k in z.files}
  d["post"] = d["post"][::-1].copy()
  np.savez(path, **d)
  logs = []
  w2 = load_wiring(cfg, conn, log=logs.append)
  assert "edge arrays do not match their fingerprint" in logs[0]
  assert w2.fingerprint == w1.fingerprint
