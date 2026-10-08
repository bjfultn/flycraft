import json
import shutil

import numpy as np
import pandas as pd
import pytest

from flycraft.config import with_overrides
from flycraft.data import connectome
from flycraft.data.build import POS_CENTROID, POS_PARTNERS, POS_SOMA, build_connectome
from flycraft.data.connectome import StaleCacheError, cache_path, load_connectome
from flycraft.data.fetch import ANNOTATIONS, NEUROTRANSMITTERS, WEIGHTS


def quiet(s):
  pass


def test_neuron_set_is_the_edge_list(conn, synth_dir):
  d, _ = synth_dir
  w = pd.read_feather(d / WEIGHTS)
  ids = np.unique(np.concatenate([w.body_pre, w.body_post]))
  np.testing.assert_array_equal(conn.body_id, ids)
  assert conn.pre.dtype == np.int32 and conn.post.dtype == np.int32 and conn.syn.dtype == np.int32
  np.testing.assert_array_equal(conn.body_id[conn.pre], w.body_pre)
  np.testing.assert_array_equal(conn.body_id[conn.post], w.body_post)
  np.testing.assert_array_equal(conn.syn, w.weight)
  assert not conn.select(r"^Unused").size  # annotated, but no edges


def test_side_rule(conn):
  r16 = conn.select(r"^R1-R6$")
  assert set(conn.side[r16]) == {"L", "R"}  # from rootSide; photoreceptors have no somaSide
  assert (conn.side == "M").any()
  assert (conn.side == "?").any()
  assert r16.size == 25 * 4 + 25 * 2 + 2


def test_transmitter_resolution_and_signs(conn, synth_dir):
  _, meta = synth_dir
  nt = dict(zip(conn.body_id.tolist(), conn.nt.tolist(), strict=True))
  assert nt[meta["missing_nt_body"]] == "unknown"
  assert set(conn.nt) <= {"acetylcholine", "gaba", "glutamate", "histamine", "dopamine",
                          "unknown"}
  assert "unclear" not in set(conn.nt)
  assert (conn.nt == "unknown").sum() > 1  # unclear+unclear falls through
  expect = np.where(np.isin(conn.nt, ["gaba", "glutamate", "histamine"]), -1, 1)
  np.testing.assert_array_equal(conn.sign, expect)
  assert conn.sign.dtype == np.int8
  assert (conn.sign[conn.select(r"^R1-R6$")] == -1).all()


def test_inhibitory_list_comes_from_config(synth_cfg, synth_dir):
  _, meta = synth_dir
  cfg = with_overrides(synth_cfg, {"connectome.inhibitory_nts": ["gaba", "glutamate"]})
  c = load_connectome(cfg, pins=meta["pins"], log=quiet)
  assert (c.sign[c.select(r"^R1-R6$")] == 1).all()


def test_positions(conn, synth_dir):
  d, _ = synth_dir
  ann = pd.read_feather(d / ANNOTATIONS).drop_duplicates("bodyId").set_index("bodyId")
  l1 = conn.select(r"^L1$")
  soma = np.stack(ann.loc[conn.body_id[l1], "somaLocation"].to_numpy()) * 0.008
  np.testing.assert_allclose(conn.pos[l1], soma, atol=1e-3)
  assert (conn.pos_source[l1] == POS_SOMA).all()
  r16 = conn.select(r"^R1-R6$")
  assert (conn.pos_source[r16] == POS_PARTNERS).all()
  # Synapse-weighted mean over partners with a soma, both directions, for one photoreceptor.
  k = r16[0]
  has = conn.pos_source == POS_SOMA
  out = (conn.pre == k) & has[conn.post]
  inc = (conn.post == k) & has[conn.pre]
  partners = np.concatenate([conn.post[out], conn.pre[inc]])
  w = np.concatenate([conn.syn[out], conn.syn[inc]]).astype(float)
  expect = (conn.pos[partners] * w[:, None]).sum(0) / w.sum()
  np.testing.assert_allclose(conn.pos[k], expect, atol=1e-3)
  orphans = np.flatnonzero(conn.superclass == "orphan")
  assert orphans.size == 2 and (conn.pos_source[orphans] == POS_CENTROID).all()
  assert np.isfinite(conn.pos).all()


def test_hex_columns(conn):
  l1 = conn.select(r"^L1$")
  assert np.isfinite(conn.hex[l1]).all()
  assert np.isnan(conn.hex[conn.select(r"^R1-R6$")]).all()


def test_manifest(conn, synth_dir):
  _, meta = synth_dir
  m = conn.manifest
  assert m["sources"] == meta["pins"]
  assert m["n_neurons"] == conn.n and m["n_edges"] == conn.pre.size
  assert m["n_self_loops"] == 1
  assert m["nt_counts"]["histamine"] == (conn.nt == "histamine").sum()
  assert m["mlx"]["n_neurons"] == 166_700 and m["mlx"]["n_edges"] == 24_469_412


def test_select(conn):
  assert conn.select(r"^PPL10[1-8]$").size == 8  # PPL201 excluded
  assert conn.select(r"^MBON\d").size == 6
  assert conn.select(r"^DNa02$", side="R").size == 1


def _copy_synth(src, dst, with_sources=True):
  dst.mkdir()
  shutil.copy(src / "malecns-v1.0.npz", dst)
  if with_sources:
    for name in (ANNOTATIONS, NEUROTRANSMITTERS, WEIGHTS):
      shutil.copy(src / name, dst)


def _age_cache(path):
  with np.load(path) as z:
    d = {k: z[k] for k in z.files}
  m = json.loads(str(d["manifest"]))
  m["build_version"] = 0
  d["manifest"] = np.array(json.dumps(m))
  np.savez(path, **d)


def test_missing_cache_says_run_prep_data(synth_cfg, tmp_path):
  cfg = with_overrides(synth_cfg, {"connectome.data_dir": str(tmp_path)})
  with pytest.raises(FileNotFoundError, match="flycraft prep-data"):
    load_connectome(cfg, log=quiet)


def test_stale_cache_is_rebuilt_from_sources(synth_cfg, synth_dir, tmp_path):
  src, meta = synth_dir
  _copy_synth(src, tmp_path / "d")
  cfg = with_overrides(synth_cfg, {"connectome.data_dir": str(tmp_path / "d")})
  _age_cache(cache_path(cfg))
  logs = []
  c = load_connectome(cfg, pins=meta["pins"], log=logs.append)
  assert c.manifest["build_version"] == 1
  assert any("rebuilding" in s for s in logs)


def test_stale_cache_without_sources_is_rejected(synth_cfg, synth_dir, tmp_path):
  src, meta = synth_dir
  _copy_synth(src, tmp_path / "d", with_sources=False)
  cfg = with_overrides(synth_cfg, {"connectome.data_dir": str(tmp_path / "d")})
  _age_cache(cache_path(cfg))
  with pytest.raises(StaleCacheError, match="prep-data"):
    load_connectome(cfg, pins=meta["pins"], log=quiet)


def test_stale_cache_with_corrupted_source_is_rejected_without_rebuilding(synth_cfg, synth_dir,
                                                                           tmp_path, monkeypatch):
  """Minor 10: a source file that exists by name but fails crc32c must not trigger a rebuild."""
  src, meta = synth_dir
  _copy_synth(src, tmp_path / "d")
  with open(tmp_path / "d" / WEIGHTS, "ab") as f:
    f.write(b"\x00")  # exists by name, but no longer matches its pinned crc32c
  cfg = with_overrides(synth_cfg, {"connectome.data_dir": str(tmp_path / "d")})
  _age_cache(cache_path(cfg))

  def boom(*a, **k):
    raise AssertionError("must not rebuild from a source that fails crc32c")

  monkeypatch.setattr(connectome, "build_connectome", boom)
  with pytest.raises(StaleCacheError, match="crc32c"):
    load_connectome(cfg, pins=meta["pins"], log=quiet)


def test_cache_from_other_sources_is_stale(synth_cfg, synth_dir, tmp_path):
  src, meta = synth_dir
  _copy_synth(src, tmp_path / "d", with_sources=False)
  cfg = with_overrides(synth_cfg, {"connectome.data_dir": str(tmp_path / "d")})
  with pytest.raises(StaleCacheError, match="pinned"):
    load_connectome(cfg, log=quiet)  # default pins are the real MaleCNS files


def test_build_is_deterministic(synth_dir, tmp_path):
  src, _ = synth_dir
  build_connectome(src, tmp_path / "a.npz", ("gaba", "glutamate", "histamine"), log=quiet)
  with np.load(tmp_path / "a.npz") as a, np.load(src / "malecns-v1.0.npz") as b:
    for k in a.files:
      np.testing.assert_array_equal(a[k], b[k])
