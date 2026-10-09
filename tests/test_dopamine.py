import numpy as np

from flycraft.brain.dopamine import compartment_edges, dopamine_map
from flycraft.brain.wiring import load_wiring
from flycraft.config import with_overrides

QUIET = lambda s: None  # noqa: E731


def test_map_counts_pam_then_ppl101_to_108(conn):
  d = dopamine_map(conn)
  types = conn.type[d.dans]
  assert d.pam.sum() == 10 and all(t.startswith("PAM") for t in types[d.pam])
  assert sorted(types[~d.pam]) == [f"PPL10{k}" for k in range(1, 9)]  # PPL201 is not a PPL1
  assert d.m.shape == (18, 6)
  np.testing.assert_allclose(d.m.sum(axis=0), 1.0)


def test_pam_compartment_is_a_pam_majority(conn):
  # synth: PAM k -> MBON k % 6 and PPL10k -> MBON k % 6, 5 synapses each (tests/synth.py)
  d = dopamine_map(conn)
  frac = dict(zip(conn.type[d.mbons], d.pam_frac, strict=True))
  np.testing.assert_allclose([frac[f"MBON0{k}"] for k in range(1, 7)],
                             [2 / 3, 0.5, 0.5, 2 / 3, 0.5, 0.5])
  assert sorted(conn.type[d.pam_mbons()]) == ["MBON01", "MBON04"]  # a tie is not a majority


def test_map_matches_a_recount_of_the_edges(conn):
  d = dopamine_map(conn)
  dans, mbons = list(d.dans), list(d.mbons)
  counts = np.zeros(d.m.shape)
  for a, b, s in zip(conn.pre, conn.post, conn.syn, strict=True):
    if a in dans and b in mbons:  # DAN -> anything else (KCs, fillers) is not in the map
      counts[dans.index(a), mbons.index(b)] += s
  np.testing.assert_allclose(d.m, counts / counts.sum(axis=0))


def test_compartment_edges_are_the_plastic_edges_onto_those_mbons(conn, synth_cfg):
  d = dopamine_map(conn)
  mb = d.pam_mbons()
  for wiring, seed in (("real", None), ("scrambled", 4)):
    cfg = with_overrides(synth_cfg, {"connectome.wiring": wiring, "connectome.scramble_seed": seed})
    w = load_wiring(cfg, conn, log=QUIET)
    e = compartment_edges(w, mb, conn.n)
    assert e.size > 0 and w.exempt[e].all() and np.isin(w.post[e], mb).all()
    rest = np.setdiff1d(np.flatnonzero(w.exempt), e)
    assert not np.isin(w.post[rest], mb).any()
    pairs = {(int(a), int(b)) for a, b in zip(w.pre[e], w.post[e], strict=True)}
    if wiring == "real":
      real_pairs = pairs
    else:
      assert pairs == real_pairs  # plastic edges are identical in every twin
