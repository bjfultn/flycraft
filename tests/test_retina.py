from types import SimpleNamespace

import numpy as np
import pytest

from flycraft import eye
from flycraft.brain.retina import Retina, RetinaGeometry, RetinaMapError, _majority_partner
from flycraft.config import EyeConfig, RetinaConfig, load_config
from flycraft.data.build import build_connectome
from flycraft.data.connectome import load_connectome
from tests.conftest import CONFIGS
from tests.synth import R16_PER_COL, make_tiny_connectome


@pytest.fixture(scope="module")
def geom(conn):
  return RetinaGeometry.build(conn)


def _retina(geom, **kw):
  return Retina(geom, RetinaConfig(**kw), EyeConfig())


def _partner_hex(conn, pre, partner_type):
  """Hex of pre's strongest partner of the given type."""
  m = (conn.pre == pre) & (conn.type[conn.post] == partner_type)
  return conn.hex[conn.post[m][np.argmax(conn.syn[m])]]


def test_affine_map_spans_the_field(geom):
  assert geom.map_mode == "affine"
  r = geom.pr_side == "R"
  assert np.nanmin(geom.pr_az[r]) == pytest.approx(-15.0)
  assert np.nanmax(geom.pr_az[r]) == pytest.approx(165.0)
  assert np.nanmin(geom.pr_el[r]) == pytest.approx(-60.0)
  assert np.nanmax(geom.pr_el[r]) == pytest.approx(75.0)


def test_eyes_mirror(geom):
  def dirs(s):
    m = geom.pr_side == s
    return {(round(a, 6), round(e, 6)) for a, e in zip(geom.pr_az[m], geom.pr_el[m], strict=True)}

  assert dirs("L") == {(-a + 0.0, e) for a, e in dirs("R")}


def test_photoreceptors_take_the_majority_column(conn, geom):
  r16 = geom.pr_idx[conn.type[geom.pr_idx] == "R1-R6"]
  assigned = [i for i in r16 if geom.assigned[geom.pr_idx == i][0]]
  assert len(assigned) == 25 * (R16_PER_COL["R"] + R16_PER_COL["L"])
  for i in assigned:
    k = np.flatnonzero(geom.pr_idx == i)[0]
    np.testing.assert_array_equal(geom.pr_hex[k], _partner_hex(conn, i, "L1"))


def test_r7_r8_ignore_lamina_partners(conn, geom):
  r78 = np.flatnonzero(np.isin(geom.pr_idx, conn.select(r"^R[78]")))
  assert r78.size == 2 * 2 * 25
  for k in r78:
    np.testing.assert_array_equal(geom.pr_hex[k], _partner_hex(conn, geom.pr_idx[k], "Mi1"))


def test_photoreceptor_without_hex_partner_is_unassigned(conn, geom):
  assert geom.info["n_unassigned"] == 2
  lost = ~geom.assigned
  assert set(geom.pr_side[lost]) == {"?"}
  assert np.isnan(geom.pr_az[lost]).all()
  img = eye.disk(60.0, 20.0, "dark_on_bright")
  idx, rate = _retina(geom).rates(img)
  k = np.isin(idx, geom.pr_idx[lost])
  np.testing.assert_allclose(rate[k], 150.0 * img.mean() / 255.0)


def test_normalization_equalizes_total_drive(geom):
  img = eye.grey(200)
  for norm in (True, False):
    idx, rate = _retina(geom, eye_normalize=norm).rates(img)
    side = dict(zip(geom.pr_idx, geom.pr_side, strict=True))
    tot = {s: rate[[side.get(i) == s for i in idx]].sum() for s in "LR"}
    if norm:
      assert tot["L"] == pytest.approx(tot["R"])
    else:
      assert tot["R"] == pytest.approx(1.5 * tot["L"])


def test_object_on_the_right_darkens_the_right_eye(geom):
  r = _retina(geom)
  base = r.rates(eye.blank("dark_on_bright"))[1]
  idx, rate = r.rates(eye.disk(75.0, 30.0, "dark_on_bright", el_deg=10.0))
  side = dict(zip(geom.pr_idx, geom.pr_side, strict=True))
  drop = {s: (base - rate)[[side.get(i) == s for i in idx]].sum() for s in "LR"}
  assert drop["R"] > 0
  assert drop["L"] == pytest.approx(0.0)


def _one_photoreceptor(az, el):
  return RetinaGeometry(
    pr_idx=np.array([0]), pr_side=np.array(["R"]), pr_az=np.array([az]), pr_el=np.array([el]),
    pr_hex=np.zeros((1, 2)), lamina_idx=np.array([1, 2]), vpn_idx=np.zeros(0, np.int64),
    vpn_side=np.zeros(0, str), vpn_az=np.zeros(0), vpn_el=np.zeros(0), map_mode="affine")


def test_sampling_wraps_the_azimuth_seam():
  img = np.zeros((eye.EYE_ROWS, eye.EYE_COLS), np.uint8)
  img[:, 0] = 255  # centred at az -177.5
  for az in (180.0, -180.0):
    r = Retina(_one_photoreceptor(az, 0.0), RetinaConfig(eye_normalize=False), EyeConfig())
    assert r.sample(img)[0] == pytest.approx(0.5)


def test_sampling_clamps_elevation():
  img = np.zeros((eye.EYE_ROWS, eye.EYE_COLS), np.uint8)
  img[0] = 255
  r = Retina(_one_photoreceptor(0.0, 90.0), RetinaConfig(eye_normalize=False), EyeConfig())
  assert r.sample(img)[0] == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [
  np.zeros((eye.EYE_ROWS, eye.EYE_COLS), np.float32),
  np.zeros((eye.EYE_COLS, eye.EYE_ROWS), np.uint8),
  np.zeros((eye.EYE_ROWS, eye.EYE_COLS, 3), np.uint8),
  [[0] * eye.EYE_COLS] * eye.EYE_ROWS,
])
def test_bad_eye_image_is_rejected(geom, bad):
  with pytest.raises(ValueError, match="uint8 30x72"):
    _retina(geom).rates(bad)


def test_lamina_tonic_and_vpn_inputs(conn, geom):
  r = _retina(geom, mode="photoreceptor+vpn", lamina_tonic_hz=20.0)
  assert r.rfc_zero_idx is r.idx
  idx, rate = r.rates(eye.blank("dark_on_bright"))
  lam = np.isin(idx, geom.lamina_idx)
  assert set(conn.type[geom.lamina_idx]) == {"L1", "L2", "L3", "L4", "L5"}
  assert lam.sum() == geom.lamina_idx.size
  assert (rate[lam] == 20.0).all()
  vpn = np.isin(idx, geom.vpn_idx)
  assert vpn.sum() == 8
  assert (rate[vpn] == 0).all()
  k = int(np.argmax(geom.vpn_side == "R"))
  img = eye.disk(geom.vpn_az[k], 15.0, "dark_on_bright", el_deg=geom.vpn_el[k])
  idx, rate = r.rates(img)
  vr = dict(zip(idx[vpn], rate[vpn], strict=True))
  assert vr[geom.vpn_idx[k]] > 0.5 * 150.0
  assert all(vr[i] == 0 for i, s in zip(geom.vpn_idx, geom.vpn_side, strict=True) if s == "L")


def test_vpn_fields_are_quadrants(geom):
  r = geom.vpn_side == "R"
  np.testing.assert_allclose(geom.vpn_az[~r], -geom.vpn_az[r])
  assert len(np.unique(geom.vpn_el[r].round(3))) == 3  # top, middle pair, bottom


@pytest.fixture(scope="module")
def noisy_conn(tmp_path_factory):
  d = tmp_path_factory.mktemp("noisy")
  meta = make_tiny_connectome(d, mi1_noise_um=200.0)
  build_connectome(d, d / "malecns-v1.0.npz", ("gaba", "glutamate", "histamine"),
                   log=lambda s: None)
  cfg = load_config(CONFIGS / "real.yaml", [f"connectome.data_dir={d}"])
  return load_connectome(cfg, pins=meta["pins"], log=lambda s: None)


def test_bad_fit_falls_back_to_bands(noisy_conn):
  g = RetinaGeometry.build(noisy_conn)
  assert g.map_mode == "bands"
  assert "Mi1 z" in g.info["map_reason"]
  for s, sign in (("R", 1), ("L", -1)):
    m = g.pr_side == s
    assert set(np.unique(sign * g.pr_az[m])) == {0.0, 30.0, 60.0, 90.0, 120.0, 150.0}
    assert (g.pr_el[m] == 0).all()


def test_affine_map_refuses_a_bad_fit(noisy_conn):
  with pytest.raises(RetinaMapError, match="Mi1 z"):
    RetinaGeometry.build(noisy_conn, "affine")


def test_bands_on_request(conn):
  assert RetinaGeometry.build(conn, "bands").map_mode == "bands"


def test_majority_partner_does_not_alias_distinct_hex_columns():
  """Minor 12: packing (side, h1, h2) into one int can alias two distinct columns (hex (1,
  -1000) and hex (0, 0) both packed to the same key); grouping must compare the real tuple
  instead, so the column with the real majority of votes wins, not whichever alias is first."""
  conn = SimpleNamespace(
    n=3,
    pre=np.array([0, 0], np.int64),
    post=np.array([1, 2], np.int64),
    syn=np.array([1, 100], np.int64),  # neuron 2 is the true majority partner by a mile
    side=np.array(["?", "R", "R"]),
  )
  src = np.array([0], np.int64)
  partner_ok = np.array([False, True, True])
  hexi = np.array([[0, 0], [1, -1000], [0, 0]], np.int64)  # neuron 2's hex is literally (0, 0)
  out = _majority_partner(conn, src, partner_ok, hexi)
  assert out[0] == 2
