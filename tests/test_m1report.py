import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from flycraft import m1report
from flycraft.config import with_overrides
from flycraft.m1report import _sanitize_for_json, chirality, run_m1, sign_rule_check

QUIET = lambda s: None  # noqa: E731
SHORT = {"calibration.g0_blank_ms": 200.0, "calibration.g1_trial_ms": 200.0,
         "calibration.g1_repeats": 3, "calibration.sample_every_ms": 50.0,
         "calibration.sample_warmup_ms": 50.0,
         "calibration.g1_azimuths_deg": [-25.0, 0.0, 25.0]}  # synth LC10a sit at +-25


def test_sign_rule_names_the_load_time_rule_not_the_build_time_rule(tmp_path, synth_cfg, synth_dir):
  """BC1: an inhibitory_nts override must change the reported rule, not just the cache's."""
  pins = synth_dir[1]["pins"]
  default_cfg = with_overrides(synth_cfg, {**SHORT, "calibration.ladder": ["vpn"]})
  override_cfg = with_overrides(synth_cfg, {**SHORT, "calibration.ladder": ["vpn"],
                                            "connectome.inhibitory_nts": ["gaba", "glutamate"]})
  out_d, out_o = tmp_path / "default", tmp_path / "override"
  run_m1(default_cfg, out_d, QUIET, pins=pins)
  run_m1(override_cfg, out_o, QUIET, pins=pins)
  rec_d = json.loads((out_d / "calibration.json").read_text())
  rec_o = json.loads((out_o / "calibration.json").read_text())
  assert rec_d["sign_rule"]["ours_inhibitory"] == ["gaba", "glutamate", "histamine"]
  assert rec_o["sign_rule"]["ours_inhibitory"] == ["gaba", "glutamate"]
  assert rec_o["sign_rule"]["neurons_differ"] == 0 and rec_o["sign_rule"]["edges_differ"] == 0
  # the fingerprint must not change with the sign rule (out of scope: wiring.fingerprint)
  assert rec_d["wiring"]["fingerprint"] == rec_o["wiring"]["fingerprint"]
  assert "inhibitory: gaba, glutamate." in (out_o / "report.md").read_text()


def test_sanitize_for_json_turns_non_finite_floats_into_strings():
  obj = {"a": float("nan"), "b": [float("inf"), -float("inf"), 1.5], "c": {"d": float("nan")},
         "e": "nan", "f": 3}
  clean = _sanitize_for_json(obj)
  assert clean == {"a": "nan", "b": ["inf", "-inf", 1.5], "c": {"d": "nan"}, "e": "nan", "f": 3}
  text = json.dumps(clean)
  assert "NaN" not in text and "Infinity" not in text


def test_sign_rule_differs_from_shiu_only_on_histamine(conn):
  s = sign_rule_check(conn)
  hist = conn.nt == "histamine"
  assert set(s["by_nt"]) == {"histamine"}
  d = s["by_nt"]["histamine"]
  assert (d["ours"], d["shiu"]) == (-1, 1)
  assert d["neurons"] == s["neurons_differ"] == int(hist.sum()) > 0
  assert d["edges"] == s["edges_differ"] == int(hist[conn.pre].sum())
  assert s["shiu_inhibitory"] == ["gaba", "glutamate"]


@pytest.mark.parametrize("z, words", [(4.0, "toward"), (-4.0, "AWAY"), (1.0, "no significant")])
def test_chirality_names_the_turn(z, words):
  assert words in chirality(SimpleNamespace(g1=SimpleNamespace(z_dn=z)), 3.0)


def test_chirality_without_a_passing_rung():
  assert "not measured" in chirality(SimpleNamespace(g1=None), 3.0)


def test_m1_report_on_a_passing_ladder(tmp_path, synth_cfg, synth_dir):
  cfg = with_overrides(synth_cfg, {**SHORT, "calibration.ladder": ["default", "vpn"]})
  assert run_m1(cfg, tmp_path, QUIET, pins=synth_dir[1]["pins"]) == 0
  rec = json.loads((tmp_path / "calibration.json").read_text())
  # G0 doubles the synth brain far past 1, where its DNa02 turn too (Minor 7)
  assert rec["result"] == "PASS on rung vpn (LC10a and DNs)"
  assert rec["ladder"]["rung"] == "vpn"
  assert [t["rung"] for t in rec["ladder"]["tried"]] == ["default", "vpn"]
  assert rec["w_scale"] == rec["ladder"]["tried"][1]["g0_scale"] > 1.0
  assert rec["decoder_calibration"]["turn_silent"] is False
  assert rec["wiring"]["label"] == "REAL WIRING"
  assert rec["retina"]["map"] == "affine" and rec["speed_s_per_sim_s"] > 0
  assert "git_sha" in rec and "flycraft_version" in rec  # best-effort code version (Minor 5)
  assert rec["config"]["retina"]["mode"] == "photoreceptor"  # the run's config, not the rung's
  report = (tmp_path / "report.md").read_text()
  for text in ("Rung **vpn** passed", "## Sign rule", "histamine", "| 25.0 |", "![retina map]"):
    assert text in report
  assert (tmp_path / "retina.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("z_lc10a, z_dn, suffix", [
  (4.0, -4.0, " (LC10a and DNs)"),
  (4.0, 1.0, " (LC10a only; DNs not significant)"),
  (-1.0, 4.0, " (DNs only; LC10a not significant)"),
  (1.0, 1.0, "")])
def test_the_result_names_the_signals_that_passed(z_lc10a, z_dn, suffix):
  assert m1report._signal_suffix(SimpleNamespace(z_lc10a=z_lc10a, z_dn=z_dn), 3.0) == suffix


def test_m1_report_on_a_failed_ladder(tmp_path, synth_cfg, synth_dir, capsys):
  cfg = with_overrides(synth_cfg, {**SHORT, "calibration.ladder": ["default"],
                                   "calibration.g1_repeats": 2, "calibration.g1_trial_ms": 100.0})
  assert run_m1(cfg, tmp_path, QUIET, pins=synth_dir[1]["pins"]) == 3  # Minor 6: FAIL exits 3
  assert "FAIL" in capsys.readouterr().err  # Minor 6: FAIL line goes to stderr
  rec = json.loads((tmp_path / "calibration.json").read_text())
  assert rec["result"] == "FAIL: no rung passed G1" and "decoder_calibration" not in rec
  assert "not measured" in (tmp_path / "report.md").read_text()
  assert np.isfinite(rec["speed_s_per_sim_s"])


def test_git_sha_is_none_when_git_hangs(monkeypatch):
  def hang(*a, **k):
    raise subprocess.TimeoutExpired(cmd="git", timeout=5)
  monkeypatch.setattr(m1report.subprocess, "run", hang)
  assert m1report._git_sha() is None
