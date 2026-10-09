from pathlib import Path

import pytest

from flycraft.config import ConfigError, load_config, to_dict, with_overrides

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_real_config_loads_with_base_defaults():
  cfg = load_config(CONFIGS / "real.yaml")
  assert cfg.connectome.wiring == "real"
  assert cfg.connectome.scramble_seed is None
  assert cfg.connectome.inhibitory_nts == ("gaba", "glutamate", "histamine")
  assert cfg.sim.dt_ms == 0.1
  assert cfg.decoder.fwd == (("DNp09", "L"), ("DNp09", "R"))
  assert cfg.calibration.ladder[0] == "default"


def test_scrambled_config_has_seed():
  cfg = load_config(CONFIGS / "scrambled.yaml")
  assert cfg.connectome.wiring == "scrambled"
  assert cfg.connectome.scramble_seed == 1


def test_override_strings_parse_as_yaml():
  cfg = load_config(CONFIGS / "scrambled.yaml", ["connectome.scramble_seed=4", "sim.dtype=float64"])
  assert cfg.connectome.scramble_seed == 4
  assert cfg.sim.dtype == "float64"


def test_with_overrides_returns_validated_copy():
  base = load_config(CONFIGS / "real.yaml")
  cfg = with_overrides(base, {"retina.lamina_tonic_hz": 20, "eye.polarity": "bright_on_dark"})
  assert cfg.retina.lamina_tonic_hz == 20
  assert cfg.eye.polarity == "bright_on_dark"
  assert base.retina.lamina_tonic_hz == 0.0
  with pytest.raises(ConfigError):
    with_overrides(base, {"eye.polarity": "plaid"})


def test_to_dict_is_json_friendly():
  d = to_dict(load_config(CONFIGS / "real.yaml"))
  assert d["decoder"]["turn_pos"] == [["DNa02", "R"]]


@pytest.mark.parametrize("override, message", [
  (["connectome.wiring=scrambled", "connectome.scramble_seed=null"], "integer scramble_seed"),
  (["connectome.wiring=scrambled", "connectome.scramble_seed=true"], "integer scramble_seed"),
  (["connectome.scramble_seed=3"], "scramble_seed: null"),
  (["connectome.include_vnc=false"], "not supported in Phase 1"),
  (["connectome.wiring=shuffled"], "connectome.wiring"),
  (["sim.mode=dense"], "sim.mode"),
  (["retina.map=polar"], "retina.map"),
  (["calibration.ladder=[default, louder]"], "ladder"),
  (["decoder.turn_pos=[[DNa02, X]]"], "decoder.turn_pos"),
  (["calibration.g1_repeats=1"], "g1_repeats"),
  (["calibration.g1_trial_ms=2020"], "whole multiple"),
  (["calibration.sample_warmup_ms=2000"], "sample_warmup_ms"),
  (["calibration.g0_precision=0"], "g0_precision"),
  (["calibration.g0_precision=1"], "g0_precision"),
  (["calibration.g0_blank_ms=0.05"], "g0_blank_ms"),
  (["calibration.g0_max_halvings=0"], "g0_max_halvings"),
  (["decoder.gain_floor_hz=0"], "gain_floor_hz"),
  (["decoder.pitch_scale_hz=0"], "pitch_scale_hz"),
  (["decoder.pitch_scale_hz=-1"], "pitch_scale_hz"),
  (["calibration.sample_every_ms=0.25"], "whole multiple"),
  (["decoder.decision_ms=0.25"], "whole multiple"),
  (["calibration.ladder=[]"], "ladder"),
  (["decoder.turn_pos=[]"], "turn_pos"),
  (["decoder.turn_neg=[]"], "turn_neg"),
  (["decoder.pitch_pos=[]"], "pitch_pos"),
  (["decoder.pitch_neg=[]"], "pitch_neg"),
  (["calibration.g1_azimuths_deg=[-200.0, 0.0, 60.0]"], "g1_azimuths_deg"),
  (["calibration.g1_azimuths_deg=[5.0, 5.0]"], "at least two distinct"),
  (["connectome.wiring=scrambled", "connectome.scramble_seed=-1"], "scramble_seed"),
  (["calibration.g2_factor=1"], "g2_factor"),
  (["calibration.g2_factor=-0.5"], "g2_factor"),
])
def test_invalid_configs_fail_with_a_clear_message(override, message):
  with pytest.raises(ConfigError, match=message):
    load_config(CONFIGS / "real.yaml", override)


@pytest.mark.parametrize("override", [
  ["sim.dt_ms=nope"],
  ["sim.dt_ms=0"],
  ["decoder.decision_frames=-5"],
  ["connectome.inhibitory_nts=gaba"],
  ["sim.seed=true"],
  ["retina.eye_normalize=1"],
  ["calibration.g1_azimuths_deg=[a, b]"],
  ["decoder.fwd=[[1, L]]"],
  ["sim.dt_ms=.nan"],
  ["sim.f_poi=.inf"],
  ["calibration.g1_azimuths_deg=[.nan, 0.0, 60.0]"],
])
def test_bad_value_is_rejected(override):
  with pytest.raises(ConfigError):
    load_config(CONFIGS / "real.yaml", override)


def test_int_is_accepted_for_float():
  cfg = load_config(CONFIGS / "real.yaml", ["sim.dt_ms=1"])
  assert cfg.sim.dt_ms == 1


def test_unknown_keys_are_errors(tmp_path):
  (tmp_path / "base.yaml").write_text((CONFIGS / "base.yaml").read_text())
  bad = tmp_path / "bad.yaml"
  bad.write_text("extends: base.yaml\nsim:\n  dtt_ms: 0.2\n")
  with pytest.raises(ConfigError, match="sim.dtt_ms"):
    load_config(bad)
  bad.write_text("extends: base.yaml\nsimm:\n  dt_ms: 0.2\n")
  with pytest.raises(ConfigError, match="simm"):
    load_config(bad)


def test_extends_cycle_is_an_error(tmp_path):
  (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
  (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
  with pytest.raises(ConfigError, match="cycle"):
    load_config(tmp_path / "a.yaml")
