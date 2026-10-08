"""YAML config with `extends`, frozen dataclasses and validation.

Every run reads one YAML file. A file may name a parent with `extends:` (a path relative
to itself); the child is deep-merged over the parent. Unknown keys are an error, so a typo
fails at startup instead of silently using a default.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

LADDER_RUNGS = ("default", "flip_polarity", "tonic_20", "tonic_50", "vpn")


class ConfigError(ValueError):
  pass


@dataclass(frozen=True)
class ConnectomeConfig:
  dataset: str = "malecns-v1.0"
  data_dir: str = "~/.cache/flycraft"
  include_vnc: bool = True
  wiring: str = "real"
  scramble_seed: int | None = None
  inhibitory_nts: tuple[str, ...] = ("gaba", "glutamate", "histamine")


@dataclass(frozen=True)
class SimConfig:
  dt_ms: float = 0.1
  device: str = "auto"
  dtype: str = "float32"
  mode: str = "event"
  seed: int = 0
  f_poi: float = 250.0


@dataclass(frozen=True)
class EyeConfig:
  polarity: str = "dark_on_bright"


@dataclass(frozen=True)
class RetinaConfig:
  mode: str = "photoreceptor"
  map: str = "auto"
  r_max_hz: float = 150.0
  eye_normalize: bool = True
  lamina_tonic_hz: float = 0.0
  vpn_rate_hz: float = 150.0
  vpn_rf_radius_deg: float = 15.0


@dataclass(frozen=True)
class DecoderConfig:
  turn_pos: tuple[tuple[str, str], ...] = (("DNa02", "R"),)
  turn_neg: tuple[tuple[str, str], ...] = (("DNa02", "L"),)
  fwd: tuple[tuple[str, str], ...] = (("DNp09", "L"), ("DNp09", "R"))
  aotu019_term: bool = False
  tau_ms: float = 150.0
  s0: float = 0.5
  max_turn_deg_s: float = 180.0
  decision_frames: int = 8
  game_fps: float = 22.4
  decision_ms: float = 50.0
  gain_floor_hz: float = 1.0


@dataclass(frozen=True)
class CalibrationConfig:
  g0_blank_ms: float = 2000.0
  g0_max_frac_100hz: float = 0.005
  g0_max_mean_hz: float = 20.0
  g0_max_halvings: int = 8
  g0_precision: float = 0.05
  g1_azimuths_deg: tuple[float, ...] = (-60.0, 0.0, 60.0)
  g1_repeats: int = 10
  g1_trial_ms: float = 2000.0
  g1_radius_deg: float = 10.0
  g1_n_se: float = 3.0
  ladder: tuple[str, ...] = LADDER_RUNGS
  sample_every_ms: float = 50.0
  sample_warmup_ms: float = 500.0


@dataclass(frozen=True)
class Config:
  connectome: ConnectomeConfig = field(default_factory=ConnectomeConfig)
  sim: SimConfig = field(default_factory=SimConfig)
  eye: EyeConfig = field(default_factory=EyeConfig)
  retina: RetinaConfig = field(default_factory=RetinaConfig)
  decoder: DecoderConfig = field(default_factory=DecoderConfig)
  calibration: CalibrationConfig = field(default_factory=CalibrationConfig)


_SECTIONS = {f.name: f.default_factory for f in dataclasses.fields(Config)}


def _deep_merge(base: dict, over: dict) -> dict:
  out = dict(base)
  for k, v in over.items():
    if isinstance(v, dict) and isinstance(out.get(k), dict):
      out[k] = _deep_merge(out[k], v)
    else:
      out[k] = v
  return out


def _oneline(e: Exception) -> str:
  """Collapse a (possibly multi-line) exception message into one line."""
  return " ".join(str(e).split())


def _read_yaml(path: Path, seen: tuple[Path, ...] = ()) -> dict:
  path = path.resolve()
  if path in seen:
    raise ConfigError(f"config extends cycle at {path}")
  try:
    data = yaml.safe_load(path.read_text()) or {}
  except yaml.YAMLError as e:
    raise ConfigError(f"{path}: invalid YAML: {_oneline(e)}") from e
  if not isinstance(data, dict):
    raise ConfigError(f"{path}: top level must be a mapping")
  parent = data.pop("extends", None)
  if parent is None:
    return data
  return _deep_merge(_read_yaml(path.parent / parent, seen + (path,)), data)


def _tuplify(v: Any) -> Any:
  if isinstance(v, list):
    return tuple(_tuplify(x) for x in v)
  return v


def _build(raw: dict) -> Config:
  sections = {}
  for name, section in raw.items():
    if name not in _SECTIONS:
      raise ConfigError(f"unknown config section: {name}")
    if not isinstance(section, dict):
      raise ConfigError(f"config section {name} must be a mapping")
    cls = type(_SECTIONS[name]())
    known = {f.name for f in dataclasses.fields(cls)}
    for k in section:
      if k not in known:
        raise ConfigError(f"unknown config key: {name}.{k}")
    sections[name] = cls(**{k: _tuplify(v) for k, v in section.items()})
  cfg = Config(**sections)
  validate(cfg)
  return cfg


def _parse_override(text: str) -> tuple[list[str], Any]:
  if "=" not in text:
    raise ConfigError(f"override must look like a.b=value: {text!r}")
  key, value = text.split("=", 1)
  key = key.strip()
  try:
    parsed = yaml.safe_load(value)
  except yaml.YAMLError as e:
    raise ConfigError(f"invalid YAML value for {key}: {_oneline(e)}") from e
  return key.split("."), parsed


def _set_path(raw: dict, keys: list[str], value: Any) -> None:
  node = raw
  for k in keys[:-1]:
    node = node.setdefault(k, {})
    if not isinstance(node, dict):
      raise ConfigError(f"cannot set {'.'.join(keys)}: {k} is not a section")
  node[keys[-1]] = value


def load_config(path: str | Path, overrides: tuple[str, ...] | list[str] = ()) -> Config:
  """Load a YAML config, following `extends`, then apply "a.b=value" overrides."""
  raw = _read_yaml(Path(path))
  for text in overrides:
    keys, value = _parse_override(text)
    _set_path(raw, keys, value)
  return _build(raw)


def to_dict(cfg: Config) -> dict:
  """Plain nested dict (lists, not tuples) for JSON records."""
  def plain(v):
    if isinstance(v, tuple):
      return [plain(x) for x in v]
    return v
  return {name: {k: plain(v) for k, v in dataclasses.asdict(getattr(cfg, name)).items()}
          for name in _SECTIONS}


def with_overrides(cfg: Config, changes: dict[str, Any]) -> Config:
  """Return a copy of cfg with dotted-key changes applied, validated."""
  raw = to_dict(cfg)
  for key, value in changes.items():
    _set_path(raw, key.split("."), value)
  return _build(raw)


def _choice(name: str, value: Any, allowed: tuple) -> None:
  if value not in allowed:
    raise ConfigError(f"{name} must be one of {list(allowed)}, got {value!r}")


def _check_kind(label: str, value: Any, default: Any) -> None:
  if isinstance(default, bool):
    if not isinstance(value, bool):
      raise ConfigError(f"{label} must be bool, got {value!r}")
  elif isinstance(default, int):
    if not isinstance(value, int) or isinstance(value, bool):
      raise ConfigError(f"{label} must be int, got {value!r}")
  elif isinstance(default, float):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
      raise ConfigError(f"{label} must be int or float, got {value!r}")
    if not math.isfinite(value):
      raise ConfigError(f"{label} must be finite, got {value!r}")
  elif isinstance(default, str):
    if not isinstance(value, str):
      raise ConfigError(f"{label} must be str, got {value!r}")
  elif isinstance(default, tuple):
    if not isinstance(value, tuple):
      raise ConfigError(f"{label} must be tuple, got {value!r}")
    if default:
      for item in value:
        _check_kind(f"{label} entry", item, default[0])
  # default is None (connectome.scramble_seed): skip, the seed rules below cover it.


def _check_value_types(cfg: Config) -> None:
  for name in _SECTIONS:
    section = getattr(cfg, name)
    for f in dataclasses.fields(section):
      default = f.default
      if default is None:
        continue
      _check_kind(f"{name}.{f.name}", getattr(section, f.name), default)


def _check_value_ranges(cfg: Config) -> None:
  if not cfg.sim.dt_ms > 0:
    raise ConfigError(f"sim.dt_ms must be > 0, got {cfg.sim.dt_ms!r}")
  if cfg.decoder.decision_frames < 1:
    raise ConfigError(f"decoder.decision_frames must be >= 1, got {cfg.decoder.decision_frames!r}")
  if not cfg.decoder.game_fps > 0:
    raise ConfigError(f"decoder.game_fps must be > 0, got {cfg.decoder.game_fps!r}")
  if not cfg.decoder.decision_ms > 0:
    raise ConfigError(f"decoder.decision_ms must be > 0, got {cfg.decoder.decision_ms!r}")
  if not cfg.decoder.tau_ms > 0:
    raise ConfigError(f"decoder.tau_ms must be > 0, got {cfg.decoder.tau_ms!r}")
  if not cfg.decoder.gain_floor_hz > 0:
    raise ConfigError(f"decoder.gain_floor_hz must be > 0, got {cfg.decoder.gain_floor_hz!r}")
  if not 0 < cfg.calibration.g0_precision < 1:
    raise ConfigError(
      f"calibration.g0_precision must be in (0, 1), got {cfg.calibration.g0_precision!r}")
  if cfg.calibration.g0_max_halvings < 1:
    raise ConfigError("calibration.g0_max_halvings must be >= 1, got "
                      f"{cfg.calibration.g0_max_halvings!r}")


def validate(cfg: Config) -> None:
  _check_value_types(cfg)
  _check_value_ranges(cfg)
  c = cfg.connectome
  _choice("connectome.wiring", c.wiring, ("real", "scrambled"))
  seed = c.scramble_seed
  if c.wiring == "scrambled" and (not isinstance(seed, int) or isinstance(seed, bool)):
    raise ConfigError("connectome.wiring=scrambled needs an integer scramble_seed")
  if c.wiring == "scrambled" and seed < 0:
    raise ConfigError(f"connectome.scramble_seed must be >= 0, got {seed!r}")
  if c.wiring == "real" and seed is not None:
    raise ConfigError("connectome.wiring=real must have scramble_seed: null")
  if not c.include_vnc:
    raise ConfigError("connectome.include_vnc=false is not supported in Phase 1")
  _choice("sim.device", cfg.sim.device, ("auto", "cpu", "cuda"))
  _choice("sim.dtype", cfg.sim.dtype, ("float32", "float64"))
  _choice("sim.mode", cfg.sim.mode, ("event", "spmv"))
  _choice("eye.polarity", cfg.eye.polarity, ("dark_on_bright", "bright_on_dark"))
  _choice("retina.mode", cfg.retina.mode, ("photoreceptor", "photoreceptor+vpn"))
  _choice("retina.map", cfg.retina.map, ("auto", "affine", "bands"))
  cal = cfg.calibration
  if not cal.ladder:
    raise ConfigError("calibration.ladder must not be empty")
  for rung in cal.ladder:
    _choice("calibration.ladder entry", rung, LADDER_RUNGS)
  if cal.g1_repeats < 2:
    raise ConfigError("calibration.g1_repeats must be at least 2 (the gate needs a variance)")
  if not cal.g0_blank_ms >= cfg.sim.dt_ms:
    raise ConfigError(
      f"calibration.g0_blank_ms must be >= sim.dt_ms, got {cal.g0_blank_ms!r} < {cfg.sim.dt_ms!r}")
  if cal.sample_every_ms <= 0 or abs(cal.g1_trial_ms / cal.sample_every_ms
                                     - round(cal.g1_trial_ms / cal.sample_every_ms)) > 1e-9:
    raise ConfigError("calibration.g1_trial_ms must be a whole multiple of sample_every_ms")
  if not cfg.sim.dt_ms > 0 or abs(cal.sample_every_ms / cfg.sim.dt_ms
                                  - round(cal.sample_every_ms / cfg.sim.dt_ms)) > 1e-9:
    raise ConfigError("calibration.sample_every_ms must be a whole multiple of sim.dt_ms")
  if abs(cfg.decoder.decision_ms / cfg.sim.dt_ms
         - round(cfg.decoder.decision_ms / cfg.sim.dt_ms)) > 1e-9:
    raise ConfigError("decoder.decision_ms must be a whole multiple of sim.dt_ms")
  if not 0 <= cal.sample_warmup_ms < cal.g1_trial_ms:
    raise ConfigError("calibration.sample_warmup_ms must be in [0, g1_trial_ms)")
  if any(not -180 <= a <= 180 for a in cal.g1_azimuths_deg):
    raise ConfigError("calibration.g1_azimuths_deg entries must be in [-180, 180], got "
                      f"{list(cal.g1_azimuths_deg)}")
  if len(set(cal.g1_azimuths_deg)) < 2:
    raise ConfigError(
      f"calibration.g1_azimuths_deg needs at least two distinct values, got "
      f"{list(cal.g1_azimuths_deg)}")
  for name in ("turn_pos", "turn_neg", "fwd"):
    if name != "fwd" and not getattr(cfg.decoder, name):
      raise ConfigError(f"decoder.{name} must not be empty")
    for group in getattr(cfg.decoder, name):
      if len(group) != 2 or group[1] not in ("L", "R"):
        raise ConfigError(f"decoder.{name} entries must be [type, L|R], got {list(group)}")
