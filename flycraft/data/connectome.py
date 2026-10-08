"""Load the connectome cache built by flycraft.data.build."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flycraft.config import Config
from flycraft.data.build import BUILD_VERSION, build_connectome, sign_from_nt
from flycraft.data.fetch import FILES, file_crc32c

PINNED_SOURCES = {name: crc for name, (crc, _) in FILES.items()}


class StaleCacheError(RuntimeError):
  pass


@dataclass(frozen=True, eq=False)
class Connectome:
  body_id: np.ndarray  # int64 (N,)
  type: np.ndarray  # str (N,)
  superclass: np.ndarray
  cell_class: np.ndarray
  side: np.ndarray  # "L" | "R" | "M" | "?"
  pos: np.ndarray  # float32 (N, 3), micrometres
  pos_source: np.ndarray  # int8: 0 soma, 1 partner mean, 2 class centroid
  hex: np.ndarray  # float32 (N, 2), NaN where unassigned
  nt: np.ndarray  # str (N,)
  sign: np.ndarray  # int8 (N,), from nt and the config's inhibitory_nts
  pre: np.ndarray  # int32 (E,)
  post: np.ndarray  # int32 (E,)
  syn: np.ndarray  # int32 (E,)
  manifest: dict

  @property
  def n(self) -> int:
    return int(self.body_id.size)

  def select(self, pattern: str, side: str | None = None) -> np.ndarray:
    """Indices whose type matches the regex (re.match), optionally on one side."""
    rx = re.compile(pattern)
    hit = np.fromiter((bool(rx.match(t)) for t in self.type), bool, self.n)
    if side is not None:
      hit &= self.side == side
    return np.flatnonzero(hit)


def cache_path(cfg: Config) -> Path:
  return Path(cfg.connectome.data_dir).expanduser() / f"{cfg.connectome.dataset}.npz"


def _stale_reason(manifest: dict, pins: dict[str, str]) -> str | None:
  if manifest.get("build_version") != BUILD_VERSION:
    return f"build_version {manifest.get('build_version')} != {BUILD_VERSION}"
  if manifest.get("sources") != pins:
    return "source files differ from the pinned versions"
  return None


def _read(path: Path, inhibitory_nts) -> Connectome:
  with np.load(path) as z:
    d = {k: z[k] for k in z.files}
  manifest = json.loads(str(d.pop("manifest")))
  # The rule actually in use for this load, which can differ from build time (build_inhibitory_nts)
  # via a config override without a cache rebuild.
  manifest["inhibitory_nts"] = list(inhibitory_nts)
  d.pop("sign")
  return Connectome(**d, sign=sign_from_nt(d["nt"], inhibitory_nts), manifest=manifest)


def load_connectome(cfg: Config, pins: dict[str, str] | None = None, log=print) -> Connectome:
  """Load the cache. A stale cache is rebuilt from the sources, or rejected without them."""
  pins = PINNED_SOURCES if pins is None else pins
  path = cache_path(cfg)
  nts = cfg.connectome.inhibitory_nts
  if not path.exists():
    raise FileNotFoundError(f"no connectome cache at {path}; run `flycraft prep-data`")
  conn = _read(path, nts)
  reason = _stale_reason(conn.manifest, pins)
  if reason is None:
    return conn
  src = path.parent
  missing = [name for name in pins if not (src / name).exists()]
  if missing:
    raise StaleCacheError(f"{path} is stale ({reason}); run `flycraft prep-data`")
  # Verify crc32c against the pins before spending time on a rebuild: a source file that
  # exists by name but doesn't match its pin would just fail staleness again after rebuilding.
  mismatched = sorted(name for name in pins if file_crc32c(src / name) != pins[name])
  if mismatched:
    raise StaleCacheError(f"{path} is stale ({reason}) and {', '.join(mismatched)} do not "
                          "match their pinned crc32c; run `flycraft prep-data`")
  log(f"{path} is stale ({reason}); rebuilding")
  build_connectome(src, path, nts, log=log)
  conn = _read(path, nts)
  reason = _stale_reason(conn.manifest, pins)
  if reason is not None:
    raise StaleCacheError(f"rebuilt {path} is still stale ({reason}); run `flycraft prep-data`")
  return conn
