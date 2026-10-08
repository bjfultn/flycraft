from pathlib import Path

import pytest

from flycraft.config import load_config
from flycraft.data.build import build_connectome
from flycraft.data.connectome import load_connectome
from tests.synth import make_tiny_connectome

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
  d = tmp_path_factory.mktemp("synth")
  meta = make_tiny_connectome(d)
  build_connectome(d, d / "malecns-v1.0.npz", ("gaba", "glutamate", "histamine"),
                   log=lambda s: None)
  return d, meta


@pytest.fixture(scope="session")
def synth_cfg(synth_dir):
  d, _ = synth_dir
  return load_config(CONFIGS / "real.yaml",
                     [f"connectome.data_dir={d}", "sim.device=cpu", "sim.dtype=float64"])


@pytest.fixture(scope="session")
def conn(synth_dir, synth_cfg):
  _, meta = synth_dir
  return load_connectome(synth_cfg, pins=meta["pins"], log=lambda s: None)
