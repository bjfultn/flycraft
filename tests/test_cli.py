import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest
from websockets.sync.client import connect

from flycraft import cli, protocol
from flycraft.brain.decoder import Calibration
from flycraft.brain.view import decode_frame
from flycraft.config import load_config, to_dict
from flycraft.data import connectome, fetch
from tests import serving

REAL = str(cli.Path(__file__).resolve().parent.parent / "configs" / "real.yaml")
SCRAMBLED = REAL.replace("real.yaml", "scrambled.yaml")


@pytest.fixture
def data(tmp_path, synth_dir, monkeypatch):
  """A private copy of the synth data, with the synth files as the pinned sources."""
  src, meta = synth_dir
  for f in src.iterdir():
    if f.suffix in (".feather", ".npz") and "scrambled" not in f.name:
      shutil.copy(f, tmp_path / f.name)
  monkeypatch.setattr(connectome, "PINNED_SOURCES", meta["pins"])
  return tmp_path


def test_wiring_prints_label_and_fingerprint(data, capsys):
  assert cli.main(["wiring", "--config", REAL, "--set", f"connectome.data_dir={data}"]) == 0
  out = capsys.readouterr().out
  assert "REAL WIRING" in out and "fingerprint " in out


def test_wiring_scrambled_twin(data, capsys):
  argv = ["wiring", "--config", SCRAMBLED, "--set", f"connectome.data_dir={data}",
          "--set", "connectome.scramble_seed=4"]
  assert cli.main(argv) == 0
  assert "SCRAMBLED #4" in capsys.readouterr().out
  assert (data / "malecns-v1.0.scrambled-4.npz").exists()


def test_bad_config_exits_2(data, capsys):
  argv = ["wiring", "--config", REAL, "--set", "sim.dtype=float16"]
  assert cli.main(argv) == 2
  assert "config error" in capsys.readouterr().err


def test_missing_config_file_exits_2(tmp_path, capsys):
  assert cli.main(["wiring", "--config", str(tmp_path / "nope.yaml")]) == 2


def test_missing_cache_says_run_prep_data(tmp_path, capsys):
  argv = ["wiring", "--config", REAL, "--set", f"connectome.data_dir={tmp_path}"]
  assert cli.main(argv) == 1
  assert "flycraft prep-data" in capsys.readouterr().err


def test_prep_data_builds_from_verified_files(data, monkeypatch, capsys):
  pins = connectome.PINNED_SOURCES
  monkeypatch.setattr(fetch, "FILES", {n: (c, (data / n).stat().st_size) for n, c in pins.items()})

  def no_network(url):
    raise AssertionError(f"tried to download {url}")

  monkeypatch.setattr(fetch, "_default_opener", no_network)
  (data / "malecns-v1.0.npz").unlink()
  assert cli.main(["prep-data", "--config", REAL, "--set", f"connectome.data_dir={data}"]) == 0
  assert (data / "malecns-v1.0.npz").exists()
  lines = capsys.readouterr().out.splitlines()  # fetch and build log lines, then the summary
  summary = json.loads("\n".join(lines[lines.index("{"):]))
  assert summary["n_neurons"] > 0 and "histamine" in summary["nt_counts"]


def test_malformed_yaml_file_exits_2_with_one_line(tmp_path, capsys):
  bad = tmp_path / "bad.yaml"
  bad.write_text("sim:\n  seed: [1, 2\n")
  assert cli.main(["wiring", "--config", str(bad)]) == 2
  err = capsys.readouterr().err.strip()
  assert err.count("\n") == 0 and "Traceback" not in err and "bad.yaml" in err


def test_malformed_set_value_exits_2_with_one_line(capsys):
  assert cli.main(["wiring", "--config", REAL, "--set", "sim.seed=[1,"]) == 2
  err = capsys.readouterr().err.strip()
  assert err.count("\n") == 0 and "Traceback" not in err


def test_negative_scramble_seed_exits_2_with_one_line(capsys):
  argv = ["wiring", "--config", SCRAMBLED, "--set", "connectome.scramble_seed=-1"]
  assert cli.main(argv) == 2
  err = capsys.readouterr().err.strip()
  assert err.count("\n") == 0 and "Traceback" not in err and "scramble_seed" in err


def test_device_cuda_without_cuda_exits_1_with_one_line(data, tmp_path, capsys):
  argv = ["m1-report", "--config", REAL, "--set", f"connectome.data_dir={data}",
          "--out", str(tmp_path / "out"), "--device", "cuda",
          "--set", "calibration.ladder=[default]"]
  assert cli.main(argv) == 1
  err = capsys.readouterr().err.strip()
  assert err.count("\n") == 0 and "Traceback" not in err and "cuda" in err.lower()


def test_m1_report_via_cli(data, tmp_path):
  out = tmp_path / "m1"
  argv = ["m1-report", "--config", REAL, "--out", str(out), "--device", "cpu",
          "--set", f"connectome.data_dir={data}", "--set", "calibration.ladder=[default]",
          "--set", "calibration.g0_blank_ms=200", "--set", "calibration.g1_repeats=2",
          "--set", "calibration.g1_trial_ms=100", "--set", "calibration.sample_warmup_ms=50"]
  assert cli.main(argv) == 3  # synth photoreceptors alone fail G1 (Minor 6: FAIL exits 3)
  rec = json.loads((out / "calibration.json").read_text())
  assert rec["config"]["sim"]["device"] == "cpu"
  assert (out / "report.md").exists()


def test_unrelated_bug_keeps_its_traceback_when_brain_is_not_loaded(monkeypatch):
  """A non-BrainError bug propagates as itself, without importing the torch-only brain module."""
  def boom(cfg):
    raise ValueError("unrelated bug")
  monkeypatch.setattr(cli, "_wiring", boom)
  monkeypatch.delitem(cli.sys.modules, "flycraft.brain.brain", raising=False)
  with pytest.raises(ValueError, match="unrelated bug"):
    cli.main(["wiring", "--config", REAL])
  assert "flycraft.brain.brain" not in cli.sys.modules


# brain and summarize (M2)

def free_port():
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


def connect_when_up(url, timeout=20.0):
  end = time.monotonic() + timeout
  while True:
    try:
      return connect(url, open_timeout=2)
    except OSError:
      assert time.monotonic() < end, "the brain did not come up"
      time.sleep(0.1)


def test_brain_serves_a_stub_until_sigterm():
  port = free_port()
  root = Path(REAL).parent.parent
  env = dict(os.environ, PYTHONPATH=str(root))
  proc = subprocess.Popen([sys.executable, "-m", "flycraft.cli", "brain", "--stub", "oracle",
                           "--config", REAL, "--port", str(port)],
                          env=env, stderr=subprocess.PIPE, text=True)
  try:
    with connect_when_up(f"ws://127.0.0.1:{port}") as ws:
      ws.send(protocol.encode(serving.hello()))
      ready = protocol.decode(ws.recv(timeout=10))
      cfg = load_config(REAL)
      assert (ready["brain_id"], ready["polarity"], ready["decision_frames"]) == (
        "stub-oracle", cfg.eye.polarity, cfg.decoder.decision_frames)
      proc.send_signal(signal.SIGTERM)
      assert protocol.decode(ws.recv(timeout=10)) == {"type": "abort", "v": 2,
                                                      "reason": "shutdown"}
    _, err = proc.communicate(timeout=10)
  finally:
    proc.kill()
  assert proc.returncode == 0
  assert f"listening on ws://127.0.0.1:{port} as stub-oracle" in err


def test_brain_on_a_port_in_use_exits_1(capsys):
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    s.listen()
    port = s.getsockname()[1]
    assert cli.main(["brain", "--stub", "random", "--config", REAL, "--port", str(port)]) == 1
  err = capsys.readouterr().err.strip().splitlines()
  assert err[-1].startswith(f"flycraft brain: cannot listen on 127.0.0.1:{port}")


def test_brain_with_an_unopenable_trace_file_exits_1(tmp_path, capsys):
  trace_dir = tmp_path / "trace_is_a_dir"
  trace_dir.mkdir()  # opening a directory for append raises OSError before the connectome loads
  argv = ["brain", "--stub", "random", "--config", REAL, "--trace", str(trace_dir)]
  assert cli.main(argv) == 1
  err = capsys.readouterr().err.strip()
  assert err.count("\n") == 0 and "Traceback" not in err and str(trace_dir) in err


# brain (M3): the connectome brain and its view

def calibration_for(data, tmp_path, capsys):
  """An M1 record that passed on the vpn rung, for the synth wiring in data."""
  assert cli.main(["wiring", "--config", REAL, "--set", f"connectome.data_dir={data}"]) == 0
  out = capsys.readouterr().out.splitlines()
  label, fingerprint = out[0].split(":")[0], out[1].split()[1]
  cfg = load_config(REAL, [f"connectome.data_dir={data}"])
  rec = {"wiring": {"label": label, "fingerprint": fingerprint},
         "config": json.loads(json.dumps(to_dict(cfg))), "ladder": {"rung": "vpn"},
         "w_scale": 3.0,
         "decoder_calibration": Calibration(k_turn=4.0, turn_silent=False).to_dict()}
  path = tmp_path / "calibration.json"
  path.write_text(json.dumps(rec))
  return path



def test_g2_via_cli(data, tmp_path, capsys):
  calib = calibration_for(data, tmp_path, capsys)
  out = tmp_path / "g2"
  argv = ["g2", "--config", REAL, "--calibration", str(calib), "--out", str(out),
          "--device", "cpu", "--set", f"connectome.data_dir={data}",
          "--set", "calibration.g1_repeats=2", "--set", "calibration.g1_trial_ms=100",
          "--set", "calibration.sample_warmup_ms=50", "--set", "calibration.sample_every_ms=50",
          "--set", "calibration.g1_azimuths_deg=[-25.0, 25.0]"]
  code = cli.main(argv)
  rec = json.loads((out / "g2.json").read_text())
  assert code == (0 if rec["g2"]["passed"] else 3)
  assert rec["rung"] == "vpn" and rec["w_scale"] == 3.0 and rec["device"] == "cpu"
  d = rec["dopamine_map"]
  assert (d["n_pam"], d["n_ppl1"], d["n_mbons"], d["n_pam_mbons"]) == (10, 8, 6, 2)
  assert d["pam_mbon_types"] == ["MBON01", "MBON04"]
  g = rec["g2"]
  assert g["factor"] == 0.5 and g["n_edges"] > 0
  assert set(g["before"]) == {"blank", "-25", "25"}
  assert set(g["before"]["25"]["pop_hz"]) == {"kc", "mbon_pam", "mbon_other", "dn_turn",
                                              "dn_fwd"}


def brain_argv(data, *more):
  return ["brain", "--config", REAL, "--set", f"connectome.data_dir={data}", *more]


def test_brain_without_a_calibration_says_run_m1_report(data, tmp_path, capsys):
  argv = brain_argv(data, "--calibration", str(tmp_path / "nope.json"), "--view-port", "0")
  assert cli.main(argv) == 1
  err = capsys.readouterr().err.strip().splitlines()
  assert err[-1].startswith("flycraft brain: ") and "m1-report" in err[-1]


def test_brain_with_its_view_port_in_use_exits_1(data, tmp_path, capsys):
  calib = calibration_for(data, tmp_path, capsys)
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    s.listen()
    view_port = s.getsockname()[1]
    argv = brain_argv(data, "--calibration", str(calib), "--port", str(free_port()),
                      "--view-port", str(view_port))
    assert cli.main(argv) == 1
  err = capsys.readouterr().err.strip().splitlines()
  assert err[-1].startswith(f"flycraft brain: cannot listen on 127.0.0.1:{view_port}")


def test_brain_serves_the_fly_and_its_view_until_sigterm(data, tmp_path, capsys):
  calib = calibration_for(data, tmp_path, capsys)
  port, view_port = free_port(), free_port()
  trace = tmp_path / "trace.jsonl"
  got, errors = {"acts": []}, []

  def client():
    # SIGTERM only once the game port answers: the brain's handler is installed by then,
    # where the default action would kill pytest.
    try:
      game = connect_when_up(f"ws://127.0.0.1:{port}", timeout=60)
    except BaseException as e:  # noqa: BLE001 - reported by the main thread
      errors.append(e)
      return
    try:
      with game, connect(f"ws://127.0.0.1:{view_port}/activity") as page:
        game.send(protocol.encode(serving.hello("watch")))
        got["ready"] = protocol.decode(game.recv(timeout=10))
        game.send(protocol.encode(protocol.make("episode_start", episode=1, seed=5,
                                                phase="watch")))
        for k in range(4):  # the page may join the hub after the first obs: send a few
          game.send(protocol.encode(serving.obs(k, marine_xy=(k, 2.0))))
          got["acts"].append(protocol.decode(game.recv(timeout=10)))
        with urllib.request.urlopen(f"http://127.0.0.1:{view_port}/meta", timeout=10) as r:
          got["meta"] = json.loads(r.read())
        got["frame"] = decode_frame(page.recv(timeout=10), got["meta"]["n"])[0]
    except BaseException as e:  # noqa: BLE001
      errors.append(e)
    finally:
      os.kill(os.getpid(), signal.SIGTERM)

  thread = threading.Thread(target=client, daemon=True)
  thread.start()

  def alarm(signum, frame):
    raise TimeoutError("flycraft brain did not return: the SIGTERM path, or the brain's own "
                       "startup, is stuck")

  previous = signal.signal(signal.SIGALRM, alarm)
  signal.alarm(90)  # connect_when_up alone waits up to 60 s: this must never hang pytest
  try:
    code = cli.main(brain_argv(data, "--calibration", str(calib), "--port", str(port),
                               "--view-port", str(view_port), "--trace", str(trace)))
  finally:
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous)
  thread.join(10)
  if errors:
    raise errors[0]
  assert code == 0
  assert got["ready"]["brain_id"] == "fly-real"
  assert [a["type"] for a in got["acts"]] == ["act"] * 4
  assert got["meta"]["condition"]["rung"] == "vpn"
  assert got["frame"]["episode"] == 1
  lines = [json.loads(line) for line in trace.read_text().splitlines()]
  assert [(x["step"], x["marine_xy"]) for x in lines] == [(k, [k, 2.0]) for k in range(4)]
  err = capsys.readouterr().err
  assert f"brain view on http://127.0.0.1:{view_port}" in err and "rung vpn" in err


def record(pilot, score, aborted=None, phase="eval", frames=960, late=0, fps=22.4):
  return {"episode": 0, "seed": 0, "phase": phase, "pilot": pilot, "score": score, "steps": 120,
          "frames": frames, "wall_s": 42.9, "late_frames": late, "max_wait_ms": 3.0,
          "aborted": aborted, "fps": fps, "counts": aborted in (None, "runaway", "stalled")}


def test_summarize_counts_runaways_and_stalls_and_drops_discarded(tmp_path, capsys):
  a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
  a.write_text("\n".join(json.dumps(r) for r in [
    record("stub-oracle", 20.0, late=2), record("stub-oracle", 24.0, fps=None),
    record("stub-oracle", 3.0, aborted="sc2"), record("stub-oracle", 9.0, aborted="runaway"),
    record("stub-oracle", 13.0, aborted="stalled")]))
  b.write_text(json.dumps(record("scripted", 25.0, phase="watch")) + "\n\n")
  assert cli.main(["summarize", str(a), str(b)]) == 0
  out = capsys.readouterr().out.splitlines()
  assert out[0].split() == ["pilot", "phase", "n", "mean", "sd", "min", "max", "runaway",
                            "stall", "discard", "fps", "late"]
  assert out[1].split() == ["scripted", "watch", "1", "25.00", "-", "25", "25", "0", "0", "0",
                            "22.4", "0.0%"]
  assert out[2].split() == ["stub-oracle", "eval", "4", "16.50", "6.76", "9", "24", "1", "1",
                            "1", "22.4", "0.1%"]


def test_summarize_rejects_a_bad_line(tmp_path, capsys):
  bad = tmp_path / "bad.jsonl"
  bad.write_text(json.dumps(record("x", 1.0)) + "\n{\"pilot\": \"x\"}\n")
  assert cli.main(["summarize", str(bad)]) == 1
  err = capsys.readouterr().err.strip()
  assert err.count("\n") == 0 and f"{bad}:2: not a client record" in err


def test_summarize_a_missing_file_exits_1(tmp_path, capsys):
  assert cli.main(["summarize", str(tmp_path / "nope.jsonl")]) == 1
  assert "nope.jsonl" in capsys.readouterr().err


def test_summarize_keeps_train_fps_apart_from_discards(tmp_path, capsys):
  path = tmp_path / "train.jsonl"
  path.write_text(json.dumps(record("scripted", 26.0, phase="train", fps=2707.4)))
  assert cli.main(["summarize", str(path)]) == 0
  assert capsys.readouterr().out.splitlines()[1].split()[-3:] == ["0", "2707.4", "0.0%"]
