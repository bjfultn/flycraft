import json
import math

import numpy as np
import pytest
from websockets.sync.client import connect

from flycraft import eye, protocol
from flycraft.brain import fly as fly_mod
from flycraft.brain.control import Runaway
from flycraft.brain.decoder import Calibration
from flycraft.brain.fly import CalibrationError, Fly, FlyCalibration, load_calibration
from flycraft.brain.retina import RetinaGeometry
from flycraft.brain.wiring import load_wiring
from flycraft.config import to_dict, with_overrides
from tests.serving import hello, obs, serving

DECODER = Calibration(k_turn=4.0, turn_silent=False, p95_turn_hz=10.0)


@pytest.fixture(scope="module")
def geom(conn):
  return RetinaGeometry.build(conn)


@pytest.fixture(scope="module")
def wiring(conn, synth_cfg):
  return load_wiring(synth_cfg, conn, log=lambda s: None)


def m1_record(cfg, wiring, rung="default"):
  """What `flycraft m1-report` writes on a pass, as far as the brain reads it."""
  return {"wiring": {"label": wiring.label, "fingerprint": wiring.fingerprint},
          "config": json.loads(json.dumps(to_dict(cfg))), "ladder": {"rung": rung},
          "w_scale": 0.5, "decoder_calibration": DECODER.to_dict()}


@pytest.fixture
def write(tmp_path):
  def _write(rec):
    path = tmp_path / "calibration.json"
    path.write_text(rec if isinstance(rec, str) else json.dumps(rec))
    return path
  return _write


@pytest.fixture
def make(conn, wiring, geom, synth_cfg):
  # The synth's LC10a -> AOTU019 -> DNa02 chain needs the vpn rung and some gain to steer.
  def _make(view=None, rung="vpn", w_scale=3.0):
    return Fly(conn, wiring, geom, synth_cfg, FlyCalibration(rung, w_scale, DECODER), view=view)
  return _make


def spot(az):
  return eye.disk(az, 10.0, "dark_on_bright", el_deg=8.5)  # where the synth's LC10a look


class Watcher:
  def __init__(self, watching=True):
    self.has_clients = watching
    self.frames = []
    self.cleared = 0

  def publish(self, info, counts, img):
    self.frames.append((info, counts.copy(), img))

  def clear(self):
    self.cleared += 1


# load_calibration

def test_reads_the_m1_record(synth_cfg, wiring, write):
  c = load_calibration(write(m1_record(synth_cfg, wiring, "vpn")), synth_cfg, wiring)
  assert (c.rung, c.w_scale, c.decoder) == ("vpn", 0.5, DECODER)


def test_device_and_data_dir_do_not_matter(synth_cfg, wiring, write):
  rec = m1_record(synth_cfg, wiring)
  rec["config"]["sim"]["device"] = "cuda"
  rec["config"]["connectome"]["data_dir"] = "/somewhere/else"
  rec["config"]["calibration"]["g1_repeats"] = 99
  assert load_calibration(write(rec), synth_cfg, wiring).rung == "default"


def test_refuses_a_calibration_for_other_wiring(synth_cfg, wiring, write):
  rec = m1_record(synth_cfg, wiring)
  rec["wiring"] = {"label": "SCRAMBLED #1", "fingerprint": "f" * 64}
  with pytest.raises(CalibrationError, match=r"is for SCRAMBLED #1 \(ffffffffffff\)"):
    load_calibration(write(rec), synth_cfg, wiring)


def test_refuses_a_calibration_from_another_config(synth_cfg, wiring, write):
  rec = m1_record(synth_cfg, wiring)
  rec["config"]["decoder"]["tau_ms"] = 99.0
  rec["config"]["sim"]["seed"] = 12345
  with pytest.raises(CalibrationError, match="decoder.tau_ms, sim.seed"):
    load_calibration(write(rec), synth_cfg, wiring)


@pytest.mark.parametrize("rec, match", [
  (None, "run `flycraft m1-report`"),
  ("{not json", "cannot read"),
  ({"anything": 1}, "not an M1 calibration record"),
  ("[]", "not an M1 calibration record"),
])
def test_a_missing_or_broken_record_is_a_user_error(synth_cfg, wiring, write, tmp_path, rec,
                                                    match):
  path = tmp_path / "nope.json" if rec is None else write(rec)
  with pytest.raises(CalibrationError, match=match):
    load_calibration(path, synth_cfg, wiring)


def test_a_failed_m1_has_no_brain_to_run(synth_cfg, wiring, write):
  rec = m1_record(synth_cfg, wiring)
  rec["ladder"]["rung"] = None
  del rec["w_scale"], rec["decoder_calibration"]
  with pytest.raises(CalibrationError, match="no rung passed G1"):
    load_calibration(write(rec), synth_cfg, wiring)


def test_an_unknown_rung_is_refused(synth_cfg, wiring, write):
  with pytest.raises(CalibrationError, match="unknown rung 'louder'"):
    load_calibration(write(m1_record(synth_cfg, wiring, "louder")), synth_cfg, wiring)


@pytest.mark.parametrize("mutate, match", [
  (lambda rec: rec["decoder_calibration"].pop("k_turn"), "missing k_turn"),
  (lambda rec: rec["decoder_calibration"].update(k_turn="nan"), "k_turn must be a finite number"),
  (lambda rec: rec["decoder_calibration"].update(turn_silent=0), "turn_silent must be bool"),
  (lambda rec: rec.update(w_scale="abc"), "w_scale must be a number"),
  (lambda rec: rec.update(w_scale=0.0), "w_scale must be finite and > 0"),
  (lambda rec: rec.update(w_scale=float("inf")), "w_scale must be finite and > 0"),
])
def test_a_broken_calibration_value_is_a_user_error(synth_cfg, wiring, write, mutate, match):
  rec = m1_record(synth_cfg, wiring, "vpn")
  mutate(rec)
  with pytest.raises(CalibrationError, match=match):
    load_calibration(write(rec), synth_cfg, wiring)


# Fly

def test_announces_its_condition(make, wiring, synth_cfg):
  f = make()
  assert (f.identity.brain_id, f.identity.wiring, f.identity.plasticity) == (
    "fly-real", wiring.label, False)
  assert f.cfg.retina.mode == "photoreceptor+vpn"  # the rung is applied
  assert f.slices == [10.0] * 5
  assert f.identity.decision_frames == synth_cfg.decoder.decision_frames


def test_a_decision_runs_its_full_ms_when_frame_ms_does_not_divide_it(conn, wiring, geom,
                                                                      synth_cfg):
  # 10 ms (FRAME_MS) is not a whole number of 0.3 ms steps, so slicing by whole steps (not by
  # milliseconds) must still cover exactly one decision's worth of brain time.
  cfg = with_overrides(synth_cfg, {"sim.dt_ms": 0.3, "decoder.decision_ms": 30.0,
                                   "calibration.sample_every_ms": 30.0,
                                   "calibration.g1_trial_ms": 1800.0})
  f = Fly(conn, wiring, geom, cfg, FlyCalibration("vpn", 3.0, DECODER))
  f.start_episode(0, 0, "train")
  f.step(eye.grey(), 0.0)
  assert f.t_ms == pytest.approx(30.0)


def play(f, seed, n=6):
  f.start_episode(0, seed, "train")
  imgs = [spot(az) for az in (-25.0, 25.0)]
  return [f.step(imgs[k % len(imgs)], 0.0) for k in range(n)]


def test_commands_are_finite_and_the_seed_reproduces_them(make):
  f = make()
  a, b = play(f, 7), play(f, 7)
  assert a == b and play(f, 8) != a
  assert any(c.dtheta != 0.0 for c in a)
  assert all(math.isfinite(c.dtheta) and 0.0 <= c.speed <= 1.0 for c in a)


def test_steers_toward_the_spot(make):
  f = make()
  for az in (-25.0, 25.0):  # left of centre is a negative (counter-clockwise) turn
    f.start_episode(0, 3, "train")
    last = [f.step(spot(az), 0.0) for _ in range(3)][-1]
    assert last.dtheta * az > 0, (az, last)


def test_start_episode_resets_the_episode(make):
  f = make()
  play(f, 1)
  f.step(eye.grey(), 1.0)
  f.start_episode(3, 3, "eval")
  assert (f.episode, f.step_no, f.score, f.t_ms) == (3, 0, 0.0, 0.0)
  assert not f._window and f._window_spikes == 0


def test_end_episode_clears_the_view(make):
  w = Watcher()
  f = make(view=w)
  f.start_episode(0, 0, "train")
  f.end_episode(0.0, 0, None)
  assert w.cleared == 1


def test_score_is_the_sum_of_rewards_and_time_is_brain_time(make):
  f = make()
  f.start_episode(0, 0, "train")
  for r in (0.0, 1.0, 0.0, 1.0):
    f.step(eye.grey(), r)
  assert (f.score, f.step_no, f.t_ms) == (2.0, 4, pytest.approx(200.0))


def test_runaway_waits_for_a_full_second_of_brain_time(make, monkeypatch):
  f = make()
  monkeypatch.setattr(fly_mod, "RUNAWAY_HZ", -1.0)  # any rate is too hot
  f.start_episode(0, 0, "train")
  for _ in range(19):  # 950 ms
    f.step(eye.grey(), 0.0)
  with pytest.raises(Runaway, match="over the last 1000 ms of brain time"):
    f.step(eye.grey(), 0.0)


def test_runaway_window_slides(make):
  f = make()
  f.start_episode(0, 0, "train")
  counts = np.zeros(f.brain.conn.n, np.int64)
  keep = np.flatnonzero(f._keep)[0]

  def at(hz):
    counts[keep] = round(hz * f._n_keep * 0.01)
    return counts
  for _ in range(100):
    f._pop_rate(at(40.0), 10.0)  # a full second at 40 Hz is fine
  hot = 0
  with pytest.raises(Runaway):
    while hot < 100:
      hot += 1
      f._pop_rate(at(60.0), 10.0)
  assert 49 <= hot <= 53  # the mean crosses 50 Hz about halfway through the swap


def test_input_neurons_do_not_count_toward_a_runaway(make, monkeypatch):
  f = make()
  monkeypatch.setattr(fly_mod, "RUNAWAY_HZ", 0.0)
  f.start_episode(0, 0, "train")
  counts = np.zeros(f.brain.conn.n, np.int64)
  counts[f.brain.retina.idx] = 1000
  for _ in range(150):
    assert f._pop_rate(counts, 10.0) == 0.0


def test_non_finite_state_is_a_runaway(make):
  f = make()
  f.start_episode(0, 0, "train")
  f.brain.lif.v[0] = float("nan")
  with pytest.raises(Runaway, match="non-finite"):
    f.step(eye.grey(), 0.0)


def test_a_watched_fly_publishes_a_frame_per_slice(make):
  w = Watcher()
  f = make(view=w)
  f.start_episode(2, 2, "watch")
  img = spot(-25.0)
  f.step(img, 1.0)
  assert len(w.frames) == 5
  infos = [i for i, _, _ in w.frames]
  assert [i["t"] for i in infos] == pytest.approx([10.0, 20.0, 30.0, 40.0, 50.0])
  assert [i["reward"] for i in infos] == [1.0, 0.0, 0.0, 0.0, 0.0]
  assert {i["episode"] for i in infos} == {2} and {i["step"] for i in infos} == {0}
  assert set(infos[0]["rates"]) >= {"turn_pos", "turn_neg", "fwd", "mdn"}
  assert w.frames[0][1].shape == (f.brain.conn.n,) and w.frames[0][2] is img
  json.dumps(infos)  # the frame header is plain JSON


def test_an_unwatched_fly_builds_no_frames(make):
  w = Watcher(watching=False)
  f = make(view=w)
  play(f, 0, 2)
  assert w.frames == []


def test_watching_does_not_change_what_the_fly_does(make):
  assert play(make(view=Watcher()), 5) == play(make(), 5)


def test_through_the_server_positions_never_matter(make):
  imgs = [spot(az).tobytes() for az in (-25.0, 25.0, -25.0)]

  def run(positions):
    with serving(make()) as (url, _), connect(url) as ws:
      ws.send(protocol.encode(hello()))
      assert protocol.decode(ws.recv(timeout=10))["brain_id"] == "fly-real"
      ws.send(protocol.encode(protocol.make("episode_start", episode=0, seed=11, phase="train")))
      acts = []
      for k, (img, (m, b)) in enumerate(zip(imgs, positions, strict=True)):
        ws.send(protocol.encode(obs(k, eye=img, marine_xy=m, beacon_xy=b)))
        acts.append(protocol.decode(ws.recv(timeout=10)))
    return acts

  a = run([((40.0, 40.0), (60.0, 40.0))] * 3)
  b = run([((1.0, 83.0), (83.0, 1.0)), ((5.0, 5.0), (6.0, 6.0)), ((70.0, 10.0), (10.0, 70.0))])
  assert a == b and all(x["type"] == "act" for x in a)
  assert any(x["dtheta"] != 0.0 for x in a)
