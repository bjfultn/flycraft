import asyncio
import contextlib
import json
import socket
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest
from websockets.sync.client import connect

from flycraft import eye, protocol
from flycraft.brain.decoder import Calibration
from flycraft.brain.fly import Fly, FlyCalibration
from flycraft.brain.retina import RetinaGeometry
from flycraft.brain.view import ViewHub, ViewServer, build_meta, decode_frame, encode_frame
from flycraft.brain.wiring import load_wiring
from flycraft.config import with_overrides

INFO = {"t": 10.0, "episode": 0, "step": 0, "score": 0.0}


def frame(t, n=5):
  return encode_frame({**INFO, "t": t}, np.arange(n), eye.grey())


def wait_for(cond, timeout=5.0):
  end = time.monotonic() + timeout
  while not cond():
    assert time.monotonic() < end, "timed out"
    time.sleep(0.01)


def test_a_frame_round_trips():
  counts = np.array([0, 1, 7, 300, 255])
  img = eye.disk(20.0, 10.0, "dark_on_bright")
  info, got, got_img = decode_frame(encode_frame(INFO, counts, img), 5)
  assert info == INFO
  assert got.tolist() == [0, 1, 7, 255, 255]  # counts saturate at 255
  np.testing.assert_array_equal(got_img, img)


def test_a_frame_needs_a_whole_eye():
  with pytest.raises(ValueError, match="2160 pixels"):
    encode_frame(INFO, np.zeros(3), np.zeros((30, 71), np.uint8))


def test_the_hub_keeps_only_the_newest_frame():
  hub = ViewHub()
  assert hub.latest() == (None, 0) and not hub.has_clients
  hub.put(b"a")  # no loop bound yet: nothing to wake, and no error
  hub.put(b"b")
  assert hub.latest() == (b"b", 2)


def test_clear_drops_the_frame_without_waking_anyone():
  hub = ViewHub()
  hub.put(b"a")
  hub.clear()  # no loop bound: clearing must not try to wake anyone either
  frame, seq = hub.latest()
  assert frame is None and seq == 1  # the old frame is gone, the sequence is untouched


@pytest.fixture(scope="module")
def fly(conn, synth_cfg):
  wiring = load_wiring(synth_cfg, conn, log=lambda s: None)
  calib = FlyCalibration("vpn", 3.0, Calibration(k_turn=4.0, turn_silent=False))
  return Fly(conn, wiring, RetinaGeometry.build(conn), synth_cfg, calib)


@contextlib.contextmanager
def viewing(meta=None, positions=None, classes=None):
  """Yield (base url, hub) for a ViewServer on a free port in a background thread."""
  hub = ViewHub()
  n = 5
  meta = meta or {"n": n}
  positions = np.zeros((n, 3)) if positions is None else positions
  classes = np.zeros(n) if classes is None else classes
  server = ViewServer(hub, meta, positions, classes, port=0, log=lambda text: None)
  loop = asyncio.new_event_loop()
  stop = asyncio.Event()
  bound: list[int] = []
  up = threading.Event()

  def ready(port):
    bound.append(port)
    up.set()

  thread = threading.Thread(target=loop.run_until_complete, args=(server.run(stop, ready),),
                            daemon=True)
  thread.start()
  assert up.wait(10), "view server did not start"
  try:
    yield f"127.0.0.1:{bound[0]}", hub
  finally:
    loop.call_soon_threadsafe(stop.set)
    thread.join(10)
    assert not thread.is_alive(), "view server did not stop"
    loop.close()


def get(url):
  with urllib.request.urlopen(f"http://{url}", timeout=10) as r:
    return r.status, r.headers["Content-Type"], r.read()


def test_serves_the_page_and_the_brain():
  pos = np.array([[1.5, 2, 3]] * 5)
  with viewing({"n": 5, "x": "y"}, pos, np.array([0, 1, 2, 1, 0])) as (base, _):
    status, kind, body = get(f"{base}/")
    assert status == 200 and kind.startswith("text/html") and b"view.js" in body
    for path in ("/view.js", "/three.module.min.js", "/OrbitControls.js"):
      status, kind, body = get(f"{base}{path}")
      assert status == 200 and kind.startswith("text/javascript") and body
    assert json.loads(get(f"{base}/meta")[2]) == {"n": 5, "x": "y"}
    got = np.frombuffer(get(f"{base}/positions.bin")[2], "<f4").reshape(5, 3)
    np.testing.assert_array_equal(got, pos)
    assert get(f"{base}/classes.bin?v=1")[2] == bytes([0, 1, 2, 1, 0])  # a query is ignored
    with pytest.raises(urllib.error.HTTPError) as e:
      get(f"{base}/../pyproject.toml")
    assert e.value.code == 404


def test_a_client_that_never_hangs_up_does_not_hold_up_shutdown():
  with viewing() as (base, _):
    host, port = base.split(":")
    sock = socket.create_connection((host, int(port)), timeout=5)
    sock.sendall(b"GET /meta HTTP/1.1\r\nHost: x\r\n\r\n")
    assert sock.recv(4096).startswith(b"HTTP/1.1 200")
    start = time.monotonic()
  try:
    assert time.monotonic() - start < 3.0
  finally:
    sock.close()


def test_a_page_gets_the_current_frame_then_the_newest():
  with viewing() as (base, hub):
    hub.publish({**INFO, "t": 10.0}, np.arange(5), eye.grey())
    with connect(f"ws://{base}/activity") as ws:
      wait_for(lambda: hub.has_clients)
      info, counts, _ = decode_frame(ws.recv(timeout=5), 5)
      assert info["t"] == 10.0 and counts.tolist() == [0, 1, 2, 3, 4]
      for t in range(20, 220, 10):  # faster than anyone reads: frames in between may drop
        hub.put(frame(float(t)))
      seen = []
      while not seen or seen[-1] != 210.0:
        seen.append(decode_frame(ws.recv(timeout=5), 5)[0]["t"])
      assert seen == sorted(seen)  # never an older frame after a newer one
    wait_for(lambda: hub.clients == 0)


def test_two_pages_each_get_frames():
  with viewing() as (base, hub):
    hub.put(frame(10.0))
    with connect(f"ws://{base}/activity") as a, connect(f"ws://{base}/activity") as b:
      wait_for(lambda: hub.clients == 2)
      assert decode_frame(a.recv(timeout=5), 5)[0]["t"] == 10.0
      assert decode_frame(b.recv(timeout=5), 5)[0]["t"] == 10.0
      hub.put(frame(20.0))
      assert decode_frame(a.recv(timeout=5), 5)[0]["t"] == 20.0
      assert decode_frame(b.recv(timeout=5), 5)[0]["t"] == 20.0
    wait_for(lambda: hub.clients == 0)


def test_meta_describes_the_fly(fly, conn):
  meta, pos, classes = build_meta(fly)
  json.dumps(meta)
  assert meta["n"] == conn.n and pos.shape == (conn.n, 3) and pos.dtype == np.float32
  assert np.isfinite(pos).all()
  assert classes.shape == (conn.n,) and classes.max() < len(meta["classes"])
  groups = {g["key"]: g for g in meta["groups"]}
  assert list(groups) == ["pr_L", "pr_R", "lc10a_L", "lc10a_R", "turn_neg", "turn_pos", "fwd",
                          "mdn", "pam", "ppl1", "kc", "mbon"]
  assert {k for k, g in groups.items() if g["dn"]} == {"turn_neg", "turn_pos", "fwd", "mdn"}
  assert groups["turn_pos"]["label"] == "DNa02 R" and groups["turn_neg"]["label"] == "DNa02 L"
  assert all(g["idx"] for k, g in groups.items() if k != "fwd")  # synth has every group
  for g in groups.values():
    assert all(0 <= i < conn.n for i in g["idx"])
  c = meta["condition"]
  assert (c["brain_id"], c["rung"], c["w_scale"], c["plasticity"]) == ("fly-real", "vpn", 3.0,
                                                                       False)
  assert c["fingerprint"] == fly.fingerprint
  assert c["max_dtheta"] == pytest.approx(180.0 * 8 / 22.4)
  assert meta["eye_shape"] == list(protocol.EYE_SHAPE)


def test_meta_marks_an_empty_group_as_none(conn, synth_cfg):
  cfg = with_overrides(synth_cfg, {"decoder.fwd": []})
  wiring = load_wiring(cfg, conn, log=lambda s: None)
  calib = FlyCalibration("vpn", 3.0, Calibration(k_turn=4.0, turn_silent=False))
  f = Fly(conn, wiring, RetinaGeometry.build(conn), cfg, calib)
  meta, _, _ = build_meta(f)
  json.dumps(meta)  # still JSON-serializable with an empty group
  fwd = {g["key"]: g for g in meta["groups"]}["fwd"]
  assert fwd["label"] == "none" and fwd["idx"] == []


def test_a_watched_fly_streams_to_the_page(fly):
  meta, pos, classes = build_meta(fly)
  with viewing(meta, pos, classes) as (base, hub):
    fly.view = hub
    try:
      with connect(f"ws://{base}/activity") as ws:
        wait_for(lambda: hub.has_clients)
        fly.start_episode(4, 4, "watch")
        fly.step(eye.disk(-25.0, 10.0, "dark_on_bright", el_deg=8.5), 0.0)
        info, img = {}, None
        while info.get("t") != pytest.approx(50.0):
          info, _, img = decode_frame(ws.recv(timeout=10), meta["n"])
        assert info["episode"] == 4 and img is not None and img.shape == protocol.EYE_SHAPE
    finally:
      fly.view = None
