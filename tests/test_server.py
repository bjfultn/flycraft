import contextlib
import inspect
import io
import json
import math
import time

import msgpack
import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from flycraft import eye, protocol
from flycraft.brain.control import Command, Controller, Identity, Runaway
from flycraft.brain.stubs import Oracle, RandomWalker
from flycraft.config import DecoderConfig
from tests.serving import hello, obs, serving

CMD = Command(5.0, 0.5, 1.5, -0.5)


class Spy(Controller):
  def __init__(self, cmd=CMD):
    self.identity = Identity("spy", "real", 3, True, "bright_on_dark", 8)
    self.cmd = cmd
    self.calls = []

  def start_episode(self, episode, seed, phase):
    self.calls.append(("start", (episode, seed, phase)))

  def step(self, eye, reward):
    self.calls.append(("step", (eye.copy(), reward)))
    if isinstance(self.cmd, Exception):
      raise self.cmd
    return self.cmd

  def end_episode(self, score, steps, aborted):
    self.calls.append(("end", (score, steps, aborted)))


def send(ws, kind, **fields):
  ws.send(protocol.encode(protocol.make(kind, **fields)))


def recv(ws):
  return protocol.decode(ws.recv(timeout=5))


def start(ws, mode="train"):
  ws.send(protocol.encode(hello(mode)))
  return recv(ws)


def wait_for(pred, timeout=5.0):
  end = time.monotonic() + timeout
  while not pred():
    assert time.monotonic() < end, "timed out"
    time.sleep(0.01)


def test_handshake_announces_the_brain():
  with serving(Spy()) as (url, _), connect(url) as ws:
    ready = start(ws)
  assert ready == {"type": "ready", "v": 2, "brain_id": "spy", "wiring": "real",
                   "scramble_seed": 3, "plasticity": True, "eye_shape": [30, 72],
                   "decision_frames": 8, "polarity": "bright_on_dark"}


def test_one_act_per_obs_including_the_last():
  spy = Spy()
  img = eye.disk(30.0, 8.0, "dark_on_bright")
  with serving(spy) as (url, _), connect(url) as ws:
    start(ws)
    send(ws, "episode_start", episode=4, seed=4_000_004, phase="eval")
    acts = []
    for k in range(3):
      ws.send(protocol.encode(obs(k, eye=img.tobytes(), last=k == 2, reward=float(k))))
      acts.append(recv(ws))
    send(ws, "episode_end", episode=4, score=2.0, steps=3, aborted=None)
    wait_for(lambda: spy.calls[-1][0] == "end")
  assert [a["step"] for a in acts] == [0, 1, 2]
  assert acts[0] == {"type": "act", "v": 2, "step": 0, "dtheta": 5.0, "speed": 0.5,
                     "turn_raw": 1.5, "fwd_raw": -0.5, "pitch": 0.0, "pitch_raw": 0.0}
  kinds = [c[0] for c in spy.calls]
  assert kinds == ["start", "step", "step", "step", "end"]
  assert spy.calls[0][1] == (4, 4_000_004, "eval")
  eye_seen, reward = spy.calls[3][1]
  assert eye_seen.shape == (30, 72) and eye_seen.dtype.name == "uint8"
  assert (eye_seen == img).all() and reward == 2.0
  assert spy.calls[-1][1] == (2.0, 3, None)


def test_a_second_client_is_busy():
  with serving(Spy()) as (url, _), connect(url) as first:
    start(first)
    with connect(url) as second:
      assert recv(second) == {"type": "error", "v": 2, "message": "busy"}
      with pytest.raises(ConnectionClosed):
        second.recv(timeout=5)
    send(first, "episode_start", episode=0, seed=0, phase="train")  # the first is unaffected
    first.send(protocol.encode(obs(0)))
    assert recv(first)["type"] == "act"


def test_a_new_client_is_welcome_after_the_old_one_leaves():
  spy = Spy()
  with serving(spy) as (url, _):
    with connect(url) as ws:
      start(ws)
      send(ws, "episode_start", episode=0, seed=0, phase="train")
      ws.send(protocol.encode(obs(0)))
      recv(ws)
    wait_for(lambda: spy.calls[-1][0] == "end")
    assert spy.calls[-1][1] == (0.0, 1, "disconnect")
    with connect(url) as ws:
      assert start(ws)["type"] == "ready"


@pytest.mark.parametrize("frame, match", [
  (msgpack.packb({"type": "hello", "v": 3, "client_version": "x", "map": "m", "screen": 84,
                  "mode": "watch"}), "protocol version 3"),
  ("a text frame", "binary frame"),
  (protocol.encode(obs(0)), "expected hello, got obs"),
])
def test_a_bad_first_message_gets_an_error_and_a_close(frame, match):
  with serving(Spy()) as (url, _), connect(url) as ws:
    ws.send(frame)
    err = recv(ws)
    assert err["type"] == "error" and match in err["message"]
    with pytest.raises(ConnectionClosed):
      ws.recv(timeout=5)


@pytest.mark.parametrize("msgs, match", [
  ([obs(0)], "unexpected obs outside an episode"),
  ([protocol.make("episode_start", episode=0, seed=0, phase="train")] * 2,
   "unexpected episode_start in an episode"),
  ([hello()], "unexpected hello outside"),
  ([protocol.make("act", step=0, dtheta=0.0, speed=0.0, turn_raw=0.0, fwd_raw=0.0, pitch=0.0,
                  pitch_raw=0.0)],
   "unexpected act outside"),
])
def test_messages_out_of_order(msgs, match):
  with serving(Spy()) as (url, _), connect(url) as ws:
    start(ws)
    for m in msgs:
      ws.send(protocol.encode(m))
    err = recv(ws)
    assert err["type"] == "error" and match in err["message"]


@pytest.mark.parametrize("cmd", [Command(math.nan, 1.0), Command(0.0, 1.5),
                                 Command(math.inf, 0.5), Runaway("population at 60 Hz")])
def test_a_runaway_command_ends_the_episode_not_the_session(cmd):
  spy = Spy(cmd)
  with serving(spy) as (url, _):
    with connect(url) as ws:
      start(ws)
      send(ws, "episode_start", episode=0, seed=0, phase="train")
      ws.send(protocol.encode(obs(0)))
      assert recv(ws) == {"type": "abort", "v": 2, "reason": "runaway"}
      wait_for(lambda: spy.calls[-1][0] == "end")
      assert spy.calls[-1][1] == (0.0, 1, "runaway")
      spy.cmd = CMD  # the next episode runs normally on the same connection
      send(ws, "episode_start", episode=1, seed=1, phase="train")
      ws.send(protocol.encode(obs(0)))
      assert recv(ws)["type"] == "act"
    wait_for(lambda: len(spy.calls) == 6)
  assert [c[0] for c in spy.calls] == ["start", "step", "end", "start", "step", "end"]
  assert spy.calls[-1][1] == (0.0, 1, "disconnect")


def test_shutdown_aborts_the_client():
  spy = Spy()
  with contextlib.ExitStack() as server:
    url, _ = server.enter_context(serving(spy))
    with connect(url) as ws:
      start(ws)
      send(ws, "episode_start", episode=0, seed=0, phase="train")
      ws.send(protocol.encode(obs(0)))
      recv(ws)
      server.close()  # stop the server with the client still connected
      assert recv(ws) == {"type": "abort", "v": 2, "reason": "shutdown"}
  assert spy.calls[-1] == ("end", (0.0, 1, "shutdown"))


def test_the_trace_has_a_line_per_obs_and_one_for_the_abort():
  spy = Spy()
  trace = io.StringIO()
  with serving(spy, trace=trace) as (url, _), connect(url) as ws:
    start(ws)
    send(ws, "episode_start", episode=2, seed=0, phase="train")
    ws.send(protocol.encode(obs(0, marine_xy=(1.0, 2.0), beacon_xy=(3.0, 4.0))))
    recv(ws)
    spy.cmd = Runaway("too hot")
    ws.send(protocol.encode(obs(1, marine_xy=(5.0, 6.0), beacon_xy=(3.0, 4.0), reward=1.0)))
    assert recv(ws)["reason"] == "runaway"
  lines = [json.loads(line) for line in trace.getvalue().splitlines()]
  assert lines == [
    {"episode": 2, "step": 0, "marine_xy": [1.0, 2.0], "beacon_xy": [3.0, 4.0], "reward": 0.0,
     "score": 0.0, "dtheta": 5.0, "speed": 0.5, "turn_raw": 1.5, "fwd_raw": -0.5, "pitch": 0.0,
     "pitch_raw": 0.0},
    {"episode": 2, "step": 1, "marine_xy": [5.0, 6.0], "beacon_xy": [3.0, 4.0], "reward": 1.0,
     "score": 0.0, "aborted": "runaway", "why": "too hot"}]


# Boundary rule (spec section 3): positions in the obs never reach the brain.

def test_step_takes_only_the_eye_and_the_reward():
  assert list(inspect.signature(Controller.step).parameters) == ["self", "eye", "reward"]


def test_positions_never_reach_the_controller():
  spy = Spy()
  sentinels = {12.345, 67.891, 23.456, 78.912}
  with serving(spy) as (url, _), connect(url) as ws:
    start(ws)
    send(ws, "episode_start", episode=0, seed=0, phase="train")
    ws.send(protocol.encode(obs(0, marine_xy=(12.345, 67.891), beacon_xy=(23.456, 78.912))))
    recv(ws)
  seen = set()
  for _, args in spy.calls:
    for a in args:
      seen.update(a.ravel().tolist() if hasattr(a, "ravel") else [a])
  assert not seen & sentinels


@pytest.mark.parametrize("make", [lambda: Oracle(DecoderConfig(), "dark_on_bright"),
                                  lambda: RandomWalker(DecoderConfig(), "dark_on_bright")])
def test_acts_do_not_depend_on_positions(make):
  imgs = [eye.disk(az, 6.0, "dark_on_bright").tobytes() for az in (-100.0, -20.0, 35.0, 170.0)]

  def run(positions):
    with serving(make()) as (url, _), connect(url) as ws:
      start(ws)
      send(ws, "episode_start", episode=0, seed=99, phase="train")
      acts = []
      for k, (img, (m, b)) in enumerate(zip(imgs, positions, strict=True)):
        ws.send(protocol.encode(obs(k, eye=img, marine_xy=m, beacon_xy=b)))
        acts.append(recv(ws))
    return acts

  a = run([((40.0, 40.0), (60.0, 40.0))] * 4)
  b = run([((1.0, 83.0), (83.0, 1.0)), ((5.0, 5.0), (6.0, 6.0)), ((0.0, 0.0), (0.0, 0.0)),
           ((70.0, 10.0), (10.0, 70.0))])
  assert a == b
  assert len({x["dtheta"] for x in a}) > 1  # the eye images do change the output
