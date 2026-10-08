import contextlib
import io
import itertools
import json
import math
import socket
import threading
import time

import numpy as np
import pytest
from websockets.sync.server import serve

from flycraft import eye, protocol
from flycraft.brain.control import Command, Controller, Identity
from flycraft.brain.stubs import Oracle, RandomWalker, object_azimuth
from flycraft.config import DecoderConfig
from flycraft.game import render
from flycraft.game.body import NOOP, SELECT, STOP, Action, Body, BodyParams
from flycraft.game.client import (
  BrainPilot,
  Frame,
  GameError,
  Link,
  LinkError,
  Pacer,
  ScriptedPilot,
  SetupError,
  episode_seed,
  main,
  play,
  play_episode,
)
from flycraft.game.tasks import TASKS
from tests.fake_sc2 import FakeGame
from tests.serving import serving

POLARITY = "dark_on_bright"
CMD = Command(10.0, 0.5)


class Brain(Controller):
  """A controller that answers cmd(episode, k) for the k-th obs and records what it saw."""

  def __init__(self, cmd=lambda episode, k: CMD, delay=0.0):
    self.identity = Identity("test", "real", None, False, POLARITY, 8)
    self.cmd, self.delay = cmd, delay
    self.calls, self.eyes = [], []
    self.episode = self.k = 0

  def start_episode(self, episode, seed, phase):
    self.calls.append(("start", episode, seed, phase))
    self.episode, self.k = episode, 0

  def step(self, eye, reward):
    self.calls.append(("step", reward))
    self.eyes.append(eye.copy())
    time.sleep(self.delay)
    self.k += 1
    return self.cmd(self.episode, self.k - 1)

  def end_episode(self, score, steps, aborted):
    self.calls.append(("end", score, steps, aborted))

  def ends(self):
    return [c[1:] for c in self.calls if c[0] == "end"]


@contextlib.contextmanager
def pilot_for(controller, mode="train", timeout=5.0):
  with serving(controller) as (url, _):
    link = Link(url, mode, "MoveToBeacon", 84, timeout)
    try:
      yield BrainPilot(link, Body())
    finally:
      link.close()


def run(make_game, pilot, episodes=1, mode="train", pacer=None):
  out = io.StringIO()
  code = play(make_game, pilot, episodes, 0, mode, 8, pacer or Pacer(None), out=out,
              log=lambda text: None)
  return code, [json.loads(line) for line in out.getvalue().splitlines()]


def wait_for(pred, timeout=5.0):
  end = time.monotonic() + timeout
  while not pred():
    assert time.monotonic() < end, "timed out"
    time.sleep(0.01)


# Pacing (spec 7.5)

class FakeClock:
  def __init__(self):
    self.t = 100.0
    self.slept = []

  def __call__(self):
    return self.t

  def sleep(self, s):
    self.slept.append(s)
    self.t += s


def test_pacer_holds_the_frame_rate():
  clock = FakeClock()
  pacer = Pacer(20.0, clock, clock.sleep)
  pacer.start()
  for _ in range(10):
    clock.t += 0.01  # each frame takes 10 ms of a 50 ms slot
    pacer.tick()
  assert clock.slept == pytest.approx([0.04] * 10)
  assert clock.t == pytest.approx(100.5) and pacer.late == 0


def test_a_late_frame_is_counted_and_the_schedule_restarts():
  clock = FakeClock()
  pacer = Pacer(20.0, clock, clock.sleep)
  pacer.start()
  clock.t += 0.2  # one slow frame, four slots late
  pacer.tick()
  assert pacer.late == 1 and clock.slept == []
  pacer.tick()  # no sprint to catch up: the next frame gets a full slot
  assert clock.slept == pytest.approx([0.05])
  clock.t += 0.07  # 20 ms over: inside the half-slot grace, so not late
  pacer.tick()
  assert pacer.late == 1
  pacer.start()
  assert pacer.late == 0


def test_no_pacer_never_sleeps():
  clock = FakeClock()
  pacer = Pacer(None, clock, clock.sleep)
  pacer.start()
  pacer.tick()
  assert clock.slept == [] and pacer.late == 0


def test_episode_seeds():
  assert episode_seed(0, 3) == 3
  assert episode_seed(7, 12) == 7_000_012


# Cadence (spec 7.5): the pilot decides every 8 frames in either mode and sees the same rewards.

class TickGame:
  """A game with no units: rewards land on given loops."""

  def __init__(self, frames=24, rewards=None):
    self.frames, self.rewards = frames, rewards or {}
    self.received = []

  def reset(self):
    self.loop, self.score = 0, 0.0
    return self._frame(0.0)

  def step(self, action, step_mul):
    self.received.append((action, step_mul))
    reward = 0.0
    for _ in range(step_mul):
      self.loop += 1
      reward += self.rewards.get(self.loop, 0.0)
    self.score += reward
    return self._frame(reward)

  def _frame(self, reward):
    return Frame(np.zeros((84, 84), np.uint8), reward, self.score, self.loop >= self.frames,
                 self.loop, True)

  def close(self):
    pass


class Recorder:
  name, max_wait_ms = "recorder", 0.0

  def __init__(self, on_decide=None):
    self.seen = []
    self.on_decide = on_decide

  def start(self, episode, seed, phase):
    self.seen.append(("start", episode, seed, phase))
    self.steps = 0

  def decide(self, frame, reward):
    self.seen.append((frame.loop, reward, frame.last))
    self.steps += 1
    if self.on_decide:
      self.on_decide()
    return Action("move", (10.0, 10.0))

  def finish(self, aborted):
    pass


@pytest.mark.parametrize("mode, step_mul, steps", [("watch", 1, 24), ("train", 8, 3),
                                                   ("eval", 8, 3)])
def test_decisions_every_8_frames(mode, step_mul, steps):
  game, pilot = TickGame(24, {3: 1.0, 5: 1.0, 8: 1.0, 20: 1.0}), Recorder()
  result = play_episode(game, pilot, 2, 9, mode, 8, Pacer(None))
  assert pilot.seen == [("start", 2, 9, mode), (0, 0.0, False), (8, 3.0, False),
                        (16, 0.0, False), (24, 1.0, True)]
  assert [m for _, m in game.received] == [step_mul] * steps
  moves = [a for a, _ in game.received if a.kind == "move"]
  assert len(moves) == 3  # in watch mode the frames in between are no_ops
  assert all(a == NOOP for a, _ in game.received if a.kind != "move")
  assert (result.score, result.steps, result.frames, result.aborted) == (4.0, 4, 24, None)


def test_a_slow_decision_makes_late_frames_in_watch_mode():
  clock = FakeClock()
  pilot = Recorder(on_decide=lambda: setattr(clock, "t", clock.t + 1.0))
  result = play_episode(TickGame(32), pilot, 0, 0, "watch", 8,
                        Pacer(22.4, clock, clock.sleep))
  assert result.late_frames == 4  # decisions at loops 0, 8, 16 and 24; the one at 32 ends it


# The scripted baseline

def frame(layer, can_move=True, last=False):
  return Frame(layer, 0.0, 0.0, last, 0, can_move)


def test_scripted_pilot_for_defeat_roaches():
  layer = np.zeros((84, 84), np.uint8)
  layer[10:13, 20:23] = render.ENEMY
  layer[30:33, 60:63] = render.ENEMY
  pilot = ScriptedPilot(TASKS["DefeatRoaches"])
  assert pilot.decide(frame(layer, can_move=False), 0.0) == SELECT
  assert pilot.decide(frame(layer), 0.0) == Action("attack", (60.0, 32.0))  # pysc2's: lowest
  assert pilot.decide(frame(np.zeros((84, 84), np.uint8)), 0.0) == NOOP


def test_scripted_pilot():
  layer = np.zeros((84, 84), np.uint8)
  layer[40:43, 60:63] = render.NEUTRAL
  pilot = ScriptedPilot()
  assert pilot.decide(frame(layer, can_move=False), 0.0) == SELECT
  assert pilot.decide(frame(layer), 0.0) == Action("move", (61.0, 41.0))
  assert pilot.decide(frame(layer, last=True), 0.0) == NOOP
  assert pilot.decide(frame(np.zeros((84, 84), np.uint8)), 0.0) == NOOP


# The brain pilot (spec 7.1, 7.5; amendment 5: turn before render)

def test_each_obs_shows_the_turn_the_last_act_asked_for():
  brain = Brain(lambda episode, k: Command(30.0, 0.0))
  game = FakeGame(seed=5, frames=48)
  with pilot_for(brain) as pilot:
    code, recs = run(lambda: game, pilot)
  assert code == 0 and recs[0]["steps"] == 7
  m = render.find(game._frame(0.0).player_relative, render.SELF).xy
  b = render.find(game._frame(0.0).player_relative, render.NEUTRAL).xy
  assert math.dist(m, b) > 10  # the marine stood still, away from the beacon
  bearing = math.degrees(math.atan2(b[1] - m[1], b[0] - m[0]))
  h0 = Body()
  h0.reset(episode_seed(0, 0))
  for k, img in enumerate(brain.eyes):
    want = eye.wrap_deg(bearing - h0.heading - 30.0 * k)
    assert eye.wrap_deg(object_azimuth(img, POLARITY) - want) == pytest.approx(0, abs=2.5), k
  assert game.received[0] == (SELECT, 8)
  assert game.received[1:] == [(STOP, 8)] * 5  # speed 0 stops the marine


def test_two_episodes_one_act_per_obs():
  brain = Brain()
  with pilot_for(brain) as pilot:
    code, recs = run(lambda: FakeGame(frames=48), pilot, episodes=2)
  assert code == 0
  assert [(r["episode"], r["seed"], r["steps"], r["frames"], r["counts"]) for r in recs] == [
    (0, 0, 7, 48, True), (1, 1, 7, 48, True)]
  starts = [c for c in brain.calls if c[0] == "start"]
  assert starts == [("start", 0, 0, "train"), ("start", 1, 1, "train")]
  assert [e[1:] for e in brain.ends()] == [(7, None), (7, None)]
  assert len(brain.eyes) == 14


def test_a_runaway_brain_ends_the_episode_and_it_counts():
  nan = Command(math.nan, 1.0)
  brain = Brain(lambda episode, k: nan if (episode, k) == (0, 2) else CMD)
  with pilot_for(brain) as pilot:
    code, recs = run(lambda: FakeGame(frames=48), pilot, episodes=2)
  assert code == 0
  assert [(r["aborted"], r["counts"], r["steps"]) for r in recs] == [
    ("runaway", True, 3), (None, True, 7)]
  assert [e[1:] for e in brain.ends()] == [(3, "runaway"), (7, None)]


def test_a_runaway_on_the_last_obs_still_counts_as_a_runaway():
  brain = Brain(lambda episode, k: Command(0.0, 2.0) if k == 6 else CMD)
  with pilot_for(brain) as pilot:
    code, recs = run(lambda: FakeGame(frames=48), pilot, episodes=2)
  assert code == 0
  assert [(r["aborted"], r["steps"]) for r in recs] == [("runaway", 7)] * 2
  assert [e[1:] for e in brain.ends()] == [(7, "runaway")] * 2


def test_a_runaway_is_scored_at_the_obs_the_brain_answered():
  # The runaway answers obs 2 (loop 16). The client hears of it at loop 24, after the game
  # scored again at loop 19; the record keeps obs 2's score, as the brain's does.
  brain = Brain(lambda episode, k: Command(math.nan, 1.0) if k == 2 else CMD)
  with pilot_for(brain) as pilot:
    code, recs = run(lambda: TickGame(48, {5: 1.0, 19: 1.0}), pilot)
  assert code == 0
  assert [(r["score"], r["steps"], r["aborted"]) for r in recs] == [(1.0, 3, "runaway")]
  assert brain.ends() == [(1.0, 3, "runaway")]


def test_a_runaway_counts_even_if_sc2_fails_before_the_client_hears_of_it():
  # Obs 5 goes out at loop 40 and SC2 fails in the very next step. The brain answered obs 5
  # with a runaway and has counted the episode, so the client must not rerun it.
  games = []
  brain = Brain(lambda episode, k: Command(math.nan, 1.0) if k == 5 else CMD)
  with pilot_for(brain) as pilot:
    code, recs = run(lambda: games.append(FakeGame(frames=480, fail_at=40)) or games[-1],
                     pilot)
  assert code == 0 and len(games) == 1
  assert [(r["aborted"], r["counts"], r["steps"]) for r in recs] == [("runaway", True, 6)]
  assert brain.ends() == [(recs[0]["score"], 6, "runaway")]
  assert [c for c in brain.calls if c[0] == "start"] == [("start", 0, 0, "train")]


def test_an_sc2_failure_is_discarded_and_the_seed_rerun():
  games = iter([FakeGame(frames=48, fail_at=16), FakeGame(frames=48)])
  made = []

  def make():
    made.append(next(games))
    return made[-1]

  brain = Brain()
  with pilot_for(brain) as pilot:
    code, recs = run(make, pilot)
  assert code == 0
  assert [(r["seed"], r["aborted"], r["counts"]) for r in recs] == [(0, "sc2", False),
                                                                    (0, None, True)]
  assert made[0].closed
  assert [c for c in brain.calls if c[0] == "start"] == [("start", 0, 0, "train")] * 2
  assert [e[-1] for e in brain.ends()] == ["sc2", None]


@pytest.mark.parametrize("at_launch", [True, False])
def test_three_sc2_failures_in_a_row_stop_the_run(at_launch):
  def make():
    if at_launch:
      raise GameError("SC2 did not start")
    return FakeGame(frames=48, fail_at=8)

  brain = Brain()
  with pilot_for(brain) as pilot:
    code, recs = run(make, pilot, episodes=5)
  assert code == 1
  assert [(r["episode"], r["aborted"]) for r in recs] == [(0, "sc2")] * 3
  assert [e[-1] for e in brain.ends()] == ([] if at_launch else ["sc2"] * 3)


def test_only_failures_in_a_row_stop_the_run():
  games = iter([FakeGame(frames=48, fail_at=8), FakeGame(frames=48, fail_after=1),
                FakeGame(frames=48, fail_at=8), FakeGame(frames=48)])
  brain = Brain()
  with pilot_for(brain) as pilot:
    code, recs = run(lambda: next(games), pilot, episodes=3)
  assert code == 0
  assert [(r["episode"], r["aborted"]) for r in recs] == [
    (0, "sc2"), (0, None), (1, "sc2"), (1, "sc2"), (1, None), (2, None)]

def test_a_brain_shutdown_stops_the_run():
  brain = Brain(delay=0.01)
  result = {}
  with contextlib.ExitStack() as server:
    url, _ = server.enter_context(serving(brain))
    link = Link(url, "train", "MoveToBeacon", 84, 5.0)

    def play_on():
      result["code"], result["recs"] = run(lambda: FakeGame(frames=8_000),
                                           BrainPilot(link, Body()))

    worker = threading.Thread(target=play_on)
    worker.start()
    wait_for(lambda: len(brain.eyes) >= 5)
    server.close()  # stop the brain mid-episode
    worker.join(10)
  link.close()
  assert result["code"] == 1
  assert [r["aborted"] for r in result["recs"]] == ["shutdown"]
  assert brain.ends()[-1][-1] == "shutdown"


def test_a_brain_stopped_after_the_last_act_keeps_the_episode():
  brain = Brain()
  with contextlib.ExitStack() as server:
    url, _ = server.enter_context(serving(brain))
    link = Link(url, "train", "MoveToBeacon", 84, 5.0)
    pilot = BrainPilot(link, Body())
    decide = pilot.decide

    def stop_after_the_last_act(frame, reward):
      action = decide(frame, reward)  # on the last frame this also collects the last act
      if frame.last:
        server.close()  # the brain stops before the client's episode_end goes out
      return action

    pilot.decide = stop_after_the_last_act
    code, recs = run(lambda: FakeGame(frames=48), pilot, episodes=2)
  link.close()
  assert code == 1
  assert [(r["episode"], r["steps"], r["aborted"], r["counts"]) for r in recs] == [
    (0, 7, None, True)]
  assert brain.ends() == [(recs[0]["score"], 7, "shutdown")]


@contextlib.contextmanager
def hanging_up_brain():
  """A brain that says ready, then hangs up on the first obs without a word."""
  ready = protocol.make("ready", brain_id="b", wiring="real", scramble_seed=None,
                        plasticity=False, eye_shape=[30, 72], decision_frames=8,
                        polarity=POLARITY)

  def handler(ws):
    ws.recv()
    ws.send(protocol.encode(ready))
    ws.recv()  # episode_start
    ws.recv()  # obs 0

  with serve(handler, "127.0.0.1", 0) as server:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
    server.shutdown()


def test_a_brain_that_vanishes_stops_the_run():
  with hanging_up_brain() as url:
    link = Link(url, "train", "MoveToBeacon", 84, 5.0)
    code, recs = run(lambda: FakeGame(frames=48), BrainPilot(link, Body()))
  assert code == 1
  assert [r["aborted"] for r in recs] == ["brain_lost"]


@pytest.mark.parametrize("pause", [0.0, 0.2])
def test_a_second_client_is_told_busy(monkeypatch, pause):
  from flycraft.game import client

  connect = client.connect

  def slow_connect(*args, **kwargs):
    ws = connect(*args, **kwargs)
    time.sleep(pause)  # with a pause the brain says busy and hangs up before our hello
    return ws

  with serving(Brain()) as (url, _):
    first = Link(url, "train", "MoveToBeacon", 84, 5.0)
    monkeypatch.setattr(client, "connect", slow_connect)
    with pytest.raises(LinkError, match="busy"):
      Link(url, "train", "MoveToBeacon", 84, 5.0)
    first.close()


def test_a_brain_slower_than_the_timeout_stops_the_run():
  brain = Brain(delay=0.5)
  with pilot_for(brain, timeout=0.1) as pilot:
    code, recs = run(lambda: FakeGame(frames=48), pilot)
  assert code == 1
  assert [r["aborted"] for r in recs] == ["brain_lost"]


def test_the_brain_waits_are_measured():
  brain = Brain(delay=0.05)
  with pilot_for(brain) as pilot:
    _, recs = run(lambda: FakeGame(frames=24), pilot)
  assert recs[0]["max_wait_ms"] >= 40


@pytest.mark.parametrize("show_marine, show_beacon", [(True, False), (False, True),
                                                      (False, False)])
def test_an_episode_survives_units_off_screen(show_marine, show_beacon):
  brain = Brain()
  game = FakeGame(frames=48, show_marine=show_marine, show_beacon=show_beacon)
  with pilot_for(brain) as pilot:
    code, recs = run(lambda: game, pilot)
  assert code == 0 and recs[0]["steps"] == 7
  assert all((img == eye.blank(POLARITY)).all() for img in brain.eyes)
  if not show_marine:
    assert all(a in (SELECT, NOOP) for a, _ in game.received)  # no move without a marine


def test_a_marine_hidden_under_the_beacon_walks_on():
  # The first real-brain run lost the marine under the beacon short of scoring, and a
  # body that stops when it loses sight of its marine stood there for the rest of the episode.
  brain = Brain(lambda episode, k: Command(0.0, 0.5))
  seen = np.zeros((84, 84), np.uint8)
  seen[39:42, 39:42] = render.SELF
  seen[45:50, 45:50] = render.NEUTRAL
  hidden = np.where(seen == render.SELF, 0, seen).astype(np.uint8)
  with pilot_for(brain) as pilot:
    pilot.start(0, 0, "train")
    pilot.body.heading = 0.0
    assert pilot.decide(frame(seen), 0.0) == SELECT
    acts = [pilot.decide(frame(img), 0.0) for img in (seen, hidden, hidden)]
    pilot.decide(frame(hidden, last=True), 0.0)
  assert acts == [Action("move", (43.0, 40.0)), Action("move", (46.0, 40.0)),
                  Action("move", (49.0, 40.0))]
  assert not (brain.eyes[2] == eye.blank(POLARITY)).all()  # the fly still sees the beacon


@pytest.mark.parametrize("seed", range(3))
def test_the_oracle_scores_where_the_beacon_hides_the_marine(seed):
  game = FakeGame(seed=seed, frames=480, beacon_r=8.0, score_r=4.5)  # hidden inside 6.5 px
  with pilot_for(Oracle(DecoderConfig(), POLARITY)) as pilot:
    code, recs = run(lambda: game, pilot)
  assert code == 0 and recs[0]["score"] >= 3


def test_the_oracle_beats_the_random_walker():
  def mean_score(controller):
    seeds = itertools.count(100)
    with pilot_for(controller) as pilot:
      code, recs = run(lambda: FakeGame(seed=next(seeds), frames=480), pilot, episodes=4)
    assert code == 0
    return np.mean([r["score"] for r in recs])

  oracle = mean_score(Oracle(DecoderConfig(), POLARITY))
  walker = mean_score(RandomWalker(DecoderConfig(), POLARITY))
  assert oracle >= 4 and oracle > walker + 3


def test_the_fly_sees_every_roach_and_traces_the_nearest():
  brain = Brain(lambda episode, k: Command(0.0, 1.0))
  layer = np.zeros((84, 84), np.uint8)
  layer[39:42, 39:42] = render.SELF
  layer[9:12, 9:12] = render.ENEMY  # far, up and left
  layer[49:52, 59:62] = render.ENEMY  # near, down and right
  trace = io.StringIO()
  with serving(brain, trace=trace) as (url, _):
    link = Link(url, "train", "DefeatRoaches", 84, 5.0)
    try:
      pilot = BrainPilot(link, Body(BodyParams(order="attack")), TASKS["DefeatRoaches"])
      pilot.start(0, 0, "train")
      pilot.body.heading = 0.0
      acts = [pilot.decide(frame(layer), 0.0) for _ in range(2)]
      pilot.decide(frame(layer, last=True), 0.0)
    finally:
      link.close()
  assert acts == [SELECT, Action("attack", (60.0, 50.0))]  # the near roach is 26.6 deg right
  dark = (brain.eyes[0] < 160).any(axis=0)  # columns with something in view
  assert dark[eye.az_to_col(-135.0).round().astype(int)]  # up-left: behind, to the left
  assert dark[eye.az_to_col(math.degrees(math.atan2(10, 20))).round().astype(int)]
  assert json.loads(trace.getvalue().splitlines()[0])["beacon_xy"] == [60.0, 50.0]


def test_the_squad_strikes_the_roach_the_fly_turns_to():
  brain = Brain(lambda episode, k: Command(26.0 if k == 0 else 0.0, 1.0))  # turn right once
  layer = np.zeros((84, 84), np.uint8)
  layer[39:42, 39:42] = render.SELF
  layer[49:52, 59:62] = render.ENEMY  # 26.6 deg right of heading 0, inside the cone
  with serving(brain) as (url, _):
    link = Link(url, "train", "DefeatRoaches", 84, 5.0)
    try:
      pilot = BrainPilot(link, Body(BodyParams(order="attack")), TASKS["DefeatRoaches"])
      pilot.start(0, 0, "train")
      pilot.body.heading = -10.0  # 36.6 deg off: outside the cone until the fly turns
      task = TASKS["DefeatRoaches"]
      assert task.strike(layer, task.targets(layer), (40.0, 40.0), -10.0) is None
      acts = [pilot.decide(frame(layer), 0.0) for _ in range(3)]
      pilot.decide(frame(layer, last=True), 0.0)
    finally:
      link.close()
  strike = Action("attack", (60.0, 50.0))
  assert acts == [SELECT, strike, strike]  # turned 26 deg: 10.6 deg off, so it attacks


def test_the_squad_never_attacks_its_own_marines():
  brain = Brain(lambda episode, k: Command(0.0, 0.5))  # a 3 px step, inside the squad
  layer = np.zeros((84, 84), np.uint8)
  layer[36:45, 36:45] = render.SELF
  layer[9:12, 9:12] = render.ENEMY  # behind and to the left, outside the cone
  with serving(brain) as (url, _):
    link = Link(url, "train", "DefeatRoaches", 84, 5.0)
    try:
      pilot = BrainPilot(link, Body(BodyParams(order="attack")), TASKS["DefeatRoaches"])
      pilot.start(0, 0, "train")
      pilot.body.heading = 0.0
      acts = [pilot.decide(frame(layer), 0.0) for _ in range(2)]
      pilot.decide(frame(layer, last=True), 0.0)
    finally:
      link.close()
  assert acts == [SELECT, Action("attack", (46.0, 40.0))]  # past the squad's edge at x = 44


@pytest.mark.parametrize("face", [None, 0.0])
def test_a_fly_set_to_face_starts_facing_the_nearest_target(face):
  brain = Brain(lambda episode, k: Command(0.0, 0.0))
  layer = np.zeros((84, 84), np.uint8)
  layer[39:42, 69:72] = render.SELF  # the squad at (70, 40)
  layer[9:12, 69:72] = render.ENEMY  # a roach straight up the screen, at (70, 10)
  layer[39:42, 9:12] = render.ENEMY  # one further off, at (10, 40)
  with serving(brain) as (url, _):
    link = Link(url, "train", "DefeatRoaches", 84, 5.0)
    try:
      pilot = BrainPilot(link, Body(BodyParams(order="attack")), TASKS["DefeatRoaches"], face)
      headings = []
      for episode in range(2):
        pilot.start(episode, 1_000_000 + episode, "train")
        start = pilot.body.heading
        pilot.decide(frame(layer), 0.0)
        headings.append((start, pilot.body.heading))
        pilot.decide(frame(layer, last=True), 0.0)
        pilot.finish(None)
    finally:
      link.close()
  if face is None:
    assert all(after == start for start, after in headings)  # the seeded random heading
  else:
    assert all(after == pytest.approx(-90.0) for _, after in headings)  # set every episode


def test_main_plays_defeat_roaches_with_attack_orders(capsys):
  games = []

  def make():
    games.append(FakeGame(frames=480, target=render.ENEMY, targets=3, beacon_r=2.0))
    return games[-1]

  with serving(Oracle(DecoderConfig(), POLARITY)) as (url, _):
    code = main(["--brain", url, "--mode", "train", "--map", "DefeatRoaches"], make_game=make)
  kinds = [a.kind for a, _ in games[0].received]  # fake_sc2 fails an attack on the marine
  assert code == 0 and kinds.count("attack") > 10 * kinds.count("move")
  capsys.readouterr()
  assert main(["--scripted", "--mode", "train", "--map", "DefeatRoaches"], make_game=make) == 0
  recs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
  assert "attack" in {a.kind for a, _ in games[1].received} and recs[0]["score"] >= 3


# Records and the command line

def test_record_fields():
  _, recs = run(lambda: FakeGame(frames=48), ScriptedPilot(), episodes=1, mode="eval")
  assert set(recs[0]) == {"episode", "seed", "phase", "pilot", "score", "steps", "frames",
                          "wall_s", "late_frames", "max_wait_ms", "aborted", "fps", "counts"}
  assert recs[0]["pilot"] == "scripted" and recs[0]["phase"] == "eval"
  assert recs[0]["fps"] > 0


def test_main_plays_the_scripted_baseline(capsys):
  games = []
  code = main(["--scripted", "--mode", "train", "--episodes", "2", "--seed", "3"],
              make_game=lambda: games.append(FakeGame(frames=96)) or games[-1])
  recs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
  assert code == 0 and len(games) == 1 and games[0].closed  # one SC2 for the whole run
  assert [(r["pilot"], r["seed"]) for r in recs] == [("scripted", 3_000_000),
                                                     ("scripted", 3_000_001)]
  assert all(r["score"] > 0 for r in recs)


def test_main_plays_a_brain(capsys):
  with serving(Oracle(DecoderConfig(), POLARITY)) as (url, _):
    code = main(["--brain", url, "--mode", "train"], make_game=lambda: FakeGame(frames=96))
  recs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
  assert code == 0 and recs[0]["pilot"] == "stub-oracle"


def test_main_without_a_brain(capsys):
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
  code = main(["--brain", f"ws://127.0.0.1:{port}"], make_game=lambda: FakeGame())
  assert code == 1
  assert "cannot reach the brain" in capsys.readouterr().err


@pytest.mark.parametrize("who", ["scripted", "brain"])
def test_a_game_that_cannot_start_stops_the_run(capsys, who):
  def make():
    raise SetupError("SC2 cannot start as configured: Map doesn't exist: Nowhere")

  brain = Brain()
  with serving(brain) as (url, _):
    argv = ["--scripted"] if who == "scripted" else ["--brain", url]
    code = main([*argv, "--mode", "train"], make_game=make)
  cap = capsys.readouterr()
  assert code == 1 and cap.out == "" and brain.calls == []
  assert "Map doesn't exist: Nowhere" in cap.err


def test_main_sizes_the_sc2_window(monkeypatch):
  sc2_compat = pytest.importorskip("flycraft.game.sc2_compat")
  seen = {}

  def launch(*args, **kwargs):
    seen.update(kwargs, args=args)
    return FakeGame(frames=48)

  monkeypatch.setattr(sc2_compat, "SC2Game", launch)
  assert main(["--scripted", "--mode", "train", "--seed", "4", "--window", "1600x1200"]) == 0
  assert seen == {"args": ("MoveToBeacon", 84, "train", 4, False), "window": (1600, 1200),
                  "race": "terran"}


def test_main_plays_defeat_marines_as_zerg(monkeypatch):
  sc2_compat = pytest.importorskip("flycraft.game.sc2_compat")
  seen = {}

  def launch(*args, **kwargs):
    seen.update(kwargs, args=args)
    return FakeGame(frames=48)

  monkeypatch.setattr(sc2_compat, "SC2Game", launch)
  assert main(["--scripted", "--mode", "train", "--map", "DefeatMarines"]) == 0
  assert (seen["args"][0], seen["race"]) == ("DefeatMarines", "zerg")


@pytest.mark.parametrize("argv", [
  ["--episodes", "0"], ["--realtime", "--mode", "train"], ["--scripted", "--brain", "ws://x"],
  ["--seed", "-1"], ["--seed", "4294967296"], ["--screen", "0"], ["--step-px", "0"],
  ["--timeout", "0"], ["--timeout", "nan"], ["--window", "12x"], ["--window", "0x960"],
  ["--map", "Nowhere"], ["--face", "-1"], ["--face", "181"], ["--face", "nan"],
  ["--scripted", "--face", "60"]])
def test_main_rejects_bad_arguments(argv):
  with pytest.raises(SystemExit) as e:
    main(argv, make_game=lambda: FakeGame())
  assert e.value.code == 2
