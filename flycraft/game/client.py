"""The game client (spec sections 3, 7 and 10): SC2 on one side, the brain's websocket on the other.

The client owns the marine's body and renders the eye image; it knows the game and nothing
about neurons. It runs on Windows next to SC2, logs to stderr and prints one JSON line per
episode to stdout. With --join it plays one Skirmish match against a person instead
(spec M6): flycraft-match hosts it, and the fly joins in real time.

Cadence (spec 7.5): one decision every decision_frames game frames (8, from the brain's ready).
- train and eval: step_mul 8, so every env step is a decision, as fast as the brain answers.
- watch: step_mul 1, paced at 22.4 frames per second. The frames between decisions are
  no_ops, which leave the current move order running.
- real time (watch with --realtime, and a match): step_mul 1, and SC2 keeps time, so the game
  runs on while the brain thinks. The client never waits for an act: it plays no_ops until the
  act is in, then decides on the newest frame. Waiting instead aimed each order from a frame
  about 14 game loops old, and the squad had walked onto the spot by the time SC2 got it, so
  SC2 read the order as "attack this zergling" (M6, docs/m6/README.md).

Pipelining with one decision of latency (spec 7.5; amendment 5, turn before render). At
decision k the pilot collects act k-1, turns the body by its dtheta, renders obs k with the
new heading and sends it, then orders the move for act k-1's speed. The brain works on obs k
while the game plays the next 8 frames. The brain answers the episode's last obs too, and
the pilot drains that act.

Bodies (--body, M6, docs/m6/README.md, "The flying fly" and "The compass fly"): the walking fly
(the default) has a heading, turns by dtheta and sees the ground around it (render.render_eye).
The flying fly turns the same way but looks down on the map from above it (render.map_eye). The
compass fly also looks down on the map, north up (render.compass_eye), and sets an absolute
heading from both brain outputs at once: dtheta for east/west, pitch for north/south.

Failures (spec sections 9 and 14):
- a runaway brain (abort "runaway") ends the episode with the score of the obs it answered,
  and it counts, even if SC2 fails before the client hears of it;
- a stalled fly (Stall, --stall) ends the episode the same way, aborted "stalled", and it
  counts: the brain asked for no turn at all, the score held and the squad stayed put for
  that many decisions in a row. Never in a match;
- SC2 failing (GameError) discards the episode and relaunches SC2 for the same seed; three
  failures in a row stop the run;
- the brain going away, shutting down, timing out or breaking the protocol stops the run;
- a game that cannot start as configured (SetupError, a bad map name) stops the run.
A stopped run exits 1. An episode played to its end is recorded before the brain is told, so
a brain that stops at that moment does not lose it.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
from websockets.exceptions import WebSocketException
from websockets.sync.client import connect

from flycraft import protocol
from flycraft.game import render
from flycraft.game.body import NOOP, SELECT, Action, Body, BodyParams
from flycraft.game.tasks import TASKS, Task, nearest

CLIENT_VERSION = "flycraft-client 0.1.0"
BRAIN_URL = "ws://127.0.0.1:8765"
GAME_FPS = 22.4  # SC2 "faster" speed
MAX_FAILURES = 3  # SC2 failures in a row before the run stops
MAX_SEED = 2**32 - 1  # SC2 takes a 32-bit random seed
WINDOW = (1280, 960)  # the SC2 window, spec section 11
COUNTED = (None, "runaway", "stalled")  # aborted values whose episodes count (spec section 9)
STALL_DECISIONS = 25  # --stall's default
STALL_PX = 2.0
COMPASS_HOLD = 0.05  # below this on both axes at once, the compass fly keeps its last heading


def _log(text: str) -> None:
  print(f"flycraft-client: {text}", file=sys.stderr, flush=True)


class GameError(Exception):
  """SC2 failed. The episode is discarded and SC2 relaunched."""


class SetupError(Exception):
  """The game cannot start as configured. Relaunching will not help, so the run stops."""


class LinkError(Exception):
  """The brain went away, timed out or broke the protocol. The run stops."""


class Stalled(Exception):
  """The fly stalled (Stall). The episode ends with the score of its last obs, and counts."""


class BrainAbort(Exception):
  """The brain sent abort in place of an act."""

  def __init__(self, reason: str):
    super().__init__(f"the brain aborted: {reason}")
    self.reason = reason


@dataclass(frozen=True)
class Frame:
  """One game step as the client sees it. reward is this step's; score is the running total."""
  player_relative: np.ndarray  # screen x screen, indexed [y, x]
  reward: float
  score: float
  last: bool
  loop: int  # the game loop
  can_move: bool  # Move_screen is available, so the marine is selected
  outcome: int | None = None  # a match's last frame: 1 the fly won, -1 it lost, 0 a tie


class Game(Protocol):
  def reset(self) -> Frame: ...

  def step(self, action: Action, step_mul: int) -> Frame: ...

  def close(self) -> None: ...


def episode_seed(run_seed: int, episode: int) -> int:
  return run_seed * 1_000_000 + episode


class Pacer:
  """Holds a loop to a frame rate and counts the frames that came in late (spec 7.5)."""

  def __init__(self, fps: float | None, clock=time.monotonic, sleep=time.sleep):
    self.period = 1.0 / fps if fps else None
    self.clock, self.sleep = clock, sleep
    self.late = 0
    self._due = 0.0

  def start(self) -> None:
    self.late = 0
    self._due = self.clock()

  def tick(self) -> None:
    """Wait for the next frame's slot. A frame more than half a period late counts as late
    and restarts the schedule, so the game never sprints to catch up."""
    if self.period is None:
      return
    self._due += self.period
    now = self.clock()
    if now < self._due:
      self.sleep(self._due - now)
    elif now - self._due > self.period / 2:
      self.late += 1
      self._due = now


class Link:
  """The client's end of the brain's websocket (spec section 10)."""

  def __init__(self, url: str, mode: str, map_name: str, screen: int, timeout: float = 30.0):
    self.timeout = timeout
    self._held = None  # a message poll() found, for the next recv
    self._stack = contextlib.ExitStack()
    try:
      self.ws = self._stack.enter_context(connect(url, open_timeout=timeout))
    except (OSError, WebSocketException) as e:
      raise LinkError(f"cannot reach the brain at {url}: {e}") from e
    self.send("hello", client_version=CLIENT_VERSION, map=map_name, screen=screen, mode=mode)
    self.ready = self.recv("ready")

  def send(self, kind: str, **fields) -> None:
    try:
      self.ws.send(protocol.encode(protocol.make(kind, **fields)))
    except WebSocketException as e:
      self._said_why()
      raise LinkError(f"lost the brain: {e}") from e

  def _said_why(self) -> None:
    """A brain that ends the session sends abort or error, then closes. If the close beat our
    send, that message is still buffered: raise it, so the run reports the brain's reason
    (shutdown, busy) rather than a lost brain."""
    try:
      msg = protocol.decode(self._next(0))
    except (TimeoutError, WebSocketException, protocol.ProtocolError):
      return
    if msg["type"] == "abort":
      raise BrainAbort(msg["reason"])
    if msg["type"] == "error":
      raise LinkError(f"the brain says: {msg['message']}")

  def poll(self) -> bool:
    """Whether recv would answer at once: a message has come (it is held for recv), or the
    brain has gone, which recv then reports."""
    if self._held is None:
      try:
        self._held = self.ws.recv(timeout=0)
      except TimeoutError:
        return False
      except WebSocketException:
        return True
    return True

  def _next(self, timeout: float):
    if self._held is not None:
      data, self._held = self._held, None
      return data
    return self.ws.recv(timeout=timeout)

  def recv(self, kind: str) -> dict:
    """The next message, which must be `kind`; an abort raises BrainAbort."""
    try:
      msg = protocol.decode(self._next(self.timeout))
    except TimeoutError as e:
      raise LinkError(f"no {kind} from the brain within {self.timeout:g} s") from e
    except WebSocketException as e:
      raise LinkError(f"lost the brain: {e}") from e
    except protocol.ProtocolError as e:
      raise LinkError(f"bad message from the brain: {e}") from e
    if msg["type"] == "abort":
      raise BrainAbort(msg["reason"])
    if msg["type"] == "error":
      raise LinkError(f"the brain says: {msg['message']}")
    if msg["type"] != kind:
      raise LinkError(f"expected {kind} from the brain, got {msg['type']}")
    return msg

  def close(self) -> None:
    self._stack.close()


class ScriptedPilot:
  """pysc2's scripted agent for the task, on the fly's decision cadence: the ceiling baseline."""

  name = "scripted"
  body_name = None  # no fly's body
  max_wait_ms = 0.0
  steps = 0

  def __init__(self, task: Task = TASKS["MoveToBeacon"]):
    self.task = task

  def start(self, episode: int, seed: int, phase: str) -> None:
    self.steps = 0

  def ready(self) -> bool:
    return True

  def decide(self, frame: Frame, reward: float) -> Action:
    self.steps += 1
    if frame.last:
      return NOOP
    if not frame.can_move:
      return SELECT
    xy = self.task.aim(frame.player_relative)
    return NOOP if xy is None else Action(self.task.order, xy)

  def drain(self) -> str | None:
    return None

  def finish(self, aborted: str | None) -> None:
    pass


class Stall:
  """Whether the fly has stalled: for `decisions` decisions in a row (so decisions + 1 obs and
  their acts), the brain asked for no turn at all, the score held, and the squad's middle
  stayed within `px` of where it was at the first. An obs with no squad on screen breaks the
  run. 0 decisions never stalls. Checked on 180 recorded episodes, at 25 decisions it never
  stopped an episode that would have scored again (docs/m6/README.md, "Stall stop")."""

  def __init__(self, decisions: int = STALL_DECISIONS, px: float = STALL_PX):
    self.decisions, self.px = decisions, px
    self.reset()

  def reset(self) -> None:
    self._rows: deque[tuple[float, tuple[float, float] | None, float]] = deque(
      maxlen=self.decisions + 1)
    self._obs: tuple[float, tuple[float, float] | None] | None = None

  def obs(self, score: float, xy: tuple[float, float] | None) -> None:
    """An obs went to the brain, with this score and the squad's middle (None: not seen)."""
    self._obs = (score, xy)

  def act(self, dtheta: float) -> bool:
    """The act for the last obs came, asking for this turn; return whether the fly stalled."""
    if self._obs is None:
      return False
    self._rows.append((*self._obs, dtheta))
    self._obs = None
    if self.decisions == 0 or len(self._rows) <= self.decisions:
      return False
    score0, xy0, _ = self._rows[0]
    return xy0 is not None and all(
      xy is not None and score == score0 and dtheta == 0.0 and math.dist(xy, xy0) <= self.px
      for score, xy, dtheta in self._rows)


class BrainPilot:
  """Flies the body from the brain's acts, one decision behind. The obs's beacon_xy is the
  target nearest the marine, for the brain's trace. With `face` set (degrees, a demo setting),
  each episode's first sight of the marine and a target turns the fly to within `face` of the
  nearest target, before the brain sees anything. With `stall` set, a fly that stalls for that
  many decisions (Stall) ends the episode: decide raises Stalled. With `search` set, the body
  makes a search turn after that many decisions in a row with no turn asked (_search). With
  `body_kind` "fly" the fly flies over the map instead of walking and looks down on it; with
  "compass" it also flies over the map, north up, and sets its heading from both brain outputs
  at once instead of turning (_steer), so stall and search do not apply."""

  def __init__(self, link: Link, body: Body, task: Task = TASKS["MoveToBeacon"],
               face: float | None = None, stall: int = 0, search: int = 0,
               body_kind: str = "walk", one_dot: bool = False):
    self.link, self.body, self.task, self.face = link, body, task, face
    self.stall = Stall(stall)
    self.search = search
    self.body_kind = body_kind
    self.one_dot = one_dot
    self.body_name = body_kind
    self.name = link.ready["brain_id"]
    self.polarity = link.ready["polarity"]
    self.max_wait_ms = 0.0
    self._open = False  # the brain has an episode open
    self._waiting = False  # an obs is waiting for its act
    self._since: float | None = None  # when ready() first found the act not in

  def start(self, episode: int, seed: int, phase: str) -> None:
    self.body.reset(seed)
    self.link.send("episode_start", episode=episode, seed=seed, phase=phase)
    self.episode, self.seed, self.steps, self.score = episode, seed, 0, 0.0
    self._faced = self.face is None
    self.stall.reset()
    self._still, self._way = 0, 0.0
    self._ways = np.random.default_rng((seed, 2))
    self.max_wait_ms = 0.0
    self._open, self._waiting = True, False
    self._since = None

  def ready(self) -> bool:
    """Whether decide can go without waiting: no act is due, or it has come. The wait for the
    brain is timed from the first call that finds it not in."""
    if not self._waiting or self.link.poll():
      return True
    if self._since is None:
      self._since = time.monotonic()
    return False

  def decide(self, frame: Frame, reward: float) -> Action:
    compass = self.body_kind == "compass"
    act = self._collect() if self._waiting else None  # act k-1
    if act is not None:
      if self.stall.act(act["dtheta"]) and not frame.last:
        raise Stalled()
      if compass:
        self._steer(act["dtheta"], act["pitch"])
      else:
        self.body.turn(act["dtheta"])
        self._search(act["dtheta"])
    marine = render.find(frame.player_relative, render.SELF)  # the squad's middle
    targets = self.task.targets(frame.player_relative)
    marine_xy = self.body.locate(marine.xy if marine else None)  # it may be under the beacon
    near = nearest(marine_xy, targets)
    if not compass and not self._faced and marine_xy is not None and near is not None:
      self.body.face(marine_xy, near.xy, self.face, self.seed)
      self._faced = True
    if self.one_dot:  # the eye gets the nearest target only, one dot as in the grid sweep
      targets_seen = [near] if near is not None else []
    else:
      targets_seen = targets
    if compass:
      img = render.compass_eye(marine_xy, targets_seen, self.polarity)
    elif self.body_kind == "fly":
      img = render.map_eye(marine_xy, targets_seen, self.body.heading, self.polarity)
    else:
      img = render.render_eye(marine_xy, targets_seen, self.body.heading, self.polarity)
    self.link.send("obs", step=self.steps, eye=img.tobytes(), reward=reward, score=frame.score,
                   last=frame.last, marine_xy=marine.xy if marine else None,
                   beacon_xy=near.xy if near else None)
    self.stall.obs(frame.score, marine.xy if marine else None)
    self.steps += 1
    self.score = frame.score
    self._waiting = True
    if frame.last:
      self._collect()  # the brain answers the last obs too; that act is never applied
      return NOOP
    if act is None:
      return SELECT  # decision 0 has no act yet; select the marine (spec 7.1)
    strike = self.task.strike(frame.player_relative, targets, marine_xy, self.body.heading)
    own = np.asarray(frame.player_relative) == render.SELF
    return self.body.motor(marine_xy, frame.can_move, act["speed"], strike, own)

  def _search(self, dtheta: float) -> None:
    """A search turn (M6, docs/m6/README.md, "Search turns"): once the brain has asked for no
    turn for `search` decisions in a row, the body turns as far as one decision allows. A
    bout of them keeps one way, drawn from the episode seed, until the brain turns. The brain
    is not asked: it sees the new heading in its next obs."""
    if not self.search:
      return
    if dtheta != 0.0:
      self._still, self._way = 0, 0.0
      return
    self._still += 1
    if self._still < self.search:
      return
    if not self._way:
      self._way = float(self._ways.choice((-1.0, 1.0)))
    self.body.turn(self._way * self.body.params.max_turn_deg)
    self._still = 0

  def _steer(self, dtheta: float, pitch: float) -> None:
    """The compass fly's heading (M6, docs/m6/README.md, "The compass fly"): dtheta sets
    east/west and pitch sets north/south, both at once, scaled by the same max_turn_deg that
    dtheta is clipped to everywhere else. Below COMPASS_HOLD on both axes together, it keeps
    its last heading rather than spin on noise near zero."""
    v_e = float(np.clip(dtheta / self.body.params.max_turn_deg, -1.0, 1.0))
    v_n = pitch
    if math.hypot(v_e, v_n) >= COMPASS_HOLD:
      self.body.heading = math.degrees(math.atan2(-v_n, v_e))

  def drain(self) -> str | None:
    """Collect the act for the obs in flight, if any; return "runaway" if the brain ended the
    episode with it."""
    if not self._waiting:
      return None
    try:
      self._collect()
    except BrainAbort as e:
      if e.reason == "runaway":
        return "runaway"
      raise
    return None

  def finish(self, aborted: str | None) -> None:
    """Close the episode at the brain, unless the brain closed it (runaway)."""
    if not self._open or self.drain() == "runaway":
      return
    self._open = False
    self.link.send("episode_end", episode=self.episode, score=self.score, steps=self.steps,
                   aborted=aborted)

  def _collect(self) -> dict:
    self._waiting = False
    t0 = time.monotonic() if self._since is None else self._since
    self._since = None
    try:
      act = self.link.recv("act")
    except BrainAbort as e:
      if e.reason == "runaway":
        self._open = False  # the brain ended the episode itself
      raise
    self.max_wait_ms = max(self.max_wait_ms, 1000 * (time.monotonic() - t0))
    if act["step"] != self.steps - 1:
      raise LinkError(f"act for step {act['step']}, expected {self.steps - 1}")
    return act


@dataclass
class Result:
  score: float | None = None
  steps: int = 0  # decisions made, so obs sent to a brain
  frames: int = 0  # game loops played
  wall_s: float = 0.0
  late_frames: int = 0
  max_wait_ms: float = 0.0
  aborted: str | None = None
  outcome: int | None = None  # a match's: 1 the fly won, -1 it lost, 0 a tie


def play_episode(game: Game, pilot, episode: int, seed: int, phase: str, decision_frames: int,
                 pacer: Pacer, realtime: bool = False) -> Result:
  """Play one episode to its last frame. A runaway or a stall ends it early, with the score of
  the last obs the brain answered; GameError, LinkError and any other BrainAbort propagate.

  The pilot decides every decision_frames by the game's clock. In real time a slow decision
  lets the game run on, and the pilot decides again as soon as that many frames have passed.
  With `realtime` the client also never waits for the pilot: a decision that is due waits, on
  no_ops, until the pilot is ready, so it is made on the newest frame. The last frame is
  always decided."""
  step_mul = 1 if phase == "watch" else decision_frames
  pilot.start(episode, seed, phase)
  frame = game.reset()
  first_loop = frame.loop
  decided = None  # the game loop of the last decision
  t0 = time.monotonic()
  pacer.start()
  reward = 0.0
  aborted = None
  try:
    while True:
      reward += frame.reward
      due = decided is None or frame.loop - decided >= decision_frames
      if frame.last or (due and (not realtime or pilot.ready())):
        action = pilot.decide(frame, reward)
        reward = 0.0
        decided = frame.loop
      else:
        action = NOOP
      if frame.last:
        break
      frame = game.step(action, step_mul)
      pacer.tick()
  except BrainAbort as e:
    if e.reason != "runaway":
      raise
    aborted = "runaway"
  except Stalled:
    aborted = "stalled"
  except GameError:
    if pilot.drain() != "runaway":
      raise
    aborted = "runaway"  # the brain ended the episode before SC2 failed, so it counts
  score = pilot.score if aborted else frame.score
  return Result(score, pilot.steps, frame.loop - first_loop, time.monotonic() - t0,
                pacer.late, round(pilot.max_wait_ms, 1), aborted,
                frame.outcome if frame.last else None)


def _emit(out, base: dict, result: Result) -> None:
  rec = base | asdict(result)
  rec["wall_s"] = round(result.wall_s, 3)
  rec["fps"] = round(result.frames / result.wall_s, 2) if result.wall_s > 0 else None
  rec["counts"] = result.aborted in COUNTED
  print(json.dumps(rec), file=out, flush=True)


def play(make_game: Callable[[], Game], pilot, episodes: int, run_seed: int, mode: str,
         decision_frames: int, pacer: Pacer, out=None, log=_log,
         max_failures: int = MAX_FAILURES, realtime: bool = False) -> int:
  """Play episodes 0..episodes-1, one JSON line each; return the exit code. SC2 failing
  max_failures times in a row stops the run (a match is never replayed: 1). `realtime`: SC2
  keeps time (play_episode)."""
  out = out or sys.stdout
  game: Game | None = None
  failures = episode = 0
  base: dict | None = None  # the episode under way that has no record yet
  try:
    while episode < episodes:
      seed = episode_seed(run_seed, episode)
      base = {"episode": episode, "seed": seed, "phase": mode, "pilot": pilot.name,
              "body": pilot.body_name}
      try:
        if game is None:
          game = make_game()
        result = play_episode(game, pilot, episode, seed, mode, decision_frames, pacer, realtime)
      except GameError as e:
        pilot.finish("sc2")
        if game is not None:
          game.close()
          game = None
        failures += 1
        _emit(out, base, Result(aborted="sc2"))
        log(f"episode {episode}: SC2 failed ({e}), {failures} in a row")
        if failures >= max_failures:
          log(f"SC2 failed {failures} times in a row; stopping" if failures > 1
              else "SC2 failed; stopping")
          return 1
        continue
      failures = 0
      _emit(out, base, result)  # the game is over: record it before telling the brain
      base = None
      pilot.finish(result.aborted)
      episode += 1
  except BrainAbort as e:
    if base is not None:
      _emit(out, base, Result(aborted=e.reason))
    log(f"the brain stopped the run: {e.reason}")
    return 1
  except LinkError as e:
    if base is not None:
      _emit(out, base, Result(aborted="brain_lost"))
    log(str(e))
    return 1
  finally:
    if game is not None:
      game.close()
  return 0


def window_arg(text: str) -> tuple[int, int]:
  w, _, h = text.lower().partition("x")
  if not (w.isdigit() and h.isdigit() and int(w) > 0 and int(h) > 0):
    raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, like 1280x960, not {text!r}")
  return int(w), int(h)


def _parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
    prog="flycraft-client",
    description="Play a minigame (MoveToBeacon, DefeatRoaches or DefeatMarines) in SC2 for a "
                "flycraft brain, or for pysc2's scripted agent, or a Skirmish match against a "
                "person (--join). DefeatMarines and Skirmish are built first, with "
                "flycraft-maps.")
  who = p.add_mutually_exclusive_group()
  who.add_argument("--brain", default=BRAIN_URL, help=f"brain websocket (default {BRAIN_URL})")
  who.add_argument("--scripted", action="store_true",
                   help="play pysc2's scripted agent, the baseline; no brain needed")
  p.add_argument("--mode", choices=protocol.MODES, default="watch",
                 help="watch: 22.4 fps with a visible window; train/eval: as fast as possible, "
                      "minimized (default watch)")
  p.add_argument("--episodes", type=int, default=1)
  p.add_argument("--seed", type=int, default=0,
                 help="run seed; episode e uses seed * 1000000 + e (default 0)")
  p.add_argument("--map", choices=sorted(TASKS), default="MoveToBeacon")
  p.add_argument("--screen", type=int, default=84)
  p.add_argument("--step-px", type=float, default=BodyParams.step_px,
                 help=f"move order length at full speed, screen px (default {BodyParams.step_px})")
  p.add_argument("--realtime", action="store_true",
                 help="watch mode only: let SC2 pace the game instead of the client")
  p.add_argument("--timeout", type=float, default=30.0,
                 help="seconds to wait for each brain message (default 30)")
  p.add_argument("--face", type=float, metavar="DEG",
                 help="demo setting: start each episode with the fly facing the target nearest "
                      "it, within DEG degrees (default: a random heading)")
  p.add_argument("--stall", type=int, default=STALL_DECISIONS, metavar="N",
                 help="end an episode once the fly has stalled for N decisions: no turn, no "
                      f"score, the squad within {STALL_PX:g} px (default {STALL_DECISIONS}; 0 "
                      "never; a brain's, and never in a match)")
  p.add_argument("--search", type=int, default=0, metavar="N",
                 help="after N decisions in a row with no turn asked, the body turns as far as "
                      "one decision allows, looking for a target (default 0, never; a brain's)")
  p.add_argument("--one-dot", action="store_true",
                 help="show the eye only the nearest target, one dot as in the grid sweep "
                      "(M6, docs/m6/README.md, \"The compass fly\"; a brain's)")
  p.add_argument("--body", choices=("walk", "fly", "compass"), default="walk",
                 help="walk: the fly walks with a heading and sees around it; fly: it flies "
                      "over the map and looks down on it; compass: it also flies over the map, "
                      "north up, and steers by both outputs at once instead of turning "
                      "(default walk; a brain's)")
  p.add_argument("--window", type=window_arg, default=WINDOW,
                 help=f"SC2 window size (default {WINDOW[0]}x{WINDOW[1]})")
  p.add_argument("--join", type=int, metavar="PORT",
                 help="play one Skirmish match against a person: join the game flycraft-match "
                      "hosts on PORT, in real time, with SC2 minimized")
  return p


def main(argv: list[str] | None = None, make_game: Callable[[], Game] | None = None) -> int:
  p = _parser()
  args = p.parse_args(argv)
  if args.episodes < 1:
    p.error("--episodes must be at least 1")
  if args.realtime and args.mode != "watch":
    p.error("--realtime is for watch mode")
  if not 0 <= args.seed <= MAX_SEED:
    p.error(f"--seed must be from 0 to {MAX_SEED}")
  if args.face is not None and (args.scripted or not 0 <= args.face <= 180):
    p.error("--face must be from 0 to 180 degrees, and is for a brain")
  task = TASKS[args.map]
  match = args.join is not None
  if task.versus and not match:
    p.error(f"{args.map} is played against a person: start flycraft-match, then --join its port")
  if match:
    if not task.versus:
      p.error(f"--join plays a match, and {args.map} is not one: use --map Skirmish")
    if args.episodes != 1 or args.mode != "watch":
      p.error("--join plays one match in real time: no --episodes, and --mode watch")
    if not 1024 <= args.join <= 65535 - 4:
      p.error("--join takes the port flycraft-match prints, from 1024 to 65531")
  if args.stall < 0:
    p.error("--stall must be 0 (off) or more decisions")
  if args.search < 0 or (args.search and args.scripted):
    p.error("--search must be 0 (off) or more decisions, and is for a brain")
  fly = args.body == "fly"
  if fly and args.scripted:
    p.error("--body fly is for a brain")
  compass = args.body == "compass"
  if compass and args.scripted:
    p.error("--body compass is for a brain")
  if compass and args.search:
    p.error("--body compass does not search")
  if compass and args.face is not None:
    p.error("--body compass does not face")
  if args.one_dot and args.scripted:
    p.error("--one-dot is for a brain")
  for flag, value in (("--screen", args.screen), ("--step-px", args.step_px),
                      ("--timeout", args.timeout)):
    if not value > 0:
      p.error(f"{flag} must be positive")
  realtime = args.realtime or match
  if make_game is None:
    from flycraft.game import sc2_compat  # pysc2 is only on the Windows side

    def launch():
      if match:
        return sc2_compat.SC2LanGame(args.join, args.screen, window=args.window, race=task.race)
      return sc2_compat.SC2Game(args.map, args.screen, args.mode, args.seed, args.realtime,
                                window=args.window, race=task.race)

    make_game = launch

  pacer = Pacer(GAME_FPS if args.mode == "watch" and not realtime else None)
  tries = 1 if match else MAX_FAILURES
  try:
    if args.scripted:
      return play(make_game, ScriptedPilot(task), args.episodes, args.seed, args.mode,
                  BodyParams.decision_frames, pacer, max_failures=tries, realtime=realtime)
    try:
      link = Link(args.brain, args.mode, args.map, args.screen, args.timeout)
    except (LinkError, BrainAbort) as e:
      _log(str(e))
      return 1
    frames = link.ready["decision_frames"]
    body = Body(BodyParams(decision_frames=frames, step_px=args.step_px, screen=args.screen,
                           order=task.order))
    try:
      pilot = BrainPilot(link, body, task, args.face, stall=0 if match else args.stall,
                         search=args.search, body_kind=args.body, one_dot=args.one_dot)
      return play(make_game, pilot, args.episodes, args.seed, args.mode, frames, pacer,
                  max_failures=tries, realtime=realtime)
    finally:
      link.close()
  except SetupError as e:
    _log(str(e))
    return 1


if __name__ == "__main__":
  sys.exit(main())
