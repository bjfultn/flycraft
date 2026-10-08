"""The game client (spec sections 3, 7 and 10): SC2 on one side, the brain's websocket on the other.

The client owns the marine's body and renders the eye image; it knows the game and nothing
about neurons. It runs on Windows next to SC2, logs to stderr and prints one JSON line per
episode to stdout.

Cadence (spec 7.5): one decision every decision_frames game frames (8, from the brain's ready).
- train and eval: step_mul 8, so every env step is a decision, as fast as the brain answers.
- watch: step_mul 1, paced at 22.4 frames per second. The frames between decisions are
  no_ops, which leave the current move order running.

Pipelining with one decision of latency (spec 7.5; amendment 5, turn before render). At
decision k the pilot collects act k-1, turns the body by its dtheta, renders obs k with the
new heading and sends it, then orders the move for act k-1's speed. The brain works on obs k
while the game plays the next 8 frames. The brain answers the episode's last obs too, and
the pilot drains that act.

Failures (spec sections 9 and 14):
- a runaway brain (abort "runaway") ends the episode with the score of the obs it answered,
  and it counts, even if SC2 fails before the client hears of it;
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
import sys
import time
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
COUNTED = (None, "runaway")  # aborted values whose episodes count (spec section 9)


def _log(text: str) -> None:
  print(f"flycraft-client: {text}", file=sys.stderr, flush=True)


class GameError(Exception):
  """SC2 failed. The episode is discarded and SC2 relaunched."""


class SetupError(Exception):
  """The game cannot start as configured. Relaunching will not help, so the run stops."""


class LinkError(Exception):
  """The brain went away, timed out or broke the protocol. The run stops."""


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
      msg = protocol.decode(self.ws.recv(timeout=0))
    except (TimeoutError, WebSocketException, protocol.ProtocolError):
      return
    if msg["type"] == "abort":
      raise BrainAbort(msg["reason"])
    if msg["type"] == "error":
      raise LinkError(f"the brain says: {msg['message']}")

  def recv(self, kind: str) -> dict:
    """The next message, which must be `kind`; an abort raises BrainAbort."""
    try:
      msg = protocol.decode(self.ws.recv(timeout=self.timeout))
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
  max_wait_ms = 0.0
  steps = 0

  def __init__(self, task: Task = TASKS["MoveToBeacon"]):
    self.task = task

  def start(self, episode: int, seed: int, phase: str) -> None:
    self.steps = 0

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


class BrainPilot:
  """Flies the body from the brain's acts, one decision behind. The obs's beacon_xy is the
  target nearest the marine, for the brain's trace. With `face` set (degrees, a demo setting),
  each episode's first sight of the marine and a target turns the fly to within `face` of the
  nearest target, before the brain sees anything."""

  def __init__(self, link: Link, body: Body, task: Task = TASKS["MoveToBeacon"],
               face: float | None = None):
    self.link, self.body, self.task, self.face = link, body, task, face
    self.name = link.ready["brain_id"]
    self.polarity = link.ready["polarity"]
    self.max_wait_ms = 0.0
    self._open = False  # the brain has an episode open
    self._waiting = False  # an obs is waiting for its act

  def start(self, episode: int, seed: int, phase: str) -> None:
    self.body.reset(seed)
    self.link.send("episode_start", episode=episode, seed=seed, phase=phase)
    self.episode, self.seed, self.steps, self.score = episode, seed, 0, 0.0
    self._faced = self.face is None
    self.max_wait_ms = 0.0
    self._open, self._waiting = True, False

  def decide(self, frame: Frame, reward: float) -> Action:
    act = self._collect() if self._waiting else None  # act k-1
    if act is not None:
      self.body.turn(act["dtheta"])
    marine = render.find(frame.player_relative, render.SELF)  # the squad's middle
    targets = self.task.targets(frame.player_relative)
    marine_xy = self.body.locate(marine.xy if marine else None)  # it may be under the beacon
    near = nearest(marine_xy, targets)
    if not self._faced and marine_xy is not None and near is not None:
      self.body.face(marine_xy, near.xy, self.face, self.seed)
      self._faced = True
    img = render.render_eye(marine_xy, targets, self.body.heading, self.polarity)
    self.link.send("obs", step=self.steps, eye=img.tobytes(), reward=reward, score=frame.score,
                   last=frame.last, marine_xy=marine.xy if marine else None,
                   beacon_xy=near.xy if near else None)
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
    t0 = time.monotonic()
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


def play_episode(game: Game, pilot, episode: int, seed: int, phase: str, decision_frames: int,
                 pacer: Pacer) -> Result:
  """Play one episode to its last frame. A runaway ends it early, with the score of the obs
  the brain answered; GameError, LinkError and any other BrainAbort propagate."""
  watch = phase == "watch"
  step_mul = 1 if watch else decision_frames
  every = decision_frames if watch else 1  # env steps per decision
  pilot.start(episode, seed, phase)
  frame = game.reset()
  first_loop = frame.loop
  t0 = time.monotonic()
  pacer.start()
  n = 0
  reward = 0.0
  aborted = None
  try:
    while True:
      reward += frame.reward
      if n % every == 0 or frame.last:
        action = pilot.decide(frame, reward)
        reward = 0.0
      else:
        action = NOOP
      if frame.last:
        break
      frame = game.step(action, step_mul)
      n += 1
      pacer.tick()
  except BrainAbort as e:
    if e.reason != "runaway":
      raise
    aborted = "runaway"
  except GameError:
    if pilot.drain() != "runaway":
      raise
    aborted = "runaway"  # the brain ended the episode before SC2 failed, so it counts
  score = pilot.score if aborted else frame.score
  return Result(score, pilot.steps, frame.loop - first_loop, time.monotonic() - t0,
                pacer.late, round(pilot.max_wait_ms, 1), aborted)


def _emit(out, base: dict, result: Result) -> None:
  rec = base | asdict(result)
  rec["wall_s"] = round(result.wall_s, 3)
  rec["fps"] = round(result.frames / result.wall_s, 2) if result.wall_s > 0 else None
  rec["counts"] = result.aborted in COUNTED
  print(json.dumps(rec), file=out, flush=True)


def play(make_game: Callable[[], Game], pilot, episodes: int, run_seed: int, mode: str,
         decision_frames: int, pacer: Pacer, out=None, log=_log) -> int:
  """Play episodes 0..episodes-1, one JSON line each; return the exit code."""
  out = out or sys.stdout
  game: Game | None = None
  failures = episode = 0
  base: dict | None = None  # the episode under way that has no record yet
  try:
    while episode < episodes:
      seed = episode_seed(run_seed, episode)
      base = {"episode": episode, "seed": seed, "phase": mode, "pilot": pilot.name}
      try:
        if game is None:
          game = make_game()
        result = play_episode(game, pilot, episode, seed, mode, decision_frames, pacer)
      except GameError as e:
        pilot.finish("sc2")
        if game is not None:
          game.close()
          game = None
        failures += 1
        _emit(out, base, Result(aborted="sc2"))
        log(f"episode {episode}: SC2 failed ({e}), {failures} in a row")
        if failures >= MAX_FAILURES:
          log(f"SC2 failed {MAX_FAILURES} times in a row; stopping")
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


def _window(text: str) -> tuple[int, int]:
  w, _, h = text.lower().partition("x")
  if not (w.isdigit() and h.isdigit() and int(w) > 0 and int(h) > 0):
    raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, like 1280x960, not {text!r}")
  return int(w), int(h)


def _parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
    prog="flycraft-client",
    description="Play a minigame (MoveToBeacon, DefeatRoaches or DefeatMarines) in SC2 for a "
                "flycraft brain, or for pysc2's scripted agent. DefeatMarines is built first, "
                "with flycraft-maps.")
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
  p.add_argument("--window", type=_window, default=WINDOW,
                 help=f"SC2 window size (default {WINDOW[0]}x{WINDOW[1]})")
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
  for flag, value in (("--screen", args.screen), ("--step-px", args.step_px),
                      ("--timeout", args.timeout)):
    if not value > 0:
      p.error(f"{flag} must be positive")
  task = TASKS[args.map]
  if make_game is None:
    from flycraft.game import sc2_compat  # pysc2 is only on the Windows side

    def launch():
      return sc2_compat.SC2Game(args.map, args.screen, args.mode, args.seed, args.realtime,
                                window=args.window, race=task.race)

    make_game = launch

  pacer = Pacer(GAME_FPS if args.mode == "watch" and not args.realtime else None)
  try:
    if args.scripted:
      return play(make_game, ScriptedPilot(task), args.episodes, args.seed, args.mode,
                  BodyParams.decision_frames, pacer)
    try:
      link = Link(args.brain, args.mode, args.map, args.screen, args.timeout)
    except (LinkError, BrainAbort) as e:
      _log(str(e))
      return 1
    frames = link.ready["decision_frames"]
    body = Body(BodyParams(decision_frames=frames, step_px=args.step_px, screen=args.screen,
                           order=task.order))
    try:
      return play(make_game, BrainPilot(link, body, task, args.face), args.episodes, args.seed,
                  args.mode, frames, pacer)
    finally:
      link.close()
  except SetupError as e:
    _log(str(e))
    return 1


if __name__ == "__main__":
  sys.exit(main())
