"""The fly's body in the game (spec section 7.1): a virtual heading and the marine's orders.

The body keeps a heading in the screen frame (x right, y down, so a positive turn is
clockwise on screen). Each decision it turns by the brain's dtheta, clipped to the turn-rate
limit, and orders the marine one step ahead along the new heading, or attacks the target the
task strikes (game/tasks.py). SC2 takes an attack aimed at one of the player's own units as an
order to shoot it, so an attack step goes past the squad, onto clear ground. It also keeps where
the marine is: SC2 draws the beacon over the marine and scores only well inside it, so near the
beacon the marine can vanish from the screen layer though it has not scored. Actions are plain
tuples; game/sc2_compat.py turns them into pysc2 calls, so this module needs no pysc2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from flycraft.eye import wrap_deg


class Action(NamedTuple):
  kind: str  # "select", "move", "attack", "stop" or "noop"
  xy: tuple[float, float] | None = None


SELECT = Action("select")
STOP = Action("stop")
NOOP = Action("noop")


@dataclass(frozen=True)
class BodyParams:
  # The first three must equal DecoderConfig's defaults; tests/test_body.py checks they do.
  max_turn_deg_s: float = 180.0
  decision_frames: int = 8
  game_fps: float = 22.4
  step_px: float = 6.0  # L: marine travel in 1.5 decisions (M2: 3.6 px median, 3.95 mean)
  screen: int = 84
  stop_below: float = 0.05
  order: str = "move"  # the order a step ahead takes: "move", or "attack" to fight on the way
  clear_px: float = 2.0  # an attack step lands at least this far from the squad's own pixels

  @property
  def max_turn_deg(self) -> float:
    """The largest turn in one decision: max_turn_deg_s times a decision's game time."""
    return self.max_turn_deg_s * self.decision_frames / self.game_fps

  @property
  def stride_px(self) -> float:
    """How far the marine walks in one decision: step_px is 1.5 decisions of it."""
    return self.step_px / 1.5


class Body:
  def __init__(self, params: BodyParams | None = None):
    self.params = params or BodyParams()
    self.heading = 0.0
    self.where: tuple[float, float] | None = None  # the marine, as far as the body knows
    self._sent: tuple[float, float] | None = None  # this decision's move target

  def reset(self, seed: int) -> None:
    """A random heading in [-180, 180), fixed by the episode seed; the marine not yet seen."""
    self.heading = float(np.random.default_rng(seed).uniform(-180.0, 180.0))
    self.where = self._sent = None

  def face(self, marine_xy, target_xy, within: float, seed: int) -> None:
    """Head within `within` degrees of target_xy, the offset fixed by the episode seed. A demo
    setting: the brain makes no turn on its own, so a fly started facing away never finds the
    target."""
    bearing = math.degrees(math.atan2(target_xy[1] - marine_xy[1], target_xy[0] - marine_xy[0]))
    self.heading = wrap_deg(bearing + np.random.default_rng((seed, 1)).uniform(-within, within))

  def locate(self, seen) -> tuple[float, float] | None:
    """Where the marine is: where it is seen, else one stride on from where it was toward the
    last move target; None until it has been seen."""
    if seen is not None:
      self.where = (float(seen[0]), float(seen[1]))
    elif self.where is not None and self._sent is not None:
      dx, dy = self._sent[0] - self.where[0], self._sent[1] - self.where[1]
      k = min(1.0, self.params.stride_px / max(math.hypot(dx, dy), 1e-9))
      self.where = (self.where[0] + k * dx, self.where[1] + k * dy)
    self._sent = None
    return self.where

  def turn(self, dtheta: float) -> float:
    """Turn by dtheta degrees (positive = clockwise on screen), clipped; return the turn made."""
    lim = self.params.max_turn_deg
    d = float(np.clip(dtheta, -lim, lim))
    self.heading = wrap_deg(self.heading + d)
    return d

  def motor(self, marine_xy, can_move: bool, speed: float, strike=None, own=None) -> Action:
    """The order for this decision: select, an attack on strike (at any speed), stop, or the
    task's order one step ahead. own (a bool mask of the player's pixels) moves an attack step
    past the squad; with no clear ground ahead on screen, the step is a move."""
    if marine_xy is None:
      return NOOP
    if not can_move:
      return SELECT
    if strike is not None:
      self._sent = (float(strike[0]), float(strike[1]))
      return Action("attack", self._sent)
    if speed < self.params.stop_below:
      return STOP
    step = self.params.step_px * speed
    th = np.radians(self.heading)
    hi = self.params.screen - 1
    x = float(np.clip(marine_xy[0] + step * np.cos(th), 0, hi))
    y = float(np.clip(marine_xy[1] + step * np.sin(th), 0, hi))
    kind = self.params.order
    if kind == "attack" and own is not None and np.any(own):
      past = self._past(own, marine_xy, step, th)
      kind, (x, y) = ("move", (x, y)) if past is None else ("attack", past)
    self._sent = (x, y)
    return Action(kind, (x, y))

  def _past(self, own, xy, step: float, th: float) -> tuple[float, float] | None:
    """The first screen pixel along the heading, from step on, at least clear_px from every
    own pixel; None if the screen ends first. Pixels, since the order is sent rounded."""
    ys, xs = np.nonzero(own)
    hi = self.params.screen - 1
    t = np.arange(step, 2.0 * self.params.screen, 0.5)
    px, py = np.round(xy[0] + t * np.cos(th)), np.round(xy[1] + t * np.sin(th))
    on = (px >= 0) & (px <= hi) & (py >= 0) & (py <= hi)
    d2 = (px[:, None] - xs[None, :]) ** 2 + (py[:, None] - ys[None, :]) ** 2
    ok = on & (d2.min(axis=1) >= self.params.clear_px ** 2)
    i = int(np.argmax(ok))
    return (float(px[i]), float(py[i])) if ok[i] else None
