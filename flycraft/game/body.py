"""The fly's body in the game (spec section 7.1): a virtual heading and the marine's orders.

The body keeps a heading in the screen frame (x right, y down, so a positive turn is
clockwise on screen). Each decision it turns by the brain's dtheta, clipped to the turn-rate
limit, and orders the marine one step ahead along the new heading. It also keeps where the
marine is: SC2 draws the beacon over the marine and scores only well inside it, so near the
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
  kind: str  # "select", "move", "stop" or "noop"
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

  def motor(self, marine_xy, can_move: bool, speed: float) -> Action:
    """The order for this decision: select, stop, or a move one step ahead."""
    if marine_xy is None:
      return NOOP
    if not can_move:
      return SELECT
    if speed < self.params.stop_below:
      return STOP
    step = self.params.step_px * speed
    th = np.radians(self.heading)
    hi = self.params.screen - 1
    x = float(np.clip(marine_xy[0] + step * np.cos(th), 0, hi))
    y = float(np.clip(marine_xy[1] + step * np.sin(th), 0, hi))
    self._sent = (x, y)
    return Action("move", (x, y))
