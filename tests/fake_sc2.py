"""A stand-in for SC2's MoveToBeacon, so client tests run without pysc2 or Windows.

The screen is a player_relative layer indexed [y, x]: the marine is a SELF disk, the beacon a
NEUTRAL disk drawn over it, as SC2 draws them. The marine walks toward its move target at
`speed_px` per frame. Coming within `score_r` of the beacon (default `beacon_r`) scores 1 and
respawns it elsewhere; SC2's beacon is drawn wider than it scores (M3: radius 6.6 px drawn,
about 4.5 scored), so there the marine can vanish under it before it scores. The marine
starts unselected, so Move_screen is unavailable until a select.

With target=ENEMY and several targets it is a crude DefeatRoaches: roaches that fall, and are
replaced, when the marine reaches them. It takes attack orders as it takes moves, except one
aimed at the marine itself: SC2 would have the squad shoot it, so that fails the test.
"""

from __future__ import annotations

import math

import numpy as np

from flycraft.game import render
from flycraft.game.body import Action
from flycraft.game.client import Frame, GameError


def _disk(layer, xy, r, value):
  ys, xs = np.ogrid[:layer.shape[0], :layer.shape[1]]
  layer[(xs - xy[0]) ** 2 + (ys - xy[1]) ** 2 <= r * r] = value


class FakeGame:
  def __init__(self, seed=0, frames=240, screen=84, speed_px=0.6, marine_r=1.5, beacon_r=3.0,
               score_r=None, fail_at=None, fail_after=None, show_beacon=True, show_marine=True,
               target=render.NEUTRAL, targets=1):
    self.rng = np.random.default_rng(seed)
    self.frames, self.screen, self.speed_px = frames, screen, speed_px
    self.marine_r, self.beacon_r = marine_r, beacon_r
    self.score_r = beacon_r if score_r is None else score_r
    self.fail_at = fail_at  # raise GameError when an episode reaches this loop
    self.fail_after = fail_after  # raise GameError resetting for episode fail_after + 1
    self.show_beacon, self.show_marine = show_beacon, show_marine
    self.target_value, self.n_targets = target, targets
    self.received: list[tuple[Action, int]] = []
    self.resets = 0
    self.closed = False

  def _spot(self):
    m = 6.0
    return tuple(float(v) for v in self.rng.uniform(m, self.screen - 1 - m, 2))

  def reset(self) -> Frame:
    self.resets += 1
    if self.fail_after is not None and self.resets > self.fail_after:
      raise GameError("fake SC2 lost its game")
    self.loop = 0
    self.score = 0.0
    self.selected = False
    self.target = None
    self.marine = self._spot()
    self.beacons = [self._spot() for _ in range(self.n_targets)]
    return self._frame(0.0)

  def step(self, action: Action, step_mul: int) -> Frame:
    self.received.append((action, step_mul))
    if action.kind == "select":
      self.selected = True
    elif action.kind in ("move", "attack"):
      if not self.selected:
        raise AssertionError(f"{action.kind} ordered without a selected marine")
      x, y = (round(v) for v in action.xy)
      if action.kind == "attack" and self.layer[y, x] == render.SELF:
        raise AssertionError(f"attack ordered on the player's own marine at {action.xy}")
      self.target = action.xy
    elif action.kind == "stop":
      self.target = None
    reward = 0.0
    for _ in range(step_mul):
      if self.fail_at is not None and self.loop >= self.fail_at:
        raise GameError("fake SC2 crashed")
      self.loop += 1
      self._walk()
      for i, beacon in enumerate(self.beacons):
        if math.dist(self.marine, beacon) <= self.score_r:
          reward += 1.0
          self.beacons[i] = self._spot()
      if self.loop >= self.frames:
        break
    self.score += reward
    return self._frame(reward)

  def _walk(self):
    if self.target is None:
      return
    dx, dy = self.target[0] - self.marine[0], self.target[1] - self.marine[1]
    d = math.hypot(dx, dy)
    if d <= self.speed_px:
      self.marine, self.target = tuple(self.target), None
    else:
      k = self.speed_px / d
      self.marine = (self.marine[0] + k * dx, self.marine[1] + k * dy)

  def _frame(self, reward) -> Frame:
    layer = np.zeros((self.screen, self.screen), np.uint8)
    if self.show_marine:
      _disk(layer, self.marine, self.marine_r, render.SELF)
    if self.show_beacon:
      for beacon in self.beacons:
        _disk(layer, beacon, self.beacon_r, self.target_value)
    self.layer = layer
    return Frame(layer, reward, self.score, self.loop >= self.frames, self.loop, self.selected)

  def close(self):
    self.closed = True
