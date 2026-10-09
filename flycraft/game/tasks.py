"""The minigames the client can play: what the fly looks at and what the body orders.

- MoveToBeacon: the beacon (one NEUTRAL patch) is the target, and the body orders a move.
- DefeatRoaches (M3b, a demo): each roach (each ENEMY patch) is a target. When a roach is
  within 30 degrees of the fly's heading, the squad attacks the one nearest the heading;
  otherwise it attack-moves a step along the heading, onto clear ground past the squad's own
  marines (SC2 takes an attack aimed at one as an order to shoot it), or moves when the screen
  ends first. So the fly picks its target by turning toward it. (Attack-moving a step at a time
  fires less than attacking a roach: docs/m3b/attack_probe.md.) The squad is every marine,
  so new marines are selected as they arrive (game/sc2_compat.py).
- DefeatMarines (M3c, a demo): DefeatRoaches with zerglings against marines, a map flycraft
  builds from DefeatRoaches (game/mapbuild.py). The same task, played as Zerg.
- Skirmish (M6, a demo): the DefeatMarines task, played against a person's marines in a match
  that flycraft-match hosts (game/match.py), so the client joins it (--join).
Each task also names pysc2's scripted agent for it, the baseline, and the race it plays.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from flycraft.game import render
from flycraft.game.render import Blob


def _centroid(layer: np.ndarray, value: int) -> tuple[float, float] | None:
  """pysc2's MoveToBeacon agent: the middle of all the target's pixels."""
  b = render.find(layer, value)
  return None if b is None else b.xy


def _lowest(layer: np.ndarray, value: int) -> tuple[float, float] | None:
  """pysc2's DefeatRoaches agent: the target pixel lowest on the screen (the first in scan
  order on that row), so the squad focuses fire on one roach."""
  ys, xs = np.nonzero(np.asarray(layer) == value)
  if xs.size == 0:
    return None
  i = int(np.argmax(ys))
  return float(xs[i]), float(ys[i])


@dataclass(frozen=True)
class Task:
  map: str
  target: int  # the player_relative value of what the fly looks at
  order: str  # the body's order a step ahead: "move" or "attack"
  split: bool  # each patch of target is its own object; else all of it is one
  scripted: Callable[[np.ndarray, int], tuple[float, float] | None]  # the baseline's aim
  cone_deg: float = 0.0  # attack a target this near the heading; 0: never, only step
  race: str = "terran"  # the player's race, as pysc2 names it
  versus: bool = False  # a match against a person, joined over LAN, not a minigame

  def targets(self, layer: np.ndarray) -> list[Blob]:
    if self.split:
      return render.blobs(layer, self.target)
    b = render.find(layer, self.target)
    return [] if b is None else [b]

  def aim(self, layer: np.ndarray) -> tuple[float, float] | None:
    """Where pysc2's scripted agent for this map sends the squad, or None."""
    return self.scripted(layer, self.target)

  def strike(self, layer: np.ndarray, targets: list[Blob], xy,
             heading_deg: float) -> tuple[float, float] | None:
    """Where the squad attacks: the target nearest the heading, if it is within cone_deg of
    it, at that target's pixel nearest its middle; None to step along the heading instead."""
    if self.cone_deg <= 0.0 or xy is None or not targets:
      return None
    off = [abs(render.egocentric(xy, t.xy, heading_deg)[0]) for t in targets]
    i = min(range(len(targets)), key=off.__getitem__)
    if off[i] > self.cone_deg:
      return None
    return render.nearest_px(layer, self.target, targets[i].xy)


TASKS = {t.map: t for t in (
  Task("MoveToBeacon", render.NEUTRAL, "move", split=False, scripted=_centroid),
  Task("DefeatRoaches", render.ENEMY, "attack", split=True, scripted=_lowest, cone_deg=30.0),
  # Zerglings only attack-move: they surround and fight what they meet, and doubled the
  # oracle's score over targeting one marine (docs/m6/README.md, Zergling orders).
  Task("DefeatMarines", render.ENEMY, "attack", split=True, scripted=_lowest, race="zerg"),
  Task("Skirmish", render.ENEMY, "attack", split=True, scripted=_lowest, race="zerg",
       versus=True))}


def nearest(xy, targets: list[Blob]) -> Blob | None:
  """The target nearest xy; the first when xy is not known; None when there are none."""
  if not targets:
    return None
  if xy is None:
    return targets[0]
  return min(targets, key=lambda b: math.dist(xy, b.xy))
