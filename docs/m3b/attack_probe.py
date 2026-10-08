"""Experiment: why attack-moving a step at a time kills so few roaches. Client-side pilots,
no brain. Not part of flycraft.

argv[1] is a variant:
  nearest        Attack_screen on the nearest roach's centroid each decision
  nearestpx      Attack_screen on the roach pixel nearest that centroid (always on a unit)
  carrot<N>      attack-move N px from the squad toward the nearest roach each decision
  carrot<N>move  the same with Move_screen (a control: no attacking on the way)
  carrot<N>clear the same, moved past the squad's own pixels as the body does (SC2 takes an
                 attack aimed at a marine as an order to shoot it)
  away<N>        attack-move N px from the squad directly AWAY from the nearest roach
Remaining args go to the client (--episodes, --seed, --mode, --map)."""
import math
import re
import sys

import numpy as np

from flycraft.game import client, render
from flycraft.game.body import NOOP, SELECT, Action, Body, BodyParams
from flycraft.game.tasks import nearest

V = sys.argv.pop(1)
M = re.fullmatch(r"(nearestpx|nearest|carrot|away)(\d*)(move|clear)?", V)
KIND, N, ORDER = M.group(1), float(M.group(2) or 0), "move" if M.group(3) == "move" else "attack"
CLEAR = M.group(3) == "clear"


class Pilot(client.ScriptedPilot):
  name = V

  def decide(self, frame, reward):
    self.steps += 1
    if frame.last:
      return NOOP
    if not frame.can_move:
      return SELECT
    layer = frame.player_relative
    squad = render.find(layer, render.SELF)
    roach = nearest(squad.xy if squad else None, render.blobs(layer, render.ENEMY))
    if squad is None or roach is None:
      return NOOP
    if KIND == "nearest":
      return Action(ORDER, roach.xy)
    if KIND == "nearestpx":
      ys, xs = np.nonzero(np.asarray(layer) == render.ENEMY)
      i = int(np.argmin((xs - roach.xy[0]) ** 2 + (ys - roach.xy[1]) ** 2))
      return Action(ORDER, (float(xs[i]), float(ys[i])))
    (sx, sy), (rx, ry) = squad.xy, roach.xy
    th = math.atan2(ry - sy, rx - sx) + (math.pi if KIND == "away" else 0.0)
    carrot = (min(max(sx + N * math.cos(th), 0), 83), min(max(sy + N * math.sin(th), 0), 83))
    if CLEAR:  # as the body does: no clear ground ahead on screen, so walk
      past = Body(BodyParams(order="attack"))._past(layer == render.SELF, squad.xy, N, th)
      return Action("attack", past) if past else Action("move", carrot)
    return Action(ORDER, carrot)


client.ScriptedPilot = Pilot
sys.exit(client.main())
