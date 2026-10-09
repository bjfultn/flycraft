"""Probe (M6 friendly fire, solo tasks): flycraft-client on DefeatMarines, counting who hits
each of the squad's zerglings. Zerglings deal 5 a hit and marines 6, so a zergling's health
drop says who hit it (ff_host.py's split); a drop with no single reading counts as unclear.
Health comes from the raw protobuf, which keeps the fractions the zerglings regenerate.

Writes one line per hit by a zergling: the decision's order, how near the order's point was to
the squad's own pixels when it was aimed (`d_then`, px), and the hit zergling's nearest marine
(`marine_d`, game units). Then one line per episode with its counts. Any pilot:

  python docs/m6/ff_lings.py runs/ffl.jsonl --mode train --episodes 5 --seed 1 [client args]

Surviving zerglings keep their damage into the next wave (the map's rule, from DefeatRoaches),
so a hurt zergling at the start line is not by itself friendly fire.
"""
import json
import math
import sys
from collections import Counter

import numpy as np
from pysc2.lib import features

from flycraft.game import client, render, sc2_compat

MARINE, ZERGLING, SELF, ENEMY = 48, 105, 1, 4
OUT = open(sys.argv[1], "w")
state = {"episode": -1, "hp": {}, "order": None}
hits = Counter()


def split(d):
  """(ling hits, marine hits) for a health drop d, or None if it has no single reading."""
  n = round(d)
  if abs(d - n) > 0.4:
    return None
  ways = [(a, (n - 5 * a) // 6) for a in range(n // 5 + 1) if (n - 5 * a) % 6 == 0]
  return ways[0] if len(ways) == 1 else None


def own_dist(layer, xy):
  if layer is None or xy is None:
    return None
  ys, xs = np.nonzero(np.asarray(layer) == render.SELF)
  if xs.size == 0:
    return None
  return round(float(np.sqrt(((xs - round(xy[0])) ** 2 + (ys - round(xy[1])) ** 2).min())), 2)


def write(rec):
  OUT.write(json.dumps(rec) + "\n")
  OUT.flush()


def close_episode():
  if state["episode"] >= 0:
    write({"episode": state["episode"], **{k: hits[k] for k in ("ling", "marine", "unclear",
                                                                 "attacks", "steps")}})


def look(obs):
  units = obs.raw_data.units
  lings = [u for u in units if u.alliance == SELF and u.unit_type == ZERGLING]
  marines = [u for u in units if u.alliance == ENEMY and u.unit_type == MARINE]
  hp = state["hp"]
  for u in lings:
    prev = hp.get(u.tag)
    if prev is not None and prev - u.health > 0.5:
      s = split(prev - u.health)
      if s is None:
        hits["unclear"] += 1
      else:
        hits["ling"] += s[0]
        hits["marine"] += s[1]
        if s[0]:
          near = min((math.hypot(m.pos.x - u.pos.x, m.pos.y - u.pos.y) for m in marines),
                     default=None)
          write({"episode": state["episode"], "loop": obs.game_loop, "tag": u.tag,
                 "hp": [round(prev, 2), round(u.health, 2)],
                 "marine_d": None if near is None else round(near, 2), **(state["order"] or {})})
    hp[u.tag] = u.health


def _interface(screen):
  return features.AgentInterfaceFormat(
    feature_dimensions=features.Dimensions(screen=screen, minimap=sc2_compat.MINIMAP),
    use_raw_units=True)


orig_take, orig_reset, orig_step = (sc2_compat.SC2Game._take, sc2_compat.SC2Game.reset,
                                    sc2_compat.SC2Game.step)


def _take(self, call):
  f = orig_take(self, call)
  look(self.env._obs[0].observation)
  self.last_layer = f.player_relative
  return f


def reset(self):
  close_episode()
  state["episode"] += 1
  state["hp"].clear()
  state["order"] = None
  hits.clear()
  return orig_reset(self)


def step(self, action, step_mul):
  hits["steps"] += 1
  if action.kind in ("attack", "move"):
    hits["attacks"] += action.kind == "attack"
    state["order"] = {"kind": action.kind, "xy": action.xy,
                      "d_then": own_dist(getattr(self, "last_layer", None), action.xy)}
  elif action.kind == "stop":
    state["order"] = None
  return orig_step(self, action, step_mul)


sc2_compat._interface = _interface
sc2_compat.SC2Game._take = _take
sc2_compat.SC2Game.reset = reset
sc2_compat.SC2Game.step = step
try:
  code = client.main(["--map", "DefeatMarines"] + sys.argv[2:])
finally:
  close_episode()
sys.exit(code)
