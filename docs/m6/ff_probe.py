"""Probe fly (M6 friendly fire): flycraft-client --map Skirmish --join 14380, logging per
decision how old the aiming frame is when the order goes out (`stale`, game frames), and how
near the fly's own units are to the order's point when it was aimed (`d_then`, `px_then`) and
when SC2 gets it (`d_now`, `px_now`). `wait_ms` is the time spent in the pilot's wait for the
act; after the fix that wait no longer happens there, so it reads 0. Run it with the stand-in
(ff_host.py) hosting:

  python docs/m6/ff_probe.py runs/ff.jsonl --brain URL
"""
import json
import sys
import time

import numpy as np

from flycraft.game import client, render, sc2_compat

OUT = open(sys.argv[1], "w")
pending = {}


def own_dist(layer, xy):
  ys, xs = np.nonzero(np.asarray(layer) == render.SELF)
  if xs.size == 0 or xy is None:
    return None
  x, y = round(xy[0]), round(xy[1])
  return round(float(np.sqrt(((xs - x) ** 2 + (ys - y) ** 2).min())), 2)


def at(layer, xy):
  return None if xy is None else int(np.asarray(layer)[round(xy[1]), round(xy[0])])


orig_collect = client.BrainPilot._collect


def _collect(self):
  t0 = time.monotonic()
  try:
    return orig_collect(self)
  finally:
    pending["wait_ms"] = round(1000 * (time.monotonic() - t0), 1)


orig_decide = client.BrainPilot.decide


def decide(self, frame, reward):
  pending.clear()
  a = orig_decide(self, frame, reward)
  pending.update(loop=frame.loop, kind=a.kind, xy=a.xy,
                 d_then=own_dist(frame.player_relative, a.xy),
                 px_then=at(frame.player_relative, a.xy))
  return a


orig_step = sc2_compat.SC2Game.step


def step(self, action, step_mul):
  f = orig_step(self, action, step_mul)
  if "kind" in pending:
    rec = dict(pending, loop_now=f.loop, stale=f.loop - pending["loop"],
               d_now=own_dist(f.player_relative, pending["xy"]),
               px_now=at(f.player_relative, pending["xy"]))
    OUT.write(json.dumps(rec) + "\n")
    OUT.flush()
    pending.clear()
  return f


client.BrainPilot._collect = _collect
client.BrainPilot.decide = decide
sc2_compat.SC2Game.step = step
sys.exit(client.main(["--map", "Skirmish", "--join", "14380"] + sys.argv[2:]))
