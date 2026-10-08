"""Measure how far the marine moves in one decision (8 game loops) on real SC2.

Orders the marine to the far side of the screen and records its centroid each decision.
Prints per-decision displacement while it is in full stride."""
import statistics

from flycraft.game import render
from flycraft.game.body import SELECT, Action
from flycraft.game.sc2_compat import SC2Game

g = SC2Game("MoveToBeacon", 84, "train", 7)
d = []
try:
  for _ in range(3):
    f = g.reset()
    f = g.step(SELECT, 8)
    for _ in range(6):
      b = render.find(f.player_relative, render.SELF)
      if b is None:
        break
      m = b.xy
      tx = 5.0 if m[0] > 42 else 78.0
      ty = 5.0 if m[1] > 42 else 78.0
      f = g.step(Action("move", (tx, ty)), 8)
      prev = getattr(render.find(f.player_relative, render.SELF), 'xy', None)
      for _ in range(30):
        f = g.step(Action("noop"), 8)
        cur = getattr(render.find(f.player_relative, render.SELF), 'xy', None)
        if cur is None or prev is None or f.last:
          break
        step = ((cur[0] - prev[0]) ** 2 + (cur[1] - prev[1]) ** 2) ** 0.5
        if step < 0.5:
          break
        d.append(step)
        prev = cur
      if f.last:
        break
finally:
  g.close()
d.sort()
print(f"n={len(d)} median={statistics.median(d):.2f} mean={statistics.fmean(d):.2f} "
      f"p10={d[len(d)//10]:.2f} p90={d[9*len(d)//10]:.2f} px/decision")
print(f"1.5 decisions = {1.5*statistics.median(d):.2f} px")
