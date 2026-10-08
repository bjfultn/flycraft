"""Does the beacon hide the marine on the screen layer before the marine scores?

The first real-brain watch run lost the marine (no SELF pixels) a few pixels from the beacon
and never scored. This walks the marine at the beacon in small steps, approaching from where
it stands, and prints at each stop: distance to the beacon centroid, marine pixels seen,
beacon pixels seen, and reward. Run on Windows with the client's venv (python hidden_marine.py).
"""
import math

import numpy as np

from flycraft.game import render
from flycraft.game.body import SELECT, Action
from flycraft.game.sc2_compat import SC2Game

STOPS = (12.0, 9.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0)


def settle(g, f, n=12):
  """Let the marine finish its move: noop until it stops or the episode ends."""
  prev = None
  for _ in range(n):
    f = g.step(Action("noop"), 8)
    m = render.find(f.player_relative, render.SELF)
    cur = m.xy if m else None
    if f.last or (cur is not None and cur == prev) or f.reward:
      break
    prev = cur
  return f


def around(layer, xy, r=6):
  x, y = round(xy[0]), round(xy[1])
  win = layer[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
  return dict(zip(*np.unique(win, return_counts=True), strict=True))


g = SC2Game("MoveToBeacon", 84, "train", 11)
try:
  for approach in range(4):
    f = g.reset()
    f = g.step(SELECT, 8)
    b = render.find(f.player_relative, render.NEUTRAL)
    m = render.find(f.player_relative, render.SELF)
    print(f"approach {approach}: beacon {b.xy} {b.n_px}px r={b.radius_px:.1f}, "
          f"marine {m.xy} {m.n_px}px r={m.radius_px:.1f}")
    ux, uy = m.xy[0] - b.xy[0], m.xy[1] - b.xy[1]
    norm = math.hypot(ux, uy)
    ux, uy = ux / norm, uy / norm
    last_seen = m.xy
    for d in STOPS:
      target = (b.xy[0] + d * ux, b.xy[1] + d * uy)
      f = g.step(Action("move", target), 8)
      got = f.reward
      f = settle(g, f)
      got += f.reward
      m2 = render.find(f.player_relative, render.SELF)
      b2 = render.find(f.player_relative, render.NEUTRAL)
      if m2:
        last_seen = m2.xy
      dist = math.hypot(last_seen[0] - b.xy[0], last_seen[1] - b.xy[1])
      seen = f"{m2.n_px:4d}px" if m2 else "HIDDEN"
      near = around(f.player_relative, last_seen)
      print(f"  ordered d={d:4.1f}: marine {seen} at {last_seen} (d={dist:.1f}), "
            f"beacon {b2.n_px if b2 else 0}px at {b2.xy if b2 else None}, reward {got}, "
            f"layer near {near}")
      if got or f.last:
        break
finally:
  g.close()
