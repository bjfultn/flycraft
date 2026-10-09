"""Where the target was when the episode began, against what the fly did: from the brain's
traces, the target's azimuth at the first decision that saw both the squad and a target, how
many decisions asked for a turn, and the score.

  python docs/m6/start_bearing.py docs/m6/trace-mc*-fly.jsonl [--run-seed 1] [--episodes]
      [--search N]

The body's heading is not in a trace, so it is replayed: Body.reset with the episode's seed
gives the starting heading, and each decision's dtheta turns it, as the client does. With
--search N it also makes the client's search turns (flycraft-client --search N). Azimuth is the
eye's: 0 dead ahead, + to the fly's right, +-180 behind.
"""
import argparse
import math
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np

from flycraft import eye
from flycraft.game.body import Body
from flycraft.game.client import episode_seed

sys.path.insert(0, str(Path(__file__).parent))
from stall_check import episodes  # noqa: E402

BINS = (0, 30, 60, 90, 115, 125, 150, 180)


def start(rows, seed, search=0):
  """(starting target azimuth or None, decisions that asked for a turn, final score, search
  turns made)."""
  body = Body()
  body.reset(seed)
  ways = np.random.default_rng((seed, 2))
  first, still, way, searched = None, 0, 0.0, 0
  for row in rows:
    m, t = row["marine_xy"], row["beacon_xy"]
    if first is None and m and t:
      first = float(eye.azimuth(math.degrees(math.atan2(t[1] - m[1], t[0] - m[0])), body.heading))
    body.turn(row["dtheta"])
    if not search:
      continue
    if row["dtheta"] != 0:  # as BrainPilot._search
      still, way = 0, 0.0
      continue
    still += 1
    if still >= search:
      way = way or float(ways.choice((-1.0, 1.0)))
      body.turn(way * body.params.max_turn_deg)
      still, searched = 0, searched + 1
  return first, sum(row["dtheta"] != 0 for row in rows), rows[-1]["score"], searched


def main():
  p = argparse.ArgumentParser()
  p.add_argument("traces", nargs="+")
  p.add_argument("--run-seed", type=int, default=1)
  p.add_argument("--episodes", action="store_true", help="also print one line per episode")
  p.add_argument("--search", type=int, default=0, help="the client's --search setting")
  args = p.parse_args()
  out = []
  for path in args.traces:
    for e, rows in enumerate(episodes(path)):
      az, turned, score, searched = start(rows, episode_seed(args.run_seed, e), args.search)
      out.append((Path(path).stem, e, az, turned, len(rows), score))
      if args.episodes:
        shown = "none" if az is None else f"{az:+.0f}"
        extra = f", {searched} search turns" if args.search else ""
        print(f"{Path(path).stem} ep {e:2d}: start azimuth {shown}, turned on {turned} of "
              f"{len(rows)} decisions, score {score:.0f}{extra}")
  print("| starting target azimuth, either side | episodes | never turned | scored 0 "
        "| mean score |")
  print("|---|---|---|---|---|")
  for lo, hi in pairwise(BINS):
    group = [o for o in out if o[2] is not None and lo <= abs(o[2]) < hi + (hi == 180)]
    if group:
      mean = sum(o[5] for o in group) / len(group)
      print(f"| {lo} to {hi} | {len(group)} | {sum(o[3] == 0 for o in group)} "
            f"| {sum(o[5] == 0 for o in group)} | {mean:.1f} |")
  unseen = [o for o in out if o[2] is None]
  if unseen:
    print(f"| never seen | {len(unseen)} | {sum(o[3] == 0 for o in unseen)} "
          f"| {sum(o[5] == 0 for o in unseen)} | {sum(o[5] for o in unseen) / len(unseen):.1f} |")


if __name__ == "__main__":
  main()
