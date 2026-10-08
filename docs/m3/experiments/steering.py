"""What did the fly see, and how did it turn? Replays the body from a `flycraft brain --trace`.

The client turns the body by act k-1's dtheta before it renders obs k (spec 7.5), from a
heading fixed by the episode seed. So the trace alone gives the exact heading behind every
eye image, and with it the beacon's azimuth as the fly saw it. Where the beacon hides the
marine, the marine is taken at its last seen position (the body dead-reckons it further).

For each episode: the reset heading, the score, and how the fly answered a beacon ahead
(|az| <= 60), to one side (60 to 120) and behind (> 120): the share of turns toward it and
the mean |dtheta|. Then, over all episodes, the same share split by which side the beacon is
on (|az| < 120 only). With --steps, one line per decision.

usage: python steering.py TRACE.jsonl --run-seed 1 [--steps]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

from flycraft.game.body import Body
from flycraft.game.client import episode_seed
from flycraft.game.render import egocentric

BANDS = (("ahead", 0.0, 60.0), ("side", 60.0, 120.0), ("behind", 120.0, 180.1))


def episodes(path: str) -> dict[int, list[dict]]:
  eps = defaultdict(list)
  with open(path) as f:
    for line in f:
      r = json.loads(line)
      eps[r["episode"]].append(r)
  return {k: sorted(v, key=lambda r: r["step"]) for k, v in sorted(eps.items())}


def replay(rows: list[dict], seed: int):
  """Yield (row, heading, az, dist) per obs; az is None with no beacon or marine yet."""
  body = Body()
  body.reset(seed)
  where = None
  prev = None
  for r in rows:
    if prev is not None and "dtheta" in prev:
      body.turn(prev["dtheta"])
    where = r["marine_xy"] or where
    az = dist = None
    if where is not None and r["beacon_xy"] is not None:
      az, dist = egocentric(where, r["beacon_xy"], body.heading)
    yield r, body.heading, az, dist
    prev = r


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("trace")
  ap.add_argument("--run-seed", type=int, required=True)
  ap.add_argument("--steps", action="store_true")
  a = ap.parse_args()
  total = {b: [0, 0, 0.0] for b, _, _ in BANDS}  # toward, turns, sum |dtheta|
  sides = {"left": [0, 0, 0.0], "right": [0, 0, 0.0]}  # toward, turns, sum dtheta
  for ep, rows in episodes(a.trace).items():
    seed = episode_seed(a.run_seed, ep)
    band = {b: [0, 0, 0.0] for b, _, _ in BANDS}
    first = None
    for r, heading, az, dist in replay(rows, seed):
      if first is None:
        first = heading
      if a.steps:
        hid = "" if r["marine_xy"] else " hidden"
        where = f"az {az:7.1f} d {dist:5.1f}" if az is not None else "no beacon/marine"
        print(f"ep {ep} step {r['step']:3d} heading {heading:7.1f} {where}{hid} "
              f"dtheta {r.get('dtheta', math.nan):6.1f} score {r['score']:.0f}"
              + (" REWARD" if r["reward"] else "") + (f" {r['aborted']}" if "aborted" in r else ""))
      if az is None or "dtheta" not in r or abs(r["dtheta"]) < 1.0:
        continue
      for b, lo, hi in BANDS:
        if lo <= abs(az) < hi:
          band[b][0] += (r["dtheta"] > 0) == (az > 0)
          band[b][1] += 1
          band[b][2] += abs(r["dtheta"])
      if abs(az) < 120.0:
        s = sides["left" if az < 0 else "right"]
        s[0] += (r["dtheta"] > 0) == (az > 0)
        s[1] += 1
        s[2] += r["dtheta"]
    score = rows[-1]["score"] + rows[-1]["reward"]
    parts = []
    for b, _, _ in BANDS:
      toward, n, mag = band[b]
      total[b][0] += toward
      total[b][1] += n
      total[b][2] += mag
      parts.append(f"{b} {toward}/{n} toward, mean |dtheta| {mag / n:.1f}" if n else f"{b} -")
    print(f"episode {ep}: reset heading {first:.1f}, {len(rows)} obs, score {score:.0f}; "
          + "; ".join(parts))
  parts = [f"{b} {t}/{n} toward ({t / n:.0%}), mean |dtheta| {m / n:.1f}" if n else f"{b} -"
           for b, (t, n, m) in total.items()]
  print("all: " + "; ".join(parts))
  parts = [f"beacon {k}: {t}/{n} toward ({t / n:.0%}), mean dtheta {m / n:+.1f}" if n else f"{k} -"
           for k, (t, n, m) in sides.items()]
  print("by side (|az| < 120): " + "; ".join(parts))


if __name__ == "__main__":
  main()
