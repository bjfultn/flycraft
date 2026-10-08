"""Does the fly turn toward the beacon? Reads `flycraft brain --trace` files.

The trace has positions but no heading, so the marine's heading is taken from its last move
(screen coordinates, y down: a growing angle is a clockwise turn on screen, the same sense as
a positive dtheta). For each obs after a move, the beacon's bearing relative to that heading
says which way the fly should turn; the command it answered with says which way it did.

Also checks the body: the heading change over the next move should follow the sign of the
dtheta just sent, or the steering convention is broken somewhere between brain and game.

usage: python chirality.py TRACE.jsonl [...]
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict


def wrap(deg: float) -> float:
  return (deg + 180.0) % 360.0 - 180.0


def angle(dx: float, dy: float) -> float:
  return math.degrees(math.atan2(dy, dx))


def episodes(paths: list[str]) -> dict:
  eps = defaultdict(list)
  for path in paths:
    with open(path) as f:
      for line in f:
        r = json.loads(line)
        if "dtheta" in r:  # abort lines carry no command
          eps[(path, r["episode"])].append(r)
  return {k: sorted(v, key=lambda r: r["step"]) for k, v in eps.items()}


def sign(x: float, dead: float = 0.0) -> int:
  return 0 if abs(x) <= dead else (1 if x > 0 else -1)


def analyse(eps: dict) -> dict:
  bins = {"left": [0, 0, 0], "ahead": [0, 0, 0], "right": [0, 0, 0]}  # turned left, none, right
  body = [0, 0]  # heading change agreed with the last dtheta, disagreed
  for rows in eps.values():
    heading = None
    for prev, cur, nxt in zip(rows, rows[1:], rows[2:], strict=False):
      if None in (prev["marine_xy"], cur["marine_xy"], nxt["marine_xy"], cur["beacon_xy"]):
        heading = None  # off screen for a moment: start over
        continue
      (px, py), (cx, cy), (nx, ny) = prev["marine_xy"], cur["marine_xy"], nxt["marine_xy"]
      if (cx, cy) != (px, py):
        heading = angle(cx - px, cy - py)
      if heading is None:
        continue
      bx, by = cur["beacon_xy"]
      rel = wrap(angle(bx - cx, by - cy) - heading)  # + = beacon clockwise of heading (right)
      where = "ahead" if abs(rel) <= 10.0 else ("right" if rel > 0 else "left")
      bins[where][sign(cur["dtheta"]) + 1] += 1
      if (nx, ny) != (cx, cy) and cur["dtheta"] != 0.0:
        turned = wrap(angle(nx - cx, ny - cy) - heading)
        if abs(turned) > 1.0:
          body[0 if sign(turned) == sign(cur["dtheta"]) else 1] += 1
  toward = bins["left"][0] + bins["right"][2]
  away = bins["left"][2] + bins["right"][0]
  return {"bins": bins, "toward": toward, "away": away, "body": body}


def main(paths: list[str]) -> None:
  eps = episodes(paths)
  a = analyse(eps)
  n = sum(len(v) for v in eps.values())
  gaps = sum(r["marine_xy"] is None or r["beacon_xy"] is None for v in eps.values() for r in v)
  print(f"{len(eps)} episodes, {n} commands, {gaps} without a marine or beacon on screen")
  print(f"{'beacon':<8}{'turned L':>9}{'none':>7}{'turned R':>9}")
  for where, (lft, none, rgt) in a["bins"].items():
    print(f"{where:<8}{lft:>9}{none:>7}{rgt:>9}")
  turns = a["toward"] + a["away"]
  if turns:
    print(f"beacon off-axis and the fly turned: {a['toward']} toward, {a['away']} away "
          f"({a['toward'] / turns:.0%} toward)")
  agree, disagree = a["body"]
  if agree + disagree:
    print(f"body check: heading followed the dtheta sign {agree} of {agree + disagree} times")


if __name__ == "__main__":
  main(sys.argv[1:])
