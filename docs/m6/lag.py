"""Does the target steer the compass fly's brain, or only the squad? (docs/m6/README.md, "One
dot"). In the game the outputs move the squad, so the target's angle depends on where the
outputs have been pointing. This regresses each output's next value on its current value and
the target's angle: the target's coefficient is the brain's own response, with the squad's
motion taken out through the current output.

  python docs/m6/lag.py docs/m6/trace-cmp1-fly.jsonl.gz docs/m6/trace-cmp3-fly.jsonl.gz:-1

A ":-1" after a trace negates its pitch_raw, for a run with pitch_pos and pitch_neg swapped, so
every run reports DNbe007 minus DNge043.
"""

from __future__ import annotations

import gzip
import json
import sys

import numpy as np

K = 60 / 84  # compass eye degrees per screen pixel (render.COMPASS_DEG_PER_PX)
MAX_JUMP_PX = 4  # a bigger jump means the nearest target changed between decisions


def _angles(r: dict) -> tuple[float, float]:
  dx = r["beacon_xy"][0] - r["marine_xy"][0]
  dy = r["beacon_xy"][1] - r["marine_xy"][1]
  return dx * K, -dy * K


def pairs(path: str, sign: float) -> np.ndarray:
  """Rows of (az, el, pitch, next pitch, turn, next turn) for consecutive decisions on one
  target, skipping each episode's first two."""
  opener = gzip.open if path.endswith(".gz") else open
  with opener(path, "rt") as f:
    rows = [json.loads(line) for line in f]
  out = []
  for a, b in zip(rows, rows[1:], strict=False):
    if a["episode"] != b["episode"] or b["step"] != a["step"] + 1 or a["step"] < 2:
      continue
    if not all(r.get(k) for r in (a, b) for k in ("marine_xy", "beacon_xy")):
      continue
    if np.hypot(*np.subtract(b["beacon_xy"], a["beacon_xy"])) > MAX_JUMP_PX:
      continue
    az, el = _angles(a)
    out.append((az, el, sign * a["pitch_raw"], sign * b["pitch_raw"], a["turn_raw"],
                b["turn_raw"]))
  return np.array(out)


def response(now: np.ndarray, nxt: np.ndarray, angle: np.ndarray) -> tuple[float, float, float]:
  """(carry, Hz per degree, z) from next = carry * now + slope * angle + c."""
  x = np.column_stack([now, angle, np.ones_like(angle)])
  coef, *_ = np.linalg.lstsq(x, nxt, rcond=None)
  res = nxt - x @ coef
  se = np.sqrt(res.var() * np.linalg.inv(x.T @ x)[1, 1])
  return coef[0], coef[1], coef[1] / se


def main(args: list[str]) -> None:
  for arg in args:
    path, _, sign = arg.partition(":")
    p = pairs(path, float(sign or 1))
    az, el, s, s1, t, t1 = p.T
    print(f"{path}: {len(p)} pairs")
    for name, now, nxt, angle in (("pitch", s, s1, el), ("turn", t, t1, az)):
      carry, slope, z = response(now, nxt, angle)
      print(f"  {name}: next = {carry:.2f} now {slope:+.3f} Hz/deg (z {z:+.1f}); "
            f"30 deg moves it {30 * slope:+.1f} Hz against an sd of {now.std():.1f}")


if __name__ == "__main__":
  main(sys.argv[1:])
