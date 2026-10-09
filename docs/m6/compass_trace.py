"""Do the compass fly's two outputs point at its targets in the game?

Reads a brain trace (runs/m3/trace-<run>.jsonl, or the .gz kept here) and, per decision, puts
the target (beacon_xy) where compass_eye puts it, east and north of the squad at
COMPASS_DEG_PER_PX, then asks how often pitch has the target's elevation sign and dtheta its
azimuth sign, and how the raw pitch goes with elevation.

  python docs/m6/compass_trace.py docs/m6/trace-cmp1-fly.jsonl.gz
"""
import gzip
import json
import sys

import numpy as np

K = 60.0 / 84  # render.COMPASS_DEG_PER_PX
MAX_TURN = 180.0 * 8 / 22.4  # degrees per decision


def main(path):
  with (gzip.open if path.endswith(".gz") else open)(path, "rt") as f:
    rows = [json.loads(line) for line in f]
  rows = [r for r in rows if r.get("marine_xy") and r.get("beacon_xy")]
  az = np.array([(r["beacon_xy"][0] - r["marine_xy"][0]) * K for r in rows])
  el = np.array([-(r["beacon_xy"][1] - r["marine_xy"][1]) * K for r in rows])
  raw = np.array([r["pitch_raw"] for r in rows])
  pitch = np.array([r["pitch"] for r in rows])
  dt = np.array([r["dtheta"] for r in rows])
  print(f"{len(rows)} decisions with a target")
  for lo, hi in ((-60, -30), (-30, -10), (-10, 10), (10, 30), (30, 60)):
    m = (el >= lo) & (el < hi)
    if m.any():
      print(f"elevation [{lo}, {hi}): n {m.sum():5d}  pitch_raw {raw[m].mean():6.1f} Hz  "
            f"pitch {pitch[m].mean():+.2f}  pitch > 0 {np.mean(pitch[m] > 0):.2f}")
  up = np.abs(el) > 15
  side = np.abs(az) > 15
  print(f"pitch sign right, |el| > 15: {np.mean(np.sign(pitch[up]) == np.sign(el[up])):.2f}")
  print(f"dtheta sign right, |az| > 15: {np.mean(np.sign(dt[side]) == np.sign(az[side])):.2f}")
  print(f"dtheta < 0: {np.mean(dt < 0):.2f}; held (both under 0.05): "
        f"{np.mean(np.hypot(np.clip(dt / MAX_TURN, -1, 1), pitch) < 0.05):.2f}")


if __name__ == "__main__":
  main(sys.argv[1])
