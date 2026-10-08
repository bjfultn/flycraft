"""Where is each brain's edge of runaway, looking up from w_scale 1 as well as down?

G0 (spec 5) only bisects down from 1, so a twin that is near silent at 1 stays there. This
probes the G0 grey scene on the vpn rung over a fixed set of scales, then finds the largest
scale that does not run away by doubling from 1 (cap --max-scale) and bisecting to the G0
precision. At that edge it runs the G1 stimulus set and the decoder calibration, to see
whether the DNs carry the eye once the network is awake.

usage: python g0_upward.py CONFIG [--scales 1 1.5 2 3 4 6 8] [--max-scale 16] [--device cuda]
                           [--set connectome.scramble_seed=2 ...]
"""

from __future__ import annotations

import argparse

from flycraft.brain.brain import Brain
from flycraft.brain.calibrate import calibrate_decoder, measure_g1, probe_g0, rung_config
from flycraft.brain.retina import RetinaGeometry
from flycraft.brain.wiring import load_wiring
from flycraft.config import load_config
from flycraft.data.connectome import load_connectome


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("config")
  ap.add_argument("--scales", type=float, nargs="*", default=[1, 1.5, 2, 3, 4, 6, 8])
  ap.add_argument("--max-scale", type=float, default=16.0)
  ap.add_argument("--device", default="cuda")
  ap.add_argument("--set", action="append", default=[], help="config override, key=value")
  a = ap.parse_args()
  cfg = load_config(a.config, [f"sim.device={a.device}", *a.set])
  conn = load_connectome(cfg)
  wiring = load_wiring(cfg, conn)
  brain = Brain(conn, wiring, RetinaGeometry.build(conn, cfg.retina.map), cfg)
  brain.reconfigure(rung_config(cfg, "vpn"))
  prec = cfg.calibration.g0_precision
  seen: dict[float, bool] = {}

  def runaway(scale: float) -> bool:
    if scale not in seen:
      brain.set_w_scale(scale)
      p = probe_g0(brain)
      seen[scale] = p.runaway
      print(f"{wiring.label} w_scale {scale:.4g}: mean {p.mean_hz:.2f} Hz, "
            f"{100 * p.frac_over_100hz:.3f}% over 100 Hz" + (" (runaway)" if p.runaway else ""),
            flush=True)
    return seen[scale]

  for s in a.scales:
    runaway(s)
  lo, hi = None, None
  s = 1.0
  while s <= a.max_scale:
    if runaway(s):
      hi = s
      break
    lo, s = s, s * 2
  if lo is None:
    print("runs away at 1: the downward G0 already handles this brain")
    return
  if hi is None:
    print(f"no runaway up to {a.max_scale:g}; edge taken as {lo:g}")
  else:
    while (hi - lo) / hi > prec:
      mid = (lo + hi) / 2
      if runaway(mid):
        hi = mid
      else:
        lo = mid
  brain.set_w_scale(lo)
  print(f"edge: w_scale {lo:.4g}", flush=True)
  g1 = measure_g1(brain, "vpn")
  for az, m in g1.means().items():
    print(f"  spot {az:+.0f}: " + ", ".join(f"{k} {v:.2f}" for k, v in m.items()))
  print(f"decoder: {calibrate_decoder(brain, g1).to_dict()}")


if __name__ == "__main__":
  main()
