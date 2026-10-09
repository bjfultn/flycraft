"""Where on the eye does a spot make the fly turn, and how small can it be at the back?

Follow-up to look_sweep.py, which found the brain turning for the calibration spot (radius
g1_radius_deg) out to 150 degrees either side on the horizon, but silent, turn exactly 0, for
spots below the horizon dead ahead. The flying fly (docs/m6/README.md, "The flying fly") puts
targets anywhere within 60 degrees of the eye's centre, up or down, so:

  grid:  the calibration spot at azimuth and elevation -60 to 60 every 20 degrees
  small: a 3 degree spot, the size the walking fly sees a target at the start of a
         DefeatMarines episode, on the horizon at the back (90 to 150 either side) and,
         to compare, in front (30 and 60 either side)

Every stimulus runs `repeats` trials from rest, as in calibration. Per stimulus it keeps the
decoder's raw (turn, fwd) per trial, how many DNs fire per trial, and the LC10a rates. Before
any simulation it also lists how many LC10a the retina drives at all for spots of radius 2, 3,
5 and 10 degrees all the way round the horizon.

Run on the GPU box from the repo root:
  .venv/bin/python docs/m6/grid_sweep.py docs/m1/calibration.json runs/grid_sweep [repeats]
"""
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from look_sweep import vpn_drive

from flycraft import eye
from flycraft.brain.brain import Brain
from flycraft.brain.calibrate import _run_trial, rung_config
from flycraft.brain.fly import load_calibration
from flycraft.brain.retina import RetinaGeometry
from flycraft.brain.wiring import load_wiring
from flycraft.config import load_config
from flycraft.data.connectome import load_connectome

GRID = [-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0]
SMALL_DEG = 3.0
SMALL_AZ = [-150.0, -135.0, -120.0, -105.0, -90.0, -60.0, -30.0,
            30.0, 60.0, 90.0, 105.0, 120.0, 135.0, 150.0]
DRIVE_RADII = [2.0, 3.0, 5.0, 10.0]


def stimuli(radius):
  """(sweep, az, el, radius) for every simulated stimulus, the grid first."""
  return ([("grid", a, e, radius) for e in GRID for a in GRID]
          + [("small", a, 0.0, SMALL_DEG) for a in SMALL_AZ])


def main(calibration, out_dir, repeats="4"):
  t0 = time.perf_counter()
  log = print
  repeats = int(repeats)
  cfg = load_config("configs/real.yaml", [])
  conn = load_connectome(cfg, log=log)
  wiring = load_wiring(cfg, conn, log=log)
  calib = load_calibration(calibration, cfg, wiring)
  geom = RetinaGeometry.build(conn, cfg.retina.map)
  brain = Brain(conn, wiring, geom, rung_config(cfg, calib.rung), w_scale=calib.w_scale,
                calib=calib.decoder)
  dns = np.flatnonzero(conn.superclass == "descending_neuron")
  lc = {s: geom.vpn_idx[geom.vpn_side == s] for s in ("L", "R")}
  pol, radius = brain.cfg.eye.polarity, brain.cfg.calibration.g1_radius_deg

  drive = []
  for r in DRIVE_RADII:
    for a in range(-180, 180, 15):
      d = vpn_drive(brain, eye.disk(float(a), r, pol))
      drive.append({"radius": r, "az": float(a), "lc10a_driven_L": d["L"][0],
                    "lc10a_driven_R": d["R"][0]})
  for r in DRIVE_RADII:
    row = [d["lc10a_driven_L"] + d["lc10a_driven_R"] for d in drive if d["radius"] == r]
    log(f"LC10a driven, radius {r:g}, az -180 to 165: {row}")

  stim = stimuli(radius)
  log(f"{wiring.label}, rung {calib.rung}, w_scale {calib.w_scale:g}, on {brain.device}: "
      f"{len(stim)} stimuli x {repeats} trials")
  dn_rates = np.zeros((len(stim), repeats, dns.size), np.float32)
  dec = np.zeros((len(stim), repeats, 2))
  lc_hz = np.zeros((len(stim), repeats, 2))
  rows = []
  for i, (sweep, az, el, rad) in enumerate(stim):
    img = eye.disk(az, rad, pol, el_deg=el)
    for k in range(repeats):
      rate, samples = _run_trial(brain, img)
      dn_rates[i, k] = rate[dns]
      dec[i, k] = np.asarray(samples, np.float64).reshape(-1, 2).mean(axis=0)
      lc_hz[i, k] = [rate[lc["L"]].mean(), rate[lc["R"]].mean()]
    d = vpn_drive(brain, img)
    row = {"sweep": sweep, "az": az, "el": el, "radius": rad,
           "lc10a_driven_L": d["L"][0], "lc10a_driven_R": d["R"][0],
           "lc10a_hz_L": round(float(lc_hz[i, :, 0].mean()), 3),
           "lc10a_hz_R": round(float(lc_hz[i, :, 1].mean()), 3),
           "turn_raw": [round(float(x), 3) for x in dec[i, :, 0]],
           "fwd_raw": [round(float(x), 3) for x in dec[i, :, 1]],
           "dns_firing": [int(x) for x in (dn_rates[i] > 0).sum(axis=1)]}
    rows.append(row)
    log(f"{sweep:<5} az {az:>5g} el {el:>4g} r {rad:>2g}: LC10a {row['lc10a_driven_L']:>3}/"
        f"{row['lc10a_driven_R']:<3} turn {np.mean(row['turn_raw']):+7.2f} "
        f"DNs firing {row['dns_firing']}", flush=True)

  rec = {
    "generated": datetime.now(UTC).isoformat(timespec="seconds"),
    "wiring": wiring.label, "calibration": str(calibration), "rung": calib.rung,
    "w_scale": calib.w_scale, "device": brain.device, "polarity": pol, "repeats": repeats,
    "lc10a_drive": drive, "stimuli": rows,
  }
  rec["wall_s"] = round(time.perf_counter() - t0, 1)
  out = Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)
  (out / "grid_sweep.json").write_text(json.dumps(rec, indent=2) + "\n")
  np.savez_compressed(out / "grid_sweep_rates.npz", dns=dns, dn_rates=dn_rates, decoder=dec,
                      lc10a_hz=lc_hz, sweep=np.array([s[0] for s in stim]),
                      az=np.array([s[1] for s in stim]), el=np.array([s[2] for s in stim]),
                      radius=np.array([s[3] for s in stim]))
  log(f"wrote {out / 'grid_sweep.json'} in {rec['wall_s']} s")


if __name__ == "__main__":
  main(*sys.argv[1:4])
