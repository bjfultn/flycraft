"""Option B after G2: could learning in the central complex (CX) reach steering?

Real flies keep visual pattern memory in the CX, and its premotor outputs, the PFL3 neurons,
are DNa02's fifth-largest direct input by type (736 of 48,125 synapses, crossed: PFL3 L feeds
DNa02 R and PFL3 R feeds DNa02 L). The mushroom body's MBONs supply 250. This scan runs the
G1 spot stimuli (the blank is skipped) on the same calibrated brain as G2 and the DN scan, and
records the CX populations and DNa02, under

  base:       the calibrated weights
  pfl_out0:   every edge out of PFL1, PFL2 and PFL3 at 0: what the CX's output drives now
  pfl3_L:     the left PFL3 cells get extra Poisson input at DRIVE_HZ, the LC10a drive's scale
  pfl3_R:     the same for the right PFL3 cells
  pfl3_L_low, pfl3_R_low: the same at DRIVE_LOW_HZ, added after the smoke run showed the full
              drive takes PFL3 to about 110 Hz, far above its rate at base

The turn signal is DNa02 R minus DNa02 L, in Hz (the decoder turns right when R > L).

Criteria, fixed before the run:
  B1, the CX hears the eye: the PFL3 population fires at >= 1 Hz at base, averaged over the
      spots. If not, the eye's input (at LC10a) does not reach the CX, and nothing learned
      there could depend on what the fly sees.
  B2, the CX output can steer: each one-sided drive moves the turn signal by >= 1 Hz and
      |z| > 3 against base, the two sides in opposite directions. B2 counts only if the drive
      took: each drive raises its own PFL3 cells by >= DRIVE_MIN_HZ over base. If it does
      not, B2 is untested, not failed.
  pfl_out0 is descriptive: how much the CX's output contributes to steering now. So are
  the low-drive arms: whether a modest change in PFL3, the size learning might make,
  moves the turn signal at all.

Run on the GPU box from the repo root:
  .venv/bin/python docs/m4/cx_scan.py docs/m1/calibration.json runs/cx_scan [key=value ...]
"""
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from flycraft.brain.brain import Brain
from flycraft.brain.calibrate import _run_trial, g1_stimuli, rung_config
from flycraft.brain.fly import load_calibration
from flycraft.brain.retina import RetinaGeometry
from flycraft.brain.wiring import load_wiring
from flycraft.config import load_config
from flycraft.data.connectome import load_connectome

DRIVE_HZ = 150.0  # retina.vpn_rate_hz: one LC10a fully inside a spot
DRIVE_LOW_HZ = 20.0
MIN_HZ = 1.0
DRIVE_MIN_HZ = 10.0
N_SE = 3.0
PFL = r"^PFL[123]$"


def populations(conn, geom):
  t, s = np.asarray(conn.type).astype(str), np.asarray(conn.side).astype(str)

  def of(pattern, side=None):
    m = np.array([bool(re.match(pattern, x)) for x in t])
    return np.flatnonzero(m if side is None else m & (s == side))

  pops = {
    "LC10a L": geom.vpn_idx[geom.vpn_side == "L"], "LC10a R": geom.vpn_idx[geom.vpn_side == "R"],
    "MeTu": of(r"^MeTu"), "TuBu": of(r"^TuBu"), "ER": of(r"^ER\d"), "EPG": of(r"^EPGt?$"),
    "Delta7": of(r"^Delta7$"), "PEN": of(r"^PEN_"), "PFN": of(r"^PFN"), "hDelta": of(r"^hDelta"),
    "FC2": of(r"^FC2"), "PFL1": of(r"^PFL1$"), "PFL2": of(r"^PFL2$"),
    "PFL3 L": of(r"^PFL3$", "L"), "PFL3 R": of(r"^PFL3$", "R"),
    "DNa02 L": of(r"^DNa02$", "L"), "DNa02 R": of(r"^DNa02$", "R"),
  }
  empty = [k for k, v in pops.items() if v.size == 0]
  if empty:
    raise RuntimeError(f"no neurons for {empty}")
  return pops


def measure(brain, stimuli, pops, repeats, log, tag):
  """{population: (spots, repeats) mean rate in Hz}."""
  out = {k: np.zeros((len(stimuli), repeats)) for k in pops}
  for i, (cond, img) in enumerate(stimuli):
    for r in range(repeats):
      rate, _ = _run_trial(brain, img)
      for k, idx in pops.items():
        out[k][i, r] = rate[idx].mean()
    log(f"{tag} {cond}: " + ", ".join(f"{k} {out[k][i].mean():.2f}" for k in
                                      ("PFL3 L", "PFL3 R", "EPG", "DNa02 L", "DNa02 R")))
  return out


def turn(rates):
  return rates["DNa02 R"] - rates["DNa02 L"]


def change(base, arm):
  """Mean over spots of (arm - base), and its z from the per-spot trial variances."""
  n = base.shape[1]
  diff = (arm.mean(axis=1) - base.mean(axis=1)).mean()
  se = np.sqrt((arm.var(axis=1, ddof=1) / n + base.var(axis=1, ddof=1) / n).sum()) / base.shape[0]
  z = diff / se if se > 0 else (0.0 if diff == 0 else float(np.sign(diff) * np.inf))
  return float(diff), float(z)


def driving(brain, idx, hz):
  """Add Poisson input at hz to idx on top of the eye's, until restored."""
  orig = brain.retina.rates

  def rates(img):
    i, r = orig(img)
    return np.concatenate([i, idx]), np.concatenate([r, np.full(idx.size, hz)])

  brain.retina.rates = rates
  return lambda: setattr(brain.retina, "rates", orig)


def main(calibration, out_dir, *overrides):
  t0 = time.perf_counter()
  log = print
  cfg = load_config("configs/real.yaml", list(overrides))
  conn = load_connectome(cfg, log=log)
  wiring = load_wiring(cfg, conn, log=log)
  calib = load_calibration(calibration, cfg, wiring)
  geom = RetinaGeometry.build(conn, cfg.retina.map)
  brain = Brain(conn, wiring, geom, rung_config(cfg, calib.rung), w_scale=calib.w_scale,
                calib=calib.decoder)
  pops = populations(conn, geom)
  t = np.asarray(conn.type).astype(str)
  pfl = np.array([bool(re.match(PFL, x)) for x in t])
  pfl_out = np.flatnonzero(pfl[wiring.pre])
  stimuli = g1_stimuli(brain.cfg)[1:]
  repeats = brain.cfg.calibration.g1_repeats
  log(f"{wiring.label}, rung {calib.rung}, w_scale {calib.w_scale:g}, on {brain.device}: "
      f"{int(pfl.sum())} PFL cells, {pfl_out.size:,} edges out; " +
      ", ".join(f"{k} {v.size}" for k, v in pops.items()))

  rates = {"base": measure(brain, stimuli, pops, repeats, log, "base")}
  w = brain.edge_weights(pfl_out)
  brain.set_edge_weights(pfl_out, np.zeros_like(w))
  try:
    rates["pfl_out0"] = measure(brain, stimuli, pops, repeats, log, "pfl_out0")
  finally:
    brain.set_edge_weights(pfl_out, w)
  if not np.array_equal(brain.edge_weights(pfl_out), w):
    raise RuntimeError("pfl_out0: weights not restored")
  for suffix, hz in (("", DRIVE_HZ), ("_low", DRIVE_LOW_HZ)):
    for side in ("L", "R"):
      arm = f"pfl3_{side}{suffix}"
      restore = driving(brain, pops[f"PFL3 {side}"], hz)
      try:
        rates[arm] = measure(brain, stimuli, pops, repeats, log, arm)
      finally:
        restore()

  base = rates["base"]
  pfl3_hz = float(np.mean([base["PFL3 L"].mean(), base["PFL3 R"].mean()]))
  b1 = pfl3_hz >= MIN_HZ
  arms = {}
  for name, r in rates.items():
    if name == "base":
      continue
    d, z = change(turn(base), turn(r))
    arms[name] = {"turn_diff_hz": round(d, 3), "turn_z": round(z, 2),
                  "means_hz": {k: round(float(v.mean()), 3) for k, v in r.items()}}
  dl, zl = arms["pfl3_L"]["turn_diff_hz"], arms["pfl3_L"]["turn_z"]
  dr, zr = arms["pfl3_R"]["turn_diff_hz"], arms["pfl3_R"]["turn_z"]
  b2 = (min(abs(dl), abs(dr)) >= MIN_HZ and min(abs(zl), abs(zr)) > N_SE
        and np.sign(dl) == -np.sign(dr) != 0)
  took = {side: float(rates[f"pfl3_{side}"][f"PFL3 {side}"].mean() - base[f"PFL3 {side}"].mean())
          for side in ("L", "R")}
  tested = min(took.values()) >= DRIVE_MIN_HZ
  rec = {
    "generated": datetime.now(UTC).isoformat(timespec="seconds"),
    "wiring": wiring.label, "calibration": str(calibration), "rung": calib.rung,
    "w_scale": calib.w_scale, "device": brain.device, "stimuli": [c for c, _ in stimuli],
    "repeats": repeats, "drive_hz": DRIVE_HZ, "drive_low_hz": DRIVE_LOW_HZ,
    "n_pfl_out_edges": int(pfl_out.size),
    "pop_sizes": {k: int(v.size) for k, v in pops.items()},
    "base_means_hz": {k: round(float(v.mean()), 3) for k, v in base.items()},
    "base_by_spot_hz": {k: [round(float(x), 3) for x in v.mean(axis=1)] for k, v in base.items()},
    "base_turn_hz": round(float(turn(base).mean()), 3),
    "arms": arms,
    "B1": {"pfl3_hz": round(pfl3_hz, 3), "pass": bool(b1)},
    "B2": {"drive_took_hz": {k: round(v, 3) for k, v in took.items()}, "tested": bool(tested),
           "pass": bool(b2 and tested)},
  }
  rec["wall_s"] = round(time.perf_counter() - t0, 1)
  out = Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)
  (out / "cx_scan.json").write_text(json.dumps(rec, indent=2) + "\n")
  np.savez_compressed(out / "cx_scan_rates.npz",
                      **{f"{a}/{k}": v for a, r in rates.items() for k, v in r.items()})
  log(f"B1 (PFL3 >= {MIN_HZ:g} Hz at base): {pfl3_hz:.2f} Hz, {'pass' if b1 else 'fail'}")
  for name, a in arms.items():
    log(f"{name}: turn {a['turn_diff_hz']:+.2f} Hz, z {a['turn_z']}")
  log(f"drive took: PFL3 L {took['L']:+.1f} Hz, PFL3 R {took['R']:+.1f} Hz "
      f"(needs >= {DRIVE_MIN_HZ:g})")
  verdict = "untested" if not tested else "pass" if b2 else "fail"
  log(f"B2 (one-sided PFL3 drive steers both ways): {verdict}")
  log(f"wrote {out / 'cx_scan.json'} in {rec['wall_s']} s")


if __name__ == "__main__":
  main(*sys.argv[1:])
