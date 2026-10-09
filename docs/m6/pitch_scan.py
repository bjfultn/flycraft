"""Does any descending neuron say up or down, the way DNa02 says left or right?

Reads grid_sweep.py's rates (the calibration spot at azimuth and elevation -60 to 60 every 20
degrees, four trials each) and, for every DN type and side, how its rate goes with the spot's
elevation and azimuth. A pitch readout for the flying fly needs a type whose rate climbs with
elevation (or falls) at every azimuth and stays readable below the horizon, so it reports per
type: the rate at each elevation (mean over azimuth), the fit rate ~ a + b el + c az, and how
much of the rate's spread the elevation explains. Also, as a ceiling, how well a ridge readout
of all the DNs gets the spot's elevation and azimuth on stimuli it was not fitted on.

Run on the GPU box from the repo root:
  .venv/bin/python docs/m6/pitch_scan.py runs/grid_sweep/grid_sweep_rates.npz [out.json]
"""
import json
import sys
from collections import defaultdict

import numpy as np

from flycraft.config import load_config
from flycraft.data.connectome import load_connectome


def fit(x, y):
  """Least squares y ~ X; (coefficients, R squared)."""
  coef, *_ = np.linalg.lstsq(x, y, rcond=None)
  res = y - x @ coef
  tot = ((y - y.mean()) ** 2).sum()
  return coef, (1.0 - (res ** 2).sum() / tot) if tot > 0 else 0.0


def ridge_cv(feat, target, groups, lam):
  """R squared of a ridge readout of target, each stimulus predicted by a fit without it."""
  pred = np.zeros_like(target)
  for g in np.unique(groups):
    tr, te = groups != g, groups == g
    mu, sd = feat[tr].mean(0), feat[tr].std(0) + 1e-9
    a = (feat[tr] - mu) / sd
    w = np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ (target[tr] - target[tr].mean()))
    pred[te] = (feat[te] - mu) / sd @ w + target[tr].mean()
  return float(1.0 - ((target - pred) ** 2).sum() / ((target - target.mean()) ** 2).sum())


def main(rates_path, out_path=None):
  z = np.load(rates_path)
  grid = z["sweep"] == "grid"
  r = z["dn_rates"][grid].astype(np.float64)  # (stimuli, trials, DNs)
  az, el = z["az"][grid].astype(float), z["el"][grid].astype(float)
  cfg = load_config("configs/real.yaml", [])
  conn = load_connectome(cfg, log=lambda *_: None)
  dns = z["dns"]
  els = np.unique(el)

  by = defaultdict(list)
  for j, i in enumerate(dns):
    by[(str(conn.type[i]), str(conn.side[i]))].append(j)

  m = r.mean(axis=1)  # (stimuli, DNs), mean over trials
  x = np.column_stack([np.ones_like(el), el, az])
  rows = []
  for (typ, side), cols in by.items():
    y = m[:, cols].mean(axis=1)
    if y.max() <= 0.0:
      continue
    coef, r2 = fit(x, y)
    _, r2_az = fit(x[:, [0, 2]], y)
    per_el = [float(y[el == e].mean()) for e in els]
    rows.append({"type": typ, "side": side, "n": len(cols),
                 "hz_by_el": [round(v, 2) for v in per_el],
                 "slope_el": round(float(coef[1]), 4), "slope_az": round(float(coef[2]), 4),
                 "r2": round(float(r2), 3), "r2_el_part": round(float(r2 - r2_az), 3),
                 "below_hz": round(float(y[el < 0].mean()), 2),
                 "above_hz": round(float(y[el > 0].mean()), 2)})
  rows.sort(key=lambda d: -d["r2_el_part"])

  # trial-level samples, a stimulus left out whole
  feat = r.reshape(-1, r.shape[2])
  keep = feat.std(0) > 0
  groups = np.repeat(np.arange(r.shape[0]), r.shape[1])
  tel, taz = np.repeat(el, r.shape[1]), np.repeat(az, r.shape[1])
  ceiling = {f"lam{lam:g}": {"el": round(ridge_cv(feat[:, keep], tel, groups, lam), 3),
                             "az": round(ridge_cv(feat[:, keep], taz, groups, lam), 3)}
             for lam in (10.0, 100.0, 1000.0)}
  # the same, from what fires below the horizon only: can it tell -20 from -60?
  lo = tel < 0
  below = {f"lam{lam:g}": round(ridge_cv(feat[lo][:, keep], tel[lo], groups[lo], lam), 3)
           for lam in (10.0, 100.0, 1000.0)}

  print(f"elevations {els.tolist()}; {int(keep.sum())} of {dns.size} DNs fire at all")
  print(f"ridge readout, held-out R2: {ceiling}; elevation below the horizon only: {below}")
  print("type side n | Hz by elevation (low to high) | slope el, az | R2, el part")
  for d in rows[:25]:
    print(f"{d['type']:>10} {d['side']} {d['n']:>2} | {d['hz_by_el']} | {d['slope_el']:+.3f} "
          f"{d['slope_az']:+.3f} | {d['r2']:.2f} {d['r2_el_part']:.2f}")
  falls = [d for d in rows if d["slope_el"] < 0 and d["r2_el_part"] > 0.3]
  print(f"types whose rate FALLS with elevation (el part > 0.3): {len(falls)}")
  for d in falls[:10]:
    print(f"  {d['type']} {d['side']}: {d['hz_by_el']}")
  if out_path:
    with open(out_path, "w") as f:
      json.dump({"elevations": els.tolist(), "ceiling": ceiling, "below_only": below,
                 "types": rows}, f, indent=1)


if __name__ == "__main__":
  main(*sys.argv[1:3])
