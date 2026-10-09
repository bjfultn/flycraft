"""Option A after G2: does mushroom-body output drive ANY descending neuron?

G2 found that halving or zeroing the PAM-compartment KC -> MBON weights moves the MBONs but not
DNa02. This scan asks the wider question on the same calibrated brain: run the G1 spot stimuli
(the blank is skipped: no input, a silent brain) and record every descending neuron, under

  base:     the calibrated weights
  kc_mbon0: every plastic KC -> MBON edge at 0 (all compartments, not only PAM)
  mbon_out0: every edge out of an MBON at 0, the upper bound: MB output removed entirely

Per DN and arm, z is the mean over the three spots of (arm - base) Hz, over its standard error
from the per-spot trial variances. With about 1,300 DNs, Bonferroni at two-sided 0.05 is
|z| > 4.12; a DN counts only if it also moves by at least 1 Hz.

Run on the GPU box from the repo root:
  .venv/bin/python docs/m4/dn_scan.py docs/m1/calibration.json runs/dn_scan
"""
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist

import numpy as np

from flycraft.brain.brain import Brain
from flycraft.brain.calibrate import _run_trial, g1_stimuli, rung_config
from flycraft.brain.fly import load_calibration
from flycraft.brain.retina import RetinaGeometry
from flycraft.brain.wiring import MBON_PATTERN, load_wiring
from flycraft.config import load_config
from flycraft.data.connectome import load_connectome

MIN_HZ = 1.0
ALPHA = 0.05


def measure(brain, stimuli, idx, repeats, log, tag):
  """(spots, repeats, len(idx)) rates in Hz."""
  out = np.zeros((len(stimuli), repeats, idx.size))
  for i, (cond, img) in enumerate(stimuli):
    for r in range(repeats):
      rate, _ = _run_trial(brain, img)
      out[i, r] = rate[idx]
    log(f"{tag} {cond}: mean DN rate {out[i].mean():.3f} Hz")
  return out


def zscores(base, arm):
  """Per neuron: mean over spots of (arm - base), its z, and both arms' mean rates."""
  n = base.shape[1]
  diff = (arm.mean(axis=1) - base.mean(axis=1)).mean(axis=0)
  var = (arm.var(axis=1, ddof=1) / n + base.var(axis=1, ddof=1) / n).sum(axis=0)
  se = np.sqrt(var) / base.shape[0]
  z = np.divide(diff, se, out=np.zeros_like(diff), where=se > 0)
  # Silent in one arm's trials but not the other's: se can be 0 with diff != 0.
  z[(se == 0) & (diff != 0)] = np.sign(diff[(se == 0) & (diff != 0)]) * np.inf
  return diff, z


def row(conn, dns, i, base_mean, arm_mean, diff, z):
  j = dns[i]
  return {"type": str(conn.type[j]), "side": str(conn.side[j]), "body_id": int(conn.body_id[j]),
          "base_hz": round(float(base_mean[i]), 3), "arm_hz": round(float(arm_mean[i]), 3),
          "diff_hz": round(float(diff[i]), 3),
          "z": float(z[i]) if np.isfinite(z[i]) else str(z[i])}


def main(calibration, out_dir):
  t0 = time.perf_counter()
  log = print
  cfg = load_config("configs/real.yaml", [])
  conn = load_connectome(cfg, log=log)
  wiring = load_wiring(cfg, conn, log=log)
  calib = load_calibration(calibration, cfg, wiring)
  geom = RetinaGeometry.build(conn, cfg.retina.map)
  brain = Brain(conn, wiring, geom, rung_config(cfg, calib.rung), w_scale=calib.w_scale,
                calib=calib.decoder)
  dns = np.flatnonzero(conn.superclass == "descending_neuron")
  mbon = np.zeros(conn.n, bool)
  mbon[conn.select(MBON_PATTERN)] = True
  arms = {"kc_mbon0": np.flatnonzero(wiring.exempt),
          "mbon_out0": np.flatnonzero(mbon[wiring.pre])}
  stimuli = g1_stimuli(brain.cfg)[1:]
  repeats = brain.cfg.calibration.g1_repeats
  log(f"{wiring.label}, rung {calib.rung}, w_scale {calib.w_scale:g}, on {brain.device}: "
      f"{dns.size} DNs, {int(mbon.sum())} MBONs, " +
      ", ".join(f"{k} {v.size:,} edges" for k, v in arms.items()))

  base = measure(brain, stimuli, dns, repeats, log, "base")
  rates = {"base": base}
  for name, edges in arms.items():
    w = brain.edge_weights(edges)
    brain.set_edge_weights(edges, np.zeros_like(w))
    try:
      rates[name] = measure(brain, stimuli, dns, repeats, log, name)
    finally:
      brain.set_edge_weights(edges, w)
    if not np.array_equal(brain.edge_weights(edges), w):
      raise RuntimeError(f"{name}: weights not restored")

  z_crit = NormalDist().inv_cdf(1 - ALPHA / 2 / dns.size)
  base_mean = base.mean(axis=(0, 1))
  rec = {
    "generated": datetime.now(UTC).isoformat(timespec="seconds"),
    "wiring": wiring.label, "calibration": str(calibration), "rung": calib.rung,
    "w_scale": calib.w_scale, "device": brain.device,
    "stimuli": [c for c, _ in stimuli], "repeats": repeats, "n_dns": int(dns.size),
    "z_crit": z_crit, "min_hz": MIN_HZ,
    "n_dns_firing_base": int((base_mean > 0).sum()),
    "n_dns_over_1hz_base": int((base_mean >= 1).sum()),
    "arms": {},
  }
  for name, edges in arms.items():
    diff, z = zscores(base, rates[name])
    arm_mean = rates[name].mean(axis=(0, 1))
    hit = (np.abs(z) > z_crit) & (np.abs(diff) >= MIN_HZ)
    order = np.argsort(-np.abs(np.where(np.isfinite(z), z, 1e9 * np.sign(z))))

    rec["arms"][name] = {
      "n_edges": int(edges.size), "n_hits": int(hit.sum()),
      "hits": [row(conn, dns, i, base_mean, arm_mean, diff, z) for i in order if hit[i]],
      "top20": [row(conn, dns, i, base_mean, arm_mean, diff, z) for i in order[:20]],
      "n_dns_firing": int((arm_mean > 0).sum()),
    }
    log(f"{name}: {int(hit.sum())} DNs pass |z| > {z_crit:.2f} and >= {MIN_HZ:g} Hz")
    for r in rec["arms"][name]["top20"][:10]:
      log(f"  {r['type']:<12}{r['side']:<3}{r['base_hz']:>8.2f}{r['arm_hz']:>8.2f}"
          f"{r['diff_hz']:>+8.2f}  z {r['z']}")
  firing = np.argsort(-base_mean)[:20]
  rec["top_base"] = [{"type": str(conn.type[dns[i]]), "side": str(conn.side[dns[i]]),
                      "hz": round(float(base_mean[i]), 3)} for i in firing]
  rec["wall_s"] = round(time.perf_counter() - t0, 1)
  out = Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)
  (out / "dn_scan.json").write_text(json.dumps(rec, indent=2) + "\n")
  np.savez_compressed(out / "dn_scan_rates.npz", dns=dns, base=rates["base"],
                      kc_mbon0=rates["kc_mbon0"], mbon_out0=rates["mbon_out0"])
  log(f"wrote {out / 'dn_scan.json'} in {rec['wall_s']} s")


if __name__ == "__main__":
  main(*sys.argv[1:3])
