"""Where can the fly see a spot, and does any descending neuron care whether it is up or down?

Two sweeps of the G1 spot (radius g1_radius_deg, the game's polarity) on the calibrated brain:

  azimuth:   elevation 0, azimuth -180 to 165 every 15 degrees, plus +-110 and +-115 where the
             game's blind spot begins (docs/m6, "Where the fly looks")
  elevation: azimuth 0, elevation -60 to 60 every 15 degrees (0 is in the azimuth sweep)

Every stimulus runs `repeats` trials. Per stimulus it keeps the LC10a input the retina hands
the brain (how many LC10a get any drive, per side, and their summed rate: no simulation needed),
the LC10a spike rates, the decoder's raw (turn, fwd), and every descending neuron's rate.

Up or down: per DN, z of mean(spot above) - mean(spot below) over all elevation-sweep trials
at azimuth 0 (15 to 60 against -15 to -60, trials pooled). With about 1,300 DNs, Bonferroni at
two-sided 0.05; a DN counts only if it also moves by at least 1 Hz.

Run on the GPU box from the repo root:
  .venv/bin/python docs/m6/look_sweep.py docs/m1/calibration.json runs/look_sweep [repeats]
"""
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist

import numpy as np

from flycraft import eye
from flycraft.brain.brain import Brain
from flycraft.brain.calibrate import _run_trial, rung_config
from flycraft.brain.fly import load_calibration
from flycraft.brain.retina import RetinaGeometry
from flycraft.brain.wiring import load_wiring
from flycraft.config import load_config
from flycraft.data.connectome import load_connectome

MIN_HZ = 1.0
ALPHA = 0.05
AZIMUTHS = sorted({*range(-180, 180, 15), -115, -110, 110, 115})
ELEVATIONS = [-60, -45, -30, -15, 15, 30, 45, 60]


def stimuli():
  """(sweep, az, el) for every stimulus, azimuth sweep first."""
  return ([("az", float(a), 0.0) for a in AZIMUTHS]
          + [("el", 0.0, float(e)) for e in ELEVATIONS])


def vpn_drive(brain, img):
  """Per side: how many LC10a the retina drives at all, and their summed input rate (Hz)."""
  _, rates = brain.retina.rates(img)
  geom = brain.geom
  drive = rates[-geom.vpn_idx.size:] if brain.retina.vpn_on else np.zeros(geom.vpn_idx.size)
  return {s: (int((drive[geom.vpn_side == s] > 0).sum()),
              round(float(drive[geom.vpn_side == s].sum()), 1)) for s in ("L", "R")}


def up_down(dn_rates, sweep, el):
  """Per DN: mean(above) - mean(below) at azimuth 0, its z, and the two means."""
  up = dn_rates[(sweep == "el") & (el > 0)].reshape(-1, dn_rates.shape[-1])
  down = dn_rates[(sweep == "el") & (el < 0)].reshape(-1, dn_rates.shape[-1])
  diff = up.mean(axis=0) - down.mean(axis=0)
  se = np.sqrt(up.var(axis=0, ddof=1) / len(up) + down.var(axis=0, ddof=1) / len(down))
  z = np.divide(diff, se, out=np.zeros_like(diff), where=se > 0)
  z[(se == 0) & (diff != 0)] = np.sign(diff[(se == 0) & (diff != 0)]) * np.inf
  return diff, z, up.mean(axis=0), down.mean(axis=0)


def main(calibration, out_dir, repeats="6"):
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
  stim = stimuli()
  log(f"{wiring.label}, rung {calib.rung}, w_scale {calib.w_scale:g}, on {brain.device}: "
      f"{dns.size} DNs, {len(stim)} stimuli x {repeats} trials, spot radius {radius:g}")
  for s in ("L", "R"):
    az = geom.vpn_az[geom.vpn_side == s]
    log(f"LC10a {s}: {az.size} with a field, centre azimuth {az.min():.0f} to {az.max():.0f}")

  dn_rates = np.zeros((len(stim), repeats, dns.size))
  dec = np.zeros((len(stim), repeats, 2))
  lc_hz = np.zeros((len(stim), repeats, 2))
  rows = []
  for i, (sweep, az, el) in enumerate(stim):
    img = eye.disk(az, radius, pol, el_deg=el)
    for r in range(repeats):
      rate, samples = _run_trial(brain, img)
      dn_rates[i, r] = rate[dns]
      dec[i, r] = np.asarray(samples, np.float64).reshape(-1, 2).mean(axis=0)
      lc_hz[i, r] = [rate[lc["L"]].mean(), rate[lc["R"]].mean()]
    drive = vpn_drive(brain, img)
    m = dec[i].mean(axis=0)
    row = {"sweep": sweep, "az": az, "el": el,
           "lc10a_driven_L": drive["L"][0], "lc10a_driven_R": drive["R"][0],
           "lc10a_input_hz_L": drive["L"][1], "lc10a_input_hz_R": drive["R"][1],
           "lc10a_hz_L": round(float(lc_hz[i, :, 0].mean()), 3),
           "lc10a_hz_R": round(float(lc_hz[i, :, 1].mean()), 3),
           "turn_raw": round(float(m[0]), 3), "fwd_raw": round(float(m[1]), 3),
           "turn_sd": round(float(dec[i, :, 0].std(ddof=1)), 3),
           "dns_firing": int((dn_rates[i].mean(axis=0) > 0).sum())}
    rows.append(row)
    log(f"{sweep} az {az:>6g} el {el:>4g}: LC10a driven {row['lc10a_driven_L']:>3}/"
        f"{row['lc10a_driven_R']:<3} spiking {row['lc10a_hz_L']:6.2f}/{row['lc10a_hz_R']:<6.2f}"
        f" turn {row['turn_raw']:+7.2f} fwd {row['fwd_raw']:6.2f}  DNs firing {row['dns_firing']}")

  sweep = np.array([s for s, _, _ in stim])
  el = np.array([e for _, _, e in stim])
  diff, z, up_hz, down_hz = up_down(dn_rates, sweep, el)
  z_crit = NormalDist().inv_cdf(1 - ALPHA / 2 / dns.size)
  hit = (np.abs(z) > z_crit) & (np.abs(diff) >= MIN_HZ)
  order = np.argsort(-np.abs(np.where(np.isfinite(z), z, 1e9 * np.sign(z))))

  def dn_row(k):
    j = dns[k]
    return {"type": str(conn.type[j]), "side": str(conn.side[j]), "body_id": int(conn.body_id[j]),
            "up_hz": round(float(up_hz[k]), 3), "down_hz": round(float(down_hz[k]), 3),
            "diff_hz": round(float(diff[k]), 3),
            "z": float(z[k]) if np.isfinite(z[k]) else str(z[k]),
            "by_el_hz": [round(float(dn_rates[i, :, k].mean()), 2) for i in
                         np.flatnonzero(sweep == "el")]}

  rec = {
    "generated": datetime.now(UTC).isoformat(timespec="seconds"),
    "wiring": wiring.label, "calibration": str(calibration), "rung": calib.rung,
    "w_scale": calib.w_scale, "device": brain.device, "radius_deg": radius, "polarity": pol,
    "repeats": repeats, "n_dns": int(dns.size), "z_crit": z_crit, "min_hz": MIN_HZ,
    "lc10a_centre_az": {s: [round(float(geom.vpn_az[geom.vpn_side == s].min()), 1),
                            round(float(geom.vpn_az[geom.vpn_side == s].max()), 1)]
                        for s in ("L", "R")},
    "stimuli": rows,
    "up_down": {"elevations": ELEVATIONS, "n_hits": int(hit.sum()),
                "hits": [dn_row(k) for k in order if hit[k]],
                "top20": [dn_row(k) for k in order[:20]]},
  }
  log(f"up vs down: {int(hit.sum())} DNs pass |z| > {z_crit:.2f} and >= {MIN_HZ:g} Hz")
  for r in rec["up_down"]["top20"][:12]:
    log(f"  {r['type']:<12}{r['side']:<3}{r['up_hz']:>8.2f}{r['down_hz']:>8.2f}"
        f"{r['diff_hz']:>+8.2f}  z {r['z']}")
  rec["wall_s"] = round(time.perf_counter() - t0, 1)
  out = Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)
  (out / "look_sweep.json").write_text(json.dumps(rec, indent=2) + "\n")
  np.savez_compressed(out / "look_sweep_rates.npz", dns=dns, dn_rates=dn_rates, decoder=dec,
                      lc10a_hz=lc_hz, sweep=sweep, az=np.array([a for _, a, _ in stim]), el=el)
  log(f"wrote {out / 'look_sweep.json'} in {rec['wall_s']} s")


if __name__ == "__main__":
  main(*sys.argv[1:4])
