"""Gate G2 on the calibrated brain (spec section 8): does plasticity reach steering?

run_g2_report builds the brain at its M1 calibration, halves every KC -> MBON weight in the PAM
compartments, reruns the G1 stimulus set and writes g2.json. Population rates (KCs, the MBONs
on each side of the split, the DNs) go in too, so a failure says where the signal stopped.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from flycraft.brain.brain import Brain
from flycraft.brain.calibrate import run_g2, rung_config
from flycraft.brain.dopamine import compartment_edges, dopamine_map
from flycraft.brain.fly import load_calibration
from flycraft.brain.retina import RetinaGeometry
from flycraft.brain.wiring import KC_PATTERN, load_wiring
from flycraft.config import Config
from flycraft.data.connectome import load_connectome


def run_g2_report(cfg: Config, calibration: str | Path, out: Path, log=print) -> int:
  """0 if G2 passes, 3 if it fails (like m1-report when no rung passes)."""
  t0 = time.perf_counter()
  conn = load_connectome(cfg, log=log)
  wiring = load_wiring(cfg, conn, log=log)
  calib = load_calibration(calibration, cfg, wiring)
  geom = RetinaGeometry.build(conn, cfg.retina.map)
  brain = Brain(conn, wiring, geom, rung_config(cfg, calib.rung), w_scale=calib.w_scale,
                calib=calib.decoder)
  dmap = dopamine_map(conn)  # always the real connectome's map
  pam_mbons = dmap.pam_mbons()
  edges = compartment_edges(wiring, pam_mbons, conn.n)
  g = brain.decoder.groups
  pops = {"kc": conn.select(KC_PATTERN), "mbon_pam": pam_mbons,
          "mbon_other": np.setdiff1d(dmap.mbons, pam_mbons), "dn_turn":
          np.concatenate([g.turn_pos, g.turn_neg]), "dn_fwd": np.asarray(g.fwd)}
  log(f"{wiring.label}, rung {calib.rung}, w_scale {calib.w_scale:g}, on {brain.device}: "
      f"{pam_mbons.size} of {dmap.mbons.size} MBONs in PAM compartments, "
      f"{edges.size:,} of {int(wiring.exempt.sum()):,} plastic edges")
  result = run_g2(brain, edges, pops, log)
  frac = dmap.pam_frac
  has_input = dmap.m.sum(axis=0) > 0
  rec = {
    "generated": datetime.now(UTC).isoformat(timespec="seconds"),
    "wiring": {"label": wiring.label, "fingerprint": wiring.fingerprint},
    "calibration": str(calibration), "rung": calib.rung, "w_scale": calib.w_scale,
    "device": brain.device,
    "dopamine_map": {
      "n_pam": int(dmap.pam.sum()), "n_ppl1": int((~dmap.pam).sum()),
      "n_mbons": int(dmap.mbons.size), "n_mbons_with_dan_input": int(has_input.sum()),
      "n_mbons_any_pam": int((frac > 0).sum()), "n_pam_mbons": int(pam_mbons.size),
      "pam_mbon_types": sorted({str(t) for t in conn.type[pam_mbons]}),
    },
    "g2": result.to_dict(),
    "wall_s": round(time.perf_counter() - t0, 1),
  }
  out.mkdir(parents=True, exist_ok=True)
  (out / "g2.json").write_text(json.dumps(rec, indent=2) + "\n")
  log(f"wrote {out / 'g2.json'}")
  return 0 if result.passed else 3
