"""The M1 report (spec section 15): what the real connectome looks like to our brain.

run_m1 builds the brain, runs the G0/G1 calibration ladder and writes three files:
report.md for people, calibration.json for later milestones, and retina.png (the map check).
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch

from flycraft import eye
from flycraft.brain.brain import Brain
from flycraft.brain.calibrate import LadderResult, calibrate_decoder, run_ladder
from flycraft.brain.retina import RetinaGeometry
from flycraft.brain.wiring import load_wiring
from flycraft.config import Config, to_dict
from flycraft.data.build import sign_from_nt
from flycraft.data.connectome import Connectome, load_connectome

# The sign rule Shiu et al. 2024 state. Their code (philshiu/Drosophila_brain_model) loads
# precomputed FlyWire signs that do not follow it everywhere (their issue #11), and FlyWire
# predicts no histamine, so this is a comparison against the stated rule.
SHIU_INHIBITORY_NTS = ("gaba", "glutamate")
SPEED_PROBE_MS = 1000.0


def sign_rule_check(conn: Connectome) -> dict:
  """Where our sign rule and Shiu's reference rule disagree, by transmitter."""
  ref = sign_from_nt(conn.nt, SHIU_INHIBITORY_NTS)
  differ = conn.sign != ref
  by_nt: dict[str, dict[str, int]] = {}
  pre_differ = differ[conn.pre]
  for nt in np.unique(conn.nt[differ]):
    m = conn.nt == nt
    by_nt[str(nt)] = {"neurons": int(m.sum()), "edges": int((pre_differ & m[conn.pre]).sum()),
                      "ours": int(conn.sign[m][0]), "shiu": int(ref[m][0])}
  return {"ours_inhibitory": list(conn.manifest.get("inhibitory_nts", [])),
          "shiu_inhibitory": list(SHIU_INHIBITORY_NTS),
          "neurons_differ": int(differ.sum()), "edges_differ": int(pre_differ.sum()),
          "by_nt": by_nt}


def chirality(ladder: LadderResult, n_se: float) -> str:
  """Which way the descending output turns for a spot on the right (spec 7.6)."""
  if ladder.g1 is None:
    return "not measured (no rung passed G1)"
  z = ladder.g1.z_dn
  if z > n_se:
    return f"turns toward the spot (z {z:+.2f})"
  if z < -n_se:
    return f"turns AWAY from the spot (z {z:+.2f}): mirrored, check the handedness chain"
  return f"no significant turn (z {z:+.2f})"


def speed_probe(brain: Brain) -> float:
  """Wall-clock seconds per simulated second on a grey scene."""
  brain.reset()
  t0 = time.perf_counter()
  brain.run(eye.grey(), SPEED_PROBE_MS)  # ends with a device-to-host copy, so it syncs
  wall = time.perf_counter() - t0
  brain.reset()
  return wall / (SPEED_PROBE_MS / 1000.0)


def plot_retina(geom: RetinaGeometry, path: Path) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots(figsize=(10, 4.5))
  for side, color in (("L", "tab:blue"), ("R", "tab:orange")):
    m = geom.pr_side == side
    ax.scatter(geom.pr_az[m], geom.pr_el[m], s=3, alpha=0.35, color=color,
               label=f"photoreceptors {side} ({int(m.sum())})")
    v = geom.vpn_side == side
    ax.scatter(geom.vpn_az[v], geom.vpn_el[v], s=24, marker="x", color=color,
               label=f"LC10a centres {side} ({int(v.sum())})")
  ax.axvline(0, color="k", lw=0.5)
  ax.set(xlim=(-180, 180), ylim=(-75, 75), xlabel="azimuth (deg, + is the fly's right)",
         ylabel="elevation (deg)", title=f"retina map: {geom.map_mode}")
  ax.legend(loc="lower center", ncol=4, fontsize=7)
  fig.tight_layout()
  fig.savefig(path, dpi=120)
  plt.close(fig)


def _fmt(x) -> str:
  if x is None:
    return "n/a"
  return f"{x:.4g}" if isinstance(x, float) else str(x)


def _sanitize_for_json(obj):
  """Recursively replace non-finite floats with the same strings calibrate._finite uses, so
  calibration.json stays valid strict JSON (RFC 8259 forbids bare NaN/Infinity tokens)."""
  if isinstance(obj, float):
    if math.isnan(obj):
      return "nan"
    if math.isinf(obj):
      return "inf" if obj > 0 else "-inf"
    return obj
  if isinstance(obj, dict):
    return {k: _sanitize_for_json(v) for k, v in obj.items()}
  if isinstance(obj, (list, tuple)):
    return [_sanitize_for_json(v) for v in obj]
  return obj


def _git_sha() -> str | None:
  """The running code's commit, best-effort (not available from an sdist/wheel install)."""
  try:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                         capture_output=True, text=True, timeout=5)
    return out.stdout.strip() if out.returncode == 0 else None
  except (OSError, subprocess.SubprocessError):
    return None


def _flycraft_version() -> str | None:
  """The installed package version, best-effort (not available from a plain source checkout)."""
  try:
    return version("flycraft")
  except PackageNotFoundError:
    return None


def _signal_suffix(g1, n_se: float) -> str:
  """Which signal(s) actually passed G1 (Minor 7): the gate rule itself is untouched."""
  if g1 is None:
    return ""
  lc, dn = abs(g1.z_lc10a) > n_se, abs(g1.z_dn) > n_se
  if lc and dn:
    return " (LC10a and DNs)"
  if lc:
    return " (LC10a only; DNs not significant)"
  if dn:
    return " (DNs only; LC10a not significant)"
  return ""


def write_report(path: Path, rec: dict) -> None:
  c, s, r = rec["connectome"], rec["sign_rule"], rec["retina"]
  lines = [
    "# flycraft M1 report", "",
    f"- generated: {rec['generated']}",
    f"- wiring: {rec['wiring']['label']} (`{rec['wiring']['fingerprint'][:16]}`); "
    f"inhibitory: {', '.join(s['ours_inhibitory'])}.",
    f"- device: {rec['device']}, dtype {rec['config']['sim']['dtype']}",
    f"- speed: {_fmt(rec['speed_s_per_sim_s'])} s wall per simulated second",
    f"- result: {rec['result']}", "",
    "## Connectome", "",
    "| | ours | drosophila-brain-mlx |", "|---|---|---|",
    f"| neurons | {c['n_neurons']:,} | {c['mlx']['n_neurons']:,} |",
    f"| edges | {c['n_edges']:,} | {c['mlx']['n_edges']:,} |", "",
    f"Synapses {c['n_synapses']:,}; self-loops {c['n_self_loops']:,}; "
    f"annotated bodies with a superclass {c['mlx']['annotated_with_superclass']:,}; "
    f"edges whose presynaptic consensus NT is ACh/GABA/Glu "
    f"{c['mlx']['edges_pre_consensus_ach_gaba_glu']:,}.", "",
    "Transmitters: " + ", ".join(f"{k} {v:,}" for k, v in c["nt_counts"].items()) + ".", "",
    "## Sign rule", "",
    f"Inhibitory here: {', '.join(s['ours_inhibitory'])}. "
    f"Shiu et al. 2024 stated rule: {', '.join(s['shiu_inhibitory'])} (their code uses "
    "precomputed FlyWire signs, which have no histamine).",
    f"Differences: {s['neurons_differ']:,} neurons, {s['edges_differ']:,} edges.", "",
  ]
  for nt, d in s["by_nt"].items():
    lines.append(f"- {nt}: {d['neurons']:,} neurons, {d['edges']:,} edges "
                 f"(ours {d['ours']:+d}, Shiu {d['shiu']:+d})")
  lines += [
    "", "## Retina map", "",
    f"Map: **{r['map']}** ({r['map_reason']}). Photoreceptors {r['n_photoreceptors']:,}, "
    f"unassigned {r['n_unassigned']:,}. R1-R6 L {r['n_r16']['L']:,} / R {r['n_r16']['R']:,}; "
    f"R7/R8 L {r['n_r78']['L']:,} / R {r['n_r78']['R']:,}. LC10a with a field "
    f"{r['n_vpn']:,}, without {r['n_vpn_without_rf']:,}.", "",
    "![retina map](retina.png)", "",
    "## Calibration ladder", "",
    "| rung | G0 w_scale | z LC10a | z DNa02 | G1 |", "|---|---|---|---|---|",
  ]
  for t in rec["ladder"]["tried"]:
    lines.append(f"| {t['rung']} | {_fmt(t['g0_scale'])} | {_fmt(t['z_lc10a'])} | "
                 f"{_fmt(t['z_dn'])} | {'pass' if t['passed'] else 'fail'} |")
  for t in rec["ladder"]["tried"]:
    if t["g0_failure"]:
      lines += ["", f"G0 failed on rung {t['rung']}: {t['g0_failure']}"]
  g1 = rec["ladder"].get("g1")
  if g1:
    lines += ["", f"Rung **{g1['rung']}** passed. Mean rates (Hz) by spot azimuth:", "",
              "| azimuth | LC10a L | LC10a R | DNa02 L | DNa02 R |", "|---|---|---|---|---|"]
    for az, m in g1["means_hz"].items():
      lines.append(f"| {az} | {m['lc10a_L']:.2f} | {m['lc10a_R']:.2f} | {m['dn_L']:.2f} | "
                   f"{m['dn_R']:.2f} |")
    probes = rec["ladder"]["g0"]["probes"]
    lines += ["", "G0 probes for this rung:", "",
              "| w_scale | mean Hz | % over 100 Hz | runaway |", "|---|---|---|---|"]
    for p in probes:
      lines.append(f"| {p['scale']:.4g} | {p['mean_hz']:.2f} | "
                   f"{100 * p['frac_over_100hz']:.3f} | {p['runaway']} |")
  lines += ["", "## Brain-chain chirality", "",
            f"Descending output for a spot on the right: {rec['chirality']}.", ""]
  if rec.get("decoder_calibration"):
    lines += ["## Decoder calibration", "", "```json",
              json.dumps(rec["decoder_calibration"], indent=2), "```", ""]
  path.write_text("\n".join(lines))


def run_m1(cfg: Config, out_dir: Path, log=print, pins: dict[str, str] | None = None) -> int:
  """Build the brain, calibrate it and write the report. Returns 0 if a rung passed G1."""
  out = Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)
  conn = load_connectome(cfg, pins=pins, log=log)
  wiring = load_wiring(cfg, conn, log)
  geom = RetinaGeometry.build(conn, cfg.retina.map)
  log(f"retina map {geom.map_mode}: {geom.info['map_reason']}")
  plot_retina(geom, out / "retina.png")
  brain = Brain(conn, wiring, geom, cfg)
  rec = {
    "generated": datetime.now(UTC).isoformat(timespec="seconds"),
    "git_sha": _git_sha(),
    "flycraft_version": _flycraft_version(),
    "device": brain.device + (f" ({torch.cuda.get_device_name()})"
                              if brain.device == "cuda" else f" ({platform.machine()})"),
    "wiring": {"label": wiring.label, "fingerprint": wiring.fingerprint, "stats": wiring.stats},
    "connectome": {k: v for k, v in conn.manifest.items() if k != "sources"},
    "sign_rule": sign_rule_check(conn),
    "retina": {**geom.info, "map": geom.map_mode},
    "config": to_dict(cfg),
  }
  ladder = run_ladder(brain, log)
  rec["ladder"] = {"rung": ladder.rung, "tried": ladder.tried,
                   "g0": ladder.g0.to_dict() if ladder.g0 else None,
                   "g1": ladder.g1.to_dict() if ladder.g1 else None}
  rec["chirality"] = chirality(ladder, cfg.calibration.g1_n_se)
  if ladder.g1 is not None:
    rec["w_scale"] = brain.w_scale
    rec["decoder_calibration"] = calibrate_decoder(brain, ladder.g1).to_dict()
    suffix = _signal_suffix(ladder.g1, cfg.calibration.g1_n_se)
    rec["result"] = f"PASS on rung {ladder.rung}{suffix}"
  else:
    rec["result"] = "FAIL: no rung passed G1"
  rec["speed_s_per_sim_s"] = speed_probe(brain)
  (out / "calibration.json").write_text(json.dumps(_sanitize_for_json(rec), indent=2))
  write_report(out / "report.md", rec)
  log(f"M1 {rec['result']}; report in {out}")
  if not ladder.passed:
    print(f"FAIL: {rec['result']}", file=sys.stderr)
    return 3
  return 0
