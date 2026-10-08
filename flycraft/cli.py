"""The `flycraft` command: prep-data, wiring, m1-report, brain and summarize (spec section 12).

Errors a user can act on (no cache, a bad download, a missing cell type, a port in use, an
unreadable record, a calibration for other wiring, an unopenable --trace file) print one line
and exit 1; a bad config exits 2; `m1-report` exits 3 if no rung passed G1. Ctrl+C exits 130
with one line, not a traceback. Anything else is a bug and keeps its traceback.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import signal
import statistics
import sys
from pathlib import Path

from flycraft.brain.decoder import DecoderError
from flycraft.brain.retina import RetinaGeometry, RetinaMapError
from flycraft.brain.stubs import STUBS
from flycraft.brain.wiring import ScrambleError, load_wiring
from flycraft.config import ConfigError, load_config
from flycraft.data.build import build_connectome
from flycraft.data.connectome import StaleCacheError, cache_path, load_connectome
from flycraft.data.fetch import FetchError, fetch_all

DEFAULT_CONFIG = "configs/real.yaml"
USER_ERRORS = (FileNotFoundError, StaleCacheError, FetchError, ScrambleError, RetinaMapError,
               DecoderError)
# User errors defined in torch-only modules, looked up rather than imported (see main).
LAZY_USER_ERRORS = (("flycraft.brain.brain", "BrainError"),
                    ("flycraft.brain.fly", "CalibrationError"))
DEFAULT_CALIBRATION = "runs/m1/calibration.json"  # where `m1-report` writes it


class RecordError(ValueError):
  pass


def _parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(prog="flycraft",
                              description="A fruit-fly connectome brain plays StarCraft II.")
  sub = p.add_subparsers(dest="cmd", required=True)

  def add(name: str, help_: str) -> argparse.ArgumentParser:
    sp = sub.add_parser(name, help=help_, description=help_)
    sp.add_argument("--config", default=DEFAULT_CONFIG,
                    help=f"YAML config (default {DEFAULT_CONFIG})")
    sp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set sim.seed=3 (repeatable)")
    return sp

  add("prep-data", "Download the MaleCNS v1.0 files (about 566 MB) and build the cache.")
  add("wiring", "Load or build this config's wiring and print its fingerprint.")
  m1 = add("m1-report", "Calibrate the brain (G0, G1 ladder) and write the M1 report.")
  m1.add_argument("--out", default="runs/m1", help="output directory (default runs/m1)")
  m1.add_argument("--device", choices=("auto", "cpu", "cuda"),
                  help="shorthand for --set sim.device=...")
  br = add("brain", "Serve a brain to the game client on 127.0.0.1: the connectome brain, "
           "with its view, or a stub.")
  br.add_argument("--stub", choices=sorted(STUBS),
                  help="serve a stub instead: oracle steers at what it sees, random turns at "
                  "random")
  br.add_argument("--port", type=int, default=8765, help="game port (default 8765)")
  br.add_argument("--calibration", default=DEFAULT_CALIBRATION,
                  help=f"the M1 calibration for this wiring (default {DEFAULT_CALIBRATION})")
  br.add_argument("--view-port", type=int, default=8766,
                  help="brain view port, http://localhost:PORT (default 8766; 0 for none)")
  br.add_argument("--trace", type=Path,
                  help="append one JSON line per obs (positions, score, command) to this file")
  sm = sub.add_parser("summarize", help="Summarize game client records.",
                      description="Per pilot and phase: counted episodes, score and pacing.")
  sm.add_argument("records", nargs="+", type=Path, help="JSON-lines files from flycraft-client")
  return p


def _prep_data(cfg) -> int:
  data_dir = Path(cfg.connectome.data_dir).expanduser()
  fetch_all(data_dir)
  manifest = build_connectome(data_dir, cache_path(cfg), cfg.connectome.inhibitory_nts)
  print(json.dumps({k: manifest[k] for k in ("n_neurons", "n_edges", "nt_counts")}, indent=2))
  return 0


def _wiring(cfg) -> int:
  conn = load_connectome(cfg)
  w = load_wiring(cfg, conn)
  print(f"{w.label}: {w.pre.size:,} edges, {int(w.exempt.sum()):,} exempt (KC to MBON)")
  print(f"fingerprint {w.fingerprint}")
  print(json.dumps(w.stats))
  return 0


def _m1_report(cfg, out: Path) -> int:
  from flycraft.m1report import run_m1  # imports torch

  return run_m1(cfg, out)


def _log_brain(text: str) -> None:
  print(f"flycraft brain: {text}", file=sys.stderr, flush=True)


def _fly(cfg, calibration: str, view_port: int):
  """The connectome brain, and its view server unless view_port is 0."""
  from flycraft.brain.fly import Fly, load_calibration  # imports torch
  from flycraft.brain.view import ViewHub, ViewServer, build_meta

  conn = load_connectome(cfg, log=_log_brain)
  wiring = load_wiring(cfg, conn, log=_log_brain)
  calib = load_calibration(calibration, cfg, wiring)
  geom = RetinaGeometry.build(conn, cfg.retina.map)
  hub = ViewHub() if view_port else None
  fly = Fly(conn, wiring, geom, cfg, calib, view=hub)
  _log_brain(f"{wiring.label}, rung {calib.rung}, w_scale {calib.w_scale:g}, "
             f"on {fly.brain.device}")
  view = ViewServer(hub, *build_meta(fly), port=view_port, log=_log_brain) if hub else None
  return fly, view


def _brain(cfg, args) -> int:
  from flycraft.brain.server import GameServer, ListenError  # imports websockets

  with contextlib.ExitStack() as stack:
    trace = None
    if args.trace:
      try:
        trace = stack.enter_context(args.trace.open("a"))
      except OSError as e:
        print(f"flycraft brain: cannot open --trace {args.trace}: {e}", file=sys.stderr)
        return 1
    if args.stub:
      controller, view = STUBS[args.stub](cfg.decoder, cfg.eye.polarity), None
    else:
      controller, view = _fly(cfg, args.calibration, args.view_port)
    server = GameServer(controller, port=args.port, trace=trace)

    async def serve() -> None:
      stop = asyncio.Event()
      loop = asyncio.get_running_loop()
      for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
      await asyncio.gather(server.run(stop), *([view.run(stop)] if view else []))

    try:
      asyncio.run(serve())
    except ListenError as e:
      print(f"flycraft brain: {e}", file=sys.stderr)
      return 1
  return 0


def _read_records(paths: list[Path]) -> list[dict]:
  rows = []
  for path in paths:
    for n, line in enumerate(path.read_text().splitlines(), 1):
      if not line.strip():
        continue
      try:
        row = json.loads(line)
        rows.append({k: row[k] for k in ("pilot", "phase", "score", "frames", "late_frames",
                                         "fps", "aborted", "counts")})
      except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise RecordError(f"{path}:{n}: not a client record ({e})") from e
  return rows


def summarize(rows: list[dict]) -> list[dict]:
  """One summary per pilot and phase. Discarded episodes count only in `discarded`."""
  groups: dict[tuple[str, str], list[dict]] = {}
  for row in rows:
    groups.setdefault((row["pilot"], row["phase"]), []).append(row)
  out = []
  for (pilot, phase), group in sorted(groups.items()):
    kept = [r for r in group if r["counts"]]
    scores = [r["score"] for r in kept]
    fps = [r["fps"] for r in kept if r["fps"] is not None]
    frames = sum(r["frames"] for r in kept)
    out.append({
      "pilot": pilot, "phase": phase, "n": len(kept), "discarded": len(group) - len(kept),
      "runaways": sum(r["aborted"] == "runaway" for r in kept),
      "mean": statistics.fmean(scores) if scores else None,
      "sd": statistics.stdev(scores) if len(scores) > 1 else None,
      "min": min(scores, default=None), "max": max(scores, default=None),
      "fps": statistics.fmean(fps) if fps else None,
      "late": sum(r["late_frames"] for r in kept) / frames if frames else None})
  return out


def _summarize(paths: list[Path]) -> int:
  def num(v, fmt):
    return "-" if v is None else format(v, fmt)

  print(f"{'pilot':<18}{'phase':<7}{'n':>4}{'mean':>7}{'sd':>6}{'min':>5}{'max':>5}"
        f"{'runaway':>8}{'discard':>8}{'fps':>8}{'late':>7}")
  for s in summarize(_read_records(paths)):
    print(f"{s['pilot']:<18}{s['phase']:<7}{s['n']:>4}{num(s['mean'], '.2f'):>7}"
          f"{num(s['sd'], '.2f'):>6}{num(s['min'], '.0f'):>5}{num(s['max'], '.0f'):>5}"
          f"{s['runaways']:>8}{s['discarded']:>8}{num(s['fps'], '.1f'):>8}"
          f"{num(s['late'], '.1%'):>7}")
  return 0


def main(argv: list[str] | None = None) -> int:
  args = _parser().parse_args(argv)
  if args.cmd == "summarize":  # needs no config
    try:
      return _summarize(args.records)
    except (OSError, RecordError) as e:
      print(f"flycraft summarize: {e}", file=sys.stderr)
      return 1
  overrides = list(args.set)
  if getattr(args, "device", None):
    overrides.append(f"sim.device={args.device}")
  try:
    cfg = load_config(args.config, overrides)
  except (ConfigError, OSError) as e:
    print(f"flycraft: config error: {e}", file=sys.stderr)
    return 2
  try:
    if args.cmd == "prep-data":
      return _prep_data(cfg)
    if args.cmd == "wiring":
      return _wiring(cfg)
    if args.cmd == "brain":
      return _brain(cfg, args)
    return _m1_report(cfg, Path(args.out))
  except KeyboardInterrupt:
    print(f"flycraft {args.cmd}: interrupted", file=sys.stderr)
    return 130
  except USER_ERRORS as e:
    print(f"flycraft {args.cmd}: {e}", file=sys.stderr)
    return 1
  except Exception as e:
    # BrainError and CalibrationError live in torch-only modules, so they are checked here
    # rather than added to USER_ERRORS, keeping `wiring`/`prep-data` free of the torch import.
    # Such an error can only exist once its module is loaded, so look it up instead of
    # importing it: an import here would replace an unrelated bug's traceback with
    # ModuleNotFoundError on a no-torch install.
    if not any(isinstance(e, getattr(sys.modules[mod], name))
               for mod, name in LAZY_USER_ERRORS if mod in sys.modules):
      raise
    print(f"flycraft {args.cmd}: {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  sys.exit(main())
