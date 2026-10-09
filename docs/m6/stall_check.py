"""The stall stop (client.Stall) replayed on recorded brain traces: which episodes it would have
ended, how many decisions that saves, and whether any of them scored again after the stop.

  python docs/m6/stall_check.py runs/m3/trace-*.jsonl [--decisions 25]

A trace has one line per decision: the obs's score and marine_xy (the squad's middle as seen)
and the act's dtheta, so the client's rule replays exactly. Each trace is also replayed with
every turn taken as 0, the rule without its turn check, to show what that check is for.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from flycraft.game.client import STALL_DECISIONS, Stall


def episodes(path):
  eps = defaultdict(list)
  for line in Path(path).read_text().splitlines():
    if line.startswith("{"):
      row = json.loads(line)
      if "dtheta" in row:  # a decision; a runaway's line has no act
        eps[row["episode"]].append(row)
  return [eps[e] for e in sorted(eps)]


def stop(rows, decisions, turns=True):
  """How many obs the client would have sent before stopping, or None. Act k is in at decision
  k + 1, and the client never stops on the episode's last frame."""
  stall = Stall(decisions)
  for k, row in enumerate(rows):
    stall.obs(row["score"], None if row["marine_xy"] is None else tuple(row["marine_xy"]))
    if stall.act(row["dtheta"] if turns else 0.0) and k + 1 < len(rows) - 1:
      return k + 1
  return None


def check(eps, decisions, turns):
  stopped = saved = lost = 0
  for rows in eps:
    n = stop(rows, decisions, turns)
    if n is None:
      continue
    stopped += 1
    saved += len(rows) - n
    lost += rows[-1]["score"] != rows[n - 1]["score"]
  return stopped, saved, lost


def main():
  p = argparse.ArgumentParser()
  p.add_argument("traces", nargs="+")
  p.add_argument("--decisions", type=int, default=STALL_DECISIONS)
  args = p.parse_args()
  print("| trace | episodes | stopped | decisions saved | scored after a stop "
        "| without the turn check: stopped, scored after |")
  print("|---|---|---|---|---|---|")
  for path in args.traces:
    eps = episodes(path)
    total = sum(len(rows) for rows in eps)
    stopped, saved, lost = check(eps, args.decisions, True)
    stopped0, _, lost0 = check(eps, args.decisions, False)
    print(f"| {Path(path).stem} | {len(eps)} | {stopped} | {saved} of {total} "
          f"({saved / total:.1%}) | {lost} | {stopped0}, {lost0} |")


if __name__ == "__main__":
  main()
