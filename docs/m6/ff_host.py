"""Probe host (M6 friendly fire): the stand-in of stand_in.py, plus who hit each zergling.
Zerglings deal 5 a hit and marines 6, so a zergling's health drop says who hit it; a drop with
no single reading counts as unclear.

  python docs/m6/ff_host.py
"""
import sys
import time
from collections import Counter

from s2clientprotocol import raw_pb2 as raw_pb
from s2clientprotocol import sc2api_pb2 as sc_pb

from flycraft.game import match

MARINE, ZERGLING, SELF, ENEMY, ATTACK = 48, 105, 1, 4, 3674
state = {"last": -100, "n": 0, "t0": time.monotonic(), "hp": {}}
hits = Counter()


def split(d):
  """(ling hits, marine hits) for a health drop d, or None if it has no single reading."""
  n = round(d)
  if abs(d - n) > 0.4:
    return None
  ways = [(a, (n - 5 * a) // 6) for a in range(n // 5 + 1) if (n - 5 * a) % 6 == 0]
  return ways[0] if len(ways) == 1 else None


def person(controller, obs):
  loop = obs.observation.game_loop
  units = obs.observation.raw_data.units
  hp = state["hp"]
  for u in units:
    if u.alliance == ENEMY and u.unit_type == ZERGLING:
      prev = hp.get(u.tag)
      if prev is not None and prev - u.health > 0.5:
        s = split(prev - u.health)
        if s is None:
          hits["unclear"] += 1
        else:
          hits["ling"] += s[0]
          hits["marine"] += s[1]
          if s[0]:
            print(f"[probe] loop {loop}: zergling {u.tag} hit by a zergling "
                  f"({prev:.1f} to {u.health:.1f})", file=sys.stderr, flush=True)
      hp[u.tag] = u.health
  state["n"] += 1
  if state["n"] % 50 == 1:
    mine = sum(u.alliance == SELF for u in units)
    theirs = sum(u.alliance == ENEMY for u in units)
    print(f"[stand-in] loop {loop} t {time.monotonic() - state['t0']:.0f}s mine {mine} "
          f"theirs {theirs} hits {dict(hits)}", file=sys.stderr, flush=True)
  if loop - state["last"] < 11:
    return
  state["last"] = loop
  marines = [u for u in units if u.alliance == SELF and u.unit_type == MARINE]
  lings = [u for u in units if u.alliance == ENEMY and u.unit_type == ZERGLING]
  if not marines or not lings:
    return
  cx = sum(m.pos.x for m in marines) / len(marines)
  cy = sum(m.pos.y for m in marines) / len(marines)
  target = min(lings, key=lambda z: (z.pos.x - cx) ** 2 + (z.pos.y - cy) ** 2)
  cmd = raw_pb.ActionRawUnitCommand(ability_id=ATTACK, unit_tags=[m.tag for m in marines],
                                    target_unit_tag=target.tag)
  controller.act(sc_pb.Action(action_raw=raw_pb.ActionRaw(unit_command=cmd)))


code = match.host(name="Stand-in", wait_s=240, end_s=3, on_obs=person)
print(f"[probe] hits on zerglings: {dict(hits)}", file=sys.stderr, flush=True)
sys.exit(code)
