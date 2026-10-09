"""flycraft-match with a scripted stand-in for the person: every 11 game frames the marines all
shoot the zergling nearest their center. Run it where flycraft-match runs, then the fly:

  python docs/m6/stand_in.py
  flycraft-client --brain URL --map Skirmish --join 14380   # or --scripted
"""
import sys
import time

from s2clientprotocol import raw_pb2 as raw_pb
from s2clientprotocol import sc2api_pb2 as sc_pb

from flycraft.game import match

MARINE, ZERGLING, SELF, ENEMY, ATTACK = 48, 105, 1, 4, 3674
state = {"last": -100, "n": 0, "t0": time.monotonic()}


def person(controller, obs):
  loop = obs.observation.game_loop
  state["n"] += 1
  if state["n"] % 50 == 1:
    units = obs.observation.raw_data.units
    mine = sum(u.alliance == SELF for u in units)
    theirs = sum(u.alliance == ENEMY for u in units)
    print(f"[stand-in] loop {loop} t {time.monotonic() - state['t0']:.0f}s mine {mine} "
          f"theirs {theirs} score {obs.observation.score.score}", file=sys.stderr, flush=True)
  if loop - state["last"] < 11:
    return
  state["last"] = loop
  units = obs.observation.raw_data.units
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


sys.exit(match.host(name="Stand-in", wait_s=240, end_s=3, on_obs=person))
