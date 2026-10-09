"""Spike: in the minigames, is any enemy ever off the screen (so only the minimap shows it)?

raw_units lists every unit; feature_units only those on screen. Also reports the camera
rectangle on the minimap and any minimap enemy pixels outside it.
"""
import collections
import json
import random
import time

from pysc2.env import sc2_env
from pysc2.lib import actions, features

from flycraft.game import sc2_compat as sc

F = actions.FUNCTIONS
ENEMY = 4


def rect(mask):
  y, x = mask.nonzero()
  return None if not len(y) else [int(x.min()), int(x.max()), int(y.min()), int(y.max())]


def run(map_name, race, policy, episodes, seed):
  sc.use_flag_defaults()
  sc.install_launch_shim(True, sc.WINDOW)
  env = sc2_env.SC2Env(
    map_name=map_name, players=[sc2_env.Agent(race)],
    agent_interface_format=features.AgentInterfaceFormat(
      feature_dimensions=features.Dimensions(screen=84, minimap=64), use_feature_units=True,
      use_raw_units=True),
    step_mul=8, random_seed=seed, visualize=False)
  sc.minimize_windows(env._sc2_procs[0].pid)
  rng = random.Random(seed)
  for ep in range(episodes):
    (ts,) = env.reset()
    stats = collections.Counter()
    cams = set()
    first = True
    while not ts.last():
      o = ts.observation
      raw_e = sum(int(u.alliance) == ENEMY for u in o.raw_units)
      scr_e = sum(int(u.alliance) == ENEMY for u in o.feature_units)
      mm = o.feature_minimap
      cam = mm.camera > 0
      cr = rect(cam)
      cams.add(tuple(cr) if cr else None)
      mm_enemy = mm.player_relative == ENEMY
      outside = int((mm_enemy & ~cam).sum())
      stats["steps"] += 1
      stats["off_screen_steps"] += raw_e > scr_e
      stats["max_off_screen"] = max(stats["max_off_screen"], raw_e - scr_e)
      stats["mm_enemy_outside_cam_steps"] += outside > 0
      if first:
        first = False
        print(json.dumps({"map": map_name, "ep": ep, "camera": cr,
                          "pathable": rect(mm.pathable > 0),
                          "mm_enemy": rect(mm_enemy), "raw_enemy": raw_e, "screen_enemy": scr_e,
                          "raw_enemy_xy": [[int(u.x), int(u.y)] for u in o.raw_units
                                           if int(u.alliance) == ENEMY][:12]}), flush=True)
      act = F.no_op()
      avail = o.available_actions
      if policy == "wander" and F.Move_screen.id in avail:
        act = F.Move_screen("now", (rng.randrange(84), rng.randrange(84)))
      elif policy == "focus" and F.Attack_screen.id in avail:
        y, x = (o.feature_screen.player_relative == ENEMY).nonzero()
        if len(y):
          i = y.argmax()
          act = F.Attack_screen("now", (x[i], y[i]))
      if act.function == F.no_op.id and F.select_army.id in avail and policy != "idle":
        act = F.select_army("select")
      (ts,) = env.step([act])
    print(json.dumps({"map": map_name, "policy": policy, "ep": ep, **stats,
                      "cameras": sorted(map(str, cams))}), flush=True)
  env.close()


t = time.monotonic()
for policy in ("wander", "focus"):
  run("DefeatRoaches", sc2_env.Race.terran, policy, 2, 1)
  run("DefeatMarines", sc2_env.Race.zerg, policy, 2, 1)
print("done", f"{time.monotonic() - t:.0f}s", flush=True)
