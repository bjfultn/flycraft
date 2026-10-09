"""flycraft-match: the person's side of a Skirmish match against the fly (spec M6).

It starts SC2 where the person can see it, creates a real-time Skirmish game for two, joins it
as player 1 (Terran), and hands the game's settings to the fly on a local port, the way pysc2's
play_vs_agent hosts a human. Unlike play_vs_agent it joins before it lets the fly in, so the
person is always player 1. The fly joins with `flycraft-client --map Skirmish --join PORT`.

When the game ends it prints one JSON line with the person's result, leaves the end screen up
for a few seconds, and closes SC2. Messages go to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time

import portpicker
from pysc2 import maps, run_configs
from pysc2.env import lan_sc2_env, sc2_env
from s2clientprotocol import sc2api_pb2 as sc_pb

from flycraft.game import sc2_compat
from flycraft.game.client import WINDOW, GameError, SetupError, window_arg

MAP = "Skirmish"
PORT = 14380  # the settings port; the game takes the 4 after it
PORTS = 5
PERSON = 1
WAIT_S = 300  # how long the person waits for the fly to join; pysc2 takes whole seconds
END_SCREEN_S = 10.0  # the end screen stays up this long
POLL_S = 0.1  # how often the person's side looks for the end of the game


def _log(msg: str) -> None:
  print(f"[match] {msg}", file=sys.stderr, flush=True)


def _settings(proc, map_inst, run_config, ports: list[int]) -> dict:
  """play_vs_agent's settings, the ones LanSC2Env reads."""
  return {
    "remote": False,
    "game_version": proc.version.game_version,
    "realtime": True,
    "map_name": map_inst.name,
    "map_path": map_inst.path,
    "map_data": map_inst.data(run_config),
    "ports": {"server": {"game": ports[1], "base": ports[2]},
              "client": {"game": ports[3], "base": ports[4]}},
  }


def _join_request(settings: dict, name: str) -> sc_pb.RequestJoinGame:
  join = sc_pb.RequestJoinGame(
    race=sc2_env.Race.terran, player_name=name,
    options=sc_pb.InterfaceOptions(raw=True, score=True))
  join.shared_port = 0  # unused
  ports = settings["ports"]
  join.server_ports.game_port = ports["server"]["game"]
  join.server_ports.base_port = ports["server"]["base"]
  join.client_ports.add(game_port=ports["client"]["game"], base_port=ports["client"]["base"])
  return join


class _Joining(threading.Thread):
  """The person's join_game, which returns only once the fly has joined too."""

  def __init__(self, controller, request: sc_pb.RequestJoinGame):
    super().__init__(name="person-join", daemon=True)
    self.controller, self.request = controller, request
    self.started_call = threading.Event()
    self.response = self.error = None

  def run(self):
    self.started_call.set()
    try:
      self.response = self.controller.join_game(self.request)
    except Exception as e:  # handed to the main thread, which reports it
      self.error = e


class _Door(threading.Thread):
  """Lets the fly in: pysc2's tcp_server waits for it to connect, then sends the settings."""

  def __init__(self, port: int, settings: dict):
    super().__init__(name="fly-door", daemon=True)
    self.port, self.settings = port, settings
    self.conn = self.error = None

  def run(self):
    try:
      self.conn = lan_sc2_env.tcp_server(
        lan_sc2_env.Addr(sc2_compat.LAN_HOST, self.port), self.settings)
    except Exception as e:  # the fly cannot get in: the main thread stops the match
      self.error = e


def host(port: int = PORT, name: str = "Human", window: tuple[int, int] = WINDOW,
         wait_s: int = WAIT_S, end_s: float = END_SCREEN_S, out=None, log=_log,
         on_obs=None) -> int:
  """Host one match and play the person's side; return the exit code. on_obs(controller, obs)
  sees every observation, so a check on SC2 can stand in for the person."""
  out = out or sys.stdout
  sc2_compat.use_flag_defaults()
  sc2_compat.install_launch_shim(False, window)
  sc2_compat.check_built(MAP)
  ports = [port + i for i in range(PORTS)]
  busy = [p for p in ports if not portpicker.is_port_free(p)]
  if busy:
    raise SetupError(f"port {', '.join(map(str, busy))} is in use, and a match takes ports "
                     f"{ports[0]} to {ports[-1]}: is another match running? Pick a --port")
  run_config = run_configs.get()
  map_inst = maps.get(MAP)
  proc = door = None
  try:
    try:
      proc = run_config.start(extra_ports=ports[1:], timeout_seconds=wait_s,
                              host=sc2_compat.LAN_HOST, window_loc=(50, 50))
      controller = proc.controller
      settings = _settings(proc, map_inst, run_config, ports)
      create = sc_pb.RequestCreateGame(realtime=True,
                                       local_map=sc_pb.LocalMap(map_path=map_inst.path))
      create.player_setup.add(type=sc_pb.Participant)
      create.player_setup.add(type=sc_pb.Participant)
      controller.save_map(map_inst.path, settings["map_data"])
      controller.create_game(create)
    except sc2_compat.SC2_ERRORS as e:
      raise GameError(f"SC2 did not start the match: {e}") from e

    joining = _Joining(controller, _join_request(settings, name))
    joining.start()
    joining.started_call.wait()  # the person asks to join before the fly can connect
    door = _Door(port, settings)
    door.start()
    log(f"waiting up to {wait_s} s for the fly: flycraft-client --brain URL "
        f"--map {MAP} --join {port}")
    while joining.is_alive():  # SC2 gives up on its own after wait_s
      joining.join(0.5)  # in steps, so Ctrl+C stops the wait on Windows too
      if door.error is not None:
        raise GameError(f"could not wait for the fly on port {port}: {door.error}")
    if joining.error is not None:
      if door.conn is None:
        raise GameError(f"the fly did not join within {wait_s} s ({joining.error})")
      raise GameError(f"the fly connected but the game did not start: {joining.error}")
    if joining.response.player_id != PERSON:
      raise SetupError(f"joined the match as player {joining.response.player_id}, not "
                       f"{PERSON}: start flycraft-match before the fly")
    log("the fly is in: the match is on")

    t0 = time.monotonic()
    try:
      while True:
        obs = controller.observe()
        if on_obs is not None:
          on_obs(controller, obs)
        if obs.player_result:
          break
        time.sleep(POLL_S)
    except sc2_compat.SC2_ERRORS as e:
      raise GameError(f"SC2 failed during the match: {e}") from e
    results = {r.player_id: r.result for r in obs.player_result}
    result = sc_pb.Result.Name(results[PERSON]) if PERSON in results else None
    rec = {"map": MAP, "player": PERSON, "result": result,
           "rounds_won": int(obs.observation.score.score),
           "game_loop": int(obs.observation.game_loop),
           "wall_s": round(time.monotonic() - t0, 1)}
    print(json.dumps(rec), file=out, flush=True)
    log(f"the match is over: {result or 'no result for the person'}")
    time.sleep(end_s)
    return 0 if result else 1
  finally:
    if door is not None and door.conn is not None:
      door.conn.close()
    if proc is not None:
      proc.close()


def _parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
    prog="flycraft-match",
    description=f"Play a {MAP} match against the fly: 4 marines against its 9 zerglings, first "
                "to 3 rounds. Start this first; then start the fly with flycraft-client "
                f"--brain URL --map {MAP} --join PORT. Build the map first with flycraft-maps.")
  p.add_argument("--port", type=int, default=PORT,
                 help=f"the port the fly joins on; the match also takes the 4 after it "
                      f"(default {PORT})")
  p.add_argument("--name", default="Human", help="the person's name in the game")
  p.add_argument("--window", type=window_arg, default=WINDOW,
                 help=f"SC2 window size (default {WINDOW[0]}x{WINDOW[1]})")
  p.add_argument("--wait", type=int, default=WAIT_S,
                 help=f"seconds to wait for the fly to join (default {WAIT_S})")
  return p


def main(argv: list[str] | None = None) -> int:
  p = _parser()
  args = p.parse_args(argv)
  if not 1024 <= args.port <= 65535 - (PORTS - 1):
    p.error(f"--port must be from 1024 to {65535 - (PORTS - 1)}")
  if not args.wait > 0:
    p.error("--wait must be positive")
  try:
    return host(args.port, args.name, args.window, args.wait)
  except (SetupError, GameError) as e:
    _log(str(e))
    return 1
  except KeyboardInterrupt:
    _log("stopped")
    return 1


if __name__ == "__main__":
  sys.exit(main())
