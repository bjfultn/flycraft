"""Everything that touches pysc2 (spec sections 7 and 13). Only the game client imports it.

- to_call and to_frame translate the body's Actions into pysc2 calls and pysc2's TimeSteps
  into Frames.
- The launch shim fixes pysc2's launch (SC2Env hands its extra_ports down to Popen, which
  rejects them), sizes the window, and starts a train-mode SC2 without taking focus; SC2Game
  then minimizes it. (SC2 started minimized finds no graphics device and quits.)
- SC2Game is the client's Game: one SC2 process for the whole run. pysc2 reads absl flags,
  which only absl's app.run parses; the client has its own argparse, so SC2Game takes the
  flags' defaults.
- SC2LanGame is the fly's side of a Skirmish match (spec M6), which flycraft-match hosts,
  through pysc2's LanSC2Env, mended here for pysc2 4.0.
- DefeatMarines and Skirmish are registered with pysc2 here, under Maps/flycraft;
  flycraft-maps builds them.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

# pysc2 imports pygame, which greets on stdout, and the client's stdout carries only records.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
from absl import flags
from pysc2 import maps, run_configs
from pysc2.env import lan_sc2_env, sc2_env
from pysc2.lib import actions, features, remote_controller, sc_process
from pysc2.lib import protocol as sc2_protocol
from s2clientprotocol import sc2api_pb2 as sc_pb

from flycraft.game.body import Action
from flycraft.game.client import WINDOW, Frame, GameError, SetupError

F = actions.FUNCTIONS
PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
SW_SHOWNOACTIVATE = 4
SW_SHOWMINNOACTIVE = 7
STARTF_USESHOWWINDOW = 0x1
MINIMAP = 64
SC2_ERRORS = (sc_process.SC2LaunchError, sc2_protocol.ConnectionError, sc2_protocol.ProtocolError,
              remote_controller.ConnectError, remote_controller.RequestError, OSError)
# pysc2 raises ValueError when the game stops advancing ("didn't advance to the expected
# game loop"), so in play it is SC2 failing too.
PLAY_ERRORS = (*SC2_ERRORS, ValueError)
BUILT = "flycraft"  # the Maps directory of the maps flycraft-maps builds
LAN_HOST = "127.0.0.1"  # a match is played on one PC
FLY = 2  # the fly's player number in a match; the person is 1


class DefeatMarines(maps.lib.Map):
  """Zerglings against marines (spec M3c). pysc2 plays every Map subclass with a filename,
  under its class name; the settings are its minigames'."""
  directory = BUILT
  filename = "DefeatMarines"
  players = 1
  score_index = 0
  game_steps_per_episode = 0
  step_mul = 8


class Skirmish(maps.lib.Map):
  """A person's 4 marines against the fly's 9 zerglings, in rounds (spec M6). Played over
  LAN, where pysc2 scores a game by its outcome, not the map's score."""
  directory = BUILT
  filename = "Skirmish"
  players = 2
  score_index = -1
  game_steps_per_episode = 0
  step_mul = 8


def _log(text: str) -> None:
  print(f"flycraft-client: {text}", file=sys.stderr, flush=True)


def to_call(action: Action, available) -> actions.FunctionCall:
  """The pysc2 call for an Action. An order the game will not take now becomes select_army
  if it can be taken, so the next decision can move, else no_op."""
  avail = {int(a) for a in available}
  if action.kind == "move" and F.Move_screen.id in avail:
    return F.Move_screen("now", (round(action.xy[0]), round(action.xy[1])))
  if action.kind == "attack" and F.Attack_screen.id in avail:
    return F.Attack_screen("now", (round(action.xy[0]), round(action.xy[1])))
  if action.kind == "stop" and F.Stop_quick.id in avail:
    return F.Stop_quick("now")
  if action.kind != "noop" and F.select_army.id in avail:
    return F.select_army("select")
  return F.no_op()


def all_selected(obs) -> bool:
  """Whether every army unit is selected, so an order reaches the whole squad. DefeatRoaches
  and DefeatMarines send new units when a wave falls; they arrive unselected."""
  selected = max(len(obs["single_select"]), len(obs["multi_select"]))
  return selected >= int(obs["player"][features.Player.army_count])


def to_frame(ts, match: bool = False) -> Frame:
  """The Frame for a TimeStep. In a match, the last step's reward is the outcome (1 a win,
  -1 a loss, 0 a tie): pysc2 scores a LAN game that way."""
  obs = ts.observation
  return Frame(
    player_relative=np.asarray(obs["feature_screen"][PLAYER_RELATIVE]),
    reward=float(ts.reward or 0.0),
    score=float(obs["score_cumulative"][0]),
    last=bool(ts.last()),
    loop=int(obs["game_loop"][0]),
    can_move=F.Move_screen.id in {int(a) for a in obs["available_actions"]}
    and all_selected(obs),
    outcome=int(ts.reward) if match and ts.last() else None)


def launch_kwargs(kwargs: dict, minimized: bool, startupinfo_cls=None) -> dict:
  """The Popen kwargs for SC2: without extra_ports, and in train or eval mode shown without
  activation (spec section 13). Not minimized: SC2 started minimized has no graphics device."""
  out = {k: v for k, v in kwargs.items() if k != "extra_ports"}
  if minimized:
    info = (startupinfo_cls or subprocess.STARTUPINFO)()
    info.dwFlags |= STARTF_USESHOWWINDOW
    info.wShowWindow = SW_SHOWNOACTIVATE
    out["startupinfo"] = info
  return out


class _Process(sc_process.StarcraftProcess):
  minimized = False
  window_size = WINDOW  # pysc2's default 640 x 480 is too small to watch

  def __init__(self, *args, **kwargs):
    kwargs.setdefault("window_size", self.window_size)
    super().__init__(*args, **kwargs)

  def _launch(self, run_config, args, **kwargs):
    return super()._launch(run_config, args, **launch_kwargs(kwargs, self.minimized))


def install_launch_shim(minimized: bool, window: tuple[int, int] = WINDOW) -> None:
  """Route pysc2's launches through _Process. run_configs looks the class up on the
  sc_process module at launch time, so replacing the attribute is enough."""
  _Process.minimized, _Process.window_size = minimized, window
  sc_process.StarcraftProcess = _Process


def use_flag_defaults() -> None:
  """pysc2 reads absl flags (its run config, SC2's paths) that only absl's app.run parses."""
  if not flags.FLAGS.is_parsed():
    flags.FLAGS.mark_as_parsed()


def minimize_windows(pid: int, user32=None) -> int:
  """Minimize pid's visible windows without activating them; return how many changed.
  The fallback for an SC2 that ignores the STARTUPINFO show flag."""
  user32 = user32 or ctypes.windll.user32
  functype = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)
  owner = ctypes.c_ulong()
  changed = []

  @functype(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
  def visit(hwnd, _):
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
    if owner.value == pid and user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
      user32.ShowWindow(hwnd, SW_SHOWMINNOACTIVE)
      changed.append(hwnd)
    return True

  user32.EnumWindows(visit, 0)
  return len(changed)


def check_built(map_name: str) -> None:
  """A map flycraft builds must be there before SC2 starts: pysc2 reads the map only once SC2
  is up, and says only that it is not found."""
  map_inst = maps.get(map_name)
  if map_inst.directory != BUILT:
    return
  data_dir = run_configs.get().data_dir  # where pysc2 will read it
  path = Path(data_dir) / "Maps" / map_inst.path
  if not path.is_file():
    raise SetupError(f"no {map_name} map at {path}: build it with "
                     f'flycraft-maps --sc2path "{data_dir}"')


def _interface(screen: int) -> features.AgentInterfaceFormat:
  return features.AgentInterfaceFormat(
    feature_dimensions=features.Dimensions(screen=screen, minimap=MINIMAP))


class SC2Game:
  """The client's Game on real SC2 (spec 7.5): step_mul 1 here, the client passes its own."""

  match = False  # a LAN match, scored by its outcome

  def __init__(self, map_name: str, screen: int, mode: str, run_seed: int,
               realtime: bool = False, window: tuple[int, int] = WINDOW, log=_log,
               race: str = "terran"):
    minimized = mode != "watch"
    use_flag_defaults()
    install_launch_shim(minimized, window)
    try:
      check_built(map_name)
      self.env = sc2_env.SC2Env(
        map_name=map_name, players=[sc2_env.Agent(sc2_env.Race[race])],
        agent_interface_format=_interface(screen), step_mul=1, realtime=realtime,
        random_seed=run_seed, visualize=False)
    except SC2_ERRORS as e:
      raise GameError(f"SC2 did not start: {e}") from e
    except (maps.lib.NoMapError, ValueError) as e:  # a bad map name or setting: fatal
      raise SetupError(f"SC2 cannot start as configured: {e}") from e
    self._available = ()
    if minimized:
      self._minimize(log)

  def _minimize(self, log) -> None:
    if sys.platform == "win32":
      n = minimize_windows(self.env._sc2_procs[0].pid)
      log(f"SC2 window: {n} minimized by pid" if n
          else "SC2 window: none found to minimize, so it may be showing")

  def _take(self, call) -> Frame:
    try:
      (ts,) = call()
    except PLAY_ERRORS as e:
      raise GameError(f"SC2 failed: {e}") from e
    self._available = ts.observation["available_actions"]
    self.player = int(ts.observation["player"][features.Player.player_id])
    return to_frame(ts, self.match)

  def reset(self) -> Frame:
    return self._take(self.env.reset)

  def step(self, action: Action, step_mul: int) -> Frame:
    call = to_call(action, self._available)
    return self._take(lambda: self.env.step([call], step_mul=step_mul))

  def close(self) -> None:
    try:
      self.env.close()
    except Exception as e:  # best effort: the run is ending or relaunching anyway
      _log(f"SC2 did not close cleanly: {e}")


class _LanEnv(lan_sc2_env.LanSC2Env):
  """pysc2 4.0's LanSC2Env, mended: it calls SC2Env's interface helper by an argument name the
  helper no longer has, and SC2Env's reset reads the players' races, which only SC2Env's own
  launch records."""

  @staticmethod
  def _get_interface(agent_interface_format, require_raw):
    return sc2_env.SC2Env._get_interface(agent_interface_format, require_raw)

  def _launch_remote(self, *args):
    super()._launch_remote(*args)
    self._requested_races = {p.player_id: p.race_requested
                             for p in self._game_info[0].player_info
                             if p.type != sc_pb.Observer}


class SC2LanGame(SC2Game):
  """The fly's side of a Skirmish match (spec M6): LanSC2Env joins the game that
  flycraft-match hosts on port, as player 2, in real time, with SC2 minimized. A match is one
  episode; LanSC2Env cannot restart, so the client plays it once."""

  match = True

  def __init__(self, port: int, screen: int, window: tuple[int, int] = WINDOW, log=_log,
               race: str = "zerg"):
    use_flag_defaults()
    install_launch_shim(True, window)
    log(f"joining the match on port {port}; flycraft-match must be waiting there")
    try:
      self.env = _LanEnv(
        host=LAN_HOST, config_port=port, race=sc2_env.Race[race], name="fly",
        agent_interface_format=_interface(screen), step_mul=1, realtime=True)
    except SC2_ERRORS as e:
      raise GameError(f"could not join the match on port {port}: {e}") from e
    except ValueError as e:  # an SC2 version this pysc2 does not know, say
      raise SetupError(f"cannot join the match as configured: {e}") from e
    self._available = ()
    self._minimize(log)

  def reset(self) -> Frame:
    frame = super().reset()
    if self.player != FLY:
      raise SetupError(f"the fly joined the match as player {self.player}, not {FLY}: the "
                       "person's flycraft-match joins first, and a match takes one fly")
    return frame
