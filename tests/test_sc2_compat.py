import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pysc2")

from absl import flags  # noqa: E402
from pysc2 import maps  # noqa: E402
from pysc2.lib import actions, features, protocol, sc_process  # noqa: E402

from flycraft.game import sc2_compat  # noqa: E402
from flycraft.game.body import NOOP, SELECT, Action  # noqa: E402
from flycraft.game.client import WINDOW, GameError, SetupError  # noqa: E402

F = actions.FUNCTIONS
ALL = [F.no_op.id, F.select_army.id, F.Move_screen.id, F.Attack_screen.id, F.Stop_quick.id]


def call(fc):
  return int(fc.function), fc.arguments


@pytest.mark.parametrize("action, available, expected", [
  (Action("move", (10.4, 20.6)), ALL, (F.Move_screen.id, [[0], [10, 21]])),
  (Action("move", (10.0, 20.0)), [F.no_op.id, F.select_army.id], (F.select_army.id, [[0]])),
  (Action("move", (10.0, 20.0)), [F.no_op.id], (F.no_op.id, [])),
  (Action("attack", (10.4, 20.6)), ALL, (F.Attack_screen.id, [[0], [10, 21]])),
  (Action("attack", (10.0, 20.0)), [F.no_op.id, F.select_army.id], (F.select_army.id, [[0]])),
  (SELECT, ALL, (F.select_army.id, [[0]])),
  (SELECT, [F.no_op.id], (F.no_op.id, [])),
  (Action("stop"), ALL, (F.Stop_quick.id, [[0]])),
  (Action("stop"), [F.no_op.id], (F.no_op.id, [])),
  (NOOP, ALL, (F.no_op.id, [])),
])
def test_to_call(action, available, expected):
  assert call(sc2_compat.to_call(action, np.array(available))) == expected


def timestep(last=False, reward=0, available=ALL, army=1, selected=1):
  screen = np.zeros((len(features.SCREEN_FEATURES), 84, 84), np.int32)
  screen[sc2_compat.PLAYER_RELATIVE, 40, 41] = features.PlayerRelative.SELF
  player = np.zeros(len(features.Player), np.int32)
  player[features.Player.army_count] = army
  units = np.zeros((selected, 7), np.int32)  # pysc2: one unit in single_select, more in multi
  single, multi = (units, units[:0]) if selected == 1 else (units[:0], units)
  obs = {"feature_screen": screen, "score_cumulative": np.array([7, 0]),
         "game_loop": np.array([96]), "available_actions": np.array(available),
         "player": player, "single_select": single, "multi_select": multi}
  step_type = 2 if last else 1
  return types.SimpleNamespace(observation=obs, reward=reward, last=lambda: step_type == 2)


def test_to_frame():
  frame = sc2_compat.to_frame(timestep(last=True, reward=1))
  assert frame.player_relative.shape == (84, 84)
  assert frame.player_relative[40, 41] == features.PlayerRelative.SELF
  assert (frame.reward, frame.score, frame.last, frame.loop, frame.can_move) == (
    1.0, 7.0, True, 96, True)
  assert sc2_compat.to_frame(timestep(available=[F.no_op.id, F.select_army.id])).can_move is False


@pytest.mark.parametrize("army, selected, can_move", [
  (1, 1, True), (9, 9, True), (14, 9, False), (5, 9, True), (0, 0, True), (1, 0, False)])
def test_orders_wait_until_the_whole_army_is_selected(army, selected, can_move):
  assert sc2_compat.to_frame(timestep(army=army, selected=selected)).can_move is can_move


class Info:
  dwFlags = 0x100
  wShowWindow = 0


def test_launch_kwargs_drop_extra_ports():
  kw = {"extra_ports": [1, 2], "stdout": None}
  assert sc2_compat.launch_kwargs(kw, minimized=False) == {"stdout": None}
  assert kw == {"extra_ports": [1, 2], "stdout": None}  # the caller's dict is untouched


def test_launch_kwargs_shown_without_focus():
  """SC2 started minimized has no graphics device and quits, so train mode starts it shown
  but not activated (SW_SHOWNOACTIVATE) and SC2Game minimizes it afterwards."""
  out = sc2_compat.launch_kwargs({"extra_ports": [1]}, minimized=True, startupinfo_cls=Info)
  info = out.pop("startupinfo")
  assert out == {}
  assert info.dwFlags == 0x101 and info.wShowWindow == 4


def test_the_shim_launches_without_extra_ports(monkeypatch):
  seen = {}

  def popen(args, **kwargs):
    seen.update(kwargs, args=args)
    return "proc"

  monkeypatch.setattr(sc_process.subprocess, "Popen", popen)
  proc = sc2_compat._Process.__new__(sc2_compat._Process)
  proc._proc = None  # never started, so __del__ has nothing to close
  run_config = types.SimpleNamespace(cwd="/sc2", env={})
  assert proc._launch(run_config, ["SC2.exe"], extra_ports=[5, 6]) == "proc"
  assert seen == {"args": ["SC2.exe"], "cwd": "/sc2", "env": {}}


def test_install_launch_shim(monkeypatch):
  monkeypatch.setattr(sc_process, "StarcraftProcess", sc_process.StarcraftProcess)
  sc2_compat.install_launch_shim(minimized=True, window=(800, 600))
  assert sc_process.StarcraftProcess is sc2_compat._Process
  assert (sc2_compat._Process.minimized, sc2_compat._Process.window_size) == (True, (800, 600))
  sc2_compat.install_launch_shim(minimized=False)
  assert (sc2_compat._Process.minimized, sc2_compat._Process.window_size) == (False, WINDOW)


class User32:
  """Windows by hwnd: (pid, visible, iconic)."""

  def __init__(self, windows):
    self.windows = windows
    self.shown = []

  def EnumWindows(self, visit, _):
    for hwnd in self.windows:
      visit(hwnd, 0)

  def GetWindowThreadProcessId(self, hwnd, ref):
    ref._obj.value = self.windows[hwnd][0]

  def IsWindowVisible(self, hwnd):
    return self.windows[hwnd][1]

  def IsIconic(self, hwnd):
    return self.windows[hwnd][2]

  def ShowWindow(self, hwnd, cmd):
    self.shown.append((hwnd, cmd))


def test_minimize_windows_touches_only_our_visible_windows():
  user32 = User32({1: (42, True, False), 2: (42, False, False), 3: (42, True, True),
                   4: (7, True, False), 5: (42, True, False)})
  assert sc2_compat.minimize_windows(42, user32) == 2
  assert user32.shown == [(1, 7), (5, 7)]


class FakeEnv:
  """SC2Env's surface. fail=(method, exception) makes that method raise."""
  made = []

  def __init__(self, fail=(None, None), **kwargs):
    flags.FLAGS.sc2_run_config  # noqa: B018 -- the real SC2Env reads it via run_configs.get()
    self.kwargs, self.steps, self.closed = kwargs, [], False
    self.fail_in, self.error = fail
    FakeEnv.made.append(self)
    self._maybe_fail("init")

  def _maybe_fail(self, where):
    if self.fail_in == where:
      raise self.error

  def reset(self):
    self._maybe_fail("reset")
    return (timestep(available=[F.no_op.id, F.select_army.id]),)

  def step(self, calls, step_mul=None):
    self.steps.append((call(calls[0]), step_mul))
    self._maybe_fail("step")
    return (timestep(),)

  def close(self):
    self.closed = True
    self._maybe_fail("close")


@pytest.fixture
def env(monkeypatch):
  FakeEnv.made = []
  monkeypatch.setattr(sc_process, "StarcraftProcess", sc_process.StarcraftProcess)
  monkeypatch.setattr(sc2_compat.sc2_env, "SC2Env", FakeEnv)
  return FakeEnv


def test_sc2game_drives_the_env(env):
  game = sc2_compat.SC2Game("MoveToBeacon", 84, "watch", run_seed=3, window=(1600, 1200))
  (made,) = env.made
  kw = made.kwargs
  assert (kw["map_name"], kw["step_mul"], kw["random_seed"], kw["realtime"]) == (
    "MoveToBeacon", 1, 3, False)
  assert kw["agent_interface_format"].feature_dimensions.screen == (84, 84)
  assert sc_process.StarcraftProcess is sc2_compat._Process
  assert (sc2_compat._Process.minimized, sc2_compat._Process.window_size) == (
    False, (1600, 1200))
  assert game.reset().can_move is False
  frame = game.step(Action("move", (30.0, 31.0)), 8)  # Move_screen is not available yet
  assert made.steps == [((F.select_army.id, [[0]]), 8)]
  assert frame.can_move is True
  game.step(Action("move", (30.0, 31.0)), 8)
  assert made.steps[-1] == ((F.Move_screen.id, [[0], [30, 31]]), 8)
  game.close()
  assert made.closed


def test_train_mode_launches_minimized(env):
  sc2_compat.SC2Game("MoveToBeacon", 84, "train", run_seed=0)
  assert sc2_compat._Process.minimized is True


@pytest.mark.parametrize(("mode", "found", "said"), [
  ("train", 1, "SC2 window: 1 minimized by pid"),
  ("eval", 0, "SC2 window: none found to minimize, so it may be showing"),
  ("watch", 1, None),
])
def test_sc2_is_minimized_by_pid_after_launch(env, monkeypatch, mode, found, said):
  seen, logged = [], []
  monkeypatch.setattr(sc2_compat.sys, "platform", "win32")
  monkeypatch.setattr(sc2_compat, "minimize_windows", lambda pid: seen.append(pid) or found)
  monkeypatch.setattr(env, "_sc2_procs", [types.SimpleNamespace(pid=77)], raising=False)
  sc2_compat.SC2Game("MoveToBeacon", 84, mode, run_seed=0, log=logged.append)
  assert (seen, logged) == (([77], [said]) if said else ([], []))


def test_sc2game_parses_pysc2s_flags(env):
  """The client never runs absl's app.run, and pysc2 reads its flags while launching."""
  flags.FLAGS.unparse_flags()
  sc2_compat.SC2Game("MoveToBeacon", 84, "eval", run_seed=0)
  assert flags.FLAGS.is_parsed()


@pytest.mark.parametrize("fail", [
  ("reset", protocol.ConnectionError("gone")),
  ("step", protocol.ProtocolError("bad")),
  ("step", ValueError("The game didn't advance to the expected game loop.")),
])
def test_sc2_errors_become_game_errors(env, monkeypatch, fail):
  monkeypatch.setattr(sc2_compat.sc2_env, "SC2Env", lambda **kw: FakeEnv(fail=fail, **kw))
  game = sc2_compat.SC2Game("MoveToBeacon", 84, "eval", run_seed=0)
  with pytest.raises(GameError):
    game.reset()
    game.step(NOOP, 8)


@pytest.mark.parametrize("error", [maps.lib.NoMapError("Map doesn't exist: MoveToBecon"),
                                   ValueError("Unknown game version: 9.9")])
def test_a_bad_setting_is_a_setup_error(env, monkeypatch, error):
  monkeypatch.setattr(sc2_compat.sc2_env, "SC2Env",
                      lambda **kw: FakeEnv(fail=("init", error), **kw))
  with pytest.raises(SetupError, match="cannot start as configured"):
    sc2_compat.SC2Game("MoveToBecon", 84, "watch", run_seed=0)


def test_close_never_raises(env, monkeypatch):
  monkeypatch.setattr(sc2_compat.sc2_env, "SC2Env",
                      lambda **kw: FakeEnv(fail=("close", RuntimeError("stuck")), **kw))
  game = sc2_compat.SC2Game("MoveToBeacon", 84, "eval", run_seed=0)
  game.close()
  assert env.made[0].closed


def test_a_failed_launch_is_a_game_error(env, monkeypatch):
  def boom(**_):
    raise sc_process.SC2LaunchError("no SC2")

  monkeypatch.setattr(sc2_compat.sc2_env, "SC2Env", boom)
  with pytest.raises(GameError, match="did not start"):
    sc2_compat.SC2Game("MoveToBeacon", 84, "watch", run_seed=0)


def test_importing_pysc2_prints_nothing_to_stdout():
  """The client's stdout carries only episode records, and pygame (which pysc2 imports)
  greets on stdout unless told not to."""
  env = {k: v for k, v in os.environ.items() if k != "PYGAME_HIDE_SUPPORT_PROMPT"}
  out = subprocess.run([sys.executable, "-c", "import flycraft.game.sc2_compat"],
                       capture_output=True, text=True, env=env, check=True,
                       cwd=Path(__file__).resolve().parents[1])
  assert out.stdout == ""
