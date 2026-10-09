import io
import json
import threading
import types
from pathlib import Path

import pytest

pytest.importorskip("pysc2")

from pysc2.lib import protocol, sc_process  # noqa: E402
from s2clientprotocol import sc2api_pb2 as sc_pb  # noqa: E402

from flycraft.game import match, sc2_compat  # noqa: E402
from flycraft.game.client import GameError, SetupError  # noqa: E402


def observation(loop, result=None, rounds=0):
  obs = sc_pb.ResponseObservation()
  obs.observation.game_loop = loop
  obs.observation.score.score = rounds
  for player_id, r in result or ():
    obs.player_result.add(player_id=player_id, result=r)
  return obs


class FakeController:
  """The person's SC2. join_game returns once the fly is in, as SC2's does."""

  def __init__(self, sc2, player_id=1, join_error=None, results=None):
    self.sc2, self.player_id, self.join_error = sc2, player_id, join_error
    self.results = results or [(1, sc_pb.Victory), (2, sc_pb.Defeat)]
    self.observed = 0

  def save_map(self, path, data):
    self.sc2.calls.append(("save_map", path, data))

  def create_game(self, create):
    self.sc2.calls.append(("create_game", create))

  def join_game(self, join):
    self.sc2.calls.append(("join_game", join))
    self.sc2.joining.set()
    if self.join_error is not None:
      raise self.join_error
    if not self.sc2.fly_in.wait(5):
      raise protocol.ConnectionError("Websocket timed out")
    return sc_pb.ResponseJoinGame(player_id=self.player_id)

  def observe(self):
    self.observed += 1
    return observation(self.observed * 22, self.results if self.observed == 3 else None, 2)


class FakeSC2:
  """pysc2's run config, its SC2 process, and the tcp server that lets the fly in."""

  def __init__(self, tmp_path, fly_connects=True, door_error=None, **controller):
    self.data_dir = str(tmp_path)
    built = tmp_path / "Maps" / "flycraft" / "Skirmish.SC2Map"
    built.parent.mkdir(parents=True)
    built.write_bytes(b"skirmish")
    self.calls, self.closed = [], []
    self.joining, self.fly_in, self.stop = threading.Event(), threading.Event(), threading.Event()
    self.fly_connects, self.door_error = fly_connects, door_error
    self.controller = FakeController(self, **controller)

  def map_data(self, path, players=None):
    return (Path(self.data_dir) / "Maps" / path).read_bytes()

  def start(self, **kwargs):
    self.calls.append(("start", kwargs))
    range(kwargs["timeout_seconds"])  # pysc2 counts the seconds it waits for SC2 this way
    return types.SimpleNamespace(controller=self.controller,
                                 version=types.SimpleNamespace(game_version="5.0.14"),
                                 close=lambda: self.closed.append("sc2"))

  def tcp_server(self, addr, settings):
    # the person's join must be under way: one that only starts after the fly is in never is
    self.calls.append(("tcp_server", tuple(addr), settings, self.joining.wait(1)))
    if self.door_error is not None:
      raise self.door_error
    if not self.fly_connects:
      self.stop.wait(5)
      raise OSError("closed")
    self.fly_in.set()
    return types.SimpleNamespace(close=lambda: self.closed.append("conn"))

  def call(self, name):
    (found,) = [c for c in self.calls if c[0] == name]
    return found


@pytest.fixture
def sc2(monkeypatch, tmp_path):
  def make(**kw):
    fake = FakeSC2(tmp_path, **kw)
    monkeypatch.setattr(sc2_compat.run_configs, "get", lambda: fake)
    monkeypatch.setattr(match.lan_sc2_env, "tcp_server", fake.tcp_server)
    monkeypatch.setattr(match, "POLL_S", 0.0)
    return fake

  monkeypatch.setattr(sc_process, "StarcraftProcess", sc_process.StarcraftProcess)
  monkeypatch.setattr(match.portpicker, "is_port_free", lambda port: True)
  return make


def host(fake, **kw):
  out, said = io.StringIO(), []
  try:
    code = match.host(14380, "BJ", (1600, 1200), wait_s=300, end_s=0, out=out,
                      log=said.append, **kw)
  finally:
    fake.stop.set()
  return code, [json.loads(line) for line in out.getvalue().splitlines()], said


def test_the_person_hosts_a_match_and_sees_the_result(sc2):
  fake = sc2()
  seen = []
  code, recs, said = host(fake, on_obs=lambda c, obs: seen.append(obs.observation.game_loop))
  assert code == 0
  assert recs == [{"map": "Skirmish", "player": 1, "result": "Victory", "rounds_won": 2,
                   "game_loop": 66, "wall_s": recs[0]["wall_s"]}]
  assert seen == [22, 44, 66]
  assert sc2_compat._Process.minimized is False  # the person's SC2 shows
  assert sc2_compat._Process.window_size == (1600, 1200)
  _, start = fake.call("start")
  assert start == {"extra_ports": [14381, 14382, 14383, 14384], "timeout_seconds": 300,
                   "host": "127.0.0.1", "window_loc": (50, 50)}
  assert fake.call("save_map")[1:] == ("flycraft/Skirmish.SC2Map", b"skirmish")
  create = fake.call("create_game")[1]
  assert create.realtime and create.local_map.map_path == "flycraft/Skirmish.SC2Map"
  assert [p.type for p in create.player_setup] == [sc_pb.Participant, sc_pb.Participant]
  join = fake.call("join_game")[1]
  assert (join.race, join.player_name, join.options.raw, join.options.score) == (
    1, "BJ", True, True)
  assert (join.server_ports.game_port, join.server_ports.base_port) == (14381, 14382)
  assert [(p.game_port, p.base_port) for p in join.client_ports] == [(14383, 14384)]
  _, addr, settings, joining = fake.call("tcp_server")
  assert addr == ("127.0.0.1", 14380)
  assert joining  # the person's join was under way before the fly could connect
  assert settings == {"remote": False, "game_version": "5.0.14", "realtime": True,
                      "map_name": "Skirmish", "map_path": "flycraft/Skirmish.SC2Map",
                      "map_data": b"skirmish",
                      "ports": {"server": {"game": 14381, "base": 14382},
                                "client": {"game": 14383, "base": 14384}}}
  assert [c[0] for c in fake.calls].index("save_map") < [c[0] for c in fake.calls].index(
    "create_game")
  assert sorted(fake.closed) == ["conn", "sc2"]
  assert any("--join 14380" in line for line in said)


def test_a_busy_port_stops_the_match_before_sc2_starts(sc2, monkeypatch):
  fake = sc2()
  monkeypatch.setattr(match.portpicker, "is_port_free", lambda port: port != 14382)
  with pytest.raises(SetupError, match="port 14382 is in use"):
    host(fake)
  assert fake.calls == []


def test_the_fly_not_joining_is_said_plainly(sc2):
  fake = sc2(fly_connects=False, join_error=protocol.ConnectionError("Websocket timed out"))
  with pytest.raises(GameError, match="the fly did not join within 300 s"):
    host(fake)
  assert fake.closed == ["sc2"]


def test_a_door_that_cannot_open_stops_the_wait(sc2):
  fake = sc2(door_error=OSError(10048, "Only one usage of each socket address"))
  with pytest.raises(GameError, match="could not wait for the fly on port 14380"):
    host(fake)
  assert fake.closed == ["sc2"]


def test_the_person_must_be_player_1(sc2):
  fake = sc2(player_id=2)
  with pytest.raises(SetupError, match="as player 2, not 1"):
    host(fake)
  assert sorted(fake.closed) == ["conn", "sc2"]


def test_a_game_with_no_result_for_the_person_fails(sc2):
  fake = sc2(results=[(2, sc_pb.Victory)])
  code, recs, _ = host(fake)
  assert code == 1 and recs[0]["result"] is None


@pytest.mark.parametrize("argv", [["--port", "1023"], ["--port", "65532"], ["--wait", "0"],
                                  ["--wait", "2.5"], ["--window", "0x9"]])
def test_main_rejects_bad_arguments(argv):
  with pytest.raises(SystemExit) as e:
    match.main(argv)
  assert e.value.code == 2


@pytest.mark.parametrize("argv,wait", [([], 300), (["--wait", "30"], 30)])
def test_main_waits_whole_seconds(monkeypatch, argv, wait):
  seen = []
  monkeypatch.setattr(match, "host", lambda *args: seen.append(args[3]) or 0)
  assert match.main(argv) == 0
  assert seen == [wait] and type(seen[0]) is int


def test_main_says_what_went_wrong(monkeypatch, capsys):
  def fail(*args):
    raise SetupError("no Skirmish map: build it with flycraft-maps")

  monkeypatch.setattr(match, "host", fail)
  assert match.main([]) == 1
  assert "build it with flycraft-maps" in capsys.readouterr().err
