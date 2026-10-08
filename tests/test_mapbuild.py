import io

import pytest

from flycraft.game import mapbuild, mpq
from flycraft.game.mapbuild import INFO, PRELOAD, SCRIPT, STRINGS, MapBuildError

mpyq = pytest.importorskip("mpyq")  # pysc2's MPQ reader, on the game side

BOM = "\ufeff"


def source() -> dict[str, bytes]:
  """A small made-up DefeatRoaches with each thing the build changes, and things it must not."""
  return {
    SCRIPT: (b'CreateUnits(1, "Marine", 0, 1, p);\nCreateUnits(1, "Roach", 0, 2, q);\n'
             b'if (UnitGetType(u) == "Marine") { }\n// Marines vs Roaches\n'),
    PRELOAD: b'<Preload>\n<Unit id="Marine"/>\n<Unit id="Roach"/>\n</Preload>\n',
    INFO: (b"IpaM\x00\x00\x00\x00\x01\x00\x00\x00Terr\x00\x00\x00\x00"
           b"\x02\x00\x00\x00Zerg\x00\x00\x00\x00Decal_Zerg_0017_01\x00"),
    STRINGS: (BOM + "DocInfo/Name=CombatFocus\r\nMapInfo/Player02/Name=Zerg\r\n"
              "Param/Value/1=Defeat Roaches\r\nParam/Value/2=Roaches Defeated\r\n").encode(),
    "Triggers": b'<Value>Roach</Value><Value>Marine</Value>',
    "t3HeightMap": bytes(range(256)),
  }


def test_the_squad_becomes_zerglings_and_the_enemy_marines():
  src = source()
  out = mapbuild.defeat_marines(src)
  assert out[SCRIPT].decode() == (
    'CreateUnits(1, "Zergling", 0, 1, p);\nCreateUnits(1, "Marine", 0, 2, q);\n'
    'if (UnitGetType(u) == "Zergling") { }\n// Marines vs Roaches\n')
  assert out[PRELOAD] == b'<Preload>\n<Unit id="Zergling"/>\n<Unit id="Marine"/>\n</Preload>\n'
  assert src == source()  # the source is not changed


def test_player_one_is_zerg_and_player_two_terran():
  info = mapbuild.defeat_marines(source())[INFO]
  assert info == (b"IpaM\x00\x00\x00\x00\x01\x00\x00\x00Zerg\x00\x00\x00\x00"
                  b"\x02\x00\x00\x00Terr\x00\x00\x00\x00Decal_Zerg_0017_01\x00")


def test_the_strings_name_the_new_map():
  text = mapbuild.defeat_marines(source())[STRINGS]
  assert text.startswith(BOM.encode())
  assert text.decode("utf-8-sig") == (
    "DocInfo/Name=CombatFocus\r\nMapInfo/Player02/Name=Terran\r\n"
    "Param/Value/1=Defeat Marines\r\nParam/Value/2=Marines Defeated\r\n")


def test_strings_without_a_bom_stay_without_one():
  files = _without(STRINGS, BOM.encode(), b"")
  assert not mapbuild.defeat_marines(files)[STRINGS].startswith(BOM.encode())


def test_nothing_else_changes():
  src, out = source(), mapbuild.defeat_marines(source())
  assert set(out) == set(src)
  assert out["Triggers"] == src["Triggers"] and out["t3HeightMap"] == src["t3HeightMap"]


def _without(name, old, new):
  files = source()
  files[name] = files[name].replace(old, new)
  return files


@pytest.mark.parametrize(("files", "says"), [
  (_without(SCRIPT, b'"Roach"', b'"Hydralisk"'), '"Roach"'),
  (_without(SCRIPT, b'"Marine"', b'"Reaper"'), '"Marine"'),
  (_without(SCRIPT, b"// Marines", b'"Zergling"'), "already names"),
  (_without(PRELOAD, b'"Roach"', b'"Hydralisk"'), "Preload.xml"),
  (_without(INFO, b"Terr\x00", b"Prot\x00"), "MapInfo"),
  (_without(INFO, b"Decal_Zerg_", b"Zerg\x00Zerg"), "MapInfo"),  # two Zerg race fields
  (_without(STRINGS, b"Defeat Roaches", b"Defeat Banelings"), "Defeat Roaches"),
  (_without(STRINGS, b"Player02/Name=Zerg", b"Player02/Name=Swarm"), "Player02"),
  (_without(STRINGS, b"DocInfo/Name=CombatFocus", b"Tip/1=Defeat Roaches fast"),
   "2 'Defeat Roaches'"),  # renamed once, so no other text changes
  ({k: v for k, v in source().items() if k != PRELOAD}, "Preload.xml"),
])
def test_a_source_it_does_not_recognize_stops_the_build(files, says):
  with pytest.raises(MapBuildError, match=says.replace('"', '.')):
    mapbuild.defeat_marines(files)


def test_build_writes_the_map_under_the_sc2_install(tmp_path):
  (tmp_path / "Maps" / "mini_games").mkdir(parents=True)
  (tmp_path / mapbuild.SOURCE).write_bytes(mpq.pack(source()))
  out = mapbuild.build(tmp_path)
  assert out == tmp_path / "Maps" / "flycraft" / "DefeatMarines.SC2Map"
  archive = mpyq.MPQArchive(io.BytesIO(out.read_bytes()), listfile=True)
  for name, data in mapbuild.defeat_marines(source()).items():
    assert archive.read_file(name) == data, name
  assert mapbuild.build(tmp_path) == out  # building again replaces it
  assert [p.name for p in out.parent.iterdir()] == ["DefeatMarines.SC2Map"]


def test_build_says_where_it_looked(tmp_path):
  with pytest.raises(MapBuildError, match="DefeatRoaches.SC2Map"):
    mapbuild.build(tmp_path)
  (tmp_path / "Maps" / "mini_games").mkdir(parents=True)
  (tmp_path / mapbuild.SOURCE).write_bytes(b"not an archive")
  with pytest.raises(MapBuildError, match="cannot read"):
    mapbuild.build(tmp_path)


def test_main(tmp_path, capsys, monkeypatch):
  assert mapbuild.main(["--sc2path", str(tmp_path)]) == 1
  assert "flycraft-maps: " in capsys.readouterr().err
  (tmp_path / "Maps" / "mini_games").mkdir(parents=True)
  (tmp_path / mapbuild.SOURCE).write_bytes(mpq.pack(source()))
  monkeypatch.setenv("SC2PATH", str(tmp_path))
  assert mapbuild.main([]) == 0
  assert "DefeatMarines.SC2Map" in capsys.readouterr().out


def test_the_default_sc2_install(monkeypatch):
  monkeypatch.delenv("SC2PATH", raising=False)
  assert str(mapbuild.default_sc2path()).replace("\\", "/") == "C:/Program Files (x86)/StarCraft II"
  monkeypatch.setenv("SC2PATH", "/games/sc2")
  assert str(mapbuild.default_sc2path()) == "/games/sc2"
