import io
import re
from pathlib import Path

import pytest

from flycraft.game import mapbuild, mpq
from flycraft.game.mapbuild import HEADER, INFO, PRELOAD, SCRIPT, STRINGS, MapBuildError

mpyq = pytest.importorskip("mpyq")  # pysc2's MPQ reader, on the game side

BOM = "\ufeff"
# DefeatRoaches' players as MapInfo holds them: id, control (1 User, 2 Computer), color, race.
P1 = b"\x01" + b"\x01\x00\x00\x00" + b"\x02\x00\x00\x00" + b"Terr\x00\x00\x00\x00"
P2 = b"\x02" + b"\x02\x00\x00\x00" + b"\x01\x00\x00\x00" + b"Zerg\x00\x00\x00\x00"


def source() -> dict[str, bytes]:
  """A small made-up DefeatRoaches with each thing the build changes, and things it must not."""
  return {
    SCRIPT: (b'CreateUnits(1, "Marine", 0, 1, p);\nCreateUnits(1, "Roach", 0, 2, q);\n'
             b'if (UnitGetType(u) == "Marine") { }\n// Marines vs Roaches\n'),
    PRELOAD: b'<Preload>\n<Unit id="Marine"/>\n<Unit id="Roach"/>\n</Preload>\n',
    INFO: b"IpaM\x27\x00\x00\x00" + P1 + P2 + b"Decal_Zerg_0017_01\x00",
    STRINGS: (BOM + "DocInfo/Name=CombatFocus\r\nMapInfo/Player02/Name=Zerg\r\n"
              "Param/Value/1=Defeat Roaches\r\nParam/Value/2=Roaches Defeated\r\n").encode(),
    HEADER: (b"H2CS\x08\x00\x00\x00\x03\x00\x00\x00"  # 3 strings; the header's first one:
             b"\x0c\x00DocInfo/NameSUne\x0b\x00CombatFocus"
             b"\x15\x00MapInfo/Player03/NameSUne\x04\x00Zerg"  # another player's Zerg stays
             b"\x15\x00MapInfo/Player02/NameSUne\x04\x00Zerg"),
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
  assert info == (b"IpaM\x27\x00\x00\x00" + P1.replace(b"Terr", b"Zerg")
                  + P2.replace(b"Zerg", b"Terr") + b"Decal_Zerg_0017_01\x00")


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


def test_skirmish_gives_player_two_to_a_second_client():
  info = mapbuild.skirmish(source())[INFO]
  user = P2[:1] + b"\x01" + P2[2:]  # control 1, User; id, color and race as they were
  assert info == b"IpaM\x27\x00\x00\x00" + P1 + user + b"Decal_Zerg_0017_01\x00"


def test_skirmish_preloads_zerglings_and_takes_its_own_name():
  out = mapbuild.skirmish(source())
  assert out[PRELOAD] == b'<Preload>\n<Unit id="Marine"/>\n<Unit id="Zergling"/>\n</Preload>\n'
  assert out[STRINGS] == (BOM + "DocInfo/Name=Skirmish\r\nMapInfo/Player02/Name=Fly\r\n"
                          "Param/Value/1=Defeat Roaches\r\nParam/Value/2=Roaches Defeated\r\n"
                          ).encode()


def test_skirmish_renames_the_map_in_its_header():
  assert mapbuild.skirmish(source())[HEADER] == (
    b"H2CS\x08\x00\x00\x00\x03\x00\x00\x00\x0c\x00DocInfo/NameSUne\x08\x00Skirmish"
    b"\x15\x00MapInfo/Player03/NameSUne\x04\x00Zerg"
    b"\x15\x00MapInfo/Player02/NameSUne\x03\x00Fly")
  assert mapbuild.defeat_marines(source())[HEADER] == source()[HEADER]


def test_skirmish_runs_flycrafts_script_and_changes_nothing_else():
  src, out = source(), mapbuild.skirmish(source())
  assert out[SCRIPT] == mapbuild.SKIRMISH_SCRIPT.read_bytes()
  assert set(out) == set(src) and src == source()
  assert out["Triggers"] == src["Triggers"] and out["t3HeightMap"] == src["t3HeightMap"]


@pytest.mark.parametrize(("files", "says"), [
  (_without(INFO, P2, P2[:1] + b"\x01" + P2[2:]), "not a Zerg computer"),  # already a User
  (_without(INFO, P2, b"\x03" + P2[1:]), "not a Zerg computer"),  # the Zerg is player 3
  (_without(INFO, b"Zerg\x00\x00", b"Prot\x00\x00"), "0 Zerg race fields"),
  (_without(INFO, b"Decal_Zerg_", b"Zerg\x00Zerg"), "2 Zerg race fields"),
  (_without(INFO, b"IpaM\x27\x00\x00\x00" + P1 + b"\x02", b""), "not a Zerg computer"),
  (_without(PRELOAD, b'"Marine"', b'"Zergling"'), "already names"),
  (_without(PRELOAD, b'"Roach"', b'"Hydralisk"'), '"Roach"'),
  (_without(STRINGS, b"DocInfo/Name=CombatFocus", b"DocInfo/Name=Other"), "CombatFocus"),
  (_without(HEADER, b"\x0b\x00CombatFocus", b"\x0b\x00CombatFocal"), "DocInfo/Name=Comb"),
  (_without(HEADER, b"\x0b\x00CombatFocus", b"\x0c\x00CombatFocus!"), "DocInfo/Name=Comb"),
  ({k: v for k, v in source().items() if k != HEADER}, "DocumentHeader"),
  ({k: v for k, v in source().items() if k != SCRIPT}, "MapScript.galaxy"),
])
def test_a_source_skirmish_does_not_recognize_stops_the_build(files, says):
  with pytest.raises(MapBuildError, match=says.replace('"', '.')):
    mapbuild.skirmish(files)


def _install(tmp_path):
  (tmp_path / "Maps" / "mini_games").mkdir(parents=True)
  (tmp_path / mapbuild.SOURCE).write_bytes(mpq.pack(source()))


@pytest.mark.parametrize("name", ["DefeatMarines", "Skirmish"])
def test_build_writes_the_map_under_the_sc2_install(tmp_path, name):
  _install(tmp_path)
  out = mapbuild.build(tmp_path, name)
  assert out == tmp_path / "Maps" / "flycraft" / f"{name}.SC2Map"
  archive = mpyq.MPQArchive(io.BytesIO(out.read_bytes()), listfile=True)
  for file, data in mapbuild.MAPS[name](source()).items():
    assert archive.read_file(file) == data, file
  assert mapbuild.build(tmp_path, name) == out  # building again replaces it
  assert [p.name for p in out.parent.iterdir()] == [f"{name}.SC2Map"]


def test_build_makes_defeat_marines_unless_told_otherwise(tmp_path):
  _install(tmp_path)
  assert mapbuild.build(tmp_path).name == "DefeatMarines.SC2Map"


def test_build_says_where_it_looked(tmp_path):
  with pytest.raises(MapBuildError, match="DefeatRoaches.SC2Map"):
    mapbuild.build(tmp_path)
  (tmp_path / "Maps" / "mini_games").mkdir(parents=True)
  (tmp_path / mapbuild.SOURCE).write_bytes(b"not an archive")
  with pytest.raises(MapBuildError, match="cannot read"):
    mapbuild.build(tmp_path)


def test_main(tmp_path, capsys, monkeypatch):
  assert mapbuild.main(["--sc2path", str(tmp_path)]) == 1
  assert "flycraft-maps: DefeatMarines: " in capsys.readouterr().err
  _install(tmp_path)
  monkeypatch.setenv("SC2PATH", str(tmp_path))
  assert mapbuild.main([]) == 0  # every map
  out = capsys.readouterr().out
  assert "DefeatMarines.SC2Map" in out and "Skirmish.SC2Map" in out
  built = tmp_path / "Maps" / "flycraft"
  for p in built.iterdir():
    p.unlink()
  assert mapbuild.main(["--map", "Skirmish"]) == 0
  assert [p.name for p in built.iterdir()] == ["Skirmish.SC2Map"]


def test_main_stops_at_the_first_map_it_cannot_build(tmp_path, capsys):
  files = source()
  files[INFO] = files[INFO].replace(P2, P2[:1] + b"\x01" + P2[2:])  # Skirmish cannot use it
  (tmp_path / "Maps" / "mini_games").mkdir(parents=True)
  (tmp_path / mapbuild.SOURCE).write_bytes(mpq.pack(files))
  assert mapbuild.main(["--sc2path", str(tmp_path)]) == 1
  assert "flycraft-maps: Skirmish: MapInfo" in capsys.readouterr().err


def test_the_default_sc2_install(monkeypatch):
  monkeypatch.delenv("SC2PATH", raising=False)
  assert str(mapbuild.default_sc2path()).replace("\\", "/") == "C:/Program Files (x86)/StarCraft II"
  monkeypatch.setenv("SC2PATH", "/games/sc2")
  assert str(mapbuild.default_sc2path()) == "/games/sc2"


SCRIPT_TEXT = mapbuild.SKIRMISH_SCRIPT.read_text()


def test_the_skirmish_script_plays_the_spec_rules():
  for rule in ["const int gv_toWin = 3;", "const int gv_maxRounds = 7;",
               "const fixed gv_roundSeconds = 120.0;", "const int gv_marineCount = 4;",
               "const int gv_zerglingCount = 9;", "const int gv_person = 1;",
               "const int gv_fly = 2;", 'gf_Spawn("Marine", gv_person,',
               'gf_Spawn("Zergling", gv_fly,', "GameOver(gv_person, lv_p1, true, true);",
               "GameOver(gv_fly, lv_p2, true, true);", "TimerStart(gv_clock, gv_roundSeconds,",
               "VisRevealArea(lv_p, RegionFromId(1),", "TriggerAddEventUnitDied(gt_Died, null);"]:
    assert rule in SCRIPT_TEXT, rule
  # Only the arena's own regions: 1 the playable area, 4 and 5 the starts.
  assert set(re.findall(r"RegionFromId\((\d+)\)", SCRIPT_TEXT)) == {"1", "4", "5"}


def test_the_skirmish_script_defines_each_function_before_its_first_use():
  """Galaxy has no forward references: a call above its function does not compile."""
  defined = {m[2]: m.start() for m in re.finditer(
    r"^(void|bool|string|int|fixed) (\w+) \(", SCRIPT_TEXT, re.MULTILINE)}
  assert {"gf_StartRound", "gf_EndRound", "gf_EndMatch", "InitMap"} <= set(defined)
  for name, at in defined.items():
    calls = [m.start() for m in re.finditer(rf"\b{name}\(", SCRIPT_TEXT)]
    assert all(c > at for c in calls), name


def test_the_skirmish_script_is_plain_ascii_and_ships_with_the_package():
  assert SCRIPT_TEXT.isascii()
  assert mapbuild.SKIRMISH_SCRIPT.parent == Path(mapbuild.__file__).parent
