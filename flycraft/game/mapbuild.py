"""flycraft-maps: build the maps flycraft plays that SC2 does not ship.

Both are built from the player's own copy of DefeatRoaches, since flycraft does not ship SC2's
maps. A source that does not hold each thing a build changes exactly as DefeatRoaches does
stops the build, so a changed map is never half converted.

DefeatMarines (spec M3c section 2) is DefeatRoaches with the squad made of zerglings and the
enemy of marines. Four things change: the units in the map script and the preload list, the
players' races in MapInfo, and the map's strings.

Skirmish (spec M6 section 2) is the arena for a person against the fly: player 2's slot becomes
a User one so a second client can join, the preload list loads zerglings for roaches, the map
is renamed (in its strings and its header, whose name SC2 shows), and the map script is
flycraft's own (skirmish.galaxy, next to this file).
"""

from __future__ import annotations

import argparse
import codecs
import os
import re
import sys
from pathlib import Path

from flycraft.game import mpq

SOURCE = Path("Maps") / "mini_games" / "DefeatRoaches.SC2Map"
BUILT = Path("Maps") / "flycraft"  # where the built maps go, under the SC2 install
SCRIPT, PRELOAD, INFO, HEADER = "MapScript.galaxy", "Preload.xml", "MapInfo", "DocumentHeader"
STRINGS = "enUS.SC2Data\\LocalizedData\\GameStrings.txt"
UNITS = (("Marine", "Zergling"), ("Roach", "Marine"))  # the squad's unit, then the enemy's
RACES = (b"Terr\x00", b"Zerg\x00")  # MapInfo's race fields: player 1's, then player 2's
TEXTS = (("Defeat Roaches", "Defeat Marines"), ("Roaches Defeated", "Marines Defeated"),
         ("Player02/Name=Zerg", "Player02/Name=Terran"))
SKIRMISH_UNITS = (("Roach", "Zergling"),)  # the preload list's; the script spawns the rest
SKIRMISH_TEXTS = (("DocInfo/Name=CombatFocus", "DocInfo/Name=Skirmish"),
                  ("Player02/Name=Zerg", "Player02/Name=Fly"))
SKIRMISH_HEADER = (("DocInfo/Name", "CombatFocus", "Skirmish"),  # the name SC2 shows
                   ("MapInfo/Player02/Name", "Zerg", "Fly"))
SKIRMISH_SCRIPT = Path(__file__).with_name("skirmish.galaxy")
USER, COMPUTER = (1).to_bytes(4, "little"), (2).to_bytes(4, "little")  # a player's control
WINDOWS_SC2 = "C:/Program Files (x86)/StarCraft II"  # where SC2 installs on Windows


class MapBuildError(Exception):
  pass


def _file(files: dict[str, bytes], name: str) -> bytes:
  if name not in files:
    raise MapBuildError(f"the source map has no {name}")
  return files[name]


def _swap_units(files: dict[str, bytes], name: str, units=UNITS) -> bytes:
  """Each quoted unit id swapped in one pass, so the new marines are not then made zerglings."""
  text = _file(files, name).decode("utf-8")
  swap = dict(units)
  for new in [new for new in swap.values() if new not in swap]:
    if f'"{new}"' in text:
      raise MapBuildError(f'{name} already names "{new}": is the source DefeatRoaches?')
  for old in swap:
    if f'"{old}"' not in text:
      raise MapBuildError(f'{name} names no "{old}": is the source DefeatRoaches?')
  quoted = re.compile('"(' + "|".join(re.escape(old) for old in swap) + ')"')
  return quoted.sub(lambda m: f'"{swap[m[1]]}"', text).encode("utf-8")


def _swap_races(info: bytes) -> bytes:
  """Player 1 becomes Zerg and player 2 Terran: the two race fields trade places."""
  at = []
  for race in RACES:
    if info.count(race) != 1:
      raise MapBuildError(f"{INFO} has {info.count(race)} {race[:4].decode()} race fields, not 1")
    at.append(info.index(race))
  out = bytearray(info)
  out[at[0]:at[0] + 4], out[at[1]:at[1] + 4] = RACES[1][:4], RACES[0][:4]
  return bytes(out)


def _player_two_user(info: bytes) -> bytes:
  """Player 2's slot made a User one, not a Computer, so a second client can join it. A player
  in MapInfo is its id (a byte), its control and its color (4 bytes each), then its race."""
  race = RACES[1]
  if info.count(race) != 1:
    raise MapBuildError(f"{INFO} has {info.count(race)} {race[:4].decode()} race fields, not 1")
  at = info.index(race)
  if at < 9 or info[at - 9] != 2 or info[at - 8:at - 4] != COMPUTER:
    raise MapBuildError(f"{INFO}: player 2 is not a Zerg computer: is the source DefeatRoaches?")
  return info[:at - 8] + USER + info[at - 4:]


def _entry(key: str, value: str) -> bytes:
  """One of DocumentHeader's strings: its key, the locale (enUS, stored backwards) and its value,
  key and value each after its length in 2 bytes."""
  k, v = key.encode(), value.encode()
  return len(k).to_bytes(2, "little") + k + b"SUne" + len(v).to_bytes(2, "little") + v


def _rename_header(header: bytes, entries) -> bytes:
  """Each (key, old, new): the header's string for key, old, made new, and only there."""
  for key, old, new in entries:
    was = _entry(key, old)
    if header.count(was) != 1:
      raise MapBuildError(f"{HEADER} has {header.count(was)} {key}={old}, not 1: "
                          "is the source DefeatRoaches?")
    header = header.replace(was, _entry(key, new))
  return header


def _rename(strings: bytes, texts=TEXTS) -> bytes:
  """Each string that names what the map changes, renamed where it is, and only there."""
  text = strings.decode("utf-8-sig")
  for old, new in texts:
    if text.count(old) != 1:
      raise MapBuildError(f"{STRINGS} has {text.count(old)} {old!r}, not 1: "
                          "is the source DefeatRoaches?")
    text = text.replace(old, new)
  return text.encode("utf-8-sig" if strings.startswith(codecs.BOM_UTF8) else "utf-8")


def defeat_marines(files: dict[str, bytes]) -> dict[str, bytes]:
  """DefeatRoaches' files (name to contents) made into DefeatMarines'."""
  out = dict(files)
  out[SCRIPT] = _swap_units(files, SCRIPT)
  out[PRELOAD] = _swap_units(files, PRELOAD)
  out[INFO] = _swap_races(_file(files, INFO))
  out[STRINGS] = _rename(_file(files, STRINGS))
  return out


def skirmish(files: dict[str, bytes]) -> dict[str, bytes]:
  """DefeatRoaches' files (name to contents) made into Skirmish's."""
  _file(files, SCRIPT)  # replaced whole, but a source without one is not DefeatRoaches
  out = dict(files)
  out[SCRIPT] = SKIRMISH_SCRIPT.read_bytes()
  out[PRELOAD] = _swap_units(files, PRELOAD, SKIRMISH_UNITS)
  out[INFO] = _player_two_user(_file(files, INFO))
  out[STRINGS] = _rename(_file(files, STRINGS), SKIRMISH_TEXTS)
  out[HEADER] = _rename_header(_file(files, HEADER), SKIRMISH_HEADER)
  return out


MAPS = {"DefeatMarines": defeat_marines, "Skirmish": skirmish}


def read(path: Path, name: str = "DefeatMarines") -> dict[str, bytes]:
  import mpyq  # pysc2's MPQ reader: only on the game side

  if not path.is_file():
    raise MapBuildError(f"no map at {path}: {name} is built from SC2's DefeatRoaches")
  try:
    archive = mpyq.MPQArchive(str(path), listfile=True)
    return {n.decode(): archive.read_file(n) or b"" for n in archive.files}
  except Exception as e:  # mpyq raises whatever its parsing hits
    raise MapBuildError(f"cannot read {path}: {e}") from e


def build(sc2path: Path, name: str = "DefeatMarines") -> Path:
  """Write the named map (one of MAPS) under the SC2 install at sc2path; return where."""
  data = mpq.pack(MAPS[name](read(sc2path / SOURCE, name)))
  out = sc2path / BUILT / f"{name}.SC2Map"
  out.parent.mkdir(parents=True, exist_ok=True)
  part = out.with_name(out.name + ".part")
  part.write_bytes(data)
  part.replace(out)  # never a half-written map where SC2 looks
  return out


def default_sc2path() -> Path:
  """SC2PATH, as pysc2 reads it, else SC2's usual install. pysc2 also finds an SC2 elsewhere
  from its launcher's files; the client then says where it looked, to pass as --sc2path."""
  return Path(os.environ.get("SC2PATH") or WINDOWS_SC2)


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(
    prog="flycraft-maps", description="Build DefeatMarines (zerglings against marines) and "
    "Skirmish (a person against the fly) from SC2's DefeatRoaches, into Maps/flycraft under "
    "the SC2 install.")
  p.add_argument("--sc2path", type=Path, default=None,
                 help=f"the SC2 install (default $SC2PATH, else {WINDOWS_SC2})")
  p.add_argument("--map", choices=sorted(MAPS), default=None,
                 help="build only this map (default: all of them)")
  args = p.parse_args(argv)
  sc2path = args.sc2path or default_sc2path()
  for name in [args.map] if args.map else MAPS:
    try:
      out = build(sc2path, name)
    except (MapBuildError, OSError) as e:
      print(f"flycraft-maps: {name}: {e}", file=sys.stderr)
      return 1
    print(f"wrote {out}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
