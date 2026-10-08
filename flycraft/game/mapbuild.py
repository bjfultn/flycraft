"""flycraft-maps: build the maps flycraft plays that SC2 does not ship (spec M3c section 2).

DefeatMarines is DefeatRoaches with the squad made of zerglings and the enemy of marines,
built from the player's own copy of DefeatRoaches, since flycraft does not ship SC2's maps.
Four things change: the units in the map script and the preload list, the players' races in
MapInfo, and the map's strings. A source that does not hold each of them exactly as
DefeatRoaches does stops the build, so a changed map is never half converted.
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
TARGET = Path("Maps") / "flycraft" / "DefeatMarines.SC2Map"
SCRIPT, PRELOAD, INFO = "MapScript.galaxy", "Preload.xml", "MapInfo"
STRINGS = "enUS.SC2Data\\LocalizedData\\GameStrings.txt"
UNITS = (("Marine", "Zergling"), ("Roach", "Marine"))  # the squad's unit, then the enemy's
RACES = (b"Terr\x00", b"Zerg\x00")  # MapInfo's race fields: player 1's, then player 2's
TEXTS = (("Defeat Roaches", "Defeat Marines"), ("Roaches Defeated", "Marines Defeated"),
         ("Player02/Name=Zerg", "Player02/Name=Terran"))
WINDOWS_SC2 = "C:/Program Files (x86)/StarCraft II"  # where SC2 installs on Windows
_QUOTED = re.compile('"(' + "|".join(re.escape(old) for old, _ in UNITS) + ')"')


class MapBuildError(Exception):
  pass


def _file(files: dict[str, bytes], name: str) -> bytes:
  if name not in files:
    raise MapBuildError(f"the source map has no {name}")
  return files[name]


def _swap_units(files: dict[str, bytes], name: str) -> bytes:
  """Each quoted unit id swapped in one pass, so the new marines are not then made zerglings."""
  text = _file(files, name).decode("utf-8")
  squad = UNITS[0][1]
  if f'"{squad}"' in text:
    raise MapBuildError(f'{name} already names "{squad}": is the source DefeatRoaches?')
  for old, _ in UNITS:
    if f'"{old}"' not in text:
      raise MapBuildError(f'{name} names no "{old}": is the source DefeatRoaches?')
  swap = dict(UNITS)
  return _QUOTED.sub(lambda m: f'"{swap[m[1]]}"', text).encode("utf-8")


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


def _rename(strings: bytes) -> bytes:
  """Each string that names the map's roaches or Zerg, renamed where it is, and only there."""
  text = strings.decode("utf-8-sig")
  for old, new in TEXTS:
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


def read(path: Path) -> dict[str, bytes]:
  import mpyq  # pysc2's MPQ reader: only on the game side

  if not path.is_file():
    raise MapBuildError(f"no map at {path}: DefeatMarines is built from SC2's DefeatRoaches")
  try:
    archive = mpyq.MPQArchive(str(path), listfile=True)
    return {n.decode(): archive.read_file(n) or b"" for n in archive.files}
  except Exception as e:  # mpyq raises whatever its parsing hits
    raise MapBuildError(f"cannot read {path}: {e}") from e


def build(sc2path: Path) -> Path:
  """Write DefeatMarines under the SC2 install at sc2path; return where."""
  data = mpq.pack(defeat_marines(read(sc2path / SOURCE)))
  out = sc2path / TARGET
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
    prog="flycraft-maps", description="Build DefeatMarines (zerglings against marines) from "
    "SC2's DefeatRoaches, into Maps/flycraft under the SC2 install.")
  p.add_argument("--sc2path", type=Path, default=None,
                 help=f"the SC2 install (default $SC2PATH, else {WINDOWS_SC2})")
  args = p.parse_args(argv)
  try:
    out = build(args.sc2path or default_sc2path())
  except (MapBuildError, OSError) as e:
    print(f"flycraft-maps: {e}", file=sys.stderr)
    return 1
  print(f"wrote {out}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
