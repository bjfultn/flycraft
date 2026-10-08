"""A minimal MPQ archive writer, enough to write an SC2 map (spec M3c section 2).

pysc2's MPQ library (mpyq) only reads. This writes the classic format: a 32-byte format-1
header, the files, then the hash table and the block table, both encrypted as MPQ requires.
Each file is stored whole as one unit, zlib-compressed when that makes it smaller, and flagged
as Blizzard flags every file of its maps. The archive gets a (listfile) naming its files; any
(listfile) or (attributes) passed in is dropped, since they describe the archive they came from.
"""

from __future__ import annotations

import struct
import zlib

FLAGS = 0x81000200  # exists, a single unit, compressed (stored as is when that gains nothing)
OFFSET, NAME_A, NAME_B, KEY = 0, 1, 2, 3  # the kinds of name hash
EMPTY = 0xFFFFFFFF  # a hash table slot no file has taken
HEADER = struct.Struct("<4sIIHHIIII")
SECTOR_SHIFT = 3  # sectors of 4 KiB; unused, since every file is one unit
METADATA = ("(listfile)", "(attributes)")
_M = 0xFFFFFFFF


def _crypt_table() -> list[int]:
  seed, table = 0x00100001, [0] * 0x500
  for i in range(0x100):
    for j in range(5):
      seed = (seed * 125 + 3) % 0x2AAAAB
      high = (seed & 0xFFFF) << 16
      seed = (seed * 125 + 3) % 0x2AAAAB
      table[i + j * 0x100] = high | (seed & 0xFFFF)
  return table


_TABLE = _crypt_table()


def _name_bytes(name: str) -> bytes:
  """The name as MPQ hashes it. Names are ASCII with backslashes, as in SC2's maps."""
  if not name or not name.isascii() or "/" in name:
    raise ValueError(f"an MPQ file name must be non-empty ASCII with backslashes: {name!r}")
  return name.upper().encode("ascii")


def hash_name(name: str, kind: int) -> int:
  """MPQ's hash of a file name, ignoring case: its slot (OFFSET), the two checks a reader
  compares (NAME_A, NAME_B), or a table's encryption key (KEY)."""
  s1, s2 = 0x7FED7FED, 0xEEEEEEEE
  for c in _name_bytes(name):
    s1 = _TABLE[(kind << 8) + c] ^ ((s1 + s2) & _M)
    s2 = (c + s1 + s2 + (s2 << 5) + 3) & _M
  return s1


def _encrypt(data: bytes, key: int) -> bytes:
  s1, s2, out = key, 0xEEEEEEEE, []
  for (v,) in struct.iter_unpack("<I", data):
    s2 = (s2 + _TABLE[0x400 + (s1 & 0xFF)]) & _M
    out.append(v ^ ((s1 + s2) & _M))
    s1 = (((~s1 << 0x15) + 0x11111111) | (s1 >> 0x0B)) & _M
    s2 = (v + s2 + (s2 << 5) + 3) & _M
  return struct.pack(f"<{len(out)}I", *out)


def _store(data: bytes) -> bytes:
  """A file's stored bytes. Readers decompress a single unit only when it is stored smaller
  than it is, so incompressible data is stored as it is."""
  packed = b"\x02" + zlib.compress(data, 9)  # 0x02: the zlib method byte
  return packed if len(packed) < len(data) else data


def pack(files: dict[str, bytes]) -> bytes:
  """An MPQ archive holding files (name to contents), with a (listfile)."""
  files = {n: d for n, d in files.items() if n not in METADATA}
  files["(listfile)"] = "".join(f"{n}\r\n" for n in files).encode()  # names checked below
  size = 16
  while size < 2 * len(files):  # at most half full, so probing stays short
    size *= 2
  slots = [(EMPTY, EMPTY, 0xFFFF, 0xFFFF, EMPTY)] * size
  blocks, body, pos = [], [], HEADER.size
  for index, (name, data) in enumerate(files.items()):
    stored = _store(data)
    blocks.append((pos, len(stored), len(data), FLAGS))
    body.append(stored)
    pos += len(stored)
    slot = hash_name(name, OFFSET) & (size - 1)
    while slots[slot][4] != EMPTY:  # taken: readers probe the next slot
      slot = (slot + 1) & (size - 1)
    slots[slot] = (hash_name(name, NAME_A), hash_name(name, NAME_B), 0, 0, index)
  hash_table = _encrypt(b"".join(struct.pack("<IIHHI", *s) for s in slots),
                        hash_name("(hash table)", KEY))
  block_table = _encrypt(b"".join(struct.pack("<IIII", *b) for b in blocks),
                         hash_name("(block table)", KEY))
  total = pos + len(hash_table) + len(block_table)
  header = HEADER.pack(b"MPQ\x1a", HEADER.size, total, 0, SECTOR_SHIFT, pos,
                       pos + len(hash_table), size, len(blocks))
  return header + b"".join(body) + hash_table + block_table
