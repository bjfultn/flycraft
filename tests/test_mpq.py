import io
import random
import struct

import pytest

from flycraft.game import mpq

mpyq = pytest.importorskip("mpyq")  # pysc2's MPQ reader, on the game side


def read(data: bytes):
  return mpyq.MPQArchive(io.BytesIO(data), listfile=True)


def test_the_table_keys_are_mpqs():
  """The keys every MPQ reader decrypts the tables with (StormLib's MPQ_KEY_*_TABLE)."""
  assert mpq.hash_name("(hash table)", mpq.KEY) == 0xC3AF3770
  assert mpq.hash_name("(block table)", mpq.KEY) == 0xEC83B3A3


def test_names_hash_without_case():
  for kind in (mpq.OFFSET, mpq.NAME_A, mpq.NAME_B, mpq.KEY):
    assert mpq.hash_name("MapScript.galaxy", kind) == mpq.hash_name("MAPSCRIPT.GALAXY", kind)


def test_every_file_reads_back():
  rng = random.Random(1)
  files = {
    "empty": b"",
    "tiny": b"ab",
    "noise": bytes(rng.getrandbits(8) for _ in range(5000)),  # stored as it is
    "Base.SC2Data\\big.txt": b"zergling " * 50_000,  # stored compressed
    **{f"f{i}": f"file {i}".encode() * (i + 1) for i in range(40)},
  }
  archive = read(mpq.pack(files))
  for name, data in files.items():
    assert (archive.read_file(name) or b"") == data, name  # mpyq returns None for empty


def find(archive, name: str) -> int | None:
  """The block a real MPQ reader finds name at: it starts at the name's slot and walks on until
  an empty one. (mpyq scans the whole table instead, so it would miss a misplaced entry.)"""
  table = archive.hash_table
  key = (mpq.hash_name(name, mpq.NAME_A), mpq.hash_name(name, mpq.NAME_B))
  i = mpq.hash_name(name, mpq.OFFSET) & (len(table) - 1)
  for _ in table:
    if table[i].block_table_index == mpq.EMPTY:
      return None
    if (table[i].hash_a, table[i].hash_b) == key:
      return table[i].block_table_index
    i = (i + 1) % len(table)
  return None


def test_colliding_slots_are_probed():
  files = {f"f{i}": bytes([i]) * 64 for i in range(40)}
  names = [*files, "(listfile)"]
  archive = read(mpq.pack(files))
  size = archive.header["hash_table_entries"]
  slots = [mpq.hash_name(n, mpq.OFFSET) & (size - 1) for n in names]
  assert len(set(slots)) < len(slots)  # some names want the same slot
  blocks = [find(archive, n) for n in names]
  assert None not in blocks and len(set(blocks)) == len(names)
  assert all(archive.read_file(n) == d for n, d in files.items())


def test_compression_is_kept_only_when_it_gains():
  data = mpq.pack({"noise": bytes(range(256)), "text": b"marine " * 1000})
  archive = read(data)
  sizes = {}
  for name in ("noise", "text"):
    block = archive.block_table[archive.get_hash_table_entry(name).block_table_index]
    assert block.flags == mpq.FLAGS  # what Blizzard sets on every file of its maps
    sizes[name] = (block.archived_size, block.size)
  assert sizes["noise"] == (256, 256)
  assert sizes["text"][0] < sizes["text"][1] == 7000


def test_the_listfile_names_every_file_and_old_metadata_is_dropped():
  files = {"a": b"1", "b\\c": b"2", "(listfile)": b"stale", "(attributes)": b"stale"}
  archive = read(mpq.pack(files))
  assert archive.read_file("(listfile)") == b"a\r\nb\\c\r\n"
  assert archive.get_hash_table_entry("(attributes)") is None


def test_the_header():
  data = mpq.pack({"a": b"1"})
  magic, header_size, size, version = struct.unpack_from("<4sIIH", data)
  assert (magic, header_size, size, version) == (b"MPQ\x1a", 32, len(data), 0)


@pytest.mark.parametrize("name", ["", "café", "a/b"])
def test_a_name_mpq_cannot_hold_is_refused(name):
  with pytest.raises(ValueError):
    mpq.pack({name: b"1"})
