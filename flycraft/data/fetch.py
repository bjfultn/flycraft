"""Download the MaleCNS v1.0 flat-connectome files and verify them by crc32c.

Each file is pinned by name, crc32c (base64, as GCS reports it) and byte size. A file is
streamed to `<name>.part` and renamed into place only after its size, the bucket's
x-goog-hash crc32c and the pinned crc32c all agree. Any failure deletes the `.part` file
and raises, so a truncated download can never be mistaken for a good one.
"""

from __future__ import annotations

import base64
import re
import urllib.request
from collections.abc import Callable
from pathlib import Path

import google_crc32c

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"

# name -> (crc32c base64, size in bytes), checked 2026-10-07
FILES = {
  ANNOTATIONS: ("vjz9cg==", 14_483_314),
  NEUROTRANSMITTERS: ("jcpNFg==", 43_282_834),
  WEIGHTS: ("iakIVA==", 508_025_642),
}

_CHUNK = 1 << 20


class FetchError(RuntimeError):
  pass


def _default_opener(url: str):
  return urllib.request.urlopen(url, timeout=60)


def file_crc32c(path: Path) -> str:
  c = google_crc32c.Checksum()
  with open(path, "rb") as fh:
    for chunk in iter(lambda: fh.read(_CHUNK), b""):
      c.update(chunk)
  return base64.b64encode(c.digest()).decode()


def _header_crc32c(headers) -> str | None:
  values = headers.get_all("x-goog-hash") or []
  for value in values:
    m = re.search(r"crc32c=([A-Za-z0-9+/=]+)", value)
    if m:
      return m.group(1)
  return None


def fetch_file(
  name: str,
  dest_dir: Path,
  opener: Callable | None = None,
  files: dict[str, tuple[str, int]] = FILES,
  log: Callable[[str], None] = print,
) -> Path:
  """Make dest_dir/name a verified copy of the pinned file, downloading if needed."""
  expected_crc, expected_size = files[name]
  dest_dir = Path(dest_dir).expanduser()
  dest_dir.mkdir(parents=True, exist_ok=True)
  dest = dest_dir / name
  if dest.exists():
    if file_crc32c(dest) == expected_crc:
      log(f"{name}: present, crc32c ok")
      return dest
    log(f"{name}: present but crc32c differs from the pinned value; downloading again")
  part = dest_dir / (name + ".part")
  opener = opener or _default_opener
  try:
    c = google_crc32c.Checksum()
    n = 0
    with opener(BASE_URL + name) as resp, open(part, "wb") as fh:
      header_crc = _header_crc32c(resp.headers)
      while chunk := resp.read(_CHUNK):
        fh.write(chunk)
        c.update(chunk)
        n += len(chunk)
    got = base64.b64encode(c.digest()).decode()
    if n != expected_size:
      raise FetchError(f"{name}: got {n} bytes, expected {expected_size} (truncated download?)")
    if header_crc is not None and got != header_crc:
      raise FetchError(f"{name}: crc32c {got} does not match x-goog-hash {header_crc}")
    if got != expected_crc:
      raise FetchError(f"{name}: crc32c {got} does not match the pinned {expected_crc}")
    part.replace(dest)
  except BaseException:
    part.unlink(missing_ok=True)
    raise
  log(f"{name}: downloaded {n} bytes, crc32c ok")
  return dest


def fetch_all(dest_dir: Path, opener: Callable | None = None, log=print) -> dict[str, Path]:
  return {name: fetch_file(name, dest_dir, opener, FILES, log) for name in FILES}
