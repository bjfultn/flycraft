import base64
import io
from email.message import Message

import google_crc32c
import pytest

from flycraft.data.fetch import FetchError, fetch_file, file_crc32c

PAYLOAD = b"flycraft test payload " * 5000


def crc(data: bytes) -> str:
  return base64.b64encode(google_crc32c.Checksum(data).digest()).decode()


class FakeResponse(io.BytesIO):
  def __init__(self, data: bytes, header_crc: str | None):
    super().__init__(data)
    self.headers = Message()
    if header_crc is not None:
      self.headers["x-goog-hash"] = "md5=AAAA"
      self.headers["x-goog-hash"] = f"crc32c={header_crc}"


class FakeOpener:
  def __init__(self, data: bytes, header_crc: str | None):
    self.data, self.header_crc, self.calls = data, header_crc, 0

  def __call__(self, url):
    self.calls += 1
    assert url.endswith("/x.feather")
    return FakeResponse(self.data, self.header_crc)


FILES = {"x.feather": (crc(PAYLOAD), len(PAYLOAD))}


def run(tmp_path, opener):
  return fetch_file("x.feather", tmp_path, opener, FILES, log=lambda s: None)


def test_download_writes_verified_file(tmp_path):
  opener = FakeOpener(PAYLOAD, crc(PAYLOAD))
  path = run(tmp_path, opener)
  assert path.read_bytes() == PAYLOAD
  assert file_crc32c(path) == crc(PAYLOAD)
  assert not (tmp_path / "x.feather.part").exists()


def test_good_existing_file_is_not_downloaded(tmp_path):
  (tmp_path / "x.feather").write_bytes(PAYLOAD)
  opener = FakeOpener(b"", None)
  run(tmp_path, opener)
  assert opener.calls == 0


def test_stale_existing_file_is_replaced(tmp_path):
  (tmp_path / "x.feather").write_bytes(b"old version")
  opener = FakeOpener(PAYLOAD, crc(PAYLOAD))
  assert run(tmp_path, opener).read_bytes() == PAYLOAD
  assert opener.calls == 1


def test_truncated_download_fails_and_leaves_nothing(tmp_path):
  short = PAYLOAD[:-100]
  with pytest.raises(FetchError, match="truncated"):
    run(tmp_path, FakeOpener(short, crc(short)))
  assert not (tmp_path / "x.feather").exists()
  assert not (tmp_path / "x.feather.part").exists()


def test_header_mismatch_fails(tmp_path):
  with pytest.raises(FetchError, match="x-goog-hash"):
    run(tmp_path, FakeOpener(PAYLOAD, crc(b"something else")))
  assert not (tmp_path / "x.feather").exists()


def test_pinned_mismatch_fails_even_when_header_agrees(tmp_path):
  other = bytes(reversed(PAYLOAD))  # same size, different content
  with pytest.raises(FetchError, match="pinned"):
    run(tmp_path, FakeOpener(other, crc(other)))
  assert not (tmp_path / "x.feather").exists()


def test_failed_redownload_keeps_no_part_file(tmp_path):
  (tmp_path / "x.feather").write_bytes(b"old version")
  with pytest.raises(FetchError):
    run(tmp_path, FakeOpener(PAYLOAD[:10], None))
  assert not (tmp_path / "x.feather.part").exists()
