import math

import msgpack
import numpy as np
import pytest

from flycraft import eye, protocol
from flycraft.protocol import ProtocolError, VersionMismatch, decode, encode, make

EYE = bytes(range(256)) * 8 + bytes(112)  # 2,160 bytes

GOOD = {
  "hello": dict(client_version="0.1.0", map="MoveToBeacon", screen=84, mode="watch"),
  "ready": dict(brain_id="stub-oracle", wiring="none", scramble_seed=None, plasticity=False,
                eye_shape=[30, 72], decision_frames=8, polarity="dark_on_bright"),
  "episode_start": dict(episode=0, seed=7_000_000, phase="train"),
  "obs": dict(step=3, eye=EYE, reward=1.0, score=4.0, last=False, marine_xy=[41.5, 40.0],
              beacon_xy=None),
  "act": dict(step=3, dtheta=-12.5, speed=1.0, turn_raw=0.0, fwd_raw=0.0, pitch=0.0,
              pitch_raw=0.0),
  "episode_end": dict(episode=0, score=21.0, steps=241, aborted=None),
  "abort": dict(reason="shutdown"),
  "error": dict(message="busy"),
}


def test_every_type_has_a_case():
  assert set(GOOD) == set(protocol.FIELDS)


@pytest.mark.parametrize("kind", sorted(GOOD))
def test_round_trip(kind):
  msg = make(kind, **GOOD[kind])
  back = decode(encode(msg))
  assert back == {"type": kind, "v": 2, **GOOD[kind]}


def test_eye_stays_bytes_and_tuples_become_lists():
  back = decode(encode(make("obs", **{**GOOD["obs"], "marine_xy": (1.0, 2.0)})))
  assert isinstance(back["eye"], bytes) and back["marine_xy"] == [1.0, 2.0]


def test_numpy_scalars_encode():
  fields = dict(GOOD["act"], step=np.int64(3), dtheta=np.float32(-12.5), speed=np.float64(1.0))
  back = decode(encode(make("act", **fields)))
  assert back["step"] == 3 and back["dtheta"] == -12.5 and type(back["dtheta"]) is float


def test_version_mismatch_is_its_own_error():
  data = msgpack.packb({"type": "hello", "v": 3, **GOOD["hello"]}, use_bin_type=True)
  with pytest.raises(VersionMismatch, match="version 3"):
    decode(data)
  with pytest.raises(VersionMismatch):
    decode(msgpack.packb({"type": "hello", **GOOD["hello"]}, use_bin_type=True))


@pytest.mark.parametrize("data, match", [
  ("text", "binary frame"),
  (b"\xc1", "undecodable"),
  (msgpack.packb([1, 2]), "is a map"),
  (msgpack.packb({"type": "nope", "v": 2}), "unknown message type"),
])
def test_bad_frames(data, match):
  with pytest.raises(ProtocolError, match=match):
    decode(data)


def test_missing_and_extra_fields():
  with pytest.raises(ProtocolError, match="missing \\['fwd_raw'\\]"):
    make("act", **{k: v for k, v in GOOD["act"].items() if k != "fwd_raw"})
  with pytest.raises(ProtocolError, match="unexpected \\['marine_xy'\\]"):
    make("act", **GOOD["act"], marine_xy=[1, 2])


def test_missing_pitch_is_rejected():
  with pytest.raises(ProtocolError, match="missing \\['pitch'\\]"):
    make("act", **{k: v for k, v in GOOD["act"].items() if k != "pitch"})


@pytest.mark.parametrize("kind, field, value", [
  ("act", "dtheta", math.nan),
  ("act", "dtheta", math.inf),
  ("act", "speed", 1.5),
  ("act", "speed", -0.1),
  ("act", "step", True),
  ("act", "pitch", 1.5),
  ("act", "pitch", -1.5),
  ("act", "pitch", math.nan),
  ("act", "pitch_raw", math.inf),
  ("obs", "eye", EYE[:-1]),
  ("obs", "eye", list(EYE)),
  ("obs", "reward", math.nan),
  ("obs", "marine_xy", [1.0]),
  ("obs", "beacon_xy", [1.0, math.nan]),
  ("obs", "last", 0),
  ("hello", "mode", "play"),
  ("ready", "eye_shape", [72, 30]),
  ("ready", "polarity", "inverted"),
  ("ready", "decision_frames", 0),
  ("episode_start", "phase", "practice"),
  ("abort", "reason", "bored"),
  ("episode_end", "aborted", 3),
])
def test_bad_values(kind, field, value):
  with pytest.raises(ProtocolError, match=f"{kind}.{field}"):
    make(kind, **{**GOOD[kind], field: value})


def test_constants_match_the_eye_module():
  assert protocol.EYE_SHAPE == (eye.EYE_ROWS, eye.EYE_COLS)
  assert set(protocol.POLARITIES) == set(eye.POLARITY)
