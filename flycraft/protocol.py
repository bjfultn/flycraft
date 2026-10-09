"""The game <-> brain protocol (spec section 10): msgpack maps over a websocket.

Both sides import this module, so it depends only on msgpack. Every message is a map with
`type` and `v`. check() rejects a wrong version first (VersionMismatch), then an unknown type,
missing or extra fields, and values of the wrong kind. A non-finite number never passes: a NaN
turn from a runaway brain must stop the client, not reach SC2 as a move target.
"""

from __future__ import annotations

import math
import numbers

import msgpack

VERSION = 2
EYE_SHAPE = (30, 72)  # flycraft.eye.EYE_ROWS, EYE_COLS; a test keeps them equal
EYE_BYTES = EYE_SHAPE[0] * EYE_SHAPE[1]
MODES = ("watch", "train", "eval")
ABORT_REASONS = ("yield", "runaway", "shutdown")
POLARITIES = ("dark_on_bright", "bright_on_dark")

FIELDS = {
  "hello": ("client_version", "map", "screen", "mode"),
  "ready": ("brain_id", "wiring", "scramble_seed", "plasticity", "eye_shape", "decision_frames",
            "polarity"),
  "episode_start": ("episode", "seed", "phase"),
  "obs": ("step", "eye", "reward", "score", "last", "marine_xy", "beacon_xy"),
  "act": ("step", "dtheta", "speed", "turn_raw", "fwd_raw", "pitch", "pitch_raw"),
  "episode_end": ("episode", "score", "steps", "aborted"),
  "abort": ("reason",),
  "error": ("message",),
}


class ProtocolError(ValueError):
  pass


class VersionMismatch(ProtocolError):
  pass


def _int(x) -> bool:
  return isinstance(x, numbers.Integral) and not isinstance(x, bool)


def _num(x) -> bool:
  return isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(x)


def _xy(x) -> bool:
  return x is None or (isinstance(x, list | tuple) and len(x) == 2 and all(_num(v) for v in x))


def _opt_str(x) -> bool:
  return x is None or isinstance(x, str)


def _opt_int(x) -> bool:
  return x is None or _int(x)


_KINDS = {
  "hello": {"client_version": lambda x: isinstance(x, str), "map": lambda x: isinstance(x, str),
            "screen": lambda x: _int(x) and x > 0, "mode": lambda x: x in MODES},
  "ready": {"brain_id": lambda x: isinstance(x, str), "wiring": lambda x: isinstance(x, str),
            "scramble_seed": _opt_int, "plasticity": lambda x: isinstance(x, bool),
            "eye_shape": lambda x: isinstance(x, list | tuple) and tuple(x) == EYE_SHAPE,
            "decision_frames": lambda x: _int(x) and x >= 1,
            "polarity": lambda x: x in POLARITIES},
  "episode_start": {"episode": lambda x: _int(x) and x >= 0, "seed": _int,
                    "phase": lambda x: x in MODES},
  "obs": {"step": lambda x: _int(x) and x >= 0,
          "eye": lambda x: isinstance(x, bytes) and len(x) == EYE_BYTES,
          "reward": _num, "score": _num, "last": lambda x: isinstance(x, bool),
          "marine_xy": _xy, "beacon_xy": _xy},
  "act": {"step": lambda x: _int(x) and x >= 0, "dtheta": _num,
          "speed": lambda x: _num(x) and 0.0 <= x <= 1.0, "turn_raw": _num, "fwd_raw": _num,
          "pitch": lambda x: _num(x) and -1.0 <= x <= 1.0, "pitch_raw": _num},
  "episode_end": {"episode": lambda x: _int(x) and x >= 0, "score": _num,
                  "steps": lambda x: _int(x) and x >= 0, "aborted": _opt_str},
  "abort": {"reason": lambda x: x in ABORT_REASONS},
  "error": {"message": lambda x: isinstance(x, str)},
}


def check(msg) -> dict:
  """Return msg if it is a valid message, else raise ProtocolError (or VersionMismatch)."""
  if not isinstance(msg, dict):
    raise ProtocolError(f"a message is a map, got {type(msg).__name__}")
  if msg.get("v") != VERSION:
    raise VersionMismatch(f"protocol version {msg.get('v')!r}, this side speaks {VERSION}")
  kind = msg.get("type")
  if kind not in FIELDS:
    raise ProtocolError(f"unknown message type {kind!r}")
  got, want = set(msg) - {"type", "v"}, set(FIELDS[kind])
  if got != want:
    missing, extra = sorted(want - got), sorted(map(str, got - want))
    raise ProtocolError(f"{kind}: missing {missing}, unexpected {extra}")
  for name, ok in _KINDS[kind].items():
    if not ok(msg[name]):
      raise ProtocolError(f"{kind}.{name}: bad value {msg[name]!r:.80}")
  return msg


def make(kind: str, **fields) -> dict:
  return check({"type": kind, "v": VERSION, **fields})


def _default(o):
  if getattr(o, "shape", None) == () and hasattr(o, "item"):  # numpy scalar
    return o.item()
  raise TypeError(f"cannot encode {type(o).__name__}")


def encode(msg: dict) -> bytes:
  return msgpack.packb(check(msg), use_bin_type=True, default=_default)


def decode(data) -> dict:
  if not isinstance(data, bytes | bytearray | memoryview):
    raise ProtocolError("expected a binary frame")
  try:
    msg = msgpack.unpackb(bytes(data), raw=False)
  except Exception as e:  # msgpack raises several unrelated types on bad input
    raise ProtocolError(f"undecodable frame: {e}") from e
  return check(msg)
