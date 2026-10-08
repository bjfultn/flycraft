"""The brain view (spec section 11): a page at http://127.0.0.1:8766 that draws every neuron and
lights it up as it spikes.

The view never slows the brain. The brain thread publishes a frame per 10 ms of brain time
into a one-frame slot, newest wins; each connected page is sent the newest frame whenever its
last send has finished, and frames it was too slow for are dropped. With no page connected,
the brain skips building frames.

Endpoints:
- GET /, /view.js, /three.module.min.js, /OrbitControls.js: the page (Three.js is vendored).
- GET /meta: JSON with the neuron count, highlighted groups as index lists, the condition
  (wiring, plasticity, rung, calibration) and the superclass names.
- GET /positions.bin: float32 N x 3, micrometres. GET /classes.bin: uint8 N superclass codes.
- WS /activity: binary frames, laid out as
  [uint32 LE length of the JSON][JSON: t, episode, step, score, reward, turn, fwd, dtheta,
  speed, rates, pop_hz][N uint8 spike counts, saturating at 255][2,160 eye bytes, 30 x 72].
"""

from __future__ import annotations

import asyncio
import json
import struct
import threading
from collections.abc import Callable
from importlib import resources

import numpy as np
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from flycraft import protocol
from flycraft.brain.server import HOST, ListenError

PORT = 8766
# After any plain HTTP response, websockets waits this long for the client to hang up; a page
# that keeps its connection would otherwise hold up Ctrl+C by the 10 s default.
CLOSE_TIMEOUT_S = 1.0
STATIC = {"/": "index.html", "/view.js": "view.js", "/three.module.min.js": "three.module.min.js",
          "/OrbitControls.js": "OrbitControls.js"}
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".json": "application/json", ".bin": "application/octet-stream"}


def encode_frame(info: dict, counts: np.ndarray, eye: np.ndarray) -> bytes:
  head = json.dumps(info).encode()
  spikes = np.minimum(counts, 255).astype(np.uint8)
  img = np.ascontiguousarray(eye, dtype=np.uint8)
  if img.size != protocol.EYE_BYTES:
    raise ValueError(f"eye image must have {protocol.EYE_BYTES} pixels, got {img.size}")
  return struct.pack("<I", len(head)) + head + spikes.tobytes() + img.tobytes()


def decode_frame(data: bytes, n: int) -> tuple[dict, np.ndarray, np.ndarray]:
  """The inverse of encode_frame, for tests: (info, counts uint8 (n,), eye uint8 (30, 72))."""
  (k,) = struct.unpack_from("<I", data)
  info = json.loads(data[4:4 + k])
  counts = np.frombuffer(data, np.uint8, n, 4 + k)
  img = np.frombuffer(data, np.uint8, protocol.EYE_BYTES, 4 + k + n)
  return info, counts, img.reshape(protocol.EYE_SHAPE)


class ViewHub:
  """The one-frame slot between the brain thread and the view's event loop."""

  def __init__(self):
    self._lock = threading.Lock()
    self._frame: bytes | None = None
    self._seq = 0
    self._pending = False
    self._loop: asyncio.AbstractEventLoop | None = None
    self._wakers: set[asyncio.Event] = set()
    self.clients = 0  # changed only on the loop; read from the brain thread

  @property
  def has_clients(self) -> bool:
    return self.clients > 0

  def bind(self, loop: asyncio.AbstractEventLoop) -> None:
    """The loop the pages are served on; put() wakes their senders there."""
    with self._lock:
      self._loop = loop

  def publish(self, info: dict, counts: np.ndarray, eye: np.ndarray) -> None:
    """From any thread: replace the slot's frame. Never waits on a page."""
    self.put(encode_frame(info, counts, eye))

  def put(self, frame: bytes) -> None:
    with self._lock:
      self._frame, self._seq = frame, self._seq + 1
      if self._pending or self._loop is None:
        return
      self._pending = True
      loop = self._loop
    try:
      loop.call_soon_threadsafe(self._wake)
    except RuntimeError:  # the loop has closed: the server is stopping
      pass

  def latest(self) -> tuple[bytes | None, int]:
    with self._lock:
      return self._frame, self._seq

  def clear(self) -> None:
    """Drop the slot's frame without waking anyone: a page that connects now gets nothing
    until the next publish."""
    with self._lock:
      self._frame = None

  def _wake(self) -> None:
    with self._lock:
      self._pending = False
    for w in self._wakers:
      w.set()

  async def stream(self, ws: ServerConnection) -> None:
    """Send ws the newest frame each time there is a new one, until it closes."""
    wake = asyncio.Event()
    wake.set()  # a new page gets the current frame at once
    self._wakers.add(wake)
    self.clients += 1

    async def send() -> None:
      seen = 0
      while True:
        await wake.wait()
        wake.clear()
        frame, seq = self.latest()
        if frame is not None and seq != seen:
          seen = seq
          await ws.send(frame)

    sender = asyncio.create_task(send())
    closed = asyncio.create_task(ws.wait_closed())
    try:
      await asyncio.wait([sender, closed], return_when=asyncio.FIRST_COMPLETED)
    finally:
      sender.cancel()
      closed.cancel()
      self._wakers.discard(wake)
      self.clients -= 1
    if sender.done() and not sender.cancelled():
      e = sender.exception()
      if e is not None and not isinstance(e, ConnectionClosed):
        raise e


def _response(status: int, reason: str, body: bytes, kind: str) -> Response:
  headers = Headers([("Content-Type", kind), ("Content-Length", str(len(body))),
                     ("Cache-Control", "no-store")])
  return Response(status, reason, headers, body)


class ViewServer:
  def __init__(self, hub: ViewHub, meta: dict, positions: np.ndarray, classes: np.ndarray,
               host: str = HOST, port: int = PORT, log: Callable[[str], None] = print):
    self.hub, self.host, self.port, self.log = hub, host, port, log
    files = resources.files("flycraft.view").joinpath("static")
    self.files = {path: (files.joinpath(name).read_bytes(), TYPES["." + name.rsplit(".", 1)[1]])
                  for path, name in STATIC.items()}
    self.files["/meta"] = (json.dumps(meta).encode(), TYPES[".json"])
    pos = np.ascontiguousarray(positions, dtype="<f4")
    self.files["/positions.bin"] = (pos.tobytes(), TYPES[".bin"])
    self.files["/classes.bin"] = (np.asarray(classes, np.uint8).tobytes(), TYPES[".bin"])

  def _http(self, connection: ServerConnection, request: Request) -> Response | None:
    path = request.path.split("?", 1)[0]
    if path == "/activity":
      return None  # the websocket handshake goes on
    if path in self.files:
      body, kind = self.files[path]
      return _response(200, "OK", body, kind)
    return _response(404, "Not Found", b"not found\n", "text/plain; charset=utf-8")

  async def run(self, stop: asyncio.Event, ready: Callable[[int], None] | None = None) -> None:
    self.hub.bind(asyncio.get_running_loop())
    try:
      server = await serve(self.hub.stream, self.host, self.port, process_request=self._http,
                           compression=None, close_timeout=CLOSE_TIMEOUT_S)
    except OSError as e:
      raise ListenError(self.port, e, "view") from e
    async with server:
      port = server.sockets[0].getsockname()[1]
      self.log(f"brain view on http://{self.host}:{port}")
      if ready is not None:
        ready(port)
      await stop.wait()


def build_meta(fly) -> tuple[dict, np.ndarray, np.ndarray]:
  """(meta, positions, class codes) for a Fly: what the page needs once, at load."""
  brain, conn, cfg = fly.brain, fly.brain.conn, fly.cfg
  geom, groups = brain.geom, brain.decoder.groups

  def name(pairs) -> str:
    if not pairs:
      return "none"
    types = sorted({t for t, _ in pairs})
    sides = sorted({s for _, s in pairs})
    return " ".join(types) + ("" if len(sides) > 1 else f" {sides[0]}")

  dec = cfg.decoder
  spec = [
    ("pr_L", "Photoreceptors L", geom.pr_idx[geom.pr_side == "L"], False),
    ("pr_R", "Photoreceptors R", geom.pr_idx[geom.pr_side == "R"], False),
    ("lc10a_L", "LC10a L", geom.vpn_idx[geom.vpn_side == "L"], False),
    ("lc10a_R", "LC10a R", geom.vpn_idx[geom.vpn_side == "R"], False),
    ("turn_neg", name(dec.turn_neg), groups.turn_neg, True),
    ("turn_pos", name(dec.turn_pos), groups.turn_pos, True),
    ("fwd", name(dec.fwd), groups.fwd, True),
    ("mdn", "MDN", groups.mdn, True),
    ("pam", "PAM", conn.select(r"^PAM"), False),
    ("ppl1", "PPL1", conn.select(r"^PPL1"), False),
    ("kc", "KC", conn.select(r"^KC"), False),
    ("mbon", "MBON", conn.select(r"^MBON"), False),
  ]
  names, codes = np.unique(np.asarray(conn.superclass, dtype=str), return_inverse=True)
  if names.size > 255:  # keep the first 255, the rest share the last code
    codes = np.minimum(codes, 255)
    names = np.append(names[:255], "other")
  pos = np.asarray(conn.pos, np.float32).copy()
  bad = ~np.isfinite(pos).all(axis=1)
  if bad.any():
    pos[bad] = np.nanmean(pos[~bad], axis=0) if (~bad).any() else 0.0
  c = fly.calib
  meta = {
    "n": conn.n,
    "classes": names.tolist(),
    "groups": [{"key": k, "label": label, "idx": np.asarray(i).tolist(), "dn": dn}
               for k, label, i, dn in spec],
    "condition": {
      "brain_id": fly.identity.brain_id, "wiring": fly.identity.wiring,
      "scramble_seed": fly.identity.scramble_seed, "plasticity": fly.identity.plasticity,
      "fingerprint": fly.fingerprint, "rung": c.rung, "w_scale": c.w_scale,
      "decoder": c.decoder.to_dict(), "polarity": cfg.eye.polarity,
      "decision_ms": dec.decision_ms, "frame_ms": fly.slices[0],
      "max_dtheta": dec.max_turn_deg_s * dec.decision_frames / dec.game_fps},
    "eye_shape": list(protocol.EYE_SHAPE),
  }
  return meta, pos, codes.astype(np.uint8)
