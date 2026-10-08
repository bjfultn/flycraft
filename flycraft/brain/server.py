"""The brain's game server (spec sections 3 and 10): one game client at a time, over a websocket.

The server owns the protocol and a Controller owns the brain. For each obs the server passes
the eye image and the reward to Controller.step() and answers with an act, including for the
episode's last obs, so the client can always drain one act per obs. marine_xy and beacon_xy
never reach the controller: they go only to the optional trace file, one JSON line per obs,
for checking on-screen behaviour after a run.

A runaway brain (step() raises Runaway, or returns a Command that fails the protocol check:
NaN, speed out of range) gets abort "runaway" in place of the act, and the episode ends, which
counts (spec section 9). The connection stays open for the next episode_start.

Other failures close the connection after one message saying why:
- a second client gets error "busy";
- a wrong protocol version, a malformed message, or a message out of order gets an error;
- stopping the server sends abort "shutdown".
An episode cut short by any of these, or by the client vanishing, is ended in the
controller with aborted set.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable
from typing import TextIO

import numpy as np
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from flycraft import protocol
from flycraft.brain.control import Controller, Runaway

HOST = "127.0.0.1"  # spec section 3: nothing listens on the LAN
PORT = 8765


def _log(text: str) -> None:
  print(f"flycraft brain: {text}", file=sys.stderr, flush=True)


class ListenError(OSError):
  def __init__(self, port: int, err: OSError, name: str):
    super().__init__(f"cannot listen on {HOST}:{port} for the {name} server: {err}")


class _Closing(Exception):
  """The session ends after sending this message."""

  def __init__(self, msg: dict, aborted: str):
    super().__init__(msg)
    self.msg, self.aborted = msg, aborted


class GameServer:
  def __init__(self, controller: Controller, host: str = HOST, port: int = PORT,
               log: Callable[[str], None] = _log, trace: TextIO | None = None):
    self.controller = controller
    self.host, self.port = host, port
    self.log = log
    self.trace = trace
    self._client: ServerConnection | None = None
    self._episode: dict | None = None  # the open episode: episode, score, steps
    self._stopping = False

  async def run(self, stop: asyncio.Event, ready: Callable[[int], None] | None = None) -> None:
    """Serve until stop is set; ready(port) is called once the socket is bound."""
    try:
      server = await serve(self._handler, self.host, self.port)
    except OSError as e:
      raise ListenError(self.port, e, "game") from e
    async with server:
      port = server.sockets[0].getsockname()[1]
      self.log(f"listening on ws://{self.host}:{port} as {self.controller.identity.brain_id}")
      if ready is not None:
        ready(port)
      await stop.wait()
      self._stopping = True
      client = self._client
      if client is not None:
        try:
          await client.send(protocol.encode(protocol.make("abort", reason="shutdown")))
          await client.close()
        except ConnectionClosed:
          pass

  async def _handler(self, ws: ServerConnection) -> None:
    if self._client is not None:
      await ws.send(protocol.encode(protocol.make("error", message="busy")))
      await ws.close()
      return
    self._client = ws
    try:
      await self._session(ws)
    except _Closing as c:
      self.log(f"closing: {c.msg}")
      self._end_episode(c.aborted)
      try:
        await ws.send(protocol.encode(c.msg))
        await ws.close()
      except ConnectionClosed:
        pass
    except ConnectionClosed:
      self.log("game client disconnected")
    finally:
      self._end_episode("shutdown" if self._stopping else "disconnect")
      self._client = None

  async def _session(self, ws: ServerConnection) -> None:
    hello = self._decode(await ws.recv())
    if hello["type"] != "hello":
      raise _Closing(self._error(f"expected hello, got {hello['type']}"), "protocol")
    ident = self.controller.identity
    self.log(f"client {hello['client_version']}: {hello['map']}, {hello['mode']} mode")
    await ws.send(protocol.encode(protocol.make(
      "ready", brain_id=ident.brain_id, wiring=ident.wiring, scramble_seed=ident.scramble_seed,
      plasticity=ident.plasticity, eye_shape=list(protocol.EYE_SHAPE),
      decision_frames=ident.decision_frames, polarity=ident.polarity)))
    async for data in ws:
      msg = self._decode(data)
      kind = msg["type"]
      if kind == "episode_start" and self._episode is None:
        self.controller.start_episode(msg["episode"], msg["seed"], msg["phase"])
        self._episode = {"episode": msg["episode"], "score": 0.0, "steps": 0}
      elif kind == "obs" and self._episode is not None:
        await ws.send(protocol.encode(await self._act(msg)))
      elif kind == "episode_end" and self._episode is not None:
        self._episode = None
        self.controller.end_episode(msg["score"], msg["steps"], msg["aborted"])
      else:
        state = "in" if self._episode is not None else "outside"
        raise _Closing(self._error(f"unexpected {kind} {state} an episode"), "protocol")

  async def _act(self, obs: dict) -> dict:
    eye = np.frombuffer(obs["eye"], dtype=np.uint8).reshape(protocol.EYE_SHAPE)
    record = {"episode": self._episode["episode"], "step": obs["step"],
              "marine_xy": obs["marine_xy"], "beacon_xy": obs["beacon_xy"],
              "reward": obs["reward"], "score": obs["score"]}
    self._episode.update(score=obs["score"], steps=obs["step"] + 1)
    try:
      cmd = await asyncio.to_thread(self.controller.step, eye, float(obs["reward"]))
      act = protocol.make("act", step=obs["step"], dtheta=cmd.dtheta, speed=cmd.speed,
                          turn_raw=cmd.turn_raw, fwd_raw=cmd.fwd_raw)
    except (Runaway, protocol.ProtocolError) as e:
      self.log(f"runaway brain: {e}")
      self._trace({**record, "aborted": "runaway", "why": str(e)})
      self._end_episode("runaway")
      return protocol.make("abort", reason="runaway")
    self._trace({**record, **{k: act[k] for k in ("dtheta", "speed", "turn_raw", "fwd_raw")}})
    return act

  def _trace(self, record: dict) -> None:
    if self.trace is not None:
      self.trace.write(json.dumps(record) + "\n")
      self.trace.flush()

  def _decode(self, data) -> dict:
    try:
      return protocol.decode(data)
    except protocol.ProtocolError as e:
      raise _Closing(self._error(str(e)), "protocol") from e

  @staticmethod
  def _error(text: str) -> dict:
    return protocol.make("error", message=text)

  def _end_episode(self, aborted: str) -> None:
    ep, self._episode = self._episode, None
    if ep is not None:
      self.controller.end_episode(ep["score"], ep["steps"], aborted)
