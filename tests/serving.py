"""Run a GameServer on a free port in a background thread, for tests that act as the client."""

from __future__ import annotations

import asyncio
import contextlib
import threading

from flycraft import protocol
from flycraft.brain.server import GameServer

EYE = bytes(protocol.EYE_BYTES)


@contextlib.contextmanager
def serving(controller, trace=None):
  """Yield (url, server); stopping sends any connected client abort "shutdown"."""
  loop = asyncio.new_event_loop()
  server = GameServer(controller, port=0, log=lambda text: None, trace=trace)
  stop = asyncio.Event()
  bound: list[int] = []
  up = threading.Event()

  def ready(port):
    bound.append(port)
    up.set()

  thread = threading.Thread(target=loop.run_until_complete, args=(server.run(stop, ready),),
                            daemon=True)
  thread.start()
  assert up.wait(10), "server did not start"
  try:
    yield f"ws://127.0.0.1:{bound[0]}", server
  finally:
    loop.call_soon_threadsafe(stop.set)
    thread.join(10)
    loop.close()


def hello(mode="train"):
  return protocol.make("hello", client_version="test", map="MoveToBeacon", screen=84, mode=mode)


def obs(step, eye=EYE, last=False, marine_xy=(40.0, 40.0), beacon_xy=(60.0, 40.0), reward=0.0):
  return protocol.make("obs", step=step, eye=eye, reward=reward, score=0.0, last=last,
                       marine_xy=list(marine_xy), beacon_xy=list(beacon_xy))
