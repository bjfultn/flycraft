"""Stub brains (spec section 15, M2): no neurons, same protocol and same body as the real one.

- Oracle steers straight at whatever it sees in the eye image and runs at full speed. It
  reads only the eye image, like the real brain, so it tests the whole render-to-body chain
  and sets the ceiling for this eye and body.
- RandomWalker turns at random within the turn limit and walks at s0. It is the floor: a
  brain that moves like the real one but has no information.
"""

from __future__ import annotations

import numpy as np

from flycraft import eye
from flycraft.brain.control import Command, Controller, Identity, max_turn_deg
from flycraft.config import DecoderConfig


def object_azimuth(img: np.ndarray, polarity: str) -> float | None:
  """Contrast-weighted circular mean azimuth of what differs from the background, or None."""
  bg, _ = eye.POLARITY[polarity]
  w = np.abs(img.astype(np.float64) - bg).sum(axis=0)  # contrast per column
  if not w.any():
    return None
  az = np.radians(eye.pixel_az(np.arange(eye.EYE_COLS)))
  return float(np.degrees(np.arctan2((w * np.sin(az)).sum(), (w * np.cos(az)).sum())))


class Oracle(Controller):
  def __init__(self, dec: DecoderConfig, polarity: str):
    self.identity = Identity("stub-oracle", "none", None, False, polarity, dec.decision_frames)
    self.max_turn = max_turn_deg(dec)

  def step(self, eye: np.ndarray, reward: float) -> Command:
    az = object_azimuth(eye, self.identity.polarity)
    if az is None:
      return Command(0.0, 1.0)
    return Command(float(np.clip(az, -self.max_turn, self.max_turn)), 1.0, turn_raw=az)


class RandomWalker(Controller):
  def __init__(self, dec: DecoderConfig, polarity: str):
    self.identity = Identity("stub-random", "none", None, False, polarity, dec.decision_frames)
    self.max_turn = max_turn_deg(dec)
    self.s0 = dec.s0
    self.rng = np.random.default_rng(0)

  def start_episode(self, episode: int, seed: int, phase: str) -> None:
    self.rng = np.random.default_rng([seed, 1])  # a stream apart from the body's heading draw

  def step(self, eye: np.ndarray, reward: float) -> Command:
    return Command(float(self.rng.uniform(-self.max_turn, self.max_turn)), self.s0)


STUBS = {"oracle": Oracle, "random": RandomWalker}
