"""What the game server drives (spec section 3): a Controller turns sight into a motor command.

The boundary rule lives in step()'s signature: a controller gets the eye image and the reward
since the last observation, and nothing else from the game. The obs message also carries the
marine and beacon positions, for logging and the view; the server never passes them on, and
tests/test_server.py checks that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flycraft.config import DecoderConfig


class Runaway(RuntimeError):
  """Raised by step() when the brain has run away (spec section 9): the server answers abort
  "runaway" and ends the episode, which counts."""


@dataclass(frozen=True)
class Command:
  dtheta: float  # degrees, positive = clockwise on screen (spec section 7.6)
  speed: float  # 0 to 1
  turn_raw: float = 0.0  # the signals behind the command, for logs and the view
  fwd_raw: float = 0.0


@dataclass(frozen=True)
class Identity:
  """What the brain tells the client in `ready`."""
  brain_id: str
  wiring: str
  scramble_seed: int | None
  plasticity: bool
  polarity: str
  decision_frames: int


class Controller:
  identity: Identity

  def start_episode(self, episode: int, seed: int, phase: str) -> None:
    """Called on episode_start. The default keeps nothing per episode."""

  def step(self, eye: np.ndarray, reward: float) -> Command:
    """One decision: eye is uint8 (30, 72); reward is what the game paid since the last obs."""
    raise NotImplementedError

  def end_episode(self, score: float, steps: int, aborted: str | None) -> None:
    """Called on episode_end, or with aborted set when the episode is cut short."""


def max_turn_deg(dec: DecoderConfig) -> float:
  """The turn limit for one decision: max_turn_deg_s times a decision's game time."""
  return dec.max_turn_deg_s * dec.decision_frames / dec.game_fps
