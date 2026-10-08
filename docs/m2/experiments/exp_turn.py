"""Experiment: the client with the body turn limit set from argv[1] (deg/s).

Not part of flycraft."""
import dataclasses
import sys

from flycraft.game import client
from flycraft.game.body import BodyParams

T = float(sys.argv.pop(1))


@dataclasses.dataclass(frozen=True)
class Params(BodyParams):
  max_turn_deg_s: float = T


client.BodyParams = Params
sys.exit(client.main())
