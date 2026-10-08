"""Experiment: where the oracle loses to scripted. Client-side pilots, no brain.

Not part of flycraft: run on the Windows side with the client's own arguments plus --scripted.

argv[1] is a variant:
  lag            scripted, but moves to the beacon seen one decision earlier
  carrot<N>      moves N px toward the beacon (exact bearing, no turn limit, no lag)
  carrot<N>lag   the same, on the previous decision's bearing
Remaining args go to the client (--episodes, --seed, --mode)."""
import math
import sys

from flycraft.game import client, render
from flycraft.game.body import NOOP, SELECT, Action

V = sys.argv.pop(1)
LAG = V.endswith("lag")
CARROT = float(V[6:-3] if LAG else V[6:]) if V.startswith("carrot") else None


class Pilot(client.ScriptedPilot):
  name = V

  def start(self, episode, seed, phase):
    super().start(episode, seed, phase)
    self.prev = None

  def decide(self, frame, reward):
    self.steps += 1
    if frame.last:
      return NOOP
    marine = render.find(frame.player_relative, render.SELF)
    beacon = render.find(frame.player_relative, render.NEUTRAL)
    seen = (marine.xy if marine else None, beacon.xy if beacon else None)
    use = self.prev if LAG else seen
    self.prev = seen
    if not frame.can_move:
      return SELECT
    if use is None or None in use or marine is None:
      return NOOP
    if CARROT is None:
      return Action("move", use[1])
    (mx, my), (bx, by) = use
    th = math.atan2(by - my, bx - mx)
    x, y = marine.xy  # the carrot hangs from where the marine is now
    return Action("move", (min(max(x + CARROT * math.cos(th), 0), 83),
                           min(max(y + CARROT * math.sin(th), 0), 83)))


client.ScriptedPilot = Pilot
sys.exit(client.main())
