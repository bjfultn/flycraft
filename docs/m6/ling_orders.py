"""Experiment (M6): do the zerglings do better attacking one marine, or attack-moving? The
client through the fly's body, with the task's cone from argv[1]: 30 is the body before this
experiment (attack the marine within 30 degrees of the heading, else attack-move a step along
it), 0 never attacks a marine directly, only attack-moves, as the body now does. Not part of
flycraft. Run it with the oracle stub serving (flycraft brain --stub oracle) and the client's
own arguments:

  python docs/m6/ling_orders.py 0 --map DefeatMarines --mode train --episodes 20 --seed 1 \
    --step-px 6
"""
import dataclasses
import sys

from flycraft.game import client, tasks

CONE = float(sys.argv.pop(1))
for name in ("DefeatMarines", "Skirmish"):
  tasks.TASKS[name] = dataclasses.replace(tasks.TASKS[name], cone_deg=CONE)
sys.exit(client.main())
