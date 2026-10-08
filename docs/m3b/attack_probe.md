# Why attack-moving a step at a time kills nothing (M3b, 2026-10-08)

The first M3b body attack-moved one step (3 to 6 px) along the fly's heading each decision.
Before the real fly ran, the baselines showed a problem: the scripted agent averaged 116.5 and
the oracle stub 22.2, and the random walker never killed a roach. The real fly's step at full
speed, aimed straight at a roach, did no better: the roaches killed all 9 marines, every
episode, and lost none. Re-issuing a short attack-move every 8 frames appears to keep the
marines walking instead of settling into fire.

[attack_probe.py](attack_probe.py) flies the squad with client-side pilots (no brain) to
separate the two things the order decides: whom the squad attacks, and how far ahead the order
points. Each variant aims at the roach nearest the squad. DefeatRoaches, train mode, seed 1,
10 episodes each (`flycraft-client --scripted` with the pilot swapped in).

| Variant | Order each decision | Mean | Median |
|---|---|---|---|
| nearestpx | Attack_screen on the nearest roach's pixel nearest its middle | 46.5 | 46 |
| nearest | Attack_screen on the nearest roach's middle | 42.0 | 46 |
| carrot3 | attack-move 3 px toward the nearest roach | -9.0 | -9 |
| carrot6 | attack-move 6 px toward it | 23.5 | 21 |
| carrot10 | attack-move 10 px toward it | 21.0 | 21 |
| carrot20 | attack-move 20 px toward it | 32.0 | 36 |
| carrot40 | attack-move 40 px toward it | 12.5 | 11 |
| carrot6move | move 6 px toward it (no attacking on the way) | -3.0 | 1 |
| away6 | attack-move 6 px directly away from it | 0.0 | 0 |

For reference, over 20 episodes: pysc2's scripted agent (attacks the lowest roach pixel, so the
whole squad focuses one roach) 116.5, median 71; the oracle stub attack-moving 22.2; the
random walker -8.9.

What it shows:

- A short attack-move aimed straight at a roach still kills nothing (carrot3, the real fly's
  step at full speed). Longer steps help a little and stop helping by 40 px.
- An attack ordered on the roach itself (nearest, nearestpx) roughly doubles the best
  attack-move, and aiming at a pixel of the roach rather than its middle is a little better.
- Focusing the squad on one roach (scripted) is better again. The roaches stay put until
  engaged (away6 scores 0), so the score measures how well the squad focuses fire.

So the demo body attacks a roach directly: when one is within 30 degrees of the fly's heading,
the squad attacks the one nearest the heading, at its pixel nearest its middle; otherwise it
attack-moves a step along the heading as before. The fly picks its target by turning toward it.
The 30 degrees is a demo setting chosen before any run, not a claim about the fly.

## Correction: the short steps were partly the squad shooting itself

SC2 takes an attack aimed at one of the player's own units as an order to shoot it. The carrot
variants aimed from the squad's middle, and a step of a few pixels from there lands inside the
squad; the first watch run showed marines firing on each other. So carrot3 measured friendly
fire as much as the attack-move. `carrot<N>clear` moves the order on along the same line to the
first pixel at least 2 px from every marine, as the body now does (and walks instead when no
clear ground lies ahead on screen). Same setup, 10 episodes each:

| Variant | Mean | Median |
|---|---|---|
| carrot3 | -9.0 | -9 |
| carrot3clear | 19.5 | 23.5 |
| carrot6 | 23.5 | 21 |
| carrot6clear | 23.5 | 21 |

With the step past the squad, the real fly's 3 px step scores about what the 6 px step does;
carrot6 is unchanged. The first bullet above is wrong: a short attack-move does kill roaches
once it stops landing on the squad. The rest stands: an attack on the roach itself (42 to 46.5)
still roughly doubles the best attack-move, and focus fire (scripted) is better again. The
reference baselines above used the body before both fixes; the spec has the current ones.
