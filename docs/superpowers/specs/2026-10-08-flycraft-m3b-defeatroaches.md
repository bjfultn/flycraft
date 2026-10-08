# flycraft M3b: the DefeatRoaches demo

- Date: 2026-10-08
- Parent spec: [2026-10-07-flycraft-design.md](2026-10-07-flycraft-design.md), section 15 (M3b)
- Status: a demo, not an experiment. Nothing here is pre-registered, and no claim about the fly
  rests on it.

## 1. Goal

A marine squad turns on the roaches and attacks, steered by the same untrained brain that walks
to beacons in M3. The exit check is BJ watching it. It is a reflex, not tactics.

## 2. The minigame

DefeatRoaches starts 9 marines against 4 roaches. Each time all 4 roaches die, 4 new roaches and
5 more marines arrive and both sides are put back on opposite sides of the map. Each roach killed
scores +10 and each marine lost scores -1. An episode ends at 120 seconds or when every marine is
dead.

pysc2's scripted agent attacks the enemy pixel lowest on the screen (the first one in its scan)
and selects the army when it cannot attack.

## 3. What changes

The brain, the protocol and the decoder do not change. The fly still sends only a turn and a
speed each decision.

- **The eye sees every roach.** The screen layer's enemy pixels are split into 8-connected
  patches, one per roach (roaches that touch on screen make one patch). Each patch is drawn as
  its own looming disk, as the beacon is in M3; where disks overlap, a pixel takes the larger cover.
  One target draws exactly the M3 image, so MoveToBeacon is unchanged.
- **The squad attacks the roach the fly faces.** When a roach is within 30 degrees of the
  fly's heading, the body orders `Attack_screen` on the roach nearest the heading, at that
  roach's pixel nearest its middle (so the order lands on a unit even when touching roaches
  make one patch). It does so at any speed. Otherwise it orders `Attack_screen` one step ahead
  along its heading, an attack-move, at the first screen pixel on from there that is at least
  2 px from every one of the squad's own pixels. SC2 takes an attack aimed at one of the
  player's own units as an order to shoot it, and a step of a few pixels from the squad's middle
  lands on its own marines. With no clear ground ahead before the screen ends, the body orders a
  move to the step instead. The fly picks its target by turning toward it. The
  30 degrees is a demo setting chosen before any run, not a claim about the fly.
- **Why not attack-move only.** The first version attack-moved a step at a time, and the
  baselines showed it barely fires: a 3 px step (the real fly's) killed no roach in 10
  episodes. Attacking the roach itself does. The probe and its numbers are in
  [docs/m3b/attack_probe.md](../../m3b/attack_probe.md). Its short attack-moves were aimed
  from the squad's middle, so part of what they measured is the squad shooting itself (the
  first watch run showed it); the probe's correction has the numbers with the step moved past
  the squad. In this minigame the score largely measures focus fire, so it says more about the
  order than about the fly's steering.
- **No attack output.** Section 15 planned an attack DN group gating `Attack_screen` on the
  object nearest the center of the frontal visual field. The demo keeps that target choice and
  replaces the gate with the cone: choosing the DN group needs a pre-registered criterion. The
  gate returns with the experiment version of the minigame.
- **The squad is one body.** Its position is the centroid of the player's own pixels, as the
  single marine's was. The trace's `beacon_xy` is the roach nearest that centroid.
- **Reinforcements are selected.** The new marines arrive unselected. Until the selection
  covers the whole army (`army_count`), the body selects the army instead of ordering, so every
  order reaches every marine.

`--map DefeatRoaches` picks the task: its target value (enemy), its order (attack), whether
targets are split, and its scripted agent. `--map MoveToBeacon` is the default and behaves as in
M3.

## 4. Baselines

All on `--map DefeatRoaches`, train mode, seed 1:

| Who | What it does |
|---|---|
| scripted | pysc2's agent: attacks the lowest roach on screen |
| random | the random-walker stub brain, with the same body (attacks inside the cone) |
| oracle | the oracle stub brain, same body: turns toward the contrast-weighted mean azimuth of everything in the eye. With several roaches in view it aims between them, not at the nearest, so it is a weaker ceiling here than on MoveToBeacon |
| real fly | the M3 brain (vpn rung, w_scale 0.84375), untrained, plasticity off |

Results, 20 episodes each (the body with the attack step past the squad; scripted does not use
the body, so its number is from the first run):

| Who | Mean | Median |
|---|---|---|
| scripted | 116.5 | 71 |
| oracle | 38.2 | 36 |
| real fly | 20.9 | 21 |
| random | 15.2 | 21 |

The real fly's mean hides a clean split on where each episode started it. In the 11 episodes
that started it within 108 degrees of the nearest roach it turned (largest turn 16 to 40
degrees an episode), engaged and scored 38.0 on average, about the oracle's mean. In the 9 that
started it more than 123 degrees away it made no turn at all, walked to the edge, and scored 0.

Then a watch-mode run of the real fly for BJ, with the brain view.

**The watch run starts the fly facing the roaches** (`--face 60`). The first watch run started
each episode at a random heading, as the baselines do. The brain turns only toward something in
its view, so a fly started facing away walked to the edge and sat (its trace: a turn of exactly
0 on every one of the episode's 241 decisions), and 5 of 6 episodes showed nothing. With
`--face DEG` the body turns, once, at the first sight of the squad and a roach, to a heading
within DEG degrees of the nearest roach (the offset fixed by the episode seed). From there the steering, the choice of roach and the attacks
are the fly's. It is a demo setting, chosen before the run and said in the video's caption; the
baselines keep the random heading. A fly that looks for things it cannot see would need
spontaneous activity the model does not have.

## 5. Tests

The fake game draws enemy disks and accepts `attack` like `move`, except an attack aimed at the
marine itself, which fails the test. So the client tests run the whole loop without SC2: the eye
shows every roach, the body sends `attack` almost always (a `move` only when no clear ground lies
ahead), the squad never attacks its own marines,
an order waits until the whole army is selected, and the scripted agent aims at the lowest roach.
The strike has its own tests: it picks the roach nearest the heading (not the nearest roach),
only inside the cone, on a roach pixel even when a patch's middle is off it, at any speed,
never on MoveToBeacon, and a fly that turns a roach into the cone attacks it.
`--face` has its own: the heading lands within the cone of the target, fixed by the seed, is
set every episode before the brain's first look, and the client refuses it outside 0 to 180
degrees or with `--scripted`.
