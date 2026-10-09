# M6 checks, 2026-10-08

Before a person plays the fly, two matches on one Windows PC checked the whole path: Skirmish
over LAN in real time, two SC2 processes, the brain on the WSL GPU (RTX 5080). The person's
side was `stand_in.py`, a scripted stand-in: every 11 game frames all 4 marines shoot the
zergling nearest their center. Real MaleCNS v1.0 wiring with the M1 calibration (`vpn` rung,
w_scale 0.8438), plasticity off, nothing trained.

```
# WSL: the brain
flycraft brain --calibration docs/m1/calibration.json --trace runs/m6/trace.jsonl
# Windows: the stand-in hosts, then the fly joins
python docs/m6/stand_in.py
flycraft-client --brain ws://127.0.0.1:8765 --map Skirmish --join 14380
```

## Result

| The fly | Rounds (fly to stand-in) | Game frames | Wall time | Fly decisions |
|---|---|---|---|---|
| pysc2's scripted agent (`--scripted`) | 3 to 0 | 965 | 42 s | 122 |
| the connectome brain | 0 to 3 | 1064 | 47 s | 73 |

- The scripted agent (pysc2's for DefeatRoaches) sends all 9 zerglings at one marine, the
  one lowest on the screen, every decision. Focus fire did not hold that off.
- The brain lost every round, and took at least one marine each round. It decided 73 times in
  1064 frames, one decision every 14.6 frames (about 650 ms; M3 measured about 430 ms with one
  SC2 on the PC). The longest wait for the brain was 750 ms, and no frame was late. It turned
  on 66 of its 73 decisions, by 33 degrees on average.
- A third match left the person's marines alone: their own return fire beat the brain 3 to 0
  (2679 frames, 117 s, 181 decisions).
- `trace-brain-vs-standin.jsonl` is the brain's trace of the second match, one line per decision.
  The trace keeps M3's field names: `marine_xy` is the fly's squad and `beacon_xy` its target.

Both sides saw the same result: the person's side printed `Victory` (or `Defeat`) and the
fly's record carried `outcome` -1 (or 1).

## Friendly fire

Watching a match, the zerglings kept attacking each other. The cause was the clock: in real
time SC2 keeps going while the brain thinks, and the client waited for each act (about 640 ms)
and then aimed the order from the frame the brain had been shown, about 15 game frames old. By
the time SC2 got an order to attack-move a step from the squad, the squad had often walked onto
that spot, and an attack order on a pixel holding a zergling is "attack this zergling". Now, in
real time, the client plays no-ops until the act is in and then decides on the newest frame
(`play_episode(..., realtime=True)`, spec section 7.5). Lockstep modes are unchanged.

Two probe matches against the same stand-in, one before the fix and one after. `ff_host.py` is
the stand-in plus a count of who hit each zergling; `ff_probe.py` is the fly's client plus a
record per decision (`ff-before.jsonl`, `ff-after.jsonl`; summarize with `ffstats.py`).

| | Before | After |
|---|---|---|
| Rounds (fly to stand-in) | 3 to 2 | 3 to 0 |
| Game frames, fly decisions | 1553, 103 | 967, 64 |
| Age of the aiming frame when the order goes out, median (max) | 15 (19) frames | 1 (1) |
| Longest wait for the brain | 782 ms | 469 ms |
| Attack orders on the fly's own units when aimed / when SC2 got them | 0 / 11 of 95 | 0 / 0 of 62 |
| Zergling hits on zerglings | 31 | 0 |
| Marine hits on zerglings | 167 | 96 |

At the before rate, a match as long as the after one would have had about 19 zergling hits.
No frame was late in either. The rounds are one match each, and the brain lost 0 to 3 to this
stand-in in the first check above, so the score says nothing about the fix.

### The fly's own zerglings, on DefeatMarines

Both probes above are matches against the stand-in. On the solo map, `ff_lings.py` counts the
same way: the client plus, for each of the squad's zerglings, who hit it, read from its health
drop (zerglings deal 5 a hit and marines 6). The real fly, 10 train episodes, seed 1, with
attack-move zerglings and the stall stop on. `ffl-fly.jsonl` is the record,
`ffl-fly-hits.jsonl` the counts per episode, `trace-ffl-fly.jsonl` the brain's trace.

```
flycraft brain --calibration docs/m1/calibration.json --trace runs/m3/trace-ffl-fly.jsonl
python docs/m6/ff_lings.py runs/ffl-fly-hits.jsonl --mode train --episodes 10 --seed 1 \
  > runs/ffl-fly.jsonl
```

| Episodes | Fought | Zergling hits on zerglings | Marine hits on zerglings | Unclear |
|---|---|---|---|---|
| 10 | 4 (scores 146, 141, 106, 109) | 0 | 361 | 0 |

The other six began with the target behind the fly and were stopped as stalled (see "Where
the fly looks"). Surviving zerglings keep their damage into the next wave (the map's rule,
taken from DefeatRoaches), so a zergling at yellow or red health before it reaches the marines
was hurt in the last wave, not by its own side.

## Zergling orders

The body attacked the marine nearest the heading, if it was within 30 degrees, and
attack-moved a step along the heading otherwise: the rule M3b chose for marines against
roaches. Zerg players do the opposite with zerglings: they surround and fight whatever they
meet, since melee units sent at one marine queue up behind each other. `ling_orders.py` ran the oracle
stub through the body on DefeatMarines with the cone at 30 (as it was) and at 0 (attack-move
only), at three step lengths. Train mode, seed 1, 20 episodes each, single SC2:

```
flycraft brain --stub oracle --view-port 0
python docs/m6/ling_orders.py 0 --map DefeatMarines --mode train --episodes 20 --seed 1 --step-px 6
```

| Orders | Step (px) | Mean | Median | SD | Range |
|---|---|---|---|---|---|
| target within 30 degrees, as it was | 6 | 189.1 | 146 | 121.8 | 26 to 359 |
| attack-move only | 6 | 397.8 | 402 | 19.2 | 358 to 442 |
| attack-move only | 20 | 405.1 | 403 | 32.6 | 317 to 444 |
| attack-move only | 40 | 69.2 | 46 | 53.3 | 11 to 176 |

For scale, M3c's DefeatMarines baselines (same map, mode, seed and count): pysc2 scripted 323,
oracle 207, random 130, real fly 62. Attack-move doubled the oracle and beat pysc2's
scripted agent, which focus-fires the lowest marine. A 40 px step sends the squad past the
fight and loses it.

**Ruling:** the Zerg tasks (DefeatMarines, Skirmish) now only attack-move, step unchanged at 6
px. DefeatRoaches keeps the 30 degree cone, which won the same comparison for marines in M3b.
This is a change to the body, not to the brain, and the video says so. The records are
`lings-cone<cone>-step<px>.jsonl`; M3c's DefeatMarines baselines were run before it and are
rerun with it.

## DefeatMarines reruns

M3c's DefeatMarines baselines (`mc1`), rerun after the zerglings went to attack-move only
(`mc2`). Same map, mode, seed and count: train, seed 1, 20 episodes, single SC2. pysc2's
scripted agent gives its own orders, not the body's, so it was not rerun. The records are
`mc1-<pilot>.jsonl` and `mc2-<pilot>.jsonl`.

| Pilot | Before: mean (median, SD, range) | After: mean (median, SD, range) |
|---|---|---|
| pysc2 scripted | 322.9 (329.5, 64.9, 56 to 366) | not rerun |
| oracle | 206.9 (198.5, 123.2, 26 to 361) | 397.8 (402, 19.2, 358 to 442) |
| random | 129.6 (143, 60.6, 35 to 220) | 147.0 (165, 59.1, 35 to 225) |
| the fly | 61.5 (62.5, 58.1, 0 to 174) | 58.0 (0, 78.6, 0 to 217) |

The oracle's rerun matches the attack-move row above episode for episode. The fly scored 0 in
7 episodes before and 11 after. Those zeros are not the change: they are the episodes that
began with the target behind the fly (next section). The seed fixes the starting heading but
not where the target is at the start, so how many episodes begin that way varies from run to
run. Over the episodes that began with the target in front, the fly's mean went from 94.6 (13
episodes) to 128.8 (9).

## Where the fly looks

`start_bearing.py` replays the body's heading through the fly's three DefeatMarines traces
(`trace-mc1-fly.jsonl`, `trace-mc2-fly.jsonl` and the friendly-fire run's
`trace-ffl-fly.jsonl`, below) and finds where the target was when each episode began: its
azimuth at the first decision that saw both the squad and a target, 0 dead ahead and 180
behind.

```
python docs/m6/start_bearing.py docs/m6/trace-mc1-fly.jsonl docs/m6/trace-mc2-fly.jsonl \
  docs/m6/trace-ffl-fly.jsonl --episodes
```

| Starting target azimuth, either side (degrees) | Episodes | Never turned | Scored 0 | Mean score |
|---|---|---|---|---|
| 0 to 30 | 4 | 0 | 0 | 101.5 |
| 30 to 60 | 11 | 0 | 0 | 111.7 |
| 60 to 90 | 5 | 0 | 0 | 141.6 |
| 90 to 115 | 7 | 1 | 1 | 78.3 |
| 115 to 125 | 3 | 3 | 3 | 0 |
| 125 to 150 | 13 | 13 | 13 | 0 |
| 150 to 180 | 7 | 7 | 7 | 0 |

The split is nearly sharp. Every episode that began with the target within 110 degrees of the
heading turned and fought. From 110 to 116 degrees it went either
way: three turned (two at 111.6 degrees, one at 114.6) and two never did (110.8 and 115.3).
Every one beyond that never turned at all. In all 24 that never turned, the brain's turn and
forward outputs were exactly 0 on every decision, and the squad walked to the map edge and
stood there. Those 24 are all 24 of the fly's zero scores. (The friendly-fire run had the
stall stop on, so its six were stopped after 33 to 90 decisions; the other two runs' went the
full 241.)

This is M3's "blind straight behind" again, and sharper. The brain's neurons do not fire on
their own, so with nothing it responds to in view its descending neurons are silent, and
nothing in the eye sees a wall. Why the edge is near 115 degrees is not settled. M3 put it
down to LC10a looking forward and to the sides, but the retina map (`docs/m1/retina.png`) has
LC10a centres out to about 160 degrees on each side, and M1 measured the vpn rung's turn only
at 0 and 60 degrees either way. A target swept around the fly would settle it. Learning on the
path from LC10a to DNa02 cannot help an episode in which that path never fires.

## Search turns

A real fly does not sit still when it sees nothing: it makes turns of its own, unprompted.
Ours does not, since its neurons do not fire without input. So the body now does it for the
brain: `flycraft-client --search N` turns the body as far as one decision allows (about 64
degrees) once the brain has asked for no turn for N decisions in a row, one way, picked from the
episode seed and kept until the brain turns. Any turn from the brain ends the search. This is a
rule in the body, not something the brain does, and the video says so. It is off by default.

The fly on DefeatMarines with `--search 5`, everything else as `mc2-fly`: train, seed 1, 20
episodes, single SC2, stall stop off (`srch5-fly.jsonl`, trace `trace-srch5-fly.jsonl`).

```
flycraft brain --calibration docs/m1/calibration.json --trace runs/m3/trace-srch5-fly.jsonl
flycraft-client --mode train --episodes 20 --seed 1 --map DefeatMarines --stall 0 --search 5
python docs/m6/start_bearing.py docs/m6/trace-srch5-fly.jsonl --episodes --search 5
```

| Pilot | Mean | Median | SD | Range | Scored 0 |
|---|---|---|---|---|---|
| the fly, search turns | 155.9 | 153 | 43.5 | 36 to 249 | 0 of 20 |
| the fly (`mc2-fly`) | 58.0 | 0 | 78.6 | 0 to 217 | 11 of 20 |
| random (`mc2-random`) | 147.0 | 165 | 59.1 | 35 to 225 | 0 of 20 |
| oracle (`mc2-oracle`) | 397.8 | 402 | 19.2 | 358 to 442 | 0 of 20 |

| Starting target azimuth, either side (degrees) | Episodes | Never turned | Scored 0 | Mean score |
|---|---|---|---|---|
| 30 to 60 | 2 | 0 | 0 | 181.0 |
| 60 to 90 | 4 | 0 | 0 | 191.2 |
| 90 to 115 | 1 | 0 | 0 | 145.0 |
| 115 to 125 | 1 | 0 | 0 | 160.0 |
| 125 to 150 | 8 | 0 | 0 | 143.9 |
| 150 to 180 | 4 | 0 | 0 | 133.8 |

Thirteen episodes began with the target behind the fly, where `mc2-fly` scored 0 every time.
All 13 fought, with a mean of 142.0. It took little: the only silent stretches came at the start
of an episode, and once the brain first turned it asked for a turn on every decision after that.
Ten episodes needed one search turn, five none, three two (the brain's first turn came 0 to 11
decisions in). Two, started at +130 and +140 degrees, needed five, 25 decisions, which looks
like the long way round. Front
starts also met a search turn when the brain was slow to start, so the seven that began in front
(mean 181.7) are not a clean comparison with `mc2-fly`'s nine (128.8).

So the fly now plays about as well as the random stand-in, and well short of the oracle, which
turns straight at the nearest marine. The stall stop would have taken none of these episodes
(`stall_check.py`, 25 decisions).

## The flying fly

The walking fly sees the ground around it, from its own heading, and can only turn left or
right. A fly can also fly, and SC2 is played from above, so `flycraft-client --body fly` makes
it a flying fly that looks down on the map as it flies over it.

What it sees (`render.map_eye`): the map as a fly 10 px over it would see it. Each target is a
disk at its azimuth from the fly's heading, as the walking fly sees it, and up from the horizon
by atan(10 / distance): 45 degrees at 10 px, 27 at 20, 7 at 80. So the eye holds the map in
polar form, which way a target is and how far. A disk's radius is at least 10 degrees, as in
M1's calibration, so even a far target is plain to see, and more when a target is close enough
to look bigger. The ground would be below the horizon; it is drawn above it because this brain
hardly answers anything below (the up-down scan, below). A target behind the fly is behind it
here too, so `--search` and `--face` work as for the walking fly.

How it steers: as the walking fly does. The brain's one turn output (DNa02 left against right)
turns the heading, and the squad flies along it at the forward output's speed.

### First try: north up

The first flying fly saw the map north up, east to the right, the squad at the centre of the eye
and a screen's width spanning 60 degrees. With one turn output it had to take turns: on even
decisions it saw the map north up and a turn right meant east; on odd ones it saw the map
turned a quarter, and a turn right meant north. Each turn set half the course, and under 0.05
of the turn limit both ways it kept its last order.

The fly on DefeatMarines, as `srch5-fly` without search turns: train, seed 1, 20 episodes,
stall stop off (`fly1-fly.jsonl`, trace `trace-fly1-fly.jsonl`).

| Pilot | Mean | Median | SD | Range | Scored 0 |
|---|---|---|---|---|---|
| the flying fly, north up | 56.4 | 38 | 44.9 | 0 to 148 | 5 of 20 |
| the fly, search turns (`srch5-fly`) | 155.9 | 153 | 43.5 | 36 to 249 | 0 of 20 |
| the fly (`mc2-fly`) | 58.0 | 0 | 78.6 | 0 to 217 | 11 of 20 |
| random (`mc2-random`) | 147.0 | 165 | 59.1 | 35 to 225 | 0 of 20 |

- Its turns hardly followed the target: a fit of the turn on the target's place in the eye
  explains 3 percent of it, and for targets more than 5 degrees off centre it turned the right
  way 33 percent of the time.
- The brain's turn changes slowly (its correlation from one decision to the next is 0.79, and
  0.11 ten decisions on), so the two alternating views smeared into one.
- It leaned left (mean turn -2.3 degrees), which the course read as a steady drift south-west:
  45 percent of its time in that quarter of the screen.
- Most turns were tiny (median 0.7 degrees, under the 3.2 that keeps the last order), so it
  kept its last order 36 to 98 percent of the time.
- The target starts about 45 px east or west of the squad. All five that scored 0 began with
  it east (5 of the 8 that did), which the quarter-turned view puts below the horizon. All 12
  that began with it west scored.

The walking fly's brain has the same slow turn (0.81) and left lean, so these are the running
brain's, not the flying body's.

### Second try, dropped before it ran: the map turned with the fly

Next was the same flat map turned with the fly, ahead up, so it could steer with its one turn
output like the walking fly. A check on the stand-in game (`tests/fake_sc2.py`) stopped it: the
oracle, which turns by where the target is left or right in the eye and nothing else, scored
1.1 a 480-frame episode on it, against 6.1 walking. On a flat map a target behind the fly sits
dead centre, like one ahead, so it never turns round. The brain reads its eye the same way
(the up-down scan found nothing that reads up and down), so the view above keeps the azimuth
true and puts the distance in the elevation. The oracle scores 6.1 on it.

## All the way round, and up and down

`docs/m6/look_sweep.py` shows the brain M1's calibration spot (radius 10 degrees) from rest,
6 trials a place: round the horizon every 15 degrees (and at 110 and 115), and dead ahead from
60 degrees down to 60 up every 15. It keeps the decoder's turn, how many LC10a the spot drives
and how many descending neurons fire (`look_sweep.json`; every DN's rate per trial in
`look_sweep_rates.npz`).

Round the horizon (turn: negative is left):

| azimuth | -165 | -150 | -135 | -105 | -60 | -30 | 0 | 30 | 60 | 75 | 90 | 105 | 120 | 150 | 165 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| turn | -0.1 | -2.9 | -10.1 | -18.1 | -22.0 | -8.7 | -1.4 | +10.5 | +5.3 | +16.4 | +2.2 | +0.4 | -3.3 | -1.5 | 0 |
| LC10a driven, L/R | 3/0 | 6/0 | 8/0 | 11/0 | 16/0 | 12/6 | 22/30 | 7/18 | 0/16 | 0/17 | 0/13 | 0/13 | 0/8 | 0/5 | 0/2 |

- A spot on the left turns the fly left all the way back to 150 degrees, hard from 45 to 135.
- The right side is weaker, and from 110 degrees back the fly turns left, the long way round.
  That is why the search-turn episodes that began with the target at 130 or 140 degrees right
  needed 5 search turns.
- Behind it (165 and 180) a spot drives at most 3 LC10a and nothing happens: the LC10a
  centres reach 161 degrees on the left and 154 on the right.
- Dead ahead it leans slightly left.

The walking fly's blind spot behind it in the game (about 115 degrees either side) is wider
than this. The game's targets start 2 to 3 degrees across, not 10, which may be why.

Up and down, dead ahead:

| elevation | -60 | -45 | -30 | -15 | 15 | 30 | 45 | 60 |
|---|---|---|---|---|---|---|---|---|
| turn | -0.2 | -1.8 | -0.1 | 0 | -2.0 | -5.9 | -5.7 | +0.4 |
| DNs firing | 209 | 448 | 137 | 175 | 472 | 440 | 458 | 508 |

Below the horizon the fly hardly turns, although the spot drives as many LC10a as anywhere
(22 to 29 a side at -15 and -30). Comparing every DN above against below, 181 pass (|z| over
4.12, at least 1 Hz apart), and 175 of them fire more above. They are not neurons that follow
the spot up and down:

- They fire as much with the spot on the horizon (18 Hz on average, within 60 degrees) as
  above it (19 Hz), and hardly at all below (1.6 Hz).
- Their rate follows how many DNs fire at all (r 0.81 over all 216 trials).
- They come in pairs that agree, left and right alike (DNg11, DNp41, DNg36_a).
- The brain, from rest, either lights up (over 200 DNs firing) or does not: in 21 of 24
  trials above the horizon, 141 of 168 round it, and 1 of 24 below.

So the scan found the brain switching on or not, and no descending neuron that says how far
up a target is. That is why the flying fly's view keeps a target's azimuth true and draws the
map above the horizon. The grid below, which looks off to the sides as well, found a pair that
says up or down.

### Off to the sides: the grid

`docs/m6/grid_sweep.py` shows the same spot at every azimuth and elevation from -60 to 60
degrees in steps of 20, 4 trials each, and a 3 degree spot round the horizon
(`grid_sweep.json`, every DN's rate per trial in `grid_sweep_rates.npz`). It took 50 minutes on
the GPU.

The turn (negative is left), rows from 60 degrees up to 60 down:

| el \ az | -60 | -40 | -20 | 0 | 20 | 40 | 60 |
|---|---|---|---|---|---|---|---|
| 60 | +2.1 | +0.5 | -0.2 | -1.7 | -3.6 | -3.1 | -8.5 |
| 40 | -5.6 | -3.3 | -5.4 | -6.3 | +3.7 | -0.2 | -1.5 |
| 20 | -9.7 | -2.3 | -7.0 | -2.7 | +0.2 | +2.1 | +15.2 |
| 0 | -14.8 | -13.6 | -6.3 | -0.3 | +1.1 | +16.3 | +8.5 |
| -20 | -22.7 | -13.6 | -1.4 | 0 | +0.7 | +11.2 | +7.9 |
| -40 | -6.7 | -6.5 | -0.9 | -2.0 | +1.5 | +11.3 | +14.5 |
| -60 | -6.4 | -14.7 | -5.2 | +0.4 | +13.9 | +17.0 | +25.2 |

- Below the horizon and off to the side the fly turns as well as on the horizon. Only dead
  ahead below is it quiet, which is all the scan above looked at.
- From 40 degrees up the turn fades and mostly goes left, whichever side the spot is on.
- The 3 degree spot turns it the right way from 60 to 105 degrees either side (-5.5 and +3.0 at
  60, -3.7 and +7.5 at 90), often with DNa02 firing alone and the rest of the brain dark. At
  30 degrees, and from 120 back, it does nothing at all, though at 30 it drives 9 or 10 LC10a.
- How many LC10a the retina drives is the same for spots of radius 2, 3 and 5 (0 to 36 round
  the horizon), and up to 52 at 10.

Up and down. Every DN type's rate, fitted on elevation and azimuth (`pitch_scan.py`,
`pitch_scan.json`): two types with plenty of spikes follow elevation in opposite directions, the
same on both sides of the brain (Hz, mean over azimuth):

| elevation | -60 | -40 | -20 | 0 | 20 | 40 | 60 |
|---|---|---|---|---|---|---|---|
| DNbe007, L / R | 11 / 15 | 12 / 16 | 17 / 18 | 36 / 38 | 38 / 42 | 40 / 43 | 28 / 31 |
| DNge043, L / R | 33 / 29 | 40 / 37 | 25 / 18 | 24 / 16 | 16 / 10 | 12 / 6 | 12 / 7 |

- DNbe007 against DNge043, less its mean on the horizon, has the right sign at 88 percent of
  the grid's places 20 degrees or more up or down (76 percent of single trials). DNa02 right
  against left has the right sign at 83 percent of the places 20 degrees or more left or right
  (74 percent of trials).
- It tells up from down, not how far. Below the horizon every place is negative, at every
  azimuth, but -40 and -60 read alike. A ridge readout of all 692 DNs that fire, fitted
  without the stimulus it predicts, gets 66 percent of the elevation's variance (41 percent of
  the azimuth's) and nothing of where below the horizon a spot is.
- The flight DNs that set the wings' stroke (DNg02) follow elevation too, but at under 1 Hz.

### The compass fly

The flying fly has one output, DNa02 right against left, and only ever turns toward where it
already faces. The grid above found a second one, DNbe007 against DNge043, that reads up or
down. `flycraft-client --body compass` steers by both at once: the turn output sets east/west,
the up/down one sets north/south, so the squad can head any way on the map without working
round to it a turn at a time.

What it sees (`render.compass_eye`): the map north up, the squad at the centre, independent of
heading. Each target sits at its screen offset from the squad, east and north scaled straight
into azimuth and elevation by the same 60 degrees a screen's width as the first flying fly used
(`COMPASS_DEG_PER_PX`). A disk is at least 10 degrees in radius, as elsewhere, and bigger when
the target is close enough to look it.

How it steers: raw = mean(DNbe007 L, R) - mean(DNge043 L, R); from the grid, its mean on the
horizon is 17.06 Hz and the 95th percentile of |raw - 17.06| is 51.8 Hz, so pitch =
clip((raw - 17.1) / 50.0, -1, 1), positive up and north. With v_e the turn output over the turn
limit and v_n the pitch, the heading is atan2(-v_n, v_e) in the screen frame (north is heading
-90), set directly rather than turned toward, so both outputs land at once instead of one
decision after the other. Below 0.05 on both axes together it keeps its last heading rather
than spin on noise near zero. There is no search and no face for a compass fly; it has no
turning to search with, and it sets its own heading from the first decision on.

Results on DefeatMarines, run the same way as the flying fly's above (20 train episodes, seed
1, stall off; `cmp1-fly.jsonl`, trace `trace-cmp1-fly.jsonl.gz`), and four watch episodes
recorded for video (`cmprec-fly.jsonl`, seed 3, scores 74, 110, 102, 36):

| pilot | body | score, mean of 20 (sd) |
|---|---|---|
| real fly | compass | 88.2 (73.1) |
| real fly | compass, one dot | 87.7 (55.5) |
| real fly | compass, one dot, pitch flipped | 54.9 (36.0) |
| real fly | compass, half speed | 84.2 (58.8) |
| real fly | walk, search 5 | 155.9 |
| real fly | fly (first flying fly) | 56.4 |
| random | walk | 147.0 |
| oracle | walk | 397.8 |

It does better than the first flying fly and worse than walking, and worse than a random
walker. The scores split: half the episodes 46 or less (two of them 0), seven from 137 to 252.

Why it is no better: in the game, neither output points at the targets. `compass_trace.py` puts
each decision's target where the compass eye draws it and asks whether the outputs have its
sign. Over the 20 episodes, pitch has the target's elevation sign 48% of the time (|el| > 15)
and the turn output its azimuth sign 43% (|az| > 15); the grid had them right for 88% and most
of its stimuli. The raw pitch sits near the grid's horizon value when the target is level
(19.6 Hz against 17.1) but drops for targets above and below alike (9.8 Hz at 10 to 30 up, 12.3
at 10 to 30 down), so it is reading how far off level the target is, not which way. The turn
output leans left, dtheta below zero on 75% of decisions, so the squad drifts west.

Two things differ from the grid, and either could be it: the game shows up to nine marines at
once, often as one wide blob, where the grid showed one dot; and the brain sees a scene that
keeps changing from a running state, where the grid started each spot from rest. The next test
is the compass fly shown only its nearest target, one dot, as in the grid.

#### One dot

`flycraft-client --one-dot` shows a brain's eye only the nearest target; the strike still
sees them all. The compass fly run that way (`cmp2-fly.jsonl`, trace `trace-cmp2-fly.jsonl.gz`)
scores the same, 87.7 (55.5). Its pitch looks as if it follows the target the wrong way round:
the raw pitch falls steadily as the target rises (23.3 Hz for targets 10 to 30 down, 21.1
level, 6.3 at 10 to 30 up), so pitch has the target's sign 22% of the time (|el| > 15). The
turn output is no better than before: 36% right, dtheta below zero on 81% of decisions.

Flipping pitch (`pitch_pos` and `pitch_neg` swapped, bias negated, everything else the same;
`cmp3-fly.jsonl`, trace `trace-cmp3-fly.jsonl.gz`) does not turn it round. Pitch then has the
target's sign 1% of the time, and the score drops to 54.9 (36.0). Whichever way pitch is wired,
the target ends up on the other side.

That is the loop, not the brain. The squad goes where the outputs send it, and the outputs
change slowly (pitch keeps 87% to 94% of its value from one decision to the next), so the
squad runs past its target and leaves it behind: the target sits opposite to wherever the
outputs have been pointing. The sign-right rates above, and the 48% and 43% before them, measure
that as much as the brain. The fairer question is whether the target predicts the brain's
next output once its current output is known (`lag.py`, a regression of the next output on
the current output and the target's angle):

| run | pitch (DNbe007 - DNge043), Hz per degree up | turn, Hz per degree east |
|---|---|---|
| cmp1, all targets | +0.14 (z +12) | +0.018 (z +12) |
| cmp2, one dot | -0.04 (z -4) | +0.016 (z +10) |
| cmp3, one dot, flipped | +0.10 (z +9) | +0.011 (z +9) |

The turn output moves toward the target's side in all three runs, and pitch toward its
elevation in two of three, with the grid's signs: the wiring was right and the flip was wrong.
But the pull is small. Thirty degrees moves turn by about 0.5 Hz a decision against a spread of
4.5 Hz, and pitch by 3 to 4 Hz against 12 to 29, while both carry a standing offset (the west
lean, and pitch's bias). The brain does respond to where the target is, too weakly and too
slowly to steer a squad that moves this fast.

#### Half speed

If the squad simply outruns that pull, slowing it should help. At half speed (`--step-px 3`,
each move order half as long; all targets and the original wiring, as in the first run;
`cmp4-fly.jsonl`, trace `trace-cmp4-fly.jsonl.gz`) it scores 84.2 (58.8), no better. The pull
per decision barely changes (pitch +0.18 Hz per degree, z +16; turn +0.016, z +12). With the
squad slower, the loop no longer hides the shape of the pitch readout: it is highest for a
level target (19.9 Hz) and drops by the same amount for targets above and below (12.5 and
12.2), as in the first run. Pitch has the target's sign 51% of the time, turn 42%, and dtheta
still leans west on 73% of decisions.

So the compass fly stops here. In the game its pitch pair reads how far off level a target is
more than which way, its turn output leans west, and the signed pull under both is too weak to
steer by, at either speed. The walking fly with search turns (155.9) stays the best real-fly
pilot.

## Stall stop

Watching the fly on DefeatMarines, its zerglings would sit in a corner for the last two minutes
of an episode, doing nothing. The client now ends a solo episode early when the fly has
stalled: over the last 25 decisions it asked for no turn at all (every `dtheta` exactly 0), the
score held, and the squad's seen middle stayed within 2 px of where the window began. A squad
the fly cannot see never counts as still. The stop never fires on an episode's last frame,
where the episode ends anyway, and never in a match.

A stalled episode counts, like a runaway: its record has `aborted: "stalled"` and the score
the fly had, and the brain hears `episode_end` with the same reason. `flycraft-client --stall
N` sets the window; `--stall 0` turns it off. `flycraft summarize` gives stalls a column.

`stall_check.py` replays the client's own rule on the brain's traces (one line per decision)
and asks which episodes it would have stopped, how much that saves, and whether any of them
scored again after the stop. Every trace is 20 episodes in train mode, seed 1, recorded before
the change: the fly on MoveToBeacon (M3), on DefeatRoaches after the friendly-fire fix (`dr3`)
and on DefeatMarines in M3c's baselines (`mc1`), with the oracle and random stand-ins on the
same two Defeat maps, and the stand-ins' DefeatMarines reruns after the zerglings went to
attack-move only (`mc2`).

```
python docs/m6/stall_check.py runs/m3/trace-*.jsonl --decisions 25
```

| trace | episodes | stopped | decisions saved | scored after a stop | without the turn check: stopped, scored after |
|---|---|---|---|---|---|
| real-train (MoveToBeacon) | 20 | 3 | 604 of 4820 (12.5%) | 0 | 3, 0 |
| dr3-fly (DefeatRoaches) | 20 | 9 | 1804 of 3556 (50.7%) | 0 | 11, 1 |
| mc1-fly (DefeatMarines) | 20 | 7 | 1397 of 4084 (34.2%) | 0 | 8, 0 |
| dr3, mc1, mc2 oracle and random | 120 | 0 | 0 of 21858 | 0 | 0, 0 |

The stop takes 19 of the fly's 60 episodes and none of the stand-ins' 120, and saves 30.5% of
the fly's decisions (3805 of 12460). No stopped episode would have scored again.

The fly's own DefeatMarines rerun (`mc2-fly`) finished after the window was set, so it is a
held-out check. Same result: no stopped episode would have scored again.

| trace | episodes | stopped | decisions saved | scored after a stop | without the turn check: stopped, scored after |
|---|---|---|---|---|---|
| mc2-fly (DefeatMarines) | 20 | 11 | 2182 of 4541 (48.1%) | 0 | 11, 0 |

Every episode the stop took, in `mc1-fly` and `mc2-fly`, began with the target behind the fly
(see "Where the fly looks") and stopped between decisions 32 and 62.

The turn check is what keeps it safe. Without it, a fly that steers but sits in place long
enough also stops, and on DefeatRoaches one of those would have scored later. At a 15 decision
window that grows: 6 lost on MoveToBeacon and 2 on each of the other two, and even with the
turn check one DefeatMarines episode that later scored is stopped. At 40 decisions nothing is
lost and the saving drops a little (11.6%, 41.8%, 31.6%). 25 is the setting.

The fly does not learn in the game yet (its brain reports `plasticity: false`), so a stop
cuts no learning short. When it does, the game's own reward gives a still fly nothing, so there
is no dopamine to miss. The spec's `game+distance` shaping, not yet built, punishes a squad held
at the screen edge, so a stall could earn punishment there; whoever builds it should check the
stop against it.
