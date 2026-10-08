# M3 run, 2026-10-08

The connectome brain plays MoveToBeacon in watch mode on one Windows PC: SC2 on Windows, the
brain on the WSL GPU (RTX 5080). Real MaleCNS v1.0 wiring with the M1 calibration (`vpn` rung, w_scale
0.8438), plasticity off, nothing trained.

```
# WSL: the brain, its view on :8766, and one trace line per decision
flycraft brain --calibration docs/m1/calibration.json --trace runs/m3/trace.jsonl
# Windows: four watch episodes
flycraft-client --mode watch --episodes 4 --seed 1 > watch.jsonl
```

Open http://localhost:8766 next to the SC2 window to watch the neurons.

- `watch-fly-real.jsonl`: the client's episode records (`flycraft summarize` reads them).
- `trace-fly-real.jsonl`: the brain's trace of the same run, one line per decision: marine and
  beacon positions, reward, the command sent and the raw decoder outputs.
- `trace-before-fix.jsonl`: the first run, two episodes before the hidden-marine fix (below).
  Both scored 0.
- `experiments/steering.py`: replays the body's heading from a trace and asks how the fly turned
  for a beacon ahead, to the side and behind, and on its left or right.
- `experiments/chirality.py`: the same question with the heading inferred from movement, plus a
  check that the body turned the way the brain said (392 of 430 times).
- `experiments/hidden_marine.py`: the probe behind the hidden-marine fix.

## Result

Scores 0, 6, 5 and 7: mean 4.5 (sd 3.1). Random scores 0.24 and the oracle body 20.6 (M2). No
runaways and no NaNs.

Steering, from `steering.py` (decisions that turned by 1 degree or more):

| Beacon | Turned toward it | Mean turn |
|---|---|---|
| ahead (within 60 degrees) | 201 of 337 (60%) | 26.4 |
| to the side (60 to 120) | 137 of 178 (77%) | 27.1 |
| behind (over 120) | 82 of 158 (52%) | 12.7 |
| on the left (within 120) | 213 of 278 (77%) | -13.3 |
| on the right (within 120) | 125 of 237 (53%) | +11.5 |

What that looks like in the game:

- **It steers at a beacon it can see.** A beacon off to one side gets a turn toward it three
  times in four. Episode 1 scored three times in its first 48 decisions, not far off the
  oracle's pace.
- **It is blind straight behind.** LC10a look forward and to the sides, so a beacon behind
  the fly drives little. Episode 3's first beacon, 135 degrees off, still got a turn toward it;
  episode 0 started with the beacon 170 degrees off, walked straight into the top-right corner within 20
  decisions and stood there for the rest: its turn output was exactly 0 for all 241 decisions.
  Nothing in the eye sees a wall.
- **It hunts.** Near the beacon (7 to 12 px) it overshoots and swings back at the full 64 degree
  turn limit, decision after decision: high gain, and a command that lands one decision late.
  That is likely why "ahead" scores lower than "to the side".
- **It is lopsided.** A beacon on the left gets a left turn 77% of the time; one on the right
  gets a right turn only 53% of the time, though the mean turn still points right. The cause is
  not known yet. Candidates: the retina map, the LC10a counts per side, the DNa02 pair. It is
  not a hangover from the last reward: split by time since a reward, the lean is the same.

On-screen chirality (spec section 12) passes: episodes 1 and 2 started with the beacon on the
right (43 and 47 degrees) and turned clockwise first; episode 3 started with it on the left and
turned counterclockwise.

## Speed

The brain answers about every 0.43 s, not the 0.22 s spec section 7.5 planned on. Watch mode
ran at 18.6 fps instead of 22.4, with about one late frame per decision (12%, longest wait 437
ms), and an episode took 103 s instead of 86. Game time stays lockstep, so the scores are the
same as they would be at full speed; only the wall time changes.

The brain view costs nothing measurable. Episodes 2 and 3 ran with a simulated page reading
the activity stream: 2,058 frames in 381 s (about 5.4 a second, 167 kB each). They ran at 18.61
and 18.79 fps, no slower than episodes 0 and 1 without it (18.45, 18.54).

## The hidden-marine bug

The first run scored 0. SC2 draws the beacon over the marine, and the marine drops off the
screen layer 5 to 7 px from the beacon's centre, before it scores (inside about 4.5 px). The
client stopped ordering a marine it could not see, so the fly stalled at the beacon's edge.
`hidden_marine.py` measured it. The fix (881df50): the body dead-reckons the hidden marine one
stride along its last order and keeps walking it.
