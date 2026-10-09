# Gate G2, 2026-10-08

Gate G2 (spec section 8) asks whether learning in the mushroom body can reach steering before
the M4 pilot spends hours training. It scales every KC to MBON weight in the PAM compartments
(the ones reward dopamine reaches) and reruns the G1 stimulus set: a blank eye and a 10 degree
spot at -60, 0 and +60 degrees, 10 trials of 2 s each, before and after. Real MaleCNS v1.0
wiring at the M1 calibration (`vpn` rung, w_scale 0.8438), on the WSL GPU.

```
flycraft g2 --calibration docs/m1/calibration.json --out runs/g2 --device cuda
flycraft g2 --calibration docs/m1/calibration.json --out runs/g2-f0 --device cuda \
  --set calibration.g2_factor=0
```

- `g2.json`, `g2.log`: the gate itself, weights x0.5. Exit 3: fail.
- `g2-factor0.json`, `g2-factor0.log`: a dose check run after the gate failed, weights x0.
  It does not gate anything.

## What is plastic

An MBON is in a PAM compartment if more than half its dopamine input (DAN to MBON synapses
in the connectome) comes from PAM neurons; a tie is not a majority. The DANs are the 316 PAM
neurons and the 16 PPL101 to PPL108. All 97 MBONs get DAN input; 77 get some from PAM, and
41 (15 types) are PAM-majority. Their KC to MBON edges are 32,626 of the 61,210 plastic edges.

## Result

Both measures must stay within 3 standard errors for a fail; both did.

| Weights | z turn response | z forward | Result |
|---|---|---|---|
| x0.5 (the gate) | -0.58 | -1.15 | fail |
| x0 (dose check) | -0.12 | -1.15 | fail |

The turn response is the mean turn with the spot at +60 minus the mean at -60, in degrees.
The gate: 41.2 before, 36.8 after halving. The dose check: 40.5 before, 39.7 after removing the
edges. The standard error of the change is about 7, so a pass needed the response to change by
about half.

Population rates (Hz, mean over the neurons and the 10 trials), spot at -60, 0 and +60:

| | KCs | PAM MBONs | other MBONs | turn DNs | forward DNs |
|---|---|---|---|---|---|
| before | 18, 44, 38 | 48, 106, 94 | 42, 86, 76 | 13.9, 0.4, 6.7 | 0, 0, 0 |
| after, x0 | 30, 10, 40 | 27, 11, 35 | 63, 25, 82 | 14.6, 1.0, 6.3 | 0, 0, 0 |

(The x0 run's "before" columns; the x0.5 run's are the same to within 1 Hz.)

Where the signal stops:

- **KCs are not silent.** The eye drives them at 10 to 44 Hz, far above a real fly's sparse
  KC code. Their rate swings between the before and after trials of the same stimulus (44 Hz
  before, 10 Hz after, spot ahead), by the same amount in both runs, so it follows the random
  input draws more than the edit.
- **The edit reaches the MBONs.** Removing the edges cut the PAM MBON rate from about 2.5
  times the KC rate to about 1 times, and left the other MBONs alone (about 2 times).
- **It does not reach the DNs.** The turn DNs (the DNa02 pair) fire the same before and
  after, within 1 Hz, even with the PAM MBONs down by half or more. Steering comes from the
  visual pathway into DNa02, and the mushroom body's output does not move it measurably.
- **The forward DNs never fire.** DNp09 runs at 0 Hz under every stimulus, before and after.
  The M1 calibration marks forward silent, so the fly has always walked at the default speed
  (`s0`, 0.5); its speed has never come from its brain. The forward measure cannot pass here,
  whatever is plastic.

Per spec section 8, the M4 pilot stops, and any change to the plastic set is pre-registered
before the main runs.

# Descending-neuron scan, 2026-10-08

G2 showed that learning in the PAM compartments does not reach steering. This scan asks the
wider question: does the mushroom body drive any descending neuron (DN), and could learning
there reach one? Same brain and calibration as G2. The three spots of the G1 set, 10 trials
each (the blank is skipped: no input, a silent brain), under three arms:

- `base`: the calibrated weights.
- `kc_mbon0`: all 61,210 plastic KC to MBON edges at 0, every compartment, not only PAM. This
  is the most any dopamine rule here could do, and far more than a learning step.
- `mbon_out0`: all 39,280 edges out of the 97 MBONs at 0. Not reachable by learning; an upper
  bound on what the mushroom body's output drives.

```
.venv/bin/python docs/m4/dn_scan.py docs/m1/calibration.json runs/dn_scan   # GPU, 17 min
python docs/m4/dn_scan_perm.py docs/m4/dn_scan_rates.npz docs/m4/dn_scan.json
```

- `dn_scan.py`, `dn_scan.json`, `dn_scan.log`, `dn_scan_rates.npz`: the scan and every DN's
  per-trial rate in every arm.
- `dn_scan_perm.py`, `dn_scan_perm.txt`: the permutation null.

Per DN, z is the change in rate averaged over the spots, over its standard error. A DN counts
if |z| > 4.12 (Bonferroni over 1,314 DNs, two-sided 0.05) and it moves by at least 1 Hz.
At base, 599 DNs fire at all and 344 at 1 Hz or more.

Ten trials per arm is few for a normal z, and the trials share network state, so the
permutation null (base and arm trials shuffled within each spot, 5,000 times) has a heavy tail:
most shuffles give 0 hits, a few give dozens. The count-level p is what to read.

| Arm | DNs that count | null 99th pct | P(null >= observed) |
|---|---|---|---|
| `kc_mbon0` | 5 | 1 | 0.004 |
| `mbon_out0` | 58 (40 up, 18 down) | 1 | 0.0006 |

The 5 `kc_mbon0` DNs, Hz averaged over the spots (base, `kc_mbon0`, `mbon_out0`):

| DN | base | kc_mbon0 | mbon_out0 |
|---|---|---|---|
| DNg63 R | 0.5 | 14.9 | 21.5 |
| DNg63 L | 0.2 | 4.4 | 2.3 |
| DNg44 R | 2.4 | 5.5 | 12.3 |
| DNbe006 R | 4.7 | 10.7 | 18.3 |
| DNp68 R | 6.7 | 0.6 | 0.0 |

All 5 move the same way in `mbon_out0`, mostly further, so the pathway is real. None is in the
decoder, and none has a motor command we could map to a game action without inventing one.

The decoder's own DNs:

| DN | base | kc_mbon0 | mbon_out0 |
|---|---|---|---|
| DNa02 L (turn) | 8.2 | 11.1 | 22.1 |
| DNa02 R (turn) | 3.2 | 3.6 | 1.8 |
| DNp09 L, R (forward) | 0 | 0 | 0 |
| MDN, 4 cells (backward) | 0.1 to 0.2 | 0.1 to 0.2 | 0.1 to 0.2 |

Removing all MBON output turns DNa02 L on for every spot (spot ahead 0.9 to 12.9 Hz, spot at
+60 0.7 to 20.7 Hz), so the mushroom body's output holds a tonic brake on one turning neuron.
Zeroing the plastic edges moves it 3 Hz, not past the threshold. Anatomy agrees: MBONs supply
250 of DNa02's 48,125 input synapses directly, and none of DNp09's.

## What it decides

The mushroom body drives DNs, but its plastic synapses reach only 5, and not the ones the
decoder reads. Reading those 5 out would mean assigning them game actions with no biological
basis, which is the same move as option B (putting learning on the visual pathway): something
we add, not the fly. So M4 learning stays stopped, and the work moves to M6 (a game against
the fly as it is). Its fighting is reflex: a visual pathway from the eye to DNa02.

# Central-complex check, 2026-10-08

The mushroom body cannot reach steering (G2, above). Real flies also learn in the central
complex (CX): visual places, and goals to steer toward. Its premotor outputs, the PFL3 cells,
are DNa02's fifth-largest input type (736 of 48,125 synapses, crossed: PFL3 L feeds DNa02 R).
Could learning there steer? `cx_scan.py` runs G1's three spots on the calibrated brain (vpn
rung, 10 repeats of 2 s each) and records the CX populations and DNa02 under: the calibrated
weights; every edge out of PFL1 to PFL3 at 0 (`pfl_out0`); and extra Poisson input to one
side's PFL3 cells at 150 Hz (the LC10a drive's scale) and at 20 Hz (`_low`). The criteria,
fixed before the run, are in its docstring: B1, PFL3 fires at 1 Hz or more at base; B2, a
one-sided drive moves the turn signal (DNa02 R minus L) by 1 Hz or more, |z| > 3, the two sides
opposite. 56 min on the GPU, sharing it with a game run.

```
.venv/bin/python docs/m4/cx_scan.py docs/m1/calibration.json runs/cx_scan
python docs/m4/cx_response.py docs/m4/cx_scan_rates.npz
```

Records: `cx_scan.json`, `cx_scan.log`, `cx_scan_rates.npz`.

## Result

B1 passes (PFL3 1.61 Hz at base) and B2 passes (each drive raised its PFL3 cells by 112 Hz and
turned the fly the crossed way). The last two columns are G2's turn measure, which asks how the
turn depends on where the spot is: the turn at the right spot minus the left, after minus
before (`cx_response.py`).

| Arm | PFL3 L, R (Hz) | Turn signal change, mean over spots | z | Turn response change | z |
|---|---|---|---|---|---|
| base | 1.67, 1.56 | | | | |
| pfl_out0 | 1.63, 1.48 | +1.76 | 0.69 | -7.33 | -0.96 |
| pfl3_L | 113.3, 0.9 | +86.51 | 34.46 | -7.33 | -1.14 |
| pfl3_R | 1.0, 113.2 | -92.69 | -40.86 | -8.93 | -1.37 |
| pfl3_L_low | 20.9, 1.6 | +14.49 | 5.83 | +5.67 | 0.76 |
| pfl3_R_low | 1.7, 20.6 | -14.47 | -6.73 | +13.47 | 2.11 |

Base rates by spot (Hz):

| Population | spot at -60 | ahead | at +60 |
|---|---|---|---|
| LC10a L | 3.37 | 5.79 | 0.00 |
| LC10a R | 0.00 | 6.61 | 2.93 |
| MeTu, TuBu | 0 | 0 | 0 |
| ER | 14.15 | 12.54 | 12.80 |
| EPG | 53.00 | 46.56 | 47.27 |
| PFN | 1.35 | 0.94 | 1.20 |
| hDelta | 5.49 | 4.43 | 4.96 |
| PFL3 L | 2.01 | 1.65 | 1.33 |
| PFL3 R | 2.08 | 1.38 | 1.21 |
| DNa02 L | 22.53 | 0.80 | 0.80 |
| DNa02 R | 0.00 | 0.07 | 9.33 |

The CX's output can steer: one side's PFL3 at 20 Hz moves the turn signal by 14.5 Hz with the
sign the anatomy predicts. Today it contributes nothing measurable (`pfl_out0`, z 0.69).

But the CX does not know where the spot is. PFL3 L and R both fire most for the left spot and
least for the right, and so do EPG, ER, PFN and hDelta. Everything here comes from the eye (the
vpn rung has no tonic input), but it reaches the CX as overall light, not as a direction.
MeTu and TuBu, the anterior visual pathway that carries what the fly sees into the CX's ring
in real flies, are silent: the retina drives photoreceptors and LC10a, and the photoreceptor
signal does not get through the optic lobe to them. So a drive adds a turn bias, the same at
every spot, and no arm passes G2's measure (largest |z| 2.11), not one side at 113 Hz and not
all CX output removed. Learning on edges inside the CX would see the same activity whichever
side the spot is on; the most it could learn is a bias one way, not "turn toward what paid".

## What it decides

B1 and B2 pass as written, but the measure that matters for learning to steer fails, so the
CX is not a place to put learning in this model as it stands. The G2-style scale check of a
CX plastic set is not run: halving inputs to PFL3 does less to DNa02 than removing all PFL
output, which fails. Two ways on:

1. Put learning where the spot's side is, on the visual path from LC10a to DNa02. This is
   something we add, not the fly, and the write-up says so. Next.
2. Drive MeTu from the eye the way the vpn rung drives LC10a, so the CX gets a direction, and
   check again. Closer to the fly, and a bigger job. Later.
