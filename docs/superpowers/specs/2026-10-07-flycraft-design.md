# flycraft: design spec

A whole-brain fruit-fly connectome plays StarCraft II. You watch it two ways at once: the live SC2 window, and a browser view of the brain's neurons firing.

- Status: approved 10-07, with four review fixes applied (sections 7.4, 7.6, 8, 9, 12, 13). Updated 10-07 with what the M1 plan's code found (sections 4, 5, 7.3, 12, 15). Updated 10-08 with what the M2 code settled (sections 3, 7.1, 7.5, 9, 10, 12, 14).
- Date: 2026-10-07
- Scope: Phase 1, the MoveToBeacon minigame. A DefeatRoaches demo follows M3 (M3b); CollectMineralShards comes later (section 15).

## 1. Goal and what counts as success

**Goal.** A spiking model of the male *Drosophila* central nervous system steers a marine in MoveToBeacon.

- It is built from the MaleCNS v1.0 connectome, about 166.7k leaky integrate-and-fire neurons.
- It sees the game through its photoreceptors.
- It acts through its descending neurons.
- It learns only through dopamine-gated plasticity in the mushroom body.

**Engineering success.** BJ can watch the fly play MoveToBeacon in the real SC2 window on the PC, with the brain view lit up next to it (M3).

**Science success.** A pre-registered test of whether the real wiring outperforms degree- and sign-matched scrambled wiring (section 9). Both outcomes count:

- It is run cleanly.
- The result is published whichever way it goes.

**Later goal.** BJ plays StarCraft II against the fly: a skirmish in M6, then a full 1v1 in M7 (section 15).

**Out of scope:**
- Training all weights by gradient descent or RL. A prior "fly plays Doom" variant trained this way showed real, random and absent wiring scoring the same (research.md §0). Training everything makes the connectome a random projection layer and erases the question.
- A physical fly body model (NeuroMechFly).
- Linux SC2.
- Multiplayer in Phase 1. Playing against BJ comes later (M6 and M7).

## 2. Prior work we build on

All of these are MIT-licensed code. We port ideas and equations; we do not vendor their code.

| Source | What we take |
|---|---|
| Shiu et al. 2024 (philshiu/Drosophila_brain_model) | LIF equations and constants (`V0 = V_RST = -52 mV`, `V_TH = -45 mV`, `T_MBR = 20 ms`, `TAU_SYN = 5 ms`, `T_RFC = 2.2 ms`, `T_DLY = 1.8 ms`, `W_SYN = 0.275`, Poisson input 150 Hz, `F_POI = 250`). |
| doomfly | Reward through dopamine neurons onto KC→MBON synapses: a 200 ms pulse into DANs gates the plasticity. |
| webergithub/fruitfly-lab | The scrambled-wiring definition: keep every neuron's in-degree, out-degree and sign, and shuffle only who connects to whom. Also the LC10a → AOTU019 → DNa02 steering circuit. Their scrambled control dropped Doom kills from 10.2 to 0.7. |
| kiminbean/drosophila-brain-mlx | Conventions for loading MaleCNS v1.0 (166,700 neurons, 24,469,412 edges) and its shuffle tool. A cross-check for our counts. |

## 3. Architecture

```
One PC (Windows 11, RTX 5080)
+--------------------------------------------+   +--------------------------------------------+
| Windows side (BJ's desktop, session 1)     |   | WSL Ubuntu 24.04                           |
|                                            |   |                                            |
|  SC2_x64.exe  <--s2client-->  flycraft     |   |  flycraft brain worker (GPU)               |
|  (visible game window)         game client |   |   - LIF sim, retina, decoder, plasticity   |
|                                (py3.11,    |ws |   - game server  ws://127.0.0.1:8765       |
|                                 pysc2)     |<->|   - view server  http://127.0.0.1:8766     |
|                                            |   |                                            |
|  browser tab: brain view  <----------------|---|   flycraft supervisor (no GPU)             |
|  (http://localhost:8766)                   |   |   - yield guard, restarts, run records     |
+--------------------------------------------+   +--------------------------------------------+
```

Windows reaches WSL at 127.0.0.1 through WSL's localhost forwarding; the spike verified this. Both servers bind to 127.0.0.1 only. Nothing listens on the LAN.

### Processes

1. **Game client** (Windows; `flycraft-client`).
   - Owns the SC2 environment and the fly's body: position, virtual heading, motion.
   - Renders what the fly sees into an "eye image".
   - Applies the brain's motor command.
   - Knows the game and nothing about neurons.
2. **Brain worker** (WSL; `flycraft brain`).
   - Owns the connectome, the simulator, photoreceptor sampling, descending-neuron decoding, plasticity and the view feed.
   - Knows neurons and nothing about StarCraft. Its only inputs are the eye image and a scalar reward.
   - `flycraft brain --stub oracle|random` serves a stub brain instead (M2); the stub is a required argument until the real brain lands in M3.
3. **Supervisor** (WSL; `flycraft experiment` / `flycraft watch`).
   - Starts the worker, launches the client through WSL interop (`cmd.exe`), runs the yield guard, and writes run records.
   - Holds no GPU memory, so it can idle while BJ games.

**Boundary rule.** The eye image and reward are the only information that crosses from game to brain. Raw positions (marine, beacon) travel in the protocol for logging and the view only. Two tests enforce that the network never sees them:
- a code-path test;
- an invariance test, in which perturbing those fields leaves the decoder output unchanged.

## 4. Connectome data

### Source

MaleCNS v1.0, CC-BY 4.0, public bucket `gs://flyem-male-cns/v1.0/` (HTTPS at `storage.googleapis.com/flyem-male-cns/v1.0/`). `flycraft prep-data` downloads three files from `connectome-data/flat-connectome/`:

| File | Size | Use |
|---|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14.5 MB | Per-neuron fields (list below). 211,577 rows. |
| `body-neurotransmitters-male-cns-v1.0.feather` | 43.3 MB | `consensus_nt` per body, falling back to `predicted_nt`. |
| `connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather` | 508 MB | Edge list: pre, post, synapse count. |

Annotation fields used:
- identity: `bodyId`, `type`, `superclass`, `class`
- side: `somaSide`, `rootSide`
- position: `somaLocation`
- optic-lobe column: `assignedOlHex1/2`

**Download integrity.** Each download is checked against the bucket's `x-goog-hash` crc32c header and against a crc32c and size pinned in the code. A mismatch aborts and leaves no partial file.

**What gets built.** A compact cache, `~/.cache/flycraft/malecns-v1.0.npz`, containing:
- a neuron table: index, bodyId, type, superclass, side, position in µm, hex column;
- edges: int32 pre, int32 post, int32 synapse count;
- one int8 sign per neuron;
- a manifest with source file names, hashes, and neuron and edge counts.

**Scope and count check.**
- The neuron set is every body in the traced-only edge list, VNC included (`include_vnc: true`).
- The build prints neuron and edge counts and compares them to drosophila-brain-mlx's 166,700 and 24,469,412.
- They differ by definition: 164,587 neurons here, and 24,458,434 edges when counted mlx's way. The manifest records both conventions, and the M1 report prints them side by side.

**Not committed.** Data is not committed to the repo. `NOTICE` carries the CC-BY attribution.

### Synapse sign

Each neuron's sign comes from its transmitter, matched case-insensitively:

| Transmitter | Sign |
|---|---|
| acetylcholine | +1 |
| GABA | -1 |
| glutamate | -1 |
| histamine | -1 |
| any other or unknown | +1 |

- The rule is configurable.
- M1 compares it with the rule Shiu et al. 2024 state (GABA and glutamate inhibitory) and reports the difference, which is histamine. Their code applies no transmitter rule: it loads precomputed FlyWire signs (their repo's issue #11 lists where those disagree), and FlyWire predicts no histamine, so the stated rule is the comparison.
- The build logs counts per transmitter.
- Edge weight: `w = sign(pre) * synapse_count * W_SYN * w_scale`.

### Positions (for the view only)

- Use `somaLocation` (8 nm voxels converted to µm) where present. 141,781 annotated bodies have it.
- Otherwise place the neuron at the synapse-weighted mean soma position of its partners, which works for photoreceptors: their somata sit in the retina outside the volume, but their lamina partners have somata.
- If no partner has a soma either, use the class centroid plus 5 µm of jitter.

## 5. Brain simulator

### Model

The Shiu 2024 LIF model, in PyTorch on CUDA, float32, `dt = 0.1 ms`.

- **Membrane and synapse:**
  - The membrane follows `dv/dt = (g - (v - V0)) / T_MBR`.
  - The synaptic conductance follows `dg/dt = -g / TAU_SYN`.
  - A spike adds `w` to the postsynaptic `g` after `T_DLY`, through a ring buffer.
- **Reset and refractory period:** after a spike, `v` resets to `V_RST` and is held there for `T_RFC`. Input that arrives while a neuron is refractory, or on the step it spikes, is dropped, as Brian2's `(unless refractory)` does.
- **Integration scheme:** matches Brian2 2.9.0 on Shiu's equations exactly in float64. The golden fixture (section 12) reproduces every spike at the same step.
- **Poisson inputs:** each input spike adds `F_POI * W_SYN` = 68.75 mV to the target's `v`, as the reference's `PoissonInput` does. Poisson-driven neurons have no refractory period, as in the reference.

### Kernels

- **Default path:** event-driven. Each step gathers the out-edges of the neurons that spiked and `index_add`s them.
- **Dense fallback:** a CSR SpMV path for testing.
- **Spike numbers** (synthetic MaleCNS-size graph):
  - event path: 4.3 s wall per simulated second;
  - SpMV: 6.3 s wall per simulated second;
  - peak memory: 2.5 GB.
- **Per decision:** 50 ms of brain time should come to about 0.22 s wall. On the real graph in M3 the brain answered about every 0.43 s (section 7.5).
- **Faster kernels:** M5 replaces the event path with a fused kernel only if the experiment budget needs it.

### State

- Membrane and synapse state persist across decisions within an episode.
- They reset at episode start (`reset_each_episode: true`), so episodes are independent for statistics.
- Plastic weights persist across episodes; that is the learning.

### Weight scale calibration (gate G0)

Shiu's constants were fit to FlyWire. MaleCNS synapse counts may run on a different scale. So each brain, real and every scrambled twin, goes through the same procedure:

1. Start at `w_scale = 1.0`.
2. Run 2 s of brain time under a uniform grey eye image.
3. The run "runs away" if more than 0.5% of neurons fire above 100 Hz, or the population mean exceeds 20 Hz. Poisson-driven input neurons are left out of both figures: the retina sets their rates, not `w_scale`.
4. If it runs away, bisect `w_scale` downward (at most 8 halvings, then bisection to 5% precision) to the largest value that does not.
5. Record the result in `calibration.json`.

Each G1 rung (7.3) gets its own G0, because a rung can change the input. If G0 fails on a rung, that rung fails and the ladder moves on.

The same rule applied to every brain is the fairness guarantee. We do not hand-tune any brain.

## 6. Real and scrambled wiring (first-class)

Wiring is a top-level config field, not a flag bolted onto the loader:

```yaml
# configs/real.yaml
extends: base.yaml
connectome:
  wiring: real
  scramble_seed: null

# configs/scrambled.yaml
extends: base.yaml
connectome:
  wiring: scrambled
  scramble_seed: 1        # one twin per seed; the experiment sets 1..6
```

**Validation:**
- `scrambled` without an integer seed is an error.
- `real` with a seed is an error.

**Scramble algorithm.** This is fruitfly-lab's definition, made exact.

1. **Exempt set.** Remove the exempt set X: every edge with a Kenyon cell (KC*) presynaptic and an MBON postsynaptic. These are the plastic edges (section 8). They stay real in every twin, so both twins learn with identical machinery.
2. **Split by sign.** Split the remaining edges by the sign of their presynaptic neuron into excitatory and inhibitory classes.
3. **Permute targets.** Within each class, apply a seeded random permutation to the postsynaptic endpoints.
   - Each edge keeps its presynaptic neuron, its synapse count and its sign; weights travel with the presynaptic side.
   - So every neuron keeps its out-degree, its outgoing weight distribution, its sign, its excitatory in-degree and its inhibitory in-degree.
4. **Repair.** Self-loops and duplicate (pre, post) pairs are repaired by swapping targets with random edges of the same class, for up to 20 passes.
   - Leftover duplicates are merged by summing synapse counts.
   - Leftovers must be under 0.1% of edges, or the build fails.
5. **Cache.** Cache the result as `malecns-v1.0.scrambled-<seed>.npz` with a sha256 fingerprint of its edge arrays.

**Shared across twins:**
- Neuron identities, positions, input neurons, decoder neurons, the dopamine-to-MBON map, the retina geometry and every procedure.
- Only who connects to whom differs.

**Twin identity is always visible:**
- Every run record carries the wiring fingerprint, so results from different twins cannot be silently mixed.
- The brain view shows a large badge: `REAL WIRING` or `SCRAMBLED #3`.

## 7. Senses and motor

### 7.1 The body (game client)

The marine is the fly's body. The client keeps a virtual heading θ, chosen at random at episode start from the episode seed.

**Each decision:**
1. **Find the marine and beacon.** Locate the marine (`player_relative == SELF`) and the beacon (`player_relative == NEUTRAL`) on the 84×84 screen layer.
   - SC2 draws the beacon over the marine, and the beacon is drawn wider than it scores (M3: drawn radius 6.6 px; the marine vanishes about 5 to 7 px from the beacon's centre and scores inside about 4.5). So a marine that has been seen but is not on the layer is under the beacon. The body then takes it to be one stride on from where it was, toward its last move target (a stride is `L / 1.5`, see step 4), and renders and moves from there. Without this a body that loses its marine stands still in that ring for the rest of the episode, which is what the first real-brain run did.
2. **Turn.** Apply the turn from the command `{dtheta, speed}` computed from the previous decision's observation. That is one decision of sensorimotor latency, the same in every mode.
   - Update the heading: θ += dtheta. The turn per decision is clipped so the turn rate stays at or below 180° per second of game time.
   - The turn comes before the render, so each eye image shows the heading the next move order uses.
3. **Render the eye image (7.2).** Use the beacon's bearing relative to the new θ and its distance.
4. **Move or stop,** at that command's speed:
   - Issue `Move_screen` to `marine + L * speed * (cos θ, sin θ)`, clipped to the screen.
   - If `speed < 0.05`, issue `Stop_quick` instead.
   - M2 picks the step length `L` as the distance a marine covers in 1.5 decisions, so it never stops between orders at full speed.
5. **Select the marine.** Select the marine at episode start, and again whenever `Move_screen` is unavailable.

**Speed.** `speed = clip(s0 + k_fwd * fwd, 0, 1)`, with `s0 = 0.5`. The body walks by default and the brain modulates it, the way a real fly walks spontaneously. `s0` is the same for every condition and baseline.

### 7.2 Eye image

The eye image is the only sensory input.

- **Format:** uint8, 30 rows × 72 columns.
  - Elevation runs from +75° to -75°, top to bottom; azimuth from -180° to +180°, left to right. Each bin is 5°.
  - 0° azimuth is the heading.
- **Content:** a uniform background with the beacon drawn as a disk centered at 0° elevation.
  - The disk is centered on the beacon's bearing.
  - Its angular radius is `atan(r_beacon / d)`, clipped to 2° to 60°, so an approaching beacon looms.
  - Polarity is configurable. The default is dark-on-bright (beacon 0, background 160), because many fly object detectors (LC10a among them) prefer dark objects.
- **Rendering:** the client renders it; the protocol carries it; the view displays it.

### 7.3 Retina (brain worker)

**Photoreceptors.**
- MaleCNS has R1-R6 (3,377: left 1,112, right 2,265), R7 subtypes (1,385) and R8 subtypes (1,329). The neuron set (bodies in the edge list) holds R1-R6 1,393 (left 501, right 892) and R7/R8 2,709: 4,102 photoreceptors.
- None has a hex column of its own (only 16 have a soma). So each photoreceptor takes the synapse-weighted majority `assignedOlHex1/2` column of its columnar postsynaptic partners:
  - L1, L2 and L3 for R1-R6;
  - any hex-assigned medulla partner for R7 and R8.
- Neural superposition means a lamina cartridge's column is the direction its photoreceptors look, so this gives viewing direction directly.
- Photoreceptors with no hex-assigned partner get the eye's mean luminance and are counted in the M1 report.

**Hex to visual angle.**
- A per-eye affine map takes hex coordinates (ranges 1-36 and 1-39) to (azimuth, elevation). It is mirrored between eyes.
- Each eye spans azimuth -15° to +165° on its own side, which gives a 30° frontal binocular overlap, and elevation -60° to +75°.
- The axis orientation comes from soma positions. Lamina soma y tracks elevation. The anterior-posterior axis is the reverse of Mi1 soma z: Mi1 z fits hex well and runs opposite to lamina z across the first optic chiasm. Each eye is normalized by the 2nd to 98th percentiles of its own lamina columns.
- On the real data this affine map passes its checks: 4,102 photoreceptors, 355 with no column, 275 LC10a.
- **M1 deliverables for this map:**
  - (a) a per-eye scatter of photoreceptor directions for visual inspection;
  - (b) a left/right test: a spot at +60° must drive right-side LC10a more than left, and vice versa.
- If the orientation can't be pinned down, the fallback keeps side and splits each eye into 6 azimuth bands by rank along the anterior-posterior hex axis. M1 reports which map is in use.

**Drive.**
- Each photoreceptor fires Poisson at `r = r_max * lum`, with `r_max = 150 Hz` and `lum` the bilinear sample of the eye image at its direction, scaled 0 to 1.
- The left eye's photoreceptors are under-reconstructed (in the neuron set, 501 versus 892 R1-R6). So `eye_normalize: true` scales each eye's rates by `mean_count / eye_count`, giving both eyes equal total drive under a uniform scene.
- Rates update once per decision.

**Input propagation risk and gate G1.** Photoreceptors are histaminergic, so under the sign rule they inhibit. Inhibiting a neuron that is already silent does nothing. Pure photoreceptor drive may therefore never reach the central brain. The check:

- Gate G1 runs offline on the real brain in M1.
- Present a dark spot at -60°, 0° and +60° azimuth, 2 s each, 10 repeats.
- **Pass:** LC10a or the decoder DNs show stimulus-dependent rates, with a left/right difference greater than 3 standard errors.

If it fails, take the first rung of this ladder that passes, applied identically to every brain afterwards:

1. Flip polarity (bright-on-dark).
2. Add tonic drive to lamina neurons, L1 to L5 (`lamina_tonic_hz`: try 20, then 50), so that disinhibition carries signal.
3. Set `retina.mode: photoreceptor+vpn`: also inject the eye image into LC10a (275 neurons) at their receptive-field centers.
   - A receptive-field center is the weighted mean direction of the cell's hex-assigned inputs.
   - This is fruitfly-lab's input point.

Photoreceptors are driven on every rung, as BJ specified.

**Known bias.** G1 is tuned on the real wiring. The ladder is three coarse, pre-defined rungs rather than a fit, and the M1 report states the rung chosen.

### 7.4 Decoder

**Turn and forward signals.**
- `turn = rate(DNa02_R) - rate(DNa02_L) - b_turn`
- `fwd = mean rate(DNp09) - b_fwd`
- Rates are exponentially filtered spike trains with `tau = 150 ms`. With only two cells per group, a raw 50 ms count is too noisy.

**Defaults and options.**
- MaleCNS has DNa02 ×2 and DNp09 ×2.
- The groups are config lists of (type, side). A missing type fails startup and names the missing types.
- fruitfly-lab's `+ (AOTU019_R - AOTU019_L)` turn term is available as a config option. It is off by default, because BJ asked for descending neurons.
- MDN (×4, backward walking) is logged but not used in Phase 1.

**Per-brain calibration.** Run on every brain, after G0, with the same procedure:
- **Baseline biases.** `b_turn` and `b_fwd` are the mean outputs under a blank grey scene.
- **Turn gain.** `k_turn` maps the 95th percentile of |turn| across the G1 stimulus set to the maximum turn rate.
- **Forward gain.** `k_fwd` maps the 95th percentile of `fwd` to `1 - s0`.
- **Gain floor.** If the 95th percentile of |turn| is below `gain_floor_hz` (1 Hz), `k_turn = 0` and `calibration.json` records `turn_silent: true`. The same rule sets `k_fwd = 0` and `fwd_silent: true`. A silent brain then walks straight at `s0`, which is the honest "no information" outcome; it never divides by zero.
- **What this means for a weak brain.** A brain with weak but informative signals gets amplified; a brain with no information gets amplified noise. The control tests information, not amplitude.

**Output.** The motor command is `dtheta = k_turn * turn * T_game`, where `T_game` is the decision length in game time (8 game loops = 0.357 s). Speed follows 7.1.

### 7.5 Timing

| Mode | step_mul | Decision cadence | Pacing |
|---|---|---|---|
| train / eval | 8 | every step | lockstep, as fast as the brain allows |
| watch | 1 | every 8th frame | 1/22.4 s per frame ("faster" game speed) |

- **Brain time per decision:** 50 ms in every mode. The brain's clock runs about 7× slower than game time; this time dilation is a stated design choice, configurable via `decision_ms`.
- **Decision cadence:** the decision cadence is the same in both modes: 8 game frames per decision. In watch mode the frames in between are no-ops, which leave the current move order running. Watch mode therefore renders smoothly; the 10-07 demo looked choppy at step_mul 8.
- **Brain speed in watch mode:** planned at about 0.22 s per decision, within the 0.357 s budget. Measured in M3: about 0.43 s, so watch mode runs at about 18.6 fps instead of 22.4, with about one late frame per decision, and an episode takes about 103 s. Game time stays lockstep, so scores are unchanged; only the wall time grows. M5's fused kernel is the fix if it matters.
- **Pipelining:** the client sends observation k and keeps stepping while the brain computes. It applies the resulting action at decision k+1. The brain answers every observation, the episode's last included, so the client always drains exactly one act per obs before `episode_end`.
- **Episode length:** at 240 decisions, a train episode takes about 55 s of wall time and a watch episode about 90 s.
- **M2 check:** M2 verifies that step_mul 1 keeps up at 22.4 Hz on the PC. The fallback is PySC2 `realtime=True`.

### 7.6 Handedness convention

Screen `y` points down, so a sign error anywhere between the screen and the turn command mirrors the fly. A mirrored fly steers away from the beacon and scores worse than a scrambled one. One convention holds everywhere:

- **Screen frame.** `x` right, `y` down, angles from `atan2(dy, dx)`. An increasing angle rotates clockwise on screen.
- **Heading.** θ uses the screen frame. `dtheta > 0` is a clockwise, rightward turn as seen on screen.
- **Azimuth.** `az = wrap(bearing - θ)`. `az > 0` means the object is to the fly's right. In the eye image it appears in the right half of the columns.
- **Eyes.** The right eye (`side == R`) covers azimuth -15° to +165°, and the left eye mirrors it.
- **Decoder.** `turn > 0` when `DNa02_R` fires more than `DNa02_L`, and it maps to `dtheta > 0`. This follows the literature, where right DNa02 activity precedes rightward turns (Rayshubskiy et al. 2020). It is fixed before any data is collected and is never fit.

The end-to-end chirality test (section 12) checks this whole chain. If the real brain's own chain comes out mirrored (right LC10a drives left DNa02 more), the M1 report says so. We do not flip the decoder to compensate.

## 8. Reward and plasticity

### Plastic edges

- Every KC→MBON edge is plastic: 4,064 KCs and 97 MBONs in MaleCNS. That is the exempt set X from section 6.
- All other weights are fixed.

### Dopamine map

- `M[d, m]` is the synapse count from DAN `d` to MBON `m`, normalized per MBON.
- Its source is the **real** connectome, used by every twin.
- DANs are PAM* (316) and PPL101-108 (16).

### Reward events

| Event | DANs driven | Drive |
|---|---|---|
| Game reward > 0 (reaching the beacon) | all PAM | 150 Hz Poisson for 200 ms |
| Punishment | PPL101-108 | Same pulse |

- Punishment in Phase 1: only if shaping is on (below).
- The DANs fire inside the simulation, so they show up in the view and affect the network like any other input.

### Dopamine signal

The plasticity signal uses the **commanded pulse**, not the DANs' simulated spikes:

`DA_m(t) = sum_d M[d, m] * pulse_d(t)`

In a scrambled twin, the DANs' own inputs are scrambled, so their spiking differs. The commanded pulse keeps the teaching signal identical across twins.

### Learning rule

The rule follows the mushroom body: dopamine depresses KC→MBON synapses in its compartment.

- **Eligibility:** `e_k(t)` is each KC's low-pass filtered spike train, with `tau_elig = 1 s` and an increment of 1 per spike.
- **Weight change:** `dw_km/dt = -eta * DA_m(t) * e_k(t) * w0_km`
- **Bounds:** clipped to `[0, w0_km]`. No sign changes.
- **When applied:** once per decision, written in place into the plastic slots of the edge-value array.

**Saturation risk.** Every reward pulses all PAM. At about 20 rewards per episode over 150 training episodes, that is about 3,000 pulses. Depression with no recovery can walk every frequently active KC→MBON weight to 0. "Learning" would then converge to a lesion of the PAM compartments, and that lesion looks the same in both twins. Two guards:

- **Floor fraction.** Each episode logs `frac_at_floor`, the fraction of plastic weights below `0.05 * w0`, overall and per MBON compartment. The view shows it.
- **Recovery term (off by default).** `dw_km/dt += eta * rho * DA_m(t) * (w0_km - w_km)`. Dopamine with no KC coincidence pulls a synapse back toward `w0`, as dopamine without odor does in the fly (Cohn et al. 2015, Berry et al. 2018). With it, a synapse settles near `w0 * rho / (rho + e_k)` rather than at 0. `rho = 0` disables it. The pilot turns it on only by the rule in section 9.

### Shaping

| Setting | Effect |
|---|---|
| `reward.source: game` (default) | Reward is only the game's +1 per beacon. |
| `game+distance` | Also pulses PAM at 1/4 strength when the beacon distance shrinks by 10 px. Also pulses PPL1 at 1/4 strength when the marine sits at the screen edge for 2 s. |

### Gate G2: does plasticity reach steering?

Pilot, real brain:

1. Scale every PAM-compartment KC→MBON weight by 0.5.
2. Rerun the G1 stimulus set.
3. **Pass:** the turn or forward response changes by more than 3 standard errors.

If it fails, KC→MBON learning cannot reach steering. The pilot stops and we report back before changing the plastic set; any change is pre-registered before the main runs.

## 9. Experiment and statistics

### Design

A 2×2 of wiring {real, scrambled} × plasticity {off, on}, plus three baselines through the same body.

| Condition | Brains | Episodes per brain |
|---|---|---|
| real, plasticity off | 1 | 50 eval |
| scrambled, plasticity off | 6 (seeds 1-6) | 50 eval |
| real, plasticity on | 6 (training seeds 1-6) | N_train train + 50 eval (frozen) |
| scrambled, plasticity on | 6 (scramble seed i, training seed i) | N_train train + 50 eval (frozen) |
| random body (dtheta uniform within the turn limit each decision, speed `s0`; its own RNG stream from the episode seed) | 1 | 50 |
| oracle body (dtheta = bearing error read from the eye image, clipped; speed 1.0) | 1 | 50 |
| PySC2 scripted agent (ceiling, about 26) | 1 | 50 |

The scripted agent acts on the frame it sees; every body acts one decision late (7.1) and turns at most `max_turn_deg_s`. M2 measured what those two limits cost on real SC2 (50 train episodes each, run seed 1; records in `docs/m2/`):

| Pilot | Mean score |
|---|---|
| scripted | 25.94 |
| scripted steering by bearing, 6 px move target (12 and 30 px give the same) | 25.90 |
| scripted, one decision late | 22.58 |
| oracle body, turn limit lifted (3600°/s) | 22.22 |
| oracle body, 360°/s | 21.96 |
| oracle body, 180°/s (the baseline) | 20.62 |
| random body | 0.24 |

So the one-decision latency costs about 3.4 points and the turn limit about 1.6; the eye and the brain link cost under 0.4. The fly's ceiling through this body is the oracle's, about 21, not 26.

### Unit of replication

The unit of replication is a **brain**: one wiring with one training history. A brain's score is its mean over 50 frozen-weight eval episodes. Episodes from one brain are not independent samples of "real wiring", and treating them as such would be pseudo-replication.

### Pilot (not counted)

One real brain, plasticity on, up to 300 episodes. It fixes three values:

- `N_train`: the episode count where the learning curve flattens, capped at 300 and defaulting to 150;
- `eta`, from {0.01, 0.03, 0.1}: the largest value that has no runaway AND keeps `frac_at_floor` under 0.5 at episode `N_train`;
- `rho`: 0 if any `eta` passes that rule. If none passes, the pilot repeats the `eta` search with `rho = 0.1`, then `rho = 0.3`, and keeps the first `rho` that lets some `eta` pass. If none passes at `rho = 0.3`, the pilot stops and reports, as G2 does;
- shaping on or off: on only if the game-reward-only pilot shows no improvement by episode 150.

The pilot also runs G2.

### Pre-registration

After the pilot, the repo commits `prereg.md` and tags it `prereg-v1`, before any main-run brain trains. It records:
- the fixed values above;
- the G1 rung;
- the hypotheses and tests below;
- the config hashes.

### Hypotheses and tests

- **Primary (H1).** Real plasticity-on brains score higher than scrambled plasticity-on brains.
  - Test: one-sided Mann-Whitney U on the 6 vs 6 brain scores, α = 0.01. That allows at most 3 of 36 inversions.
  - Report: the median difference with a bootstrap 95% CI over brains, and Cliff's delta.
- **Secondary, innate wiring (H2).** The rank of the real plasticity-off brain among the 7 plasticity-off brains. Reported descriptively; one brain cannot be tested.
- **Secondary, learning gain (H3).** Each brain's eval score with plasticity on, minus its wiring's plasticity-off score. Real versus scrambled, Mann-Whitney, α = 0.05.
- **Exploratory, after the main run.** Lesions on the best real brain: silence LC10a, AOTU019, or all PAM. Clearly labeled exploratory.

### Abort accounting

| Abort | Rule |
|---|---|
| Brain-caused (runaway: the mean population rate over the last 1 s of brain time is above 50 Hz, input neurons left out; or NaN) | Ends the episode with the score of the obs the brain answered, so the client's record and the brain's agree. It counts, even if SC2 fails before the client hears of it. |
| Environment-caused (yield, SC2 crash, client disconnect) | The episode is discarded and rerun with the same seed. |

**Seeds.** Episode `e` of a run with seed `r` has seed `r * 1,000,000 + e`. It seeds the body's starting heading and any brain-side RNG. SC2 takes `r` once per launch and places the beacons itself, so a rerun after an SC2 relaunch has the same seed but not the same beacon positions.

### Budget

- Main experiment: about 2,900 episodes (12 × 200 + 350 + 150 at `N_train = 150`).
- At about 60 s per episode: about 48 hours of PC time.
- Pilot: about 5 hours.
- At BJ's off-hours this spreads over one to two weeks.
- M5 (faster kernel) triggers only if the projected main-run time exceeds 80 hours.

### Publish either way

- The README results section covers the table, learning curves and the H1-H3 outcomes, whatever they are.
- Raw `episodes.jsonl` and calibration records go in `results/`.

## 10. Protocol

msgpack over websocket at `ws://127.0.0.1:8765`. Every message is a map with `type` and `v: 1`; on a version mismatch the brain closes the connection with an error.

| type | direction | fields |
|---|---|---|
| `hello` | client → brain | `client_version`, `map`, `screen` (84), `mode` (`watch`/`train`/`eval`) |
| `ready` | brain → client | `brain_id`, `wiring`, `scramble_seed`, `plasticity`, `eye_shape` [30, 72], `decision_frames` (8), `polarity` |
| `episode_start` | client → brain | `episode`, `seed`, `phase` (`train`/`eval`/`watch`) |
| `obs` | client → brain | `step`, `eye` (bytes, 2,160), `reward` (since last obs), `score`, `last`, `marine_xy`, `beacon_xy` (log/view only) |
| `act` | brain → client | `step`, `dtheta`, `speed`, `turn_raw`, `fwd_raw` |
| `episode_end` | client → brain | `episode`, `score`, `steps` (obs sent), `aborted` (null or reason) |
| `abort` | brain → client | `reason` (`yield`, `runaway`, `shutdown`) |
| `error` | either | `message` |

`flycraft/protocol.py` defines these messages and depends only on msgpack, so both sides import the same module.

**Session rules.**
- One client at a time. A second client gets error `busy` and is closed; the first is unaffected.
- The brain answers every `obs` with an `act`, the episode's last obs included. The client renders with the eye polarity `ready` names.
- A runaway brain gets abort `runaway` in place of the act. That ends the episode, which counts (section 9), but not the session: the brain has already ended the episode, the client sends no `episode_end`, and the next `episode_start` proceeds.
- An episode cut off by anything else (disconnect, shutdown, a protocol error) is ended on the brain side with `aborted` set to that reason. Every other abort or error closes the connection after the one message.
- The client writes one JSON line per episode to stdout (seed, phase, pilot, score, steps, frames, wall time, late frames, the longest wait for an act, `aborted`, whether it counts). The brain records nothing in M2; the M4 supervisor collects these into `episodes.jsonl`.

## 11. Brain view

**Serving.** The brain worker serves a static page at `http://127.0.0.1:8766`. BJ opens `http://localhost:8766` in a browser on the PC next to the SC2 window. The default layout puts SC2 on the left half and the browser on the right; the SC2 window size is configurable, default 1280×960, via the launch shim (`flycraft-client --window WxH`).

**Page:**
- **Library:** Three.js, vendored (MIT), as a `Points` cloud of every neuron at its position.
- **Brain cloud:**
  - Dim class colors at rest. Each spike flashes bright and fades with a 100 ms brain-time constant.
  - Highlighted groups with legends: photoreceptors by eye, LC10a, decoder DNs (large labeled markers), PAM and PPL1, KCs, MBONs.
  - Orbit controls.
- **Panel:**
  - The wiring badge, plasticity on/off, episode, step and score.
  - A reward flash.
  - Turn and forward meters.
  - The fly's-eye image.
  - Decoder DN rate sparklines and population mean rate.
  - A **Pause** button, which writes the manual pause flag (section 13).

**Endpoints:**
- `GET /meta`: JSON with neuron count, groups as index lists, condition and calibration.
- `GET /positions.bin`: float32 N×3, about 2 MB, loaded once.
- `WS /activity`: frames every 10 ms of brain time (5 per decision). Each frame carries:
  - dense uint8 spike counts per neuron (about 167 KB, saturating at 255);
  - decoder values, reward, score;
  - the eye image.

At watch pacing this is about 14 frames per second, about 2.3 MB/s over localhost.

**Rule.** The view never slows the brain. The simulation runs in its own thread. The view queue holds one frame, newest wins, and old frames are dropped.

## 12. Code layout and testing

```
flycraft/
  protocol.py            message schema, encode/decode, version check
  config.py              YAML load with `extends`, dataclasses, validation
  eye.py                 eye image geometry and calibration stimuli
  cli.py                 the `flycraft` command
  m1report.py            the M1 report
  data/fetch.py          download + crc32c verification
  data/build.py          neuron table, edges, signs, positions -> npz
  data/connectome.py     load the cache; reject or rebuild it when stale
  brain/lif.py           LIF state and step (event + SpMV paths)
  brain/spikes.py        spike-raster comparison (golden test)
  brain/brain.py         connectome, wiring, retina, LIF and decoder as one Brain
  brain/wiring.py        real/scrambled edge sets, exempt set, fingerprint
  brain/retina.py        photoreceptor directions, eye sampling, Poisson drive
  brain/decoder.py       DN rate filters, biases, gains, command
  brain/plasticity.py    eligibility, DA map, KC->MBON updates
  brain/calibrate.py     G0, G1, G2, decoder calibration
  brain/server.py        game websocket + view HTTP/websocket
  game/client.py         PySC2 loop, watch/train pacing, pipelining
  game/body.py           heading, motion, action translation
  game/render.py         eye image from screen layers
  game/sc2_compat.py     pysc2 shim (extra_ports pop, window size, unfocused launch)
  game/probe.py          Windows-side yield probe: one JSON snapshot via ctypes/winreg
  runner/supervisor.py   worker lifecycle, restarts, run records
  runner/yieldguard.py   busy/clear/resume decisions from probe snapshots
  runner/experiment.py   condition matrix, scheduling, resume
  runner/analyze.py      stats and plots
  view/static/           index.html, view.js, three.module.min.js
configs/                 base.yaml, real.yaml, scrambled.yaml, experiment.yaml
tools/make_golden.py     Brian2 reference run for the golden fixture
tools/dev.Dockerfile     CPU test image
tests/                   unit tests (CPU), tests/integration (the PC)
```

### Packaging

One `pyproject.toml` with two extras:
- `[brain]`: torch, numpy, pandas, pyarrow, websockets, msgpack, pyyaml.
- `[game]`: pysc2 pinned to 0df53d3 as the archive `https://github.com/google-deepmind/pysc2/archive/0df53d38c153972f1e368572ba65b1442a0fd41f.zip` (no git needed on Windows), `protobuf==3.20.3`, websockets, msgpack, numpy.

Python 3.11 on both sides.

### Unit tests (CPU, GitHub Actions)

- **LIF:**
  - The analytic subthreshold response to a single input.
  - Spike threshold.
  - The 1.8 ms delay.
  - Refractory hold.
  - Golden fixture: a 50-neuron network run in Brian2 with Shiu's equations and a fixed input spike train. In float64, on both kernel paths, our sim matches all 722 spikes at the same step. The float32 test allows one step of drift for at least 99% of spikes. The fixture is committed, so CI does not need Brian2.
- **GPU and CPU:** identical spikes over 200 ms on the small network. Run on the PC only.
- **Scramble:**
  - Per-neuron out-degree, excitatory in-degree and inhibitory in-degree are preserved.
  - Signs and the outgoing weight multiset are preserved.
  - The exempt set is untouched.
  - Determinism per seed, and different seeds differ.
  - Leftover duplicates stay under 0.1%.
- **Retina:**
  - Direction mapping is mirrored between eyes.
  - Bilinear sampling.
  - Eye normalization equalizes total drive.
- **Decoder:**
  - Bias and gain calibration on synthetic rates.
  - Gain floor: a silent turn or forward signal gives a gain of exactly 0, the matching `*_silent` flag, and finite commands.
  - Missing cell types fail loudly.
- **Plasticity:**
  - Depression sign.
  - Floor at 0, cap at w0.
  - Only the exempt-set slots change.
  - The commanded-pulse DA path.
  - `frac_at_floor`, overall and per compartment.
  - The recovery term: with `rho > 0`, repeated pulses settle at `w0 * rho / (rho + e_k)` rather than 0; with `rho = 0`, it changes nothing.
- **End-to-end chirality (code chain, CPU).** No brain, so a mirror anywhere in our code fails it:
  1. The marine faces screen-up, and the beacon sits to its screen-right.
  2. The eye-image disk lands in the right half (`az > 0`).
  3. The right-eye photoreceptors sample a darker image than the left-eye ones.
  4. Synthetic rates with `DNa02_R > DNa02_L` give `dtheta > 0`.
  5. After the body applies the command, the next move target is rotated clockwise on screen, toward the beacon. (Step 5 needs the body, so it lands in M2.)

  It runs for the mirror case too, beacon on the left with every sign flipped. It shares no bearing code with the oracle's test path: it computes expected values from fixed geometry written into the test.
- **Body and render:**
  - Egocentric bearing for known geometries.
  - Looming size.
  - Heading integration and turn clipping.
  - Move target clipping.
- **Protocol:** round-trip for every message type, and rejection of a version mismatch.
- **Boundary invariance:** changing `marine_xy` and `beacon_xy` with the same `eye` gives an identical brain output.
- **Yield guard:** a fixture probe snapshot for each busy rule, the never-busy list, and the resume gate (clear time and user idle each short of 10 minutes keep it waiting) (section 13).

**CI** runs `ruff check .` and `pytest -q -m "not gpu and not integration"`. Lint only: the code is hand-wrapped, so `ruff format` is not enforced.

### Integration tests (on the PC, manual)

- Oracle stub brain: one MoveToBeacon episode through real SC2 scores at least 15.
- Watch mode holds 22.4 frames per second.
- **Chirality, brain chain (M1, real brain, offline).** A dark spot at +60° drives right LC10a above left. The report states whether `DNa02_R` exceeds `DNa02_L`, and by how many standard errors. The mirror case is run too.
- **Chirality, on screen (M3).** A beacon placed to the marine's right gives a clockwise first turn in the real game, checked from `marine_xy` across the first decisions.
- The probe reports the right snapshot on the PC: a fullscreen window counts as busy, and the idle time grows while nobody touches the machine.
- In train mode, SC2 launches without taking focus.
- The real brain plays one episode with the view connected.

## 13. Running on a gaming PC: yield guard

The PC is BJ's gaming machine, shared with this project. The rules:

**Never auto-start.**
- No scheduler, startup entry or cron.
- Runs start only from an explicit `flycraft experiment start` or `flycraft watch`, started by hand on request.
- A run started on request may resume itself after a yield. It never starts a new run.

**The probe.** `flycraft-probe` (Windows Python, `game/probe.py`) prints one JSON snapshot. The supervisor runs it through `cmd.exe` interop. Interop processes land in BJ's console session (session 1), so the Windows calls see his desktop. The snapshot holds:
- `steam_app_id`, from `HKCU\Software\Valve\Steam\RunningAppID`;
- `processes`, the running image names with PIDs;
- `foreground`, the foreground window's PID, image name, and whether it covers its whole monitor (`GetForegroundWindow`, `GetWindowRect`, `MonitorFromWindow`). The desktop shell (`Progman`, `WorkerW`) never counts as fullscreen;
- `idle_s`, the seconds since the last keyboard or mouse input (`GetLastInputInfo`);
- `pause`, whether the pause flag exists.

`runner/yieldguard.py` makes every decision from these snapshots, so its unit tests run on fixture snapshots.

**Busy signals**, checked before each episode and every 30 s:
1. Steam `RunningAppID` is not 0.
2. A process on the denylist is running. Seed list:
   - `iRacingSim64DX11.exe`
   - `AC2-Win64-Shipping.exe`
   - `Le Mans Ultimate.exe`
   - `rFactor2.exe`
   - `AMS2AVX.exe`
   - any `SC2_x64.exe` we did not launch
3. The manual pause flag `%USERPROFILE%\flycraft\PAUSE` exists. It is set from the view's Pause button or `flycraft pause`.
4. **Catch-all:** the foreground window is fullscreen and belongs to a process that is not ours. Ours means the SC2 we launched, by PID. This catches games outside Steam and the denylist, and full-screen video.

In watch mode, BJ is at the machine on purpose, so only rule 3 applies. Train and eval apply all four.

**Never busy signals.** Processes that are always running at idle never count. Observed on 10-07:
- the sim-racing services, telemetry tools and overlays that sit in the tray (six on 10-07)

**No GPU-utilization check.** WDDM does not attribute GPU memory per process reliably. An optional total-utilization heuristic exists and is off by default.

**On busy:**
1. The client aborts the episode and closes SC2. The episode is discarded.
2. The worker saves a checkpoint at the last completed episode and exits, freeing about 2.5 GB of VRAM.
3. The supervisor resumes from the checkpoint only when both hold: 10 continuous minutes of clear signals, and `idle_s` of at least 600. If BJ quits a game and starts browsing, the run waits until he has been away for 10 minutes. It does not pop SC2 onto his desktop mid-browse.

**Not stealing focus.** One SC2 process serves a whole worker session; it is not relaunched per episode. In train and eval mode, the launch shim starts SC2 shown but not activated (`SW_SHOWNOACTIVATE` in `STARTUPINFO.wShowWindow`), and the client minimizes the window by PID right after launch without activating it (`SW_SHOWMINNOACTIVE`). M2 found that SC2 started minimized cannot get a graphics device: it shows a "graphics device not available" error and quits. Watch mode launches normally, since BJ wants the window.

**Optional hours.** `yield.allowed_hours` (for example `22:00-07:00` Mountain) restricts training further. It is off by default.

**Install footprint.** User-level only, with every addition logged in `%USERPROFILE%\flycraft\INSTALL_LOG.txt` along with removal steps. This continues the spike's practice.

## 14. Error handling and records

**Startup failures**, which fail fast with a clear message:
- Missing data, a hash mismatch or an unknown cell type.
- A protocol version mismatch.
- Invalid wiring and seed combinations.
- A game that cannot start as configured: an unknown map, or client arguments out of range (a seed outside SC2's 32-bit range, a non-positive screen, step or timeout).

**Client or SC2 failure mid-episode.**
- The episode is discarded and the client relaunched.
- Three consecutive failures stop the run with an error.
- A lost brain (disconnect, shutdown, no act within the timeout) stops the client's run at once: there is nothing to rerun against.
- An episode played to its end is recorded before the client tells the brain, so a brain that stops at that moment does not cost the episode.
- pysc2's own failures (connection, protocol, and "the game didn't advance") all count as SC2 failures.

**Runaway or NaN in the brain.** Abort per the rule in section 9. Logged with the population rate trace.

**Run record**, in `~/flycraft-runs/<run_id>/` in WSL:
- `run.json`: config, git sha, wiring fingerprint, SC2 build, pysc2 commit.
- `calibration.json`
- `episodes.jsonl`: one line per episode with phase, seed, score, steps, wall time and abort reason.
- `checkpoints/`: plastic weights plus RNG states, about 0.5 MB each.
- `logs/`

After each run, a copy is pulled off the PC for analysis.

## 15. Milestones

| ID | Delivers | Exit check |
|---|---|---|
| M1 | Data build, LIF with golden test, wiring and scramble, retina geometry, decoder, G0 and G1 run offline on the WSL GPU | M1 report: counts, sign-rule check, retina map, G0 scale, G1 rung, brain-chain chirality. The repo goes public here (MIT, `github.com/bjfultn/flycraft`). |
| M2 | Windows client, body, eye render, protocol, watch pacing; oracle and random stub brains; baselines recorded | Oracle body plays smoothly in watch mode and scores near the scripted agent. |
| M3 | Real brain in the loop, plus the brain view | BJ watches the fly play MoveToBeacon with neurons lighting up: the first real demo. |
| M3b | DefeatRoaches demo: `attack` and `target` outputs, a moving-target eye input, untrained weights. A demo, not an experiment: no pre-registration, its own short spec | BJ watches a marine squad turn on the roaches and attack, driven by the LC10a pursuit chain. A reflex, not tactics. |
| M4 | Plasticity, G2, yield guard, supervisor, experiment runner, analysis; pilot, pre-registration, main experiment | Results table and README write-up, published either way. |
| M5 | Fused CUDA/Triton kernel, at or below 1.5 s per simulated second | Only if the M4 projection exceeds 80 h. |
| M6 | Skirmish: BJ plays the fly over LAN through PySC2 `play_vs_agent`, on a small map | BJ plays a match against the fly. |
| M7 | Full 1v1: a scripted bot runs the economy and production; the fly commands the army | BJ plays a full game against the fly. |

**M2 result (2026-10-08): passed.** In watch mode the oracle scored 22, 18 and 20 at 22.4 fps with 2 late frames in 5,760. Over 50 train episodes it scored 20.62 against scripted's 25.94 (0.79×). The gap is the body's designed limits, not a fault: lifting the turn limit gives 22.22, and scripted made one decision late gives 22.58 (section 9).

**M3 result (2026-10-08): the demo runs; the exit check is BJ watching it.** Untrained, with plasticity off, the real wiring scored 0, 6, 5 and 7 in four watch episodes (mean 4.5), against random's 0.24 and the oracle's 20.6. The on-screen chirality check passes. It turns toward a beacon to its side 77% of the time and barely sees one behind it (LC10a look forward): episode 0 started with the beacon 170 degrees off, walked into a corner and never turned. It hunts at close range and leans left (a beacon on the right gets a right turn only 53% of the time); the lean is unexplained. Watch mode ran at 18.6 fps (section 7.5). Details in `docs/m3/README.md`.

**Later minigames** (a separate spec after Phase 1 results):
- **CollectMineralShards.** Two marines: one body per marine, each with its own eye image.
- **DefeatRoaches.** Moved up to M3b as a demo (BJ, 2026-10-07); the experiment version still waits for this spec. Adds `attack` and `target` outputs. An attack DN group, chosen by a criterion pre-registered in that spec, triggers `Attack_screen` on the object nearest the center of the frontal visual field.
- **Protocol.** The `act` message gets new fields under a protocol version bump.

## 16. Risks

| Risk | Mitigation |
|---|---|
| Histaminergic photoreceptor drive never propagates | G1 ladder, ending in LC10a injection |
| KC→MBON learning cannot move the steering output | G2 stops the pilot before the main runs |
| Depression-only learning saturates into a PAM-compartment lesion | `frac_at_floor` logged per episode; the eta rule caps it at 0.5; a DA-gated recovery term is the pre-set fallback |
| A sign flip between screen and steering mirrors the fly | One handedness convention (7.6); end-to-end chirality tests in code, brain and game |
| A silent twin's decoder gain divides by zero | Gain floor: a silent signal gets gain 0 and the brain walks straight |
| Resuming after a yield pops SC2 onto BJ's desktop | Resume needs 10 min of user idle; train mode launches SC2 minimized without focus |
| MaleCNS weight scale differs from FlyWire's | G0 per-brain calibration |
| Retina orientation wrong (front/back mirrored) | M1 scatter plus left/right test; side-band fallback |
| Scrambled twins run away or go silent, giving an unfair control | Same G0 and decoder calibration for every brain; G0 records reported |
| Pseudo-replication | Brain-level unit of analysis, 6 vs 6 |
| Compute time | Budget about 48 h; M5 gated at 80 h |
| BJ's gaming interrupts runs | Yield guard, per-episode checkpoints, discard-and-rerun |
| An SC2 or pysc2 update breaks the client | pysc2 pinned at 0df53d3 with the shim; the SC2 build is recorded per run; s2clientprotocol 5.0.16 drives Base95841 today |
| Left eye under-reconstructed, giving a turning bias | Eye normalization, plus decoder bias calibration on a blank scene |

## 17. License and attribution

- **Code:** MIT.
- **NOTICE** credits:
  - MaleCNS v1.0 (CC-BY 4.0, with its citation);
  - Shiu et al. 2024 (*Nature*) for the model;
  - doomfly, fruitfly-lab and drosophila-brain-mlx for methods;
  - PySC2 (Apache-2.0);
  - Three.js (MIT, vendored).
- **Not redistributed:** StarCraft II, its maps and the connectome data. Users fetch them: `flycraft prep-data` for the data; the mini-games zip under Blizzard's AI and Machine Learning license.
