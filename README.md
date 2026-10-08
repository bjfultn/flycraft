# flycraft

A whole-brain fruit-fly connectome plays StarCraft II.

flycraft runs a spiking model of the male *Drosophila* central nervous system (MaleCNS v1.0,
164,587 leaky integrate-and-fire neurons) and gives it a body in SC2. The fly sees the
game through its photoreceptors, steers through its descending neurons, and learns only
through dopamine-gated plasticity in the mushroom body. Every run has a scrambled-wiring twin
with the same degrees and signs, so we can ask whether the real wiring is doing the work.

Status: milestone M3. The connectome steers a marine in the MoveToBeacon minigame, with a live
view of its neurons. Untrained, it scores 4.5 a game against 0.24 for a random walker and 20.6
for an oracle with the same body (`docs/m3`). Learning comes in M4.

## Install

Python 3.11. The brain needs PyTorch; install the build for your machine first
(https://pytorch.org), then:

```
pip install -e ".[brain]"
```

## Use

```
flycraft prep-data     # download MaleCNS v1.0 (about 566 MB) and build the cache
flycraft wiring        # print the real wiring's fingerprint
flycraft wiring --config configs/scrambled.yaml --set connectome.scramble_seed=2
flycraft m1-report --device cuda   # calibrate the brain, write runs/m1/report.md
```

To play, run the brain on the machine with the GPU and the game client where SC2 is installed
(we use WSL and Windows on one PC; the client reaches the brain on localhost):

```
flycraft brain --calibration docs/m1/calibration.json   # game on :8765, brain view on :8766
flycraft brain --stub oracle                             # or a stand-in brain, no GPU
```

```
pip install -e ".[game]"                      # on the SC2 machine
flycraft-client --mode watch --episodes 4 --seed 1 > watch.jsonl
flycraft-client --scripted --mode train --episodes 50 > scripted.jsonl
flycraft summarize watch.jsonl scripted.jsonl
```

`--map DefeatRoaches` plays the roach minigame instead of MoveToBeacon: the fly sees every
roach, and the squad attacks the roach it turns toward (within 30 degrees of its heading), else
attack-moves where it steers ([M3b spec](docs/superpowers/specs/2026-10-08-flycraft-m3b-defeatroaches.md)).

Open http://localhost:8766 to watch the brain while it plays. Watch mode shows the game at
normal speed; train mode runs minimized and as fast as the brain allows.

`--set KEY=VALUE` overrides any config value. Exit codes: 0 success, 1 an error you can act
on (the message says what), 2 a bad config, 3 `m1-report` ran but no rung passed G1.

## Design

The design spec, with the pre-registered experiment and the milestones, is
[docs/superpowers/specs/2026-10-07-flycraft-design.md](docs/superpowers/specs/2026-10-07-flycraft-design.md).
Code comments cite its sections ("spec section 7.4"). Each milestone's run records and
write-up are in `docs/m1` to `docs/m3`.

## Tests

```
pip install -e ".[brain,dev]"
pytest -q
```

The LIF simulator is checked spike for spike against a Brian2 run of Shiu et al.'s model
(`tools/make_golden.py`; the fixture is committed so the tests do not need Brian2). Tests marked
`gpu` or `integration` need a CUDA device or the real data.

## Credits

See NOTICE. The connectome is CC-BY 4.0 from Janelia FlyEM and Google; the model follows
Shiu et al. 2024 (*Nature*). flycraft does not ship the data, StarCraft II or its maps.

## License

MIT.
