# M1 run, 2026-10-07

Output of `flycraft m1-report` on MaleCNS v1.0, run on an RTX 5080 under WSL2.

- `report.md`: the human-readable report (counts, sign-rule check, retina map, G0, G1, chirality).
- `calibration.json`: the machine-readable record M2 loads.
- `retina.png`: the photoreceptor and LC10a map.
- `scrambled-1.txt`: `flycraft wiring` for the real wiring and scrambled twin 1.

Result: G1 passed on the `vpn` rung (LC10a z +126.8, DNa02 z +4.42) at w_scale 0.8438.
The photoreceptor-only rungs did not reach LC10a.

The run used the M1 code as first written, before the final review fixes. Those fixes only
touch validation, error handling and reporting, plus two edge cases that do not occur in this
data; the real-data retina map and signs were checked identical before and after. This record
predates the `git_sha` and `flycraft_version` fields and the signal-naming headline.
