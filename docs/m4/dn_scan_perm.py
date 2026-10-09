"""Permutation null for dn_scan.py: how many DNs would pass by chance?

Within each spot, the base and arm trials are shuffled together and split again, and the same
test is rerun. Trials share network state, so DN responses are correlated and the null count has
a heavy tail; the count-level p is the share of shuffles with at least the observed hits.

  python docs/m4/dn_scan_perm.py docs/m4/dn_scan_rates.npz docs/m4/dn_scan.json
"""
import json
import sys

import numpy as np

PERMS = 5000


def zscores(base, arm):
  n = base.shape[1]
  diff = (arm.mean(axis=1) - base.mean(axis=1)).mean(axis=0)
  var = (arm.var(axis=1, ddof=1) / n + base.var(axis=1, ddof=1) / n).sum(axis=0)
  se = np.sqrt(var) / base.shape[0]
  z = np.divide(diff, se, out=np.zeros_like(diff), where=se > 0)
  z[(se == 0) & (diff != 0)] = np.inf
  return diff, z


def hits(base, arm, z_crit, min_hz):
  diff, z = zscores(base, arm)
  return (np.abs(z) > z_crit) & (np.abs(diff) >= min_hz)


def main(rates_path, scan_path):
  rates, scan = np.load(rates_path), json.loads(open(scan_path).read())
  z_crit, min_hz = scan["z_crit"], scan["min_hz"]
  rng = np.random.default_rng(0)
  base = rates["base"]
  n = base.shape[1]
  for arm in ("kc_mbon0", "mbon_out0"):
    obs = int(hits(base, rates[arm], z_crit, min_hz).sum())
    both = np.concatenate([base, rates[arm]], axis=1)
    counts = np.zeros(PERMS, np.int64)
    for k in range(PERMS):
      p = np.stack([rng.permutation(s) for s in both])
      counts[k] = hits(p[:, :n], p[:, n:], z_crit, min_hz).sum()
    p_count = (np.sum(counts >= obs) + 1) / (PERMS + 1)
    print(f"{arm}: observed {obs} hits; null mean {counts.mean():.2f}, "
          f"95th {np.percentile(counts, 95):g}, 99th {np.percentile(counts, 99):g}, "
          f"max {counts.max()}; P(null >= observed) = {p_count:.4f} ({PERMS} shuffles)")


if __name__ == "__main__":
  main(*sys.argv[1:3])
