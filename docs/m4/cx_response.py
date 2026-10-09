"""Gate G2's turn measure on the CX scan's arms: does the arm change how the fly's turn depends
on where the spot is? The turn response is the turn signal (DNa02 R minus DNa02 L, Hz) at the
rightmost spot minus the leftmost; its z is after minus before over the trial variances.

  python docs/m4/cx_response.py docs/m4/cx_scan_rates.npz
"""
import sys

import numpy as np


def turn(z, arm):
  return z[f"{arm}/DNa02 R"] - z[f"{arm}/DNa02 L"]  # (spots, repeats)


def main(path):
  z = np.load(path)
  arms = sorted({k.split("/")[0] for k in z.files} - {"base"})
  b = turn(z, "base")
  print(f"base turn response {(b[-1] - b[0]).mean():.2f} Hz")
  for arm in arms:
    a = turn(z, arm)
    d = (a[-1] - a[0]).mean() - (b[-1] - b[0]).mean()
    se = np.sqrt(sum(x.var(ddof=1) / x.size for x in (a[-1], a[0], b[-1], b[0])))
    print(f"{arm}: turn response {(a[-1] - a[0]).mean():.2f} Hz, change {d:+.2f}, z {d / se:.2f}")


if __name__ == "__main__":
  main(*sys.argv[1:])
