"""Summary of an ff_probe.py record: python docs/m6/ffstats.py docs/m6/ff-after.jsonl"""
import json
import statistics as st
import sys
from collections import Counter

recs = [json.loads(line) for line in open(sys.argv[1])]
att = [r for r in recs if r["kind"] == "attack"]
stale = [r["stale"] for r in recs]
waits = [r["wait_ms"] for r in recs if r.get("wait_ms") is not None]
dnow = [r["d_now"] for r in att if r["d_now"] is not None]
print(f"decisions {len(recs)}, kinds {dict(Counter(r['kind'] for r in recs))}")
print(f"stale loops: median {st.median(stale)}, max {max(stale)}")
if waits:
  print(f"probe wait_ms: median {st.median(waits)}, max {max(waits)}")
print(f"attack orders {len(att)}: on own pixels when aimed {sum(r['px_then'] == 1 for r in att)}, "
      f"when sent {sum(r['px_now'] == 1 for r in att)}; d_now < 2 px: {sum(d < 2 for d in dnow)}, "
      f"min d_now {min(dnow) if dnow else None}")
