"""Gate 1: is there a gap for the IR teacher to close?

Run on yourself before building the study. Three blocks of the same session:
  A calib9   9-point calibration in room A (the "old" calibration)
  B test16   16-dot test in room B (other room, light, seat)
  B calib9   a fresh 9-point calibration in room B (the best case, "ceiling")

All four corrections are scored on the same room B test frames:
  generic        no personal correction
  stale offset   room A offsets               <- the study's baseline
  stale affine   room A offsets + screen map
  fresh offset   room B offsets               <- ceiling for the baseline
  fresh affine   room B offsets + screen map

The gap = stale offset - fresh offset. That is the most any room A personalisation
(including the IR teacher) can gain over the baseline in room B.
  gap below 0.5 deg : room B is too similar to room A. Make the change bigger
                      (sit further back, other light) or the study cannot show anything.
  gap 0.5 deg or more: there is room for the teacher to help. Proceed to Gate 2.
The interval comes from resampling the 16 test dots, so it shows how sure one
session can be, not how other people will behave.

Usage:
  python study/gate1.py --session PILOT1
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import find_block, write_csv  # noqa: E402
from analysis_lib import (Block, bootstrap_mean_diff, dot_errors, fit_calibration,  # noqa: E402
                          head_summary, rotation_between_deg)

THRESHOLD_DEG = 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()

    a_cal = Block(find_block(args.session, "A", "calib9"))
    b_test = Block(find_block(args.session, "B", "test16"))
    b_cal = Block(find_block(args.session, "B", "calib9"))

    stale, fresh = fit_calibration(a_cal), fit_calibration(b_cal)
    conditions = {
        "generic": dot_errors(b_test),
        "stale offset": dot_errors(b_test, stale["offsets"]),
        "stale affine": dot_errors(b_test, stale["offsets"], stale["mapping"]),
        "fresh offset": dot_errors(b_test, fresh["offsets"]),
        "fresh affine": dot_errors(b_test, fresh["offsets"], fresh["mapping"]),
    }
    dots = sorted(set.intersection(*[{r["dot"] for r in rows} for rows in conditions.values()]))
    per_dot = {c: np.array([next(r["deg"] for r in rows if r["dot"] == d) for d in dots])
               for c, rows in conditions.items()}

    dist_a, head_a = head_summary(a_cal)
    dist_b, head_b = head_summary(b_test)
    print(f"\nSession {args.session}: room A calib {a_cal.folder.name}, room B test {b_test.folder.name}")
    print(f"Room A notes: '{a_cal.meta.get('note', '')}' | room B notes: '{b_test.meta.get('note', '')}'")
    print(f"Eye distance: room A {dist_a / 10:.1f} cm, room B {dist_b / 10:.1f} cm | "
          f"head rotation A vs B: {rotation_between_deg(head_a, head_b):.1f} deg")
    print(f"Test dots used: {len(dots)} of 16\n")
    print(f"  {'condition':14s} {'mean deg':>8s} {'median':>7s} {'mean cm':>8s}")
    for c, rows in conditions.items():
        mm = np.mean([r["mm"] for r in rows if r["dot"] in dots])
        print(f"  {c:14s} {per_dot[c].mean():8.2f} {np.median(per_dot[c]):7.2f} {mm / 10:8.2f}")

    gap, lo, hi = bootstrap_mean_diff(per_dot["stale offset"], per_dot["fresh offset"])
    print(f"\nGap (stale offset - fresh offset): {gap:+.2f} deg, 95% interval {lo:+.2f} to {hi:+.2f}")
    if gap >= THRESHOLD_DEG and lo > 0:
        verdict = "PASS: a clear gap. The teacher has something to close. Go to Gate 2."
    elif gap >= THRESHOLD_DEG:
        verdict = "UNSURE: the gap is large on average but the interval reaches zero. Repeat once more."
    else:
        verdict = (f"FAIL: gap below {THRESHOLD_DEG} deg. Room B is too close to room A for this person. "
                   "Make room B more different (distance, light, seat) and run again.")
    print(verdict)

    rows = [{"dot": d, **{c: round(float(per_dot[c][k]), 3) for c in per_dot}} for k, d in enumerate(dots)]
    out = b_test.folder.parent / "gate1.csv"
    write_csv(out, rows, ["dot", *per_dot])
    (b_test.folder.parent / "gate1.txt").write_text(
        f"gap {gap:+.3f} [{lo:+.3f}, {hi:+.3f}] deg\n{verdict}\n"
        + "\n".join(f"{c}: {per_dot[c].mean():.3f}" for c in per_dot) + "\n")
    print(f"Saved {out} and gate1.txt")


if __name__ == "__main__":
    main()
