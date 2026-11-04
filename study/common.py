"""Shared pieces for the study scripts: where data goes, the dot layouts, block files.

One participant = one session folder:
  data/<session>/camera.json                  focal length measured for this person (measure_distance.py)
  data/<session>/<room>_<block>_<time>/       one recorded block (record_block.py)
      meta.json        settings, screen size, dot order, software versions
      events.csv       what the screen showed and when (dot on, settle over, dot off, cues)
      webcam.avi       raw webcam video          webcam_frames.csv  time of every frame
      ir.avi           raw IR video (if used)    ir_frames.csv      time of every frame
      gaze.npz         per-frame eye predictions and geometry (process_block.py)

All times are seconds on one clock (time.perf_counter of the recording process),
so webcam frames, IR frames and screen events can be matched afterwards.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gm import GAZEMAP_DIR, STUDY_DIR  # noqa: E402

DATA_DIR = STUDY_DIR / "data"

# Same dot positions as GazeMap (fractions of the window): 3x3 calibration grid and 4x4 test grid.
GRID9 = [(x, y) for y in (0.06, 0.5, 0.94) for x in (0.06, 0.5, 0.94)]
GRID16 = [((c + 0.5) / 4, (r + 0.5) / 4) for r in range(4) for c in range(4)]
CHECK3 = [(0.2, 0.2), (0.8, 0.5), (0.3, 0.85)]
BLOCKS = {"calib9": GRID9, "test16": GRID16, "check3": CHECK3}

SETTLE_SECONDS = 1.0   # eyes travel to the dot; frames are kept but marked, never used for fitting
COLLECT_SECONDS = 1.5  # frames used for fitting and testing


def session_dir(session):
    if not session.replace("_", "").replace("-", "").isalnum():
        raise SystemExit("Session id: letters, digits, - and _ only (e.g. P03 or PILOT1). No names.")
    path = DATA_DIR / session
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_camera(session):
    """Focal length measured for this participant, or None (then GazeMap's default guess is used)."""
    path = DATA_DIR / session / "camera.json"
    return json.loads(path.read_text()) if path.exists() else None


def find_block(session, room, block):
    """The most recent recording of this room and block type, e.g. ("P03", "B", "test16")."""
    found = sorted((DATA_DIR / session).glob(f"{room}_{block}_*"))
    found = [f for f in found if (f / "meta.json").exists()]
    if not found:
        raise SystemExit(f"No {room}_{block} block in {DATA_DIR / session}. Record it first.")
    return found[-1]


def git_commit(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def dot_phases(events):
    """From events.csv rows: list of (dot index, fx, fy, settle_end time, off time)."""
    dots, current = [], None
    for e in events:
        t = float(e["t"])
        if e["event"] == "dot_on":
            current = {"dot": int(e["dot"]), "fx": float(e["fx"]), "fy": float(e["fy"]), "on": t}
        elif e["event"] == "settled" and current is not None:
            current["settled"] = t
        elif e["event"] == "dot_off" and current is not None:
            if "settled" in current:
                dots.append((current["dot"], current["fx"], current["fy"], current["settled"], t))
            current = None
    return dots


def frame_targets(frame_times, events):
    """For every frame time: (dot index, fx, fy) if it falls in a dot's collect phase, else (-1, nan, nan)."""
    out = np.full((len(frame_times), 3), np.nan)
    out[:, 0] = -1
    for dot, fx, fy, start, end in dot_phases(events):
        m = (frame_times >= start) & (frame_times < end)
        out[m] = (dot, fx, fy)
    return out


__all__ = ["DATA_DIR", "GAZEMAP_DIR", "STUDY_DIR", "BLOCKS", "GRID9", "GRID16", "CHECK3",
           "SETTLE_SECONDS", "COLLECT_SECONDS", "session_dir", "load_camera", "find_block",
           "git_commit", "write_csv", "read_csv", "dot_phases", "frame_targets"]
