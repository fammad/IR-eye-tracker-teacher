"""Link to the GazeMap repository, without changing it.

This repository started as a copy of GazeMap (with its history), so GazeMap's
code sits in the repo root. The study scripts import its network, data loading
and live-demo functions from there.

Where GazeMap's code is found, in this order:
  1. the GAZEMAP_DIR environment variable
  2. this repository's root, if gaze_model.py is there
  3. a folder called 00_GazeMap next to this repository

GazeMap uses paths relative to its own folder ("dataset/...", "results/final/models"),
so they are turned into absolute paths here. Then the study scripts can be run
from anywhere.
"""

import os
import sys
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parent
_default = STUDY_DIR if (STUDY_DIR / "gaze_model.py").exists() else STUDY_DIR.parent / "00_GazeMap"
GAZEMAP_DIR = Path(os.environ.get("GAZEMAP_DIR", _default)).resolve()

if not (GAZEMAP_DIR / "gaze_model.py").exists():
    raise SystemExit(f"GazeMap not found at {GAZEMAP_DIR}. Set GAZEMAP_DIR to the GazeMap folder.")

sys.path.insert(0, str(GAZEMAP_DIR))

import gaze_model  # noqa: E402

gaze_model.BASE = GAZEMAP_DIR / "dataset" / "Data" / "Normalized"
MODEL_DIR = GAZEMAP_DIR / "results" / "final" / "models"


def demo():
    """GazeMap's live-demo module, with its file paths pointing into the GazeMap folder."""
    import gaze_demo

    gaze_demo.FACE_MODEL_PATH = GAZEMAP_DIR / "dataset" / "6 points-based face model.mat"
    gaze_demo.LANDMARKER_PATH = GAZEMAP_DIR / "face_landmarker.task"
    return gaze_demo
