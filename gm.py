"""Connects the study scripts to the webcam model in gazemap/.

The webcam model is GazeMap (github.com/fammad/GazeMap). Its code was copied into
gazemap/ with its git history; the trained weights are not retrained here but
taken from GazeMap's release v1.0.0 (15 leave-one-person-out models).

GazeMap's code opens files by paths relative to the folder it is run from
("dataset/...", "results/final/models", "face_landmarker.task"). Those files stay
in the repository root, and here they are turned into absolute paths, so the
study scripts can be run from anywhere.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GAZEMAP_DIR = ROOT / "gazemap"          # the webcam model code
STUDY_DIR = ROOT                        # kept for older imports

MODELS_RELEASE = "v1.0.0"
MODELS_URL = f"https://github.com/fammad/GazeMap/releases/download/{MODELS_RELEASE}/gazemap_models.zip"
MODEL_DIR = ROOT / "results" / "final" / "models"
N_MODELS = 15

if not (GAZEMAP_DIR / "gaze_model.py").exists():
    raise SystemExit(f"Webcam model code not found in {GAZEMAP_DIR}.")

sys.path.insert(0, str(GAZEMAP_DIR))

import gaze_model  # noqa: E402

gaze_model.BASE = ROOT / "dataset" / "Data" / "Normalized"


def demo():
    """GazeMap's live pipeline (landmarks, head pose, eye crops, screen geometry), with absolute paths."""
    import gaze_demo

    gaze_demo.FACE_MODEL_PATH = ROOT / "dataset" / "6 points-based face model.mat"
    gaze_demo.LANDMARKER_PATH = ROOT / "face_landmarker.task"
    gaze_demo.MODELS_URL = MODELS_URL
    return gaze_demo


def ensure_models():
    """Download GazeMap's trained models (release v1.0.0, about 100 MB) once, if they are not here yet."""
    if len(list(MODEL_DIR.glob("lopo_p*.pt"))) < N_MODELS:
        demo().download_models(MODEL_DIR)
    found = len(list(MODEL_DIR.glob("lopo_p*.pt")))
    if found < N_MODELS:
        raise SystemExit(f"Only {found} of {N_MODELS} models in {MODEL_DIR}. Download {MODELS_URL} "
                         "and unzip it there.")
    return MODEL_DIR
