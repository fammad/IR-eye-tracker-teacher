"""Turn a recorded block into per-frame gaze data (gaze.npz). Safe to run again at any time.

For every webcam frame, GazeMap's own pipeline runs unchanged: MediaPipe face
points -> head pose -> both eyes cut out and straightened -> the 15 fold models
-> pitch/yaw per eye. Nothing is personalised here; offsets and fine-tuning are
fitted later from these numbers.

The focal length comes from the session's camera.json (measure_distance.py).
Without it, GazeMap's default guess (image width) is used and a warning is printed.

Saved per frame (arrays in gaze.npz, one row per webcam frame):
  t            frame time (s, shared clock)
  face         1 if a face was found
  dot, fx, fy  the dot on screen during its collect phase, else -1 / nan
  raw          (2, 2)   predicted pitch, yaw per eye [right, left], camera-side (left un-mirrored)
  R            (2, 3, 3) normalization rotation per eye
  center       (2, 3)   eye centre in camera coordinates (mm)
  head_R       (3, 3)   head rotation
  eye_img      (2, 36, 60) the eye images the network saw (uint8, left as seen, not mirrored)

Usage:
  python study/process_block.py data/P03/A_calib9_20261120-151200
  python study/process_block.py data/P03            # every block of the session
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import frame_targets, read_csv  # noqa: E402
from gm import MODEL_DIR, demo  # noqa: E402


def process(folder, gd, models, face_model):
    meta = json.loads((folder / "meta.json").read_text())
    events = read_csv(folder / "events.csv")
    times = np.array([float(r["t"]) for r in read_csv(folder / "webcam_frames.csv")])
    focal = meta.get("focal_px")
    if focal is None:
        print(f"  WARNING {folder.name}: no measured focal length, using GazeMap's default guess.")
    else:
        measured_w = meta["camera_json"]["image_size"][0]
        video_w = meta["webcam"]["size"][0]
        if measured_w != video_w:  # focal length in pixels scales with image width
            focal = focal * video_w / measured_w
            print(f"  NOTE {folder.name}: distance was measured at width {measured_w}, video is {video_w}; "
                  f"focal scaled to {focal:.0f} px.")

    cap = cv2.VideoCapture(str(folder / meta["webcam"]["file"]))
    detect = gd.make_landmarker()
    n = len(times)
    out = {
        "t": times, "face": np.zeros(n, np.int8),
        "raw": np.full((n, 2, 2), np.nan), "R": np.full((n, 2, 3, 3), np.nan),
        "center": np.full((n, 2, 3), np.nan), "head_R": np.full((n, 3, 3), np.nan),
        "eye_img": np.zeros((n, 2, 36, 60), np.uint8),
    }
    targets = frame_targets(times, events)
    out["dot"], out["fx"], out["fy"] = targets[:, 0].astype(int), targets[:, 1], targets[:, 2]

    start = time.time()
    for k in range(n):
        ok, frame = cap.read()
        if not ok:
            print(f"  {folder.name}: video ended at frame {k} of {n}")
            break
        h, w = frame.shape[:2]
        camera_matrix = gd.default_camera_matrix(w, h, focal)
        result = gd.process_frame(frame, detect, int(times[k] * 1000), face_model, camera_matrix, models)
        if result is None:
            continue
        eyes = result[0]
        out["face"][k] = 1
        for e, r in enumerate(eyes):
            out["raw"][k, e] = r["raw"]
            out["R"][k, e] = r["R"]
            out["center"][k, e] = r["center"]
            out["eye_img"][k, e] = r["eye_img"]
        out["head_R"][k] = eyes[0]["head_R"]
    cap.release()

    np.savez_compressed(folder / "gaze.npz", **out)
    used = (out["dot"] >= 0)
    print(f"  {folder.name}: {n} frames, face in {out['face'].mean():.0%}, "
          f"{int((used & (out['face'] == 1)).sum())} usable dot frames, {time.time() - start:.0f} s")
    missing = [d for d in range(len(meta["dots"]))
               if ((out["dot"] == d) & (out["face"] == 1)).sum() < 5]
    if missing and meta.get("status") == "complete":
        print(f"  WARNING: dots {missing} have fewer than 5 usable frames.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    folders = []
    for p in map(Path, args.paths):
        folders += [p] if (p / "meta.json").exists() else sorted(f.parent for f in p.glob("*/meta.json"))
    if not folders:
        raise SystemExit("No recorded blocks found.")

    gd = demo()
    models = gd.load_gaze_models(MODEL_DIR)
    face_model = gd.load_face_model()
    for folder in folders:
        process(folder, gd, models, face_model)


if __name__ == "__main__":
    main()
