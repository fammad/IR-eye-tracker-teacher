"""Measure the focal length for one participant, from one tape-measured distance.

Why: GazeMap guesses the webcam focal length (image width) and assumes everyone's
face has the size of the dataset's average face model. If either is off, the code
thinks you sit at 55 cm when you sit at 65 cm. Every gaze ray then starts from the
wrong point and the dot targets used as labels are wrong. Both errors scale the
estimated distance by one factor, so one known distance fixes both, for this
person and this laptop.

How:
  1. Sit so your eyes are DISTANCE cm from the screen (tape measure from the
     camera to the bridge of the nose). Look at the camera.
  2. SPACE: 3 seconds of face points are recorded.
  3. The focal length is scaled until the estimated eye distance equals the
     measured one (repeated on the same points until it agrees within 0.5%).
  4. Saved to data/<session>/camera.json. All recordings of this session use it.

Run it once per participant, at the start, in room A.

Usage:
  python study/measure_distance.py --session P03 --distance-cm 60
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import session_dir  # noqa: E402
from gm import demo  # noqa: E402

RECORD_SECONDS = 3.0


def eye_distance_mm(points, face_model, camera_matrix, gd):
    """Median distance camera -> midpoint between the eyes, over the recorded frames."""
    d = []
    for p in points:
        R, t = gd.estimate_head_pose(p, face_model, camera_matrix)
        right, left = gd.eye_centers(R, t, face_model)
        d.append(np.linalg.norm((right + left) / 2))
    return float(np.median(d))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True)
    parser.add_argument("--distance-cm", type=float, required=True, help="tape-measured eye distance")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    gd = demo()
    face_model = gd.load_face_model()
    detect = gd.make_landmarker()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Cannot open the camera.")

    window = "Measure distance"
    cv2.namedWindow(window)
    t0 = time.time()
    points, size, recording_until = [], None, None
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        size = frame.shape[1], frame.shape[0]
        found = detect(frame, int((time.time() - t0) * 1000))
        if recording_until and found is not None:
            points.append(found)
        view = frame.copy()
        if found is not None:
            for x, y in found.astype(int):
                cv2.circle(view, (x, y), 4, (0, 255, 0), -1)
        msg = (f"Recording... {len(points)} frames" if recording_until else
               f"Eyes at {args.distance_cm:.0f} cm, look at the camera. SPACE = record, ESC = cancel")
        cv2.putText(view, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if found is not None
                    else (0, 0, 255), 2)
        cv2.imshow(window, view)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return
        if key == ord(" ") and recording_until is None and found is not None:
            recording_until = time.time() + RECORD_SECONDS
        if recording_until and time.time() > recording_until:
            break
    cap.release()
    cv2.destroyAllWindows()

    if len(points) < 20:
        raise SystemExit(f"Only {len(points)} frames with a face. Check the light and try again.")

    true_mm = args.distance_cm * 10
    focal = float(size[0])  # GazeMap's default guess
    first = eye_distance_mm(points, face_model, gd.default_camera_matrix(*size, focal), gd)
    estimate = first
    for _ in range(10):
        focal *= true_mm / estimate
        estimate = eye_distance_mm(points, face_model, gd.default_camera_matrix(*size, focal), gd)
        if abs(estimate - true_mm) / true_mm < 0.005:
            break

    result = {
        "focal_px": round(focal, 1), "image_size": list(size), "measured_cm": args.distance_cm,
        "default_estimate_cm": round(first / 10, 1), "final_estimate_cm": round(estimate / 10, 2),
        "frames": len(points), "measured": datetime.now().isoformat(timespec="seconds"),
    }
    path = session_dir(args.session) / "camera.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"With the default focal length ({size[0]} px) your eyes looked {first / 10:.1f} cm away; "
          f"tape says {args.distance_cm:.0f} cm.")
    print(f"Focal length set to {focal:.0f} px -> estimate {estimate / 10:.1f} cm. Saved {path}")
    if abs(first - true_mm) / true_mm > 0.25:
        print("WARNING: the default was off by more than 25%. Re-measure the tape distance to be sure.")


if __name__ == "__main__":
    main()
