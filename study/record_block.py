"""Record one block of dots: raw webcam (and IR) video, frame times and screen events.

Nothing is estimated while recording. The script only shows dots and saves
everything raw, with one clock for all streams. Gaze is computed afterwards by
process_block.py. Why:
  - a bug found later (focal length, normalization, pupil detection) can be
    fixed and the same recording processed again, so no participant is wasted
  - every condition in the analysis is built from exactly the same frames
  - the webcam runs at full frame rate, not slowed down by the network

Blocks:
  calib9  3x3 grid, the normal 9-point calibration
  test16  4x4 grid, the accuracy test
  check3  3 dots, a quick drift check during the IR session

Each dot: shown for SETTLE (1.0 s, eyes travel, frames marked and never used)
then COLLECT (1.5 s, frames used). Dot order is shuffled with a logged seed so
people cannot anticipate the next dot.

Usage:
  python study/record_block.py --session PILOT1 --room A --block calib9
  python study/record_block.py --session PILOT1 --room B --block test16
  python study/record_block.py --session P03 --room A --block calib9 --ir-camera 1
Before the first block of a person, run measure_distance.py once.
Keys: SPACE start, ESC abort (an aborted block is kept but marked "aborted").
"""

import argparse
import json
import platform
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BLOCKS, COLLECT_SECONDS, GAZEMAP_DIR, SETTLE_SECONDS, git_commit,  # noqa: E402
                    load_camera, session_dir, write_csv)
from gm import STUDY_DIR, demo  # noqa: E402

CODECS = {"mjpg": ("MJPG", ".avi"), "mp4v": ("mp4v", ".mp4")}


class CameraThread(threading.Thread):
    """Reads one camera as fast as it delivers and writes every frame with its time on the shared clock."""

    def __init__(self, name, index, folder, clock0, codec, size=None):
        super().__init__(daemon=True)
        self.name, self.clock0 = name, clock0
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise SystemExit(f"Cannot open the {name} camera (index {index}). On macOS, allow camera "
                             "access for Terminal in System Settings > Privacy & Security > Camera.")
        if size:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        ok, frame = self.cap.read()
        if not ok:
            raise SystemExit(f"The {name} camera opened but gives no image.")
        self.size = (frame.shape[1], frame.shape[0])
        fourcc, ext = CODECS[codec]
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.path = folder / f"{name}{ext}"
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*fourcc), fps, self.size)
        if codec == "mjpg":
            self.writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)
        self.times, self.latest, self.recording = [], frame, False
        self.stop_flag = threading.Event()

    def run(self):
        while not self.stop_flag.is_set():
            ok, frame = self.cap.read()
            t = time.perf_counter() - self.clock0
            if not ok:
                continue
            self.latest = frame
            if self.recording:
                self.writer.write(frame)
                self.times.append(t)

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=2)
        self.cap.release()
        self.writer.release()


def face_preview(window, webcam, detect, width, height, ir=None):
    """Live check before the block: is the face found, is the IR eye image there. SPACE starts, ESC cancels."""
    t0 = time.perf_counter()
    while True:
        frame = webcam.latest.copy()
        found = detect(frame, int((time.perf_counter() - t0) * 1000)) is not None
        canvas = np.zeros((height, width, 3), np.uint8)
        view = cv2.resize(frame, (width // 3, int(width / 3 * frame.shape[0] / frame.shape[1])))
        canvas[60:60 + view.shape[0], 40:40 + view.shape[1]] = view
        if ir is not None:
            iv = cv2.resize(ir.latest, (width // 4, int(width / 4 * ir.latest.shape[0] / ir.latest.shape[1])))
            canvas[60:60 + iv.shape[0], width // 3 + 80:width // 3 + 80 + iv.shape[1]] = iv
        colour = (0, 220, 0) if found else (0, 0, 255)
        text = "Face found. Sit still, look at the dots. SPACE = start" if found else \
            "No face found: check light and position. ESC = cancel"
        cv2.putText(canvas, text, (40, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            return False
        if key == ord(" ") and found:
            return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, help="participant id, e.g. P03 (never a name)")
    parser.add_argument("--room", required=True, choices=["A", "B"])
    parser.add_argument("--block", required=True, choices=sorted(BLOCKS))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--ir-camera", type=int, default=None, help="index of the IR camera, if worn")
    parser.add_argument("--codec", default="mjpg", choices=sorted(CODECS),
                        help="mjpg keeps eye detail (bigger files), mp4v is smaller")
    parser.add_argument("--screen-width-mm", type=float, default=345.0)
    parser.add_argument("--camera-offset-mm", type=float, default=5.0)
    parser.add_argument("--note", default="", help="free text saved with the block, e.g. 'lamp on, blinds closed'")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    gd = demo()
    folder = session_dir(args.session) / f"{args.room}_{args.block}_{datetime.now():%Y%m%d-%H%M%S}"
    folder.mkdir()
    camera = load_camera(args.session)
    if camera is None:
        print("WARNING: no camera.json for this session. Run measure_distance.py first, "
              "otherwise the eye distance (and the gaze labels) use a rough focal-length guess.")

    seed = args.seed if args.seed is not None else int(time.time()) % 100000
    rng = np.random.default_rng(seed)
    dots = BLOCKS[args.block]
    order = rng.permutation(len(dots)).tolist()

    clock0 = time.perf_counter()
    webcam = CameraThread("webcam", args.camera, folder, clock0, args.codec)
    ir = CameraThread("ir", args.ir_camera, folder, clock0, args.codec) if args.ir_camera is not None else None
    webcam.start()
    if ir:
        ir.start()

    window = "GazeMap study recording"
    width, height = gd.open_fullscreen(window)
    detect = gd.make_landmarker()
    events = []

    def log(event, **kw):
        events.append({"t": round(time.perf_counter() - clock0, 4), "event": event, **kw})

    status = "aborted"
    try:
        if not face_preview(window, webcam, detect, width, height, ir):
            raise KeyboardInterrupt
        webcam.recording = True
        if ir:
            ir.recording = True
        log("start")

        if ir:  # sync cue: blinks are visible in both cameras
            canvas = np.zeros((height, width, 3), np.uint8)
            cv2.putText(canvas, "Blink twice, slowly, now", (width // 2 - 300, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
            cv2.imshow(window, canvas)
            log("cue_blink")
            t_end = time.perf_counter() + 3
            while time.perf_counter() < t_end:
                cv2.waitKey(10)
            log("cue_blink_end")

        for n, i in enumerate(order):
            fx, fy = dots[i]
            center = (int(fx * width), int(fy * height))
            log("dot_on", dot=i, fx=fx, fy=fy, px=center[0], py=center[1])
            start, settled = time.perf_counter(), False
            total = SETTLE_SECONDS + COLLECT_SECONDS
            while (elapsed := time.perf_counter() - start) < total:
                if not settled and elapsed >= SETTLE_SECONDS:
                    log("settled", dot=i)
                    settled = True
                canvas = np.zeros((height, width, 3), np.uint8)
                radius = int(18 - 10 * min(elapsed / total, 1))
                cv2.circle(canvas, center, radius, (255, 255, 255), -1)
                cv2.circle(canvas, center, 3, (0, 0, 255), -1)
                cv2.putText(canvas, f"{n + 1}/{len(order)}", (20, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1)
                cv2.imshow(window, canvas)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    log("dot_off", dot=i)
                    raise KeyboardInterrupt
            log("dot_off", dot=i)
        log("end")
        status = "complete"
    except KeyboardInterrupt:
        log("abort")
    finally:
        webcam.stop()
        if ir:
            ir.stop()
        cv2.destroyWindow(window)

    write_csv(folder / "events.csv", events, ["t", "event", "dot", "fx", "fy", "px", "py"])
    streams = {"webcam": webcam}
    if ir:
        streams["ir"] = ir
    for name, cam in streams.items():
        write_csv(folder / f"{name}_frames.csv", [{"frame": k, "t": round(t, 4)} for k, t in enumerate(cam.times)],
                  ["frame", "t"])

    def fps(cam):
        return round((len(cam.times) - 1) / (cam.times[-1] - cam.times[0]), 1) if len(cam.times) > 1 else 0

    meta = {
        "session": args.session, "room": args.room, "block": args.block, "status": status,
        "recorded": datetime.now().isoformat(timespec="seconds"), "note": args.note,
        "seed": seed, "dot_order": order, "dots": dots,
        "settle_seconds": SETTLE_SECONDS, "collect_seconds": COLLECT_SECONDS,
        "window_px": [width, height], "screen_width_mm": args.screen_width_mm,
        "camera_offset_mm": args.camera_offset_mm,
        "webcam": {"index": args.camera, "size": webcam.size, "frames": len(webcam.times), "fps": fps(webcam),
                   "file": webcam.path.name},
        "ir": {"index": args.ir_camera, "size": ir.size, "frames": len(ir.times), "fps": fps(ir),
               "file": ir.path.name} if ir else None,
        "focal_px": camera["focal_px"] if camera else None,
        "camera_json": camera,
        "software": {"gazemap_commit": git_commit(GAZEMAP_DIR), "study_commit": git_commit(STUDY_DIR),
                     "python": platform.python_version(), "opencv": cv2.__version__,
                     "machine": platform.platform()},
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{status}: {folder}\n  webcam {meta['webcam']['frames']} frames at {meta['webcam']['fps']} fps"
          + (f", IR {meta['ir']['frames']} frames at {meta['ir']['fps']} fps" if ir else ""))
    if status == "complete":
        print(f"Next: python study/process_block.py {folder}")


if __name__ == "__main__":
    main()
