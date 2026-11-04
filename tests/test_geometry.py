"""Checks the study maths on synthetic blocks with known answers. No camera needed.

Builds fake processed blocks where the network's error is a known constant offset
per eye, then checks that:
  1. a calibration fitted on a block recovers that offset
  2. the corrected test error is near zero when the offset is right
  3. Gate 1 sees a gap exactly when room A and room B offsets differ

Usage: python tests/test_geometry.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "study"))

import common  # noqa: E402
from analysis_lib import (Block, angle_deg, dot_errors, fit_calibration, gd,  # noqa: E402
                          vectors_to_angles)

W, H = 1728, 1080


def normalization_rotation(head_R, center):
    forward = center / np.linalg.norm(center)
    down = np.cross(forward, head_R[:, 0])
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    return np.stack([right, down, forward])


def make_block(folder, dots, offsets, rng, eye_mid=(0.0, -40.0, 600.0), noise_deg=0.0, frames_per_dot=40):
    """Fake processed block. offsets: (2, 2) radians the network is WRONG by (true = raw + offsets)."""
    folder.mkdir(parents=True)
    meta = {"window_px": [W, H], "screen_width_mm": 345.0, "camera_offset_mm": 5.0,
            "dots": dots, "status": "complete", "note": "synthetic"}
    (folder / "meta.json").write_text(json.dumps(meta))
    screen = gd.Screen(W, H, 345.0, 5.0)
    head_R = np.diag([1.0, -1.0, -1.0])  # face looking at the camera
    mid = np.array(eye_mid)
    centers = np.stack([mid + [-31, 0, 0], mid + [31, 0, 0]])  # right eye on camera's left
    rows = {k: [] for k in ("t", "face", "dot", "fx", "fy", "raw", "R", "center", "head_R")}
    t = 0.0
    for d, (fx, fy) in enumerate(dots):
        target = screen.to_camera(fx * W, fy * H)
        for _ in range(frames_per_dot):
            R = np.stack([normalization_rotation(head_R, c) for c in centers])
            true = np.stack([vectors_to_angles(R[e] @ ((target - centers[e]) / np.linalg.norm(target - centers[e])))
                             for e in (0, 1)])
            raw = true - offsets + rng.normal(0, np.radians(noise_deg), (2, 2))
            for k, v in (("t", t), ("face", 1), ("dot", d), ("fx", fx), ("fy", fy), ("raw", raw),
                         ("R", R), ("center", centers), ("head_R", head_R)):
                rows[k].append(v)
            t += 1 / 30
    np.savez(folder / "gaze.npz", **{k: np.array(v) for k, v in rows.items()},
             eye_img=np.zeros((len(rows["t"]), 2, 36, 60), np.uint8))


def main():
    rng = np.random.default_rng(0)
    tmp = Path(tempfile.mkdtemp())
    try:
        off_a = np.radians([[1.5, -2.0], [1.0, 2.5]])
        off_b = off_a + np.radians([[1.0, 0.5], [1.0, 0.5]])

        make_block(tmp / "A_calib9_1", common.GRID9, off_a, rng, noise_deg=0.3)
        make_block(tmp / "B_test16_1", common.GRID16, off_b, rng, eye_mid=(30, -40, 700), noise_deg=0.3)
        make_block(tmp / "B_calib9_1", common.GRID9, off_b, rng, eye_mid=(30, -40, 700), noise_deg=0.3)

        a, bt, bc = Block(tmp / "A_calib9_1"), Block(tmp / "B_test16_1"), Block(tmp / "B_calib9_1")

        # 1. offsets recovered
        fit_a = fit_calibration(a)
        err = np.degrees(np.abs(fit_a["offsets"] - off_a)).max()
        assert err < 0.1, f"offset not recovered: off by {err:.3f} deg"
        print(f"1. room A offset recovered within {err:.3f} deg")

        # 2. fresh calibration -> near zero error; generic -> large
        fresh = np.mean([r["deg"] for r in dot_errors(bt, fit_calibration(bc)["offsets"])])
        generic = np.mean([r["deg"] for r in dot_errors(bt)])
        assert fresh < 0.2, f"fresh calibration error too big: {fresh:.3f}"
        assert generic > 2.0, f"generic error suspiciously small: {generic:.3f}"
        print(f"2. room B test: generic {generic:.2f} deg, fresh offset {fresh:.3f} deg")

        # 3. stale calibration leaves about the offset change (~1.1 deg)
        stale = np.mean([r["deg"] for r in dot_errors(bt, fit_a["offsets"])])
        expected = angle_deg(np.array([0, 0, -1.0]),
                             np.array([np.sin(np.radians(0.5)), np.sin(np.radians(1.0)), -1.0]))
        assert abs(stale - expected) < 0.3, f"stale error {stale:.3f}, expected about {expected:.3f}"
        print(f"3. stale offset {stale:.2f} deg (expected about {expected:.2f}): gap detected")

        # 4. affine on top of correct offsets does not hurt in place
        aff = np.mean([r["deg"] for r in dot_errors(bt, *fit_calibration(bc).values())])
        assert aff < 0.3, f"affine after correct offsets: {aff:.3f}"
        print(f"4. fresh affine {aff:.3f} deg")

        # 5. frame_targets only marks collect-phase frames
        events = [{"t": "0.0", "event": "dot_on", "dot": "0", "fx": "0.5", "fy": "0.5"},
                  {"t": "1.0", "event": "settled", "dot": "0", "fx": "", "fy": ""},
                  {"t": "2.5", "event": "dot_off", "dot": "0", "fx": "", "fy": ""}]
        tg = common.frame_targets(np.array([0.5, 1.2, 2.6]), events)
        assert tg[0, 0] == -1 and tg[1, 0] == 0 and tg[2, 0] == -1
        print("5. settle-phase and after-dot frames are excluded")

        # 6. Gate 1 end to end on a synthetic session
        session = common.DATA_DIR / "SELFTEST"
        try:
            for name in ("A_calib9_1", "B_test16_1", "B_calib9_1"):
                shutil.copytree(tmp / name, session / name)
            import gate1
            sys.argv = ["gate1.py", "--session", "SELFTEST"]
            gate1.main()
            text = (session / "gate1.txt").read_text()
            assert text.startswith("gap +1.0") or text.startswith("gap +1.1"), text
            assert "PASS" in text
            print("6. Gate 1 runs end to end and reports the known gap")
        finally:
            shutil.rmtree(session, ignore_errors=True)
        print("\nall checks passed")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
