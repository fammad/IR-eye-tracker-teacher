"""Calibration and accuracy maths for recorded blocks. Same geometry as GazeMap's demo, vectorised.

Two kinds of personal correction, both fitted on a calibration block:
  offset  per eye, a fixed pitch/yaw correction (median error per dot, then mean over dots,
          so one blink or glance away cannot pull it). Survives head movement.
  affine  offset first, then a screen-space scale/shift/tilt of the combined gaze
          (GazeMap's fit_mapping). Better in place, known to break when the person moves.

Accuracy on a test block, per dot: the median screen point over the dot's collect frames,
then the angle between "eye -> dot" and "eye -> estimated point". The block's error is the
mean over dots. Same as GazeMap's --test.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gm import demo  # noqa: E402

gd = demo()


def angles_to_vectors(a):
    p, y = a[..., 0], a[..., 1]
    return np.stack([-np.cos(p) * np.sin(y), -np.sin(p), -np.cos(p) * np.cos(y)], axis=-1)


def vectors_to_angles(v):
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return np.stack([np.arcsin(np.clip(-v[..., 1], -1, 1)), np.arctan2(-v[..., 0], -v[..., 2])], axis=-1)


def angle_deg(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


class Block:
    """One processed block: per-frame gaze data plus its screen geometry."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / "meta.json").read_text())
        if not (self.folder / "gaze.npz").exists():
            raise SystemExit(f"{self.folder.name} is not processed yet. Run process_block.py on it.")
        g = np.load(self.folder / "gaze.npz")
        self.g = {k: g[k] for k in g.files}
        w, h = self.meta["window_px"]
        self.size = (w, h)
        self.screen = gd.Screen(w, h, self.meta["screen_width_mm"], self.meta["camera_offset_mm"])
        self.use = (self.g["dot"] >= 0) & (self.g["face"] == 1)

    @property
    def dots(self):
        return sorted(set(self.g["dot"][self.use].tolist()))

    def target_mm(self, fx, fy):
        """Dot position in camera coordinates (mm) for arrays of window fractions."""
        return np.stack([self.screen.to_camera(x * self.size[0], y * self.size[1]) for x, y in zip(fx, fy)])

    def gaze(self, offsets=None):
        """Per frame: eye midpoint (mm) and combined gaze direction in camera space, with optional offsets."""
        raw = self.g["raw"].copy()
        if offsets is not None:
            raw = raw + offsets[None]
        per_eye = angles_to_vectors(raw)                                   # (n, 2, 3) normalized space
        cam = np.einsum("neji,nej->nei", self.g["R"], per_eye)             # R^T v
        cam = cam / np.linalg.norm(cam, axis=-1, keepdims=True)
        combined = cam.sum(axis=1)
        return self.g["center"].mean(axis=1), combined / np.linalg.norm(combined, axis=-1, keepdims=True)

    def true_angles(self):
        """Per frame and eye: the pitch/yaw that would point exactly at the dot (normalized space)."""
        target = self.target_mm(self.g["fx"][self.use], self.g["fy"][self.use])
        out = np.full(self.g["raw"].shape, np.nan)
        d = target[:, None, :] - self.g["center"][self.use]
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        out[self.use] = vectors_to_angles(np.einsum("neij,nej->nei", self.g["R"][self.use], d))
        return out

    def screen_points(self, offsets=None):
        """Per frame: where the combined gaze hits the screen, as window fractions (nan if it misses)."""
        origin, gaze = self.gaze(offsets)
        out = np.full((len(origin), 2), np.nan)
        ok = gaze[:, 2] < -1e-6
        t = -origin[ok, 2] / gaze[ok, 2]
        hit = origin[ok] + t[:, None] * gaze[ok]
        px = np.stack([self.screen.to_pixels(p) for p in hit]) if len(hit) else np.zeros((0, 2))
        out[ok] = px / np.array(self.size)
        return out, origin


def fit_offsets(block):
    """Per-eye pitch/yaw offset: median error at each dot, then the mean over dots. Shape (2, 2)."""
    err = block.true_angles() - block.g["raw"]
    per_dot = [np.nanmedian(err[block.use & (block.g["dot"] == d)], axis=0) for d in block.dots]
    if len(per_dot) < 5:
        raise ValueError(f"Only {len(per_dot)} usable dots in {block.folder.name}; need at least 5.")
    return np.mean(per_dot, axis=0)


def dot_estimates(block, offsets=None):
    """Per dot: (dot, target fractions, median estimated fractions, median eye midpoint)."""
    points, origin = block.screen_points(offsets)
    out = []
    for d in block.dots:
        m = block.use & (block.g["dot"] == d) & ~np.isnan(points[:, 0])
        if m.sum() < 3:
            continue
        target = np.array([block.g["fx"][m][0], block.g["fy"][m][0]])
        out.append((d, target, np.median(points[m], axis=0), np.median(origin[m], axis=0)))
    return out


def fit_calibration(block):
    """Everything a calibration block gives: offsets, and an affine screen mapping on top of them."""
    offsets = fit_offsets(block)
    est = dot_estimates(block, offsets)
    estimates = np.array([e[2] for e in est])
    targets = np.array([e[1] for e in est])
    mapping = gd.fit_mapping(estimates, targets, "affine") if len(est) >= 6 else None
    return {"offsets": offsets, "mapping": mapping}


def dot_errors(block, offsets=None, mapping=None):
    """Angular error (deg) per test dot, plus the error in mm on screen."""
    rows = []
    for d, target, est, origin in dot_estimates(block, offsets):
        if mapping is not None:
            est = gd.apply_mapping(mapping, est)
        t_mm = block.target_mm([target[0]], [target[1]])[0]
        e_mm = block.target_mm([est[0]], [est[1]])[0]
        rows.append({"dot": d, "deg": float(angle_deg(t_mm - origin, e_mm - origin)),
                     "mm": float(np.linalg.norm(t_mm - e_mm)), "dist_mm": float(np.linalg.norm(origin))})
    return rows


def bootstrap_mean_diff(a, b, n=5000, seed=0):
    """Paired difference a - b over dots: mean and 95% interval by resampling dots."""
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    diffs = (a[idx] - b[idx]).mean(axis=1)
    return float((a - b).mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def head_summary(block):
    """Median eye distance (mm) and median head rotation of the frames used."""
    mid = block.g["center"][block.use].mean(axis=1)
    dist = float(np.median(np.linalg.norm(mid, axis=1)))
    R = np.nanmedian(block.g["head_R"][block.use], axis=0)
    u, _, vt = np.linalg.svd(R)
    return dist, u @ vt


def rotation_between_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))
