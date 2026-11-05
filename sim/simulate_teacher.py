"""Offline teacher simulation on MPIIGaze. No hardware, no participants.

Question: if a temporary teacher (the IR tracker) gives the webcam model many
labelled frames of one person, does that beat a normal 9-point calibration on
OTHER days? And how accurate must the teacher be for that?

MPIIGaze stands in for the study:
  one recording day of a person  = room A (the session with the teacher)
  all other days of that person  = room B (new place, light, head position)
  the true MPIIGaze labels       = a perfect teacher; noise is added to make it imperfect
Every person is predicted by the fold model that never saw them (LOPO), so the
starting point is the same "generic" model as in the study.

Three experiments, each repeated with different source days:
  E1 amount     : 9 grid frames vs 30 / 100 / 300 teacher frames from the same day
  E2 diversity  : the same number of frames from one day, from four days, or
                  from one day but spread over head poses
  E3 quality    : teacher frames with random error (mean 0 to 5 degrees), a
                  constant bias (0, 1, 2 degrees) or an error that grows towards
                  the screen edges (5, 10, 20 % stretch)

Personalisation methods (all per eye, nothing else in the network changes):
  offset  : add the mean error on the training frames to every prediction
  ridge   : refit the last layer (502 features -> pitch, yaw), pulled towards the
            generic weights by a penalty lambda. A huge lambda gives the offset.
  hybrid  : ridge on teacher frames, then the constant part re-fitted on the 9
            exact grid frames (teacher learns the shape, 9 points fix the bias)

Lambda is chosen honestly: for each person, the lambda that worked best on the
OTHER 14 people. The single best lambda over everyone is saved in frozen.json;
the real study uses that value and never tunes on participants.

Error: mean angular error per eye in degrees, person-equal average, as in GazeMap.
Caveat: one MPIIGaze day spans several short sessions spread over a day, so it
is more varied than one 8-minute study session. Treat the gains as an upper bound.

Usage (from anywhere, GazeMap in ../00_GazeMap or GAZEMAP_DIR set):
  python sim/simulate_teacher.py                  # all people, 5 repeats
  python sim/simulate_teacher.py --people p09 p10 --repeats 2   # quick check
Outputs go to results/sim/ in the study folder.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gm import MODEL_DIR, STUDY_DIR, gaze_model  # noqa: E402

OUT_DIR = STUDY_DIR / "results" / "sim"
SEED = 7
MIN_DAY_FRAMES = 30          # a day needs this many frames to be used as a source
AMOUNTS = [9, 30, 100, 300]  # E1
DIVERSITY_K = 100            # E2, frames per condition
QUALITY_K = 300              # E3
NOISES_DEG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
BIASES_DEG = [0.0, 1.0, 2.0]
SCALES = [0.05, 0.1, 0.2]       # E3, error growing towards the edges
LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e6]  # 1e6 is effectively the offset
BIAS_PENALTY = 1e-6          # the constant term is almost free to move
MEAN_TO_SIGMA = np.sqrt(np.pi / 2)  # 2D Gaussian: mean angular error = sigma * sqrt(pi/2)


# ---------- angles ----------

def to_vectors(angles):
    pitch, yaw = angles[..., 0], angles[..., 1]
    return np.stack([-np.cos(pitch) * np.sin(yaw), -np.sin(pitch), -np.cos(pitch) * np.cos(yaw)], axis=-1)


def angular_error(pred, true):
    dot = (to_vectors(pred) * to_vectors(true)).sum(axis=-1)
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


# ---------- network pieces ----------

def load_fold(person, config):
    model = gaze_model.build_model(config)
    model.load_state_dict(torch.load(MODEL_DIR / f"lopo_{person}.pt", map_location="cpu"))
    return model.eval()


def last_layer_features(model, images, head, batch=4096):
    """Everything up to the last layer: 500 hidden units + 2 head angles = 502 numbers per eye image."""
    out = []
    with torch.no_grad():
        for s in range(0, len(images), batch):
            x = torch.from_numpy(images[s:s + batch]).float().div(255).unsqueeze(1)
            x = model.pool(torch.relu(model.conv1(x)))
            x = model.pool(torch.relu(model.conv2(x)))
            x = torch.relu(model.fc1(x.flatten(start_dim=1)))
            x = torch.cat([x, torch.from_numpy(head[s:s + batch])], dim=1)
            out.append(x.numpy().astype(np.float64))
    return np.concatenate(out)


def extract(person, config, eyes):
    """Fold model that never saw this person -> its last layer as a (503, 2) matrix (502 weights, then
    the bias) and the 502 features of every eye image. eyes: [(images, head) for right, left]."""
    model = load_fold(person, config)
    theta0 = np.vstack([model.fc2.weight.detach().numpy().T,
                        model.fc2.bias.detach().numpy()[None]]).astype(np.float64)
    F = [last_layer_features(model, images, head) for images, head in eyes]

    # Self-check: features x original last layer must reproduce the network's own output.
    images, head = eyes[0]
    with torch.no_grad():
        direct = model(torch.from_numpy(images[:64]).float().div(255).unsqueeze(1),
                       torch.from_numpy(head[:64])).numpy()
    if np.abs(direct - with_ones(F[0][:64]) @ theta0).max() > 1e-4:
        raise RuntimeError("Feature split does not reproduce the network output. Stop.")
    return theta0, F


# ---------- personalisation ----------

def with_ones(F):
    return np.column_stack([F, np.ones(len(F))])


def fit_ridge(F, Y, theta0, lam):
    """Last layer (503 x 2 incl. bias) that fits Y, pulled towards theta0 with strength lam."""
    A = with_ones(F)
    n = len(A)
    penalty = np.full(A.shape[1], lam)
    penalty[-1] = BIAS_PENALTY
    lhs = A.T @ A / n + np.diag(penalty)
    rhs = A.T @ (Y - A @ theta0) / n
    return theta0 + np.linalg.solve(lhs, rhs)


def fit_offset(F, Y, theta0):
    theta = theta0.copy()
    theta[-1] += (Y - with_ones(F) @ theta0).mean(axis=0)
    return theta


def refit_bias(theta, F, Y):
    """Keep the shape of theta, move only its constant so the mean error on (F, Y) is zero."""
    return fit_offset(F, Y, theta)


# ---------- frame selection ----------

def grid_nine(gaze, rng):
    """9 frames spread over the gaze range, one per cell of a 3x3 pitch/yaw grid (same as GazeMap's evaluate.py)."""
    p_edges = np.quantile(gaze[:, 0], [1 / 3, 2 / 3])
    y_edges = np.quantile(gaze[:, 1], [1 / 3, 2 / 3])
    cell = np.digitize(gaze[:, 0], p_edges) * 3 + np.digitize(gaze[:, 1], y_edges)
    chosen = [rng.choice(np.where(cell == c)[0]) for c in range(9) if (cell == c).any()]
    rest = np.setdiff1d(np.arange(len(gaze)), chosen)
    while len(chosen) < 9 and len(rest):
        pick = rng.choice(rest)
        chosen.append(pick)
        rest = rest[rest != pick]
    return np.array(chosen)


def spread_over_heads(head, k, rng):
    """k frames that cover the head poses of a day as widely as possible (farthest-point sampling)."""
    k = min(k, len(head))
    chosen = [int(rng.integers(len(head)))]
    dist = np.linalg.norm(head - head[chosen[0]], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(head - head[nxt], axis=1))
    return np.array(chosen)


# ---------- one person ----------

class Person:
    """Both eyes of every frame of one person, paired, with last-layer features from the fold model that never saw them."""

    def __init__(self, person, config):
        data = gaze_model.load_all([person])
        r = np.where(data["eye"] == 0)[0]
        l = np.where(data["eye"] == 1)[0]
        r = r[np.lexsort((data["sample_index"][r], data["file_path"][r]))]
        l = l[np.lexsort((data["sample_index"][l], data["file_path"][l]))]
        assert np.array_equal(data["file_path"][r], data["file_path"][l])
        assert np.array_equal(data["sample_index"][r], data["sample_index"][l])

        self.theta0, self.F = extract(person, config, [(data["images"][idx], data["head"][idx]) for idx in (r, l)])
        self.Y = [data["gaze"][idx].astype(np.float64) for idx in (r, l)]
        self.head = data["head"][r].astype(np.float64)
        self.day = np.array([Path(p).stem for p in data["file_path"][r]])
        self.name = person

    def days_with(self, n):
        names, counts = np.unique(self.day, return_counts=True)
        return [d for d, c in zip(names, counts) if c >= n]

    def predict(self, thetas, frames):
        return [with_ones(self.F[e][frames]) @ thetas[e] for e in (0, 1)]

    def error(self, thetas, frames):
        """Mean angular error over both eyes of the given frames."""
        pred = self.predict(thetas, frames)
        return float(np.mean([angular_error(pred[e], self.Y[e][frames]).mean() for e in (0, 1)]))

    def teacher_labels(self, frames, noise_deg, bias_deg, rng, scale=0.0):
        """True labels plus teacher error. One gaze point per frame, so both eyes share the same error
        (yaw sign flipped for the mirrored left eye).
          noise_deg : random error per frame, given as its mean angular size
          bias_deg  : one constant error for the whole session (e.g. the frame slipped)
          scale     : error that grows towards the screen edges (e.g. a poor IR calibration):
                      0.1 stretches every gaze angle by 10% around the session's average gaze"""
        n = len(frames)
        sigma = np.radians(noise_deg) / MEAN_TO_SIGMA
        shift = rng.normal(0, sigma, (n, 2)) if noise_deg > 0 else np.zeros((n, 2))
        if bias_deg > 0:
            angle = rng.uniform(0, 2 * np.pi)
            shift = shift + np.radians(bias_deg) * np.array([np.cos(angle), np.sin(angle)])
        if scale:
            true = self.Y[0][frames]
            shift = shift + scale * (true - true.mean(axis=0))
        mirror = np.array([1.0, -1.0])
        return [self.Y[0][frames] + shift, self.Y[1][frames] + shift * mirror]

    def fit(self, method, frames, labels, lam=None, exact9=None):
        if method == "offset":
            return [fit_offset(self.F[e][frames], labels[e], self.theta0) for e in (0, 1)]
        thetas = [fit_ridge(self.F[e][frames], labels[e], self.theta0, lam) for e in (0, 1)]
        if method == "hybrid":
            thetas = [refit_bias(thetas[e], self.F[e][exact9], self.Y[e][exact9]) for e in (0, 1)]
        return thetas


# ---------- experiments ----------

def run_person(p, repeats, rng):
    rows = []
    generic = [p.theta0, p.theta0]

    def add(exp, repeat, condition, method, frames, test, labels, lam=None, exact9=None, **extra):
        thetas = generic if method == "generic" else p.fit(method, frames, labels, lam, exact9)
        rows.append({"experiment": exp, "person": p.name, "repeat": repeat, "condition": condition,
                     "method": method, "lam": "" if lam is None else lam,
                     "n_train": 0 if frames is None else len(frames), "n_test": int(len(test)),
                     "mae": p.error(thetas, test), **extra})

    def add_all_methods(exp, repeat, condition, frames, test, labels, exact9=None, **extra):
        add(exp, repeat, condition, "offset", frames, test, labels, **extra)
        for lam in LAMBDAS:
            add(exp, repeat, condition, "ridge", frames, test, labels, lam=lam, **extra)
            if exact9 is not None:
                add(exp, repeat, condition, "hybrid", frames, test, labels, lam=lam, exact9=exact9, **extra)

    sources = p.days_with(MIN_DAY_FRAMES)
    big = p.days_with(max(AMOUNTS)) or p.days_with(DIVERSITY_K) or sources
    for repeat in range(repeats):
        if len(sources) < 2:
            break
        day = big[rng.integers(len(big))]
        in_day = np.where(p.day == day)[0]
        test = np.where(p.day != day)[0]
        nine = in_day[grid_nine(p.Y[0][in_day], rng)]
        exact = lambda f: [p.Y[0][f], p.Y[1][f]]  # noqa: E731

        # E1 amount
        add("E1", repeat, "generic", "generic", None, test, None)
        add("E1", repeat, "9-point", "offset", nine, test, exact(nine))
        for k in AMOUNTS[1:]:
            frames = rng.choice(in_day, min(k, len(in_day)), replace=False)
            add_all_methods("E1", repeat, f"teacher {k}", frames, test, exact(frames), k=k)

        # E3 quality (same source day and test days as E1)
        frames = rng.choice(in_day, min(QUALITY_K, len(in_day)), replace=False)
        add("E3", repeat, "9-point", "offset", nine, test, exact(nine))
        for bias in BIASES_DEG:
            for noise in NOISES_DEG:
                labels = p.teacher_labels(frames, noise, bias, rng)
                add_all_methods("E3", repeat, f"noise {noise} bias {bias}", frames, test, labels,
                                exact9=nine, noise=noise, bias=bias)
        for scale in SCALES:
            labels = p.teacher_labels(frames, 1.0, 0.0, rng, scale=scale)
            add_all_methods("E3", repeat, f"scale {scale:.0%} noise 1.0", frames, test, labels,
                            exact9=nine, noise=1.0, scale=scale)

        # E2 diversity: four source days, test on the rest
        four_pool = p.days_with(DIVERSITY_K // 4)
        wide = [d for d in p.days_with(DIVERSITY_K) if d in four_pool]
        if len(four_pool) >= 5 and wide:
            first = wide[rng.integers(len(wide))]
            others = rng.choice([d for d in four_pool if d != first], 3, replace=False)
            used = [first, *others]
            test2 = np.where(~np.isin(p.day, used))[0]
            first_frames = np.where(p.day == first)[0]
            narrow = rng.choice(first_frames, DIVERSITY_K, replace=False)
            heads = first_frames[spread_over_heads(p.head[first_frames], DIVERSITY_K, rng)]
            spread = np.concatenate([rng.choice(np.where(p.day == d)[0], DIVERSITY_K // 4, replace=False)
                                     for d in used])
            nine2 = first_frames[grid_nine(p.Y[0][first_frames], rng)]
            add("E2", repeat, "9-point", "offset", nine2, test2, exact(nine2))
            for name, frames in (("one day", narrow), ("one day, spread heads", heads), ("four days", spread)):
                add_all_methods("E2", repeat, name, frames, test2, exact(frames))
    return rows


# ---------- summary ----------

def choose_lambda(rows):
    """Person-level error for every (experiment, condition, method), with ridge/hybrid lambda chosen on the other people.
    Returns {key: {person: mae}} and {key: best lambda over everyone}."""
    table = {}
    for r in rows:
        key = (r["experiment"], r["condition"], r["method"])
        table.setdefault(key, {}).setdefault(r["lam"], {}).setdefault(r["person"], []).append(r["mae"])

    per_person, frozen = {}, {}
    for key, by_lam in table.items():
        means = {lam: {p: np.mean(v) for p, v in d.items()} for lam, d in by_lam.items()}
        people = sorted(set.intersection(*[set(m) for m in means.values()]))
        if len(by_lam) == 1:
            per_person[key] = {p: means[next(iter(means))][p] for p in people}
            continue
        frozen[key] = min(means, key=lambda lam: np.mean([means[lam][p] for p in people]))
        per_person[key] = {}
        for p in people:
            others = [q for q in people if q != p] or people
            lam = min(means, key=lambda lam: np.mean([means[lam][q] for q in others]))
            per_person[key][p] = means[lam][p]
    return per_person, frozen


def summarise(rows):
    per_person, frozen = choose_lambda(rows)
    lines, table_rows = [], []
    for exp, title in (("E1", "E1 amount: teacher frames from one day, tested on other days"),
                       ("E2", "E2 diversity: 100 frames, from one day or from four days"),
                       ("E3", "E3 teacher quality: 300 frames with random error (mean deg), bias (deg) or edge stretch")):
        keys = [k for k in per_person if k[0] == exp]
        if not keys:
            continue
        base_key = (exp, "9-point", "offset")
        base = per_person.get(base_key, {})
        lines.append(f"\n{title}")
        lines.append(f"  {'condition':28s} {'method':8s} {'MAE':>6s} {'SE':>5s} {'vs 9-point':>10s} {'better for':>11s}")
        for key in keys:
            vals = per_person[key]
            people = sorted(vals)
            m = np.array([vals[p] for p in people])
            common = [p for p in people if p in base]
            diff = np.array([vals[p] - base[p] for p in common]) if common and key != base_key else None
            better = f"{(diff < 0).sum()}/{len(diff)}" if diff is not None else ""
            delta = f"{diff.mean():+.2f}" if diff is not None else ""
            lines.append(f"  {key[1]:28s} {key[2]:8s} {m.mean():6.2f} {m.std() / np.sqrt(len(m)):5.2f} "
                         f"{delta:>10s} {better:>11s}")
            table_rows.append({"experiment": exp, "condition": key[1], "method": key[2],
                               "mae": m.mean(), "se": m.std() / np.sqrt(len(m)), "people": len(m),
                               "delta_vs_9point": diff.mean() if diff is not None else "",
                               "delta_se": diff.std() / np.sqrt(len(diff)) if diff is not None and len(diff) else "",
                               "better_than_9point": better, "lambda_all": frozen.get(key, "")})

    # Break-even teacher error: largest noise where ridge on the teacher still beats 9-point on average.
    base = per_person.get(("E3", "9-point", "offset"), {})
    for method in ("ridge", "hybrid"):
        for bias in BIASES_DEG:
            ok = []
            for noise in NOISES_DEG:
                vals = per_person.get(("E3", f"noise {noise} bias {bias}", method), {})
                common = [p for p in vals if p in base]
                if common and np.mean([vals[p] - base[p] for p in common]) < 0:
                    ok.append(noise)
            if not ok:
                verdict = "never, even with a perfect teacher"
            elif max(ok) == max(NOISES_DEG):
                verdict = f"at least up to {max(ok)} deg (the largest tested)"
            else:
                verdict = f"up to {max(ok)} deg"
            lines.append(f"\nTeacher beats 9-point ({method}, bias {bias} deg): {verdict}")
    return "\n".join(lines), table_rows, frozen


def plot(table_rows, path):
    """Paired difference to the 9-point baseline (below zero = better), +- standard error over people."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def get(exp, cond, meth):
        return next((r for r in table_rows if r["experiment"] == exp and r["condition"] == cond
                     and r["method"] == meth and r["delta_se"] != ""), None)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for method in ("offset", "ridge"):
        pts = [(k, get("E1", f"teacher {k}", method)) for k in AMOUNTS[1:]]
        pts = [(k, r) for k, r in pts if r]
        if pts:
            axes[0].errorbar([k for k, _ in pts], [r["delta_vs_9point"] for _, r in pts],
                             [r["delta_se"] for _, r in pts], marker="o", capsize=3, label=method)
    axes[0].set_xscale("log")
    axes[0].set_xticks(AMOUNTS[1:], [str(k) for k in AMOUNTS[1:]])
    axes[0].minorticks_off()
    axes[0].set_xlabel("teacher frames from one day")
    axes[0].set_ylabel("error vs 9-point on other days (deg)")
    axes[0].set_title("E1: more labels from the same session")

    names = ["one day", "one day, spread heads", "four days"]
    for j, method in enumerate(("offset", "ridge")):
        rs = [get("E2", n, method) for n in names]
        xs = [i + (j - 0.5) * 0.3 for i, r in enumerate(rs) if r]
        rs = [r for r in rs if r]
        axes[1].bar(xs, [r["delta_vs_9point"] for r in rs], 0.3, yerr=[r["delta_se"] for r in rs],
                    capsize=3, label=method)
    axes[1].set_xticks(range(len(names)), names, fontsize=8)
    axes[1].set_title("E2: same number of labels, more variety")

    for method in ("ridge", "hybrid"):
        for bias in BIASES_DEG:
            pts = [(n, get("E3", f"noise {n} bias {bias}", method)) for n in NOISES_DEG]
            pts = [(n, r) for n, r in pts if r]
            if pts:
                axes[2].errorbar([n for n, _ in pts], [r["delta_vs_9point"] for _, r in pts],
                                 [r["delta_se"] for _, r in pts], marker="o", capsize=3,
                                 ls="-" if method == "ridge" else "--", label=f"{method}, bias {bias}°")
    axes[2].set_xlabel("teacher random error (mean deg)")
    axes[2].set_title("E3: how good must the teacher be?")
    for ax in axes:
        ax.axhline(0, color="grey", lw=1)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--people", nargs="*", default=None)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    with open(MODEL_DIR.parent / "config.json") as f:
        config = json.load(f)
    people = args.people or gaze_model.list_people()
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for person in people:
        start = time.time()
        p = Person(person, config)
        person_rows = run_person(p, args.repeats, rng)
        rows += person_rows
        gen = [r["mae"] for r in person_rows if r["condition"] == "generic"]
        nine = [r["mae"] for r in person_rows if r["experiment"] == "E1" and r["condition"] == "9-point"]
        print(f"{person}: {len(p.day)} frames, {len(set(p.day))} days | generic {np.mean(gen):.2f}° "
              f"| 9-point {np.mean(nine):.2f}° | {time.time() - start:.0f} s", flush=True)

    with open(OUT_DIR / "sim_rows.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)

    text, table_rows, frozen = summarise(rows)
    header = (f"Teacher simulation on MPIIGaze | people: {len(set(r['person'] for r in rows))} "
              f"| repeats: {args.repeats} | seed {SEED}\n"
              "MAE = mean angular error per eye on held-out days, person-equal mean. "
              "'vs 9-point' < 0 means better than 9-point.\n"
              "Ridge/hybrid lambda chosen on the other people (leave-one-person-out).")
    print("\n" + header + text)
    (OUT_DIR / "summary.txt").write_text(header + text + "\n")
    with open(OUT_DIR / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)
    (OUT_DIR / "frozen.json").write_text(json.dumps(
        {f"{k[0]} | {k[1]} | {k[2]}": v for k, v in frozen.items()}, indent=1))
    plot(table_rows, OUT_DIR / "sim.png")
    print(f"\nSaved: {OUT_DIR}/summary.txt, summary.csv, sim_rows.csv, frozen.json, sim.png")


if __name__ == "__main__":
    main()
