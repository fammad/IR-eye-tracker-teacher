# IR Eye Tracker Teacher

[![CI](https://github.com/fammad/IR-eye-tracker-teacher/actions/workflows/ci.yml/badge.svg)](https://github.com/fammad/IR-eye-tracker-teacher/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

A webcam eye tracker is cheap but rough, about 4 to 5 degrees off for a new person.
An infrared eye tracker is accurate but needs hardware on your face. This project
uses both, briefly: you wear a low-cost IR tracker for a few minutes, it labels
what the webcam sees, and you take it off. The webcam model, personalised with
those labels, keeps working on its own.

The question: does it stay more accurate in a new room, with new light and a new
seat, than the same webcam model after a normal 9-point calibration?

Course project for EQ2700 at KTH, autumn 2026. Built on
[GazeMap](https://github.com/fammad/GazeMap), whose trained webcam model it uses
unchanged.

This README is also a blueprint: every stage below says what it does, how to run
it, what comes out, and how to change it for your own setup.

## Status

| Stage | State |
|---|---|
| Webcam base model | done (GazeMap v1.0.0) |
| Offline teacher simulation | done: hybrid with 300 IR-style labels beats 9-point by 0.4° on MPIIGaze (14/15 people) |
| Recording, processing, distance calibration | done, first real run pending |
| Gate 1 (is there a gap to close?) | tool done, not run yet |
| IR tracker hardware | being built |
| IR pipeline (pupil, glint, calibration) | planned |
| Session recorder and user study | planned |
| Analysis and figures | planned |

## The pipeline

```
 1 build the IR tracker          hardware, docs/hardware/BUILD.md
 2 webcam base model             GazeMap v1.0.0 weights (or retrain)
 3 per-person setup              measure distance -> focal length
 4 record room A                 webcam + IR video, one clock, dots + natural tasks
 5 IR labels                     pupil + glint -> gaze point per frame        (planned)
 6 personalise                   refit the webcam model's last layer on IR labels
 7 test in room B                webcam only, new room, light and seat
 8 compare and visualise         generic / 9-point / IR / hybrid / fresh calibration
```

### 1. Build the IR tracker

A near-eye camera with 850 nm illumination on a 3D-printed glasses frame, sitting
below the eye so it stays out of the webcam's eye crop. Parts, printing, wiring and
eye-safety limits: [docs/hardware/BUILD.md](docs/hardware/BUILD.md).

### 2. Webcam base model

A small CNN (2 conv layers, 500 hidden units, head pose input) trained on
MPIIGaze with leave-one-person-out: 15 models, each never saw one person.
The 15 weights come from [GazeMap release v1.0.0](https://github.com/fammad/GazeMap/releases/tag/v1.0.0)
and download once on first use into `results/final/models/` (about 100 MB,
not stored in this repository).

Starting point, from GazeMap's own evaluation:

| Condition (MPIIGaze, unseen person) | Error |
|---|---|
| single eye | 6.03° |
| both eyes fused | 4.97° |
| + 9-frame offset calibration, tested on other days | 3.98° |
| live webcam, same session, one user | 1.55° |

Retraining is only needed if you change the network. Then: download MPIIGaze into
`dataset/` and run `python gazemap/train_lopo.py --all` (about 4 hours on an M1).

### 3. Per-person setup

The webcam's focal length and the person's face size are unknown, so the code
can misjudge how far away the eyes are. One tape-measured distance fixes both:

```bash
python study/measure_distance.py --session P01 --distance-cm 60
```

### 4. Record

Everything is recorded raw and processed afterwards, so a bug found later never
costs a participant. Dots: `calib9` (3x3), `test16` (4x4), `check3` (drift).

```bash
python study/record_block.py --session P01 --room A --block calib9 --note "desk lamp, blinds closed"
python study/process_block.py data/P01
```

### 5. IR labels (planned)

Pupil and corneal glint in each IR frame, a 9-point calibration to the screen,
then a gaze point for every webcam frame on the shared clock.

### 6. Personalise

Only the last layer of the webcam model is refitted (502 inputs to pitch and yaw),
pulled towards the original weights. Closed form, takes seconds, cannot diverge.
The pull strength (0.1) is fixed from the simulation, never tuned on participants.
Hybrid, the main method: the same, then the constant part is re-fitted on the 9
calibration dots. The IR labels set the shape, the 9 points set the position, so a
slipping frame cannot shift everything.

### 7. Test in room B

A different room and light, same laptop, webcam only. 16 dots at two seat
positions, then a fresh 9-point calibration as the best case.

### 8. Compare and visualise

| Condition | Uses |
|---|---|
| generic | nothing personal |
| 9-point offset (baseline) | room A 9-point |
| IR still, hybrid | IR labels from the room A dot test + 9-point |
| IR natural, ridge only | IR labels from natural tasks |
| IR natural, hybrid (main) | IR labels from natural tasks + 9-point constant |
| fresh 9-point (ceiling) | room B 9-point |

One primary comparison, fixed in advance: room B error, IR natural hybrid vs 9-point.
Full plan: [docs/ANALYSIS_PLAN.md](docs/ANALYSIS_PLAN.md).

## Try the parts that work now

```bash
git clone https://github.com/fammad/IR-eye-tracker-teacher.git
cd IR-eye-tracker-teacher
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python tests/test_geometry.py             # checks the maths on synthetic data
python gazemap/gaze_demo.py --calibrate   # GazeMap's live demo
python sim/simulate_teacher.py            # needs MPIIGaze in dataset/
```

Run everything from the repository root. Step-by-step study protocol, including
Gate 1: [docs/STUDY.md](docs/STUDY.md).

## Make it yours

- **Other laptop or screen:** pass `--screen-width-mm` and `--camera-offset-mm`
  (camera centre to the top edge of the screen image) to `record_block.py`.
- **Other webcam:** run `measure_distance.py` again; the focal length is per camera.
- **Other IR camera:** pass its index with `--ir-camera`; check the eye image in
  the preview before the first block.
- **Your own base model:** retrain (stage 2) and put the 15 `.pt` files in
  `results/final/models/`.

## Repository layout

| Folder | Contents |
|---|---|
| `gazemap/` | the webcam model from GazeMap: network, live pipeline, training, evaluation |
| `study/` | distance calibration, recording, processing, Gate 1 |
| `sim/` | offline teacher simulation on MPIIGaze |
| `tests/` | checks with known answers |
| `tools/` | helpers, e.g. downloading the stimulus images |
| `docs/` | study protocol, analysis plan, hardware guide, credits |
| `gm.py` | connects the study scripts to `gazemap/` |

Not in git: `dataset/` (MPIIGaze), `results/final/models/` (downloaded),
`data/` (recordings with faces), `stimuli/`.

## Privacy

Recordings contain faces. They stay in `data/`, which is never committed.
Sessions are named by code (P01), never by name.

## Credits, licence, citation

Credits and related work: [docs/CREDITS.md](docs/CREDITS.md).
Code: [Apache 2.0](LICENSE), keep the [NOTICE](NOTICE). The trained weights are
CC BY-NC-SA 4.0 (non-commercial), as the MPIIGaze data they come from.
To cite this project, use GitHub's "Cite this repository" ([CITATION.cff](CITATION.cff)).
