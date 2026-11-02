# IR teacher study: how to run it

Can a cheap IR eye tracker, worn for a few minutes and then removed, personalise
the GazeMap webcam model better than a normal 9-point calibration?

The study scripts live in `study/`, `sim/` and `tests/`. They use the GazeMap code in the
repository root through `gm.py`.

## Order of work

| Step | What | Script | Decides |
|---|---|---|---|
| 0 | Check the maths | `tests/test_geometry.py` | code is trustworthy |
| 1 | Offline simulation on MPIIGaze | `sim/simulate_teacher.py` | how much a teacher can help, how accurate it must be |
| 2 | Gate 1 on yourself | `measure_distance.py`, `record_block.py`, `process_block.py`, `gate1.py` | is there a gap in room B at all |
| 3 | Gate 2: IR accuracy on yourself | IR pipeline (next) | is the teacher good enough |
| 4 | Pilot, then participants | session recorder (next) | |
| 5 | Analysis | analysis script (next) | |

Do not start step 4 before steps 1 to 3 have passed.

## Gate 1, step by step (about 20 minutes, no IR needed)

```bash
cd 02_IR_tracker
source ../00_GazeMap/.gazemap-env/bin/activate   # the GazeMap environment works here too

# room A
python study/measure_distance.py --session PILOT1 --distance-cm 60
python study/record_block.py --session PILOT1 --room A --block calib9 --note "room A, desk lamp, blinds closed"

# walk to room B, sit differently (further back, other light)
python study/record_block.py --session PILOT1 --room B --block test16 --note "room B, ceiling light, 75 cm"
python study/record_block.py --session PILOT1 --room B --block calib9 --note "room B, fresh calibration"

# afterwards, anywhere
python study/process_block.py data/PILOT1
python study/gate1.py --session PILOT1
```

Gate 1 prints the gap between the room A calibration and a fresh room B
calibration on the same test dots. Below 0.5 degrees: room B is too similar,
make the change bigger and repeat.

## Simulation

```bash
python sim/simulate_teacher.py --people p12 p14 --repeats 2   # 1 minute smoke test
python sim/simulate_teacher.py                                # all 15 people, about 10 to 20 minutes
```

Results in `results/sim/`: `summary.txt` (read this), `sim.png`, `frozen.json`
(the ridge strength the study must use, chosen before any participant).

## Rules that keep the study honest

1. Room B data never goes into training, filtering or choosing settings.
2. Every setting (ridge strength, frame filters, recipe) is fixed on the pilot and
   the simulation, then frozen. Participants are not used to tune anything.
3. One primary comparison, written down before the first participant:
   room B error, IR natural use vs 9-point offset. See `docs/ANALYSIS_PLAN.md`.
4. Raw video is kept, so any processing bug can be fixed and rerun.

## Data and privacy

`data/` holds face video. It is git-ignored. Session ids are codes (P01, P02),
never names. The list linking codes to people is kept on paper or separately,
and deleted when the course is graded, as written in the consent form.

## Files

| File | Purpose |
|---|---|
| `gm.py` | connects the study scripts to the GazeMap code in the repo root |
| `study/common.py` | folders, dot layouts, event and frame timing |
| `study/measure_distance.py` | focal length per participant from one tape distance |
| `study/record_block.py` | records a dot block: raw webcam (and IR) video, frame times, screen events |
| `study/process_block.py` | runs GazeMap's pipeline on a recorded block, saves per-frame gaze |
| `study/analysis_lib.py` | offset and affine calibration, per-dot errors, bootstrap |
| `study/gate1.py` | Gate 1 report |
| `sim/simulate_teacher.py` | the offline teacher simulation |
| `tests/test_geometry.py` | checks the maths on synthetic data with known answers |
