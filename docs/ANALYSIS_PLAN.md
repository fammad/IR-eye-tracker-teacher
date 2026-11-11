# Analysis plan (draft, fix and date before the first participant)

Written before any participant data exists. Changes after that date are listed
at the bottom with a reason, and reported in the paper.

Date frozen: ____________

## Question

Does a short session with a removable IR eye tracker personalise a webcam gaze
model better than a 9-point calibration, when tested in a different room?

## Design

One visit per participant, within-subject.

- Room A (blinds closed, controlled light): distance measurement, 9-point webcam
  calibration without the frame, IR frame on, IR calibration, IR 16-dot test,
  about 8 minutes of natural tasks with small head movements and one light change,
  3-dot drift checks, frame off.
- Short break, walk to room B.
- Room B (different room and light, same laptop): 16-dot test at two seat
  positions (normal, sat back), then a fresh 9-point calibration.

No room B data is used for training, filtering or choosing any setting.

## Personalisation method (fixed now)

The webcam model's last layer (502 inputs to pitch and yaw, per eye) is refitted
on the IR-labelled frames by ridge regression towards the original weights, with
strength lambda = 0.1. Lambda was chosen on MPIIGaze in the offline simulation
(`sim/simulate_teacher.py`, `results/sim/frozen.json`, leave-one-person-out) and
is not tuned on any participant.

Hybrid: after the ridge refit, the constant term of each eye is re-fitted so the
mean error on the room A 9-point calibration frames is zero. The IR labels set
the shape of the correction, the 9 points set its position. This protects
against a constant IR error, such as the frame slipping on the nose.

## Conditions (all built offline from room A data, scored on the same room B frames)

| Id | Condition | Uses |
|---|---|---|
| C0 | Generic | nothing personal |
| C1 | 9-point offset (baseline) | room A 9-point |
| C2 | IR still, hybrid | IR labels from the room A 16-dot block + room A 9-point |
| C3 | IR natural, ridge only | IR labels from the natural tasks |
| C4 | IR natural, hybrid (main) | C3 + room A 9-point constant |
| C5 | Fresh 9-point (ceiling) | room B 9-point, reference only |

## Primary outcome

Mean angular error on the room B 16-dot test (both seat positions), C4 vs C1.
Per participant: mean over dots. Test: paired Wilcoxon signed-rank over
participants, two-sided, alpha 0.05. Also reported: mean difference with a
bootstrap 95% interval over participants, and how many participants improved.

Expected size, from the simulation: hybrid with 300 teacher frames beat 9-point
by 0.40 degrees on MPIIGaze (14 of 15 people). The simulation's teacher was more
varied than one session, so the study may find less.

## Secondary (reported, not used to claim the main result)

- Gap closed: (C1 - C4) / (C1 - C5), per participant, median
- C3 vs C4: ridge only vs hybrid
- C2 vs C4: does natural use with a light change add anything over still dots
- Wearing time: C4 rebuilt from the first 1, 2, 4, 8 minutes of natural use
- Error split into horizontal and vertical, and upper screen half separately
- IR teacher accuracy (room A 16-dot test) and drift against the C4 gain

## IR quality checks, decided now

Measured in room A for every participant, before any room B result is looked at:
- IR 16-dot error, overall and corners vs centre (edge stretch)
- drift: change of the IR error between the first and last 3-dot check

## Exclusions, decided now

- Participant excluded from the IR conditions if the IR 16-dot error in room A
  is above 2.5 degrees, IR drift is above 2 degrees, or IR calibration fails twice.
  Reported with the count.
- Participant excluded from the IR conditions if fewer than 300 IR-labelled
  frames survive cleaning.
- Frames excluded when: no face found, eye openness below threshold, IR pupil
  or glint not found, IR confidence below threshold. Thresholds set on the pilot.
- A test dot with fewer than 5 usable frames is dropped from all conditions for that person.

## Sample

Target 10 to 12, minimum 8 completed. No interim look at the primary outcome.

## Changes after freezing

| Date | Change | Reason |
|---|---|---|
