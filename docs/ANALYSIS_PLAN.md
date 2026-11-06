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
  about 8 minutes of natural tasks with small head movements and one light change.
- Short break, walk to room B.
- Room B (different room and light, same laptop): 16-dot test at two seat
  positions (normal, sat back), then a fresh 9-point calibration.

No room B data is used for training, filtering or choosing any setting.

## Conditions (all built offline from room A data, scored on the same room B frames)

| Id | Condition | Uses |
|---|---|---|
| C0 | Generic | nothing personal |
| C1 | 9-point offset (baseline) | room A 9-point |
| C2 | IR still | IR labels from the room A 16-dot block |
| C3 | IR natural (main) | IR labels from the natural tasks |
| C4 | Hybrid | C3, constant re-fitted on the 9-point frames |
| C5 | Fresh 9-point (ceiling) | room B 9-point, reference only |

Personalisation method for C2 to C4: last-layer ridge with the strength frozen
from the simulation (`results/sim/frozen.json`) and checked on the pilot.

## Primary outcome

Mean angular error on the room B 16-dot test (both seat positions), C3 vs C1.
Per participant: mean over dots. Test: paired Wilcoxon signed-rank over
participants, two-sided, alpha 0.05. Also reported: mean difference with a
bootstrap 95% interval over participants, and how many participants improved.

## Secondary (reported, not used to claim the main result)

- Gap closed: (C1 - C3) / (C1 - C5), per participant, median
- C2 vs C3: does natural use add anything over still dots (diversity)
- C4 vs C3 and C1 (hybrid)
- Wearing time: C3 rebuilt from the first 1, 2, 4, 8 minutes
- Error split into horizontal and vertical, and upper screen half separately
- IR teacher accuracy (16-dot test in room A) against the C3 gain

## Exclusions, decided now

- Participant excluded from the IR conditions if IR 16-dot error in room A is above 2.5 degrees,
  or IR calibration fails twice. Reported with the count.
- Frames excluded when: no face found, eye openness below threshold, IR pupil
  or glint not found, IR confidence below threshold. Thresholds set on the pilot.
- A test dot with fewer than 5 usable frames is dropped from all conditions for that person.

## Sample

Target 10 to 12, minimum 8 completed. No interim look at the primary outcome.

## Changes after freezing

| Date | Change | Reason |
|---|---|---|
