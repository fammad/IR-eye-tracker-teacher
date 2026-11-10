# Building the IR eye tracker

Status: being built. This guide is filled in as the hardware is finished, with
photos, part numbers and print files. Sections marked TODO are not done yet.

## What it is

One near-eye camera that sees in near infrared, one 850 nm LED next to it, both
on a 3D-printed glasses frame. The LED lights the eye evenly and puts a small
bright reflection (the glint) on the cornea. The pupil shows as a dark disc.
The glint-to-pupil vector changes as the eye turns, and a short calibration maps
it to a point on the screen.

## Placement rules

- The camera sits **at least 2 cm below the eye centre**, so it stays out of the
  webcam's eye crop (about 38 x 22 mm around each eye).
- The arm comes in **from the temple**, never across the cheek, so the mouth
  corners stay visible: the webcam head pose uses them.
- Check once that the webcam does not see the LED glowing. If it does, the
  webcam model learns "eye with a light next to it".

## Parts (TODO)

| Part | Example | Notes |
|---|---|---|
| IR camera module | TODO | IR-cut filter removed or none, USB, close focus |
| 850 nm LED | TODO | with a resistor sized for the current limit below |
| Visible-light blocking filter | TODO | optional, helps in bright rooms |
| Frame | 3D print, files in `docs/hardware/print/` | TODO |

## Eye safety

Infrared light is invisible, so the eye does not blink or look away from it.
Keep the irradiance at the eye well under the IEC 62471 exempt-group limit for
the cornea (100 W/m² for long exposure). In practice: one small LED, driven at a
few milliamps through a resistor, a few centimetres from the eye. Measure or
calculate it for your LED before anyone wears it, and write the number here.
TODO: measured value.

## Software check

```bash
python study/record_block.py --session TEST --room A --block check3 --ir-camera 1
```

The preview shows the webcam and the IR image side by side. The pupil must be
clearly dark and the glint visible in every gaze direction on the screen.
