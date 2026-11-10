# Credits and related work

## Built on

- **GazeMap** ([github.com/fammad/GazeMap](https://github.com/fammad/GazeMap)): the
  webcam gaze model, its training and its live pipeline. Trained weights: release
  [v1.0.0](https://github.com/fammad/GazeMap/releases/tag/v1.0.0).
- **MPIIGaze**: Zhang, Sugano, Fritz and Bulling, *MPIIGaze: Real-World Dataset and
  Deep Appearance-Based Gaze Estimation*, IEEE TPAMI 2019
  ([paper](https://arxiv.org/abs/1711.09017)). Training data for the base model.
- **MediaPipe Face Landmarker** (Google): face points for head pose and eye crops.

## Closest related work

- Jindal, Kaur and Manduchi, *Tracker/Camera Calibration for Accurate Automatic Gaze
  Annotation of Images and Videos*, ETRA 2022. An IR tracker labels camera images
  during natural viewing.
- Park et al., *Towards End-to-end Video-based Eye-Tracking* (EVE), ECCV 2020.
  Webcam, screen and eye tracker recorded together.
- Chen and Shi, *Offset Calibration for Appearance-Based Gaze Estimation via Gaze
  Decomposition*, WACV 2020, and *Towards High Performance Low Complexity
  Calibration*, TPAMI 2022. Most of the person-specific error is a constant offset.
- Park et al., *Few-Shot Adaptive Gaze Estimation* (FAZE), ICCV 2019.

## Also by the author

- [gaze-XR-training](https://github.com/fammad/gaze-XR-training): infrared eye
  tracking in a Meta Quest 3, with a Unity toolkit.
