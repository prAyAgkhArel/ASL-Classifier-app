# Real-Time Sign Language Translator for Deaf-Mute Communication

**An Android-based sign language recognition prototype combining computer vision, deep learning, and optional embedded hardware for accessible communication.**

**Major Project | Bachelor of Electronics, Communication and Information Engineering**  
**Tribhuvan University — Institute of Engineering, Purwanchal Campus, Dharan, Nepal**  
**March 2026**

## Overview

This project develops a portable, real-time sign language translator that captures gestures through an Android device's front camera and converts recognized signs into readable text. It is designed to help bridge communication gaps between sign-language users and people unfamiliar with signing.

The application uses a **hybrid recognition pipeline**: a static **letter mode** based on single-hand landmarks and a dynamic **word mode** based on hand-and-pose sequences. Recognition runs **on the Android device** using TensorFlow Lite; no external server is required for the main inference loop.

The result is an **academic prototype**, not a complete or production-ready sign language interpreter.

## Features

- **Live camera recognition:** Uses CameraX for front-camera preview and frame analysis.
- **Letter recognition:** Identifies static alphabet signs and control classes such as `space` and `del`.
- **Word recognition:** Classifies selected isolated dynamic signs using a sequence-based model with **54 word classes**.
- **On-device ML:** Runs TensorFlow Lite inference locally on Android.
- **Landmark-based processing:** Uses MediaPipe Tasks to track hands and body pose.
- **Clear status feedback:** Jetpack Compose interface indicates recognition mode, capture state, and predictions.
- **Reverse fingerspelling:** Supports representing input text using corresponding ASL fingerspelling signs.
- **Additional output paths:** Explores speech feedback and an optional ESP32/OLED external text display.

## Technology Stack

| Layer | Technologies |
| --- | --- |
| Mobile application | Kotlin, Android Studio, Jetpack Compose |
| Camera | Android CameraX |
| Landmark detection | MediaPipe Tasks: HandLandmarker and PoseLandmarker |
| Model inference | TensorFlow Lite |
| Temporal word recognition | Bidirectional LSTM (BiLSTM) |
| Optional embedded output | ESP32, OLED display, Arduino IDE |
| Development tools | Git, GitHub |

## System Architecture

```text
                 Android Front Camera
                          |
                 CameraX Frame Analysis
                          |
                MediaPipe Landmark Detection
                          |
                 Recognition Mode Selection
                     /           \
                    /             \
             Letter Mode        Word Mode
             (one hand)     (pose + two hands)
                  |                 |
           21 landmarks       30-frame sequence
                  |                 |
           63 features        516 features/step
                  |                 |
           Letter Model       BiLSTM Word Model
                    \             /
                     \           /
                  TensorFlow Lite
                          |
                 Recognized Text + UI
                          |
             Optional Speech / ESP32-OLED
```

### Letter Recognition

The letter pipeline extracts **21 hand landmarks**, each with three coordinates, then normalizes them relative to the wrist. The resulting **63-value vector** is used for static-sign classification. The pipeline also accounts for handedness and front-camera mirroring.

### Word Recognition

Dynamic signs require motion information. The word pipeline combines pose and hand landmarks into **258 raw features per frame**, normalizes and resamples a captured clip to **30 time steps**, and adds frame-to-frame velocity features. This produces an input tensor of **`(30, 516)`** for a BiLSTM-based classifier covering **54 selected words**.

## Model Evaluation

The following figures are reported from the model development and evaluation experiments in the project report. They **are not end-to-end accuracy guarantees for live Android operation**.

| Metric | Letter Model | Word Model |
| --- | ---: | ---: |
| Training accuracy | ~93.92% | ~86.37% |
| Validation accuracy | ~98.13% | Best ~68.15% |
| Classes | 28 | 54 |
| Input representation | 21 × 3 landmarks | 30 × 516 temporal features |

Additional **word-model classification** results:

- **Top-1 accuracy:** 67.04%
- **Top-3 accuracy:** 84.07%
- **Top-5 accuracy:** 90.74%

The static letter model performed more consistently under controlled framing. Live word recognition was more sensitive to sign boundaries, motion, camera orientation, and environmental conditions.

## Hardware Extension

The project also explores an optional **ESP32-based external display pathway**. The Android application performs the heavy computer-vision and ML processing; the ESP32 is intended to receive stable recognized text and show it on an OLED display. This is an **auxiliary integration pathway**, not a dependency for running the Android recognition system.

## Running the Project

The report describes an Android Studio/Kotlin application, but it does **not specify the repository's exact directory layout, build versions, model filenames, or installation commands**. Once the source code and trained model assets are available in this repository:

1. Open the Android project in **Android Studio**.
2. Review the project's Gradle configuration and resolve its declared dependencies.
3. Confirm that the TensorFlow Lite model and label assets referenced by the application are present and compatible.
4. Build and run the application on a compatible Android device with a working front camera.
5. Grant camera permission, choose **Letter** or **Word** mode, and present a sign to the camera.

> **Important:** Keep the runtime landmark ordering, normalization, tensor shapes, and model-label mappings aligned with those used during model training. See the project report for the processing design. Device and SDK compatibility must be verified against the actual source repository.

## Known Limitations

- Supports a limited vocabulary rather than full conversational or sentence-level translation.
- Dynamic word predictions can be unstable during rapid transitions or ambiguous sign boundaries.
- Accuracy varies with lighting, framing, hand visibility, individual signing style, and device performance.
- Primarily targets one signer using the front camera.
- Optional hardware integration is less mature than the Android application.
- The project report does not establish one definitive end-to-end accuracy metric for the complete live hybrid application.

## Future Improvements

- More reliable sign start/end detection and background rejection.
- Larger and more diverse signer datasets and vocabulary.
- Improved temporal smoothing and prediction confidence handling.
- Expanded user testing with sign-language users.
- Refined text-to-speech and fully integrated external display support.
- Automated input/model compatibility checks and broader Android device optimization.

## Project Team

- **Hitesh Labh**
- **Mausam Lamichhane**
- **Prayag Kharel**
- **Sugam Subedi**

**Supervisor:** Asst. Prof. Manoj Kumar Guragai  
**Department:** Electronics and Computer Engineering  
**Institution:** Purwanchal Campus, Institute of Engineering, Tribhuvan University

## Project Report

*Real-Time Sign Language Translator for Deaf-Mute Communication*, major project report, March 2026. The report documents the design, implementation, training/evaluation results, engineering challenges, and limitations of the prototype.

## Attribution and Usage

This project was completed as academic work by the named team members. The accompanying report reserves academic rights and requires proper acknowledgment for scholarly use; copying, publishing, or commercial use of report content may require permission from the authors and department. **No open-source software license is specified in the report.** Consult the authors before assuming permission to redistribute or reuse the source code.
