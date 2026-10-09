# Berlin EmoDB 1.3.0 four-class baseline

## Data and split

- Source: Berlin EmoDB 1.3.0, Zenodo record 7447302. The downloaded archive was 39,981,818 bytes; MD5 `9d21362dbc5676ef3ab4745d83ced0db`; ZIP CRC and all 535 WAV formats were checked.
- Zenodo record metadata lists CC BY 4.0; the archive's embedded audformat metadata says CC0-1.0. Model provenance conservatively uses CC BY 4.0 and retains attribution. No raw audio is included in this repository.
- Gold files use 16 kHz mono PCM16. Filename codes: N neutral, F happy, W angry, T sad. A/fear is excluded, never mapped to angry; boredom and disgust are excluded too.
- Official test speakers 12, 14, 15, 16 remained held out. Official train speakers were divided into train speakers 08, 10, 11, 13 and validation speakers 03, 09. No speaker, original recording, or overlapping crop crosses subsets.
- Fixed windows: 1.5 seconds, 0.5 second stride. Train / validation / test contain 577 / 254 / 569 accepted-manifest windows before audio quality rejection; the test windows come from 136 distinct utterances and four speakers. Metrics below are window-level; overlapping windows from one utterance are correlated.
- Training used 20/10/5/0 dB white-noise augmentation on training windows only. Validation fit the one-vs-rest sigmoid calibration and selected the confidence threshold; the official test set was not used for model selection or threshold tuning.

## Model selection and clean held-out result

Full 24-D was selected from validation results (correctness-F1 0.883, accepted error 0.209, coverage 0.996, Brier 0.330). The 8-D simple baseline had validation correctness-F1 0.721, accepted error 0.426, coverage 0.949, and Brier 0.583. The full model was then evaluated once on the official held-out test speakers.

| Held-out clean metric | Result |
|---|---:|
| Macro-F1 | 0.570 |
| UAR | 0.613 |
| Coverage | 0.953 (542 / 569 windows) |
| Accepted error rate | 0.315 |
| Brier score | 0.429 |
| ECE | 0.055 |

Per-class precision/recall among accepted windows:

| Emotion | Precision | Recall |
|---|---:|---:|
| Angry | 0.653 | 0.965 |
| Happy | 0.556 | 0.049 |
| Neutral | 0.560 | 0.718 |
| Sad | 0.853 | 0.720 |

Confusion matrix (rows=true, columns=predicted; order: angry, happy, neutral, sad):

```text
[[194,  3,  4,   0],
 [ 92,  5,  5,   0],
 [  1,  1, 56,  20],
 [ 10,  0, 35, 116]]
```

Happy recall is inadequate. This is a real trained experimental model, not a dependable four-class emotion recognizer. EmoDB is German acted speech; results do not establish performance on natural conversation, Mandarin, or other speaker populations.

## Robustness

| Condition | Coverage | Macro-F1 | UAR | Accepted error |
|---|---:|---:|---:|---:|
| Clean | 0.953 | 0.570 | 0.613 | 0.315 |
| White noise, 20 dB | 0.341 | 0.496 | 0.558 | 0.433 |
| White noise, 10 dB | 0.000 | Not defined | Not defined | Not defined |
| White noise, 5 dB | 0.000 | Not defined | Not defined | Not defined |
| White noise, 0 dB | 0.000 | Not defined | Not defined | Not defined |
| Other voice, SIR 6 dB | 0.828 | 0.402 | 0.468 | 0.539 |
| Other voice, SIR 0 dB | 0.793 | 0.192 | 0.320 | 0.754 |
| Other voice, SIR -6 dB | 0.629 | 0.111 | 0.276 | 0.838 |

At 10/5/0 dB the audio-quality gate rejected every window, so no classification metric is defined. The low-SIR overlap results are poor and are not evidence of target-speaker separation. Background music, fan, recorded room noise, and reverb were `NOT_EVALUATED` because no licensed source manifest was available.

## Lightweight and live-path measurements

- JSON model: 12,219 bytes; 180 numeric parameters; CPU / NumPy inference only.
- On the final held-out evaluation run: 24-D feature extraction median 9.89 ms / p95 14.48 ms; NumPy model inference median 0.074 ms / p95 0.121 ms; end-to-end per window median 1.61 ms / p95 14.76 ms.
- All evaluation conditions together: 27.47 process CPU seconds, 27.67 wall seconds, 0.993 average CPU cores, peak RSS 153,378,816 bytes.
- WAV replay emits four-class scores through the same 1.5 s rolling buffer and 0.5 s update runtime. A separate 5-second local microphone smoke run connected successfully with zero dropped frames and rejected silence. No speech sample was captured; this does not measure emotion accuracy or target identity.

## Identity and remaining limits

The output mode is `EMOTION_ONLY_EXPERIMENTAL` with identity `NOT_EVALUATED`. It never creates a target-active event or lighting command. The target speaker verifier, Mandarin/natural-speech performance, licensed environmental noise evaluation, and MCU port remain unverified.

The full machine-readable metrics, per-condition reasons, latency, CPU, and RSS are in [`emodb_1_3_0_evaluation.json`](emodb_1_3_0_evaluation.json). Model attribution and commands are in [`../models/README.md`](../models/README.md).
