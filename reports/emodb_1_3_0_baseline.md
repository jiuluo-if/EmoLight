# Berlin EmoDB 1.3.0 four-class baseline

## Data and split

- Source: Berlin EmoDB 1.3.0, Zenodo record 7447302. The downloaded archive was 39,981,818 bytes; MD5 `9d21362dbc5676ef3ab4745d83ced0db`; ZIP CRC and all 535 WAV formats were checked.
- Zenodo record metadata lists CC BY 4.0; the archive's embedded audformat metadata says CC0-1.0. Model provenance conservatively uses CC BY 4.0 and retains attribution. No raw audio is included in this repository.
- Gold files use 16 kHz mono PCM16. Filename codes: N neutral, F happy, W angry, T sad. A/fear is excluded, never mapped to angry; boredom and disgust are excluded too.
- The four official held-out test speakers remained isolated; four official training speakers were used for train and two for validation. No speaker, original recording, or overlapping crop crosses subsets. Public reports contain only speaker-aggregated summaries, not dataset speaker identifiers.
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

The current fresh run reproduced the published model's fitted arrays exactly. At the original-recording level, mean calibrated scores over quality-eligible overlapping windows produced Macro-F1 **0.608**, UAR **0.653**, coverage **0.985** (134 / 136 recordings), and accepted error **0.269**. A 1,000-resample class-stratified bootstrap over recordings gave 95% intervals: Macro-F1 **[0.557, 0.657]**, UAR **[0.602, 0.704]**, coverage **[0.963, 1.000]**, and accepted error **[0.224, 0.313]**. Aggregated happy recall was **0 / 25** accepted recordings; its bootstrap interval was **[0.000, 0.000]**.

Across speakers, recording-aggregated Macro-F1 ranged **0.509–0.679**, UAR **0.614–0.729**, coverage **0.971–1.000**, and accepted error **0.136–0.333**; happy recall was zero for each speaker. These per-speaker measurements are retained as anonymous ranges, without publishing speaker IDs.

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

## Validation-only model exploration

Candidates were trained with the same manifest, speaker split, seed, and feature pipeline; the listed change is the only training option varied. Selection used only the two validation speakers.

| Validation candidate | Macro-F1 | UAR | Happy recall | Coverage | Accepted error | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| LinearSVC, white-noise augmentation 20/10/5/0 dB | 0.707 | 0.710 | 0.206 | 0.996 | 0.209 | 0.330 | 0.062 |
| LinearSVC, happy weight 2 | 0.707 | 0.710 | 0.206 | 0.996 | 0.209 | 0.330 | 0.063 |
| LinearSVC, happy weight 4 | 0.707 | 0.710 | 0.206 | 0.996 | 0.209 | 0.330 | 0.064 |
| Logistic Regression, happy weight 1 | 0.644 | 0.674 | 0.000 | 0.953 | 0.196 | 0.331 | 0.106 |
| LinearSVC, white-noise augmentation 20/10 dB | 0.723 | 0.719 | 0.265 | 1.000 | 0.204 | 0.310 | 0.057 |

The reduced-augmentation candidate raised validation happy recall from 0.206 to 0.265 (7/34 to 9/34 accepted windows); the small sample comes from two speakers and overlapping windows.

The reduced-augmentation change is small and based on only 34 happy validation windows from two speakers; windows overlap within source recordings. The candidate JSON is 13,930 bytes versus 11,949 bytes for the sanitized baseline (+17% metadata/provenance), with the same 180 numeric parameters. The existing baseline therefore remains the default based on the validation evidence and deployment-size preference. Held-out candidate metrics below are outcome reporting only; they did not select or tune a model.

This reduced-augmentation candidate was evaluated once on the held-out speakers after the validation-only decision. On held-out windows it scored Macro-F1 **0.614**, UAR **0.645**, coverage **0.960**, accepted error **0.282**, and happy recall **0.097**. After original-recording aggregation it scored Macro-F1 **0.590**, UAR **0.634**, coverage **1.000**, accepted error **0.294**, and happy recall **0 / 27**. Its recording-level 95% bootstrap intervals were Macro-F1 **[0.536, 0.644]**, UAR **[0.583, 0.690]**, and happy recall **[0.000, 0.000]**. The baseline's recording-level intervals overlap these ranges. These held-out measurements are reported without changing the validation-selected decision.

Across speakers, the candidate's recording-level Macro-F1 ranged **0.470–0.692**, UAR **0.591–0.729**, coverage **1.000**, and accepted error **0.136–0.382**; happy recall remained zero for each speaker.

The existing 8-D energy/rhythm comparison was also measured on the same held-out data: window Macro-F1/UAR were **0.462/0.528** and original-recording Macro-F1/UAR were **0.531/0.572**; happy recall remained zero at the recording level. The 24-D profile had already won the validation comparison; these held-out numbers are an outcome report, not a selection criterion. They do not isolate which individual pitch, energy, or rhythm feature drives happy/angry confusion, so individual-feature ablation remains unverified.

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

- JSON model: 11,949 bytes; 180 numeric parameters; CPU / NumPy inference only.
- On the fresh baseline held-out evaluation run: 24-D feature extraction median 8.52 ms / p95 9.77 ms; NumPy model inference median 0.062 ms / p95 0.080 ms; end-to-end per test item median 1.17 ms / p95 10.56 ms.
- All evaluation conditions together: 25.95 process CPU seconds, 26.49 wall seconds, 0.980 average CPU cores, peak RSS 154,456,064 bytes.
- WAV replay emits four-class scores through the same 1.5 s rolling buffer and 0.5 s update runtime. A separate 5-second local microphone smoke run connected successfully with zero dropped frames and rejected silence. No speech sample was captured; this does not measure emotion accuracy or target identity.

## Identity and remaining limits

The output mode is `EMOTION_ONLY_EXPERIMENTAL` with identity `NOT_EVALUATED`. It never creates a target-active event or lighting command. The target speaker verifier, Mandarin/natural-speech performance, licensed environmental noise evaluation, and MCU port remain unverified.

The full machine-readable metrics, per-condition reasons, latency, CPU, and RSS are in [`emodb_1_3_0_evaluation.json`](emodb_1_3_0_evaluation.json). Model attribution and commands are in [`../models/README.md`](../models/README.md).
