# Target-speaker verification and Personal VAD feasibility

## Scope and evidence

This is a source-based feasibility review only. No speaker model was added to EmoLight or run on this machine. The comparison distinguishes frame-level target-speaker activity detection (Personal VAD) from utterance-level speaker verification; they answer different questions.

## Candidate families

| Candidate | Published or artifact size | Enrollment/runtime shape | Evidence and limits |
|---|---|---|---|
| Personal VAD 1.0 | Paper reports a 130K-parameter network. A separate, independent implementation reports 129,795 parameters, 507.75 KB Safetensors, 627.24 KB FP32 TFLite, and 262.38 KB 8-bit TFLite. | Target-conditioned, frame-level three-way output: non-speech, target speech, non-target speech. Its embedding-conditioned design consumes an enrolled target d-vector; the paper describes aggregating enrollment recordings into a stored target vector. | The paper's 130K network is small, but it depends on a speaker embedding produced by a separate model during enrollment. The open implementation reports mAP/frame metrics and ROC-AUC on its own multilingual data, not EmoLight-domain FAR/FRR. The open model explicitly says it is an independent reproduction, not an official Google release. |
| Personal VAD 2.0 | Paper reports a 2.8 MB Conformer model and about 75% model-size reduction with 8-bit quantization (roughly 0.7 MB by arithmetic). | Streaming speaker-conditioned VAD optimized as part of an on-device ASR system. Enrollment and enrollment-less operation are both studied. | The paper evaluates downstream ASR WER; those results do not establish speaker-verifier FAR/FRR. The paper does not establish an MCU port or EmoLight CPU/RAM performance. |
| SpeechBrain ECAPA-TDNN verifier | The currently published `embedding_model.ckpt` is 83.3 MB; the checkpoint is PyTorch/pickle-based. Parameter count, INT8 size, and MCU memory were not established in this review. | Extract utterance embeddings, then compare with a stored enrollment embedding using cosine scoring. | The model card reports 0.80% EER on cleaned VoxCeleb1 test and training on VoxCeleb1+2. This is a useful PC reference, but the checkpoint is far outside the project's size target and the reported EER does not transfer to a different microphone, population, or noise domain. |

The independent reproduction describes the original papers' internal 3-layer LSTM speaker embedding network as having 4.88 million parameters. At four bytes per FP32 parameter, its weights alone would be about 19.5 MB; this is a derived estimate that excludes runtime buffers. The embedding-conditioned Personal VAD design can avoid running that large extractor continuously after enrollment, but a fully local product still needs a local registration path or a separately validated enrollment tool. A single 256-D FP32 enrollment vector is about 1 KiB, excluding metadata and secure storage.

The independent Personal VAD reproduction reports its own 256-D speaker subspace fitted on 98 Multilingual LibriSpeech speakers and ROC-AUC 0.9521 on 35 unseen speakers. That result is not FAR/FRR, and its data and evaluation conditions are not the target EmoLight deployment. It is a candidate for a reproducible research baseline, not evidence of readiness.

## Requirements before integration

- Keep enrollment audio and voice templates local, obtain informed consent, and make deletion available. Enrollment tooling and the always-on resident model have separate CPU, memory, and privacy costs.
- Define the target operating point before selecting a threshold. Evaluate false acceptance rate (FAR) and false rejection rate (FRR) on held-out sessions and non-target people; include DET/ROC and EER as summaries, not as a replacement for the selected operating point.
- Test clean speech, low SNR, room changes, microphone changes, short enrollment, similar voices, and overlapping speakers. Split by person and session so windows from one utterance do not appear as independent trials.
- Measure streaming latency P50/P95, peak RAM, model and enrollment sizes, CPU use, and dropped frames on the intended PC. Measure the combined resident path, not only the Personal VAD network.
- For MCU feasibility, export and benchmark the complete quantized inference path, feature extractor, enrollment storage, and tensor arena on the named board. No board or FAR/FRR acceptance target is configured here, so MCU and identity readiness remain `NOT_EVALUATED`.

## Decision

Do not configure `SpeakerVerifier` from these literature numbers. Preserve fail-closed target mode. A later opt-in experiment can compare a small embedding-conditioned Personal VAD against a separate compact verifier on consented, speaker-exclusive target/non-target recordings. Only an independent FAR/FRR result for the intended environment can change `NOT_CONFIGURED`.

AI-assisted source review was used; cited measurements remain claims of their respective papers or model card, not EmoLight measurements.

## Sources

- Ding et al., [Personal VAD: Speaker-Conditioned Voice Activity Detection](https://arxiv.org/abs/1908.04284).
- Ding et al., [Personal VAD 2.0: Optimizing Personal Voice Activity Detection for On-Device Speech Recognition](https://arxiv.org/abs/2204.03793).
- Independent reproduction, [Personal VAD 1.0 ET/WPL model card and artifacts](https://huggingface.co/wq2012/personal-vad-v1-et-wpl).
- SpeechBrain, [ECAPA-TDNN VoxCeleb speaker verification model card](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb).
