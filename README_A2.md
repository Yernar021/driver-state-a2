# Driver State Monitoring + Liveness — Assignment 2

Same topic as A1 (driver state from a face camera), upgraded with:

- **SOTA encoder:** DINOv2 ViT-S/14 (frozen) instead of A1's MobileNetV2, compared by leave-one-person-out (LOPO).
- **Hybrid model:** geometry + 52 MediaPipe blendshapes + PCA32 image embedding → random forest.
- **Liveness:** training-free blink rule (EAR < 0.8 × own 10 s median; live if a blink in the last 10 s).
- **Web app (Gradio):** camera/upload → state + liveness → DINOv2 attention heatmap + timeline + latency.

## Install

```
pip install torch transformers gradio mediapipe opencv-python scikit-learn scipy pandas joblib matplotlib
```
`code/face_landmarker.task` (MediaPipe) must be present. DINOv2 weights (`facebook/dinov2-small`) download on first run.

## Data

DMD (Vicomtech, CC BY-NC-ND 4.0) must be requested and extracted by the user; it is **not** redistributed here,
nor are models derived from it. Steps 02–03 build `data_processed/dmd/{frames,labels.csv,landmarks}`.
Spoofing recordings of volunteers are private and not included.

## Run (from the project root)

| Step | Command | Output |
|---|---|---|
| MobileNetV2 embeddings (A1 backbone) | `m0b_honest_colab.ipynb` | `results/emb_all_v2.npy` |
| DINOv2 embeddings | `python code/18_dinov2_embed.py` | `results/emb_dinov2_s.npy` |
| Baseline vs SOTA, curves, ablation | `python code/19_a2_experiments.py --task drowsy` / `--task attention` | `results/a2/` |
| Final models | `python code/20_train_final_a2.py --task drowsy --backbone dino` (and attention) | `models/eb_*_dino.joblib` |
| Web app | `python code/app.py` (`--share` for a public link) | http://127.0.0.1:7860 |
| Liveness evaluation | `python code/16_liveness.py` then `python code/17_blink_rule.py` | `results/spoof/` |

Fixed settings: seed 0; 5 fps for state, 15 fps for blinks; 10 s label-free calibration; 1 s window;
PCA32 fitted on ≤ 20,000 training frames; RF 300 trees, min leaf 3, balanced; MLP 128-64, 30 epochs.

## App input

MP4/MOV/WebM, 20–60 s, one face, looking ahead for the first 10 s. The consent box must be ticked.
Tick "synthetic" for any AI-generated clip: outputs are then watermarked SYNTHETIC.

## Ethics

Consent is required in the app; videos are processed in memory and not stored. No DMD frames are shown.
AI assistance (Claude, Anthropic) was used for programming assistance and language editing.
