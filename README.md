# AI Speak — Serbian Speech-to-Blendshape Avatar Animation

A research/ML project that generates real-time facial animation (ARKit-style blendshapes) directly from Serbian speech audio — the kind of model that drives a talking 3D avatar's lip-sync and facial expression from a voice recording, with no video or motion-capture input needed.


## Concept

Given raw speech audio, the model predicts a sequence of 52 ARKit blendshape weights (the standard set used to drive facial rigs in game engines / AR avatars — eyebrows, eyes, jaw, mouth, tongue, etc.) at 60 fps, synchronized to the audio. This is the "AVATAR" pipeline: audio in, animated face out.

Pipeline stages, in the order they appear in the notebook:

1. **Audio feature extraction** — a Wav2Vec2 model fine-tuned on Serbian speech (`AdvancedAudioProcessor`) turns raw waveform into rich audio embeddings, averaging the last 4 hidden layers.
2. **Dataset** — `ChampionshipDataset` pairs cached audio features with per-frame blendshape targets and phoneme alignments across multiple speakers (recordings of people reading books — "knjige"), with a configurable lookahead window so the model can anticipate upcoming mouth shapes.
3. **Model architecture** — a VQ-based (`VectorQuantizer`) sequence model with an adversarial discriminator (this notebook variant, `NODISC`, trains/evaluates without the discriminator loss for comparison).
4. **Training** — custom loss functions including a Pearson-correlation loss between predicted and target blendshape curves, trained with a `OneCycleLR` schedule.
5. **Post-processing** — a Savitzky-Golay-based denoiser (`AggressiveDenoiser` / `MinimalPostProcessor`) smooths upper-face motion more aggressively than mouth/jaw motion, to keep speech articulation sharp while keeping eyes/brows natural.
6. **Evaluation** — per-speaker Pearson correlation, RTF (real-time factor, i.e. inference speed vs. audio duration) measurement, correlation-per-blendshape breakdowns, and an ablation study comparing the full model against raw (no post-processing) predictions.

## Tech stack

| Purpose | Library |
|---|---|
| Deep learning | PyTorch, `torch.optim` (OneCycleLR scheduler) |
| Speech representation | Hugging Face `transformers` (`Wav2Vec2FeatureExtractor`, `Wav2Vec2Model`) fine-tuned for Serbian, `torchaudio` |
| Signal processing | `scipy.signal` (Savitzky-Golay filter), `scipy.interpolate`, `scipy.stats.pearsonr` |
| Data handling | NumPy, pandas |
| Visualization / reporting | Matplotlib, Seaborn |
| Environment | Google Colab + Google Drive (dataset, cached features, model checkpoints all read/written there) |

## Repository structure

```
.
└── AVATAR_trifon_srpskiwav_knjige_5speakera_NODISC_(1).ipynb
```

Notebook sections (as numbered in-code):

1. Environment setup & imports
2. Dataset & preprocessing (Serbian phoneme set, ARKit blendshape names, `ChampionshipDataset`)
3. Google Drive mount
4. Audio feature extraction & caching
5. Model architecture (VQ layer + discriminator)
6. Loss functions & optimization criteria
7. Trainer logic / training loop
8. Training run configuration
9. Inference & post-processing
10–11. Validation system & results analysis
12. Post-processing (lip-sync smoothing)
13. Validation CSV export
— plus dedicated cells for training curves, per-speaker results tables, correlation-per-blendshape figures, RTF measurement, and an ablation study (with/without post-processing).

## Running it

This notebook is written to run in **Google Colab**, not locally:

1. Open it via the badge at the top of the notebook, or directly:
   `https://colab.research.google.com/github/tijanicica/ai-speak/blob/main/AVATAR_trifon_srpskiwav_knjige_5speakera_NODISC_(1).ipynb`
2. Run the setup/import cell — it detects CUDA and reports available VRAM.
3. Mount Google Drive when prompted (the notebook expects the dataset, cached audio features, the fine-tuned Serbian Wav2Vec2 checkpoint, and trained model checkpoints under a Drive path such as `/content/drive/MyDrive/AI-SPEAK/...`).
4. Run cells top-to-bottom: feature pre-caching → model/training setup → training loop → inference/validation/analysis.

Since the dataset and pretrained checkpoints referenced (Drive paths, speaker recordings) are private and not included in this repo, the notebook is meant to be read as a reference implementation rather than executed end-to-end without that data.
