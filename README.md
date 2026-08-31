# 🎙️ Voice Anti-Spoofing & Deepfake Detection Pipeline

An end-to-end Machine Learning pipeline designed to detect genuine (real) versus fake (TTS-generated and Replay attack) human voices using robust spectral statistics, acoustic feature engineering, and calibrated decision boundaries.

---

## 📌 Project Overview

With the rapid advancement of Text-to-Speech (TTS) models and voice-cloning technology, differentiating authentic human speech from synthetic or replayed audio has become a critical security challenge.

This project implements a lightweight, data-leak-free classification pipeline that:
- Extracts comprehensive spectral, dynamic, and temporal acoustic features.
- Aggregates features via global summary statistics to avoid spatial/speaker memorization.
- Calibrates classification thresholds using **Equal Error Rate (EER)** to balance False Acceptance Rate (FAR) and False Rejection Rate (FRR).

---

## 🔬 Feature Engineering

Instead of relying solely on raw waveforms or uncompressed spectrograms, the pipeline extracts a multi-faceted feature set:

| Feature Set | Dimensionality | Acoustic Purpose |
| :--- | :--- | :--- |
| **MFCCs (1–40)** | 40 coefficients | Captures vocal tract shape and spectral envelope |
| **Delta ($\Delta$)** | 40 coefficients | Measures rate of spectral change (velocity) |
| **Delta-Delta ($\Delta\Delta$)** | 40 coefficients | Measures acceleration of spectral transitions (prosody) |
| **Spectral Centroid** | 1 (mean + std) | Represents brightness / center of mass of spectrum |
| **Spectral Rolloff** | 1 (mean + std) | Identifies frequency cutoff and transmission roll-offs |
| **Zero-Crossing Rate** | 1 (mean + std) | Differentiates voiced speech from high-frequency artifacts |

**Statistical Aggregation:** Time-axis pooling (mean and standard deviation) is computed across all spectral frames, compressing high-dimensional audio into a robust 1D feature vector (~164 features) to prevent overfitting on small-scale datasets.

---

## 📂 Repository Structure

```text
├── saved_models/              # Serialized model, scaler, and configuration files
│   ├── classifier.pkl
│   ├── scaler.pkl
│   └── config.pkl
├── figures/                   # Evaluation plots and diagrams
│   ├── confusion_matrices.png
│   ├── threshold_calibration.png
│   └── voice_comparison.png
├── notebooks/
│   └── voice_anti_spoofing_pipeline.ipynb
├── .gitignore
├── requirements.txt
└── README.md

## 📂 Dataset

The audio dataset used for training and evaluating this pipeline is hosted on Kaggle:
* **Dataset Link:** [Kaggle Audio Dataset (faisalziyadahmed/audio-data-set)](https://www.kaggle.com/datasets/faisalziyadahmed/audio-data-set)[cite: 1]

### Expected Directory Structure
```text
data/
└── Audio/
    ├── real/         # Genuine recorded human voices (156 samples)
    └── fake/
        ├── tts/      # Text-to-Speech synthesized voices (70 samples)
        └── replay/   # Replay attack voice recordings (150 samples)
