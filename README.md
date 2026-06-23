# Learning Normal for Open-World Anomaly Detection in Campus CCTV

This repository contains **unsupervised video anomaly detection for campus CCTV footage**. The goal is to learn normal campus activity patterns and flag strong out-of-distribution events, especially weapon-related anomalies, using reconstruction error.

The system uses a two-stage pipeline:

1. **VideoMAE feature extraction** to convert short video clips into rich spatiotemporal feature vectors.
2. **Memory-Augmented Autoencoder (MemAE)** to learn normal behavior and identify anomalies based on reconstruction error.

---

## Project Overview

Campus CCTV footage can contain diverse lighting conditions, occlusions, crowds, motion patterns, and background changes. Since it is difficult to collect labeled examples for every possible anomaly, this project uses an **unsupervised reconstruction-based approach**.

The model is trained on normal campus activity. At test time, clips that reconstruct poorly are assigned higher reconstruction errors and are classified as anomalies when their error exceeds an unsupervised threshold.

---

## Methodology

### 1. Feature Extraction with VideoMAE

A pretrained **VideoMAE** model is used as a feature extractor. Video clips are sampled using 16-frame windows, and each clip is converted into a 768-dimensional feature representation.

### 2. Memory-Augmented Autoencoder

The extracted VideoMAE features are passed into a **Memory-Augmented Autoencoder (MemAE)**. The autoencoder learns to reconstruct normal feature patterns, while the memory module stores representative normal latent prototypes.

### 3. Anomaly Scoring

At inference time, the reconstruction error is used as the anomaly score:

```text
If reconstruction error > threshold: anomaly
If reconstruction error <= threshold: normal
```

The threshold is selected using the **95th percentile of the training reconstruction error distribution**.

---
## Dataset

The project uses the **ShanghaiTech Campus dataset** for normal and in-domain campus video evaluation, along with a **synthetic weapon anomaly dataset** for testing strong out-of-distribution weapon-related anomalies.

The datasets and related project files can be accessed here:

[Google Drive Dataset Folder](https://drive.google.com/drive/folders/1e92myC8NSD9pFQEx4sc3UWeWVSH8vM3Z?usp=sharing)

After opening the link, add the dataset folder as a shortcut to `MyDrive` so the notebook paths work correctly in Google Colab.

## How to Run

1. Open the notebook in Google Colab.
2. Enable GPU from `Runtime > Change runtime type > GPU`.
3. Add the provided Google Drive dataset folder as a shortcut to `MyDrive`.
4. Update the path configuration in the first notebook cell if needed.
5. Mount Google Drive.
6. Run the dependency installation cell.
7. Do **not** run cells labeled `DO NOT RUN` unless you want to regenerate datasets or features.
8. Run the remaining cells sequentially.

---

## Results

| Evaluation Set          | Mean Reconstruction Error |     Min-Max Error |               Percentiles 50/90/95/99 |
| ----------------------- | ------------------------: | ----------------: | ------------------------------------: |
| Training normal clips   |                   0.00047 | 0.00020 - 0.00535 | 0.00042 / 0.00066 / 0.00082 / 0.00124 |
| ShanghaiTech test clips |                   0.00170 | 0.00029 - 0.00735 | 0.00111 / 0.00347 / 0.00562 / 0.00681 |
| Weapon anomaly clips    |                   0.01071 | 0.00517 - 0.01402 | 0.01063 / 0.01382 / 0.01392 / 0.01399 |

Using the unsupervised threshold, the evaluated weapon anomaly clips were flagged at:

```text
100% anomaly detection rate
```

---
## Limitations

The model performs well on strong out-of-distribution threats such as weapon-related anomalies, which was the main scope of this project. However, it is less reliable for more subtle or everyday abnormal activities in the ShanghaiTech test set, such as a pedestrian being on the wrong side of a truck or a person playing on the road.

This limitation is expected because the model was trained using an unsupervised reconstruction-based setup and was mainly evaluated for weapon-based anomalies. As a result, the system is better suited for detecting clear visual deviations from normal campus activity than for distinguishing all possible types of unusual behavior.

## Technologies Used

* Python
* Google Colab
* PyTorch
* Torchvision
* Hugging Face Transformers
* VideoMAE
* NumPy
* Matplotlib
* scikit-learn
* PyAV

---

## Author

**Samavia Jaffery**
