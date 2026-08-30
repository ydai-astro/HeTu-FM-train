# HeTu-FM-train

Training code for **HeTu-FM**, a foundation-model-based framework for radio-source identification, instance segmentation, and morphological classification.

HeTu-FM is developed for large-scale radio astronomical surveys and is currently applied to radio-source analysis using data from the **Rapid ASKAP Continuum Survey at 1367.5 MHz (RACS-mid)**.

> **Paper**
> *HeTu-FM: A Foundation-Model-Based Framework for Radio Galaxy Identification and Classification at Survey Scale*
> Yao Dai et al.
> Submitted to *The Astrophysical Journal Supplement Series (ApJS)*.

---

## Overview

Large radio surveys contain a wide variety of compact and extended radio sources. Complex structures such as radio galaxies can be difficult to describe using traditional source-extraction methods alone.

HeTu-FM applies modern computer-vision models to jointly perform:

* radio-source detection;
* morphological classification;
* instance segmentation;
* source-level visual representation learning.

This repository focuses on the **training and evaluation of the HeTu-FM visual model**.

The downstream astronomical analysis pipeline is maintained separately.

---

## Radio-source Classes

The current model is designed to identify four representative radio-source morphologies:

| Class | Description                         |
| ----- | ----------------------------------- |
| CS    | Compact Source                      |
| CJ    | Core-Jet Source                     |
| FR I  | Fanaroff–Riley Type I radio galaxy  |
| FR II | Fanaroff–Riley Type II radio galaxy |

These classes contain both compact and extended radio structures and are used to evaluate the capability of the model to recognize complex radio morphology.

---

## Model

HeTu-FM uses an instance-segmentation framework to simultaneously predict:

```text
Input radio image
        │
        ▼
Visual backbone
        │
        ▼
Multi-scale image features
        │
        ▼
Detection / Segmentation Head
        │
        ├── Source class
        ├── Bounding box
        ├── Instance mask
        └── Confidence score
```

The model is designed to provide both morphological classification and spatial information for individual radio sources.

The resulting instance masks can subsequently be used by downstream astronomical analysis pipelines.

---

## Dataset

The current HeTu-FM experiments are primarily based on **RACS-mid** radio continuum images obtained with the Australian Square Kilometre Array Pathfinder (ASKAP).

The training samples contain radio-source images and corresponding instance-level annotations.

Typical annotation information includes:

* source morphology class;
* source bounding box;
* instance segmentation mask.

The original RACS-mid survey data are not redistributed in this repository.

RACS data can be obtained from the official survey resources:

https://research.csiro.au/racs/

---

## Dataset Structure

A typical dataset may be organized as:

```text
dataset/
├── train/
│   ├── images/
│   └── annotations/
├── val/
│   ├── images/
│   └── annotations/
└── test/
    ├── images/
    └── annotations/
```

The exact annotation format should follow the dataset configuration used by the training framework in this repository.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ydai-astro/HeTu-FM-train.git
cd HeTu-FM-train
```

Create an isolated Python environment before installing the required packages.

For example:

```bash
conda create -n hetufm python=3.10
conda activate hetufm
```

Then install the dependencies required by the training framework.

```bash
pip install -r requirements.txt
```

If the repository uses a different environment file, please follow the corresponding dependency configuration included in the project.

---

## Training

Before training, configure:

* training dataset path;
* validation dataset path;
* number of morphology classes;
* pretrained model weights;
* output directory;
* training hyperparameters.

A typical training procedure is:

```bash
python <training_script.py> <config_file>
```

Replace the command above with the corresponding training script and configuration file included in this repository.

Training outputs generally include:

```text
outputs/
├── checkpoints/
├── logs/
└── evaluation_results/
```

---

## Evaluation

The trained model can be evaluated on an independent validation or test dataset.

The model produces source-level predictions including:

```text
class label
confidence score
bounding box
instance mask
```

Evaluation can therefore consider several complementary aspects:

* source detection performance;
* morphology classification performance;
* instance-segmentation performance.

For astronomical applications, performance should also be inspected across sources with different morphology, signal-to-noise ratio, angular scale, and image complexity.

---

## Inference Output

For an input radio image, HeTu-FM produces instance-level predictions:

```text
Radio image
    │
    ▼
HeTu-FM
    │
    ├── class
    ├── score
    ├── bounding box
    └── segmentation mask
```

These outputs provide the interface between the machine-learning model and subsequent astronomical analysis.

---

## Downstream Scientific Analysis

This repository focuses only on **HeTu-FM model training and evaluation**.

Operations such as:

* WCS-based celestial-coordinate reconstruction;
* astronomical source measurement;
* survey-scale catalogue construction;
* duplicate-source removal;
* catalogue cross-matching;
* statistical and astrophysical analysis;

belong to the downstream HeTu scientific-analysis workflow and are **not part of this training repository**.

The corresponding code is available at:

### HeTu-FM Scientific Analysis

https://github.com/ydai-astro/HeTu-Foundation-Model-scientific-analysis

The relationship between the two repositories is:

```text
HeTu-FM-train
      │
      │ model training
      ▼
trained model
      │
      │ survey inference
      ▼
HeTu-FM scientific analysis
      │
      ▼
astronomical source catalogue
```

---

## Research Goal

The goal of HeTu-FM is to investigate how modern visual models can support the automated interpretation of large radio surveys.

Rather than treating radio-source analysis only as an image-classification problem, HeTu-FM aims to provide instance-level representations that can be connected to subsequent astronomical measurements and scientific analysis.

This training repository provides the machine-learning component of that workflow.

---

## Citation

If this repository is useful for your research, please cite:

```bibtex
@article{dai2026hetufm,
  author  = {Dai, Yao and others},
  title   = {HeTu-FM: A Foundation-Model-Based Framework for Radio Galaxy Identification and Classification at Survey Scale},
  journal = {The Astrophysical Journal Supplement Series},
  note    = {submitted}
}
```

The citation information will be updated after publication.

---

## Contact

**Yao Dai**

Shanghai Astronomical Observatory, Chinese Academy of Sciences

GitHub:
https://github.com/ydai-astro

For questions about the HeTu-FM training framework, please open an issue in this repository.
