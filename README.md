# HeTu-FM: Training Framework for Survey-Scale Radio Source Identification and Classification

**HeTu-FM** is an AI-for-Science framework developed for automated radio-source identification, instance segmentation, and morphological classification in large-area radio surveys.

This repository contains the **training and model adaptation components** of HeTu-FM. The framework is designed to bridge modern visual foundation models and survey-scale radio astronomy, enabling complex radio structures to be transformed from image-level detections into scientifically usable source representations.

The current HeTu-FM project is primarily developed and validated using data from the **Rapid ASKAP Continuum Survey at 1367.5 MHz (RACS-mid)**.

> **Paper:**
> *HeTu-FM: A Foundation-Model-Based Framework for Radio Galaxy Identification and Classification at Survey Scale*
> Yao Dai et al.
> Submitted to *The Astrophysical Journal Supplement Series (ApJS)*.

---

## 1. Scientific Motivation

Next-generation radio surveys are rapidly increasing both the volume and complexity of astronomical imaging data.

For extended radio sources, the scientific problem is substantially more difficult than simply determining whether an object is present in an image. A practical survey-scale system must simultaneously address:

* identification of radio sources in complex backgrounds;
* separation of spatially overlapping or extended emission;
* morphological classification of radio galaxies;
* reliable localization of individual radio components;
* transfer of image-space predictions into celestial coordinates;
* construction of source catalogues suitable for subsequent astrophysical analysis.

Traditional source-extraction pipelines remain highly effective for many compact sources, but complex extended radio morphologies can be difficult to represent using predefined Gaussian components alone.

HeTu-FM therefore explores a complementary approach in which **large-scale visual representation learning and instance-level astronomical source analysis are integrated into a unified framework**.

The broader goal is to move computer vision in radio astronomy from

**image recognition**

toward

**survey-scale scientific catalogue construction and source discovery**.

---

## 2. HeTu-FM Framework

HeTu-FM follows a **pretrain-and-adapt** paradigm for radio astronomical images.

The framework combines a modern hierarchical visual backbone with an instance-segmentation architecture so that source detection, morphological recognition, and spatial segmentation can be performed simultaneously.

The main pipeline can be summarized as:

```text
Radio Survey Images
        │
        ▼
Visual Representation Backbone
        │
        ▼
Multi-scale Feature Representation
        │
        ▼
Instance Detection & Segmentation
        │
        ├── Bounding Box
        ├── Instance Mask
        ├── Morphological Class
        └── Confidence Score
        │
        ▼
Astronomical Coordinate Reconstruction
        │
        ▼
Source Measurement & Deduplication
        │
        ▼
Scientific Catalogue
```

The current model uses a hierarchical **InternImage-family visual backbone with deformable convolution** together with a two-stage instance-segmentation framework.

The purpose of this design is not merely to improve image-level classification accuracy, but to obtain spatially resolved representations that can subsequently support astronomical measurement and catalogue construction.

---

## 3. Radio Morphology Classes

The current HeTu-FM training task focuses on four representative radio-source morphologies:

| Class     | Description                         |
| --------- | ----------------------------------- |
| **CS**    | Compact Source                      |
| **CJ**    | Core-Jet Source                     |
| **FR I**  | Fanaroff–Riley Type I radio galaxy  |
| **FR II** | Fanaroff–Riley Type II radio galaxy |

These classes span compact, asymmetric, and extended double-lobed radio structures and therefore provide a useful benchmark for evaluating whether visual models can learn physically meaningful radio morphology.

---

## 4. Dataset

The primary training and validation data are constructed from **RACS-mid** observations obtained with the Australian Square Kilometre Array Pathfinder (ASKAP).

RACS-mid provides wide-area radio continuum imaging at approximately 1.37 GHz and offers an important precursor dataset for developing automated analysis methods toward the SKA era.

### Annotation

HeTu-FM uses instance-level annotations containing:

* source class;
* bounding box;
* instance segmentation mask;
* image/source association.

Segmentation annotations are generated through a combination of machine-assisted annotation and manual astronomical verification.

The labelled dataset is used for supervised adaptation of the visual representation to radio-source morphology.

### Data availability

The original RACS-mid survey images are **not redistributed in this repository**.

Users should obtain the survey data from the official RACS data release and prepare the training samples according to the corresponding HeTu-FM data-processing pipeline.

RACS:

https://research.csiro.au/racs/

---

## 5. Training Strategy

The HeTu-FM training procedure is designed to transfer general visual representation capability to radio astronomical morphology.

The major stages are:

```text
Pre-trained visual representation
              │
              ▼
Radio-domain dataset construction
              │
              ▼
Instance-level source annotation
              │
              ▼
Supervised radio-domain adaptation
              │
              ▼
Detection + Segmentation + Classification
              │
              ▼
Survey-scale inference
```

During radio-domain adaptation, the model jointly learns to recognize source morphology and recover the spatial extent of individual radio sources.

This is important for HeTu because the instance mask is subsequently used not only as a computer-vision prediction, but also as an interface to downstream astronomical operations such as:

* celestial-coordinate reconstruction;
* radio flux-density measurement;
* source-size estimation;
* catalogue cross-matching;
* duplicate-source removal;
* unusual-source candidate selection.

---

## 6. Training Workflow

A typical HeTu-FM experiment contains the following stages.

### Step 1 — Prepare RACS-mid data

Download the required RACS-mid observations and generate radio-source image samples.

The corresponding FITS/WCS information should be retained because celestial-coordinate reconstruction is required during the downstream scientific analysis.

### Step 2 — Prepare annotations

Each training source should contain the information required for instance segmentation and morphology classification, including:

```text
image
 ├── source class
 ├── bounding box
 └── instance mask
```

### Step 3 — Configure the model

Select the visual backbone, initialization weights, number of morphology classes, training dataset, validation dataset, and optimization settings.

### Step 4 — Train / fine-tune

Run the corresponding training configuration contained in this repository.

> The exact command and configuration path will be documented here according to the final public repository structure.

### Step 5 — Evaluate

Evaluate the trained model on an independent validation/test set using detection, classification, and segmentation metrics.

For scientific applications, model performance should additionally be examined as a function of observational properties such as source morphology, signal-to-noise ratio, angular scale, and background complexity.

---

## 7. From Machine Learning to Scientific Catalogue Construction

The neural-network output represents only the first stage of the complete HeTu-FM scientific workflow.

The downstream HeTu pipeline further converts model predictions into astronomical measurements:

```text
HeTu-FM inference
      │
      ▼
Instance masks
      │
      ▼
WCS coordinate reconstruction
      │
      ▼
AI + Physics source measurement
      │
      ▼
Global celestial-coordinate deduplication
      │
      ▼
Catalogue cross-matching
      │
      ▼
Radio-source catalogue
      │
      ▼
Rare / unusual source discovery
```

This separation between **model training** and **scientific analysis** is intentional.

This repository focuses on learning robust radio-source representations, whereas astronomical catalogue construction and physical analysis are maintained separately.

---

## 8. Downstream Scientific Analysis

The downstream HeTu-FM catalogue-construction and scientific-analysis code is available at:

**HeTu-FM Scientific Analysis**

https://github.com/ydai-astro/HeTu-Foundation-Model-scientific-analysis

That repository includes components associated with:

* survey-scale source processing;
* WCS-based localization;
* source catalogue generation;
* astronomical parameter measurement;
* catalogue cross-matching;
* statistical analysis of the resulting source population.

Together, the two repositories form the current HeTu-FM workflow:

```text
HeTu-FM-train
      │
      │ model training
      │
      ▼
trained HeTu-FM model
      │
      │ survey inference
      ▼
HeTu-FM scientific analysis
      │
      ▼
radio-source catalogue
      │
      ▼
astrophysical analysis
```

---

## 9. Research Scope

HeTu-FM is being developed as part of a broader effort toward scalable AI-assisted radio astronomy.

Current research directions include:

* radio visual foundation models;
* survey-scale instance segmentation;
* robust extended-source localization;
* physically constrained source measurement;
* morphology-aware catalogue construction;
* low-surface-brightness source recovery;
* anomalous and rare radio-source discovery;
* cross-survey transfer and generalization;
* intelligent astronomical data-processing pipelines.

An important long-term objective is to develop transferable AI infrastructure capable of supporting the much larger data volumes expected from the **Square Kilometre Array (SKA)**.

---

## 10. HeTu Research Ecosystem

HeTu is evolving from an astronomical object-detection system toward a broader AI-for-Science framework for radio-survey analysis.

```text
HeTu
 │
 ├── Radio-source detection
 │
 ├── Instance segmentation
 │
 ├── Morphological classification
 │
 ├── Foundation-model representation
 │
 ├── Physical source measurement
 │
 ├── Survey-scale catalogue construction
 │
 └── Rare-source discovery
```

Related work also explores the recovery and identification of Galactic supernova remnants using context-expanded radio-source representations and visual foundation models.

---

## 11. Citation

If this repository contributes to your research, please cite the corresponding HeTu-FM paper.

```bibtex
@article{dai_hetufm,
  author  = {Dai, Yao and others},
  title   = {HeTu-FM: A Foundation-Model-Based Framework for Radio Galaxy Identification and Classification at Survey Scale},
  journal = {The Astrophysical Journal Supplement Series},
  note    = {submitted}
}
```

The BibTeX information will be updated after publication.

---

## 12. Acknowledgements

This project makes use of radio continuum observations from the **Rapid ASKAP Continuum Survey (RACS)**.

ASKAP is part of the Australia Telescope National Facility and is operated by CSIRO.

We also acknowledge the open-source computer-vision and astronomical software communities whose tools support the development of the HeTu framework.

---

## 13. Contact

**Yao Dai**

Shanghai Astronomical Observatory, Chinese Academy of Sciences

GitHub:
https://github.com/ydai-astro

For questions related to HeTu-FM training, astronomical applications, or collaboration, please open an issue in this repository.
