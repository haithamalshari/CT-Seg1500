
# CT-Seg1500

CT-Seg1500 is a **harmonized, slice-thickness-aware, transformer-ready multi-source brain CT cohort and preprocessing framework** for intracranial hemorrhage (ICH) **segmentation and diagnosis**. :contentReference[oaicite:0]{index=0}

---

## TL;DR

- **1,582 CT volumes** (**942 segmented** + **640 normal**) :contentReference[oaicite:1]{index=1}  
- Aggregated from **5 public datasets + RSNA-derived normals** :contentReference[oaicite:2]{index=2}  
- Slice-thickness harmonization (≈ **5 mm target**) + HU sanity correction + QC :contentReference[oaicite:3]{index=3}  
- Multi-format releases:
  - NIfTI (`.nii.gz`)
  - Raw NumPy (`.npy`)
  - Precomputed **3-channel** NumPy (`.npy`) (two windowed + raw) :contentReference[oaicite:4]{index=4}  
- Full audit trail (reports, deltas, per-case features, summaries) :contentReference[oaicite:5]{index=5}  

---

## Overview Pipeline

![Preprocessing Pipeline](CT-Seg1500-Info/figures/Flow%20Diagram%2001.png) :contentReference[oaicite:6]{index=6}

The pipeline integrates multiple datasets into a unified representation:

- Multi-source merge (CQ500-51, CT-ICH, HemSeg500, BHSD, INSTANCE2022, RSNA-normal) :contentReference[oaicite:7]{index=7}  
- Pairing and integrity checks
- File standardization
- Slice-thickness (z-spacing) harmonization
- HU sanity correction and QC
- Final releases + full audit reports :contentReference[oaicite:8]{index=8}  

---

## Dataset Composition

### Final Cohort (Preprocessed)

| Branch     | Dataset        | Cases |
|------------|---------------:|------:|
| Normal     | RSNA-normal     | 640 |
| Segmented  | BHSD            | 191 |
| Segmented  | CQ500-51        | 51  |
| Segmented  | CT-ICH          | 75  |
| Segmented  | HemSeg500       | 525 |
| Segmented  | INSTANCE2022    | 100 |
| **Total**  |                | **1,582** |

:contentReference[oaicite:9]{index=9}

### Raw Cohort (Before Preprocessing)

| Branch     | Dataset        | Cases |
|------------|---------------:|------:|
| Normal     | RSNA-normal     | 640 |
| Segmented  | BHSD            | 192 |
| Segmented  | CQ500-51        | 51  |
| Segmented  | CT-ICH          | 75  |
| Segmented  | HemSeg500       | 525 |
| Segmented  | INSTANCE2022    | 100 |
| **Total**  |                | **1,583** |

> One BHSD case was removed during preprocessing due to integrity issues (documented in reports). :contentReference[oaicite:10]{index=10}

---

## Dataset Structure

![Dataset folder layout](CT-Seg1500-Info/figures/Folder%20layout.png)

Each case consists of:
- 3D CT volume
- Corresponding segmentation mask (binary)

---

## Provided Dataset Formats

### 1) NIfTI (`.nii.gz`)
- Standardized 3D volumes
- Consistent extensions
- Preserved geometry (affine/header) :contentReference[oaicite:11]{index=11}

### 2) Raw NumPy (`.npy`)
- Direct conversion from NIfTI
- CT: `float32`
- Mask: `uint8` :contentReference[oaicite:12]{index=12}

### 3) Precomputed NumPy (3-channel)
Per-slice channels:
- Brain window + CLAHE
- Subdural window + CLAHE
- Raw HU mapped to uint8 :contentReference[oaicite:13]{index=13}

Designed for:
- Efficient transformer training
- Reduced CPU/I/O overhead
- Reproducible input formatting :contentReference[oaicite:14]{index=14}

---

## Key Contributions

- Multi-dataset aggregation across heterogeneous CT sources :contentReference[oaicite:15]{index=15}  
- Slice-thickness-aware harmonization (≈5 mm target) :contentReference[oaicite:16]{index=16}  
- HU normalization and sanity correction :contentReference[oaicite:17]{index=17}  
- Precomputed multi-channel representation for deep learning :contentReference[oaicite:18]{index=18}  
- Full raw → preprocessed audit traceability :contentReference[oaicite:19]{index=19}  
- Reproducible preprocessing and dataset construction pipeline :contentReference[oaicite:20]{index=20}  

---

## Evidence & Reports

All transformations are logged under:
```text
CT-Seg1500-Info/reports/
````



The evidence bundle includes:

* Raw vs preprocessed deltas (case-level + dataset-level)
* Slice-thickness statistics and resampling estimates
* Per-case geometry/features (HU, spacing, volume, etc.)
* Dataset summaries (combined + per dataset)
* Conversion and repair logs 

---

## Reproducibility

Full step-by-step instructions (environment + expected inputs/outputs + how to run each script) are in:

➡️ **[instructions.md](instructions.md)**

---

## Repository Layout (current)

```text
CT-Seg1500/
│
├── CT-Seg1500-Info/
│   ├── reports/        # CSV logs, deltas, per-case features, summaries
│   ├── metadata/       # JSON structure summaries
│   └── figures/        # PNG figures used in the paper
│
├── Labels/             # label merge CSVs (optional public release)
├── Splits/             # train/val JSONs (1ch + 3ch)
├── Scripts/            # preprocessing scripts (added/updated over time)
│
├── requirements.txt
├── environment.yml
├── instructions.md
└── README.md
```

---

## Source Datasets

| Dataset                        | Content                          | Role in CT-Seg1500      | Public Status                        | Official Source                                                                                                                    |
| ------------------------------ | -------------------------------- | ----------------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| RSNA ICH (Kaggle)              | DICOM CT + labels                | Normal branch (derived) | Requires Kaggle access               | [https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection) |
| CQ500-51 (Seq-CQ500)           | CT volumes + voxel masks         | Segmented branch        | Redistributable (if license permits) | [https://zenodo.org/records/8063221](https://zenodo.org/records/8063221)                                                           |
| CT-ICH (PhysioNet)             | CT volumes + masks + labels      | Segmented branch        | Restricted (DUA required)            | [https://physionet.org/content/ct-ich/1.3.1/](https://physionet.org/content/ct-ich/1.3.1/)                                         |
| INSTANCE2022 (Grand Challenge) | CT volumes + masks               | Segmented branch        | Restricted (challenge rules)         | [https://instance.grand-challenge.org/](https://instance.grand-challenge.org/)                                                     |
| HemSeg500                      | RSNA-derived scans + voxel masks | Segmented branch        | Derived (depends on RSNA access)     | [https://github.com/songchangwei/3DCT-SD-IVH-ICH](https://github.com/songchangwei/3DCT-SD-IVH-ICH)                                 |
| BHSD (Hugging Face)            | Reconstructed volumes + masks    | Segmented branch        | Redistributable (MIT license)        | [https://huggingface.co/datasets/Wendy-Fly/BHSD](https://huggingface.co/datasets/Wendy-Fly/BHSD)                                   |

⚠️ Users must comply with original dataset licenses before use or redistribution. 

---

## How to cite

GitHub will expose citation metadata via `CITATION.cff` (see the “Cite this repository” box). 
If/when a DOI is minted via Zenodo, the DOI will be added to `CITATION.cff` and (optionally) a README badge.

---

## License

This repository provides:

* code,
* derived reports,
* dataset indices/metadata.

Original datasets remain under their respective licenses. 

---

## Contact

For questions, issues, or collaboration: open a GitHub issue.

```
```
