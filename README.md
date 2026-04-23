# CT-Seg1500

**CT-Seg1500** is a slice-thickness-aware harmonization and preprocessing framework for multi-source brain CT in transformer-based intracranial hemorrhage (ICH) segmentation and diagnosis.

Due to licensing and legal constraints affecting most source datasets, this repository releases the **framework, metadata, reports, and reconstruction pipeline**, rather than redistributing the final derived dataset itself.

---

## TL;DR

* **1,582 CT volumes** in the built cohort (**942 segmented + 640 normal**)
* Aggregated from **5 public datasets + RSNA-derived normals**
* **Slice-thickness-aware harmonization** (~5 mm target)
* **Standardized HU handling** and quality control
* Framework generates three output formats:

  * **Compressed NIfTI (.nii.gz)**
  * **Raw NumPy (.npy)**
  * **Precomputed three-channel NumPy**
* **Audit trail, reproducible case index, and build pipeline**
* Due to source-dataset licensing constraints, this repository releases the **framework and reconstruction pipeline**, not the redistributed final dataset

---

## Overview Pipeline

![Preprocessing Pipeline](CT-Seg1500/CT-Seg1500-Info/figures/Flow%20Diagram%2001.png)

The pipeline integrates multiple datasets into a unified representation:

- Multi-source merge (CQ500-51, CT-ICH, HemSeg500, BHSD, INSTANCE2022, RSNA-normal)
- Pairing and integrity checks
- File standardization
- Slice-thickness (z-spacing) harmonization
- HU sanity correction and QC
- Final releases + full audit reports

---

## Dataset Composition

### Final Cohort (Preprocessed)

| Branch     | Dataset        | Cases |
|------------|---------------|------|
| Normal     | RSNA-normal   | 640  |
| Segmented  | BHSD          | 191  |
| Segmented  | CQ500-51      | 51   |
| Segmented  | CT-ICH        | 75   |
| Segmented  | HemSeg500     | 525  |
| Segmented  | INSTANCE2022  | 100  |
| **Total**  |               | **1,582** |

---

### Raw Cohort (Before Preprocessing)

| Branch     | Dataset        | Cases |
|------------|---------------|------|
| Normal     | RSNA-normal   | 640  |
| Segmented  | BHSD          | 192  |
| Segmented  | CQ500-51      | 51   |
| Segmented  | CT-ICH        | 75   |
| Segmented  | HemSeg500     | 525  |
| Segmented  | INSTANCE2022  | 100  |
| **Total**  |               | **1,583** |

> One BHSD case was removed during preprocessing due to integrity issues (fully documented in reports).

---

## Dataset Structure

![Dataset folder layout](CT-Seg1500/CT-Seg1500-Info/figures/Folder%20layout.png)

Each case consists of:
- 3D CT volume
- Corresponding segmentation mask (binary)

---

## Provided Dataset Formats

### 1) NIfTI (.nii.gz)
- Standardized 3D volumes
- Consistent extensions
- Preserved geometry (affine/header)

### 2) Raw NumPy (.npy)
- Direct conversion from NIfTI
- CT: `float32`
- Mask: `uint8`

### 3) Precomputed NumPy (3-channel)
Per-slice channels:
- Brain window + CLAHE
- Subdural window + CLAHE
- Raw HU mapped to uint8

Designed for:
- Efficient transformer training
- Reduced I/O bottlenecks

---

## Key Contributions

- Multi-dataset aggregation across heterogeneous CT sources
- Slice-thickness-aware harmonization (~5 mm)
- HU normalization and sanity correction
- Precomputed multi-channel representation for deep learning
- Full raw → preprocessed audit traceability
- Reproducible preprocessing and dataset construction pipeline

---

## Preprocessing Pipeline (Summary)

The preprocessing pipeline includes:

- Merging
- CT–mask pairing validation
- Standardization to `.nii.gz`
- HU sanity checks and normalization
- Slice-thickness policy:
  - Resample thin slices (<4 mm) → ~5 mm
  - Preserve thicker scans (no artificial interpolation)
- Export to NPY formats

All transformations are logged in:
```text
CT-Seg1500-Info/reports/
```


---

## Evidence & Reports

The repository includes a complete audit and validation suite:

### Available Reports
- Raw vs preprocessed deltas (case-level and dataset-level)
- Slice-thickness statistics and resampling estimates
- Per-case features (HU, spacing, volume)
- Dataset summaries (combined and per dataset)
- Conversion and repair logs

### Key Findings
- CQ500 median slices reduced from 244 → 32
- ~90% of CQ500 required resampling
- BHSD minimally affected (~10%)
- Final dataset fully standardized to `.nii.gz`

---

## Repository Layout
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

| Dataset | Content | Role in CT-Seg1500 | Public Status | Official Source |
|--------|--------|-------------------|--------------|----------------|
| RSNA ICH (Kaggle) | DICOM CT + labels | Normal branch (derived) | Requires Kaggle access | https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection |
| CQ500-51 (Seq-CQ500) | CT volumes + voxel masks | Segmented branch | Redistributable (if license permits) | https://zenodo.org/records/8063221 |
| CT-ICH (PhysioNet) | CT volumes + masks + labels | Segmented branch | Restricted (DUA required) | https://physionet.org/content/ct-ich/1.3.1/ |
| INSTANCE2022 (Grand Challenge) | CT volumes + masks | Segmented branch | Restricted (challenge rules) | https://instance.grand-challenge.org/ |
| HemSeg500 | RSNA-derived scans + voxel masks | Segmented branch | Derived (depends on RSNA access) | https://github.com/songchangwei/3DCT-SD-IVH-ICH |
| BHSD (Hugging Face) | Reconstructed volumes + masks | Segmented branch | Redistributable (MIT license) | https://huggingface.co/datasets/Wendy-Fly/BHSD |

⚠️ Users must comply with original dataset licenses before use or redistribution.

---

## Reproducibility

Full step-by-step instructions (environment + expected inputs/outputs + how to run each script) are in:

➡️ **[instructions.md](instructions.md)**

---

## Outputs:

- NIfTI volumes
- Raw NPY
- Precomputed NPY

## How to cite

GitHub will expose citation metadata via `CITATION.cff` (see the “Cite this repository” box).
If/when a DOI is minted via Zenodo, the DOI will be added to `CITATION.cff` and the README badge.

## License

This repository provides:

- code,
- derived reports,
- dataset indices.

Original datasets remain under their respective licenses.


## Contact

For questions, issues, or collaboration:

Open a GitHub issue

