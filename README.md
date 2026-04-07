# CT-Seg1500

CT-Seg1500 is a harmonized, slice-thickness-aware, transformer-ready multi-source brain CT cohort and preprocessing framework for intracranial hemorrhage segmentation and diagnosis.

This repository contains:
- reproducible preprocessing scripts,
- full QC/audit reports (CSV/JSON),
- dataset statistics tables and figures,
- and split/label artifacts used in experiments.

> **Transformer-ready** in this project means: standardized 3D NIfTI volumes and two fast-loading NumPy exports (raw NPY and precomputed multi-channel NPY) designed for efficient training pipelines.

---

## Dataset structure (final)

The final dataset is organized into two branches:

```text
CT-Seg1500/
├── Segmented Scans/
│   ├── ct_scans/
│   └── masks/
└── Normal Scans/
    ├── ct_scans/
    └── qmasks/
```


**Final cohort size**
- Segmented (abnormal) scans: **942 CT + 942 masks**
- Normal scans: **640 CT + 640 masks**
- Total: **1,582 scans** (each scan has a CT + a mask)

---

## Provided dataset forms

The dataset is provided (or prepared for release) in three synchronized forms:

1) **NIfTI master (`.nii.gz`)**
   - Unified 3D volumes
   - Standardized extensions
   - Geometry preserved (affine/header where available)

2) **Raw NumPy (`.npy`)**
   - Direct `.nii.gz → .npy` conversion for fast loading
   - Typical dtypes: CT `float32`, mask `uint8`

3) **Precomputed NumPy (`.npy`, uint8, 3 channels per slice)**
   - Per-slice channels:
     - brain window + CLAHE
     - subdural window + CLAHE
     - raw HU mapped to uint8
   - Masks remain binary volumes

---

## What preprocessing was applied (high-level)

Across merged sources, the pipeline includes:
- duplicate removal (HemSeg500 vs previous merged abnormal cohort),
- CT/mask pairing validation,
- HU sanity correction / normalization checks,
- selective slice-thickness harmonization:
  - resample scans with very small z-spacing (e.g., < 4 mm) to ~5 mm,
  - keep thicker scans rather than synthesizing intermediate slices,
- final standardization to `.nii.gz`,
- export to raw NPY and precomputed NPY for transformer training efficiency.

All actions are logged in `reports/` as CSV/JSON artifacts.

---

## Repository layout

- `scripts/`  
  Core preprocessing scripts (standardize NIfTI, build raw NPY, build precomputed NPY).

- `reports/preprocessing/`  
  QC/audit logs: duplicate reports, missing-pair reports, fix/resample action logs, NIfTI standardization audits, and export build reports.

- `reports/dataset_stats/`  
  Case index files, raw-vs-preprocessed deltas, per-dataset summaries, and the figures/tables used in the dataset paper.

- `docs/dataset-paper-assets/`  
  Paper-ready figures and tables (mirrors items from dataset_stats in a cleaner layout for writing).

- `reports/labels/` and `splits/`  
  Diagnosis label merges (presence/extent) and frozen train/val split indices.

---

## How to cite

GitHub will expose citation metadata via `CITATION.cff` (see the “Cite this repository” box).
If/when a DOI is minted via Zenodo, the DOI will be added to `CITATION.cff` and the README badge.

---

## License and data availability

This repository contains **code + derived reports** produced from multiple public sources.
Upstream datasets may have different licensing/redistribution requirements.

This repository is intended to provide:
- preprocessing code,
- cohort definition + audit logs,
- and reproducible build artifacts.

Please consult upstream dataset terms before redistributing any source data.
(See `docs/methodology/` for details and recommended wording for the dataset paper.)

---

## Contact

For questions or collaboration, open an issue in this repository.
