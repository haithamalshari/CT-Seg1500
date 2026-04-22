
# instructions.md — CT-Seg1500 build & preprocessing guide

This document provides **reproducible, end-to-end instructions** for acquiring the source datasets (where permitted), organizing them into a unified input layout, and running the **CT-Seg1500** preprocessing pipeline using the scripts in this repository.

> **Important**
> - This repository contains **preprocessing code + derived reports/figures/indices**.
> - The **original datasets** are governed by their own access rules and licenses/terms. Ensure compliance before downloading, reconstructing, or redistributing any data.
> - Before running any script, you must **edit paths** (`root_input_dir`, `output_dir`, etc.) to match your environment.

---

## 1) Source datasets and acquisition

### Direct download sources
- **CQ500-51** (segmented subset):  
  https://zenodo.org/records/8063221
- **CT-ICH (v1.3.1)**:  
  https://physionet.org/content/ct-ich/1.3.1/
- **BHSD**:  
  https://huggingface.co/datasets/Wendy-Fly/BHSD
- **INSTANCE2022 (Instance22)**:  
  https://instance.grand-challenge.org/Instance2022/

### Reconstructed/derived sources (require RSNA access)
- **RSNA ICH competition (required upstream source)**:  
  https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/

- **HemSeg500 build notebook (Kaggle)**:  
  https://www.kaggle.com/code/haithamalshari/build-hemseg-500-dataset  
  > Requires RSNA competition access above.

- **RSNA-derived normals build notebook (Kaggle)**:  
  https://www.kaggle.com/code/haithamalshari/rsna-derived-normal-ct-masks  
  > Requires RSNA competition access above.

---

## 2) Before you start

### 2.1 Environment
Use either `requirements.txt` or `environment.yml` provided in the repo.

**Option A — pip**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

**Option B — conda**

```bash
conda env create -f environment.yml
conda activate ctseg1500
```

### 2.2 Hardware & storage

* Building `.nii.gz` outputs is moderate.
* Building **raw NPY + precomputed 3-channel NPY** requires substantial space.

✅ Ensure the final output directory has **at least 150 GB free** before running:

* `5-NPY-and-Precomputed-NPY.py`

---

## 3) Required input folder structure

All datasets must be placed under a single `root_input_dir` with the following layout:

```text
root_input_dir/
  BHSD/
    ct_scans/
    masks/
  CQ500-51/
    ct_scans/
    masks/
  CT-ICH/
    ct_scans/
    masks/
  HemSeg500/
    ct_scans/
    masks/
  Instance22/
    ct_scans/
    masks/
  RSNA-Normal/
    ct_scans/
    masks/
```

### Notes

* Each dataset folder must contain exactly:

  * `ct_scans/` → CT volumes (NIfTI)
  * `masks/` → segmentation masks (NIfTI)
* Filenames must be consistent per case:

  * a CT volume and its mask must share the same stem (e.g., `case123.nii.gz` in both folders).
* The CQ500-51 release often needs reorganization to match this format (see Script 1).

---

## 4) Pipeline overview (what each script does)

The CT-Seg1500 pipeline is implemented as **five scripts**, designed to be run in order.

### Script 1 — `1-Reorganizing-CQ500-51-files.py`

**Purpose:** Normalize/reorganize CQ500-51 into the required `ct_scans/` and `masks/` layout so it can be merged consistently.

**What you do:**

1. Download/extract CQ500-51.
2. Set:

   * `root_input_dir` (where the raw CQ500-51 files are)
   * `output_dir` (where the reorganized CQ500-51 should be written)
3. Run the script.

**Expected output:**

```text
root_input_dir/CQ500-51/
  ct_scans/
  masks/
```

---

### Script 2 — `2-Merging_and_Case_Index_Raw.py`

**Purpose:** Merge all source datasets into the unified CT-Seg1500 cohort and generate the **raw case index**.

**Inputs:**

* `root_input_dir/` containing:
  `BHSD, CQ500-51, CT-ICH, HemSeg500, Instance22, RSNA-Normal`
  (each with `ct_scans/` and `masks/`)

**Outputs:**

* A merged dataset folder (raw form)
* `case_index_raw.csv`
* Merge logs / summary CSVs (as configured)

**What it produces conceptually:**

* A unified cohort with consistent case naming
* A machine-readable table mapping case IDs to paths and dataset source

---

### Script 3 — `3-Preprocessing-CT-Seg1500.py`

**Purpose:** Perform preprocessing + harmonization and produce **audit-grade preprocessing reports**.

This includes (as implemented in your pipeline):

* CT/mask pairing validation
* filename/extension standardization (e.g., `.nii.gz`)
* slice-thickness-aware decisions (e.g., target ≈ 5 mm, selective resampling rules)
* HU sanity correction where applicable
* structured reports of actions taken and final retained cases

**Outputs:**

* Preprocessed dataset output folder
* `case_index_preprocessed.csv`
* preprocessing logs and summaries (CSV)
* thickness/geometry reports (CSV)
* metadata summaries (JSON)

---

### Script 4 — `4-Stats_Figures_code_01.py`

**Purpose:** Generate the **statistics tables** and **paper-ready figures** from the raw vs preprocessed indices and geometry reports.

**Outputs (typical):**

* CSV tables (counts and deltas)
* PNG figures (distributions and summaries)
* “Info bundle” used for documentation/paper artifacts

---

### Script 5 — `5-NPY-and-Precomputed-NPY.py`

**Purpose:** Export the final dataset into two transformer-ready NumPy forms:

1. **Raw NPY**

* CT: typically `float32`
* Mask: typically `uint8`

2. **Precomputed 3-channel NPY**
   Per-slice channels:

* brain window + CLAHE
* subdural window + CLAHE
* raw HU mapped to `uint8`

**Important: disk requirement**

* Ensure `output_dir` has **≥ 150 GB free**.

**Outputs:**

* `CT-Seg-1500-NPY/` (raw NPY)
* `CT-Seg-1500-Precomputed/` (precomputed 3ch NPY)
* build reports / audits (CSV)

---

## 5) How to run (recommended sequence)

### 5.1 Set common paths

Before running, open each script and set:

* `root_input_dir` → where the organized source dataset folders live
* `output_dir` → where outputs should be written

Recommended:

* Keep `root_input_dir` on a fast disk (SSD if possible)
* Write outputs to a disk with sufficient space

---

### 5.2 Run scripts in order

#### Step 1 — CQ500-51 reorganization (only needed once)

```bash
python Scripts/1-Reorganizing-CQ500-51-files.py
```

#### Step 2 — merge + raw case index

```bash
python Scripts/2-Merging_and_Case_Index_Raw.py
```

#### Step 3 — preprocessing + reports

```bash
python Scripts/3-Preprocessing-CT-Seg1500.py
```

#### Step 4 — stats + figures

```bash
python Scripts/4-Stats_Figures_code_01.py
```

#### Step 5 — NPY exports (requires ≥150GB free)

```bash
python Scripts/5-NPY-and-Precomputed-NPY.py
```

---

## 6) Outputs you should expect

After running the full pipeline, you will have:

* **Preprocessed NIfTI dataset** (`.nii.gz`)
* **Raw NPY dataset** (`.npy`)
* **Precomputed 3-channel NPY dataset** (`.npy`)
* **Full traceability artifacts**, including:

  * pairing reports
  * preprocessing action logs
  * geometry & slice thickness summaries
  * raw vs preprocessed deltas
  * final indices
  * paper-ready figures

---

## 7) Troubleshooting notes

### Path errors

Most failures come from incorrect directory variables. Double-check:

* `root_input_dir` points to the folder that contains `BHSD/`, `CQ500-51/`, etc.
* Every dataset has both `ct_scans/` and `masks/`

### Pairing mismatches

If a dataset has CTs without masks (or mismatched filenames), Script 2/3 will report it in pairing/audit CSVs. Fix the dataset folder before re-running.

### Storage exhaustion during NPY export

If the NPY/precomputed build fails mid-run, move the output directory to a drive with ≥150GB free space and re-run Script 5.

---

## 8) Citation & reproducibility statement (for paper/repo)

This repository provides:

* preprocessing scripts,
* derived indices/logs/reports,
* and paper figures.

It does **not** distribute the upstream datasets directly. For any third-party dataset, users must obtain access via the official links listed above and comply with each dataset’s terms.


