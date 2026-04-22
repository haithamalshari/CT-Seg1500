# ============================================================
# Cell 1 — Imports
# ============================================================
from pathlib import Path
import os
import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
import cv2

from tqdm.auto import tqdm

# ============================================================
# Cell 2 — Configuration
# ============================================================
def ensure_valid_cwd():
    try:
        _ = os.getcwd()
    except FileNotFoundError:
        os.chdir(str(Path.home()))

def normalize_dir(path_str: str) -> Path:
    p = os.path.expandvars(os.path.expanduser(path_str.strip()))
    return Path(p).resolve()

ensure_valid_cwd()

# ------------------------------------------------------------------
# INPUT dataset:
# CT-Seg1500/
#   Segmented Scans/ct_scans, masks
#   Normal Scans/ct_scans, masks
# ------------------------------------------------------------------
SRC_ROOT = normalize_dir("/path/to/CT-Seg1500")

# ------------------------------------------------------------------
# OUTPUT folders
# ------------------------------------------------------------------
OUT_NPY_ROOT = normalize_dir("/path/to/T-Seg1500-NPY")
OUT_PRECOMP_ROOT = normalize_dir("/path/to/CT-Seg1500-NPY-Precomputd")

# reports
REPORT_DIR = normalize_dir("/path/to/CT-Seg1500-NPY-reports")

TARGET_BRANCHES = ["Segmented Scans", "Normal Scans"]
TARGET_SUBDIRS = ["ct_scans", "masks"]

# If True, remove existing output folders before rebuilding
RESET_OUTPUTS = True

# Zip outputs at the end
MAKE_ZIPS = False #True

print("SRC_ROOT        =", SRC_ROOT)
print("OUT_NPY_ROOT    =", OUT_NPY_ROOT)
print("OUT_PRECOMP_ROOT=", OUT_PRECOMP_ROOT)
print("REPORT_DIR      =", REPORT_DIR)

print("===========================================================================")
print("[Storage note] Expected local free space: CT-Seg1500-NPY ≈75 GB; CT-Seg1500-NPY-Precomputd ≈60 GB; keep extra headroom for reports/zips if enabled.")
print("===========================================================================")

# ============================================================
# Cell 3 — Validate input and create outputs
# ============================================================
if not SRC_ROOT.exists():
    raise FileNotFoundError(f"Missing SRC_ROOT: {SRC_ROOT}")

for branch in TARGET_BRANCHES:
    for sub in TARGET_SUBDIRS:
        p = SRC_ROOT / branch / sub
        if not p.exists():
            raise FileNotFoundError(f"Missing required folder: {p}")


# Safe reset/delete helpers for local + external drives
# Handles macOS AppleDouble files like ._ct_scans

def _on_rm_error(func, path, exc_info):
    """
    Robust rmtree error handler.
    - Ignore FileNotFoundError
    - Try chmod for permission errors
    """
    exc = exc_info[1]

    if isinstance(exc, FileNotFoundError):
        return

    try:
        os.chmod(path, os.stat.S_IWRITE)
        func(path)
    except FileNotFoundError:
        return
    except Exception:
        raise

def remove_macos_junk(root: Path):
    """
    Remove macOS metadata files from a folder tree:
      .DS_Store
      ._*
    """
    root = Path(root)
    if not root.exists():
        return

    for p in root.rglob("*"):
        try:
            if p.is_file() and (p.name == ".DS_Store" or p.name.startswith("._")):
                p.unlink(missing_ok=True)
        except Exception:
            pass

def safe_rmtree(path: Path):
    """
    Safer folder removal for external drives.
    """
    path = Path(path)

    if not path.exists():
        return

    # safety guard: never delete filesystem roots
    if path.parent == path:
        raise ValueError(f"Refusing to delete filesystem root: {path}")

    remove_macos_junk(path)

    try:
        shutil.rmtree(path, onerror=_on_rm_error)
    except FileNotFoundError:
        pass

def reset_dir(path: Path):
    """
    Delete folder if exists, then recreate it.
    """
    path = Path(path)

    if path.exists():
        safe_rmtree(path)

    path.mkdir(parents=True, exist_ok=True)

if RESET_OUTPUTS:
    reset_dir(OUT_NPY_ROOT)
    reset_dir(OUT_PRECOMP_ROOT)
    reset_dir(REPORT_DIR)
else:
    OUT_NPY_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_PRECOMP_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

for root in [OUT_NPY_ROOT, OUT_PRECOMP_ROOT]:
    for branch in TARGET_BRANCHES:
        for sub in TARGET_SUBDIRS:
            (root / branch / sub).mkdir(parents=True, exist_ok=True)

print("Input validated and output folders created.")

# ============================================================
# Cell 4 — File helpers
# ============================================================
def is_nifti_file(path: Path) -> bool:
    name = path.name
    if name.startswith("."):
        return False
    low = name.lower()
    return low.endswith(".nii") or low.endswith(".nii.gz")

def stem_nii(path: Path) -> str:
    name = path.name
    low = name.lower()
    if low.endswith(".nii.gz"):
        return name[:-7]
    if low.endswith(".nii"):
        return name[:-4]
    return path.stem

def list_nii_files(folder: Path):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and is_nifti_file(p)])

def save_npy(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr)

def count_npy(folder: Path):
    return len(list(Path(folder).glob("*.npy")))

def zip_folder(src_dir: Path, zip_path: Path):
    src_dir = Path(src_dir)
    zip_path = Path(zip_path)

    if zip_path.exists():
        zip_path.unlink()

    subprocess.check_call([
        "bash", "-lc",
        f"cd '{src_dir.parent}' && zip -qr '{zip_path.name}' '{src_dir.name}'"
    ])

    print("Created zip:", zip_path)

# ============================================================
# Cell 5 — Robust NIfTI loader
# ============================================================
def load_volume_zyx(path: Path):
    """
    Load NIfTI as NumPy array in [D,H,W] = [Z,Y,X] order.

    Returns:
      arr, backend
    backend:
      - "sitk"
      - "nibabel_fallback"
    """
    path = Path(path)

    # 1) SimpleITK path
    try:
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img)  # [Z,Y,X]
        return arr, "sitk"

    except Exception:
        # 2) nibabel fallback
        nii = nib.load(str(path))
        arr = np.asanyarray(nii.dataobj)

        if arr.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {arr.shape} for {path}")

        # nibabel commonly exposes [X,Y,Z], convert to [Z,Y,X]
        arr = np.transpose(arr, (2, 1, 0))
        return arr, "nibabel_fallback"
    
# ============================================================
# Cell 6 — Windowing + CLAHE helpers
# ============================================================
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def apply_window_u8(img_hu: np.ndarray, center: float, width: float) -> np.ndarray:
    low = center - width / 2.0
    high = center + width / 2.0

    x = np.clip(img_hu, low, high)
    x = (x - low) / max(1e-6, high - low)
    return (x * 255.0).astype(np.uint8)

def apply_clahe_u8(x_uint8: np.ndarray) -> np.ndarray:
    return _CLAHE.apply(x_uint8)

def raw_to_u8(img_hu: np.ndarray, lo: float = -1000, hi: float = 3000) -> np.ndarray:
    x = np.clip(img_hu, lo, hi)
    x = (x - lo) / max(1e-6, hi - lo)
    return (x * 255.0).astype(np.uint8)

def make_3ch_precomputed(arr_zyx: np.ndarray) -> np.ndarray:
    """
    Input:
      arr_zyx: CT volume [D,H,W], float-like HU

    Output:
      X: [D,3,H,W], uint8
    """
    arr = arr_zyx.astype(np.float32, copy=False)
    D, H, W = arr.shape

    X = np.zeros((D, 3, H, W), dtype=np.uint8)

    for z in range(D):
        hu = arr[z]

        brain = apply_clahe_u8(apply_window_u8(hu, center=40, width=80))
        subdural = apply_clahe_u8(apply_window_u8(hu, center=80, width=200))
        raw = raw_to_u8(hu, lo=-1000, hi=3000)

        X[z, 0] = brain
        X[z, 1] = subdural
        X[z, 2] = raw

    return X

# ============================================================
# Cell 7 — Build paired file table
# ============================================================
pair_rows = []

for branch in TARGET_BRANCHES:
    ct_dir = SRC_ROOT / branch / "ct_scans"
    mask_dir = SRC_ROOT / branch / "masks"

    ct_map = {stem_nii(p): p for p in list_nii_files(ct_dir)}
    mask_map = {stem_nii(p): p for p in list_nii_files(mask_dir)}

    all_stems = sorted(set(ct_map) | set(mask_map))

    for stem in all_stems:
        pair_rows.append({
            "branch": branch,
            "case_id": stem,
            "ct_path": str(ct_map[stem]) if stem in ct_map else None,
            "mask_path": str(mask_map[stem]) if stem in mask_map else None,
            "ct_exists": stem in ct_map,
            "mask_exists": stem in mask_map,
            "paired": stem in ct_map and stem in mask_map,
        })

pairs_df = pd.DataFrame(pair_rows)
pairs_csv = REPORT_DIR / "ct_seg1500_pair_report.csv"
pairs_df.to_csv(pairs_csv, index=False)

print("Saved:", pairs_csv)
print(pairs_df["paired"].value_counts(dropna=False).to_string())

unpaired = pairs_df[pairs_df["paired"] == False].copy()
if len(unpaired) > 0:
    print("\nUnpaired cases:")
    print(unpaired.to_string(index=False))
else:
    print("\nAll cases are paired.")


# ============================================================
# Cell 8 — Build CT-Seg1500-NPY
# ============================================================
npy_rows = []

paired_df = pairs_df[pairs_df["paired"] == True].copy()

for row in tqdm(paired_df.to_dict("records"), desc="Building raw NPY", unit="case"):
    branch = row["branch"]
    case_id = row["case_id"]
    ct_path = Path(row["ct_path"])
    mask_path = Path(row["mask_path"])

    out_ct = OUT_NPY_ROOT / branch / "ct_scans" / f"{case_id}.npy"
    out_mask = OUT_NPY_ROOT / branch / "masks" / f"{case_id}.npy"

    try:
        ct_arr, ct_backend = load_volume_zyx(ct_path)
        mask_arr, mask_backend = load_volume_zyx(mask_path)

        ct_arr = ct_arr.astype(np.float32)
        mask_arr = (mask_arr > 0).astype(np.uint8)

        save_npy(out_ct, ct_arr)
        save_npy(out_mask, mask_arr)

        npy_rows.append({
            "branch": branch,
            "case_id": case_id,
            "status": "ok",
            "ct_backend": ct_backend,
            "mask_backend": mask_backend,
            "ct_shape": str(ct_arr.shape),
            "mask_shape": str(mask_arr.shape),
            "ct_dtype": str(ct_arr.dtype),
            "mask_dtype": str(mask_arr.dtype),
            "ct_out": str(out_ct),
            "mask_out": str(out_mask),
            "error": None,
        })

    except Exception as e:
        npy_rows.append({
            "branch": branch,
            "case_id": case_id,
            "status": "failed",
            "ct_backend": None,
            "mask_backend": None,
            "ct_shape": None,
            "mask_shape": None,
            "ct_dtype": None,
            "mask_dtype": None,
            "ct_out": str(out_ct),
            "mask_out": str(out_mask),
            "error": repr(e),
        })

npy_report = pd.DataFrame(npy_rows)
npy_report_csv = REPORT_DIR / "ct_seg1500_npy_build_report.csv"
npy_report.to_csv(npy_report_csv, index=False)

print("Saved:", npy_report_csv)
print(npy_report["status"].value_counts(dropna=False).to_string())


# ============================================================
# Cell 9 — Build CT-Seg1500-NPY-Precomputd
# ============================================================
precomp_rows = []

for row in tqdm(paired_df.to_dict("records"), desc="Building precomputed NPY", unit="case"):
    branch = row["branch"]
    case_id = row["case_id"]
    ct_path = Path(row["ct_path"])
    mask_path = Path(row["mask_path"])

    out_ct = OUT_PRECOMP_ROOT / branch / "ct_scans" / f"{case_id}.npy"
    out_mask = OUT_PRECOMP_ROOT / branch / "masks" / f"{case_id}.npy"

    try:
        ct_arr, ct_backend = load_volume_zyx(ct_path)
        mask_arr, mask_backend = load_volume_zyx(mask_path)

        X = make_3ch_precomputed(ct_arr)
        Y = (mask_arr > 0).astype(np.uint8)

        save_npy(out_ct, X)
        save_npy(out_mask, Y)

        precomp_rows.append({
            "branch": branch,
            "case_id": case_id,
            "status": "ok",
            "ct_backend": ct_backend,
            "mask_backend": mask_backend,
            "ct_shape": str(X.shape),
            "mask_shape": str(Y.shape),
            "ct_dtype": str(X.dtype),
            "mask_dtype": str(Y.dtype),
            "ct_out": str(out_ct),
            "mask_out": str(out_mask),
            "error": None,
        })

    except Exception as e:
        precomp_rows.append({
            "branch": branch,
            "case_id": case_id,
            "status": "failed",
            "ct_backend": None,
            "mask_backend": None,
            "ct_shape": None,
            "mask_shape": None,
            "ct_dtype": None,
            "mask_dtype": None,
            "ct_out": str(out_ct),
            "mask_out": str(out_mask),
            "error": repr(e),
        })

precomp_report = pd.DataFrame(precomp_rows)
precomp_report_csv = REPORT_DIR / "ct_seg1500_precomputed_build_report.csv"
precomp_report.to_csv(precomp_report_csv, index=False)

print("Saved:", precomp_report_csv)
print(precomp_report["status"].value_counts(dropna=False).to_string())


# ============================================================
# Cell 10 — Audit output counts
# ============================================================
audit_rows = []

for dataset_name, root in [
    ("NPY", OUT_NPY_ROOT),
    ("Precomputed", OUT_PRECOMP_ROOT),
]:
    for branch in TARGET_BRANCHES:
        for sub in TARGET_SUBDIRS:
            folder = root / branch / sub
            audit_rows.append({
                "dataset": dataset_name,
                "branch": branch,
                "subdir": sub,
                "n_npy_files": count_npy(folder),
            })

audit_df = pd.DataFrame(audit_rows)
audit_csv = REPORT_DIR / "ct_seg1500_npy_output_audit.csv"
audit_df.to_csv(audit_csv, index=False)

print("Saved:", audit_csv)
print(audit_df.to_string(index=False))


# ============================================================
# Cell 11 — Create compact summary JSON
# ============================================================
summary = {
    "source_root": str(SRC_ROOT),
    "output_npy_root": str(OUT_NPY_ROOT),
    "output_precomputed_root": str(OUT_PRECOMP_ROOT),
    "n_pairs_total": int(len(paired_df)),
    "n_unpaired": int((pairs_df["paired"] == False).sum()),
    "npy": {
        "ok": int((npy_report["status"] == "ok").sum()),
        "failed": int((npy_report["status"] == "failed").sum()),
        "ct_sitk_reads": int((npy_report["ct_backend"] == "sitk").sum()),
        "ct_nibabel_fallback_reads": int((npy_report["ct_backend"] == "nibabel_fallback").sum()),
        "mask_sitk_reads": int((npy_report["mask_backend"] == "sitk").sum()),
        "mask_nibabel_fallback_reads": int((npy_report["mask_backend"] == "nibabel_fallback").sum()),
    },
    "precomputed": {
        "ok": int((precomp_report["status"] == "ok").sum()),
        "failed": int((precomp_report["status"] == "failed").sum()),
        "ct_sitk_reads": int((precomp_report["ct_backend"] == "sitk").sum()),
        "ct_nibabel_fallback_reads": int((precomp_report["ct_backend"] == "nibabel_fallback").sum()),
        "mask_sitk_reads": int((precomp_report["mask_backend"] == "sitk").sum()),
        "mask_nibabel_fallback_reads": int((precomp_report["mask_backend"] == "nibabel_fallback").sum()),
    },
    "audit": audit_df.to_dict(orient="records"),
}

summary_json = REPORT_DIR / "ct_seg1500_npy_summary.json"
with open(summary_json, "w") as f:
    json.dump(summary, f, indent=2)

print("Saved:", summary_json)
print(json.dumps(summary, indent=2))

# ============================================================
# Cell 12 — Zip outputs
# ============================================================
if MAKE_ZIPS:
    npy_zip = OUT_NPY_ROOT.parent / f"{OUT_NPY_ROOT.name}.zip"
    precomp_zip = OUT_PRECOMP_ROOT.parent / f"{OUT_PRECOMP_ROOT.name}.zip"
    report_zip = REPORT_DIR.parent / f"{REPORT_DIR.name}.zip"

    zip_folder(OUT_NPY_ROOT, npy_zip)
    zip_folder(OUT_PRECOMP_ROOT, precomp_zip)
    zip_folder(REPORT_DIR, report_zip)

    print("\nCreated:")
    print(npy_zip)
    print(precomp_zip)
    print(report_zip)

    print("\nSizes:")
    print("NPY zip GB       :", round(npy_zip.stat().st_size / (1024**3), 3))
    print("Precomputed GB   :", round(precomp_zip.stat().st_size / (1024**3), 3))
    print("Reports zip MB   :", round(report_zip.stat().st_size / (1024**2), 3))
else:
    print("MAKE_ZIPS=False, skipping zipping.")