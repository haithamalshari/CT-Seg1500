# ============================================================
# Cell 1 — Imports
# ============================================================
from pathlib import Path
import os
import json
import shutil
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from tqdm.auto import tqdm


# ============================================================
# Cell 2 — Configuration
# ============================================================
from pathlib import Path
import os

def ensure_valid_cwd():
    try:
        _ = os.getcwd()
    except FileNotFoundError:
        os.chdir(str(Path.home()))

def normalize_dir(path_str: str) -> Path:
    p = os.path.expandvars(os.path.expanduser(path_str.strip()))
    return Path(p).resolve()

ensure_valid_cwd()

# INPUT: merged raw dataset with branch layout
MERGED_RAW_ROOT = normalize_dir("/path/to/CT-Seg1500-Raw")

# OUTPUT: preprocessed dataset with same branch layout
PREPROCESSED_ROOT = normalize_dir("/path/to/CT-Seg1500-Preprocessed")

# processing policy
TARGET_DZ_MM = 5.0
RESAMPLE_BELOW_MM = 4.0

# only obvious offset correction
ENABLE_HU_OFFSET_FIX = True

# report paths
CASE_INDEX_RAW_CSV = MERGED_RAW_ROOT / "case_index_raw.csv"
CASE_INDEX_PREPROCESSED_CSV = PREPROCESSED_ROOT / "case_index_preprocessed.csv"
ACTION_LOG_CSV = PREPROCESSED_ROOT / "preprocess_action_log.csv"
THICKNESS_REPORT_CSV = PREPROCESSED_ROOT / "thickness_report.csv"
SUMMARY_JSON = PREPROCESSED_ROOT / "preprocess_summary.json"

print("MERGED_RAW_ROOT   =", MERGED_RAW_ROOT)
print("PREPROCESSED_ROOT =", PREPROCESSED_ROOT)

# ============================================================
# Cell 3 — Validate input and create output folders
# ============================================================
if not MERGED_RAW_ROOT.exists():
    raise FileNotFoundError(f"Missing MERGED_RAW_ROOT: {MERGED_RAW_ROOT}")

if not CASE_INDEX_RAW_CSV.exists():
    raise FileNotFoundError(f"Missing case_index_raw.csv: {CASE_INDEX_RAW_CSV}")

RAW_SEG_CT_DIR = MERGED_RAW_ROOT / "Segmented Scans" / "ct_scans"
RAW_SEG_MASK_DIR = MERGED_RAW_ROOT / "Segmented Scans" / "masks"

RAW_NOR_CT_DIR = MERGED_RAW_ROOT / "Normal Scans" / "ct_scans"
RAW_NOR_MASK_DIR = MERGED_RAW_ROOT / "Normal Scans" / "masks"

required_input_dirs = [
    RAW_SEG_CT_DIR,
    RAW_SEG_MASK_DIR,
    RAW_NOR_CT_DIR,
    RAW_NOR_MASK_DIR,
]

for d in required_input_dirs:
    if not d.exists():
        raise FileNotFoundError(f"Missing required input folder: {d}")

PREPROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

PREP_SEG_CT_DIR = PREPROCESSED_ROOT / "Segmented Scans" / "ct_scans"
PREP_SEG_MASK_DIR = PREPROCESSED_ROOT / "Segmented Scans" / "masks"

PREP_NOR_CT_DIR = PREPROCESSED_ROOT / "Normal Scans" / "ct_scans"
PREP_NOR_MASK_DIR = PREPROCESSED_ROOT / "Normal Scans" / "masks"

required_output_dirs = [
    PREP_SEG_CT_DIR,
    PREP_SEG_MASK_DIR,
    PREP_NOR_CT_DIR,
    PREP_NOR_MASK_DIR,
]

for d in required_output_dirs:
    d.mkdir(parents=True, exist_ok=True)

print("Input validated and branch-based output folders created.")

# ============================================================
# Cell 4 — Basic filename and path helpers
# ============================================================
def is_nifti_file(path: Path) -> bool:
    name = path.name
    if name.startswith("."):
        return False
    low = name.lower()
    return low.endswith(".nii") or low.endswith(".nii.gz")


def split_nii_name(filename: str) -> Tuple[str, str]:
    low = filename.lower()
    if low.endswith(".nii.gz"):
        return filename[:-7], ".nii.gz"
    if low.endswith(".nii"):
        return filename[:-4], ".nii"
    raise ValueError(f"Not a NIfTI file: {filename}")


def to_niigz_name(case_id: str) -> str:
    return f"{case_id}.nii.gz"


def get_branch_folder(branch: str) -> str:
    """
    Maps internal branch names to final folder names.
    """
    if branch == "segmented":
        return "Segmented Scans"
    if branch == "normal":
        return "Normal Scans"
    raise ValueError(f"Unknown branch: {branch}")


def resolve_input_path(path_value: str) -> Path:
    """
    Supports both:
      1) absolute paths
      2) short local paths from case_index_raw.csv, e.g.
         Segmented Scans/ct_scans/case_id.nii.gz
    """
    p = Path(str(path_value))

    if p.is_absolute():
        return p

    return MERGED_RAW_ROOT / p


def get_output_dirs_for_branch(branch: str) -> Tuple[Path, Path]:
    """
    Returns:
      ct_output_dir, mask_output_dir
    """
    branch_folder = get_branch_folder(branch)
    ct_out_dir = PREPROCESSED_ROOT / branch_folder / "ct_scans"
    mask_out_dir = PREPROCESSED_ROOT / branch_folder / "masks"
    return ct_out_dir, mask_out_dir


def make_preprocessed_relative_paths(branch: str, case_id: str) -> Tuple[str, str]:
    """
    Returns short portable paths for case_index_preprocessed.csv.
    """
    branch_folder = get_branch_folder(branch)
    filename = to_niigz_name(case_id)

    ct_rel_path = f"{branch_folder}/ct_scans/{filename}"
    mask_rel_path = f"{branch_folder}/masks/{filename}"

    return ct_rel_path, mask_rel_path

# ============================================================
# Cell 5 — Robust image readers with backend reporting
# ============================================================
def make_identity_direction(img: sitk.Image) -> sitk.Image:
    dim = img.GetDimension()
    out = sitk.Image(img)
    out.SetDirection(tuple(np.eye(dim, dtype=np.float64).flatten()))
    return out


def safe_sitk_read_with_backend(path: str, force_identity_direction: bool = True):
    """
    Returns:
      img, backend
    backend in {"sitk", "nibabel_fallback"}

    Strategy:
    - try SimpleITK first
    - fallback to nibabel if SimpleITK fails
    - transpose nibabel array from (X,Y,Z) -> (Z,Y,X)
    """
    path = str(path)

    # 1) Try SimpleITK
    try:
        img = sitk.ReadImage(path)
        if force_identity_direction:
            img = make_identity_direction(img)
        return img, "sitk"

    except Exception:
        # 2) Fallback to nibabel
        nii = nib.load(path)
        arr = np.asanyarray(nii.dataobj)

        if arr.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {arr.shape} for {path}")

        # nibabel usually exposes voxel array as (X,Y,Z)
        # convert to pipeline standard (Z,Y,X)
        arr = np.transpose(arr, (2, 1, 0))

        img = sitk.GetImageFromArray(arr)

        zooms = nii.header.get_zooms()[:3]
        img.SetSpacing((float(zooms[0]), float(zooms[1]), float(zooms[2])))

        aff = nii.affine
        try:
            img.SetOrigin((float(aff[0, 3]), float(aff[1, 3]), float(aff[2, 3])))
        except Exception:
            pass

        if force_identity_direction:
            img = make_identity_direction(img)

        return img, "nibabel_fallback"
    

# ============================================================
# Cell 6 — Utility functions for QC and HU sanity
# ============================================================
def sitk_to_arr(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img)  # z,y,x

def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return default
        return x
    except Exception:
        return default

def sample_voxels_from_proxy(dataobj, shape, n_slices=16, xy_stride=4, max_voxels=400_000):
    """
    Sample scaled voxel values from a nibabel ArrayProxy without loading
    the full volume. This is the same idea as your earlier HU sanity code.
    """
    if len(shape) < 3:
        return np.array([], dtype=np.float32)

    x, y, z = shape[0], shape[1], shape[2]
    if z <= 0:
        return np.array([], dtype=np.float32)

    if z <= n_slices:
        z_idxs = np.arange(z, dtype=int)
    else:
        z_idxs = np.linspace(0, z - 1, n_slices, dtype=int)

    chunks = []
    total = 0

    for zi in z_idxs:
        sl = np.asanyarray(dataobj[:, :, int(zi)])
        sl = sl[::xy_stride, ::xy_stride].astype(np.float32, copy=False)
        flat = sl.ravel()

        if flat.size == 0:
            continue

        remaining = max_voxels - total
        if remaining <= 0:
            break

        if flat.size > remaining:
            idx = np.random.choice(flat.size, size=remaining, replace=False)
            flat = flat[idx]

        chunks.append(flat)
        total += flat.size

    if not chunks:
        return np.array([], dtype=np.float32)

    return np.concatenate(chunks, axis=0)

def binarize_mask(arr: np.ndarray) -> np.ndarray:
    """
    Convert any mask to binary uint8:
    background -> 0
    foreground -> 1
    """
    arr = np.asarray(arr)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return (arr > 0).astype(np.uint8)

def compute_stats(arr: np.ndarray):
    if arr.size == 0:
        return {}

    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}

    q = np.percentile(arr, [1, 5, 50, 95, 99]).astype(float)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p01": float(q[0]),
        "p05": float(q[1]),
        "p50": float(q[2]),
        "p95": float(q[3]),
        "p99": float(q[4]),
        "n": int(arr.size),
    }

def hu_heuristic(stats):
    """
    Same permissive HU-like rule from your earlier QC code.
    """
    p01 = stats.get("p01", np.nan)
    p99 = stats.get("p99", np.nan)
    mn  = stats.get("min", np.nan)
    mx  = stats.get("max", np.nan)

    if any(np.isnan(v) for v in [p01, p99, mn, mx]):
        return False

    cond_air = (p01 < -700) or (mn < -900)
    cond_bone_or_high = (p99 > 200) or (mx > 600)

    return bool(cond_air and cond_bone_or_high)

def needs_offset_fix(stats: dict) -> bool:
    """
    Heuristic for obvious shifted / unsigned-like CT values.
    """
    return (
        stats.get("min", np.nan) >= 0 and
        stats.get("p01", np.nan) > -50 and
        stats.get("p99", np.nan) > 1000 and
        stats.get("max", np.nan) <= 4095
    )

def fix_ct_offset_if_needed(arr: np.ndarray, enable_fix: bool = True):
    stats = compute_stats(arr)
    if enable_fix and needs_offset_fix(stats):
        return arr.astype(np.int16) - 1024, "offset_minus_1024"
    return arr, "no_offset_fix"


# ============================================================
# Cell 7 — Resampling helpers
# ============================================================
def compute_new_size(old_size, old_spacing, new_spacing):
    return [
        max(1, int(round(osz * osp / nsp)))
        for osz, osp, nsp in zip(old_size, old_spacing, new_spacing)
    ]

def resample_image_to_spacing(img: sitk.Image, out_spacing_xyz, is_label: bool) -> sitk.Image:
    old_spacing = img.GetSpacing()
    old_size = img.GetSize()
    new_size = compute_new_size(old_size, old_spacing, out_spacing_xyz)

    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing(tuple(float(v) for v in out_spacing_xyz))
    rs.SetSize([int(v) for v in new_size])
    rs.SetOutputDirection(img.GetDirection())
    rs.SetOutputOrigin(img.GetOrigin())
    rs.SetTransform(sitk.Transform())
    rs.SetDefaultPixelValue(0)
    rs.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return rs.Execute(img)

def downsample_z_ct_by_slab_average(img: sitk.Image, target_z: float = 5.0) -> sitk.Image:
    arr = sitk_to_arr(img).astype(np.float32)
    sx, sy, sz = img.GetSpacing()

    factor = max(1, int(round(target_z / sz)))
    if factor <= 1:
        return img

    slabs = []
    for i in range(0, arr.shape[0], factor):
        slab = arr[i:i + factor]
        slabs.append(slab.mean(axis=0))

    out_arr = np.stack(slabs, axis=0)
    out = sitk.GetImageFromArray(out_arr)
    out.SetSpacing((sx, sy, sz * factor))
    out.SetOrigin(img.GetOrigin())
    out.SetDirection(img.GetDirection())
    return out

def downsample_z_mask_by_slab_max(img: sitk.Image, target_z: float = 5.0) -> sitk.Image:
    arr = (sitk_to_arr(img) > 0).astype(np.uint8)
    sx, sy, sz = img.GetSpacing()

    factor = max(1, int(round(target_z / sz)))
    if factor <= 1:
        return img

    slabs = []
    for i in range(0, arr.shape[0], factor):
        slab = arr[i:i + factor]
        slabs.append(slab.max(axis=0))

    out_arr = np.stack(slabs, axis=0).astype(np.uint8)
    out = sitk.GetImageFromArray(out_arr)
    out.SetSpacing((sx, sy, sz * factor))
    out.SetOrigin(img.GetOrigin())
    out.SetDirection(img.GetDirection())
    return out

def resample_pair_to_target(ct_img: sitk.Image, mask_img: sitk.Image, target_z: float = 5.0):
    sx, sy, sz = ct_img.GetSpacing()
    target_spacing = (float(sx), float(sy), float(target_z))

    # CT: slab average first
    ct2 = downsample_z_ct_by_slab_average(ct_img, target_z=target_z)
    if abs(ct2.GetSpacing()[2] - target_z) > 1e-3:
        ct2 = resample_image_to_spacing(ct2, target_spacing, is_label=False)

    # Mask: slab max first, then align to CT grid
    ms2 = downsample_z_mask_by_slab_max(mask_img, target_z=target_z)

    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(ct2)
    rs.SetTransform(sitk.Transform())
    rs.SetInterpolator(sitk.sitkNearestNeighbor)
    rs.SetDefaultPixelValue(0)
    ms2 = rs.Execute(ms2)
    ms2 = sitk.Cast(ms2 > 0, sitk.sitkUInt8)

    return ct2, ms2

# ============================================================
# Cell 8 — Save helper
# ============================================================
def write_pair_as_niigz(ct_img: sitk.Image, mask_img: sitk.Image, case_id: str, branch: str):
    ct_out_dir, mask_out_dir = get_output_dirs_for_branch(branch)

    ct_out = ct_out_dir / to_niigz_name(case_id)
    mask_out = mask_out_dir / to_niigz_name(case_id)

    sitk.WriteImage(ct_img, str(ct_out), useCompression=True)
    sitk.WriteImage(mask_img, str(mask_out), useCompression=True)

    return ct_out, mask_out

# ============================================================
# Cell 9 — Load raw case index
# ============================================================
case_index_raw = pd.read_csv(CASE_INDEX_RAW_CSV)

required_cols = [
    "case_id", "branch", "source_dataset",
    "ct_ext", "mask_ext", "ct_path", "mask_path"
]

for c in required_cols:
    if c not in case_index_raw.columns:
        raise ValueError(f"Missing required column in case_index_raw.csv: {c}")

print("Loaded raw case index:", len(case_index_raw))
print(case_index_raw.head(10).to_string(index=False))


# ============================================================
# Cell 10 — Preprocessing scan/report pass
# ============================================================
SAMPLES_PER_VOLUME_SLICES = 16
XY_STRIDE = 4
MAX_VOXELS_PER_VOLUME = 400_000

thickness_rows = []

for row in tqdm(case_index_raw.to_dict("records"), desc="Scanning raw cases", unit="case"):
    case_id = row["case_id"]
    branch = row["branch"]
    source_dataset = row["source_dataset"]

    ct_path = resolve_input_path(row["ct_path"])
    mask_path = resolve_input_path(row["mask_path"])

    try:
        # robust read for geometry / spacing / actual preprocessing
        ct_img, ct_backend = safe_sitk_read_with_backend(ct_path, force_identity_direction=True)
        mask_img, mask_backend = safe_sitk_read_with_backend(mask_path, force_identity_direction=True)

        ct_arr = sitk_to_arr(ct_img)
        mask_arr = sitk_to_arr(mask_img)

        # HU-like QC via nibabel proxy sampling
        nii = nib.load(str(ct_path))  # lazy load
        vox = sample_voxels_from_proxy(
            nii.dataobj,
            shape=nii.shape,
            n_slices=SAMPLES_PER_VOLUME_SLICES,
            xy_stride=XY_STRIDE,
            max_voxels=MAX_VOXELS_PER_VOLUME,
        )
        stats = compute_stats(vox)
        looks_like_HU = hu_heuristic(stats) if stats else False
        offset_fix_flag = needs_offset_fix(stats) if stats else False

        sx, sy, sz = ct_img.GetSpacing()

        # recommendation logic
        if not looks_like_HU:
            recommendation = "drop_non_hu_like"
        elif float(sz) < RESAMPLE_BELOW_MM:
            recommendation = "resample_to_5mm"
        else:
            recommendation = "keep_original"

        thickness_rows.append({
            "case_id": case_id,
            "branch": branch,
            "source_dataset": source_dataset,
            "ct_shape": str(tuple(int(x) for x in ct_arr.shape)),
            "mask_shape": str(tuple(int(x) for x in mask_arr.shape)),
            "ct_sx": float(sx),
            "ct_sy": float(sy),
            "ct_sz": float(sz),

            # HU-like QC fields
            "hu_min": stats.get("min", np.nan) if stats else np.nan,
            "hu_max": stats.get("max", np.nan) if stats else np.nan,
            "hu_p01": stats.get("p01", np.nan) if stats else np.nan,
            "hu_p05": stats.get("p05", np.nan) if stats else np.nan,
            "hu_p50": stats.get("p50", np.nan) if stats else np.nan,
            "hu_p95": stats.get("p95", np.nan) if stats else np.nan,
            "hu_p99": stats.get("p99", np.nan) if stats else np.nan,
            "looks_like_HU": bool(looks_like_HU),
            "needs_offset_fix": bool(offset_fix_flag),

            "mask_empty": bool((mask_arr > 0).sum() == 0),
            "ct_reader_backend": ct_backend,
            "mask_reader_backend": mask_backend,
            "recommendation": recommendation,
            "status": "ok",
            "error": None,
        })

    except Exception as e:
        thickness_rows.append({
            "case_id": case_id,
            "branch": branch,
            "source_dataset": source_dataset,
            "ct_shape": None,
            "mask_shape": None,
            "ct_sx": None,
            "ct_sy": None,
            "ct_sz": None,

            "hu_min": np.nan,
            "hu_max": np.nan,
            "hu_p01": np.nan,
            "hu_p05": np.nan,
            "hu_p50": np.nan,
            "hu_p95": np.nan,
            "hu_p99": np.nan,
            "looks_like_HU": False,
            "needs_offset_fix": False,

            "mask_empty": None,
            "ct_reader_backend": None,
            "mask_reader_backend": None,
            "recommendation": "unresolved_error",
            "status": "failed",
            "error": repr(e),
        })

thickness_report = pd.DataFrame(thickness_rows)
thickness_report.to_csv(THICKNESS_REPORT_CSV, index=False)

print("Saved:", THICKNESS_REPORT_CSV)
print("\nRecommendation counts:")
print(thickness_report["recommendation"].value_counts(dropna=False).to_string())

print("\nStatus counts:")
print(thickness_report["status"].value_counts(dropna=False).to_string())

print("\nHU-like counts:")
print(thickness_report["looks_like_HU"].value_counts(dropna=False).to_string())

# ============================================================
# Cell 11 — Main preprocessing loop (drop non-HU-like cases)
# ============================================================
action_rows = []
preprocessed_rows = []

# make lookup from scan/report pass
scan_lookup = thickness_report.set_index("case_id").to_dict("index")

for row in tqdm(case_index_raw.to_dict("records"), desc="Preprocessing cases", unit="case"):
    case_id = row["case_id"]
    branch = row["branch"]
    source_dataset = row["source_dataset"]

    ct_path = resolve_input_path(row["ct_path"])
    mask_path = resolve_input_path(row["mask_path"])

    scan_info = scan_lookup.get(case_id, None)
    if scan_info is None:
        action_rows.append({
            "case_id": case_id,
            "branch": branch,
            "source_dataset": source_dataset,
            "ct_reader_backend": None,
            "mask_reader_backend": None,
            "hu_action": "missing_scan_info",
            "resample_action": "missing_scan_info",
            "status": "failed",
            "error": "case_id not found in thickness_report",
        })
        continue

    recommendation = scan_info["recommendation"]

    # Drop unresolved errors
    if recommendation == "unresolved_error":
        action_rows.append({
            "case_id": case_id,
            "branch": branch,
            "source_dataset": source_dataset,
            "ct_reader_backend": scan_info.get("ct_reader_backend"),
            "mask_reader_backend": scan_info.get("mask_reader_backend"),
            "hu_action": "skip_unresolved_error",
            "resample_action": "skip_unresolved_error",
            "status": "dropped_unresolved_error",
            "error": scan_info.get("error"),
        })
        continue

    # Drop non-HU-like scans
    if recommendation == "drop_non_hu_like":
        action_rows.append({
            "case_id": case_id,
            "branch": branch,
            "source_dataset": source_dataset,
            "ct_reader_backend": scan_info.get("ct_reader_backend"),
            "mask_reader_backend": scan_info.get("mask_reader_backend"),
            "hu_action": "dropped_non_hu_like",
            "resample_action": "not_processed",
            "status": "dropped_non_hu_like",
            "error": None,
        })
        continue

    try:
        # robust read
        ct_img, ct_backend = safe_sitk_read_with_backend(ct_path, force_identity_direction=True)
        mask_img, mask_backend = safe_sitk_read_with_backend(mask_path, force_identity_direction=True)

        # arrays
        ct_arr = sitk_to_arr(ct_img).astype(np.float32)
        mask_arr = sitk_to_arr(mask_img)

        # HU sanity / obvious offset correction
        ct_arr_fixed, hu_action = fix_ct_offset_if_needed(ct_arr, enable_fix=ENABLE_HU_OFFSET_FIX)
        mask_arr_fixed = binarize_mask(mask_arr)

        # rebuild sitk images on original grid
        ct_fixed = sitk.GetImageFromArray(ct_arr_fixed.astype(np.float32))
        ct_fixed.SetSpacing(ct_img.GetSpacing())
        ct_fixed.SetOrigin(ct_img.GetOrigin())
        ct_fixed.SetDirection(ct_img.GetDirection())

        mask_fixed = sitk.GetImageFromArray(mask_arr_fixed.astype(np.uint8))
        mask_fixed.SetSpacing(mask_img.GetSpacing())
        mask_fixed.SetOrigin(mask_img.GetOrigin())
        mask_fixed.SetDirection(mask_img.GetDirection())

        # thickness policy
        sx, sy, sz = ct_fixed.GetSpacing()
        if recommendation == "resample_to_5mm":
            ct_final, mask_final = resample_pair_to_target(ct_fixed, mask_fixed, target_z=TARGET_DZ_MM)
            resample_action = "resampled_to_5mm"
        else:
            ct_final = ct_fixed
            mask_final = sitk.Cast(mask_fixed > 0, sitk.sitkUInt8)
            resample_action = "kept_original"

        # write standardized .nii.gz into branch-based output folder
        ct_out, mask_out = write_pair_as_niigz(ct_final, mask_final, case_id, branch)

        # store short portable paths
        ct_rel_path, mask_rel_path = make_preprocessed_relative_paths(branch, case_id)

        preprocessed_rows.append({
            "case_id": case_id,
            "branch": branch,
            "source_dataset": source_dataset,
            "ct_ext": ".nii.gz",
            "mask_ext": ".nii.gz",
            "ct_path": ct_rel_path,
            "mask_path": mask_rel_path,
        })

        action_rows.append({
            "case_id": case_id,
            "branch": branch,
            "source_dataset": source_dataset,
            "ct_reader_backend": ct_backend,
            "mask_reader_backend": mask_backend,
            "hu_action": hu_action,
            "resample_action": resample_action,
            "status": "ok",
            "error": None,
        })

    except Exception as e:
        action_rows.append({
            "case_id": case_id,
            "branch": branch,
            "source_dataset": source_dataset,
            "ct_reader_backend": None,
            "mask_reader_backend": None,
            "hu_action": "failed",
            "resample_action": "failed",
            "status": "failed",
            "error": repr(e),
        })

action_log = pd.DataFrame(action_rows)
action_log.to_csv(ACTION_LOG_CSV, index=False)

case_index_preprocessed = pd.DataFrame(preprocessed_rows)
case_index_preprocessed = case_index_preprocessed.sort_values(
    by=["branch", "source_dataset", "case_id"]
).reset_index(drop=True)
case_index_preprocessed.to_csv(CASE_INDEX_PREPROCESSED_CSV, index=False)

print("Saved:", ACTION_LOG_CSV)
print("Saved:", CASE_INDEX_PREPROCESSED_CSV)

print("\nAction log status counts:")
print(action_log["status"].value_counts(dropna=False).to_string())


# ============================================================
# Cell 12 — Final audit and summary
# ============================================================
seg_ct_files = [p for p in PREP_SEG_CT_DIR.iterdir() if p.is_file() and is_nifti_file(p)]
seg_mask_files = [p for p in PREP_SEG_MASK_DIR.iterdir() if p.is_file() and is_nifti_file(p)]

nor_ct_files = [p for p in PREP_NOR_CT_DIR.iterdir() if p.is_file() and is_nifti_file(p)]
nor_mask_files = [p for p in PREP_NOR_MASK_DIR.iterdir() if p.is_file() and is_nifti_file(p)]

final_ct_count = len(seg_ct_files) + len(nor_ct_files)
final_mask_count = len(seg_mask_files) + len(nor_mask_files)

n_segmented_index = int((case_index_preprocessed["branch"] == "segmented").sum())
n_normal_index = int((case_index_preprocessed["branch"] == "normal").sum())

assert len(seg_ct_files) == n_segmented_index, "Segmented CT file count does not match segmented index row count"
assert len(seg_mask_files) == n_segmented_index, "Segmented mask file count does not match segmented index row count"

assert len(nor_ct_files) == n_normal_index, "Normal CT file count does not match normal index row count"
assert len(nor_mask_files) == n_normal_index, "Normal mask file count does not match normal index row count"

assert final_ct_count == len(case_index_preprocessed), "Total CT file count does not match index row count"
assert final_mask_count == len(case_index_preprocessed), "Total mask file count does not match index row count"

summary = {
    "input_cases": int(len(case_index_raw)),
    "preprocessed_cases": int(len(case_index_preprocessed)),
    "failed_cases": int((action_log["status"] == "failed").sum()),
    "dropped_non_hu_like": int((action_log["status"] == "dropped_non_hu_like").sum()),
    "dropped_unresolved_error": int((action_log["status"] == "dropped_unresolved_error").sum()),

    "segmented_ct_files": int(len(seg_ct_files)),
    "segmented_mask_files": int(len(seg_mask_files)),
    "normal_ct_files": int(len(nor_ct_files)),
    "normal_mask_files": int(len(nor_mask_files)),
    "final_ct_files": int(final_ct_count),
    "final_mask_files": int(final_mask_count),

    "resampled_to_5mm": int((action_log["resample_action"] == "resampled_to_5mm").sum()),
    "kept_original": int((action_log["resample_action"] == "kept_original").sum()),
    "offset_minus_1024": int((action_log["hu_action"] == "offset_minus_1024").sum()),
    "no_offset_fix": int((action_log["hu_action"] == "no_offset_fix").sum()),
    "sitk_ct_reads": int((action_log["ct_reader_backend"] == "sitk").sum()),
    "nibabel_ct_fallback_reads": int((action_log["ct_reader_backend"] == "nibabel_fallback").sum()),
    "sitk_mask_reads": int((action_log["mask_reader_backend"] == "sitk").sum()),
    "nibabel_mask_fallback_reads": int((action_log["mask_reader_backend"] == "nibabel_fallback").sum()),
}

with open(SUMMARY_JSON, "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))

print("\nFinal output structure:")
print(PREPROCESSED_ROOT)
print("├── Segmented Scans/")
print("│   ├── ct_scans/")
print("│   └── masks/")
print("├── Normal Scans/")
print("│   ├── ct_scans/")
print("│   └── masks/")
print("├── case_index_preprocessed.csv")
print("├── preprocess_action_log.csv")
print("├── thickness_report.csv")
print("└── preprocess_summary.json")