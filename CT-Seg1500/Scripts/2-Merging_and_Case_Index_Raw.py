# ============================================================
# Cell 1 — Imports
# ============================================================
from pathlib import Path
import shutil
import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm

# ============================================================
# Cell 2 — Configuration
# ============================================================
# Input: folder that contains the 6 source dataset folders
# Example structure:
# root_input_dir/
#   BHSD/
#   CQ500-51/
#   CT-ICH/
#   HemSeg500/
#   Instance22/
#   RSNA-Normal/

ROOT_INPUT_DIR = Path("/path/to/CT-Seg1500 Source Datasets")
OUTPUT_DIR = Path("/path/to/CT-Seg1500-Raw")

# Expected source dataset folder names
SOURCE_DATASETS = [
    "BHSD",
    "CQ500-51",
    "CT-ICH",
    "HemSeg500",
    "Instance22",
    "RSNA-Normal",
]

# Only RSNA-Normal is considered normal; all others are segmented
NORMAL_DATASETS = {"RSNA-Normal"}

# Add prefix only to CT-ICH to avoid filename collisions with BHSD
PREFIX_RULES = {
    "CT-ICH": "ct-"
}

print("ROOT_INPUT_DIR =", ROOT_INPUT_DIR)
print("OUTPUT_DIR     =", OUTPUT_DIR)


# ============================================================
# Cell 3 — Helpers
# ============================================================


def is_nifti_file(path: Path) -> bool:
    """
    Accept only real NIfTI files and ignore macOS metadata junk like:
      ._filename.nii.gz
      .DS_Store
    """
    name = path.name

    # ignore hidden/macOS metadata files
    if name.startswith("._") or name.startswith("."):
        return False

    low = name.lower()
    return low.endswith(".nii") or low.endswith(".nii.gz")


def split_nii_name(filename: str) -> Tuple[str, str]:
    """
    Returns:
      stem, extension

    Example:
      '001.nii.gz' -> ('001', '.nii.gz')
      'CQ500-CT-4.nii' -> ('CQ500-CT-4', '.nii')
    """
    name = filename
    if name.lower().endswith(".nii.gz"):
        return name[:-7], ".nii.gz"
    elif name.lower().endswith(".nii"):
        return name[:-4], ".nii"
    else:
        raise ValueError(f"Not a NIfTI filename: {filename}")


def collect_nifti_map(folder: Path) -> Dict[str, Path]:
    """
    Returns a mapping:
      case_stem -> full file path

    Assumes one unique file per stem in the folder.
    """
    out = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and is_nifti_file(p):
            stem, _ = split_nii_name(p.name)
            if stem in out:
                raise ValueError(
                    f"Duplicate case stem found in folder {folder}: '{stem}' "
                    f"for files:\n- {out[stem].name}\n- {p.name}"
                )
            out[stem] = p
    return out


def ensure_source_structure(dataset_root: Path) -> Tuple[Path, Path]:
    ct_dir = dataset_root / "ct_scans"
    mask_dir = dataset_root / "masks"

    if not ct_dir.exists():
        raise FileNotFoundError(f"Missing ct_scans folder: {ct_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing masks folder: {mask_dir}")

    return ct_dir, mask_dir


def get_branch(source_dataset: str) -> str:
    return "normal" if source_dataset in NORMAL_DATASETS else "segmented"


def get_branch_folder(branch: str) -> str:
    """
    Maps internal branch names to final dataset folder names.
    """
    if branch == "segmented":
        return "Segmented Scans"
    if branch == "normal":
        return "Normal Scans"
    raise ValueError(f"Unknown branch: {branch}")


def make_output_dirs(output_root: Path) -> Dict[str, Dict[str, Path]]:
    """
    Creates and returns final branch-based output directories.

    Final structure:
      output_root/
      ├── Segmented Scans/
      │   ├── ct_scans/
      │   └── masks/
      └── Normal Scans/
          ├── ct_scans/
          └── masks/
    """
    out_dirs = {}

    for branch in ["segmented", "normal"]:
        branch_folder = get_branch_folder(branch)

        ct_out = output_root / branch_folder / "ct_scans"
        mask_out = output_root / branch_folder / "masks"

        ct_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        out_dirs[branch] = {
            "ct": ct_out,
            "mask": mask_out,
        }

    return out_dirs


def maybe_add_prefix(case_id: str, source_dataset: str) -> str:
    prefix = PREFIX_RULES.get(source_dataset, "")
    return f"{prefix}{case_id}" if prefix else case_id


# ============================================================
# Cell 4 — Validate sources and preview pair counts with progress
# ============================================================
preview_rows = []

for ds_name in tqdm(SOURCE_DATASETS, desc="Scanning source datasets", unit="dataset"):
    ds_root = ROOT_INPUT_DIR / ds_name
    if not ds_root.exists():
        raise FileNotFoundError(f"Missing source dataset folder: {ds_root}")

    ct_dir, mask_dir = ensure_source_structure(ds_root)

    ct_map = collect_nifti_map(ct_dir)
    mask_map = collect_nifti_map(mask_dir)

    ct_stems = set(ct_map.keys())
    mask_stems = set(mask_map.keys())

    paired = sorted(ct_stems & mask_stems)
    ct_only = sorted(ct_stems - mask_stems)
    mask_only = sorted(mask_stems - ct_stems)

    preview_rows.append({
        "source_dataset": ds_name,
        "branch": get_branch(ds_name),
        "n_ct_files": len(ct_map),
        "n_mask_files": len(mask_map),
        "n_paired": len(paired),
        "n_ct_only": len(ct_only),
        "n_mask_only": len(mask_only),
    })

preview_df = pd.DataFrame(preview_rows)
print(preview_df.to_string(index=False))


# ============================================================
# Cell 5 — Merge datasets into branch-based target folders with progress
# ============================================================
out_dirs = make_output_dirs(OUTPUT_DIR)

merge_rows: List[dict] = []
copy_log_rows: List[dict] = []

for ds_name in SOURCE_DATASETS:
    ds_root = ROOT_INPUT_DIR / ds_name
    ct_dir, mask_dir = ensure_source_structure(ds_root)

    ct_map = collect_nifti_map(ct_dir)
    mask_map = collect_nifti_map(mask_dir)

    paired_stems = sorted(set(ct_map.keys()) & set(mask_map.keys()))

    branch = get_branch(ds_name)
    branch_folder = get_branch_folder(branch)

    ct_out_dir = out_dirs[branch]["ct"]
    mask_out_dir = out_dirs[branch]["mask"]

    print(f"\n[{ds_name}] branch: {branch} | paired cases to copy: {len(paired_stems)}")

    for stem in tqdm(paired_stems, desc=f"Merging {ds_name}", unit="case"):
        ct_src = ct_map[stem]
        mask_src = mask_map[stem]

        _, ct_ext = split_nii_name(ct_src.name)
        _, mask_ext = split_nii_name(mask_src.name)

        case_id = maybe_add_prefix(stem, ds_name)

        ct_filename = f"{case_id}{ct_ext}"
        mask_filename = f"{case_id}{mask_ext}"

        ct_dst = ct_out_dir / ct_filename
        mask_dst = mask_out_dir / mask_filename

        # guard against collisions in merged output branch
        if ct_dst.exists():
            raise FileExistsError(f"Target CT already exists: {ct_dst}")
        if mask_dst.exists():
            raise FileExistsError(f"Target mask already exists: {mask_dst}")

        shutil.copy2(ct_src, ct_dst)
        shutil.copy2(mask_src, mask_dst)

        # Store short local paths for portable repo/dataset use
        ct_rel_path = f"{branch_folder}/ct_scans/{ct_filename}"
        mask_rel_path = f"{branch_folder}/masks/{mask_filename}"

        merge_rows.append({
            "case_id": case_id,
            "branch": branch,
            "source_dataset": ds_name,
            "ct_ext": ct_ext,
            "mask_ext": mask_ext,
            "ct_path": ct_rel_path,
            "mask_path": mask_rel_path,
        })

        copy_log_rows.append({
            "source_dataset": ds_name,
            "branch": branch,
            "original_stem": stem,
            "final_case_id": case_id,
            "ct_src": str(ct_src),
            "mask_src": str(mask_src),
            "ct_dst": str(ct_dst),
            "mask_dst": str(mask_dst),
        })

print("\nMerging finished.")
print("Total merged cases:", len(merge_rows))


# ============================================================
# Cell 6 — Build and save case_index_raw.csv
# ============================================================
print("Building case_index_raw DataFrame...")
case_index_raw = pd.DataFrame(merge_rows)

print("Sorting case index...")
case_index_raw = case_index_raw.sort_values(
    by=["branch", "source_dataset", "case_id"]
).reset_index(drop=True)

case_index_csv = OUTPUT_DIR / "case_index_raw.csv"
print(f"Saving case index to: {case_index_csv}")
case_index_raw.to_csv(case_index_csv, index=False)

print("Saved:", case_index_csv)
print(case_index_raw.head(20).to_string(index=False))

# ============================================================
# Cell 7 — Optional copy log and summary tables
# ============================================================
copy_log_df = pd.DataFrame(copy_log_rows)
copy_log_csv = OUTPUT_DIR / "merge_copy_log.csv"
copy_log_df.to_csv(copy_log_csv, index=False)

summary_by_source = (
    case_index_raw.groupby(["branch", "source_dataset"])
    .size()
    .reset_index(name="n_cases")
    .sort_values(["branch", "source_dataset"])
    .reset_index(drop=True)
)

summary_csv = OUTPUT_DIR / "merge_summary_by_source.csv"
summary_by_source.to_csv(summary_csv, index=False)

print("Saved copy log:", copy_log_csv)
print("Saved summary :", summary_csv)
print(summary_by_source)


# ============================================================
# Cell 8 — Final structure audit
# ============================================================

seg_ct_dir = OUTPUT_DIR / "Segmented Scans" / "ct_scans"
seg_mask_dir = OUTPUT_DIR / "Segmented Scans" / "masks"

nor_ct_dir = OUTPUT_DIR / "Normal Scans" / "ct_scans"
nor_mask_dir = OUTPUT_DIR / "Normal Scans" / "masks"

seg_ct_files = [p for p in seg_ct_dir.iterdir() if p.is_file() and is_nifti_file(p)]
seg_mask_files = [p for p in seg_mask_dir.iterdir() if p.is_file() and is_nifti_file(p)]

nor_ct_files = [p for p in nor_ct_dir.iterdir() if p.is_file() and is_nifti_file(p)]
nor_mask_files = [p for p in nor_mask_dir.iterdir() if p.is_file() and is_nifti_file(p)]

n_segmented_index = int((case_index_raw["branch"] == "segmented").sum())
n_normal_index = int((case_index_raw["branch"] == "normal").sum())

print("Segmented ct_scans files :", len(seg_ct_files))
print("Segmented masks files    :", len(seg_mask_files))
print("Segmented index rows     :", n_segmented_index)

print("Normal ct_scans files    :", len(nor_ct_files))
print("Normal masks files       :", len(nor_mask_files))
print("Normal index rows        :", n_normal_index)

print("Total CT files           :", len(seg_ct_files) + len(nor_ct_files))
print("Total mask files         :", len(seg_mask_files) + len(nor_mask_files))
print("Index rows               :", len(case_index_raw))

assert len(seg_ct_files) == n_segmented_index, "Segmented CT file count does not match segmented index row count"
assert len(seg_mask_files) == n_segmented_index, "Segmented mask file count does not match segmented index row count"

assert len(nor_ct_files) == n_normal_index, "Normal CT file count does not match normal index row count"
assert len(nor_mask_files) == n_normal_index, "Normal mask file count does not match normal index row count"

assert (len(seg_ct_files) + len(nor_ct_files)) == len(case_index_raw), "Total CT file count does not match index row count"
assert (len(seg_mask_files) + len(nor_mask_files)) == len(case_index_raw), "Total mask file count does not match index row count"

print("\nFinal output structure:")
print(OUTPUT_DIR)
print("├── Segmented Scans/")
print("│   ├── ct_scans/")
print("│   └── masks/")
print("├── Normal Scans/")
print("│   ├── ct_scans/")
print("│   └── masks/")
print("└── case_index_raw.csv")