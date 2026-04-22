# ============================================================
# Cell 1 — Imports
# ============================================================
from pathlib import Path
import os
import json
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib


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

# Dataset roots
RAW_ROOT = normalize_dir(
    "/path/to/CT-Seg1500-Raw"
)

PREPROCESSED_ROOT = normalize_dir(
    "/path/to/CT-Seg1500-Preprocessed"
)

# Output folder for stats, tables, and figures
REPORT_OUT_ROOT = PREPROCESSED_ROOT / "final_dataset_stats"
REPORT_OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Raw-side reports
CASE_RAW_CSV = RAW_ROOT / "case_index_raw.csv"
MERGE_SUMMARY_CSV = RAW_ROOT / "merge_summary_by_source.csv"

# Preprocessed-side reports
CASE_PREP_CSV = PREPROCESSED_ROOT / "case_index_preprocessed.csv"
THICKNESS_CSV = PREPROCESSED_ROOT / "thickness_report.csv"
ACTION_CSV = PREPROCESSED_ROOT / "preprocess_action_log.csv"
SUMMARY_JSON = PREPROCESSED_ROOT / "preprocess_summary.json"

required_files = [
    CASE_RAW_CSV,
    MERGE_SUMMARY_CSV,
    CASE_PREP_CSV,
    THICKNESS_CSV,
    ACTION_CSV,
    SUMMARY_JSON,
]

for p in required_files:
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")

# Validate branch-based dataset layout
required_dirs = [
    RAW_ROOT / "Segmented Scans" / "ct_scans",
    RAW_ROOT / "Segmented Scans" / "masks",
    RAW_ROOT / "Normal Scans" / "ct_scans",
    RAW_ROOT / "Normal Scans" / "masks",

    PREPROCESSED_ROOT / "Segmented Scans" / "ct_scans",
    PREPROCESSED_ROOT / "Segmented Scans" / "masks",
    PREPROCESSED_ROOT / "Normal Scans" / "ct_scans",
    PREPROCESSED_ROOT / "Normal Scans" / "masks",
]

for d in required_dirs:
    if not d.exists():
        raise FileNotFoundError(f"Missing required dataset folder: {d}")

print("RAW_ROOT          =", RAW_ROOT)
print("PREPROCESSED_ROOT =", PREPROCESSED_ROOT)
print("REPORT_OUT_ROOT   =", REPORT_OUT_ROOT)


# ============================================================
# Cell 3 — Load inputs
# ============================================================
case_raw = pd.read_csv(CASE_RAW_CSV)
merge_summary = pd.read_csv(MERGE_SUMMARY_CSV)
case_prep = pd.read_csv(CASE_PREP_CSV)
thickness_report = pd.read_csv(THICKNESS_CSV)
action_log = pd.read_csv(ACTION_CSV)

with open(SUMMARY_JSON, "r") as f:
    preprocess_summary = json.load(f)

print("case_raw rows       :", len(case_raw))
print("case_prep rows      :", len(case_prep))
print("thickness_report    :", len(thickness_report))
print("action_log          :", len(action_log))
print("merge_summary rows  :", len(merge_summary))
print("preprocess_summary  :")
print(json.dumps(preprocess_summary, indent=2))

# ============================================================
# Cell 4 — Basic helpers
# ============================================================
def parse_shape_string(shape_str):
    """
    Input example: '(28, 512, 512)' from thickness_report
    Returns: tuple or (nan, nan, nan)
    """
    if pd.isna(shape_str):
        return (np.nan, np.nan, np.nan)
    try:
        t = ast.literal_eval(shape_str)
        if isinstance(t, tuple) and len(t) == 3:
            return t
    except Exception:
        pass
    return (np.nan, np.nan, np.nan)

def add_shape_columns(df, shape_col="ct_shape", prefix="ct"):
    shapes = df[shape_col].apply(parse_shape_string)
    df = df.copy()
    df[f"{prefix}_D"] = shapes.apply(lambda x: x[0])
    df[f"{prefix}_H"] = shapes.apply(lambda x: x[1])
    df[f"{prefix}_W"] = shapes.apply(lambda x: x[2])
    return df

def iqr(series):
    s = pd.Series(series).dropna()
    if len(s) == 0:
        return np.nan
    return float(np.percentile(s, 75) - np.percentile(s, 25))

# ============================================================
# Cell 5 — Prepare enriched thickness/action tables
# ============================================================
thickness_enriched = add_shape_columns(thickness_report, shape_col="ct_shape", prefix="ct")
thickness_enriched = add_shape_columns(thickness_enriched, shape_col="mask_shape", prefix="mask")

# merge action log onto thickness report
full_qc = thickness_enriched.merge(
    action_log,
    on=["case_id", "branch", "source_dataset"],
    how="left",
    suffixes=("", "_action")
)

print("full_qc rows:", len(full_qc))
print(full_qc.head(5).to_string(index=False))


# ============================================================
# Cell 6 — Raw vs preprocessed cohort counts by dataset
# ============================================================
raw_counts = (
    case_raw.groupby(["branch", "source_dataset"])
    .size()
    .reset_index(name="n_cases_raw")
)

prep_counts = (
    case_prep.groupby(["branch", "source_dataset"])
    .size()
    .reset_index(name="n_cases_preprocessed")
)

counts_by_dataset = raw_counts.merge(
    prep_counts,
    on=["branch", "source_dataset"],
    how="outer"
).fillna(0)

counts_by_dataset["n_cases_raw"] = counts_by_dataset["n_cases_raw"].astype(int)
counts_by_dataset["n_cases_preprocessed"] = counts_by_dataset["n_cases_preprocessed"].astype(int)
counts_by_dataset["delta"] = counts_by_dataset["n_cases_preprocessed"] - counts_by_dataset["n_cases_raw"]

counts_by_dataset = counts_by_dataset.sort_values(["branch", "source_dataset"]).reset_index(drop=True)

counts_by_dataset_csv = REPORT_OUT_ROOT / "counts_raw_vs_preprocessed_by_dataset.csv"
counts_by_dataset.to_csv(counts_by_dataset_csv, index=False)

print("Saved:", counts_by_dataset_csv)
print(counts_by_dataset.to_string(index=False))

# ============================================================
# Cell 7 — Removed cases from raw -> preprocessed
# ============================================================
raw_key = case_raw[["case_id", "branch", "source_dataset"]].copy()
prep_key = case_prep[["case_id", "branch", "source_dataset"]].copy()

raw_ids = set(raw_key["case_id"].tolist())
prep_ids = set(prep_key["case_id"].tolist())

removed_ids = sorted(raw_ids - prep_ids)
added_ids = sorted(prep_ids - raw_ids)

removed_df = raw_key[raw_key["case_id"].isin(removed_ids)].copy().sort_values(["branch", "source_dataset", "case_id"])
added_df = prep_key[prep_key["case_id"].isin(added_ids)].copy().sort_values(["branch", "source_dataset", "case_id"])

removed_csv = REPORT_OUT_ROOT / "removed_cases_raw_to_preprocessed.csv"
added_csv = REPORT_OUT_ROOT / "added_cases_raw_to_preprocessed.csv"

removed_df.to_csv(removed_csv, index=False)
added_df.to_csv(added_csv, index=False)

print("Removed cases:", len(removed_df))
print(removed_df.to_string(index=False) if len(removed_df) > 0 else "None")

print("\nAdded cases:", len(added_df))
print(added_df.to_string(index=False) if len(added_df) > 0 else "None")


# ============================================================
# Cell 8 — Preprocessing/QC action summary by dataset
# ============================================================
action_summary = (
    full_qc.groupby(["branch", "source_dataset"])
    .agg(
        n_cases=("case_id", "count"),
        n_keep_original=("recommendation", lambda x: int((x == "keep_original").sum())),
        n_resample_to_5mm=("recommendation", lambda x: int((x == "resample_to_5mm").sum())),
        n_drop_non_hu_like=("recommendation", lambda x: int((x == "drop_non_hu_like").sum())),
        n_unresolved_error=("recommendation", lambda x: int((x == "unresolved_error").sum())),
        hu_like_true=("looks_like_HU", lambda x: int((x == True).sum())),
        hu_like_false=("looks_like_HU", lambda x: int((x == False).sum())),
        needs_offset_fix_true=("needs_offset_fix", lambda x: int((x == True).sum())),
        mask_empty_n=("mask_empty", lambda x: int((x == True).sum())),
        sitk_ct_reads=("ct_reader_backend", lambda x: int((x == "sitk").sum())),
        nibabel_ct_fallback_reads=("ct_reader_backend", lambda x: int((x == "nibabel_fallback").sum())),
        sitk_mask_reads=("mask_reader_backend", lambda x: int((x == "sitk").sum())),
        nibabel_mask_fallback_reads=("mask_reader_backend", lambda x: int((x == "nibabel_fallback").sum())),
    )
    .reset_index()
)

action_summary["resample_pct"] = 100.0 * action_summary["n_resample_to_5mm"] / action_summary["n_cases"]
action_summary["drop_non_hu_like_pct"] = 100.0 * action_summary["n_drop_non_hu_like"] / action_summary["n_cases"]

action_summary_csv = REPORT_OUT_ROOT / "preprocessing_action_summary_by_dataset.csv"
action_summary.to_csv(action_summary_csv, index=False)

print("Saved:", action_summary_csv)
print(action_summary.to_string(index=False))

# ============================================================
# Cell 9 — Build lightweight geometry/thickness summaries for RAW + PREPROCESSED
# Creates:
#   summary_by_dataset_raw.csv
#   summary_by_dataset_preprocessed.csv
#   geometry_thickness_summary_by_dataset_raw.csv
#   geometry_thickness_summary_by_dataset_preprocessed.csv
# ============================================================
import numpy as np
import pandas as pd
import nibabel as nib

def resolve_dataset_path(path_value: str, dataset_root: Path) -> Path:
    """
    Supports both:
      1) absolute paths
      2) short local paths from case_index CSV files, e.g.
         Segmented Scans/ct_scans/case_id.nii.gz
         Normal Scans/ct_scans/case_id.nii.gz
    """
    p = Path(str(path_value))

    if p.is_absolute():
        return p

    return dataset_root / p


def safe_header_geometry(ct_path):
    """
    Read lightweight NIfTI header-level geometry info only:
      - shape
      - voxel spacing

    Returns:
      D, H, W, vx, vy, vz

    Note:
      nibabel shape is usually (X, Y, Z), so:
        H = shape[0]
        W = shape[1]
        D = shape[2]
    """
    try:
        ct_path = Path(ct_path)

        img = nib.load(str(ct_path))
        shape = img.shape
        zooms = img.header.get_zooms()[:3]

        if len(shape) != 3:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        H = int(shape[0])
        W = int(shape[1])
        D = int(shape[2])

        vx = float(zooms[0])
        vy = float(zooms[1])
        vz = float(zooms[2])

        return D, H, W, vx, vy, vz

    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def iqr(series):
    s = pd.Series(series).dropna()
    if len(s) == 0:
        return np.nan
    return float(np.percentile(s, 75) - np.percentile(s, 25))


def add_geom_from_index(df_index: pd.DataFrame, snapshot_name: str, dataset_root: Path):
    rows = []

    for _, r in df_index.iterrows():
        ct_path = resolve_dataset_path(r["ct_path"], dataset_root)

        D, H, W, vx, vy, vz = safe_header_geometry(ct_path)

        rows.append({
            "snapshot": snapshot_name,
            "case_id": r["case_id"],
            "branch": r["branch"],
            "source_dataset": r["source_dataset"],
            "D": D,
            "H": H,
            "W": W,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "ct_path_resolved": str(ct_path),
        })

    return pd.DataFrame(rows)


def add_vz_bins(df):
    df = df.copy()
    vz = df["vz"]

    df["vz_bin_<1"]    = (vz < 1).astype(int)
    df["vz_bin_1_2"]   = ((vz >= 1) & (vz < 2)).astype(int)
    df["vz_bin_2_4"]   = ((vz >= 2) & (vz < 4)).astype(int)
    df["vz_bin_4_6p5"] = ((vz >= 4) & (vz <= 6.5)).astype(int)
    df["vz_bin_>6p5"]  = (vz > 6.5).astype(int)

    return df


def summarize_geom_by_dataset(df):
    df = add_vz_bins(df)

    out = (
        df.groupby(["branch", "source_dataset"])
        .agg(
            n_cases=("case_id", "count"),
            D_med=("D", "median"),
            D_iqr=("D", iqr),
            D_min=("D", "min"),
            D_max=("D", "max"),
            vz_med=("vz", "median"),
            vz_iqr=("vz", iqr),
            vz_min=("vz", "min"),
            vz_max=("vz", "max"),
            vz_bin_lt1=("vz_bin_<1", "sum"),
            vz_bin_1_2=("vz_bin_1_2", "sum"),
            vz_bin_2_4=("vz_bin_2_4", "sum"),
            vz_bin_4_6p5=("vz_bin_4_6p5", "sum"),
            vz_bin_gt6p5=("vz_bin_>6p5", "sum"),
        )
        .reset_index()
    )

    # Rename to match later plotting cells
    out = out.rename(columns={
        "vz_bin_lt1": "vz_bin_<1",
        "vz_bin_gt6p5": "vz_bin_>6p5",
    })

    return out


print("Building lightweight RAW geometry features...")
geom_raw_cases = add_geom_from_index(
    df_index=case_raw,
    snapshot_name="raw",
    dataset_root=RAW_ROOT,
)

print("Building lightweight PREPROCESSED geometry features...")
geom_prep_cases = add_geom_from_index(
    df_index=case_prep,
    snapshot_name="preprocessed",
    dataset_root=PREPROCESSED_ROOT,
)

# Optional case-level geometry files for debugging/audit
geom_raw_cases_csv = REPORT_OUT_ROOT / "geometry_cases_raw.csv"
geom_prep_cases_csv = REPORT_OUT_ROOT / "geometry_cases_preprocessed.csv"

geom_raw_cases.to_csv(geom_raw_cases_csv, index=False)
geom_prep_cases.to_csv(geom_prep_cases_csv, index=False)

summary_by_dataset_raw = summarize_geom_by_dataset(geom_raw_cases)
summary_by_dataset_preprocessed = summarize_geom_by_dataset(geom_prep_cases)

# Save for reuse
summary_by_dataset_raw.to_csv(
    REPORT_OUT_ROOT / "summary_by_dataset_raw.csv",
    index=False,
)

summary_by_dataset_preprocessed.to_csv(
    REPORT_OUT_ROOT / "summary_by_dataset_preprocessed.csv",
    index=False,
)

# Also save under geometry_* names because later plotting cells use them
summary_by_dataset_raw.to_csv(
    REPORT_OUT_ROOT / "geometry_thickness_summary_by_dataset_raw.csv",
    index=False,
)

summary_by_dataset_preprocessed.to_csv(
    REPORT_OUT_ROOT / "geometry_thickness_summary_by_dataset_preprocessed.csv",
    index=False,
)

print("Saved:")
print(geom_raw_cases_csv)
print(geom_prep_cases_csv)
print(REPORT_OUT_ROOT / "summary_by_dataset_raw.csv")
print(REPORT_OUT_ROOT / "summary_by_dataset_preprocessed.csv")
print(REPORT_OUT_ROOT / "geometry_thickness_summary_by_dataset_raw.csv")
print(REPORT_OUT_ROOT / "geometry_thickness_summary_by_dataset_preprocessed.csv")

print("\nRAW summary:")
print(summary_by_dataset_raw.to_string(index=False))

print("\nPREPROCESSED summary:")
print(summary_by_dataset_preprocessed.to_string(index=False))


# ============================================================
# Cell 10 — Combined branch-level summary
# ============================================================
ok_qc = full_qc[full_qc["status"] == "ok"].copy()

combined_summary = (
    ok_qc.groupby(["branch"])
    .agg(
        n_cases=("case_id", "count"),
        ct_sz_min=("ct_sz", "min"),
        ct_sz_med=("ct_sz", "median"),
        ct_sz_iqr=("ct_sz", iqr),
        ct_sz_max=("ct_sz", "max"),
        ct_D_med=("ct_D", "median"),
        ct_D_iqr=("ct_D", iqr),
        hu_like_true=("looks_like_HU", lambda x: int((x == True).sum())),
        hu_like_false=("looks_like_HU", lambda x: int((x == False).sum())),
        n_resample_to_5mm=("recommendation", lambda x: int((x == "resample_to_5mm").sum())),
        n_keep_original=("recommendation", lambda x: int((x == "keep_original").sum())),
    )
    .reset_index()
)

combined_summary_csv = REPORT_OUT_ROOT / "combined_branch_summary.csv"
combined_summary.to_csv(combined_summary_csv, index=False)

print("Saved:", combined_summary_csv)
print(combined_summary.to_string(index=False))


# ============================================================
# Cell 11 — Figure 1: Raw vs preprocessed counts by dataset
# ============================================================
plot_df = counts_by_dataset.copy()

# readable ordering: segmented first, then normal
plot_df["branch_order"] = plot_df["branch"].map({"segmented": 0, "normal": 1}).fillna(99)
plot_df = plot_df.sort_values(["branch_order", "source_dataset"]).reset_index(drop=True)

plot_df["label"] = plot_df["source_dataset"] + " (" + plot_df["branch"] + ")"

x = np.arange(len(plot_df))
w = 0.38

plt.figure(figsize=(11, 5.5))
plt.bar(x - w/2, plot_df["n_cases_raw"].values, width=w, label="Raw")
plt.bar(x + w/2, plot_df["n_cases_preprocessed"].values, width=w, label="Preprocessed")

plt.xticks(x, plot_df["label"], rotation=30, ha="right")
plt.ylabel("Number of scans")
plt.xlabel("Source dataset")
plt.title("Raw vs Preprocessed Counts by Dataset")
plt.legend()
plt.tight_layout()

fig_path = REPORT_OUT_ROOT / "fig_counts_raw_vs_preprocessed.png"
plt.savefig(fig_path, dpi=200)
plt.close()

print("Saved:", fig_path)

# ============================================================
# Cell 12 — Figure 2: Thickness distribution by branch
# ============================================================

# Use branch-level counts from raw and preprocessed data
# Build raw thickness bins from case_index_raw + thickness-style logic only if raw thickness summary is available.
# Preferred: use raw/preprocessed summary tables if they already exist.

raw_summary_path = REPORT_OUT_ROOT / "summary_by_dataset_raw.csv"
prep_summary_path = REPORT_OUT_ROOT / "summary_by_dataset_preprocessed.csv"

if not raw_summary_path.exists() or not prep_summary_path.exists():
    raise FileNotFoundError(
        "This plot expects summary_by_dataset_raw.csv and summary_by_dataset_preprocessed.csv "
        "to exist in REPORT_OUT_ROOT."
    )

raw = pd.read_csv(raw_summary_path)
prep = pd.read_csv(prep_summary_path)

BIN_COLS = ["vz_bin_<1", "vz_bin_1_2", "vz_bin_2_4", "vz_bin_4_6p5", "vz_bin_>6p5"]
BIN_LABELS = ["<1", "1–2", "2–4", "4–6.5", ">6.5"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

for ax, branch_name in zip(axes, ["segmented", "normal"]):
    raw_b = raw[raw["branch"] == branch_name].copy()
    prep_b = prep[prep["branch"] == branch_name].copy()

    raw_counts = raw_b[BIN_COLS].sum().values.astype(int)
    prep_counts = prep_b[BIN_COLS].sum().values.astype(int)

    x = np.arange(len(BIN_COLS))
    w = 0.38

    ax.bar(x - w/2, raw_counts, width=w, label="Raw")
    ax.bar(x + w/2, prep_counts, width=w, label="Preprocessed")

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS)
    ax.set_xlabel("Slice thickness vz (mm) bins")
    ax.set_ylabel("Number of scans")
    ax.set_title(f"{branch_name.capitalize()} branch")
    ax.legend()

fig.suptitle("Slice-Thickness Distribution: Raw vs Preprocessed", y=1.02, fontsize=13)
fig.tight_layout()

fig_path = REPORT_OUT_ROOT / "fig_vz_distribution.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()

print("Saved:", fig_path)

# ============================================================
# Cell 13 — Figure 3: Resampled percentage by dataset
# ============================================================
plot_df = action_summary.copy()

# keep datasets in a readable order: segmented first, then normal
plot_df["branch_order"] = plot_df["branch"].map({"segmented": 0, "normal": 1}).fillna(99)
plot_df = plot_df.sort_values(["branch_order", "resample_pct"], ascending=[True, False]).reset_index(drop=True)

# build x labels with branch info only when needed
plot_df["label"] = plot_df["source_dataset"] + " (" + plot_df["branch"] + ")"

x = np.arange(len(plot_df))

plt.figure(figsize=(11, 5.5))
plt.bar(x, plot_df["resample_pct"].values)

plt.xticks(x, plot_df["label"], rotation=30, ha="right")
plt.ylabel("Resampled scans (%)")
plt.xlabel("Source dataset")
plt.title("Percentage of Scans Resampled to ~5 mm by Dataset")
plt.tight_layout()

fig_path = REPORT_OUT_ROOT / "fig_resample_pct_by_dataset.png"
plt.savefig(fig_path, dpi=200)
plt.close()

print("Saved:", fig_path)


# ============================================================
# Cell 14 — Figure 4: Median slice count (D) by dataset
# ============================================================

raw_geom = pd.read_csv(REPORT_OUT_ROOT / "geometry_thickness_summary_by_dataset_raw.csv")
prep_geom = pd.read_csv(REPORT_OUT_ROOT / "geometry_thickness_summary_by_dataset_preprocessed.csv")

m = raw_geom.merge(
    prep_geom,
    on=["branch", "source_dataset"],
    how="inner",
    suffixes=("_raw", "_prep")
)

# Fixed order
dataset_order = ["CQ500-51", "CT-ICH", "BHSD", "HemSeg500", "Instance22", "RSNA-Normal"]
m["source_dataset"] = pd.Categorical(m["source_dataset"], categories=dataset_order, ordered=True)
m = m.sort_values("source_dataset").reset_index(drop=True)

x = np.arange(len(m))
w = 0.38

plt.figure(figsize=(12, 5.5))
plt.bar(x - w/2, m["D_med_raw"].values, width=w, label="Raw")
plt.bar(x + w/2, m["D_med_prep"].values, width=w, label="Preprocessed")

plt.xticks(x, m["source_dataset"], rotation=30, ha="right")
plt.ylabel("Median number of slices (D)")
plt.xlabel("Source dataset")
plt.title("Median Slice Count by Dataset: Raw vs Preprocessed")
plt.legend()
plt.tight_layout()

fig_path = REPORT_OUT_ROOT / "fig_D_median_by_dataset.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()

print("Saved:", fig_path)


# ============================================================
# Cell 15 — Compact master reporting JSON
# ============================================================
master_summary = {
    "raw_case_count": int(len(case_raw)),
    "preprocessed_case_count": int(len(case_prep)),
    "removed_case_count": int(len(removed_df)),
    "added_case_count": int(len(added_df)),
    "preprocess_summary": preprocess_summary,
    "counts_by_dataset": counts_by_dataset.to_dict(orient="records"),
    "combined_branch_summary": combined_summary.to_dict(orient="records"),
}

master_summary_json = REPORT_OUT_ROOT / "dataset_master_summary.json"
with open(master_summary_json, "w") as f:
    json.dump(master_summary, f, indent=2)

print("Saved:", master_summary_json)