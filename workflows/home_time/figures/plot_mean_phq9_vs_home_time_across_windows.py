from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, linregress

# =============================================================================
# SETTINGS
# =============================================================================

# Main file containing PHQ-8/PHQ-9 + multiple home-time windows
DATA_PATH = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/phq_temporal_windows/"
    "phq8_phq9_home_time_all_assessments.csv"
)

OUT_DIR = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/paper_replication"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_PNG = OUT_DIR / "mean_phq9_vs_home_time_across_windows.png"
FIG_PDF = OUT_DIR / "mean_phq9_vs_home_time_across_windows.pdf"
SUMMARY_CSV = OUT_DIR / "mean_phq9_vs_home_time_across_windows_summary.csv"
PARTICIPANT_CSV = OUT_DIR / "mean_phq9_vs_home_time_across_windows_participant_means.csv"

# If True: use only participants who contribute at least one paired observation
# in ALL requested windows. This makes panel-to-panel comparison cleaner.
# If False: each panel uses its own maximum available sample.
USE_COMMON_PARTICIPANT_SET = True

# =============================================================================
# LOAD
# =============================================================================

df = pd.read_csv(DATA_PATH, low_memory=False)

print("=" * 100)
print("MEAN PHQ-9 vs MEAN HOME TIME ACROSS TEMPORAL WINDOWS")
print("=" * 100)
print("Input file:", DATA_PATH)
print("Rows:", len(df))
print("Columns:")
for c in df.columns:
    print(" ", c)

# =============================================================================
# IDENTIFY REQUIRED COLUMNS
# =============================================================================

def first_present(candidates, cols):
    for c in candidates:
        if c in cols:
            return c
    return None

cols = set(df.columns)

participant_col = first_present(
    ["participant_id"],
    cols
)

phq9_col = first_present(
    ["phq9_total", "depression_score"],
    cols
)

date_col = first_present(
    ["assessment_date", "date", "timestamp"],
    cols
)

if participant_col is None:
    raise RuntimeError("Could not find participant_id column.")

if phq9_col is None:
    raise RuntimeError(
        "Could not find a PHQ-9 column. Tried: phq9_total, depression_score"
    )

WINDOW_CANDIDATES = {
    "Previous 1 day": [
        "home_prev_1d_pct",
        "home_1d_pct",
        "home_previous_1d_pct",
        "time_at_home_prev_1d_pct",
    ],
    "Previous 2 days": [
        "home_prev_2d_pct",
        "home_2d_pct",
        "home_previous_2d_pct",
        "time_at_home_prev_2d_pct",
    ],
    "Previous 3 days": [
        "home_prev_3d_pct",
        "home_3d_pct",
        "home_previous_3d_pct",
        "time_at_home_prev_3d_pct",
    ],
    "Previous 7 days": [
        "home_prev_7d_pct",
        "home_7d_pct",
        "home_previous_7d_pct",
        "time_at_home_prev_7d_pct",
    ],
    "Previous 14 days": [
        "home_14d_all_pct",
        "home_14d_pct",
        "home_prev_14d_pct",
        "home_previous_14d_pct",
        "time_at_home_prev_14d_pct",
    ],
}

resolved_windows = {}
for label, candidates in WINDOW_CANDIDATES.items():
    hit = first_present(candidates, cols)
    if hit is None:
        print(f"[WARNING] No column found for window: {label}")
        print(" Tried:", candidates)
    else:
        resolved_windows[label] = hit

if len(resolved_windows) == 0:
    raise RuntimeError("Could not find any requested home-time window columns.")

print("\nResolved window columns:")
for k, v in resolved_windows.items():
    print(f"  {k}: {v}")

# =============================================================================
# CLEAN CORE VARIABLES
# =============================================================================

df = df.copy()
df[participant_col] = df[participant_col].astype(str)
df["phq9"] = pd.to_numeric(df[phq9_col], errors="coerce")

# Keep valid PHQ-9 rows
df = df[df["phq9"].between(0, 27)].copy()

# Optional date parsing, only for debugging / export
if date_col is not None:
    df["_assessment_date"] = pd.to_datetime(df[date_col], errors="coerce")

# =============================================================================
# FIND COMMON PARTICIPANT SET IF REQUESTED
# =============================================================================

window_pair_participants = {}

for label, col in resolved_windows.items():
    temp = df.copy()
    temp["_home"] = pd.to_numeric(temp[col], errors="coerce")
    temp = temp[temp["_home"].between(0, 100)]
    part_ids = set(temp[participant_col].dropna().unique())
    window_pair_participants[label] = part_ids
    print(f"\n{label}:")
    print("  valid paired rows:", len(temp))
    print("  participants:", len(part_ids))

if USE_COMMON_PARTICIPANT_SET:
    common_parts = None
    for s in window_pair_participants.values():
        common_parts = s if common_parts is None else common_parts.intersection(s)
    common_parts = sorted(common_parts)
    print("\nUsing common participant set across all windows.")
    print("Common participants:", len(common_parts))
else:
    common_parts = None
    print("\nUsing window-specific participant sets.")

# =============================================================================
# PANEL-SPECIFIC PARTICIPANT MEANS
# =============================================================================

panel_results = []
participant_export_rows = []

for label, col in resolved_windows.items():
    temp = df[[participant_col, "phq9", col] + (["_assessment_date"] if "_assessment_date" in df.columns else [])].copy()
    temp["_home"] = pd.to_numeric(temp[col], errors="coerce")
    temp = temp.dropna(subset=[participant_col, "phq9", "_home"]).copy()
    temp = temp[temp["_home"].between(0, 100)].copy()

    if common_parts is not None:
        temp = temp[temp[participant_col].isin(common_parts)].copy()

    participant_means = (
        temp.groupby(participant_col, as_index=False)
        .agg(
            mean_phq9=("phq9", "mean"),
            mean_home_time_pct=("_home", "mean"),
            n_paired_assessments=("phq9", "size"),
            median_phq9=("phq9", "median"),
            median_home_time_pct=("_home", "median"),
        )
    )

    n_participants = len(participant_means)
    n_pairs = len(temp)

    if n_participants < 3:
        print(f"[WARNING] Too few participants for {label}; skipping.")
        continue

    x = participant_means["mean_phq9"].to_numpy()
    y = participant_means["mean_home_time_pct"].to_numpy()

    pearson_r, pearson_p = pearsonr(x, y)
    spearman_rho, spearman_p = spearmanr(x, y)
    reg = linregress(x, y)

    panel_results.append({
        "window_label": label,
        "home_column": col,
        "n_participants": n_participants,
        "n_paired_rows": n_pairs,
        "median_paired_assessments_per_participant": participant_means["n_paired_assessments"].median(),
        "mean_phq9_across_participants": participant_means["mean_phq9"].mean(),
        "mean_home_time_pct_across_participants": participant_means["mean_home_time_pct"].mean(),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "linear_slope_home_pct_per_phq9": reg.slope,
        "linear_intercept": reg.intercept,
    })

    out = participant_means.copy()
    out["window_label"] = label
    out["home_column"] = col
    participant_export_rows.append(out)

# Save tables
summary_df = pd.DataFrame(panel_results)
summary_df.to_csv(SUMMARY_CSV, index=False)

participant_export = pd.concat(participant_export_rows, ignore_index=True)
participant_export.to_csv(PARTICIPANT_CSV, index=False)

print("\nSummary results:")
print(summary_df.to_string(index=False))

# =============================================================================
# PLOTTING
# =============================================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 11

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

panel_order = list(resolved_windows.keys())

# Scatter panels
for i, label in enumerate(panel_order):
    ax = axes[i]

    row = summary_df.loc[summary_df["window_label"] == label]
    if row.empty:
        ax.set_visible(False)
        continue

    home_col = row["home_column"].iloc[0]

    temp = df[[participant_col, "phq9", home_col]].copy()
    temp["_home"] = pd.to_numeric(temp[home_col], errors="coerce")
    temp = temp.dropna(subset=[participant_col, "phq9", "_home"]).copy()
    temp = temp[temp["_home"].between(0, 100)].copy()

    if common_parts is not None:
        temp = temp[temp[participant_col].isin(common_parts)].copy()

    participant_means = (
        temp.groupby(participant_col, as_index=False)
        .agg(
            mean_phq9=("phq9", "mean"),
            mean_home_time_pct=("_home", "mean"),
            n_paired_assessments=("phq9", "size"),
        )
    )

    x = participant_means["mean_phq9"].to_numpy()
    y = participant_means["mean_home_time_pct"].to_numpy()

    reg = linregress(x, y)

    ax.scatter(
        x,
        y,
        s=30,
        alpha=0.65,
    )

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = reg.intercept + reg.slope * x_line
    ax.plot(x_line, y_line, linewidth=1.6)

    r = row["pearson_r"].iloc[0]
    p = row["pearson_p"].iloc[0]
    rho = row["spearman_rho"].iloc[0]
    n = int(row["n_participants"].iloc[0])

    ax.text(
        0.03,
        0.97,
        f"N = {n}\nPearson r = {r:.3f}\nSpearman ρ = {rho:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )

    ax.set_title(label)
    ax.set_xlabel("Mean PHQ-9 across paired assessments")
    ax.set_ylabel("Mean home time across paired assessments (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(left=max(0, x.min() - 0.5), right=min(27, x.max() + 0.5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Summary panel
summary_ax = axes[5]

plot_df = summary_df.copy()
plot_df["window_rank"] = range(len(plot_df), 0, -1)

summary_ax.axvline(0, linestyle="--", linewidth=1)

summary_ax.scatter(
    plot_df["pearson_r"],
    plot_df["window_rank"],
    s=70,
    label="Pearson r",
)

summary_ax.scatter(
    plot_df["spearman_rho"],
    plot_df["window_rank"],
    s=70,
    marker="s",
    label="Spearman ρ",
)

summary_ax.set_yticks(plot_df["window_rank"])
summary_ax.set_yticklabels(plot_df["window_label"])
summary_ax.set_xlabel("Correlation coefficient")
summary_ax.set_title("Summary across windows")
summary_ax.legend(frameon=False, loc="best")
summary_ax.spines["top"].set_visible(False)
summary_ax.spines["right"].set_visible(False)

main_title = (
    "Average PHQ-9 and average home time across participants,\n"
    "compared across temporal windows"
)
fig.suptitle(main_title, fontsize=18, y=0.98)

fig.tight_layout(rect=[0, 0, 1, 0.95])

fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
fig.savefig(FIG_PDF, bbox_inches="tight")
plt.close(fig)

print("\nSaved:")
print(" ", FIG_PNG)
print(" ", FIG_PDF)
print(" ", SUMMARY_CSV)
print(" ", PARTICIPANT_CSV)
