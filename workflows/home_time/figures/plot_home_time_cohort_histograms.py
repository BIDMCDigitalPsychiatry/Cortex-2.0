from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

WEEKLY_PATH = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/"
    "gps_clinical_weekly_features_with_home.parquet"
)

OUTDIR = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/figures"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTDIR / "home_time_cohort_histograms.png"
OUT_PDF = OUTDIR / "home_time_cohort_histograms.pdf"

# ============================================================
# LOAD WEEKLY DATA
# ============================================================

df = pd.read_parquet(WEEKLY_PATH)

print("Loaded:", WEEKLY_PATH)
print("Rows:", len(df))
print("Columns:", len(df.columns))

# Keep interpretable weekly home-time windows
home = df.copy()
home["time_at_home_pct"] = pd.to_numeric(home["time_at_home_pct"], errors="coerce")
home["depression_score"] = pd.to_numeric(home["depression_score"], errors="coerce")
home["anxiety_score"] = pd.to_numeric(home["anxiety_score"], errors="coerce")

home_valid = home[home["time_at_home_pct"].notna()].copy()

print("Weekly windows with home time:", len(home_valid))
print("Participants with home time:", home_valid["participant_id"].nunique())

# ============================================================
# PANEL A: NUMBER OF WEEKLY WINDOWS PER PARTICIPANT
# ============================================================

windows_per_participant = (
    home_valid.groupby("participant_id")
    .size()
    .rename("n_windows")
)

# ============================================================
# PANEL B: PHQ-9 DISTRIBUTION ACROSS WEEKLY WINDOWS
# ============================================================

phq = home_valid["depression_score"].dropna()

# ============================================================
# PANEL C: AGE DISTRIBUTION
# Try to discover a demographics/roster file automatically
# ============================================================

def load_table(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return None


def find_age_series(weekly_df: pd.DataFrame):
    # If age is already in weekly file, use it
    for c in weekly_df.columns:
        cl = c.lower()
        if cl in {"age", "age_years", "participant_age"}:
            s = pd.to_numeric(weekly_df[c], errors="coerce").dropna()
            if len(s) > 0:
                return s.drop_duplicates(), f"embedded column: {c}"

    search_root = Path("/shared/Digital Clinic")

    candidate_patterns = [
        "**/*demograph*.csv",
        "**/*demograph*.parquet",
        "**/*demograph*.xlsx",
        "**/*roster*.csv",
        "**/*roster*.parquet",
        "**/*roster*.xlsx",
        "**/*participant*.csv",
        "**/*participant*.parquet",
        "**/*participant*.xlsx",
        "**/*clinical*.csv",
        "**/*clinical*.parquet",
        "**/*clinical*.xlsx",
    ]

    candidate_files = []
    for pattern in candidate_patterns:
        candidate_files.extend(search_root.glob(pattern))

    # de-duplicate and keep smaller likely metadata files first
    seen = set()
    deduped = []
    for p in candidate_files:
        if p in seen:
            continue
        seen.add(p)
        deduped.append(p)

    age_col_candidates = [
        "age",
        "age_years",
        "participant_age",
        "baseline_age",
    ]
    pid_candidates = ["participant_id", "Participant_ID", "participant"]
    uid_candidates = ["mindlamp_uid", "MindLAMP_ID", "mindlamp id", "lamp_id"]

    weekly_ids = weekly_df[["participant_id", "mindlamp_uid"]].drop_duplicates()

    for path in deduped:
        try:
            tab = load_table(path)
            if tab is None or len(tab) == 0:
                continue
        except Exception:
            continue

        cols = list(tab.columns)
        colmap = {c.lower(): c for c in cols}

        age_col = None
        for c in age_col_candidates:
            if c.lower() in colmap:
                age_col = colmap[c.lower()]
                break
        if age_col is None:
            continue

        pid_col = None
        for c in pid_candidates:
            if c.lower() in colmap:
                pid_col = colmap[c.lower()]
                break

        uid_col = None
        for c in uid_candidates:
            if c.lower() in colmap:
                uid_col = colmap[c.lower()]
                break

        if pid_col is None and uid_col is None:
            continue

        subcols = [age_col]
        if pid_col is not None:
            subcols.append(pid_col)
        if uid_col is not None:
            subcols.append(uid_col)

        sub = tab[subcols].copy()
        sub[age_col] = pd.to_numeric(sub[age_col], errors="coerce")

        merged = None
        if pid_col is not None:
            temp = weekly_ids.merge(
                sub[[pid_col, age_col]].rename(columns={pid_col: "participant_id"}),
                on="participant_id",
                how="left",
            )
            if temp[age_col].notna().sum() > 0:
                merged = temp[[age_col]]

        if merged is None and uid_col is not None:
            temp = weekly_ids.merge(
                sub[[uid_col, age_col]].rename(columns={uid_col: "mindlamp_uid"}),
                on="mindlamp_uid",
                how="left",
            )
            if temp[age_col].notna().sum() > 0:
                merged = temp[[age_col]]

        if merged is not None:
            ages = pd.to_numeric(merged[age_col], errors="coerce").dropna().drop_duplicates()
            if len(ages) > 0:
                return ages, str(path)

    return None, None


ages, age_source = find_age_series(home_valid)

if ages is not None:
    print("Age source:", age_source)
    print("Participants with age:", len(ages))
else:
    print("Could not find an age file automatically.")

# ============================================================
# PLOT
# ============================================================

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
})

fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))
fig.subplots_adjust(wspace=0.35, top=0.86)

# ---------- Panel A ----------
ax = axes[0]
a_vals = windows_per_participant.values

if len(a_vals) > 0:
    bins = np.arange(a_vals.min() - 0.5, a_vals.max() + 1.5, 1)
    ax.hist(a_vals, bins=bins, color="dimgray", edgecolor="black", linewidth=0.5)
ax.set_title("(A)", loc="left", fontweight="bold", fontsize=11, pad=2)
ax.set_xlabel("Number of weekly windows per participant")
ax.set_ylabel("Number of participants")
ax.set_xlim(left=0)
ax.text(
    0.03, 0.97,
    f"N = {len(a_vals)}",
    transform=ax.transAxes,
    ha="left", va="top"
)

# ---------- Panel B ----------
ax = axes[1]
b_vals = phq.values

if len(b_vals) > 0:
    phq_min = int(np.floor(np.nanmin(b_vals)))
    phq_max = int(np.ceil(np.nanmax(b_vals)))
    bins = np.arange(phq_min - 0.5, phq_max + 1.5, 1)
    ax.hist(b_vals, bins=bins, color="dimgray", edgecolor="black", linewidth=0.5)
ax.set_title("(B)", loc="left", fontweight="bold", fontsize=11, pad=2)
ax.set_xlabel("PHQ-9 total score")
ax.set_ylabel("Number of weekly windows")
ax.text(
    0.03, 0.97,
    f"N = {len(b_vals)}",
    transform=ax.transAxes,
    ha="left", va="top"
)

# ---------- Panel C ----------
ax = axes[2]
ax.set_title("(C)", loc="left", fontweight="bold", fontsize=11, pad=2)

if ages is not None and len(ages) > 0:
    c_vals = ages.values
    age_min = int(np.floor(np.nanmin(c_vals) / 5.0) * 5)
    age_max = int(np.ceil(np.nanmax(c_vals) / 5.0) * 5)
    bins = np.arange(age_min, age_max + 5, 5)
    ax.hist(c_vals, bins=bins, color="dimgray", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of participants")
    ax.text(
        0.03, 0.97,
        f"N = {len(c_vals)}",
        transform=ax.transAxes,
        ha="left", va="top"
    )
else:
    ax.axis("off")
    ax.text(
        0.5, 0.55,
        "Age data not found automatically",
        ha="center", va="center", fontsize=13
    )
    ax.text(
        0.5, 0.42,
        "If you want, send me the age-file path\nand I will patch the script.",
        ha="center", va="center", fontsize=11
    )

fig.suptitle("Cohort characteristics for home-time analysis", fontsize=20, fontweight="bold")

fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")

print()
print("Saved:")
print(OUT_PNG)
print(OUT_PDF)
