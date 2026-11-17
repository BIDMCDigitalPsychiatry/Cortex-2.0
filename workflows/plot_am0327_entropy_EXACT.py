from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PID = "AMO327"
WINDOW = "AMO327__20231205"
DAY = "2023-12-01"

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical"
)

A_PATH = ROOT / "home_time/entropy_exact/AMO327__20231205_place_assignments.parquet"
P_PATH = ROOT / "home_time/entropy_exact/AMO327__20231205_places.parquet"
D_PATH = ROOT / "gps_clinical_weekly_daily_features.parquet"
H_PATH = ROOT / "home_time/participant_home_anchors.csv"

OUT = ROOT / "home_time/figures/AMO327_location_entropy_EXACT.png"

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10.5


# ============================================================
# EXACT SAVED PRODUCTION ASSIGNMENTS
# ============================================================

a = pd.read_parquet(A_PATH)
p = pd.read_parquet(P_PATH)

a["date"] = pd.to_datetime(a["date"]).dt.strftime("%Y-%m-%d")

day = a[a["date"].eq(DAY)].copy()

if day.empty:
    raise RuntimeError(f"No assignments for {DAY}")

occ = (
    day.groupby("place_candidate_id")["duration_ms"]
    .sum()
    .reset_index()
)

occ["fraction"] = occ["duration_ms"] / occ["duration_ms"].sum()

occ = (
    occ.sort_values(
        ["fraction", "place_candidate_id"],
        ascending=[False, True]
    )
    .reset_index(drop=True)
)

occ["rank"] = np.arange(1, len(occ) + 1)
occ["label"] = "Place " + occ["rank"].astype(str)
occ["pct"] = occ["fraction"] * 100

H = float(-(occ["fraction"] * np.log(occ["fraction"])).sum())
Neff = float(np.exp(H))


# ============================================================
# VERIFY AGAINST PRODUCTION DAILY OUTPUT
# ============================================================

d = pd.read_parquet(
    D_PATH,
    filters=[("participant_id", "==", PID)]
)

d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")

target = d[
    d["clinical_window_id"].astype(str).eq(WINDOW)
    & d["date"].eq(DAY)
]

if len(target) != 1:
    raise RuntimeError(f"Expected 1 production row, got {len(target)}")

target = target.iloc[0]

H_prod = float(target["location_entropy_nats"])

if abs(H - H_prod) > 1e-9:
    raise RuntimeError(
        f"ENTROPY MISMATCH: assignments={H:.9f}, production={H_prod:.9f}"
    )

print(f"VERIFIED: assignment entropy = {H:.9f}")
print(f"VERIFIED: production entropy = {H_prod:.9f}")


# ============================================================
# PLACE COORDINATES
# ============================================================

lat_candidates = [
    "representative_latitude",
    "latitude",
    "anchor_latitude",
]

lon_candidates = [
    "representative_longitude",
    "longitude",
    "anchor_longitude",
]

lat_col = next(c for c in lat_candidates if c in p.columns)
lon_col = next(c for c in lon_candidates if c in p.columns)

coords = (
    p[["place_candidate_id", lat_col, lon_col]]
    .drop_duplicates("place_candidate_id")
)

occ = occ.merge(
    coords,
    on="place_candidate_id",
    how="left"
)


# ============================================================
# PRIVACY-SAFE RELATIVE COORDINATES
# ============================================================

home = pd.read_csv(H_PATH, low_memory=False)
home = home[home["participant_id"].astype(str).eq(PID)]

hlat = next(
    c for c in [
        "home_latitude",
        "representative_latitude",
        "home_center_latitude"
    ]
    if c in home.columns
)

hlon = next(
    c for c in [
        "home_longitude",
        "representative_longitude",
        "home_center_longitude"
    ]
    if c in home.columns
)

lat0 = float(home.iloc[0][hlat])
lon0 = float(home.iloc[0][hlon])

R = 6371000.0

occ["x"] = (
    np.radians(occ[lon_col] - lon0)
    * R
    * np.cos(np.radians(lat0))
    / 1000
)

occ["y"] = (
    np.radians(occ[lat_col] - lat0)
    * R
    / 1000
)


# ============================================================
# DAILY SERIES
# ============================================================

daily = d[
    (d["date"] >= "2023-11-29")
    & (d["date"] <= "2023-12-12")
].copy()

daily = (
    daily.sort_values(["date", "clinical_window_id"])
    .drop_duplicates("date")
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 6.2)
)


# ------------------------------------------------------------
# A. Exact detected places
# ------------------------------------------------------------

ax = axes[0]

sizes = 30 + occ["pct"] * 10

ax.scatter(
    occ["x"],
    occ["y"],
    s=sizes,
    color="0.72",
    edgecolors="0.4",
    linewidths=0.6,
    alpha=0.8
)

dom = occ.iloc[0]

ax.scatter(
    dom["x"],
    dom["y"],
    s=30 + dom["pct"] * 10,
    color="0.25",
    edgecolors="black",
    linewidths=1.2,
    zorder=4
)

# label only top 4
offsets = [
    (10, 16),     # Place 1
    (10, -20),    # Place 2
    (18, 14),     # Place 3
    (-42, -6),    # Place 4
]

for (_, r), off in zip(occ.head(4).iterrows(), offsets):
    ax.annotate(
        f"{r['label']}  {r['pct']:.1f}%",
        (r["x"], r["y"]),
        xytext=off,
        textcoords="offset points",
        fontsize=8.5,
        ha="right" if off[0] < 0 else "left",
        arrowprops=dict(
            arrowstyle="-",
            linewidth=0.6,
            color="0.4"
        ),
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
            pad=1
        )
    )

ax.set_title("A. Detected places and assigned time\nDec 01")
ax.set_xlabel("East–west position relative to home (km)")
ax.set_ylabel("North–south position relative to home (km)")
ax.set_aspect("equal", adjustable="datalim")

ax.text(
    0.02, 0.02,
    "Bubble size = assigned observed time",
    transform=ax.transAxes,
    fontsize=8.5
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ------------------------------------------------------------
# B. Exact probabilities entering entropy
# ------------------------------------------------------------

ax = axes[1]

top = occ.head(7)[["label", "pct"]].copy()

if len(occ) > 7:
    other = pd.DataFrame({
        "label": ["Other places"],
        "pct": [occ.iloc[7:]["pct"].sum()]
    })
    top = pd.concat([top, other], ignore_index=True)

y = np.arange(len(top))

ax.barh(
    y,
    top["pct"],
    color="0.38"
)

ax.set_yticks(y)
ax.set_yticklabels(top["label"])
ax.invert_yaxis()
ax.set_xlabel("% of assigned place time")
ax.set_title("B. Place-time distribution used for entropy")

for i, v in enumerate(top["pct"]):
    ax.text(
        v + 0.5,
        i,
        f"{v:.1f}%",
        va="center",
        fontsize=8.5
    )

ax.text(
    0.98, 0.10,
    (
        f"Entropy H = {H_prod:.2f} nats\n"
        f"Effective places = {Neff:.2f}\n"
        f"Dominant place = {occ.iloc[0]['pct']:.1f}%\n"
        f"Observed places = {int(target['n_observed_places'])}\n"
        f"Assigned place time = "
        f"{100*float(target['place_time_fraction_of_supported']):.1f}% "
        "of supported GPS"
    ),
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=9,
    bbox=dict(
        boxstyle="round,pad=0.3",
        facecolor="white",
        edgecolor="0.8"
    )
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ------------------------------------------------------------
# C. Daily production entropy
# ------------------------------------------------------------

ax = axes[2]

x = np.arange(len(daily))

vals = pd.to_numeric(
    daily["location_entropy_nats"],
    errors="coerce"
).to_numpy()

day_pos = daily.index[daily["date"].eq(DAY)]

if len(day_pos) != 1:
    raise RuntimeError("Could not identify Dec 01 in daily series")

day_pos = int(day_pos[0])

for i, v in enumerate(vals):
    if np.isfinite(v):
        ax.bar(
            i,
            v,
            width=0.72,
            color="0.25" if i == day_pos else "0.72"
        )

ax.text(
    day_pos,
    vals[day_pos] + 0.03,
    f"{vals[day_pos]:.2f}",
    ha="center",
    fontweight="bold",
    fontsize=9
)

ax.set_xticks(x)
ax.set_xticklabels(
    [
        pd.Timestamp(z).strftime("%b %d")
        for z in daily["date"]
    ],
    rotation=45,
    ha="right"
)

ax.set_ylabel("Location entropy (nats)")
ax.set_title("C. Daily Cortex-2.0 location entropy")

ax.text(
    0.98, 0.96,
    (
        "14-day window\n"
        f"Median daily entropy = "
        f"{daily['location_entropy_nats'].median():.2f} nats"
    ),
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=9
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# coverage strip
pos = ax.get_position()

cov_ax = fig.add_axes(
    [
        pos.x0,
        pos.y0 - 0.042,
        pos.width,
        0.025
    ]
)

cov_ax.set_xlim(-0.5, len(daily)-0.5)
cov_ax.set_ylim(0, 1)

for i, cov in enumerate(daily["supported_fraction_of_day"]):
    if pd.notna(cov):
        shade = np.clip(
            0.92 - 0.72 * float(cov),
            0.18,
            0.92
        )

        cov_ax.add_patch(
            Rectangle(
                (i - 0.36, 0.12),
                0.72,
                0.76,
                facecolor=str(shade),
                edgecolor="none"
            )
        )

cov_ax.set_xticks([])
cov_ax.set_yticks([])

for sp in cov_ax.spines.values():
    sp.set_visible(False)

cov_ax.text(
    -0.02, 0.5,
    "GPS coverage",
    transform=cov_ax.transAxes,
    ha="right",
    va="center",
    fontsize=8
)


fig.suptitle(
    (
        "AMO327: How Cortex-2.0 calculates location entropy\n"
        "Observed time is converted into a probability distribution "
        "across detected places"
    ),
    fontsize=16
)

fig.subplots_adjust(
    top=0.78,
    bottom=0.27,
    wspace=0.38
)

fig.savefig(
    OUT,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

print()
print("NEW EXACT FIGURE SAVED:")
print(OUT)
