"""Analyze AMO327 reconstruction loss by donor type and geometry."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = (
    Path.home() / "cortex2_visualizations/outputs"
    / "AMO327_all_gap_resampling"
)

data = pd.read_csv(ROOT / "all_imputation_draws.csv")

keys = [
    "clinical_window_id",
    "gap_start_ms",
    "gap_end_ms",
]

required = keys + [
    "donor_type", "loss_pct", "duration_min",
    "full_transformed_m", "resampled_m",
    "endpoint_distance_m", "reconstructed_speed_kmh",
]

missing = set(required) - set(data.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

data = data.loc[data.donor_type.isin(["trip", "stay"])].copy()

if data.empty:
    raise RuntimeError("No trip or stay reconstructions found.")

# A donor can contain substantial movement even when its
# endpoints are close. This measures route indirectness,
# not whether the patient was actually stationary.
data["endpoint_to_path_ratio"] = (
    data.endpoint_distance_m
    / data.full_transformed_m.replace(0, np.nan)
)

data["endpoint_to_path_ratio"] = (
    data.endpoint_to_path_ratio.clip(lower=0)
)

# Each gap × donor type gets one summary, preventing gaps with
# many repeated draws from dominating the comparison.
group_cols = keys + ["donor_type"]

by_gap_type = data.groupby(
    group_cols, as_index=False
).agg(
    n_draws=("loss_pct", "size"),
    median_loss_pct=("loss_pct", "median"),
    median_full_m=("full_transformed_m", "median"),
    median_resampled_m=("resampled_m", "median"),
    median_speed_kmh=("reconstructed_speed_kmh", "median"),
    median_indirectness=("endpoint_to_path_ratio", "median"),
    duration_min=("duration_min", "first"),
)

by_gap_type.to_csv(
    ROOT / "gap_by_donor_type.csv", index=False
)

# Classify unique gaps by which donor types were actually used.
type_sets = (
    data.groupby(keys)["donor_type"]
    .agg(lambda v: "+".join(sorted(set(v))))
    .reset_index(name="donor_types_used")
)

type_counts = (
    type_sets.donor_types_used
    .value_counts()
    .rename_axis("donor_types_used")
    .reset_index(name="unique_gaps")
)

type_counts.to_csv(
    ROOT / "gap_donor_type_counts.csv", index=False
)

# Summary: each gap contributes once per donor type represented.
summary = by_gap_type.groupby("donor_type").agg(
    gaps=("median_loss_pct", "size"),
    median_loss_pct=("median_loss_pct", "median"),
    mean_loss_pct=("median_loss_pct", "mean"),
    p90_loss_pct=(
        "median_loss_pct", lambda s: s.quantile(.90)
    ),
    pct_gaps_loss_gt10=(
        "median_loss_pct", lambda s: 100*(s > 10).mean()
    ),
    pct_gaps_loss_gt20=(
        "median_loss_pct", lambda s: 100*(s > 20).mean()
    ),
    median_full_m=("median_full_m", "median"),
    median_resampled_m=("median_resampled_m", "median"),
    median_speed_kmh=("median_speed_kmh", "median"),
    median_indirectness=("median_indirectness", "median"),
).reset_index()

summary["draws"] = summary.donor_type.map(
    data.donor_type.value_counts()
)

summary.to_csv(
    ROOT / "donor_type_summary.csv", index=False
)

# Paired analysis: only gaps for which both donor types
# appeared among the successful reconstructions.
wide = by_gap_type.pivot(
    index=keys,
    columns="donor_type",
    values="median_loss_pct",
).dropna(subset=["trip", "stay"])

paired = wide.reset_index().copy()

if not paired.empty:
    paired["stay_minus_trip_loss_pct"] = (
        paired["stay"] - paired["trip"]
    )

paired.to_csv(
    ROOT / "paired_trip_stay_gaps.csv",
    index=False
)

print("\nAMO327 — TRIP VS STAY DONOR ANALYSIS")
print("=" * 58)

print("\nUNIQUE GAPS BY DONOR TYPES USED")
print(type_counts.to_string(index=False))

print("\nSUMMARY BY DONOR TYPE")
print(summary.round(2).to_string(index=False))

print("\nWITHIN-GAP COMPARISON")
print("Gaps with both donor types:", len(paired))

if not paired.empty:
    print(
        "Median stay-minus-trip loss:",
        f"{paired.stay_minus_trip_loss_pct.median():.1f}",
        "percentage points"
    )
    print(
        "Gaps where stay has greater loss:",
        f"{100*(paired.stay_minus_trip_loss_pct > 0).mean():.1f}%"
    )

print("\nHIGH-LOSS TRIP-DONOR GAPS")
trip = by_gap_type.loc[
    by_gap_type.donor_type.eq("trip")
]
print(
    trip.nlargest(8, "median_loss_pct")[
        [
            "gap_start_ms", "duration_min",
            "n_draws", "median_loss_pct",
            "median_full_m", "median_resampled_m",
            "median_indirectness"
        ]
    ].round(2).to_string(index=False)
)

# Figure: simple, interpretable comparisons.
fig, (a, b) = plt.subplots(
    1, 2, figsize=(12.5, 4.8)
)
fig.subplots_adjust(
    left=.08, right=.97, top=.82,
    bottom=.17, wspace=.26
)

colors = {
    "trip": "#287EAD",
    "stay": "#27917D"
}

for i, kind in enumerate(["trip", "stay"]):
    vals = by_gap_type.loc[
        by_gap_type.donor_type.eq(kind),
        "median_loss_pct"
    ].dropna().to_numpy()

    if not len(vals):
        continue

    a.boxplot(
        [vals],
        positions=[i],
        widths=.34,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#173C52", linewidth=2),
        boxprops=dict(
            facecolor=colors[kind], alpha=.25,
            edgecolor=colors[kind]
        ),
        whiskerprops=dict(color=colors[kind]),
        capprops=dict(color=colors[kind])
    )

    # Show every gap-level estimate, without random jitter.
    spread = np.linspace(-.12, .12, len(vals))
    a.scatter(
        i + spread,
        np.sort(vals),
        s=17, color=colors[kind], alpha=.65,
        edgecolors="none", zorder=4
    )

a.set_xticks([0, 1], ["Trip donors", "Stay donors"])
a.set_ylabel("Median path length lost per gap (%)")
a.set_title(
    "A  Resampling loss by donor type",
    loc="left", fontsize=12, weight="bold"
)

# Endpoint-to-path ratio is close to zero for looping or
# indirect movement, and close to one for nearly direct paths.
for kind in ["trip", "stay"]:
    subset = by_gap_type.loc[
        by_gap_type.donor_type.eq(kind)
    ]
    b.scatter(
        subset.median_indirectness,
        subset.median_loss_pct,
        label=f"{kind.capitalize()} donors",
        s=38, color=colors[kind],
        alpha=.75, edgecolors="white",
        linewidths=.45
    )

b.set_xlabel(
    "Endpoint displacement / full donor path length"
)
b.set_ylabel("Median path length lost per gap (%)")
b.set_title(
    "B  Loss versus route indirectness",
    loc="left", fontsize=12, weight="bold"
)
b.legend(frameon=False, fontsize=9)

for ax in (a, b):
    ax.grid(color="#E6ECF0", linewidth=.7)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

fig.suptitle(
    "AMO327 | Does donor movement type affect resampling loss?",
    fontsize=16, weight="bold", y=.965
)

fig.text(
    .5, .055,
    "Each point summarizes one gap within a donor type; "
    "loss is relative to the transformed donor, not ground truth.",
    ha="center", fontsize=9, color="#597080"
)

fig.savefig(
    ROOT / "trip_vs_stay_resampling.png",
    dpi=210, facecolor="white"
)
plt.close(fig)

print("\nSaved results to:", ROOT)
