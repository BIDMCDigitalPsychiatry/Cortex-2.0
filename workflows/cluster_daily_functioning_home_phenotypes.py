from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/daily_functioning"
)

IN = (
    ROOT
    / "daily_functioning_slope_participant_phenotypes.csv"
)

OUT_TABLE = (
    ROOT
    / "daily_functioning_home_phenotype_clusters.csv"
)

OUT_SUMMARY = (
    ROOT
    / "daily_functioning_home_phenotype_cluster_summary.csv"
)

OUT_ENRICHMENT = (
    ROOT
    / "daily_functioning_home_phenotype_slope_enrichment.csv"
)

OUT_SILHOUETTE = (
    ROOT
    / "daily_functioning_home_phenotype_silhouette.csv"
)

OUT_PNG = (
    ROOT
    / "daily_functioning_home_phenotype_clusters.png"
)

OUT_PDF = (
    ROOT
    / "daily_functioning_home_phenotype_clusters.pdf"
)

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    IN,
    low_memory=False,
)

features = [
    "functioning_mean",
    "functioning_sd",
    "home_mean_pct",
    "home_sd_pct",
    "weekend_minus_weekday_home_pct",
    "depression_score_mean",
    "depression_score_sd",
    "anxiety_score_mean",
    "anxiety_score_sd",
]

for c in features:
    df[c] = pd.to_numeric(
        df[c],
        errors="coerce",
    )

print("=" * 100)
print("PHENOTYPE CLUSTERING")
print("=" * 100)

print("\nParticipants:", len(df))

print("\nFeatures:")
for c in features:
    print(
        f"  {c}: "
        f"{df[c].notna().sum()}/{len(df)} available"
    )

# ============================================================
# IMPUTE VERY LIMITED MISSINGNESS
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

X_imp = imputer.fit_transform(
    df[features]
)

scaler = StandardScaler()

X = scaler.fit_transform(
    X_imp
)

# ============================================================
# CHOOSE K BY SILHOUETTE
# ============================================================

rows = []

for k in range(2, 7):

    model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward",
    )

    labels = model.fit_predict(X)

    score = silhouette_score(
        X,
        labels,
    )

    cluster_sizes = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    rows.append(
        {
            "k": k,
            "silhouette_score": score,
            "smallest_cluster": min(
                cluster_sizes.values()
            ),
            "largest_cluster": max(
                cluster_sizes.values()
            ),
        }
    )

sil = pd.DataFrame(rows)

print("\nSilhouette scores:")
print(
    sil.round(4).to_string(
        index=False
    )
)

best_k = int(
    sil.loc[
        sil["silhouette_score"].idxmax(),
        "k",
    ]
)

print(
    "\nBest k by silhouette:",
    best_k,
)

sil.to_csv(
    OUT_SILHOUETTE,
    index=False,
)

# ============================================================
# FINAL CLUSTERING
# ============================================================

model = AgglomerativeClustering(
    n_clusters=best_k,
    linkage="ward",
)

df["cluster"] = (
    model.fit_predict(X) + 1
)

# ============================================================
# CLUSTER CHARACTERISTICS
# ============================================================

summary_rows = []

for cluster, g in df.groupby("cluster"):

    row = {
        "cluster": cluster,
        "n_participants": len(g),
    }

    for c in features:
        row[f"{c}_mean"] = (
            g[c].mean()
        )
        row[f"{c}_median"] = (
            g[c].median()
        )

    summary_rows.append(row)

summary = pd.DataFrame(
    summary_rows
)

print()
print("=" * 100)
print("CLUSTER CHARACTERISTICS")
print("=" * 100)

print(
    summary.round(3).to_string(
        index=False
    )
)

# ============================================================
# SLOPE-GROUP ENRICHMENT
#
# IMPORTANT:
# slope_group was NOT used for clustering.
# We examine it only after clusters are defined.
# ============================================================

enrichment = (
    pd.crosstab(
        df["cluster"],
        df["slope_group"],
        margins=False,
    )
)

for col in [
    "negative",
    "positive",
    "zero",
]:
    if col not in enrichment.columns:
        enrichment[col] = 0

enrichment = enrichment[
    [
        "negative",
        "positive",
        "zero",
    ]
]

enrichment["total"] = (
    enrichment.sum(axis=1)
)

enrichment["negative_pct"] = (
    100
    * enrichment["negative"]
    / enrichment["total"]
)

enrichment["positive_pct"] = (
    100
    * enrichment["positive"]
    / enrichment["total"]
)

enrichment["zero_pct"] = (
    100
    * enrichment["zero"]
    / enrichment["total"]
)

print()
print("=" * 100)
print("SLOPE DIRECTION BY DISCOVERED CLUSTER")
print("=" * 100)

print(
    enrichment.round(1).to_string()
)

# ============================================================
# MEAN SLOPE PER CLUSTER
# ============================================================

slope_summary = (
    df.groupby("cluster")[
        "within_slope_per_10pp"
    ]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std",
        ]
    )
)

print()
print("Slope distribution by cluster:")
print(
    slope_summary.round(4)
    .to_string()
)

# ============================================================
# PCA ONLY FOR VISUALIZATION
# ============================================================

pca = PCA(
    n_components=2
)

pcs = pca.fit_transform(
    X
)

df["PC1"] = pcs[:, 0]
df["PC2"] = pcs[:, 1]

print()
print(
    "PCA explained variance:",
    np.round(
        pca.explained_variance_ratio_,
        3,
    ),
)

# ============================================================
# FIGURE
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10

fig, ax = plt.subplots(
    figsize=(8.0, 6.3)
)

for cluster, g in df.groupby("cluster"):

    ax.scatter(
        g["PC1"],
        g["PC2"],
        s=38,
        alpha=0.75,
        label=(
            f"Cluster {cluster} "
            f"(n={len(g)})"
        ),
    )

ax.set_xlabel(
    "Phenotype PC1"
)

ax.set_ylabel(
    "Phenotype PC2"
)

ax.set_title(
    "Participant phenotypes from functioning, home behavior,\n"
    "depression, and anxiety"
)

ax.legend(
    frameon=False
)

ax.spines["top"].set_visible(
    False
)

ax.spines["right"].set_visible(
    False
)

fig.tight_layout()

fig.savefig(
    OUT_PNG,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
)

plt.close(fig)

# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUT_TABLE,
    index=False,
)

summary.to_csv(
    OUT_SUMMARY,
    index=False,
)

enrichment.reset_index().to_csv(
    OUT_ENRICHMENT,
    index=False,
)

print("\nSaved:")
print(" ", OUT_TABLE)
print(" ", OUT_SUMMARY)
print(" ", OUT_ENRICHMENT)
print(" ", OUT_SILHOUETTE)
print(" ", OUT_PNG)
print(" ", OUT_PDF)
