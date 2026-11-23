import runpy
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from cortex2_engine.features.gps_gap_imputation import (
    ImputationConfig,
    _matching_route_sections,
    _bridge_matched_route,
)

ctx = runpy.run_path(
    "workflows/visualization_diagnostics/check_dec7_reverse_route.py"
)

left, right = ctx["left"], ctx["right"]
cfg = ImputationConfig()

workflow = runpy.run_path(
    "workflows/run_gps_imputed_clinical_features.py"
)
trip_donors, stay_donors, stats = (
    workflow["build_participant_donors"](ctx["clean"])
)

matches = _matching_route_sections(
    left, right, trip_donors, cfg
)
assert matches

score, _, donor = matches[0]
filled = _bridge_matched_route(left, right, donor, cfg)
assert filled is not None and not filled.empty

project = ctx["project"]

# Complete reconstructed trajectory, including real gap endpoints.
import pandas as pd
reconstructed = pd.concat([
    pd.DataFrame([left]),
    filled,
    pd.DataFrame([right])
], ignore_index=True)

dx, dy = project(donor)
rx, ry = project(reconstructed)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))

axes[0].plot(
    dx / 1000, dy / 1000,
    color="#2878A8", lw=2
)
axes[0].set_title(
    "A  Automatically selected observed donor",
    loc="left", weight="bold"
)

axes[1].plot(
    dx / 1000, dy / 1000,
    color="#2878A8", lw=2,
    alpha=.35, label="Observed donor"
)
axes[1].plot(
    rx / 1000, ry / 1000,
    color="#D76A35", lw=2.5,
    label="Reconstructed gap"
)
axes[1].scatter(
    [rx[0], rx[-1]],
    [ry[0], ry[-1]],
    s=45, color="#222222",
    zorder=5, label="Actual gap endpoints"
)
axes[1].legend(frameon=False, fontsize=9)
axes[1].set_title(
    "B  December 12 reconstructed route",
    loc="left", weight="bold"
)

all_x = np.r_[dx, rx] / 1000
all_y = np.r_[dy, ry] / 1000
pad = .15

for ax in axes:
    ax.set_xlim(all_x.min()-pad, all_x.max()+pad)
    ax.set_ylim(all_y.min()-pad, all_y.max()+pad)
    ax.set_aspect("equal")
    ax.set_xlabel("East–west (km)")
    ax.set_ylabel("North–south (km)")
    ax.grid(alpha=.15)

fig.suptitle(
    "AMO327 | Geographic donor reconstruction",
    weight="bold", fontsize=16
)

fig.tight_layout(rect=[0, 0, 1, .94])

out = (
    Path.home()
    / "cortex2_visualizations/outputs"
    / "AMO327_Dec12_compact_reconstruction.png"
)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, facecolor="white")
plt.close(fig)

print("Saved:", out)
