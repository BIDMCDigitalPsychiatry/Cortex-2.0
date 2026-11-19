from pathlib import Path
from PIL import Image, ImageOps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

OUT = Path.home() / "cortex2_visualizations/outputs"

SOURCE_ABCD = OUT / "AMO327_20231212_01_observed_gps_to_trips.png"
SOURCE_E = OUT / "AMO327_20231212_imputation_panel_E.png"
DEST = OUT / "AMO327_GPS_mobility_pipeline_with_imputation.png"

for p in (SOURCE_ABCD, SOURCE_E):
    if not p.exists():
        raise FileNotFoundError(f"Missing source figure: {p}")

original = Image.open(SOURCE_ABCD).convert("RGB")
imputation = Image.open(SOURCE_E).convert("RGB")

def crop_fraction(image, bounds):
    """Crop normalized image coordinates: left, top, right, bottom."""
    w, h = image.size
    l, t, r, b = bounds
    return image.crop((
        round(l * w), round(t * h),
        round(r * w), round(b * h)
    ))

# These regions correspond to the four existing plot panels.
# They can be fine-tuned if the source figure's margins change.
A = crop_fraction(original, (0.045, 0.080, 0.505, 0.480))
B = crop_fraction(original, (0.505, 0.080, 0.995, 0.480))
C = crop_fraction(original, (0.045, 0.475, 0.505, 0.900))
D = crop_fraction(original, (0.505, 0.475, 0.995, 0.900))

# Panel E includes both donor and reconstruction plots, along
# with the labels and provenance explanation.
E = crop_fraction(imputation, (0.015, 0.015, 0.995, 0.985))

fig = plt.figure(figsize=(20, 17), facecolor="white")
gs = GridSpec(
    3, 2,
    figure=fig,
    height_ratios=[1.0, 1.28, 0.95],
    width_ratios=[1.0, 1.18],
    left=.045, right=.965,
    top=.895, bottom=.075,
    hspace=.22, wspace=.09
)

def show_panel(ax, image):
    ax.imshow(image, interpolation="nearest")
    ax.set_axis_off()

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_e = fig.add_subplot(gs[1, 1])
ax_d = fig.add_subplot(gs[2, :])

show_panel(ax_a, A)
show_panel(ax_b, B)
show_panel(ax_c, C)
show_panel(ax_e, E)
show_panel(ax_d, D)

# Main figure heading and subtitle.
fig.text(
    .5, .972,
    "How GPS becomes a mobility measure",
    ha="center", va="top",
    fontsize=24, fontweight="bold", color="#183B52"
)
fig.text(
    .5, .945,
    "AMO327  |  December 6–12, 2023  |  Real GPS observations",
    ha="center", va="top",
    fontsize=12, color="#607B8F"
)

# Clarify what the dotted line in C represents.
ax_c.text(
    .02, -.035,
    "Dotted line marks missing GPS, not an observed route.",
    transform=ax_c.transAxes,
    ha="left", va="top",
    fontsize=9.5, color="#A9742A"
)

# Explain the transformation and avoid claiming the imputed path is true.
ax_e.text(
    .5, -.045,
    "Donor trajectory → rotation, scaling and time adjustment",
    transform=ax_e.transAxes,
    ha="center", va="top",
    fontsize=10.5, color="#526C7C"
)

fig.text(
    .5, .045,
    "Imputed GPS positions are model-generated candidates, not "
    "observed or independently validated travel routes.",
    ha="center", fontsize=10.5, color="#526C7C"
)

fig.savefig(
    DEST,
    dpi=230,
    facecolor="white"
)
plt.close(fig)

print("Saved:", DEST)
print("Original source figures unchanged.")
