from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from cortex2_engine.features.gps_life_space import build_life_space_polygon, cartesian_transform, compute_geographic_centroid

"""
Smoke test for life space polygon. Only for visual inpsection, no assertions or changes to the data or codebase.
"""


USER_GOOD = 'U0015175710'
USER_BAD = 'U0010007464'
USERS_FIRST_10 = ['U0010007464', 'U0014599842', 'U0015175710', 'U0069924528', 'U0089491651', 'U0133360788', 'U0151129791', 'U0155183941', 'U0188390646', 'U0203726787']
GPS_PATH = Path('/Users/henryrenninger/dropbox_ali/Digital Clinic/passive/gps')
output_directory = Path('/Users/henryrenninger/BIDMC/Cortex/Life Space Smoke Test/Plots')


def load_user_gps(user, gps_path):
    files = list(gps_path.glob(f'gps_{user}_*.csv'))
    df = pd.concat(
        (pd.read_csv(f, index_col=0) for f in files),
        ignore_index=True,
    )
    gps = df.sort_values('timestamp').reset_index(drop=True)
    gps.rename(columns={'latitude': 'representative_latitude', 'longitude': 'representative_longitude'}, inplace=True)
    return gps

def plot_life_space(points, hull, area):
    lats = points['representative_latitude'].to_numpy(float)
    lons = points['representative_longitude'].to_numpy(float)
    h_lats = hull[:, 0]
    h_lons = hull[:, 1]

    lat_0, lon_0 = compute_geographic_centroid(lats, lons)
    px, py = cartesian_transform(lats, lons, lat_0, lon_0)
    hx, hy = cartesian_transform(h_lats, h_lons, lat_0, lon_0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(px / 1000, py / 1000, s=4, alpha=0.3, color='gray', label='GPS points')
    transposed_hull = np.array([hx / 1000, hy / 1000]).T
    ax.add_patch(Polygon(transposed_hull, closed=True, facecolor='blue', edgecolor='blue', alpha=0.2, linewidth=1.8))
    ax.scatter(hx / 1000, hy / 1000, s=25, color='blue', label='Hull vertices')
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('West -> East (km)')
    ax.set_ylabel('South -> North (km)')
    ax.set_title(f'Life space: {area / 1000000:.2f} km^2')
    ax.legend()
    ax.grid(alpha=0.15)
    #plt.show()

# Run on the first 10 users and save plots
for user in USERS_FIRST_10:
    gps = load_user_gps(user, GPS_PATH)
    area, hull = build_life_space_polygon(gps)
    points = gps.dropna(subset=['representative_latitude', 'representative_longitude'])
    plot_life_space(points, hull, area)
    plt.savefig(output_directory / f'life_space_{user}.png')
    plt.close()

# gps = load_user_gps(USER_GOOD, GPS_PATH)
# area, hull = build_life_space_polygon(gps)

# print(f'Life space area: {area:.2f} m^2')
# print(f'Life space hull vertices:\n{hull}')

# points = gps.dropna(subset=['representative_latitude', 'representative_longitude'])
# plot_life_space(points, hull, area)


# unique_ids = {
#     f.stem.split("_")[1] 
#     for f in GPS_PATH.glob("gps_*_*.csv") 
#     if f.is_file()
# }

# print(sorted(unique_ids))