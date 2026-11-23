from pathlib import Path
import shutil

p = Path("src/cortex2_engine/features/gps_gap_imputation.py")
s = p.read_text()

marker = "# CORTEX2_GEOGRAPHIC_ROUTE_V1"

if marker in s:
    raise SystemExit("Route-matching patch already applied.")

helper = '''
# CORTEX2_GEOGRAPHIC_ROUTE_V1
def _matching_route_sections(left, right, trip_donors, cfg):
    """Find observed route subsections near both gap endpoints."""
    matches = []

    for donor_index, donor in enumerate(trip_donors):
        d = _require_fixes(donor)
        if len(d) < cfg.min_donor_fixes:
            continue
        if "imputed" in d and d.imputed.any():
            continue

        times = d.timestamp.to_numpy(dtype=np.int64)
        if np.diff(times).max(initial=0) > cfg.max_supported_gap_ms:
            continue

        lat = d.latitude.to_numpy(dtype=float)
        lon = d.longitude.to_numpy(dtype=float)

        def distances(anchor):
            a = np.deg2rad(float(anchor.latitude))
            b = np.deg2rad(lat)
            dl = np.deg2rad(lon - float(anchor.longitude))
            h = (np.sin((b-a)/2)**2
                 + np.cos(a)*np.cos(b)*np.sin(dl/2)**2)
            return 2*EARTH_RADIUS_M*np.arcsin(
                np.sqrt(np.clip(h, 0, 1))
            )

        ds = distances(left)
        de = distances(right)

        k = min(8, len(d))
        starts = np.argpartition(ds, k-1)[:k]
        ends = np.argpartition(de, k-1)[:k]

        for i in starts:
            for j in ends:
                if abs(int(i)-int(j))+1 < cfg.min_donor_fixes:
                    continue

                a, b = sorted((int(i), int(j)))
                if times[b]-times[a] < cfg.min_template_span_ms:
                    continue

                score = float(ds[i]+de[j])

                # Compare endpoint mismatch with gap displacement,
                # without imposing a donor-duration limit.
                displacement = float(_haversine(
                    left.latitude, left.longitude,
                    right.latitude, right.longitude
                ))
                if score >= displacement:
                    continue

                route = d.iloc[a:b+1].copy().reset_index(drop=True)
                if i > j:
                    route = route.iloc[::-1].reset_index(drop=True)

                matches.append((score, donor_index, route))

    matches.sort(key=lambda x: x[0])
    return matches


def _bridge_matched_route(left, right, route, cfg):
    """Preserve geographically matched geometry; allocate time by distance."""
    d = route.copy()
    if len(d) < cfg.min_donor_fixes:
        return None

    dt = int(right.timestamp-left.timestamp)
    if dt <= cfg.max_supported_gap_ms:
        return None

    lat0 = float(left.latitude)
    lon0 = float(left.longitude)
    coslat = np.cos(np.deg2rad(lat0))

    if abs(coslat) < 1e-8:
        return None

    x = (EARTH_RADIUS_M
         * np.deg2rad(d.longitude.to_numpy(dtype=float)-lon0)
         * coslat)
    y = (EARTH_RADIUS_M
         * np.deg2rad(d.latitude.to_numpy(dtype=float)-lat0))

    x -= x[0]
    y -= y[0]

    step = np.hypot(np.diff(x), np.diff(y))
    progress = np.r_[0.0, np.cumsum(step)]

    if progress[-1] <= 0:
        return None

    progress /= progress[-1]
    keep = np.r_[True, np.diff(progress) > 1e-12]

    progress = progress[keep]
    x = x[keep]
    y = y[keep]
    accuracy = d.accuracy.to_numpy(dtype=float)[keep]

    if len(progress) < 2:
        return None

    xe = EARTH_RADIUS_M*np.deg2rad(
        float(right.longitude)-lon0
    )*coslat
    ye = EARTH_RADIUS_M*np.deg2rad(
        float(right.latitude)-lat0
    )

    # Gradually correct endpoint offset without rotating/scaling
    # the whole observed trajectory.
    x += progress*(xe-x[-1])
    y += progress*(ye-y[-1])

    n = max(2, int(np.ceil(
        dt/min(cfg.max_supported_gap_ms, 5000)
    )))
    grid = np.linspace(0, 1, n+1)
    timestamps = np.rint(
        int(left.timestamp)+dt*grid
    ).astype(np.int64)

    mid = (
        (timestamps > int(left.timestamp))
        & (timestamps < int(right.timestamp))
    )

    return pd.DataFrame({
        "timestamp": timestamps[mid],
        "latitude": lat0 + np.rad2deg(
            np.interp(grid[mid], progress, y)/EARTH_RADIUS_M
        ),
        "longitude": lon0 + np.rad2deg(
            np.interp(grid[mid], progress, x)
            /(EARTH_RADIUS_M*coslat)
        ),
        "accuracy": np.interp(
            grid[mid], progress, accuracy
        ),
        "imputed": True,
    })

'''

anchor = "def impute_timelines(\n"
assert s.count(anchor) == 1
s = s.replace(anchor, helper + anchor, 1)

old = '''            piece = None
            chosen = None
            kind = "unresolved"
            if candidate_pairs:
'''

new = '''            piece = None
            chosen = None
            kind = "unresolved"
            geographic_error_m = np.nan

            route_matches = _matching_route_sections(
                left, right, donors, cfg
            )

            for score, donor_index, route in route_matches:
                proposed = _bridge_matched_route(
                    left, right, route, cfg
                )
                if proposed is not None:
                    piece = proposed
                    chosen = donor_index
                    kind = "geographic_trip"
                    geographic_error_m = score
                    break

            if piece is None and candidate_pairs:
'''

assert s.count(old) == 1
s = s.replace(old, new, 1)

old = '''                "n_candidate_donors": len(candidate_pairs),
            })'''

new = '''                "n_candidate_donors": len(candidate_pairs),
                "geographic_endpoint_error_m": geographic_error_m,
            })'''

assert s.count(old) == 1, "Diagnostics block differs from expected source"
s = s.replace(old, new, 1)

compile(s, str(p), "exec")

backup = p.with_suffix(
    p.suffix + ".before_geographic_route.bak"
)
if backup.exists():
    raise SystemExit(f"Backup already exists: {backup}")

shutil.copy2(p, backup)
p.write_text(s)

print("Patched:", p)
print("Backup:", backup)
