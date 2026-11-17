"""Shared Cortex-2.0 GPS quality control and temporal coverage diagnostics."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from cortex2_engine.features.gps_intervals import build_gps_intervals

@dataclass(frozen=True)
class GPSQCConfig:
    max_accuracy_m: float | None = None
    max_speed_kmh: float = 1000.0
    max_gap_ms: int = 60_000
    min_coverage_fraction: float | None = None


def clean_gps_fixes(fixes, config: GPSQCConfig = GPSQCConfig()):
    """Deduplicate and apply positional-accuracy QC; never silently fill gaps."""
    required={"timestamp", "latitude", "longitude", "accuracy"}
    missing=required-set(fixes.columns)
    if missing:
        raise ValueError(f"Missing GPS fields: {sorted(missing)}")
    f=fixes.copy()
    for c in required:
        f[c]=pd.to_numeric(f[c], errors="coerce")
    f=f.dropna(subset=list(required))
    f=f.loc[f.latitude.between(-90,90)&f.longitude.between(-180,180)&(f.accuracy>0)]
    if config.max_accuracy_m is not None:
        f=f.loc[f.accuracy<=config.max_accuracy_m]
    f=f.sort_values("timestamp",kind="mergesort").drop_duplicates("timestamp").reset_index(drop=True)
    return f


def quality_and_coverage(fixes, *, start_ms, end_ms, config=GPSQCConfig()):
    f=clean_gps_fixes(fixes, config=config)
    e=build_gps_intervals(f,window_start_ms=int(start_ms),window_end_ms=int(end_ms),max_gap_ms=config.max_gap_ms)
    if len(e):
        e=e.copy()
        e["plausible_speed"]=(pd.to_numeric(e.speed_kmh,errors="coerce")<=config.max_speed_kmh).fillna(False)
        e["qc_supported"]=e.supported.fillna(False)&e.plausible_speed
    else:
        e["qc_supported"]=pd.Series(dtype=bool)
    duration=int(end_ms)-int(start_ms)
    supported_ms=float(e.loc[e.qc_supported,"duration_ms"].sum()) if len(e) else 0.0
    fraction=supported_ms/duration if duration>0 else np.nan
    gaps=e.loc[~e.supported.fillna(False),"duration_ms"].to_numpy(float) if len(e) else np.array([])
    return f,e,{
        "n_fixes":len(f),"supported_duration_ms":supported_ms,
        "coverage_fraction":fraction,
        "n_long_gaps":len(gaps),"max_long_gap_ms":float(gaps.max()) if len(gaps) else 0.0,
        "meets_coverage_threshold":bool(np.isfinite(fraction) and
            (config.min_coverage_fraction is None or fraction>=config.min_coverage_fraction)),
    }
