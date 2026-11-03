"""Daily missingness phenotype: a six-class label of how a day's data went missing.

Construct
---------
Which missingness regime dominated a participant-day, using the two onset
types established in ``docs/gps_missingness_wifi_module.md``:

- *GPS-specific* gap hours: GPS absent while the accelerometer is observed.
- *Shared* gap hours: GPS and accelerometer both absent.

plus battery context from ``screen.battery_level_fraction``. The full
specification is ``docs/features/missingness.md``.

Classes (first rule that applies wins)
--------------------------------------
1. ``no_signal`` — neither GPS nor accelerometer observed in any eligible hour.
2. ``complete`` — gap hours (either stream missing) are at most
   ``max_gap_fraction_complete`` of eligible hours.
3. ``battery_depletion`` — shared gap hours are at least
   ``dominance_fraction`` of gap hours, **and** at least half of the day's
   shared onsets follow an hour whose last battery reading is at most
   ``low_battery_fraction``.
4. ``shared_interruption`` — shared gap hours dominate, without low-battery
   onsets (the app or OS suspended sensing).
5. ``gps_specific_dropout`` — GPS-only gap hours are at least
   ``dominance_fraction`` of gap hours.
6. ``mixed`` — anything else (accelerometer-only gaps, or no regime
   dominates).

Eligibility
-----------
Hours where either source is unavailable or partial (the ``*_source_unavailable``
and ``*_partial_source`` flags of the hourly table) are not eligible. They
are neither gaps nor observations. A day needs ``min_eligible_hours``
eligible hours, otherwise the label is NaN with
``insufficient_observability`` (``source_absent`` if no hour is eligible
because the sources are unavailable).

What this is NOT
----------------
Not a causal mechanism (MCAR/MAR/MNAR) and not a diagnosis of device
failure. ``battery_depletion`` means "shared gaps that began at low battery",
which is consistent with, but does not prove, the phone switching off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cortex2_engine.features.status import FeatureStatus
from cortex2_engine.missingness.transitions import add_hourly_missingness_transitions
from cortex2_engine.quality.completeness import (
    completeness_weight_from_mask,
    multi_stream_weight,
)
from cortex2_engine.time import HOUR_MS, local_dates

PHENOTYPES = (
    "no_signal",
    "complete",
    "battery_depletion",
    "shared_interruption",
    "gps_specific_dropout",
    "mixed",
)

_SOURCE_FLAGS = (
    "gps_source_unavailable",
    "acc_source_unavailable",
    "gps_partial_source",
    "acc_partial_source",
)


def classify_day(
    *,
    n_eligible_hours: int,
    gps_specific_gap_hours: int,
    shared_gap_hours: int,
    acc_only_gap_hours: int,
    n_shared_onsets: int,
    n_low_battery_shared_onsets: int,
    max_gap_fraction_complete: float = 0.1,
    dominance_fraction: float = 0.6,
) -> str:
    """Apply the six-class rules to one day's hour counts.

    Parameters
    ----------
    n_eligible_hours
        Eligible hours (> 0).
    gps_specific_gap_hours, shared_gap_hours, acc_only_gap_hours
        Eligible hours with only GPS, both, or only ACC missing.
    n_shared_onsets, n_low_battery_shared_onsets
        Shared onsets, and those preceded by low battery.
    max_gap_fraction_complete
        Largest gap share still labelled ``complete`` (0.1).
    dominance_fraction
        Share of gap hours a regime needs to dominate (0.6).

    Returns
    -------
    str
        One of :data:`PHENOTYPES`.
    """

    if n_eligible_hours <= 0:
        raise ValueError("n_eligible_hours must be > 0")

    gap_hours = gps_specific_gap_hours + shared_gap_hours + acc_only_gap_hours

    if shared_gap_hours == n_eligible_hours:
        return "no_signal"

    if gap_hours <= max_gap_fraction_complete * n_eligible_hours:
        return "complete"

    if shared_gap_hours >= dominance_fraction * gap_hours:
        if n_shared_onsets and n_low_battery_shared_onsets * 2 >= n_shared_onsets:
            return "battery_depletion"
        return "shared_interruption"

    if gps_specific_gap_hours >= dominance_fraction * gap_hours:
        return "gps_specific_dropout"

    return "mixed"


def daily_missingness_phenotype(
    hourly: pd.DataFrame,
    *,
    tz: str,
    day_start: str = "00:00",
    min_eligible_hours: int = 12,
    max_gap_fraction_complete: float = 0.1,
    dominance_fraction: float = 0.6,
    low_battery_fraction: float = 0.15,
) -> pd.DataFrame:
    """Label each participant-day with its missingness phenotype.

    Parameters
    ----------
    hourly
        Episode hourly table (``missingness.hourly_context`` output, or any
        table with the columns ``add_hourly_missingness_transitions``
        requires plus ``start_ms``; ``battery_last`` is optional).
    tz
        IANA timezone for local days.
    day_start
        Local clock time at which a day starts.
    min_eligible_hours
        Eligible hours a day needs (12).
    max_gap_fraction_complete, dominance_fraction
        See :func:`classify_day`.
    low_battery_fraction
        Battery level at or below which an onset counts as low-battery
        (0.15).

    Returns
    -------
    pandas.DataFrame
        One row per ``episode_id`` × ``date_local`` with
        ``missingness_phenotype``, ``missingness_phenotype_status``, hour
        and onset counts, and ``completeness_weight`` (geometric mean of the
        GPS and ACC hourly weights over eligible hours).
    """

    if not 0 < dominance_fraction <= 1 or not 0 <= max_gap_fraction_complete < 1:
        raise ValueError("dominance_fraction in (0, 1]; max_gap_fraction_complete in [0, 1)")

    if "start_ms" not in hourly.columns:
        raise ValueError("Missing required hourly column: start_ms")

    x = add_hourly_missingness_transitions(hourly)

    for column in _SOURCE_FLAGS:
        x[column] = x[column].fillna(False).astype(bool)

    x["date_local"] = [
        d.isoformat() for d in local_dates(x["start_ms"].to_numpy(), tz, day_start)
    ]

    battery = (
        pd.to_numeric(x["battery_last"], errors="coerce")
        if "battery_last" in x.columns
        else pd.Series(np.nan, index=x.index)
    )
    prev_battery = battery.groupby(x["episode_id"], sort=False).shift(1)
    x["_low_battery_onset"] = x["shared_gap_onset"].astype(bool) & (
        prev_battery <= low_battery_fraction
    ).fillna(False)

    rows = []

    for (episode, date), day in x.groupby(["episode_id", "date_local"], sort=True):
        eligible = day[day["source_eligible"].astype(bool)]
        gps = eligible["gps_observed"].astype(bool).to_numpy()
        acc = eligible["acc_observed"].astype(bool).to_numpy()

        counts = {
            "n_hours": len(day),
            "n_eligible_hours": len(eligible),
            "gps_specific_gap_hours": int((~gps & acc).sum()),
            "shared_gap_hours": int((~gps & ~acc).sum()),
            "acc_only_gap_hours": int((gps & ~acc).sum()),
            "n_gps_specific_onsets": int(eligible["gps_specific_gap_onset"].astype(bool).sum()),
            "n_shared_onsets": int(eligible["shared_gap_onset"].astype(bool).sum()),
            "n_low_battery_shared_onsets": int(eligible["_low_battery_onset"].sum()),
        }

        row = {"episode_id": episode, "date_local": date}

        if "participant_id" in day.columns:
            row["participant_id"] = day["participant_id"].iloc[0]

        row.update(counts)

        if len(eligible) == 0:
            status = FeatureStatus.SOURCE_ABSENT
        elif len(eligible) < min_eligible_hours:
            status = FeatureStatus.INSUFFICIENT_OBSERVABILITY
        else:
            status = FeatureStatus.OBSERVED

        row["missingness_phenotype"] = (
            classify_day(
                n_eligible_hours=counts["n_eligible_hours"],
                gps_specific_gap_hours=counts["gps_specific_gap_hours"],
                shared_gap_hours=counts["shared_gap_hours"],
                acc_only_gap_hours=counts["acc_only_gap_hours"],
                n_shared_onsets=counts["n_shared_onsets"],
                n_low_battery_shared_onsets=counts["n_low_battery_shared_onsets"],
                max_gap_fraction_complete=max_gap_fraction_complete,
                dominance_fraction=dominance_fraction,
            )
            if status is FeatureStatus.OBSERVED
            else np.nan
        )
        row["missingness_phenotype_status"] = status.value

        row["completeness_weight"] = (
            multi_stream_weight({
                "gps": completeness_weight_from_mask(~gps, bin_size_ms=HOUR_MS),
                "accelerometer": completeness_weight_from_mask(~acc, bin_size_ms=HOUR_MS),
            })
            if len(eligible)
            else np.nan
        )

        rows.append(row)

    return pd.DataFrame(rows)
