import numpy as np
import pandas as pd

from cortex2_engine.quality.observability import (
    temporal_quality_metrics,
)


def _observation_summary(
    df,
    start_ms,
    end_ms,
):
    z = df[
        (df["timestamp_ms"] >= start_ms)
        &
        (df["timestamp_ms"] < end_ms)
    ]

    if len(z) == 0:
        return {
            "n_observations": 0,
            "first_observed_ms": np.nan,
            "last_observed_ms": np.nan,
        }

    return {
        "n_observations":
            int(len(z)),

        "first_observed_ms":
            int(
                z["timestamp_ms"].min()
            ),

        "last_observed_ms":
            int(
                z["timestamp_ms"].max()
            ),
    }


def gps_quality(
    df,
    *,
    start_ms,
    end_ms,
    bin_seconds=600,
):
    metrics = temporal_quality_metrics(
        df["timestamp_ms"],
        start_ms=start_ms,
        end_ms=end_ms,
        bin_seconds=bin_seconds,
    )

    metrics.update(
        _observation_summary(
            df,
            start_ms,
            end_ms,
        )
    )

    z = df[
        (df["timestamp_ms"] >= start_ms)
        &
        (df["timestamp_ms"] < end_ms)
    ]

    if (
        len(z)
        and "accuracy_m" in z.columns
    ):
        accuracy = pd.to_numeric(
            z["accuracy_m"],
            errors="coerce",
        ).dropna()

        metrics[
            "accuracy_m_median"
        ] = (
            accuracy.median()
            if len(accuracy)
            else np.nan
        )

        metrics[
            "accuracy_m_p90"
        ] = (
            accuracy.quantile(0.90)
            if len(accuracy)
            else np.nan
        )

    else:
        metrics[
            "accuracy_m_median"
        ] = np.nan

        metrics[
            "accuracy_m_p90"
        ] = np.nan

    return metrics


def accelerometer_quality(
    df,
    *,
    start_ms,
    end_ms,
    bin_seconds=1,
):
    metrics = temporal_quality_metrics(
        df["timestamp_ms"],
        start_ms=start_ms,
        end_ms=end_ms,
        bin_seconds=bin_seconds,
    )

    metrics.update(
        _observation_summary(
            df,
            start_ms,
            end_ms,
        )
    )

    return metrics


def screen_quality(
    df,
    *,
    start_ms,
    end_ms,
):
    z = df[
        (df["timestamp_ms"] >= start_ms)
        &
        (df["timestamp_ms"] < end_ms)
    ]

    if len(z) == 0:
        return {
            "observed": 0,
            "n_events": 0,
            "first_observed_ms": np.nan,
            "last_observed_ms": np.nan,
        }

    return {
        "observed": 1,

        "n_events":
            int(len(z)),

        "first_observed_ms":
            int(
                z["timestamp_ms"].min()
            ),

        "last_observed_ms":
            int(
                z["timestamp_ms"].max()
            ),
    }
