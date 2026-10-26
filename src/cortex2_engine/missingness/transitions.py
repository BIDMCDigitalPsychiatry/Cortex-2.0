import pandas as pd


def add_hourly_missingness_transitions(df):
    """
    Add GPS missingness transitions to an hourly Cortex 2.0 table.

    A GPS gap onset is defined as:
        previous eligible hour: GPS observed
        current eligible hour:  GPS not observed

    This does not infer a causal missingness mechanism.
    """

    required = {
        "episode_id",
        "hour_index",
        "gps_observed",
        "acc_observed",
        "screen_event_observed",
        "gps_dq",
        "acc_dq",
        "gps_source_unavailable",
        "acc_source_unavailable",
        "gps_partial_source",
        "acc_partial_source",
        "gps_boundary_unknown",
        "acc_boundary_unknown",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    x = df.sort_values(
        [
            "episode_id",
            "hour_index",
        ]
    ).copy()

    same_episode = (
        x["episode_id"]
        .eq(
            x["episode_id"].shift()
        )
    )

    consecutive = (
        same_episode
        &
        x["hour_index"].eq(
            x["hour_index"].shift() + 1
        )
    )

    x["source_eligible"] = ~(
        x["gps_source_unavailable"]
        |
        x["acc_source_unavailable"]
        |
        x["gps_partial_source"]
        |
        x["acc_partial_source"]
    )

    prev_source_eligible = (
        x["source_eligible"]
        .shift()
        .fillna(False)
    )

    x["transition_eligible"] = (
        consecutive
        &
        x["source_eligible"]
        &
        prev_source_eligible
    )

    x["boundary_known"] = ~(
        x["gps_boundary_unknown"]
        |
        x["acc_boundary_unknown"]
    )

    x["gps_gap"] = ~(
        x["gps_observed"]
        .astype(bool)
    )

    prev_gps_observed = (
        x["gps_observed"]
        .shift()
        .fillna(False)
        .astype(bool)
    )

    prev_gps_gap = ~prev_gps_observed

    x["gps_gap_onset"] = (
        x["transition_eligible"]
        &
        x["gps_gap"]
        &
        prev_gps_observed
    )

    x["gps_gap_persistence"] = (
        x["transition_eligible"]
        &
        x["gps_gap"]
        &
        prev_gps_gap
    )

    x["gps_recovery"] = (
        x["transition_eligible"]
        &
        x["gps_observed"].astype(bool)
        &
        prev_gps_gap
    )

    # GPS disappears while ACC remains positively observed.
    x["gps_specific_gap_onset"] = (
        x["gps_gap_onset"]
        &
        x["acc_observed"].astype(bool)
    )

    # Both GPS and ACC disappear at GPS onset.
    x["shared_gap_onset"] = (
        x["gps_gap_onset"]
        &
        ~x["acc_observed"].astype(bool)
    )

    # Lagged context. These are especially useful for later
    # onset models because they precede the missingness event.
    for col in [
        "gps_dq",
        "acc_dq",
        "gps_observed",
        "acc_observed",
        "screen_event_observed",
    ]:
        x[f"prev_{col}"] = (
            x.groupby(
                "episode_id",
                sort=False,
            )[col]
            .shift(1)
        )

    return x
