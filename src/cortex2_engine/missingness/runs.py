import pandas as pd


EXCLUDED_PATTERNS = {
    "both_observed",
    "source_unavailable_or_partial",
}


def assign_contiguous_runs(
    df,
    *,
    episode_col="episode_id",
    order_col="hour_index",
    state_col="gap_pattern",
):
    """
    Assign contiguous run IDs without merging across:
    - episodes
    - state changes
    - discontinuous hour indices
    """

    x = df.sort_values(
        [
            episode_col,
            order_col,
        ]
    ).copy()

    previous_episode = (
        x[episode_col].shift()
    )

    previous_order = (
        x[order_col].shift()
    )

    previous_state = (
        x[state_col].shift()
    )

    new_run = (
        x[episode_col]
        .ne(previous_episode)
        |
        x[state_col]
        .ne(previous_state)
        |
        x[order_col]
        .ne(previous_order + 1)
    )

    x["run_id"] = (
        new_run.cumsum()
        .astype("int64")
    )

    return x
