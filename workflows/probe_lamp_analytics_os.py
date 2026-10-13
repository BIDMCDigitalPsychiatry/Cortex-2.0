from pathlib import Path
import getpass
import os

import pandas as pd
import LAMP


RISK = Path(
    "../raphe_outputs/digital_clinic/"
    "quality/hourly_missingness/"
    "gps_onset_contextual_risk_set.parquet"
)


# ============================================================
# CONNECT
# ============================================================

access_key = os.getenv(
    "LAMP_ACCESS_KEY"
)

secret_key = os.getenv(
    "LAMP_SECRET_KEY"
)

server = os.getenv(
    "LAMP_SERVER_ADDRESS",
    "api.lamp.digital",
)


if not access_key:

    access_key = input(
        "LAMP access key/email: "
    ).strip()


if not secret_key:

    secret_key = getpass.getpass(
        "LAMP password/secret: "
    )


print()
print(
    "Connecting to:",
    server,
)

LAMP.connect(
    access_key,
    secret_key,
    server,
)

print(
    "Connected."
)


# ============================================================
# SELECT A SMALL SAMPLE OF DIGITAL CLINIC UIDs
# ============================================================

risk = pd.read_parquet(
    RISK,
    columns=[
        "episode_id",
        "source_participant_id",
    ],
)

uids = (
    risk[
        "source_participant_id"
    ]
    .dropna()
    .astype(str)
    .drop_duplicates()
    .head(12)
    .tolist()
)


print()
print("=" * 78)
print("LAMP ANALYTICS OS PROBE")
print("=" * 78)

print(
    "Participants tested:",
    len(uids),
)


rows = []


def unwrap(response):

    if isinstance(
        response,
        dict,
    ):

        data = response.get(
            "data"
        )

        if isinstance(
            data,
            list,
        ):
            return data

    if isinstance(
        response,
        list,
    ):
        return response

    return []


def payload(event):

    d = event.get(
        "data",
        {}
    )

    if isinstance(
        d,
        dict,
    ):
        return d

    return {}


for i, uid in enumerate(
    uids,
    start=1,
):

    try:

        response = (
            LAMP.SensorEvent
            .all_by_participant(
                uid,
                origin="lamp.analytics",
            )
        )

        events = unwrap(
            response
        )

    except Exception as e:

        rows.append({
            "uid":
                uid,

            "analytics_events":
                None,

            "login_events":
                None,

            "device_types":
                "ERROR",

            "latest_user_agent":
                repr(e)[:200],
        })

        continue


    login_events = []

    for event in events:

        p = payload(
            event
        )

        action = p.get(
            "action",
            event.get(
                "action"
            ),
        )

        if action == "login":

            login_events.append(
                event
            )


    device_types = []

    user_agents = []

    for event in login_events:

        p = payload(
            event
        )

        device_type = p.get(
            "device_type",
            event.get(
                "device_type"
            ),
        )

        user_agent = p.get(
            "user_agent",
            event.get(
                "user_agent"
            ),
        )

        if device_type:
            device_types.append(
                str(device_type)
            )

        if user_agent:
            user_agents.append(
                str(user_agent)
            )


    rows.append({
        "uid":
            uid,

        "analytics_events":
            len(events),

        "login_events":
            len(login_events),

        "device_types":
            ", ".join(
                sorted(
                    set(
                        device_types
                    )
                )
            ),

        "latest_user_agent":
            (
                user_agents[0][:180]
                if user_agents
                else ""
            ),
    })


out = pd.DataFrame(
    rows
)


print()
print(
    out.to_string(
        index=False
    )
)


print()
print("=" * 78)

print(
    "Participants with login events:",
    int(
        out[
            "login_events"
        ]
        .fillna(0)
        .gt(0)
        .sum()
    ),
)

print(
    "Participants with OS:",
    int(
        out[
            "device_types"
        ]
        .isin(
            [
                "iOS",
                "Android",
            ]
        )
        .sum()
    ),
)

print("=" * 78)
