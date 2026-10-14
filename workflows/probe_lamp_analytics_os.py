from pathlib import Path

import pandas as pd
import LAMP


OS_MAP = Path(
    "../raphe_outputs/digital_clinic/"
    "inventory/"
    "risk_episode_ios_sensorkit_evidence_timeaware.csv"
)


# ============================================================
# LOAD 3 CONFIRMED IOS + 3 UNRESOLVED EPISODES
# ============================================================

df = pd.read_csv(
    OS_MAP
)


ios = (
    df[
        df[
            "sensorkit_ios_evidence_in_episode"
        ] == True
    ]
    .head(3)
)


unresolved = (
    df[
        df[
            "sensorkit_ios_evidence_in_episode"
        ] == False
    ]
    .head(3)
)


sample = pd.concat(
    [
        ios,
        unresolved,
    ],
    ignore_index=True,
)


# ============================================================
# CONNECT USING EXISTING CLUSTER CONFIGURATION
# ============================================================

print("=" * 78)
print("CONNECTING TO LAMP")
print("=" * 78)

LAMP.connect()

print("LAMP.connect() completed.")


# ============================================================
# HELPERS
# ============================================================

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


# ============================================================
# QUERY ANALYTICS
# ============================================================

rows = []


for _, r in sample.iterrows():

    uid = str(
        r["source_participant_id"]
    )

    start_ms = int(
        r["episode_start_ms"]
    )

    end_ms = int(
        r["episode_end_ms"]
    )

    expected_group = (
        "confirmed_iOS"
        if bool(
            r[
                "sensorkit_ios_evidence_in_episode"
            ]
        )
        else "OS_unresolved"
    )


    print()
    print(
        "Querying:",
        uid,
        "|",
        expected_group,
    )


    try:

        response = (
            LAMP.SensorEvent
            .all_by_participant(
                uid,
                origin="lamp.analytics",
                _from=start_ms,
                to=end_ms,
            )
        )

        events = unwrap(
            response
        )

    except Exception as e:

        rows.append({
            "uid":
                uid,

            "expected_group":
                expected_group,

            "analytics_events":
                None,

            "login_events":
                None,

            "device_types":
                "ERROR",

            "user_agents":
                repr(e)[:180],
        })

        continue


    login_events = []

    for event in events:

        data = event.get(
            "data",
            {}
        )

        if not isinstance(
            data,
            dict,
        ):
            continue

        if (
            data.get("action")
            == "login"
        ):

            login_events.append(
                event
            )


    device_types = []

    user_agents = []


    for event in login_events:

        data = event.get(
            "data",
            {}
        )

        device_type = data.get(
            "device_type"
        )

        user_agent = data.get(
            "user_agent"
        )

        if device_type:

            device_types.append(
                str(
                    device_type
                )
            )

        if user_agent:

            user_agents.append(
                str(
                    user_agent
                )
            )


    rows.append({
        "uid":
            uid,

        "expected_group":
            expected_group,

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

        "user_agents":
            " | ".join(
                sorted(
                    set(
                        user_agents
                    )
                )[:3]
            )[:300],
    })


# ============================================================
# REPORT
# ============================================================

out = pd.DataFrame(
    rows
)


print()
print("=" * 78)
print("LAMP ANALYTICS OS PROBE")
print("=" * 78)

print(
    out.to_string(
        index=False
    )
)


print()
print("=" * 78)

print(
    "Episodes with explicit OS:",
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
    "/",
    len(out),
)

print("=" * 78)
