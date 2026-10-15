from pathlib import Path
import json
import os
import time

import pandas as pd
import LAMP
import LAMP.rest


ROOT = Path(
    "../raphe_outputs/digital_clinic"
)

EPISODES_PATH = (
    ROOT
    / "inventory/"
    "risk_episode_ios_sensorkit_evidence_timeaware.csv"
)

OUT_MAP = (
    ROOT
    / "inventory/"
    "risk_episode_lamp_analytics_os.csv"
)

OUT_EVENTS = (
    ROOT
    / "inventory/"
    "lamp_analytics_phone_os_events.parquet"
)


PHONE_TYPES = {
    "iOS",
    "Android",
}

API_CAP = 1000

MIN_SPLIT_MS = (
    60
    * 60
    * 1000
)


# ============================================================
# PROXY SUPPORT
# ============================================================

proxy = (
    os.environ.get("HTTPS_PROXY")
    or os.environ.get("https_proxy")
)

if not proxy:
    raise RuntimeError(
        "HTTPS_PROXY / https_proxy is not set."
    )


_original_connect = LAMP.connect


def apply_proxy():

    seen = set()

    for name in dir(LAMP):

        obj = getattr(
            LAMP,
            name,
        )

        client = getattr(
            obj,
            "api_client",
            None,
        )

        if (
            client is None
            or id(client) in seen
        ):
            continue

        config = getattr(
            client,
            "configuration",
            None,
        )

        if config is None:
            continue

        seen.add(
            id(client)
        )

        config.proxy = proxy

        client.rest_client = (
            LAMP.rest.RESTClientObject(
                config
            )
        )


def connect_with_proxy(
    *args,
    **kwargs,
):

    result = _original_connect(
        *args,
        **kwargs,
    )

    apply_proxy()

    return result


LAMP.connect = connect_with_proxy

apply_proxy()
LAMP.connect()


# ============================================================
# API HELPERS
# ============================================================

def unwrap(response):

    if isinstance(
        response,
        dict,
    ):

        data = response.get(
            "data",
            []
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


def fetch_once(
    uid,
    start_ms,
    end_ms,
):

    response = (
        LAMP.SensorEvent
        .all_by_participant(
            uid,
            origin="lamp.analytics",
            _from=int(start_ms),
            to=int(end_ms),
        )
    )

    return unwrap(
        response
    )


def fetch_complete(
    uid,
    start_ms,
    end_ms,
    depth=0,
):

    events = fetch_once(
        uid,
        start_ms,
        end_ms,
    )

    # Less than API cap means the interval
    # is very likely complete.
    if len(events) < API_CAP:
        return events


    interval = (
        end_ms
        - start_ms
    )

    # Safety stop for an extremely dense interval.
    if interval <= MIN_SPLIT_MS:

        print(
            "  WARNING:",
            uid,
            "still returned",
            len(events),
            "events within <=1 hour",
        )

        return events


    midpoint = (
        start_ms
        + interval // 2
    )


    left = fetch_complete(
        uid,
        start_ms,
        midpoint,
        depth + 1,
    )

    right = fetch_complete(
        uid,
        midpoint + 1,
        end_ms,
        depth + 1,
    )

    return (
        left
        + right
    )


def event_key(event):

    return (
        event.get("timestamp"),
        event.get("sensor"),
        json.dumps(
            event.get(
                "data",
                {}
            ),
            sort_keys=True,
            default=str,
        ),
    )


# ============================================================
# LOAD EPISODES
# ============================================================

episodes = pd.read_csv(
    EPISODES_PATH
)


print("=" * 80)
print("LAMP.ANALYTICS EPISODE OS EXTRACTION")
print("=" * 80)

print(
    "Episodes:",
    len(episodes),
)


summary_rows = []
phone_event_rows = []


# ============================================================
# PULL
# ============================================================

for i, row in episodes.iterrows():

    episode_id = str(
        row["episode_id"]
    )

    uid = str(
        row["source_participant_id"]
    )

    start_ms = int(
        row["episode_start_ms"]
    )

    end_ms = int(
        row["episode_end_ms"]
    )


    if (
        i == 0
        or (i + 1) % 10 == 0
        or i + 1 == len(episodes)
    ):

        print(
            f"[{i + 1:3d}/{len(episodes)}] "
            f"{episode_id} | {uid}"
        )


    try:

        events = fetch_complete(
            uid,
            start_ms,
            end_ms,
        )

        # Remove boundary duplicates caused by splitting.
        unique = {}

        for event in events:

            unique[
                event_key(event)
            ] = event

        events = list(
            unique.values()
        )

        error = ""

    except Exception as exc:

        events = []

        error = (
            f"{type(exc).__name__}: "
            f"{str(exc)[:250]}"
        )


    phone_types = set()

    phone_events = 0
    phone_login_events = 0

    ios_events = 0
    android_events = 0


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


        device_type = data.get(
            "device_type"
        )

        if device_type not in PHONE_TYPES:
            continue


        phone_events += 1

        event_type = data.get(
            "type"
        )

        if event_type == "login":
            phone_login_events += 1


        if device_type == "iOS":
            ios_events += 1

        elif device_type == "Android":
            android_events += 1


        phone_types.add(
            device_type
        )


        phone_event_rows.append({
            "episode_id":
                episode_id,

            "participant_id":
                row["participant_id"],

            "source_participant_id":
                uid,

            "timestamp":
                event.get(
                    "timestamp"
                ),

            "event_type":
                event_type,

            "device_type":
                device_type,

            "user_agent":
                data.get(
                    "user_agent"
                ),
        })


    if phone_types == {"iOS"}:

        os_class = "iOS"

    elif phone_types == {"Android"}:

        os_class = "Android"

    elif phone_types == {
        "iOS",
        "Android",
    }:

        os_class = "mixed"

    else:

        os_class = "unknown"


    summary_rows.append({
        "episode_id":
            episode_id,

        "participant_id":
            row["participant_id"],

        "source_participant_id":
            uid,

        "episode_start_ms":
            start_ms,

        "episode_end_ms":
            end_ms,

        "analytics_events":
            len(events),

        "phone_os_events":
            phone_events,

        "phone_login_events":
            phone_login_events,

        "ios_events":
            ios_events,

        "android_events":
            android_events,

        "phone_os":
            os_class,

        "sensorkit_ios_evidence_in_episode":
            row[
                "sensorkit_ios_evidence_in_episode"
            ],

        "error":
            error,
    })


    # Be gentle with the API.
    time.sleep(
        0.05
    )


# ============================================================
# SAVE
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

phone_events = pd.DataFrame(
    phone_event_rows
)


OUT_MAP.parent.mkdir(
    parents=True,
    exist_ok=True,
)


summary.to_csv(
    OUT_MAP,
    index=False,
)


if not phone_events.empty:

    phone_events[
        "timestamp"
    ] = pd.to_numeric(
        phone_events["timestamp"],
        errors="coerce",
    )

    phone_events.to_parquet(
        OUT_EVENTS,
        index=False,
    )


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 80)
print("EPISODE OS RESULTS")
print("=" * 80)

print(
    summary[
        "phone_os"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)


print()
print("=" * 80)
print("SENSORKIT EVIDENCE × LAMP OS")
print("=" * 80)

cross = pd.crosstab(
    summary[
        "sensorkit_ios_evidence_in_episode"
    ],
    summary[
        "phone_os"
    ],
    margins=True,
)

print(
    cross.to_string()
)


print()
print("=" * 80)
print("MIXED / UNKNOWN / ERRORS")
print("=" * 80)

print(
    "Mixed:",
    int(
        (
            summary["phone_os"]
            == "mixed"
        ).sum()
    ),
)

print(
    "Unknown:",
    int(
        (
            summary["phone_os"]
            == "unknown"
        ).sum()
    ),
)

print(
    "API errors:",
    int(
        summary[
            "error"
        ]
        .astype(str)
        .ne("")
        .sum()
    ),
)


print()
print("Saved OS map:")
print(OUT_MAP)

print()
print("Saved phone OS events:")
print(OUT_EVENTS)

print()
print("=" * 80)
print("DONE")
print("=" * 80)
