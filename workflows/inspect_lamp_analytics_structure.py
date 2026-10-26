from pathlib import Path
from collections import Counter
import os
import re

import pandas as pd
import LAMP
import LAMP.rest


OS_MAP = Path(
    "../cortex2_outputs/digital_clinic/"
    "inventory/"
    "risk_episode_ios_sensorkit_evidence_timeaware.csv"
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
# PICK ONE CONFIRMED-IOS EPISODE
# ============================================================

episodes = pd.read_csv(
    OS_MAP
)

row = (
    episodes[
        episodes[
            "sensorkit_ios_evidence_in_episode"
        ] == True
    ]
    .iloc[0]
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


print("=" * 80)
print("LAMP.ANALYTICS STRUCTURE INSPECTION")
print("=" * 80)

print("UID:", uid)
print("Episode:", row["episode_id"])


# ============================================================
# QUERY
# ============================================================

response = (
    LAMP.SensorEvent
    .all_by_participant(
        uid,
        origin="lamp.analytics",
        _from=start_ms,
        to=end_ms,
    )
)


if isinstance(
    response,
    dict,
):

    events = response.get(
        "data",
        []
    )

else:

    events = response


print(
    "Analytics events returned:",
    len(events),
)


# ============================================================
# TOP-LEVEL / DATA KEYS
# ============================================================

top_keys = Counter()
data_keys = Counter()
data_keysets = Counter()


for event in events:

    if not isinstance(
        event,
        dict,
    ):
        continue

    top_keys.update(
        event.keys()
    )

    data = event.get(
        "data"
    )

    if isinstance(
        data,
        dict,
    ):

        data_keys.update(
            data.keys()
        )

        data_keysets[
            tuple(
                sorted(
                    data.keys()
                )
            )
        ] += 1


print()
print("Top-level event keys:")

for key, n in (
    top_keys.most_common()
):

    print(
        f"  {key}: {n}"
    )


print()
print("Most common data keys:")

for key, n in (
    data_keys.most_common(30)
):

    print(
        f"  {key}: {n}"
    )


print()
print("Most common data key combinations:")

for keys, n in (
    data_keysets.most_common(15)
):

    print(
        f"  {n:5d}  {keys}"
    )


# ============================================================
# RECURSIVELY FIND OS / DEVICE / LOGIN-RELATED FIELDS
# ============================================================

KEY_PATTERN = re.compile(
    r"login|device|user.?agent|platform|"
    r"operating.?system|(^|_)os($|_)",
    re.IGNORECASE,
)

VALUE_PATTERN = re.compile(
    r"\bios\b|android|iphone|ipad|login",
    re.IGNORECASE,
)


matches = Counter()


def walk(
    obj,
    path="",
):

    if isinstance(
        obj,
        dict,
    ):

        for key, value in obj.items():

            new_path = (
                f"{path}.{key}"
                if path
                else str(key)
            )

            if (
                KEY_PATTERN.search(
                    str(key)
                )
            ):

                if isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                        bool,
                    ),
                ):

                    text = str(
                        value
                    )

                    matches[
                        (
                            new_path,
                            text[:250],
                        )
                    ] += 1

            walk(
                value,
                new_path,
            )


    elif isinstance(
        obj,
        list,
    ):

        for item in obj:

            walk(
                item,
                path + "[]",
            )


    elif isinstance(
        obj,
        str,
    ):

        if VALUE_PATTERN.search(
            obj
        ):

            matches[
                (
                    path,
                    obj[:250],
                )
            ] += 1


for event in events:

    walk(
        event
    )


print()
print("=" * 80)
print("OS / DEVICE / LOGIN-RELATED VALUES")
print("=" * 80)

if not matches:

    print(
        "No matching OS/device/login fields found "
        "in this episode."
    )

else:

    for (
        path,
        value,
    ), n in matches.most_common(50):

        print()
        print(
            f"{n:5d}  {path}"
        )

        print(
            f"       {value}"
        )


print()
print("=" * 80)
print("DONE")
print("=" * 80)
