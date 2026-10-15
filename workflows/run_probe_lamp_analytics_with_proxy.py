import os
import runpy

import LAMP
import LAMP.rest


# ============================================================
# PROXY
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
    configured = 0

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

        configured += 1

    return configured


def connect_with_proxy(
    *args,
    **kwargs,
):

    result = _original_connect(
        *args,
        **kwargs,
    )

    n = apply_proxy()

    print(
        "LAMP API clients configured "
        f"with proxy: {n}"
    )

    return result


LAMP.connect = connect_with_proxy

# Also configure clients that already exist.
n = apply_proxy()

print(
    "Initial LAMP clients configured "
    f"with proxy: {n}"
)


# ============================================================
# RUN EXISTING ANALYTICS PROBE
# ============================================================

runpy.run_path(
    "workflows/probe_lamp_analytics_os.py",
    run_name="__main__",
)
