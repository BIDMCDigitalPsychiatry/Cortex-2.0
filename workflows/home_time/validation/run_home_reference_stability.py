from pathlib import Path
import re
import subprocess
import sys

ROOT = Path("/shared/Digital Clinic/Cortex-2.0")
SOURCE = ROOT / "workflows/home_time/build/build_weekly_home_time.py"

STABILITY_ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic/"
    "feature_reliability/gps_clinical/home_time/home_reference_stability"
)

STABILITY_ROOT.mkdir(parents=True, exist_ok=True)

WINDOWS = [3, 5, 7, 10, 14, 21, 30]

original = SOURCE.read_text()

for n_days in WINDOWS:

    print()
    print("=" * 90)
    print(f"HOME REFERENCE STABILITY: {n_days} DAYS")
    print("=" * 90)

    s = original

    # --------------------------------------------------------
    # Change ONLY the calibration duration.
    # --------------------------------------------------------

    s, count = re.subn(
        r"HOME_REFERENCE_DAYS\s*=\s*\d+",
        f"HOME_REFERENCE_DAYS = {n_days}",
        s,
        count=1,
    )

    if count != 1:
        raise RuntimeError(
            "Could not patch HOME_REFERENCE_DAYS."
        )

    # --------------------------------------------------------
    # Redirect the normal home_time output directory.
    #
    # Production script contains a Path expression ending in:
    #     / "home_time"
    #
    # Add a run-specific subdirectory beneath it.
    # --------------------------------------------------------

    replacement = (
        '/ "home_time"\n'
        f'    / "home_reference_stability"\n'
        f'    / "days_{n_days}"'
    )

    s, count = re.subn(
        r'/\s*"home_time"',
        replacement,
        s,
        count=1,
    )

    if count != 1:
        raise RuntimeError(
            "Could not redirect home_time output root."
        )

    tmp = Path(
        f"/tmp/build_weekly_home_time_stability_{n_days}.py"
    )

    tmp.write_text(s)

    subprocess.run(
        [
            sys.executable,
            str(tmp),
        ],
        cwd=str(ROOT),
        env={
            **__import__("os").environ,
            "PYTHONPATH": "src",
        },
        check=True,
    )

print()
print("=" * 90)
print("ALL CALIBRATION-WINDOW RUNS COMPLETED")
print("=" * 90)
print(STABILITY_ROOT)
