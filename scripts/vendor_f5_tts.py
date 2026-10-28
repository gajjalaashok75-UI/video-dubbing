#!/usr/bin/env python3
"""
Fetch IndicF5's own `f5_tts` fork into `engine/vendor/f5_tts`.

Why this exists: the IndicF5 checkpoint is NOT compatible with the `f5-tts`
release that pip installs. Both export `CFM`/`DiT` and both load the
checkpoint's 447 tensors with zero missing keys, so nothing errors - but the
release's forward math differs (`AdaLayerNormZero_Final` vs `AdaLayerNorm_Final`
in `DiT.norm_out`, text padding, the CFM duration/timestep schedule), and it
returns fluent audio that is the wrong words. `tts_engines._use_vendored_f5_tts`
refuses to synthesize unless the fork is the `f5_tts` that got imported.

The fork lives in AI4Bharat's own IndicF5 repository rather than on PyPI, so it
has to be placed in the tree at setup time. This script does that, verifies the
marker the guard checks for, and is safe to re-run.

    python scripts/vendor_f5_tts.py            # fetch if missing
    python scripts/vendor_f5_tts.py --check     # report, change nothing
    python scripts/vendor_f5_tts.py --force     # refetch over an existing copy
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = REPO_ROOT / "engine" / "vendor" / "f5_tts"

UPSTREAM = "https://github.com/AI4Bharat/IndicF5"
# The class the release does not have. `_use_vendored_f5_tts` tests for exactly
# this name, so this script verifies exactly this name: any other check could
# pass while the guard still refuses to run.
MARKER = "AdaLayerNormZero_Final"
MARKER_FILE = Path("model") / "backbones" / "dit.py"


class VendorError(RuntimeError):
    pass


def is_vendored(path: Path | None = None) -> bool:
    """Whether `path` holds the fork (marker present in dit.py), not the release."""
    root = Path(path) if path is not None else VENDOR_DIR
    dit = root / MARKER_FILE
    if not dit.is_file():
        return False
    try:
        return MARKER in dit.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def _clone(dest: Path) -> None:
    """Shallow-clone the upstream repo into `dest`."""
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", UPSTREAM, str(dest)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError as e:
        raise VendorError(
            "git is not installed, so IndicF5's fork cannot be fetched. "
            f"Install git, or copy {UPSTREAM}/f5_tts to {VENDOR_DIR} by hand."
        ) from e
    except subprocess.CalledProcessError as e:
        raise VendorError(
            f"Cloning {UPSTREAM} failed:\n{(e.stdout or '').strip()}"
        ) from e


def install(force: bool = False, verbose: bool = True) -> bool:
    """
    Place the fork at `engine/vendor/f5_tts`. Returns True if it is now present.

    Idempotent: an already-correct copy is left alone unless `force` is set.
    """
    if is_vendored() and not force:
        if verbose:
            print(f"IndicF5's f5_tts fork is already vendored at {VENDOR_DIR}")
        return True

    with tempfile.TemporaryDirectory(prefix="indicf5-") as tmp:
        checkout = Path(tmp) / "IndicF5"
        if verbose:
            print(f"Fetching {UPSTREAM} ...")
        _clone(checkout)

        source = checkout / "f5_tts"
        if not is_vendored(source):
            raise VendorError(
                f"{UPSTREAM} no longer ships the fork this app needs: "
                f"{source / MARKER_FILE} does not contain {MARKER}. "
                "The engine refuses to synthesize with the pip release because "
                "it generates the wrong words; see _use_vendored_f5_tts in "
                "engine/tts_engines.py before changing this."
            )

        # Replace wholesale: a stale half-copy is worse than none, because the
        # marker check would pass while individual modules stayed out of date.
        parent = VENDOR_DIR.parent
        parent.mkdir(parents=True, exist_ok=True)
        staging = parent / "f5_tts.incoming"
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(source, staging)
        if VENDOR_DIR.exists():
            shutil.rmtree(VENDOR_DIR)
        staging.rename(VENDOR_DIR)

    if not is_vendored():
        raise VendorError(f"copied the fork but {VENDOR_DIR / MARKER_FILE} is missing {MARKER}")
    if verbose:
        print(f"Vendored IndicF5's f5_tts fork at {VENDOR_DIR}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--check", action="store_true",
                        help="report whether the fork is vendored; change nothing")
    parser.add_argument("--force", action="store_true",
                        help="refetch even if a vendored copy is already present")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="only print on failure")
    args = parser.parse_args(argv)

    if args.check:
        if is_vendored():
            if not args.quiet:
                print(f"IndicF5's f5_tts fork: present ({VENDOR_DIR})")
            return 0
        print(f"IndicF5's f5_tts fork: MISSING ({VENDOR_DIR}); "
              "run scripts/vendor_f5_tts.py")
        return 1

    try:
        install(force=args.force, verbose=not args.quiet)
    except VendorError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
