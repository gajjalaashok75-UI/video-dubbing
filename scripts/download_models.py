#!/usr/bin/env python3
"""
Pre-download every model VideoTrans needs.

The studio can do this for you on its setup page; this is the same code with a
command line in front of it, for people who would rather not click.

    python scripts/download_models.py            # everything
    python scripts/download_models.py --core     # required models only
    python scripts/download_models.py --check    # report only, download nothing

Works on macOS, Windows and Linux.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))

import models  # noqa: E402
import source_separation as separation  # noqa: E402


def _report_checkpoint() -> None:
    """Show whether Demucs' htdemucs checkpoint is cached, and where."""
    present = separation.checkpoint_present()
    mark = "present" if present else "MISSING"
    path = os.path.join(separation._hub_checkpoints_dir(),
                        separation.DEMUCS_CHECKPOINT)
    print(f"  [{mark:>7}] {'Background model':<20} "
          f"{separation.DEMUCS_CHECKPOINT_BYTES // 1_000_000:>5} MB   "
          f"{separation.MODEL} (background preservation)")
    print(f"            torch hub cache: {path}")


def _fetch_checkpoint() -> bool:
    """
    Pre-fetch the Demucs checkpoint, verifying its SHA-256.

    Meta's host for this file (`dl.fbaipublicfiles.com`) has had an expired TLS
    certificate, and Demucs downloads it through torch.hub on the FIRST
    separation, deep inside a render. When that fetch fails Demucs is disabled
    silently and the output loses its music and effects. Fetching it here - with
    a mirror list and a checksum - turns that into a visible, one-time step.
    """
    if separation.checkpoint_present():
        return True
    print("  Background model (htdemucs) is not cached; fetching it once.")
    return separation.ensure_checkpoint()


def main() -> int:
    ap = argparse.ArgumentParser(description="Download VideoTrans's models")
    ap.add_argument("--core", action="store_true",
                    help="skip optional models (evaluation harness)")
    ap.add_argument("--check", action="store_true",
                    help="report what is present without downloading")
    ap.add_argument("--no-background", action="store_true",
                    help="skip the Demucs background-separation checkpoint")
    args = ap.parse_args()

    rows = [r for r in models.inventory() if r["required"] or not args.core]
    want_background = not args.no_background
    background_present = separation.checkpoint_present()

    print("\n  VideoTrans models")
    print("  " + "-" * 62)
    print(f"  Cache: {models.cache_root()}")
    print(f"  Free disk: {models.free_space_gb():.1f} GB\n")

    for r in rows:
        mark = "present" if r["present"] else "MISSING"
        tag = "" if r["required"] else "  (optional)"
        if r.get("gated"):
            tag += "  (gated - needs HF_TOKEN)"
        print(f"  [{mark:>7}] {r['label']:<20} {r['size']:>7}   {r['id']}{tag}")
        if r["why"]:
            print(f"            {r['why']}")

    if want_background:
        _report_checkpoint()

    todo = [r for r in rows if not r["present"]]
    checkpoint_todo = want_background and not background_present
    n_todo = len(todo) + (1 if checkpoint_todo else 0)

    if args.check:
        print(f"\n  {n_todo} item(s) still to download.\n")
        return 0
    if not n_todo:
        print("\n  Everything is already downloaded. You can start the studio.\n")
        return 0

    if models.free_space_gb() < 8:
        print(f"\n  Only {models.free_space_gb():.1f} GB free. "
              f"About 8 GB is needed. Aborting.\n")
        return 1

    print(f"\n  Downloading {n_todo} item(s). This is resumable; if it is")
    print("  interrupted, run the command again and it continues.\n")

    # The Demucs checkpoint is not a Hub snapshot or a vendored weight, so it is
    # fetched here rather than through `models`. Do it first: it is small, and
    # without it every render silently drops the soundtrack.
    if checkpoint_todo and not _fetch_checkpoint():
        print("  The background model could not be fetched; renders will keep")
        print("  the dialogue only until it is. Re-run this command to retry.\n")

    weights = {w[1]: w for w in models.WEIGHTS}
    skipped = []
    for i, r in enumerate(todo, 1):
        print(f"  ({i}/{len(todo)}) {r['label']} - {r['size']}")
        if r["kind"] == "hub":
            try:
                models.fetch_hub(r["id"], gated=r.get("gated", False))
                print("          done\n")
            except Exception as exc:                         # noqa: BLE001
                if not r.get("gated"):
                    raise
                # Expected when the licence has not been accepted or no token
                # is set. Say how to fix it, then carry on with the rest.
                reason = (str(exc) if isinstance(exc, models.GatedError)
                          else f"could not be fetched: {exc}")
                print(f"          skipped: {reason}\n")
                skipped.append((r["label"], reason))
        else:
            label, rel, size, want_bytes, want_sha, mirrors = weights[r["id"]]
            if not models.fetch_weight(label, os.path.join(models.ROOT, rel),
                                       size, want_bytes, want_sha, mirrors):
                return 1

    if skipped:
        print("  Some gated models were skipped, so their languages will not")
        print("  speak until they are downloaded:")
        for label, _ in skipped:
            print(f"    - {label}")
        print("\n  To enable them: accept the licence on the model page, put")
        print("  HF_TOKEN=hf_... in the .env file beside run.sh, then re-run")
        print("  this command. See docs/GATED-MODELS.md for the full steps.\n")
        return 0

    print("  All models downloaded. Start the studio next.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
