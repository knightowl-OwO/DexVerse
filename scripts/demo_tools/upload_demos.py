# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Push demonstrations from the local ``demonstrations/`` tree to Hugging Face.

Inverse of ``download_demos.py``: a local tree laid out as
``<category>/<task>/<file>.pkl`` is mirrored to ``demonstrations/<category>/
<task>/<file>.pkl`` in the dataset repo. Maintainer tool -- uploading needs a
write token (``huggingface-cli login``).

Examples
--------
    # The curated baseline set (the tasks in baseline_manifest.txt)
    python scripts/demo_tools/upload_demos.py --baseline

    # One category, or one task
    python scripts/demo_tools/upload_demos.py --category functional
    python scripts/demo_tools/upload_demos.py --task functional/Dexverse-GraspPan-v0

    # Preview without pushing
    python scripts/demo_tools/upload_demos.py --baseline --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_REPO = "dexverse/DexVerse_Dataset"
DEFAULT_REPO_TYPE = "dataset"
REMOTE_ROOT = "demonstrations"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "source" / "dexverse" / "demonstrations"
BASELINE_MANIFEST = DEFAULT_SOURCE / "baseline_manifest.txt"


def _baseline_entries() -> list[str]:
    if not BASELINE_MANIFEST.is_file():
        raise SystemExit(f"--baseline manifest not found: {BASELINE_MANIFEST}")
    return [
        line.strip().strip("/")
        for line in BASELINE_MANIFEST.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _collect(args) -> list[tuple[Path, str]]:
    """Return (local file, path under ``demonstrations/``) pairs to upload."""
    source_root = args.source_root.expanduser().resolve()
    if not source_root.is_dir():
        raise SystemExit(f"--source-root not found: {source_root}")

    subtrees = [source_root] if args.all else []
    for entry in (_baseline_entries() if args.baseline else []) + (args.category or []) + (args.task or []):
        subtree = source_root / entry.strip("/")
        if not subtree.is_dir():
            raise SystemExit(f"not found under {source_root}: {entry}")
        subtrees.append(subtree)

    files = {
        path.resolve(): path.resolve().relative_to(source_root).as_posix()
        for subtree in subtrees
        for path in sorted(subtree.rglob("*"))
        if path.is_file() and path.name != BASELINE_MANIFEST.name
    }
    return sorted(files.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HF repo id (default: %(default)s)")
    parser.add_argument("--repo-type", default=DEFAULT_REPO_TYPE, choices=("dataset", "model", "space"))
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE,
                        help="Local root holding <category>/<task>/... (default: source/dexverse/demonstrations)")
    parser.add_argument("--all", action="store_true", help="Upload everything under --source-root.")
    parser.add_argument("--baseline", action="store_true",
                        help=f"Upload the tasks listed in {BASELINE_MANIFEST.name}.")
    parser.add_argument("--category", action="append", help="Upload one category subdir. Repeatable.")
    parser.add_argument("--task", action="append", help="Upload one <category>/<task> subdir. Repeatable.")
    parser.add_argument("--commit-message", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded.")
    args = parser.parse_args()

    if not (args.all or args.baseline or args.category or args.task):
        parser.error("Nothing selected. Pass --all, --baseline, --category or --task.")
    plan = _collect(args)
    if not plan:
        raise SystemExit("Nothing to upload: the selected paths contain no files.")

    print(f"Repo:        {args.repo} ({args.repo_type})")
    print(f"Source root: {args.source_root}")
    print(f"{len(plan)} file(s):")
    for local, remote in plan:
        print(f"  {local.name}  ->  {REMOTE_ROOT}/{remote}")
    if args.dry_run:
        return 0

    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError as exc:
        raise SystemExit("huggingface_hub not installed. Run `pip install huggingface_hub`.") from exc

    HfApi().create_commit(
        repo_id=args.repo,
        repo_type=args.repo_type,
        operations=[
            CommitOperationAdd(path_in_repo=f"{REMOTE_ROOT}/{remote}", path_or_fileobj=str(local))
            for local, remote in plan
        ],
        commit_message=args.commit_message or f"Upload {len(plan)} demonstration file(s)",
    )
    print(f"Done. Pushed {len(plan)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
