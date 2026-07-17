"""Integrity guards shared by serialized experiment queues."""

from __future__ import annotations

from pathlib import Path


def archive_incomplete_persistent_run(
    outdir: Path,
    completion_filename: str = "summary.json",
) -> Path | None:
    """Archive a partial persistent-worker run so it restarts from epoch zero.

    Persistent data-loader workers carry augmentation RNG state that is not
    recoverable from a checkpoint.  A completed run is left untouched, while a
    non-empty incomplete directory is atomically renamed beside the canonical
    output directory.  Existing archives are never overwritten.
    """
    outdir = Path(outdir)
    if (outdir / completion_filename).exists() or not outdir.exists():
        return None
    if not any(outdir.iterdir()):
        return None

    archive = outdir.with_name(f"{outdir.name}_partial_archive")
    suffix = 2
    while archive.exists():
        archive = outdir.with_name(f"{outdir.name}_partial_archive_{suffix}")
        suffix += 1
    outdir.rename(archive)
    return archive
