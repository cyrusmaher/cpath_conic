#!/usr/bin/env python3
"""Build and validate a static GitHub Pages copy of the CoNIC dashboard."""

from __future__ import annotations

import argparse
import json
import shutil
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse


PUBLIC_PAGES = ("index.html", "gallery/index.html")
MAX_SITE_BYTES = 1_000_000_000
MAX_FILE_BYTES = 100_000_000


class AssetParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        key = "src" if tag in {"img", "script"} else "href" if tag in {"a", "link"} else None
        if key and attributes.get(key):
            self.references.append(attributes[key] or "")


def local_references(page: Path) -> list[Path]:
    parser = AssetParser()
    parser.feed(page.read_text())
    resolved = []
    for reference in parser.references:
        parsed = urlparse(reference)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        resolved.append((page.parent / unquote(parsed.path)).resolve())
    return resolved


def validate_site(site: Path) -> dict:
    site = site.resolve()
    if not site.is_dir():
        raise FileNotFoundError(f"GitHub Pages site directory does not exist: {site}")
    for public_page in PUBLIC_PAGES:
        page = site / public_page
        if not page.is_file():
            raise FileNotFoundError(f"Required public dashboard page is missing: {page}")
        markup = page.read_text().lower()
        for family in ("viewport", "description"):
            if f'name="{family}"' not in markup and f"name='{family}'" not in markup:
                raise ValueError(f"Public dashboard page is missing {family} metadata: {page}")
    if not (site / ".nojekyll").is_file():
        raise FileNotFoundError(f"GitHub Pages marker is missing: {site / '.nojekyll'}")

    html_files = sorted(site.rglob("*.html"))
    missing = []
    escaped = []
    checked = set()
    for page in html_files:
        for target in local_references(page):
            try:
                target.relative_to(site)
            except ValueError:
                escaped.append({"page": str(page.relative_to(site)), "target": str(target)})
                continue
            checked.add(target)
            if not target.exists():
                missing.append({"page": str(page.relative_to(site)), "target": str(target)})
    if escaped:
        examples = "\n".join(f"  {item['page']} -> {item['target']}" for item in escaped[:20])
        raise ValueError(
            f"GitHub Pages build has {len(escaped)} local references outside the site root. "
            f"Use relative, site-contained paths so project Pages URLs work:\n{examples}"
        )
    if missing:
        examples = "\n".join(f"  {item['page']} -> {item['target']}" for item in missing[:20])
        raise FileNotFoundError(f"GitHub Pages build has {len(missing)} missing local references:\n{examples}")

    all_files = [path for path in site.rglob("*") if path.is_file()]
    oversized = [(path, path.stat().st_size) for path in all_files if path.stat().st_size > MAX_FILE_BYTES]
    if oversized:
        examples = "\n".join(
            f"  {path.relative_to(site)}: {size / 1_000_000:.1f} MB" for path, size in oversized[:20]
        )
        raise ValueError(f"GitHub rejects files over 100 MB; found {len(oversized)}:\n{examples}")
    total_bytes = sum(path.stat().st_size for path in all_files)
    if total_bytes > MAX_SITE_BYTES:
        raise ValueError(f"Prepared site is {total_bytes / 1_000_000_000:.2f} GB; keep it below 1 GB")
    # Exclude the generated report from its own payload statistics so a build
    # and a subsequent --check-only invocation return the same values.
    files = [path for path in all_files if path.name != "build_manifest.json"]
    content_bytes = sum(path.stat().st_size for path in files)
    return {
        "html_files": len(html_files),
        "local_references": len(checked),
        "files": len(files),
        "bytes": content_bytes,
        "largest_file_bytes": max((path.stat().st_size for path in files), default=0),
        "build_manifest_excluded_from_payload_statistics": True,
        "project_pages_compatible": True,
    }


def build(source: Path, site: Path) -> dict:
    if not (source / "gallery" / "index.html").exists():
        raise FileNotFoundError(f"Dashboard is not rendered: {source / 'gallery' / 'index.html'}")
    if site.exists():
        shutil.rmtree(site)
    site.mkdir(parents=True)
    pages = []
    for name in PUBLIC_PAGES:
        src, dst = source / name, site / name
        if not src.exists():
            raise FileNotFoundError(f"Required public dashboard page is missing: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        pages.append(src)
    for page in pages:
        for referenced in local_references(page):
            try:
                relative = referenced.relative_to(source.resolve())
            except ValueError as error:
                raise ValueError(f"Local reference escapes dashboard root: {page} -> {referenced}") from error
            if not referenced.exists():
                raise FileNotFoundError(f"Dashboard reference is missing: {page} -> {referenced}")
            if referenced.is_dir():
                continue
            target = site / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(referenced, target)
    (site / ".nojekyll").write_text("")
    report = validate_site(site)
    (site / "build_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("outputs/conic_review_hovernet"))
    parser.add_argument("--site", type=Path, default=Path("docs"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    report = validate_site(args.site) if args.check_only else build(args.source, args.site)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
