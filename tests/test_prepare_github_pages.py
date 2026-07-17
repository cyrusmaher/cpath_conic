from pathlib import Path

import pytest

from scripts.prepare_github_pages import validate_site


ROOT = Path(__file__).resolve().parents[1]


def _minimal_site(root: Path, gallery_reference: str = "../asset.png") -> Path:
    site = root / "docs"
    gallery = site / "gallery"
    gallery.mkdir(parents=True)
    (site / ".nojekyll").write_text("")
    metadata = '<meta name="viewport" content="width=device-width"><meta name="description" content="test">'
    (site / "index.html").write_text(metadata + '<a href="gallery/index.html">dashboard</a>')
    (gallery / "index.html").write_text(metadata + f'<img src="{gallery_reference}">')
    (site / "asset.png").write_bytes(b"image")
    return site


def test_validate_site_accepts_project_relative_assets(tmp_path: Path) -> None:
    site = _minimal_site(tmp_path)
    first_report = validate_site(site)
    (site / "build_manifest.json").write_text("{}")
    report = validate_site(site)

    assert report == first_report
    assert report["project_pages_compatible"] is True
    assert report["html_files"] == 2
    assert report["local_references"] == 2


@pytest.mark.parametrize("reference", ["/asset.png", "../../outside.png"])
def test_validate_site_rejects_references_outside_bundle(tmp_path: Path, reference: str) -> None:
    site = _minimal_site(tmp_path, gallery_reference=reference)
    (tmp_path / "outside.png").write_bytes(b"outside")

    with pytest.raises(ValueError, match="outside the site root"):
        validate_site(site)


def test_validate_site_requires_nojekyll(tmp_path: Path) -> None:
    site = _minimal_site(tmp_path)
    (site / ".nojekyll").unlink()

    with pytest.raises(FileNotFoundError, match="marker is missing"):
        validate_site(site)


def test_pages_deployment_remains_manual_only() -> None:
    workflow = (ROOT / ".github" / "workflows" / "deploy-dashboard-pages.yml").read_text()

    trigger_block = workflow.split("permissions:", 1)[0]
    assert "workflow_dispatch:" in trigger_block
    assert "push:" not in trigger_block
    assert "pull_request:" not in trigger_block
    assert "actions/configure-pages@v5" in workflow
    assert "actions/upload-pages-artifact@v4" in workflow
    assert "actions/deploy-pages@" in workflow
    assert "  build:\n" in workflow
    assert "  deploy:\n    needs: build\n" in workflow
