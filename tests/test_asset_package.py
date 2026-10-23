"""Build files round-trip through a package addressed by build hash; only admitted assets travel."""

import pytest

from robotsmith.assets import dedup, package, store


def _asset(assets, license, name="cube", review="pass"):
    d = assets / name
    d.mkdir(parents=True)
    (d / "model.urdf").write_text(f'<robot name="{name}"><link name="base"/></robot>\n', encoding="utf-8")
    (d / "body.obj").write_bytes(b"v 0 0 0\n")
    store.new_asset(d, "rigid", "blender", {}, license=license)
    if review:
        store.append_check(d, "llm_review", review, {"claims": []}, by="agent:t")
    store.append_check(d, "dedup_check", "none", {"rule": dedup.RULE, "candidates": []}, by="check:dedup@1")
    return d


def _current(d):
    data = store.read_asset(d)
    return store.build_hash(d, data["build"]["files"]) == data["build"]["hash"]


def test_fetch_restores_missing_and_tampered_build_files(tmp_path):
    assets, pkg = tmp_path / "assets", str(tmp_path / "store")
    d = _asset(assets, "Apache-2.0")
    package.publish(assets, pkg)
    (d / "model.urdf").unlink()
    (d / "body.obj").write_bytes(b"tampered\n")

    assert package.fetch(assets, pkg) == ["cube"]
    assert _current(d)
    assert package.fetch(assets, pkg) == []


def test_fetch_refuses_a_package_copy_that_does_not_match(tmp_path):
    assets, pkg = tmp_path / "assets", tmp_path / "store"
    d = _asset(assets, "Apache-2.0")
    package.publish(assets, str(pkg))
    digest = store.read_asset(d)["build"]["hash"]
    (pkg / "cube" / digest / "body.obj").write_bytes(b"corrupt\n")
    (d / "body.obj").unlink()

    with pytest.raises(ValueError, match="do not match"):
        package.fetch(assets, str(pkg))


@pytest.mark.parametrize("license, reasons", [
    ("CC-BY-4.0", []),
    (None, ["source.license missing"]),
    ("CC-BY-NC-4.0", ["license CC-BY-NC-4.0 not redistributable"]),
    ("MIT AND CC-BY-NC-4.0", ["license MIT AND CC-BY-NC-4.0 not redistributable"]),
])
def test_admission_needs_a_redistributable_license(tmp_path, license, reasons):
    assert store.draft_reasons(_asset(tmp_path / "assets", license)) == reasons


def test_admission_needs_a_review_pass_on_the_current_build(tmp_path):
    d = _asset(tmp_path / "assets", "MIT", review=None)
    assert store.draft_reasons(d) == ["no llm_review on the current build"]
    store.append_check(d, "llm_review", "fail", {"claims": []}, by="agent:t")
    assert store.draft_reasons(d) == ["llm_review failed"]
    store.append_check(d, "llm_review", "pass", {"claims": []}, by="agent:t")
    assert store.draft_reasons(d) == []

    data = store.read_asset(d)
    store.write_asset(d, data | {"revision": 2})
    assert store.draft_reasons(d) == ["no llm_review on the current build", "no dedup_check on the current build"]


def test_drafts_are_neither_published_nor_fetched(tmp_path):
    assets, pkg = tmp_path / "assets", tmp_path / "store"
    admitted = _asset(assets, "MIT")
    draft = _asset(assets, None, name="draft_box")
    package.publish(assets, str(pkg))
    assert (pkg / "cube").is_dir() and not (pkg / "draft_box").exists()

    for d in (admitted, draft):
        (d / "model.urdf").unlink()
    assert package.fetch(assets, str(pkg)) == ["cube"]
    assert _current(admitted) and not (draft / "model.urdf").exists()


def test_every_cc_by_library_asset_carries_attribution():
    for d in package.asset_dirs(package.DEFAULT_ROOT):
        source = store.read_asset(d)["source"]
        if (source.get("license") or "").startswith("CC-BY"):
            assert source.get("attribution"), d.name


def test_every_library_asset_has_a_category():
    for d in package.asset_dirs(package.DEFAULT_ROOT):
        assert store.read_asset(d).get("category"), d.name
