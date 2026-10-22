"""Build files round-trip through a package addressed by build hash; publishing needs a license."""

import pytest

from robotsmith.assets import package, store


def _asset(assets, license):
    d = assets / "objects" / "cube"
    d.mkdir(parents=True)
    (d / "model.urdf").write_text('<robot name="cube"><link name="base"/></robot>\n', encoding="utf-8")
    (d / "body.obj").write_bytes(b"v 0 0 0\n")
    store.new_asset(d, "rigid", "blender", {}, license=license)
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


@pytest.mark.parametrize("license", [None, "CC-BY-NC-4.0", "MIT AND CC-BY-NC-4.0"])
def test_publish_refuses_assets_it_may_not_redistribute(tmp_path, license):
    _asset(tmp_path / "assets", license)
    with pytest.raises(package.PublishRefused, match="cube"):
        package.publish(tmp_path / "assets", str(tmp_path / "store"))
    assert not (tmp_path / "store").exists()


def test_every_library_asset_may_be_redistributed():
    for d in package.asset_dirs(package.DEFAULT_ROOT):
        source = store.read_asset(d)["source"]
        assert package.redistributable(source.get("license")), d.name
        if source["license"].startswith("CC-BY"):
            assert source.get("attribution"), d.name
