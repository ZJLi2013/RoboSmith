"""Admission refuses duplicates: same build hash outright, near candidates until a reviewer judges them;
search_assets returns admitted assets by category, size and shared words."""

import pytest

from robotsmith.assets import dedup, store
from robotsmith.assets.library import AssetLibrary


def _asset(assets, name, size_cm, category="drawer_cabinet", description="", review="pass", mark=""):
    d = assets / name
    d.mkdir(parents=True)
    sx, sy, sz = (v / 100 for v in size_cm)
    box = f'<geometry><box size="{sx} {sy} {sz}"/></geometry>'
    (d / "model.urdf").write_text(
        f'<robot name="t"><link name="base"><visual>{box}</visual><collision>{box}</collision></link></robot>\n',
        encoding="utf-8")
    (d / "body.obj").write_bytes(f"v 0 0 0\n# {mark}\n".encode())
    store.new_asset(d, "articulated", "blender", {"description": store.annotation(description, "agent:t")},
                    license="CC0-1.0", category=category)
    store.append_check(d, "llm_review", review, {"claims": []}, by="agent:t")
    dedup.check(d)
    return d


def test_dedup_same_build_hash_is_refused(tmp_path):
    a = _asset(tmp_path, "a", [37, 42, 25])
    b = _asset(tmp_path, "b", [37, 42, 25])
    assert store.draft_reasons(a) == []
    assert store.draft_reasons(b) == ["same build hash as a"]


def test_dedup_near_candidate_waits_for_review(tmp_path):
    _asset(tmp_path, "a", [37, 42, 25], mark="a")
    b = _asset(tmp_path, "b", [40, 43, 26], mark="b")
    assert store.draft_reasons(b) == ["near-duplicate candidates without dedup_review: a"]


def test_dedup_variant_admits_and_duplicate_refuses(tmp_path):
    _asset(tmp_path, "a", [37, 42, 25], mark="a")
    b = _asset(tmp_path, "b", [40, 43, 26], mark="b")
    c = _asset(tmp_path, "c", [41, 37, 24], mark="c")
    dedup.judge(b, {"a": {"verdict": "variant", "reason": "different handle"}}, by="agent:t")
    assert store.draft_reasons(b) == []
    assert [x["id"] for x in dedup.check(c)["metrics"]["candidates"]] == ["a"]
    dedup.judge(c, {"a": {"verdict": "duplicate", "reason": "same shape"}}, by="agent:t")
    assert store.draft_reasons(c) == ["duplicate of a (dedup_review)"]
    with pytest.raises(ValueError):
        dedup.judge(c, {"a": {"verdict": "maybe"}}, by="agent:t")


def test_dedup_rule_bounds(tmp_path):
    _asset(tmp_path, "a", [37, 42, 25], mark="a")
    assert dedup.check(_asset(tmp_path, "far", [37, 42, 32], mark="far"))["result"] == "none"
    assert dedup.check(_asset(tmp_path, "mug", [37, 42, 25], category="mug", mark="mug"))["result"] == "none"
    assert dedup.check(_asset(tmp_path, "turned", [42, 37, 25], mark="turned"))["result"] == "candidates"


def test_dedup_check_is_recorded_once_per_build(tmp_path):
    _asset(tmp_path, "a", [37, 42, 25], mark="a")
    b = _asset(tmp_path, "b", [40, 43, 26], mark="b")
    n = len((b / store.CHECKS_FILE).read_text(encoding="utf-8").splitlines())
    assert dedup.check(b)["result"] == "candidates"
    assert len((b / store.CHECKS_FILE).read_text(encoding="utf-8").splitlines()) == n


def test_dedup_ignores_rebuilt_or_draft_candidates(tmp_path):
    a = _asset(tmp_path, "a", [37, 42, 25], mark="a")
    b = _asset(tmp_path, "b", [40, 43, 26], mark="b")
    data = store.read_asset(a)
    (a / "body.obj").write_bytes(b"v 1 1 1\n")
    data["revision"] += 1
    data["build"]["hash"] = store.build_hash(a, data["build"]["files"])
    store.write_asset(a, data)
    assert store.draft_reasons(b) == []
    assert "no llm_review on the current build" in store.draft_reasons(a)


def test_dedup_without_check_or_comparable_size_is_draft(tmp_path):
    a = _asset(tmp_path, "a", [37, 42, 25], mark="a")
    data = store.read_asset(a)
    (a / "body.obj").write_bytes(b"v 2 2 2\n")
    data["build"]["hash"] = store.build_hash(a, data["build"]["files"])
    store.write_asset(a, data)
    store.append_check(a, "llm_review", "pass", {"claims": []}, by="agent:t")
    assert store.draft_reasons(a) == ["no dedup_check on the current build"]
    _asset(tmp_path, "other", [10, 10, 10], mark="other")
    with pytest.raises(ValueError):
        _asset(tmp_path, "flat", [37, 42, 0], mark="flat")


def _library(tmp_path):
    _asset(tmp_path, "cab_a", [37, 42, 25], description="Drawer cabinet with a pull handle.", mark="a")
    cab_b = _asset(tmp_path, "cab_b", [40, 43, 26], description="Single-drawer nightstand.", mark="b")
    dedup.judge(cab_b, {"cab_a": {"verdict": "variant", "reason": "t"}}, by="agent:t")
    _asset(tmp_path, "cab_c", [60, 60, 80], description="Tall cabinet.", mark="c")
    _asset(tmp_path, "mug", [8, 8, 10], category="mug", description="Coffee mug.", mark="m")
    _asset(tmp_path, "cab_draft", [37, 42, 25], description="Drawer cabinet.", review="fail", mark="d")
    return AssetLibrary(tmp_path)


def test_search_filters_category_and_ranks_by_size(tmp_path):
    lib = _library(tmp_path)
    hits = lib.search_assets("single-drawer cabinet", category="drawer_cabinet", size_cm=[41, 43, 26])
    assert [h["id"] for h in hits] == ["cab_b", "cab_a", "cab_c"]
    assert set(hits[0]) >= {"id", "status", "build_hash", "size_cm"}
    assert lib.search_assets("anything", category="banana") == []


def test_search_without_category_needs_a_shared_word(tmp_path):
    lib = _library(tmp_path)
    assert [h["id"] for h in lib.search_assets("coffee cup")] == ["mug"]
    assert lib.search_assets("banana") == []
    assert lib.search_assets("nightstand with one drawer")[0]["id"] == "cab_b"


def test_search_excludes_drafts_unless_asked(tmp_path):
    lib = _library(tmp_path)
    assert "cab_draft" not in [h["id"] for h in lib.search_assets("cabinet", category="drawer_cabinet")]
    hit = [h for h in lib.search_assets("cabinet", category="drawer_cabinet", include_drafts=True) if h["id"] == "cab_draft"]
    assert hit and hit[0]["status"] == "draft"
