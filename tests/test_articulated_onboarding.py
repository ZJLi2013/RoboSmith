"""Articulated asset onboarding.

Loader reads URDF joints (type/axis/limits) at load time; AssetMetadata carries
only thin semantic annotations (task_joints/handles/joint_init). Rigid assets stay
backward compatible (empty joints, empty annotation fields).
"""

from pathlib import Path

from robotsmith.assets.library import AssetLibrary
from robotsmith.assets.schema import parse_urdf_joints

REPO_ROOT = Path(__file__).resolve().parents[1]
DRAWER_URDF = REPO_ROOT / "assets" / "articraft" / "drawer_cabinet" / "model.urdf"

_RIGID_URDF = """<robot name="rigid_box">
  <link name="box">
    <visual><geometry><box size="0.05 0.05 0.05" /></geometry></visual>
  </link>
</robot>
"""

_ARTIC_URDF = """<robot name="hinged_lid">
  <link name="base" />
  <link name="lid" />
  <joint name="lid_hinge" type="revolute">
    <parent link="base" />
    <child link="lid" />
    <axis xyz="0 -1 0" />
    <limit effort="5" velocity="3" lower="0" upper="1.2" />
  </joint>
</robot>
"""


class TestParseUrdfJoints:
    def test_rigid_has_no_joints(self, tmp_path):
        p = tmp_path / "model.urdf"
        p.write_text(_RIGID_URDF, encoding="utf-8")
        assert parse_urdf_joints(p) == []

    def test_revolute_joint_fields(self, tmp_path):
        p = tmp_path / "model.urdf"
        p.write_text(_ARTIC_URDF, encoding="utf-8")
        joints = parse_urdf_joints(p)
        assert len(joints) == 1
        j = joints[0]
        assert j.name == "lid_hinge"
        assert j.type == "revolute"
        assert j.parent == "base" and j.child == "lid"
        assert j.axis == (0.0, -1.0, 0.0)
        assert j.lower == 0.0 and j.upper == 1.2
        assert j.is_movable

    def test_drawer_prismatic_joint(self):
        joints = parse_urdf_joints(DRAWER_URDF)
        movable = [j for j in joints if j.is_movable]
        assert len(movable) == 1
        j = movable[0]
        assert j.name == "drawer_slide"
        assert j.type == "prismatic"
        assert j.axis == (1.0, 0.0, 0.0)
        assert j.lower == 0.0 and j.upper == 0.35


class TestArticulatedAssetLoad:
    def test_library_discovers_and_annotates_drawer(self):
        lib = AssetLibrary(REPO_ROOT / "assets")
        drawer = lib.get("drawer_cabinet")
        assert drawer is not None
        assert drawer.is_articulated
        assert [j.name for j in drawer.movable_joints] == ["drawer_slide"]
        assert drawer.get_joint("drawer_slide").type == "prismatic"
        # Thin semantic annotations come from metadata, kinematics from the URDF.
        assert drawer.metadata.task_joints == ["drawer_slide"]
        assert drawer.metadata.handles == [
            {"name": "handle", "link": "drawer_handle", "local_pos": [0.268, 0.0, 0.2125]}
        ]
        assert drawer.metadata.joint_init == {"drawer_slide": 0.0}

    def test_rigid_asset_not_articulated(self):
        lib = AssetLibrary(REPO_ROOT / "assets")
        block = lib.get("die_01")
        assert block is not None
        assert not block.is_articulated
        assert block.metadata.task_joints == []
