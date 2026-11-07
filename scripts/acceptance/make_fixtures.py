"""Write the URDF-primitive fixtures of the acceptance families into assets/ (drafts until reviewed).

  python scripts/acceptance/make_fixtures.py

Each fixture is one fixed link of named boxes; origin at the footprint centre on the floor.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from robotsmith.assets import store

# name -> (size xyz, centre xyz); all in metres.
WALL_T = 0.02
CAGE, CAGE_T = 0.12, 0.01
FIXTURES = {
    "mat_01": {
        "category": "mat", "rgba": "0.2 0.4 0.8 1",
        "description": "Flat 16 x 16 cm placing mat, 1 cm thick; place target `top`.",
        "boxes": {"top": ((0.16, 0.16, 0.01), (0, 0, 0.005))},
        "place_targets": [{"name": "top"}],
    },
    "wall_01": {
        "category": "wall", "rgba": "0.6 0.6 0.6 1",
        "description": "Low wall 40 cm long (local y), 2 cm thick (local x), 6 cm tall.",
        "boxes": {"wall": ((WALL_T, 0.40, 0.06), (0, 0, 0.03))},
    },
    "wall_02": {
        "category": "wall", "rgba": "0.6 0.6 0.6 1",
        "description": "Barrier wall 40 cm long (local y), 2 cm thick (local x), 20 cm tall.",
        "boxes": {"wall": ((WALL_T, 0.40, 0.20), (0, 0, 0.10))},
    },
    "cage_01": {
        "category": "cage", "rgba": "0.7 0.3 0.3 1",
        "description": "Closed cage 12 x 12 x 12 cm (1 cm walls and roof, open bottom, no other opening).",
        "boxes": {
            "side_px": ((CAGE_T, CAGE, CAGE), ((CAGE - CAGE_T) / 2, 0, CAGE / 2)),
            "side_nx": ((CAGE_T, CAGE, CAGE), (-(CAGE - CAGE_T) / 2, 0, CAGE / 2)),
            "side_py": ((CAGE - 2 * CAGE_T, CAGE_T, CAGE), (0, (CAGE - CAGE_T) / 2, CAGE / 2)),
            "side_ny": ((CAGE - 2 * CAGE_T, CAGE_T, CAGE), (0, -(CAGE - CAGE_T) / 2, CAGE / 2)),
            "roof": ((CAGE - 2 * CAGE_T, CAGE - 2 * CAGE_T, CAGE_T), (0, 0, CAGE - CAGE_T / 2)),
        },
    },
}


def urdf(name: str, spec: dict) -> str:
    def geom(size):
        return f'<geometry><box size="{size[0]:g} {size[1]:g} {size[2]:g}" /></geometry>'

    parts = []
    for part, (size, xyz) in spec["boxes"].items():
        origin = f'<origin xyz="{xyz[0]:g} {xyz[1]:g} {xyz[2]:g}" rpy="0 0 0" />'
        parts.append(f'    <visual name="{part}">{origin}{geom(size)}<material name="paint" /></visual>')
        parts.append(f'    <collision name="{part}">{origin}{geom(size)}</collision>')
    body = "\n".join(parts)
    return (
        f'<robot name="{name}">\n  <material name="paint"><color rgba="{spec["rgba"]}" /></material>\n'
        f'  <link name="base">\n    <inertial><origin xyz="0 0 0" rpy="0 0 0" /><mass value="2" />'
        f'<inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01" /></inertial>\n'
        f"{body}\n  </link>\n</robot>\n"
    )


def main() -> None:
    for name, spec in FIXTURES.items():
        d = REPO / "assets" / name
        d.mkdir(exist_ok=True)
        (d / "model.urdf").write_text(urdf(name, spec), encoding="utf-8", newline="\n")
        by = "agent:acceptance"
        annotations = {"description": store.annotation(spec["description"], by),
                       "friction": store.annotation(0.6, by)}
        if "place_targets" in spec:
            annotations["place_targets"] = store.annotation(spec["place_targets"], by)
        store.new_asset(d, "fixture", "builtin_primitive", annotations, license="Apache-2.0",
                        category=spec["category"])
        print(name, store.draft_reasons(d))


if __name__ == "__main__":
    main()
