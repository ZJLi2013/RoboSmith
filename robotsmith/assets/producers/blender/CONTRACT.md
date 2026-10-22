# Asset contract

Build the requested item with Blender Python and nothing else: no downloads, no external
asset files. The item will be loaded into a robot manipulation simulator, so it must be
usable there, not only look right.

## How to build

- Write `build.py` and run it with `./run_blender.sh build.py` (Blender 4.4, headless).
- Units are metres. +Z is up. Each item rests on the plane z = 0, centred at x = y = 0, and
  its front faces +X.
- Keep the scale and the names given in the request.

## Collision

The simulator collides with boxes. Every mesh object in a part becomes one collision box:
the object's axis-aligned bounding box. So build everything that must collide as its own
axis-aligned object (each panel of a drawer, a handle's neck, its knob) and give each
object a unique, descriptive name (`tray_floor`, `handle_neck`). An open container made of
one object becomes a solid block.

A rigid part with a rounded surface (a fruit, a bottle) can set `"collision": "mesh"` in its
`item.json` entry instead: the whole part then collides as its visual mesh, which the
simulator decomposes into convex pieces.

## What to write

For each separate item, a folder `out/<item>/` with:

- one `<part>.glb` per rigid part (apply transforms before export; keep object names;
  colours and materials are welcome);
- `item.json`:

```json
{
  "name": "drawer_cabinet",
  "kind": "articulated",
  "parts": [
    {"name": "carcass", "mesh": "carcass.glb", "mass_kg": 6.0, "material": "wood"},
    {"name": "drawer", "mesh": "drawer.glb", "mass_kg": 0.8, "material": "wood"},
    {"name": "handle", "mesh": "handle.glb", "mass_kg": 0.02, "material": "metal", "parent": "drawer"}
  ],
  "joints": [
    {"name": "drawer_slide", "type": "prismatic", "parent": "carcass", "child": "drawer",
     "axis": [1, 0, 0], "origin": [0, 0, 0], "lower": 0.0, "upper": 0.2}
  ],
  "handles": [{"part": "handle", "object": "handle_neck"}],
  "place_targets": [{"name": "drawer_opening", "part": "drawer", "surface": "tray_floor"}]
}
```

- `kind` is `rigid` or `articulated`. The first part is the root.
- A moving part is the `child` of one joint: `{"name", "type": "prismatic" | "revolute",
  "parent", "child", "axis": [x, y, z], "origin": [x, y, z], "lower", "upper"}` (metres or
  radians; `origin` is a point on the joint axis in item coordinates). Every other part is
  rigidly attached to its `parent` part, or to the root when it has none.
- A handle names the object a gripper closes on.
- A place target names the object whose top face holds what is put there (a drawer's
  floor, a shelf board).
- Several separate items (for example two chopsticks) are several `out/<item>/` folders.

## Check your work

`./run_blender.sh tools/render.py` renders `out/renders/front.png`, `side.png`, `top.png`
and `iso.png` over a 10 cm ground grid and prints the overall bounding box (`BBOX_M`).
Look at the renders and fix what is wrong. Finish when the item matches the request.
