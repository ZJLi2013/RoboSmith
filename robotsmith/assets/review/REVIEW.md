# Asset review

You decide whether a generated asset may enter the asset library. Accept only what you can
back with evidence from this directory; anything you could not check counts against it.

## What is here

- `request.md`: what was asked for. Names, sizes and joints in it are requirements.
- `renders/`: views of the visual model (front, side, top, iso).
- `asset/`: the compiled asset: `model.urdf`, meshes, `asset.json`.
- `readings.json`: what the asset declares and what the scenario runtime derives from it: joints and
  limits, link masses, every named collision box with its bounds at rest, handles (grasp points)
  and place targets. A place target is the collision box an object is put on (`box`); on a moving
  part it also names the joint it rides on and `lip_local`, the front edge of that box's top face
  along the joint at the closed pose.
- `tools/probe.py`: simulator helper, see below.

## How the asset is used

A scenario loads it as a fixed fixture on a table. A robot gripper (fingers open up to 8 cm,
about 2 cm wide) grasps a handle from above, pulls it along its joint, and lets go. Later a small
object (about 4 cm) is lowered onto a place target and must come to rest there. On a moving part
the robot lowers it at `lip_local + axis * q / 2` in the asset frame (q = the joint position at
that moment): the middle of the stretch the joint has pulled out, at the height of the box top.
Elsewhere it is the centre of the box top. The simulator only
collides the named collision boxes; the visual meshes never touch anything. A jointed part is
carried on its joint and cannot sag or leave its axis, so runners and rails need not be modelled;
a sliding part counts as held as long as it still overlaps the part it slides in, along the joint,
at full travel.

## Simulator

`./run_sim.sh <script.py>` runs a Python script against this directory in the simulator (about
one minute per run) and prints its output. Use at most 6 runs. Keep each script under `probes/`
and save what it printed next to it (`probes/01_open.py` -> `probes/01_open.out`).

```python
from probe import Probe
p = Probe("asset", cubes={"c": ((0.04, 0.04, 0.04), (0.2, 0.0, 0.3))})  # optional free test boxes, 0.1 kg
p.step(30)                         # 30 frames = 1 s
p.joints()                         # {joint: position}
p.set_joint("j", 0.1, velocity=0)  # teleport a joint
p.drive_joint("j", 0.2)            # drag a joint there like a gripper would, then release
p.link_pose("link")                # world position, rotation
p.box_world()                      # {collision box: (lo, hi)} world bounds at the current pose
p.box_overlaps(0.001)              # collision boxes on different links that interpenetrate
p.cube_pos("c")                    # where a test box is now
p.push("link", (0, 0, -20), steps=30)  # world force in N on a link
p.contacts()                       # current contacts with force
```

Joints are passive with damping, as in a scenario: a released joint keeps most of where it was put.

## Check at least

1. The asset matches the request: names, overall size, parts and joints.
2. The collision boxes follow the visual model; nothing the robot or an object must reach into is
   filled in, and nothing visible is missing.
3. Every joint works over its whole declared range: parts that move stay attached to, and
   supported by, the part they slide or turn in, and links do not interpenetrate.
4. Each handle can be grasped from above and pulled along its joint.
5. An object put on each place target comes to rest on it, inside any rim.

## Output

Write `review.json`:

```json
{
  "decision": "accept",
  "claims": [
    {"claim": "short statement", "verdict": "pass", "evidence": "readings.json key, render, or probes/NN_x.out with the numbers"}
  ],
  "feedback": "on reject: what the generator must change, concrete enough to act on"
}
```

`decision` is `accept` only if every claim passes. `verdict` is `pass` or `fail`.
