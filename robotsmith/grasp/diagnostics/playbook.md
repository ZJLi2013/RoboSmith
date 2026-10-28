Match the evidence against the signatures below. A direction is supported only when its signature's readings are
present; if no signature matches, answer direction unknown, owner unknown, next step report_unknown.

| signature (what the readings show) | direction | owner | next step |
|---|---|---|---|
| Planner object points / candidate tips are offset from the simulator's collision AABB by a constant amount (often equal to a URDF origin such as the inertial origin) | coordinate_frame | primitive | report_primitive_defect |
| Recorded finger opening (sum of the two finger joints) differs from the candidate's closing width, e.g. the fingers close across the other box axis | closing_axis | primitive | report_primitive_defect |
| At the end of closing a finger touches the table instead of the target, closing speed drops sharply before the target width, no two-finger target contact | support_clearance | primitive | report_primitive_defect |
| Both fingers touch the target and stop, yet lift is ~0; lowering the object's mass (diagnostic tool) makes the same candidates lift | grip_control | primitive | report_primitive_defect |
| The object settled far from the requested position or off the work area before any grasp (large settle displacement, z below the table) | scene_placement | scene | flag_scene_placement |
| The planner produced no usable candidate (e.g. every box axis wider than the gripper opening) while the object has a narrower graspable part (see horizontal extent by height) | add_candidate | asset | propose_candidate_pose |

Notes:
- IK rejection by itself ("IK discontinuity") is not a diagnosed signature; without other readings it is unknown.
- Owner primitive means every asset is affected: report it, do not patch the asset or tune the controller for it.
