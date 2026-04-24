"""Quick debug: run 1-episode pick_bowl with detailed logging."""
from robotsmith.grasp import TemplateGraspPlanner
from robotsmith.gen.franka import to_numpy
import numpy as np

table_z = 0.775
h = 0.0331
p = TemplateGraspPlanner(z_offset=table_z)
plans = p.plan(np.array([0.5, 0.0, 0.79]), category="bowl", object_height=h)
g = plans[0]
bowl_top = table_z + h
bowl_mid = table_z + h / 2
print("=== Grasp Plan ===")
print(f"table_z={table_z}, bowl_h={h}")
print(f"bowl_top={bowl_top:.4f}, bowl_mid={bowl_mid:.4f}")
print(f"grasp_z={g.grasp_pos[2]:.4f}  (vs bowl_top: {g.grasp_pos[2]-bowl_top:+.4f})")
print(f"1. pre_grasp   = [{g.pre_grasp_pos[0]:.3f}, {g.pre_grasp_pos[1]:.3f}, {g.pre_grasp_pos[2]:.3f}]")
if g.side_approach_pos is not None:
    print(f"2. side_appr   = [{g.side_approach_pos[0]:.3f}, {g.side_approach_pos[1]:.3f}, {g.side_approach_pos[2]:.3f}]")
print(f"3. grasp       = [{g.grasp_pos[0]:.3f}, {g.grasp_pos[1]:.3f}, {g.grasp_pos[2]:.3f}]")
print(f"4. retreat     = [{g.retreat_pos[0]:.3f}, {g.retreat_pos[1]:.3f}, {g.retreat_pos[2]:.3f}]")
print(f"finger: open={g.finger_open}, closed={g.finger_closed}")
print()

# Check the bowl diameter at this scale to see if gripper can reach
# bowl_02 at scale=0.5, bounding box x-extent ~ 0.14m -> diameter ~ 14cm
# Franka max opening = 0.08m = 8cm -> needs to grip at narrow point
# At scale=0.5, the bowl visual mesh x-extent was [-0.07, 0.07] = 14cm
# That's wider than Franka's 8cm opening!
print("=== Gripper vs Bowl Size ===")
print(f"finger_open = {g.finger_open}m = {g.finger_open*100:.1f}cm")
print("Franka max opening = 0.08m = 8cm")
print("bowl_02 @ scale=0.5 diameter ~ 14cm -> TOO WIDE for gripper!")
print("bowl_02 @ scale=0.35 diameter ~ 9.8cm -> still too wide")
print("The mid-wall grasp works by gripping the narrower upper rim or tilted portion")
print()

# Check actual object_height_m computed by backend
print("=== What does object_height_m compute? ===")
print("object_height_m is world-frame Z extent after rotation.")
print("For bowl_02, the mesh is Y-up, rotated to Z-up by stable pose quat.")
print(f"h={h} means the bowl is {h*100:.1f}cm tall in world frame.")
