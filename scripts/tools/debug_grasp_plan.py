"""Quick debug: print the grasp plan for bowl category."""
from robotsmith.grasp import TemplateGraspPlanner
import numpy as np

table_z = 0.775
h = 0.0331
p = TemplateGraspPlanner(z_offset=table_z)
plans = p.plan(np.array([0.5, 0.0, 0.79]), category="bowl", object_height=h)
g = plans[0]
bowl_top = table_z + h
print(f"table_z={table_z}, bowl_h={h}, bowl_top={bowl_top:.4f}")
print(f"grasp_pos      = {g.grasp_pos}  (z vs bowl_top: {g.grasp_pos[2]-bowl_top:+.4f})")
print(f"pre_grasp_pos  = {g.pre_grasp_pos}")
print(f"side_approach   = {g.side_approach_pos}")
print(f"retreat_pos    = {g.retreat_pos}")
print(f"finger_open={g.finger_open} closed={g.finger_closed}")
