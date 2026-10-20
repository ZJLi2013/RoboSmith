import numpy as np

from robotsmith.grasp.bucket_policy import pose_axis_metrics
from robotsmith.grasp.planner import GraspPlan


def _plan(*, approach_dir, finger_dir) -> GraspPlan:
    return GraspPlan(
        grasp_pos=np.zeros(3),
        grasp_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        pre_grasp_pos=np.zeros(3),
        pre_grasp_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        retreat_pos=np.array([0.0, 0.0, 0.1]),
        retreat_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        finger_open=0.04,
        finger_closed=0.01,
        metadata={
            "approach_dir": approach_dir,
            "finger_dir": finger_dir,
        },
    )


def test_pose_axis_metrics_normalizes_axes():
    plan = _plan(approach_dir=[0.0, 0.0, 2.0], finger_dir=[3.0, 0.0, 0.0])

    finger_z, palm_z, path_z = pose_axis_metrics(plan)

    assert finger_z == 0.0
    assert palm_z == 0.0
    assert path_z == 1.0
