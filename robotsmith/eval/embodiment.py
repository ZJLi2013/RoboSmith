"""A RoboSmith scenario as an Inspect Robots ``Embodiment``: an external policy sends absolute fingertip targets,
RoboSmith steps the physics and writes the judge log. No execution gate checks the evaluatee's motion."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from inspect_robots.embodiment import SEEDABLE, EmbodimentInfo
from inspect_robots.scene import Scene
from inspect_robots.spaces import ActionSemantics, Box, ObservationSpace, StateField, StateSpec
from inspect_robots.types import Action, Observation, StepResult

from robotsmith.eval.judge import exam_context, held, violations
from robotsmith.eval.log import JudgeLog
from robotsmith.grasp.geometric import GripperSpec
from robotsmith.scenario.load import load_scenario
from robotsmith.sim.franka import to_numpy

CONTROL_HZ = 10.0
SETTLE_STEPS = 30
GRIPPER = GripperSpec()
LABELS = ("x", "y", "z", "gripper")
DOCS = """\
Franka Panda arm, one hand pointing straight down; the hand's yaw is fixed.
Actions are absolute targets in the world frame, metres: x, y, z of the fingertip centre (between the two finger
pads), and gripper from 0 (closed) to 1 (open, 8 cm between the fingers). The table top is at z = {table_z:.3f}.
state[eef_state] reads the fingertip centre and the gripper opening the same way.
state[object_pos.<name>] is the simulator's centre of each scene object (privileged information).
Objects: {objects}."""


class ExamMonitor:
    """The one reader of ``SimEnv.on_step`` in an exam run: logs each physics step's contacts and counts the steps
    ``gate.certified`` rejects under the context the physics gives."""

    def __init__(self, env, log: JudgeLog, target: str, rest: set[str]):
        self.log, self.target, self.rest = log, target, rest
        self.k = self.violating = 0
        env.on_step = self._on_step

    def _on_step(self, env) -> None:
        contacts = env.contacts_now()
        if violations(contacts, self.target, self.rest):
            self.violating += 1
        self.log.physics(self.k, contacts, held(contacts, self.target))
        self.k += 1


class RoboSmithEmbodiment:
    """One attempt at one seed of a scenario. ``close(ended)`` settles, judges the success tree and writes
    ``final``; the caller owns it (``eval`` does not close an embodiment it was handed)."""

    def __init__(self, scenario: str | Path, *, seed: int, attempt: int, log_path: str | Path):
        self.scenario_path = Path(scenario)
        self.seed, self.attempt, self.log_path = seed, attempt, Path(log_path)
        self.scenario = load_scenario(self.scenario_path, seed=seed, agent_plan=True)
        self.target, self.rest = exam_context(self.scenario.task.success.to_dict())
        self.objects = [o.name_override for o in self.scenario.scene.objects]
        cfg = self.scenario.scene
        self.table_z = cfg.table_height + cfg.table_size[2] / 2.0
        self.sha256 = hashlib.sha256(self.scenario_path.read_bytes()).hexdigest()
        low = np.array([0.2, -0.5, self.table_z, 0.0])
        high = np.array([0.8, 0.5, self.table_z + 0.5, 1.0])
        self.info = EmbodimentInfo(
            name="robosmith",
            action_space=Box(shape=(4,), low=low, high=high, semantics=ActionSemantics(
                control_mode="eef_abs_pose", gripper="continuous", frame="world", dim_labels=LABELS)),
            observation_space=ObservationSpace(state=StateSpec(fields=(
                StateField("eef_state", (4,), "m"),
                *(StateField(f"object_pos.{n}", (3,), "m") for n in self.objects)))),
            control_hz=CONTROL_HZ,
            is_simulated=True,
            capabilities=frozenset({SEEDABLE}),
            docs=DOCS.format(table_z=self.table_z, objects=", ".join(self.objects)),
            environment_id=self.scenario.task.name,
            environment_revision=self.sha256[:12],
            morphology="robosmith-franka",
        )
        self.env = self.log = self.monitor = None
        self.closed = False

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        from robotsmith.scenario_runtime.runner import collect_object_positions, reset_scene
        from robotsmith.sim.sim_env import SimEnv

        if self.log is not None:
            raise RuntimeError("one RoboSmithEmbodiment runs one attempt")
        self.env = env = SimEnv.build(self.scenario.scene, seed=self.seed)
        reset_scene(env, SETTLE_STEPS)
        self.initial_positions = collect_object_positions(env)
        self.q = to_numpy(env.franka.get_dofs_position(env.motors_dof)).astype(np.float32)
        self.quat = to_numpy(env.end_effector.get_quat()).astype(np.float32).reshape(-1)
        self.steps = 0
        self.log = JudgeLog(self.log_path)
        self.log.header(seed=self.seed, attempt=self.attempt, scenario=self.scenario_path.as_posix(),
                        scenario_sha256=self.sha256, success=self.scenario.task.success.to_dict(),
                        target=self.target, rest=sorted(self.rest))
        self.monitor = ExamMonitor(env, self.log, self.target, self.rest)
        return self._observe()

    def step(self, action: Action) -> StepResult:
        from robotsmith.sim.world_state import collect_world_state

        a = np.clip(np.asarray(action.data, dtype=np.float64), self.info.action_space.low,
                    self.info.action_space.high)
        hand = a[:3] + np.array([0.0, 0.0, GRIPPER.tcp_offset])
        self.q = self.env.solve_ik(hand, quat=self.quat, finger_pos=float(a[3]) * GRIPPER.finger_open,
                                   init_qpos=self.q)
        self.env.franka.control_dofs_position(self.q, self.env.motors_dof)
        for _ in range(round(1.0 / (CONTROL_HZ * float(self.env.scene.dt)))):
            self.env.step()
        self.log.control(self.steps, a, collect_world_state(self.env))
        self.steps += 1
        return StepResult(observation=self._observe())

    def _observe(self) -> Observation:
        env = self.env
        tip = to_numpy(env.end_effector.get_pos()).astype(np.float64).reshape(-1)[:3]
        tip = tip - np.array([0.0, 0.0, GRIPPER.tcp_offset])
        fingers = to_numpy(env.franka.get_dofs_position(env.finger_dof)).astype(np.float64)
        state = {"eef_state": np.r_[tip, float(fingers.mean()) / GRIPPER.finger_open]}
        for name, ent in env.entity_map.items():
            state[f"object_pos.{name}"] = to_numpy(ent.get_pos()).astype(np.float64).reshape(-1)[:3]
        return Observation(state=state, instruction=self.scenario.task.instruction)

    def close(self, ended: str = "policy_error") -> None:
        """Settle holding the last command, judge the success tree, write ``final`` and end the Genesis session
        (one per process at a time, so the next attempt can build its scene)."""
        import genesis as gs

        from robotsmith.scenario_runtime.runner import evaluate_scenario_task, resolve_static_target_positions

        if self.closed or self.log is None:
            return
        self.closed = True
        targets = resolve_static_target_positions(self.env, self.scenario.target_frames)
        success, state = evaluate_scenario_task(self.env, self.scenario.task, initial_positions=self.initial_positions,
                                                reference_positions=targets, hold=self.q)
        self.log.final(ended=ended, success=bool(success), state=state)
        self.log.close()
        self.env.on_step = None
        gs.destroy()
