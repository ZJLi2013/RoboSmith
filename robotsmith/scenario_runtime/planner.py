"""Agent planning: an LLM writes teacher-schema steps from the instruction and the scene's objects, the existing
skills execute them with per-step judging, and plan-level failures go back to the LLM.

Two roles: ``certifier`` also sees the success tree (its successful plan proves the task solvable); ``baseline``
does not. No memory is kept across episodes.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from robotsmith.scenario.adapters import bindings_from_steps, to_legacy_skills
from robotsmith.scenario.load import steps_to_intents
from robotsmith.scenario_runtime.runner import EpisodeRun, collect_object_positions, reset_scene

ROLES = ("certifier", "baseline")
MAX_REPLANS = 2
SKILLS_DOC = ("pick <object>  |  place <object>.<target> (places the held object)  |  "
              "open <object>  |  close <object>")


class PlanError(ValueError):
    """The agent's reply is not a valid step list for this scene."""


def scene_objects(scene, assets: dict, positions: dict) -> list[dict]:
    """The object list the agent reads: id, asset, movable or fixed, live position, size, targets, joints."""
    out = []
    for obj in scene.objects:
        asset = assets[obj.name]
        pos = positions.get(obj.name, obj.fixed_position)
        out.append({
            "id": obj.name,
            "asset": obj.asset,
            "kind": "fixed" if asset.is_fixture else "movable",
            "pos": [round(float(v), 3) for v in np.asarray(pos).reshape(-1)[:3]],
            "size": [round(v / 100.0, 3) for v in asset.metadata.size_cm],
            "place_targets": [f"{obj.name}.{t['name']}" for t in asset.metadata.place_targets],
            "joints": [j.name for j in asset.movable_joints],
        })
    return out


def render_request(instruction: str, objects: list[dict], *, success: dict | None = None,
                   done: list[dict] | None = None, failure: dict | None = None, held: str | None = None) -> str:
    lines = [f"Task: {instruction}", "Objects (world frame, metres):"]
    for o in objects:
        extra = []
        if o["place_targets"]:
            extra.append("place targets: " + ", ".join(o["place_targets"]))
        if o["joints"]:
            extra.append("joints: " + ", ".join(o["joints"]))
        lines.append(f"  {o['id']}  {o['asset']}  {o['kind']}  pos {o['pos']}  size {o['size']}"
                     + (f"  ({'; '.join(extra)})" if extra else ""))
    lines.append(f"Skills: {SKILLS_DOC}")
    if success is not None:
        lines.append(f"Success condition: {json.dumps(success)}")
    if done is not None:
        lines.append(f"Done so far: {json.dumps(done)}")
        lines.append(f"Failed step: {json.dumps(failure)}")
        lines.append(f"Held now: {held or 'nothing'}")
        lines.append("Reply with the remaining steps only.")
    lines.append('Reply with JSON only: {"steps": [{"pick": "<object>"}, {"place": {"target": "<object>.<target>"}}]}')
    return "\n".join(lines)


def call_llm(prompt: str) -> tuple[str, int]:
    """One OpenAI Responses call through the gateway named by ``ROBOSMITH_LLM_*``; returns the text and tokens."""
    base = os.environ["ROBOSMITH_LLM_BASE_URL"].rstrip("/")
    url = f"{base}/responses?api-version={os.environ['ROBOSMITH_LLM_API_VERSION']}"
    headers = {"content-type": "application/json", "Ocp-Apim-Subscription-Key": os.environ["ROBOSMITH_LLM_KEY"]}
    body = {"model": os.environ["ROBOSMITH_LLM_MODEL"], "input": prompt, "max_output_tokens": 4000,
            "reasoning": {"effort": "low"}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read())
    text = "".join(part.get("text", "") for item in data.get("output", []) if item.get("type") == "message"
                   for part in item.get("content", []))
    return text, int(data.get("usage", {}).get("total_tokens", 0))


def parse_steps(text: str, assets: dict, yaw_deg: dict, held: str | None = None):
    """The reply's steps as intents and bindings; ``PlanError`` when they do not fit the scene."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise PlanError("no JSON object in the reply")
    try:
        steps = json.loads(match.group(0))["steps"]
        intents, _ = steps_to_intents(steps, assets)
        bindings = bindings_from_steps(intents, assets, yaw_deg, held=held)
    except Exception as exc:
        raise PlanError(str(exc)) from exc
    if any(b.get("reason") == "no preceding pick" for b in bindings):
        raise PlanError("place with nothing held")
    return steps, intents, bindings


def route_failure(entry: dict, positions: dict, binding: dict | None) -> str:
    """``plan`` / ``physical`` / ``unknown`` from execution evidence, per the M6 contract."""
    if entry.get("skill") == "pick" or entry.get("cause") in ("grip?", "motion?"):
        return "physical"
    if entry.get("skill") == "place" and binding and binding.get("status") == "bound":
        params = binding["params"]
        lo, hi = np.asarray(params["box"][0], dtype=float), np.asarray(params["box"][1], dtype=float)
        parent = np.asarray(positions.get(params["parent"], [0, 0, 0]), dtype=float)
        c, s = np.cos(np.radians(params["yaw_deg"])), np.sin(np.radians(params["yaw_deg"]))
        for name, pos in positions.items():
            if name in (params["object"], params["parent"]):
                continue
            dx, dy = np.asarray(pos, dtype=float)[:2] - parent[:2]
            local = np.array([c * dx + s * dy, -s * dx + c * dy])
            if np.all(local >= lo[:2]) and np.all(local <= hi[:2]):
                return "plan"
    return "unknown"


@dataclass
class PlannerRecord:
    role: str
    plans: list[list[dict]] = field(default_factory=list)
    routing: list[dict] = field(default_factory=list)
    tokens: int = 0
    stopped: str | None = None

    def to_dict(self) -> dict:
        return {"role": self.role, "plans": self.plans, "routing": self.routing, "tokens": self.tokens,
                "stopped": self.stopped}


def run_agentic_episode(env, package, assets: dict, *, role: str, executor, motion_params, dataset,
                        record_fn: Callable, settle_steps: int, target_frames: dict, camera_names,
                        env_state_names, llm: Callable[[str], tuple[str, int]] = call_llm):
    """Plan, execute, judge each step; on a plan-level failure re-plan the rest (at most ``MAX_REPLANS``)."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    task, scene = package.legacy_task, package.scenario_scene
    yaw_deg = {o.name: o.yaw_deg for o in scene.objects}
    success = task.success.to_dict() if role == "certifier" else None
    record = PlannerRecord(role=role)
    reset_scene(env, settle_steps)
    run = EpisodeRun(env, task, executor=executor, motion_params=motion_params, dataset=dataset,
                     record_fn=record_fn, target_frames=target_frames, camera_names=camera_names,
                     env_state_names=env_state_names)
    done: list[dict] = []
    failure: dict | None = None
    for attempt in range(MAX_REPLANS + 1):
        objects = scene_objects(scene, assets, collect_object_positions(env))
        prompt = render_request(task.instruction, objects, success=success,
                                done=done if attempt else None, failure=failure, held=run.held)
        text, tokens = llm(prompt)
        record.tokens += tokens
        try:
            steps, intents, bindings = parse_steps(text, assets, yaw_deg, held=run.held)
        except PlanError as exc:
            record.plans.append([])
            failure = {"route": "plan", "reason": f"invalid reply: {exc}"}
            record.routing.append(failure)
            continue
        record.plans.append(steps)
        skills = to_legacy_skills(intents, package.scenario_task)
        start = len(run.skills)
        failed = run.run_steps(skills, bindings, stop_on_fail=True)
        if failed is None:
            record.stopped = "plan completed"
            return run.result(), record
        k = len(run.skills) - start - 1
        done.extend(steps[:k])
        route = route_failure(failed, {n: np.asarray(p).tolist() for n, p in collect_object_positions(env).items()},
                              bindings[k] if k < len(bindings) else None)
        failure = {"route": route, "step": steps[k], "skill": failed.get("skill"), "cause": failed.get("cause")}
        record.routing.append(failure)
        if route != "plan":
            record.stopped = f"{route} failure reported"
            return run.result(), record
    record.stopped = "re-plan budget used"
    return run.result(), record
