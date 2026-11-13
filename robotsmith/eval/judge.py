"""The exam's verdict, read from a judge log alone: no simulator, no evaluatee, no LLM.

Contacts are judged by ``gate.certified`` under a context the physics gives at each step: the task's one moved
object is the target, the table and the success tree's surfaces are rest, and the target is held when two
different finger links touch it. Nothing the evaluatee writes is read."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from robotsmith.eval.log import read
from robotsmith.execution.gate import FINGER_LINKS, SUPPORT, Span, certified
from robotsmith.tasks.predicates import evaluate_success
from robotsmith.tasks.task_spec import SuccessNode

ENDINGS = ("done", "give_up", "horizon", "policy_error")


def exam_context(success: dict) -> tuple[str, set[str]]:
    """``(target, rest)`` of a success tree: its one moved object, and the table plus the surfaces it names."""

    def leaves(node: dict):
        if node["op"] == "leaf":
            yield node["params"]
        for term in node.get("terms", []):
            yield from leaves(term)

    params = list(leaves(success))
    moved = {p["object"] for p in params if "object" in p}
    if len(moved) != 1:
        raise ValueError(f"the exam judges tasks that move one object; this one moves {sorted(moved)}")
    return moved.pop(), {SUPPORT, *(p["parent"] for p in params if "parent" in p)}


def held(contacts, target: str) -> bool:
    fingers = set()
    for a, b in contacts:
        for (entity, link), (other, _) in ((a, b), (b, a)):
            if entity == "robot" and link in FINGER_LINKS and other == target:
                fingers.add(link)
    return len(fingers) >= 2


def violations(contacts, target: str, rest: set[str]) -> list:
    h = held(contacts, target)
    span = Span("exam", 0, 1, holding=h, allow="rest" if h else "grip")
    return [p for p in contacts if not certified(p, span, target, rest)]


@dataclass(frozen=True)
class Verdict:
    seed: int | None
    attempt: int | None
    ended: str | None
    success: bool | None
    certified: bool
    physics_steps: int
    violating_steps: int
    first_violation: dict | None
    env_error: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def judge(path: str | Path) -> Verdict:
    lines = read(path)
    head = lines[0] if lines and lines[0]["kind"] == "header" else None
    if head is None:
        return Verdict(None, None, None, None, False, 0, 0, None, "no header line")
    target, rest = exam_context(head["success"])
    steps = bad_steps = 0
    first = None
    for line in lines:
        if line["kind"] != "physics":
            continue
        steps += 1
        bad = violations(line["contacts"], target, rest)
        if bad:
            bad_steps += 1
            if first is None:
                first = {"k": line["k"], "pair": bad[0], "held": held(line["contacts"], target)}
    final = [line for line in lines if line["kind"] == "final"]
    if not final:
        return Verdict(head["seed"], head["attempt"], None, None, False, steps, bad_steps, first, "no final line")
    success = bool(evaluate_success(SuccessNode.from_dict(head["success"]), final[-1]["state"]))
    return Verdict(head["seed"], head["attempt"], final[-1]["ended"], success, success and not bad_steps, steps,
                   bad_steps, first, None)
