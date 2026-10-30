"""Mistake clustering (plan Task 28, hard-problems §8): graded mistakes grouped by the reason
keys of the best move, printed as ``skill.*`` candidates beside the app's skill list.

Input: the ``mistake`` lines of ``big2-teach mine``'s ``raw.jsonl`` (``best_keys``, ``loss``,
``grade``, ``level``). ``reason.value.gap`` and ``reason.counterfactual.outcomes`` fire on
almost every decision, so they are left out of the key sets.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from big2.teach.drills import APP_PROMPTS, SKILL_RULES

#: Keys that say nothing about the kind of mistake.
IGNORED = frozenset({"reason.value.gap", "reason.counterfactual.outcomes"})
NONE = "(none)"


@dataclass
class Cluster:
    keys: tuple[str, ...]
    count: int
    mean_loss: float
    skill: str
    grades: Counter[str] = field(default_factory=Counter)
    levels: Counter[str] = field(default_factory=Counter)

    @property
    def in_app(self) -> bool:
        return self.skill in APP_PROMPTS


def skill_for_keys(keys: tuple[str, ...]) -> str:
    for key, skill in SKILL_RULES:
        if key in keys:
            return skill
    if "reason.tempo.plays_to_out" in keys:
        return "skill.finish_in_two"
    return "skill.general"


def clusters(raw: list[dict[str, Any]], min_size: int = 5) -> list[Cluster]:
    """Mistake clusters with at least ``min_size`` members, largest first."""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for m in raw:
        if m.get("type") != "mistake":
            continue
        keys = tuple(sorted(set(m.get("best_keys", [])) - IGNORED)) or (NONE,)
        groups.setdefault(keys, []).append(m)
    out = []
    for keys, ms in groups.items():
        if len(ms) < min_size:
            continue
        out.append(
            Cluster(
                keys=keys,
                count=len(ms),
                mean_loss=sum(float(m["loss"]) for m in ms) / len(ms),
                skill=skill_for_keys(keys),
                grades=Counter(m["grade"] for m in ms),
                levels=Counter(m["level"] for m in ms),
            )
        )
    out.sort(key=lambda c: -c.count)
    return out


def format_clusters(cs: list[Cluster], app_skills: list[str] | None = None) -> str:
    """A table of clusters; ``app_skills`` (the app's list) marks the unmatched ones."""
    app = set(app_skills if app_skills is not None else APP_PROMPTS)
    lines = ["count  mean loss  skill                    in app  keys"]
    for c in cs:
        mark = "yes" if c.skill in app else "NEW"
        lines.append(
            f"{c.count:>5}  {c.mean_loss:9.2f}  {c.skill:<24} {mark:<6}  {', '.join(c.keys)}"
        )
    unmatched = sorted({c.skill for c in cs} - app)
    lines.append("")
    lines.append("new-skill candidates: " + (", ".join(unmatched) if unmatched else "none"))
    missing = sorted(app - {c.skill for c in cs})
    lines.append("app skills with no cluster: " + (", ".join(missing) if missing else "none"))
    return "\n".join(lines)
