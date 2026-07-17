"""Presentation model for the CoNIC dashboard.

The experiment matrix accumulates free-text ``status`` strings and long
``notes`` narrations as runs complete. This module turns those into a closed
vocabulary the dashboard can render as chips, a recipe/findings split, and the
experiment trajectory. It is deliberately pure: every function takes and returns
plain data, so both a full render and a cached re-render produce the same page.
"""
from __future__ import annotations

import re

# Verdict vocabulary. Order matters only for display grouping.
OUTCOMES = {
    "promoted": "Promoted",
    "mixed": "Partly promoted",
    "rejected": "Not adopted",
    "running": "Running",
    "planned": "Planned",
    "cancelled": "Cancelled",
    "baseline": "Baseline",
    "external": "External control",
}

OUTCOME_HINTS = {
    "promoted": "Part of the selected final model.",
    "mixed": "Promoted for one objective or one follow-up only; not the final model.",
    "rejected": "Scored but not part of the final model — it failed a matched control or was superseded.",
    "running": "In flight; numbers shown are provisional.",
    "planned": "Designed, not yet run.",
    "cancelled": "Stopped before a verdict.",
    "baseline": "The reference the deltas are measured against.",
    "external": "Reproduction on someone else's fold; never differenced against our runs.",
}

_REJECT_PATTERNS = (
    "rejected",
    "not promoted",
    "failed guard",
    "failed transfer",
    "did not transfer",
    "indistinguishable from zero",
)


def classify_outcome(row: dict) -> str:
    """Map a row's free-text status and kind onto the verdict vocabulary."""
    kind = (row.get("kind") or "").lower()
    status = (row.get("status") or "").lower()
    if kind == "baseline":
        return "baseline"
    if kind == "benchmark":
        return "external"
    if kind == "future-best" or status in {"not run yet", "planned"}:
        return "planned"
    if "cancelled" in status or "deprioritized" in status:
        return "cancelled"

    rejected = any(pattern in status for pattern in _REJECT_PATTERNS)
    # "not promoted" also contains "promoted", so demand a promotion that is not
    # the negated form before believing it.
    promoted = "promoted" in re.sub(r"not promoted", "", status)
    if promoted and rejected:
        return "mixed"
    if promoted:
        # E41-style rows advance to a follow-up audit rather than to a recipe.
        return "mixed" if "not standalone" in status else "promoted"
    if rejected:
        return "rejected"
    if any(word in status for word in ("training", "live", "running", "waiter", "awaits")):
        return "running"
    if status.startswith("implementation validated") or "implemented" in status:
        return "running"
    # Anything scored but not promoted is, in practice, not adopted — the same
    # outcome as an explicit rejection from the reader's point of view.
    return "rejected"


def split_note(row: dict, recipe_lookup: dict[str, str] | None = None) -> tuple[str, str]:
    """Separate the fixed recipe description from accumulated result narration.

    ``render_review`` records both fields directly. For a cached summary written
    before that change, fall back to stripping the known recipe prefix.
    """
    if row.get("recipe"):
        return row["recipe"], row.get("findings", "")
    notes = row.get("notes", "") or ""
    recipe = (recipe_lookup or {}).get(row.get("id", ""))
    if recipe and notes.startswith(recipe):
        return recipe, notes[len(recipe):].strip()
    return notes, ""


def experiment_number(identifier: str) -> int | None:
    """Sequence position from an experiment ID (``E04-stack`` -> 4)."""
    match = re.match(r"E(\d+)", identifier or "")
    return int(match.group(1)) if match else None


def is_comparable(row: dict) -> bool:
    """Whether a row's metrics may be differenced against our internal baseline.

    Mirrors the promotion rule in ``render_review.build_performance_summary`` so
    the trajectory's frontier ends exactly on the headline best.
    """
    return (
        row.get("r2") is not None
        and row.get("mpq") is not None
        and row.get("kind") != "benchmark"
        and not row.get("leaderboard_ineligible")
    )


def normalize_rows(rows: list[dict], recipe_lookup: dict[str, str] | None = None) -> list[dict]:
    """Attach verdict, recipe/findings, and sequence fields to each row."""
    normalized = []
    for row in rows:
        item = dict(row)
        outcome = classify_outcome(row)
        recipe, findings = split_note(row, recipe_lookup)
        item["outcome"] = outcome
        item["outcome_label"] = OUTCOMES[outcome]
        item["outcome_hint"] = OUTCOME_HINTS[outcome]
        item["recipe"] = recipe
        item["findings"] = findings
        item["sequence"] = experiment_number(row.get("id", ""))
        item["comparable"] = is_comparable(row)
        normalized.append(item)
    return normalized


def outcome_tally(rows: list[dict]) -> list[dict]:
    """Count rows per verdict, in vocabulary order, skipping empty buckets."""
    counts: dict[str, int] = {}
    for row in rows:
        outcome = row.get("outcome") or classify_outcome(row)
        counts[outcome] = counts.get(outcome, 0) + 1
    return [
        {"outcome": name, "label": OUTCOMES[name], "count": counts[name], "hint": OUTCOME_HINTS[name]}
        for name in OUTCOMES
        if counts.get(name)
    ]


def build_trajectory(rows: list[dict], targets: dict) -> dict:
    """Experiment-ordered scores plus the running-best frontier per metric.

    Only rows carrying directly comparable internal-test numbers are plotted, so
    the frontier is a truthful record of what our own held-out test ever saw.
    """
    points = []
    for row in rows:
        sequence = row.get("sequence", experiment_number(row.get("id", "")))
        if sequence is None or not is_comparable(row):
            continue
        points.append(
            {
                "id": row.get("id"),
                "sequence": sequence,
                "method": row.get("method", ""),
                "outcome": row.get("outcome") or classify_outcome(row),
                "r2": float(row["r2"]),
                "mpq": float(row["mpq"]),
            }
        )
    points.sort(key=lambda item: (item["sequence"], item["id"] or ""))

    series = {}
    for metric in ("r2", "mpq"):
        best = float("-inf")
        frontier = []
        for point in points:
            if point[metric] > best:
                best = point[metric]
                frontier.append({"sequence": point["sequence"], "value": best, "id": point["id"]})
        series[metric] = {
            "target": targets.get(metric),
            "frontier": frontier,
            "best": best if frontier else None,
        }
    return {"points": points, "series": series}
