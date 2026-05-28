"""Self-analysis reports from eval history."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

RESULTS_PATH = Path("evals/results.json")
REPORT_PATH = Path("evals/improvement_report.md")


def load_latest_results(path: Path = RESULTS_PATH) -> dict[str, Any]:
    """Load latest eval results if present."""

    if not path.exists():
        return {"summary": {}, "category_summary": {}, "recurring_failures": {}, "lessons": []}
    return json.loads(path.read_text(encoding="utf-8"))


def build_self_analysis(results: dict[str, Any]) -> dict[str, Any]:
    """Compute strongest skills, weakest skills, and improvement proposals."""

    category_summary = results.get("category_summary", {})
    ranked = sorted(
        category_summary.items(),
        key=lambda item: (item[1].get("pass_rate", 0.0), item[1].get("average_score", 0.0)),
        reverse=True,
    )
    strongest = [category for category, _ in ranked[:3]]
    weakest = [category for category, _ in ranked[-3:]][::-1] if ranked else []
    recurring = results.get("recurring_failures", {})
    lessons = results.get("lessons", [])
    proposed = [
        lesson["lesson"] if isinstance(lesson, dict) and "lesson" in lesson else str(lesson)
        for lesson in lessons[:10]
    ]
    if not proposed and category_summary:
        low_categories = [
            category
            for category, values in category_summary.items()
            if values.get("pass_rate", 0.0) < 0.9 or values.get("average_score", 0.0) < 0.8
        ]
        proposed = [f"Add targeted regression tests for {category}." for category in low_categories]

    return {
        "strongest_skills": strongest,
        "weakest_skills": weakest,
        "recurring_failure_patterns": recurring,
        "proposed_improvements": proposed,
        "overall_average_score": mean(
            values.get("average_score", 0.0) for values in category_summary.values()
        )
        if category_summary
        else 0.0,
    }


def render_report(analysis: dict[str, Any]) -> str:
    """Render markdown self-analysis."""

    lines = [
        "# Self-Analysis Report",
        "",
        f"Overall average category score: {analysis['overall_average_score']:.2f}",
        "",
        "## Strongest Skills",
    ]
    lines.extend(f"- {skill}" for skill in analysis["strongest_skills"] or ["No eval data yet."])
    lines.extend(["", "## Weakest Skills"])
    lines.extend(f"- {skill}" for skill in analysis["weakest_skills"] or ["No eval data yet."])
    lines.extend(["", "## Recurring Failure Patterns"])
    patterns = analysis["recurring_failure_patterns"]
    if patterns:
        lines.extend(f"- {pattern}: {count}" for pattern, count in patterns.items())
    else:
        lines.append("- No recurring failures in the latest run.")
    lines.extend(["", "## Proposed Improvements"])
    lines.extend(
        f"- {improvement}" for improvement in analysis["proposed_improvements"] or ["Keep current regression suite running."]
    )
    return "\n".join(lines) + "\n"


def run_self_analysis() -> str:
    """Generate and save the self-analysis report."""

    report = render_report(build_self_analysis(load_latest_results()))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    return report


def main() -> None:
    """CLI entrypoint."""

    print(run_self_analysis())


if __name__ == "__main__":
    main()
