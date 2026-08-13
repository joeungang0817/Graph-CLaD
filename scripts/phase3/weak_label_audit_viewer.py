"""Interactive Colab viewer for the Phase 3 holding weak-label audit.

The viewer never assigns a verdict automatically.  It renders the causal
robot-object trajectory and action window for each selected audit item, then
persists explicit human decisions to the versioned review CSV and summary.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DECISIONS = ("pass", "label_error", "ambiguous")
ERROR_TYPES = (
    "false_onset",
    "missed_onset",
    "false_release",
    "missed_release",
    "hard_negative_is_holding",
    "contact_mapping_error",
    "temporal_alignment_error",
    "insufficient_evidence",
    "other",
)


def load_audit_bundle(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    evidence_path = root / "holding_weak_label_audit_evidence_v1.jsonl.gz"
    review_path = root / "holding_weak_label_audit_review_v1.csv"
    with gzip.open(evidence_path, "rt", encoding="utf-8") as handle:
        evidence = [json.loads(line) for line in handle if line.strip()]
    with review_path.open("r", encoding="utf-8-sig", newline="") as handle:
        review = list(csv.DictReader(handle))
    evidence_ids = [str(row["audit_id"]) for row in evidence]
    review_ids = [str(row["audit_id"]) for row in review]
    if evidence_ids != review_ids:
        raise ValueError("Evidence and review CSV audit_id order differs")
    return evidence, review


def review_summary(review: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    decisions = Counter(str(row.get("reviewer_decision", "")).strip() for row in review)
    reviewed = sum(decisions[name] for name in DECISIONS)
    error_types = Counter(
        str(row.get("label_error_type", "")).strip()
        for row in review
        if str(row.get("label_error_type", "")).strip()
    )
    per_cell: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in review:
        decision = str(row.get("reviewer_decision", "")).strip() or "unreviewed"
        key = f"task{row.get('task_id')}:{row.get('event_type')}"
        per_cell[key][decision] += 1
    return {
        "status": "complete" if reviewed == len(review) else ("in_progress" if reviewed else "pending"),
        "total": len(review),
        "reviewed": reviewed,
        "passed": decisions["pass"],
        "label_errors": decisions["label_error"],
        "ambiguous": decisions["ambiguous"],
        "pass_rate": decisions["pass"] / reviewed if reviewed else None,
        "error_type_counts": dict(sorted(error_types.items())),
        "task_event_counts": {
            key: dict(sorted(counts.items())) for key, counts in sorted(per_cell.items())
        },
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "claim_limit": "Human review summary; ambiguous items are not counted as passed.",
    }


def save_review_bundle(root: Path, review: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    review_path = root / "holding_weak_label_audit_review_v1.csv"
    manifest_path = root / "holding_weak_label_audit_manifest_v1.json"
    summary_path = root / "holding_weak_label_audit_review_summary_v1.json"
    fields = [
        "audit_id",
        "task_id",
        "event_type",
        "sample_id",
        "episode_id",
        "start_step",
        "target_step",
        "edge_source",
        "edge_target",
        "reviewer_decision",
        "label_error_type",
        "reviewer_notes",
    ]
    temporary = review_path.with_suffix(review_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in review)
    os.replace(temporary, review_path)
    summary = review_summary(review)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["review_summary"] = summary
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return summary


def _action_vector(value: Any) -> list[float]:
    if isinstance(value, Mapping):
        value = value.get("action", [])
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [float(item) for item in value]


def launch_audit_viewer(root: str | Path):
    """Display and return an interactive notebook audit viewer."""
    import html

    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import numpy as np
    from IPython.display import HTML, clear_output, display

    root = Path(root)
    evidence, review = load_audit_bundle(root)
    review_by_id = {str(row["audit_id"]): row for row in review}
    state = {"indices": list(range(len(evidence))), "position": 0, "loading": False}

    task_filter = widgets.Dropdown(
        options=[("all tasks", "all"), ("task 0", "0"), ("task 1", "1"), ("task 2", "2")],
        description="Task",
    )
    event_filter = widgets.Dropdown(
        options=[("all events", "all"), ("onset", "onset"), ("release", "release"), ("hard negative", "hard_negative")],
        description="Event",
    )
    status_filter = widgets.Dropdown(
        options=[("all", "all"), ("unreviewed", "unreviewed"), ("reviewed", "reviewed"), ("label error", "label_error"), ("ambiguous", "ambiguous")],
        description="Status",
    )
    previous_button = widgets.Button(description="← Previous")
    next_button = widgets.Button(description="Next →")
    save_button = widgets.Button(description="Save", button_style="success")
    save_next_button = widgets.Button(description="Save & Next", button_style="success")
    decision = widgets.ToggleButtons(
        options=[("unreviewed", ""), ("pass", "pass"), ("label error", "label_error"), ("ambiguous", "ambiguous")],
        description="Decision",
    )
    error_type = widgets.Dropdown(
        options=[("none", "")] + [(name, name) for name in ERROR_TYPES],
        description="Error",
    )
    notes = widgets.Textarea(description="Notes", layout=widgets.Layout(width="95%"))
    progress = widgets.HTML()
    message = widgets.HTML()
    output = widgets.Output()

    def current_pair():
        if not state["indices"]:
            return None, None
        index = state["indices"][state["position"]]
        row = evidence[index]
        return row, review_by_id[str(row["audit_id"])]

    def filtered_indices() -> list[int]:
        result = []
        for index, row in enumerate(evidence):
            reviewed_row = review_by_id[str(row["audit_id"])]
            verdict = str(reviewed_row.get("reviewer_decision", "")).strip()
            if task_filter.value != "all" and str(row["task_id"]) != task_filter.value:
                continue
            if event_filter.value != "all" and str(row["event_type"]) != event_filter.value:
                continue
            if status_filter.value == "unreviewed" and verdict:
                continue
            if status_filter.value == "reviewed" and not verdict:
                continue
            if status_filter.value in {"label_error", "ambiguous"} and verdict != status_filter.value:
                continue
            result.append(index)
        return result

    def evidence_table(row: Mapping[str, Any]) -> str:
        cells = []
        for label in ("current", "future"):
            holding = row[label]["edge"]["relations"]["holding"]
            contact = row[label]["edge"]["relations"]["contact"]
            proof = holding.get("evidence", {}) or {}
            cells.append(
                "<tr>"
                f"<td>{label}</td><td>{holding.get('value')}</td><td>{holding.get('state')}</td>"
                f"<td>{contact.get('value')}</td><td>{proof.get('closed_gripper')}</td>"
                f"<td>{proof.get('finger_contact')}</td><td>{proof.get('relative_pose_stable')}</td>"
                f"<td>{proof.get('object_followed_eef')}</td><td>{holding.get('confidence')}</td>"
                "</tr>"
            )
        return (
            "<table><tr><th>frame</th><th>holding</th><th>state</th><th>contact</th>"
            "<th>closed</th><th>finger contact</th><th>stable</th><th>followed</th>"
            "<th>confidence</th></tr>" + "".join(cells) + "</table>"
        )

    def render_plot(row: Mapping[str, Any]):
        trajectory = row.get("trajectory", []) or []
        actions = [_action_vector(value) for value in row.get("action_window", [])]
        steps = [int(point["step"]) for point in trajectory]
        distances = [point.get("distance") for point in trajectory]
        source = np.asarray([point.get("source_position") for point in trajectory], dtype=float)
        target = np.asarray([point.get("target_position") for point in trajectory], dtype=float)
        relative = target - source if len(source) else np.empty((0, 3))
        holding = np.asarray([bool(point.get("holding")) for point in trajectory], dtype=float)
        contact = np.asarray([bool(point.get("contact")) for point in trajectory], dtype=float)
        action_array = np.asarray(actions, dtype=float) if actions else np.empty((0, 0))

        figure, axes = plt.subplots(1, 3, figsize=(15, 3.8))
        if steps:
            axes[0].plot(steps, distances, marker="o", label="robot-object distance")
            axes[0].set_xlabel("frame step")
            axes[0].set_ylabel("distance")
            axes[0].grid(alpha=0.25)
            marker_height = max([float(value) for value in distances if value is not None] + [0.05])
            axes[0].step(steps, holding * marker_height, where="mid", label="holding", alpha=0.7)
            axes[0].step(steps, contact * marker_height * 0.8, where="mid", label="contact", alpha=0.7)
            axes[0].legend(fontsize=8)
            for axis, name in enumerate(("x", "y", "z")):
                axes[1].plot(steps, relative[:, axis], marker="o", label=f"relative {name}")
            follow_residual = np.linalg.norm((target - target[0]) - (source - source[0]), axis=1)
            axes[1].plot(steps, follow_residual, "k--", label="follow residual")
            axes[1].axhline(0.03, color="red", linestyle=":", label="0.03 threshold")
            axes[1].set_xlabel("frame step")
            axes[1].set_title("Pair-local trajectory")
            axes[1].grid(alpha=0.25)
            axes[1].legend(fontsize=8)
        else:
            axes[0].text(0.5, 0.5, "trajectory unavailable", ha="center")
            axes[1].text(0.5, 0.5, "trajectory unavailable", ha="center")
        if action_array.size:
            action_steps = np.arange(len(action_array)) + int(row["start_step"])
            axes[2].plot(action_steps, np.linalg.norm(action_array[:, :-1], axis=1), marker="o", label="arm action norm")
            axes[2].plot(action_steps, action_array[:, -1], marker="o", label="gripper action")
            axes[2].axhline(0, color="black", linewidth=0.7)
            axes[2].set_xlabel("action step")
            axes[2].set_title("Action window")
            axes[2].grid(alpha=0.25)
            axes[2].legend(fontsize=8)
        else:
            axes[2].text(0.5, 0.5, "action unavailable", ha="center")
        figure.tight_layout()
        display(figure)
        plt.close(figure)

    def render():
        row, reviewed_row = current_pair()
        with output:
            clear_output(wait=True)
            if row is None:
                display(HTML("<b>No items match the current filters.</b>"))
                return
            state["loading"] = True
            decision.value = str(reviewed_row.get("reviewer_decision", ""))
            error_type.value = str(reviewed_row.get("label_error_type", ""))
            notes.value = str(reviewed_row.get("reviewer_notes", ""))
            state["loading"] = False
            qa = row.get("trajectory_qa", {}) or {}
            title = (
                f"<h3>{html.escape(str(row['audit_id']))}: task {row['task_id']} / "
                f"{html.escape(str(row['event_type']))}</h3>"
                f"<p><b>episode</b> {html.escape(str(row['episode_id']))}; "
                f"<b>pair</b> {html.escape(str(row['edge']['source']))} → "
                f"{html.escape(str(row['edge']['target']))}; <b>steps</b> "
                f"{row['start_step']} → {row['target_step']}; "
                f"<b>trajectory QA</b> {qa.get('status')} "
                f"({qa.get('available_steps')}/{qa.get('requested_steps')})</p>"
            )
            display(HTML(title + evidence_table(row)))
            render_plot(row)
        summary = review_summary(review)
        progress.value = (
            f"<b>Filtered:</b> {state['position'] + 1 if state['indices'] else 0}/{len(state['indices'])} &nbsp; "
            f"<b>Total reviewed:</b> {summary['reviewed']}/{summary['total']} &nbsp; "
            f"pass {summary['passed']}, errors {summary['label_errors']}, ambiguous {summary['ambiguous']}"
        )

    def refilter(*_):
        state["indices"] = filtered_indices()
        state["position"] = 0
        render()

    def move(delta: int):
        if state["indices"]:
            state["position"] = max(0, min(len(state["indices"]) - 1, state["position"] + delta))
            render()

    def save(go_next: bool):
        row, reviewed_row = current_pair()
        if row is None:
            return
        verdict = str(decision.value)
        error = str(error_type.value)
        if verdict not in DECISIONS:
            message.value = "<span style='color:#b00020'>Choose pass, label error, or ambiguous.</span>"
            return
        if verdict in {"label_error", "ambiguous"} and error not in ERROR_TYPES:
            message.value = "<span style='color:#b00020'>Choose an error/evidence type.</span>"
            return
        if verdict == "pass":
            error = ""
        reviewed_row["reviewer_decision"] = verdict
        reviewed_row["label_error_type"] = error
        reviewed_row["reviewer_notes"] = str(notes.value).strip()
        summary = save_review_bundle(root, review)
        message.value = (
            f"<span style='color:#137333'>Saved {row['audit_id']} at "
            f"{summary['last_updated_utc']}.</span>"
        )
        if go_next and state["position"] < len(state["indices"]) - 1:
            state["position"] += 1
        render()

    previous_button.on_click(lambda _: move(-1))
    next_button.on_click(lambda _: move(1))
    save_button.on_click(lambda _: save(False))
    save_next_button.on_click(lambda _: save(True))
    for control in (task_filter, event_filter, status_filter):
        control.observe(refilter, names="value")

    controls = widgets.VBox(
        [
            widgets.HBox([task_filter, event_filter, status_filter]),
            widgets.HBox([previous_button, next_button, save_button, save_next_button]),
            progress,
            output,
            decision,
            error_type,
            notes,
            message,
        ]
    )
    display(controls)
    render()
    return controls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_root", type=Path)
    args = parser.parse_args()
    evidence, review = load_audit_bundle(args.audit_root)
    print(
        json.dumps(
            {
                "items": len(evidence),
                "review_summary": review_summary(review),
                "trajectory_qa_failures": sum(
                    row.get("trajectory_qa", {}).get("status") != "pass"
                    for row in evidence
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
