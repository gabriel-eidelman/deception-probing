#!/usr/bin/env python3
"""Convert DATASET entries to SCENARIOS format (field remap only, no content edits)."""


def convert(entry: dict) -> dict:
    return {
        "system": entry["system"],
        "user": entry["message"],
        "description": f"{entry['task_domain']} / {entry['elicitation_strategy']} (label={entry['label']})",
        "task_domain": entry["task_domain"],
        "elicitation_strategy": entry["elicitation_strategy"],
        "label": entry["label"],
    }


def to_scenarios(dataset: list) -> list:
    return [convert(e) for e in dataset]


def dump_scenarios(scenarios: list) -> str:
    """Pretty-print as a Python literal in the SCENARIOS style."""
    lines = ["SCENARIOS = ["]
    last_key = None
    for s in scenarios:
        key = (s["task_domain"], s["elicitation_strategy"])
        if key != last_key:
            lines.append(f"    # ── {s['task_domain']} × {s['elicitation_strategy']} ──")
            last_key = key
        lines.append("    {")
        for field in ("system", "user", "description",
                      "task_domain", "elicitation_strategy"):
            val = s[field].replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'        "{field}": "{val}",')
        lines.append(f'        "label": {s["label"]},')
        lines.append("    },")
    lines.append("]")
    return "\n".join(lines)


if __name__ == "__main__":
    from pilot_v2 import DATASET  # or paste DATASET inline

    scenarios = to_scenarios(DATASET)
    with open("scenarios.py", "w", encoding="utf-8") as f:
        f.write(dump_scenarios(scenarios))
        f.write("\n")