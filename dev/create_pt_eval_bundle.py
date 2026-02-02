"""
Create the PT-CORE evaluation bundle.

Downloads Portuguese evaluation datasets from HuggingFace and converts them
to the JSONL format expected by nanochat's core_eval.py (multiple_choice type).

Output structure:
    pt_eval_bundle/
    ├── pt_core.yaml          # task configs (same schema as core.yaml)
    ├── eval_meta_data.csv     # random baselines per task
    └── eval_data/
        ├── sst2_pt.jsonl
        ├── goldenswag_pt.jsonl
        ├── mmlu_pt.jsonl
        └── scala_pt.jsonl

Usage:
    python -m dev.create_pt_eval_bundle
"""

import json
import os
import re
import csv

import yaml
from datasets import load_dataset

OUTPUT_DIR = "pt_eval_bundle"

TASKS = [
    {
        "label": "sst2_pt",
        "hf_dataset": "duarteocarmo/sst2-pt-mini",
        "icl_task_type": "multiple_choice",
        "num_fewshot": [12],
        "continuation_delimiter": "\nSentimento: ",
        "random_baseline": 50.0,
        "split": "test",
    },
    {
        "label": "goldenswag_pt",
        "hf_dataset": "duarteocarmo/goldenswag-pt-mini",
        "icl_task_type": "multiple_choice",
        "num_fewshot": [5],
        "continuation_delimiter": "\n",
        "random_baseline": 25.0,
        "split": "test",
    },
    {
        "label": "mmlu_pt",
        "hf_dataset": "duarteocarmo/mmlu-pt-mini",
        "icl_task_type": "multiple_choice",
        "num_fewshot": [5],
        "continuation_delimiter": "\nResposta: ",
        "random_baseline": 25.0,
        "split": "test",
    },
    {
        "label": "scala_pt",
        "hf_dataset": "duarteocarmo/scala-pt",
        "icl_task_type": "multiple_choice",
        "num_fewshot": [12],
        "continuation_delimiter": "\nGramaticalmente correcto: ",
        "random_baseline": 50.0,
        "split": "test",
    },
]


# ---------------------------------------------------------------------------
# Converters: each transforms a HF dataset row into the CORE MC format
# {"query": str, "choices": list[str], "gold": int}
# ---------------------------------------------------------------------------


def convert_sst2_pt(row):
    label_map = {"positive": 0, "negative": 1}
    return {
        "query": row["text"],
        "choices": ["positivo", "negativo"],
        "gold": label_map[row["label"]],
    }


def convert_scala_pt(row):
    label_map = {"correct": 0, "incorrect": 1}
    return {
        "query": row["text"],
        "choices": ["sim", "não"],
        "gold": label_map[row["label"]],
    }


def parse_options_from_text(text):
    """Split EuroEval-style text into (query, [option_texts]).

    Expected format:
        Some passage or question text
        Opções:
        a. First option
        b. Second option
        c. Third option
        d. Fourth option
    """
    parts = text.split("\nOpções:\n")
    if len(parts) != 2:
        raise ValueError(
            f"Expected exactly one '\\nOpções:\\n' separator, got {len(parts) - 1}"
        )
    query = parts[0].strip()
    options_text = parts[1].strip()
    matches = re.findall(r"[a-d]\.\s*(.*?)(?=\n[a-d]\.\s|$)", options_text, re.DOTALL)
    choices = [m.strip() for m in matches]
    if len(choices) != 4:
        raise ValueError(f"Expected 4 options, parsed {len(choices)}")
    return query, choices


def convert_goldenswag_pt(row):
    """GoldenSwag: passage + 4 natural continuations.

    We split the passage from the options and use full continuation texts as choices.
    The model evaluates the loss of each continuation after the passage.
    """
    label_map = {"a": 0, "b": 1, "c": 2, "d": 3}
    query, choices = parse_options_from_text(row["text"])
    return {
        "query": query,
        "choices": choices,
        "gold": label_map[row["label"]],
    }


def convert_mmlu_pt(row):
    """MMLU: question + 4 options, model picks a letter.

    We keep the full text (question + options) as query and use letter labels
    as choices, since the question often references the options ("Qual delas...").
    """
    label_map = {"a": 0, "b": 1, "c": 2, "d": 3}
    return {
        "query": row["text"],
        "choices": ["a", "b", "c", "d"],
        "gold": label_map[row["label"]],
    }


CONVERTERS = {
    "sst2_pt": convert_sst2_pt,
    "goldenswag_pt": convert_goldenswag_pt,
    "mmlu_pt": convert_mmlu_pt,
    "scala_pt": convert_scala_pt,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    eval_data_dir = os.path.join(OUTPUT_DIR, "eval_data")
    os.makedirs(eval_data_dir, exist_ok=True)

    for task in TASKS:
        label = task["label"]
        converter = CONVERTERS[label]
        print(
            f"Processing {label} from {task['hf_dataset']} (split={task['split']})..."
        )

        dataset = load_dataset(task["hf_dataset"], split=task["split"])

        output_path = os.path.join(eval_data_dir, f"{label}.jsonl")
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for row in dataset:
                converted = converter(dict(row))
                f.write(json.dumps(converted, ensure_ascii=False) + "\n")
                count += 1
        print(f"  -> {count} examples written to {output_path}")

    # Write pt_core.yaml
    config = {
        "icl_tasks": [
            {
                "label": t["label"],
                "icl_task_type": t["icl_task_type"],
                "dataset_uri": f"{t['label']}.jsonl",
                "num_fewshot": t["num_fewshot"],
                "continuation_delimiter": t["continuation_delimiter"],
            }
            for t in TASKS
        ]
    }
    config_path = os.path.join(OUTPUT_DIR, "pt_core.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"Wrote config to {config_path}")

    # Write eval_meta_data.csv
    meta_path = os.path.join(OUTPUT_DIR, "eval_meta_data.csv")
    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Eval Task", "Random baseline"])
        for t in TASKS:
            writer.writerow([t["label"], t["random_baseline"]])
    print(f"Wrote baselines to {meta_path}")

    print(f"\nBundle ready at ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
