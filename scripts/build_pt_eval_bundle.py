"""
Build a Portuguese evaluation bundle from EuroEval datasets.

Maps EuroEval Portuguese datasets to the CORE eval format used by nanochat.
Downloads datasets from HuggingFace, converts them to the CORE JSONL format,
and creates the config files needed by evaluate_model().

Usage:
    python -m scripts.build_pt_eval_bundle

The output is placed in {base_dir}/eval_bundle_pt/ with the same structure
as the English eval_bundle/:
    eval_bundle_pt/
        core.yaml           # task config
        eval_meta_data.csv  # random baselines
        eval_data/          # JSONL task files

Mapping (CORE English -> EuroEval Portuguese):
    mmlu_fewshot         -> mmlu_pt          (multiple_choice, 4 options)
    hellaswag            -> goldenswag_pt    (multiple_choice, 4 options)
    hellaswag_zeroshot   -> goldenswag_pt_zeroshot (multiple_choice, 4 options)
    boolq                -> boolq_pt         (multiple_choice, 2 options)
    winogrande           -> winogrande_pt    (multiple_choice, 2 options)
    (new) linguistic acc -> scala_pt         (multiple_choice, 2 options)
    (new) sentiment      -> sst2_pt          (multiple_choice, 2 options)
"""

import os
import re
import csv
import json
import random

import yaml

from nanochat.common import print0, get_base_dir

# ---------------------------------------------------------------------------
# EuroEval HF dataset IDs and their properties

PT_DATASETS = {
    "mmlu_pt": {
        "hf_id": "EuroEval/mmlu-pt-mini",
        "task_type": "multiple_choice",
        "num_options": 4,
        "random_baseline": 25.0,
        "num_fewshot": 5,
        "continuation_delimiter": " ",
        "description": "MMLU translated to Portuguese (maps to mmlu_fewshot)",
    },
    "goldenswag_pt": {
        "hf_id": "EuroEval/goldenswag-pt-mini",
        "task_type": "multiple_choice",
        "num_options": 4,
        "random_baseline": 25.0,
        "num_fewshot": 10,
        "continuation_delimiter": " ",
        "description": "HellaSwag adapted for Portuguese (maps to hellaswag)",
    },
    "goldenswag_pt_zeroshot": {
        "hf_id": "EuroEval/goldenswag-pt-mini",
        "task_type": "multiple_choice",
        "num_options": 4,
        "random_baseline": 25.0,
        "num_fewshot": 0,
        "continuation_delimiter": " ",
        "description": "HellaSwag adapted for Portuguese, 0-shot (maps to hellaswag_zeroshot)",
    },
    "boolq_pt": {
        "hf_id": "EuroEval/boolq-pt",
        "task_type": "multiple_choice",
        "num_options": 2,
        "random_baseline": 50.0,
        "num_fewshot": 10,
        "continuation_delimiter": " ",
        "description": "BoolQ translated to Portuguese (maps to boolq)",
    },
    "winogrande_pt": {
        "hf_id": "EuroEval/winogrande-pt",
        "task_type": "multiple_choice",
        "num_options": 2,
        "random_baseline": 50.0,
        "num_fewshot": 0,
        "continuation_delimiter": " ",
        "description": "Winogrande translated to Portuguese (maps to winogrande)",
    },
    "scala_pt": {
        "hf_id": "EuroEval/scala-pt",
        "task_type": "multiple_choice",
        "num_options": 2,
        "random_baseline": 50.0,
        "num_fewshot": 10,
        "continuation_delimiter": " ",
        "description": "Linguistic acceptability for Portuguese (new task, no English CORE equivalent)",
    },
    "sst2_pt": {
        "hf_id": "EuroEval/sst2-pt-mini",
        "task_type": "multiple_choice",
        "num_options": 2,
        "random_baseline": 50.0,
        "num_fewshot": 10,
        "continuation_delimiter": " ",
        "description": "Sentiment classification for Portuguese (new task, no English CORE equivalent)",
    },
}

# ---------------------------------------------------------------------------
# Parsing: extract query + choices from EuroEval pre-formatted text

# Patterns for the choices/options separator line in multiple languages
CHOICES_SEPARATORS = re.compile(
    r"\n(Choices|Opções|Options)\s*:\s*\n", re.IGNORECASE
)
# Pattern for individual option lines: "a. text", "b. text", etc.
OPTION_PATTERN = re.compile(r"^([a-z])\.\s+(.+)$", re.MULTILINE)

LABEL_TO_INDEX = {chr(ord("a") + i): i for i in range(26)}


def parse_euroeval_mc(text: str, label: str, num_options: int):
    """
    Parse a EuroEval pre-formatted multiple-choice text into CORE format.

    EuroEval format:
        <query text>
        Choices:          (or Opções:)
        a. <option 1>
        b. <option 2>
        ...

    Returns dict with keys: query, choices (list[str]), gold (int index)
    or None if parsing fails.
    """
    # Split on the choices separator
    parts = CHOICES_SEPARATORS.split(text)
    if len(parts) < 3:
        # fallback: try splitting on last occurrence of "a. " preceded by newline
        # This handles cases where the separator keyword is absent
        idx = text.rfind("\na. ")
        if idx == -1:
            return None
        query = text[:idx].strip()
        options_text = text[idx:]
    else:
        query = parts[0].strip()
        options_text = parts[-1]  # everything after the separator

    # Extract options
    options = OPTION_PATTERN.findall(options_text)
    if len(options) < num_options:
        return None

    # Take exactly num_options options
    choices = [opt_text.strip() for _, opt_text in options[:num_options]]
    gold = LABEL_TO_INDEX.get(label.strip().lower())
    if gold is None or gold >= num_options:
        return None

    return {"query": query, "choices": choices, "gold": gold}


# ---------------------------------------------------------------------------
# Download + convert

def download_and_convert(task_name: str, task_config: dict, output_dir: str):
    """
    Download a EuroEval dataset from HuggingFace and convert it to CORE JSONL.
    Returns the number of examples written, or 0 on failure.
    """
    from datasets import load_dataset

    hf_id = task_config["hf_id"]
    num_options = task_config["num_options"]
    print0(f"  Downloading {hf_id} ...")

    try:
        ds = load_dataset(hf_id, trust_remote_code=True)
    except Exception as e:
        print0(f"  WARNING: Could not download {hf_id}: {e}")
        return 0

    # Merge all splits (train + val + test) into one pool, like the English bundle
    all_examples = []
    for split_name in ds:
        split = ds[split_name]
        for row in split:
            text = row.get("text", "")
            label = row.get("label", "")
            parsed = parse_euroeval_mc(text, label, num_options)
            if parsed is not None:
                all_examples.append(parsed)

    if not all_examples:
        print0(f"  WARNING: No valid examples parsed from {hf_id}")
        return 0

    # Shuffle deterministically for reproducibility
    rng = random.Random(1337)
    rng.shuffle(all_examples)

    # Write to JSONL
    output_path = os.path.join(output_dir, f"{task_name}.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print0(f"  Wrote {len(all_examples)} examples to {output_path}")
    return len(all_examples)


def build_bundle(output_base_dir: str):
    """Build the complete Portuguese eval bundle."""
    eval_data_dir = os.path.join(output_base_dir, "eval_data")
    os.makedirs(eval_data_dir, exist_ok=True)

    # Download and convert each dataset
    successful_tasks = {}
    for task_name, task_config in PT_DATASETS.items():
        n = download_and_convert(task_name, task_config, eval_data_dir)
        if n > 0:
            successful_tasks[task_name] = task_config

    if not successful_tasks:
        print0("ERROR: No datasets were successfully downloaded. Check HuggingFace access.")
        return False

    # Write core.yaml
    icl_tasks = []
    for task_name, cfg in successful_tasks.items():
        icl_tasks.append({
            "label": task_name,
            "icl_task_type": cfg["task_type"],
            "dataset_uri": f"{task_name}.jsonl",
            "num_fewshot": [cfg["num_fewshot"]],
            "continuation_delimiter": cfg["continuation_delimiter"],
        })

    config = {"icl_tasks": icl_tasks}
    config_path = os.path.join(output_base_dir, "core.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print0(f"Wrote config to {config_path}")

    # Write eval_meta_data.csv (random baselines)
    meta_path = os.path.join(output_base_dir, "eval_meta_data.csv")
    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Eval Task", "Random baseline"])
        for task_name, cfg in successful_tasks.items():
            writer.writerow([task_name, cfg["random_baseline"]])
    print0(f"Wrote metadata to {meta_path}")

    print0(f"\nPortuguese eval bundle built with {len(successful_tasks)} tasks:")
    for task_name, cfg in successful_tasks.items():
        print0(f"  {task_name}: {cfg['description']}")

    return True


# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build Portuguese eval bundle from EuroEval datasets")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: {base_dir}/eval_bundle_pt)")
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        base_dir = get_base_dir()
        output_dir = os.path.join(base_dir, "eval_bundle_pt")

    print0(f"Building Portuguese eval bundle in: {output_dir}")
    success = build_bundle(output_dir)
    if success:
        print0("\nDone! Use --core-metric-lang=pt in base_train.py or --lang=pt in base_eval.py")
    else:
        print0("\nFailed to build Portuguese eval bundle.")


if __name__ == "__main__":
    main()
