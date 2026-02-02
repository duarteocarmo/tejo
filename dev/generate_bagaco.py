"""
Extract Portuguese (.pt) subset from FineWeb-2 using HuggingFace datasets streaming.
Keeps all columns: text, id, dump, url, date, file_path, language, language_score,
language_script, minhash_cluster_size, top_langs.
"""

import os
import time
import shutil
from types import SimpleNamespace

from datasets import load_dataset
from huggingface_hub import HfApi
import pyarrow.parquet as pq
import pyarrow as pa

LIMIT = 100  # set to None to scan everything

config = SimpleNamespace(
    docs_per_shard=100_000,
    row_group_size=1024,
    output_dir="./bagaco_data",
    repo_id="duarteocarmo/fineweb2-bagaco",
)


def build_dataset():
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-2",
        name="por_Latn",
        split="train",
        streaming=True,
    )
    dataset = dataset.filter(lambda doc: ".pt/" in doc["url"])

    if os.path.exists(config.output_dir):
        shutil.rmtree(config.output_dir)
        print(f"Removed existing directory {config.output_dir}")

    os.makedirs(config.output_dir, exist_ok=True)
    print(f"Created output directory at {config.output_dir}")

    shard_docs = []
    shard_index = 0
    total_docs = 0
    total_time_spent = 0
    t0 = time.time()

    for doc in dataset:
        if LIMIT is not None and total_docs >= LIMIT:
            print(f"Reached limit of {LIMIT} documents, stopping.")
            break

        shard_docs.append(doc)
        total_docs += 1

        shard_size = len(shard_docs)
        docs_multiple_of_row_group_size = shard_size % config.row_group_size == 0

        if shard_size < config.docs_per_shard or not docs_multiple_of_row_group_size:
            continue

        shard_path = os.path.join(config.output_dir, f"shard_{shard_index:05d}.parquet")
        shard_table = pa.Table.from_pylist(shard_docs)
        pq.write_table(
            shard_table,
            shard_path,
            row_group_size=config.row_group_size,
            use_dictionary=False,
            compression="zstd",
            compression_level=3,
            write_statistics=False,
        )

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        total_time_spent += dt

        print(
            f"Wrote {shard_path} | #docs in shard: {shard_size} | "
            f"total docs: {total_docs} | time: {dt:.2f}s | "
            f"total time: {total_time_spent:.2f}s"
        )

        shard_docs = []
        shard_index += 1

    # flush remaining rows
    remaining = len(shard_docs)
    if remaining > 0:
        shard_path = os.path.join(config.output_dir, f"shard_{shard_index:05d}.parquet")
        shard_table = pa.Table.from_pylist(shard_docs)
        pq.write_table(
            shard_table,
            shard_path,
            row_group_size=config.row_group_size,
            use_dictionary=False,
            compression="zstd",
            compression_level=3,
            write_statistics=False,
        )
        print(
            f"Wrote final {shard_path} | #docs: {remaining} | total docs: {total_docs}"
        )

    print(f"Done. {total_docs} documents across {shard_index + 1} shards.")


README = """\
---
language:
  - por
  - pt
license: odc-by
task_categories:
  - text-generation
source_datasets:
  - HuggingFaceFW/fineweb-2
tags:
  - portuguese
  - web-corpus
dataset_info:
  features:
    - name: text
      dtype: string
    - name: id
      dtype: string
    - name: dump
      dtype: string
    - name: url
      dtype: string
    - name: date
      dtype: string
    - name: file_path
      dtype: string
    - name: language
      dtype: string
    - name: language_score
      dtype: float64
    - name: language_script
      dtype: string
    - name: minhash_cluster_size
      dtype: int64
    - name: top_langs
      dtype: string
    - name: wordlist_ratio
      dtype: float64
---

# Bagaço 🍷🇵🇹

[Fineweb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) filtered by Portuguese URLs (i.e., containing `.pt/`) and Portuguese language.


## Filtering

- **Source**: `HuggingFaceFW/fineweb-2`, subset `por_Latn`, split `train`
- **Filter**: URLs containing `.pt/` (Portuguese top-level domain)
"""


def upload():
    token = os.getenv("HF_TOKEN")
    assert token is not None, "HF_TOKEN environment variable not set."
    api = HfApi(token=token)

    readme_path = os.path.join(config.output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(README)
    print(f"Wrote {readme_path}")

    api.upload_large_folder(
        folder_path=config.output_dir,
        repo_id=config.repo_id,
        repo_type="dataset",
    )


if __name__ == "__main__":
    import os as _os

    build_dataset()
    # upload()
    _os._exit(0)  # avoid PyGILState_Release crash from datasets background threads
