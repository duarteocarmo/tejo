"""
Repackage the FinewebEdu-100B dataset into shards:
"""

import os
import time
from types import SimpleNamespace
import shutil
import polars


import pyarrow.parquet as pq
import pyarrow as pa


config = SimpleNamespace(
    seed=42,
    chars_per_shard=250_000_000,
    row_group_size=1024,
    output_dir="./base_data",
    token_target=15_000_000_000,
    chars_per_token=3,
    batch_size=50_000,
)


def build_dataset():
    total_chars_target = config.token_target * config.chars_per_token

    df = (
        polars.scan_parquet(
            "hf://datasets/HuggingFaceFW/fineweb-2/data/por_Latn/train/*.parquet"
        )
        .filter(polars.col("url").str.contains(".pt", literal=True))
        .select(["text", "url"])
    )

    if os.path.exists(config.output_dir):
        shutil.rmtree(config.output_dir)
        print(f"Removed existing directory {config.output_dir}")

    os.makedirs(config.output_dir, exist_ok=True)
    print(f"Created output directory at {config.output_dir}")

    shard_docs = []
    shard_index = 0
    shard_characters = 0
    total_time_spent = 0
    total_chars_collected = 0
    t0 = time.time()

    for batch_df in df.collect_batches(chunk_size=config.batch_size):
        if total_chars_collected >= total_chars_target:
            print("Reached target number of characters, stopping.")
            break

        for doc in batch_df.iter_rows(named=True):
            text = doc["text"]
            shard_docs.append(text)
            shard_characters += len(text)
            total_chars_collected += len(text)
            collected_enough_chars = shard_characters >= config.chars_per_shard
            docs_multiple_of_row_group_size = (
                len(shard_docs) % config.row_group_size == 0
            )
            if not collected_enough_chars or not docs_multiple_of_row_group_size:
                continue

            shard_path = os.path.join(
                config.output_dir, f"shard_{shard_index:05d}.parquet"
            )
            shard_table = pa.Table.from_pydict({"text": shard_docs})
            pq.write_table(
                shard_table,
                shard_path,
                row_group_size=config.row_group_size,
                use_dictionary=False,  # this is usually used for categorical data
                compression="zstd",  # Valid values: {‘NONE’, ‘SNAPPY’, ‘GZIP’, ‘BROTLI’, ‘LZ4’, ‘ZSTD’}
                compression_level=3,
                write_statistics=False,  # not needed for text
            )
            t1 = time.time()
            dt = t1 - t0  # for this shard alone
            t0 = t1
            total_time_spent += dt
            perc_done = (total_chars_collected / total_chars_target) * 100
            estimated_total_time = (
                total_time_spent / total_chars_collected
            ) * total_chars_target
            estimated_time_remaining = estimated_total_time - total_time_spent
            estimated_time_remaining_hours = estimated_time_remaining / 3600
            print(
                f"Wrote {shard_path}. #documents: {len(shard_docs)} | "
                f"#characters: {shard_characters} | time: {dt:.2f}s | "
                f"total time: {total_time_spent:.2f}s | {perc_done:.2f}% done | "
                f"total chars: {total_chars_collected} | "
                f"estimated time remaining: {estimated_time_remaining_hours:.2f} hours"
            )
            shard_docs = []
            shard_characters = 0
            shard_index += 1


# # Demonstration of how the data was later uploaded to HuggingFace
# def upload():
#     import os
#     from huggingface_hub import HfApi
#
#     token = os.getenv("HF_TOKEN")
#     api = HfApi(token=token)
#     api.upload_large_folder(
#         folder_path=output_dir,
#         repo_id="karpathy/fineweb-edu-100b-shuffle",
#         repo_type="dataset",
#     )
#
#
# # upload()
if __name__ == "__main__":
    build_dataset()
