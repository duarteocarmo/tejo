"""Create a local DuckDB table from the fineweb2-bagaco HuggingFace dataset."""

import duckdb

DB_PATH = "fineweb2_bagaco.duckdb"
HF_DATASET = "hf://datasets/duarteocarmo/fineweb2-bagaco/shard_*.parquet"
TABLE_NAME = "fineweb2_bagaco"


def main():
    con = duckdb.connect(database=DB_PATH)
    con.install_extension("httpfs")
    con.load_extension("httpfs")

    con.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
    con.execute(f"CREATE TABLE {TABLE_NAME} AS SELECT * FROM '{HF_DATASET}'")

    row_count = con.execute(f"SELECT count(*) FROM {TABLE_NAME}").fetchone()[0]
    print(f"Created table '{TABLE_NAME}' in {DB_PATH} with {row_count:,} rows")

    con.close()


if __name__ == "__main__":
    main()
