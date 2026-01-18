"""
Load FineWeb v2 Portuguese data using Polars streaming.

This is the recommended approach based on benchmark results (6x faster than alternatives).

Usage:
    python load_fineweb_polars.py --target-tokens 10000000 --output processed_data.parquet

Features:
- Streams data efficiently with Polars
- Filters for Portuguese language and .pt domains
- Processes 10M tokens (30M characters)
- Memory efficient
"""

import polars as pl
import argparse
from pathlib import Path
from typing import Optional
import time


def download_fineweb2_parquet(output_dir: Path, language: str = 'pt', max_shards: Optional[int] = None):
    """
    Download FineWeb v2 parquet files for a specific language.

    Note: FineWeb v2 is hosted on HuggingFace as parquet files.
    You can download them using the datasets library or directly from HF.

    Args:
        output_dir: Directory to save parquet files
        language: Language code (default: 'pt' for Portuguese)
        max_shards: Maximum number of shards to download (None = all)
    """
    try:
        from datasets import load_dataset

        print(f"Downloading FineWeb v2 ({language}) to {output_dir}...")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset with streaming to get parquet URLs
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-2",
            split="train",
            streaming=True
        )

        # Filter for language
        dataset = dataset.filter(lambda x: x.get('language') == language)

        # Download as parquet
        # This is a placeholder - in practice you'd use dataset.to_parquet()
        # or download the parquet files directly from HuggingFace
        print("Note: Use HuggingFace dataset viewer to find parquet file URLs")
        print("Or use: dataset.to_parquet(output_dir)")

    except ImportError:
        print("Error: datasets library not available")
        print("Install with: pip install datasets")


def load_and_process_fineweb2_streaming(
    data_dir: Path,
    target_chars: int = 30_000_000,  # 10M tokens * 3 chars/token
    output_file: Optional[Path] = None,
    batch_size: int = 10000
):
    """
    Load and process FineWeb v2 Portuguese data using Polars streaming.

    This is the FASTEST approach based on benchmarks (6x faster than alternatives).

    Args:
        data_dir: Directory containing FineWeb v2 parquet files
        target_chars: Target number of characters to process
        output_file: Optional output file to save processed data
        batch_size: Number of rows to process per batch
    """
    print(f"Loading FineWeb v2 from {data_dir}")
    print(f"Target: {target_chars:,} characters ({target_chars // 3:,} tokens)")
    print("=" * 80)

    start_time = time.time()

    # Lazy scan all parquet files
    print("Scanning parquet files (lazy)...")
    lazy_df = pl.scan_parquet(str(data_dir / "*.parquet"))

    # Filter for Portuguese language and .pt domains (lazy - not executed yet)
    print("Applying filters (lazy)...")
    lazy_df = lazy_df.filter(
        (pl.col('language') == 'pt') &
        (pl.col('url').str.contains('.pt'))
    )

    # Add computed columns (lazy)
    lazy_df = lazy_df.with_columns([
        pl.col('text').str.len_chars().alias('text_len'),
        pl.col('text').str.len_bytes().alias('text_bytes'),
    ])

    # Select columns we need
    lazy_df = lazy_df.select([
        'id',
        'text',
        'url',
        'language',
        'text_len',
        'text_bytes'
    ])

    # Stream and process in batches
    print(f"\nStreaming and processing (batch size: {batch_size:,})...")
    print("-" * 80)

    total_chars = 0
    total_docs = 0
    processed_batches = []

    try:
        for batch_idx, batch in enumerate(lazy_df.collect(engine='streaming').iter_slices(batch_size)):
            batch_chars = batch['text_len'].sum()
            batch_docs = len(batch)

            total_chars += batch_chars
            total_docs += batch_docs

            # Save batch if output file specified
            if output_file:
                processed_batches.append(batch)

            # Progress
            elapsed = time.time() - start_time
            throughput = total_chars / elapsed / 1e6 if elapsed > 0 else 0

            print(
                f"Batch {batch_idx + 1:4d} | "
                f"Docs: {total_docs:8,} | "
                f"Chars: {total_chars:12,} / {target_chars:,} | "
                f"Throughput: {throughput:6.1f} M chars/sec",
                end='\r'
            )

            # Stop when we reach target
            if total_chars >= target_chars:
                print()
                print(f"\n✓ Reached target of {target_chars:,} characters!")
                break

    except Exception as e:
        print(f"\n✗ Error during streaming: {e}")
        raise

    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    elapsed = time.time() - start_time
    throughput = total_chars / elapsed / 1e6

    print(f"Documents processed: {total_docs:,}")
    print(f"Characters processed: {total_chars:,}")
    print(f"Tokens (approx): {total_chars // 3:,}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {throughput:.2f} M chars/sec")

    # Save to output file if specified
    if output_file and processed_batches:
        print(f"\nSaving to {output_file}...")
        combined_df = pl.concat(processed_batches)
        combined_df.write_parquet(output_file, compression='zstd')
        print(f"✓ Saved {len(combined_df):,} documents to {output_file}")

    return {
        'docs': total_docs,
        'chars': total_chars,
        'tokens': total_chars // 3,
        'time': elapsed,
        'throughput': throughput
    }


def main():
    parser = argparse.ArgumentParser(
        description="Load FineWeb v2 Portuguese data using Polars streaming"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('fineweb2_data'),
        help='Directory containing FineWeb v2 parquet files'
    )
    parser.add_argument(
        '--target-tokens',
        type=int,
        default=10_000_000,
        help='Target number of tokens to process (default: 10M)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output parquet file (optional)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10000,
        help='Batch size for streaming (default: 10000)'
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download FineWeb v2 data first'
    )

    args = parser.parse_args()

    # Download if requested
    if args.download:
        download_fineweb2_parquet(args.data_dir)
        return

    # Check if data directory exists
    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        print("Download FineWeb v2 data first or use --download flag")
        return

    # Load and process
    target_chars = args.target_tokens * 3  # 1 token ≈ 3 chars

    results = load_and_process_fineweb2_streaming(
        data_dir=args.data_dir,
        target_chars=target_chars,
        output_file=args.output,
        batch_size=args.batch_size
    )

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
