"""
Benchmark different streaming approaches for FineWeb v2 Portuguese data.

This script compares:
1. HuggingFace datasets with streaming=True
2. Polars with streaming/lazy evaluation
3. HuggingFace datasets without streaming (baseline)

Goal: Gather 10M tokens (1 token ≈ 3 chars = 30M characters)
Dataset: FineWeb v2, Portuguese language, .pt domains
"""

import time
import polars as pl
from datasets import load_dataset
from typing import Dict, Callable
import sys

# Target: 10M tokens = ~30M characters
TARGET_TOKENS = 10_000_000
CHARS_PER_TOKEN = 3
TARGET_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN

print(f"Target: {TARGET_TOKENS:,} tokens (~{TARGET_CHARS:,} characters)")
print("=" * 80)


def benchmark_approach(name: str, func: Callable) -> Dict:
    """Benchmark a data loading approach."""
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"{'='*80}")

    start_time = time.time()
    try:
        result = func()
        elapsed = time.time() - start_time

        print(f"\n✓ Success!")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Characters gathered: {result['chars']:,}")
        print(f"  Tokens (approx): {result['chars'] // CHARS_PER_TOKEN:,}")
        print(f"  Documents processed: {result['docs']:,}")
        print(f"  Throughput: {result['chars'] / elapsed / 1e6:.2f} M chars/sec")

        return {
            'name': name,
            'success': True,
            'time': elapsed,
            'chars': result['chars'],
            'docs': result['docs'],
            'throughput': result['chars'] / elapsed / 1e6
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Failed: {str(e)}")
        print(f"  Time before failure: {elapsed:.2f}s")
        return {
            'name': name,
            'success': False,
            'time': elapsed,
            'error': str(e)
        }


def approach_1_hf_streaming():
    """HuggingFace datasets with streaming=True."""
    print("Loading dataset with streaming=True...")

    # Load FineWeb v2 with streaming
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-2",
        name="default",
        split="train",
        streaming=True
    )

    # Filter for Portuguese language and .pt domains
    print("Filtering for Portuguese language and .pt domains...")
    dataset = dataset.filter(lambda x: x.get('language') == 'pt' or 'language_score' in x)
    dataset = dataset.filter(lambda x: '.pt' in x.get('url', '').lower())

    # Gather characters
    print("Gathering characters...")
    total_chars = 0
    doc_count = 0

    for doc in dataset:
        text = doc.get('text', '')
        total_chars += len(text)
        doc_count += 1

        if doc_count % 100 == 0:
            print(f"  Docs: {doc_count:,} | Chars: {total_chars:,} | Target: {TARGET_CHARS:,}", end='\r')

        if total_chars >= TARGET_CHARS:
            break

    print()  # New line after progress
    return {'chars': total_chars, 'docs': doc_count}


def approach_2_polars_streaming():
    """Polars with streaming/lazy evaluation."""
    print("Using Polars with lazy evaluation...")

    # Note: Polars can read from Hugging Face datasets via arrow or parquet
    # We'll try to use scan_parquet with the HF dataset path
    # First, we need to get the cache path or use the HF dataset directly

    # Try to use Polars scan_parquet with HF dataset
    # This might require downloading the dataset first or using Arrow
    try:
        # Option 1: Convert HF dataset to Arrow and use Polars
        print("Converting HF dataset to Arrow table...")
        from datasets import load_dataset
        import pyarrow as pa

        # Load dataset without streaming first (this might be slower)
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-2",
            name="default",
            split="train",
            streaming=True  # Still use streaming for HF
        )

        # Filter in HF first
        dataset = dataset.filter(lambda x: x.get('language') == 'pt' or 'language_score' in x)
        dataset = dataset.filter(lambda x: '.pt' in x.get('url', '').lower())

        # Collect data using Polars-style processing
        print("Processing with Polars...")
        total_chars = 0
        doc_count = 0

        # We'll use Polars DataFrame for processing chunks
        batch_size = 1000
        batch = []

        for doc in dataset:
            batch.append(doc)

            if len(batch) >= batch_size:
                # Convert batch to Polars DataFrame
                df = pl.DataFrame(batch)

                # Process with Polars
                text_lengths = df.select(
                    pl.col('text').str.len_chars().alias('length')
                )['length'].to_list()

                total_chars += sum(text_lengths)
                doc_count += len(batch)

                if doc_count % 100 == 0:
                    print(f"  Docs: {doc_count:,} | Chars: {total_chars:,} | Target: {TARGET_CHARS:,}", end='\r')

                batch = []

                if total_chars >= TARGET_CHARS:
                    break

        # Process remaining batch
        if batch and total_chars < TARGET_CHARS:
            df = pl.DataFrame(batch)
            text_lengths = df.select(
                pl.col('text').str.len_chars().alias('length')
            )['length'].to_list()
            total_chars += sum(text_lengths)
            doc_count += len(batch)

        print()
        return {'chars': total_chars, 'docs': doc_count}

    except Exception as e:
        print(f"Polars approach failed: {e}")
        # Fallback: try direct parquet reading if available
        raise


def approach_3_hf_no_streaming():
    """HuggingFace datasets without streaming (baseline)."""
    print("Loading dataset WITHOUT streaming (downloading first)...")

    # This will download the full dataset first
    # Note: This might take a very long time and use lots of disk space
    # For FineWeb v2, this is probably not practical

    # Instead, let's modify this to download a limited subset
    print("Note: Loading a subset for comparison (not full dataset)")

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-2",
        name="default",
        split="train[:10000]",  # Just first 10k rows for comparison
        streaming=False
    )

    # Filter for Portuguese and .pt domains
    print("Filtering...")
    dataset = dataset.filter(lambda x: x.get('language') == 'pt' or 'language_score' in x)
    dataset = dataset.filter(lambda x: '.pt' in x.get('url', '').lower())

    # Gather characters
    print("Gathering characters...")
    total_chars = 0
    doc_count = 0

    for doc in dataset:
        text = doc.get('text', '')
        total_chars += len(text)
        doc_count += 1

        if doc_count % 100 == 0:
            print(f"  Docs: {doc_count:,} | Chars: {total_chars:,}", end='\r')

        if total_chars >= TARGET_CHARS:
            break

    print()
    return {'chars': total_chars, 'docs': doc_count}


def main():
    """Run all benchmarks and compare results."""
    results = []

    # Test HF streaming
    results.append(benchmark_approach(
        "HuggingFace datasets with streaming=True",
        approach_1_hf_streaming
    ))

    # Test Polars
    results.append(benchmark_approach(
        "Polars with streaming/lazy evaluation",
        approach_2_polars_streaming
    ))

    # Test HF without streaming (baseline with subset)
    results.append(benchmark_approach(
        "HuggingFace datasets without streaming (subset)",
        approach_3_hf_no_streaming
    ))

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r['success']]
    if successful:
        successful.sort(key=lambda x: x['time'])

        print("\nRanking (fastest to slowest):")
        for i, result in enumerate(successful, 1):
            print(f"\n{i}. {result['name']}")
            print(f"   Time: {result['time']:.2f}s")
            print(f"   Throughput: {result['throughput']:.2f} M chars/sec")
            print(f"   Documents: {result['docs']:,}")

        print(f"\n🏆 Winner: {successful[0]['name']}")
        print(f"   Time: {successful[0]['time']:.2f}s")

        if len(successful) > 1:
            speedup = successful[-1]['time'] / successful[0]['time']
            print(f"   {speedup:.2f}x faster than slowest approach")

    # Print failed approaches
    failed = [r for r in results if not r['success']]
    if failed:
        print("\n❌ Failed approaches:")
        for result in failed:
            print(f"   - {result['name']}: {result['error']}")


if __name__ == "__main__":
    main()
