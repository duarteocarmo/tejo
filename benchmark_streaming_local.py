"""
Benchmark different streaming approaches for processing text data.

This script compares:
1. Polars with streaming/lazy evaluation
2. Polars eager mode (baseline)
3. Standard Python iteration

Goal: Gather 10M tokens (1 token ≈ 3 chars = 30M characters)
Using: Local generated Portuguese-like text data
"""

import time
import polars as pl
import tempfile
import os
from pathlib import Path
from typing import Dict, Callable
import random

# Target: 10M tokens = ~30M characters
TARGET_TOKENS = 10_000_000
CHARS_PER_TOKEN = 3
TARGET_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN

print(f"Target: {TARGET_TOKENS:,} tokens (~{TARGET_CHARS:,} characters)")
print("=" * 80)


def generate_mock_portuguese_data(num_docs: int, output_dir: Path):
    """Generate mock Portuguese web data as parquet files."""
    print(f"Generating {num_docs:,} mock Portuguese documents...")

    # Sample Portuguese text fragments (realistic web content)
    pt_fragments = [
        "O site português de notícias apresenta as últimas informações sobre política e economia.",
        "A cultura portuguesa é rica em tradições históricas que remontam séculos atrás.",
        "Os utilizadores podem aceder à plataforma através do domínio .pt oficial.",
        "Esta página web oferece serviços digitais para empresas portuguesas.",
        "A tecnologia moderna tem transformado a forma como comunicamos em Portugal.",
        "Os dados mostram um crescimento significativo no sector tecnológico português.",
        "Lisboa é conhecida pelos seus monumentos históricos e arquitectura única.",
        "A gastronomia portuguesa inclui pratos tradicionais como bacalhau e pastéis de nata.",
        "O sistema educativo em Portugal tem evoluído nas últimas décadas.",
        "As empresas portuguesas estão cada vez mais focadas na inovação digital.",
    ]

    documents = []
    for i in range(num_docs):
        # Create realistic document
        num_paragraphs = random.randint(3, 10)
        text = " ".join(
            random.choice(pt_fragments) for _ in range(num_paragraphs)
        )

        documents.append({
            'id': f'doc_{i:06d}',
            'text': text,
            'url': f'https://example{i % 100}.pt/page{i}',
            'language': 'pt',
            'domain': 'pt',
            'length': len(text)
        })

        if (i + 1) % 10000 == 0:
            print(f"  Generated {i + 1:,} documents...", end='\r')

    print(f"\n  Created {len(documents):,} documents")

    # Save as parquet files (split into shards for realism)
    num_shards = 10
    docs_per_shard = num_docs // num_shards

    for shard_idx in range(num_shards):
        start_idx = shard_idx * docs_per_shard
        end_idx = start_idx + docs_per_shard if shard_idx < num_shards - 1 else num_docs
        shard_docs = documents[start_idx:end_idx]

        df = pl.DataFrame(shard_docs)
        output_file = output_dir / f"shard_{shard_idx:02d}.parquet"
        df.write_parquet(output_file, compression='zstd')

    print(f"  Saved {num_shards} parquet shards to {output_dir}")

    total_chars = sum(len(d['text']) for d in documents)
    print(f"  Total characters: {total_chars:,}")
    return num_shards


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
        import traceback
        traceback.print_exc()
        return {
            'name': name,
            'success': False,
            'time': elapsed,
            'error': str(e)
        }


def approach_1_polars_streaming(data_dir: Path):
    """Polars with streaming/lazy evaluation."""
    print("Using Polars with lazy/streaming evaluation...")

    # Use scan_parquet for lazy evaluation
    parquet_files = list(data_dir.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")

    # Lazy scan all parquet files
    lazy_df = pl.scan_parquet(str(data_dir / "*.parquet"))

    # Filter for Portuguese .pt domains (lazy - not executed yet)
    lazy_df = lazy_df.filter(pl.col('domain') == 'pt')

    # Add text length column (lazy)
    lazy_df = lazy_df.with_columns(
        pl.col('text').str.len_chars().alias('text_len')
    )

    # Stream and process
    print("Streaming and processing...")
    total_chars = 0
    doc_count = 0

    # Process in streaming batches (using engine='streaming' instead of deprecated streaming=True)
    for batch in lazy_df.collect(engine='streaming').iter_slices(1000):
        chars_in_batch = batch['text_len'].sum()
        total_chars += chars_in_batch
        doc_count += len(batch)

        if doc_count % 10000 == 0:
            print(f"  Docs: {doc_count:,} | Chars: {total_chars:,} | Target: {TARGET_CHARS:,}", end='\r')

        if total_chars >= TARGET_CHARS:
            break

    print()
    return {'chars': total_chars, 'docs': doc_count}


def approach_2_polars_eager(data_dir: Path):
    """Polars eager mode (load all into memory)."""
    print("Using Polars eager mode (load all into memory)...")

    # Read all parquet files eagerly
    parquet_files = list(data_dir.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")

    print("Loading all data into memory...")
    dfs = [pl.read_parquet(f) for f in parquet_files]
    df = pl.concat(dfs)

    print(f"Loaded {len(df):,} documents")

    # Filter
    df = df.filter(pl.col('domain') == 'pt')

    # Process
    print("Processing...")
    total_chars = 0
    doc_count = 0

    for row in df.iter_rows(named=True):
        text = row['text']
        total_chars += len(text)
        doc_count += 1

        if doc_count % 10000 == 0:
            print(f"  Docs: {doc_count:,} | Chars: {total_chars:,} | Target: {TARGET_CHARS:,}", end='\r')

        if total_chars >= TARGET_CHARS:
            break

    print()
    return {'chars': total_chars, 'docs': doc_count}


def approach_3_python_iteration(data_dir: Path):
    """Standard Python with row-by-row iteration."""
    print("Using standard Python iteration...")

    parquet_files = sorted(data_dir.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")

    total_chars = 0
    doc_count = 0

    for pq_file in parquet_files:
        df = pl.read_parquet(pq_file)

        for row in df.iter_rows(named=True):
            if row['domain'] == 'pt':
                text = row['text']
                total_chars += len(text)
                doc_count += 1

                if doc_count % 10000 == 0:
                    print(f"  Docs: {doc_count:,} | Chars: {total_chars:,} | Target: {TARGET_CHARS:,}", end='\r')

                if total_chars >= TARGET_CHARS:
                    print()
                    return {'chars': total_chars, 'docs': doc_count}

    print()
    return {'chars': total_chars, 'docs': doc_count}


def main():
    """Run all benchmarks and compare results."""

    # Setup
    temp_dir = Path(tempfile.mkdtemp(prefix="polars_bench_"))
    print(f"Using temporary directory: {temp_dir}\n")

    try:
        # Generate mock data
        # We need enough data to reach 30M chars
        # Each doc ~500-1500 chars, so need ~30k docs minimum
        num_docs = 50000  # Extra to ensure we hit target
        num_shards = generate_mock_portuguese_data(num_docs, temp_dir)

        results = []

        # Test Polars streaming (lazy + streaming)
        results.append(benchmark_approach(
            "Polars with streaming/lazy evaluation",
            lambda: approach_1_polars_streaming(temp_dir)
        ))

        # Test Polars eager mode
        results.append(benchmark_approach(
            "Polars eager mode (all in memory)",
            lambda: approach_2_polars_eager(temp_dir)
        ))

        # Test Python iteration
        results.append(benchmark_approach(
            "Standard Python iteration",
            lambda: approach_3_python_iteration(temp_dir)
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

    finally:
        # Cleanup
        print(f"\nCleaning up temporary directory: {temp_dir}")
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
