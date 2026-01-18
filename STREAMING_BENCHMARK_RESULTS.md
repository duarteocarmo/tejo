# Streaming Approaches Benchmark Results

Comparison of different streaming approaches for loading FineWeb v2 Portuguese data.

**Goal:** Gather 10M tokens (where 1 token ≈ 3 chars = 30M characters)

## Test Results (Local Mock Data)

Tested with 50,000 mock Portuguese documents (~25M characters):

### Rankings (Fastest to Slowest)

| Rank | Approach | Time | Throughput | Speedup |
|------|----------|------|------------|---------|
| 🏆 1 | **Polars with streaming/lazy evaluation** | 0.02s | 1,124 M chars/sec | 6.07x |
| 2 | Standard Python iteration | 0.13s | 195 M chars/sec | 1.05x |
| 3 | Polars eager mode (all in memory) | 0.13s | 185 M chars/sec | 1.00x |

### Winner: Polars Streaming!

**Polars with streaming/lazy evaluation is 6x faster** than the other approaches.

## Key Findings

### 1. Polars Streaming (Winner)
```python
# Use scan_parquet for lazy evaluation
lazy_df = pl.scan_parquet("data/*.parquet")
lazy_df = lazy_df.filter(pl.col('language') == 'pt')
lazy_df = lazy_df.with_columns(pl.col('text').str.len_chars().alias('text_len'))

# Stream and process
for batch in lazy_df.collect(engine='streaming').iter_slices(1000):
    # Process batch
    pass
```

**Advantages:**
- ✅ Extremely fast (6x faster than alternatives)
- ✅ Memory efficient (streaming/lazy evaluation)
- ✅ Optimized query execution
- ✅ Built-in parallelization

**When to use:** Large datasets, memory-constrained environments

### 2. Standard Python Iteration
```python
for pq_file in parquet_files:
    df = pl.read_parquet(pq_file)
    for row in df.iter_rows(named=True):
        # Process row
        pass
```

**Advantages:**
- ✅ Simple and straightforward
- ✅ Easy to understand and debug
- ✅ More flexible for custom processing

**Disadvantages:**
- ❌ Slower (6x slower than Polars streaming)
- ❌ Row-by-row iteration is inefficient

### 3. Polars Eager Mode
```python
# Load all data into memory
dfs = [pl.read_parquet(f) for f in parquet_files]
df = pl.concat(dfs)
df = df.filter(pl.col('domain') == 'pt')

# Process
for row in df.iter_rows(named=True):
    # Process row
    pass
```

**Advantages:**
- ✅ All data available at once
- ✅ Can use full Polars API

**Disadvantages:**
- ❌ Requires loading all data into memory
- ❌ Not suitable for large datasets
- ❌ Slowest approach in our test

## Recommendation for FineWeb v2

For processing FineWeb v2 Portuguese data (filtered by .pt domains):

### **Use Polars Streaming** 🏆

```python
import polars as pl

# Scan parquet files lazily (FineWeb v2 is distributed as parquet)
lazy_df = pl.scan_parquet("path/to/fineweb2/*.parquet")

# Filter for Portuguese + .pt domains (lazy - not executed yet)
lazy_df = lazy_df.filter(
    (pl.col('language') == 'pt') &
    (pl.col('url').str.contains('.pt'))
)

# Add computed columns (lazy)
lazy_df = lazy_df.with_columns(
    pl.col('text').str.len_chars().alias('text_len')
)

# Stream and process in batches
total_chars = 0
target_chars = 30_000_000  # 10M tokens

for batch in lazy_df.collect(engine='streaming').iter_slices(10000):
    total_chars += batch['text_len'].sum()

    # Your processing logic here
    # ...

    if total_chars >= target_chars:
        break

print(f"Processed {total_chars:,} characters")
```

## HuggingFace Datasets Streaming

For comparison, here's how to use HuggingFace datasets with streaming:

```python
from datasets import load_dataset

# Load with streaming
dataset = load_dataset(
    "HuggingFaceFW/fineweb-2",
    split="train",
    streaming=True
)

# Filter
dataset = dataset.filter(lambda x: x.get('language') == 'pt')
dataset = dataset.filter(lambda x: '.pt' in x.get('url', '').lower())

# Process
total_chars = 0
for doc in dataset:
    text = doc.get('text', '')
    total_chars += len(text)

    if total_chars >= 30_000_000:
        break
```

**Note:** HuggingFace datasets streaming is convenient but typically slower than Polars streaming for parquet files. Use HF datasets when you need the HuggingFace ecosystem integration.

## Performance Tips

1. **Use Polars streaming** for maximum performance
2. **Batch processing** (process 1000-10000 rows at a time, not row-by-row)
3. **Lazy evaluation** (build query first, execute once)
4. **Parquet format** is highly optimized for columnar access
5. **Filter early** (apply filters in the lazy query, not after loading)

## Environment

- Python 3.11
- Polars 1.37.1
- PyArrow 23.0.0
- Test data: 50,000 documents, ~25M characters

## Scripts

- `benchmark_streaming_local.py` - Local benchmark with mock data (runs offline)
- `benchmark_streaming.py` - Full benchmark with FineWeb v2 (requires internet access)
