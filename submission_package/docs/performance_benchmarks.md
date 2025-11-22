# Performance Benchmarks

## Environment

- CPU: <insert CPU model>
- RAM: <insert RAM>
- Python: <version>
- Model: TF-IDF + LogisticRegression
- Model file: artifacts/checkpoints/baseline.joblib

## Benchmark results

- Single-call average latency: 0.0004529 s
- Batch (1000) average per-example latency: 0.0000190 s
- Throughput: ~52,600 requests/sec (batch)

## Methodology

- Warm-up: 10 runs
- Measured: 1000 runs
- Tool: `src/benchmark.py` (simple time-based benchmarking)
