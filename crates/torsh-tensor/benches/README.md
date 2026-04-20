# torsh-tensor Benchmarks

## Performance Regression Framework

### Quick start

Record a baseline (run this before making performance changes):

    cargo bench --bench regression_baselines -- --save-baseline v0.1.2-pre

After your changes, compare against the baseline:

    cargo bench --bench regression_baselines -- --baseline v0.1.2-pre

Check for regressions (fails with exit code 1 if any bench slowed by >10%):

    bash scripts/check_perf_regression.sh

Use a tighter threshold (e.g., 5%):

    bash scripts/check_perf_regression.sh 5

### Benchmarks included

- `add_into_f32/{256,4096,65536}` — out-of-place f32 element-wise add (raw SIMD helper)
- `add_assign_f32/{256,4096,65536}` — in-place f32 element-wise add (raw SIMD helper)
- `relu_inplace/{256,4096,65536}` — in-place ReLU activation (raw SIMD helper)
- `clamp_inplace/{256,4096,65536}` — in-place clamp activation (raw SIMD helper)
- `tensor_add_f32/{256,4096,65536}` — `Tensor::add` (high-level, includes storage overhead)
- `tensor_relu_f32/{256,4096,65536}` — `Tensor::relu` (high-level, includes storage overhead)

### Notes

Baselines are machine-specific. Only compare runs from the same machine and CPU.
The threshold check uses **relative percentage change**, not absolute ns/iter.

### SIMD-gated benchmarks (require `--features simd`)

- `simd_performance` — AVX2/NEON SIMD vs scalar comparison
- `zero_copy_simd_benchmark` — zero-copy tensor view SIMD operations
