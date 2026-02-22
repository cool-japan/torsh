# Fuzzing for torsh-quantization

This directory contains fuzz targets for testing the robustness of the quantization library using [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz).

## Prerequisites

Install cargo-fuzz:
```bash
cargo install cargo-fuzz
```

## Running Fuzz Tests

### Fuzz Per-Tensor Quantization
```bash
cargo fuzz run fuzz_quantize_per_tensor
```

### Fuzz Observer Updates
```bash
cargo fuzz run fuzz_observer_update
```

### Fuzz Specialized Schemes
```bash
cargo fuzz run fuzz_specialized_schemes
```

## Fuzz Targets

- **fuzz_quantize_per_tensor**: Tests per-tensor affine quantization with arbitrary inputs
- **fuzz_observer_update**: Tests observer parameter calculation robustness
- **fuzz_specialized_schemes**: Tests INT4, binary, and ternary quantization

## Coverage

Run with coverage instrumentation:
```bash
cargo fuzz coverage fuzz_quantize_per_tensor
```

## Continuous Fuzzing

For continuous integration, run with a timeout:
```bash
cargo fuzz run fuzz_quantize_per_tensor -- -max_total_time=300
```

## Corpus

Fuzz targets automatically build a corpus of interesting inputs in:
```
fuzz/corpus/fuzz_<target_name>/
```

## Crashes

Any crashes found will be saved to:
```
fuzz/artifacts/fuzz_<target_name>/
```

## Integration with CI

Add to your CI pipeline:
```yaml
- name: Fuzz testing
  run: |
    cargo install cargo-fuzz
    cargo fuzz run fuzz_quantize_per_tensor -- -max_total_time=60
    cargo fuzz run fuzz_observer_update -- -max_total_time=60
    cargo fuzz run fuzz_specialized_schemes -- -max_total_time=60
```
