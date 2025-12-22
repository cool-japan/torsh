# Benchmark Comparison Report

Generated: 2025-11-13 22:03:37 UTC

## Summary

- **Total Comparisons**: 9
- **Improvements**: 9 (100.0%)
- **Regressions**: 0 (0.0%)
- **No Change**: 0 (0.0%)

## Detailed Comparison

| Benchmark | Baseline (ns) | Candidate (ns) | Speedup | Change | Significance | Verdict |
|-----------|---------------|----------------|---------|--------|--------------|----------|
| conv2d_128 | 4000.00 | 2400.00 | 1.67x | +66.7% | *** | 🚀 |
| matmul_1024 | 64000.00 | 48000.00 | 1.33x | +33.3% | *** | 🚀 |
| conv2d_256 | 32000.00 | 19200.00 | 1.67x | +66.7% | *** | 🚀 |
| reduction_sum_1024 | 100.00 | 85.00 | 1.18x | +17.6% | *** | ✅ |
| conv2d_64 | 500.00 | 300.00 | 1.67x | +66.7% | *** | 🚀 |
| matmul_256 | 1000.00 | 750.00 | 1.33x | +33.3% | *** | 🚀 |
| matmul_512 | 8000.00 | 6000.00 | 1.33x | +33.3% | *** | 🚀 |
| reduction_sum_4096 | 400.00 | 340.00 | 1.18x | +17.6% | *** | ✅ |
| reduction_sum_16384 | 1600.00 | 1360.00 | 1.18x | +17.6% | *** | ✅ |

### Legend
- Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant
- Verdict: 🚀 Major improvement (>20%), ✅ Improvement (5-20%), ➖ No change (<5%), ⚠️ Regression (5-20%), 🔴 Major regression (>20%)
