//! Optimized kernels for CPU backend without external BLAS dependency
//!
//! Phase 4: Intelligent chunking is wired into matmul (cache-optimal block sizes),
//! elementwise ops, and reduction ops via `ChunkingUtils` / `WorkloadType`.

use crate::cpu::error::CpuResult;
use crate::cpu::scirs2_chunking::{ChunkingStrategy, ChunkingUtils, WorkloadType};
// ✅ SciRS2 POLICY: Use scirs2_core::parallel_ops instead of direct rayon
use scirs2_core::parallel_ops::*;
use torsh_core::error::{Result, TorshError};

// Re-export commonly used functions for easier access
pub use parallel_ops::parallel_sum;

/// Cache-blocked matrix multiplication driven by Phase-4 intelligent chunking.
///
/// Block sizes for the three tiling dimensions (bm, bn, bk) are chosen by
/// [`ChunkingUtils::matrix_blocks`] so that each panel fits in L2 cache, rather
/// than relying on the former compile-time constant of 64.
#[allow(clippy::too_many_arguments)]
pub fn optimized_matmul(
    a: &[f32],
    b: &[f32],
    result: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    transpose_a: bool,
    transpose_b: bool,
) -> Result<()> {
    if a.len() != m * k || b.len() != k * n || result.len() != m * n {
        return Err(TorshError::ShapeMismatch {
            expected: vec![m * k, k * n, m * n],
            got: vec![a.len(), b.len(), result.len()],
        });
    }

    // Initialize result to zero
    result.fill(0.0);

    // Phase 4: derive cache-optimal block sizes from hardware topology.
    // element_size = 4 bytes (f32).
    let (block_m, block_n, block_k) = ChunkingUtils::matrix_blocks(m, n, k, 4);

    // Handle transposition by choosing appropriate indexing functions
    let get_a = |i: usize, j: usize| -> f32 {
        if transpose_a {
            a[j * m + i] // A^T
        } else {
            a[i * k + j] // A
        }
    };

    let get_b = |i: usize, j: usize| -> f32 {
        if transpose_b {
            b[j * k + i] // B^T
        } else {
            b[i * n + j] // B
        }
    };

    // Cache-blocked matrix multiplication with hardware-derived block sizes
    for ii in (0..m).step_by(block_m) {
        for jj in (0..n).step_by(block_n) {
            for kk in (0..k).step_by(block_k) {
                let i_end = (ii + block_m).min(m);
                let j_end = (jj + block_n).min(n);
                let k_end = (kk + block_k).min(k);

                for i in ii..i_end {
                    for j in jj..j_end {
                        let mut sum = 0.0;

                        // Inner loop vectorization hint
                        for kk_inner in kk..k_end {
                            sum += get_a(i, kk_inner) * get_b(kk_inner, j);
                        }

                        result[i * n + j] += sum;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Chunking-aware element-wise operation over two f32 slices.
///
/// Uses `WorkloadType::Elementwise` to derive the cache-optimal chunk size,
/// then applies `op` to each element pair in parallel.  Falls back gracefully
/// to a scalar loop for small inputs (<= one chunk).
pub fn chunked_elementwise<F>(a: &[f32], b: &[f32], result: &mut [f32], op: F) -> Result<()>
where
    F: Fn(f32, f32) -> f32 + Sync + Send,
{
    let len = a.len();
    if b.len() != len || result.len() != len {
        return Err(TorshError::ShapeMismatch {
            expected: vec![len, len],
            got: vec![b.len(), result.len()],
        });
    }

    let strategy = ChunkingStrategy::new(WorkloadType::Elementwise, 4);
    let chunk_sz = strategy.chunk_size(len);

    if len <= chunk_sz {
        // Single-chunk path — no parallel overhead for small arrays
        for idx in 0..len {
            result[idx] = op(a[idx], b[idx]);
        }
    } else {
        // Multi-chunk parallel path
        let chunks = strategy.split_range(0, len);
        // Parallel execution: each chunk is an independent stripe
        result
            .par_chunks_mut(chunk_sz)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let start = chunk_idx * chunk_sz;
                let end = (start + out_chunk.len()).min(len);
                let in_a = &a[start..end];
                let in_b = &b[start..end];
                for (i, r) in out_chunk.iter_mut().enumerate() {
                    *r = op(in_a[i], in_b[i]);
                }
            });
        // suppress unused variable warning for `chunks`
        let _ = chunks;
    }

    Ok(())
}

/// Chunking-aware reduction (sum) over an f32 slice.
///
/// Uses `WorkloadType::Reduction` to select a chunk size that keeps partial
/// accumulators in L2 cache, avoiding last-level-cache pressure on wide arrays.
pub fn chunked_sum(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    let strategy = ChunkingStrategy::new(WorkloadType::Reduction, 4);
    let chunk_sz = strategy.chunk_size(data.len());

    if data.len() <= chunk_sz {
        // Small array: single accumulator, no overhead
        data.iter().sum()
    } else {
        // Accumulate partial sums per chunk, then sum the partials
        let chunks = strategy.split_range(0, data.len());
        let partial_sums: Vec<f32> = chunks
            .iter()
            .map(|&(start, end)| data[start..end].iter().sum::<f32>())
            .collect();
        partial_sums.iter().sum()
    }
}

/// Chunking-aware mean over an f32 slice.
pub fn chunked_mean(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    chunked_sum(data) / data.len() as f32
}

/// Basic matrix multiplication for benchmarking
///
/// This is a simplified version of optimized_matmul without blocking
/// for benchmarking purposes.
pub fn optimized_matmul_basic(
    a: &[f32],
    b: &[f32],
    result: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    optimized_matmul(a, b, result, m, n, k, false, false)
}

/// Optimized dot product using loop unrolling
pub fn optimized_dot(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(TorshError::ShapeMismatch {
            expected: vec![a.len()],
            got: vec![b.len()],
        });
    }

    let len = a.len();
    let mut sum = 0.0f32;

    // Process 4 elements at a time for better performance
    let chunks = len / 4;
    let remainder = len % 4;

    for i in 0..chunks {
        let base = i * 4;
        sum += a[base] * b[base]
            + a[base + 1] * b[base + 1]
            + a[base + 2] * b[base + 2]
            + a[base + 3] * b[base + 3];
    }

    // Handle remaining elements
    for i in (chunks * 4)..(chunks * 4 + remainder) {
        sum += a[i] * b[i];
    }

    Ok(sum)
}

/// Optimized matrix-vector multiplication
pub fn optimized_matvec(
    matrix: &[f32],
    vector: &[f32],
    result: &mut [f32],
    m: usize,
    n: usize,
    transpose: bool,
) -> Result<()> {
    if matrix.len() != m * n {
        return Err(TorshError::ShapeMismatch {
            expected: vec![m * n],
            got: vec![matrix.len()],
        });
    }

    let (expected_vec_len, expected_result_len) = if transpose { (m, n) } else { (n, m) };

    if vector.len() != expected_vec_len || result.len() != expected_result_len {
        return Err(TorshError::ShapeMismatch {
            expected: vec![expected_vec_len, expected_result_len],
            got: vec![vector.len(), result.len()],
        });
    }

    result.fill(0.0);

    if transpose {
        // Matrix^T * vector
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..m {
                sum += matrix[j * n + i] * vector[j];
            }
            result[i] = sum;
        }
    } else {
        // Matrix * vector
        for i in 0..m {
            let mut sum = 0.0;
            for j in 0..n {
                sum += matrix[i * n + j] * vector[j];
            }
            result[i] = sum;
        }
    }

    Ok(())
}

/// Parallel reduction operations
pub mod parallel_ops {
    use super::*;

    /// Parallel sum using divide-and-conquer
    pub fn parallel_sum(data: &[f32]) -> f32 {
        data.par_iter().sum()
    }

    /// Parallel mean calculation
    pub fn parallel_mean(data: &[f32]) -> f32 {
        if data.is_empty() {
            0.0
        } else {
            parallel_sum(data) / data.len() as f32
        }
    }

    /// Parallel element-wise operations
    pub fn parallel_elementwise<F>(a: &[f32], b: &[f32], result: &mut [f32], op: F)
    where
        F: Fn(f32, f32) -> f32 + Sync + Send,
    {
        result
            .par_iter_mut()
            .zip(a.par_iter().zip(b.par_iter()))
            .for_each(|(r, (&a_val, &b_val))| *r = op(a_val, b_val));
    }

    /// Parallel unary operations
    pub fn parallel_unary<F>(input: &[f32], output: &mut [f32], op: F)
    where
        F: Fn(f32) -> f32 + Sync + Send,
    {
        output
            .par_iter_mut()
            .zip(input.par_iter())
            .for_each(|(out, &inp)| *out = op(inp));
    }
}

/// Advanced optimization kernels
pub mod advanced {
    use super::*;

    /// Simple convolution implementation
    #[allow(clippy::too_many_arguments)]
    pub fn optimized_conv2d(
        input: &[f32],
        weight: &[f32],
        output: &mut [f32],
        batch_size: usize,
        in_channels: usize,
        input_height: usize,
        input_width: usize,
        out_channels: usize,
        kernel_height: usize,
        kernel_width: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> CpuResult<()> {
        let output_height = (input_height + 2 * pad_h - kernel_height) / stride_h + 1;
        let output_width = (input_width + 2 * pad_w - kernel_width) / stride_w + 1;

        // Simple convolution implementation
        for batch in 0..batch_size {
            for out_c in 0..out_channels {
                for out_h in 0..output_height {
                    for out_w in 0..output_width {
                        let mut sum = 0.0f32;

                        for in_c in 0..in_channels {
                            for kh in 0..kernel_height {
                                for kw in 0..kernel_width {
                                    let input_h = out_h * stride_h + kh;
                                    let input_w = out_w * stride_w + kw;

                                    if input_h >= pad_h && input_w >= pad_w {
                                        let ih = input_h - pad_h;
                                        let iw = input_w - pad_w;

                                        if ih < input_height && iw < input_width {
                                            let input_idx =
                                                batch * in_channels * input_height * input_width
                                                    + in_c * input_height * input_width
                                                    + ih * input_width
                                                    + iw;
                                            let weight_idx =
                                                out_c * in_channels * kernel_height * kernel_width
                                                    + in_c * kernel_height * kernel_width
                                                    + kh * kernel_width
                                                    + kw;

                                            sum += input[input_idx] * weight[weight_idx];
                                        }
                                    }
                                }
                            }
                        }

                        let output_idx = batch * out_channels * output_height * output_width
                            + out_c * output_height * output_width
                            + out_h * output_width
                            + out_w;
                        output[output_idx] = sum;
                    }
                }
            }
        }

        Ok(())
    }

    /// Optimized batch normalization
    #[allow(clippy::too_many_arguments)]
    pub fn optimized_batch_norm(
        input: &[f32],
        output: &mut [f32],
        mean: &[f32],
        variance: &[f32],
        weight: Option<&[f32]>,
        bias: Option<&[f32]>,
        eps: f32,
        batch_size: usize,
        channels: usize,
        spatial_size: usize,
    ) -> CpuResult<()> {
        if input.len() != batch_size * channels * spatial_size {
            return Err(crate::cpu::error::cpu_errors::invalid_parameter_error(
                "Input size mismatch".to_string(),
            ));
        }

        output
            .par_chunks_mut(channels * spatial_size)
            .enumerate()
            .for_each(|(batch, batch_output)| {
                let batch_input =
                    &input[batch * channels * spatial_size..(batch + 1) * channels * spatial_size];

                for c in 0..channels {
                    let inv_std = 1.0 / (variance[c] + eps).sqrt();
                    let gamma = weight.map(|w| w[c]).unwrap_or(1.0);
                    let beta = bias.map(|b| b[c]).unwrap_or(0.0);

                    for s in 0..spatial_size {
                        let idx = c * spatial_size + s;
                        let normalized = (batch_input[idx] - mean[c]) * inv_std;
                        batch_output[idx] = gamma * normalized + beta;
                    }
                }
            });

        Ok(())
    }

    /// Optimized softmax with numerical stability
    pub fn optimized_softmax(
        input: &[f32],
        output: &mut [f32],
        batch_size: usize,
        num_classes: usize,
    ) -> CpuResult<()> {
        if input.len() != batch_size * num_classes || output.len() != input.len() {
            return Err(crate::cpu::error::cpu_errors::invalid_parameter_error(
                "Size mismatch".to_string(),
            ));
        }

        output
            .par_chunks_mut(num_classes)
            .enumerate()
            .for_each(|(batch, batch_output)| {
                let batch_input = &input[batch * num_classes..(batch + 1) * num_classes];

                // Find max for numerical stability
                let max_val = batch_input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                // Compute exp(x - max) and sum
                let mut sum = 0.0f32;
                for (i, &x) in batch_input.iter().enumerate() {
                    let exp_val = (x - max_val).exp();
                    batch_output[i] = exp_val;
                    sum += exp_val;
                }

                // Normalize
                let inv_sum = 1.0 / sum;
                for val in batch_output.iter_mut() {
                    *val *= inv_sum;
                }
            });

        Ok(())
    }

    /// Memory-efficient matrix multiplication with threading
    pub fn threaded_matmul(
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> CpuResult<()> {
        if a.len() != m * k || b.len() != k * n || result.len() != m * n {
            return Err(crate::cpu::error::cpu_errors::invalid_parameter_error(
                "Shape mismatch".to_string(),
            ));
        }

        result.fill(0.0);

        // Parallel over rows
        result.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            for j in 0..n {
                let mut sum = 0.0f32;
                for k_idx in 0..k {
                    sum += a[i * k + k_idx] * b[k_idx * n + j];
                }
                row[j] = sum;
            }
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_optimized_matmul_basic() {
        // 2x3 * 3x2 = 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2,3], [4,5,6]]
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2], [3,4], [5,6]]
        let mut result = vec![0.0; 4];

        optimized_matmul(&a, &b, &mut result, 2, 2, 3, false, false)
            .expect("optimized matmul should succeed");

        // Expected: [[22, 28], [49, 64]] in row-major order
        assert_abs_diff_eq!(result[0], 22.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[1], 28.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[2], 49.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[3], 64.0, epsilon = 1e-6);
    }

    #[test]
    fn test_optimized_dot() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let result = optimized_dot(&a, &b).expect("optimized dot should succeed");
        assert_abs_diff_eq!(result, 32.0, epsilon = 1e-6); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_optimized_matvec() {
        // 2x3 matrix * 3x1 vector = 2x1 vector
        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2,3], [4,5,6]]
        let vector = vec![1.0, 2.0, 3.0];
        let mut result = vec![0.0; 2];

        optimized_matvec(&matrix, &vector, &mut result, 2, 3, false)
            .expect("optimized matvec should succeed");

        // Expected: [14, 32] (1*1+2*2+3*3=14, 4*1+5*2+6*3=32)
        assert_abs_diff_eq!(result[0], 14.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[1], 32.0, epsilon = 1e-6);
    }

    #[test]
    fn test_parallel_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = parallel_ops::parallel_sum(&data);
        assert_abs_diff_eq!(result, 15.0, epsilon = 1e-6);
    }

    // --- Phase 4: Intelligent chunking tests ---

    #[test]
    fn test_chunked_elementwise_add_small() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![10.0f32, 20.0, 30.0, 40.0];
        let mut out = vec![0.0f32; 4];
        chunked_elementwise(&a, &b, &mut out, |x, y| x + y).expect("chunked add should succeed");
        assert_abs_diff_eq!(out[0], 11.0, epsilon = 1e-6);
        assert_abs_diff_eq!(out[1], 22.0, epsilon = 1e-6);
        assert_abs_diff_eq!(out[2], 33.0, epsilon = 1e-6);
        assert_abs_diff_eq!(out[3], 44.0, epsilon = 1e-6);
    }

    #[test]
    fn test_chunked_elementwise_mul_large() {
        // Large enough to trigger the multi-chunk parallel path
        let n = 100_000usize;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|_| 2.0f32).collect();
        let mut out = vec![0.0f32; n];
        chunked_elementwise(&a, &b, &mut out, |x, y| x * y).expect("chunked mul should succeed");
        for (i, &v) in out.iter().enumerate() {
            assert_abs_diff_eq!(v, (i as f32) * 2.0, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_chunked_elementwise_shape_mismatch() {
        let a = vec![1.0f32; 4];
        let b = vec![1.0f32; 3];
        let mut out = vec![0.0f32; 4];
        assert!(chunked_elementwise(&a, &b, &mut out, |x, y| x + y).is_err());
    }

    #[test]
    fn test_chunked_sum_empty() {
        assert_abs_diff_eq!(chunked_sum(&[]), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn test_chunked_sum_small() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert_abs_diff_eq!(chunked_sum(&data), 15.0, epsilon = 1e-6);
    }

    #[test]
    fn test_chunked_sum_large() {
        // Sum of 0..N should equal N*(N-1)/2
        let n = 50_000usize;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let expected = (n * (n - 1) / 2) as f32;
        let got = chunked_sum(&data);
        // Allow relative tolerance for large float sums
        assert!((got - expected).abs() / expected < 1e-3);
    }

    #[test]
    fn test_chunked_mean_small() {
        let data = vec![2.0f32, 4.0, 6.0, 8.0];
        assert_abs_diff_eq!(chunked_mean(&data), 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_chunked_mean_empty() {
        assert_abs_diff_eq!(chunked_mean(&[]), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn test_optimized_matmul_uses_chunking_large() {
        // 128x128 matmul — exercises all three tiling dimensions with chunking-derived block sizes
        let m = 128;
        let n = 128;
        let k = 64;
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.001).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.001).collect();
        let mut result = vec![0.0f32; m * n];

        optimized_matmul(&a, &b, &mut result, m, n, k, false, false)
            .expect("large chunked matmul should succeed");

        // Spot-check: result[0][0] = row0 of A dot col0 of B
        let expected_00: f32 = (0..k).map(|l| a[l] * b[l * n]).sum();
        assert_abs_diff_eq!(result[0], expected_00, epsilon = 1e-3);
    }
}
