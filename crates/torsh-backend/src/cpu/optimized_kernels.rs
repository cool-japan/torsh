//! Optimized kernels for CPU backend without external BLAS dependency

use crate::cpu::error::CpuResult;
// ✅ SciRS2 POLICY: Use scirs2_core::parallel_ops instead of direct rayon
use scirs2_core::parallel_ops::*;
use torsh_core::error::{Result, TorshError};

// Re-export commonly used functions for easier access
pub use parallel_ops::parallel_sum;

/// Cache-blocked matrix multiplication for better performance
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

    // Cache blocking parameters
    const BLOCK_SIZE: usize = 64;

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

    // Cache-blocked matrix multiplication
    for ii in (0..m).step_by(BLOCK_SIZE) {
        for jj in (0..n).step_by(BLOCK_SIZE) {
            for kk in (0..k).step_by(BLOCK_SIZE) {
                let i_end = (ii + BLOCK_SIZE).min(m);
                let j_end = (jj + BLOCK_SIZE).min(n);
                let k_end = (kk + BLOCK_SIZE).min(k);

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

        optimized_matmul(&a, &b, &mut result, 2, 2, 3, false, false).unwrap();

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

        let result = optimized_dot(&a, &b).unwrap();
        assert_abs_diff_eq!(result, 32.0, epsilon = 1e-6); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_optimized_matvec() {
        // 2x3 matrix * 3x1 vector = 2x1 vector
        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2,3], [4,5,6]]
        let vector = vec![1.0, 2.0, 3.0];
        let mut result = vec![0.0; 2];

        optimized_matvec(&matrix, &vector, &mut result, 2, 3, false).unwrap();

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
}
