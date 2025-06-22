//! Optimized kernels for CPU backend without external BLAS dependency

use rayon::prelude::*;
use torsh_core::error::{Result, TorshError};

/// Cache-blocked matrix multiplication for better performance
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

    // Cache blocking parameters - tune these for your system
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

/// Optimized dot product using loop unrolling and SIMD-like operations
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

/// Parallel reduction operations for large arrays
pub mod parallel_ops {
    use super::*;

    /// Parallel sum using divide-and-conquer
    pub fn parallel_sum(data: &[f32]) -> f32 {
        #[cfg(feature = "rayon-threads")]
        {
            use rayon::prelude::*;
            data.par_iter().sum()
        }
        #[cfg(not(feature = "rayon-threads"))]
        {
            // Fallback to simple sum
            data.iter().sum()
        }
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
        #[cfg(feature = "rayon-threads")]
        {
            use rayon::prelude::*;
            result
                .par_iter_mut()
                .zip(a.par_iter().zip(b.par_iter()))
                .for_each(|(r, (&a_val, &b_val))| *r = op(a_val, b_val));
        }
        #[cfg(not(feature = "rayon-threads"))]
        {
            result
                .iter_mut()
                .zip(a.iter().zip(b.iter()))
                .for_each(|(r, (&a_val, &b_val))| *r = op(a_val, b_val));
        }
    }

    /// Parallel unary operations
    pub fn parallel_unary<F>(input: &[f32], output: &mut [f32], op: F)
    where
        F: Fn(f32) -> f32 + Sync + Send,
    {
        #[cfg(feature = "rayon-threads")]
        {
            use rayon::prelude::*;
            output
                .par_iter_mut()
                .zip(input.par_iter())
                .for_each(|(out, &inp)| *out = op(inp));
        }
        #[cfg(not(feature = "rayon-threads"))]
        {
            output
                .iter_mut()
                .zip(input.iter())
                .for_each(|(out, &inp)| *out = op(inp));
        }
    }
}

/// Advanced optimization kernels
pub mod advanced {
    use super::*;
    use crate::simd::SimdOps;

    /// Optimized convolution using im2col transformation
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
    ) -> Result<()> {
        let output_height = (input_height + 2 * pad_h - kernel_height) / stride_h + 1;
        let output_width = (input_width + 2 * pad_w - kernel_width) / stride_w + 1;

        // im2col transformation - convert convolution to matrix multiplication
        let col_buffer_size =
            in_channels * kernel_height * kernel_width * output_height * output_width;
        let mut col_buffer = vec![0.0f32; col_buffer_size];

        for batch in 0..batch_size {
            // im2col transformation for current batch
            im2col_cpu(
                &input[batch * in_channels * input_height * input_width..],
                in_channels,
                input_height,
                input_width,
                kernel_height,
                kernel_width,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                &mut col_buffer,
            )?;

            // Matrix multiplication: weight * col_buffer = output
            optimized_matmul(
                weight,
                &col_buffer,
                &mut output[batch * out_channels * output_height * output_width..],
                out_channels,
                output_height * output_width,
                in_channels * kernel_height * kernel_width,
                false,
                false,
            )?;
        }

        Ok(())
    }

    /// im2col CPU implementation
    fn im2col_cpu(
        data: &[f32],
        channels: usize,
        height: usize,
        width: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        col_buffer: &mut [f32],
    ) -> Result<()> {
        let output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
        let output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;

        col_buffer
            .par_chunks_mut(output_h * output_w)
            .enumerate()
            .for_each(|(c, chunk)| {
                let c_im = c / (kernel_h * kernel_w);
                let rest = c % (kernel_h * kernel_w);
                let kh = rest / kernel_w;
                let kw = rest % kernel_w;

                for (h_col, row) in chunk.chunks_mut(output_w).enumerate() {
                    for (w_col, out_val) in row.iter_mut().enumerate() {
                        let h_im = h_col * stride_h + kh;
                        let w_im = w_col * stride_w + kw;

                        if h_im >= pad_h
                            && w_im >= pad_w
                            && h_im < height + pad_h
                            && w_im < width + pad_w
                        {
                            let h_idx = h_im - pad_h;
                            let w_idx = w_im - pad_w;
                            if h_idx < height && w_idx < width {
                                *out_val = data[c_im * height * width + h_idx * width + w_idx];
                            }
                        }
                    }
                }
            });

        Ok(())
    }

    /// Optimized batch normalization
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
    ) -> Result<()> {
        if input.len() != batch_size * channels * spatial_size {
            return Err(TorshError::InvalidArgument(
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

    /// Optimized ReLU activation with in-place operation
    pub fn optimized_relu_inplace(data: &mut [f32]) {
        // Fallback to scalar implementation since SIMD needs separate input/output
        for val in data.iter_mut() {
            *val = val.max(0.0);
        }
    }

    /// Optimized softmax with numerical stability
    pub fn optimized_softmax(
        input: &[f32],
        output: &mut [f32],
        batch_size: usize,
        num_classes: usize,
    ) -> Result<()> {
        if input.len() != batch_size * num_classes || output.len() != input.len() {
            return Err(TorshError::InvalidArgument("Size mismatch".to_string()));
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

    /// Cache-optimized transpose operation
    pub fn optimized_transpose(
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        if input.len() != rows * cols || output.len() != rows * cols {
            return Err(TorshError::InvalidArgument("Size mismatch".to_string()));
        }

        const BLOCK_SIZE: usize = 32;

        for i in (0..rows).step_by(BLOCK_SIZE) {
            for j in (0..cols).step_by(BLOCK_SIZE) {
                let i_end = (i + BLOCK_SIZE).min(rows);
                let j_end = (j + BLOCK_SIZE).min(cols);

                for ii in i..i_end {
                    for jj in j..j_end {
                        output[jj * rows + ii] = input[ii * cols + jj];
                    }
                }
            }
        }

        Ok(())
    }

    /// Memory-efficient matrix multiplication with custom threading
    pub fn threaded_matmul(
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if a.len() != m * k || b.len() != k * n || result.len() != m * n {
            return Err(TorshError::ShapeMismatch {
                expected: vec![m * k, k * n, m * n],
                got: vec![a.len(), b.len(), result.len()],
            });
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
