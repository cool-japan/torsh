//! Sparse convolution layers
//!
//! This module provides sparse convolution implementations optimized for both sparse inputs
//! and sparse kernels. These layers can significantly reduce computational complexity when
//! dealing with sparse data, which is common in computer vision tasks after activation
//! functions like ReLU, or in specialized applications like medical imaging and satellite data.

use crate::{CooTensor, CsrTensor, SparseTensor, TorshResult};
use scirs2_core::random::{Random, Rng};
use std::collections::HashMap;
use torsh_core::{Shape, TorshError};
use torsh_tensor::{
    creation::{randn, zeros},
    Tensor,
};

/// Sparse 2D Convolution layer
///
/// Implements 2D convolution operations optimized for sparse inputs and/or sparse kernels.
/// This can significantly reduce computation when either the input or kernel has many zeros.
///
/// # Mathematical Formulation
/// For standard convolution: y[i,j] = Σ_m Σ_n w[m,n] * x[i+m, j+n]
/// For sparse convolution: Only compute terms where w[m,n] ≠ 0
///
/// # Use Cases
/// - Computer vision with sparse activations (post-ReLU)
/// - Medical imaging with sparse features
/// - Pruned neural networks
/// - Specialized domains with naturally sparse data
#[derive(Debug, Clone)]
pub struct SparseConv2d {
    /// Convolution kernel in sparse format (out_channels, in_channels, kernel_height, kernel_width)
    kernel: CsrTensor,
    /// Optional bias vector (out_channels,)
    bias: Option<Tensor>,
    /// Number of input channels
    in_channels: usize,
    /// Number of output channels
    out_channels: usize,
    /// Kernel size (height, width)
    kernel_size: (usize, usize),
    /// Stride (height, width)
    stride: (usize, usize),
    /// Padding (height, width)
    padding: (usize, usize),
    /// Dilation (height, width)
    dilation: (usize, usize),
}

impl SparseConv2d {
    /// Create a new sparse 2D convolution layer
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Convolution kernel size (height, width)
    /// * `stride` - Stride for convolution (default: (1, 1))
    /// * `padding` - Padding for convolution (default: (0, 0))
    /// * `dilation` - Dilation for convolution (default: (1, 1))
    /// * `sparsity` - Kernel sparsity level (0.0 = dense, 1.0 = fully sparse)
    /// * `use_bias` - Whether to include learnable bias
    ///
    /// # Returns
    /// * `TorshResult<Self>` - New sparse 2D convolution layer or error
    ///
    /// # Example
    /// ```rust
    /// use torsh_sparse::nn::convolution::SparseConv2d;
    ///
    /// // Create sparse conv: 3->64 channels, 3x3 kernel, 90% sparse
    /// let conv = SparseConv2d::new(
    ///     3, 64, (3, 3),
    ///     Some((1, 1)), Some((1, 1)), None,
    ///     0.9, true
    /// ).unwrap();
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        dilation: Option<(usize, usize)>,
        sparsity: f32,
        use_bias: bool,
    ) -> TorshResult<Self> {
        if !(0.0..=1.0).contains(&sparsity) {
            return Err(TorshError::InvalidArgument(
                "Sparsity must be between 0.0 and 1.0".to_string(),
            ));
        }

        if kernel_size.0 == 0 || kernel_size.1 == 0 {
            return Err(TorshError::InvalidArgument(
                "Kernel size must be greater than 0".to_string(),
            ));
        }

        let stride = stride.unwrap_or((1, 1));
        let padding = padding.unwrap_or((0, 0));
        let dilation = dilation.unwrap_or((1, 1));

        if stride.0 == 0 || stride.1 == 0 {
            return Err(TorshError::InvalidArgument(
                "Stride must be greater than 0".to_string(),
            ));
        }

        if dilation.0 == 0 || dilation.1 == 0 {
            return Err(TorshError::InvalidArgument(
                "Dilation must be greater than 0".to_string(),
            ));
        }

        // Generate sparse kernel
        let kernel =
            Self::generate_sparse_kernel(out_channels, in_channels, kernel_size, sparsity)?;

        // Generate bias if requested
        let bias = if use_bias {
            Some(zeros::<f32>(&[out_channels])?)
        } else {
            None
        };

        Ok(Self {
            kernel,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
        })
    }

    /// Forward pass for sparse convolution
    ///
    /// # Arguments
    /// * `input` - Input tensor (batch_size, in_channels, height, width)
    ///
    /// # Returns
    /// * `TorshResult<Tensor>` - Output tensor (batch_size, out_channels, out_height, out_width)
    pub fn forward(&self, input: &Tensor) -> TorshResult<Tensor> {
        // Validate input shape: (batch_size, in_channels, height, width)
        let input_shape = input.shape();
        if input_shape.ndim() != 4 {
            return Err(TorshError::InvalidArgument(
                "Input must be 4D tensor (batch_size, in_channels, height, width)".to_string(),
            ));
        }

        let batch_size = input_shape.dims()[0];
        let input_channels = input_shape.dims()[1];
        let input_height = input_shape.dims()[2];
        let input_width = input_shape.dims()[3];

        if input_channels != self.in_channels {
            return Err(TorshError::InvalidArgument(format!(
                "Input channels {} don't match layer input channels {}",
                input_channels, self.in_channels
            )));
        }

        // Calculate output dimensions
        let output_height =
            (input_height + 2 * self.padding.0 - self.dilation.0 * (self.kernel_size.0 - 1) - 1)
                / self.stride.0
                + 1;
        let output_width =
            (input_width + 2 * self.padding.1 - self.dilation.1 * (self.kernel_size.1 - 1) - 1)
                / self.stride.1
                + 1;

        let mut output =
            zeros::<f32>(&[batch_size, self.out_channels, output_height, output_width])?;

        // Perform sparse convolution for each batch item
        for b in 0..batch_size {
            self.conv2d_single(
                input,
                &mut output,
                b,
                input_height,
                input_width,
                output_height,
                output_width,
            )?;
        }

        // Add bias if present
        if let Some(ref bias) = self.bias {
            for b in 0..batch_size {
                for c in 0..self.out_channels {
                    let bias_val = bias.get(&[c])?;
                    for h in 0..output_height {
                        for w in 0..output_width {
                            let current = output.get(&[b, c, h, w])?;
                            output.set(&[b, c, h, w], current + bias_val)?;
                        }
                    }
                }
            }
        }

        Ok(output)
    }

    /// Perform convolution for a single batch item
    #[allow(clippy::too_many_arguments)]
    fn conv2d_single(
        &self,
        input: &Tensor,
        output: &mut Tensor,
        batch_idx: usize,
        input_height: usize,
        input_width: usize,
        output_height: usize,
        output_width: usize,
    ) -> TorshResult<()> {
        // Get sparse kernel triplets (we need to interpret the flat CSR as 4D)
        let kernel_coo = self.kernel.to_coo()?;
        let kernel_triplets = kernel_coo.triplets();

        // Perform sparse convolution by iterating over non-zero kernel elements
        for (kernel_flat_idx, _, kernel_val) in kernel_triplets {
            // Convert flat index back to 4D coordinates
            let (out_c, in_c, kh, kw) = self.flat_to_4d_kernel(kernel_flat_idx);

            // For each output position
            for out_h in 0..output_height {
                for out_w in 0..output_width {
                    // Calculate input position
                    let in_h = out_h * self.stride.0 + kh * self.dilation.0;
                    let in_w = out_w * self.stride.1 + kw * self.dilation.1;

                    // Apply padding offset
                    if in_h >= self.padding.0 && in_w >= self.padding.1 {
                        let padded_in_h = in_h - self.padding.0;
                        let padded_in_w = in_w - self.padding.1;

                        // Check bounds
                        if padded_in_h < input_height && padded_in_w < input_width {
                            let input_val =
                                input.get(&[batch_idx, in_c, padded_in_h, padded_in_w])?;
                            let current_out = output.get(&[batch_idx, out_c, out_h, out_w])?;
                            output.set(
                                &[batch_idx, out_c, out_h, out_w],
                                current_out + kernel_val * input_val,
                            )?;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Convert flat kernel index to 4D coordinates (out_c, in_c, kh, kw)
    fn flat_to_4d_kernel(&self, flat_idx: usize) -> (usize, usize, usize, usize) {
        let kernel_size_total = self.kernel_size.0 * self.kernel_size.1;
        let channel_size = self.in_channels * kernel_size_total;

        let out_c = flat_idx / channel_size;
        let remaining = flat_idx % channel_size;
        let in_c = remaining / kernel_size_total;
        let remaining = remaining % kernel_size_total;
        let kh = remaining / self.kernel_size.1;
        let kw = remaining % self.kernel_size.1;

        (out_c, in_c, kh, kw)
    }

    /// Convert 4D kernel coordinates to flat index
    #[allow(dead_code)]
    fn kernel_4d_to_flat(&self, out_c: usize, in_c: usize, kh: usize, kw: usize) -> usize {
        let kernel_size_total = self.kernel_size.0 * self.kernel_size.1;
        let channel_size = self.in_channels * kernel_size_total;

        out_c * channel_size + in_c * kernel_size_total + kh * self.kernel_size.1 + kw
    }

    /// Generate sparse convolution kernel
    fn generate_sparse_kernel(
        out_channels: usize,
        in_channels: usize,
        kernel_size: (usize, usize),
        sparsity: f32,
    ) -> TorshResult<CsrTensor> {
        let total_elements = out_channels * in_channels * kernel_size.0 * kernel_size.1;
        let nnz = ((total_elements as f32) * (1.0 - sparsity)) as usize;

        let mut row_indices = Vec::with_capacity(nnz);
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        // Generate random sparse pattern
        let mut positions = std::collections::HashSet::new();
        while positions.len() < nnz {
            let mut rng = scirs2_core::random::thread_rng();
            let flat_idx = rng.gen_range(0..total_elements);
            positions.insert(flat_idx);
        }

        // For CSR representation, we'll use a 2D matrix where:
        // - rows represent output channels
        // - columns represent (in_channels * kernel_height * kernel_width)
        let kernel_size_total = kernel_size.0 * kernel_size.1;
        let channel_size = in_channels * kernel_size_total;

        for flat_idx in positions {
            let out_c = flat_idx / channel_size;
            let col_idx = flat_idx % channel_size; // This represents (in_c, kh, kw) flattened

            row_indices.push(out_c);
            col_indices.push(col_idx);

            // He initialization for convolution layers
            let fan_in = in_channels * kernel_size.0 * kernel_size.1;
            let std_dev = (2.0 / fan_in as f32).sqrt();
            let mut rng = scirs2_core::random::thread_rng();
            values.push((rng.gen::<f32>() * 2.0 - 1.0) * std_dev);
        }

        let shape = Shape::new(vec![out_channels, channel_size]);
        let coo = CooTensor::new(row_indices, col_indices, values, shape)?;
        CsrTensor::from_coo(&coo)
    }

    /// Get the number of parameters
    pub fn num_parameters(&self) -> usize {
        let kernel_params = self.kernel.nnz();
        let bias_params = self.bias.as_ref().map_or(0, |b| b.shape().numel());
        kernel_params + bias_params
    }

    /// Get kernel sparsity
    pub fn kernel_sparsity(&self) -> f32 {
        let total_elements =
            self.out_channels * self.in_channels * self.kernel_size.0 * self.kernel_size.1;
        let nnz = self.kernel.nnz();
        1.0 - (nnz as f32 / total_elements as f32)
    }

    /// Get input channels
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Get output channels
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Get kernel size
    pub fn kernel_size(&self) -> (usize, usize) {
        self.kernel_size
    }

    /// Get stride
    pub fn stride(&self) -> (usize, usize) {
        self.stride
    }

    /// Get padding
    pub fn padding(&self) -> (usize, usize) {
        self.padding
    }

    /// Get dilation
    pub fn dilation(&self) -> (usize, usize) {
        self.dilation
    }
}

/// Sparse 1D Convolution layer
///
/// Implements 1D convolution operations optimized for sparse inputs and/or sparse kernels.
/// This is particularly useful for time series data, audio processing, and NLP applications
/// where sparsity can naturally occur or be induced through pruning.
#[derive(Debug, Clone)]
pub struct SparseConv1d {
    /// Convolution kernel in sparse format (out_channels, in_channels, kernel_length)
    kernel: CsrTensor,
    /// Optional bias vector (out_channels,)
    bias: Option<Tensor>,
    /// Number of input channels
    in_channels: usize,
    /// Number of output channels
    out_channels: usize,
    /// Kernel size
    kernel_size: usize,
    /// Stride
    stride: usize,
    /// Padding
    padding: usize,
    /// Dilation
    dilation: usize,
}

impl SparseConv1d {
    /// Create a new sparse 1D convolution layer
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Convolution kernel size
    /// * `stride` - Stride for convolution (default: 1)
    /// * `padding` - Padding for convolution (default: 0)
    /// * `dilation` - Dilation for convolution (default: 1)
    /// * `sparsity` - Kernel sparsity level (0.0 = dense, 1.0 = fully sparse)
    /// * `use_bias` - Whether to include learnable bias
    ///
    /// # Returns
    /// * `TorshResult<Self>` - New sparse 1D convolution layer or error
    ///
    /// # Example
    /// ```rust
    /// use torsh_sparse::nn::convolution::SparseConv1d;
    ///
    /// // Create sparse 1D conv: 16->32 channels, kernel size 3, 80% sparse
    /// let conv = SparseConv1d::new(16, 32, 3, None, None, None, 0.8, true).unwrap();
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
        dilation: Option<usize>,
        sparsity: f32,
        use_bias: bool,
    ) -> TorshResult<Self> {
        if !(0.0..=1.0).contains(&sparsity) {
            return Err(TorshError::InvalidArgument(
                "Sparsity must be between 0.0 and 1.0".to_string(),
            ));
        }

        if kernel_size == 0 {
            return Err(TorshError::InvalidArgument(
                "Kernel size must be greater than 0".to_string(),
            ));
        }

        let stride = stride.unwrap_or(1);
        let padding = padding.unwrap_or(0);
        let dilation = dilation.unwrap_or(1);

        if stride == 0 {
            return Err(TorshError::InvalidArgument(
                "Stride must be greater than 0".to_string(),
            ));
        }

        if dilation == 0 {
            return Err(TorshError::InvalidArgument(
                "Dilation must be greater than 0".to_string(),
            ));
        }

        // Generate sparse kernel
        let kernel =
            Self::generate_sparse_kernel_1d(out_channels, in_channels, kernel_size, sparsity)?;

        // Generate bias if requested
        let bias = if use_bias {
            Some(zeros::<f32>(&[out_channels])?)
        } else {
            None
        };

        Ok(Self {
            kernel,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
        })
    }

    /// Forward pass for sparse 1D convolution
    ///
    /// # Arguments
    /// * `input` - Input tensor (batch_size, in_channels, length)
    ///
    /// # Returns
    /// * `TorshResult<Tensor>` - Output tensor (batch_size, out_channels, out_length)
    pub fn forward(&self, input: &Tensor) -> TorshResult<Tensor> {
        // Validate input shape: (batch_size, in_channels, length)
        let input_shape = input.shape();
        if input_shape.ndim() != 3 {
            return Err(TorshError::InvalidArgument(
                "Input must be 3D tensor (batch_size, in_channels, length)".to_string(),
            ));
        }

        let batch_size = input_shape.dims()[0];
        let input_channels = input_shape.dims()[1];
        let input_length = input_shape.dims()[2];

        if input_channels != self.in_channels {
            return Err(TorshError::InvalidArgument(format!(
                "Input channels {} don't match layer input channels {}",
                input_channels, self.in_channels
            )));
        }

        // Calculate output length
        let output_length =
            (input_length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
                / self.stride
                + 1;

        let mut output = zeros::<f32>(&[batch_size, self.out_channels, output_length])?;

        // Perform sparse convolution for each batch item
        for b in 0..batch_size {
            self.conv1d_single(input, &mut output, b, input_length, output_length)?;
        }

        // Add bias if present
        if let Some(ref bias) = self.bias {
            for b in 0..batch_size {
                for c in 0..self.out_channels {
                    let bias_val = bias.get(&[c])?;
                    for l in 0..output_length {
                        let current = output.get(&[b, c, l])?;
                        output.set(&[b, c, l], current + bias_val)?;
                    }
                }
            }
        }

        Ok(output)
    }

    /// Perform 1D convolution for a single batch item
    fn conv1d_single(
        &self,
        input: &Tensor,
        output: &mut Tensor,
        batch_idx: usize,
        input_length: usize,
        output_length: usize,
    ) -> TorshResult<()> {
        // Get sparse kernel triplets
        let kernel_coo = self.kernel.to_coo()?;
        let kernel_triplets = kernel_coo.triplets();

        // Perform sparse convolution by iterating over non-zero kernel elements
        for (kernel_flat_idx, _, kernel_val) in kernel_triplets {
            // Convert flat index back to 3D coordinates
            let (out_c, in_c, k_pos) = self.flat_to_3d_kernel(kernel_flat_idx);

            // For each output position
            for out_pos in 0..output_length {
                // Calculate input position
                let in_pos = out_pos * self.stride + k_pos * self.dilation;

                // Apply padding offset
                if in_pos >= self.padding {
                    let padded_in_pos = in_pos - self.padding;

                    // Check bounds
                    if padded_in_pos < input_length {
                        let input_val = input.get(&[batch_idx, in_c, padded_in_pos])?;
                        let current_out = output.get(&[batch_idx, out_c, out_pos])?;
                        output.set(
                            &[batch_idx, out_c, out_pos],
                            current_out + kernel_val * input_val,
                        )?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Convert flat kernel index to 3D coordinates (out_c, in_c, k_pos)
    fn flat_to_3d_kernel(&self, flat_idx: usize) -> (usize, usize, usize) {
        let channel_size = self.in_channels * self.kernel_size;

        let out_c = flat_idx / channel_size;
        let remaining = flat_idx % channel_size;
        let in_c = remaining / self.kernel_size;
        let k_pos = remaining % self.kernel_size;

        (out_c, in_c, k_pos)
    }

    /// Generate sparse 1D convolution kernel
    fn generate_sparse_kernel_1d(
        out_channels: usize,
        in_channels: usize,
        kernel_size: usize,
        sparsity: f32,
    ) -> TorshResult<CsrTensor> {
        let total_elements = out_channels * in_channels * kernel_size;
        let nnz = ((total_elements as f32) * (1.0 - sparsity)) as usize;

        let mut row_indices = Vec::with_capacity(nnz);
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        // Generate random sparse pattern
        let mut positions = std::collections::HashSet::new();
        while positions.len() < nnz {
            let mut rng = scirs2_core::random::thread_rng();
            let flat_idx = rng.gen_range(0..total_elements);
            positions.insert(flat_idx);
        }

        // For CSR representation, we'll use a 2D matrix where:
        // - rows represent output channels
        // - columns represent (in_channels * kernel_size)
        let channel_size = in_channels * kernel_size;

        for flat_idx in positions {
            let out_c = flat_idx / channel_size;
            let col_idx = flat_idx % channel_size; // This represents (in_c, k_pos) flattened

            row_indices.push(out_c);
            col_indices.push(col_idx);

            // He initialization for convolution layers
            let fan_in = in_channels * kernel_size;
            let std_dev = (2.0 / fan_in as f32).sqrt();
            let mut rng = scirs2_core::random::thread_rng();
            values.push((rng.gen::<f32>() * 2.0 - 1.0) * std_dev);
        }

        let shape = Shape::new(vec![out_channels, channel_size]);
        let coo = CooTensor::new(row_indices, col_indices, values, shape)?;
        CsrTensor::from_coo(&coo)
    }

    /// Get the number of parameters
    pub fn num_parameters(&self) -> usize {
        let kernel_params = self.kernel.nnz();
        let bias_params = self.bias.as_ref().map_or(0, |b| b.shape().numel());
        kernel_params + bias_params
    }

    /// Get kernel sparsity
    pub fn kernel_sparsity(&self) -> f32 {
        let total_elements = self.out_channels * self.in_channels * self.kernel_size;
        let nnz = self.kernel.nnz();
        1.0 - (nnz as f32 / total_elements as f32)
    }

    /// Get input channels
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Get output channels
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Get kernel size
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// Get stride
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Get padding
    pub fn padding(&self) -> usize {
        self.padding
    }

    /// Get dilation
    pub fn dilation(&self) -> usize {
        self.dilation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::ones;

    #[test]
    fn test_sparse_conv2d_creation() {
        let conv =
            SparseConv2d::new(3, 16, (3, 3), Some((1, 1)), Some((1, 1)), None, 0.5, true).unwrap();
        assert_eq!(conv.in_channels(), 3);
        assert_eq!(conv.out_channels(), 16);
        assert_eq!(conv.kernel_size(), (3, 3));
        assert!(conv.num_parameters() > 0);
    }

    #[test]
    fn test_sparse_conv2d_forward() {
        let conv = SparseConv2d::new(2, 4, (3, 3), None, Some((1, 1)), None, 0.3, false).unwrap();
        let input = ones::<f32>(&[1, 2, 5, 5]).unwrap();
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 4, 5, 5]); // Same size due to padding
    }

    #[test]
    fn test_sparse_conv1d_creation() {
        let conv = SparseConv1d::new(8, 16, 5, None, None, None, 0.7, true).unwrap();
        assert_eq!(conv.in_channels(), 8);
        assert_eq!(conv.out_channels(), 16);
        assert_eq!(conv.kernel_size(), 5);
        assert!(conv.num_parameters() > 0);
    }

    #[test]
    fn test_sparse_conv1d_forward() {
        let conv = SparseConv1d::new(4, 8, 3, None, Some(1), None, 0.4, false).unwrap();
        let input = ones::<f32>(&[2, 4, 10]).unwrap();
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 8, 10]); // Same size due to padding
    }

    #[test]
    fn test_output_size_calculation() {
        // Test 2D convolution output size
        let conv = SparseConv2d::new(1, 1, (3, 3), Some((2, 2)), None, None, 0.0, false).unwrap();
        let input = ones::<f32>(&[1, 1, 8, 8]).unwrap();
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 1, 3, 3]); // (8 - 3 + 1) / 2 = 3

        // Test 1D convolution output size
        let conv1d = SparseConv1d::new(1, 1, 3, Some(2), None, None, 0.0, false).unwrap();
        let input1d = ones::<f32>(&[1, 1, 10]).unwrap();
        let output1d = conv1d.forward(&input1d).unwrap();
        assert_eq!(output1d.shape().dims(), &[1, 1, 4]); // (10 - 3 + 1) / 2 = 4
    }

    #[test]
    fn test_invalid_parameters() {
        assert!(SparseConv2d::new(1, 1, (0, 3), None, None, None, 0.5, false).is_err());
        assert!(SparseConv2d::new(1, 1, (3, 3), Some((0, 1)), None, None, 0.5, false).is_err());
        assert!(SparseConv2d::new(1, 1, (3, 3), None, None, None, 1.5, false).is_err());

        assert!(SparseConv1d::new(1, 1, 0, None, None, None, 0.5, false).is_err());
        assert!(SparseConv1d::new(1, 1, 3, Some(0), None, None, 0.5, false).is_err());
        assert!(SparseConv1d::new(1, 1, 3, None, None, None, -0.1, false).is_err());
    }

    #[test]
    fn test_dimension_validation() {
        let conv = SparseConv2d::new(3, 16, (3, 3), None, None, None, 0.5, false).unwrap();
        let wrong_input = ones::<f32>(&[1, 2, 5, 5]).unwrap(); // Wrong channels
        assert!(conv.forward(&wrong_input).is_err());

        let conv1d = SparseConv1d::new(4, 8, 3, None, None, None, 0.5, false).unwrap();
        let wrong_input1d = ones::<f32>(&[1, 3, 10]).unwrap(); // Wrong channels
        assert!(conv1d.forward(&wrong_input1d).is_err());
    }

    #[test]
    fn test_sparsity_measurement() {
        let conv = SparseConv2d::new(2, 4, (3, 3), None, None, None, 0.8, false).unwrap();
        let sparsity = conv.kernel_sparsity();
        assert!(sparsity >= 0.7 && sparsity <= 0.9); // Should be around 0.8

        let conv1d = SparseConv1d::new(2, 4, 5, None, None, None, 0.6, false).unwrap();
        let sparsity1d = conv1d.kernel_sparsity();
        assert!(sparsity1d >= 0.5 && sparsity1d <= 0.7); // Should be around 0.6
    }

    #[test]
    fn test_bias_addition() {
        let conv = SparseConv2d::new(1, 2, (1, 1), None, None, None, 0.0, true).unwrap();
        let input = ones::<f32>(&[1, 1, 3, 3]).unwrap();
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 2, 3, 3]);

        let conv1d = SparseConv1d::new(1, 2, 1, None, None, None, 0.0, true).unwrap();
        let input1d = ones::<f32>(&[1, 1, 5]).unwrap();
        let output1d = conv1d.forward(&input1d).unwrap();
        assert_eq!(output1d.shape().dims(), &[1, 2, 5]);
    }
}
