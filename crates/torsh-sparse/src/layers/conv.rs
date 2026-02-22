//! Sparse convolution layers
//!
//! This module provides convolution operations optimized for sparse tensors,
//! including 1D convolution, 2D convolution, and graph convolution layers.

use crate::{CooTensor, CsrTensor, SparseTensor, TorshResult};

use torsh_core::{Shape, TorshError};
use torsh_tensor::{
    creation::{randn, zeros},
    Tensor,
};

/// Graph Convolutional Network (GCN) layer
///
/// Implements the GCN operation: H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
/// where A is the adjacency matrix, D is the degree matrix, H is node features, W is weights
pub struct GraphConvolution {
    /// Weight matrix for feature transformation
    weight: Tensor,
    /// Optional bias vector
    bias: Option<Tensor>,
    /// Input feature dimension
    in_features: usize,
    /// Output feature dimension
    out_features: usize,
    /// Whether to add self-loops to adjacency matrix
    add_self_loops: bool,
    /// Whether to normalize adjacency matrix
    normalize: bool,
}

impl GraphConvolution {
    /// Create a new graph convolution layer
    pub fn new(
        in_features: usize,
        out_features: usize,
        use_bias: bool,
        add_self_loops: bool,
        normalize: bool,
    ) -> TorshResult<Self> {
        // Initialize weight matrix with Xavier/Glorot initialization
        let _std_dev = (2.0 / (in_features + out_features) as f32).sqrt();
        let weight = randn::<f32>(&[in_features, out_features])?;

        // Initialize bias if requested
        let bias = if use_bias {
            Some(zeros::<f32>(&[out_features])?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
            add_self_loops,
            normalize,
        })
    }

    /// Forward pass through the graph convolution layer
    pub fn forward(&self, node_features: &Tensor, adjacency: &CsrTensor) -> TorshResult<Tensor> {
        // Validate input dimensions
        let feature_shape = node_features.shape();
        if feature_shape.ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "Node features must be 2D tensor (num_nodes x in_features)".to_string(),
            ));
        }

        let num_nodes = feature_shape.dims()[0];
        let input_features = feature_shape.dims()[1];

        if input_features != self.in_features {
            return Err(TorshError::InvalidArgument(format!(
                "Input features {} don't match layer input features {}",
                input_features, self.in_features
            )));
        }

        // Validate adjacency matrix
        let adj_shape = adjacency.shape();
        if adj_shape.dims() != [num_nodes, num_nodes] {
            return Err(TorshError::InvalidArgument(
                "Adjacency matrix must be square and match number of nodes".to_string(),
            ));
        }

        // Prepare adjacency matrix (add self-loops if requested)
        let adj_processed = if self.add_self_loops {
            self.add_self_loops_to_adjacency(adjacency)?
        } else {
            adjacency.clone()
        };

        // Normalize adjacency matrix if requested
        let adj_normalized = if self.normalize {
            self.normalize_adjacency(&adj_processed)?
        } else {
            adj_processed
        };

        // Apply linear transformation: H * W
        let transformed_features = zeros::<f32>(&[num_nodes, self.out_features])?;
        for i in 0..num_nodes {
            for j in 0..self.out_features {
                let mut sum = 0.0;
                for k in 0..self.in_features {
                    sum += node_features.get(&[i, k])? * self.weight.get(&[k, j])?;
                }
                transformed_features.set(&[i, j], sum)?;
            }
        }

        // Apply graph convolution: A_norm * (H * W)
        let output = zeros::<f32>(&[num_nodes, self.out_features])?;
        for i in 0..num_nodes {
            let (neighbors, weights) = adj_normalized.get_row(i)?;
            for j in 0..self.out_features {
                let mut sum = 0.0;
                for (&neighbor, &weight) in neighbors.iter().zip(weights.iter()) {
                    sum += weight * transformed_features.get(&[neighbor, j])?;
                }
                output.set(&[i, j], sum)?;
            }
        }

        // Add bias if present
        if let Some(ref bias) = self.bias {
            for i in 0..num_nodes {
                for j in 0..self.out_features {
                    let current = output.get(&[i, j])?;
                    output.set(&[i, j], current + bias.get(&[j])?)?;
                }
            }
        }

        Ok(output)
    }

    /// Add self-loops to adjacency matrix
    fn add_self_loops_to_adjacency(&self, adjacency: &CsrTensor) -> TorshResult<CsrTensor> {
        let coo = adjacency.to_coo()?;
        let mut triplets = coo.triplets();
        let num_nodes = adjacency.shape().dims()[0];

        // Add self-loops (diagonal entries with value 1.0)
        let mut self_loop_set = std::collections::HashSet::new();
        for (row, col, _) in &triplets {
            if row == col {
                self_loop_set.insert(*row);
            }
        }

        // Add missing self-loops
        for i in 0..num_nodes {
            if !self_loop_set.contains(&i) {
                triplets.push((i, i, 1.0));
            }
        }

        let (row_indices, col_indices, values): (Vec<_>, Vec<_>, Vec<_>) =
            triplets.into_iter().fold(
                (Vec::new(), Vec::new(), Vec::new()),
                |(mut rows, mut cols, mut vals), (r, c, v)| {
                    rows.push(r);
                    cols.push(c);
                    vals.push(v);
                    (rows, cols, vals)
                },
            );

        let new_coo = CooTensor::new(row_indices, col_indices, values, adjacency.shape().clone())?;
        CsrTensor::from_coo(&new_coo)
    }

    /// Normalize adjacency matrix using D^(-1/2) * A * D^(-1/2)
    fn normalize_adjacency(&self, adjacency: &CsrTensor) -> TorshResult<CsrTensor> {
        let num_nodes = adjacency.shape().dims()[0];

        // Calculate degree for each node
        let mut degrees = vec![0.0; num_nodes];
        let coo = adjacency.to_coo()?;
        let triplets = coo.triplets();

        for (row, _col, val) in &triplets {
            degrees[*row] += val;
        }

        // Calculate D^(-1/2)
        let inv_sqrt_degrees: Vec<f32> = degrees
            .iter()
            .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
            .collect();

        // Apply normalization: D^(-1/2) * A * D^(-1/2)
        let normalized_triplets: Vec<_> = triplets
            .into_iter()
            .map(|(row, col, val)| {
                let normalized_val = inv_sqrt_degrees[row] * val * inv_sqrt_degrees[col];
                (row, col, normalized_val)
            })
            .collect();

        let (row_indices, col_indices, values): (Vec<_>, Vec<_>, Vec<_>) =
            normalized_triplets.into_iter().fold(
                (Vec::new(), Vec::new(), Vec::new()),
                |(mut rows, mut cols, mut vals), (r, c, v)| {
                    rows.push(r);
                    cols.push(c);
                    vals.push(v);
                    (rows, cols, vals)
                },
            );

        let normalized_coo =
            CooTensor::new(row_indices, col_indices, values, adjacency.shape().clone())?;
        CsrTensor::from_coo(&normalized_coo)
    }

    /// Get the number of parameters
    pub fn num_parameters(&self) -> usize {
        let weight_params = self.in_features * self.out_features;
        let bias_params = self.bias.as_ref().map_or(0, |b| b.shape().numel());
        weight_params + bias_params
    }

    /// Get input feature dimension
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output feature dimension
    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

/// Sparse 2D Convolution layer
///
/// Implements 2D convolution operations optimized for sparse inputs and/or sparse kernels.
/// This can significantly reduce computation when either the input or kernel has many zeros.
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

        let stride = stride.unwrap_or((1, 1));
        let padding = padding.unwrap_or((0, 0));
        let dilation = dilation.unwrap_or((1, 1));

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
            values.push((rng.random::<f32>() * 2.0 - 1.0) * std_dev);
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
}

/// Sparse 1D Convolution layer
///
/// Implements 1D convolution operations optimized for sparse inputs and/or sparse kernels.
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

        let stride = stride.unwrap_or(1);
        let padding = padding.unwrap_or(0);
        let dilation = dilation.unwrap_or(1);

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
            values.push((rng.random::<f32>() * 2.0 - 1.0) * std_dev);
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
}
