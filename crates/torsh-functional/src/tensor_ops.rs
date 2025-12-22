///! Additional tensor operations for PyTorch compatibility
///!
///! This module provides commonly used PyTorch functional operations including
///! one-hot encoding, linear transformations, and utility functions.
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Creates a one-hot encoded tensor from class indices
///
/// **PyTorch Equivalent:** `torch.nn.functional.one_hot(tensor, num_classes=-1)`
///
/// # Arguments
/// * `tensor` - 1D tensor of class indices (integers)
/// * `num_classes` - Number of classes. If -1, uses max(tensor) + 1
///
/// # Returns
/// 2D tensor of shape `[N, num_classes]` where N is the length of input tensor
///
/// # Example
/// ```ignore
/// use torsh_functional::one_hot;
/// let indices = Tensor::from_vec(vec![0.0, 1.0, 2.0, 0.0], &[4]).unwrap();
/// let encoded = one_hot(&indices, 3).unwrap();
/// // Result shape: [4, 3]
/// // [[1, 0, 0],
/// //  [0, 1, 0],
/// //  [0, 0, 1],
/// //  [1, 0, 0]]
/// ```
pub fn one_hot(tensor: &Tensor, num_classes: i32) -> TorshResult<Tensor> {
    // Validate input is 1D
    if tensor.shape().ndim() != 1 {
        return Err(TorshError::InvalidArgument(
            "one_hot expects 1D tensor input".to_string(),
        ));
    }

    let data = tensor.data()?;
    let n = data.len();

    // Determine number of classes
    let nc = if num_classes < 0 {
        // Find max value and add 1
        let max_val = data.iter().fold(0.0_f32, |a, &b| a.max(b));
        (max_val as usize) + 1
    } else {
        num_classes as usize
    };

    // Create one-hot encoded data
    let mut one_hot_data = vec![0.0_f32; n * nc];
    for (i, &class_idx) in data.iter().enumerate() {
        let class = class_idx as usize;
        if class < nc {
            one_hot_data[i * nc + class] = 1.0;
        }
    }

    Tensor::from_data(one_hot_data, vec![n, nc], tensor.device())
}

/// Applies a linear transformation to incoming data: y = xW^T + b
///
/// **PyTorch Equivalent:** `torch.nn.functional.linear(input, weight, bias=None)`
///
/// # Arguments
/// * `input` - Input tensor of shape `[*, in_features]`
/// * `weight` - Weight matrix of shape `[out_features, in_features]`
/// * `bias` - Optional bias vector of shape `[out_features]`
///
/// # Returns
/// Output tensor of shape `[*, out_features]`
///
/// # Example
/// ```ignore
/// use torsh_functional::linear;
/// let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();
/// let weight = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]).unwrap();
/// let bias = Tensor::from_vec(vec![0.1, 0.2], &[2]).unwrap();
/// let output = linear(&input, &weight, Some(&bias)).unwrap();
/// // Output shape: [1, 2]
/// ```
pub fn linear(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> TorshResult<Tensor> {
    // Validate dimensions
    if weight.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument(
            "Weight must be a 2D tensor".to_string(),
        ));
    }

    // Compute matrix multiplication: input @ weight^T
    let output = input.matmul(&weight.transpose(-1, -2)?)?;

    // Add bias if provided
    if let Some(b) = bias {
        // Validate bias shape
        if b.shape().ndim() != 1 {
            return Err(TorshError::InvalidArgument(
                "Bias must be a 1D tensor".to_string(),
            ));
        }
        output.add(b)
    } else {
        Ok(output)
    }
}

/// Computes pairwise distance between vectors
///
/// **PyTorch Equivalent:** `torch.nn.functional.pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False)`
///
/// # Arguments
/// * `x1` - First input tensor of shape `[N, D]`
/// * `x2` - Second input tensor of shape `[N, D]`
/// * `p` - Norm degree (default: 2.0 for Euclidean distance)
/// * `eps` - Small value to avoid division by zero
///
/// # Returns
/// Tensor of shape `[N]` containing pairwise distances
///
/// # Mathematical Definition
/// ```text
/// dist(x1, x2) = ||x1 - x2||_p = (Σ|x1_i - x2_i|^p)^(1/p)
/// ```
pub fn pairwise_distance(x1: &Tensor, x2: &Tensor, p: f32, eps: f32) -> TorshResult<Tensor> {
    // Validate shapes match
    if x1.shape().dims() != x2.shape().dims() {
        return Err(TorshError::InvalidArgument(
            "Input tensors must have the same shape".to_string(),
        ));
    }

    // Compute difference
    let diff = x1.sub(x2)?;

    // Compute p-norm
    if (p - 2.0).abs() < eps {
        // L2 norm (Euclidean distance): sqrt(sum(diff^2))
        let squared = diff.pow_scalar(2.0)?;
        let sum = squared.sum_dim(&[-1], false)?;
        sum.sqrt()
    } else if (p - 1.0).abs() < eps {
        // L1 norm (Manhattan distance): sum(|diff|)
        let abs_diff = diff.abs()?;
        abs_diff.sum_dim(&[-1], false)
    } else {
        // General Lp norm
        let abs_diff = diff.abs()?;
        let powered = abs_diff.pow_scalar(p)?;
        let sum = powered.sum_dim(&[-1], false)?;
        sum.pow_scalar(1.0 / p)
    }
}

/// Computes cosine similarity between vectors
///
/// **PyTorch Equivalent:** `torch.nn.functional.cosine_similarity(x1, x2, dim=1, eps=1e-8)`
///
/// # Arguments
/// * `x1` - First input tensor
/// * `x2` - Second input tensor (same shape as x1)
/// * `dim` - Dimension along which to compute cosine similarity
/// * `eps` - Small value to avoid division by zero
///
/// # Returns
/// Tensor containing cosine similarity values
///
/// # Mathematical Definition
/// ```text
/// cos_sim(x1, x2) = (x1 · x2) / (||x1|| * ||x2||)
/// ```
///
/// # Example
/// ```ignore
/// use torsh_functional::cosine_similarity;
/// let x1 = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap();
/// let x2 = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[1, 3]).unwrap();
/// let similarity = cosine_similarity(&x1, &x2, 1, 1e-8).unwrap();
/// ```
pub fn cosine_similarity(x1: &Tensor, x2: &Tensor, dim: i32, eps: f32) -> TorshResult<Tensor> {
    // Validate shapes match
    if x1.shape().dims() != x2.shape().dims() {
        return Err(TorshError::InvalidArgument(
            "Input tensors must have the same shape".to_string(),
        ));
    }

    // Compute dot product
    let dot_product = x1.mul(x2)?.sum_dim(&[dim], false)?;

    // Compute norms
    let x1_norm = x1.pow_scalar(2.0)?.sum_dim(&[dim], false)?.sqrt()?;
    let x2_norm = x2.pow_scalar(2.0)?.sum_dim(&[dim], false)?.sqrt()?;

    // Compute denominator with epsilon for numerical stability
    let denominator = x1_norm.mul(&x2_norm)?.clamp(eps, f32::MAX)?;

    // Cosine similarity = dot_product / denominator
    dot_product.div(&denominator)
}

/// Looks up embeddings from an embedding weight matrix
///
/// **PyTorch Equivalent:** `torch.nn.functional.embedding(input, weight, padding_idx=None, ...)`
///
/// # Arguments
/// * `weight` - Embedding weight matrix of shape `[num_embeddings, embedding_dim]`
/// * `indices` - Tensor of indices to look up (any shape)
///
/// # Returns
/// Tensor of shape `[*indices.shape, embedding_dim]` containing the embeddings
///
/// # Example
/// ```ignore
/// use torsh_functional::embedding;
/// let weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
/// let indices = Tensor::from_vec(vec![0.0, 1.0, 0.0], &[3]).unwrap();
/// let embedded = embedding(&weight, &indices).unwrap();
/// // Output shape: [3, 3]
/// ```
pub fn embedding(weight: &Tensor, indices: &Tensor) -> TorshResult<Tensor> {
    // Validate weight is 2D
    if weight.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument(
            "Weight must be a 2D tensor [num_embeddings, embedding_dim]".to_string(),
        ));
    }

    let num_embeddings = weight.shape().dims()[0];
    let embedding_dim = weight.shape().dims()[1];

    // Get indices data
    let indices_data = indices.data()?;
    let indices_shape_binding = indices.shape();
    let indices_shape = indices_shape_binding.dims();

    // Calculate output shape: [*indices_shape, embedding_dim]
    let mut output_shape = indices_shape.to_vec();
    output_shape.push(embedding_dim);

    // Get weight data
    let weight_data = weight.data()?;

    // Create output data by looking up embeddings
    let mut output_data = Vec::with_capacity(indices_data.len() * embedding_dim);

    for &idx in indices_data.iter() {
        let idx_usize = idx as usize;
        if idx_usize >= num_embeddings {
            return Err(TorshError::InvalidArgument(format!(
                "Index {} out of bounds for embedding with {} entries",
                idx_usize, num_embeddings
            )));
        }

        // Copy the embedding vector
        let start = idx_usize * embedding_dim;
        let end = start + embedding_dim;
        output_data.extend_from_slice(&weight_data[start..end]);
    }

    Tensor::from_data(output_data, output_shape, weight.device())
}

/// Rearranges elements in a tensor for upsampling (pixel shuffle)
///
/// **PyTorch Equivalent:** `torch.nn.functional.pixel_shuffle(input, upscale_factor)`
///
/// # Arguments
/// * `input` - Input tensor of shape `[B, C*r^2, H, W]` where r is upscale factor
/// * `upscale_factor` - Factor to increase spatial resolution by
///
/// # Returns
/// Tensor of shape `[B, C, H*r, W*r]`
///
/// # Description
/// This is commonly used in super-resolution networks. It rearranges depth
/// (channels) into spatial dimensions.
///
/// # Example
/// ```ignore
/// use torsh_functional::pixel_shuffle;
/// let input = Tensor::zeros(&[1, 4, 2, 2]).unwrap();  // C=4, r=2, so output C=1
/// let output = pixel_shuffle(&input, 2).unwrap();
/// // Output shape: [1, 1, 4, 4]
/// ```
pub fn pixel_shuffle(input: &Tensor, upscale_factor: usize) -> TorshResult<Tensor> {
    // Validate input is 4D [B, C, H, W]
    if input.shape().ndim() != 4 {
        return Err(TorshError::InvalidArgument(
            "pixel_shuffle expects 4D input [B, C, H, W]".to_string(),
        ));
    }

    let shape_binding = input.shape();
    let shape = shape_binding.dims();
    let batch_size = shape[0];
    let channels = shape[1];
    let height = shape[2];
    let width = shape[3];

    let r = upscale_factor;
    let r_squared = r * r;

    // Validate channels is divisible by r^2
    if channels % r_squared != 0 {
        return Err(TorshError::InvalidArgument(format!(
            "Channels {} must be divisible by upscale_factor^2 = {}",
            channels, r_squared
        )));
    }

    let output_channels = channels / r_squared;
    let output_height = height * r;
    let output_width = width * r;

    // Reshape: [B, C, H, W] -> [B, C/(r^2), r, r, H, W]
    // Then permute: [B, C/(r^2), r, r, H, W] -> [B, C/(r^2), H, r, W, r]
    // Then reshape: [B, C/(r^2), H, r, W, r] -> [B, C/(r^2), H*r, W*r]

    // For simplicity, use a direct implementation
    let data = input.data()?;
    let mut output_data =
        vec![0.0_f32; batch_size * output_channels * output_height * output_width];

    for b in 0..batch_size {
        for c in 0..output_channels {
            for h in 0..height {
                for w in 0..width {
                    for r_h in 0..r {
                        for r_w in 0..r {
                            let input_c = c * r_squared + r_h * r + r_w;
                            let input_idx = ((b * channels + input_c) * height + h) * width + w;
                            let output_h = h * r + r_h;
                            let output_w = w * r + r_w;
                            let output_idx = ((b * output_channels + c) * output_height + output_h)
                                * output_width
                                + output_w;
                            output_data[output_idx] = data[input_idx];
                        }
                    }
                }
            }
        }
    }

    Tensor::from_data(
        output_data,
        vec![batch_size, output_channels, output_height, output_width],
        input.device(),
    )
}

/// Rearranges elements in a tensor for downsampling (reverse of pixel shuffle)
///
/// **PyTorch Equivalent:** `torch.nn.functional.pixel_unshuffle(input, downscale_factor)`
///
/// # Arguments
/// * `input` - Input tensor of shape `[B, C, H*r, W*r]` where r is downscale factor
/// * `downscale_factor` - Factor to decrease spatial resolution by
///
/// # Returns
/// Tensor of shape `[B, C*r^2, H, W]`
///
/// # Description
/// This is the reverse of pixel_shuffle. It rearranges spatial dimensions into depth (channels).
///
/// # Example
/// ```ignore
/// use torsh_functional::pixel_unshuffle;
/// let input = Tensor::zeros(&[1, 1, 4, 4]).unwrap();
/// let output = pixel_unshuffle(&input, 2).unwrap();
/// // Output shape: [1, 4, 2, 2]
/// ```
pub fn pixel_unshuffle(input: &Tensor, downscale_factor: usize) -> TorshResult<Tensor> {
    // Validate input is 4D [B, C, H, W]
    if input.shape().ndim() != 4 {
        return Err(TorshError::InvalidArgument(
            "pixel_unshuffle expects 4D input [B, C, H, W]".to_string(),
        ));
    }

    let shape_binding = input.shape();
    let shape = shape_binding.dims();
    let batch_size = shape[0];
    let channels = shape[1];
    let height = shape[2];
    let width = shape[3];

    let r = downscale_factor;

    // Validate height and width are divisible by r
    if height % r != 0 || width % r != 0 {
        return Err(TorshError::InvalidArgument(format!(
            "Height {} and width {} must be divisible by downscale_factor {}",
            height, width, r
        )));
    }

    let output_channels = channels * r * r;
    let output_height = height / r;
    let output_width = width / r;

    // Direct implementation
    let data = input.data()?;
    let mut output_data =
        vec![0.0_f32; batch_size * output_channels * output_height * output_width];

    for b in 0..batch_size {
        for c in 0..channels {
            for h in 0..output_height {
                for w in 0..output_width {
                    for r_h in 0..r {
                        for r_w in 0..r {
                            let input_h = h * r + r_h;
                            let input_w = w * r + r_w;
                            let input_idx =
                                ((b * channels + c) * height + input_h) * width + input_w;
                            let output_c = c * r * r + r_h * r + r_w;
                            let output_idx = ((b * output_channels + output_c) * output_height + h)
                                * output_width
                                + w;
                            output_data[output_idx] = data[input_idx];
                        }
                    }
                }
            }
        }
    }

    Tensor::from_data(
        output_data,
        vec![batch_size, output_channels, output_height, output_width],
        input.device(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_hot_basic() -> TorshResult<()> {
        let indices = Tensor::from_vec(vec![0.0, 1.0, 2.0, 0.0], &[4])?;
        let encoded = one_hot(&indices, 3)?;

        assert_eq!(encoded.shape().dims(), &[4, 3]);
        let data = encoded.data()?;

        // Check first row [1, 0, 0]
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], 0.0);
        assert_eq!(data[2], 0.0);

        // Check second row [0, 1, 0]
        assert_eq!(data[3], 0.0);
        assert_eq!(data[4], 1.0);
        assert_eq!(data[5], 0.0);

        Ok(())
    }

    #[test]
    fn test_linear_without_bias() -> TorshResult<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3])?;
        let weight = Tensor::from_vec(vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0], &[2, 3])?;
        let output = linear(&input, &weight, None)?;

        assert_eq!(output.shape().dims(), &[1, 2]);
        let data = output.data()?;

        // First output: 1*1 + 2*1 + 3*1 = 6
        assert!((data[0] - 6.0).abs() < 1e-5);
        // Second output: 1*2 + 2*2 + 3*2 = 12
        assert!((data[1] - 12.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_pairwise_distance_euclidean() -> TorshResult<()> {
        let x1 = Tensor::from_vec(vec![0.0, 0.0, 3.0, 4.0], &[2, 2])?;
        let x2 = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], &[2, 2])?;
        let dist = pairwise_distance(&x1, &x2, 2.0, 1e-6)?;

        assert_eq!(dist.shape().dims(), &[2]);
        let data = dist.data()?;

        // First distance: sqrt(0^2 + 0^2) = 0
        assert!((data[0] - 0.0).abs() < 1e-5);
        // Second distance: sqrt(3^2 + 4^2) = 5
        assert!((data[1] - 5.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_cosine_similarity_basic() -> TorshResult<()> {
        // Parallel vectors should have similarity 1.0
        let x1 = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3])?;
        let x2 = Tensor::from_vec(vec![2.0, 4.0, 6.0], &[1, 3])?;
        let sim = cosine_similarity(&x1, &x2, 1, 1e-8)?;

        let data = sim.data()?;
        // Cosine similarity should be close to 1.0 for parallel vectors
        assert!((data[0] - 1.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_embedding_basic() -> TorshResult<()> {
        // Create simple embedding weight matrix [vocab_size=4, embedding_dim=3]
        let weight = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, // word 0
                4.0, 5.0, 6.0, // word 1
                7.0, 8.0, 9.0, // word 2
                10.0, 11.0, 12.0, // word 3
            ],
            &[4, 3],
        )?;

        // Lookup indices [0, 2, 1]
        let indices = Tensor::from_vec(vec![0.0, 2.0, 1.0], &[3])?;
        let embedded = embedding(&weight, &indices)?;

        assert_eq!(embedded.shape().dims(), &[3, 3]);
        let data = embedded.data()?;

        // Check first embedding (word 0): [1, 2, 3]
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], 2.0);
        assert_eq!(data[2], 3.0);

        // Check second embedding (word 2): [7, 8, 9]
        assert_eq!(data[3], 7.0);
        assert_eq!(data[4], 8.0);
        assert_eq!(data[5], 9.0);

        Ok(())
    }
}
