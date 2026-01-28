//! Linear transformations and attention operations
//!
//! This module provides linear transformations, embedding operations,
//! and attention mechanisms for neural networks.

use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

// =============================================================================
// LINEAR TRANSFORMATIONS
// =============================================================================

/// Linear transformation function
pub fn linear(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    // Linear layer computation: y = xW^T + b
    // Input: [batch_size, in_features]
    // Weight: [out_features, in_features]
    // Output: [batch_size, out_features]

    // We need to compute input @ weight.T
    // Since weight is [out_features, in_features], we need weight.T which is [in_features, out_features]
    // Then input[batch, in_features] @ weight.T[in_features, out_features] = output[batch, out_features]

    // For now, let's use the weight directly but fix the dimensions
    // weight should be transposed but our transpose only works for 2D
    // So we'll change the initialization to be [in_features, out_features]
    let output = input.matmul(weight)?;

    if let Some(bias_tensor) = bias {
        output.add(bias_tensor)
    } else {
        Ok(output)
    }
}

/// Bilinear transformation function
pub fn bilinear(
    input1: &Tensor,
    input2: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    // Bilinear transformation: y = x1^T * W * x2 + b
    // For now, implement a simplified version
    let temp = input1.matmul(weight)?;
    let output = temp.matmul(input2)?;

    if let Some(bias_tensor) = bias {
        output.add(bias_tensor)
    } else {
        Ok(output)
    }
}

// =============================================================================
// EMBEDDING OPERATIONS
// =============================================================================

/// Embedding lookup function
///
/// Performs embedding lookup from a weight matrix given input indices.
///
/// # Arguments
///
/// * `input` - Input tensor of indices (shape: [batch_size, seq_len] or similar)
/// * `weight` - Embedding weight matrix (shape: [vocab_size, embedding_dim])
/// * `padding_idx` - Optional index to zero out in the embedding
///
/// # Returns
///
/// Embedded tensor of shape [..., embedding_dim]
pub fn embedding(input: &Tensor<i64>, weight: &Tensor, padding_idx: Option<i64>) -> Result<Tensor> {
    let input_shape = input.shape();
    let input_dims = input_shape.dims();
    let weight_shape = weight.shape();
    let weight_dims = weight_shape.dims();

    // Validate weight shape (should be 2D: [vocab_size, embedding_dim])
    if weight_dims.len() != 2 {
        return Err(TorshError::InvalidShape(format!(
            "Embedding weight must be 2D, got shape {:?}",
            weight_dims
        )));
    }

    let vocab_size = weight_dims[0];
    let embedding_dim = weight_dims[1];

    // Get input indices
    let input_data = input.to_vec()?;
    let total_elements: usize = input_dims.iter().product();

    // Validate indices are within vocab_size
    for &idx in &input_data {
        if idx < 0 || idx >= vocab_size as i64 {
            return Err(TorshError::InvalidArgument(format!(
                "Index {} out of range for vocabulary size {}",
                idx, vocab_size
            )));
        }
    }

    // Perform embedding lookup using index_select
    // Flatten input indices to 1D for lookup
    let indices_tensor = Tensor::from_data(
        input_data.clone(),
        vec![total_elements],
        torsh_core::device::DeviceType::Cpu,
    )?;

    // Select rows from weight matrix (dim=0)
    let mut embedded = weight.index_select(0, &indices_tensor)?;

    // Handle padding_idx if specified
    if let Some(pad_idx) = padding_idx {
        // Zero out embeddings at padding positions
        let embedded_data = embedded.to_vec()?;
        let mut masked_data = embedded_data;

        for (i, &idx) in input_data.iter().enumerate() {
            if idx == pad_idx {
                // Zero out this embedding
                for j in 0..embedding_dim {
                    masked_data[i * embedding_dim + j] = 0.0;
                }
            }
        }

        embedded = Tensor::from_data(
            masked_data,
            vec![total_elements, embedding_dim],
            torsh_core::device::DeviceType::Cpu,
        )?;
    }

    // Reshape to match input shape + [embedding_dim]
    let mut output_shape: Vec<i32> = input_dims.iter().map(|&x| x as i32).collect();
    output_shape.push(embedding_dim as i32);

    embedded.reshape(&output_shape)
}

/// Embedding bag operation (sum, mean, max aggregation)
pub fn embedding_bag(
    input: &Tensor<i64>,
    weight: &Tensor,
    offsets: Option<&Tensor<i64>>,
    scale_grad_by_freq: bool,
    mode: &str,
    sparse: bool,
    per_sample_weights: Option<&Tensor>,
    include_last_offset: bool,
    padding_idx: Option<i64>,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    // Implement embedding bag: aggregate embeddings for variable-length sequences
    // Used for efficient text processing (bag-of-words, mean pooling, etc.)

    let _ = (scale_grad_by_freq, sparse); // Not used in forward pass

    let input_data = input.to_vec()?;
    let weight_shape_binding = weight.shape();
    let weight_shape = weight_shape_binding.dims();
    let num_embeddings = weight_shape[0];
    let embedding_dim = weight_shape[1];
    let weight_data = weight.to_vec()?;

    // Determine mode
    let reduction_mode = match mode {
        "sum" => 0,
        "mean" => 1,
        "max" => 2,
        _ => {
            return Err(TorshError::InvalidArgument(format!(
                "Invalid mode: {}",
                mode
            )))
        }
    };

    // Process offsets to determine bag boundaries
    let (num_bags, bag_boundaries) = if let Some(offsets_tensor) = offsets {
        let offsets_data = offsets_tensor.to_vec()?;
        let num_bags = if include_last_offset {
            offsets_data.len() - 1
        } else {
            offsets_data.len()
        };

        let mut boundaries = Vec::with_capacity(num_bags + 1);
        for &offset in offsets_data.iter() {
            boundaries.push(offset as usize);
        }
        if !include_last_offset {
            boundaries.push(input_data.len());
        }

        (num_bags, boundaries)
    } else {
        // No offsets: treat each input as a separate bag
        (input_data.len(), (0..=input_data.len()).collect())
    };

    // Get per-sample weights if provided
    let sample_weights = if let Some(weights_tensor) = per_sample_weights {
        Some(weights_tensor.to_vec()?)
    } else {
        None
    };

    // Initialize output tensors
    let mut output_data = vec![0.0f32; num_bags * embedding_dim];
    let mut offset2bag_data = vec![0.0f32; input_data.len()];
    let mut bag_size_data = vec![0.0f32; num_bags];
    let mut max_indices_data = vec![0.0f32; num_bags * embedding_dim];

    // Process each bag
    for bag_idx in 0..num_bags {
        let start = bag_boundaries[bag_idx];
        let end = bag_boundaries[bag_idx + 1];
        let bag_size = end - start;

        bag_size_data[bag_idx] = bag_size as f32;

        // Track max indices for max mode
        let mut max_indices_for_bag = vec![0usize; embedding_dim];
        let mut max_values_for_bag = vec![f32::NEG_INFINITY; embedding_dim];

        // Aggregate embeddings for this bag
        for (local_idx, global_idx) in (start..end).enumerate() {
            let idx = input_data[global_idx];

            // Mark offset2bag mapping
            offset2bag_data[global_idx] = bag_idx as f32;

            // Skip padding index if specified
            if let Some(padding) = padding_idx {
                if idx == padding {
                    continue;
                }
            }

            // Bounds check
            if idx < 0 || idx >= num_embeddings as i64 {
                return Err(TorshError::InvalidArgument(format!(
                    "Index {} out of bounds for embedding with {} entries",
                    idx, num_embeddings
                )));
            }

            // Get embedding vector
            let emb_start = (idx as usize) * embedding_dim;
            let emb_end = emb_start + embedding_dim;
            let embedding = &weight_data[emb_start..emb_end];

            // Apply per-sample weight if provided
            let weight_factor = if let Some(ref weights) = sample_weights {
                weights[global_idx]
            } else {
                1.0
            };

            // Aggregate based on mode
            let output_start = bag_idx * embedding_dim;
            for d in 0..embedding_dim {
                let weighted_value = embedding[d] * weight_factor;

                match reduction_mode {
                    0 | 1 => {
                        // sum or mean (mean will be divided later)
                        output_data[output_start + d] += weighted_value;
                    }
                    2 => {
                        // max
                        if weighted_value > max_values_for_bag[d] {
                            max_values_for_bag[d] = weighted_value;
                            max_indices_for_bag[d] = local_idx;
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }

        // Finalize reduction
        let output_start = bag_idx * embedding_dim;
        if reduction_mode == 1 && bag_size > 0 {
            // Mean: divide by bag size
            for d in 0..embedding_dim {
                output_data[output_start + d] /= bag_size as f32;
            }
        } else if reduction_mode == 2 {
            // Max: use max values
            for d in 0..embedding_dim {
                output_data[output_start + d] = max_values_for_bag[d];
                max_indices_data[bag_idx * embedding_dim + d] = max_indices_for_bag[d] as f32;
            }
        }
    }

    // Create output tensors
    let output = Tensor::from_data(
        output_data,
        vec![num_bags, embedding_dim],
        torsh_core::device::DeviceType::Cpu,
    )?;

    let offset2bag = Tensor::from_data(
        offset2bag_data,
        vec![input_data.len()],
        torsh_core::device::DeviceType::Cpu,
    )?;

    let bag_size = Tensor::from_data(
        bag_size_data,
        vec![num_bags],
        torsh_core::device::DeviceType::Cpu,
    )?;

    let max_indices = Tensor::from_data(
        max_indices_data,
        vec![num_bags, embedding_dim],
        torsh_core::device::DeviceType::Cpu,
    )?;

    Ok((output, offset2bag, bag_size, max_indices))
}

/// One-hot encoding
pub fn one_hot(input: &Tensor<i64>, num_classes: Option<usize>) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();
    let input_data = input.to_vec()?;

    // Determine number of classes
    let num_classes = if let Some(classes) = num_classes {
        classes
    } else {
        input_data.iter().map(|&x| x as usize).max().unwrap_or(0) + 1
    };

    // Create one-hot encoding
    let batch_size = input_shape.iter().product::<usize>();
    let mut one_hot_data = vec![0.0f32; batch_size * num_classes];

    for (i, &class_idx) in input_data.iter().enumerate() {
        if class_idx >= 0 && (class_idx as usize) < num_classes {
            one_hot_data[i * num_classes + class_idx as usize] = 1.0;
        }
    }

    let mut output_shape = input_shape.to_vec();
    output_shape.push(num_classes);

    Tensor::from_data(
        one_hot_data,
        output_shape,
        torsh_core::device::DeviceType::Cpu,
    )
}

// =============================================================================
// ATTENTION MECHANISMS
// =============================================================================

/// Multi-head attention function
///
/// Implements multi-head attention mechanism as described in "Attention is All You Need".
///
/// # Arguments
///
/// * `query` - Query tensor of shape [batch, seq_len, embed_dim]
/// * `key` - Key tensor of shape [batch, seq_len, embed_dim]
/// * `value` - Value tensor of shape [batch, seq_len, embed_dim]
/// * `embed_dim` - Total dimension of the model
/// * `num_heads` - Number of parallel attention heads
/// * `attn_mask` - Optional attention mask
/// * `key_padding_mask` - Optional key padding mask
/// * `need_weights` - Whether to return attention weights
/// * `attn_dropout` - Dropout probability for attention weights
/// * `training` - Whether in training mode
///
/// # Returns
///
/// Tuple of (output tensor, optional attention weights)
#[allow(clippy::too_many_arguments)]
pub fn multi_head_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    embed_dim: usize,
    num_heads: usize,
    attn_mask: Option<&Tensor>,
    key_padding_mask: Option<&Tensor<bool>>,
    need_weights: bool,
    attn_dropout: f32,
    training: bool,
) -> Result<(Tensor, Option<Tensor>)> {
    // Validate inputs
    if embed_dim % num_heads != 0 {
        return Err(TorshError::InvalidArgument(format!(
            "embed_dim ({}) must be divisible by num_heads ({})",
            embed_dim, num_heads
        )));
    }

    let head_dim = embed_dim / num_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let query_shape = query.shape();
    let query_dims = query_shape.dims();

    // Expected shape: [batch_size, seq_len, embed_dim]
    if query_dims.len() < 3 {
        return Err(TorshError::InvalidShape(format!(
            "Query tensor must have at least 3 dimensions, got shape {:?}",
            query_dims
        )));
    }

    let batch_size = query_dims[0];
    let seq_len = query_dims[1];

    // Reshape Q, K, V for multi-head attention
    // [batch, seq_len, embed_dim] -> [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
    let q_reshaped = query.reshape(&[
        batch_size as i32,
        seq_len as i32,
        num_heads as i32,
        head_dim as i32,
    ])?;
    let k_reshaped = key.reshape(&[
        batch_size as i32,
        seq_len as i32,
        num_heads as i32,
        head_dim as i32,
    ])?;
    let v_reshaped = value.reshape(&[
        batch_size as i32,
        seq_len as i32,
        num_heads as i32,
        head_dim as i32,
    ])?;

    // Transpose to [batch, num_heads, seq_len, head_dim]
    let q_transposed = q_reshaped.permute(&[0, 2, 1, 3])?;
    let k_transposed = k_reshaped.permute(&[0, 2, 1, 3])?;
    let v_transposed = v_reshaped.permute(&[0, 2, 1, 3])?;

    // Compute scaled dot-product attention for all heads in parallel
    // Q @ K^T scaled by sqrt(head_dim)
    let scores = q_transposed.matmul(&k_transposed.transpose(-1, -2)?)?;
    let scaled_scores = scores.mul_scalar(scale)?;

    // Apply attention mask if provided
    let masked_scores = if let Some(mask) = attn_mask {
        scaled_scores.add(mask)?
    } else {
        scaled_scores
    };

    // Apply key padding mask if provided
    // key_padding_mask shape: [batch, seq_len] where True indicates positions to mask
    let final_scores = if let Some(padding_mask) = key_padding_mask {
        // Convert boolean mask to float mask with -inf for masked positions
        let mask_data = padding_mask.to_vec()?;
        let mask_shape = padding_mask.shape();
        let mask_dims = mask_shape.dims();

        // Create float mask: -inf for True (masked), 0.0 for False (not masked)
        let float_mask_data: Vec<f32> = mask_data
            .iter()
            .map(|&masked| if masked { -f32::INFINITY } else { 0.0 })
            .collect();

        // Create float mask tensor [batch, seq_len]
        let float_mask = Tensor::from_data(
            float_mask_data,
            mask_dims.to_vec(),
            torsh_core::device::DeviceType::Cpu,
        )?;

        // Reshape to [batch, 1, 1, seq_len] for broadcasting with [batch, num_heads, seq_len, seq_len]
        let float_mask = float_mask.unsqueeze(1)?.unsqueeze(1)?;

        // Broadcast and add to scores
        masked_scores.add(&float_mask)?
    } else {
        masked_scores
    };

    // Apply softmax
    let attn_weights = crate::functional::activation::softmax(&final_scores, Some(-1))?;

    // Apply dropout if in training mode
    let attn_weights = if training && attn_dropout > 0.0 {
        crate::functional::activation::dropout(&attn_weights, attn_dropout, training)?
    } else {
        attn_weights
    };

    // Apply attention to values
    // [batch, num_heads, seq_len, seq_len] @ [batch, num_heads, seq_len, head_dim]
    // -> [batch, num_heads, seq_len, head_dim]
    let attn_output = attn_weights.matmul(&v_transposed)?;

    // Transpose back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
    let attn_output = attn_output.permute(&[0, 2, 1, 3])?;

    // Reshape to [batch, seq_len, embed_dim]
    let output = attn_output.reshape(&[batch_size as i32, seq_len as i32, embed_dim as i32])?;

    // Return attention weights if requested
    let weights = if need_weights {
        Some(attn_weights)
    } else {
        None
    };

    Ok((output, weights))
}

/// Scaled dot-product attention
pub fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    attn_mask: Option<&Tensor>,
    dropout_p: f32,
    is_causal: bool,
    scale: Option<f32>,
) -> Result<Tensor> {
    // Simplified implementation of scaled dot-product attention
    // attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    let binding = query.shape();
    let dims = binding.dims();
    let d_k = dims.last().unwrap_or(&1);
    let scale_factor = scale.unwrap_or(1.0 / (*d_k as f32).sqrt());

    // Q @ K^T
    let scores = query.matmul(&key.transpose(-1, -2)?)?;

    // Scale
    let scaled_scores = scores.mul_scalar(scale_factor)?;

    // Apply mask if provided
    let masked_scores = if let Some(mask) = attn_mask {
        // Add mask (assuming mask contains -inf for positions to mask)
        scaled_scores.add(mask)?
    } else {
        scaled_scores
    };

    // Apply causal mask if needed
    let final_scores = if is_causal {
        // Create causal mask: lower triangular matrix filled with 0, upper with -inf
        let seq_len = dims[dims.len() - 2]; // second to last dimension is sequence length
        let mut causal_mask_data = vec![-f32::INFINITY; seq_len * seq_len];

        for i in 0..seq_len {
            for j in 0..=i {
                causal_mask_data[i * seq_len + j] = 0.0;
            }
        }

        // For broadcasting, use shape [seq_len, seq_len]
        let causal_mask = Tensor::from_data(
            causal_mask_data,
            vec![seq_len, seq_len],
            torsh_core::device::DeviceType::Cpu,
        )?;

        // Broadcast and add mask
        masked_scores.add(&causal_mask)?
    } else {
        masked_scores
    };

    // Softmax
    let attn_weights = crate::functional::activation::softmax(&final_scores, Some(-1))?;

    // Apply dropout
    let attn_weights = if dropout_p > 0.0 {
        crate::functional::activation::dropout(&attn_weights, dropout_p, true)?
    } else {
        attn_weights
    };

    // Apply attention to values
    let output = attn_weights.matmul(value)?;

    Ok(output)
}

/// Multi-query attention (MQA)
///
/// MQA uses a single key and value head shared across all query heads,
/// reducing memory bandwidth requirements and improving inference speed.
///
/// # Arguments
///
/// * `query` - Query tensor of shape [batch, seq_len, embed_dim]
/// * `key` - Key tensor of shape [batch, seq_len, head_dim] (single head)
/// * `value` - Value tensor of shape [batch, seq_len, head_dim] (single head)
/// * `num_heads` - Number of query heads
/// * `attn_mask` - Optional attention mask
/// * `dropout_p` - Dropout probability
///
/// # Returns
///
/// Output tensor of shape [batch, seq_len, embed_dim]
pub fn multi_query_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_heads: usize,
    attn_mask: Option<&Tensor>,
    dropout_p: f32,
) -> Result<Tensor> {
    let query_shape = query.shape();
    let query_dims = query_shape.dims();

    if query_dims.len() < 3 {
        return Err(TorshError::InvalidShape(format!(
            "Query tensor must have at least 3 dimensions, got shape {:?}",
            query_dims
        )));
    }

    let batch_size = query_dims[0];
    let seq_len = query_dims[1];
    let embed_dim = query_dims[2];

    if embed_dim % num_heads != 0 {
        return Err(TorshError::InvalidArgument(format!(
            "embed_dim ({}) must be divisible by num_heads ({})",
            embed_dim, num_heads
        )));
    }

    let head_dim = embed_dim / num_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Reshape query: [batch, seq_len, num_heads, head_dim]
    let q_reshaped = query.reshape(&[
        batch_size as i32,
        seq_len as i32,
        num_heads as i32,
        head_dim as i32,
    ])?;
    // Transpose to [batch, num_heads, seq_len, head_dim]
    let q_transposed = q_reshaped.permute(&[0, 2, 1, 3])?;

    // Key and value are single-headed, expand to match query heads
    // [batch, seq_len, head_dim] -> [batch, 1, seq_len, head_dim]
    let k_expanded = key.unsqueeze(1)?;
    let v_expanded = value.unsqueeze(1)?;

    // Compute attention scores: Q @ K^T
    let scores = q_transposed.matmul(&k_expanded.transpose(-1, -2)?)?;
    let scaled_scores = scores.mul_scalar(scale)?;

    // Apply mask if provided
    let masked_scores = if let Some(mask) = attn_mask {
        scaled_scores.add(mask)?
    } else {
        scaled_scores
    };

    // Apply softmax
    let attn_weights = crate::functional::activation::softmax(&masked_scores, Some(-1))?;

    // Apply dropout
    let attn_weights = if dropout_p > 0.0 {
        crate::functional::activation::dropout(&attn_weights, dropout_p, true)?
    } else {
        attn_weights
    };

    // Apply attention to values
    let attn_output = attn_weights.matmul(&v_expanded)?;

    // Transpose back and reshape
    let attn_output = attn_output.permute(&[0, 2, 1, 3])?;
    attn_output.reshape(&[batch_size as i32, seq_len as i32, embed_dim as i32])
}

/// Grouped query attention (GQA)
///
/// GQA is a generalization of multi-query attention where queries are divided
/// into groups, with each group sharing key and value heads. This provides
/// a trade-off between standard multi-head attention and multi-query attention.
///
/// # Arguments
///
/// * `query` - Query tensor of shape [batch, seq_len, embed_dim]
/// * `key` - Key tensor of shape [batch, seq_len, num_kv_heads * head_dim]
/// * `value` - Value tensor of shape [batch, seq_len, num_kv_heads * head_dim]
/// * `num_heads` - Number of query heads
/// * `num_kv_heads` - Number of key/value heads (must divide num_heads)
/// * `attn_mask` - Optional attention mask
/// * `dropout_p` - Dropout probability
///
/// # Returns
///
/// Output tensor of shape [batch, seq_len, embed_dim]
pub fn grouped_query_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    attn_mask: Option<&Tensor>,
    dropout_p: f32,
) -> Result<Tensor> {
    let query_shape = query.shape();
    let query_dims = query_shape.dims();

    if query_dims.len() < 3 {
        return Err(TorshError::InvalidShape(format!(
            "Query tensor must have at least 3 dimensions, got shape {:?}",
            query_dims
        )));
    }

    let batch_size = query_dims[0];
    let seq_len = query_dims[1];
    let embed_dim = query_dims[2];

    // Validate that num_heads is divisible by num_kv_heads
    if num_heads % num_kv_heads != 0 {
        return Err(TorshError::InvalidArgument(format!(
            "num_heads ({}) must be divisible by num_kv_heads ({})",
            num_heads, num_kv_heads
        )));
    }

    if embed_dim % num_heads != 0 {
        return Err(TorshError::InvalidArgument(format!(
            "embed_dim ({}) must be divisible by num_heads ({})",
            embed_dim, num_heads
        )));
    }

    let head_dim = embed_dim / num_heads;
    let group_size = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Reshape query: [batch, seq_len, num_heads, head_dim]
    let q_reshaped = query.reshape(&[
        batch_size as i32,
        seq_len as i32,
        num_heads as i32,
        head_dim as i32,
    ])?;
    // Transpose to [batch, num_heads, seq_len, head_dim]
    let q_transposed = q_reshaped.permute(&[0, 2, 1, 3])?;

    // Reshape key and value: [batch, seq_len, num_kv_heads, head_dim]
    let k_reshaped = key.reshape(&[
        batch_size as i32,
        seq_len as i32,
        num_kv_heads as i32,
        head_dim as i32,
    ])?;
    let v_reshaped = value.reshape(&[
        batch_size as i32,
        seq_len as i32,
        num_kv_heads as i32,
        head_dim as i32,
    ])?;

    // Transpose to [batch, num_kv_heads, seq_len, head_dim]
    let k_transposed = k_reshaped.permute(&[0, 2, 1, 3])?;
    let v_transposed = v_reshaped.permute(&[0, 2, 1, 3])?;

    // Repeat k and v for each group
    // [batch, num_kv_heads, seq_len, head_dim] -> [batch, num_heads, seq_len, head_dim]
    // Manually implement repeat_interleave by repeating each head group_size times
    let k_data = k_transposed.to_vec()?;
    let v_data = v_transposed.to_vec()?;

    let mut k_expanded_data = Vec::with_capacity(batch_size * num_heads * seq_len * head_dim);
    let mut v_expanded_data = Vec::with_capacity(batch_size * num_heads * seq_len * head_dim);

    for b in 0..batch_size {
        for kv_head in 0..num_kv_heads {
            // Repeat this head group_size times
            for _ in 0..group_size {
                for s in 0..seq_len {
                    for h in 0..head_dim {
                        let idx = b * num_kv_heads * seq_len * head_dim
                            + kv_head * seq_len * head_dim
                            + s * head_dim
                            + h;
                        k_expanded_data.push(k_data[idx]);
                        v_expanded_data.push(v_data[idx]);
                    }
                }
            }
        }
    }

    let k_expanded = Tensor::from_data(
        k_expanded_data,
        vec![batch_size, num_heads, seq_len, head_dim],
        torsh_core::device::DeviceType::Cpu,
    )?;
    let v_expanded = Tensor::from_data(
        v_expanded_data,
        vec![batch_size, num_heads, seq_len, head_dim],
        torsh_core::device::DeviceType::Cpu,
    )?;

    // Compute attention scores: Q @ K^T
    let scores = q_transposed.matmul(&k_expanded.transpose(-1, -2)?)?;
    let scaled_scores = scores.mul_scalar(scale)?;

    // Apply mask if provided
    let masked_scores = if let Some(mask) = attn_mask {
        scaled_scores.add(mask)?
    } else {
        scaled_scores
    };

    // Apply softmax
    let attn_weights = crate::functional::activation::softmax(&masked_scores, Some(-1))?;

    // Apply dropout
    let attn_weights = if dropout_p > 0.0 {
        crate::functional::activation::dropout(&attn_weights, dropout_p, true)?
    } else {
        attn_weights
    };

    // Apply attention to values
    let attn_output = attn_weights.matmul(&v_expanded)?;

    // Transpose back and reshape
    let attn_output = attn_output.permute(&[0, 2, 1, 3])?;
    attn_output.reshape(&[batch_size as i32, seq_len as i32, embed_dim as i32])
}

// =============================================================================
// POSITIONAL ENCODING
// =============================================================================

/// Sinusoidal positional encoding
pub fn sinusoidal_positional_encoding(
    seq_len: usize,
    d_model: usize,
    max_len: Option<usize>,
) -> Result<Tensor> {
    let _max_len = max_len.unwrap_or(10000);
    let mut pe_data = vec![0.0f32; seq_len * d_model];

    for pos in 0..seq_len {
        for i in (0..d_model).step_by(2) {
            let div_term = (i as f32 / d_model as f32 * (-10000.0f32.ln())).exp();
            let pos_f = pos as f32;

            // sin for even indices
            pe_data[pos * d_model + i] = (pos_f * div_term).sin();

            // cos for odd indices
            if i + 1 < d_model {
                pe_data[pos * d_model + i + 1] = (pos_f * div_term).cos();
            }
        }
    }

    Tensor::from_data(
        pe_data,
        vec![seq_len, d_model],
        torsh_core::device::DeviceType::Cpu,
    )
}

/// Learnable positional encoding
pub fn learnable_positional_encoding(seq_len: usize, d_model: usize) -> Result<Tensor> {
    // Initialize with small random values
    torsh_tensor::creation::randn(&[seq_len, d_model])
}

/// Rotary positional encoding (RoPE)
///
/// Applies rotary position embeddings to input tensor. RoPE encodes position
/// information through rotation in complex space, allowing for better length
/// extrapolation in transformer models.
///
/// # Arguments
///
/// * `input` - Input tensor of shape [..., seq_len, d_model]
/// * `position_ids` - Position indices tensor of shape \[seq_len\]
/// * `theta` - Base wavelength for frequency calculation (typically 10000.0)
///
/// # Returns
///
/// Rotated tensor with same shape as input
pub fn rotary_positional_encoding(
    input: &Tensor,
    position_ids: &Tensor<i64>,
    theta: f32,
) -> Result<Tensor> {
    let input_shape = input.shape();
    let input_dims = input_shape.dims();

    if input_dims.len() < 2 {
        return Err(TorshError::InvalidShape(format!(
            "Input must have at least 2 dimensions, got shape {:?}",
            input_dims
        )));
    }

    let seq_len = input_dims[input_dims.len() - 2];
    let d_model = input_dims[input_dims.len() - 1];

    // Ensure d_model is even (required for complex rotation)
    if d_model % 2 != 0 {
        return Err(TorshError::InvalidArgument(format!(
            "d_model ({}) must be even for RoPE",
            d_model
        )));
    }

    let position_data = position_ids.to_vec()?;
    let input_data = input.to_vec()?;

    // Calculate frequencies
    let half_d = d_model / 2;
    let mut freqs = Vec::with_capacity(half_d);
    for i in 0..half_d {
        let freq = 1.0 / theta.powf(2.0 * i as f32 / d_model as f32);
        freqs.push(freq);
    }

    // Apply rotation
    let total_elements: usize = input_dims[..input_dims.len() - 2].iter().product();
    let mut output_data = vec![0.0f32; input_data.len()];

    for batch_idx in 0..total_elements {
        for pos_idx in 0..seq_len {
            let position = if pos_idx < position_data.len() {
                position_data[pos_idx] as f32
            } else {
                pos_idx as f32
            };

            for i in 0..half_d {
                let angle = position * freqs[i];
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                let base_idx = batch_idx * seq_len * d_model + pos_idx * d_model;
                let idx1 = base_idx + 2 * i;
                let idx2 = base_idx + 2 * i + 1;

                // Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
                let x1 = input_data[idx1];
                let x2 = input_data[idx2];

                output_data[idx1] = x1 * cos_val - x2 * sin_val;
                output_data[idx2] = x1 * sin_val + x2 * cos_val;
            }
        }
    }

    Tensor::from_data(
        output_data,
        input_dims.to_vec(),
        torsh_core::device::DeviceType::Cpu,
    )
}

// =============================================================================
// NORMALIZATION IN ATTENTION
// =============================================================================

/// RMS normalization (used in some modern architectures)
pub fn rms_norm(input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    // RMS normalization: x / sqrt(mean(x^2) + eps) * weight
    let squared = input.pow(2.0)?;
    let last_dim = input.shape().dims().len() - 1;
    let mean_squared = squared.mean(Some(&[last_dim]), true)?;
    let eps_tensor = torsh_tensor::creation::full_like(&mean_squared, eps)?;
    let rms = mean_squared.add(&eps_tensor)?.sqrt()?;
    let normalized = input.div(&rms)?;
    normalized.mul(weight)
}

/// Pre-normalization layer normalization
pub fn pre_norm_layer_norm(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    crate::functional::norm::layer_norm_enhanced(
        input,
        weight.shape().dims(),
        Some(weight),
        bias,
        eps,
    )
}

/// Post-normalization layer normalization
pub fn post_norm_layer_norm(
    input: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let combined = input.add(residual)?;
    crate::functional::norm::layer_norm_enhanced(
        &combined,
        weight.shape().dims(),
        Some(weight),
        bias,
        eps,
    )
}

// =============================================================================
// GATING MECHANISMS
// =============================================================================

/// Gated Linear Unit (GLU)
pub fn glu(input: &Tensor, dim: i32) -> Result<Tensor> {
    // GLU: σ(X * W + b) ⊙ (X * V + c)
    // Split input into two halves along dim
    let input_shape = input.shape();
    let split_size = input_shape.dims()[dim as usize] / 2;

    // For now, implement a simplified version
    let first_half = input.narrow(dim, 0i64, split_size)?;
    let second_half = input.narrow(dim, split_size as i64, split_size)?;

    let gated = crate::functional::activation::sigmoid(&second_half)?;
    first_half.mul(&gated)
}

/// Swish Gated Linear Unit (SwiGLU)
pub fn swiglu(input: &Tensor, dim: i32) -> Result<Tensor> {
    // SwiGLU: Swish(X * W) ⊙ (X * V)
    let input_shape = input.shape();
    let split_size = input_shape.dims()[dim as usize] / 2;

    let first_half = input.narrow(dim, 0i64, split_size)?;
    let second_half = input.narrow(dim, split_size as i64, split_size)?;

    let swish_result = crate::functional::activation::swish(&first_half)?;
    swish_result.mul(&second_half)
}

/// GELU Gated Linear Unit (GeGLU)
pub fn geglu(input: &Tensor, dim: i32) -> Result<Tensor> {
    // GeGLU: GELU(X * W) ⊙ (X * V)
    let input_shape = input.shape();
    let split_size = input_shape.dims()[dim as usize] / 2;

    let first_half = input.narrow(dim, 0i64, split_size)?;
    let second_half = input.narrow(dim, split_size as i64, split_size)?;

    let gelu_result = crate::functional::activation::gelu(&first_half)?;
    gelu_result.mul(&second_half)
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/// Apply attention mask
pub fn apply_attention_mask(
    scores: &Tensor,
    mask: &Tensor<bool>,
    mask_value: f32,
) -> Result<Tensor> {
    let mask_tensor = torsh_tensor::creation::full_like(scores, mask_value)?;
    scores.where_tensor(mask, &mask_tensor)
}

/// Create causal mask
pub fn create_causal_mask(seq_len: usize) -> Result<Tensor<bool>> {
    let mut mask_data = vec![false; seq_len * seq_len];

    for i in 0..seq_len {
        for j in 0..=i {
            mask_data[i * seq_len + j] = true;
        }
    }

    Tensor::<bool>::from_data(
        mask_data,
        vec![seq_len, seq_len],
        torsh_core::device::DeviceType::Cpu,
    )
}

/// Create padding mask
pub fn create_padding_mask(input_lengths: &[usize], max_len: usize) -> Result<Tensor> {
    let batch_size = input_lengths.len();
    let mut mask_data = vec![0.0f32; batch_size * max_len];

    for (batch_idx, &length) in input_lengths.iter().enumerate() {
        for seq_idx in 0..length.min(max_len) {
            mask_data[batch_idx * max_len + seq_idx] = 1.0f32;
        }
    }

    Tensor::from_data(
        mask_data,
        vec![batch_size, max_len],
        torsh_core::device::DeviceType::Cpu,
    )
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_tensor::Tensor;

    #[test]
    fn test_linear_basic() -> Result<()> {
        // Test basic linear transformation: y = xW + b
        // Input: [2, 3] (2 samples, 3 input features)
        // Weight: [3, 4] (3 input features, 4 output features)
        // Bias: [4] (4 output features)
        let input = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, // Sample 1
                4.0, 5.0, 6.0, // Sample 2
            ],
            &[2, 3],
        )?;

        let weight = Tensor::from_vec(
            vec![
                1.0, 0.0, 0.0, 0.0, // Feature 1 weights
                0.0, 1.0, 0.0, 0.0, // Feature 2 weights
                0.0, 0.0, 1.0, 0.0, // Feature 3 weights
            ],
            &[3, 4],
        )?;

        let bias = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[4])?;

        let output = linear(&input, &weight, Some(&bias))?;

        assert_eq!(output.shape().dims(), &[2, 4]);

        let output_data = output.to_vec()?;

        // Sample 1: [1, 2, 3] @ [[1,0,0,0],[0,1,0,0],[0,0,1,0]] + [0.1,0.2,0.3,0.4]
        // = [1, 2, 3, 0] + [0.1, 0.2, 0.3, 0.4] = [1.1, 2.2, 3.3, 0.4]
        assert_relative_eq!(output_data[0], 1.1, epsilon = 1e-5);
        assert_relative_eq!(output_data[1], 2.2, epsilon = 1e-5);
        assert_relative_eq!(output_data[2], 3.3, epsilon = 1e-5);
        assert_relative_eq!(output_data[3], 0.4, epsilon = 1e-5);

        Ok(())
    }

    #[test]
    fn test_linear_no_bias() -> Result<()> {
        // Test linear transformation without bias
        let input = Tensor::from_vec(vec![1.0, 2.0], &[1, 2])?;
        let weight = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[2, 2])?;

        let output = linear(&input, &weight, None)?;

        let output_data = output.to_vec()?;

        // [1, 2] @ [[2, 3], [4, 5]] = [1*2 + 2*4, 1*3 + 2*5] = [10, 13]
        assert_relative_eq!(output_data[0], 10.0, epsilon = 1e-5);
        assert_relative_eq!(output_data[1], 13.0, epsilon = 1e-5);

        Ok(())
    }

    #[test]
    fn test_glu_basic() -> Result<()> {
        // Test Gated Linear Unit
        // GLU splits input in half along dim, then: first_half * sigmoid(second_half)
        let input = Tensor::from_vec(
            vec![
                1.0, 2.0, 0.0, 0.0, // Will split to [1, 2] and [0, 0]
                2.0, 3.0, 1.0, 1.0, // Will split to [2, 3] and [1, 1]
            ],
            &[2, 4],
        )?;

        let output = glu(&input, 1)?; // Split along dimension 1

        assert_eq!(output.shape().dims(), &[2, 2]); // Output is half the size

        let output_data = output.to_vec()?;

        // First sample: [1, 2] * sigmoid([0, 0])
        // sigmoid(0) ≈ 0.5
        assert_relative_eq!(output_data[0], 1.0 * 0.5, epsilon = 1e-5);
        assert_relative_eq!(output_data[1], 2.0 * 0.5, epsilon = 1e-5);

        Ok(())
    }

    #[test]
    fn test_swiglu() -> Result<()> {
        // Test Swish-Gated Linear Unit
        // SwiGLU: swish(first_half) * second_half
        let input = Tensor::from_vec(
            vec![
                0.0, 1.0, 2.0, 3.0, // Will split to [0, 1] and [2, 3]
            ],
            &[1, 4],
        )?;

        let output = swiglu(&input, 1)?;

        assert_eq!(output.shape().dims(), &[1, 2]);

        let output_data = output.to_vec()?;

        // swish(0) * 2 and swish(1) * 3
        // swish(0) = 0 * sigmoid(0) = 0
        // swish(1) = 1 * sigmoid(1) ≈ 0.731
        assert_relative_eq!(output_data[0], 0.0, epsilon = 1e-5);
        assert!(output_data[1] > 2.0 && output_data[1] < 2.5); // swish(1) * 3

        Ok(())
    }

    #[test]
    fn test_geglu() -> Result<()> {
        // Test GELU-Gated Linear Unit
        // GeGLU: gelu(first_half) * second_half
        let input = Tensor::from_vec(
            vec![
                0.0, 1.0, 2.0, 3.0, // Will split to [0, 1] and [2, 3]
            ],
            &[1, 4],
        )?;

        let output = geglu(&input, 1)?;

        assert_eq!(output.shape().dims(), &[1, 2]);

        let output_data = output.to_vec()?;

        // gelu(0) * 2 ≈ 0, gelu(1) * 3
        // gelu(0) ≈ 0
        assert_relative_eq!(output_data[0], 0.0, epsilon = 1e-1);
        assert!(output_data[1] > 2.0); // gelu(1) > 0.8, so output > 2.4

        Ok(())
    }

    #[test]
    fn test_rms_norm_basic() -> Result<()> {
        // Test RMS normalization: x / sqrt(mean(x^2) + eps) * weight
        let input = Tensor::from_vec(
            vec![
                3.0, 4.0, // RMS = sqrt((9 + 16) / 2) = sqrt(12.5) ≈ 3.536
            ],
            &[1, 2],
        )?;

        let weight = Tensor::from_vec(vec![1.0, 1.0], &[2])?;

        let output = rms_norm(&input, &weight, 1e-5)?;

        let output_data = output.to_vec()?;

        // Expected: [3.0 / 3.536, 4.0 / 3.536] * [1, 1]
        let expected_rms = (12.5_f32).sqrt();
        assert_relative_eq!(output_data[0], 3.0 / expected_rms, epsilon = 1e-3);
        assert_relative_eq!(output_data[1], 4.0 / expected_rms, epsilon = 1e-3);

        Ok(())
    }

    #[test]
    fn test_rms_norm_with_weight() -> Result<()> {
        // Test RMS normalization with non-uniform weight
        let input = Tensor::from_vec(vec![2.0, 2.0, 2.0, 2.0], &[1, 4])?;

        let weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4])?;

        let output = rms_norm(&input, &weight, 1e-5)?;

        let output_data = output.to_vec()?;

        // RMS of [2, 2, 2, 2] = sqrt(4) = 2.0
        // Normalized: [1, 1, 1, 1]
        // After weight: [1, 2, 3, 4]
        assert_relative_eq!(output_data[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(output_data[1], 2.0, epsilon = 1e-5);
        assert_relative_eq!(output_data[2], 3.0, epsilon = 1e-5);
        assert_relative_eq!(output_data[3], 4.0, epsilon = 1e-5);

        Ok(())
    }

    #[test]
    fn test_rms_norm_batched() -> Result<()> {
        // Test RMS normalization with batch dimension
        let input = Tensor::from_vec(
            vec![
                3.0, 4.0, // Batch 1: RMS = sqrt(12.5)
                5.0, 12.0, // Batch 2: RMS = sqrt(84.5)
            ],
            &[2, 2],
        )?;

        let weight = Tensor::from_vec(vec![1.0, 1.0], &[2])?;

        let output = rms_norm(&input, &weight, 1e-5)?;

        assert_eq!(output.shape().dims(), &[2, 2]);

        let output_data = output.to_vec()?;

        // Each row should be normalized independently
        let rms1 = (12.5_f32).sqrt();
        let rms2 = (84.5_f32).sqrt();

        assert_relative_eq!(output_data[0], 3.0 / rms1, epsilon = 1e-3);
        assert_relative_eq!(output_data[1], 4.0 / rms1, epsilon = 1e-3);
        assert_relative_eq!(output_data[2], 5.0 / rms2, epsilon = 1e-3);
        assert_relative_eq!(output_data[3], 12.0 / rms2, epsilon = 1e-3);

        Ok(())
    }

    #[test]
    fn test_linear_shape_preservation() -> Result<()> {
        // Test that linear preserves batch dimensions correctly
        let input = Tensor::from_vec(
            vec![1.0; 3 * 5], // 3 samples, 5 features
            &[3, 5],
        )?;

        let weight = Tensor::from_vec(
            vec![1.0; 5 * 7], // 5 input features, 7 output features
            &[5, 7],
        )?;

        let output = linear(&input, &weight, None)?;

        assert_eq!(output.shape().dims(), &[3, 7]);

        Ok(())
    }
}
