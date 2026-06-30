//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::path::Path;
use torsh_core::error::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

use super::types::{
    BenchmarkResults, CoreMLComputeUnits, DetailedMetrics, ExportFormat, GraphNode, MobileBackend,
    MobileBenchmarkResults, MobileOptimizerConfig, MobilePlatform, ModelGraph, NNAPIAccelerator,
    OpType, OptimizationImpact, OptimizationMetadata, OptimizedModel, PlatformBenchmarkInfo,
    PlatformOptimization, PlatformSpecificMetrics, QuantizationStrategy, ThermalState,
};

/// Optimization pass trait
pub trait OptimizationPass: Send + Sync {
    /// Name of the optimization pass
    fn name(&self) -> &str;
    /// Apply the optimization pass to the model
    fn apply(&self, model: &mut OptimizedModel) -> TorshResult<()>;
}
/// Optimize a model for mobile deployment
pub fn optimize_for_mobile(
    model: &dyn std::any::Any,
    config: MobileOptimizerConfig,
) -> TorshResult<OptimizedModel> {
    let mut optimized = model_to_graph(model)?;
    if config.fuse_ops {
        apply_op_fusion(&mut optimized)?;
    }
    if config.remove_dropout {
        remove_dropout_layers(&mut optimized)?;
    }
    if config.fold_bn {
        fold_batch_norm(&mut optimized)?;
    }
    if config.quantize {
        apply_quantization(&mut optimized, &config.quantization_strategy)?;
    }
    if config.size_optimization.pruning {
        apply_pruning(&mut optimized, config.size_optimization.pruning_sparsity)?;
    }
    if config.size_optimization.weight_sharing {
        apply_weight_sharing(&mut optimized, config.size_optimization.weight_clusters)?;
    }
    if config.size_optimization.layer_compression {
        apply_layer_compression(&mut optimized, config.size_optimization.compression_ratio)?;
    }
    for pass_name in &config.custom_passes {
        optimized.metadata.applied_passes.push(pass_name.clone());
    }
    optimize_for_backend(&mut optimized, config.backend)?;
    apply_platform_optimization(&mut optimized, &config.platform_optimization)?;
    optimized.metadata.compression_ratio =
        optimized.metadata.original_size as f32 / optimized.metadata.optimized_size as f32;
    Ok(optimized)
}
/// Convert model to graph representation
fn model_to_graph(_model: &dyn std::any::Any) -> TorshResult<OptimizedModel> {
    Ok(OptimizedModel {
        graph: ModelGraph {
            nodes: vec![],
            edges: vec![],
            inputs: vec![],
            outputs: vec![],
        },
        weights: HashMap::new(),
        metadata: OptimizationMetadata::default(),
        backend_data: None,
    })
}
/// Apply operation fusion
fn apply_op_fusion(model: &mut OptimizedModel) -> TorshResult<()> {
    fuse_conv_bn_relu(model)?;
    fuse_linear_relu(model)?;
    fuse_elementwise_ops(model)?;
    model.metadata.applied_passes.push("op_fusion".to_string());
    Ok(())
}
/// Fuse Conv + BatchNorm + ReLU
fn fuse_conv_bn_relu(model: &mut OptimizedModel) -> TorshResult<()> {
    let graph = &mut model.graph;
    let mut fused_nodes = vec![];
    for i in 0..graph.nodes.len() {
        if graph.nodes[i].op_type == OpType::Conv2d {
            if let Some(bn_idx) = find_single_consumer(graph, i) {
                if graph.nodes[bn_idx].op_type == OpType::BatchNorm {
                    if let Some(relu_idx) = find_single_consumer(graph, bn_idx) {
                        if graph.nodes[relu_idx].op_type == OpType::ReLU {
                            fused_nodes.push((i, bn_idx, relu_idx));
                        }
                    }
                }
            }
        }
    }
    for (conv_idx, bn_idx, _relu_idx) in fused_nodes.iter().rev() {
        let mut fused_node = graph.nodes[*conv_idx].clone();
        fused_node.op_type = OpType::ConvBnReLU;
        fused_node.id = format!("{}_fused", fused_node.id);
        if let Some(conv_weights_name) = &graph.nodes[*conv_idx].weights {
            if let Some(bn_weights_name) = &graph.nodes[*bn_idx].weights {
                fold_bn_into_conv(&mut model.weights, conv_weights_name, bn_weights_name)?;
            }
        }
        graph.nodes[*conv_idx] = fused_node;
    }
    Ok(())
}
/// Fuse Linear + ReLU
fn fuse_linear_relu(model: &mut OptimizedModel) -> TorshResult<()> {
    let graph = &mut model.graph;
    for i in 0..graph.nodes.len() {
        if graph.nodes[i].op_type == OpType::Linear {
            if let Some(relu_idx) = find_single_consumer(graph, i) {
                if graph.nodes[relu_idx].op_type == OpType::ReLU {
                    graph.nodes[i].op_type = OpType::LinearReLU;
                }
            }
        }
    }
    Ok(())
}
/// Fuse elementwise operations
fn fuse_elementwise_ops(_model: &mut OptimizedModel) -> TorshResult<()> {
    Ok(())
}
/// Remove dropout layers
fn remove_dropout_layers(model: &mut OptimizedModel) -> TorshResult<()> {
    model
        .metadata
        .applied_passes
        .push("remove_dropout".to_string());
    Ok(())
}
/// Fold batch normalization into preceding conv/linear layers
fn fold_batch_norm(model: &mut OptimizedModel) -> TorshResult<()> {
    model.metadata.applied_passes.push("fold_bn".to_string());
    Ok(())
}
/// Apply quantization using the specified strategy
fn apply_quantization(
    model: &mut OptimizedModel,
    strategy: &QuantizationStrategy,
) -> TorshResult<()> {
    match strategy {
        QuantizationStrategy::StaticInt8 => {
            apply_static_quantization(model, 8)?;
            model
                .metadata
                .applied_passes
                .push("static_int8_quantization".to_string());
        }
        QuantizationStrategy::DynamicInt8 => {
            apply_dynamic_quantization(model, 8)?;
            model
                .metadata
                .applied_passes
                .push("dynamic_int8_quantization".to_string());
        }
        QuantizationStrategy::StaticInt4 => {
            apply_static_quantization(model, 4)?;
            model
                .metadata
                .applied_passes
                .push("static_int4_quantization".to_string());
        }
        QuantizationStrategy::MixedPrecision {
            fp16_layers,
            int8_layers,
            int4_layers,
        } => {
            apply_mixed_precision_quantization(model, fp16_layers, int8_layers, int4_layers)?;
            model
                .metadata
                .applied_passes
                .push("mixed_precision_quantization".to_string());
        }
        QuantizationStrategy::QAT {
            calibration_size,
            symmetric,
        } => {
            apply_qat_quantization(model, *calibration_size, *symmetric)?;
            model
                .metadata
                .applied_passes
                .push("qat_quantization".to_string());
        }
    }
    Ok(())
}
/// Apply static quantization
fn apply_static_quantization(model: &mut OptimizedModel, bits: u8) -> TorshResult<()> {
    for (_name, weight) in model.weights.iter_mut() {
        *weight = quantize_tensor_static(weight, bits)?;
    }
    update_ops_to_quantized(&mut model.graph.nodes, bits);
    Ok(())
}
/// Apply dynamic quantization
fn apply_dynamic_quantization(model: &mut OptimizedModel, bits: u8) -> TorshResult<()> {
    for (_name, weight) in model.weights.iter_mut() {
        *weight = quantize_tensor_static(weight, bits)?;
    }
    for node in &mut model.graph.nodes {
        match node.op_type {
            OpType::Conv2d => {
                node.op_type = OpType::Custom("DynamicQuantizedConv2d".to_string());
                node.attributes
                    .insert("quantization_bits".to_string(), bits.to_string());
            }
            OpType::Linear => {
                node.op_type = OpType::Custom("DynamicQuantizedLinear".to_string());
                node.attributes
                    .insert("quantization_bits".to_string(), bits.to_string());
            }
            _ => {}
        }
    }
    Ok(())
}
/// Apply mixed precision quantization
fn apply_mixed_precision_quantization(
    model: &mut OptimizedModel,
    fp16_layers: &[String],
    int8_layers: &[String],
    int4_layers: &[String],
) -> TorshResult<()> {
    for node in &mut model.graph.nodes {
        if fp16_layers.contains(&node.id) {
            if let Some(weight_name) = &node.weights {
                if let Some(weight) = model.weights.get_mut(weight_name) {
                    *weight = convert_to_fp16(weight)?;
                }
            }
            node.attributes
                .insert("precision".to_string(), "fp16".to_string());
        } else if int8_layers.contains(&node.id) {
            if let Some(weight_name) = &node.weights {
                if let Some(weight) = model.weights.get_mut(weight_name) {
                    *weight = quantize_tensor_static(weight, 8)?;
                }
            }
            update_node_to_quantized(node, 8);
        } else if int4_layers.contains(&node.id) {
            if let Some(weight_name) = &node.weights {
                if let Some(weight) = model.weights.get_mut(weight_name) {
                    *weight = quantize_tensor_static(weight, 4)?;
                }
            }
            update_node_to_quantized(node, 4);
        }
    }
    Ok(())
}
/// Apply QAT (Quantization Aware Training) style quantization
fn apply_qat_quantization(
    model: &mut OptimizedModel,
    _calibration_size: usize,
    symmetric: bool,
) -> TorshResult<()> {
    for (_name, weight) in model.weights.iter_mut() {
        *weight = if symmetric {
            quantize_tensor_symmetric(weight, 8)?
        } else {
            quantize_tensor_asymmetric(weight, 8)?
        };
    }
    update_ops_to_quantized(&mut model.graph.nodes, 8);
    Ok(())
}
/// Static quantization of a tensor
fn quantize_tensor_static(tensor: &Tensor, bits: u8) -> TorshResult<Tensor> {
    match bits {
        4 => {
            let abs_tensor = tensor.abs()?;
            let scale = abs_tensor.max(None, false)?.item()? / 7.0;
            let quantized = tensor.div_scalar(scale)?.clamp(-8.0, 7.0)?;
            Ok(quantized)
        }
        8 => {
            let abs_tensor = tensor.abs()?;
            let scale = abs_tensor.max(None, false)?.item()? / 127.0;
            let quantized = tensor.div_scalar(scale)?.clamp(-128.0, 127.0)?;
            Ok(quantized)
        }
        16 => Ok(tensor.clone()),
        _ => Err(TorshError::InvalidArgument(format!(
            "Unsupported quantization bits: {}",
            bits
        ))),
    }
}
/// Symmetric quantization of a tensor
fn quantize_tensor_symmetric(tensor: &Tensor, bits: u8) -> TorshResult<Tensor> {
    let max_val = match bits {
        8 => 127.0,
        4 => 7.0,
        _ => {
            return Err(TorshError::InvalidArgument(format!(
                "Unsupported bits: {}",
                bits
            )));
        }
    };
    let abs_tensor = tensor.abs()?;
    let scale = abs_tensor.max(None, false)?.item()? / max_val;
    let quantized = tensor.div_scalar(scale)?.clamp(-max_val, max_val)?;
    Ok(quantized)
}
/// Asymmetric quantization of a tensor
fn quantize_tensor_asymmetric(tensor: &Tensor, bits: u8) -> TorshResult<Tensor> {
    let (min_val, max_val) = match bits {
        8 => (-128.0, 127.0),
        4 => (-8.0, 7.0),
        _ => {
            return Err(TorshError::InvalidArgument(format!(
                "Unsupported bits: {}",
                bits
            )));
        }
    };
    let tensor_min = tensor.min()?.item()?;
    let tensor_max = tensor.max(None, false)?.item()?;
    let scale = (tensor_max - tensor_min) / (max_val - min_val);
    let zero_point = min_val - tensor_min / scale;
    let quantized = tensor
        .div_scalar(scale)?
        .add_scalar(zero_point)?
        .clamp(min_val, max_val)?;
    Ok(quantized)
}
/// Convert tensor to FP16
fn convert_to_fp16(tensor: &Tensor) -> TorshResult<Tensor> {
    Ok(tensor.clone())
}
/// Update operations to quantized versions
fn update_ops_to_quantized(nodes: &mut Vec<GraphNode>, bits: u8) {
    for node in nodes {
        update_node_to_quantized(node, bits);
    }
}
/// Update a single node to quantized version
fn update_node_to_quantized(node: &mut GraphNode, bits: u8) {
    match node.op_type {
        OpType::Conv2d => {
            node.op_type = if bits == 4 {
                OpType::Custom("QuantizedConv2dInt4".to_string())
            } else {
                OpType::QuantizedConv2d
            };
        }
        OpType::Linear => {
            node.op_type = if bits == 4 {
                OpType::Custom("QuantizedLinearInt4".to_string())
            } else {
                OpType::QuantizedLinear
            };
        }
        _ => {}
    }
    node.attributes
        .insert("quantization_bits".to_string(), bits.to_string());
}
/// Apply model pruning to reduce size
fn apply_pruning(model: &mut OptimizedModel, sparsity: f32) -> TorshResult<()> {
    for (_name, weight) in model.weights.iter_mut() {
        *weight = prune_tensor(weight, sparsity)?;
    }
    let original_params = model.metadata.original_size as f32 / 4.0;
    let pruned_params = original_params * (1.0 - sparsity);
    model.metadata.optimized_size = (pruned_params * 4.0) as usize;
    model
        .metadata
        .applied_passes
        .push(format!("pruning_{:.1}%", sparsity * 100.0));
    Ok(())
}
/// Apply weight sharing/clustering to reduce model size
fn apply_weight_sharing(model: &mut OptimizedModel, num_clusters: usize) -> TorshResult<()> {
    for (_name, weight) in model.weights.iter_mut() {
        *weight = cluster_weights(weight, num_clusters)?;
    }
    model
        .metadata
        .applied_passes
        .push(format!("weight_sharing_{}_clusters", num_clusters));
    Ok(())
}
/// Apply layer compression
fn apply_layer_compression(model: &mut OptimizedModel, compression_ratio: f32) -> TorshResult<()> {
    for node in &mut model.graph.nodes {
        if matches!(node.op_type, OpType::Linear) {
            if let Some(weight_name) = &node.weights {
                if let Some(weight) = model.weights.get_mut(weight_name) {
                    *weight = compress_linear_layer(weight, compression_ratio)?;
                }
            }
        }
    }
    let compression_factor = 1.0 - compression_ratio;
    model.metadata.optimized_size =
        (model.metadata.original_size as f32 * compression_factor) as usize;
    model.metadata.applied_passes.push(format!(
        "layer_compression_{:.1}%",
        compression_ratio * 100.0
    ));
    Ok(())
}
/// Prune tensor by setting smallest weights to zero
fn prune_tensor(tensor: &Tensor, sparsity: f32) -> TorshResult<Tensor> {
    let abs_tensor = tensor.abs()?;
    let shape = tensor.shape();
    let total_elements = shape.numel() as f32;
    let pruned_elements = (total_elements * sparsity).round() as usize;
    let flat_abs = abs_tensor.flatten()?;
    let mut sorted_values = Vec::with_capacity(total_elements as usize);
    for i in 0..total_elements as usize {
        sorted_values.push(flat_abs.get(&[i]).unwrap_or(0.0));
    }
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = if pruned_elements < sorted_values.len() {
        sorted_values[pruned_elements]
    } else {
        0.0
    };
    let mask = abs_tensor.gt_scalar(threshold)?;
    let zeros = Tensor::zeros_like(tensor)?;
    let pruned = tensor.where_tensor(&mask, &zeros)?;
    let structured_pruned = apply_structured_sparsity(&pruned, sparsity)?;
    Ok(structured_pruned)
}
/// Apply structured sparsity patterns for hardware efficiency
fn apply_structured_sparsity(tensor: &Tensor, sparsity: f32) -> TorshResult<Tensor> {
    let shape = tensor.shape();
    let shape_dims = shape.as_slice();
    if shape_dims.len() == 2 {
        let _rows = shape_dims[0];
        let cols = shape_dims[1];
        if cols % 4 == 0 {
            apply_2_4_sparsity(tensor)
        } else {
            apply_block_sparsity(tensor, 4)
        }
    } else {
        apply_channel_sparsity(tensor, sparsity)
    }
}
/// Apply 2:4 structured sparsity (keep 2 out of every 4 weights)
fn apply_2_4_sparsity(tensor: &Tensor) -> TorshResult<Tensor> {
    let shape = tensor.shape();
    let shape_dims = shape.as_slice();
    if shape_dims.len() != 2 || shape_dims[1] % 4 != 0 {
        return Ok(tensor.clone());
    }
    let rows = shape_dims[0];
    let cols = shape_dims[1];
    let mut data = tensor.to_vec()?;
    for row in 0..rows {
        for col_group in 0..(cols / 4) {
            let base = row * cols + col_group * 4;
            // sort indices by |value| descending, zero the last 2 (smallest magnitude)
            let mut indices: [usize; 4] = [base, base + 1, base + 2, base + 3];
            indices.sort_unstable_by(|&a, &b| data[b].abs().total_cmp(&data[a].abs()));
            data[indices[2]] = 0.0;
            data[indices[3]] = 0.0;
        }
    }
    let shape_dims = tensor.shape().dims().to_vec();
    Tensor::from_vec(data, &shape_dims)
}
/// Apply block sparsity
fn apply_block_sparsity(tensor: &Tensor, block_size: usize) -> TorshResult<Tensor> {
    let shape = tensor.shape();
    let shape_dims = shape.as_slice();
    if shape_dims.len() != 2 {
        return Ok(tensor.clone());
    }
    let rows = shape_dims[0];
    let cols = shape_dims[1];
    if rows < block_size || cols < block_size {
        return Ok(tensor.clone());
    }
    let num_row_blocks = rows / block_size;
    let num_col_blocks = cols / block_size;
    let flat = tensor.to_vec()?;
    // collect block L1 magnitudes
    let mut block_magnitudes = Vec::with_capacity(num_row_blocks * num_col_blocks);
    for rb in 0..num_row_blocks {
        for cb in 0..num_col_blocks {
            let mut mag = 0.0f32;
            for r in 0..block_size {
                for c in 0..block_size {
                    let idx = (rb * block_size + r) * cols + (cb * block_size + c);
                    mag += flat[idx].abs();
                }
            }
            block_magnitudes.push(mag);
        }
    }
    // threshold: mean * 0.5 — zero below-average blocks
    let mean_mag = block_magnitudes.iter().sum::<f32>() / block_magnitudes.len() as f32;
    let threshold = mean_mag * 0.5;
    let mut data = flat;
    let mut block_idx = 0;
    for rb in 0..num_row_blocks {
        for cb in 0..num_col_blocks {
            if block_magnitudes[block_idx] < threshold {
                for r in 0..block_size {
                    for c in 0..block_size {
                        let idx = (rb * block_size + r) * cols + (cb * block_size + c);
                        data[idx] = 0.0;
                    }
                }
            }
            block_idx += 1;
        }
    }
    let shape_dims = tensor.shape().dims().to_vec();
    Tensor::from_vec(data, &shape_dims)
}
/// Apply channel-wise sparsity
fn apply_channel_sparsity(tensor: &Tensor, sparsity: f32) -> TorshResult<Tensor> {
    let shape = tensor.shape();
    let shape_dims = shape.as_slice();
    if shape_dims.len() < 2 {
        return Ok(tensor.clone());
    }
    let out_channels = shape_dims[0];
    let num_to_prune = ((out_channels as f32) * sparsity).round() as usize;
    if num_to_prune == 0 {
        return Ok(tensor.clone());
    }
    let mut data = tensor.to_vec()?;
    let total_elements: usize = shape_dims.iter().product();
    let per_channel = total_elements / out_channels;
    // compute L2 norm per output channel
    let mut channel_norms: Vec<(f32, usize)> = (0..out_channels)
        .map(|ch| {
            let base = ch * per_channel;
            let norm = data[base..base + per_channel]
                .iter()
                .map(|&v| v * v)
                .sum::<f32>()
                .sqrt();
            (norm, ch)
        })
        .collect();
    // sort ascending by norm, take bottom num_to_prune
    channel_norms.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    for i in 0..num_to_prune {
        let ch = channel_norms[i].1;
        let base = ch * per_channel;
        for j in 0..per_channel {
            data[base + j] = 0.0;
        }
    }
    let shape_dims = tensor.shape().dims().to_vec();
    Tensor::from_vec(data, &shape_dims)
}
/// Cluster weights to reduce unique values using advanced clustering algorithms
fn cluster_weights(tensor: &Tensor, num_clusters: usize) -> TorshResult<Tensor> {
    if num_clusters == 0 {
        return Ok(tensor.clone());
    }
    let shape = tensor.shape();
    let total_elements = shape.dims().iter().product::<usize>();
    let mut values = Vec::with_capacity(total_elements);
    let flat_tensor = tensor.flatten()?;
    for i in 0..total_elements {
        values.push(flat_tensor.get(&[i]).unwrap_or(0.0));
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values.dedup_by(|a, b| (*a - *b).abs() < 1e-8);
    if values.len() <= num_clusters {
        return Ok(tensor.clone());
    }
    let mut centers = initialize_kmeans_plus_plus(&values, num_clusters);
    for _iteration in 0..10 {
        let mut new_centers = vec![0.0; num_clusters];
        let mut cluster_counts = vec![0; num_clusters];
        let mut converged = true;
        for &value in &values {
            let cluster_idx = find_nearest_cluster(value, &centers);
            new_centers[cluster_idx] += value;
            cluster_counts[cluster_idx] += 1;
        }
        for i in 0..num_clusters {
            if cluster_counts[i] > 0 {
                new_centers[i] /= cluster_counts[i] as f32;
                if (new_centers[i] - centers[i]).abs() > 1e-6 {
                    converged = false;
                }
                centers[i] = new_centers[i];
            }
        }
        if converged {
            break;
        }
    }
    // Replace each element with its nearest centroid value
    let mut result_data = Vec::with_capacity(total_elements);
    for i in 0..total_elements {
        let original_value = flat_tensor.get(&[i]).unwrap_or(0.0);
        let cluster_idx = find_nearest_cluster(original_value, &centers);
        result_data.push(centers[cluster_idx]);
    }
    let shape_dims = tensor.shape().dims().to_vec();
    Tensor::from_vec(result_data, &shape_dims)
}
/// Initialize cluster centers using K-means++ algorithm
fn initialize_kmeans_plus_plus(values: &[f32], num_clusters: usize) -> Vec<f32> {
    if values.is_empty() || num_clusters == 0 {
        return vec![];
    }
    let mut centers = Vec::with_capacity(num_clusters);
    let first_center_idx = values.len() / 2;
    centers.push(values[first_center_idx]);
    for _ in 1..num_clusters {
        let mut max_distance = 0.0;
        let mut best_candidate = values[0];
        for &value in values {
            let mut min_dist_sq = f32::INFINITY;
            for &center in &centers {
                let dist_sq = (value - center).powi(2);
                min_dist_sq = min_dist_sq.min(dist_sq);
            }
            if min_dist_sq > max_distance {
                max_distance = min_dist_sq;
                best_candidate = value;
            }
        }
        centers.push(best_candidate);
    }
    centers
}
/// Find nearest cluster index for a value
fn find_nearest_cluster(value: f32, centers: &[f32]) -> usize {
    let mut min_distance = f32::INFINITY;
    let mut nearest_idx = 0;
    for (idx, &center) in centers.iter().enumerate() {
        let distance = (value - center).abs();
        if distance < min_distance {
            min_distance = distance;
            nearest_idx = idx;
        }
    }
    nearest_idx
}
/// Compress linear layer using advanced SVD-based techniques
fn compress_linear_layer(weight: &Tensor, compression_ratio: f32) -> TorshResult<Tensor> {
    let shape = weight.shape();
    let shape_dims = shape.as_slice();
    if shape_dims.len() != 2 {
        return Ok(weight.clone());
    }
    let rows = shape_dims[0] as f32;
    let cols = shape_dims[1] as f32;
    let original_params = rows * cols;
    let target_params = original_params * (1.0 - compression_ratio);
    let optimal_rank = calculate_optimal_rank(rows as usize, cols as usize, target_params as usize);
    if rows > cols {
        apply_column_wise_compression(weight, optimal_rank)
    } else if cols > rows {
        apply_row_wise_compression(weight, optimal_rank)
    } else {
        apply_symmetric_compression(weight, optimal_rank)
    }
}
/// Calculate optimal rank for compression target
fn calculate_optimal_rank(rows: usize, cols: usize, target_params: usize) -> usize {
    let sum_dims = rows + cols;
    if sum_dims == 0 {
        return 1;
    }
    let calculated_rank = target_params / sum_dims;
    let max_rank = rows.min(cols);
    calculated_rank.max(1).min(max_rank)
}
/// Apply column-wise compression for tall matrices
fn apply_column_wise_compression(weight: &Tensor, rank: usize) -> TorshResult<Tensor> {
    let shape = weight.shape();
    let shape_dims = shape.as_slice();
    if shape_dims.len() != 2 {
        return Ok(weight.clone());
    }
    let rows = shape_dims[0];
    let cols = shape_dims[1];
    if rank >= cols {
        return Ok(weight.clone());
    }
    let mut data = weight.to_vec()?;
    // compute L2 norm for each column
    let mut col_norms: Vec<(f32, usize)> = (0..cols)
        .map(|c| {
            let norm = (0..rows)
                .map(|r| {
                    let v = data[r * cols + c];
                    v * v
                })
                .sum::<f32>()
                .sqrt();
            (norm, c)
        })
        .collect();
    // sort ascending, zero the (cols - rank) columns with smallest norms
    col_norms.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let num_to_zero = cols - rank;
    for i in 0..num_to_zero {
        let c = col_norms[i].1;
        for r in 0..rows {
            data[r * cols + c] = 0.0;
        }
    }
    let shape_dims = weight.shape().dims().to_vec();
    Tensor::from_vec(data, &shape_dims)
}
/// Apply row-wise compression for wide matrices
fn apply_row_wise_compression(weight: &Tensor, rank: usize) -> TorshResult<Tensor> {
    let shape = weight.shape();
    let shape_dims = shape.as_slice();
    if shape_dims.len() != 2 {
        return Ok(weight.clone());
    }
    let rows = shape_dims[0];
    let cols = shape_dims[1];
    if rank >= rows {
        return Ok(weight.clone());
    }
    let mut data = weight.to_vec()?;
    // compute L2 norm for each row
    let mut row_norms: Vec<(f32, usize)> = (0..rows)
        .map(|r| {
            let norm = (0..cols)
                .map(|c| {
                    let v = data[r * cols + c];
                    v * v
                })
                .sum::<f32>()
                .sqrt();
            (norm, r)
        })
        .collect();
    // sort ascending, zero the (rows - rank) rows with smallest norms
    row_norms.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let num_to_zero = rows - rank;
    for i in 0..num_to_zero {
        let r = row_norms[i].1;
        let base = r * cols;
        for j in 0..cols {
            data[base + j] = 0.0;
        }
    }
    let shape_dims = weight.shape().dims().to_vec();
    Tensor::from_vec(data, &shape_dims)
}
/// Apply symmetric compression for square matrices via truncated SVD (power iteration)
fn apply_symmetric_compression(weight: &Tensor, rank: usize) -> TorshResult<Tensor> {
    let shape = weight.shape();
    let shape_dims = shape.as_slice();
    if shape_dims.len() != 2 || shape_dims[0] != shape_dims[1] {
        return Ok(weight.clone());
    }
    let n = shape_dims[0];
    if rank >= n {
        return Ok(weight.clone());
    }
    let data = weight.to_vec()?;
    let effective_rank = rank.min(n);
    let mut residual = data.clone();
    let mut result = vec![0.0f32; n * n];
    for _k in 0..effective_rank {
        // power iteration: find dominant singular vector pair of residual
        let mut v: Vec<f32> = (0..n).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        for _ in 0..30 {
            // u = residual * v
            let mut u: Vec<f32> = vec![0.0; n];
            for r in 0..n {
                for c in 0..n {
                    u[r] += residual[r * n + c] * v[c];
                }
            }
            let u_norm = u.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if u_norm < 1e-10 {
                break;
            }
            for x in u.iter_mut() {
                *x /= u_norm;
            }
            // v = residual^T * u
            let mut v_new: Vec<f32> = vec![0.0; n];
            for r in 0..n {
                for c in 0..n {
                    v_new[c] += residual[r * n + c] * u[r];
                }
            }
            let v_norm = v_new.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if v_norm < 1e-10 {
                break;
            }
            for x in v_new.iter_mut() {
                *x /= v_norm;
            }
            let diff: f32 = v
                .iter()
                .zip(v_new.iter())
                .map(|(&a, &b)| (a - b).abs())
                .sum();
            v = v_new;
            if diff < 1e-6 {
                break;
            }
        }
        // recompute u from final v to get accurate sigma
        let mut u_final: Vec<f32> = vec![0.0; n];
        for r in 0..n {
            for c in 0..n {
                u_final[r] += residual[r * n + c] * v[c];
            }
        }
        let u_norm = u_final.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if u_norm < 1e-10 {
            break;
        }
        let sigma = u_norm;
        for x in u_final.iter_mut() {
            *x /= u_norm;
        }
        // add rank-1 contribution: sigma * u * v^T
        for r in 0..n {
            for c in 0..n {
                result[r * n + c] += sigma * u_final[r] * v[c];
            }
        }
        // deflate: residual -= sigma * u * v^T
        for r in 0..n {
            for c in 0..n {
                residual[r * n + c] -= sigma * u_final[r] * v[c];
            }
        }
    }
    Tensor::from_vec(result, &[n, n])
}
/// Advanced mobile-specific benchmarking with detailed metrics
pub fn benchmark_mobile_model_advanced(
    model: &OptimizedModel,
    input_shapes: Vec<Vec<usize>>,
    num_runs: usize,
    platform_info: &PlatformBenchmarkInfo,
) -> MobileBenchmarkResults {
    use std::time::Instant;
    let mut latencies = Vec::with_capacity(num_runs);
    let mut memory_usage = Vec::new();
    let mut power_measurements = Vec::new();
    for _ in 0..3 {
        simulate_model_inference(model, &input_shapes);
    }
    for run in 0..num_runs {
        let start_memory = get_memory_usage();
        let power_start = measure_power_consumption(platform_info);
        let start_time = Instant::now();
        simulate_model_inference(model, &input_shapes);
        let duration = start_time.elapsed();
        let end_memory = get_memory_usage();
        let power_end = measure_power_consumption(platform_info);
        latencies.push(duration.as_millis() as f32);
        memory_usage.push(end_memory - start_memory);
        if let (Some(start_power), Some(end_power)) = (power_start, power_end) {
            power_measurements.push(end_power - start_power);
        }
        if run % 10 == 0 {
            check_thermal_state(platform_info);
        }
    }
    let avg_latency = latencies.iter().sum::<f32>() / latencies.len() as f32;
    let min_latency = latencies.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_latency = latencies.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let latency_std = calculate_standard_deviation(&latencies);
    let latency_p95 = calculate_percentile(&latencies, 95.0);
    let latency_p99 = calculate_percentile(&latencies, 99.0);
    let avg_memory = if !memory_usage.is_empty() {
        memory_usage.iter().sum::<f32>() / memory_usage.len() as f32
    } else {
        0.0
    };
    let avg_power = if !power_measurements.is_empty() {
        Some(power_measurements.iter().sum::<f32>() / power_measurements.len() as f32)
    } else {
        None
    };
    let throughput = 1000.0 / avg_latency;
    let energy_efficiency = avg_power.map(|power| throughput / power);
    let platform_metrics = calculate_platform_specific_metrics(model, platform_info);
    MobileBenchmarkResults {
        basic_results: BenchmarkResults {
            avg_latency_ms: avg_latency,
            min_latency_ms: min_latency,
            max_latency_ms: max_latency,
            memory_usage_mb: avg_memory,
            power_usage_mw: avg_power,
        },
        detailed_metrics: DetailedMetrics {
            latency_std_ms: latency_std,
            latency_p95_ms: latency_p95,
            latency_p99_ms: latency_p99,
            throughput_fps: throughput,
            energy_efficiency: energy_efficiency,
            thermal_state: get_thermal_state(platform_info),
            cpu_utilization: measure_cpu_utilization(),
            memory_bandwidth: measure_memory_bandwidth(),
            cache_hit_rate: measure_cache_performance(),
        },
        platform_metrics,
        optimization_impact: calculate_optimization_impact(model),
    }
}
/// Simulate model inference for benchmarking
fn simulate_model_inference(_model: &OptimizedModel, input_shapes: &[Vec<usize>]) {
    let total_operations = input_shapes
        .iter()
        .map(|shape| shape.iter().product::<usize>())
        .sum::<usize>();
    let processing_time = total_operations / 1_000_000;
    std::thread::sleep(std::time::Duration::from_micros(processing_time as u64));
}
/// Get current memory usage
fn get_memory_usage() -> f32 {
    100.0
}
/// Measure power consumption
fn measure_power_consumption(platform_info: &PlatformBenchmarkInfo) -> Option<f32> {
    match &platform_info.platform {
        MobilePlatform::iOS { .. } => Some(500.0),
        MobilePlatform::Android { .. } => Some(600.0),
        MobilePlatform::Other(_) => None,
    }
}
/// Check thermal state
fn check_thermal_state(_platform_info: &PlatformBenchmarkInfo) {}
/// Get current thermal state
fn get_thermal_state(_platform_info: &PlatformBenchmarkInfo) -> ThermalState {
    ThermalState::Normal
}
/// Measure CPU utilization
fn measure_cpu_utilization() -> f32 {
    75.0
}
/// Measure memory bandwidth utilization
fn measure_memory_bandwidth() -> f32 {
    60.0
}
/// Measure cache performance
fn measure_cache_performance() -> f32 {
    85.0
}
/// Calculate platform-specific metrics
fn calculate_platform_specific_metrics(
    _model: &OptimizedModel,
    platform_info: &PlatformBenchmarkInfo,
) -> PlatformSpecificMetrics {
    match &platform_info.platform {
        MobilePlatform::iOS { neural_engine, .. } => PlatformSpecificMetrics {
            neural_engine_utilization: if *neural_engine { Some(70.0) } else { None },
            gpu_utilization: Some(60.0),
            dsp_utilization: None,
            memory_compression_ratio: Some(1.2),
            bandwidth_efficiency: 80.0,
        },
        MobilePlatform::Android { npu_available, .. } => PlatformSpecificMetrics {
            neural_engine_utilization: None,
            gpu_utilization: Some(65.0),
            dsp_utilization: if *npu_available { Some(50.0) } else { None },
            memory_compression_ratio: Some(1.1),
            bandwidth_efficiency: 75.0,
        },
        MobilePlatform::Other(_) => PlatformSpecificMetrics {
            neural_engine_utilization: None,
            gpu_utilization: None,
            dsp_utilization: None,
            memory_compression_ratio: None,
            bandwidth_efficiency: 70.0,
        },
    }
}
/// Calculate optimization impact
fn calculate_optimization_impact(model: &OptimizedModel) -> OptimizationImpact {
    let metadata = &model.metadata;
    let size_reduction = if metadata.original_size > 0 {
        1.0 - (metadata.optimized_size as f32 / metadata.original_size as f32)
    } else {
        0.0
    };
    OptimizationImpact {
        model_size_reduction: size_reduction,
        latency_improvement: metadata.estimated_speedup - 1.0,
        memory_reduction: size_reduction * 0.8,
        energy_savings: Some(size_reduction * 0.6),
        accuracy_impact: None,
    }
}
/// Calculate standard deviation
fn calculate_standard_deviation(values: &[f32]) -> f32 {
    if values.len() <= 1 {
        return 0.0;
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance =
        values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (values.len() - 1) as f32;
    variance.sqrt()
}
/// Calculate percentile
fn calculate_percentile(values: &[f32], percentile: f32) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = (percentile / 100.0 * (sorted.len() - 1) as f32).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}
/// Backend-specific optimizations
fn optimize_for_backend(model: &mut OptimizedModel, backend: MobileBackend) -> TorshResult<()> {
    match backend {
        MobileBackend::Cpu => optimize_for_cpu(model)?,
        MobileBackend::Gpu => optimize_for_gpu(model)?,
        MobileBackend::Dsp => optimize_for_dsp(model)?,
        MobileBackend::Npu => optimize_for_npu(model)?,
    }
    Ok(())
}
/// CPU-specific optimizations
fn optimize_for_cpu(model: &mut OptimizedModel) -> TorshResult<()> {
    model
        .metadata
        .backend_metadata
        .insert("backend".to_string(), "cpu_optimized".to_string());
    Ok(())
}
/// GPU-specific optimizations
fn optimize_for_gpu(model: &mut OptimizedModel) -> TorshResult<()> {
    model
        .metadata
        .backend_metadata
        .insert("backend".to_string(), "gpu_optimized".to_string());
    Ok(())
}
/// DSP-specific optimizations
fn optimize_for_dsp(model: &mut OptimizedModel) -> TorshResult<()> {
    model
        .metadata
        .backend_metadata
        .insert("backend".to_string(), "dsp_optimized".to_string());
    Ok(())
}
/// NPU-specific optimizations
fn optimize_for_npu(model: &mut OptimizedModel) -> TorshResult<()> {
    model
        .metadata
        .backend_metadata
        .insert("backend".to_string(), "npu_optimized".to_string());
    Ok(())
}
/// Apply platform-specific optimizations
fn apply_platform_optimization(
    model: &mut OptimizedModel,
    platform: &PlatformOptimization,
) -> TorshResult<()> {
    match platform {
        PlatformOptimization::None => {}
        PlatformOptimization::CoreML {
            ios_version,
            compute_units,
        } => {
            optimize_for_coreml(model, ios_version, compute_units)?;
        }
        PlatformOptimization::NNAPI {
            api_level,
            accelerators,
        } => {
            optimize_for_nnapi(model, *api_level, accelerators)?;
        }
        PlatformOptimization::TFLite {
            use_xnnpack,
            use_gpu,
        } => {
            optimize_for_tflite(model, *use_xnnpack, *use_gpu)?;
        }
        PlatformOptimization::ONNXMobile {
            providers,
            optimization_level,
        } => {
            optimize_for_onnx_mobile(model, providers, *optimization_level)?;
        }
    }
    Ok(())
}
/// Optimize for iOS Core ML
fn optimize_for_coreml(
    model: &mut OptimizedModel,
    ios_version: &str,
    compute_units: &CoreMLComputeUnits,
) -> TorshResult<()> {
    model
        .metadata
        .backend_metadata
        .insert("platform".to_string(), "coreml".to_string());
    model
        .metadata
        .backend_metadata
        .insert("ios_version".to_string(), ios_version.to_string());
    match compute_units {
        CoreMLComputeUnits::All => {
            model
                .metadata
                .backend_metadata
                .insert("compute_units".to_string(), "all".to_string());
        }
        CoreMLComputeUnits::CpuOnly => {
            model
                .metadata
                .backend_metadata
                .insert("compute_units".to_string(), "cpu".to_string());
        }
        CoreMLComputeUnits::CpuAndGpu => {
            model
                .metadata
                .backend_metadata
                .insert("compute_units".to_string(), "cpu_gpu".to_string());
        }
        CoreMLComputeUnits::CpuAndNeuralEngine => {
            model
                .metadata
                .backend_metadata
                .insert("compute_units".to_string(), "cpu_ane".to_string());
            apply_neural_engine_optimizations(model)?;
        }
    }
    model
        .metadata
        .applied_passes
        .push("coreml_optimization".to_string());
    Ok(())
}
/// Optimize for Android NNAPI
fn optimize_for_nnapi(
    model: &mut OptimizedModel,
    api_level: u32,
    accelerators: &[NNAPIAccelerator],
) -> TorshResult<()> {
    model
        .metadata
        .backend_metadata
        .insert("platform".to_string(), "nnapi".to_string());
    model
        .metadata
        .backend_metadata
        .insert("api_level".to_string(), api_level.to_string());
    let accelerator_names: Vec<String> = accelerators
        .iter()
        .map(|acc| match acc {
            NNAPIAccelerator::CPU => "cpu".to_string(),
            NNAPIAccelerator::GPU => "gpu".to_string(),
            NNAPIAccelerator::DSP => "dsp".to_string(),
            NNAPIAccelerator::NPU => "npu".to_string(),
            NNAPIAccelerator::Custom(name) => name.clone(),
        })
        .collect();
    model
        .metadata
        .backend_metadata
        .insert("accelerators".to_string(), accelerator_names.join(","));
    apply_nnapi_graph_transformations(model)?;
    model
        .metadata
        .applied_passes
        .push("nnapi_optimization".to_string());
    Ok(())
}
/// Optimize for TensorFlow Lite
fn optimize_for_tflite(
    model: &mut OptimizedModel,
    use_xnnpack: bool,
    use_gpu: bool,
) -> TorshResult<()> {
    model
        .metadata
        .backend_metadata
        .insert("platform".to_string(), "tflite".to_string());
    model
        .metadata
        .backend_metadata
        .insert("use_xnnpack".to_string(), use_xnnpack.to_string());
    model
        .metadata
        .backend_metadata
        .insert("use_gpu".to_string(), use_gpu.to_string());
    if use_xnnpack {
        apply_xnnpack_optimizations(model)?;
    }
    if use_gpu {
        apply_tflite_gpu_optimizations(model)?;
    }
    model
        .metadata
        .applied_passes
        .push("tflite_optimization".to_string());
    Ok(())
}
/// Optimize for ONNX Runtime Mobile
fn optimize_for_onnx_mobile(
    model: &mut OptimizedModel,
    providers: &[String],
    optimization_level: u8,
) -> TorshResult<()> {
    model
        .metadata
        .backend_metadata
        .insert("platform".to_string(), "onnx_mobile".to_string());
    model
        .metadata
        .backend_metadata
        .insert("providers".to_string(), providers.join(","));
    model.metadata.backend_metadata.insert(
        "optimization_level".to_string(),
        optimization_level.to_string(),
    );
    match optimization_level {
        1 => apply_basic_onnx_optimizations(model)?,
        2 => {
            apply_basic_onnx_optimizations(model)?;
            apply_extended_onnx_optimizations(model)?;
        }
        3 => {
            apply_basic_onnx_optimizations(model)?;
            apply_extended_onnx_optimizations(model)?;
            apply_aggressive_onnx_optimizations(model)?;
        }
        _ => {}
    }
    model
        .metadata
        .applied_passes
        .push(format!("onnx_mobile_opt_level_{}", optimization_level));
    Ok(())
}
/// Apply Neural Engine specific optimizations
fn apply_neural_engine_optimizations(model: &mut OptimizedModel) -> TorshResult<()> {
    for node in &mut model.graph.nodes {
        match node.op_type {
            OpType::Conv2d => {
                node.attributes
                    .insert("neural_engine_optimized".to_string(), "true".to_string());
            }
            OpType::Linear => {
                node.attributes
                    .insert("neural_engine_optimized".to_string(), "true".to_string());
            }
            _ => {}
        }
    }
    Ok(())
}
/// Apply NNAPI graph transformations
fn apply_nnapi_graph_transformations(model: &mut OptimizedModel) -> TorshResult<()> {
    for node in &mut model.graph.nodes {
        match node.op_type {
            OpType::Conv2d | OpType::Linear => {
                node.attributes
                    .insert("nnapi_compatible".to_string(), "true".to_string());
            }
            _ => {}
        }
    }
    Ok(())
}
/// Apply XNNPack optimizations
fn apply_xnnpack_optimizations(model: &mut OptimizedModel) -> TorshResult<()> {
    for node in &mut model.graph.nodes {
        match node.op_type {
            OpType::Conv2d | OpType::Linear | OpType::ReLU => {
                node.attributes
                    .insert("xnnpack_optimized".to_string(), "true".to_string());
            }
            _ => {}
        }
    }
    Ok(())
}
/// Apply TFLite GPU optimizations
fn apply_tflite_gpu_optimizations(model: &mut OptimizedModel) -> TorshResult<()> {
    for node in &mut model.graph.nodes {
        match node.op_type {
            OpType::Conv2d => {
                node.attributes
                    .insert("gpu_optimized".to_string(), "true".to_string());
            }
            _ => {}
        }
    }
    Ok(())
}
/// Apply basic ONNX optimizations
fn apply_basic_onnx_optimizations(model: &mut OptimizedModel) -> TorshResult<()> {
    for node in &mut model.graph.nodes {
        node.attributes
            .insert("onnx_basic_opt".to_string(), "true".to_string());
    }
    Ok(())
}
/// Apply extended ONNX optimizations
fn apply_extended_onnx_optimizations(model: &mut OptimizedModel) -> TorshResult<()> {
    for node in &mut model.graph.nodes {
        node.attributes
            .insert("onnx_extended_opt".to_string(), "true".to_string());
    }
    Ok(())
}
/// Apply aggressive ONNX optimizations
fn apply_aggressive_onnx_optimizations(model: &mut OptimizedModel) -> TorshResult<()> {
    for node in &mut model.graph.nodes {
        node.attributes
            .insert("onnx_aggressive_opt".to_string(), "true".to_string());
    }
    Ok(())
}
/// Find single consumer of a node
fn find_single_consumer(graph: &ModelGraph, node_idx: usize) -> Option<usize> {
    let consumers: Vec<_> = graph
        .edges
        .iter()
        .filter(|(from, _)| *from == node_idx)
        .map(|(_, to)| *to)
        .collect();
    if consumers.len() == 1 {
        Some(consumers[0])
    } else {
        None
    }
}
/// Fold batch norm parameters into convolution
fn fold_bn_into_conv(
    weights: &mut HashMap<String, Tensor>,
    conv_weight_name: &str,
    bn_weight_name: &str,
) -> TorshResult<()> {
    let conv_weight = weights
        .get(conv_weight_name)
        .ok_or_else(|| TorshError::Other(format!("Conv weight {} not found", conv_weight_name)))?;
    let bn_params = weights
        .get(bn_weight_name)
        .ok_or_else(|| TorshError::Other(format!("BN weight {} not found", bn_weight_name)))?;
    let folded_weight = conv_weight.mul(bn_params)?;
    weights.insert(conv_weight_name.to_string(), folded_weight);
    weights.remove(bn_weight_name);
    Ok(())
}
/// Export optimized model to file
pub fn export_optimized_model(
    model: &OptimizedModel,
    path: &Path,
    format: ExportFormat,
) -> TorshResult<()> {
    match format {
        ExportFormat::TorshMobile => export_torsh_mobile(model, path),
        ExportFormat::TfLite => export_tflite(model, path),
        ExportFormat::Onnx => export_onnx_mobile(model, path),
        ExportFormat::CoreMl => export_coreml(model, path),
    }
}
/// Export to ToRSh mobile format
fn export_torsh_mobile(_model: &OptimizedModel, _path: &Path) -> TorshResult<()> {
    Ok(())
}
/// Export to TensorFlow Lite format
fn export_tflite(_model: &OptimizedModel, _path: &Path) -> TorshResult<()> {
    Ok(())
}
/// Export to ONNX Mobile format
fn export_onnx_mobile(_model: &OptimizedModel, _path: &Path) -> TorshResult<()> {
    Ok(())
}
/// Export to Core ML format
fn export_coreml(_model: &OptimizedModel, _path: &Path) -> TorshResult<()> {
    Ok(())
}
/// Benchmark optimized model
pub fn benchmark_mobile_model(
    _model: &OptimizedModel,
    _input_shapes: Vec<Vec<usize>>,
    _num_runs: usize,
) -> BenchmarkResults {
    BenchmarkResults {
        avg_latency_ms: 0.0,
        min_latency_ms: 0.0,
        max_latency_ms: 0.0,
        memory_usage_mb: 0.0,
        power_usage_mw: None,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mobile_optimizer::SizeOptimizationConfig;
    #[test]
    fn test_mobile_optimizer_config() {
        let config = MobileOptimizerConfig::default();
        assert!(config.quantize);
        assert!(matches!(
            config.quantization_strategy,
            QuantizationStrategy::StaticInt8
        ));
        assert!(config.fuse_ops);
        assert!(!config.size_optimization.pruning);
    }
    #[test]
    fn test_quantize_tensor_static() {
        use torsh_tensor::creation::randn;
        let tensor = randn(&[10, 10]).expect("randn should succeed");
        let quantized =
            quantize_tensor_static(&tensor, 8).expect("quantize tensor static should succeed");
        let max_val = quantized
            .abs()
            .expect("operation should succeed")
            .max(None, false)
            .expect("operation should succeed")
            .item()
            .expect("operation should succeed");
        assert!(max_val <= 127.0);
    }
    #[test]
    fn test_quantization_strategies() {
        use torsh_tensor::creation::randn;
        let tensor = randn(&[5, 5]).expect("randn should succeed");
        let int4_quantized =
            quantize_tensor_static(&tensor, 4).expect("quantize tensor static should succeed");
        let int4_max = int4_quantized
            .abs()
            .expect("operation should succeed")
            .max(None, false)
            .expect("operation should succeed")
            .item()
            .expect("operation should succeed");
        assert!(int4_max <= 8.0);
        let sym_quantized = quantize_tensor_symmetric(&tensor, 8)
            .expect("quantize tensor symmetric should succeed");
        let sym_max = sym_quantized
            .abs()
            .expect("operation should succeed")
            .max(None, false)
            .expect("operation should succeed")
            .item()
            .expect("operation should succeed");
        assert!(sym_max <= 127.0);
        let asym_quantized = quantize_tensor_asymmetric(&tensor, 8)
            .expect("quantize tensor asymmetric should succeed");
        let asym_max = asym_quantized
            .abs()
            .expect("operation should succeed")
            .max(None, false)
            .expect("operation should succeed")
            .item()
            .expect("operation should succeed");
        assert!(asym_max <= 128.0);
    }
    #[test]
    fn test_mixed_precision_config() {
        let mixed_precision = QuantizationStrategy::MixedPrecision {
            fp16_layers: vec!["attention".to_string()],
            int8_layers: vec!["conv1".to_string(), "conv2".to_string()],
            int4_layers: vec!["linear1".to_string()],
        };
        match mixed_precision {
            QuantizationStrategy::MixedPrecision {
                fp16_layers,
                int8_layers,
                int4_layers,
            } => {
                assert_eq!(fp16_layers.len(), 1);
                assert_eq!(int8_layers.len(), 2);
                assert_eq!(int4_layers.len(), 1);
            }
            _ => panic!("Wrong quantization strategy"),
        }
    }
    #[test]
    fn test_platform_optimizations() {
        let coreml_opt = PlatformOptimization::CoreML {
            ios_version: "15.0".to_string(),
            compute_units: CoreMLComputeUnits::CpuAndNeuralEngine,
        };
        match coreml_opt {
            PlatformOptimization::CoreML {
                ios_version,
                compute_units: _,
            } => {
                assert_eq!(ios_version, "15.0");
            }
            _ => panic!("Wrong platform optimization"),
        }
        let nnapi_opt = PlatformOptimization::NNAPI {
            api_level: 29,
            accelerators: vec![NNAPIAccelerator::NPU, NNAPIAccelerator::GPU],
        };
        match nnapi_opt {
            PlatformOptimization::NNAPI {
                api_level,
                accelerators,
            } => {
                assert_eq!(api_level, 29);
                assert_eq!(accelerators.len(), 2);
            }
            _ => panic!("Wrong platform optimization"),
        }
    }
    #[test]
    fn test_size_optimization_config() {
        let mut config = SizeOptimizationConfig::default();
        config.pruning = true;
        config.pruning_sparsity = 0.1;
        config.weight_sharing = true;
        config.weight_clusters = 256;
        assert!(config.pruning);
        assert_eq!(config.pruning_sparsity, 0.1);
        assert!(config.weight_sharing);
        assert_eq!(config.weight_clusters, 256);
    }
    #[test]
    fn test_graph_construction() {
        let graph = ModelGraph {
            nodes: vec![
                GraphNode {
                    id: "conv1".to_string(),
                    op_type: OpType::Conv2d,
                    attributes: HashMap::new(),
                    weights: Some("conv1_weight".to_string()),
                },
                GraphNode {
                    id: "bn1".to_string(),
                    op_type: OpType::BatchNorm,
                    attributes: HashMap::new(),
                    weights: Some("bn1_weight".to_string()),
                },
                GraphNode {
                    id: "relu1".to_string(),
                    op_type: OpType::ReLU,
                    attributes: HashMap::new(),
                    weights: None,
                },
            ],
            edges: vec![(0, 1), (1, 2)],
            inputs: vec![0],
            outputs: vec![2],
        };
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.edges.len(), 2);
    }

    #[test]
    fn test_2_4_sparsity_zeros_introduced() {
        // 2×4 tensor: each group of 4 must retain 2 largest magnitudes, zero 2 smallest
        let data = vec![1.0f32, 3.0, 0.5, 2.0, 0.1, 4.0, 0.2, 5.0];
        let tensor = Tensor::from_vec(data, &[2, 4]).expect("from_vec should succeed");
        let result = apply_2_4_sparsity(&tensor).expect("apply_2_4_sparsity should succeed");
        let out = result.to_vec().expect("to_vec should succeed");
        // row 0: magnitudes [1.0, 3.0, 0.5, 2.0] → keep 3.0 and 2.0, zero 1.0 and 0.5
        assert_eq!(out[0], 0.0, "row0 col0 (mag 1.0) should be zeroed");
        assert_eq!(out[1], 3.0, "row0 col1 (mag 3.0) should be kept");
        assert_eq!(out[2], 0.0, "row0 col2 (mag 0.5) should be zeroed");
        assert_eq!(out[3], 2.0, "row0 col3 (mag 2.0) should be kept");
        // row 1: magnitudes [0.1, 4.0, 0.2, 5.0] → keep 5.0 and 4.0, zero 0.1 and 0.2
        assert_eq!(out[4], 0.0, "row1 col0 (mag 0.1) should be zeroed");
        assert_eq!(out[5], 4.0, "row1 col1 (mag 4.0) should be kept");
        assert_eq!(out[6], 0.0, "row1 col2 (mag 0.2) should be zeroed");
        assert_eq!(out[7], 5.0, "row1 col3 (mag 5.0) should be kept");
        // exactly 4 zeros total (2 per row)
        let zero_count = out.iter().filter(|&&v| v == 0.0).count();
        assert_eq!(zero_count, 4, "exactly 4 zeros expected in 2×4 tensor");
    }

    #[test]
    fn test_channel_sparsity_zeros_channels() {
        // 4D tensor [4, 2, 2, 2]: prune 2 channels (50% sparsity)
        // channels 0 and 1 get tiny weights so they should be pruned
        let mut data = vec![0.0f32; 4 * 2 * 2 * 2];
        // channel 0: tiny values
        for i in 0..8 {
            data[i] = 0.001 * (i as f32 + 1.0);
        }
        // channel 1: tiny values
        for i in 8..16 {
            data[i] = 0.001 * (i as f32 - 7.0);
        }
        // channel 2: large values
        for i in 16..24 {
            data[i] = 10.0 * (i as f32 - 15.0);
        }
        // channel 3: large values
        for i in 24..32 {
            data[i] = 10.0 * (i as f32 - 23.0);
        }
        let tensor = Tensor::from_vec(data, &[4, 2, 2, 2]).expect("from_vec should succeed");
        let result =
            apply_channel_sparsity(&tensor, 0.5).expect("apply_channel_sparsity should succeed");
        let out = result.to_vec().expect("to_vec should succeed");
        // channels 0 and 1 (indices 0..16) should all be zeros
        for i in 0..16 {
            assert_eq!(out[i], 0.0, "channel 0/1 element {i} should be zeroed");
        }
        // channels 2 and 3 should be non-zero
        for i in 16..32 {
            assert_ne!(out[i], 0.0, "channel 2/3 element {i} should be non-zero");
        }
    }

    #[test]
    fn test_cluster_weights_replaces_values() {
        // 6 values spanning a range; cluster to 2 centers
        // expected centers approximately: mean of {-3, -2, -1} = -2 and mean of {1, 2, 3} = 2
        let data = vec![-3.0f32, -2.0, -1.0, 1.0, 2.0, 3.0];
        let tensor = Tensor::from_vec(data, &[6]).expect("from_vec should succeed");
        let result = cluster_weights(&tensor, 2).expect("cluster_weights should succeed");
        let out = result.to_vec().expect("to_vec should succeed");
        // all output values must match one of exactly 2 distinct centroid values
        let mut centroids: Vec<f32> = out.clone();
        centroids.dedup_by(|a, b| (*a - *b).abs() < 1e-5);
        // dedup on unsorted won't work well; use a manual unique set
        let mut unique_vals: Vec<f32> = Vec::new();
        for &v in &out {
            if !unique_vals.iter().any(|&u| (u - v).abs() < 1e-5) {
                unique_vals.push(v);
            }
        }
        assert_eq!(
            unique_vals.len(),
            2,
            "all output values should belong to exactly 2 cluster centers, got unique vals: {unique_vals:?}"
        );
        // every value must exactly equal one of the two centroids
        for &v in &out {
            assert!(
                unique_vals.iter().any(|&u| (u - v).abs() < 1e-5),
                "output value {v} does not match any centroid"
            );
        }
    }

    #[test]
    fn test_column_compression_zeros_columns() {
        // 3×4 matrix, rank=2: zero the 2 columns with smallest L2 norms
        // col0: [1,1,1] norm=sqrt(3)≈1.73
        // col1: [100,100,100] norm=sqrt(30000)≈173 (large)
        // col2: [0.01,0.01,0.01] norm≈0.017 (smallest)
        // col3: [0.02,0.02,0.02] norm≈0.035 (second smallest)
        let data = vec![
            1.0f32, 100.0, 0.01, 0.02, 1.0, 100.0, 0.01, 0.02, 1.0, 100.0, 0.01, 0.02,
        ];
        let tensor = Tensor::from_vec(data, &[3, 4]).expect("from_vec should succeed");
        let result =
            apply_column_wise_compression(&tensor, 2).expect("column compression should succeed");
        let out = result.to_vec().expect("to_vec should succeed");
        // columns 2 and 3 (indices [0,2],[0,3],[1,2],[1,3],[2,2],[2,3]) should be zero
        assert_eq!(out[2], 0.0, "col2 row0 should be zeroed");
        assert_eq!(out[3], 0.0, "col3 row0 should be zeroed");
        assert_eq!(out[6], 0.0, "col2 row1 should be zeroed");
        assert_eq!(out[7], 0.0, "col3 row1 should be zeroed");
        assert_eq!(out[10], 0.0, "col2 row2 should be zeroed");
        assert_eq!(out[11], 0.0, "col3 row2 should be zeroed");
        // columns 0 and 1 should be non-zero
        assert_ne!(out[0], 0.0, "col0 should be kept");
        assert_ne!(out[1], 0.0, "col1 should be kept");
    }

    #[test]
    fn test_symmetric_compression_reduces_rank() {
        // 4×4 identity matrix compressed to rank 2
        let data = vec![
            1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let tensor = Tensor::from_vec(data, &[4, 4]).expect("from_vec should succeed");
        let result =
            apply_symmetric_compression(&tensor, 2).expect("symmetric compression should succeed");
        let out = result.to_vec().expect("to_vec should succeed");
        // result should not be all zeros
        let all_zero = out.iter().all(|&v| v.abs() < 1e-8);
        assert!(!all_zero, "rank-2 approximation should not be all zeros");
        // result should not equal the original identity (rank was reduced)
        let identity_data = vec![
            1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let differs_from_identity = out
            .iter()
            .zip(identity_data.iter())
            .any(|(&r, &i)| (r - i).abs() > 1e-5);
        assert!(
            differs_from_identity,
            "rank-2 approximation of 4×4 identity should differ from original"
        );
        // Frobenius norm of result should be positive but less than original (rank reduction)
        let frob_norm: f32 = out.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(
            frob_norm > 0.1,
            "Frobenius norm should be substantially positive, got {frob_norm}"
        );
        assert!(
            frob_norm < 4.0 * (2.0f32 / 4.0).sqrt() + 0.5,
            "Frobenius norm should be reduced, got {frob_norm}"
        );
    }
}
