//! Quantization context for managing quantization state

use super::types::{
    CalibrationData, QuantizationAnnotation, QuantizationParams, QuantizationScheme,
};
use crate::{FxGraph, Node, TorshResult};
use petgraph::graph::NodeIndex;
use std::collections::HashMap;
use torsh_core::error::TorshError;

/// Quantization context for managing quantization state
pub struct QuantizationContext {
    annotations: HashMap<NodeIndex, QuantizationAnnotation>,
    calibration_data: HashMap<NodeIndex, CalibrationData>,
    global_scheme: QuantizationScheme,
}

impl QuantizationContext {
    /// Create new quantization context
    pub fn new(scheme: QuantizationScheme) -> Self {
        Self {
            annotations: HashMap::new(),
            calibration_data: HashMap::new(),
            global_scheme: scheme,
        }
    }

    /// Add quantization annotation for a node
    pub fn annotate_node(&mut self, node: NodeIndex, annotation: QuantizationAnnotation) {
        self.annotations.insert(node, annotation);
    }

    /// Get all annotations (for testing)
    #[cfg(test)]
    pub fn annotations(&self) -> &HashMap<NodeIndex, QuantizationAnnotation> {
        &self.annotations
    }

    /// Get quantization annotation for a node
    pub fn get_annotation(&self, node: NodeIndex) -> Option<&QuantizationAnnotation> {
        self.annotations.get(&node)
    }

    /// Start calibration for a node
    pub fn start_calibration(&mut self, node: NodeIndex) {
        self.calibration_data.insert(node, CalibrationData::new());
    }

    /// Update calibration data for a node
    pub fn update_calibration(&mut self, node: NodeIndex, values: &[f32]) -> TorshResult<()> {
        if let Some(data) = self.calibration_data.get_mut(&node) {
            data.update(values);
            Ok(())
        } else {
            Err(TorshError::InvalidArgument(format!(
                "Calibration not started for node {:?}",
                node
            )))
        }
    }

    /// Finalize calibration and compute quantization parameters
    pub fn finalize_calibration(&mut self, node: NodeIndex) -> TorshResult<QuantizationParams> {
        if let Some(data) = self.calibration_data.remove(&node) {
            Ok(data.compute_params(self.global_scheme))
        } else {
            Err(TorshError::InvalidArgument(format!(
                "No calibration data for node {:?}",
                node
            )))
        }
    }

    /// Prepare graph for quantization-aware training (QAT)
    pub fn prepare_qat(&mut self, graph: &mut FxGraph) -> TorshResult<()> {
        // Insert fake quantize nodes before operations that benefit from quantization
        let mut insertions = Vec::new();

        for (idx, node) in graph.nodes() {
            if let Node::Call(op_name, _) = node {
                if self.should_quantize_operation(op_name) {
                    insertions.push(idx);
                }
            }
        }

        // Insert fake quantize nodes
        for node_idx in insertions {
            self.insert_fake_quantize_node(graph, node_idx)?;
        }

        Ok(())
    }

    /// Convert graph to quantized format
    pub fn quantize_graph(&self, graph: &mut FxGraph) -> TorshResult<()> {
        // Replace operations with quantized versions
        let mut replacements = Vec::new();

        for (idx, node) in graph.nodes() {
            if let Node::Call(op_name, args) = node {
                if self.should_quantize_operation(op_name) {
                    let quantized_op = self.get_quantized_operation(op_name);
                    replacements.push((idx, quantized_op, args.clone()));
                }
            }
        }

        // Apply replacements
        for (idx, quantized_op, args) in replacements {
            if let Some(node) = graph.graph.node_weight_mut(idx) {
                *node = Node::Call(quantized_op, args);
            }
        }

        Ok(())
    }

    /// Check if an operation should be quantized
    fn should_quantize_operation(&self, op_name: &str) -> bool {
        matches!(op_name, "linear" | "conv2d" | "matmul" | "add" | "mul")
    }

    /// Get quantized version of an operation
    fn get_quantized_operation(&self, op_name: &str) -> String {
        match op_name {
            "linear" => "quantized_linear".to_string(),
            "conv2d" => "quantized_conv2d".to_string(),
            "matmul" => "quantized_matmul".to_string(),
            "add" => "quantized_add".to_string(),
            "mul" => "quantized_mul".to_string(),
            _ => format!("quantized_{op_name}"),
        }
    }

    /// Insert fake quantize node for QAT
    fn insert_fake_quantize_node(
        &mut self,
        _graph: &mut FxGraph,
        target_idx: NodeIndex,
    ) -> TorshResult<()> {
        // In a full implementation, this would insert a FakeQuantize node
        // before the target operation. For now, we'll just annotate the node.

        let annotation = QuantizationAnnotation {
            input_params: vec![Some(QuantizationParams::symmetric(self.global_scheme, 1.0))],
            output_params: Some(QuantizationParams::symmetric(self.global_scheme, 1.0)),
            calibration_data: None,
        };

        self.annotate_node(target_idx, annotation);
        Ok(())
    }
}
