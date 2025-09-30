//! Post-training quantization (PTQ) utilities

use super::context::QuantizationContext;
use super::types::{QuantizationAnnotation, QuantizationScheme};
use crate::{FxGraph, TorshResult};
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

/// Post-training quantization (PTQ) utilities
pub struct PTQUtils;

impl PTQUtils {
    /// Perform post-training quantization
    pub fn quantize_post_training(
        graph: &mut FxGraph,
        calibration_data: HashMap<NodeIndex, Vec<f32>>,
        scheme: QuantizationScheme,
    ) -> TorshResult<QuantizationContext> {
        let mut context = QuantizationContext::new(scheme);

        // Calibrate each node
        for (node_idx, data) in calibration_data {
            context.start_calibration(node_idx);
            context.update_calibration(node_idx, &data)?;
            let params = context.finalize_calibration(node_idx)?;

            let annotation = QuantizationAnnotation {
                input_params: vec![Some(params.clone())],
                output_params: Some(params),
                calibration_data: None,
            };
            context.annotate_node(node_idx, annotation);
        }

        // Quantize the graph
        context.quantize_graph(graph)?;
        Ok(context)
    }
}
