//! Quantization-aware training (QAT) utilities

use super::context::QuantizationContext;
use super::types::QuantizationScheme;
use crate::{FxGraph, TorshResult};

/// Quantization-aware training (QAT) utilities
pub struct QATUtils;

impl QATUtils {
    /// Prepare a model for QAT
    pub fn prepare_qat(
        graph: &mut FxGraph,
        scheme: QuantizationScheme,
    ) -> TorshResult<QuantizationContext> {
        let mut context = QuantizationContext::new(scheme);
        context.prepare_qat(graph)?;
        Ok(context)
    }

    /// Convert QAT model to quantized model
    pub fn convert_qat_to_quantized(
        graph: &mut FxGraph,
        context: &QuantizationContext,
    ) -> TorshResult<()> {
        context.quantize_graph(graph)
    }
}
