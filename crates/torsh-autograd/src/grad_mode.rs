//! Gradient computation mode management

use crate::{is_grad_enabled, set_grad_enabled};
// use scirs2::autograd as ag;

/// Inference mode guard - disables both gradient computation and graph building
pub struct InferenceModeGuard {
    prev_grad_state: bool,
    #[allow(dead_code)]
    prev_inference_mode: bool,
}

impl Default for InferenceModeGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceModeGuard {
    pub fn new() -> Self {
        let prev_grad_state = is_grad_enabled();
        let prev_inference_mode = false; // scirs2 doesn't have inference mode

        // Disable gradients (inference mode doesn't exist in scirs2)
        set_grad_enabled(false);

        Self {
            prev_grad_state,
            prev_inference_mode,
        }
    }
}

impl Drop for InferenceModeGuard {
    fn drop(&mut self) {
        // Restore previous state (scirs2 doesn't have inference mode)
        set_grad_enabled(self.prev_grad_state);
    }
}

/// Enter inference mode for maximum performance
pub fn inference_mode() -> InferenceModeGuard {
    InferenceModeGuard::new()
}

/// Set anomaly detection mode for debugging
pub fn set_anomaly_enabled(enabled: bool) {
    // Note: scirs2 doesn't have anomaly detection
    // This would need to be implemented in ToRSh
    let _ = enabled; // Suppress unused warning
}

/// Check if anomaly detection is enabled
pub fn is_anomaly_enabled() -> bool {
    // Note: scirs2 doesn't have anomaly detection
    // Return false as placeholder
    false
}

/// Detect anomalies in gradient computation
pub struct AnomalyModeGuard {
    prev_state: bool,
}

impl Default for AnomalyModeGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl AnomalyModeGuard {
    pub fn new() -> Self {
        let prev_state = is_anomaly_enabled();
        set_anomaly_enabled(true);
        Self { prev_state }
    }
}

impl Drop for AnomalyModeGuard {
    fn drop(&mut self) {
        set_anomaly_enabled(self.prev_state);
    }
}

/// Enable anomaly detection for debugging
pub fn detect_anomaly() -> AnomalyModeGuard {
    AnomalyModeGuard::new()
}

/// Graph mode settings
pub struct GraphMode {
    /// Whether to retain graph after backward
    pub retain_graph: bool,
    /// Whether to create graph for higher-order derivatives
    pub create_graph: bool,
    /// Whether to accumulate gradients
    pub accumulate_grad: bool,
}

impl Default for GraphMode {
    fn default() -> Self {
        Self {
            retain_graph: false,
            create_graph: false,
            accumulate_grad: true,
        }
    }
}

// Thread-local graph mode settings
thread_local! {
    static GRAPH_MODE: std::cell::RefCell<GraphMode> = std::cell::RefCell::new(GraphMode::default());
}

/// Set graph mode settings
pub fn set_graph_mode(mode: GraphMode) {
    GRAPH_MODE.with(|m| {
        *m.borrow_mut() = mode;
    });
}

/// Get current graph mode settings
pub fn get_graph_mode() -> GraphMode {
    GRAPH_MODE.with(|m| {
        let mode = m.borrow();
        GraphMode {
            retain_graph: mode.retain_graph,
            create_graph: mode.create_graph,
            accumulate_grad: mode.accumulate_grad,
        }
    })
}

/// Profiling mode for performance analysis
pub struct ProfilingModeGuard {
    #[allow(dead_code)]
    prev_state: bool,
}

impl Default for ProfilingModeGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfilingModeGuard {
    pub fn new() -> Self {
        let prev_state = false; // scirs2 doesn't have profiling functions
                                // Note: scirs2 doesn't have enable_profiling function
        Self { prev_state }
    }
}

impl Drop for ProfilingModeGuard {
    fn drop(&mut self) {
        // Note: scirs2 doesn't have profiling functions
        // Would need to implement custom profiling in ToRSh
    }
}

/// Enable profiling mode
pub fn profile() -> ProfilingModeGuard {
    ProfilingModeGuard::new()
}

/// Gradient clipping utilities
pub mod clip {
    use torsh_core::dtype::FloatElement;
    use torsh_tensor::Tensor;
    // Temporarily disable scirs2 integration
    // use scirs2::autograd::tensor_ops as T;

    /// Clip gradients by global norm
    pub fn clip_grad_norm<T: FloatElement>(
        tensors: &mut [Tensor<T>],
        max_norm: f32,
        norm_type: f32,
    ) -> f32 {
        // Temporarily disabled - would use scirs2 for gradient computation
        let _ = (tensors, max_norm, norm_type); // Suppress unused warnings

        // TODO: Implement proper gradient clipping when scirs2 integration is ready
        // For now, return a placeholder value
        0.0
    }

    /// Clip gradients by value
    pub fn clip_grad_value<T: FloatElement>(tensors: &mut [Tensor<T>], clip_value: f32) {
        // Temporarily disabled - would use scirs2 for gradient computation
        let _ = (tensors, clip_value); // Suppress unused warnings

        // TODO: Implement proper gradient clipping when scirs2 integration is ready
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_mode() {
        assert!(is_grad_enabled());

        {
            let _guard = inference_mode();
            assert!(!is_grad_enabled());
        }

        assert!(is_grad_enabled());
    }

    #[test]
    fn test_anomaly_mode() {
        // Anomaly detection is not implemented yet in scirs2
        // This test verifies that it doesn't crash
        assert!(!is_anomaly_enabled());

        {
            let _guard = detect_anomaly();
            // Anomaly detection always returns false for now
            assert!(!is_anomaly_enabled());
        }

        assert!(!is_anomaly_enabled());
    }
}
