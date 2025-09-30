//! Gradient computation mode management

use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::sync::{Arc, Mutex};
use torsh_core::error::{Result, TorshError};

/// Global gradient mode state
#[derive(Debug, Clone)]
struct GradMode {
    /// Stack of gradient states for nested contexts
    enabled_stack: Vec<bool>,
    /// Current gradient enabled state
    current_enabled: bool,
}

impl GradMode {
    fn new() -> Self {
        Self {
            enabled_stack: Vec::new(),
            current_enabled: true, // Default: gradients enabled
        }
    }

    fn push(&mut self, enabled: bool) {
        self.enabled_stack.push(self.current_enabled);
        self.current_enabled = enabled;
    }

    fn pop(&mut self) {
        if let Some(prev_enabled) = self.enabled_stack.pop() {
            self.current_enabled = prev_enabled;
        }
    }

    fn is_enabled(&self) -> bool {
        self.current_enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.current_enabled = enabled;
        // Clear stack when explicitly setting
        self.enabled_stack.clear();
    }
}

/// Global gradient mode state
static GRAD_MODE: Lazy<Arc<RwLock<GradMode>>> =
    Lazy::new(|| Arc::new(RwLock::new(GradMode::new())));

/// Check if gradient computation is currently enabled
pub fn is_grad_enabled() -> bool {
    GRAD_MODE.read().is_enabled()
}

/// Set gradient computation mode
pub fn set_grad_enabled(enabled: bool) {
    GRAD_MODE.write().set_enabled(enabled);
}

/// Push a new gradient state onto the stack
pub fn push_grad_enabled(enabled: bool) {
    GRAD_MODE.write().push(enabled);
}

/// Pop the most recent gradient state from the stack
pub fn pop_grad_enabled() {
    GRAD_MODE.write().pop();
}

/// Execute a function with a specific gradient mode
pub fn with_grad_mode<F, R>(enabled: bool, f: F) -> R
where
    F: FnOnce() -> R,
{
    push_grad_enabled(enabled);
    let result = f();
    pop_grad_enabled();
    result
}

/// Inference mode guard - disables both gradient computation and graph building for zero overhead
pub struct InferenceModeGuard {
    prev_inference_mode: bool,
    #[allow(dead_code)]
    prev_grad_enabled: bool,
}

impl Default for InferenceModeGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceModeGuard {
    pub fn new() -> Self {
        let prev_inference_mode = is_inference_mode();
        let prev_grad_enabled = is_grad_enabled();

        // Set inference mode and disable gradients for zero overhead
        set_inference_mode(true);
        push_grad_enabled(false);

        Self {
            prev_inference_mode,
            prev_grad_enabled,
        }
    }
}

impl Drop for InferenceModeGuard {
    fn drop(&mut self) {
        // Restore previous gradient state using stack mechanism
        pop_grad_enabled();
        // Restore previous inference mode
        set_inference_mode(self.prev_inference_mode);
    }
}

// Thread-local inference mode state for zero-overhead optimization
thread_local! {
    static INFERENCE_MODE: std::cell::RefCell<bool> = const { std::cell::RefCell::new(false) };
}

/// Check if we're in inference mode (zero-overhead autograd)
pub fn is_inference_mode() -> bool {
    INFERENCE_MODE.with(|mode| *mode.borrow())
}

/// Set inference mode for zero-overhead inference
pub fn set_inference_mode(enabled: bool) {
    INFERENCE_MODE.with(|mode| {
        *mode.borrow_mut() = enabled;
    });
}

/// Enter inference mode for maximum performance
pub fn inference_mode() -> InferenceModeGuard {
    InferenceModeGuard::new()
}

/// No-grad guard - disables gradient computation for performance
pub struct NoGradGuard {
    #[allow(dead_code)]
    prev_grad_enabled: bool,
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl NoGradGuard {
    pub fn new() -> Self {
        let prev_grad_enabled = is_grad_enabled();
        push_grad_enabled(false);

        Self { prev_grad_enabled }
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        pop_grad_enabled();
    }
}

/// Enter no-grad mode for zero-overhead inference
pub fn no_grad() -> NoGradGuard {
    NoGradGuard::new()
}

/// Execute a closure with gradients disabled
pub fn with_no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = no_grad();
    f()
}

/// Execute a closure with inference mode enabled
pub fn with_inference_mode<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = inference_mode();
    f()
}

// Thread-local anomaly detection state
thread_local! {
    static ANOMALY_ENABLED: std::cell::RefCell<bool> = const { std::cell::RefCell::new(false) };
}

/// Set anomaly detection mode for debugging
pub fn set_anomaly_enabled(enabled: bool) {
    ANOMALY_ENABLED.with(|a| {
        *a.borrow_mut() = enabled;
    });
}

/// Check if anomaly detection is enabled
pub fn is_anomaly_enabled() -> bool {
    ANOMALY_ENABLED.with(|a| *a.borrow())
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

/// Zero-overhead inference optimizations
pub mod zero_overhead {
    use super::*;

    thread_local! {
        static ZERO_OVERHEAD_MODE: std::cell::RefCell<bool> = const { std::cell::RefCell::new(false) };
        static OPERATION_COUNT: std::cell::RefCell<u64> = const { std::cell::RefCell::new(0) };
        static MEMORY_ALLOCATION_COUNT: std::cell::RefCell<u64> = const { std::cell::RefCell::new(0) };
    }

    /// Check if zero-overhead mode is enabled
    pub fn is_zero_overhead_enabled() -> bool {
        ZERO_OVERHEAD_MODE.with(|mode| *mode.borrow())
    }

    /// Enable/disable zero-overhead mode
    pub fn set_zero_overhead_mode(enabled: bool) {
        ZERO_OVERHEAD_MODE.with(|mode| {
            *mode.borrow_mut() = enabled;
        });

        if enabled {
            // Reset counters when enabling zero-overhead mode
            OPERATION_COUNT.with(|count| *count.borrow_mut() = 0);
            MEMORY_ALLOCATION_COUNT.with(|count| *count.borrow_mut() = 0);
            tracing::debug!("Zero-overhead autograd mode enabled");
        } else {
            tracing::debug!("Zero-overhead autograd mode disabled");
        }
    }

    /// Increment operation count (for profiling zero-overhead mode)
    #[inline(always)]
    pub fn increment_operation_count() {
        if cfg!(debug_assertions) && is_zero_overhead_enabled() {
            OPERATION_COUNT.with(|count| {
                *count.borrow_mut() += 1;
            });
        }
    }

    /// Increment memory allocation count (for profiling zero-overhead mode)
    #[inline(always)]
    pub fn increment_memory_allocation_count() {
        if cfg!(debug_assertions) && is_zero_overhead_enabled() {
            MEMORY_ALLOCATION_COUNT.with(|count| {
                *count.borrow_mut() += 1;
            });
        }
    }

    /// Get operation count since zero-overhead mode was enabled
    pub fn get_operation_count() -> u64 {
        OPERATION_COUNT.with(|count| *count.borrow())
    }

    /// Get memory allocation count since zero-overhead mode was enabled
    pub fn get_memory_allocation_count() -> u64 {
        MEMORY_ALLOCATION_COUNT.with(|count| *count.borrow())
    }

    /// Zero-overhead guard that completely bypasses autograd
    pub struct ZeroOverheadGuard {
        prev_inference_mode: bool,
        #[allow(dead_code)]
        prev_grad_enabled: bool,
        prev_zero_overhead: bool,
    }

    impl Default for ZeroOverheadGuard {
        fn default() -> Self {
            Self::new()
        }
    }

    impl ZeroOverheadGuard {
        pub fn new() -> Self {
            let prev_inference_mode = is_inference_mode();
            let prev_grad_enabled = is_grad_enabled();
            let prev_zero_overhead = is_zero_overhead_enabled();

            // Enable all optimization modes
            set_inference_mode(true);
            push_grad_enabled(false);
            set_zero_overhead_mode(true);

            Self {
                prev_inference_mode,
                prev_grad_enabled,
                prev_zero_overhead,
            }
        }

        /// Get performance stats for this zero-overhead session
        pub fn get_stats(&self) -> ZeroOverheadStats {
            ZeroOverheadStats {
                operation_count: get_operation_count(),
                memory_allocation_count: get_memory_allocation_count(),
            }
        }
    }

    impl Drop for ZeroOverheadGuard {
        fn drop(&mut self) {
            // Restore previous states
            pop_grad_enabled();
            set_inference_mode(self.prev_inference_mode);
            set_zero_overhead_mode(self.prev_zero_overhead);
        }
    }

    /// Performance statistics for zero-overhead mode
    #[derive(Debug, Clone)]
    pub struct ZeroOverheadStats {
        pub operation_count: u64,
        pub memory_allocation_count: u64,
    }

    impl ZeroOverheadStats {
        /// Check if the mode was truly zero-overhead
        pub fn is_zero_overhead(&self) -> bool {
            self.operation_count == 0 && self.memory_allocation_count == 0
        }

        /// Get efficiency score (closer to 1.0 is better)
        pub fn efficiency_score(&self) -> f64 {
            let total_ops = self.operation_count + self.memory_allocation_count;
            if total_ops == 0 {
                1.0 // Perfect efficiency
            } else {
                1.0 / (1.0 + total_ops as f64)
            }
        }
    }

    /// Enter zero-overhead mode
    pub fn zero_overhead() -> ZeroOverheadGuard {
        ZeroOverheadGuard::new()
    }

    /// Execute a closure with zero-overhead autograd
    pub fn with_zero_overhead<F, R>(f: F) -> (R, ZeroOverheadStats)
    where
        F: FnOnce() -> R,
    {
        let guard = zero_overhead();
        let result = f();
        let stats = guard.get_stats();
        (result, stats)
    }

    /// Macro for zero-overhead tensor operations
    #[macro_export]
    macro_rules! zero_overhead_op {
        ($expr:expr) => {{
            use $crate::grad_mode::zero_overhead;
            if zero_overhead::is_zero_overhead_enabled() {
                // In zero-overhead mode, bypass all autograd tracking
                $expr
            } else {
                // Normal autograd mode
                zero_overhead::increment_operation_count();
                $expr
            }
        }};
    }

    /// Compile-time optimization hints for zero-overhead mode
    pub mod hints {
        /// Hint that this operation should be optimized away in zero-overhead mode
        #[inline(always)]
        pub fn likely_zero_overhead() {
            if super::is_zero_overhead_enabled() {
                // Compiler hint that this branch is likely
                std::hint::black_box(());
            }
        }

        /// Hint that this operation involves gradient computation
        #[inline(always)]
        pub fn unlikely_in_inference() {
            if !super::is_zero_overhead_enabled() {
                // Compiler hint that this branch is unlikely in inference
                std::hint::black_box(());
            }
        }
    }
}

/// Lazy gradient computation support
pub mod lazy {
    use parking_lot::Mutex;
    use std::sync::Arc;

    thread_local! {
        static LAZY_GRAD_ENABLED: std::cell::RefCell<bool> = const { std::cell::RefCell::new(false) };
    }

    /// Check if lazy gradient computation is enabled
    pub fn is_lazy_grad_enabled() -> bool {
        LAZY_GRAD_ENABLED.with(|enabled| *enabled.borrow())
    }

    /// Enable/disable lazy gradient computation
    pub fn set_lazy_grad_enabled(enabled: bool) {
        LAZY_GRAD_ENABLED.with(|lazy_enabled| {
            *lazy_enabled.borrow_mut() = enabled;
        });

        if enabled {
            tracing::debug!("Lazy gradient computation enabled");
        } else {
            tracing::debug!("Lazy gradient computation disabled");
        }
    }

    /// Lazy gradient computation guard
    pub struct LazyGradGuard {
        prev_state: bool,
    }

    impl Default for LazyGradGuard {
        fn default() -> Self {
            Self::new()
        }
    }

    impl LazyGradGuard {
        pub fn new() -> Self {
            let prev_state = is_lazy_grad_enabled();
            set_lazy_grad_enabled(true);
            Self { prev_state }
        }
    }

    impl Drop for LazyGradGuard {
        fn drop(&mut self) {
            set_lazy_grad_enabled(self.prev_state);
        }
    }

    /// Lazy gradient computation closure
    pub type LazyGradComputation<T> =
        Arc<Mutex<Option<Box<dyn Fn() -> Result<T, torsh_core::error::TorshError> + Send + Sync>>>>;

    /// Lazy gradient holder
    pub struct LazyGradient<T> {
        computation: LazyGradComputation<T>,
        is_computed: Arc<Mutex<bool>>,
        cached_result: Arc<Mutex<Option<T>>>,
    }

    impl<T> LazyGradient<T>
    where
        T: Clone + Send + Sync + 'static,
    {
        /// Create a new lazy gradient
        pub fn new<F>(computation: F) -> Self
        where
            F: Fn() -> Result<T, torsh_core::error::TorshError> + Send + Sync + 'static,
        {
            Self {
                computation: Arc::new(Mutex::new(Some(Box::new(computation)))),
                is_computed: Arc::new(Mutex::new(false)),
                cached_result: Arc::new(Mutex::new(None)),
            }
        }

        /// Compute the gradient if not already computed
        pub fn compute(&self) -> Result<T, torsh_core::error::TorshError> {
            let mut is_computed = self.is_computed.lock();
            if *is_computed {
                // Return cached result
                let cached = self.cached_result.lock();
                if let Some(ref result) = *cached {
                    return Ok(result.clone());
                }
            }

            // Compute gradient
            let mut computation = self.computation.lock();
            if let Some(compute_fn) = computation.take() {
                tracing::debug!("Computing lazy gradient");
                let result = compute_fn()?;

                // Cache result
                *self.cached_result.lock() = Some(result.clone());
                *is_computed = true;

                Ok(result)
            } else {
                Err(torsh_core::error::TorshError::AutogradError(
                    "Lazy gradient computation already consumed".to_string(),
                ))
            }
        }

        /// Check if gradient has been computed
        pub fn is_computed(&self) -> bool {
            *self.is_computed.lock()
        }

        /// Force evaluation of the lazy gradient
        pub fn force(&self) -> Result<T, torsh_core::error::TorshError> {
            self.compute()
        }
    }

    impl<T> Clone for LazyGradient<T>
    where
        T: Clone,
    {
        fn clone(&self) -> Self {
            Self {
                computation: Arc::clone(&self.computation),
                is_computed: Arc::clone(&self.is_computed),
                cached_result: Arc::clone(&self.cached_result),
            }
        }
    }

    /// Enter lazy gradient computation mode
    pub fn lazy_grad() -> LazyGradGuard {
        LazyGradGuard::new()
    }

    /// Execute a closure with lazy gradient computation
    pub fn with_lazy_grad<F, R>(f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let _guard = lazy_grad();
        f()
    }
}

/* TODO: Re-enable gradient clipping once tensor integration is complete
/// Gradient clipping utilities
pub mod clip {
    use torsh_core::dtype::FloatElement;
    // use torsh_tensor::Tensor;  // Commented out to avoid circular dependency
    // use crate::AutogradTensor; // Commented out - trait is generic
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
*/

/// Enable gradient computation context guard
pub struct EnableGradGuard {
    #[allow(dead_code)]
    prev_grad_enabled: bool,
}

impl Default for EnableGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl EnableGradGuard {
    pub fn new() -> Self {
        let prev_grad_enabled = is_grad_enabled();
        push_grad_enabled(true);

        Self { prev_grad_enabled }
    }
}

impl Drop for EnableGradGuard {
    fn drop(&mut self) {
        pop_grad_enabled();
    }
}

/// Enable gradient computation
pub fn enable_grad() -> EnableGradGuard {
    EnableGradGuard::new()
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
        assert!(!is_anomaly_enabled());

        {
            let _guard = detect_anomaly();
            assert!(is_anomaly_enabled());
        }

        assert!(!is_anomaly_enabled());
    }
}
