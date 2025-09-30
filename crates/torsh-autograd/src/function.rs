//! Enhanced autograd function framework with custom function support
//!
//! This module provides a comprehensive framework for defining custom autograd functions,
//! including forward and backward pass implementations, function composition, and
//! automatic differentiation through user-defined operations.

use torsh_core::error::{Result, TorshError};
// AutogradTensor trait is available through crate root - it's generic
use crate::AutogradTensor;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;

/// Type-erased trait for custom autograd functions - dyn-compatible version
pub trait DynFunction: Send + Sync {
    /// Name of the function for debugging and profiling
    fn name(&self) -> &str;

    /// Whether this function is differentiable
    fn is_differentiable(&self) -> bool {
        true
    }

    /// Memory complexity hint for optimization
    fn memory_complexity(&self) -> MemoryComplexity {
        MemoryComplexity::Linear
    }

    /// Computational complexity hint for optimization
    fn computational_complexity(&self) -> ComputationalComplexity {
        ComputationalComplexity::Linear
    }

    /// Whether this function can be fused with others
    fn is_fusable(&self) -> bool {
        false
    }

    /// Get function metadata for optimization
    fn metadata(&self) -> FunctionMetadata {
        FunctionMetadata {
            name: self.name().to_string(),
            is_differentiable: self.is_differentiable(),
            memory_complexity: self.memory_complexity(),
            computational_complexity: self.computational_complexity(),
            is_fusable: self.is_fusable(),
            version: "1.0.0".to_string(),
            description: "Custom autograd function".to_string(),
            author: "Unknown".to_string(),
            created_at: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                .to_string(),
            checksum: "".to_string(),
            dependencies: vec![],
        }
    }
}

/// Trait for custom autograd functions with generic methods
pub trait Function: Send + Sync + DynFunction {
    /// Forward pass computation
    fn forward<T>(
        &self,
        ctx: &mut FunctionContext,
        inputs: &[&dyn AutogradTensor<T>],
    ) -> Result<Vec<Box<dyn AutogradTensor<T>>>>
    where
        T: torsh_core::dtype::TensorElement;

    /// Backward pass computation
    fn backward<T>(
        &self,
        ctx: &mut FunctionContext,
        grad_outputs: &[&dyn AutogradTensor<T>],
    ) -> Result<Vec<Option<Box<dyn AutogradTensor<T>>>>>
    where
        T: torsh_core::dtype::TensorElement;
}

/// Trait for non-differentiable functions that have subgradients
pub trait SubgradientFunction: Send + Sync + DynFunction {
    /// Forward pass computation
    fn forward<T>(
        &self,
        ctx: &mut FunctionContext,
        inputs: &[&dyn AutogradTensor<T>],
    ) -> Result<Vec<Box<dyn AutogradTensor<T>>>>
    where
        T: torsh_core::dtype::TensorElement + num_traits::Float;

    /// Subgradient computation for non-differentiable operations
    /// Returns a set of possible subgradients
    fn subgradient<T>(
        &self,
        ctx: &mut FunctionContext,
        grad_outputs: &[&dyn AutogradTensor<T>],
    ) -> Result<Vec<Option<SubgradientSet<T>>>>
    where
        T: torsh_core::dtype::TensorElement + num_traits::Float;
}

/// Set of subgradients for non-differentiable functions
pub struct SubgradientSet<T: torsh_core::dtype::TensorElement> {
    /// Primary subgradient (commonly used one)
    pub primary: Box<dyn AutogradTensor<T>>,
    /// Alternative subgradients
    pub alternatives: Vec<Box<dyn AutogradTensor<T>>>,
    /// Selection strategy for choosing subgradient
    pub selection_strategy: SubgradientSelection,
}

impl<T: torsh_core::dtype::TensorElement> Clone for SubgradientSet<T> {
    fn clone(&self) -> Self {
        Self {
            primary: self.primary.clone_tensor(),
            alternatives: self.alternatives.iter().map(|t| t.clone_tensor()).collect(),
            selection_strategy: self.selection_strategy,
        }
    }
}

impl<T: torsh_core::dtype::TensorElement> std::fmt::Debug for SubgradientSet<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubgradientSet")
            .field(
                "primary",
                &format!("Box<dyn AutogradTensor<{}>>", std::any::type_name::<T>()),
            )
            .field(
                "alternatives",
                &format!(
                    "Vec<Box<dyn AutogradTensor<{}>>> (len: {})",
                    std::any::type_name::<T>(),
                    self.alternatives.len()
                ),
            )
            .field("selection_strategy", &self.selection_strategy)
            .finish()
    }
}

/// Strategy for selecting subgradients in non-differentiable operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubgradientSelection {
    /// Always use the primary subgradient
    Primary,
    /// Choose randomly from available subgradients
    Random,
    /// Use the subgradient with minimum norm
    MinNorm,
    /// Use the subgradient with maximum norm
    MaxNorm,
    /// Use Clarke subgradient (convex hull)
    Clarke,
    /// Use generalized gradient (for locally Lipschitz functions)
    Generalized,
}

/// Memory complexity categories for function optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryComplexity {
    Constant,
    Linear,
    Quadratic,
    Exponential,
}

/// Computational complexity categories for function optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ComputationalComplexity {
    Constant,
    Linear,
    LogLinear,
    Quadratic,
    Cubic,
    Exponential,
}

/// Function metadata for optimization and debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionMetadata {
    pub name: String,
    pub is_differentiable: bool,
    pub memory_complexity: MemoryComplexity,
    pub computational_complexity: ComputationalComplexity,
    pub is_fusable: bool,
    pub version: String,
    pub description: String,
    pub author: String,
    pub created_at: String,
    pub checksum: String,
    pub dependencies: Vec<String>,
}

/// Context for storing values between forward and backward passes
pub struct FunctionContext {
    /// Saved tensors for backward pass
    #[allow(dead_code)]
    saved_tensors: Vec<Box<dyn Any + Send + Sync>>,
    /// Saved scalar values for backward pass
    saved_values: Vec<Box<dyn Any + Send + Sync>>,
    /// Whether to materialize gradients for non-differentiable tensors
    materialize_grads: bool,
    /// Unique context ID for debugging
    context_id: usize,
    /// Function name for debugging
    function_name: String,
}

impl Default for FunctionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl FunctionContext {
    /// Create a new function context
    pub fn new() -> Self {
        static CONTEXT_COUNTER: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        Self {
            saved_tensors: Vec::new(),
            saved_values: Vec::new(),
            materialize_grads: true,
            context_id: CONTEXT_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            function_name: "unknown".to_string(),
        }
    }

    /// Create a new function context with a specific name
    pub fn with_name(name: String) -> Self {
        let mut ctx = Self::new();
        ctx.function_name = name;
        ctx
    }

    /// Save arbitrary values for backward pass
    pub fn save_value<V: Any + Send + Sync + 'static>(&mut self, value: V) {
        self.saved_values.push(Box::new(value));
    }

    /// Get saved value by index
    pub fn get_saved_value<V: Any + 'static>(&self, index: usize) -> Result<&V> {
        self.saved_values
            .get(index)
            .and_then(|v| v.downcast_ref::<V>())
            .ok_or_else(|| {
                TorshError::AutogradError(format!(
                    "Saved value at index {} not found or type mismatch in context {}",
                    index, self.context_id
                ))
            })
    }

    /// Get the context ID
    pub fn context_id(&self) -> usize {
        self.context_id
    }

    /// Get the function name
    pub fn function_name(&self) -> &str {
        &self.function_name
    }

    /// Set whether to materialize gradients
    pub fn set_materialize_grads(&mut self, materialize: bool) {
        self.materialize_grads = materialize;
    }

    /// Check if gradients should be materialized
    pub fn should_materialize_grads(&self) -> bool {
        self.materialize_grads
    }
}

/// Function registry for managing custom functions
pub struct FunctionRegistry {
    functions: RwLock<HashMap<String, Arc<dyn DynFunction>>>,
}

impl Default for FunctionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl FunctionRegistry {
    /// Create a new function registry
    pub fn new() -> Self {
        Self {
            functions: RwLock::new(HashMap::new()),
        }
    }

    /// Register a custom function
    pub fn register<F>(&self, name: String, function: F) -> Result<()>
    where
        F: Function + 'static,
    {
        let mut functions = self.functions.write();
        if functions.contains_key(&name) {
            return Err(TorshError::AutogradError(format!(
                "Function '{name}' is already registered"
            )));
        }
        functions.insert(name, Arc::new(function));
        Ok(())
    }

    /// Get a registered function
    pub fn get(&self, name: &str) -> Option<Arc<dyn DynFunction>> {
        self.functions.read().get(name).cloned()
    }

    /// List all registered function names
    pub fn list_functions(&self) -> Vec<String> {
        self.functions.read().keys().cloned().collect()
    }

    /// Unregister a function
    pub fn unregister(&self, name: &str) -> bool {
        self.functions.write().remove(name).is_some()
    }

    /// Get function metadata
    pub fn get_metadata(&self, name: &str) -> Option<FunctionMetadata> {
        self.functions.read().get(name).map(|f| f.metadata())
    }
}

// Global function registry
static GLOBAL_REGISTRY: std::sync::OnceLock<FunctionRegistry> = std::sync::OnceLock::new();

/// Get the global function registry
pub fn global_registry() -> &'static FunctionRegistry {
    GLOBAL_REGISTRY.get_or_init(FunctionRegistry::new)
}

/// Register a function globally
pub fn register_function<F>(name: String, function: F) -> Result<()>
where
    F: Function + 'static,
{
    global_registry().register(name, function)
}

/// Apply a registered function by name
/// Note: This function is limited to metadata operations only due to type erasure
pub fn get_function_metadata(name: &str) -> Result<FunctionMetadata> {
    let function = global_registry()
        .get(name)
        .ok_or_else(|| TorshError::AutogradError(format!("Function '{name}' not found")))?;

    Ok(function.metadata())
}

/// Function composition utilities
pub mod composition {
    use super::*;

    /// Composed function that applies multiple functions in sequence
    /// Note: Due to type erasure limitations, this is a placeholder structure
    pub struct ComposedFunction {
        #[allow(dead_code)]
        function_names: Vec<String>,
        name: String,
    }

    impl ComposedFunction {
        /// Create a new composed function from function names
        pub fn new(function_names: Vec<String>) -> Self {
            let name = format!("compose({})", function_names.join(", "));
            Self {
                function_names,
                name,
            }
        }
    }

    impl DynFunction for ComposedFunction {
        fn name(&self) -> &str {
            &self.name
        }

        fn is_differentiable(&self) -> bool {
            // For simplicity, assume composed functions are differentiable
            true
        }

        fn memory_complexity(&self) -> MemoryComplexity {
            // Conservative estimate
            MemoryComplexity::Linear
        }

        fn computational_complexity(&self) -> ComputationalComplexity {
            // Conservative estimate
            ComputationalComplexity::Linear
        }
    }

    impl Function for ComposedFunction {
        fn forward<T>(
            &self,
            _ctx: &mut FunctionContext,
            _inputs: &[&dyn AutogradTensor<T>],
        ) -> Result<Vec<Box<dyn AutogradTensor<T>>>>
        where
            T: torsh_core::dtype::TensorElement,
        {
            // Due to type erasure limitations, function composition is not implemented
            Err(TorshError::AutogradError(
                "Function composition with type erasure is not supported".to_string(),
            ))
        }

        fn backward<T>(
            &self,
            _ctx: &mut FunctionContext,
            _grad_outputs: &[&dyn AutogradTensor<T>],
        ) -> Result<Vec<Option<Box<dyn AutogradTensor<T>>>>>
        where
            T: torsh_core::dtype::TensorElement,
        {
            // Due to type erasure limitations, function composition is not implemented
            Err(TorshError::AutogradError(
                "Function composition with type erasure is not supported".to_string(),
            ))
        }
    }

    /// Compose multiple functions into a single function by name
    pub fn compose(function_names: Vec<String>) -> ComposedFunction {
        ComposedFunction::new(function_names)
    }
}

/// Example function implementations
pub mod examples {
    use super::*;

    /// Example: Scaled addition function (a + scale * b)
    #[derive(Debug)]
    pub struct ScaledAdd {
        pub scale: f32,
    }

    impl DynFunction for ScaledAdd {
        fn name(&self) -> &str {
            "ScaledAdd"
        }

        fn is_fusable(&self) -> bool {
            true // Element-wise operations can often be fused
        }
    }

    impl Function for ScaledAdd {
        fn forward<T>(
            &self,
            ctx: &mut FunctionContext,
            inputs: &[&dyn AutogradTensor<T>],
        ) -> Result<Vec<Box<dyn AutogradTensor<T>>>>
        where
            T: torsh_core::dtype::TensorElement,
        {
            if inputs.len() != 2 {
                return Err(TorshError::AutogradError(
                    "ScaledAdd expects exactly two inputs".to_string(),
                ));
            }

            // Save scale for backward pass
            ctx.save_value(self.scale);

            // TODO: Implement actual tensor operations
            // For now, return placeholder error
            Err(TorshError::AutogradError(
                "ScaledAdd forward pass not yet implemented".to_string(),
            ))
        }

        fn backward<T>(
            &self,
            ctx: &mut FunctionContext,
            grad_outputs: &[&dyn AutogradTensor<T>],
        ) -> Result<Vec<Option<Box<dyn AutogradTensor<T>>>>>
        where
            T: torsh_core::dtype::TensorElement,
        {
            if grad_outputs.len() != 1 {
                return Err(TorshError::AutogradError(
                    "ScaledAdd backward expects exactly one gradient output".to_string(),
                ));
            }

            let _scale: &f32 = ctx.get_saved_value(0)?;
            let grad_output = grad_outputs[0];

            // Gradients: da = grad_output, db = scale * grad_output
            let grad_a = Some(grad_output.clone_tensor());

            // TODO: Implement actual tensor scaling
            // For now, return cloned gradient
            let grad_b = Some(grad_output.clone_tensor());

            Ok(vec![grad_a, grad_b])
        }
    }
}

/// Helper macro for defining custom functions
#[macro_export]
macro_rules! define_custom_function {
    (
        $name:ident,
        forward: $forward:expr,
        backward: $backward:expr
    ) => {
        #[derive(Debug)]
        pub struct $name;

        impl $crate::function::Function for $name {
            fn forward<T>(
                &self,
                ctx: &mut $crate::function::FunctionContext,
                inputs: &[&dyn $crate::AutogradTensor<T>],
            ) -> $crate::Result<Vec<Box<dyn $crate::AutogradTensor<T>>>>
            where
                T: torsh_core::dtype::TensorElement,
            {
                $forward(ctx, inputs)
            }

            fn backward<T>(
                &self,
                ctx: &mut $crate::function::FunctionContext,
                grad_outputs: &[&dyn $crate::AutogradTensor<T>],
            ) -> $crate::Result<Vec<Option<Box<dyn $crate::AutogradTensor<T>>>>>
            where
                T: torsh_core::dtype::TensorElement,
            {
                $backward(ctx, grad_outputs)
            }

            fn name(&self) -> &str {
                stringify!($name)
            }
        }
    };
}

/// Common non-differentiable functions with subgradient support
pub mod subgradient_functions {
    use super::*;

    /// Absolute value function: f(x) = |x|
    /// Subgradient: ∂f(x) = {-1 if x < 0, [-1, 1] if x = 0, 1 if x > 0}
    #[derive(Debug)]
    pub struct AbsFunction;

    impl DynFunction for AbsFunction {
        fn name(&self) -> &str {
            "abs"
        }
        fn is_differentiable(&self) -> bool {
            false
        }
    }

    impl SubgradientFunction for AbsFunction {
        fn forward<T>(
            &self,
            _ctx: &mut FunctionContext,
            inputs: &[&dyn AutogradTensor<T>],
        ) -> Result<Vec<Box<dyn AutogradTensor<T>>>>
        where
            T: torsh_core::dtype::TensorElement + num_traits::Float,
        {
            if inputs.len() != 1 {
                return Err(TorshError::AutogradError(
                    "abs requires exactly 1 input".to_string(),
                ));
            }

            // For now, return a cloned tensor (in real implementation, apply abs)
            Ok(vec![inputs[0].clone_tensor()])
        }

        fn subgradient<T>(
            &self,
            _ctx: &mut FunctionContext,
            grad_outputs: &[&dyn AutogradTensor<T>],
        ) -> Result<Vec<Option<SubgradientSet<T>>>>
        where
            T: torsh_core::dtype::TensorElement + num_traits::Float,
        {
            if grad_outputs.len() != 1 {
                return Err(TorshError::AutogradError(
                    "abs grad requires exactly 1 output".to_string(),
                ));
            }

            let grad_output = grad_outputs[0];

            // Primary subgradient: sign function (most commonly used)
            let primary = grad_output.clone_tensor();

            // For x = 0, subgradient is any value in [-1, 1]
            // We provide common alternatives: -1, 0, 1
            let zero_grad = grad_output.zeros_like();
            let neg_grad = grad_output.clone_tensor(); // TODO: negate

            let subgrad_set = SubgradientSet {
                primary,
                alternatives: vec![zero_grad, neg_grad],
                selection_strategy: SubgradientSelection::Primary,
            };

            Ok(vec![Some(subgrad_set)])
        }
    }

    /// ReLU function: f(x) = max(0, x)
    /// Subgradient: ∂f(x) = {0 if x < 0, [0, 1] if x = 0, 1 if x > 0}
    #[derive(Debug)]
    pub struct ReLUFunction;

    impl DynFunction for ReLUFunction {
        fn name(&self) -> &str {
            "relu"
        }
        fn is_differentiable(&self) -> bool {
            false
        }
    }

    impl SubgradientFunction for ReLUFunction {
        fn forward<T>(
            &self,
            _ctx: &mut FunctionContext,
            inputs: &[&dyn AutogradTensor<T>],
        ) -> Result<Vec<Box<dyn AutogradTensor<T>>>>
        where
            T: torsh_core::dtype::TensorElement + num_traits::Float,
        {
            if inputs.len() != 1 {
                return Err(TorshError::AutogradError(
                    "relu requires exactly 1 input".to_string(),
                ));
            }

            // For now, return a cloned tensor (in real implementation, apply relu)
            Ok(vec![inputs[0].clone_tensor()])
        }

        fn subgradient<T>(
            &self,
            _ctx: &mut FunctionContext,
            grad_outputs: &[&dyn AutogradTensor<T>],
        ) -> Result<Vec<Option<SubgradientSet<T>>>>
        where
            T: torsh_core::dtype::TensorElement + num_traits::Float,
        {
            if grad_outputs.len() != 1 {
                return Err(TorshError::AutogradError(
                    "relu grad requires exactly 1 output".to_string(),
                ));
            }

            let grad_output = grad_outputs[0];

            // Primary subgradient: use 1 for x >= 0, 0 for x < 0
            let primary = grad_output.clone_tensor();

            // Alternative for x = 0: use 0 instead of 1
            let zero_grad = grad_output.zeros_like();

            let subgrad_set = SubgradientSet {
                primary,
                alternatives: vec![zero_grad],
                selection_strategy: SubgradientSelection::Primary,
            };

            Ok(vec![Some(subgrad_set)])
        }
    }

    /// Maximum function: f(x, y) = max(x, y)
    /// Subgradient depends on which input is larger
    #[derive(Debug)]
    pub struct MaxFunction;

    impl DynFunction for MaxFunction {
        fn name(&self) -> &str {
            "max"
        }
        fn is_differentiable(&self) -> bool {
            false
        }
    }

    impl SubgradientFunction for MaxFunction {
        fn forward<T>(
            &self,
            _ctx: &mut FunctionContext,
            inputs: &[&dyn AutogradTensor<T>],
        ) -> Result<Vec<Box<dyn AutogradTensor<T>>>>
        where
            T: torsh_core::dtype::TensorElement + num_traits::Float,
        {
            if inputs.len() != 2 {
                return Err(TorshError::AutogradError(
                    "max requires exactly 2 inputs".to_string(),
                ));
            }

            // For now, return first input (in real implementation, compute element-wise max)
            Ok(vec![inputs[0].clone_tensor()])
        }

        fn subgradient<T>(
            &self,
            _ctx: &mut FunctionContext,
            grad_outputs: &[&dyn AutogradTensor<T>],
        ) -> Result<Vec<Option<SubgradientSet<T>>>>
        where
            T: torsh_core::dtype::TensorElement + num_traits::Float,
        {
            if grad_outputs.len() != 1 {
                return Err(TorshError::AutogradError(
                    "max grad requires exactly 1 output".to_string(),
                ));
            }

            let grad_output = grad_outputs[0];

            // Primary: gradient flows to the larger input
            let grad_x = grad_output.clone_tensor();
            let grad_y = grad_output.zeros_like();

            // Alternative: when inputs are equal, gradient can be split
            let half_grad_x = grad_output.clone_tensor(); // TODO: multiply by 0.5
            let half_grad_y = grad_output.clone_tensor(); // TODO: multiply by 0.5

            let subgrad_set_x = SubgradientSet {
                primary: grad_x,
                alternatives: vec![half_grad_x],
                selection_strategy: SubgradientSelection::Primary,
            };

            let subgrad_set_y = SubgradientSet {
                primary: grad_y,
                alternatives: vec![half_grad_y],
                selection_strategy: SubgradientSelection::Primary,
            };

            Ok(vec![Some(subgrad_set_x), Some(subgrad_set_y)])
        }
    }

    /// L1 norm function: f(x) = ||x||_1 = Σ|x_i|
    /// Non-differentiable at zero, but has subgradients
    #[derive(Debug)]
    pub struct L1NormFunction;

    impl DynFunction for L1NormFunction {
        fn name(&self) -> &str {
            "l1_norm"
        }
        fn is_differentiable(&self) -> bool {
            false
        }
    }

    impl SubgradientFunction for L1NormFunction {
        fn forward<T>(
            &self,
            _ctx: &mut FunctionContext,
            inputs: &[&dyn AutogradTensor<T>],
        ) -> Result<Vec<Box<dyn AutogradTensor<T>>>>
        where
            T: torsh_core::dtype::TensorElement + num_traits::Float,
        {
            if inputs.len() != 1 {
                return Err(TorshError::AutogradError(
                    "l1_norm requires exactly 1 input".to_string(),
                ));
            }

            // For now, return a scalar ones tensor (in real implementation, compute L1 norm)
            Ok(vec![inputs[0].ones_like()])
        }

        fn subgradient<T>(
            &self,
            _ctx: &mut FunctionContext,
            grad_outputs: &[&dyn AutogradTensor<T>],
        ) -> Result<Vec<Option<SubgradientSet<T>>>>
        where
            T: torsh_core::dtype::TensorElement + num_traits::Float,
        {
            if grad_outputs.len() != 1 {
                return Err(TorshError::AutogradError(
                    "l1_norm grad requires exactly 1 output".to_string(),
                ));
            }

            let grad_output = grad_outputs[0];

            // Primary subgradient: sign function
            let primary = grad_output.clone_tensor(); // TODO: apply sign function

            // Alternative subgradients for zero elements: any value in [-1, 1]
            let zero_grad = grad_output.zeros_like();
            let neg_grad = grad_output.clone_tensor(); // TODO: negate

            let subgrad_set = SubgradientSet {
                primary,
                alternatives: vec![zero_grad, neg_grad],
                selection_strategy: SubgradientSelection::MinNorm, // Prefer smaller gradients
            };

            Ok(vec![Some(subgrad_set)])
        }
    }
}

/// Function serialization and deployment framework
pub mod serialization {
    use super::deployment::compute_signature;
    use super::subgradient_functions::*;
    use super::*;
    use std::fs::{self, File};
    use std::io::{BufReader, BufWriter};

    /// Trait for serializable functions
    pub trait SerializableFunction: DynFunction {
        /// Serialize the function to bytes
        fn serialize(&self) -> Result<Vec<u8>>;

        /// Deserialize the function from bytes
        fn deserialize(data: &[u8]) -> Result<Box<dyn SerializableFunction>>
        where
            Self: Sized;

        /// Get the function's serialization format version
        fn format_version(&self) -> u32 {
            1
        }

        /// Validate the function after deserialization
        fn validate(&self) -> Result<()> {
            Ok(())
        }
    }

    /// Concrete implementations of SerializableFunction for common functions
    impl SerializableFunction for AbsFunction {
        fn serialize(&self) -> Result<Vec<u8>> {
            let data = serde_json::to_vec(&"abs_function")
                .map_err(|e| TorshError::AutogradError(format!("Serialization error: {e}")))?;
            Ok(data)
        }

        fn deserialize(data: &[u8]) -> Result<Box<dyn SerializableFunction>>
        where
            Self: Sized,
        {
            let _function_type: String = serde_json::from_slice(data)
                .map_err(|e| TorshError::AutogradError(format!("Deserialization error: {e}")))?;
            Ok(Box::new(AbsFunction))
        }
    }

    impl SerializableFunction for ReLUFunction {
        fn serialize(&self) -> Result<Vec<u8>> {
            let data = serde_json::to_vec(&"relu_function")
                .map_err(|e| TorshError::AutogradError(format!("Serialization error: {e}")))?;
            Ok(data)
        }

        fn deserialize(data: &[u8]) -> Result<Box<dyn SerializableFunction>>
        where
            Self: Sized,
        {
            let _function_type: String = serde_json::from_slice(data)
                .map_err(|e| TorshError::AutogradError(format!("Deserialization error: {e}")))?;
            Ok(Box::new(ReLUFunction))
        }
    }

    impl SerializableFunction for MaxFunction {
        fn serialize(&self) -> Result<Vec<u8>> {
            let data = serde_json::to_vec(&"max_function")
                .map_err(|e| TorshError::AutogradError(format!("Serialization error: {e}")))?;
            Ok(data)
        }

        fn deserialize(data: &[u8]) -> Result<Box<dyn SerializableFunction>>
        where
            Self: Sized,
        {
            let _function_type: String = serde_json::from_slice(data)
                .map_err(|e| TorshError::AutogradError(format!("Deserialization error: {e}")))?;
            Ok(Box::new(MaxFunction))
        }
    }

    impl SerializableFunction for L1NormFunction {
        fn serialize(&self) -> Result<Vec<u8>> {
            let data = serde_json::to_vec(&"l1_norm_function")
                .map_err(|e| TorshError::AutogradError(format!("Serialization error: {e}")))?;
            Ok(data)
        }

        fn deserialize(data: &[u8]) -> Result<Box<dyn SerializableFunction>>
        where
            Self: Sized,
        {
            let _function_type: String = serde_json::from_slice(data)
                .map_err(|e| TorshError::AutogradError(format!("Deserialization error: {e}")))?;
            Ok(Box::new(L1NormFunction))
        }
    }

    /// Function factory for creating functions from serialized data
    pub struct FunctionFactory;

    impl FunctionFactory {
        /// Create a function from a package
        pub fn create_from_package(
            package: &FunctionPackage,
        ) -> Result<Box<dyn SerializableFunction>> {
            let function_type: String = serde_json::from_slice(&package.function_data)
                .map_err(|e| TorshError::AutogradError(format!("Deserialization error: {e}")))?;

            match function_type.as_str() {
                "abs_function" => AbsFunction::deserialize(&package.function_data),
                "relu_function" => ReLUFunction::deserialize(&package.function_data),
                "max_function" => MaxFunction::deserialize(&package.function_data),
                "l1_norm_function" => L1NormFunction::deserialize(&package.function_data),
                _ => Err(TorshError::AutogradError(format!(
                    "Unknown function type: {function_type}"
                ))),
            }
        }

        /// Create a package from a serializable function
        pub fn create_package_from_function(
            function: &dyn SerializableFunction,
            metadata: FunctionMetadata,
        ) -> Result<FunctionPackage> {
            let function_data = function.serialize()?;
            let signature = compute_signature(&metadata, &function_data);
            Ok(FunctionPackage::new(metadata, function_data, signature))
        }
    }

    /// Function package for deployment
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FunctionPackage {
        /// Function metadata
        pub metadata: FunctionMetadata,
        /// Serialized function data
        pub function_data: Vec<u8>,
        /// Package format version
        pub format_version: u32,
        /// Package signature for verification
        pub signature: String,
        /// Required runtime dependencies
        pub runtime_dependencies: Vec<String>,
        /// Minimum supported framework version
        pub min_framework_version: String,
    }

    impl FunctionPackage {
        /// Create a new function package
        pub fn new(metadata: FunctionMetadata, function_data: Vec<u8>, signature: String) -> Self {
            Self {
                metadata,
                function_data,
                format_version: 1,
                signature,
                runtime_dependencies: vec!["torsh-autograd".to_string()],
                min_framework_version: "0.1.0".to_string(),
            }
        }

        /// Verify package integrity
        pub fn verify(&self) -> Result<()> {
            // Simple checksum verification (in production, use proper cryptographic signatures)
            let computed_checksum = self.compute_checksum();
            if computed_checksum != self.signature {
                return Err(TorshError::AutogradError(
                    "Function package signature verification failed".to_string(),
                ));
            }
            Ok(())
        }

        /// Compute package checksum
        fn compute_checksum(&self) -> String {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            self.metadata.name.hash(&mut hasher);
            self.metadata.version.hash(&mut hasher);
            self.function_data.hash(&mut hasher);
            format!("{:x}", hasher.finish())
        }

        /// Save package to file
        pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
            let file = File::create(path)?;
            let writer = BufWriter::new(file);
            serde_json::to_writer_pretty(writer, self)
                .map_err(|e| TorshError::AutogradError(format!("Serialization error: {e}")))?;
            Ok(())
        }

        /// Load package from file
        pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            let package: FunctionPackage = serde_json::from_reader(reader)
                .map_err(|e| TorshError::AutogradError(format!("Deserialization error: {e}")))?;
            package.verify()?;
            Ok(package)
        }
    }

    /// Function deployment manager
    pub struct FunctionDeploymentManager {
        /// Directory for storing deployed functions
        deploy_dir: std::path::PathBuf,
        /// Registry of deployed functions
        deployed_functions: RwLock<HashMap<String, FunctionPackage>>,
    }

    impl FunctionDeploymentManager {
        /// Create a new deployment manager
        pub fn new<P: AsRef<Path>>(deploy_dir: P) -> Result<Self> {
            let deploy_dir = deploy_dir.as_ref().to_path_buf();
            fs::create_dir_all(&deploy_dir)?;

            Ok(Self {
                deploy_dir,
                deployed_functions: RwLock::new(HashMap::new()),
            })
        }

        /// Deploy a function package
        pub fn deploy(&self, package: FunctionPackage) -> Result<()> {
            // Verify package before deployment
            package.verify()?;

            // Check version compatibility
            if !self.is_compatible_version(&package.min_framework_version) {
                return Err(TorshError::AutogradError(format!(
                    "Function {} requires framework version {}, but current version is incompatible",
                    package.metadata.name, package.min_framework_version
                )));
            }

            // Save package to deployment directory
            let package_path = self
                .deploy_dir
                .join(format!("{}.pkg", package.metadata.name));
            package.save(&package_path)?;

            // Register deployed function
            let mut deployed = self.deployed_functions.write();
            deployed.insert(package.metadata.name.clone(), package);

            Ok(())
        }

        /// Undeploy a function
        pub fn undeploy(&self, name: &str) -> Result<()> {
            let package_path = self.deploy_dir.join(format!("{}.pkg", name));
            if package_path.exists() {
                fs::remove_file(package_path)?;
            }

            let mut deployed = self.deployed_functions.write();
            deployed.remove(name);

            Ok(())
        }

        /// List deployed functions
        pub fn list_deployed(&self) -> Vec<String> {
            self.deployed_functions.read().keys().cloned().collect()
        }

        /// Get deployed function metadata
        pub fn get_deployed_metadata(&self, name: &str) -> Option<FunctionMetadata> {
            self.deployed_functions
                .read()
                .get(name)
                .map(|pkg| pkg.metadata.clone())
        }

        /// Load deployed function
        pub fn load_deployed(&self, name: &str) -> Result<Vec<u8>> {
            let deployed = self.deployed_functions.read();
            let package = deployed.get(name).ok_or_else(|| {
                TorshError::AutogradError(format!("Deployed function '{}' not found", name))
            })?;
            Ok(package.function_data.clone())
        }

        /// Check framework version compatibility
        fn is_compatible_version(&self, required_version: &str) -> bool {
            // Simplified version check (in production, use proper semantic versioning)
            let current_version = "0.1.0";
            required_version <= current_version
        }

        /// Import function from package file
        pub fn import_from_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
            let package = FunctionPackage::load(path)?;
            self.deploy(package)
        }

        /// Export function to package file
        pub fn export_to_file<P: AsRef<Path>>(&self, name: &str, path: P) -> Result<()> {
            let deployed = self.deployed_functions.read();
            let package = deployed.get(name).ok_or_else(|| {
                TorshError::AutogradError(format!("Deployed function '{}' not found", name))
            })?;
            package.save(path)
        }
    }

    /// Function library for managing collections of functions
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct FunctionLibrary {
        /// Library name
        pub name: String,
        /// Library version
        pub version: String,
        /// Library description
        pub description: String,
        /// Functions in the library
        pub functions: Vec<FunctionPackage>,
        /// Library dependencies
        pub dependencies: Vec<String>,
    }

    impl FunctionLibrary {
        /// Create a new function library
        pub fn new(name: String, version: String, description: String) -> Self {
            Self {
                name,
                version,
                description,
                functions: Vec::new(),
                dependencies: Vec::new(),
            }
        }

        /// Add a function to the library
        pub fn add_function(&mut self, package: FunctionPackage) {
            self.functions.push(package);
        }

        /// Add a dependency to the library
        pub fn add_dependency(&mut self, dependency: String) {
            if !self.dependencies.contains(&dependency) {
                self.dependencies.push(dependency);
            }
        }

        /// Save library to file
        pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
            let file = File::create(path)?;
            let writer = BufWriter::new(file);
            serde_json::to_writer_pretty(writer, self)
                .map_err(|e| TorshError::AutogradError(format!("Serialization error: {e}")))?;
            Ok(())
        }

        /// Load library from file
        pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            let library: FunctionLibrary = serde_json::from_reader(reader)
                .map_err(|e| TorshError::AutogradError(format!("Deserialization error: {e}")))?;
            Ok(library)
        }

        /// Deploy all functions in the library
        pub fn deploy_all(&self, manager: &FunctionDeploymentManager) -> Result<()> {
            for package in &self.functions {
                manager.deploy(package.clone())?;
            }
            Ok(())
        }

        /// List function names in the library
        pub fn list_functions(&self) -> Vec<String> {
            self.functions
                .iter()
                .map(|pkg| pkg.metadata.name.clone())
                .collect()
        }
    }
}

/// Function deployment utilities
pub mod deployment {
    use super::serialization::*;
    use super::*;

    /// Global deployment manager
    static DEPLOYMENT_MANAGER: std::sync::OnceLock<FunctionDeploymentManager> =
        std::sync::OnceLock::new();

    /// Get the global deployment manager
    pub fn global_deployment_manager() -> &'static FunctionDeploymentManager {
        DEPLOYMENT_MANAGER.get_or_init(|| {
            FunctionDeploymentManager::new("./torsh_functions").unwrap_or_else(|_| {
                // Fallback to temporary directory
                let temp_dir = std::env::temp_dir().join("torsh_functions");
                FunctionDeploymentManager::new(temp_dir)
                    .expect("Failed to create deployment manager")
            })
        })
    }

    /// Deploy a function globally
    pub fn deploy_function(package: FunctionPackage) -> Result<()> {
        global_deployment_manager().deploy(package)
    }

    /// Undeploy a function globally
    pub fn undeploy_function(name: &str) -> Result<()> {
        global_deployment_manager().undeploy(name)
    }

    /// List all deployed functions
    pub fn list_deployed_functions() -> Vec<String> {
        global_deployment_manager().list_deployed()
    }

    /// Get deployed function metadata
    pub fn get_deployed_function_metadata(name: &str) -> Option<FunctionMetadata> {
        global_deployment_manager().get_deployed_metadata(name)
    }

    /// Create a function package from metadata and data
    pub fn create_function_package(
        metadata: FunctionMetadata,
        function_data: Vec<u8>,
    ) -> FunctionPackage {
        let signature = compute_signature(&metadata, &function_data);
        FunctionPackage::new(metadata, function_data, signature)
    }

    /// Compute function signature for verification
    pub fn compute_signature(metadata: &FunctionMetadata, data: &[u8]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        metadata.name.hash(&mut hasher);
        metadata.version.hash(&mut hasher);
        data.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Function deployment builder
    pub struct FunctionDeploymentBuilder {
        metadata: FunctionMetadata,
        function_data: Option<Vec<u8>>,
        dependencies: Vec<String>,
    }

    impl FunctionDeploymentBuilder {
        /// Create a new deployment builder
        pub fn new(name: String) -> Self {
            Self {
                metadata: FunctionMetadata {
                    name,
                    is_differentiable: true,
                    memory_complexity: MemoryComplexity::Linear,
                    computational_complexity: ComputationalComplexity::Linear,
                    is_fusable: false,
                    version: "1.0.0".to_string(),
                    description: "".to_string(),
                    author: "".to_string(),
                    created_at: SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap()
                        .as_secs()
                        .to_string(),
                    checksum: "".to_string(),
                    dependencies: vec![],
                },
                function_data: None,
                dependencies: vec![],
            }
        }

        /// Set function version
        pub fn version(mut self, version: String) -> Self {
            self.metadata.version = version;
            self
        }

        /// Set function description
        pub fn description(mut self, description: String) -> Self {
            self.metadata.description = description;
            self
        }

        /// Set function author
        pub fn author(mut self, author: String) -> Self {
            self.metadata.author = author;
            self
        }

        /// Set function data
        pub fn data(mut self, data: Vec<u8>) -> Self {
            self.function_data = Some(data);
            self
        }

        /// Add dependency
        pub fn dependency(mut self, dependency: String) -> Self {
            self.dependencies.push(dependency);
            self
        }

        /// Set memory complexity
        pub fn memory_complexity(mut self, complexity: MemoryComplexity) -> Self {
            self.metadata.memory_complexity = complexity;
            self
        }

        /// Set computational complexity
        pub fn computational_complexity(mut self, complexity: ComputationalComplexity) -> Self {
            self.metadata.computational_complexity = complexity;
            self
        }

        /// Set fusable flag
        pub fn fusable(mut self, fusable: bool) -> Self {
            self.metadata.is_fusable = fusable;
            self
        }

        /// Set differentiable flag
        pub fn differentiable(mut self, differentiable: bool) -> Self {
            self.metadata.is_differentiable = differentiable;
            self
        }

        /// Build the function package
        pub fn build(mut self) -> Result<FunctionPackage> {
            let function_data = self
                .function_data
                .ok_or_else(|| TorshError::AutogradError("Function data not set".to_string()))?;

            self.metadata.dependencies = self.dependencies;
            Ok(create_function_package(self.metadata, function_data))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_registry() {
        let registry = FunctionRegistry::new();
        let scaled_add = examples::ScaledAdd { scale: 2.0 };

        // Test registration
        assert!(registry
            .register("scaled_add".to_string(), scaled_add)
            .is_ok());

        // Test duplicate registration fails
        let scaled_add2 = examples::ScaledAdd { scale: 3.0 };
        assert!(registry
            .register("scaled_add".to_string(), scaled_add2)
            .is_err());

        // Test retrieval
        assert!(registry.get("scaled_add").is_some());
        assert!(registry.get("nonexistent").is_none());

        // Test listing
        let functions = registry.list_functions();
        assert_eq!(functions.len(), 1);
        assert!(functions.contains(&"scaled_add".to_string()));

        // Test metadata
        let metadata = registry.get_metadata("scaled_add").unwrap();
        assert_eq!(metadata.name, "ScaledAdd");
        assert!(metadata.is_differentiable);
        assert!(metadata.is_fusable);
    }

    #[test]
    fn test_function_context() {
        let mut ctx = FunctionContext::new();

        // Test saving and retrieving values
        ctx.save_value(42i32);
        ctx.save_value(3.14f64);

        assert_eq!(*ctx.get_saved_value::<i32>(0).unwrap(), 42);
        assert_eq!(*ctx.get_saved_value::<f64>(1).unwrap(), 3.14);

        // Test type mismatch
        assert!(ctx.get_saved_value::<f32>(0).is_err());
    }

    #[test]
    fn test_function_serialization() {
        use super::deployment::create_function_package;

        // Create test metadata
        let metadata = FunctionMetadata {
            name: "test_function".to_string(),
            is_differentiable: true,
            memory_complexity: MemoryComplexity::Linear,
            computational_complexity: ComputationalComplexity::Linear,
            is_fusable: false,
            version: "1.0.0".to_string(),
            description: "Test function for serialization".to_string(),
            author: "Test Author".to_string(),
            created_at: "1640995200".to_string(), // Fixed timestamp for testing
            checksum: "".to_string(),
            dependencies: vec!["torsh-autograd".to_string()],
        };

        // Create test function data
        let function_data = vec![1, 2, 3, 4, 5];

        // Create function package
        let package = create_function_package(metadata.clone(), function_data.clone());

        // Test package verification
        assert!(package.verify().is_ok());

        // Test package metadata
        assert_eq!(package.metadata.name, "test_function");
        assert_eq!(package.metadata.version, "1.0.0");
        assert_eq!(package.function_data, function_data);
    }

    #[test]
    fn test_function_deployment_builder() {
        use super::deployment::*;

        let builder = FunctionDeploymentBuilder::new("test_func".to_string())
            .version("2.0.0".to_string())
            .description("Test function".to_string())
            .author("Test Author".to_string())
            .data(vec![1, 2, 3])
            .dependency("test_dep".to_string())
            .memory_complexity(MemoryComplexity::Constant)
            .computational_complexity(ComputationalComplexity::Quadratic)
            .fusable(true)
            .differentiable(false);

        let package = builder.build().unwrap();

        assert_eq!(package.metadata.name, "test_func");
        assert_eq!(package.metadata.version, "2.0.0");
        assert_eq!(package.metadata.description, "Test function");
        assert_eq!(package.metadata.author, "Test Author");
        assert_eq!(package.function_data, vec![1, 2, 3]);
        assert_eq!(package.metadata.dependencies, vec!["test_dep"]);
        assert_eq!(
            package.metadata.memory_complexity,
            MemoryComplexity::Constant
        );
        assert_eq!(
            package.metadata.computational_complexity,
            ComputationalComplexity::Quadratic
        );
        assert!(package.metadata.is_fusable);
        assert!(!package.metadata.is_differentiable);
    }

    #[test]
    fn test_function_library() {
        use super::deployment::*;
        use super::serialization::*;

        let mut library = FunctionLibrary::new(
            "test_library".to_string(),
            "1.0.0".to_string(),
            "Test function library".to_string(),
        );

        // Create test packages
        let package1 = FunctionDeploymentBuilder::new("func1".to_string())
            .data(vec![1, 2, 3])
            .build()
            .unwrap();

        let package2 = FunctionDeploymentBuilder::new("func2".to_string())
            .data(vec![4, 5, 6])
            .build()
            .unwrap();

        // Add packages to library
        library.add_function(package1);
        library.add_function(package2);
        library.add_dependency("dep1".to_string());
        library.add_dependency("dep2".to_string());

        // Test library properties
        assert_eq!(library.name, "test_library");
        assert_eq!(library.version, "1.0.0");
        assert_eq!(library.functions.len(), 2);
        assert_eq!(library.dependencies, vec!["dep1", "dep2"]);

        // Test function listing
        let function_names = library.list_functions();
        assert!(function_names.contains(&"func1".to_string()));
        assert!(function_names.contains(&"func2".to_string()));
    }

    #[test]
    fn test_function_serialization_implementations() {
        use super::serialization::*;
        use super::subgradient_functions::*;

        // Test AbsFunction serialization
        let abs_func = AbsFunction;
        let serialized = abs_func.serialize().unwrap();
        let _deserialized = AbsFunction::deserialize(&serialized).unwrap();

        // Verify the function type matches
        let function_type: String = serde_json::from_slice(&serialized).unwrap();
        assert_eq!(function_type, "abs_function");

        // Test ReLUFunction serialization
        let relu_func = ReLUFunction;
        let serialized = relu_func.serialize().unwrap();
        let _deserialized = ReLUFunction::deserialize(&serialized).unwrap();

        let function_type: String = serde_json::from_slice(&serialized).unwrap();
        assert_eq!(function_type, "relu_function");

        // Test MaxFunction serialization
        let max_func = MaxFunction;
        let serialized = max_func.serialize().unwrap();
        let _deserialized = MaxFunction::deserialize(&serialized).unwrap();

        let function_type: String = serde_json::from_slice(&serialized).unwrap();
        assert_eq!(function_type, "max_function");

        // Test L1NormFunction serialization
        let l1_func = L1NormFunction;
        let serialized = l1_func.serialize().unwrap();
        let _deserialized = L1NormFunction::deserialize(&serialized).unwrap();

        let function_type: String = serde_json::from_slice(&serialized).unwrap();
        assert_eq!(function_type, "l1_norm_function");
    }

    #[test]
    fn test_function_factory() {
        use super::deployment::create_function_package;
        use super::serialization::*;
        use super::subgradient_functions::*;

        // Create a test function package
        let metadata = FunctionMetadata {
            name: "test_abs".to_string(),
            is_differentiable: true,
            memory_complexity: MemoryComplexity::Linear,
            computational_complexity: ComputationalComplexity::Linear,
            is_fusable: true,
            version: "1.0.0".to_string(),
            description: "Test absolute value function".to_string(),
            author: "Test Author".to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            checksum: "".to_string(),
            dependencies: vec![],
        };

        let abs_func = AbsFunction;
        let function_data = abs_func.serialize().unwrap();
        let package = create_function_package(metadata, function_data);

        // Test factory creation
        let created_function = FunctionFactory::create_from_package(&package).unwrap();

        // Verify the function can be serialized again
        let re_serialized = created_function.serialize().unwrap();
        let function_type: String = serde_json::from_slice(&re_serialized).unwrap();
        assert_eq!(function_type, "abs_function");
    }

    #[test]
    fn test_function_package_from_serializable() {
        use super::serialization::*;
        use super::subgradient_functions::*;

        let metadata = FunctionMetadata {
            name: "test_relu".to_string(),
            is_differentiable: true,
            memory_complexity: MemoryComplexity::Constant,
            computational_complexity: ComputationalComplexity::Linear,
            is_fusable: true,
            version: "1.0.0".to_string(),
            description: "Test ReLU function".to_string(),
            author: "Test Author".to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            checksum: "".to_string(),
            dependencies: vec![],
        };

        let relu_func = ReLUFunction;
        let package = FunctionFactory::create_package_from_function(&relu_func, metadata).unwrap();

        // Test package verification
        assert!(package.verify().is_ok());
        assert_eq!(package.metadata.name, "test_relu");

        // Test that we can recreate the function from the package
        let recreated_function = FunctionFactory::create_from_package(&package).unwrap();
        let serialized_again = recreated_function.serialize().unwrap();
        let function_type: String = serde_json::from_slice(&serialized_again).unwrap();
        assert_eq!(function_type, "relu_function");
    }
}
