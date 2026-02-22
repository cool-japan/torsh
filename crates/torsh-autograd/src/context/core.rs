//! Core autograd context and computation graph management

use crate::grad_mode;
use crate::{gradient_storage::UnifiedGradientStorage, GradientStorage};
use petgraph::visit::EdgeRef;
use petgraph::{Direction, Graph};
use std::collections::HashMap;
use std::sync::Arc;
use torsh_core::error::{Result, TorshError};

/// Node in the computation graph
#[derive(Clone)]
pub struct GraphNode {
    /// Unique identifier for this node
    pub id: usize,
    /// Name of the operation that created this tensor
    pub operation: String,
    /// Input tensor IDs
    pub inputs: Vec<usize>,
    /// Output tensor ID
    pub output: usize,
    /// Gradient function for backward pass
    pub grad_fn: Option<Arc<dyn GradientFunction>>,
    /// Whether this node requires gradient computation
    pub requires_grad: bool,
}

impl std::fmt::Debug for GraphNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphNode")
            .field("id", &self.id)
            .field("operation", &self.operation)
            .field("inputs", &self.inputs)
            .field("output", &self.output)
            .field("grad_fn", &self.grad_fn.as_ref().map(|gf| gf.name()))
            .field("requires_grad", &self.requires_grad)
            .finish()
    }
}

/// Trait for gradient functions in the computation graph
pub trait GradientFunction: Send + Sync + std::fmt::Debug {
    /// Compute gradients for this operation
    fn backward(&self, grad_output: &[f32]) -> Result<Vec<Vec<f32>>>;

    /// Get the name of this operation
    fn name(&self) -> &str;
}

/// Autograd context with computation graph management
pub struct AutogradContext {
    /// Computation graph storing operations and dependencies
    pub(crate) computation_graph: Graph<GraphNode, ()>,
    /// Map tensor IDs to graph node indices
    pub(crate) tensor_to_node: HashMap<usize, petgraph::graph::NodeIndex>,
    /// Counter for generating unique tensor IDs
    next_tensor_id: usize,
    /// Whether this context owns gradient computation
    owns_grad: bool,
    /// Unified gradient storage
    pub(crate) gradient_storage: UnifiedGradientStorage,
    /// Internal gradient cache for backward pass computation
    pub(crate) gradient_cache: HashMap<usize, Vec<f32>>,
    /// Whether to retain the graph after backward pass
    pub(crate) retain_graph: bool,
    /// Memory threshold for automatic graph management
    pub(crate) memory_threshold: Option<usize>,
    /// Whether to automatically optimize the graph
    pub(crate) auto_optimize: bool,
    /// Whether anomaly detection is enabled
    anomaly_detection_enabled: bool,
}

impl Default for AutogradContext {
    fn default() -> Self {
        Self {
            computation_graph: Graph::new(),
            tensor_to_node: HashMap::new(),
            next_tensor_id: 0,
            owns_grad: true,
            gradient_storage: UnifiedGradientStorage::new(),
            gradient_cache: HashMap::new(),
            retain_graph: false,
            memory_threshold: None,
            auto_optimize: false,
            anomaly_detection_enabled: false,
        }
    }
}

impl AutogradContext {
    /// Create a new autograd context
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if gradients are enabled in this context
    pub fn is_grad_enabled(&self) -> bool {
        grad_mode::is_grad_enabled() && self.owns_grad
    }

    /// Generate a new unique tensor ID
    pub fn new_tensor_id(&mut self) -> usize {
        let id = self.next_tensor_id;
        self.next_tensor_id += 1;
        id
    }

    /// Enable anomaly detection for autograd operations
    pub fn enable_anomaly_detection(&mut self) {
        self.anomaly_detection_enabled = true;
        tracing::debug!("Anomaly detection enabled");
    }

    /// Disable anomaly detection for autograd operations
    pub fn disable_anomaly_detection(&mut self) {
        self.anomaly_detection_enabled = false;
        tracing::debug!("Anomaly detection disabled");
    }

    /// Check if anomaly detection is enabled
    pub fn is_anomaly_detection_enabled(&self) -> bool {
        self.anomaly_detection_enabled
    }

    /// Add a new operation to the computation graph (respects inference mode for zero overhead)
    pub fn add_operation(
        &mut self,
        operation: String,
        inputs: Vec<usize>,
        output: usize,
        requires_grad: bool,
        grad_fn: Option<Arc<dyn GradientFunction>>,
    ) -> Result<()> {
        // Skip graph building in inference mode for zero overhead
        if crate::grad_mode::is_inference_mode() || !self.is_grad_enabled() {
            return Ok(());
        }

        let node = GraphNode {
            id: output,
            operation,
            inputs: inputs.clone(),
            output,
            grad_fn,
            requires_grad,
        };

        let node_index = self.computation_graph.add_node(node);
        self.tensor_to_node.insert(output, node_index);

        // Add edges from input tensors to this operation
        for input_id in inputs {
            if let Some(&input_node_index) = self.tensor_to_node.get(&input_id) {
                self.computation_graph
                    .add_edge(input_node_index, node_index, ());
            }
        }

        Ok(())
    }

    /// Execute a function within this context
    pub fn run<F, R>(&mut self, f: F) -> Result<R>
    where
        F: FnOnce(&mut Self) -> Result<R>,
    {
        f(self)
    }

    /// Clear the computation graph
    pub fn clear_graph(&mut self) {
        self.computation_graph.clear();
        self.tensor_to_node.clear();
        self.gradient_cache.clear();
        // Note: We don't clear gradient_storage here because gradients should persist
        // after the backward pass for later access. Use clear_gradients() explicitly if needed.
    }

    /// Set whether to retain the graph after backward pass
    pub fn set_retain_graph(&mut self, retain: bool) {
        self.retain_graph = retain;
    }

    /// Get the number of operations in the graph
    pub fn graph_size(&self) -> usize {
        self.computation_graph.node_count()
    }

    /// Perform backward pass starting from a tensor
    pub fn backward_from_tensor(&mut self, tensor_id: usize, grad_output: Vec<f32>) -> Result<()> {
        // Initialize the gradient for the output tensor
        let gradient_tensor = torsh_tensor::Tensor::from_data(
            grad_output.clone(),
            vec![grad_output.len()],
            torsh_core::DeviceType::Cpu,
        )?;
        self.gradient_storage
            .store_gradient(tensor_id, gradient_tensor)?;

        // Also initialize the gradient cache for backward pass computation
        self.gradient_cache.insert(tensor_id, grad_output);

        // Find the node corresponding to this tensor
        let &node_index = self.tensor_to_node.get(&tensor_id).ok_or_else(|| {
            TorshError::AutogradError("Tensor not found in computation graph".to_string())
        })?;

        // Perform topological sort to get the correct order for gradient computation
        let mut visited = std::collections::HashSet::new();
        let mut topo_order = Vec::new();
        self.topological_sort_dfs(node_index, &mut visited, &mut topo_order);
        topo_order.reverse();

        // Compute gradients in reverse topological order
        for &idx in &topo_order {
            let node = &self.computation_graph[idx];
            if !node.requires_grad {
                continue;
            }

            if let Some(ref grad_fn) = node.grad_fn {
                // Get the gradient flowing into this node
                let output_grad = self
                    .gradient_cache
                    .get(&node.output)
                    .ok_or_else(|| TorshError::AutogradError("Missing gradient".to_string()))?
                    .clone();

                // Compute gradients for inputs
                let input_grads = grad_fn.backward(&output_grad)?;

                // Accumulate gradients for input tensors
                for (i, input_id) in node.inputs.iter().enumerate() {
                    if i < input_grads.len() {
                        let grad = input_grads[i].clone();
                        let entry = self
                            .gradient_cache
                            .entry(*input_id)
                            .or_insert_with(|| vec![0.0; grad.len()]);
                        for (j, &val) in grad.iter().enumerate() {
                            if j < entry.len() {
                                entry[j] += val;
                            }
                        }
                    }
                }
            }
        }

        // Store all computed gradients in gradient storage
        for (&gradient_tensor_id, gradient_vec) in &self.gradient_cache {
            if !gradient_vec.is_empty() {
                let gradient_tensor = torsh_tensor::Tensor::from_data(
                    gradient_vec.clone(),
                    vec![gradient_vec.len()],
                    torsh_core::DeviceType::Cpu,
                )?;
                self.gradient_storage
                    .store_gradient(gradient_tensor_id, gradient_tensor)?;
            }
        }

        // Clear the graph if not retaining
        if !self.retain_graph {
            self.clear_graph();
        }

        Ok(())
    }

    /// Depth-first search for topological sorting
    fn topological_sort_dfs(
        &self,
        node_index: petgraph::graph::NodeIndex,
        visited: &mut std::collections::HashSet<petgraph::graph::NodeIndex>,
        topo_order: &mut Vec<petgraph::graph::NodeIndex>,
    ) {
        if visited.contains(&node_index) {
            return;
        }
        visited.insert(node_index);

        // Visit all neighbors (dependencies) first
        for edge in self
            .computation_graph
            .edges_directed(node_index, Direction::Incoming)
        {
            self.topological_sort_dfs(edge.source(), visited, topo_order);
        }

        topo_order.push(node_index);
    }

    /// Get gradient for a tensor
    pub fn get_gradient(&self, tensor_id: usize) -> Result<Option<torsh_tensor::Tensor>> {
        self.gradient_storage.get_gradient(tensor_id)
    }

    /// Check if a tensor has computed gradient
    pub fn has_gradient(&self, tensor_id: usize) -> bool {
        self.gradient_storage.has_gradient(tensor_id)
    }

    /// Store gradient for a tensor
    pub fn store_gradient(&self, tensor_id: usize, gradient: torsh_tensor::Tensor) -> Result<()> {
        self.gradient_storage.store_gradient(tensor_id, gradient)
    }

    /// Clear gradient for a tensor
    pub fn clear_gradient(&self, tensor_id: usize) -> Result<()> {
        self.gradient_storage.clear_gradient(tensor_id)
    }

    /// Add operation with automatic optimization checks
    pub fn add_operation_with_optimization(
        &mut self,
        operation: String,
        inputs: Vec<usize>,
        output: usize,
        requires_grad: bool,
        grad_fn: Option<Arc<dyn GradientFunction>>,
    ) -> Result<()> {
        // Add the operation normally
        self.add_operation(operation, inputs, output, requires_grad, grad_fn)?;

        // Check if we should trigger optimizations
        if self.auto_optimize {
            // Trigger optimization every N operations to avoid overhead
            if self.computation_graph.node_count() % 100 == 0 {
                // This will be handled by optimization module
                // self.check_memory_pressure()?;
            }
        }

        Ok(())
    }

    /// Create a new context with dynamic graph management enabled
    pub fn new_dynamic() -> Self {
        let mut ctx = Self::new();
        ctx.memory_threshold = Some(64 * 1024 * 1024); // 64MB default threshold
        ctx
    }

    /// Enable automatic graph optimization
    pub fn enable_auto_optimization(&mut self, enable: bool) {
        self.auto_optimize = enable;
        if enable {
            tracing::debug!("Automatic graph optimization enabled");
        } else {
            tracing::debug!("Automatic graph optimization disabled");
        }
    }
}

// Thread-local autograd context
thread_local! {
    static THREAD_CONTEXT: std::cell::RefCell<Option<AutogradContext>> =
        const { std::cell::RefCell::new(None) };
}

/// Get or create the thread-local autograd context
pub fn get_or_create_context() -> Result<AutogradContext> {
    THREAD_CONTEXT.with(|ctx| {
        let mut ctx_ref = ctx.borrow_mut();
        if ctx_ref.is_none() {
            *ctx_ref = Some(AutogradContext::new());
        }
        // Safe to expect here: we just ensured ctx_ref is Some above
        Ok(ctx_ref
            .take()
            .expect("AutogradContext should exist after initialization"))
    })
}

/// Execute a function with an autograd context
pub fn with_context<F, R>(f: F) -> Result<R>
where
    F: FnOnce(&mut AutogradContext) -> Result<R>,
{
    let mut ctx = get_or_create_context()?;
    let result = f(&mut ctx);

    // Store context back
    THREAD_CONTEXT.with(|thread_ctx| {
        *thread_ctx.borrow_mut() = Some(ctx);
    });

    result
}
