//! Dynamic computation graph implementation for neural network modules
//!
//! This module provides a flexible execution framework that allows for conditional execution,
//! loops, parallel processing, and runtime modification of the computation graph structure.

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::{boxed::Box, collections::HashMap, vec::Vec};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

// Use parking_lot::Mutex for both std and no_std
use parking_lot::Mutex;

/// Dynamic Graph execution node types
///
/// Represents different types of computational nodes that can be executed
/// in a dynamic computation graph, enabling complex control flow patterns.
#[derive(Debug, Clone)]
pub enum GraphNode {
    /// Execute a specific module by name
    Module(String),
    /// Conditional execution based on predicate
    Conditional {
        condition: String,
        true_branch: Box<GraphNode>,
        false_branch: Option<Box<GraphNode>>,
    },
    /// Sequential execution of multiple nodes
    Sequence(Vec<GraphNode>),
    /// Parallel execution with results combined
    Parallel {
        nodes: Vec<GraphNode>,
        combiner: String, // Name of combiner function
    },
    /// Loop execution
    Loop {
        body: Box<GraphNode>,
        condition: String,
        max_iterations: usize,
    },
    /// Custom execution function
    Function(String),
}

/// Dynamic computation graph that can modify its structure at runtime
///
/// This container provides a flexible execution framework for neural networks
/// that need complex control flow, conditional execution, or runtime adaptation.
/// It supports modules, conditions, combiners, and custom functions that can
/// be composed into arbitrary computational graphs.
///
/// # Examples
///
/// ```rust
/// use torsh_nn::container::{DynamicGraph, GraphNode};
/// use torsh_nn::linear::Linear;
///
/// let mut graph = DynamicGraph::new();
/// graph.add_module("linear1".to_string(), Linear::new(784, 128)?);
/// graph.add_module("linear2".to_string(), Linear::new(128, 10)?);
///
/// // Create a simple sequential execution graph
/// let seq_graph = DynamicGraph::sequential(vec![
///     "linear1".to_string(),
///     "linear2".to_string(),
/// ]);
/// graph.set_graph(seq_graph);
///
/// let output = graph.forward(&input)?;
/// ```
pub struct DynamicGraph {
    base: ModuleBase,
    /// Named modules available for execution
    modules: HashMap<String, Box<dyn Module>>,
    /// Named condition functions
    conditions: HashMap<String, Box<dyn Fn(&Tensor) -> bool + Send + Sync>>,
    /// Named combiner functions for parallel execution
    combiners: HashMap<String, Box<dyn Fn(Vec<Tensor>) -> Result<Tensor> + Send + Sync>>,
    /// Named custom functions
    functions: HashMap<String, Box<dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync>>,
    /// Current execution graph
    graph: GraphNode,
    /// Execution history for debugging
    execution_history: Mutex<Vec<String>>,
}

impl DynamicGraph {
    /// Create a new dynamic graph with a simple sequential structure
    pub fn new() -> Self {
        let mut graph = Self {
            base: ModuleBase::new(),
            modules: HashMap::new(),
            conditions: HashMap::new(),
            combiners: HashMap::new(),
            functions: HashMap::new(),
            graph: GraphNode::Sequence(Vec::new()),
            execution_history: Mutex::new(Vec::new()),
        };

        // Add default combiners
        graph.add_combiner(
            "concat".to_string(),
            Box::new(|tensors: Vec<Tensor>| {
                if tensors.is_empty() {
                    return Err(TorshError::InvalidArgument(
                        "No tensors to concatenate".to_string(),
                    ));
                }
                // Simple concatenation along the last dimension
                let mut result = tensors[0].clone();
                for _tensor in tensors.iter().skip(1) {
                    // Simple concatenation along last dimension - replace with proper implementation when available
                    result = result.clone(); // TODO: implement proper concatenation
                }
                Ok(result)
            }),
        );

        graph.add_combiner(
            "add".to_string(),
            Box::new(|tensors: Vec<Tensor>| {
                if tensors.is_empty() {
                    return Err(TorshError::InvalidArgument("No tensors to add".to_string()));
                }
                let mut result = tensors[0].clone();
                for tensor in tensors.iter().skip(1) {
                    result = result.add_op(tensor)?;
                }
                Ok(result)
            }),
        );

        graph.add_combiner(
            "mean".to_string(),
            Box::new(|tensors: Vec<Tensor>| {
                if tensors.is_empty() {
                    return Err(TorshError::InvalidArgument(
                        "No tensors to average".to_string(),
                    ));
                }
                let mut result = tensors[0].clone();
                for tensor in tensors.iter().skip(1) {
                    result = result.add_op(tensor)?;
                }
                let count = tensors.len() as f32;
                result = result.div_scalar(count)?;
                Ok(result)
            }),
        );

        graph
    }

    /// Add a module to the graph
    pub fn add_module<M: Module + 'static>(&mut self, name: String, module: M) {
        self.modules.insert(name, Box::new(module));
    }

    /// Add a condition function
    pub fn add_condition<F>(&mut self, name: String, condition: F)
    where
        F: Fn(&Tensor) -> bool + Send + Sync + 'static,
    {
        self.conditions.insert(name, Box::new(condition));
    }

    /// Add a combiner function for parallel execution
    pub fn add_combiner<F>(&mut self, name: String, combiner: F)
    where
        F: Fn(Vec<Tensor>) -> Result<Tensor> + Send + Sync + 'static,
    {
        self.combiners.insert(name, Box::new(combiner));
    }

    /// Add a custom function
    pub fn add_function<F>(&mut self, name: String, function: F)
    where
        F: Fn(&Tensor) -> Result<Tensor> + Send + Sync + 'static,
    {
        self.functions.insert(name, Box::new(function));
    }

    /// Set the execution graph
    pub fn set_graph(&mut self, graph: GraphNode) {
        self.graph = graph;
    }

    /// Create a sequential graph from module names
    pub fn sequential(module_names: Vec<String>) -> GraphNode {
        GraphNode::Sequence(
            module_names
                .into_iter()
                .map(|name| GraphNode::Module(name))
                .collect(),
        )
    }

    /// Create a conditional graph
    pub fn conditional(
        condition: String,
        true_branch: GraphNode,
        false_branch: Option<GraphNode>,
    ) -> GraphNode {
        GraphNode::Conditional {
            condition,
            true_branch: Box::new(true_branch),
            false_branch: false_branch.map(Box::new),
        }
    }

    /// Create a parallel graph
    pub fn parallel(nodes: Vec<GraphNode>, combiner: String) -> GraphNode {
        GraphNode::Parallel { nodes, combiner }
    }

    /// Create a loop graph
    pub fn loop_graph(body: GraphNode, condition: String, max_iterations: usize) -> GraphNode {
        GraphNode::Loop {
            body: Box::new(body),
            condition,
            max_iterations,
        }
    }

    /// Execute a graph node
    fn execute_node(&self, node: &GraphNode, input: &Tensor) -> Result<Tensor> {
        let mut history = self.execution_history.lock();

        match node {
            GraphNode::Module(name) => {
                history.push(format!("Module: {}", name));
                let module = self.modules.get(name).ok_or_else(|| {
                    TorshError::InvalidArgument(format!("Module '{}' not found", name))
                })?;
                module.forward(input)
            }

            GraphNode::Conditional {
                condition,
                true_branch,
                false_branch,
            } => {
                history.push(format!("Conditional: {}", condition));
                let cond_fn = self.conditions.get(condition).ok_or_else(|| {
                    TorshError::InvalidArgument(format!("Condition '{}' not found", condition))
                })?;

                if cond_fn(input) {
                    history.push("Taking true branch".to_string());
                    self.execute_node(true_branch, input)
                } else if let Some(false_branch) = false_branch {
                    history.push("Taking false branch".to_string());
                    self.execute_node(false_branch, input)
                } else {
                    history.push("No false branch, returning input".to_string());
                    Ok(input.clone())
                }
            }

            GraphNode::Sequence(nodes) => {
                history.push("Sequence execution".to_string());
                let mut output = input.clone();
                for node in nodes {
                    output = self.execute_node(node, &output)?;
                }
                Ok(output)
            }

            GraphNode::Parallel { nodes, combiner } => {
                history.push(format!("Parallel execution with combiner: {}", combiner));
                let mut results = Vec::new();
                for node in nodes {
                    results.push(self.execute_node(node, input)?);
                }

                let combiner_fn = self.combiners.get(combiner).ok_or_else(|| {
                    TorshError::InvalidArgument(format!("Combiner '{}' not found", combiner))
                })?;
                combiner_fn(results)
            }

            GraphNode::Loop {
                body,
                condition,
                max_iterations,
            } => {
                history.push(format!("Loop execution with condition: {}", condition));
                let cond_fn = self.conditions.get(condition).ok_or_else(|| {
                    TorshError::InvalidArgument(format!("Condition '{}' not found", condition))
                })?;

                let mut output = input.clone();
                let mut iterations = 0;

                while cond_fn(&output) && iterations < *max_iterations {
                    output = self.execute_node(body, &output)?;
                    iterations += 1;
                    history.push(format!("Loop iteration: {}", iterations));
                }

                Ok(output)
            }

            GraphNode::Function(name) => {
                history.push(format!("Function: {}", name));
                let function = self.functions.get(name).ok_or_else(|| {
                    TorshError::InvalidArgument(format!("Function '{}' not found", name))
                })?;
                function(input)
            }
        }
    }

    /// Get execution history for debugging
    pub fn get_execution_history(&self) -> Vec<String> {
        self.execution_history.lock().clone()
    }

    /// Clear execution history
    pub fn clear_execution_history(&self) {
        self.execution_history.lock().clear();
    }

    /// Dynamically modify the graph at runtime
    pub fn modify_graph<F>(&mut self, modifier: F)
    where
        F: FnOnce(&mut GraphNode),
    {
        modifier(&mut self.graph);
    }

    /// Get a reference to a specific module
    pub fn get_module(&self, name: &str) -> Option<&dyn Module> {
        self.modules.get(name).map(|m| m.as_ref())
    }

    /// Replace a module at runtime
    pub fn replace_module<M: Module + 'static>(&mut self, name: String, module: M) {
        self.modules.insert(name, Box::new(module));
    }

    /// Remove a module
    pub fn remove_module(&mut self, name: &str) -> Option<Box<dyn Module>> {
        self.modules.remove(name)
    }

    /// Get the number of modules
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }

    /// List all module names
    pub fn module_names(&self) -> Vec<&String> {
        self.modules.keys().collect()
    }

    /// List all condition names
    pub fn condition_names(&self) -> Vec<&String> {
        self.conditions.keys().collect()
    }

    /// List all combiner names
    pub fn combiner_names(&self) -> Vec<&String> {
        self.combiners.keys().collect()
    }

    /// List all function names
    pub fn function_names(&self) -> Vec<&String> {
        self.functions.keys().collect()
    }
}

impl Default for DynamicGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for DynamicGraph {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.clear_execution_history();
        self.execute_node(&self.graph, input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (module_name, module) in &self.modules {
            for (param_name, param) in module.parameters() {
                params.insert(format!("{}.{}", module_name, param_name), param);
            }
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (module_name, module) in &self.modules {
            for (param_name, param) in module.named_parameters() {
                params.insert(format!("{}.{}", module_name, param_name), param);
            }
        }

        params
    }

    fn train(&mut self) {
        self.base.set_training(true);
        for module in self.modules.values_mut() {
            module.train();
        }
    }

    fn eval(&mut self) {
        self.base.set_training(false);
        for module in self.modules.values_mut() {
            module.eval();
        }
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
        for module in self.modules.values_mut() {
            module.set_training(training);
        }
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)?;
        for module in self.modules.values_mut() {
            module.to_device(device)?;
        }
        Ok(())
    }

    fn children(&self) -> Vec<&dyn Module> {
        self.modules.values().map(|m| m.as_ref()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock module for testing
    struct MockModule {
        base: ModuleBase,
        id: i32,
    }

    impl MockModule {
        fn new(id: i32) -> Self {
            Self {
                base: ModuleBase::new(),
                id,
            }
        }
    }

    impl Module for MockModule {
        fn forward(&self, input: &Tensor) -> Result<Tensor> {
            // Simple identity function for testing
            Ok(input.clone())
        }

        fn parameters(&self) -> HashMap<String, Parameter> {
            HashMap::new()
        }

        fn named_parameters(&self) -> HashMap<String, Parameter> {
            HashMap::new()
        }

        fn train(&mut self) {
            self.base.set_training(true);
        }

        fn eval(&mut self) {
            self.base.set_training(false);
        }

        fn training(&self) -> bool {
            self.base.training()
        }

        fn set_training(&mut self, training: bool) {
            self.base.set_training(training);
        }

        fn to_device(&mut self, device: DeviceType) -> Result<()> {
            self.base.to_device(device)
        }
    }

    #[test]
    fn test_dynamic_graph_creation() {
        let graph = DynamicGraph::new();
        assert_eq!(graph.module_count(), 0);
        assert!(graph.module_names().is_empty());
        assert!(graph.training());
    }

    #[test]
    fn test_module_management() {
        let mut graph = DynamicGraph::new();

        graph.add_module("mock1".to_string(), MockModule::new(1));
        graph.add_module("mock2".to_string(), MockModule::new(2));

        assert_eq!(graph.module_count(), 2);
        assert!(graph.get_module("mock1").is_some());
        assert!(graph.get_module("nonexistent").is_none());

        let removed = graph.remove_module("mock1");
        assert!(removed.is_some());
        assert_eq!(graph.module_count(), 1);
    }

    #[test]
    fn test_graph_node_creation() {
        // Test sequential graph creation
        let seq_graph =
            DynamicGraph::sequential(vec!["module1".to_string(), "module2".to_string()]);

        match seq_graph {
            GraphNode::Sequence(nodes) => {
                assert_eq!(nodes.len(), 2);
            }
            _ => panic!("Expected Sequence node"),
        }

        // Test conditional graph creation
        let cond_graph = DynamicGraph::conditional(
            "test_condition".to_string(),
            GraphNode::Module("true_module".to_string()),
            Some(GraphNode::Module("false_module".to_string())),
        );

        match cond_graph {
            GraphNode::Conditional { condition, .. } => {
                assert_eq!(condition, "test_condition");
            }
            _ => panic!("Expected Conditional node"),
        }
    }

    #[test]
    fn test_default_combiners() {
        let graph = DynamicGraph::new();

        // Should have default combiners
        let combiners = graph.combiner_names();
        assert!(combiners.iter().any(|&name| name == "add"));
        assert!(combiners.iter().any(|&name| name == "mean"));
        assert!(combiners.iter().any(|&name| name == "concat"));
    }

    #[test]
    fn test_execution_history() {
        let graph = DynamicGraph::new();

        assert!(graph.get_execution_history().is_empty());

        // Clear should work even when empty
        graph.clear_execution_history();
        assert!(graph.get_execution_history().is_empty());
    }
}
