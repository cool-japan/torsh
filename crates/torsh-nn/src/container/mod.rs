//! Container modules for organizing neural network layers and parameters
//!
//! This module provides various container types for organizing and managing neural network
//! components:
//!
//! ## Basic Containers
//! - [`Sequential`] - Executes modules in sequence
//! - [`ModuleList`] - List of modules without forward pass definition
//! - [`ModuleDict`] - Dictionary of modules without forward pass definition
//! - [`FunctionModule`] - Wraps functions as modules
//!
//! ## Parameter Containers
//! - [`ParameterList`] - List of trainable parameters
//! - [`ParameterDict`] - Dictionary of named trainable parameters
//!
//! ## Lazy Containers
//! - [`LazySequential`] - Sequential container with deferred module creation
//! - [`LazyModuleList`] - Module list with deferred module creation
//! - [`LazyModuleDict`] - Module dictionary with deferred module creation
//!
//! ## Dynamic Containers
//! - [`GraphNode`] - Represents nodes in dynamic computation graphs
//! - [`DynamicGraph`] - Runtime-modifiable computation graph

// Module declarations
pub mod basic;
pub mod dynamic_graph;
pub mod lazy;
pub mod parameters;

// Re-export all public types for backward compatibility
pub use basic::{FunctionModule, ModuleDict, ModuleList, Sequential};
pub use dynamic_graph::{DynamicGraph, GraphNode};
pub use lazy::{LazyModuleDict, LazyModuleList, LazySequential};
pub use parameters::{ParameterDict, ParameterList};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::activation::ReLU;
    use crate::layers::linear::Linear;
    use crate::Module;
    use torsh_core::error::Result;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_sequential_basic() -> Result<()> {
        let model = Sequential::new()
            .add(Linear::new(10, 20, true))
            .add(ReLU::new())
            .add(Linear::new(20, 5, true));

        let input = randn::<f32>(&[4, 10])?;
        let output = model.forward(&input)?;

        assert_eq!(output.shape().dims(), &[4, 5]);
        Ok(())
    }

    #[test]
    fn test_module_list() -> Result<()> {
        let mut module_list = ModuleList::new();
        module_list.push(Linear::new(8, 16, true));
        module_list.push(ReLU::new());
        module_list.push(Linear::new(16, 4, true));

        assert_eq!(module_list.len(), 3);
        assert!(!module_list.is_empty());

        // Test individual module access
        let linear1 = module_list.get(0).unwrap();
        let input = randn::<f32>(&[2, 8])?;
        let output = linear1.forward(&input)?;
        assert_eq!(output.shape().dims(), &[2, 16]);

        Ok(())
    }

    #[test]
    fn test_module_dict() -> Result<()> {
        let mut module_dict = ModuleDict::new();
        module_dict.insert("linear1".to_string(), Linear::new(6, 12, true));
        module_dict.insert("activation".to_string(), ReLU::new());
        module_dict.insert("linear2".to_string(), Linear::new(12, 3, true));

        assert_eq!(module_dict.len(), 3);
        assert!(!module_dict.is_empty());

        // Test individual module access
        let linear1 = module_dict.get("linear1").unwrap();
        let input = randn::<f32>(&[3, 6])?;
        let output = linear1.forward(&input)?;
        assert_eq!(output.shape().dims(), &[3, 12]);

        Ok(())
    }

    #[test]
    fn test_function_module() -> Result<()> {
        let relu_fn = FunctionModule::new(|x: &torsh_tensor::Tensor| x.clamp(0.0, f32::INFINITY));

        let input = randn::<f32>(&[2, 4])?;
        let output = relu_fn.forward(&input)?;
        assert_eq!(output.shape().dims(), &[2, 4]);

        Ok(())
    }

    #[test]
    fn test_parameter_list() -> Result<()> {
        use crate::Parameter;
        use torsh_tensor::creation::randn;

        let mut param_list = ParameterList::new();

        let weight1 = Parameter::new(randn::<f32>(&[10, 20])?);
        let weight2 = Parameter::new(randn::<f32>(&[20, 5])?);

        param_list.push(weight1.clone());
        param_list.push(weight2.clone());

        assert_eq!(param_list.len(), 2);
        assert!(!param_list.is_empty());

        let params = param_list.parameters();
        assert_eq!(params.len(), 2);
        assert!(params.contains_key("0"));
        assert!(params.contains_key("1"));

        Ok(())
    }

    #[test]
    fn test_parameter_dict() -> Result<()> {
        use crate::Parameter;
        use torsh_tensor::creation::randn;

        let mut param_dict = ParameterDict::new();

        let weight = Parameter::new(randn::<f32>(&[8, 16])?);
        let bias = Parameter::new(randn::<f32>(&[16])?);

        param_dict.insert("weight".to_string(), weight.clone());
        param_dict.insert("bias".to_string(), bias.clone());

        assert_eq!(param_dict.len(), 2);
        assert!(!param_dict.is_empty());

        let params = param_dict.parameters();
        assert_eq!(params.len(), 2);
        assert!(params.contains_key("weight"));
        assert!(params.contains_key("bias"));

        Ok(())
    }

    #[test]
    fn test_lazy_sequential() -> Result<()> {
        let lazy_seq = LazySequential::new()
            .add_factory(|input_shape| {
                let in_features = input_shape[input_shape.len() - 1];
                Ok(Box::new(Linear::new(in_features, 32, true)) as Box<dyn crate::Module>)
            })
            .add_factory(|_| Ok(Box::new(ReLU::new()) as Box<dyn crate::Module>))
            .add_factory(|_| Ok(Box::new(Linear::new(32, 10, true)) as Box<dyn crate::Module>));

        assert!(!lazy_seq.is_initialized());
        assert_eq!(lazy_seq.len(), 3);

        let input = randn::<f32>(&[4, 16])?;
        let output = lazy_seq.forward(&input)?;

        assert!(lazy_seq.is_initialized());
        assert_eq!(output.shape().dims(), &[4, 10]);

        Ok(())
    }

    #[test]
    #[ignore] // Slow test - takes over 14 minutes
    fn test_dynamic_graph_simple() -> Result<()> {
        let mut graph = DynamicGraph::new();

        graph.add_module("linear".to_string(), Linear::new(5, 8, true));
        graph.add_module("relu".to_string(), ReLU::new());

        graph.set_graph(DynamicGraph::sequential(vec![
            "linear".to_string(),
            "relu".to_string(),
        ]));

        let input = randn::<f32>(&[3, 5])?;
        let output = graph.forward(&input)?;

        assert_eq!(output.shape().dims(), &[3, 8]);

        let history = graph.get_execution_history();
        assert!(!history.is_empty());

        Ok(())
    }

    #[test]
    fn test_container_parameters() -> Result<()> {
        let model = Sequential::new()
            .add(Linear::new(4, 8, true))
            .add(ReLU::new())
            .add(Linear::new(8, 2, false)); // No bias

        let params = model.parameters();
        let named_params = model.named_parameters();

        // Should have 3 parameters: weight1, bias1, weight2
        assert_eq!(params.len(), 3);
        assert_eq!(named_params.len(), 3);

        assert!(named_params.contains_key("0.weight"));
        assert!(named_params.contains_key("0.bias"));
        assert!(named_params.contains_key("2.weight"));
        assert!(!named_params.contains_key("2.bias")); // No bias for second linear

        Ok(())
    }

    #[test]
    fn test_container_training_modes() -> Result<()> {
        let mut model = Sequential::new()
            .add(Linear::new(3, 6, true))
            .add(ReLU::new());

        // Test training mode
        assert!(model.training());
        model.eval();
        assert!(!model.training());
        model.train();
        assert!(model.training());

        // Test set_training
        model.set_training(false);
        assert!(!model.training());
        model.set_training(true);
        assert!(model.training());

        Ok(())
    }
}

// Additional comprehensive tests from the original container.rs
#[cfg(test)]
mod dynamic_graph_tests {
    use super::*;
    use crate::layers::activation::ReLU;
    use crate::layers::linear::Linear;
    use crate::Module;
    use torsh_core::error::Result;
    use torsh_tensor::creation::randn;

    #[test]
    #[ignore] // Slow test - takes over 14 minutes
    fn test_dynamic_graph_sequential() -> Result<()> {
        let mut graph = DynamicGraph::new();

        // Add modules
        graph.add_module("linear1".to_string(), Linear::new(10, 20, true));
        graph.add_module("relu".to_string(), ReLU::new());
        graph.add_module("linear2".to_string(), Linear::new(20, 5, true));

        // Set sequential graph
        graph.set_graph(DynamicGraph::sequential(vec![
            "linear1".to_string(),
            "relu".to_string(),
            "linear2".to_string(),
        ]));

        let input = randn::<f32>(&[4, 10])?;
        let output = graph.forward(&input)?;

        assert_eq!(output.shape().dims(), &[4, 5]);

        // Check execution history
        let history = graph.get_execution_history();
        assert!(history.len() > 0);
        assert!(history.iter().any(|h| h.contains("linear1")));
        assert!(history.iter().any(|h| h.contains("relu")));
        assert!(history.iter().any(|h| h.contains("linear2")));

        Ok(())
    }

    #[test]
    #[ignore] // Slow test - takes over 14 minutes
    fn test_dynamic_graph_conditional() -> Result<()> {
        let mut graph = DynamicGraph::new();

        // Add modules
        graph.add_module("path_a".to_string(), Linear::new(8, 16, true));
        graph.add_module("path_b".to_string(), Linear::new(8, 16, true));

        // Add condition based on input sum
        graph.add_condition("input_sum_positive".to_string(), |tensor| {
            let sum: f32 = tensor.to_vec().map(|v| v.iter().sum()).unwrap_or(0.0);
            sum > 0.0
        });

        // Set conditional graph
        graph.set_graph(DynamicGraph::conditional(
            "input_sum_positive".to_string(),
            GraphNode::Module("path_a".to_string()),
            Some(GraphNode::Module("path_b".to_string())),
        ));

        // Test with positive input
        let positive_input = randn::<f32>(&[2, 8])?.add_scalar(10.0)?; // Add constant to ensure positive
        let output1 = graph.forward(&positive_input)?;
        assert_eq!(output1.shape().dims(), &[2, 16]);

        let history1 = graph.get_execution_history();
        assert!(history1.iter().any(|h| h.contains("path_a")));

        // Test with negative input
        let negative_input = randn::<f32>(&[2, 8])?.add_scalar(-10.0)?; // Subtract to ensure negative
        let output2 = graph.forward(&negative_input)?;
        assert_eq!(output2.shape().dims(), &[2, 16]);

        let history2 = graph.get_execution_history();
        assert!(history2.iter().any(|h| h.contains("path_b")));

        Ok(())
    }

    #[test]
    #[ignore] // Slow test - takes over 14 minutes
    fn test_dynamic_graph_modification() -> Result<()> {
        let mut graph = DynamicGraph::new();

        // Add initial modules
        graph.add_module("initial".to_string(), Linear::new(5, 10, true));
        graph.set_graph(GraphNode::Module("initial".to_string()));

        let input = randn::<f32>(&[2, 5])?;
        let output1 = graph.forward(&input)?;
        assert_eq!(output1.shape().dims(), &[2, 10]);

        // Dynamically modify the graph
        graph.add_module("additional".to_string(), ReLU::new());
        graph.modify_graph(|graph_node| {
            *graph_node = GraphNode::Sequence(vec![
                GraphNode::Module("initial".to_string()),
                GraphNode::Module("additional".to_string()),
            ]);
        });

        let output2 = graph.forward(&input)?;
        assert_eq!(output2.shape().dims(), &[2, 10]);

        Ok(())
    }
}

#[cfg(test)]
mod lazy_tests {
    use super::*;
    use crate::layers::activation::ReLU;
    use crate::layers::linear::Linear;
    use crate::Module;
    use torsh_core::error::Result;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_lazy_sequential_basic() -> Result<()> {
        // Create a lazy sequential with factory functions
        let lazy_seq = LazySequential::new()
            .add_factory(|input_shape| {
                // First layer: Linear with input features inferred from shape
                let in_features = input_shape[input_shape.len() - 1];
                Ok(Box::new(Linear::new(in_features, 64, true)) as Box<dyn crate::Module>)
            })
            .add_factory(|_input_shape| {
                // ReLU doesn't depend on input shape
                Ok(Box::new(ReLU::new()) as Box<dyn crate::Module>)
            })
            .add_factory(|_input_shape| {
                // Output layer: always 64 -> 10
                Ok(Box::new(Linear::new(64, 10, true)) as Box<dyn crate::Module>)
            });

        assert!(!lazy_seq.is_initialized());
        assert_eq!(lazy_seq.len(), 3);

        // Create input tensor with 32 features
        let input = randn::<f32>(&[8, 32])?;

        // Forward pass should initialize and execute
        let output = lazy_seq.forward(&input)?;

        assert!(lazy_seq.is_initialized());
        assert_eq!(output.shape().dims(), &[8, 10]);

        // Parameters should be available after initialization
        let params = lazy_seq.parameters();
        assert!(!params.is_empty());

        Ok(())
    }

    #[test]
    fn test_lazy_module_list() -> Result<()> {
        let mut lazy_list = LazyModuleList::new();

        // Add factory functions
        lazy_list.push_factory(|input_shape| {
            let in_features = input_shape[input_shape.len() - 1];
            Ok(Box::new(Linear::new(in_features, 128, true)) as Box<dyn crate::Module>)
        });

        lazy_list.push_factory(|_| Ok(Box::new(ReLU::new()) as Box<dyn crate::Module>));

        assert!(!lazy_list.is_initialized());
        assert_eq!(lazy_list.len(), 2);

        // Initialize with input shape
        lazy_list.initialize_lazy(&[4, 20])?;

        assert!(lazy_list.is_initialized());

        Ok(())
    }

    #[test]
    fn test_lazy_module_dict() -> Result<()> {
        let mut lazy_dict = LazyModuleDict::new();

        // Add factory functions with named keys
        lazy_dict.insert_factory("encoder".to_string(), |input_shape| {
            let in_features = input_shape[input_shape.len() - 1];
            Ok(Box::new(Linear::new(in_features, 256, true)) as Box<dyn crate::Module>)
        });

        lazy_dict.insert_factory("activation".to_string(), |_| {
            Ok(Box::new(ReLU::new()) as Box<dyn crate::Module>)
        });

        lazy_dict.insert_factory("decoder".to_string(), |_| {
            Ok(Box::new(Linear::new(256, 10, true)) as Box<dyn crate::Module>)
        });

        assert!(!lazy_dict.is_initialized());
        assert_eq!(lazy_dict.len(), 3);

        // Initialize with input shape
        lazy_dict.initialize_lazy(&[2, 100])?;

        assert!(lazy_dict.is_initialized());

        // Test key iteration
        let keys: Vec<_> = lazy_dict.module_keys();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&"encoder".to_string()));
        assert!(keys.contains(&"activation".to_string()));
        assert!(keys.contains(&"decoder".to_string()));

        Ok(())
    }
}
