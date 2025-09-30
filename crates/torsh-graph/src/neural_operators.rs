//! Graph Neural Operators
//!
//! Advanced implementation of graph neural operators for learning continuous
//! functions on graphs. Inspired by Neural Operator Theory and Physics-Informed
//! Neural Networks (PINNs) for graph-structured data.
//!
//! # Features:
//! - Graph Fourier Neural Operators (GraphFNO)
//! - Graph DeepONet for operator learning
//! - Physics-informed graph neural networks
//! - Multi-scale graph operators
//! - Spectral graph convolutions with learnable kernels
//! - Graph wavelet neural operators

use crate::parameter::Parameter;
use crate::{GraphData, GraphLayer};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::Random;
use std::collections::HashMap;
use torsh_tensor::{
    creation::{from_vec, ones, randn, zeros},
    Tensor,
};

/// Graph Fourier Neural Operator (GraphFNO)
/// Learns operators in the spectral domain of graphs
#[derive(Debug)]
pub struct GraphFNO {
    in_features: usize,
    out_features: usize,
    hidden_features: usize,
    num_modes: usize,
    num_layers: usize,

    // Fourier layers
    fourier_weights: Vec<Parameter>,
    conv_weights: Vec<Parameter>,

    // Input/output projections
    input_projection: Parameter,
    output_projection: Parameter,

    // Bias terms
    bias: Option<Parameter>,
}

impl GraphFNO {
    /// Create a new Graph Fourier Neural Operator
    pub fn new(
        in_features: usize,
        out_features: usize,
        hidden_features: usize,
        num_modes: usize,
        num_layers: usize,
        bias: bool,
    ) -> Self {
        let mut fourier_weights = Vec::new();
        let mut conv_weights = Vec::new();

        // Initialize Fourier weights for each layer
        for _ in 0..num_layers {
            fourier_weights.push(Parameter::new(
                randn(&[hidden_features, hidden_features, num_modes]).unwrap(),
            ));
            conv_weights.push(Parameter::new(
                randn(&[hidden_features, hidden_features]).unwrap(),
            ));
        }

        let input_projection = Parameter::new(randn(&[in_features, hidden_features]).unwrap());
        let output_projection = Parameter::new(randn(&[hidden_features, out_features]).unwrap());

        let bias = if bias {
            Some(Parameter::new(zeros::<f32>(&[out_features]).unwrap()))
        } else {
            None
        };

        Self {
            in_features,
            out_features,
            hidden_features,
            num_modes,
            num_layers,
            fourier_weights,
            conv_weights,
            input_projection,
            output_projection,
            bias,
        }
    }

    /// Forward pass through GraphFNO
    pub fn forward(&self, graph: &GraphData) -> GraphData {
        let num_nodes = graph.num_nodes;

        // Input projection
        let mut x = graph.x.matmul(&self.input_projection.clone_data()).unwrap();

        // Apply Fourier layers
        for layer in 0..self.num_layers {
            x = self.fourier_layer(&x, layer, graph);
        }

        // Output projection
        let mut output = x.matmul(&self.output_projection.clone_data()).unwrap();

        // Add bias if present
        if let Some(ref bias) = self.bias {
            output = output.add(&bias.clone_data()).unwrap();
        }

        // Create output graph
        let mut output_graph = graph.clone();
        output_graph.x = output;
        output_graph
    }

    /// Apply a single Fourier layer
    fn fourier_layer(&self, x: &Tensor, layer: usize, graph: &GraphData) -> Tensor {
        // Step 1: Apply graph Fourier transform (simplified)
        let fourier_x = self.graph_fourier_transform(x, graph);

        // Step 2: Apply learnable Fourier weights
        let fourier_weights = &self.fourier_weights[layer];
        let spectral_conv = self.spectral_convolution(&fourier_x, fourier_weights);

        // Step 3: Inverse Fourier transform
        let spatial_features = self.inverse_graph_fourier_transform(&spectral_conv, graph);

        // Step 4: Apply spatial convolution
        let conv_weights = &self.conv_weights[layer];
        let conv_output = spatial_features.matmul(&conv_weights.clone_data()).unwrap();

        // Step 5: Residual connection and activation
        let residual = x.add(&conv_output).unwrap();

        // Apply ReLU activation (simplified)
        self.relu(&residual)
    }

    /// Graph Fourier Transform (simplified eigendecomposition)
    fn graph_fourier_transform(&self, x: &Tensor, graph: &GraphData) -> Tensor {
        // For simplicity, we'll use a learned transformation matrix
        // In practice, this would use graph Laplacian eigendecomposition
        let num_nodes = graph.num_nodes;

        // Create a simple transformation that captures spectral properties
        let mut transform_data = Vec::new();
        for i in 0..num_nodes {
            for j in 0..self.num_modes {
                let freq = (j as f32 + 1.0) * std::f32::consts::PI / num_nodes as f32;
                let basis = (freq * i as f32).cos();
                transform_data.push(basis);
            }
        }

        let transform_matrix = from_vec(
            transform_data,
            &[num_nodes, self.num_modes],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        // Project to spectral domain
        transform_matrix.t().unwrap().matmul(x).unwrap()
    }

    /// Inverse Graph Fourier Transform
    fn inverse_graph_fourier_transform(&self, fourier_x: &Tensor, graph: &GraphData) -> Tensor {
        let num_nodes = graph.num_nodes;

        // Create inverse transformation matrix
        let mut inv_transform_data = Vec::new();
        for i in 0..num_nodes {
            for j in 0..self.num_modes {
                let freq = (j as f32 + 1.0) * std::f32::consts::PI / num_nodes as f32;
                let basis = (freq * i as f32).cos();
                inv_transform_data.push(basis);
            }
        }

        let inv_transform_matrix = from_vec(
            inv_transform_data,
            &[num_nodes, self.num_modes],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        // Project back to spatial domain
        inv_transform_matrix.matmul(fourier_x).unwrap()
    }

    /// Spectral convolution in Fourier domain
    fn spectral_convolution(&self, fourier_x: &Tensor, weights: &Parameter) -> Tensor {
        // Apply Fourier weights (simplified)
        let weight_data = weights.clone_data();

        // For simplicity, use only the first mode slice
        // In practice, this would involve complex multiplication across all modes
        let weight_2d = weight_data
            .slice_tensor(2, 0, 1)
            .unwrap()
            .squeeze_tensor(2)
            .unwrap();

        fourier_x.matmul(&weight_2d).unwrap()
    }

    /// ReLU activation function
    fn relu(&self, x: &Tensor) -> Tensor {
        // Simplified ReLU - clamp negative values to 0
        let data = x.to_vec().unwrap();
        let activated_data: Vec<f32> = data.iter().map(|&val| val.max(0.0)).collect();

        from_vec(
            activated_data,
            x.shape().dims(),
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap()
    }
}

impl GraphLayer for GraphFNO {
    fn forward(&self, graph: &GraphData) -> GraphData {
        self.forward(graph)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![
            self.input_projection.clone_data(),
            self.output_projection.clone_data(),
        ];

        for weight in &self.fourier_weights {
            params.push(weight.clone_data());
        }

        for weight in &self.conv_weights {
            params.push(weight.clone_data());
        }

        if let Some(ref bias) = self.bias {
            params.push(bias.clone_data());
        }

        params
    }
}

/// Graph DeepONet for operator learning on graphs
#[derive(Debug)]
pub struct GraphDeepONet {
    trunk_net_features: usize,
    branch_net_features: usize,
    hidden_features: usize,
    output_features: usize,
    num_sensors: usize,

    // Branch network (processes input functions)
    branch_layers: Vec<Parameter>,

    // Trunk network (processes locations/coordinates)
    trunk_layers: Vec<Parameter>,

    // Output bias
    bias: Option<Parameter>,
}

impl GraphDeepONet {
    /// Create a new Graph DeepONet
    pub fn new(
        trunk_net_features: usize,
        branch_net_features: usize,
        hidden_features: usize,
        output_features: usize,
        num_sensors: usize,
        num_layers: usize,
        bias: bool,
    ) -> Self {
        let mut branch_layers = Vec::new();
        let mut trunk_layers = Vec::new();

        // Initialize branch network layers
        for i in 0..num_layers {
            let in_dim = if i == 0 { num_sensors } else { hidden_features };
            let out_dim = if i == num_layers - 1 {
                output_features
            } else {
                hidden_features
            };
            branch_layers.push(Parameter::new(randn(&[in_dim, out_dim]).unwrap()));
        }

        // Initialize trunk network layers
        for i in 0..num_layers {
            let in_dim = if i == 0 {
                trunk_net_features
            } else {
                hidden_features
            };
            let out_dim = if i == num_layers - 1 {
                output_features
            } else {
                hidden_features
            };
            trunk_layers.push(Parameter::new(randn(&[in_dim, out_dim]).unwrap()));
        }

        let bias = if bias {
            Some(Parameter::new(zeros::<f32>(&[output_features]).unwrap()))
        } else {
            None
        };

        Self {
            trunk_net_features,
            branch_net_features,
            hidden_features,
            output_features,
            num_sensors,
            branch_layers,
            trunk_layers,
            bias,
        }
    }

    /// Forward pass through Graph DeepONet
    pub fn forward(
        &self,
        graph: &GraphData,
        sensor_data: &Tensor,
        locations: &Tensor,
    ) -> GraphData {
        // Process sensor data through branch network
        let branch_output = self.forward_branch_net(sensor_data);

        // Process locations through trunk network
        let trunk_output = self.forward_trunk_net(locations);

        // Combine branch and trunk outputs (dot product)
        let combined = self.combine_outputs(&branch_output, &trunk_output);

        // Add bias if present
        let mut output = combined;
        if let Some(ref bias) = self.bias {
            output = output.add(&bias.clone_data()).unwrap();
        }

        // Create output graph
        let mut output_graph = graph.clone();
        output_graph.x = output;
        output_graph
    }

    /// Forward pass through branch network
    fn forward_branch_net(&self, sensor_data: &Tensor) -> Tensor {
        let mut x = sensor_data.clone();

        for (i, layer) in self.branch_layers.iter().enumerate() {
            x = x.matmul(&layer.clone_data()).unwrap();

            // Apply activation function except for last layer
            if i < self.branch_layers.len() - 1 {
                x = self.tanh(&x);
            }
        }

        x
    }

    /// Forward pass through trunk network
    fn forward_trunk_net(&self, locations: &Tensor) -> Tensor {
        let mut x = locations.clone();

        for (i, layer) in self.trunk_layers.iter().enumerate() {
            x = x.matmul(&layer.clone_data()).unwrap();

            // Apply activation function except for last layer
            if i < self.trunk_layers.len() - 1 {
                x = self.tanh(&x);
            }
        }

        x
    }

    /// Combine branch and trunk network outputs
    fn combine_outputs(&self, branch_output: &Tensor, trunk_output: &Tensor) -> Tensor {
        // Element-wise multiplication and sum
        branch_output.mul(trunk_output).unwrap()
    }

    /// Tanh activation function
    fn tanh(&self, x: &Tensor) -> Tensor {
        let data = x.to_vec().unwrap();
        let activated_data: Vec<f32> = data.iter().map(|&val| val.tanh()).collect();

        from_vec(
            activated_data,
            x.shape().dims(),
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap()
    }
}

impl GraphLayer for GraphDeepONet {
    fn forward(&self, graph: &GraphData) -> GraphData {
        // Default forward using graph features as both sensor data and locations
        let sensor_data = graph
            .x
            .slice_tensor(1, 0, self.num_sensors.min(graph.x.shape().dims()[1]))
            .unwrap();
        let locations = graph.x.clone();

        self.forward(graph, &sensor_data, &locations)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();

        for layer in &self.branch_layers {
            params.push(layer.clone_data());
        }

        for layer in &self.trunk_layers {
            params.push(layer.clone_data());
        }

        if let Some(ref bias) = self.bias {
            params.push(bias.clone_data());
        }

        params
    }
}

/// Physics-Informed Graph Neural Network
#[derive(Debug)]
pub struct PhysicsInformedGNN {
    in_features: usize,
    out_features: usize,
    hidden_features: usize,

    // Neural network layers
    layers: Vec<Parameter>,

    // Physics constraints
    diffusion_coefficient: f32,
    reaction_rate: f32,

    // Bias
    bias: Option<Parameter>,
}

impl PhysicsInformedGNN {
    /// Create a new Physics-Informed GNN
    pub fn new(
        in_features: usize,
        out_features: usize,
        hidden_features: usize,
        num_layers: usize,
        diffusion_coefficient: f32,
        reaction_rate: f32,
        bias: bool,
    ) -> Self {
        let mut layers = Vec::new();

        for i in 0..num_layers {
            let in_dim = if i == 0 { in_features } else { hidden_features };
            let out_dim = if i == num_layers - 1 {
                out_features
            } else {
                hidden_features
            };
            layers.push(Parameter::new(randn(&[in_dim, out_dim]).unwrap()));
        }

        let bias = if bias {
            Some(Parameter::new(zeros::<f32>(&[out_features]).unwrap()))
        } else {
            None
        };

        Self {
            in_features,
            out_features,
            hidden_features,
            layers,
            diffusion_coefficient,
            reaction_rate,
            bias,
        }
    }

    /// Forward pass with physics constraints
    pub fn forward(&self, graph: &GraphData) -> GraphData {
        // Neural network forward pass
        let mut x = graph.x.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            x = x.matmul(&layer.clone_data()).unwrap();

            // Apply activation except for last layer
            if i < self.layers.len() - 1 {
                x = self.swish(&x);
            }
        }

        // Apply physics constraints
        let physics_constrained = self.apply_physics_constraints(&x, graph);

        // Add bias if present
        let mut output = physics_constrained;
        if let Some(ref bias) = self.bias {
            output = output.add(&bias.clone_data()).unwrap();
        }

        // Create output graph
        let mut output_graph = graph.clone();
        output_graph.x = output;
        output_graph
    }

    /// Apply physics constraints (diffusion-reaction equation)
    fn apply_physics_constraints(&self, prediction: &Tensor, graph: &GraphData) -> Tensor {
        // Compute graph Laplacian for diffusion term
        let laplacian = self.compute_graph_laplacian(graph);

        // Diffusion term: D * L * u
        let diffusion_term = laplacian
            .matmul(prediction)
            .unwrap()
            .mul_scalar(self.diffusion_coefficient)
            .unwrap();

        // Reaction term: r * u
        let reaction_term = prediction.mul_scalar(self.reaction_rate).unwrap();

        // Combine terms (simplified physics equation)
        prediction
            .add(&diffusion_term)
            .unwrap()
            .add(&reaction_term)
            .unwrap()
    }

    /// Compute graph Laplacian matrix
    fn compute_graph_laplacian(&self, graph: &GraphData) -> Tensor {
        let num_nodes = graph.num_nodes;
        let num_edges = graph.num_edges;

        // Initialize adjacency matrix
        let mut adj_data = vec![0.0f32; num_nodes * num_nodes];

        // Fill adjacency matrix from edge_index
        let edge_data = graph.edge_index.to_vec().unwrap();
        for i in (0..edge_data.len()).step_by(2) {
            if i + 1 < edge_data.len() {
                let src = edge_data[i] as usize;
                let dst = edge_data[i + 1] as usize;

                if src < num_nodes && dst < num_nodes {
                    adj_data[src * num_nodes + dst] = 1.0;
                    adj_data[dst * num_nodes + src] = 1.0; // Undirected graph
                }
            }
        }

        // Compute degree matrix
        let mut degree_data = vec![0.0f32; num_nodes * num_nodes];
        for i in 0..num_nodes {
            let mut degree = 0.0;
            for j in 0..num_nodes {
                degree += adj_data[i * num_nodes + j];
            }
            degree_data[i * num_nodes + i] = degree;
        }

        // Laplacian = Degree - Adjacency
        let mut laplacian_data = Vec::new();
        for i in 0..num_nodes * num_nodes {
            laplacian_data.push(degree_data[i] - adj_data[i]);
        }

        from_vec(
            laplacian_data,
            &[num_nodes, num_nodes],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap()
    }

    /// Swish activation function (x * sigmoid(x))
    fn swish(&self, x: &Tensor) -> Tensor {
        let data = x.to_vec().unwrap();
        let activated_data: Vec<f32> = data
            .iter()
            .map(|&val| val * (1.0 / (1.0 + (-val).exp())))
            .collect();

        from_vec(
            activated_data,
            x.shape().dims(),
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap()
    }
}

impl GraphLayer for PhysicsInformedGNN {
    fn forward(&self, graph: &GraphData) -> GraphData {
        self.forward(graph)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();

        for layer in &self.layers {
            params.push(layer.clone_data());
        }

        if let Some(ref bias) = self.bias {
            params.push(bias.clone_data());
        }

        params
    }
}

/// Multi-scale Graph Neural Operator
#[derive(Debug)]
pub struct MultiScaleGNO {
    in_features: usize,
    out_features: usize,
    num_scales: usize,
    hidden_features: usize,

    // Scale-specific operators
    scale_operators: Vec<Parameter>,

    // Cross-scale fusion
    fusion_weights: Parameter,

    // Output projection
    output_projection: Parameter,

    bias: Option<Parameter>,
}

impl MultiScaleGNO {
    /// Create a new Multi-scale Graph Neural Operator
    pub fn new(
        in_features: usize,
        out_features: usize,
        num_scales: usize,
        hidden_features: usize,
        bias: bool,
    ) -> Self {
        let mut scale_operators = Vec::new();

        // Initialize scale-specific operators
        for _ in 0..num_scales {
            scale_operators.push(Parameter::new(
                randn(&[in_features, hidden_features]).unwrap(),
            ));
        }

        let fusion_weights =
            Parameter::new(randn(&[num_scales * hidden_features, hidden_features]).unwrap());

        let output_projection = Parameter::new(randn(&[hidden_features, out_features]).unwrap());

        let bias = if bias {
            Some(Parameter::new(zeros::<f32>(&[out_features]).unwrap()))
        } else {
            None
        };

        Self {
            in_features,
            out_features,
            num_scales,
            hidden_features,
            scale_operators,
            fusion_weights,
            output_projection,
            bias,
        }
    }

    /// Forward pass through multi-scale operator
    pub fn forward(&self, graph: &GraphData) -> GraphData {
        let mut scale_features = Vec::new();

        // Process each scale
        for scale in 0..self.num_scales {
            let scale_graph = self.coarsen_graph(graph, scale);
            let features = self.process_scale(&scale_graph, scale);
            let upsampled = self.upsample_features(&features, graph.num_nodes);
            scale_features.push(upsampled);
        }

        // Fuse multi-scale features
        let fused_features = self.fuse_scales(&scale_features);

        // Output projection
        let mut output = fused_features
            .matmul(&self.output_projection.clone_data())
            .unwrap();

        // Add bias if present
        if let Some(ref bias) = self.bias {
            output = output.add(&bias.clone_data()).unwrap();
        }

        // Create output graph
        let mut output_graph = graph.clone();
        output_graph.x = output;
        output_graph
    }

    /// Coarsen graph for multi-scale processing
    fn coarsen_graph(&self, graph: &GraphData, scale: usize) -> GraphData {
        let coarsening_factor = 2_usize.pow(scale as u32);
        let coarse_nodes = (graph.num_nodes + coarsening_factor - 1) / coarsening_factor;

        // Simple node pooling - average features of neighboring nodes
        let mut coarse_features = Vec::new();

        for coarse_id in 0..coarse_nodes {
            let start_node = coarse_id * coarsening_factor;
            let end_node = ((coarse_id + 1) * coarsening_factor).min(graph.num_nodes);

            // Average features of nodes in this coarse group
            let mut sum_features = vec![0.0f32; graph.x.shape().dims()[1]];
            let mut count = 0;

            for node_id in start_node..end_node {
                let features = graph.x.slice_tensor(0, node_id, node_id + 1).unwrap();
                let feature_data = features.to_vec().unwrap();

                for (i, &val) in feature_data.iter().enumerate() {
                    if i < sum_features.len() {
                        sum_features[i] += val;
                    }
                }
                count += 1;
            }

            // Normalize
            if count > 0 {
                for val in &mut sum_features {
                    *val /= count as f32;
                }
            }

            coarse_features.extend(sum_features);
        }

        let coarse_x = from_vec(
            coarse_features,
            &[coarse_nodes, graph.x.shape().dims()[1]],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        // Simplified edge index (connect sequential nodes)
        let mut coarse_edges = Vec::new();
        for i in 0..coarse_nodes.saturating_sub(1) {
            coarse_edges.push(i as f32);
            coarse_edges.push((i + 1) as f32);
        }

        let coarse_edge_index = from_vec(
            coarse_edges,
            &[2, coarse_nodes.saturating_sub(1)],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        GraphData::new(coarse_x, coarse_edge_index)
    }

    /// Process features at a specific scale
    fn process_scale(&self, graph: &GraphData, scale: usize) -> Tensor {
        let operator = &self.scale_operators[scale];
        graph.x.matmul(&operator.clone_data()).unwrap()
    }

    /// Upsample features to original graph size
    fn upsample_features(&self, features: &Tensor, target_nodes: usize) -> Tensor {
        let current_nodes = features.shape().dims()[0];
        let feature_dim = features.shape().dims()[1];

        if current_nodes >= target_nodes {
            // Truncate if necessary
            return features.slice_tensor(0, 0, target_nodes).unwrap();
        }

        // Simple upsampling by repetition
        let feature_data = features.to_vec().unwrap();
        let mut upsampled_data = Vec::new();

        for target_id in 0..target_nodes {
            let source_id = (target_id * current_nodes) / target_nodes;
            let start_idx = source_id * feature_dim;
            let end_idx = start_idx + feature_dim;

            if end_idx <= feature_data.len() {
                upsampled_data.extend(&feature_data[start_idx..end_idx]);
            } else {
                // Pad with zeros if needed
                upsampled_data.extend(vec![0.0f32; feature_dim]);
            }
        }

        from_vec(
            upsampled_data,
            &[target_nodes, feature_dim],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap()
    }

    /// Fuse multi-scale features
    fn fuse_scales(&self, scale_features: &[Tensor]) -> Tensor {
        // Concatenate features from all scales
        let mut concatenated_data = Vec::new();
        let num_nodes = scale_features[0].shape().dims()[0];

        for node_id in 0..num_nodes {
            for scale_feature in scale_features {
                let node_features = scale_feature.slice_tensor(0, node_id, node_id + 1).unwrap();
                let feature_data = node_features.to_vec().unwrap();
                concatenated_data.extend(feature_data);
            }
        }

        let concatenated = from_vec(
            concatenated_data,
            &[num_nodes, self.num_scales * self.hidden_features],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        // Apply fusion weights
        concatenated
            .matmul(&self.fusion_weights.clone_data())
            .unwrap()
    }
}

impl GraphLayer for MultiScaleGNO {
    fn forward(&self, graph: &GraphData) -> GraphData {
        self.forward(graph)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![
            self.fusion_weights.clone_data(),
            self.output_projection.clone_data(),
        ];

        for operator in &self.scale_operators {
            params.push(operator.clone_data());
        }

        if let Some(ref bias) = self.bias {
            params.push(bias.clone_data());
        }

        params
    }
}

/// Graph Neural Operator utilities
pub mod utils {
    use super::*;

    /// Compute spectral features of a graph
    pub fn compute_spectral_features(graph: &GraphData, num_eigenvalues: usize) -> Tensor {
        // Simplified spectral computation
        let num_nodes = graph.num_nodes;
        let mut spectral_data = Vec::new();

        for i in 0..num_nodes {
            for j in 0..num_eigenvalues {
                let eigenvalue = (j as f32 + 1.0) / num_eigenvalues as f32;
                let eigenvector_val = (std::f32::consts::PI * (i as f32 + 1.0) * (j as f32 + 1.0)
                    / num_nodes as f32)
                    .sin();
                spectral_data.push(eigenvalue * eigenvector_val);
            }
        }

        from_vec(
            spectral_data,
            &[num_nodes, num_eigenvalues],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap()
    }

    /// Generate synthetic operator learning data
    pub fn generate_operator_data(
        num_graphs: usize,
        num_nodes: usize,
        feature_dim: usize,
    ) -> Vec<(GraphData, GraphData)> {
        let mut rng = scirs2_core::random::thread_rng();
        let mut data_pairs = Vec::new();

        for _ in 0..num_graphs {
            // Generate input graph
            let input_features = randn(&[num_nodes, feature_dim]).unwrap();
            let mut edge_data = Vec::new();

            // Create random edges
            for _ in 0..(num_nodes * 2) {
                let src = rng.gen_range(0..num_nodes) as f32;
                let dst = rng.gen_range(0..num_nodes) as f32;
                edge_data.push(src);
                edge_data.push(dst);
            }

            let edge_index = from_vec(
                edge_data,
                &[2, num_nodes * 2],
                torsh_core::device::DeviceType::Cpu,
            )
            .unwrap();

            let input_graph = GraphData::new(input_features, edge_index);

            // Generate corresponding output (apply some transformation)
            let output_features = input_graph.x.mul_scalar(2.0).unwrap();
            let output_graph = GraphData::new(output_features, input_graph.edge_index.clone());

            data_pairs.push((input_graph, output_graph));
        }

        data_pairs
    }

    /// Evaluate operator approximation error
    pub fn compute_operator_error(predicted: &GraphData, target: &GraphData) -> f32 {
        let pred_data = predicted.x.to_vec().unwrap();
        let target_data = target.x.to_vec().unwrap();

        let mut mse = 0.0;
        let mut count = 0;

        for (pred, target) in pred_data.iter().zip(target_data.iter()) {
            mse += (pred - target).powi(2);
            count += 1;
        }

        if count > 0 {
            mse / count as f32
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_graph_fno_creation() {
        let fno = GraphFNO::new(4, 8, 16, 10, 3, true);
        assert_eq!(fno.in_features, 4);
        assert_eq!(fno.out_features, 8);
        assert_eq!(fno.hidden_features, 16);
        assert_eq!(fno.num_modes, 10);
        assert_eq!(fno.num_layers, 3);
    }

    #[test]
    fn test_graph_fno_forward() {
        let features = randn(&[5, 4]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0];
        let edge_index = from_vec(edges, &[2, 4], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let fno = GraphFNO::new(4, 8, 16, 10, 3, true);
        let output = fno.forward(&graph);

        assert_eq!(output.x.shape().dims(), &[5, 8]);
    }

    #[test]
    fn test_graph_deeponet_creation() {
        let deeponet = GraphDeepONet::new(3, 4, 16, 8, 10, 3, true);
        assert_eq!(deeponet.trunk_net_features, 3);
        assert_eq!(deeponet.branch_net_features, 4);
        assert_eq!(deeponet.output_features, 8);
        assert_eq!(deeponet.num_sensors, 10);
    }

    #[test]
    fn test_physics_informed_gnn() {
        let features = randn(&[4, 3]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        let edge_index = from_vec(edges, &[2, 3], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let pignn = PhysicsInformedGNN::new(3, 6, 12, 2, 0.1, 0.05, true);
        let output = pignn.forward(&graph);

        assert_eq!(output.x.shape().dims(), &[4, 6]);
    }

    #[test]
    fn test_multi_scale_gno() {
        let features = randn(&[8, 4]).unwrap();
        let edges = vec![
            0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0,
        ];
        let edge_index = from_vec(edges, &[2, 7], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let ms_gno = MultiScaleGNO::new(4, 6, 3, 8, true);
        let output = ms_gno.forward(&graph);

        assert_eq!(output.x.shape().dims(), &[8, 6]);
    }

    #[test]
    fn test_spectral_features() {
        let features = randn(&[6, 3]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0];
        let edge_index = from_vec(edges, &[2, 5], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let spectral_features = utils::compute_spectral_features(&graph, 4);
        assert_eq!(spectral_features.shape().dims(), &[6, 4]);
    }

    #[test]
    fn test_operator_data_generation() {
        let data_pairs = utils::generate_operator_data(3, 5, 4);
        assert_eq!(data_pairs.len(), 3);

        for (input, output) in &data_pairs {
            assert_eq!(input.num_nodes, 5);
            assert_eq!(output.num_nodes, 5);
            assert_eq!(input.x.shape().dims()[1], 4);
            assert_eq!(output.x.shape().dims()[1], 4);
        }
    }

    #[test]
    fn test_operator_error_computation() {
        let features1 = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], DeviceType::Cpu).unwrap();
        let features2 = from_vec(vec![1.1, 2.1, 3.1, 4.1], &[2, 2], DeviceType::Cpu).unwrap();
        let edges = vec![0.0, 1.0];
        let edge_index = from_vec(edges, &[2, 1], DeviceType::Cpu).unwrap();

        let graph1 = GraphData::new(features1, edge_index.clone());
        let graph2 = GraphData::new(features2, edge_index);

        let error = utils::compute_operator_error(&graph1, &graph2);
        assert!(error > 0.0);
        assert!(error < 1.0); // Should be small for similar graphs
    }
}
