//! Geometric Graph Neural Networks
//!
//! This module provides geometric deep learning capabilities for graph-structured
//! data with spatial coordinates. It includes geometric graph construction methods,
//! spatial convolutions, and geometric transformations inspired by scirs2-spatial.
//!
//! # Features:
//! - Geometric graph construction (k-NN, radius, Delaunay)
//! - Point cloud to graph conversion
//! - Spatial graph convolutions with distance-based weighting
//! - Geometric transformations (rotation, translation, scaling)
//! - 3D mesh processing
//! - Geometric pooling operations

use crate::parameter::Parameter;
use crate::{GraphData, GraphLayer};
use scirs2_core::random::thread_rng;
use std::cmp::Ordering;
use std::collections::HashMap;
use torsh_tensor::{
    creation::{from_vec, randn, zeros},
    Tensor,
};

/// Point in 3D space
#[derive(Debug, Clone, Copy)]
pub struct Point3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Point3D {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn distance(&self, other: &Point3D) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z - other.z).powi(2))
            .sqrt()
    }

    pub fn dot(&self, other: &Point3D) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn norm(&self) -> f32 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }
}

/// Geometric graph construction methods
pub struct GeometricGraphBuilder;

impl GeometricGraphBuilder {
    /// Build k-nearest neighbors graph from point cloud
    pub fn knn_graph(points: &[Point3D], k: usize, features: Option<Tensor>) -> GraphData {
        let num_points = points.len();
        let mut edges = Vec::new();
        let mut edge_weights = Vec::new();

        for i in 0..num_points {
            // Find k nearest neighbors
            let mut distances: Vec<(usize, f32)> = (0..num_points)
                .filter(|&j| j != i)
                .map(|j| (j, points[i].distance(&points[j])))
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            for (j, dist) in distances.iter().take(k) {
                edges.push(i as f32);
                edges.push(*j as f32);
                edge_weights.push(*dist);
            }
        }

        let num_edges = edges.len() / 2;
        let edge_index =
            from_vec(edges, &[2, num_edges], torsh_core::device::DeviceType::Cpu).unwrap();

        // Use provided features or create default features
        let x = features.unwrap_or_else(|| {
            let coords: Vec<f32> = points.iter().flat_map(|p| vec![p.x, p.y, p.z]).collect();
            from_vec(
                coords,
                &[num_points, 3],
                torsh_core::device::DeviceType::Cpu,
            )
            .unwrap()
        });

        let mut graph = GraphData::new(x, edge_index);

        // Store edge weights as edge attributes
        let edge_attr = from_vec(
            edge_weights,
            &[num_edges, 1],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();
        graph.edge_attr = Some(edge_attr);

        graph
    }

    /// Build radius graph (connect all points within radius)
    pub fn radius_graph(points: &[Point3D], radius: f32, features: Option<Tensor>) -> GraphData {
        let num_points = points.len();
        let mut edges = Vec::new();
        let mut edge_weights = Vec::new();

        for i in 0..num_points {
            for j in (i + 1)..num_points {
                let dist = points[i].distance(&points[j]);

                if dist <= radius {
                    edges.push(i as f32);
                    edges.push(j as f32);
                    edges.push(j as f32);
                    edges.push(i as f32);
                    edge_weights.push(dist);
                    edge_weights.push(dist);
                }
            }
        }

        let num_edges = edges.len() / 2;
        let edge_index = if num_edges > 0 {
            from_vec(edges, &[2, num_edges], torsh_core::device::DeviceType::Cpu).unwrap()
        } else {
            from_vec(vec![], &[2, 0], torsh_core::device::DeviceType::Cpu).unwrap()
        };

        let x = features.unwrap_or_else(|| {
            let coords: Vec<f32> = points.iter().flat_map(|p| vec![p.x, p.y, p.z]).collect();
            from_vec(
                coords,
                &[num_points, 3],
                torsh_core::device::DeviceType::Cpu,
            )
            .unwrap()
        });

        let mut graph = GraphData::new(x, edge_index);

        if num_edges > 0 {
            let edge_attr = from_vec(
                edge_weights,
                &[num_edges, 1],
                torsh_core::device::DeviceType::Cpu,
            )
            .unwrap();
            graph.edge_attr = Some(edge_attr);
        }

        graph
    }

    /// Build Delaunay triangulation graph (2D simplified version)
    pub fn delaunay_graph_2d(points: &[(f32, f32)], features: Option<Tensor>) -> GraphData {
        let num_points = points.len();

        // Simplified Delaunay: connect points that are close
        // Full Delaunay would require more complex algorithms
        let mut edges = Vec::new();
        let mut visited_pairs: std::collections::HashSet<(usize, usize)> =
            std::collections::HashSet::new();

        for i in 0..num_points {
            // Find k nearest neighbors for simplified triangulation
            let k = 5;
            let mut distances: Vec<(usize, f32)> = (0..num_points)
                .filter(|&j| j != i)
                .map(|j| {
                    let dx = points[i].0 - points[j].0;
                    let dy = points[i].1 - points[j].1;
                    (j, (dx * dx + dy * dy).sqrt())
                })
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            for (j, _) in distances.iter().take(k) {
                let pair = if i < *j { (i, *j) } else { (*j, i) };

                if !visited_pairs.contains(&pair) {
                    visited_pairs.insert(pair);
                    edges.push(i as f32);
                    edges.push(*j as f32);
                    edges.push(*j as f32);
                    edges.push(i as f32);
                }
            }
        }

        let num_edges = edges.len() / 2;
        let edge_index =
            from_vec(edges, &[2, num_edges], torsh_core::device::DeviceType::Cpu).unwrap();

        let x = features.unwrap_or_else(|| {
            let coords: Vec<f32> = points.iter().flat_map(|(x, y)| vec![*x, *y]).collect();
            from_vec(
                coords,
                &[num_points, 2],
                torsh_core::device::DeviceType::Cpu,
            )
            .unwrap()
        });

        GraphData::new(x, edge_index)
    }
}

/// Geometric convolution layer with distance-based attention
#[derive(Debug)]
pub struct GeometricConv {
    in_features: usize,
    out_features: usize,
    hidden_dim: usize,

    // MLP for message generation
    message_mlp: Vec<Parameter>,

    // Distance encoding
    distance_encoder: Parameter,

    // Output projection
    output_weight: Parameter,

    bias: Option<Parameter>,
}

impl GeometricConv {
    /// Create a new geometric convolution layer
    pub fn new(in_features: usize, out_features: usize, hidden_dim: usize, use_bias: bool) -> Self {
        // MLP layers for message generation
        let message_layer1 = Parameter::new(randn(&[in_features * 2 + 1, hidden_dim]).unwrap());
        let message_layer2 = Parameter::new(randn(&[hidden_dim, hidden_dim]).unwrap());

        let distance_encoder = Parameter::new(randn(&[1, hidden_dim]).unwrap());
        let output_weight = Parameter::new(randn(&[hidden_dim, out_features]).unwrap());

        let bias = if use_bias {
            Some(Parameter::new(zeros(&[out_features]).unwrap()))
        } else {
            None
        };

        Self {
            in_features,
            out_features,
            hidden_dim,
            message_mlp: vec![message_layer1, message_layer2],
            distance_encoder,
            output_weight,
            bias,
        }
    }

    /// Forward pass through geometric convolution
    pub fn forward(&self, graph: &GraphData) -> GraphData {
        let num_nodes = graph.num_nodes;
        let num_edges = graph.num_edges;

        // Get edge distances if available
        let edge_distances = if let Some(ref edge_attr) = graph.edge_attr {
            edge_attr.to_vec().expect("conversion should succeed")
        } else {
            vec![1.0; num_edges]
        };

        // Aggregate messages
        let edge_data = graph
            .edge_index
            .to_vec()
            .expect("conversion should succeed");
        let mut aggregated = vec![0.0; num_nodes * self.hidden_dim];

        let node_features = graph.x.to_vec().expect("conversion should succeed");

        for edge_idx in 0..num_edges {
            let src = edge_data[edge_idx * 2] as usize;
            let dst = edge_data[edge_idx * 2 + 1] as usize;

            if src >= num_nodes || dst >= num_nodes {
                continue;
            }

            // Get source and destination features
            let src_features = &node_features[src * self.in_features..(src + 1) * self.in_features];
            let dst_features = &node_features[dst * self.in_features..(dst + 1) * self.in_features];

            // Distance encoding
            let dist = edge_distances[edge_idx.min(edge_distances.len() - 1)];

            // Concatenate features and distance
            let mut message_input = Vec::new();
            message_input.extend_from_slice(src_features);
            message_input.extend_from_slice(dst_features);
            message_input.push(dist);

            // Compute message through MLP (simplified)
            let message = self.compute_message(&message_input);

            // Aggregate to destination node
            for (i, &val) in message.iter().enumerate() {
                aggregated[dst * self.hidden_dim + i] += val;
            }
        }

        // Apply output projection
        let mut output_features = vec![0.0; num_nodes * self.out_features];

        for node in 0..num_nodes {
            let agg_features = &aggregated[node * self.hidden_dim..(node + 1) * self.hidden_dim];
            let output_proj = self
                .output_weight
                .clone_data()
                .to_vec()
                .expect("conversion should succeed");

            for out_idx in 0..self.out_features {
                let mut sum = 0.0;
                for hid_idx in 0..self.hidden_dim {
                    sum +=
                        agg_features[hid_idx] * output_proj[hid_idx * self.out_features + out_idx];
                }

                if let Some(ref bias) = self.bias {
                    let bias_data = bias
                        .clone_data()
                        .to_vec()
                        .expect("conversion should succeed");
                    if out_idx < bias_data.len() {
                        sum += bias_data[out_idx];
                    }
                }

                output_features[node * self.out_features + out_idx] = sum;
            }
        }

        let output = from_vec(
            output_features,
            &[num_nodes, self.out_features],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        let mut output_graph = graph.clone();
        output_graph.x = output;
        output_graph
    }

    /// Compute message from concatenated features and distance
    fn compute_message(&self, input: &[f32]) -> Vec<f32> {
        // Layer 1
        let layer1_weights = self.message_mlp[0]
            .clone_data()
            .to_vec()
            .expect("conversion should succeed");
        let input_dim = self.in_features * 2 + 1;
        let mut hidden = vec![0.0; self.hidden_dim];

        for h in 0..self.hidden_dim {
            let mut sum = 0.0;
            for i in 0..input_dim.min(input.len()) {
                sum += input[i] * layer1_weights[i * self.hidden_dim + h];
            }
            hidden[h] = sum.max(0.0); // ReLU
        }

        // Layer 2
        let layer2_weights = self.message_mlp[1]
            .clone_data()
            .to_vec()
            .expect("conversion should succeed");
        let mut output = vec![0.0; self.hidden_dim];

        for h in 0..self.hidden_dim {
            let mut sum = 0.0;
            for i in 0..self.hidden_dim {
                sum += hidden[i] * layer2_weights[i * self.hidden_dim + h];
            }
            output[h] = sum.max(0.0); // ReLU
        }

        output
    }
}

impl GraphLayer for GeometricConv {
    fn forward(&self, graph: &GraphData) -> GraphData {
        self.forward(graph)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();

        for layer in &self.message_mlp {
            params.push(layer.clone_data());
        }

        params.push(self.distance_encoder.clone_data());
        params.push(self.output_weight.clone_data());

        if let Some(ref bias) = self.bias {
            params.push(bias.clone_data());
        }

        params
    }
}

/// Geometric transformations for point clouds and graphs
pub struct GeometricTransformer;

impl GeometricTransformer {
    /// Apply rotation to point cloud
    pub fn rotate_3d(points: &mut [Point3D], axis: &Point3D, angle: f32) {
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();

        // Normalize axis
        let norm = axis.norm();
        if norm == 0.0 {
            return;
        }

        let ux = axis.x / norm;
        let uy = axis.y / norm;
        let uz = axis.z / norm;

        // Rotation matrix (Rodrigues' rotation formula)
        for point in points.iter_mut() {
            let x = point.x;
            let y = point.y;
            let z = point.z;

            // Dot product with axis
            let dot = ux * x + uy * y + uz * z;

            // Cross product with axis
            let cross_x = uy * z - uz * y;
            let cross_y = uz * x - ux * z;
            let cross_z = ux * y - uy * x;

            // Apply rotation
            point.x = x * cos_theta + cross_x * sin_theta + ux * dot * (1.0 - cos_theta);
            point.y = y * cos_theta + cross_y * sin_theta + uy * dot * (1.0 - cos_theta);
            point.z = z * cos_theta + cross_z * sin_theta + uz * dot * (1.0 - cos_theta);
        }
    }

    /// Apply translation to point cloud
    pub fn translate_3d(points: &mut [Point3D], offset: &Point3D) {
        for point in points.iter_mut() {
            point.x += offset.x;
            point.y += offset.y;
            point.z += offset.z;
        }
    }

    /// Apply scaling to point cloud
    pub fn scale_3d(points: &mut [Point3D], scale: f32) {
        for point in points.iter_mut() {
            point.x *= scale;
            point.y *= scale;
            point.z *= scale;
        }
    }

    /// Normalize point cloud to unit sphere
    pub fn normalize_to_unit_sphere(points: &mut [Point3D]) {
        if points.is_empty() {
            return;
        }

        // Find center
        let mut center = Point3D::new(0.0, 0.0, 0.0);
        for point in points.iter() {
            center.x += point.x;
            center.y += point.y;
            center.z += point.z;
        }
        center.x /= points.len() as f32;
        center.y /= points.len() as f32;
        center.z /= points.len() as f32;

        // Translate to origin
        Self::translate_3d(points, &Point3D::new(-center.x, -center.y, -center.z));

        // Find max distance
        let max_dist = points
            .iter()
            .map(|p| p.norm())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(1.0);

        // Scale to unit sphere
        if max_dist > 0.0 {
            Self::scale_3d(points, 1.0 / max_dist);
        }
    }
}

/// Geometric pooling operations
pub struct GeometricPooling;

impl GeometricPooling {
    /// Voxel-based pooling (divide space into voxels and pool within each)
    pub fn voxel_pool(
        points: &[Point3D],
        features: &Tensor,
        voxel_size: f32,
    ) -> (Vec<Point3D>, Tensor) {
        let feature_data = features.to_vec().expect("conversion should succeed");
        let feature_dim = features.shape().dims()[1];

        // Compute voxel indices
        let mut voxel_map: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();

        for (i, point) in points.iter().enumerate() {
            let vx = (point.x / voxel_size).floor() as i32;
            let vy = (point.y / voxel_size).floor() as i32;
            let vz = (point.z / voxel_size).floor() as i32;

            voxel_map
                .entry((vx, vy, vz))
                .or_insert_with(Vec::new)
                .push(i);
        }

        // Pool points and features within each voxel
        let mut pooled_points = Vec::new();
        let mut pooled_features = Vec::new();

        for (_voxel, indices) in voxel_map {
            if indices.is_empty() {
                continue;
            }

            // Average position
            let mut avg_point = Point3D::new(0.0, 0.0, 0.0);
            for &idx in &indices {
                avg_point.x += points[idx].x;
                avg_point.y += points[idx].y;
                avg_point.z += points[idx].z;
            }
            avg_point.x /= indices.len() as f32;
            avg_point.y /= indices.len() as f32;
            avg_point.z /= indices.len() as f32;

            pooled_points.push(avg_point);

            // Average features
            let mut avg_features = vec![0.0; feature_dim];
            for &idx in &indices {
                for d in 0..feature_dim {
                    avg_features[d] += feature_data[idx * feature_dim + d];
                }
            }
            for val in &mut avg_features {
                *val /= indices.len() as f32;
            }

            pooled_features.extend(avg_features);
        }

        let pooled_tensor = from_vec(
            pooled_features,
            &[pooled_points.len(), feature_dim],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        (pooled_points, pooled_tensor)
    }

    /// Farthest point sampling
    pub fn farthest_point_sampling(
        points: &[Point3D],
        features: &Tensor,
        num_samples: usize,
    ) -> (Vec<Point3D>, Tensor) {
        let num_points = points.len();
        let feature_dim = features.shape().dims()[1];
        let feature_data = features.to_vec().expect("conversion should succeed");

        if num_samples >= num_points {
            return (points.to_vec(), features.clone());
        }

        let mut selected = Vec::new();
        let mut distances = vec![f32::MAX; num_points];

        // Start with random point
        let mut rng = thread_rng();
        let first_idx = rng.gen_range(0..num_points);
        selected.push(first_idx);

        // Update distances
        for i in 0..num_points {
            distances[i] = points[i].distance(&points[first_idx]);
        }

        // Iteratively select farthest point
        for _ in 1..num_samples {
            let farthest_idx = distances
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap();

            selected.push(farthest_idx);

            // Update distances
            for i in 0..num_points {
                let dist = points[i].distance(&points[farthest_idx]);
                distances[i] = distances[i].min(dist);
            }
        }

        // Extract selected points and features
        let sampled_points: Vec<_> = selected.iter().map(|&idx| points[idx]).collect();
        let sampled_features: Vec<_> = selected
            .iter()
            .flat_map(|&idx| {
                let start = idx * feature_dim;
                let end = start + feature_dim;
                &feature_data[start..end]
            })
            .copied()
            .collect();

        let sampled_tensor = from_vec(
            sampled_features,
            &[num_samples, feature_dim],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap();

        (sampled_points, sampled_tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point3d_distance() {
        let p1 = Point3D::new(0.0, 0.0, 0.0);
        let p2 = Point3D::new(3.0, 4.0, 0.0);

        assert!((p1.distance(&p2) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_knn_graph() {
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
            Point3D::new(1.0, 1.0, 0.0),
        ];

        let graph = GeometricGraphBuilder::knn_graph(&points, 2, None);

        assert_eq!(graph.num_nodes, 4);
        assert_eq!(graph.x.shape().dims()[1], 3); // 3D coordinates
        assert!(graph.edge_attr.is_some());
    }

    #[test]
    fn test_radius_graph() {
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(0.5, 0.0, 0.0),
            Point3D::new(2.0, 0.0, 0.0),
        ];

        let graph = GeometricGraphBuilder::radius_graph(&points, 1.0, None);

        assert_eq!(graph.num_nodes, 3);
        assert!(graph.num_edges >= 2); // At least points 0 and 1 connected
    }

    #[test]
    fn test_geometric_conv() {
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
        ];

        let graph = GeometricGraphBuilder::knn_graph(&points, 2, None);
        let conv = GeometricConv::new(3, 6, 8, true);

        let output = conv.forward(&graph);

        assert_eq!(output.num_nodes, 3);
        assert_eq!(output.x.shape().dims()[1], 6);
    }

    #[test]
    fn test_geometric_rotation() {
        let mut points = vec![Point3D::new(1.0, 0.0, 0.0)];

        let axis = Point3D::new(0.0, 0.0, 1.0);
        let angle = std::f32::consts::PI / 2.0;

        GeometricTransformer::rotate_3d(&mut points, &axis, angle);

        // After 90 degree rotation around Z-axis, (1,0,0) -> (0,1,0)
        assert!((points[0].x - 0.0).abs() < 1e-5);
        assert!((points[0].y - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_to_unit_sphere() {
        let mut points = vec![
            Point3D::new(2.0, 0.0, 0.0),
            Point3D::new(0.0, 2.0, 0.0),
            Point3D::new(0.0, 0.0, 2.0),
        ];

        GeometricTransformer::normalize_to_unit_sphere(&mut points);

        // All points should be within unit sphere
        for point in &points {
            assert!(point.norm() <= 1.0 + 1e-5);
        }
    }

    #[test]
    fn test_voxel_pooling() {
        let points = vec![
            Point3D::new(0.1, 0.1, 0.1),
            Point3D::new(0.2, 0.2, 0.2),
            Point3D::new(1.1, 1.1, 1.1),
        ];

        let features = randn(&[3, 4]).unwrap();

        let (pooled_points, pooled_features) =
            GeometricPooling::voxel_pool(&points, &features, 1.0);

        assert!(pooled_points.len() <= 3);
        assert_eq!(pooled_features.shape().dims()[1], 4);
    }

    #[test]
    fn test_farthest_point_sampling() {
        let points = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
            Point3D::new(0.0, 0.0, 1.0),
        ];

        let features = randn(&[4, 3]).unwrap();

        let (sampled_points, sampled_features) =
            GeometricPooling::farthest_point_sampling(&points, &features, 2);

        assert_eq!(sampled_points.len(), 2);
        assert_eq!(sampled_features.shape().dims(), &[2, 3]);
    }
}
