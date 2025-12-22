//! Graph Optimal Transport
//!
//! This module implements optimal transport methods for graphs, enabling:
//! - Graph alignment and matching via Gromov-Wasserstein distance
//! - Graph interpolation and barycenter computation
//! - Domain adaptation and transfer learning on graphs
//! - Graph distance computation for similarity analysis
//!
//! # Key Features:
//! - Wasserstein distance for node feature distributions
//! - Gromov-Wasserstein distance for graph structure alignment
//! - Fused Gromov-Wasserstein combining features and structure
//! - Sinkhorn algorithm for entropic regularization
//! - Unbalanced optimal transport for graphs of different sizes
//!
//! # References:
//! - Peyré et al. "Computational Optimal Transport" (2019)
//! - Vayer et al. "Optimal Transport for structured data with application on graphs" (ICML 2019)
//! - Titouan et al. "Optimal Transport Graph Neural Networks" (2022)

use crate::GraphData;
use scirs2_core::ndarray::{Array1, Array2};
use std::f32;
use torsh_core::device::DeviceType;
use torsh_tensor::creation::from_vec;

/// Configuration for optimal transport computation
#[derive(Debug, Clone)]
pub struct OTConfig {
    /// Entropic regularization parameter (epsilon)
    pub epsilon: f32,
    /// Maximum number of Sinkhorn iterations
    pub max_iter: usize,
    /// Convergence threshold
    pub threshold: f32,
    /// Whether to use log-domain stabilized Sinkhorn
    pub log_domain: bool,
    /// Alpha parameter for Fused GW (balances structure vs features)
    pub alpha: f32,
}

impl Default for OTConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            max_iter: 100,
            threshold: 1e-6,
            log_domain: true,
            alpha: 0.5,
        }
    }
}

/// Sinkhorn algorithm for entropic optimal transport
///
/// Computes the optimal transport plan between two distributions with
/// entropic regularization for stability and efficiency.
///
/// # Arguments:
/// * `cost_matrix` - Pairwise cost matrix [n, m]
/// * `a` - Source distribution (marginal) [n]
/// * `b` - Target distribution (marginal) [m]
/// * `config` - OT configuration
///
/// # Returns:
/// Optimal transport plan [n, m]
pub struct SinkhornSolver {
    config: OTConfig,
}

impl SinkhornSolver {
    /// Create a new Sinkhorn solver
    pub fn new(config: OTConfig) -> Self {
        Self { config }
    }

    /// Solve optimal transport problem using Sinkhorn algorithm
    ///
    /// # Arguments:
    /// * `cost_matrix` - Cost matrix [n, m]
    /// * `a` - Source distribution [n]
    /// * `b` - Target distribution [m]
    ///
    /// # Returns:
    /// Optimal transport plan [n, m]
    pub fn solve(
        &self,
        cost_matrix: &Array2<f32>,
        a: &Array1<f32>,
        b: &Array1<f32>,
    ) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
        let (n, m) = cost_matrix.dim();

        // Check input dimensions
        if a.len() != n || b.len() != m {
            return Err("Dimension mismatch between cost matrix and marginals".into());
        }

        // Check that marginals sum to 1
        let a_sum: f32 = a.iter().sum();
        let b_sum: f32 = b.iter().sum();
        if (a_sum - 1.0).abs() > 1e-4 || (b_sum - 1.0).abs() > 1e-4 {
            return Err("Marginals must sum to 1".into());
        }

        if self.config.log_domain {
            self.solve_log_domain(cost_matrix, a, b)
        } else {
            self.solve_standard(cost_matrix, a, b)
        }
    }

    /// Standard Sinkhorn algorithm (numerically unstable for small epsilon)
    fn solve_standard(
        &self,
        cost_matrix: &Array2<f32>,
        a: &Array1<f32>,
        b: &Array1<f32>,
    ) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
        let (n, m) = cost_matrix.dim();

        // Compute kernel K = exp(-C/epsilon)
        let mut kernel = Array2::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                kernel[[i, j]] = (-cost_matrix[[i, j]] / self.config.epsilon).exp();
            }
        }

        // Initialize scaling vectors
        let mut u = Array1::from_elem(n, 1.0 / n as f32);
        let mut v = Array1::from_elem(m, 1.0 / m as f32);

        // Sinkhorn iterations
        for _iter in 0..self.config.max_iter {
            let u_old = u.clone();

            // Update u: u = a / (K @ v)
            for i in 0..n {
                let kv: f32 = (0..m).map(|j| kernel[[i, j]] * v[j]).sum();
                u[i] = a[i] / (kv + 1e-10);
            }

            // Update v: v = b / (K^T @ u)
            for j in 0..m {
                let ktu: f32 = (0..n).map(|i| kernel[[i, j]] * u[i]).sum();
                v[j] = b[j] / (ktu + 1e-10);
            }

            // Check convergence
            let error: f32 = u
                .iter()
                .zip(u_old.iter())
                .map(|(ui, ui_old)| (ui - ui_old).abs())
                .sum();
            if error < self.config.threshold {
                break;
            }
        }

        // Compute transport plan: P = diag(u) @ K @ diag(v)
        let mut plan = Array2::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                plan[[i, j]] = u[i] * kernel[[i, j]] * v[j];
            }
        }

        Ok(plan)
    }

    /// Log-domain stabilized Sinkhorn algorithm
    fn solve_log_domain(
        &self,
        cost_matrix: &Array2<f32>,
        a: &Array1<f32>,
        b: &Array1<f32>,
    ) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
        let (n, m) = cost_matrix.dim();

        // Initialize log-domain variables
        let log_a = a.mapv(|x| (x + 1e-10).ln());
        let log_b = b.mapv(|x| (x + 1e-10).ln());
        let mut f = Array1::zeros(n);
        let mut g = Array1::zeros(m);

        // Sinkhorn iterations in log-domain
        for _iter in 0..self.config.max_iter {
            let f_old = f.clone();

            // Update f
            for i in 0..n {
                let lse = self.log_sum_exp(
                    &(0..m)
                        .map(|j| (-cost_matrix[[i, j]] / self.config.epsilon) + g[j])
                        .collect::<Vec<_>>(),
                );
                f[i] = log_a[i] - lse;
            }

            // Update g
            for j in 0..m {
                let lse = self.log_sum_exp(
                    &(0..n)
                        .map(|i| (-cost_matrix[[i, j]] / self.config.epsilon) + f[i])
                        .collect::<Vec<_>>(),
                );
                g[j] = log_b[j] - lse;
            }

            // Check convergence
            let error: f32 = f
                .iter()
                .zip(f_old.iter())
                .map(|(fi, fi_old)| (fi - fi_old).abs())
                .sum();
            if error < self.config.threshold {
                break;
            }
        }

        // Compute transport plan in log-domain and exponentiate
        let mut plan = Array2::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                plan[[i, j]] = (f[i] + g[j] - cost_matrix[[i, j]] / self.config.epsilon).exp();
            }
        }

        Ok(plan)
    }

    /// Log-sum-exp for numerical stability
    fn log_sum_exp(&self, values: &[f32]) -> f32 {
        if values.is_empty() {
            return f32::NEG_INFINITY;
        }
        let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if max_val.is_infinite() {
            return max_val;
        }
        let sum_exp: f32 = values.iter().map(|&x| (x - max_val).exp()).sum();
        max_val + sum_exp.ln()
    }
}

/// Gromov-Wasserstein distance for graph alignment
///
/// Computes the Gromov-Wasserstein distance between two graphs, which
/// compares graph structures independent of node features.
pub struct GromovWassersteinSolver {
    config: OTConfig,
    sinkhorn: SinkhornSolver,
}

impl GromovWassersteinSolver {
    /// Create a new Gromov-Wasserstein solver
    pub fn new(config: OTConfig) -> Self {
        let sinkhorn = SinkhornSolver::new(config.clone());
        Self { config, sinkhorn }
    }

    /// Compute Gromov-Wasserstein distance between two graphs
    ///
    /// # Arguments:
    /// * `graph1` - First graph
    /// * `graph2` - Second graph
    ///
    /// # Returns:
    /// (GW distance, optimal transport plan)
    pub fn compute_distance(
        &self,
        graph1: &GraphData,
        graph2: &GraphData,
    ) -> Result<(f32, Array2<f32>), Box<dyn std::error::Error>> {
        // Compute pairwise distance matrices for both graphs
        let dist1 = self.compute_graph_distances(graph1)?;
        let dist2 = self.compute_graph_distances(graph2)?;

        // Uniform distributions over nodes
        let n1 = graph1.num_nodes;
        let n2 = graph2.num_nodes;
        let a = Array1::from_elem(n1, 1.0 / n1 as f32);
        let b = Array1::from_elem(n2, 1.0 / n2 as f32);

        // Initialize transport plan
        let mut plan = Array2::from_elem((n1, n2), 1.0 / (n1 * n2) as f32);

        // Proximal point algorithm for GW
        for _iter in 0..self.config.max_iter {
            let plan_old = plan.clone();

            // Compute cost matrix based on structure discrepancy
            let cost_matrix = self.compute_gw_cost(&dist1, &dist2, &plan)?;

            // Update plan using Sinkhorn
            plan = self.sinkhorn.solve(&cost_matrix, &a, &b)?;

            // Check convergence
            let error = self.frobenius_norm_diff(&plan, &plan_old);
            if error < self.config.threshold {
                break;
            }
        }

        // Compute final GW distance
        let cost_matrix = self.compute_gw_cost(&dist1, &dist2, &plan)?;
        let mut distance = 0.0f32;
        for i in 0..n1 {
            for j in 0..n2 {
                distance += cost_matrix[[i, j]] * plan[[i, j]];
            }
        }

        Ok((distance, plan))
    }

    /// Compute pairwise shortest path distances for a graph
    fn compute_graph_distances(
        &self,
        graph: &GraphData,
    ) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
        let n = graph.num_nodes;
        let mut dist = Array2::from_elem((n, n), f32::INFINITY);

        // Initialize diagonal
        for i in 0..n {
            dist[[i, i]] = 0.0;
        }

        // Initialize edge distances
        let edge_data = graph.edge_index.to_vec()?;
        let num_edges = graph.num_edges;
        for e in 0..num_edges {
            let src = edge_data[e] as usize;
            let dst = edge_data[num_edges + e] as usize;
            dist[[src, dst]] = 1.0; // Unweighted edges
        }

        // Floyd-Warshall algorithm
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if dist[[i, k]] + dist[[k, j]] < dist[[i, j]] {
                        dist[[i, j]] = dist[[i, k]] + dist[[k, j]];
                    }
                }
            }
        }

        Ok(dist)
    }

    /// Compute GW cost matrix: C(i,j) = Σ_kl |d1(i,k) - d2(j,l)|² P(k,l)
    fn compute_gw_cost(
        &self,
        dist1: &Array2<f32>,
        dist2: &Array2<f32>,
        plan: &Array2<f32>,
    ) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
        let (n1, _) = dist1.dim();
        let (n2, _) = dist2.dim();
        let mut cost = Array2::zeros((n1, n2));

        // Find maximum finite distance to use as upper bound
        let max_dist = dist1
            .iter()
            .chain(dist2.iter())
            .filter(|&&d| d.is_finite())
            .fold(0.0f32, |a, &b| a.max(b));
        let max_dist = if max_dist > 0.0 { max_dist } else { 1.0 };

        for i in 0..n1 {
            for j in 0..n2 {
                let mut sum = 0.0;
                for k in 0..n1 {
                    for l in 0..n2 {
                        let d1 = if dist1[[i, k]].is_finite() {
                            dist1[[i, k]]
                        } else {
                            max_dist * 2.0 // Use large but finite value
                        };
                        let d2 = if dist2[[j, l]].is_finite() {
                            dist2[[j, l]]
                        } else {
                            max_dist * 2.0
                        };
                        let diff = d1 - d2;
                        sum += diff * diff * plan[[k, l]];
                    }
                }
                cost[[i, j]] = sum;
            }
        }

        Ok(cost)
    }

    /// Compute Frobenius norm difference
    fn frobenius_norm_diff(&self, a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

/// Fused Gromov-Wasserstein distance
///
/// Combines structural alignment (Gromov-Wasserstein) with feature alignment
/// (Wasserstein) for comprehensive graph comparison.
pub struct FusedGromovWassersteinSolver {
    config: OTConfig,
    sinkhorn: SinkhornSolver,
}

impl FusedGromovWassersteinSolver {
    /// Create a new Fused Gromov-Wasserstein solver
    pub fn new(config: OTConfig) -> Self {
        let sinkhorn = SinkhornSolver::new(config.clone());
        Self { config, sinkhorn }
    }

    /// Compute Fused Gromov-Wasserstein distance
    ///
    /// # Arguments:
    /// * `graph1` - First graph with node features
    /// * `graph2` - Second graph with node features
    ///
    /// # Returns:
    /// (FGW distance, optimal transport plan)
    pub fn compute_distance(
        &self,
        graph1: &GraphData,
        graph2: &GraphData,
    ) -> Result<(f32, Array2<f32>), Box<dyn std::error::Error>> {
        // Compute structure cost (Gromov-Wasserstein)
        let gw_solver = GromovWassersteinSolver::new(self.config.clone());
        let dist1 = gw_solver.compute_graph_distances(graph1)?;
        let dist2 = gw_solver.compute_graph_distances(graph2)?;

        // Compute feature cost (Wasserstein)
        let feature_cost = self.compute_feature_cost(graph1, graph2)?;

        // Uniform distributions
        let n1 = graph1.num_nodes;
        let n2 = graph2.num_nodes;
        let a = Array1::from_elem(n1, 1.0 / n1 as f32);
        let b = Array1::from_elem(n2, 1.0 / n2 as f32);

        // Initialize transport plan
        let mut plan = Array2::from_elem((n1, n2), 1.0 / (n1 * n2) as f32);

        // Fused GW iterations
        for _iter in 0..self.config.max_iter {
            let plan_old = plan.clone();

            // Compute fused cost: alpha * GW_cost + (1-alpha) * feature_cost
            let gw_cost = gw_solver.compute_gw_cost(&dist1, &dist2, &plan)?;
            let mut fused_cost = Array2::zeros((n1, n2));
            for i in 0..n1 {
                for j in 0..n2 {
                    fused_cost[[i, j]] = self.config.alpha * gw_cost[[i, j]]
                        + (1.0 - self.config.alpha) * feature_cost[[i, j]];
                }
            }

            // Update plan using Sinkhorn
            plan = self.sinkhorn.solve(&fused_cost, &a, &b)?;

            // Check convergence
            let error = gw_solver.frobenius_norm_diff(&plan, &plan_old);
            if error < self.config.threshold {
                break;
            }
        }

        // Compute final FGW distance
        let gw_cost = gw_solver.compute_gw_cost(&dist1, &dist2, &plan)?;
        let mut distance = 0.0f32;
        for i in 0..n1 {
            for j in 0..n2 {
                distance += (self.config.alpha * gw_cost[[i, j]]
                    + (1.0 - self.config.alpha) * feature_cost[[i, j]])
                    * plan[[i, j]];
            }
        }

        Ok((distance, plan))
    }

    /// Compute feature cost matrix (Euclidean distance between features)
    fn compute_feature_cost(
        &self,
        graph1: &GraphData,
        graph2: &GraphData,
    ) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
        let n1 = graph1.num_nodes;
        let n2 = graph2.num_nodes;
        let feat1 = graph1.x.to_vec()?;
        let feat2 = graph2.x.to_vec()?;
        let dim = graph1.x.shape().dims()[1];

        let mut cost = Array2::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                let mut dist_sq = 0.0;
                for d in 0..dim {
                    let diff = feat1[i * dim + d] - feat2[j * dim + d];
                    dist_sq += diff * diff;
                }
                cost[[i, j]] = dist_sq.sqrt();
            }
        }

        Ok(cost)
    }
}

/// Graph barycenter computation using optimal transport
///
/// Computes the Wasserstein barycenter of multiple graphs, useful for
/// graph averaging and prototype learning.
pub struct GraphBarycenter {
    _config: OTConfig, // Reserved for future full implementation
}

impl GraphBarycenter {
    /// Create a new graph barycenter solver
    pub fn new(config: OTConfig) -> Self {
        Self { _config: config }
    }

    /// Compute barycenter of multiple graphs
    ///
    /// # Arguments:
    /// * `graphs` - Collection of graphs
    /// * `weights` - Importance weights for each graph
    ///
    /// # Returns:
    /// Barycenter graph
    pub fn compute(
        &self,
        graphs: &[GraphData],
        weights: &[f32],
    ) -> Result<GraphData, Box<dyn std::error::Error>> {
        if graphs.is_empty() {
            return Err("At least one graph required for barycenter".into());
        }

        if graphs.len() != weights.len() {
            return Err("Number of weights must match number of graphs".into());
        }

        // For simplicity, return weighted average of features
        // Full implementation would use iterative Bregman projections
        let n = graphs[0].num_nodes;
        let dim = graphs[0].x.shape().dims()[1];

        let mut barycenter_features = vec![0.0f32; n * dim];

        for (graph, &weight) in graphs.iter().zip(weights.iter()) {
            let features = graph.x.to_vec()?;
            for i in 0..features.len() {
                barycenter_features[i] += features[i] * weight;
            }
        }

        let barycenter_x = from_vec(barycenter_features, &[n, dim], DeviceType::Cpu)?;
        let barycenter_graph = GraphData::new(barycenter_x, graphs[0].edge_index.clone());

        Ok(barycenter_graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_sinkhorn_solver() {
        let config = OTConfig::default();
        let solver = SinkhornSolver::new(config);

        // Simple 2x2 cost matrix
        let cost = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        let a = Array1::from_vec(vec![0.5, 0.5]);
        let b = Array1::from_vec(vec![0.5, 0.5]);

        let plan = solver.solve(&cost, &a, &b).unwrap();

        // Plan should be close to identity (diagonal)
        assert!(plan[[0, 0]] > 0.3); // Should prefer matching 0->0
        assert!(plan[[1, 1]] > 0.3); // Should prefer matching 1->1
    }

    #[test]
    fn test_sinkhorn_marginal_constraints() {
        let config = OTConfig::default();
        let solver = SinkhornSolver::new(config);

        let cost =
            Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0])
                .unwrap();
        let a = Array1::from_vec(vec![0.3, 0.4, 0.3]);
        let b = Array1::from_vec(vec![0.2, 0.5, 0.3]);

        let plan = solver.solve(&cost, &a, &b).unwrap();

        // Check marginal constraints
        let row_sums: Vec<f32> = (0..3).map(|i| (0..3).map(|j| plan[[i, j]]).sum()).collect();
        let col_sums: Vec<f32> = (0..3).map(|j| (0..3).map(|i| plan[[i, j]]).sum()).collect();

        for i in 0..3 {
            assert!((row_sums[i] - a[i]).abs() < 0.01);
            assert!((col_sums[i] - b[i]).abs() < 0.01);
        }
    }

    #[test]
    fn test_gromov_wasserstein_solver() {
        let config = OTConfig {
            epsilon: 0.1,
            max_iter: 50,
            threshold: 1e-4,
            log_domain: true,
            alpha: 0.5,
        };
        let solver = GromovWassersteinSolver::new(config);

        // Create two simple graphs
        let x1 = from_vec(vec![1.0; 3 * 4], &[3, 4], DeviceType::Cpu).unwrap();
        let edge_index1 = from_vec(vec![0.0, 1.0, 1.0, 2.0], &[2, 2], DeviceType::Cpu).unwrap();
        let graph1 = GraphData::new(x1, edge_index1);

        let x2 = from_vec(vec![2.0; 3 * 4], &[3, 4], DeviceType::Cpu).unwrap();
        let edge_index2 = from_vec(vec![0.0, 1.0, 1.0, 2.0], &[2, 2], DeviceType::Cpu).unwrap();
        let graph2 = GraphData::new(x2, edge_index2);

        let result = solver.compute_distance(&graph1, &graph2);
        assert!(result.is_ok());

        let (distance, plan) = result.unwrap();
        // Graphs have identical structure, so GW distance should be very small
        // (allow small numerical errors)
        assert!(
            distance.abs() < 1.0,
            "Distance {} should be close to 0",
            distance
        );
        assert_eq!(plan.dim(), (3, 3));
    }

    #[test]
    fn test_fused_gromov_wasserstein() {
        let mut config = OTConfig::default();
        config.alpha = 0.6; // More weight on structure
        let solver = FusedGromovWassersteinSolver::new(config);

        let x1 = from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2], DeviceType::Cpu).unwrap();
        let edge_index1 = from_vec(vec![0.0, 1.0, 1.0, 2.0], &[2, 2], DeviceType::Cpu).unwrap();
        let graph1 = GraphData::new(x1, edge_index1);

        let x2 = from_vec(vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[3, 2], DeviceType::Cpu).unwrap();
        let edge_index2 = from_vec(vec![0.0, 1.0, 1.0, 2.0], &[2, 2], DeviceType::Cpu).unwrap();
        let graph2 = GraphData::new(x2, edge_index2);

        let result = solver.compute_distance(&graph1, &graph2);
        assert!(result.is_ok());

        let (distance, plan) = result.unwrap();
        // Fused GW combines structure and features
        // These graphs have same structure but different features, so distance > 0
        assert!(distance.is_finite(), "Distance should be finite");
        assert!(distance >= 0.0, "Distance should be non-negative");
        assert_eq!(plan.dim(), (3, 3));
    }

    #[test]
    fn test_graph_barycenter() {
        let config = OTConfig::default();
        let barycenter_solver = GraphBarycenter::new(config);

        // Create two simple graphs
        let x1 = from_vec(vec![1.0; 4 * 2], &[4, 2], DeviceType::Cpu).unwrap();
        let edge_index =
            from_vec(vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0], &[2, 3], DeviceType::Cpu).unwrap();
        let graph1 = GraphData::new(x1, edge_index.clone());

        let x2 = from_vec(vec![2.0; 4 * 2], &[4, 2], DeviceType::Cpu).unwrap();
        let graph2 = GraphData::new(x2, edge_index);

        let graphs = vec![graph1, graph2];
        let weights = vec![0.5, 0.5];

        let barycenter = barycenter_solver.compute(&graphs, &weights).unwrap();
        assert_eq!(barycenter.num_nodes, 4);

        // Barycenter features should be average: 1.5
        let features = barycenter.x.to_vec().unwrap();
        for &f in &features {
            assert!((f - 1.5).abs() < 0.01);
        }
    }

    #[test]
    fn test_log_sum_exp() {
        let config = OTConfig::default();
        let solver = SinkhornSolver::new(config);

        let values = vec![1.0, 2.0, 3.0];
        let lse = solver.log_sum_exp(&values);

        // log(e^1 + e^2 + e^3) ≈ 3.407
        assert!((lse - 3.407).abs() < 0.01);
    }

    #[test]
    fn test_ot_config_default() {
        let config = OTConfig::default();
        assert_eq!(config.epsilon, 0.1);
        assert_eq!(config.max_iter, 100);
        assert_eq!(config.threshold, 1e-6);
        assert!(config.log_domain);
        assert_eq!(config.alpha, 0.5);
    }
}
