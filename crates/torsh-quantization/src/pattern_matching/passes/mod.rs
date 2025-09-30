//! Optimization passes for pattern matching and graph transformation.
//!
//! This module provides a unified interface for various optimization passes
//! that can be applied to computation graphs. The passes include:
//!
//! - **Pattern Optimization**: Applies pattern-based optimizations like fusion and elimination
//! - **Dead Code Elimination**: Removes unreachable and unused nodes from the graph
//! - **Constant Folding**: Evaluates constant expressions at compile time
//!
//! # Usage
//!
//! ```rust
//! use torsh_quantization::pattern_matching::passes::{PassManager, PassConfig};
//! use torsh_quantization::pattern_matching::ComputationGraph;
//!
//! let mut graph = ComputationGraph::new();
//! let mut pass_manager = PassManager::new();
//! let result = pass_manager.run_all(&mut graph)?;
//! ```

use crate::error::TorshResult;
use crate::pattern_matching::graph::ComputationGraph;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub mod constant_folding;
pub mod dead_code_elimination;
pub mod pattern_optimization;

pub use constant_folding::{ConstantFoldingPass, ConstantValue, FoldingConfig, FoldingStatistics};
pub use dead_code_elimination::{
    DeadCodeEliminationPass, EliminationConfig, EliminationStatistics,
};
pub use pattern_optimization::{
    OptimizationConfig, OptimizationStatistics, PatternOptimizationPass,
};

/// Configuration for the pass manager and execution order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassConfig {
    /// Enable pattern optimization pass
    pub enable_pattern_optimization: bool,
    /// Enable dead code elimination pass
    pub enable_dead_code_elimination: bool,
    /// Enable constant folding pass
    pub enable_constant_folding: bool,
    /// Maximum number of iterations for iterative passes
    pub max_iterations: usize,
    /// Convergence threshold for iterative optimization
    pub convergence_threshold: f64,
    /// Enable verbose logging during pass execution
    pub verbose: bool,
    /// Custom pass execution order (if None, uses default order)
    pub custom_order: Option<Vec<PassType>>,
}

impl Default for PassConfig {
    fn default() -> Self {
        Self {
            enable_pattern_optimization: true,
            enable_dead_code_elimination: true,
            enable_constant_folding: true,
            max_iterations: 10,
            convergence_threshold: 1e-6,
            verbose: false,
            custom_order: None,
        }
    }
}

/// Types of optimization passes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PassType {
    PatternOptimization,
    DeadCodeElimination,
    ConstantFolding,
}

/// Result of running optimization passes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassResult {
    /// Total execution time for all passes
    pub total_time: Duration,
    /// Per-pass execution times
    pub pass_times: HashMap<PassType, Duration>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether the optimization converged
    pub converged: bool,
    /// Statistics from pattern optimization
    pub pattern_stats: Option<OptimizationStatistics>,
    /// Statistics from dead code elimination
    pub elimination_stats: Option<EliminationStatistics>,
    /// Statistics from constant folding
    pub folding_stats: Option<FoldingStatistics>,
    /// Total nodes before optimization
    pub nodes_before: usize,
    /// Total nodes after optimization
    pub nodes_after: usize,
    /// Overall improvement score (0.0 to 1.0)
    pub improvement_score: f64,
}

/// Pass manager for coordinating multiple optimization passes
#[derive(Debug)]
pub struct PassManager {
    config: PassConfig,
    pattern_pass: Option<PatternOptimizationPass>,
    elimination_pass: Option<DeadCodeEliminationPass>,
    folding_pass: Option<ConstantFoldingPass>,
}

impl PassManager {
    /// Create a new pass manager with default configuration
    pub fn new() -> Self {
        Self::with_config(PassConfig::default())
    }

    /// Create a new pass manager with custom configuration
    pub fn with_config(config: PassConfig) -> Self {
        let pattern_pass = if config.enable_pattern_optimization {
            Some(PatternOptimizationPass::new())
        } else {
            None
        };

        let elimination_pass = if config.enable_dead_code_elimination {
            Some(DeadCodeEliminationPass::new())
        } else {
            None
        };

        let folding_pass = if config.enable_constant_folding {
            Some(ConstantFoldingPass::new())
        } else {
            None
        };

        Self {
            config,
            pattern_pass,
            elimination_pass,
            folding_pass,
        }
    }

    /// Configure pattern optimization pass
    pub fn configure_pattern_optimization(&mut self, config: OptimizationConfig) {
        if let Some(ref mut pass) = self.pattern_pass {
            pass.configure(config);
        }
    }

    /// Configure dead code elimination pass
    pub fn configure_dead_code_elimination(&mut self, config: EliminationConfig) {
        if let Some(ref mut pass) = self.elimination_pass {
            pass.configure(config);
        }
    }

    /// Configure constant folding pass
    pub fn configure_constant_folding(&mut self, config: FoldingConfig) {
        if let Some(ref mut pass) = self.folding_pass {
            pass.configure(config);
        }
    }

    /// Run all enabled passes on the computation graph
    pub fn run_all(&mut self, graph: &mut ComputationGraph) -> TorshResult<PassResult> {
        let start_time = Instant::now();
        let nodes_before = graph.nodes.len();

        let mut pass_times = HashMap::new();
        let mut pattern_stats = None;
        let mut elimination_stats = None;
        let mut folding_stats = None;
        let mut iterations = 0;
        let mut converged = false;

        // Determine pass execution order
        let pass_order = self.config.custom_order.clone().unwrap_or_else(|| {
            vec![
                PassType::PatternOptimization,
                PassType::ConstantFolding,
                PassType::DeadCodeElimination,
            ]
        });

        // Run passes iteratively until convergence or max iterations
        let mut prev_node_count = graph.nodes.len();

        for iteration in 0..self.config.max_iterations {
            iterations = iteration + 1;
            let mut changed = false;

            for &pass_type in &pass_order {
                if !self.is_pass_enabled(pass_type) {
                    continue;
                }

                let pass_start = Instant::now();
                let pass_changed = self.run_pass(
                    pass_type,
                    graph,
                    &mut pattern_stats,
                    &mut elimination_stats,
                    &mut folding_stats,
                )?;
                let pass_duration = pass_start.elapsed();

                pass_times
                    .entry(pass_type)
                    .and_modify(|d| *d += pass_duration)
                    .or_insert(pass_duration);

                changed |= pass_changed;

                if self.config.verbose {
                    println!(
                        "Pass {:?} iteration {} completed in {:?}, changed: {}",
                        pass_type,
                        iteration + 1,
                        pass_duration,
                        pass_changed
                    );
                }
            }

            // Check for convergence
            let current_node_count = graph.nodes.len();
            let relative_change = if prev_node_count > 0 {
                (prev_node_count as f64 - current_node_count as f64).abs() / prev_node_count as f64
            } else {
                0.0
            };

            if !changed || relative_change < self.config.convergence_threshold {
                converged = true;
                if self.config.verbose {
                    println!("Optimization converged after {} iterations", iteration + 1);
                }
                break;
            }

            prev_node_count = current_node_count;
        }

        let total_time = start_time.elapsed();
        let nodes_after = graph.nodes.len();
        let improvement_score = if nodes_before > 0 {
            (nodes_before as f64 - nodes_after as f64) / nodes_before as f64
        } else {
            0.0
        }
        .max(0.0)
        .min(1.0);

        Ok(PassResult {
            total_time,
            pass_times,
            iterations,
            converged,
            pattern_stats,
            elimination_stats,
            folding_stats,
            nodes_before,
            nodes_after,
            improvement_score,
        })
    }

    /// Run a single pass
    pub fn run_single_pass(
        &mut self,
        pass_type: PassType,
        graph: &mut ComputationGraph,
    ) -> TorshResult<bool> {
        let mut pattern_stats = None;
        let mut elimination_stats = None;
        let mut folding_stats = None;

        self.run_pass(
            pass_type,
            graph,
            &mut pattern_stats,
            &mut elimination_stats,
            &mut folding_stats,
        )
    }

    /// Check if a pass is enabled
    fn is_pass_enabled(&self, pass_type: PassType) -> bool {
        match pass_type {
            PassType::PatternOptimization => self.config.enable_pattern_optimization,
            PassType::DeadCodeElimination => self.config.enable_dead_code_elimination,
            PassType::ConstantFolding => self.config.enable_constant_folding,
        }
    }

    /// Run a specific pass and update statistics
    fn run_pass(
        &mut self,
        pass_type: PassType,
        graph: &mut ComputationGraph,
        pattern_stats: &mut Option<OptimizationStatistics>,
        elimination_stats: &mut Option<EliminationStatistics>,
        folding_stats: &mut Option<FoldingStatistics>,
    ) -> TorshResult<bool> {
        match pass_type {
            PassType::PatternOptimization => {
                if let Some(ref mut pass) = self.pattern_pass {
                    let result = pass.optimize(graph)?;
                    *pattern_stats = Some(pass.get_statistics().clone());
                    Ok(result.optimizations_applied > 0)
                } else {
                    Ok(false)
                }
            }
            PassType::DeadCodeElimination => {
                if let Some(ref mut pass) = self.elimination_pass {
                    let result = pass.eliminate_dead_code(graph)?;
                    *elimination_stats = Some(pass.get_statistics().clone());
                    Ok(result.nodes_removed > 0)
                } else {
                    Ok(false)
                }
            }
            PassType::ConstantFolding => {
                if let Some(ref mut pass) = self.folding_pass {
                    let result = pass.fold_constants(graph)?;
                    *folding_stats = Some(pass.get_statistics().clone());
                    Ok(result.nodes_folded > 0)
                } else {
                    Ok(false)
                }
            }
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &PassConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: PassConfig) {
        self.config = config;

        // Update enabled passes
        if !config.enable_pattern_optimization {
            self.pattern_pass = None;
        } else if self.pattern_pass.is_none() {
            self.pattern_pass = Some(PatternOptimizationPass::new());
        }

        if !config.enable_dead_code_elimination {
            self.elimination_pass = None;
        } else if self.elimination_pass.is_none() {
            self.elimination_pass = Some(DeadCodeEliminationPass::new());
        }

        if !config.enable_constant_folding {
            self.folding_pass = None;
        } else if self.folding_pass.is_none() {
            self.folding_pass = Some(ConstantFoldingPass::new());
        }
    }

    /// Reset all pass statistics
    pub fn reset_statistics(&mut self) {
        if let Some(ref mut pass) = self.pattern_pass {
            pass.reset_statistics();
        }
        if let Some(ref mut pass) = self.elimination_pass {
            pass.reset_statistics();
        }
        if let Some(ref mut pass) = self.folding_pass {
            pass.reset_statistics();
        }
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for pass management
pub mod utils {
    use super::*;

    /// Create a pass manager with only pattern optimization enabled
    pub fn pattern_only() -> PassManager {
        PassManager::with_config(PassConfig {
            enable_pattern_optimization: true,
            enable_dead_code_elimination: false,
            enable_constant_folding: false,
            ..Default::default()
        })
    }

    /// Create a pass manager with only dead code elimination enabled
    pub fn elimination_only() -> PassManager {
        PassManager::with_config(PassConfig {
            enable_pattern_optimization: false,
            enable_dead_code_elimination: true,
            enable_constant_folding: false,
            ..Default::default()
        })
    }

    /// Create a pass manager with only constant folding enabled
    pub fn folding_only() -> PassManager {
        PassManager::with_config(PassConfig {
            enable_pattern_optimization: false,
            enable_dead_code_elimination: false,
            enable_constant_folding: true,
            ..Default::default()
        })
    }

    /// Create a pass manager optimized for fast compilation
    pub fn fast_compile() -> PassManager {
        PassManager::with_config(PassConfig {
            enable_pattern_optimization: true,
            enable_dead_code_elimination: true,
            enable_constant_folding: false, // Skip constant folding for speed
            max_iterations: 3,
            convergence_threshold: 1e-3,
            verbose: false,
            custom_order: Some(vec![
                PassType::DeadCodeElimination,
                PassType::PatternOptimization,
            ]),
        })
    }

    /// Create a pass manager optimized for maximum optimization
    pub fn max_optimization() -> PassManager {
        PassManager::with_config(PassConfig {
            enable_pattern_optimization: true,
            enable_dead_code_elimination: true,
            enable_constant_folding: true,
            max_iterations: 20,
            convergence_threshold: 1e-8,
            verbose: false,
            custom_order: Some(vec![
                PassType::ConstantFolding,
                PassType::PatternOptimization,
                PassType::DeadCodeElimination,
                PassType::PatternOptimization, // Second round for better optimization
            ]),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern_matching::graph::*;

    #[test]
    fn test_pass_manager_creation() {
        let manager = PassManager::new();
        assert!(manager.config.enable_pattern_optimization);
        assert!(manager.config.enable_dead_code_elimination);
        assert!(manager.config.enable_constant_folding);
    }

    #[test]
    fn test_pass_config_serialization() {
        let config = PassConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: PassConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(
            config.enable_pattern_optimization,
            deserialized.enable_pattern_optimization
        );
        assert_eq!(config.max_iterations, deserialized.max_iterations);
    }

    #[test]
    fn test_utility_managers() {
        let pattern_only = utils::pattern_only();
        assert!(pattern_only.config.enable_pattern_optimization);
        assert!(!pattern_only.config.enable_dead_code_elimination);
        assert!(!pattern_only.config.enable_constant_folding);

        let fast_compile = utils::fast_compile();
        assert_eq!(fast_compile.config.max_iterations, 3);
        assert!(!fast_compile.config.enable_constant_folding);

        let max_opt = utils::max_optimization();
        assert_eq!(max_opt.config.max_iterations, 20);
        assert!(max_opt.config.enable_constant_folding);
    }

    #[test]
    fn test_pass_manager_with_empty_graph() {
        let mut manager = PassManager::new();
        let mut graph = ComputationGraph::new();

        let result = manager.run_all(&mut graph).unwrap();
        assert_eq!(result.nodes_before, 0);
        assert_eq!(result.nodes_after, 0);
        assert!(result.converged);
    }
}
