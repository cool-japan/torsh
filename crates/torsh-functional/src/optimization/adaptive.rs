//! Adaptive algorithm selection for optimization
//!
//! This module provides intelligent algorithm selection based on tensor characteristics
//! and historical performance data to automatically choose the best optimization method.

use torsh_core::Result as TorshResult;
use torsh_tensor::Tensor;

/// Tensor characteristics for algorithm selection
#[derive(Debug, Clone)]
pub struct TensorCharacteristics {
    /// Number of elements in the tensor
    pub size: usize,
    /// Condition number estimate (ratio of largest to smallest singular value)
    pub condition_number: f32,
    /// Sparsity ratio (fraction of zero elements)
    pub sparsity: f32,
    /// Numerical precision characteristics
    pub numerical_precision: f32,
    /// Memory layout efficiency
    pub memory_layout_score: f32,
    /// Computational complexity estimate
    pub computational_complexity: f32,
}

impl TensorCharacteristics {
    /// Analyze tensor characteristics
    pub fn analyze(tensor: &Tensor) -> TorshResult<Self> {
        let data = tensor.data()?;
        let size = data.len();

        // Calculate sparsity
        let zero_count = data.iter().filter(|&&x| x.abs() < 1e-12).count();
        let sparsity = zero_count as f32 / size as f32;

        // Estimate condition number using simple heuristic
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b.abs()));
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| {
            if b.abs() > 1e-12 {
                a.min(b.abs())
            } else {
                a
            }
        });
        let condition_number = if min_val > 0.0 {
            max_val / min_val
        } else {
            f32::INFINITY
        };

        // Estimate numerical precision based on dynamic range
        let mean_val = data.iter().sum::<f32>() / size as f32;
        let std_dev =
            (data.iter().map(|&x| (x - mean_val).powi(2)).sum::<f32>() / size as f32).sqrt();
        let numerical_precision = if std_dev > 0.0 {
            mean_val.abs() / std_dev
        } else {
            1.0
        };

        // Simple memory layout score based on tensor shape
        let shape_binding = tensor.shape();
        let shape = shape_binding.dims();
        let memory_layout_score = match shape.len() {
            1 => 1.0, // Linear - optimal
            2 => 0.8, // Matrix - good
            3 => 0.6, // 3D - fair
            _ => 0.4, // Higher dimensional - poor
        };

        // Computational complexity estimate based on size and dimensionality
        let computational_complexity = size as f32 * shape.len() as f32;

        Ok(TensorCharacteristics {
            size,
            condition_number,
            sparsity,
            numerical_precision,
            memory_layout_score,
            computational_complexity,
        })
    }
}

/// Algorithm recommendation based on tensor characteristics
#[derive(Debug, Clone)]
pub enum OptimizationAlgorithm {
    /// Gradient descent (for well-conditioned, dense problems)
    GradientDescent,
    /// Momentum gradient descent (for noisy or ill-conditioned problems)
    MomentumGradientDescent,
    /// Adam optimizer (for sparse or noisy gradients)
    Adam,
    /// L-BFGS (for smooth, deterministic problems)
    LBFGS,
    /// Conjugate gradient (for large, sparse, positive definite systems)
    ConjugateGradient,
}

/// Adaptive algorithm selector
pub struct AdaptiveAlgorithmSelector {
    /// Performance history for different algorithms
    performance_history: std::collections::HashMap<String, Vec<f32>>,
    /// Learning rate for adaptation
    #[allow(dead_code)]
    adaptation_rate: f32,
}

impl AdaptiveAlgorithmSelector {
    /// Create new adaptive algorithm selector
    pub fn new() -> Self {
        Self {
            performance_history: std::collections::HashMap::new(),
            adaptation_rate: 0.1,
        }
    }

    /// Select best optimization algorithm based on tensor characteristics
    pub fn select_algorithm(
        &self,
        characteristics: &TensorCharacteristics,
    ) -> OptimizationAlgorithm {
        // Rule-based selection with learned adjustments

        // For very sparse problems, prefer specialized sparse algorithms
        if characteristics.sparsity > 0.8 {
            return OptimizationAlgorithm::ConjugateGradient;
        }

        // For large problems, use memory-efficient methods
        if characteristics.size > 1_000_000 {
            return match characteristics.condition_number {
                cn if cn > 1000.0 => OptimizationAlgorithm::Adam,
                _ => OptimizationAlgorithm::LBFGS,
            };
        }

        // For well-conditioned, smooth problems
        if characteristics.condition_number < 100.0 && characteristics.numerical_precision > 0.1 {
            return OptimizationAlgorithm::LBFGS;
        }

        // For ill-conditioned problems
        if characteristics.condition_number > 1000.0 {
            return OptimizationAlgorithm::Adam;
        }

        // For moderately sized, general problems
        if characteristics.size < 10_000 {
            OptimizationAlgorithm::MomentumGradientDescent
        } else {
            OptimizationAlgorithm::Adam
        }
    }

    /// Record performance for algorithm adaptation
    pub fn record_performance(&mut self, algorithm: &OptimizationAlgorithm, performance: f32) {
        let key = format!("{:?}", algorithm);
        self.performance_history
            .entry(key)
            .or_insert_with(Vec::new)
            .push(performance);

        // Keep only recent history (last 100 entries)
        if let Some(history) = self
            .performance_history
            .get_mut(&format!("{:?}", algorithm))
        {
            if history.len() > 100 {
                history.drain(0..history.len() - 100);
            }
        }
    }

    /// Get performance score for an algorithm
    pub fn get_algorithm_score(&self, algorithm: &OptimizationAlgorithm) -> f32 {
        let key = format!("{:?}", algorithm);
        if let Some(history) = self.performance_history.get(&key) {
            if !history.is_empty() {
                history.iter().sum::<f32>() / history.len() as f32
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Adaptive selection that learns from performance history
    pub fn select_algorithm_adaptive(
        &self,
        characteristics: &TensorCharacteristics,
    ) -> OptimizationAlgorithm {
        let base_selection = self.select_algorithm(characteristics);

        // Check if we have enough history to make adaptive decisions
        let base_score = self.get_algorithm_score(&base_selection);

        // Try alternative algorithms if base algorithm performs poorly
        if base_score < 0.5 && self.performance_history.len() > 3 {
            let alternatives = vec![
                OptimizationAlgorithm::Adam,
                OptimizationAlgorithm::MomentumGradientDescent,
                OptimizationAlgorithm::LBFGS,
            ];

            let best_alternative = alternatives.iter().max_by(|a, b| {
                self.get_algorithm_score(a)
                    .partial_cmp(&self.get_algorithm_score(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            if let Some(best) = best_alternative {
                if self.get_algorithm_score(best) > base_score + 0.1 {
                    return best.clone();
                }
            }
        }

        base_selection
    }
}

impl Default for AdaptiveAlgorithmSelector {
    fn default() -> Self {
        Self::new()
    }
}

/// Analyze problem characteristics and recommend optimization strategy
pub fn analyze_optimization_problem(
    objective_values: &[f32],
    gradient_norms: &[f32],
    tensor_characteristics: &TensorCharacteristics,
) -> (OptimizationAlgorithm, String) {
    let mut recommendations = Vec::new();

    // Analyze convergence pattern
    if objective_values.len() > 5 {
        let recent_improvement = objective_values
            .windows(2)
            .rev()
            .take(5)
            .map(|w| w[0] - w[1])
            .sum::<f32>()
            / 5.0;

        if recent_improvement < 1e-8 {
            recommendations.push(
                "Problem appears to have slow convergence - consider Adam or momentum methods",
            );
        }
    }

    // Analyze gradient characteristics
    if gradient_norms.len() > 3 {
        let gradient_variance = {
            let mean = gradient_norms.iter().sum::<f32>() / gradient_norms.len() as f32;
            gradient_norms
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>()
                / gradient_norms.len() as f32
        };

        if gradient_variance > 1.0 {
            recommendations.push("High gradient variance detected - Adam optimizer recommended");
        }
    }

    // Use selector for algorithm choice
    let selector = AdaptiveAlgorithmSelector::new();
    let algorithm = selector.select_algorithm(tensor_characteristics);

    // Generate recommendation summary
    let recommendation_text = if recommendations.is_empty() {
        format!(
            "Recommended algorithm: {:?} based on tensor characteristics",
            algorithm
        )
    } else {
        format!(
            "Recommended algorithm: {:?}. Additional notes: {}",
            algorithm,
            recommendations.join("; ")
        )
    };

    (algorithm, recommendation_text)
}

/// Auto-configure optimization parameters based on problem characteristics
pub fn auto_configure_optimization(
    characteristics: &TensorCharacteristics,
    algorithm: &OptimizationAlgorithm,
) -> TorshResult<String> {
    let config = match algorithm {
        OptimizationAlgorithm::GradientDescent => {
            let lr = if characteristics.condition_number > 100.0 {
                0.001 // Conservative for ill-conditioned problems
            } else {
                0.01 // Standard rate for well-conditioned problems
            };
            format!(
                "GradientDescentParams {{ learning_rate: {}, max_iter: {}, tolerance: {} }}",
                lr, 1000, 1e-6
            )
        }

        OptimizationAlgorithm::MomentumGradientDescent => {
            let lr = if characteristics.size > 100_000 {
                0.001
            } else {
                0.01
            };
            let momentum = if characteristics.condition_number > 100.0 {
                0.95
            } else {
                0.9
            };
            format!(
                "MomentumParams {{ learning_rate: {}, momentum: {}, max_iter: {}, tolerance: {} }}",
                lr, momentum, 1000, 1e-6
            )
        }

        OptimizationAlgorithm::Adam => {
            let lr = if characteristics.sparsity > 0.5 {
                0.001
            } else {
                0.01
            };
            format!("AdamParams {{ learning_rate: {}, beta1: 0.9, beta2: 0.999, eps: 1e-8, max_iter: {}, tolerance: {} }}",
                    lr, 2000, 1e-6)
        }

        OptimizationAlgorithm::LBFGS => {
            let m = if characteristics.size > 10_000 { 5 } else { 10 };
            format!(
                "BFGSParams {{ memory_size: {}, max_iter: {}, tolerance: {} }}",
                m, 500, 1e-6
            )
        }

        OptimizationAlgorithm::ConjugateGradient => {
            format!(
                "ConjugateGradientParams {{ max_iter: {}, tolerance: {}, restart_frequency: {} }}",
                1000,
                1e-6,
                characteristics.size.min(50)
            )
        }
    };

    Ok(config)
}
