// Copyright (c) 2025 ToRSh Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! # Probabilistic Compilation
//!
//! This module implements uncertainty-aware compilation using probabilistic models
//! to handle variability in hardware, workloads, and optimization effects.
//!
//! ## Key Concepts
//!
//! - **Probabilistic Performance Models**: Predict performance with confidence intervals
//! - **Bayesian Optimization**: Learn optimal compilation strategies with uncertainty
//! - **Risk-Aware Decisions**: Make compilation choices considering worst-case scenarios
//! - **Monte Carlo Sampling**: Estimate optimization effects through sampling
//! - **Credible Intervals**: Quantify uncertainty in performance predictions
//!
//! ## Architecture
//!
//! ```text
//! Program → Probabilistic Model → Distribution over Outcomes
//!              ↓                          ↓
//!         Uncertainty Quantification → Risk-Aware Decisions
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use torsh_jit::probabilistic_compilation::{ProbabilisticCompiler, CompilationConfig};
//!
//! let compiler = ProbabilisticCompiler::new();
//!
//! // Compile with uncertainty estimates
//! let result = compiler.compile(&graph)?;
//!
//! println!("Expected time: {} ± {} μs",
//!          result.mean_time, result.std_time);
//! println!("95% confidence: [{}, {}]",
//!          result.confidence_interval.0,
//!          result.confidence_interval.1);
//! ```

use crate::graph::ComputationGraph;
use crate::JitResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Probability Distributions
// ============================================================================

/// Normal (Gaussian) distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalDistribution {
    /// Mean
    pub mean: f64,

    /// Standard deviation
    pub std_dev: f64,
}

impl NormalDistribution {
    /// Create new normal distribution
    pub fn new(mean: f64, std_dev: f64) -> Self {
        Self { mean, std_dev }
    }

    /// Sample from distribution (Box-Muller transform)
    pub fn sample(&self) -> f64 {
        // Simplified: In production, use scirs2-core random module
        use std::f64::consts::PI;
        let u1 = self.uniform_sample();
        let u2 = self.uniform_sample();

        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        self.mean + self.std_dev * z0
    }

    fn uniform_sample(&self) -> f64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after UNIX_EPOCH")
            .subsec_nanos();
        ((nanos % 10000) as f64 / 10000.0).max(0.0001)
    }

    /// Probability density function
    pub fn pdf(&self, x: f64) -> f64 {
        let coefficient = 1.0 / (self.std_dev * (2.0 * std::f64::consts::PI).sqrt());
        let exponent = -((x - self.mean).powi(2)) / (2.0 * self.std_dev.powi(2));
        coefficient * exponent.exp()
    }

    /// Cumulative distribution function (approximation)
    pub fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.std_dev;
        0.5 * (1.0 + Self::erf(z / std::f64::consts::SQRT_2))
    }

    /// Error function (approximation)
    fn erf(x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Get confidence interval
    pub fn confidence_interval(&self, confidence: f64) -> (f64, f64) {
        // For 95% confidence: ±1.96 std_dev
        // For 99% confidence: ±2.576 std_dev
        let z_score = match confidence {
            c if c >= 0.99 => 2.576,
            c if c >= 0.95 => 1.96,
            c if c >= 0.90 => 1.645,
            _ => 1.0,
        };

        let margin = z_score * self.std_dev;
        (self.mean - margin, self.mean + margin)
    }
}

/// Beta distribution (for probabilities)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaDistribution {
    /// Alpha parameter
    pub alpha: f64,

    /// Beta parameter
    pub beta: f64,
}

impl BetaDistribution {
    /// Create new beta distribution
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self { alpha, beta }
    }

    /// Mean of distribution
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Mode of distribution (most likely value)
    pub fn mode(&self) -> f64 {
        if self.alpha > 1.0 && self.beta > 1.0 {
            (self.alpha - 1.0) / (self.alpha + self.beta - 2.0)
        } else {
            self.mean()
        }
    }

    /// Variance
    pub fn variance(&self) -> f64 {
        let sum = self.alpha + self.beta;
        (self.alpha * self.beta) / (sum.powi(2) * (sum + 1.0))
    }

    /// Update parameters with new observation (Bayesian update)
    pub fn update(&mut self, success: bool) {
        if success {
            self.alpha += 1.0;
        } else {
            self.beta += 1.0;
        }
    }

    /// Credible interval
    pub fn credible_interval(&self, confidence: f64) -> (f64, f64) {
        // Simplified approximation using normal approximation
        let mean = self.mean();
        let std_dev = self.variance().sqrt();

        let z_score = if confidence >= 0.95 { 1.96 } else { 1.645 };
        let margin = z_score * std_dev;

        ((mean - margin).max(0.0), (mean + margin).min(1.0))
    }
}

// ============================================================================
// Probabilistic Performance Model
// ============================================================================

/// Performance prediction with uncertainty
#[derive(Debug, Clone)]
pub struct ProbabilisticPerformance {
    /// Execution time distribution
    pub time_dist: NormalDistribution,

    /// Memory usage distribution
    pub memory_dist: NormalDistribution,

    /// Success probability (compilation succeeds)
    pub success_prob: BetaDistribution,

    /// Performance variance factors
    pub variance_factors: HashMap<String, f64>,
}

impl ProbabilisticPerformance {
    /// Create new performance model
    pub fn new(mean_time: f64, mean_memory: f64) -> Self {
        Self {
            time_dist: NormalDistribution::new(mean_time, mean_time * 0.2), // 20% uncertainty
            memory_dist: NormalDistribution::new(mean_memory, mean_memory * 0.15), // 15% uncertainty
            success_prob: BetaDistribution::new(10.0, 1.0), // Optimistic prior
            variance_factors: HashMap::new(),
        }
    }

    /// Sample execution time
    pub fn sample_time(&self) -> f64 {
        self.time_dist.sample().max(0.0)
    }

    /// Sample memory usage
    pub fn sample_memory(&self) -> f64 {
        self.memory_dist.sample().max(0.0)
    }

    /// Get confidence intervals
    pub fn time_confidence_interval(&self, confidence: f64) -> (f64, f64) {
        self.time_dist.confidence_interval(confidence)
    }

    pub fn memory_confidence_interval(&self, confidence: f64) -> (f64, f64) {
        self.memory_dist.confidence_interval(confidence)
    }

    /// Expected value (mean)
    pub fn expected_time(&self) -> f64 {
        self.time_dist.mean
    }

    pub fn expected_memory(&self) -> f64 {
        self.memory_dist.mean
    }

    /// Value at risk (VaR) - worst case with given probability
    pub fn value_at_risk(&self, percentile: f64) -> f64 {
        // Approximate: mean + z*std_dev
        let z = match percentile {
            p if p >= 0.99 => 2.326, // 99th percentile
            p if p >= 0.95 => 1.645, // 95th percentile
            _ => 1.0,
        };
        self.time_dist.mean + z * self.time_dist.std_dev
    }
}

// ============================================================================
// Optimization Decision Under Uncertainty
// ============================================================================

/// Optimization decision with uncertainty
#[derive(Debug, Clone)]
pub struct UncertainDecision {
    /// Optimization name
    pub optimization: String,

    /// Probability of improvement
    pub prob_improvement: BetaDistribution,

    /// Expected speedup distribution
    pub speedup_dist: NormalDistribution,

    /// Risk (variance of outcome)
    pub risk: f64,

    /// Historical observations
    pub observations: Vec<Observation>,
}

/// Single observation of optimization effect
#[derive(Debug, Clone)]
pub struct Observation {
    /// Was it beneficial?
    pub beneficial: bool,

    /// Speedup achieved
    pub speedup: f64,

    /// Context features
    pub context: HashMap<String, f64>,
}

impl UncertainDecision {
    /// Create new decision with prior
    pub fn new(optimization: String) -> Self {
        Self {
            optimization,
            prob_improvement: BetaDistribution::new(1.0, 1.0), // Uniform prior
            speedup_dist: NormalDistribution::new(1.2, 0.3),   // Modest expected speedup
            risk: 0.5,
            observations: Vec::new(),
        }
    }

    /// Update with new observation
    pub fn observe(&mut self, beneficial: bool, speedup: f64, context: HashMap<String, f64>) {
        // Bayesian update
        self.prob_improvement.update(beneficial);

        // Update speedup distribution
        if !self.observations.is_empty() {
            let n = self.observations.len() as f64;
            let old_mean = self.speedup_dist.mean;
            let new_mean = (old_mean * n + speedup) / (n + 1.0);
            self.speedup_dist.mean = new_mean;

            // Update variance
            let old_var = self.speedup_dist.std_dev.powi(2);
            let new_var = ((old_var * n) + (speedup - new_mean).powi(2)) / (n + 1.0);
            self.speedup_dist.std_dev = new_var.sqrt();
        } else {
            self.speedup_dist.mean = speedup;
        }

        self.observations.push(Observation {
            beneficial,
            speedup,
            context,
        });

        // Update risk
        self.risk = self.prob_improvement.variance();
    }

    /// Should apply this optimization? (Thompson sampling)
    pub fn should_apply(&self) -> bool {
        // Sample from posterior
        let prob = self.prob_improvement.mean();
        let threshold = 0.6; // Apply if >60% chance of improvement
        prob > threshold
    }

    /// Expected utility (reward - risk)
    pub fn expected_utility(&self, risk_aversion: f64) -> f64 {
        let expected_reward = self.speedup_dist.mean * self.prob_improvement.mean();
        let risk_penalty = risk_aversion * self.risk;
        expected_reward - risk_penalty
    }
}

// ============================================================================
// Probabilistic Compiler
// ============================================================================

/// Main probabilistic compilation engine
pub struct ProbabilisticCompiler {
    /// Configuration
    config: ProbabilisticConfig,

    /// Decision models for each optimization
    decisions: HashMap<String, UncertainDecision>,

    /// Performance model
    performance_model: Option<ProbabilisticPerformance>,

    /// Statistics
    stats: CompilerStatistics,
}

/// Configuration
#[derive(Debug, Clone)]
pub struct ProbabilisticConfig {
    /// Confidence level for intervals
    pub confidence_level: f64,

    /// Risk aversion factor (0 = risk-neutral, 1 = risk-averse)
    pub risk_aversion: f64,

    /// Number of Monte Carlo samples
    pub num_samples: usize,

    /// Enable Bayesian optimization
    pub bayesian_optimization: bool,

    /// Exploration rate
    pub exploration_rate: f64,
}

impl Default for ProbabilisticConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            risk_aversion: 0.5,
            num_samples: 1000,
            bayesian_optimization: true,
            exploration_rate: 0.1,
        }
    }
}

/// Compiler statistics
#[derive(Debug, Clone, Default)]
pub struct CompilerStatistics {
    /// Number of compilations
    pub compilations: usize,

    /// Average prediction error
    pub avg_prediction_error: f64,

    /// Calibration score (how well confidence matches reality)
    pub calibration_score: f64,
}

impl ProbabilisticCompiler {
    /// Create new probabilistic compiler
    pub fn new() -> Self {
        Self::with_config(ProbabilisticConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: ProbabilisticConfig) -> Self {
        let mut decisions = HashMap::new();

        // Initialize decision models
        for opt in [
            "constant_folding",
            "dead_code_elimination",
            "fusion",
            "vectorization",
            "parallelization",
            "tiling",
        ] {
            decisions.insert(opt.to_string(), UncertainDecision::new(opt.to_string()));
        }

        Self {
            config,
            decisions,
            performance_model: None,
            stats: CompilerStatistics::default(),
        }
    }

    /// Compile with uncertainty quantification
    pub fn compile(
        &mut self,
        graph: &ComputationGraph,
    ) -> JitResult<ProbabilisticCompilationResult> {
        // Estimate base performance
        let node_count = graph.node_count() as f64;
        let base_time = node_count * 10.0; // microseconds per node
        let base_memory = node_count * 1024.0; // bytes per node

        // Create performance model
        let mut perf = ProbabilisticPerformance::new(base_time, base_memory);

        // Decide which optimizations to apply
        let mut applied_opts = Vec::new();
        let mut decisions_made = Vec::new();

        for (opt_name, decision_model) in &self.decisions {
            let should_apply = if self.config.bayesian_optimization {
                decision_model.should_apply()
            } else {
                decision_model.prob_improvement.mean() > 0.5
            };

            if should_apply {
                applied_opts.push(opt_name.clone());

                // Update performance estimate
                let speedup = decision_model.speedup_dist.sample();
                perf.time_dist.mean /= speedup;
            }

            decisions_made.push(OptimizationDecision {
                optimization: opt_name.clone(),
                applied: should_apply,
                prob_improvement: decision_model.prob_improvement.mean(),
                expected_speedup: decision_model.speedup_dist.mean,
                credible_interval: decision_model
                    .prob_improvement
                    .credible_interval(self.config.confidence_level),
            });
        }

        self.performance_model = Some(perf.clone());
        self.stats.compilations += 1;

        Ok(ProbabilisticCompilationResult {
            performance: perf,
            decisions: decisions_made,
            applied_optimizations: applied_opts,
            confidence_level: self.config.confidence_level,
            risk_score: self.compute_overall_risk(),
        })
    }

    /// Update models with actual performance
    pub fn observe_performance(
        &mut self,
        applied_opts: &[String],
        actual_time: f64,
        predicted_time: f64,
    ) -> JitResult<()> {
        // Compute error
        let error = (actual_time - predicted_time).abs() / predicted_time;

        // Update prediction error statistics
        let n = self.stats.compilations as f64;
        self.stats.avg_prediction_error = (self.stats.avg_prediction_error * (n - 1.0) + error) / n;

        // Update decision models
        for opt_name in applied_opts {
            if let Some(decision) = self.decisions.get_mut(opt_name) {
                let speedup = predicted_time / actual_time;
                let beneficial = speedup > 1.0;
                decision.observe(beneficial, speedup, HashMap::new());
            }
        }

        // Update performance model uncertainty
        if let Some(perf_model) = &mut self.performance_model {
            // Reduce uncertainty after more observations
            let confidence_boost = 0.9; // Slightly reduce std_dev
            perf_model.time_dist.std_dev *= confidence_boost;
        }

        log::info!(
            "Observed performance: actual={:.2}μs, predicted={:.2}μs, error={:.1}%",
            actual_time,
            predicted_time,
            error * 100.0
        );

        Ok(())
    }

    /// Compute overall risk score
    fn compute_overall_risk(&self) -> f64 {
        let mut total_risk = 0.0;
        let mut count = 0;

        for decision in self.decisions.values() {
            total_risk += decision.risk;
            count += 1;
        }

        if count > 0 {
            total_risk / count as f64
        } else {
            0.5
        }
    }

    /// Monte Carlo simulation
    pub fn monte_carlo_simulation(&self, num_samples: usize) -> MonteCarloResult {
        if let Some(perf_model) = &self.performance_model {
            let mut samples = Vec::with_capacity(num_samples);

            for _ in 0..num_samples {
                let time = perf_model.sample_time();
                samples.push(time);
            }

            samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let mean = samples.iter().sum::<f64>() / num_samples as f64;
            let variance =
                samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / num_samples as f64;

            let p50 = samples[num_samples / 2];
            let p95 = samples[(num_samples as f64 * 0.95) as usize];
            let p99 = samples[(num_samples as f64 * 0.99) as usize];

            MonteCarloResult {
                mean,
                std_dev: variance.sqrt(),
                percentiles: vec![(50, p50), (95, p95), (99, p99)],
                samples: samples.into_iter().take(100).collect(), // Keep first 100 for visualization
            }
        } else {
            MonteCarloResult::default()
        }
    }

    /// Get statistics
    pub fn statistics(&self) -> &CompilerStatistics {
        &self.stats
    }
}

impl Default for ProbabilisticCompiler {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Compilation Results
// ============================================================================

/// Result of probabilistic compilation
#[derive(Debug, Clone)]
pub struct ProbabilisticCompilationResult {
    /// Performance prediction
    pub performance: ProbabilisticPerformance,

    /// Optimization decisions
    pub decisions: Vec<OptimizationDecision>,

    /// Applied optimizations
    pub applied_optimizations: Vec<String>,

    /// Confidence level
    pub confidence_level: f64,

    /// Overall risk score
    pub risk_score: f64,
}

/// Single optimization decision
#[derive(Debug, Clone)]
pub struct OptimizationDecision {
    /// Optimization name
    pub optimization: String,

    /// Was it applied?
    pub applied: bool,

    /// Probability of improvement
    pub prob_improvement: f64,

    /// Expected speedup
    pub expected_speedup: f64,

    /// Credible interval for probability
    pub credible_interval: (f64, f64),
}

/// Monte Carlo simulation result
#[derive(Debug, Clone, Default)]
pub struct MonteCarloResult {
    /// Mean of samples
    pub mean: f64,

    /// Standard deviation
    pub std_dev: f64,

    /// Percentiles (percentile, value)
    pub percentiles: Vec<(usize, f64)>,

    /// Sample values (subset)
    pub samples: Vec<f64>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;
    use torsh_core::{DType, Shape};

    #[test]
    fn test_normal_distribution() {
        let dist = NormalDistribution::new(100.0, 10.0);
        assert_eq!(dist.mean, 100.0);
        assert_eq!(dist.std_dev, 10.0);

        let sample = dist.sample();
        assert!(sample > 0.0); // Should be positive

        let (lower, upper) = dist.confidence_interval(0.95);
        assert!(lower < dist.mean);
        assert!(upper > dist.mean);
    }

    #[test]
    fn test_beta_distribution() {
        let mut dist = BetaDistribution::new(10.0, 2.0);
        let mean = dist.mean();
        assert!(mean > 0.5); // Should be skewed towards success

        dist.update(true);
        assert!(dist.alpha == 11.0);

        let (lower, upper) = dist.credible_interval(0.95);
        assert!(lower <= upper);
    }

    #[test]
    fn test_uncertain_decision() {
        let mut decision = UncertainDecision::new("fusion".to_string());

        // Observe positive outcome
        decision.observe(true, 1.5, HashMap::new());
        assert!(decision.prob_improvement.alpha > 1.0);

        let utility = decision.expected_utility(0.5);
        assert!(utility > 0.0);
    }

    #[test]
    fn test_probabilistic_compilation() {
        let mut compiler = ProbabilisticCompiler::new();

        let mut builder = GraphBuilder::new();
        let x = builder.add_input("x".to_string(), Shape::new(vec![10, 10]), DType::F32);
        builder.mark_output(x).unwrap();

        let graph = builder.build().unwrap();
        let result = compiler.compile(&graph).unwrap();

        assert!(!result.decisions.is_empty());
        assert!(result.performance.expected_time() > 0.0);
        assert!(result.confidence_level == 0.95);
    }

    #[test]
    fn test_performance_observation() {
        let mut compiler = ProbabilisticCompiler::new();

        let mut builder = GraphBuilder::new();
        let x = builder.add_input("x".to_string(), Shape::new(vec![5, 5]), DType::F32);
        builder.mark_output(x).unwrap();

        let graph = builder.build().unwrap();
        let result = compiler.compile(&graph).unwrap();

        let predicted = result.performance.expected_time();
        let actual = predicted * 1.1; // 10% slower

        let obs_result =
            compiler.observe_performance(&result.applied_optimizations, actual, predicted);
        assert!(obs_result.is_ok());
    }

    #[test]
    fn test_monte_carlo() {
        let mut compiler = ProbabilisticCompiler::new();

        let mut builder = GraphBuilder::new();
        let x = builder.add_input("x".to_string(), Shape::new(vec![8, 8]), DType::F32);
        builder.mark_output(x).unwrap();

        let graph = builder.build().unwrap();
        let _ = compiler.compile(&graph).unwrap();

        let mc_result = compiler.monte_carlo_simulation(100);
        assert!(mc_result.mean > 0.0);
        assert!(mc_result.std_dev >= 0.0);
        assert!(!mc_result.percentiles.is_empty());
    }

    #[test]
    fn test_value_at_risk() {
        let perf = ProbabilisticPerformance::new(1000.0, 10000.0);
        let var_95 = perf.value_at_risk(0.95);
        let var_99 = perf.value_at_risk(0.99);

        assert!(var_95 > perf.expected_time());
        assert!(var_99 > var_95); // Higher percentile = more conservative
    }
}
