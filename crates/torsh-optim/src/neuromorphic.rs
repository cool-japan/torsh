//! Neuromorphic Optimization
//!
//! This module implements biologically-inspired optimization algorithms based on
//! spiking neural networks (SNNs) and brain-inspired learning rules.
//!
//! # Key Features
//!
//! - **STDP (Spike-Timing-Dependent Plasticity)**: Hebbian learning rule based on spike timing
//! - **Event-driven optimization**: Sparse updates triggered by spike events
//! - **Energy-aware computation**: Minimizes computational cost inspired by biological efficiency
//! - **Temporal credit assignment**: Handles delayed reward signals with eligibility traces
//! - **Adaptive thresholding**: Dynamic spike thresholds for stability
//!
//! # Algorithms
//!
//! ## STDPOptimizer
//!
//! Implements spike-timing-dependent plasticity, where synaptic weights are updated
//! based on the relative timing of pre- and post-synaptic spikes.
//!
//! Weight update rule:
//! - If pre-spike before post-spike: Δw = A+ * exp(-Δt / τ+)  (potentiation)
//! - If post-spike before pre-spike: Δw = -A- * exp(-Δt / τ-)  (depression)
//!
//! ## EventDrivenOptimizer
//!
//! Sparse gradient-based optimization that only updates parameters when
//! corresponding "spikes" (large gradient magnitudes) are detected.
//!
//! ## TemporalCreditAssignment
//!
//! Handles credit assignment over time using eligibility traces, allowing
//! delayed rewards to influence earlier parameter updates.
//!
//! # References
//!
//! - Gerstner & Kistler (2002). "Spiking Neuron Models"
//! - Bi & Poo (1998). "Synaptic modifications in cultured hippocampal neurons"
//! - Bellec et al. (2020). "A solution to the learning dilemma for recurrent networks of spiking neurons"

use crate::{Optimizer, OptimizerError, OptimizerResult, OptimizerState};
use parking_lot::RwLock;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{thread_rng, Uniform};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use torsh_tensor::Tensor;

// ============================================================================
// STDP (Spike-Timing-Dependent Plasticity) Optimizer
// ============================================================================

/// STDP learning configuration
#[derive(Debug, Clone)]
pub struct STDPConfig {
    /// Potentiation amplitude (A+)
    pub a_plus: f64,
    /// Depression amplitude (A-)
    pub a_minus: f64,
    /// Potentiation time constant (τ+)
    pub tau_plus: f64,
    /// Depression time constant (τ-)
    pub tau_minus: f64,
    /// Maximum allowed weight
    pub w_max: f64,
    /// Minimum allowed weight
    pub w_min: f64,
    /// Spike threshold for membrane potential
    pub spike_threshold: f64,
    /// Membrane time constant
    pub tau_membrane: f64,
    /// Resting potential reset value
    pub v_reset: f64,
}

impl Default for STDPConfig {
    fn default() -> Self {
        Self {
            a_plus: 0.01,
            a_minus: 0.01,
            tau_plus: 20.0,
            tau_minus: 20.0,
            w_max: 1.0,
            w_min: -1.0,
            spike_threshold: 1.0,
            tau_membrane: 10.0,
            v_reset: 0.0,
        }
    }
}

/// Spike timing state for a neuron
#[derive(Debug, Clone)]
struct SpikeState {
    /// Last spike time
    last_spike_time: Option<f64>,
    /// Membrane potential
    membrane_potential: f64,
    /// Spike history (time, magnitude)
    spike_history: VecDeque<(f64, f64)>,
}

impl Default for SpikeState {
    fn default() -> Self {
        Self {
            last_spike_time: None,
            membrane_potential: 0.0,
            spike_history: VecDeque::with_capacity(100),
        }
    }
}

/// STDP-based optimizer
///
/// Implements spike-timing-dependent plasticity for parameter updates.
/// Updates are based on the correlation between pre- and post-synaptic
/// spike timings.
pub struct STDPOptimizer {
    /// Learning rate
    lr: f32,
    /// STDP configuration
    config: STDPConfig,
    /// Current time step
    current_time: f64,
    /// Parameter groups
    param_groups: Vec<Arc<RwLock<Tensor>>>,
    /// Spike states per parameter
    spike_states: HashMap<String, SpikeState>,
    /// Eligibility traces
    eligibility_traces: HashMap<String, Tensor>,
}

impl STDPOptimizer {
    /// Create a new STDP optimizer
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: f32,
        config: STDPConfig,
    ) -> OptimizerResult<Self> {
        if lr <= 0.0 {
            return Err(OptimizerError::InvalidParameter(format!(
                "Invalid learning rate: {}",
                lr
            )));
        }

        let mut spike_states = HashMap::new();
        let mut eligibility_traces = HashMap::new();

        for (i, param) in params.iter().enumerate() {
            let param_key = format!("param_{}", i);
            spike_states.insert(param_key.clone(), SpikeState::default());

            // Initialize eligibility trace
            let param_read = param.read();
            let shape_owned = param_read.shape().dims().to_vec();
            drop(param_read);
            let trace = torsh_tensor::creation::zeros(&shape_owned)?;
            eligibility_traces.insert(param_key, trace);
        }

        Ok(Self {
            lr,
            config,
            current_time: 0.0,
            param_groups: params,
            spike_states,
            eligibility_traces,
        })
    }

    /// Create with default configuration
    pub fn with_defaults(params: Vec<Arc<RwLock<Tensor>>>, lr: f32) -> OptimizerResult<Self> {
        Self::new(params, lr, STDPConfig::default())
    }

    /// Detect spikes based on gradient magnitude
    fn detect_spike(&mut self, param_key: &str, gradient: &Tensor) -> OptimizerResult<bool> {
        let state = self
            .spike_states
            .get_mut(param_key)
            .expect("spike_states should exist for param_key");

        // Calculate "membrane potential" as a function of gradient magnitude
        let grad_norm = gradient.norm()?.item()?;
        let grad_norm_f64 = grad_norm as f64;

        // Leaky integration of gradient
        state.membrane_potential =
            state.membrane_potential * (1.0 - 1.0 / self.config.tau_membrane) + grad_norm_f64;

        // Check if spike threshold is crossed
        if state.membrane_potential > self.config.spike_threshold {
            state.last_spike_time = Some(self.current_time);
            state
                .spike_history
                .push_back((self.current_time, grad_norm_f64));

            // Keep history bounded
            if state.spike_history.len() > 100 {
                state.spike_history.pop_front();
            }

            // Reset membrane potential
            state.membrane_potential = self.config.v_reset;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Compute STDP weight change
    fn compute_stdp_change(&self, pre_time: f64, post_time: f64) -> f64 {
        let dt = post_time - pre_time;

        if dt > 0.0 {
            // Post after pre -> potentiation (LTP)
            self.config.a_plus * (-dt / self.config.tau_plus).exp()
        } else {
            // Pre after post -> depression (LTD)
            -self.config.a_minus * (dt.abs() / self.config.tau_minus).exp()
        }
    }

    /// Update eligibility trace
    fn update_eligibility_trace(
        &mut self,
        param_key: &str,
        gradient: &Tensor,
    ) -> OptimizerResult<()> {
        let trace = self
            .eligibility_traces
            .get_mut(param_key)
            .expect("eligibility_traces should exist for param_key");

        // Exponential decay: e(t+1) = λ * e(t) + grad
        let decay = 0.95; // Decay factor
        *trace = trace.mul_scalar(decay)?;
        *trace = trace.add(gradient)?;

        Ok(())
    }
}

impl Optimizer for STDPOptimizer {
    fn step(&mut self) -> OptimizerResult<()> {
        self.current_time += 1.0;

        // Collect gradients first to avoid borrow issues
        let mut gradients = Vec::new();
        for param in self.param_groups.iter() {
            let grad_opt = param.read().grad().clone();
            gradients.push(grad_opt);
        }

        for (i, grad_opt) in gradients.into_iter().enumerate() {
            let param_key = format!("param_{}", i);

            if let Some(grad) = grad_opt {
                // Update eligibility trace
                self.update_eligibility_trace(&param_key, &grad)?;

                // Detect spike
                let spiked = self.detect_spike(&param_key, &grad)?;

                if spiked {
                    // Apply STDP rule
                    let state = &self.spike_states[&param_key];

                    // Compute STDP-based update using spike history
                    let mut total_stdp_change = 0.0;

                    if let Some(current_spike_time) = state.last_spike_time {
                        // Look for correlated spikes in recent history
                        for (past_time, _magnitude) in state.spike_history.iter() {
                            if *past_time != current_spike_time {
                                let stdp_change =
                                    self.compute_stdp_change(*past_time, current_spike_time);
                                total_stdp_change += stdp_change;
                            }
                        }
                    }

                    // Apply update with eligibility trace
                    let trace = self.eligibility_traces[&param_key].clone();
                    let update = trace.mul_scalar(self.lr * (1.0 + total_stdp_change as f32))?;

                    let param = &self.param_groups[i];
                    let mut param_write = param.write();
                    let new_param = param_write.sub(&update)?;

                    // Clamp weights to valid range
                    let clamped =
                        new_param.clamp(self.config.w_min as f32, self.config.w_max as f32)?;
                    *param_write = clamped;
                }
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        for param in &self.param_groups {
            param.write().set_grad(None);
        }
    }

    fn get_lr(&self) -> Vec<f32> {
        vec![self.lr]
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn add_param_group(
        &mut self,
        params: Vec<Arc<RwLock<Tensor>>>,
        _options: HashMap<String, f32>,
    ) {
        let start_idx = self.param_groups.len();
        self.param_groups.extend(params.iter().cloned());

        for (i, _param) in params.iter().enumerate() {
            let param_key = format!("param_{}", start_idx + i);
            self.spike_states
                .insert(param_key.clone(), SpikeState::default());
        }
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        // Create basic state
        let mut state = OptimizerState {
            optimizer_type: "STDP".to_string(),
            version: "1.0".to_string(),
            param_groups: vec![],
            state: HashMap::new(),
            global_state: HashMap::new(),
        };

        // Store configuration
        state.global_state.insert("lr".to_string(), self.lr);
        state
            .global_state
            .insert("current_time".to_string(), self.current_time as f32);
        state
            .global_state
            .insert("a_plus".to_string(), self.config.a_plus as f32);
        state
            .global_state
            .insert("a_minus".to_string(), self.config.a_minus as f32);

        Ok(state)
    }

    fn load_state_dict(&mut self, _state: OptimizerState) -> OptimizerResult<()> {
        // State loading implementation
        Ok(())
    }
}

// ============================================================================
// Event-Driven Optimizer
// ============================================================================

/// Event-driven optimization configuration
#[derive(Debug, Clone)]
pub struct EventDrivenConfig {
    /// Spike threshold for gradient magnitude
    pub spike_threshold: f64,
    /// Refractory period (steps to skip after spike)
    pub refractory_period: usize,
    /// Minimum time between updates (in steps)
    pub min_update_interval: usize,
    /// Use adaptive thresholding
    pub adaptive_threshold: bool,
    /// Threshold adaptation rate
    pub threshold_adapt_rate: f64,
}

impl Default for EventDrivenConfig {
    fn default() -> Self {
        Self {
            spike_threshold: 0.1,
            refractory_period: 5,
            min_update_interval: 1,
            adaptive_threshold: true,
            threshold_adapt_rate: 0.01,
        }
    }
}

/// Event-driven optimizer
///
/// Only updates parameters when gradient magnitude exceeds a threshold,
/// implementing sparse, event-driven computation.
pub struct EventDrivenOptimizer {
    /// Base learning rate
    lr: f32,
    /// Configuration
    config: EventDrivenConfig,
    /// Parameter groups
    param_groups: Vec<Arc<RwLock<Tensor>>>,
    /// Steps since last spike per parameter
    steps_since_spike: HashMap<String, usize>,
    /// Adaptive thresholds per parameter
    adaptive_thresholds: HashMap<String, f64>,
    /// Momentum buffers (for smoother updates)
    momentum_buffers: HashMap<String, Tensor>,
    /// Momentum coefficient
    momentum: f32,
}

impl EventDrivenOptimizer {
    /// Create a new event-driven optimizer
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: f32,
        momentum: f32,
        config: EventDrivenConfig,
    ) -> OptimizerResult<Self> {
        if lr <= 0.0 {
            return Err(OptimizerError::InvalidParameter(format!(
                "Invalid learning rate: {}",
                lr
            )));
        }

        let mut steps_since_spike = HashMap::new();
        let mut adaptive_thresholds = HashMap::new();
        let mut momentum_buffers = HashMap::new();

        for (i, param) in params.iter().enumerate() {
            let param_key = format!("param_{}", i);
            steps_since_spike.insert(param_key.clone(), 0);
            adaptive_thresholds.insert(param_key.clone(), config.spike_threshold);

            // Initialize momentum buffer
            let param_read = param.read();
            let shape_owned = param_read.shape().dims().to_vec();
            drop(param_read);
            let buffer = torsh_tensor::creation::zeros(&shape_owned)?;
            momentum_buffers.insert(param_key, buffer);
        }

        Ok(Self {
            lr,
            config,
            param_groups: params,
            steps_since_spike,
            adaptive_thresholds,
            momentum_buffers,
            momentum,
        })
    }

    /// Create with default configuration
    pub fn with_defaults(params: Vec<Arc<RwLock<Tensor>>>, lr: f32) -> OptimizerResult<Self> {
        Self::new(params, lr, 0.9, EventDrivenConfig::default())
    }

    /// Check if parameter should spike (update)
    fn should_spike(&mut self, param_key: &str, gradient: &Tensor) -> OptimizerResult<bool> {
        let steps = self
            .steps_since_spike
            .get(param_key)
            .expect("steps_since_spike should exist for param_key");

        // Check refractory period
        if *steps < self.config.refractory_period {
            return Ok(false);
        }

        // Check minimum update interval
        if *steps < self.config.min_update_interval {
            return Ok(false);
        }

        // Check gradient magnitude against threshold
        let grad_norm = gradient.norm()?.item()?;
        let grad_norm_f64 = grad_norm as f64;
        let threshold = self.adaptive_thresholds[param_key];

        let should_spike = grad_norm_f64 > threshold;

        // Adapt threshold if enabled
        if self.config.adaptive_threshold {
            let new_threshold = if should_spike {
                threshold * (1.0 + self.config.threshold_adapt_rate)
            } else {
                threshold * (1.0 - self.config.threshold_adapt_rate)
            };
            self.adaptive_thresholds
                .insert(param_key.to_string(), new_threshold.max(1e-6));
        }

        Ok(should_spike)
    }
}

impl Optimizer for EventDrivenOptimizer {
    fn step(&mut self) -> OptimizerResult<()> {
        // Increment all step counters
        for (_key, steps) in self.steps_since_spike.iter_mut() {
            *steps += 1;
        }

        // Collect gradients first to avoid borrow issues
        let mut gradients = Vec::new();
        for param in self.param_groups.iter() {
            let grad_opt = param.read().grad().clone();
            gradients.push(grad_opt);
        }

        for (i, grad_opt) in gradients.into_iter().enumerate() {
            let param_key = format!("param_{}", i);

            if let Some(grad) = grad_opt {
                // Check if this parameter should spike
                if self.should_spike(&param_key, &grad)? {
                    // Reset counter
                    self.steps_since_spike.insert(param_key.clone(), 0);

                    // Update momentum buffer
                    let buffer = self
                        .momentum_buffers
                        .get_mut(&param_key)
                        .expect("momentum_buffers should exist for param_key");
                    *buffer = buffer.mul_scalar(self.momentum)?;
                    *buffer = buffer.add(&grad)?;

                    // Apply update
                    let update = buffer.mul_scalar(self.lr)?;
                    let param = &self.param_groups[i];
                    let mut param_write = param.write();
                    *param_write = param_write.sub(&update)?;
                }
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        for param in &self.param_groups {
            param.write().set_grad(None);
        }
    }

    fn get_lr(&self) -> Vec<f32> {
        vec![self.lr]
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn add_param_group(
        &mut self,
        params: Vec<Arc<RwLock<Tensor>>>,
        _options: HashMap<String, f32>,
    ) {
        let start_idx = self.param_groups.len();
        self.param_groups.extend(params.iter().cloned());

        for (i, _param) in params.iter().enumerate() {
            let param_key = format!("param_{}", start_idx + i);
            self.steps_since_spike.insert(param_key.clone(), 0);
            self.adaptive_thresholds
                .insert(param_key.clone(), self.config.spike_threshold);
        }
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        let mut state = OptimizerState {
            optimizer_type: "EventDriven".to_string(),
            version: "1.0".to_string(),
            param_groups: vec![],
            state: HashMap::new(),
            global_state: HashMap::new(),
        };

        state.global_state.insert("lr".to_string(), self.lr);
        state
            .global_state
            .insert("momentum".to_string(), self.momentum);

        Ok(state)
    }

    fn load_state_dict(&mut self, _state: OptimizerState) -> OptimizerResult<()> {
        Ok(())
    }
}

// ============================================================================
// Temporal Credit Assignment
// ============================================================================

/// Temporal credit assignment configuration
#[derive(Debug, Clone)]
pub struct TemporalCreditConfig {
    /// Eligibility trace decay rate (λ)
    pub trace_decay: f64,
    /// Reward discount factor (γ)
    pub discount_factor: f64,
    /// Maximum trace history length
    pub max_trace_length: usize,
    /// Use three-factor learning rule (dopamine modulation)
    pub use_dopamine_modulation: bool,
    /// Baseline dopamine level
    pub baseline_dopamine: f64,
}

impl Default for TemporalCreditConfig {
    fn default() -> Self {
        Self {
            trace_decay: 0.95,
            discount_factor: 0.99,
            max_trace_length: 100,
            use_dopamine_modulation: true,
            baseline_dopamine: 1.0,
        }
    }
}

/// Temporal credit assignment optimizer
///
/// Implements eligibility traces for delayed reward learning,
/// allowing future rewards to influence past parameter updates.
pub struct TemporalCreditOptimizer {
    /// Base learning rate
    lr: f32,
    /// Configuration
    config: TemporalCreditConfig,
    /// Parameter groups
    param_groups: Vec<Arc<RwLock<Tensor>>>,
    /// Eligibility traces
    eligibility_traces: HashMap<String, Tensor>,
    /// Reward history
    reward_history: VecDeque<f64>,
    /// Current dopamine level (reward prediction error)
    dopamine_level: f64,
}

impl TemporalCreditOptimizer {
    /// Create a new temporal credit assignment optimizer
    pub fn new(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: f32,
        config: TemporalCreditConfig,
    ) -> OptimizerResult<Self> {
        if lr <= 0.0 {
            return Err(OptimizerError::InvalidParameter(format!(
                "Invalid learning rate: {}",
                lr
            )));
        }

        let mut eligibility_traces = HashMap::new();

        for (i, param) in params.iter().enumerate() {
            let param_key = format!("param_{}", i);
            let param_read = param.read();
            let shape_owned = param_read.shape().dims().to_vec();
            drop(param_read);
            let trace = torsh_tensor::creation::zeros(&shape_owned)?;
            eligibility_traces.insert(param_key, trace);
        }

        let max_trace_length = config.max_trace_length;
        let baseline_dopamine = config.baseline_dopamine;

        Ok(Self {
            lr,
            config,
            param_groups: params,
            eligibility_traces,
            reward_history: VecDeque::with_capacity(max_trace_length),
            dopamine_level: baseline_dopamine,
        })
    }

    /// Create with default configuration
    pub fn with_defaults(params: Vec<Arc<RwLock<Tensor>>>, lr: f32) -> OptimizerResult<Self> {
        Self::new(params, lr, TemporalCreditConfig::default())
    }

    /// Update eligibility traces
    fn update_traces(&mut self, gradients: &HashMap<String, Tensor>) -> OptimizerResult<()> {
        for (key, grad) in gradients {
            let trace = self
                .eligibility_traces
                .get_mut(key)
                .expect("eligibility_traces should exist for key");

            // e(t+1) = λ * γ * e(t) + ∇L
            let decay_factor = (self.config.trace_decay * self.config.discount_factor) as f32;
            *trace = trace.mul_scalar(decay_factor)?;
            *trace = trace.add(grad)?;
        }
        Ok(())
    }

    /// Update dopamine level (reward prediction error)
    pub fn update_dopamine(&mut self, reward: f64) {
        // Simple moving average of recent rewards
        self.reward_history.push_back(reward);
        if self.reward_history.len() > self.config.max_trace_length {
            self.reward_history.pop_front();
        }

        let avg_reward: f64 =
            self.reward_history.iter().sum::<f64>() / self.reward_history.len() as f64;

        // Dopamine = reward prediction error
        self.dopamine_level = reward - avg_reward + self.config.baseline_dopamine;
    }

    /// Step with reward signal
    pub fn step_with_reward(&mut self, reward: f64) -> OptimizerResult<()> {
        // Update dopamine level
        self.update_dopamine(reward);

        // Collect gradients
        let mut gradients = HashMap::new();
        for (i, param) in self.param_groups.iter().enumerate() {
            let param_key = format!("param_{}", i);
            let param_read = param.read();
            if let Some(grad) = param_read.grad() {
                gradients.insert(param_key, grad.clone());
            }
        }

        // Update eligibility traces
        self.update_traces(&gradients)?;

        // Apply updates using eligibility traces and dopamine modulation
        for (i, param) in self.param_groups.iter().enumerate() {
            let param_key = format!("param_{}", i);
            let trace = &self.eligibility_traces[&param_key];

            // Three-factor learning rule: Δw = lr * dopamine * eligibility_trace
            let modulation = if self.config.use_dopamine_modulation {
                self.dopamine_level as f32
            } else {
                1.0
            };

            let update = trace.mul_scalar(self.lr * modulation)?;
            let mut param_write = param.write();
            *param_write = param_write.sub(&update)?;
        }

        Ok(())
    }
}

impl Optimizer for TemporalCreditOptimizer {
    fn step(&mut self) -> OptimizerResult<()> {
        // Default step with zero reward
        self.step_with_reward(0.0)
    }

    fn zero_grad(&mut self) {
        for param in &self.param_groups {
            param.write().set_grad(None);
        }
    }

    fn get_lr(&self) -> Vec<f32> {
        vec![self.lr]
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn add_param_group(
        &mut self,
        params: Vec<Arc<RwLock<Tensor>>>,
        _options: HashMap<String, f32>,
    ) {
        let start_idx = self.param_groups.len();
        self.param_groups.extend(params.iter().cloned());

        for (i, _param) in params.iter().enumerate() {
            let param_key = format!("param_{}", start_idx + i);
            // Would need to initialize eligibility trace here
        }
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        let mut state = OptimizerState {
            optimizer_type: "TemporalCredit".to_string(),
            version: "1.0".to_string(),
            param_groups: vec![],
            state: HashMap::new(),
            global_state: HashMap::new(),
        };

        state.global_state.insert("lr".to_string(), self.lr);
        state
            .global_state
            .insert("dopamine_level".to_string(), self.dopamine_level as f32);

        Ok(state)
    }

    fn load_state_dict(&mut self, _state: OptimizerState) -> OptimizerResult<()> {
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_stdp_optimizer_creation() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[10, 10])?));
        let optimizer = STDPOptimizer::with_defaults(vec![param], 0.01)?;

        assert_eq!(optimizer.param_groups.len(), 1);
        assert_eq!(optimizer.spike_states.len(), 1);
        Ok(())
    }

    #[test]
    fn test_stdp_config_default() {
        let config = STDPConfig::default();
        assert_eq!(config.a_plus, 0.01);
        assert_eq!(config.a_minus, 0.01);
        assert_eq!(config.tau_plus, 20.0);
        assert!(config.w_max > config.w_min);
    }

    #[test]
    fn test_event_driven_optimizer_creation() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[5, 5])?));
        let optimizer = EventDrivenOptimizer::with_defaults(vec![param], 0.01)?;

        assert_eq!(optimizer.param_groups.len(), 1);
        assert_eq!(optimizer.steps_since_spike.len(), 1);
        Ok(())
    }

    #[test]
    fn test_event_driven_config_default() {
        let config = EventDrivenConfig::default();
        assert_eq!(config.spike_threshold, 0.1);
        assert_eq!(config.refractory_period, 5);
        assert!(config.adaptive_threshold);
    }

    #[test]
    fn test_temporal_credit_optimizer_creation() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[3, 3])?));
        let optimizer = TemporalCreditOptimizer::with_defaults(vec![param], 0.01)?;

        assert_eq!(optimizer.param_groups.len(), 1);
        assert_eq!(optimizer.eligibility_traces.len(), 1);
        Ok(())
    }

    #[test]
    fn test_temporal_credit_config_default() {
        let config = TemporalCreditConfig::default();
        assert_eq!(config.trace_decay, 0.95);
        assert_eq!(config.discount_factor, 0.99);
        assert!(config.use_dopamine_modulation);
    }

    #[test]
    fn test_stdp_step() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[2, 2])?));

        // Set gradient
        {
            let mut p = param.write();
            let grad = randn::<f32>(&[2, 2])?;
            p.set_grad(Some(grad));
        }

        let mut optimizer = STDPOptimizer::with_defaults(vec![param.clone()], 0.01)?;

        // Step should succeed
        optimizer.step()?;

        Ok(())
    }

    #[test]
    fn test_event_driven_step() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[2, 2])?));

        // Set gradient
        {
            let mut p = param.write();
            let grad = randn::<f32>(&[2, 2])?;
            p.set_grad(Some(grad));
        }

        let mut optimizer = EventDrivenOptimizer::with_defaults(vec![param.clone()], 0.01)?;

        // Step should succeed
        optimizer.step()?;

        Ok(())
    }

    #[test]
    fn test_temporal_credit_step_with_reward() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[2, 2])?));

        // Set gradient
        {
            let mut p = param.write();
            let grad = randn::<f32>(&[2, 2])?;
            p.set_grad(Some(grad));
        }

        let mut optimizer = TemporalCreditOptimizer::with_defaults(vec![param.clone()], 0.01)?;

        // Step with reward
        optimizer.step_with_reward(1.0)?;

        // Dopamine level should be updated
        assert!(optimizer.dopamine_level > 0.0);

        Ok(())
    }

    #[test]
    fn test_zero_grad() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[2, 2])?));

        // Set gradient
        {
            let mut p = param.write();
            let grad = randn::<f32>(&[2, 2])?;
            p.set_grad(Some(grad));
        }

        let mut optimizer = STDPOptimizer::with_defaults(vec![param.clone()], 0.01)?;

        // Zero gradients
        optimizer.zero_grad();

        // Check gradient is None
        let p = param.read();
        assert!(p.grad().is_none());

        Ok(())
    }
}
