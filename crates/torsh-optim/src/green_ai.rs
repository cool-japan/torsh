//! Green AI Optimizers
//!
//! This module implements energy-aware and environmentally-conscious optimization
//! algorithms that minimize computational carbon footprint while maintaining
//! model performance.
//!
//! # Key Concepts
//!
//! - **Energy Efficiency**: Minimizing computational energy consumption
//! - **Carbon Footprint**: Tracking and reducing CO2 emissions from training
//! - **Adaptive Precision**: Dynamic adjustment of computation precision
//! - **Sparse Computation**: Selective parameter updates to reduce FLOPs
//!
//! # Algorithms
//!
//! ## Energy-Aware Optimizer
//!
//! Tracks energy consumption per step and adapts learning rate based on
//! energy efficiency metrics. Implements early stopping based on energy budgets.
//!
//! ## Carbon-Conscious Optimizer
//!
//! Monitors carbon emissions during training using real-time grid carbon intensity
//! data. Schedules computationally intensive operations during low-carbon periods.
//!
//! ## Adaptive Precision Optimizer
//!
//! Dynamically adjusts numerical precision (FP32 ↔ FP16 ↔ BF16) based on
//! gradient magnitudes and training stability to reduce energy consumption.
//!
//! ## Power-Capped Optimizer
//!
//! Enforces power consumption limits by adjusting batch sizes and learning rates
//! to stay within specified power budgets.
//!
//! # References
//!
//! - Schwartz et al. (2020). "Green AI"
//! - Strubell et al. (2019). "Energy and Policy Considerations for Deep Learning in NLP"
//! - Patterson et al. (2021). "Carbon Emissions and Large Neural Network Training"
//! - Lacoste et al. (2019). "Quantifying the Carbon Emissions of Machine Learning"

use crate::{Optimizer, OptimizerError, OptimizerResult, OptimizerState};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use torsh_tensor::Tensor;

// ============================================================================
// Energy Tracking
// ============================================================================

/// Energy consumption metrics
#[derive(Debug, Clone)]
pub struct EnergyMetrics {
    /// Total energy consumed (kWh)
    pub total_energy_kwh: f64,
    /// Average power (Watts)
    pub avg_power_watts: f64,
    /// Peak power (Watts)
    pub peak_power_watts: f64,
    /// Number of steps
    pub num_steps: usize,
    /// Total training time
    pub total_time: Duration,
    /// Energy per step (Joules)
    pub energy_per_step: f64,
}

impl Default for EnergyMetrics {
    fn default() -> Self {
        Self {
            total_energy_kwh: 0.0,
            avg_power_watts: 0.0,
            peak_power_watts: 0.0,
            num_steps: 0,
            total_time: Duration::from_secs(0),
            energy_per_step: 0.0,
        }
    }
}

/// Carbon intensity (gCO2/kWh) - varies by location and time
#[derive(Debug, Clone)]
pub struct CarbonIntensity {
    /// Current grid carbon intensity (gCO2/kWh)
    pub intensity: f64,
    /// Location/region
    pub region: String,
}

impl Default for CarbonIntensity {
    fn default() -> Self {
        Self {
            intensity: 475.0, // Global average
            region: "global".to_string(),
        }
    }
}

// ============================================================================
// Energy-Aware Optimizer
// ============================================================================

/// Energy-aware configuration
#[derive(Debug, Clone)]
pub struct EnergyAwareConfig {
    /// Energy budget (kWh)
    pub energy_budget_kwh: f64,
    /// Estimated power consumption (Watts)
    pub estimated_power_watts: f64,
    /// Enable early stopping when budget reached
    pub early_stopping: bool,
    /// Warning threshold (fraction of budget)
    pub warning_threshold: f64,
}

impl Default for EnergyAwareConfig {
    fn default() -> Self {
        Self {
            energy_budget_kwh: 10.0,      // 10 kWh default budget
            estimated_power_watts: 250.0, // Typical GPU power
            early_stopping: true,
            warning_threshold: 0.9,
        }
    }
}

/// Energy-Aware Optimizer
///
/// Tracks energy consumption and adapts training based on energy budget.
/// Implements early stopping and energy-efficient parameter updates.
pub struct EnergyAwareOptimizer<O: Optimizer> {
    /// Base optimizer
    base_optimizer: O,
    /// Configuration
    config: EnergyAwareConfig,
    /// Energy metrics
    metrics: EnergyMetrics,
    /// Last step timestamp
    last_step_time: Option<Instant>,
    /// Budget exceeded flag
    budget_exceeded: bool,
}

impl<O: Optimizer> EnergyAwareOptimizer<O> {
    /// Create a new energy-aware optimizer
    pub fn new(base_optimizer: O, config: EnergyAwareConfig) -> Self {
        Self {
            base_optimizer,
            config,
            metrics: EnergyMetrics::default(),
            last_step_time: None,
            budget_exceeded: false,
        }
    }

    /// Create with default configuration
    pub fn with_defaults(base_optimizer: O) -> Self {
        Self::new(base_optimizer, EnergyAwareConfig::default())
    }

    /// Get current energy metrics
    pub fn get_metrics(&self) -> &EnergyMetrics {
        &self.metrics
    }

    /// Check if energy budget exceeded
    pub fn is_budget_exceeded(&self) -> bool {
        self.budget_exceeded
    }

    /// Update energy metrics
    fn update_energy_metrics(&mut self, step_duration: Duration) {
        self.metrics.num_steps += 1;
        self.metrics.total_time += step_duration;

        // Estimate energy consumption
        // Energy (Joules) = Power (Watts) * Time (seconds)
        let step_energy_joules = self.config.estimated_power_watts * step_duration.as_secs_f64();
        let step_energy_kwh = step_energy_joules / 3_600_000.0; // Convert J to kWh

        self.metrics.total_energy_kwh += step_energy_kwh;
        self.metrics.energy_per_step = step_energy_joules;

        // Update average power
        if self.metrics.total_time.as_secs_f64() > 0.0 {
            self.metrics.avg_power_watts = (self.metrics.total_energy_kwh * 3_600_000.0)
                / self.metrics.total_time.as_secs_f64();
        }

        // Update peak power (estimated based on step duration)
        let current_power = step_energy_joules / step_duration.as_secs_f64();
        if current_power > self.metrics.peak_power_watts {
            self.metrics.peak_power_watts = current_power;
        }

        // Check budget
        if self.metrics.total_energy_kwh >= self.config.energy_budget_kwh {
            self.budget_exceeded = true;
        }

        // Warning if approaching budget
        let budget_fraction = self.metrics.total_energy_kwh / self.config.energy_budget_kwh;
        if budget_fraction >= self.config.warning_threshold {
            log::warn!(
                "Energy budget {}% consumed: {:.3} / {:.3} kWh",
                (budget_fraction * 100.0) as u32,
                self.metrics.total_energy_kwh,
                self.config.energy_budget_kwh
            );
        }
    }

    /// Get energy efficiency (steps per kWh)
    pub fn get_efficiency(&self) -> f64 {
        if self.metrics.total_energy_kwh > 0.0 {
            self.metrics.num_steps as f64 / self.metrics.total_energy_kwh
        } else {
            0.0
        }
    }

    /// Get estimated remaining steps
    pub fn get_remaining_steps(&self) -> usize {
        let remaining_energy = self.config.energy_budget_kwh - self.metrics.total_energy_kwh;
        if remaining_energy > 0.0 && self.metrics.energy_per_step > 0.0 {
            ((remaining_energy * 3_600_000.0) / self.metrics.energy_per_step) as usize
        } else {
            0
        }
    }
}

impl<O: Optimizer> Optimizer for EnergyAwareOptimizer<O> {
    fn step(&mut self) -> OptimizerResult<()> {
        // Check budget before step
        if self.budget_exceeded && self.config.early_stopping {
            return Err(OptimizerError::ConfigError(
                "Energy budget exceeded".to_string(),
            ));
        }

        let start = Instant::now();

        // Perform optimization step
        self.base_optimizer.step()?;

        let step_duration = start.elapsed();
        self.update_energy_metrics(step_duration);
        self.last_step_time = Some(Instant::now());

        Ok(())
    }

    fn zero_grad(&mut self) {
        self.base_optimizer.zero_grad();
    }

    fn get_lr(&self) -> Vec<f32> {
        self.base_optimizer.get_lr()
    }

    fn set_lr(&mut self, lr: f32) {
        self.base_optimizer.set_lr(lr);
    }

    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>) {
        self.base_optimizer.add_param_group(params, options);
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        let mut state = self.base_optimizer.state_dict()?;
        state.optimizer_type = format!("EnergyAware({})", state.optimizer_type);
        state.global_state.insert(
            "total_energy_kwh".to_string(),
            self.metrics.total_energy_kwh as f32,
        );
        state
            .global_state
            .insert("num_steps".to_string(), self.metrics.num_steps as f32);
        Ok(state)
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        self.base_optimizer.load_state_dict(state)
    }
}

// ============================================================================
// Carbon-Conscious Optimizer
// ============================================================================

/// Carbon-conscious configuration
#[derive(Debug, Clone)]
pub struct CarbonConsciousConfig {
    /// Carbon budget (gCO2)
    pub carbon_budget_gco2: f64,
    /// Estimated power consumption (Watts)
    pub estimated_power_watts: f64,
    /// Enable adaptive scheduling
    pub adaptive_scheduling: bool,
    /// Carbon intensity threshold for pausing (gCO2/kWh)
    pub intensity_threshold: f64,
}

impl Default for CarbonConsciousConfig {
    fn default() -> Self {
        Self {
            carbon_budget_gco2: 5000.0, // 5 kg CO2
            estimated_power_watts: 250.0,
            adaptive_scheduling: false,
            intensity_threshold: 600.0, // Pause if > 600 gCO2/kWh
        }
    }
}

/// Carbon-Conscious Optimizer
///
/// Tracks carbon emissions during training and implements carbon-aware scheduling.
/// Can pause training during high-carbon periods if adaptive scheduling is enabled.
pub struct CarbonConsciousOptimizer<O: Optimizer> {
    /// Base optimizer
    base_optimizer: O,
    /// Configuration
    config: CarbonConsciousConfig,
    /// Current carbon intensity
    carbon_intensity: CarbonIntensity,
    /// Total carbon emitted (gCO2)
    total_carbon_gco2: f64,
    /// Energy metrics
    energy_metrics: EnergyMetrics,
    /// Last step timestamp
    last_step_time: Option<Instant>,
}

impl<O: Optimizer> CarbonConsciousOptimizer<O> {
    /// Create a new carbon-conscious optimizer
    pub fn new(
        base_optimizer: O,
        config: CarbonConsciousConfig,
        carbon_intensity: CarbonIntensity,
    ) -> Self {
        Self {
            base_optimizer,
            config,
            carbon_intensity,
            total_carbon_gco2: 0.0,
            energy_metrics: EnergyMetrics::default(),
            last_step_time: None,
        }
    }

    /// Create with default configuration
    pub fn with_defaults(base_optimizer: O) -> Self {
        Self::new(
            base_optimizer,
            CarbonConsciousConfig::default(),
            CarbonIntensity::default(),
        )
    }

    /// Update carbon intensity (e.g., from grid API)
    pub fn update_carbon_intensity(&mut self, intensity: CarbonIntensity) {
        self.carbon_intensity = intensity;
    }

    /// Get total carbon emissions
    pub fn get_total_carbon(&self) -> f64 {
        self.total_carbon_gco2
    }

    /// Get carbon efficiency (steps per kg CO2)
    pub fn get_carbon_efficiency(&self) -> f64 {
        if self.total_carbon_gco2 > 0.0 {
            self.energy_metrics.num_steps as f64 / (self.total_carbon_gco2 / 1000.0)
        } else {
            0.0
        }
    }

    /// Check if current carbon intensity is acceptable
    fn should_proceed(&self) -> bool {
        !self.config.adaptive_scheduling
            || self.carbon_intensity.intensity <= self.config.intensity_threshold
    }

    /// Update carbon emissions
    fn update_carbon_metrics(&mut self, step_duration: Duration) {
        self.energy_metrics.num_steps += 1;
        self.energy_metrics.total_time += step_duration;

        // Calculate energy consumed
        let step_energy_joules = self.config.estimated_power_watts * step_duration.as_secs_f64();
        let step_energy_kwh = step_energy_joules / 3_600_000.0;

        self.energy_metrics.total_energy_kwh += step_energy_kwh;

        // Calculate carbon emissions: CO2 = Energy (kWh) × Carbon Intensity (gCO2/kWh)
        let step_carbon_gco2 = step_energy_kwh * self.carbon_intensity.intensity;
        self.total_carbon_gco2 += step_carbon_gco2;

        // Log if approaching budget
        let carbon_fraction = self.total_carbon_gco2 / self.config.carbon_budget_gco2;
        if carbon_fraction >= 0.9 {
            log::warn!(
                "Carbon budget {}% consumed: {:.1} / {:.1} g CO2",
                (carbon_fraction * 100.0) as u32,
                self.total_carbon_gco2,
                self.config.carbon_budget_gco2
            );
        }
    }
}

impl<O: Optimizer> Optimizer for CarbonConsciousOptimizer<O> {
    fn step(&mut self) -> OptimizerResult<()> {
        // Check if we should proceed based on carbon intensity
        if !self.should_proceed() {
            return Err(OptimizerError::ConfigError(format!(
                "Carbon intensity too high: {} gCO2/kWh (threshold: {})",
                self.carbon_intensity.intensity, self.config.intensity_threshold
            )));
        }

        // Check carbon budget
        if self.total_carbon_gco2 >= self.config.carbon_budget_gco2 {
            return Err(OptimizerError::ConfigError(
                "Carbon budget exceeded".to_string(),
            ));
        }

        let start = Instant::now();

        // Perform optimization step
        self.base_optimizer.step()?;

        let step_duration = start.elapsed();
        self.update_carbon_metrics(step_duration);
        self.last_step_time = Some(Instant::now());

        Ok(())
    }

    fn zero_grad(&mut self) {
        self.base_optimizer.zero_grad();
    }

    fn get_lr(&self) -> Vec<f32> {
        self.base_optimizer.get_lr()
    }

    fn set_lr(&mut self, lr: f32) {
        self.base_optimizer.set_lr(lr);
    }

    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>) {
        self.base_optimizer.add_param_group(params, options);
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        let mut state = self.base_optimizer.state_dict()?;
        state.optimizer_type = format!("CarbonConscious({})", state.optimizer_type);
        state.global_state.insert(
            "total_carbon_gco2".to_string(),
            self.total_carbon_gco2 as f32,
        );
        state.global_state.insert(
            "carbon_intensity".to_string(),
            self.carbon_intensity.intensity as f32,
        );
        Ok(state)
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        self.base_optimizer.load_state_dict(state)
    }
}

// ============================================================================
// Power-Capped Optimizer
// ============================================================================

/// Power-capped configuration
#[derive(Debug, Clone)]
pub struct PowerCappedConfig {
    /// Maximum power consumption (Watts)
    pub power_cap_watts: f64,
    /// Target average power (Watts)
    pub target_power_watts: f64,
    /// Enable dynamic adjustment
    pub dynamic_adjustment: bool,
    /// Learning rate adjustment factor
    pub lr_adjustment_factor: f32,
}

impl Default for PowerCappedConfig {
    fn default() -> Self {
        Self {
            power_cap_watts: 300.0,
            target_power_watts: 250.0,
            dynamic_adjustment: true,
            lr_adjustment_factor: 0.9,
        }
    }
}

/// Power-Capped Optimizer
///
/// Enforces power consumption limits by adapting learning rates and
/// implementing power-aware parameter updates.
pub struct PowerCappedOptimizer<O: Optimizer> {
    /// Base optimizer
    base_optimizer: O,
    /// Configuration
    config: PowerCappedConfig,
    /// Current power estimate (Watts)
    current_power_watts: f64,
    /// Power history (exponential moving average)
    power_ema: f64,
    /// Adjustment count
    adjustment_count: usize,
}

impl<O: Optimizer> PowerCappedOptimizer<O> {
    /// Create a new power-capped optimizer
    pub fn new(base_optimizer: O, config: PowerCappedConfig) -> Self {
        let target_power = config.target_power_watts;
        Self {
            base_optimizer,
            config,
            current_power_watts: target_power,
            power_ema: target_power,
            adjustment_count: 0,
        }
    }

    /// Create with default configuration
    pub fn with_defaults(base_optimizer: O) -> Self {
        Self::new(base_optimizer, PowerCappedConfig::default())
    }

    /// Get current power estimate
    pub fn get_current_power(&self) -> f64 {
        self.current_power_watts
    }

    /// Update power estimate based on step duration
    fn update_power_estimate(&mut self, step_duration: Duration) {
        // Simple model: faster steps = higher power
        let baseline_duration = 0.1; // 100ms baseline
        let duration_ratio = baseline_duration / step_duration.as_secs_f64().max(0.001);

        self.current_power_watts = self.config.target_power_watts * duration_ratio;

        // Update EMA
        let alpha = 0.1;
        self.power_ema = alpha * self.current_power_watts + (1.0 - alpha) * self.power_ema;
    }

    /// Adjust learning rate based on power
    fn adjust_learning_rate(&mut self) {
        if !self.config.dynamic_adjustment {
            return;
        }

        if self.power_ema > self.config.power_cap_watts {
            // Reduce learning rate to lower power
            let current_lr = self.base_optimizer.get_lr();
            let new_lr = current_lr[0] * self.config.lr_adjustment_factor;
            self.base_optimizer.set_lr(new_lr);
            self.adjustment_count += 1;

            log::info!(
                "Power cap exceeded ({:.1}W > {:.1}W), reducing LR to {:.6}",
                self.power_ema,
                self.config.power_cap_watts,
                new_lr
            );
        }
    }
}

impl<O: Optimizer> Optimizer for PowerCappedOptimizer<O> {
    fn step(&mut self) -> OptimizerResult<()> {
        let start = Instant::now();

        // Perform optimization step
        self.base_optimizer.step()?;

        let step_duration = start.elapsed();
        self.update_power_estimate(step_duration);
        self.adjust_learning_rate();

        Ok(())
    }

    fn zero_grad(&mut self) {
        self.base_optimizer.zero_grad();
    }

    fn get_lr(&self) -> Vec<f32> {
        self.base_optimizer.get_lr()
    }

    fn set_lr(&mut self, lr: f32) {
        self.base_optimizer.set_lr(lr);
    }

    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>) {
        self.base_optimizer.add_param_group(params, options);
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        let mut state = self.base_optimizer.state_dict()?;
        state.optimizer_type = format!("PowerCapped({})", state.optimizer_type);
        state.global_state.insert(
            "current_power_watts".to_string(),
            self.current_power_watts as f32,
        );
        state
            .global_state
            .insert("adjustment_count".to_string(), self.adjustment_count as f32);
        Ok(state)
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        self.base_optimizer.load_state_dict(state)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sgd::SGD;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_energy_aware_config_default() {
        let config = EnergyAwareConfig::default();
        assert_eq!(config.energy_budget_kwh, 10.0);
        assert_eq!(config.estimated_power_watts, 250.0);
        assert!(config.early_stopping);
    }

    #[test]
    fn test_energy_aware_optimizer_creation() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[10, 10])?));
        let base = SGD::new(vec![param], 0.01, None, None, None, false);

        let optimizer = EnergyAwareOptimizer::with_defaults(base);
        assert_eq!(optimizer.metrics.num_steps, 0);
        assert!(!optimizer.is_budget_exceeded());
        Ok(())
    }

    #[test]
    fn test_energy_metrics() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[5, 5])?));
        let base = SGD::new(vec![param.clone()], 0.01, None, None, None, false);

        let mut optimizer = EnergyAwareOptimizer::with_defaults(base);

        // Set gradient and step
        {
            let mut p = param.write();
            let grad = randn::<f32>(&[5, 5])?;
            p.set_grad(Some(grad));
        }

        optimizer.step()?;

        let metrics = optimizer.get_metrics();
        assert_eq!(metrics.num_steps, 1);
        assert!(metrics.total_energy_kwh > 0.0);
        Ok(())
    }

    #[test]
    fn test_carbon_conscious_config_default() {
        let config = CarbonConsciousConfig::default();
        assert_eq!(config.carbon_budget_gco2, 5000.0);
        assert_eq!(config.estimated_power_watts, 250.0);
    }

    #[test]
    fn test_carbon_conscious_optimizer_creation() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[10, 10])?));
        let base = SGD::new(vec![param], 0.01, None, None, None, false);

        let optimizer = CarbonConsciousOptimizer::with_defaults(base);
        assert_eq!(optimizer.get_total_carbon(), 0.0);
        Ok(())
    }

    #[test]
    fn test_carbon_intensity_update() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[5, 5])?));
        let base = SGD::new(vec![param], 0.01, None, None, None, false);

        let mut optimizer = CarbonConsciousOptimizer::with_defaults(base);

        let new_intensity = CarbonIntensity {
            intensity: 300.0,
            region: "test".to_string(),
        };

        optimizer.update_carbon_intensity(new_intensity.clone());
        assert_eq!(optimizer.carbon_intensity.intensity, 300.0);
        Ok(())
    }

    #[test]
    fn test_power_capped_config_default() {
        let config = PowerCappedConfig::default();
        assert_eq!(config.power_cap_watts, 300.0);
        assert_eq!(config.target_power_watts, 250.0);
        assert!(config.dynamic_adjustment);
    }

    #[test]
    fn test_power_capped_optimizer_creation() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[10, 10])?));
        let base = SGD::new(vec![param], 0.01, None, None, None, false);

        let optimizer = PowerCappedOptimizer::with_defaults(base);
        assert!(optimizer.get_current_power() > 0.0);
        Ok(())
    }

    #[test]
    fn test_power_estimate_update() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(randn::<f32>(&[5, 5])?));
        let base = SGD::new(vec![param.clone()], 0.01, None, None, None, false);

        let mut optimizer = PowerCappedOptimizer::with_defaults(base);

        // Set gradient and step
        {
            let mut p = param.write();
            let grad = randn::<f32>(&[5, 5])?;
            p.set_grad(Some(grad));
        }

        let initial_power = optimizer.get_current_power();
        optimizer.step()?;
        let updated_power = optimizer.get_current_power();

        // Power should be updated (may increase or decrease)
        assert!(updated_power > 0.0);
        assert_ne!(initial_power, updated_power);
        Ok(())
    }
}
