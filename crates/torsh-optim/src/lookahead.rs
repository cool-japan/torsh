//! Lookahead optimizer implementation
//!
//! Lookahead is a meta-optimizer that can be wrapped around any base optimizer
//! to reduce variance and provide more stable convergence. It maintains a set
//! of "slow weights" that are updated less frequently based on the trajectory
//! of the "fast weights" optimized by the base optimizer.
//!
//! Reference: "Lookahead Optimizer: k steps forward, 1 step back"
//! <https://arxiv.org/abs/1907.08610>

use crate::{Optimizer, OptimizerError, OptimizerResult, OptimizerState, ParamGroupState};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_core::error::Result;
use torsh_tensor::Tensor;

/// Lookahead optimizer wrapper
///
/// Wraps any base optimizer to provide more stable training by maintaining
/// slow weights that are updated based on the fast weight trajectory.
pub struct Lookahead<O: Optimizer> {
    /// Base optimizer (fast weights)
    base_optimizer: O,
    /// Slow weights that are updated less frequently
    slow_weights: HashMap<String, Tensor>,
    /// Step size for slow weight updates (α in the paper)
    alpha: f32,
    /// Number of fast weight updates before slow weight update (k in the paper)
    k: usize,
    /// Current step count for tracking fast weight updates
    step_count: usize,
}

impl<O: Optimizer> Lookahead<O> {
    /// Create a new Lookahead optimizer
    pub fn new(base_optimizer: O, alpha: f32, k: usize) -> Self {
        Self {
            base_optimizer,
            slow_weights: HashMap::new(),
            alpha,
            k,
            step_count: 0,
        }
    }

    /// Create Lookahead with default hyperparameters
    pub fn with_defaults(base_optimizer: O) -> Self {
        Self::new(base_optimizer, 0.5, 5)
    }

    /// Get parameter key for slow weight storage.
    ///
    /// Keyed by the `Arc<RwLock<Tensor>>` handle identity, NOT the tensor's data
    /// buffer pointer. Base optimizers update parameters by *replacing* the inner
    /// tensor (`*param = param.sub(..)`), which allocates a new data buffer each
    /// step; a data-pointer key would therefore change between initialization and
    /// the slow-weight update, silently losing the mapping. The `Arc` handle, by
    /// contrast, is stable for the lifetime of the parameter.
    fn get_param_key(param_ref: &Arc<RwLock<Tensor>>) -> String {
        format!("param_{:p}", Arc::as_ptr(param_ref))
    }

    /// Initialize slow weights if not already done
    fn initialize_slow_weights(&mut self, params: &[Arc<RwLock<Tensor>>]) -> Result<()> {
        for param_ref in params {
            let param_key = Self::get_param_key(param_ref);

            if !self.slow_weights.contains_key(&param_key) {
                // Initialize slow weights as a copy of current parameters
                let param = param_ref.read();
                self.slow_weights.insert(param_key, param.clone());
            }
        }
        Ok(())
    }

    /// Update slow weights based on fast weight trajectory
    fn update_slow_weights(&mut self, params: &[Arc<RwLock<Tensor>>]) -> Result<()> {
        for param_ref in params {
            let param_key = Self::get_param_key(param_ref);

            if let Some(slow_weight) = self.slow_weights.get_mut(&param_key) {
                let param = param_ref.read();
                // Lookahead update: φ_{t+1} = φ_t + α(θ_{t+k} - φ_t)
                // where φ is slow weights, θ is fast weights
                let diff = param.sub(slow_weight)?;
                let update = diff.mul_scalar(self.alpha)?;
                *slow_weight = slow_weight.add(&update)?;

                // Update fast weights to slow weights
                drop(param); // Release read lock
                let mut param_mut = param_ref.write();
                *param_mut = slow_weight.clone();
            }
        }
        Ok(())
    }

    /// Get a reference to the base optimizer
    pub fn base_optimizer(&self) -> &O {
        &self.base_optimizer
    }

    /// Get a mutable reference to the base optimizer
    pub fn base_optimizer_mut(&mut self) -> &mut O {
        &mut self.base_optimizer
    }

    /// Get the current alpha value
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Set the alpha value
    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = alpha;
    }

    /// Get the current k value
    pub fn k(&self) -> usize {
        self.k
    }

    /// Set the k value
    pub fn set_k(&mut self, k: usize) {
        self.k = k;
    }

    /// Get the current step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }
}

impl<O: Optimizer> Optimizer for Lookahead<O> {
    fn step(&mut self) -> OptimizerResult<()> {
        // Get the parameter handles from the base optimizer. These are shared
        // `Arc<RwLock<Tensor>>` clones pointing at the same tensors the base
        // optimizer updates, so reading them after `base_optimizer.step()` sees
        // the freshly-updated fast weights.
        let params = self.base_optimizer.parameters();

        // Lookahead is meaningless without access to the wrapped optimizer's
        // parameters: the slow-weight trajectory could never be tracked. An
        // empty list means the base optimizer does not expose its parameters,
        // so fail loudly instead of silently degrading into the base optimizer.
        if params.is_empty() {
            return Err(OptimizerError::StateError(
                "Lookahead requires access to the base optimizer's parameters, but \
                 `Optimizer::parameters()` returned an empty list. The wrapped optimizer \
                 must implement `parameters()` to expose its parameter tensors."
                    .to_string(),
            ));
        }

        // Initialize slow weights if this is the first step
        if self.slow_weights.is_empty() {
            self.initialize_slow_weights(&params)?;
        }

        // Perform base optimizer step (fast weights update)
        self.base_optimizer.step()?;
        self.step_count += 1;

        // Every k steps, update slow weights and reset fast weights
        if self.step_count % self.k == 0 {
            self.update_slow_weights(&params)?;
        }

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

    fn parameters(&self) -> Vec<Arc<RwLock<Tensor>>> {
        self.base_optimizer.parameters()
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        let mut base_state = self.base_optimizer.state_dict()?;

        // Add slow weights to state
        for (key, tensor) in &self.slow_weights {
            let mut param_state = HashMap::new();
            param_state.insert("slow_weight".to_string(), tensor.clone());
            base_state
                .state
                .insert(format!("lookahead_{}", key), param_state);
        }

        // Add Lookahead-specific state
        let mut lookahead_options = HashMap::new();
        lookahead_options.insert("alpha".to_string(), self.alpha);
        lookahead_options.insert("k".to_string(), self.k as f32);
        lookahead_options.insert("step_count".to_string(), self.step_count as f32);

        let lookahead_group = ParamGroupState {
            lr: 0.0, // Not applicable for meta-optimizer
            options: lookahead_options,
            param_count: 0,
        };

        base_state.param_groups.push(lookahead_group);
        Ok(base_state)
    }

    fn load_state_dict(&mut self, mut state: OptimizerState) -> OptimizerResult<()> {
        // Extract Lookahead-specific state
        if let Some(lookahead_group) = state.param_groups.pop() {
            if let Some(&alpha) = lookahead_group.options.get("alpha") {
                self.alpha = alpha;
            }
            if let Some(&k) = lookahead_group.options.get("k") {
                self.k = k as usize;
            }
            if let Some(&step_count) = lookahead_group.options.get("step_count") {
                self.step_count = step_count as usize;
            }
        }

        // Extract slow weights
        self.slow_weights.clear();
        let mut base_state_entries = HashMap::new();

        for (key, param_state) in state.state {
            if key.starts_with("lookahead_") {
                if let Some(tensor) = param_state.get("slow_weight") {
                    let param_key = key
                        .strip_prefix("lookahead_")
                        .expect("prefix should exist after starts_with check")
                        .to_string();
                    self.slow_weights.insert(param_key, tensor.clone());
                }
            } else {
                base_state_entries.insert(key, param_state);
            }
        }

        // Restore base optimizer state
        let base_state = OptimizerState {
            param_groups: state.param_groups,
            state: base_state_entries,
            global_state: state.global_state,
            optimizer_type: state.optimizer_type,
            version: state.version,
        };

        self.base_optimizer.load_state_dict(base_state)
    }
}

/// Convenience function to create Lookahead with Adam
pub fn lookahead_adam(params: Vec<Arc<RwLock<Tensor>>>, lr: f32) -> Lookahead<crate::adam::Adam> {
    let adam = crate::adam::Adam::new(params, Some(lr), None, None, None, false);
    Lookahead::with_defaults(adam)
}

/// Convenience function to create Lookahead with SGD
pub fn lookahead_sgd(params: Vec<Arc<RwLock<Tensor>>>, lr: f32) -> Lookahead<crate::sgd::SGD> {
    let sgd = crate::sgd::SGD::new(params, lr, None, None, None, false);
    Lookahead::with_defaults(sgd)
}

/// Convenience function to create Lookahead with RAdam
pub fn lookahead_radam(
    params: Vec<Arc<RwLock<Tensor>>>,
    lr: f32,
) -> Lookahead<crate::radam::RAdam> {
    let radam = crate::radam::RAdam::new(params, Some(lr), None, None, None, None);
    Lookahead::with_defaults(radam)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adam::Adam;
    use parking_lot::RwLock;
    use std::sync::Arc;
    use torsh_tensor::creation::ones;

    #[test]
    fn test_lookahead_creation() {
        let param = Arc::new(RwLock::new(ones(&[2, 3]).unwrap()));
        let adam = Adam::new(vec![param], Some(0.001), None, None, None, false);
        let optimizer = Lookahead::new(adam, 0.5, 5);

        assert_eq!(optimizer.alpha(), 0.5);
        assert_eq!(optimizer.k(), 5);
        assert_eq!(optimizer.step_count(), 0);
    }

    #[test]
    fn test_lookahead_with_defaults() {
        let param = Arc::new(RwLock::new(ones(&[2, 3]).unwrap()));
        let adam = Adam::new(vec![param], Some(0.001), None, None, None, false);
        let optimizer = Lookahead::with_defaults(adam);

        assert_eq!(optimizer.alpha(), 0.5);
        assert_eq!(optimizer.k(), 5);
        assert_eq!(optimizer.step_count(), 0);
    }

    #[test]
    fn test_lookahead_setters() {
        let param = Arc::new(RwLock::new(ones(&[2, 3]).unwrap()));
        let adam = Adam::new(vec![param], Some(0.001), None, None, None, false);
        let mut optimizer = Lookahead::new(adam, 0.5, 5);

        optimizer.set_alpha(0.8);
        optimizer.set_k(10);

        assert_eq!(optimizer.alpha(), 0.8);
        assert_eq!(optimizer.k(), 10);
    }

    #[test]
    fn test_lookahead_base_optimizer_access() {
        let param = Arc::new(RwLock::new(ones(&[2, 3]).unwrap()));
        let adam = Adam::new(vec![param], Some(0.001), None, None, None, false);
        let mut optimizer = Lookahead::new(adam, 0.5, 5);

        // Test immutable access
        let lr = optimizer.base_optimizer().get_lr();
        assert_eq!(lr[0], 0.001);

        // Test mutable access
        optimizer.base_optimizer_mut().set_lr(0.002);
        let new_lr = optimizer.base_optimizer().get_lr();
        assert_eq!(new_lr[0], 0.002);
    }

    #[test]
    fn test_lookahead_lr_operations() {
        let param = Arc::new(RwLock::new(ones(&[2, 3]).unwrap()));
        let adam = Adam::new(vec![param], Some(0.001), None, None, None, false);
        let mut optimizer = Lookahead::new(adam, 0.5, 5);

        assert_eq!(optimizer.get_lr()[0], 0.001);

        optimizer.set_lr(0.002);
        assert_eq!(optimizer.get_lr()[0], 0.002);
    }

    #[test]
    fn test_lookahead_zero_grad() {
        let param = Arc::new(RwLock::new(ones(&[2, 2]).unwrap()));

        // Set a gradient
        {
            let mut p = param.write();
            let grad = ones(&[2, 2]).unwrap();
            p.set_grad(Some(grad));
            assert!(p.grad().is_some());
        }

        let adam = Adam::new(vec![param.clone()], Some(0.001), None, None, None, false);
        let mut optimizer = Lookahead::new(adam, 0.5, 5);
        optimizer.zero_grad();

        // Check gradient is cleared
        let p = param.read();
        assert!(p.grad().is_none());
    }

    #[test]
    fn test_lookahead_step_counting() {
        let param = Arc::new(RwLock::new(ones(&[2, 2]).unwrap()));

        // Set a gradient
        {
            let mut p = param.write();
            let grad = ones(&[2, 2]).unwrap().mul_scalar(0.1).unwrap();
            p.set_grad(Some(grad));
        }

        let adam = Adam::new(vec![param.clone()], Some(0.001), None, None, None, false);
        let mut optimizer = Lookahead::new(adam, 0.5, 3);

        // Step 1-2: should only update fast weights
        optimizer.step().unwrap();
        assert_eq!(optimizer.step_count(), 1);

        // Set gradient again
        {
            let mut p = param.write();
            let grad = ones(&[2, 2]).unwrap().mul_scalar(0.1).unwrap();
            p.set_grad(Some(grad));
        }

        optimizer.step().unwrap();
        assert_eq!(optimizer.step_count(), 2);

        // Set gradient again
        {
            let mut p = param.write();
            let grad = ones(&[2, 2]).unwrap().mul_scalar(0.1).unwrap();
            p.set_grad(Some(grad));
        }

        // Step 3: should update both fast and slow weights
        optimizer.step().unwrap();
        assert_eq!(optimizer.step_count(), 3);
    }

    #[test]
    fn test_lookahead_convenience_functions() {
        let param = Arc::new(RwLock::new(ones(&[2, 2]).unwrap()));

        let _adam_lookahead = lookahead_adam(vec![param.clone()], 0.001);
        let _sgd_lookahead = lookahead_sgd(vec![param.clone()], 0.01);
        let _radam_lookahead = lookahead_radam(vec![param.clone()], 0.001);

        // Just test that they compile and create successfully
    }

    #[test]
    fn test_lookahead_exposes_base_parameters() {
        let param = Arc::new(RwLock::new(ones(&[2, 3]).unwrap()));
        let adam = Adam::new(vec![param.clone()], Some(0.001), None, None, None, false);
        let optimizer = Lookahead::new(adam, 0.5, 5);

        // Lookahead must surface the base optimizer's parameters so the slow-weight
        // update can run; an empty list would mean the optimizer silently no-ops.
        let params = optimizer.parameters();
        assert_eq!(params.len(), 1);
        assert!(Arc::ptr_eq(&params[0], &param));
    }

    #[test]
    fn test_lookahead_updates_slow_weights() {
        use crate::sgd::SGD;

        // With k = 1 the slow-weight blend runs after every inner step. Using SGD as
        // the base optimizer makes the fast-weight step deterministic, so we can
        // assert the exact Lookahead update φ_new = φ_old + α(θ_fast - φ_old).
        //
        // SGD (lr = 0.1) on param = 1.0 with grad = 1.0 gives the fast weight
        //   θ_fast = 1.0 - 0.1 * 1.0 = 0.9
        // and the slow-weight blend with α = 0.5, φ_old = 1.0 gives
        //   φ_new = 1.0 + 0.5 * (0.9 - 1.0) = 0.95
        // which is written back onto the live parameter. This is only possible if
        // Lookahead can reach the base optimizer's parameters via `parameters()`.
        let param = Arc::new(RwLock::new(ones(&[2, 2]).unwrap()));

        {
            let mut p = param.write();
            let grad = ones(&[2, 2]).unwrap();
            p.set_grad(Some(grad));
        }

        let sgd = SGD::new(vec![param.clone()], 0.1, None, None, None, false);
        let mut optimizer = Lookahead::new(sgd, 0.5, 1);

        optimizer.step().unwrap();

        let result = param.read().to_vec().unwrap();
        for v in &result {
            assert!(
                (v - 0.95).abs() < 1e-5,
                "expected blended slow weight 0.95, got {v}"
            );
        }
    }

    #[test]
    fn test_lookahead_state_dict() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(ones(&[2, 2]).unwrap()));
        let adam = Adam::new(vec![param], Some(0.001), None, None, None, false);
        let optimizer = Lookahead::new(adam, 0.6, 7);

        let state = optimizer.state_dict()?;
        // Should have one extra param group for Lookahead state
        assert!(!state.param_groups.is_empty());

        // The last param group should contain Lookahead options
        let lookahead_group = state.param_groups.last().unwrap();
        assert_eq!(lookahead_group.options.get("alpha"), Some(&0.6));
        assert_eq!(lookahead_group.options.get("k"), Some(&7.0));

        Ok(())
    }
}
