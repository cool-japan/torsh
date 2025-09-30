//! Ranger optimizer implementation
//!
//! Ranger combines RAdam (Rectified Adam) with Lookahead to provide both
//! the benefits of rectified adaptive learning rates and stable convergence.
//! It's essentially Lookahead(RAdam), providing a powerful optimization
//! algorithm that works well across many different tasks.
//!
//! Ranger = RAdam + Lookahead
//!
//! References:
//! - RAdam: "On the Variance of the Adaptive Learning Rate and Beyond"
//! - Lookahead: "Lookahead Optimizer: k steps forward, 1 step back"
//! - Ranger: Combined by Less Wright (fastai community)

use crate::{lookahead::Lookahead, radam::RAdam};
use parking_lot::RwLock;
use std::sync::Arc;
use torsh_tensor::Tensor;

/// Ranger optimizer (RAdam + Lookahead)
///
/// This is a type alias for Lookahead wrapped around RAdam, providing
/// the benefits of both optimizers in a single, easy-to-use package.
pub type Ranger = Lookahead<RAdam>;

impl Ranger {
    /// Create a new Ranger optimizer with default hyperparameters
    ///
    /// Uses:
    /// - RAdam with lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    /// - Lookahead with alpha=0.5, k=5
    pub fn new_ranger(params: Vec<Arc<RwLock<Tensor>>>, lr: f32) -> Self {
        let radam = RAdam::new(params, Some(lr), None, None, None, None);
        Lookahead::with_defaults(radam)
    }

    /// Create Ranger with custom RAdam parameters but default Lookahead parameters
    #[allow(clippy::too_many_arguments)]
    pub fn with_radam_params(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) -> Self {
        let radam = RAdam::new(
            params,
            Some(lr),
            Some(beta1),
            Some(beta2),
            Some(eps),
            Some(weight_decay),
        );
        Lookahead::with_defaults(radam)
    }

    /// Create Ranger with custom Lookahead parameters but default RAdam parameters
    pub fn with_lookahead_params(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: f32,
        alpha: f32,
        k: usize,
    ) -> Self {
        let radam = RAdam::new(params, Some(lr), None, None, None, None);
        Lookahead::new(radam, alpha, k)
    }

    /// Create Ranger with full customization of both RAdam and Lookahead parameters
    #[allow(clippy::too_many_arguments)]
    pub fn with_all_params(
        params: Vec<Arc<RwLock<Tensor>>>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        alpha: f32,
        k: usize,
    ) -> Self {
        let radam = RAdam::new(
            params,
            Some(lr),
            Some(beta1),
            Some(beta2),
            Some(eps),
            Some(weight_decay),
        );
        Lookahead::new(radam, alpha, k)
    }

    /// Get the underlying RAdam optimizer
    pub fn radam(&self) -> &RAdam {
        self.base_optimizer()
    }

    /// Get a mutable reference to the underlying RAdam optimizer
    pub fn radam_mut(&mut self) -> &mut RAdam {
        self.base_optimizer_mut()
    }

    /// Get RAdam's beta1 parameter
    pub fn beta1(&self) -> f32 {
        self.radam().beta1()
    }

    /// Get RAdam's beta2 parameter
    pub fn beta2(&self) -> f32 {
        self.radam().beta2()
    }

    /// Get RAdam's epsilon parameter
    pub fn eps(&self) -> f32 {
        self.radam().eps()
    }

    /// Get RAdam's weight decay parameter
    pub fn weight_decay(&self) -> f32 {
        self.radam().weight_decay()
    }

    /// Set RAdam's beta1 parameter
    pub fn set_beta1(&mut self, beta1: f32) {
        self.radam_mut().set_beta1(beta1);
    }

    /// Set RAdam's beta2 parameter
    pub fn set_beta2(&mut self, beta2: f32) {
        self.radam_mut().set_beta2(beta2);
    }

    /// Set RAdam's epsilon parameter
    pub fn set_eps(&mut self, eps: f32) {
        self.radam_mut().set_eps(eps);
    }

    /// Set RAdam's weight decay parameter
    pub fn set_weight_decay(&mut self, weight_decay: f32) {
        self.radam_mut().set_weight_decay(weight_decay);
    }

    /// Create a "Ranger21" variant with enhanced parameters
    ///
    /// Ranger21 is an improved version that uses different hyperparameters
    /// based on empirical findings for better performance.
    pub fn ranger21(params: Vec<Arc<RwLock<Tensor>>>, lr: f32) -> Self {
        // Ranger21 typically uses:
        // - Higher beta1 (0.95 instead of 0.9)
        // - Different lookahead parameters (alpha=0.8, k=6)
        // - Specific epsilon value
        let radam = RAdam::new(
            params,
            Some(lr),
            Some(0.95),  // beta1
            Some(0.999), // beta2
            Some(1e-7),  // eps
            Some(0.0),   // weight_decay
        );
        Lookahead::new(radam, 0.8, 6) // alpha=0.8, k=6
    }

    /// Create Ranger with warm restart support
    ///
    /// This creates a Ranger optimizer that's designed to work well with
    /// cosine annealing warm restarts.
    pub fn with_warm_restarts(params: Vec<Arc<RwLock<Tensor>>>, lr: f32) -> Self {
        // Parameters optimized for warm restart training
        let radam = RAdam::new(
            params,
            Some(lr),
            Some(0.9),   // beta1
            Some(0.999), // beta2
            Some(1e-6),  // eps (slightly higher for stability)
            Some(0.0),   // weight_decay
        );
        Lookahead::new(radam, 0.6, 4) // More aggressive lookahead
    }

    /// Enable/disable rectification in the underlying RAdam optimizer
    pub fn set_rectification(&mut self, enabled: bool) {
        self.radam_mut().set_rectification(enabled);
    }

    /// Check if rectification is enabled
    pub fn is_rectification_enabled(&self) -> bool {
        self.radam().is_rectification_enabled()
    }

    /// Get the current rectification coefficient (if rectification is enabled)
    pub fn rectification_coefficient(&self) -> Option<f32> {
        self.radam().rectification_coefficient()
    }
}

/// Builder pattern for Ranger optimizer configuration
pub struct RangerBuilder {
    params: Vec<Arc<RwLock<Tensor>>>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    alpha: f32,
    k: usize,
}

impl RangerBuilder {
    /// Create a new Ranger builder
    pub fn new(params: Vec<Arc<RwLock<Tensor>>>) -> Self {
        Self {
            params,
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            alpha: 0.5,
            k: 5,
        }
    }

    /// Set learning rate
    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }

    /// Set RAdam beta1 parameter
    pub fn beta1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set RAdam beta2 parameter
    pub fn beta2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set RAdam epsilon parameter
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Set RAdam weight decay parameter
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set Lookahead alpha parameter
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set Lookahead k parameter
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Build the Ranger optimizer
    pub fn build(self) -> Ranger {
        Ranger::with_all_params(
            self.params,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            self.weight_decay,
            self.alpha,
            self.k,
        )
    }
}

/// Convenience functions for common Ranger configurations
impl Ranger {
    /// Get a builder for this optimizer
    pub fn builder(params: Vec<Arc<RwLock<Tensor>>>) -> RangerBuilder {
        RangerBuilder::new(params)
    }

    /// Create Ranger optimized for computer vision tasks
    pub fn for_vision(params: Vec<Arc<RwLock<Tensor>>>, lr: f32) -> Self {
        Self::with_all_params(
            params, lr, 0.9,   // beta1
            0.999, // beta2
            1e-8,  // eps
            1e-4,  // weight_decay (common for vision)
            0.5,   // alpha
            5,     // k
        )
    }

    /// Create Ranger optimized for natural language processing tasks
    pub fn for_nlp(params: Vec<Arc<RwLock<Tensor>>>, lr: f32) -> Self {
        Self::with_all_params(
            params, lr, 0.9,   // beta1
            0.999, // beta2
            1e-8,  // eps
            0.01,  // weight_decay (common for NLP)
            0.6,   // alpha (slightly more aggressive)
            4,     // k (shorter lookahead for faster convergence)
        )
    }

    /// Create Ranger optimized for reinforcement learning
    pub fn for_rl(params: Vec<Arc<RwLock<Tensor>>>, lr: f32) -> Self {
        Self::with_all_params(
            params, lr, 0.95,  // beta1 (higher for RL stability)
            0.999, // beta2
            1e-6,  // eps (slightly higher for stability)
            0.0,   // weight_decay (typically not used in RL)
            0.8,   // alpha (more aggressive for exploration)
            3,     // k (shorter for faster adaptation)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Optimizer, OptimizerResult};
    use parking_lot::RwLock;
    use std::sync::Arc;
    use torsh_tensor::creation::ones;

    #[test]
    fn test_ranger_creation() {
        let param = Arc::new(RwLock::new(ones(&[2, 3]).unwrap()));
        let optimizer = Ranger::new_ranger(vec![param], 0.001);

        assert_eq!(optimizer.get_lr()[0], 0.001);
        assert_eq!(optimizer.alpha(), 0.5);
        assert_eq!(optimizer.k(), 5);
        assert_eq!(optimizer.beta1(), 0.9);
        assert_eq!(optimizer.beta2(), 0.999);
    }

    #[test]
    fn test_ranger_with_radam_params() {
        let param = Arc::new(RwLock::new(ones(&[2, 3]).unwrap()));
        let optimizer = Ranger::with_radam_params(vec![param], 0.002, 0.95, 0.9999, 1e-7, 0.01);

        assert_eq!(optimizer.get_lr()[0], 0.002);
        assert_eq!(optimizer.beta1(), 0.95);
        assert_eq!(optimizer.beta2(), 0.9999);
        assert_eq!(optimizer.eps(), 1e-7);
        assert_eq!(optimizer.weight_decay(), 0.01);
        // Default Lookahead parameters
        assert_eq!(optimizer.alpha(), 0.5);
        assert_eq!(optimizer.k(), 5);
    }

    #[test]
    fn test_ranger_with_lookahead_params() {
        let param = Arc::new(RwLock::new(ones(&[2, 3]).unwrap()));
        let optimizer = Ranger::with_lookahead_params(vec![param], 0.003, 0.8, 6);

        assert_eq!(optimizer.get_lr()[0], 0.003);
        assert_eq!(optimizer.alpha(), 0.8);
        assert_eq!(optimizer.k(), 6);
        // Default RAdam parameters
        assert_eq!(optimizer.beta1(), 0.9);
        assert_eq!(optimizer.beta2(), 0.999);
    }

    #[test]
    fn test_ranger_with_all_params() {
        let param = Arc::new(RwLock::new(ones(&[2, 3]).unwrap()));
        let optimizer =
            Ranger::with_all_params(vec![param], 0.005, 0.95, 0.9999, 1e-7, 0.02, 0.7, 8);

        assert_eq!(optimizer.get_lr()[0], 0.005);
        assert_eq!(optimizer.beta1(), 0.95);
        assert_eq!(optimizer.beta2(), 0.9999);
        assert_eq!(optimizer.eps(), 1e-7);
        assert_eq!(optimizer.weight_decay(), 0.02);
        assert_eq!(optimizer.alpha(), 0.7);
        assert_eq!(optimizer.k(), 8);
    }

    #[test]
    fn test_ranger21() {
        let param = Arc::new(RwLock::new(ones(&[2, 3]).unwrap()));
        let optimizer = Ranger::ranger21(vec![param], 0.001);

        assert_eq!(optimizer.get_lr()[0], 0.001);
        assert_eq!(optimizer.beta1(), 0.95);
        assert_eq!(optimizer.beta2(), 0.999);
        assert_eq!(optimizer.eps(), 1e-7);
        assert_eq!(optimizer.alpha(), 0.8);
        assert_eq!(optimizer.k(), 6);
    }

    #[test]
    fn test_ranger_builder() {
        let param = Arc::new(RwLock::new(ones(&[2, 3]).unwrap()));
        let optimizer = Ranger::builder(vec![param])
            .lr(0.002)
            .beta1(0.95)
            .beta2(0.9999)
            .eps(1e-7)
            .weight_decay(0.01)
            .alpha(0.6)
            .k(4)
            .build();

        assert_eq!(optimizer.get_lr()[0], 0.002);
        assert_eq!(optimizer.beta1(), 0.95);
        assert_eq!(optimizer.beta2(), 0.9999);
        assert_eq!(optimizer.eps(), 1e-7);
        assert_eq!(optimizer.weight_decay(), 0.01);
        assert_eq!(optimizer.alpha(), 0.6);
        assert_eq!(optimizer.k(), 4);
    }

    #[test]
    fn test_ranger_setters() {
        let param = Arc::new(RwLock::new(ones(&[2, 3]).unwrap()));
        let mut optimizer = Ranger::new_ranger(vec![param], 0.001);

        optimizer.set_beta1(0.95);
        optimizer.set_beta2(0.9999);
        optimizer.set_eps(1e-7);
        optimizer.set_weight_decay(0.02);
        optimizer.set_alpha(0.7);
        optimizer.set_k(7);

        assert_eq!(optimizer.beta1(), 0.95);
        assert_eq!(optimizer.beta2(), 0.9999);
        assert_eq!(optimizer.eps(), 1e-7);
        assert_eq!(optimizer.weight_decay(), 0.02);
        assert_eq!(optimizer.alpha(), 0.7);
        assert_eq!(optimizer.k(), 7);
    }

    #[test]
    fn test_ranger_domain_specific() {
        let param = Arc::new(RwLock::new(ones(&[2, 3]).unwrap()));

        let vision_opt = Ranger::for_vision(vec![param.clone()], 0.001);
        assert_eq!(vision_opt.weight_decay(), 1e-4);

        let nlp_opt = Ranger::for_nlp(vec![param.clone()], 0.001);
        assert_eq!(nlp_opt.weight_decay(), 0.01);
        assert_eq!(nlp_opt.alpha(), 0.6);
        assert_eq!(nlp_opt.k(), 4);

        let rl_opt = Ranger::for_rl(vec![param.clone()], 0.001);
        assert_eq!(rl_opt.beta1(), 0.95);
        assert_eq!(rl_opt.alpha(), 0.8);
        assert_eq!(rl_opt.k(), 3);
    }

    #[test]
    fn test_ranger_step() -> OptimizerResult<()> {
        let param = Arc::new(RwLock::new(ones(&[2, 2]).unwrap()));

        // Set a gradient
        {
            let mut p = param.write();
            let grad = ones(&[2, 2]).unwrap().mul_scalar(0.1)?;
            p.set_grad(Some(grad));
        }

        let mut optimizer = Ranger::new_ranger(vec![param.clone()], 0.01);

        optimizer.step()?;
        assert_eq!(optimizer.step_count(), 1);

        Ok(())
    }

    #[test]
    fn test_ranger_rectification() {
        let param = Arc::new(RwLock::new(ones(&[2, 2]).unwrap()));
        let mut optimizer = Ranger::new_ranger(vec![param], 0.01);

        // RAdam always uses rectification by design, so it should always be enabled
        assert!(optimizer.is_rectification_enabled());

        // Attempting to disable rectification is a no-op for RAdam
        optimizer.set_rectification(false);
        // Rectification remains enabled because RAdam always uses it
        assert!(optimizer.is_rectification_enabled());

        // Setting rectification to true should also work (still enabled)
        optimizer.set_rectification(true);
        assert!(optimizer.is_rectification_enabled());
    }

    #[test]
    fn test_ranger_warm_restarts() {
        let param = Arc::new(RwLock::new(ones(&[2, 3]).unwrap()));
        let optimizer = Ranger::with_warm_restarts(vec![param], 0.001);

        assert_eq!(optimizer.eps(), 1e-6);
        assert_eq!(optimizer.alpha(), 0.6);
        assert_eq!(optimizer.k(), 4);
    }
}
