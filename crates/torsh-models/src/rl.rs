//! Reinforcement Learning Models
//!
//! This module provides implementations of popular reinforcement learning algorithms
//! including Deep Q-Networks (DQN), Proximal Policy Optimization (PPO), and
//! Asynchronous Advantage Actor-Critic (A3C).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_core::{error::Result as TorshResult, DType, DeviceType};
use torsh_nn::prelude::{Linear, LogSoftmax, ReLU, Softmax, Tanh, GELU};
use torsh_nn::{Module, Parameter};
use torsh_tensor::{creation, Tensor};

/// Configuration for Deep Q-Network (DQN)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DQNConfig {
    /// Input state dimensions
    pub state_dim: usize,
    /// Number of possible actions
    pub action_dim: usize,
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Activation function type
    pub activation: String,
    /// Whether to use dueling DQN architecture
    pub dueling: bool,
    /// Whether to use double DQN
    pub double_dqn: bool,
    /// Replay buffer size
    pub buffer_size: usize,
    /// Discount factor (gamma)
    pub gamma: f32,
    /// Learning rate
    pub learning_rate: f32,
    /// Epsilon for epsilon-greedy exploration
    pub epsilon: f32,
    /// Epsilon decay rate
    pub epsilon_decay: f32,
    /// Minimum epsilon value
    pub epsilon_min: f32,
    /// Target network update frequency
    pub target_update_freq: usize,
}

impl Default for DQNConfig {
    fn default() -> Self {
        Self {
            state_dim: 4,
            action_dim: 2,
            hidden_dims: vec![128, 128],
            activation: "relu".to_string(),
            dueling: false,
            double_dqn: false,
            buffer_size: 10000,
            gamma: 0.99,
            learning_rate: 1e-3,
            epsilon: 1.0,
            epsilon_decay: 0.995,
            epsilon_min: 0.01,
            target_update_freq: 100,
        }
    }
}

/// Deep Q-Network implementation
#[derive(Debug)]
pub struct DQN {
    config: DQNConfig,
    /// Main Q-network layers
    layers: Vec<Linear>,
    /// Value stream (for dueling DQN)
    value_stream: Option<Linear>,
    /// Advantage stream (for dueling DQN)
    advantage_stream: Option<Linear>,
    /// Training mode flag
    training: bool,
}

impl DQN {
    /// Create a new DQN model
    pub fn new(config: DQNConfig) -> TorshResult<Self> {
        let mut layers = Vec::new();
        let mut prev_dim = config.state_dim;

        // Build hidden layers
        for &hidden_dim in &config.hidden_dims {
            layers.push(Linear::new(prev_dim, hidden_dim, true));
            prev_dim = hidden_dim;
        }

        // Output layer setup depends on dueling architecture
        let (value_stream, advantage_stream) = if config.dueling {
            // Dueling DQN: separate value and advantage streams
            let value = Linear::new(prev_dim, 1, true);
            let advantage = Linear::new(prev_dim, config.action_dim, true);
            (Some(value), Some(advantage))
        } else {
            // Standard DQN: single output layer
            layers.push(Linear::new(prev_dim, config.action_dim, true));
            (None, None)
        };

        Ok(Self {
            config,
            layers,
            value_stream,
            advantage_stream,
            training: true,
        })
    }

    /// Select action using epsilon-greedy policy
    pub fn select_action(&self, state: &Tensor, epsilon: f32) -> TorshResult<usize> {
        use scirs2_core::random::{Random, Rng};
        let mut rng = Random::seed(42);

        if rng.gen::<f32>() < epsilon {
            // Random action
            Ok(rng.gen_range(0..self.config.action_dim))
        } else {
            // Greedy action
            let q_values = self.forward(state)?;
            let action = q_values.argmax(Some(1))?;
            Ok(action.item()? as usize)
        }
    }

    /// Compute target Q-values for training
    pub fn compute_targets(
        &self,
        rewards: &Tensor,
        next_states: &Tensor,
        dones: &Tensor,
        target_net: &DQN,
    ) -> TorshResult<Tensor> {
        let next_q_values = if self.config.double_dqn {
            // Double DQN: use main network to select actions, target network to evaluate
            let next_actions = self.forward(next_states)?.argmax(Some(1))?;
            let target_q_values = target_net.forward(next_states)?;
            target_q_values.gather(1, &next_actions.unsqueeze(1)?)?
        } else {
            // Standard DQN: use target network for both selection and evaluation
            target_net.forward(next_states)?.max(Some(1), false)?
        };

        // Compute targets: r + gamma * max(Q'(s', a')) * (1 - done)
        let gamma_tensor =
            torsh_tensor::creation::full(next_q_values.shape().dims(), self.config.gamma)?;
        let not_done = Tensor::ones_like(dones)?.sub(dones)?;
        let gamma_next = gamma_tensor.mul(&next_q_values)?;
        let gamma_next_done = gamma_next.mul(&not_done)?;
        let targets = rewards.add(&gamma_next_done)?;

        Ok(targets)
    }
}

impl Module for DQN {
    fn forward(&self, input: &Tensor) -> TorshResult<Tensor> {
        let mut x = input.clone();

        // Forward through hidden layers
        for (i, layer) in self.layers[..self.layers.len() - if self.config.dueling { 0 } else { 1 }]
            .iter()
            .enumerate()
        {
            x = layer.forward(&x)?;

            // Apply activation
            x = match self.config.activation.as_str() {
                "relu" => ReLU::new().forward(&x)?,
                "gelu" => GELU::new(false).forward(&x)?,
                "tanh" => Tanh::new().forward(&x)?,
                _ => ReLU::new().forward(&x)?,
            };
        }

        if self.config.dueling {
            // Dueling architecture: V(s) + A(s,a) - mean(A(s,*))
            let value = self.value_stream.as_ref().unwrap().forward(&x)?;
            let advantage = self.advantage_stream.as_ref().unwrap().forward(&x)?;
            let advantage_mean = advantage.mean(Some(&[1]), false)?.unsqueeze(1)?;
            let q_values = value.add(&advantage)?.sub(&advantage_mean)?;
            Ok(q_values)
        } else {
            // Standard architecture: direct Q-value output
            let final_layer = self.layers.last().unwrap();
            final_layer.forward(&x)
        }
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        // Add parameters from all layers
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_params = layer.parameters();
            for (name, param) in layer_params {
                params.insert(format!("layer_{}.{}", i, name), param);
            }
        }

        // Add dueling architecture parameters
        if let Some(ref value_stream) = self.value_stream {
            let value_params = value_stream.parameters();
            for (name, param) in value_params {
                params.insert(format!("value_stream.{}", name), param);
            }
        }

        if let Some(ref advantage_stream) = self.advantage_stream {
            let advantage_params = advantage_stream.parameters();
            for (name, param) in advantage_params {
                params.insert(format!("advantage_stream.{}", name), param);
            }
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.training
    }

    fn train(&mut self) {
        self.training = true;
        for layer in &mut self.layers {
            layer.train();
        }
        if let Some(ref mut value_stream) = self.value_stream {
            value_stream.train();
        }
        if let Some(ref mut advantage_stream) = self.advantage_stream {
            advantage_stream.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        for layer in &mut self.layers {
            layer.eval();
        }
        if let Some(ref mut value_stream) = self.value_stream {
            value_stream.eval();
        }
        if let Some(ref mut advantage_stream) = self.advantage_stream {
            advantage_stream.eval();
        }
    }

    fn to_device(&mut self, device: DeviceType) -> TorshResult<()> {
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        if let Some(ref mut value_stream) = self.value_stream {
            value_stream.to_device(device)?;
        }
        if let Some(ref mut advantage_stream) = self.advantage_stream {
            advantage_stream.to_device(device)?;
        }
        Ok(())
    }
}

/// Configuration for Proximal Policy Optimization (PPO)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PPOConfig {
    /// State space dimension
    pub state_dim: usize,
    /// Action space dimension (for discrete actions)
    pub action_dim: usize,
    /// Whether action space is continuous
    pub continuous_actions: bool,
    /// Actor network hidden dimensions
    pub actor_hidden_dims: Vec<usize>,
    /// Critic network hidden dimensions
    pub critic_hidden_dims: Vec<usize>,
    /// Activation function
    pub activation: String,
    /// Learning rate for actor
    pub actor_lr: f32,
    /// Learning rate for critic
    pub critic_lr: f32,
    /// Discount factor
    pub gamma: f32,
    /// GAE lambda parameter
    pub gae_lambda: f32,
    /// PPO clip parameter
    pub clip_epsilon: f32,
    /// Number of PPO epochs per update
    pub ppo_epochs: usize,
    /// Batch size for PPO updates
    pub batch_size: usize,
    /// Value function loss coefficient
    pub value_coef: f32,
    /// Entropy bonus coefficient
    pub entropy_coef: f32,
}

impl Default for PPOConfig {
    fn default() -> Self {
        Self {
            state_dim: 4,
            action_dim: 2,
            continuous_actions: false,
            actor_hidden_dims: vec![64, 64],
            critic_hidden_dims: vec![64, 64],
            activation: "tanh".to_string(),
            actor_lr: 3e-4,
            critic_lr: 1e-3,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_epsilon: 0.2,
            ppo_epochs: 4,
            batch_size: 64,
            value_coef: 0.5,
            entropy_coef: 0.01,
        }
    }
}

/// Actor network for PPO
#[derive(Debug)]
pub struct PPOActor {
    config: PPOConfig,
    layers: Vec<Linear>,
    /// Output layer for action logits/means
    action_head: Linear,
    /// Output layer for action log stds (continuous actions only)
    log_std_head: Option<Linear>,
    training: bool,
}

impl PPOActor {
    pub fn new(config: PPOConfig) -> TorshResult<Self> {
        let mut layers = Vec::new();
        let mut prev_dim = config.state_dim;

        // Build hidden layers
        for &hidden_dim in &config.actor_hidden_dims {
            layers.push(Linear::new(prev_dim, hidden_dim, true));
            prev_dim = hidden_dim;
        }

        // Output layers
        let action_head = Linear::new(prev_dim, config.action_dim, true);
        let log_std_head = if config.continuous_actions {
            Some(Linear::new(prev_dim, config.action_dim, true))
        } else {
            None
        };

        Ok(Self {
            config,
            layers,
            action_head,
            log_std_head,
            training: true,
        })
    }

    /// Sample action from policy
    pub fn sample_action(&self, state: &Tensor) -> TorshResult<(Tensor, Tensor)> {
        let action_logits = self.forward(state)?;

        if self.config.continuous_actions {
            // Continuous actions: sample from Gaussian distribution
            let mean = action_logits;
            let log_std = self.log_std_head.as_ref().unwrap().forward(state)?;
            let std = log_std.exp()?;

            // Sample action: mean + std * noise
            let noise = torsh_tensor::creation::randn(mean.shape().dims())?;
            let action = mean.add(&std.mul(&noise)?)?;

            // Compute log probability
            let log_prob = self.gaussian_log_prob(&action, &mean, &std)?;

            Ok((action, log_prob))
        } else {
            // Discrete actions: sample from categorical distribution
            let log_probs = LogSoftmax::new(Some(1)).forward(&action_logits)?;
            let probs = log_probs.exp()?;

            // Sample action
            let action = self.categorical_sample(&probs)?;
            let action_data = action.to_vec()?;
            let action_i64_data: Vec<i64> = action_data.iter().map(|&x| x as i64).collect();
            let action_i64 = Tensor::<i64>::from_data(
                action_i64_data,
                action.shape().dims().to_vec(),
                action.device(),
            )?;
            let log_prob = log_probs.gather(1, &action_i64.unsqueeze(1)?)?;

            Ok((action, log_prob))
        }
    }

    /// Compute log probability of actions under current policy
    pub fn log_prob(&self, state: &Tensor, action: &Tensor) -> TorshResult<Tensor> {
        let action_logits = self.forward(state)?;

        if self.config.continuous_actions {
            let mean = action_logits;
            let log_std = self.log_std_head.as_ref().unwrap().forward(state)?;
            let std = log_std.exp()?;
            self.gaussian_log_prob(action, &mean, &std)
        } else {
            let log_probs = LogSoftmax::new(Some(1)).forward(&action_logits)?;
            let action_data = action.to_vec()?;
            let action_i64_data: Vec<i64> = action_data.iter().map(|&x| x as i64).collect();
            let action_i64 = Tensor::<i64>::from_data(
                action_i64_data,
                action.shape().dims().to_vec(),
                action.device(),
            )?;
            log_probs.gather(1, &action_i64.unsqueeze(1)?)
        }
    }

    fn gaussian_log_prob(
        &self,
        action: &Tensor,
        mean: &Tensor,
        std: &Tensor,
    ) -> TorshResult<Tensor> {
        // Log probability of Gaussian: -0.5 * ((action - mean) / std)^2 - log(std) - 0.5 * log(2π)
        let var = std.pow(2.0)?;
        let diff = action.sub(&mean)?;
        let diff_squared = diff.pow(2.0)?;
        let div_result = diff_squared.div(&var)?;
        let var_log = var.log()?;
        let pi_term = creation::tensor_scalar((2.0 * std::f32::consts::PI).ln())?;
        let sum_terms = div_result.add(&var_log)?.add(&pi_term)?;
        let log_prob = sum_terms.mul_scalar(-0.5)?;
        log_prob.sum() // Sum over action dimensions
    }

    fn categorical_sample(&self, probs: &Tensor) -> TorshResult<Tensor> {
        // Multinomial sampling
        use scirs2_core::random::{Random, Rng};
        let mut rng = Random::seed(42);
        let random_val = rng.gen::<f32>();

        // Simple implementation: use multinomial sampling
        let cumprobs = probs.cumsum(1)?;
        let random_tensor =
            torsh_tensor::creation::full(&[probs.shape().dims()[0], 1], random_val)?;
        let bool_tensor = cumprobs.gt(&random_tensor)?;
        let bool_data = bool_tensor.to_vec()?;
        let i64_data: Vec<i64> = bool_data
            .iter()
            .map(|&x| if x { 1i64 } else { 0i64 })
            .collect();
        let i64_tensor = Tensor::<i64>::from_data(
            i64_data,
            bool_tensor.shape().dims().to_vec(),
            bool_tensor.device(),
        )?;
        let action = i64_tensor.sum()?.to_f32_simd()?;

        Ok(action)
    }
}

impl Module for PPOActor {
    fn forward(&self, input: &Tensor) -> TorshResult<Tensor> {
        let mut x = input.clone();

        // Forward through hidden layers
        for layer in &self.layers {
            x = layer.forward(&x)?;
            x = match self.config.activation.as_str() {
                "relu" => ReLU::new().forward(&x)?,
                "gelu" => GELU::new(false).forward(&x)?,
                "tanh" => Tanh::new().forward(&x)?,
                _ => Tanh::new().forward(&x)?,
            };
        }

        // Action head
        self.action_head.forward(&x)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_params = layer.parameters();
            for (name, param) in layer_params {
                params.insert(format!("layer_{}.{}", i, name), param);
            }
        }

        let action_params = self.action_head.parameters();
        for (name, param) in action_params {
            params.insert(format!("action_head.{}", name), param);
        }

        if let Some(ref log_std_head) = self.log_std_head {
            let log_std_params = log_std_head.parameters();
            for (name, param) in log_std_params {
                params.insert(format!("log_std_head.{}", name), param);
            }
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.training
    }

    fn train(&mut self) {
        self.training = true;
        for layer in &mut self.layers {
            layer.train();
        }
        self.action_head.train();
        if let Some(ref mut log_std_head) = self.log_std_head {
            log_std_head.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        for layer in &mut self.layers {
            layer.eval();
        }
        self.action_head.eval();
        if let Some(ref mut log_std_head) = self.log_std_head {
            log_std_head.eval();
        }
    }

    fn to_device(&mut self, device: DeviceType) -> TorshResult<()> {
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        self.action_head.to_device(device)?;
        if let Some(ref mut log_std_head) = self.log_std_head {
            log_std_head.to_device(device)?;
        }
        Ok(())
    }
}

/// Critic network for PPO
#[derive(Debug)]
pub struct PPOCritic {
    config: PPOConfig,
    layers: Vec<Linear>,
    value_head: Linear,
    training: bool,
}

impl PPOCritic {
    pub fn new(config: PPOConfig) -> TorshResult<Self> {
        let mut layers = Vec::new();
        let mut prev_dim = config.state_dim;

        // Build hidden layers
        for &hidden_dim in &config.critic_hidden_dims {
            layers.push(Linear::new(prev_dim, hidden_dim, true));
            prev_dim = hidden_dim;
        }

        // Value output layer
        let value_head = Linear::new(prev_dim, 1, true);

        Ok(Self {
            config,
            layers,
            value_head,
            training: true,
        })
    }
}

impl Module for PPOCritic {
    fn forward(&self, input: &Tensor) -> TorshResult<Tensor> {
        let mut x = input.clone();

        // Forward through hidden layers
        for layer in &self.layers {
            x = layer.forward(&x)?;
            x = match self.config.activation.as_str() {
                "relu" => ReLU::new().forward(&x)?,
                "gelu" => GELU::new(false).forward(&x)?,
                "tanh" => Tanh::new().forward(&x)?,
                _ => Tanh::new().forward(&x)?,
            };
        }

        // Value head
        self.value_head.forward(&x)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_params = layer.parameters();
            for (name, param) in layer_params {
                params.insert(format!("layer_{}.{}", i, name), param);
            }
        }

        let value_params = self.value_head.parameters();
        for (name, param) in value_params {
            params.insert(format!("value_head.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.training
    }

    fn train(&mut self) {
        self.training = true;
        for layer in &mut self.layers {
            layer.train();
        }
        self.value_head.train();
    }

    fn eval(&mut self) {
        self.training = false;
        for layer in &mut self.layers {
            layer.eval();
        }
        self.value_head.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> TorshResult<()> {
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        self.value_head.to_device(device)
    }
}

/// Complete PPO model with actor and critic
#[derive(Debug)]
pub struct PPO {
    pub config: PPOConfig,
    pub actor: PPOActor,
    pub critic: PPOCritic,
    training: bool,
}

impl PPO {
    pub fn new(config: PPOConfig) -> TorshResult<Self> {
        let actor = PPOActor::new(config.clone())?;
        let critic = PPOCritic::new(config.clone())?;

        Ok(Self {
            config,
            actor,
            critic,
            training: true,
        })
    }

    /// Compute GAE advantages
    pub fn compute_gae(
        &self,
        rewards: &Tensor,
        values: &Tensor,
        next_values: &Tensor,
        dones: &Tensor,
    ) -> TorshResult<(Tensor, Tensor)> {
        let mut advantages = Vec::new();
        let mut returns = Vec::new();
        let mut gae = torsh_tensor::creation::zeros(&[1])?;

        let seq_len = rewards.shape().dims()[0];

        for t in (0..seq_len).rev() {
            let reward = rewards.select(0, t as i64)?;
            let value = values.select(0, t as i64)?;
            let next_value = if t == seq_len - 1 {
                torsh_tensor::creation::zeros(value.shape().dims())?
            } else {
                next_values.select(0, t as i64)?
            };
            let done = dones.select(0, t as i64)?;

            let gamma_tensor = creation::tensor_scalar(self.config.gamma)?;
            let ones_like_done = Tensor::ones_like(&done)?;
            let not_done = ones_like_done.sub(&done)?;
            let gamma_next_value = gamma_tensor.mul(&next_value)?;
            let gamma_next_value_masked = gamma_next_value.mul(&not_done)?;
            let delta = reward.add(&gamma_next_value_masked)?.sub(&value)?;

            let gae_lambda_tensor = creation::tensor_scalar(self.config.gae_lambda)?;
            let gamma_gae_lambda = gamma_tensor.mul(&gae_lambda_tensor)?;
            let gamma_gae_lambda_not_done = gamma_gae_lambda.mul(&not_done)?;
            let gamma_gae_lambda_not_done_gae = gamma_gae_lambda_not_done.mul(&gae)?;
            gae = delta.add(&gamma_gae_lambda_not_done_gae)?;

            advantages.insert(0, gae.clone());
            returns.insert(0, gae.add(&value)?);
        }

        // Stack advantages - implement using unsqueeze + cat
        let unsqueezed_advantages: Result<Vec<Tensor>, torsh_core::error::TorshError> =
            advantages.iter().map(|t| t.unsqueeze(0)).collect();
        let unsqueezed_advantages = unsqueezed_advantages?;
        let advantages_refs: Vec<&Tensor> = unsqueezed_advantages.iter().collect();
        let advantages_tensor = Tensor::cat(&advantages_refs, 0)?;

        // Stack returns - implement using unsqueeze + cat
        let unsqueezed_returns: Result<Vec<Tensor>, _> =
            returns.iter().map(|t| t.unsqueeze(0)).collect();
        let unsqueezed_returns = unsqueezed_returns?;
        let returns_refs: Vec<&Tensor> = unsqueezed_returns.iter().collect();
        let returns_tensor = Tensor::cat(&returns_refs, 0)?;

        Ok((advantages_tensor, returns_tensor))
    }

    /// Compute PPO loss
    pub fn compute_ppo_loss(
        &self,
        states: &Tensor,
        actions: &Tensor,
        old_log_probs: &Tensor,
        advantages: &Tensor,
        returns: &Tensor,
    ) -> TorshResult<(Tensor, Tensor, Tensor)> {
        // Actor loss
        let new_log_probs = self.actor.log_prob(states, actions)?;
        let ratio = new_log_probs.sub(old_log_probs)?.exp()?;
        let clipped_ratio = ratio.clamp(
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon,
        )?;

        let policy_loss1 = ratio.mul(&advantages)?;
        let policy_loss2 = clipped_ratio.mul(&advantages)?;
        let policy_loss = policy_loss1
            .minimum(&policy_loss2)?
            .mean(None, false)?
            .neg()?;

        // Critic loss
        let values = self.critic.forward(states)?;
        let value_loss = returns.sub(&values)?.pow(2.0)?.mean(None, false)?;

        // Entropy bonus (approximation for discrete actions)
        let entropy = if self.config.continuous_actions {
            // For continuous: 0.5 * log(2πe * σ²)
            let action_logits = self.actor.forward(states)?;
            let log_std = self.actor.log_std_head.as_ref().unwrap().forward(states)?;
            let constant_term = creation::tensor_scalar(
                0.5 * (2.0 * std::f32::consts::PI * std::f32::consts::E).ln(),
            )?;
            let entropy = log_std.add(&constant_term)?;
            entropy.mean(None, false)?
        } else {
            // For discrete: -Σ(p * log(p))
            let action_logits = self.actor.forward(states)?;
            let probs = Softmax::new(Some(1)).forward(&action_logits)?;
            let log_probs = LogSoftmax::new(Some(1)).forward(&action_logits)?;
            probs
                .mul(&log_probs)?
                .sum_dim(&[], false)?
                .mean(None, false)?
                .neg()?
        };

        Ok((policy_loss, value_loss, entropy))
    }
}

impl Module for PPO {
    fn forward(&self, input: &Tensor) -> TorshResult<Tensor> {
        // Forward pass through actor by default
        self.actor.forward(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        // Actor parameters
        let actor_params = self.actor.parameters();
        for (name, param) in actor_params {
            params.insert(format!("actor.{}", name), param);
        }

        // Critic parameters
        let critic_params = self.critic.parameters();
        for (name, param) in critic_params {
            params.insert(format!("critic.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.training
    }

    fn train(&mut self) {
        self.training = true;
        self.actor.train();
        self.critic.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.actor.eval();
        self.critic.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> TorshResult<()> {
        self.actor.to_device(device)?;
        self.critic.to_device(device)
    }
}

/// Configuration for A3C (Asynchronous Advantage Actor-Critic)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A3CConfig {
    /// State space dimension
    pub state_dim: usize,
    /// Action space dimension
    pub action_dim: usize,
    /// Whether actions are continuous
    pub continuous_actions: bool,
    /// Shared network hidden dimensions
    pub shared_hidden_dims: Vec<usize>,
    /// Actor head hidden dimensions
    pub actor_hidden_dims: Vec<usize>,
    /// Critic head hidden dimensions
    pub critic_hidden_dims: Vec<usize>,
    /// Activation function
    pub activation: String,
    /// Learning rate
    pub learning_rate: f32,
    /// Discount factor
    pub gamma: f32,
    /// Entropy regularization coefficient
    pub entropy_coef: f32,
    /// Value loss coefficient
    pub value_coef: f32,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: f32,
    /// Number of steps before update
    pub update_steps: usize,
}

impl Default for A3CConfig {
    fn default() -> Self {
        Self {
            state_dim: 4,
            action_dim: 2,
            continuous_actions: false,
            shared_hidden_dims: vec![256],
            actor_hidden_dims: vec![],
            critic_hidden_dims: vec![],
            activation: "relu".to_string(),
            learning_rate: 1e-4,
            gamma: 0.99,
            entropy_coef: 0.01,
            value_coef: 0.5,
            max_grad_norm: 5.0,
            update_steps: 5,
        }
    }
}

/// A3C model with shared feature extraction and separate actor/critic heads
#[derive(Debug)]
pub struct A3C {
    config: A3CConfig,
    /// Shared feature extraction layers
    shared_layers: Vec<Linear>,
    /// Actor head layers
    actor_layers: Vec<Linear>,
    /// Critic head layers
    critic_layers: Vec<Linear>,
    /// Actor output layer
    actor_head: Linear,
    /// Critic output layer
    critic_head: Linear,
    /// Log standard deviation for continuous actions
    log_std_head: Option<Linear>,
    training: bool,
}

impl A3C {
    pub fn new(config: A3CConfig) -> TorshResult<Self> {
        let mut shared_layers = Vec::new();
        let mut prev_dim = config.state_dim;

        // Build shared layers
        for &hidden_dim in &config.shared_hidden_dims {
            shared_layers.push(Linear::new(prev_dim, hidden_dim, true));
            prev_dim = hidden_dim;
        }

        let shared_output_dim = prev_dim;

        // Build actor head
        let mut actor_layers = Vec::new();
        let mut actor_dim = shared_output_dim;
        for &hidden_dim in &config.actor_hidden_dims {
            actor_layers.push(Linear::new(actor_dim, hidden_dim, true));
            actor_dim = hidden_dim;
        }
        let actor_head = Linear::new(actor_dim, config.action_dim, true);

        // Build critic head
        let mut critic_layers = Vec::new();
        let mut critic_dim = shared_output_dim;
        for &hidden_dim in &config.critic_hidden_dims {
            critic_layers.push(Linear::new(critic_dim, hidden_dim, true));
            critic_dim = hidden_dim;
        }
        let critic_head = Linear::new(critic_dim, 1, true);

        // Log std for continuous actions
        let log_std_head = if config.continuous_actions {
            Some(Linear::new(actor_dim, config.action_dim, true))
        } else {
            None
        };

        Ok(Self {
            config,
            shared_layers,
            actor_layers,
            critic_layers,
            actor_head,
            critic_head,
            log_std_head,
            training: true,
        })
    }

    /// Forward pass through shared layers
    fn forward_shared(&self, input: &Tensor) -> TorshResult<Tensor> {
        let mut x = input.clone();

        for layer in &self.shared_layers {
            x = layer.forward(&x)?;
            x = match self.config.activation.as_str() {
                "relu" => ReLU::new().forward(&x)?,
                "gelu" => GELU::new(false).forward(&x)?,
                "tanh" => Tanh::new().forward(&x)?,
                _ => ReLU::new().forward(&x)?,
            };
        }

        Ok(x)
    }

    /// Forward pass through actor head
    pub fn forward_actor(&self, shared_features: &Tensor) -> TorshResult<Tensor> {
        let mut x = shared_features.clone();

        for layer in &self.actor_layers {
            x = layer.forward(&x)?;
            x = match self.config.activation.as_str() {
                "relu" => ReLU::new().forward(&x)?,
                "gelu" => GELU::new(false).forward(&x)?,
                "tanh" => Tanh::new().forward(&x)?,
                _ => ReLU::new().forward(&x)?,
            };
        }

        self.actor_head.forward(&x)
    }

    /// Forward pass through critic head
    pub fn forward_critic(&self, shared_features: &Tensor) -> TorshResult<Tensor> {
        let mut x = shared_features.clone();

        for layer in &self.critic_layers {
            x = layer.forward(&x)?;
            x = match self.config.activation.as_str() {
                "relu" => ReLU::new().forward(&x)?,
                "gelu" => GELU::new(false).forward(&x)?,
                "tanh" => Tanh::new().forward(&x)?,
                _ => ReLU::new().forward(&x)?,
            };
        }

        self.critic_head.forward(&x)
    }

    /// Get action and value from state
    pub fn act_and_value(&self, state: &Tensor) -> TorshResult<(Tensor, Tensor, Tensor)> {
        let shared_features = self.forward_shared(state)?;
        let action_logits = self.forward_actor(&shared_features)?;
        let value = self.forward_critic(&shared_features)?;

        if self.config.continuous_actions {
            // Continuous actions
            let mean = action_logits;
            let log_std = self
                .log_std_head
                .as_ref()
                .unwrap()
                .forward(&shared_features)?;
            let std = log_std.exp()?;

            // Sample action
            let noise = torsh_tensor::creation::randn(mean.shape().dims())?;
            let action = mean.add(&std.mul(&noise)?)?;

            Ok((action, mean, value))
        } else {
            // Discrete actions
            let log_probs = LogSoftmax::new(Some(1)).forward(&action_logits)?;
            let probs = log_probs.exp()?;

            // Sample action (simplified multinomial)
            use scirs2_core::random::{Random, Rng};
            let mut rng = Random::seed(42);
            let random_val = rng.gen::<f32>();
            let cumprobs = probs.cumsum(1)?;
            let random_tensor =
                torsh_tensor::creation::full(&[probs.shape().dims()[0], 1], random_val)?;
            let bool_tensor = cumprobs.gt(&random_tensor)?;
            let bool_data = bool_tensor.to_vec()?;
            let i64_data: Vec<i64> = bool_data
                .iter()
                .map(|&x| if x { 1i64 } else { 0i64 })
                .collect();
            let i64_tensor = Tensor::<i64>::from_data(
                i64_data,
                bool_tensor.shape().dims().to_vec(),
                bool_tensor.device(),
            )?;
            let action = i64_tensor.sum_dim(&[1], false)?.to_f32_simd()?;

            Ok((action, action_logits, value))
        }
    }

    /// Compute A3C loss
    pub fn compute_loss(
        &self,
        states: &Tensor,
        actions: &Tensor,
        rewards: &Tensor,
        next_values: &Tensor,
        dones: &Tensor,
    ) -> TorshResult<(Tensor, Tensor, Tensor)> {
        let shared_features = self.forward_shared(states)?;
        let action_logits = self.forward_actor(&shared_features)?;
        let values = self.forward_critic(&shared_features)?;

        // Compute returns and advantages
        let gamma_next_values = next_values.mul_scalar(self.config.gamma)?;
        let ones_minus_dones = Tensor::ones_like(dones)?.sub(dones)?;
        let discounted_next = gamma_next_values.mul(&ones_minus_dones)?;
        let returns = rewards.add(&discounted_next)?;
        let advantages = returns.sub(&values)?;

        // Policy loss
        let log_probs = if self.config.continuous_actions {
            // Continuous actions: Gaussian log probability
            let mean = action_logits.clone();
            let log_std = self
                .log_std_head
                .as_ref()
                .unwrap()
                .forward(&shared_features)?;
            let std = log_std.exp()?;
            let var = std.pow(2.0)?;
            let diff_squared = actions.sub(&mean)?.pow(2.0)?;
            let div_by_var = diff_squared.div(&var)?;
            let log_var = var.log()?;
            let log_2pi = creation::tensor_scalar((2.0 * std::f32::consts::PI).ln())?;
            let sum_terms = div_by_var.add(&log_var)?.add(&log_2pi)?;
            sum_terms.mul_scalar(-0.5)?.sum_dim(&[1], false)?
        } else {
            // Discrete actions: categorical log probability
            let log_probs_all = LogSoftmax::new(Some(1)).forward(&action_logits)?;
            let actions_i64 = actions.to_dtype(DType::I64)?;
            let actions_f32: Vec<f32> = actions_i64.to_vec()?;
            let actions_data: Vec<i64> = actions_f32.into_iter().map(|x| x as i64).collect();
            let batch_size = actions_data.len();
            let actions_unsqueezed_data: Vec<i64> = actions_data.into_iter().map(|x| x).collect();
            let actions_unsqueezed = Tensor::from_data(
                actions_unsqueezed_data,
                vec![batch_size, 1],
                torsh_core::DeviceType::Cpu,
            )?;
            log_probs_all.gather(1, &actions_unsqueezed)?.squeeze(1)?
        };

        let advantages_detached = advantages.detach();
        let policy_loss = log_probs
            .mul(&advantages_detached)?
            .mean(None, false)?
            .neg()?;

        // Value loss
        let value_loss = advantages.pow(2.0)?.mean(None, false)?;

        // Entropy bonus
        let entropy = if self.config.continuous_actions {
            let log_std = self
                .log_std_head
                .as_ref()
                .unwrap()
                .forward(&shared_features)?;
            let constant_term = creation::tensor_scalar(
                0.5 * (2.0 * std::f32::consts::PI * std::f32::consts::E).ln(),
            )?;
            log_std.add(&constant_term)?.mean(None, false)
        } else {
            let probs = Softmax::new(Some(1)).forward(&action_logits)?;
            let log_probs_all = LogSoftmax::new(Some(1)).forward(&action_logits)?;
            probs
                .mul(&log_probs_all)?
                .sum_dim(&[1], false)?
                .mean(None, false)?
                .neg()
        }?;

        Ok((policy_loss, value_loss, entropy))
    }
}

impl Module for A3C {
    fn forward(&self, input: &Tensor) -> TorshResult<Tensor> {
        let shared_features = self.forward_shared(input)?;
        self.forward_actor(&shared_features)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        // Shared layer parameters
        for (i, layer) in self.shared_layers.iter().enumerate() {
            let layer_params = layer.parameters();
            for (name, param) in layer_params {
                params.insert(format!("shared_{}.{}", i, name), param);
            }
        }

        // Actor layer parameters
        for (i, layer) in self.actor_layers.iter().enumerate() {
            let layer_params = layer.parameters();
            for (name, param) in layer_params {
                params.insert(format!("actor_{}.{}", i, name), param);
            }
        }

        // Critic layer parameters
        for (i, layer) in self.critic_layers.iter().enumerate() {
            let layer_params = layer.parameters();
            for (name, param) in layer_params {
                params.insert(format!("critic_{}.{}", i, name), param);
            }
        }

        // Actor head parameters
        let actor_head_params = self.actor_head.parameters();
        for (name, param) in actor_head_params {
            params.insert(format!("actor_head.{}", name), param);
        }

        // Critic head parameters
        let critic_head_params = self.critic_head.parameters();
        for (name, param) in critic_head_params {
            params.insert(format!("critic_head.{}", name), param);
        }

        // Log std head parameters
        if let Some(ref log_std_head) = self.log_std_head {
            let log_std_params = log_std_head.parameters();
            for (name, param) in log_std_params {
                params.insert(format!("log_std_head.{}", name), param);
            }
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.training
    }

    fn train(&mut self) {
        self.training = true;
        for layer in &mut self.shared_layers {
            layer.train();
        }
        for layer in &mut self.actor_layers {
            layer.train();
        }
        for layer in &mut self.critic_layers {
            layer.train();
        }
        self.actor_head.train();
        self.critic_head.train();
        if let Some(ref mut log_std_head) = self.log_std_head {
            log_std_head.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        for layer in &mut self.shared_layers {
            layer.eval();
        }
        for layer in &mut self.actor_layers {
            layer.eval();
        }
        for layer in &mut self.critic_layers {
            layer.eval();
        }
        self.actor_head.eval();
        self.critic_head.eval();
        if let Some(ref mut log_std_head) = self.log_std_head {
            log_std_head.eval();
        }
    }

    fn to_device(&mut self, device: DeviceType) -> TorshResult<()> {
        for layer in &mut self.shared_layers {
            layer.to_device(device)?;
        }
        for layer in &mut self.actor_layers {
            layer.to_device(device)?;
        }
        for layer in &mut self.critic_layers {
            layer.to_device(device)?;
        }
        self.actor_head.to_device(device)?;
        self.critic_head.to_device(device)?;
        if let Some(ref mut log_std_head) = self.log_std_head {
            log_std_head.to_device(device)?;
        }
        Ok(())
    }
}

/// RL model types enum for unified interface
#[derive(Debug)]
pub enum RLModel {
    DQN(DQN),
    PPO(PPO),
    A3C(A3C),
}

impl Module for RLModel {
    fn forward(&self, input: &Tensor) -> TorshResult<Tensor> {
        match self {
            RLModel::DQN(model) => model.forward(input),
            RLModel::PPO(model) => model.forward(input),
            RLModel::A3C(model) => model.forward(input),
        }
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        match self {
            RLModel::DQN(model) => model.parameters(),
            RLModel::PPO(model) => model.parameters(),
            RLModel::A3C(model) => model.parameters(),
        }
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        match self {
            RLModel::DQN(model) => model.named_parameters(),
            RLModel::PPO(model) => model.named_parameters(),
            RLModel::A3C(model) => model.named_parameters(),
        }
    }

    fn training(&self) -> bool {
        match self {
            RLModel::DQN(model) => model.training(),
            RLModel::PPO(model) => model.training(),
            RLModel::A3C(model) => model.training(),
        }
    }

    fn train(&mut self) {
        match self {
            RLModel::DQN(model) => model.train(),
            RLModel::PPO(model) => model.train(),
            RLModel::A3C(model) => model.train(),
        }
    }

    fn eval(&mut self) {
        match self {
            RLModel::DQN(model) => model.eval(),
            RLModel::PPO(model) => model.eval(),
            RLModel::A3C(model) => model.eval(),
        }
    }

    fn to_device(&mut self, device: DeviceType) -> TorshResult<()> {
        match self {
            RLModel::DQN(model) => model.to_device(device),
            RLModel::PPO(model) => model.to_device(device),
            RLModel::A3C(model) => model.to_device(device),
        }
    }
}
