//! Reinforcement Learning Models
//!
//! This module provides implementations of popular RL algorithms and architectures
//! including DQN, Actor-Critic, PPO, and DDPG models.

use scirs2_core::random::{Random, Rng};
use std::collections::HashMap;
use torsh_core::error::{Result, TorshError};
use torsh_core::DeviceType;
use torsh_nn::prelude::*;
use torsh_tensor::stats::StatMode;
use torsh_tensor::Tensor;

/// Deep Q-Network (DQN) for discrete action spaces
#[derive(Debug)]
pub struct DQN {
    /// Feature extraction network
    pub feature_net: Sequential,
    /// Value head for Q-values
    pub value_head: Linear,
    /// Input dimensions
    pub input_dim: usize,
    /// Number of actions
    pub num_actions: usize,
    /// Hidden dimensions
    pub hidden_dims: Vec<usize>,
}

impl DQN {
    /// Create a new DQN model
    pub fn new(input_dim: usize, num_actions: usize, hidden_dims: Vec<usize>) -> Self {
        let mut feature_net = Sequential::new();

        let mut prev_dim = input_dim;
        for &hidden_dim in &hidden_dims {
            feature_net = feature_net
                .add(Linear::new(prev_dim, hidden_dim, true))
                .add(ReLU::new());
            prev_dim = hidden_dim;
        }

        let value_head = Linear::new(prev_dim, num_actions, true);

        Self {
            feature_net,
            value_head,
            input_dim,
            num_actions,
            hidden_dims,
        }
    }

    /// Get Q-values for given states
    pub fn q_values(&self, states: &Tensor) -> Result<Tensor> {
        let features = self.feature_net.forward(states)?;
        self.value_head.forward(&features)
    }

    /// Select action with epsilon-greedy policy
    pub fn select_action(&self, state: &Tensor, epsilon: f32) -> Result<usize> {
        let mut rng = Random::seed(42);
        if rng.gen::<f32>() < epsilon {
            // Random action
            Ok(rng.gen_range(0..self.num_actions))
        } else {
            // Greedy action
            let q_vals = self.q_values(state)?;
            let q_data = q_vals.to_vec()?;
            let best_action = q_data
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            Ok(best_action)
        }
    }
}

impl Module for DQN {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        self.q_values(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.feature_net.named_parameters());
        params.extend(self.value_head.named_parameters());
        params
    }

    fn load_state_dict(
        &mut self,
        state_dict: &HashMap<String, Tensor<f32>>,
        strict: bool,
    ) -> Result<()> {
        self.feature_net.load_state_dict(state_dict, strict)?;
        self.value_head.load_state_dict(state_dict, strict)?;
        Ok(())
    }
}

/// Actor-Critic Network for continuous control
#[derive(Debug)]
pub struct ActorCritic {
    /// Actor network (policy)
    pub actor: Sequential,
    /// Critic network (value function)
    pub critic: Sequential,
    /// Input dimensions
    pub state_dim: usize,
    /// Action dimensions
    pub action_dim: usize,
    /// Hidden dimensions for actor
    pub actor_hidden: Vec<usize>,
    /// Hidden dimensions for critic
    pub critic_hidden: Vec<usize>,
}

impl ActorCritic {
    /// Create a new Actor-Critic model
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        actor_hidden: Vec<usize>,
        critic_hidden: Vec<usize>,
    ) -> Self {
        // Build actor network
        let mut actor = Sequential::new();
        let mut prev_dim = state_dim;
        for &hidden_dim in &actor_hidden {
            actor = actor
                .add(Linear::new(prev_dim, hidden_dim, true))
                .add(ReLU::new());
            prev_dim = hidden_dim;
        }
        actor = actor
            .add(Linear::new(prev_dim, action_dim, true))
            .add(Tanh::new()); // Actions typically bounded [-1, 1]

        // Build critic network
        let mut critic = Sequential::new();
        prev_dim = state_dim;
        for &hidden_dim in &critic_hidden {
            critic = critic
                .add(Linear::new(prev_dim, hidden_dim, true))
                .add(ReLU::new());
            prev_dim = hidden_dim;
        }
        critic = critic.add(Linear::new(prev_dim, 1, true)); // Single value output

        Self {
            actor,
            critic,
            state_dim,
            action_dim,
            actor_hidden,
            critic_hidden,
        }
    }

    /// Get action from actor network
    pub fn get_action(&self, state: &Tensor) -> Result<Tensor> {
        self.actor.forward(state)
    }

    /// Get value from critic network
    pub fn get_value(&self, state: &Tensor) -> Result<Tensor> {
        self.critic.forward(state)
    }

    /// Evaluate action and state
    pub fn evaluate(&self, state: &Tensor, _action: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        // Get action probabilities and value
        let action_probs = self.get_action(state)?;
        let value = self.get_value(state)?;

        // Compute log probabilities (simplified for continuous actions)
        // In practice, this would involve proper probability distributions
        let log_probs = action_probs.log_softmax(-1)?;

        Ok((action_probs, log_probs, value))
    }
}

impl Module for ActorCritic {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        // Forward pass returns both actor and critic outputs
        let action = self.actor.forward(input)?;
        let value = self.critic.forward(input)?;

        // Concatenate outputs (in practice might return a tuple)
        let action_binding = action.shape();
        let action_shape = action_binding.dims();
        let value_binding = value.shape();
        let _value_shape = value_binding.dims();
        let batch_size = action_shape[0];

        // Create combined output tensor
        let action_data = action.to_vec()?;
        let value_data = value.to_vec()?;

        let mut combined_data = action_data;
        combined_data.extend(value_data);

        use torsh_tensor::creation::from_vec;
        from_vec(
            combined_data,
            &[batch_size, self.action_dim + 1],
            input.device(),
        )
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.actor.named_parameters());
        params.extend(self.critic.named_parameters());
        params
    }

    fn load_state_dict(
        &mut self,
        state_dict: &HashMap<String, Tensor<f32>>,
        strict: bool,
    ) -> Result<()> {
        self.actor.load_state_dict(state_dict, strict)?;
        self.critic.load_state_dict(state_dict, strict)?;
        Ok(())
    }
}

/// Proximal Policy Optimization (PPO) Agent
#[derive(Debug)]
pub struct PPOAgent {
    /// Actor-Critic network
    pub network: ActorCritic,
    /// Clip parameter for PPO
    pub clip_param: f32,
    /// Value function coefficient
    pub value_coeff: f32,
    /// Entropy coefficient
    pub entropy_coeff: f32,
    /// GAE lambda
    pub gae_lambda: f32,
    /// Discount factor
    pub gamma: f32,
}

impl PPOAgent {
    /// Create a new PPO agent
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        actor_hidden: Vec<usize>,
        critic_hidden: Vec<usize>,
    ) -> Self {
        Self {
            network: ActorCritic::new(state_dim, action_dim, actor_hidden, critic_hidden),
            clip_param: 0.2,
            value_coeff: 0.5,
            entropy_coeff: 0.01,
            gae_lambda: 0.95,
            gamma: 0.99,
        }
    }

    /// Compute PPO loss
    pub fn compute_loss(
        &self,
        states: &Tensor,
        actions: &Tensor,
        old_log_probs: &Tensor,
        returns: &Tensor,
        advantages: &Tensor,
    ) -> Result<Tensor> {
        let (action_probs, log_probs, values) = self.network.evaluate(states, actions)?;

        // Policy loss with clipping
        let ratio = (&log_probs - old_log_probs).exp()?;
        let advantages_norm = self.normalize_advantages(advantages)?;

        let policy_loss1 = &ratio * &advantages_norm;
        let ratio_clipped = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)?;
        let policy_loss2 = &ratio_clipped * &advantages_norm;
        let policy_loss = policy_loss1
            .minimum(&policy_loss2)?
            .mean(None, false)?
            .neg()?;

        // Value loss
        let value_loss = (&values - returns).pow(2.0)?.mean(None, false)?;

        // Entropy loss (simplified)
        let entropy = (&action_probs * &action_probs.log()?)
            .sum()?
            .mean(None, false)?
            .neg()?;

        // Total loss
        let value_weighted = value_loss.mul_scalar(self.value_coeff)?;
        let entropy_weighted = entropy.mul_scalar(self.entropy_coeff)?;
        let total_loss = (&policy_loss + &value_weighted).sub(&entropy_weighted)?;

        Ok(total_loss)
    }

    /// Normalize advantages
    fn normalize_advantages(&self, advantages: &Tensor) -> Result<Tensor> {
        let mean = advantages.mean(None, false)?;
        let std = advantages.std(None, false, StatMode::Sample)?;
        let eps = 1e-8;

        let numerator = advantages - &mean;
        let denominator = std.add_scalar(eps)?;
        Ok(numerator.div(&denominator)?)
    }

    /// Compute GAE advantages
    pub fn compute_gae_advantages(
        &self,
        rewards: &[f32],
        values: &[f32],
        dones: &[bool],
        next_value: f32,
    ) -> Vec<f32> {
        let mut advantages = vec![0.0; rewards.len()];
        let mut gae = 0.0;

        for i in (0..rewards.len()).rev() {
            let delta =
                rewards[i] + self.gamma * next_value * (1.0 - dones[i] as u8 as f32) - values[i];
            gae = delta + self.gamma * self.gae_lambda * gae * (1.0 - dones[i] as u8 as f32);
            advantages[i] = gae;
        }

        advantages
    }
}

/// Deep Deterministic Policy Gradient (DDPG) for continuous control
pub struct DDPGAgent {
    /// Actor network
    pub actor: Sequential,
    /// Critic network
    pub critic: Sequential,
    /// Target actor network
    pub target_actor: Sequential,
    /// Target critic network
    pub target_critic: Sequential,
    /// State dimension
    pub state_dim: usize,
    /// Action dimension
    pub action_dim: usize,
    /// Tau for soft updates
    pub tau: f32,
}

impl std::fmt::Debug for DDPGAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DDPGAgent")
            .field("state_dim", &self.state_dim)
            .field("action_dim", &self.action_dim)
            .field("tau", &self.tau)
            .finish()
    }
}

impl DDPGAgent {
    /// Create a new DDPG agent
    pub fn new(state_dim: usize, action_dim: usize, hidden_dims: Vec<usize>) -> Self {
        let actor = Self::create_actor_network(state_dim, action_dim, &hidden_dims);
        let critic = Self::create_critic_network(state_dim, action_dim, &hidden_dims);

        // Create target networks (separate instances with same architecture)
        let target_actor = Self::create_actor_network(state_dim, action_dim, &hidden_dims);
        let target_critic = Self::create_critic_network(state_dim, action_dim, &hidden_dims);

        Self {
            actor,
            critic,
            target_actor,
            target_critic,
            state_dim,
            action_dim,
            tau: 0.005,
        }
    }

    /// Create actor network architecture
    fn create_actor_network(
        state_dim: usize,
        action_dim: usize,
        hidden_dims: &[usize],
    ) -> Sequential {
        let mut actor = Sequential::new();
        let mut prev_dim = state_dim;
        for &hidden_dim in hidden_dims {
            actor = actor
                .add(Linear::new(prev_dim, hidden_dim, true))
                .add(ReLU::new());
            prev_dim = hidden_dim;
        }
        actor = actor
            .add(Linear::new(prev_dim, action_dim, true))
            .add(Tanh::new());
        actor
    }

    /// Create critic network architecture  
    fn create_critic_network(
        state_dim: usize,
        action_dim: usize,
        hidden_dims: &[usize],
    ) -> Sequential {
        let mut critic = Sequential::new();
        let mut prev_dim = state_dim + action_dim;
        for &hidden_dim in hidden_dims {
            critic = critic
                .add(Linear::new(prev_dim, hidden_dim, true))
                .add(ReLU::new());
            prev_dim = hidden_dim;
        }
        critic = critic.add(Linear::new(prev_dim, 1, true));
        critic
    }

    /// Get action from actor
    pub fn get_action(&self, state: &Tensor) -> Result<Tensor> {
        self.actor.forward(state)
    }

    /// Get Q-value from critic
    pub fn get_q_value(&self, state: &Tensor, action: &Tensor) -> Result<Tensor> {
        // Concatenate state and action
        let state_data = state.to_vec()?;
        let action_data = action.to_vec()?;
        let batch_size = state.shape().dims()[0];

        let mut combined_data = state_data;
        combined_data.extend(action_data);

        let state_action = Tensor::from_data(
            combined_data,
            vec![batch_size, self.state_dim + self.action_dim],
            state.device(),
        )?;

        self.critic.forward(&state_action)
    }

    /// Soft update target networks
    pub fn soft_update_targets(&mut self) -> Result<()> {
        // In practice, this would update target network parameters
        // target_param = tau * param + (1 - tau) * target_param
        // This is a simplified placeholder
        Ok(())
    }

    /// Compute actor loss
    pub fn actor_loss(&self, states: &Tensor) -> Result<Tensor> {
        let actions = self.get_action(states)?;
        let q_values = self.get_q_value(states, &actions)?;
        Ok(q_values.mean(None, false)?.neg()?)
    }

    /// Compute critic loss
    pub fn critic_loss(
        &self,
        states: &Tensor,
        actions: &Tensor,
        rewards: &Tensor,
        next_states: &Tensor,
        dones: &Tensor,
    ) -> Result<Tensor> {
        // Current Q-values
        let current_q = self.get_q_value(states, actions)?;

        // Target Q-values
        let next_actions = self.target_actor.forward(next_states)?;
        let next_q = self.get_q_value(next_states, &next_actions)?;
        let ones_tensor = torsh_tensor::creation::ones(dones.shape().dims())?;
        let discount_mask = &ones_tensor - dones;
        let discounted_next_q = &next_q * &discount_mask;
        let target_q = rewards + &discounted_next_q;

        // MSE loss
        let loss_tensor = &current_q - &target_q;
        loss_tensor.pow(2.0)?.mean(None, false)
    }
}

/// Replay Buffer for experience replay
#[derive(Debug)]
pub struct ReplayBuffer {
    /// Buffer capacity
    pub capacity: usize,
    /// Current size
    pub size: usize,
    /// Current position
    pub position: usize,
    /// State buffer
    pub states: Vec<Vec<f32>>,
    /// Action buffer
    pub actions: Vec<Vec<f32>>,
    /// Reward buffer
    pub rewards: Vec<f32>,
    /// Next state buffer
    pub next_states: Vec<Vec<f32>>,
    /// Done buffer
    pub dones: Vec<bool>,
}

impl ReplayBuffer {
    /// Create a new replay buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            size: 0,
            position: 0,
            states: Vec::with_capacity(capacity),
            actions: Vec::with_capacity(capacity),
            rewards: Vec::with_capacity(capacity),
            next_states: Vec::with_capacity(capacity),
            dones: Vec::with_capacity(capacity),
        }
    }

    /// Add experience to buffer
    pub fn push(
        &mut self,
        state: Vec<f32>,
        action: Vec<f32>,
        reward: f32,
        next_state: Vec<f32>,
        done: bool,
    ) {
        if self.size < self.capacity {
            self.states.push(state);
            self.actions.push(action);
            self.rewards.push(reward);
            self.next_states.push(next_state);
            self.dones.push(done);
            self.size += 1;
        } else {
            self.states[self.position] = state;
            self.actions[self.position] = action;
            self.rewards[self.position] = reward;
            self.next_states[self.position] = next_state;
            self.dones[self.position] = done;
        }

        self.position = (self.position + 1) % self.capacity;
    }

    /// Sample batch from buffer
    pub fn sample(&self, batch_size: usize) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        if self.size < batch_size {
            return Err(TorshError::InvalidArgument(
                "Not enough samples in buffer".to_string(),
            ));
        }

        let mut indices = Vec::new();
        let mut rng = Random::seed(42);
        for _ in 0..batch_size {
            indices.push(rng.gen_range(0..self.size));
        }

        let state_dim = self.states[0].len();
        let action_dim = self.actions[0].len();

        let mut batch_states = Vec::new();
        let mut batch_actions = Vec::new();
        let mut batch_rewards = Vec::new();
        let mut batch_next_states = Vec::new();
        let mut batch_dones = Vec::new();

        for &idx in &indices {
            batch_states.extend(&self.states[idx]);
            batch_actions.extend(&self.actions[idx]);
            batch_rewards.push(self.rewards[idx]);
            batch_next_states.extend(&self.next_states[idx]);
            batch_dones.push(if self.dones[idx] { 1.0 } else { 0.0 });
        }

        let states = Tensor::from_data(batch_states, vec![batch_size, state_dim], DeviceType::Cpu)?;
        let actions =
            Tensor::from_data(batch_actions, vec![batch_size, action_dim], DeviceType::Cpu)?;
        let rewards = Tensor::from_data(batch_rewards, vec![batch_size, 1], DeviceType::Cpu)?;
        let next_states = Tensor::from_data(
            batch_next_states,
            vec![batch_size, state_dim],
            DeviceType::Cpu,
        )?;
        let dones = Tensor::from_data(batch_dones, vec![batch_size, 1], DeviceType::Cpu)?;

        Ok((states, actions, rewards, next_states, dones))
    }

    /// Check if buffer has enough samples
    pub fn can_sample(&self, batch_size: usize) -> bool {
        self.size >= batch_size
    }
}

/// Factory functions for creating popular RL models
pub mod factory {
    use super::*;

    /// Create a DQN for Atari games
    pub fn atari_dqn() -> DQN {
        // Standard DQN architecture for Atari
        // Input: 84x84x4 preprocessed frames -> 28224 features
        DQN::new(28224, 18, vec![512, 512]) // 18 actions for most Atari games
    }

    /// Create a DQN for CartPole
    pub fn cartpole_dqn() -> DQN {
        DQN::new(4, 2, vec![128, 128]) // 4 state features, 2 actions
    }

    /// Create an Actor-Critic for continuous control
    pub fn continuous_actor_critic(state_dim: usize, action_dim: usize) -> ActorCritic {
        ActorCritic::new(
            state_dim,
            action_dim,
            vec![400, 300], // Actor hidden dims
            vec![400, 300], // Critic hidden dims
        )
    }

    /// Create a PPO agent for MuJoCo environments
    pub fn mujoco_ppo(state_dim: usize, action_dim: usize) -> PPOAgent {
        PPOAgent::new(
            state_dim,
            action_dim,
            vec![64, 64], // Actor hidden dims
            vec![64, 64], // Critic hidden dims
        )
    }

    /// Create a DDPG agent for continuous control
    pub fn ddpg_agent(state_dim: usize, action_dim: usize) -> DDPGAgent {
        DDPGAgent::new(state_dim, action_dim, vec![400, 300])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::*;

    #[test]
    fn test_dqn_creation() {
        let dqn = DQN::new(4, 2, vec![64, 64]);
        assert_eq!(dqn.input_dim, 4);
        assert_eq!(dqn.num_actions, 2);
    }

    #[test]
    fn test_actor_critic_creation() {
        let ac = ActorCritic::new(8, 4, vec![64, 64], vec![64, 64]);
        assert_eq!(ac.state_dim, 8);
        assert_eq!(ac.action_dim, 4);
    }

    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(100);

        buffer.push(
            vec![1.0, 2.0, 3.0],
            vec![0.5],
            1.0,
            vec![1.1, 2.1, 3.1],
            false,
        );

        assert_eq!(buffer.size, 1);
        assert!(buffer.can_sample(1));
        assert!(!buffer.can_sample(2));
    }

    #[test]
    fn test_factory_functions() {
        let cartpole_dqn = factory::cartpole_dqn();
        assert_eq!(cartpole_dqn.input_dim, 4);
        assert_eq!(cartpole_dqn.num_actions, 2);

        let atari_dqn = factory::atari_dqn();
        assert_eq!(atari_dqn.input_dim, 28224);
        assert_eq!(atari_dqn.num_actions, 18);
    }
}
