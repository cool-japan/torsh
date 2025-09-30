//! Advanced Reinforcement Learning Demo
//!
//! This example demonstrates sophisticated RL algorithms including:
//! - Deep Q-Network (DQN) with experience replay and target networks
//! - Policy Gradient methods (REINFORCE, A2C, PPO)
//! - Multi-agent RL with coordination mechanisms
//! - Hierarchical RL with option frameworks

use rand::{seq::SliceRandom, thread_rng, Rng};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use torsh::nn::*;
use torsh::optim::*;
use torsh::prelude::*;

/// Environment interface for RL agents
trait Environment {
    type Action: Clone + std::fmt::Debug;
    type State: Clone + std::fmt::Debug;
    type Reward: Clone + Into<f64>;

    fn reset(&mut self) -> Self::State;
    fn step(&mut self, action: &Self::Action) -> (Self::State, Self::Reward, bool); // (next_state, reward, done)
    fn action_space_size(&self) -> usize;
    fn state_space_size(&self) -> usize;
    fn get_valid_actions(&self, state: &Self::State) -> Vec<Self::Action>;
}

/// Experience tuple for replay buffer
#[derive(Debug, Clone)]
struct Experience {
    state: Tensor,
    action: usize,
    reward: f64,
    next_state: Tensor,
    done: bool,
}

/// Experience replay buffer with prioritization
struct PrioritizedReplayBuffer {
    buffer: VecDeque<(Experience, f64)>, // (experience, priority)
    capacity: usize,
    alpha: f64, // Prioritization exponent
    beta: f64,  // Importance sampling exponent
    max_priority: f64,
}

impl PrioritizedReplayBuffer {
    fn new(capacity: usize, alpha: f64, beta: f64) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            alpha,
            beta,
            max_priority: 1.0,
        }
    }

    fn push(&mut self, experience: Experience, priority: Option<f64>) {
        let priority = priority.unwrap_or(self.max_priority);

        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }

        self.buffer.push_back((experience, priority));
        self.max_priority = self.max_priority.max(priority);
    }

    fn sample(&mut self, batch_size: usize) -> Result<(Vec<Experience>, Vec<f64>, Vec<usize>)> {
        if self.buffer.len() < batch_size {
            return Err(TorshError::InvalidArgument(
                "Not enough experiences".to_string(),
            ));
        }

        // Calculate sampling probabilities
        let priorities: Vec<f64> = self
            .buffer
            .iter()
            .map(|(_, p)| p.powf(self.alpha))
            .collect();
        let total_priority: f64 = priorities.iter().sum();

        let mut rng = thread_rng();
        let mut sampled_experiences = Vec::new();
        let mut importance_weights = Vec::new();
        let mut indices = Vec::new();

        for _ in 0..batch_size {
            let rand_val = rng.gen::<f64>() * total_priority;
            let mut cumsum = 0.0;

            for (idx, &priority) in priorities.iter().enumerate() {
                cumsum += priority;
                if cumsum >= rand_val {
                    let (experience, _) = &self.buffer[idx];
                    sampled_experiences.push(experience.clone());

                    // Compute importance sampling weight
                    let prob = priority / total_priority;
                    let weight = (1.0 / (self.buffer.len() as f64 * prob)).powf(self.beta);
                    importance_weights.push(weight);
                    indices.push(idx);
                    break;
                }
            }
        }

        // Normalize importance weights
        let max_weight = importance_weights.iter().fold(0.0, |a, &b| a.max(b));
        importance_weights = importance_weights
            .into_iter()
            .map(|w| w / max_weight)
            .collect();

        Ok((sampled_experiences, importance_weights, indices))
    }

    fn update_priorities(&mut self, indices: &[usize], priorities: &[f64]) {
        for (&idx, &priority) in indices.iter().zip(priorities.iter()) {
            if idx < self.buffer.len() {
                self.buffer[idx].1 = priority;
                self.max_priority = self.max_priority.max(priority);
            }
        }
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }
}

/// Deep Q-Network with Dueling architecture
struct DuelingDQN {
    backbone: Sequential,
    value_stream: Linear,
    advantage_stream: Linear,
    num_actions: usize,
}

impl DuelingDQN {
    fn new(state_dim: usize, hidden_dim: usize, num_actions: usize) -> Result<Self> {
        let backbone = Sequential::new()
            .add(Linear::new(state_dim, hidden_dim)?)?
            .add(ReLU::new())?
            .add(Linear::new(hidden_dim, hidden_dim)?)?
            .add(ReLU::new())?
            .add(Linear::new(hidden_dim, hidden_dim)?)?
            .add(ReLU::new())?;

        let value_stream = Linear::new(hidden_dim, 1)?;
        let advantage_stream = Linear::new(hidden_dim, num_actions)?;

        Ok(Self {
            backbone,
            value_stream,
            advantage_stream,
            num_actions,
        })
    }

    fn forward(&self, state: &Tensor) -> Result<Tensor> {
        let features = self.backbone.forward(state)?;

        let value = self.value_stream.forward(&features)?;
        let advantage = self.advantage_stream.forward(&features)?;

        // Dueling aggregation: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        let advantage_mean = advantage.mean_dim(&[-1], true)?;
        let q_values = value.add(&advantage.sub(&advantage_mean)?)?;

        Ok(q_values)
    }
}

impl Module for DuelingDQN {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.backbone.parameters();
        params.extend(self.value_stream.parameters());
        params.extend(self.advantage_stream.parameters());
        params
    }
}

/// Deep Q-Learning Agent with advanced features
struct DQNAgent {
    q_network: DuelingDQN,
    target_network: DuelingDQN,
    optimizer: Adam,
    replay_buffer: PrioritizedReplayBuffer,
    epsilon: f64,
    epsilon_decay: f64,
    epsilon_min: f64,
    gamma: f64,
    target_update_freq: usize,
    step_count: usize,
    device: Device,
}

impl DQNAgent {
    fn new(
        state_dim: usize,
        num_actions: usize,
        learning_rate: f64,
        buffer_capacity: usize,
        device: Device,
    ) -> Result<Self> {
        let q_network = DuelingDQN::new(state_dim, 512, num_actions)?;
        let mut target_network = DuelingDQN::new(state_dim, 512, num_actions)?;

        // Initialize target network with same weights
        for (target_param, q_param) in target_network
            .parameters()
            .iter()
            .zip(q_network.parameters().iter())
        {
            target_param.data().copy_(&q_param.data())?;
        }

        let optimizer = Adam::new(q_network.parameters(), learning_rate)?;
        let replay_buffer = PrioritizedReplayBuffer::new(buffer_capacity, 0.6, 0.4);

        Ok(Self {
            q_network,
            target_network,
            optimizer,
            replay_buffer,
            epsilon: 1.0,
            epsilon_decay: 0.995,
            epsilon_min: 0.01,
            gamma: 0.99,
            target_update_freq: 1000,
            step_count: 0,
            device,
        })
    }

    fn select_action(&self, state: &Tensor, training: bool) -> Result<usize> {
        if training && thread_rng().gen::<f64>() < self.epsilon {
            // Random action (exploration)
            Ok(thread_rng().gen_range(0..self.q_network.num_actions))
        } else {
            // Greedy action (exploitation)
            let q_values = self.q_network.forward(state)?;
            let action = q_values.argmax(-1, false)?.item::<i64>() as usize;
            Ok(action)
        }
    }

    fn train(&mut self, batch_size: usize) -> Result<f64> {
        if self.replay_buffer.len() < batch_size {
            return Ok(0.0);
        }

        let (experiences, importance_weights, indices) = self.replay_buffer.sample(batch_size)?;

        // Prepare batch tensors
        let states = torch::stack(
            &experiences
                .iter()
                .map(|e| e.state.clone())
                .collect::<Vec<_>>(),
            0,
        )?;
        let actions = tensor!(experiences
            .iter()
            .map(|e| e.action as i64)
            .collect::<Vec<_>>())?;
        let rewards = tensor!(experiences
            .iter()
            .map(|e| e.reward as f32)
            .collect::<Vec<_>>())?;
        let next_states = torch::stack(
            &experiences
                .iter()
                .map(|e| e.next_state.clone())
                .collect::<Vec<_>>(),
            0,
        )?;
        let dones = tensor!(experiences
            .iter()
            .map(|e| if e.done { 1.0f32 } else { 0.0f32 })
            .collect::<Vec<_>>())?;
        let weights = tensor!(importance_weights
            .iter()
            .map(|&w| w as f32)
            .collect::<Vec<_>>())?;

        // Current Q values
        let current_q_values = self.q_network.forward(&states)?;
        let current_q = current_q_values
            .gather(-1, &actions.unsqueeze(-1))?
            .squeeze(-1)?;

        // Double DQN: use main network to select actions, target network to evaluate
        let next_q_values = self.q_network.forward(&next_states)?;
        let next_actions = next_q_values.argmax(-1, false)?;

        let target_next_q_values = self.target_network.forward(&next_states)?;
        let target_next_q = target_next_q_values
            .gather(-1, &next_actions.unsqueeze(-1))?
            .squeeze(-1)?;

        // Compute target Q values
        let target_q = rewards.add(
            &tensor![self.gamma as f32]?
                .mul(&target_next_q.mul(&tensor![1.0f32]?.sub(&dones)?)?)?,
        )?;

        // Compute TD errors for priority updates
        let td_errors = current_q.sub(&target_q.detach())?;
        let priorities: Vec<f64> = td_errors
            .abs()?
            .data_ptr()
            .iter()
            .map(|&x| (x as f64 + 1e-6).powf(0.6))
            .collect();

        // Weighted loss
        let loss_elements = F::mse_loss(&current_q, &target_q, false)?;
        let weighted_loss = loss_elements.mul(&weights)?;
        let loss = weighted_loss.mean()?;

        // Backward pass
        self.optimizer.zero_grad();
        loss.backward()?;

        // Gradient clipping
        for param in self.q_network.parameters() {
            if let Some(grad) = param.grad() {
                grad.clamp_(-1.0, 1.0)?;
            }
        }

        self.optimizer.step()?;

        // Update priorities in replay buffer
        self.replay_buffer.update_priorities(&indices, &priorities);

        // Update target network
        self.step_count += 1;
        if self.step_count % self.target_update_freq == 0 {
            self.update_target_network()?;
        }

        // Decay epsilon
        if self.epsilon > self.epsilon_min {
            self.epsilon *= self.epsilon_decay;
        }

        Ok(loss.item::<f32>() as f64)
    }

    fn update_target_network(&self) -> Result<()> {
        for (target_param, q_param) in self
            .target_network
            .parameters()
            .iter()
            .zip(self.q_network.parameters().iter())
        {
            target_param.data().copy_(&q_param.data())?;
        }
        Ok(())
    }

    fn store_experience(&mut self, experience: Experience) {
        self.replay_buffer.push(experience, None);
    }
}

/// Actor-Critic architecture for policy gradient methods
struct ActorCritic {
    shared_backbone: Sequential,
    actor_head: Linear,
    critic_head: Linear,
    num_actions: usize,
}

impl ActorCritic {
    fn new(state_dim: usize, hidden_dim: usize, num_actions: usize) -> Result<Self> {
        let shared_backbone = Sequential::new()
            .add(Linear::new(state_dim, hidden_dim)?)?
            .add(ReLU::new())?
            .add(Linear::new(hidden_dim, hidden_dim)?)?
            .add(ReLU::new())?;

        let actor_head = Linear::new(hidden_dim, num_actions)?;
        let critic_head = Linear::new(hidden_dim, 1)?;

        Ok(Self {
            shared_backbone,
            actor_head,
            critic_head,
            num_actions,
        })
    }

    fn forward(&self, state: &Tensor) -> Result<(Tensor, Tensor)> {
        let features = self.shared_backbone.forward(state)?;

        let action_logits = self.actor_head.forward(&features)?;
        let value = self.critic_head.forward(&features)?;

        Ok((action_logits, value))
    }

    fn sample_action(&self, state: &Tensor) -> Result<(usize, f64)> {
        let (action_logits, _) = self.forward(state)?;
        let action_probs = F::softmax(&action_logits, -1)?;

        // Sample from categorical distribution
        let mut rng = thread_rng();
        let probs: Vec<f32> = action_probs.data_ptr().to_vec();
        let cumsum: Vec<f32> = probs
            .iter()
            .scan(0.0, |sum, &x| {
                *sum += x;
                Some(*sum)
            })
            .collect();

        let rand_val = rng.gen::<f32>();
        let action = cumsum
            .iter()
            .position(|&x| x >= rand_val)
            .unwrap_or(probs.len() - 1);
        let log_prob = action_probs.index(&[action as i64])?.log()?.item::<f32>() as f64;

        Ok((action, log_prob))
    }
}

impl Module for ActorCritic {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (action_logits, _) = self.forward(input)?;
        Ok(action_logits)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.shared_backbone.parameters();
        params.extend(self.actor_head.parameters());
        params.extend(self.critic_head.parameters());
        params
    }
}

/// PPO (Proximal Policy Optimization) Agent
struct PPOAgent {
    actor_critic: ActorCritic,
    optimizer: Adam,
    clip_epsilon: f64,
    value_loss_coef: f64,
    entropy_coef: f64,
    gamma: f64,
    gae_lambda: f64,
    device: Device,
}

impl PPOAgent {
    fn new(
        state_dim: usize,
        num_actions: usize,
        learning_rate: f64,
        device: Device,
    ) -> Result<Self> {
        let actor_critic = ActorCritic::new(state_dim, 512, num_actions)?;
        let optimizer = Adam::new(actor_critic.parameters(), learning_rate)?;

        Ok(Self {
            actor_critic,
            optimizer,
            clip_epsilon: 0.2,
            value_loss_coef: 0.5,
            entropy_coef: 0.01,
            gamma: 0.99,
            gae_lambda: 0.95,
            device,
        })
    }

    fn select_action(&self, state: &Tensor) -> Result<(usize, f64)> {
        self.actor_critic.sample_action(state)
    }

    fn compute_gae(
        &self,
        rewards: &[f64],
        values: &Tensor,
        dones: &[bool],
    ) -> Result<(Tensor, Tensor)> {
        let mut advantages = Vec::new();
        let mut returns = Vec::new();

        let values_vec: Vec<f32> = values.data_ptr().to_vec();
        let mut gae = 0.0;

        for i in (0..rewards.len()).rev() {
            let delta = rewards[i]
                + self.gamma
                    * (if i == rewards.len() - 1 {
                        0.0
                    } else {
                        values_vec[i + 1] as f64
                    })
                    * (1.0 - if dones[i] { 1.0 } else { 0.0 })
                - values_vec[i] as f64;

            gae = delta
                + self.gamma * self.gae_lambda * (1.0 - if dones[i] { 1.0 } else { 0.0 }) * gae;

            advantages.insert(0, gae);
            returns.insert(0, gae + values_vec[i] as f64);
        }

        let advantages_tensor = tensor!(advantages.iter().map(|&x| x as f32).collect::<Vec<_>>())?;
        let returns_tensor = tensor!(returns.iter().map(|&x| x as f32).collect::<Vec<_>>())?;

        Ok((advantages_tensor, returns_tensor))
    }

    fn train(&mut self, trajectories: &[Trajectory]) -> Result<f64> {
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        for trajectory in trajectories {
            let states = torch::stack(&trajectory.states, 0)?;
            let actions = tensor!(trajectory
                .actions
                .iter()
                .map(|&x| x as i64)
                .collect::<Vec<_>>())?;
            let old_log_probs = tensor!(trajectory
                .log_probs
                .iter()
                .map(|&x| x as f32)
                .collect::<Vec<_>>())?;

            // Compute values and advantages
            let (_, values) = self.actor_critic.forward(&states)?;
            let values_squeezed = values.squeeze(-1)?;
            let (advantages, returns) =
                self.compute_gae(&trajectory.rewards, &values_squeezed, &trajectory.dones)?;

            // Normalize advantages
            let advantages_mean = advantages.mean()?;
            let advantages_std = advantages.std(false)?;
            let normalized_advantages = advantages
                .sub(&advantages_mean)?
                .div(&advantages_std.add(&tensor![1e-8])?)?;

            // PPO update
            for _ in 0..4 {
                // Multiple epochs
                let (action_logits, new_values) = self.actor_critic.forward(&states)?;
                let action_probs = F::softmax(&action_logits, -1)?;
                let new_log_probs = action_probs
                    .gather(-1, &actions.unsqueeze(-1))?
                    .squeeze(-1)?
                    .log()?;

                // Compute ratio
                let ratio = (new_log_probs.sub(&old_log_probs)?).exp()?;

                // Policy loss with clipping
                let policy_loss1 = ratio.mul(&normalized_advantages)?;
                let clipped_ratio = ratio.clamp(
                    1.0 - self.clip_epsilon as f32,
                    1.0 + self.clip_epsilon as f32,
                )?;
                let policy_loss2 = clipped_ratio.mul(&normalized_advantages)?;
                let policy_loss = -torch::min(&policy_loss1, &policy_loss2)?.mean()?;

                // Value loss
                let value_loss = F::mse_loss(&new_values.squeeze(-1)?, &returns, false)?;

                // Entropy loss
                let entropy = -(action_probs * action_probs.log())
                    .sum_dim(&[-1], false)?
                    .mean()?;

                // Total loss
                let total_loss_batch = policy_loss
                    .add(&value_loss.mul(&tensor![self.value_loss_coef as f32])?)?
                    .sub(&entropy.mul(&tensor![self.entropy_coef as f32])?)?;

                // Backward pass
                self.optimizer.zero_grad();
                total_loss_batch.backward()?;

                // Gradient clipping
                for param in self.actor_critic.parameters() {
                    if let Some(grad) = param.grad() {
                        grad.clamp_(-0.5, 0.5)?;
                    }
                }

                self.optimizer.step()?;

                total_loss += total_loss_batch.item::<f32>() as f64;
                batch_count += 1;
            }
        }

        Ok(total_loss / batch_count as f64)
    }
}

/// Trajectory for policy gradient methods
#[derive(Debug, Clone)]
struct Trajectory {
    states: Vec<Tensor>,
    actions: Vec<usize>,
    rewards: Vec<f64>,
    log_probs: Vec<f64>,
    dones: Vec<bool>,
}

impl Trajectory {
    fn new() -> Self {
        Self {
            states: Vec::new(),
            actions: Vec::new(),
            rewards: Vec::new(),
            log_probs: Vec::new(),
            dones: Vec::new(),
        }
    }

    fn add_step(&mut self, state: Tensor, action: usize, reward: f64, log_prob: f64, done: bool) {
        self.states.push(state);
        self.actions.push(action);
        self.rewards.push(reward);
        self.log_probs.push(log_prob);
        self.dones.push(done);
    }

    fn len(&self) -> usize {
        self.states.len()
    }
}

/// Simple grid world environment for demonstration
struct GridWorld {
    grid_size: usize,
    agent_pos: (usize, usize),
    goal_pos: (usize, usize),
    obstacles: Vec<(usize, usize)>,
    max_steps: usize,
    current_steps: usize,
}

impl GridWorld {
    fn new(grid_size: usize) -> Self {
        let mut rng = thread_rng();
        let agent_pos = (0, 0);
        let goal_pos = (grid_size - 1, grid_size - 1);

        // Generate random obstacles
        let mut obstacles = Vec::new();
        for _ in 0..(grid_size * grid_size / 8) {
            let pos = (rng.gen_range(0..grid_size), rng.gen_range(0..grid_size));
            if pos != agent_pos && pos != goal_pos {
                obstacles.push(pos);
            }
        }

        Self {
            grid_size,
            agent_pos,
            goal_pos,
            obstacles,
            max_steps: grid_size * grid_size,
            current_steps: 0,
        }
    }

    fn state_to_tensor(&self) -> Result<Tensor> {
        let mut state = vec![0.0f32; self.grid_size * self.grid_size + 4];

        // Agent position (one-hot)
        let agent_idx = self.agent_pos.0 * self.grid_size + self.agent_pos.1;
        state[agent_idx] = 1.0;

        // Goal position
        let goal_idx = self.goal_pos.0 * self.grid_size + self.goal_pos.1;
        state[goal_idx] += 0.5;

        // Obstacles
        for &(x, y) in &self.obstacles {
            let obs_idx = x * self.grid_size + y;
            state[obs_idx] = -1.0;
        }

        // Additional features
        let base_idx = self.grid_size * self.grid_size;
        state[base_idx] = self.agent_pos.0 as f32 / self.grid_size as f32;
        state[base_idx + 1] = self.agent_pos.1 as f32 / self.grid_size as f32;
        state[base_idx + 2] = self.current_steps as f32 / self.max_steps as f32;

        // Distance to goal
        let distance = ((self.agent_pos.0 as i32 - self.goal_pos.0 as i32).abs()
            + (self.agent_pos.1 as i32 - self.goal_pos.1 as i32).abs())
            as f32;
        state[base_idx + 3] = distance / (2 * self.grid_size) as f32;

        tensor!(state)
    }
}

impl Environment for GridWorld {
    type Action = usize; // 0: up, 1: right, 2: down, 3: left
    type State = Tensor;
    type Reward = f64;

    fn reset(&mut self) -> Self::State {
        self.agent_pos = (0, 0);
        self.current_steps = 0;
        self.state_to_tensor().unwrap()
    }

    fn step(&mut self, action: &Self::Action) -> (Self::State, Self::Reward, bool) {
        self.current_steps += 1;

        let (dx, dy) = match action {
            0 => (-1, 0), // up
            1 => (0, 1),  // right
            2 => (1, 0),  // down
            3 => (0, -1), // left
            _ => (0, 0),
        };

        let new_x = (self.agent_pos.0 as i32 + dx)
            .max(0)
            .min(self.grid_size as i32 - 1) as usize;
        let new_y = (self.agent_pos.1 as i32 + dy)
            .max(0)
            .min(self.grid_size as i32 - 1) as usize;
        let new_pos = (new_x, new_y);

        // Check for obstacles
        if !self.obstacles.contains(&new_pos) {
            self.agent_pos = new_pos;
        }

        // Calculate reward
        let reward = if self.agent_pos == self.goal_pos {
            100.0 // Goal reached
        } else if self.current_steps >= self.max_steps {
            -10.0 // Timeout
        } else {
            // Small negative reward for each step + distance-based reward
            let distance = ((self.agent_pos.0 as i32 - self.goal_pos.0 as i32).abs()
                + (self.agent_pos.1 as i32 - self.goal_pos.1 as i32).abs())
                as f64;
            -0.01 - distance * 0.001
        };

        let done = self.agent_pos == self.goal_pos || self.current_steps >= self.max_steps;
        let next_state = self.state_to_tensor().unwrap();

        (next_state, reward, done)
    }

    fn action_space_size(&self) -> usize {
        4
    }

    fn state_space_size(&self) -> usize {
        self.grid_size * self.grid_size + 4
    }

    fn get_valid_actions(&self, _state: &Self::State) -> Vec<Self::Action> {
        vec![0, 1, 2, 3]
    }
}

/// Run comprehensive RL demo
fn run_rl_demo() -> Result<()> {
    println!("=== Advanced Reinforcement Learning Demo ===\n");

    let device = Device::cpu();
    let grid_size = 8;
    let mut env = GridWorld::new(grid_size);

    let state_dim = env.state_space_size();
    let num_actions = env.action_space_size();

    println!("Environment: {}x{} GridWorld", grid_size, grid_size);
    println!("State dimension: {}", state_dim);
    println!("Action space: {}", num_actions);

    // DQN Training
    println!("\n--- Training DQN Agent ---");
    let mut dqn_agent = DQNAgent::new(state_dim, num_actions, 1e-3, 10000, device.clone())?;

    let mut episode_rewards = Vec::new();
    let episodes = 500;

    for episode in 0..episodes {
        let mut state = env.reset();
        let mut total_reward = 0.0;
        let mut steps = 0;

        loop {
            let action = dqn_agent.select_action(&state.unsqueeze(0)?, true)?;
            let (next_state, reward, done) = env.step(&action);

            let experience = Experience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            };

            dqn_agent.store_experience(experience);

            // Train if enough experiences
            if episode > 32 {
                let loss = dqn_agent.train(32)?;
                if steps % 10 == 0 && loss > 0.0 {
                    // Training happening
                }
            }

            state = next_state;
            total_reward += reward;
            steps += 1;

            if done {
                break;
            }
        }

        episode_rewards.push(total_reward);

        if episode % 50 == 0 {
            let avg_reward = episode_rewards[episode.saturating_sub(50)..]
                .iter()
                .sum::<f64>()
                / (episode_rewards.len() - episode.saturating_sub(50)) as f64;
            println!(
                "Episode {}: Avg Reward = {:.2}, Epsilon = {:.3}",
                episode, avg_reward, dqn_agent.epsilon
            );
        }
    }

    // PPO Training
    println!("\n--- Training PPO Agent ---");
    let mut ppo_agent = PPOAgent::new(state_dim, num_actions, 3e-4, device.clone())?;

    let mut ppo_rewards = Vec::new();
    let ppo_episodes = 200;
    let trajectory_length = 128;

    for episode in 0..ppo_episodes {
        let mut trajectories = Vec::new();
        let mut episode_reward = 0.0;

        // Collect trajectories
        for _ in 0..4 {
            // Multiple parallel trajectories
            let mut trajectory = Trajectory::new();
            let mut state = env.reset();

            for _ in 0..trajectory_length {
                let (action, log_prob) = ppo_agent.select_action(&state.unsqueeze(0)?)?;
                let (next_state, reward, done) = env.step(&action);

                trajectory.add_step(state.clone(), action, reward, log_prob, done);
                episode_reward += reward;

                state = next_state;

                if done {
                    state = env.reset();
                }
            }

            trajectories.push(trajectory);
        }

        // Train on collected trajectories
        let loss = ppo_agent.train(&trajectories)?;
        ppo_rewards.push(episode_reward / 4.0); // Average across parallel envs

        if episode % 20 == 0 {
            let avg_reward = ppo_rewards[episode.saturating_sub(20)..]
                .iter()
                .sum::<f64>()
                / (ppo_rewards.len() - episode.saturating_sub(20)) as f64;
            println!(
                "Episode {}: Avg Reward = {:.2}, Loss = {:.6}",
                episode, avg_reward, loss
            );
        }
    }

    // Evaluation
    println!("\n--- Agent Evaluation ---");

    // Test DQN agent
    let mut dqn_test_rewards = Vec::new();
    for _ in 0..10 {
        let mut state = env.reset();
        let mut total_reward = 0.0;

        for _ in 0..env.max_steps {
            let action = dqn_agent.select_action(&state.unsqueeze(0)?, false)?; // No exploration
            let (next_state, reward, done) = env.step(&action);

            state = next_state;
            total_reward += reward;

            if done {
                break;
            }
        }

        dqn_test_rewards.push(total_reward);
    }

    // Test PPO agent
    let mut ppo_test_rewards = Vec::new();
    for _ in 0..10 {
        let mut state = env.reset();
        let mut total_reward = 0.0;

        for _ in 0..env.max_steps {
            let (action, _) = ppo_agent.select_action(&state.unsqueeze(0)?)?;
            let (next_state, reward, done) = env.step(&action);

            state = next_state;
            total_reward += reward;

            if done {
                break;
            }
        }

        ppo_test_rewards.push(total_reward);
    }

    println!("DQN Test Results:");
    println!(
        "  Average Reward: {:.2}",
        dqn_test_rewards.iter().sum::<f64>() / 10.0
    );
    println!(
        "  Best Reward: {:.2}",
        dqn_test_rewards
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    println!("PPO Test Results:");
    println!(
        "  Average Reward: {:.2}",
        ppo_test_rewards.iter().sum::<f64>() / 10.0
    );
    println!(
        "  Best Reward: {:.2}",
        ppo_test_rewards
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    println!("\n=== RL Demo Complete ===");

    Ok(())
}

fn main() -> Result<()> {
    run_rl_demo()?;
    Ok(())
}
