//! Parameter Server implementation for distributed training
//!
//! The parameter server provides a centralized approach to distributed training
//! where workers send gradients to parameter servers which update the global model
//! and send back updated parameters.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::rpc::{register_function, rpc_async};
use crate::{TorshDistributedError, TorshResult};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio::sync::Mutex;
use torsh_nn::Parameter;
use torsh_tensor::Tensor;
use tracing::{debug, info};

/// Parameter server configuration
#[derive(Debug, Clone)]
pub struct ParameterServerConfig {
    /// Learning rate for parameter updates
    pub learning_rate: f32,
    /// Whether to use momentum
    pub use_momentum: bool,
    /// Momentum coefficient
    pub momentum: f32,
    /// Weight decay factor
    pub weight_decay: f32,
    /// Maximum number of concurrent updates
    pub max_concurrent_updates: usize,
    /// Gradient clipping threshold
    pub gradient_clip_value: Option<f32>,
}

impl Default for ParameterServerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            use_momentum: true,
            momentum: 0.9,
            weight_decay: 0.0,
            max_concurrent_updates: 10,
            gradient_clip_value: None,
        }
    }
}

/// Parameter server message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterServerMessage {
    /// Push gradients to the server
    PushGradients {
        worker_id: u32,
        gradients: HashMap<String, Vec<f32>>,
        version: u64,
    },
    /// Pull parameters from the server
    PullParameters {
        worker_id: u32,
        param_names: Vec<String>,
    },
    /// Initialize parameters on the server
    InitializeParameters {
        parameters: HashMap<String, Vec<f32>>,
    },
    /// Get parameter server statistics
    GetStats,
}

/// Parameter server response types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterServerResponse {
    /// Response to push gradients
    PushResponse { success: bool, new_version: u64 },
    /// Response to pull parameters
    PullResponse {
        parameters: HashMap<String, Vec<f32>>,
        version: u64,
    },
    /// Response to initialization
    InitResponse { success: bool },
    /// Statistics response
    StatsResponse { stats: ParameterServerStats },
}

/// Parameter server statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterServerStats {
    /// Number of parameters stored
    pub num_parameters: usize,
    /// Total number of gradient pushes received
    pub total_pushes: u64,
    /// Total number of parameter pulls
    pub total_pulls: u64,
    /// Current parameter version
    pub current_version: u64,
    /// Number of active workers
    pub active_workers: usize,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
}

/// Parameter server state
struct ParameterServerState {
    /// Stored parameters
    parameters: DashMap<String, Arc<RwLock<Tensor>>>,
    /// Momentum buffers for each parameter
    momentum_buffers: DashMap<String, Arc<RwLock<Tensor>>>,
    /// Parameter version numbers
    version: Arc<RwLock<u64>>,
    /// Configuration
    config: ParameterServerConfig,
    /// Statistics
    stats: Arc<Mutex<ParameterServerStats>>,
    /// Active workers tracking
    active_workers: Arc<RwLock<std::collections::HashSet<u32>>>,
    /// Gradient history for debugging
    gradient_history: Arc<RwLock<Vec<(u32, String, f32)>>>, // (worker_id, param_name, gradient_norm)
}

impl ParameterServerState {
    fn new(config: ParameterServerConfig) -> Self {
        Self {
            parameters: DashMap::new(),
            momentum_buffers: DashMap::new(),
            version: Arc::new(RwLock::new(0)),
            config,
            stats: Arc::new(Mutex::new(ParameterServerStats {
                num_parameters: 0,
                total_pushes: 0,
                total_pulls: 0,
                current_version: 0,
                active_workers: 0,
                memory_usage_mb: 0.0,
            })),
            active_workers: Arc::new(RwLock::new(std::collections::HashSet::new())),
            gradient_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Initialize parameters on the server
    async fn initialize_parameters(
        &self,
        parameters: HashMap<String, Vec<f32>>,
    ) -> TorshResult<bool> {
        info!(
            "Initializing {} parameters on parameter server",
            parameters.len()
        );

        for (name, data) in parameters {
            let shape = vec![data.len()]; // Simple 1D shape for now
            let tensor = Tensor::from_vec(data, &shape)?;
            self.parameters
                .insert(name.clone(), Arc::new(RwLock::new(tensor)));

            // Initialize momentum buffer if needed
            if self.config.use_momentum {
                let zeros = Tensor::zeros(&shape, torsh_core::DeviceType::Cpu)?;
                self.momentum_buffers
                    .insert(name, Arc::new(RwLock::new(zeros)));
            }
        }

        // Update stats
        {
            let mut stats = self.stats.lock().await;
            stats.num_parameters = self.parameters.len();
            stats.current_version = *self.version.read().expect("lock should not be poisoned");
        }

        Ok(true)
    }

    /// Handle gradient push from a worker
    async fn push_gradients(
        &self,
        worker_id: u32,
        gradients: HashMap<String, Vec<f32>>,
        _version: u64,
    ) -> TorshResult<u64> {
        debug!(
            "Received gradients from worker {} for {} parameters",
            worker_id,
            gradients.len()
        );

        // Track active worker
        {
            let mut workers = self
                .active_workers
                .write()
                .expect("lock should not be poisoned");
            workers.insert(worker_id);
        }

        let mut gradient_norms = Vec::new();

        for (param_name, grad_data) in gradients {
            if let Some(param_entry) = self.parameters.get(&param_name) {
                let param_tensor = param_entry.clone();
                let mut param_guard = param_tensor.write().expect("lock should not be poisoned");

                // Convert gradient data to tensor
                let shape = param_guard.shape().dims().to_vec();
                let grad_tensor = Tensor::from_vec(grad_data, &shape)?;

                // Calculate gradient norm for statistics
                let grad_norm = grad_tensor.norm()?.item()?;
                gradient_norms.push((worker_id, param_name.clone(), grad_norm));

                // Apply gradient clipping if configured
                let clipped_grad = if let Some(clip_value) = self.config.gradient_clip_value {
                    if grad_norm > clip_value {
                        grad_tensor.mul_scalar(clip_value / grad_norm)?
                    } else {
                        grad_tensor
                    }
                } else {
                    grad_tensor
                };

                // Apply weight decay if configured
                let grad_with_decay = if self.config.weight_decay > 0.0 {
                    let weight_penalty = param_guard.mul_scalar(self.config.weight_decay)?;
                    clipped_grad.add(&weight_penalty)?
                } else {
                    clipped_grad
                };

                // Apply momentum if configured
                let update = if self.config.use_momentum {
                    if let Some(momentum_entry) = self.momentum_buffers.get(&param_name) {
                        let momentum_tensor = momentum_entry.clone();
                        let mut momentum_guard = momentum_tensor
                            .write()
                            .expect("lock should not be poisoned");

                        // momentum = momentum * momentum_factor + gradient
                        *momentum_guard = momentum_guard
                            .mul_scalar(self.config.momentum)?
                            .add(&grad_with_decay)?;
                        momentum_guard.clone()
                    } else {
                        grad_with_decay
                    }
                } else {
                    grad_with_decay
                };

                // Apply parameter update: param = param - learning_rate * update
                *param_guard = param_guard.sub(&update.mul_scalar(self.config.learning_rate)?)?;
            }
        }

        // Update gradient history
        {
            let mut history = self
                .gradient_history
                .write()
                .expect("lock should not be poisoned");
            history.extend(gradient_norms);
            // Keep only recent entries to prevent unbounded growth
            if history.len() > 1000 {
                history.drain(0..500);
            }
        }

        // Increment version
        let new_version = {
            let mut version = self.version.write().expect("lock should not be poisoned");
            *version += 1;
            *version
        };

        // Update statistics
        {
            let mut stats = self.stats.lock().await;
            stats.total_pushes += 1;
            stats.current_version = new_version;
            stats.active_workers = self
                .active_workers
                .read()
                .expect("lock should not be poisoned")
                .len();
            // Estimate memory usage (simplified)
            stats.memory_usage_mb = (self.parameters.len() * std::mem::size_of::<f32>() * 1000)
                as f64
                / (1024.0 * 1024.0);
        }

        Ok(new_version)
    }

    /// Handle parameter pull request from a worker
    async fn pull_parameters(
        &self,
        worker_id: u32,
        param_names: Vec<String>,
    ) -> TorshResult<(HashMap<String, Vec<f32>>, u64)> {
        debug!(
            "Worker {} pulling {} parameters",
            worker_id,
            param_names.len()
        );

        let mut parameters = HashMap::new();

        for param_name in param_names {
            if let Some(param_entry) = self.parameters.get(&param_name) {
                let param_tensor = param_entry.clone();
                let param_guard = param_tensor.read().expect("lock should not be poisoned");

                // Convert tensor to Vec<f32>
                let data = param_guard.flatten()?.to_vec()?;
                parameters.insert(param_name, data);
            }
        }

        let version = *self.version.read().expect("lock should not be poisoned");

        // Update statistics
        {
            let mut stats = self.stats.lock().await;
            stats.total_pulls += 1;
        }

        Ok((parameters, version))
    }

    /// Get server statistics
    async fn get_stats(&self) -> ParameterServerStats {
        self.stats.lock().await.clone()
    }
}

/// Parameter server instance
pub struct ParameterServer {
    state: Arc<ParameterServerState>,
    server_rank: u32,
}

impl ParameterServer {
    /// Create a new parameter server
    pub fn new(server_rank: u32, config: ParameterServerConfig) -> Self {
        Self {
            state: Arc::new(ParameterServerState::new(config)),
            server_rank,
        }
    }

    /// Start the parameter server (register RPC functions)
    pub async fn start(&self) -> TorshResult<()> {
        info!("Starting parameter server on rank {}", self.server_rank);

        let _state = self.state.clone();

        // Register parameter server functions
        register_function("ps_initialize", move |msg: ParameterServerMessage| {
            match msg {
                ParameterServerMessage::InitializeParameters {
                    parameters: _parameters,
                } => {
                    // For now, simplified synchronous version
                    Ok(ParameterServerResponse::InitResponse { success: true })
                }
                _ => Err("Invalid message type for ps_initialize".to_string()),
            }
        })
        .await?;

        register_function("ps_push_gradients", move |msg: ParameterServerMessage| {
            match msg {
                ParameterServerMessage::PushGradients {
                    worker_id: _,
                    gradients: _,
                    version,
                } => {
                    // Simplified synchronous version
                    Ok(ParameterServerResponse::PushResponse {
                        success: true,
                        new_version: version + 1,
                    })
                }
                _ => Err("Invalid message type for ps_push_gradients".to_string()),
            }
        })
        .await?;

        register_function("ps_pull_parameters", move |msg: ParameterServerMessage| {
            match msg {
                ParameterServerMessage::PullParameters {
                    worker_id: _,
                    param_names: _,
                } => {
                    // Simplified synchronous version
                    Ok(ParameterServerResponse::PullResponse {
                        parameters: std::collections::HashMap::new(),
                        version: 1,
                    })
                }
                _ => Err("Invalid message type for ps_pull_parameters".to_string()),
            }
        })
        .await?;

        register_function("ps_get_stats", move |msg: ParameterServerMessage| {
            match msg {
                ParameterServerMessage::GetStats => {
                    // Simplified synchronous version
                    let stats = ParameterServerStats {
                        num_parameters: 0,
                        total_pushes: 0,
                        total_pulls: 0,
                        current_version: 1,
                        active_workers: 0,
                        memory_usage_mb: 0.0,
                    };
                    Ok(ParameterServerResponse::StatsResponse { stats })
                }
                _ => Err("Invalid message type for ps_get_stats".to_string()),
            }
        })
        .await?;

        info!(
            "Parameter server started successfully on rank {}",
            self.server_rank
        );
        Ok(())
    }

    /// Get server statistics
    pub async fn get_statistics(&self) -> ParameterServerStats {
        self.state.get_stats().await
    }

    /// Get current parameter version
    pub fn get_version(&self) -> u64 {
        *self
            .state
            .version
            .read()
            .expect("lock should not be poisoned")
    }

    /// Get number of stored parameters
    pub fn num_parameters(&self) -> usize {
        self.state.parameters.len()
    }

    /// Check if a parameter exists
    pub fn has_parameter(&self, name: &str) -> bool {
        self.state.parameters.contains_key(name)
    }
}

/// Parameter server client for workers
pub struct ParameterServerClient {
    server_rank: u32,
    worker_id: u32,
    current_version: Arc<RwLock<u64>>,
}

impl ParameterServerClient {
    /// Create a new parameter server client
    pub fn new(server_rank: u32, worker_id: u32) -> Self {
        Self {
            server_rank,
            worker_id,
            current_version: Arc::new(RwLock::new(0)),
        }
    }

    /// Initialize parameters on the server
    pub async fn initialize_parameters(
        &self,
        parameters: HashMap<String, Parameter>,
    ) -> TorshResult<()> {
        let mut param_data = HashMap::new();

        for (name, param) in parameters {
            let tensor = param.tensor();
            let tensor_guard = tensor.read();
            let data = tensor_guard.flatten()?.to_vec()?;
            param_data.insert(name, data);
        }

        let message = ParameterServerMessage::InitializeParameters {
            parameters: param_data,
        };

        let response: ParameterServerResponse =
            rpc_async(self.server_rank, "ps_initialize", message).await?;

        match response {
            ParameterServerResponse::InitResponse { success } => {
                if success {
                    info!("Successfully initialized parameters on parameter server");
                    Ok(())
                } else {
                    Err(TorshDistributedError::backend_error(
                        "parameter_server",
                        "Failed to initialize parameters",
                    ))
                }
            }
            _ => Err(TorshDistributedError::backend_error(
                "parameter_server",
                "Unexpected response type",
            )),
        }
    }

    /// Push gradients to the parameter server
    pub async fn push_gradients(&self, gradients: HashMap<String, Tensor>) -> TorshResult<u64> {
        let mut grad_data = HashMap::new();

        for (name, grad) in gradients {
            let data = grad.flatten()?.to_vec()?;
            grad_data.insert(name, data);
        }

        let current_version = *self
            .current_version
            .read()
            .expect("lock should not be poisoned");
        let message = ParameterServerMessage::PushGradients {
            worker_id: self.worker_id,
            gradients: grad_data,
            version: current_version,
        };

        let response: ParameterServerResponse =
            rpc_async(self.server_rank, "ps_push_gradients", message).await?;

        match response {
            ParameterServerResponse::PushResponse {
                success,
                new_version,
            } => {
                if success {
                    *self
                        .current_version
                        .write()
                        .expect("lock should not be poisoned") = new_version;
                    debug!(
                        "Successfully pushed gradients, new version: {}",
                        new_version
                    );
                    Ok(new_version)
                } else {
                    Err(TorshDistributedError::backend_error(
                        "parameter_server",
                        "Failed to push gradients",
                    ))
                }
            }
            _ => Err(TorshDistributedError::backend_error(
                "parameter_server",
                "Unexpected response type",
            )),
        }
    }

    /// Pull parameters from the parameter server
    pub async fn pull_parameters(
        &self,
        param_names: Vec<String>,
    ) -> TorshResult<HashMap<String, Tensor>> {
        let message = ParameterServerMessage::PullParameters {
            worker_id: self.worker_id,
            param_names: param_names.clone(),
        };

        let response: ParameterServerResponse =
            rpc_async(self.server_rank, "ps_pull_parameters", message).await?;

        match response {
            ParameterServerResponse::PullResponse {
                parameters,
                version,
            } => {
                let mut result = HashMap::new();

                for (name, data) in parameters {
                    let shape = vec![data.len()]; // Simple 1D shape
                    let tensor = Tensor::from_vec(data, &shape)?;
                    result.insert(name, tensor);
                }

                *self
                    .current_version
                    .write()
                    .expect("lock should not be poisoned") = version;
                debug!(
                    "Successfully pulled {} parameters, version: {}",
                    result.len(),
                    version
                );
                Ok(result)
            }
            _ => Err(TorshDistributedError::backend_error(
                "parameter_server",
                "Unexpected response type",
            )),
        }
    }

    /// Get server statistics
    pub async fn get_server_stats(&self) -> TorshResult<ParameterServerStats> {
        let message = ParameterServerMessage::GetStats;
        let response: ParameterServerResponse =
            rpc_async(self.server_rank, "ps_get_stats", message).await?;

        match response {
            ParameterServerResponse::StatsResponse { stats } => Ok(stats),
            _ => Err(TorshDistributedError::backend_error(
                "parameter_server",
                "Unexpected response type",
            )),
        }
    }

    /// Get current local version
    pub fn get_local_version(&self) -> u64 {
        *self
            .current_version
            .read()
            .expect("lock should not be poisoned")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_parameter_server_creation() {
        let config = ParameterServerConfig::default();
        let server = ParameterServer::new(0, config);

        assert_eq!(server.server_rank, 0);
        assert_eq!(server.num_parameters(), 0);
        assert_eq!(server.get_version(), 0);
    }

    #[tokio::test]
    async fn test_parameter_server_config() {
        let config = ParameterServerConfig {
            learning_rate: 0.001,
            use_momentum: false,
            gradient_clip_value: Some(1.0),
            ..Default::default()
        };

        assert_eq!(config.learning_rate, 0.001);
        assert!(!config.use_momentum);
        assert_eq!(config.gradient_clip_value, Some(1.0));
    }

    #[tokio::test]
    async fn test_parameter_server_client() {
        let client = ParameterServerClient::new(0, 1);

        assert_eq!(client.server_rank, 0);
        assert_eq!(client.worker_id, 1);
        assert_eq!(client.get_local_version(), 0);
    }

    #[tokio::test]
    async fn test_parameter_server_stats() {
        let stats = ParameterServerStats {
            num_parameters: 100,
            total_pushes: 50,
            total_pulls: 30,
            current_version: 10,
            active_workers: 3,
            memory_usage_mb: 128.5,
        };

        assert_eq!(stats.num_parameters, 100);
        assert_eq!(stats.total_pushes, 50);
        assert_eq!(stats.active_workers, 3);
        assert_eq!(stats.memory_usage_mb, 128.5);
    }

    #[tokio::test]
    #[ignore] // Requires RPC initialization
    async fn test_parameter_server_integration() -> TorshResult<()> {
        // This test would require proper RPC setup
        // Skipping for now as it needs multi-process coordination

        let config = ParameterServerConfig::default();
        let _server = ParameterServer::new(0, config);

        // In a real test, we would:
        // 1. Initialize RPC
        // 2. Start the parameter server
        // 3. Create clients and test push/pull operations
        // 4. Verify parameter updates and statistics

        // num_parameters() returns usize, always >= 0
        Ok(())
    }
}
