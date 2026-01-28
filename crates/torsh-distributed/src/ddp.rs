//! Distributed Data Parallel (DDP) implementation

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
#![allow(clippy::await_holding_lock)]
use crate::backend::ReduceOp;
use crate::collectives::all_reduce;
use crate::{process_group::ProcessGroup, TorshResult};
use log::info;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinHandle;
use torsh_core::{error::Result, DeviceType};
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;

/// Configuration for gradient bucketing
#[derive(Debug, Clone)]
pub struct BucketConfig {
    /// Maximum size of each bucket in MB
    pub max_bucket_size_mb: f32,
    /// Whether to enable gradient bucketing
    pub enabled: bool,
    /// Minimum bucket size to avoid tiny buckets
    pub min_bucket_size_mb: f32,
}

impl Default for BucketConfig {
    fn default() -> Self {
        Self {
            max_bucket_size_mb: 25.0,
            enabled: true,
            min_bucket_size_mb: 1.0,
        }
    }
}

/// A bucket of gradients for efficient communication
#[derive(Debug)]
struct GradientBucket {
    /// Parameters in this bucket
    parameters: Vec<String>,
    /// Total size in bytes
    total_size: usize,
    /// Whether this bucket is ready for synchronization
    _ready: bool,
}

impl GradientBucket {
    fn new() -> Self {
        Self {
            parameters: Vec::new(),
            total_size: 0,
            _ready: false,
        }
    }

    fn add_parameter(&mut self, name: String, size: usize) {
        self.parameters.push(name);
        self.total_size += size;
    }

    fn size_mb(&self) -> f32 {
        self.total_size as f32 / (1024.0 * 1024.0)
    }
}

/// Statistics about gradient synchronization
#[derive(Debug, Clone)]
pub struct GradientSyncStats {
    /// Total number of parameters that require gradients
    pub total_parameters: usize,
    /// Number of parameters that currently have gradients
    pub parameters_with_grad: usize,
    /// Total size of gradients in MB
    pub total_gradient_size_mb: f32,
    /// Number of gradient buckets
    pub num_buckets: usize,
    /// World size (number of processes)
    pub world_size: u32,
}

/// Information about a gradient bucket
#[derive(Debug, Clone)]
pub struct BucketInfo {
    /// Bucket index
    pub index: usize,
    /// Size of bucket in MB
    pub size_mb: f32,
    /// Number of parameters in bucket
    pub num_parameters: usize,
    /// Names of parameters in bucket
    pub parameter_names: Vec<String>,
}

/// Message sent to gradient synchronization worker
#[derive(Debug)]
struct GradientMessage {
    /// Parameter name
    param_name: String,
    /// Gradient tensor to synchronize
    gradient: Tensor,
    /// Bucket index this parameter belongs to
    bucket_index: usize,
}

/// State for tracking unused parameters
#[derive(Debug, Default)]
struct UnusedParameterTracker {
    /// All parameters that require gradients
    all_parameters: HashSet<String>,
    /// Parameters that have been used in the current iteration
    used_parameters: HashSet<String>,
    /// Whether unused parameter detection is enabled
    enabled: bool,
    /// Iteration counter
    iteration: u64,
}

/// Overlap computation configuration
#[derive(Debug, Clone)]
pub struct OverlapConfig {
    /// Whether to enable computation/communication overlap
    pub enabled: bool,
    /// Maximum number of pending gradient synchronizations
    pub max_pending_syncs: usize,
    /// Timeout for gradient synchronization (in seconds)
    pub sync_timeout_secs: u64,
    /// Whether to enable unused parameter detection
    pub detect_unused_parameters: bool,
}

impl Default for OverlapConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_pending_syncs: 4,
            sync_timeout_secs: 30,
            detect_unused_parameters: true,
        }
    }
}

/// Distributed Data Parallel wrapper for models
pub struct DistributedDataParallel<M: Module> {
    module: M,
    process_group: Arc<ProcessGroup>,
    _device_ids: Vec<usize>,
    _output_device: Option<usize>,
    _broadcast_buffers: bool,
    _bucket_cap_mb: f32,
    bucket_config: BucketConfig,
    gradient_buckets: Vec<GradientBucket>,
    /// Mapping from parameter name to bucket index
    param_to_bucket: HashMap<String, usize>,
    /// Overlap computation configuration
    overlap_config: OverlapConfig,
    /// Channel for sending gradients to background worker
    gradient_sender: Option<mpsc::UnboundedSender<GradientMessage>>,
    /// Handle to background gradient synchronization task
    sync_task_handle: Option<JoinHandle<()>>,
    /// Semaphore to limit concurrent gradient synchronizations
    sync_semaphore: Arc<Semaphore>,
    /// Unused parameter tracker
    unused_param_tracker: Arc<Mutex<UnusedParameterTracker>>,
    /// Bucket readiness tracking (bucket_index -> ready gradients count)
    bucket_ready_count: Arc<Mutex<HashMap<usize, usize>>>,
}

impl<M: Module> DistributedDataParallel<M> {
    /// Create a new DDP wrapper
    pub fn new(
        module: M,
        process_group: Arc<ProcessGroup>,
        device_ids: Vec<usize>,
        output_device: Option<usize>,
        broadcast_buffers: bool,
        bucket_cap_mb: f32,
    ) -> TorshResult<Self> {
        let bucket_config = BucketConfig {
            max_bucket_size_mb: bucket_cap_mb,
            enabled: true,
            min_bucket_size_mb: 1.0,
        };

        Self::new_with_configs(
            module,
            process_group,
            device_ids,
            output_device,
            broadcast_buffers,
            bucket_config,
            OverlapConfig::default(),
        )
    }

    /// Create a new DDP wrapper with custom bucket configuration
    pub fn new_with_bucket_config(
        module: M,
        process_group: Arc<ProcessGroup>,
        device_ids: Vec<usize>,
        output_device: Option<usize>,
        broadcast_buffers: bool,
        bucket_config: BucketConfig,
    ) -> TorshResult<Self> {
        Self::new_with_configs(
            module,
            process_group,
            device_ids,
            output_device,
            broadcast_buffers,
            bucket_config,
            OverlapConfig::default(),
        )
    }

    /// Create a new DDP wrapper with custom configurations
    pub fn new_with_configs(
        module: M,
        process_group: Arc<ProcessGroup>,
        device_ids: Vec<usize>,
        output_device: Option<usize>,
        broadcast_buffers: bool,
        bucket_config: BucketConfig,
        overlap_config: OverlapConfig,
    ) -> TorshResult<Self> {
        let sync_semaphore = Arc::new(Semaphore::new(overlap_config.max_pending_syncs));
        let unused_param_tracker = Arc::new(Mutex::new(UnusedParameterTracker::default()));
        let bucket_ready_count = Arc::new(Mutex::new(HashMap::new()));

        let mut ddp = Self {
            module,
            process_group,
            _device_ids: device_ids,
            _output_device: output_device,
            _broadcast_buffers: broadcast_buffers,
            _bucket_cap_mb: bucket_config.max_bucket_size_mb,
            bucket_config,
            gradient_buckets: Vec::new(),
            param_to_bucket: HashMap::new(),
            overlap_config,
            gradient_sender: None,
            sync_task_handle: None,
            sync_semaphore,
            unused_param_tracker,
            bucket_ready_count,
        };

        // Initialize gradient buckets
        ddp.initialize_buckets()?;

        // Initialize unused parameter tracking if enabled
        if ddp.overlap_config.detect_unused_parameters {
            ddp.initialize_unused_parameter_tracking()?;
        }

        // Start background gradient synchronization worker if overlap is enabled
        if ddp.overlap_config.enabled {
            ddp.start_gradient_sync_worker()?;
        }

        Ok(ddp)
    }

    /// Initialize gradient buckets based on parameter sizes
    fn initialize_buckets(&mut self) -> TorshResult<()> {
        if !self.bucket_config.enabled {
            return Ok(());
        }

        let parameters = self.module.named_parameters();
        let mut current_bucket = GradientBucket::new();
        let mut bucket_index = 0;

        // Sort parameters by size (largest first for better packing)
        let mut param_sizes: Vec<(String, usize)> = parameters
            .iter()
            .map(|(name, param)| {
                let tensor = param.tensor();
                let tensor_guard = tensor.read();
                let size = tensor_guard.numel() * std::mem::size_of::<f32>(); // Assume f32 for now
                (name.clone(), size)
            })
            .collect();

        param_sizes.sort_by(|a, b| b.1.cmp(&a.1)); // Sort descending by size

        for (param_name, param_size) in param_sizes {
            // Check if adding this parameter would exceed bucket capacity
            let new_size_mb = (current_bucket.total_size + param_size) as f32 / (1024.0 * 1024.0);

            if new_size_mb > self.bucket_config.max_bucket_size_mb
                && !current_bucket.parameters.is_empty()
            {
                // Finalize current bucket and start a new one
                self.gradient_buckets.push(current_bucket);
                current_bucket = GradientBucket::new();
                bucket_index += 1;
            }

            // Add parameter to current bucket
            current_bucket.add_parameter(param_name.clone(), param_size);
            self.param_to_bucket.insert(param_name, bucket_index);
        }

        // Add the last bucket if it has parameters
        if !current_bucket.parameters.is_empty() {
            self.gradient_buckets.push(current_bucket);
        }

        info!(
            "📦 Initialized {} gradient buckets",
            self.gradient_buckets.len()
        );
        for (i, bucket) in self.gradient_buckets.iter().enumerate() {
            info!(
                "  Bucket {}: {:.2} MB, {} parameters",
                i,
                bucket.size_mb(),
                bucket.parameters.len()
            );
        }

        Ok(())
    }

    /// Initialize unused parameter tracking
    fn initialize_unused_parameter_tracking(&mut self) -> TorshResult<()> {
        let parameters = self.module.named_parameters();
        let mut tracker = self
            .unused_param_tracker
            .lock()
            .expect("lock should not be poisoned");

        tracker.enabled = true;
        tracker.all_parameters.clear();
        tracker.used_parameters.clear();

        // Add all parameters that require gradients
        for (name, param) in parameters {
            let tensor = param.tensor();
            let tensor_guard = tensor.read();
            if tensor_guard.requires_grad() {
                tracker.all_parameters.insert(name);
            }
        }

        info!(
            "🔍 Initialized unused parameter detection for {} parameters",
            tracker.all_parameters.len()
        );

        Ok(())
    }

    /// Start the background gradient synchronization worker
    fn start_gradient_sync_worker(&mut self) -> TorshResult<()> {
        let (sender, mut receiver) = mpsc::unbounded_channel::<GradientMessage>();

        let process_group = Arc::clone(&self.process_group);
        let sync_semaphore = Arc::clone(&self.sync_semaphore);
        let _bucket_ready_count = Arc::clone(&self.bucket_ready_count);
        let _gradient_buckets_len = self.gradient_buckets.len();
        let timeout_duration = Duration::from_secs(self.overlap_config.sync_timeout_secs);

        // Clone bucket info for the worker
        let bucket_param_counts: HashMap<usize, usize> = self
            .gradient_buckets
            .iter()
            .enumerate()
            .map(|(i, bucket)| (i, bucket.parameters.len()))
            .collect();

        let handle = tokio::spawn(async move {
            let mut pending_gradients: HashMap<usize, Vec<(String, Tensor)>> = HashMap::new();

            while let Some(grad_msg) = receiver.recv().await {
                // Acquire semaphore permit to limit concurrent operations
                let _permit =
                    match tokio::time::timeout(timeout_duration, sync_semaphore.acquire()).await {
                        Ok(Ok(permit)) => permit,
                        Ok(Err(_)) => {
                            info!("  Gradient sync semaphore closed, stopping worker");
                            break;
                        }
                        Err(_) => {
                            info!(
                                "  Gradient sync timeout, dropping gradient for {}",
                                grad_msg.param_name
                            );
                            continue;
                        }
                    };

                let bucket_index = grad_msg.bucket_index;

                // Add gradient to pending list for this bucket
                pending_gradients
                    .entry(bucket_index)
                    .or_insert_with(Vec::new)
                    .push((grad_msg.param_name, grad_msg.gradient));

                // Check if bucket is ready (all parameters have gradients)
                let expected_count = bucket_param_counts.get(&bucket_index).copied().unwrap_or(0);
                let current_count = pending_gradients
                    .get(&bucket_index)
                    .map(|v| v.len())
                    .unwrap_or(0);

                if current_count >= expected_count && expected_count > 0 {
                    // Bucket is ready - process all gradients in this bucket
                    if let Some(bucket_gradients) = pending_gradients.remove(&bucket_index) {
                        // Process bucket asynchronously with improved synchronization
                        let pg = Arc::clone(&process_group);
                        tokio::spawn(async move {
                            match Self::sync_bucket_gradients(bucket_gradients, &pg).await {
                                Ok(synchronized_gradients) => {
                                    info!(
                                        " Successfully synchronized bucket {} with {} gradients",
                                        bucket_index,
                                        synchronized_gradients.len()
                                    );
                                    // Note: In this async worker context, we cannot directly set gradients back to parameters
                                    // This would need to be handled by the main thread or through a callback mechanism
                                }
                                Err(e) => {
                                    info!("  Failed to sync bucket {}: {}", bucket_index, e);
                                }
                            }
                        });
                    }
                }
            }

            info!(" Gradient synchronization worker stopped");
        });

        self.gradient_sender = Some(sender);
        self.sync_task_handle = Some(handle);

        info!(" Started background gradient synchronization worker");
        Ok(())
    }

    /// Synchronize a bucket of gradients with efficient flattening
    async fn sync_bucket_gradients(
        gradients: Vec<(String, Tensor)>,
        process_group: &ProcessGroup,
    ) -> TorshResult<Vec<(String, Tensor)>> {
        let start_time = Instant::now();

        if gradients.is_empty() {
            return Ok(Vec::new());
        }

        // Improved implementation with efficient bucket flattening and synchronization
        if gradients.len() > 1 {
            // Multiple gradients - use efficient bucket flattening
            Self::sync_bucket_gradients_flattened(gradients, process_group).await
        } else {
            // Single gradient - direct synchronization
            Self::sync_single_gradient(gradients, process_group).await
        }
        .inspect(|_result| {
            let elapsed = start_time.elapsed();
            if elapsed > Duration::from_millis(100) {
                info!(
                    "⏱️  Bucket sync took {:.2}ms",
                    elapsed.as_secs_f32() * 1000.0
                );
            }
        })
    }

    /// Synchronize gradients using flattening for efficiency
    async fn sync_bucket_gradients_flattened(
        gradients: Vec<(String, Tensor)>,
        process_group: &ProcessGroup,
    ) -> TorshResult<Vec<(String, Tensor)>> {
        // Step 1: Flatten all gradients into a single tensor for efficient communication
        let mut gradient_shapes = Vec::new();
        let mut gradient_sizes = Vec::new();
        let mut flattened_data = Vec::new();

        for (param_name, grad) in &gradients {
            let shape = grad.shape();
            let numel = grad.numel();
            gradient_shapes.push((param_name.clone(), shape.dims().to_vec()));
            gradient_sizes.push(numel);

            // Flatten the gradient and extract its data
            let flattened_grad = grad.flatten()?;
            let grad_data = flattened_grad.data()?;
            flattened_data.extend_from_slice(&grad_data);
        }

        // Step 2: Create a single flattened tensor containing all gradients
        let total_size = flattened_data.len();
        let mut flattened_tensor =
            Tensor::from_data(flattened_data, vec![total_size], gradients[0].1.device())?;

        // Step 3: Perform a single all-reduce operation on the flattened tensor
        all_reduce(&mut flattened_tensor, ReduceOp::Sum, process_group).await?;

        // Average by world size
        let world_size = process_group.world_size() as f32;
        flattened_tensor = flattened_tensor.div_scalar(world_size)?;

        // Step 4: Unflatten and distribute back to individual gradients
        let flattened_data = flattened_tensor.data()?;
        let mut result_gradients = Vec::new();
        let mut current_offset = 0;

        for ((param_name, original_shape), size) in
            gradient_shapes.iter().zip(gradient_sizes.iter())
        {
            // Extract data for this gradient
            let grad_data = &flattened_data[current_offset..current_offset + size];

            // Reconstruct the gradient tensor with original shape
            let reconstructed_grad = Tensor::from_data(
                grad_data.to_vec(),
                original_shape.clone(),
                gradients[0].1.device(),
            )?;

            result_gradients.push((param_name.clone(), reconstructed_grad));
            current_offset += size;
        }

        Ok(result_gradients)
    }

    /// Synchronize a single gradient directly
    async fn sync_single_gradient(
        mut gradients: Vec<(String, Tensor)>,
        process_group: &ProcessGroup,
    ) -> TorshResult<Vec<(String, Tensor)>> {
        if let Some((param_name, mut grad)) = gradients.pop() {
            all_reduce(&mut grad, ReduceOp::Sum, process_group).await?;

            // Average by world size
            let world_size = process_group.world_size() as f32;
            grad = grad.div_scalar(world_size)?;

            Ok(vec![(param_name, grad)])
        } else {
            Ok(Vec::new())
        }
    }

    /// Synchronize gradients across all processes
    pub async fn sync_gradients(&mut self) -> TorshResult<()> {
        if self.bucket_config.enabled && !self.gradient_buckets.is_empty() {
            // Use bucketed gradient synchronization for better performance
            self.sync_gradients_bucketed().await
        } else {
            // Fall back to naive synchronization
            self.sync_gradients_naive().await
        }
    }

    /// Synchronize gradients using naive approach (one parameter at a time)
    async fn sync_gradients_naive(&mut self) -> TorshResult<()> {
        #[allow(clippy::await_holding_lock)]
        let parameters = self.module.parameters();

        for (_name, param) in parameters {
            let tensor = param.tensor();
            let tensor_guard = tensor.read();

            // Check if this parameter requires gradients and has a gradient
            if tensor_guard.requires_grad() {
                if let Some(mut grad) = tensor_guard.grad() {
                    // Perform all-reduce on the gradient
                    all_reduce(&mut grad, ReduceOp::Sum, &self.process_group).await?;

                    // Average by world size (divide by number of processes)
                    let world_size = self.process_group.world_size() as f32;
                    grad = grad.div_scalar(world_size)?;

                    // Set the synchronized gradient back to the parameter
                    tensor_guard.set_grad(Some(grad));
                }
            }
        }

        Ok(())
    }

    /// Synchronize gradients using bucketing for better communication efficiency
    async fn sync_gradients_bucketed(&mut self) -> TorshResult<()> {
        let parameters = self.module.named_parameters();

        // Process each bucket
        for bucket in &self.gradient_buckets {
            let mut bucket_gradients = Vec::new();
            let mut bucket_params = Vec::new();

            // Collect gradients for this bucket
            for param_name in &bucket.parameters {
                if let Some(param) = parameters.get(param_name) {
                    let tensor = param.tensor();
                    let tensor_guard = tensor.read();

                    if tensor_guard.requires_grad() {
                        if let Some(grad) = tensor_guard.grad() {
                            bucket_gradients.push(grad);
                            bucket_params.push(param_name.clone());
                        }
                    }
                }
            }

            if !bucket_gradients.is_empty() {
                // Sophisticated implementation with efficient bucket flattening:
                // 1. Flatten all gradients in the bucket into a single tensor
                // 2. Perform a single all-reduce operation on the flattened tensor
                // 3. Unflatten and distribute back to individual gradients

                // Prepare gradients with parameter names for synchronization
                let gradients_with_names: Vec<(String, Tensor)> = bucket_gradients
                    .into_iter()
                    .zip(bucket_params.iter())
                    .map(|(grad, param_name)| (param_name.clone(), grad))
                    .collect();

                // Synchronize the entire bucket efficiently
                match Self::sync_bucket_gradients_flattened(
                    gradients_with_names,
                    &self.process_group,
                )
                .await
                {
                    Ok(synchronized_gradients) => {
                        // Set the synchronized gradients back to their parameters
                        for (param_name, synchronized_grad) in synchronized_gradients {
                            if let Some(param) = parameters.get(&param_name) {
                                let tensor = param.tensor();
                                let tensor_guard = tensor.read();
                                tensor_guard.set_grad(Some(synchronized_grad));
                            }
                        }
                    }
                    Err(e) => {
                        info!("  Failed to sync bucket gradients efficiently, falling back to individual sync: {}", e);

                        #[allow(clippy::await_holding_lock)]
                        // Fallback to individual gradient synchronization
                        for param_name in &bucket_params {
                            if let Some(param) = parameters.get(param_name) {
                                let tensor = param.tensor();
                                let tensor_guard = tensor.read();

                                if let Some(mut grad) = tensor_guard.grad() {
                                    all_reduce(&mut grad, ReduceOp::Sum, &self.process_group)
                                        .await?;
                                    let world_size = self.process_group.world_size() as f32;
                                    grad = grad.div_scalar(world_size)?;
                                    tensor_guard.set_grad(Some(grad));
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Register gradient synchronization hooks
    /// This should be called during the backward pass to automatically sync gradients
    pub fn register_gradient_hooks(&self) -> TorshResult<()> {
        // In a complete implementation, this would register hooks on each parameter
        // that automatically call all_reduce when gradients are computed

        // For now, we'll use a simpler approach where sync_gradients is called manually
        // after backward() but before optimizer.step()
        Ok(())
    }

    /// Register a gradient for asynchronous synchronization (overlap mode)
    /// This should be called when a gradient becomes available during backward pass
    pub fn register_gradient_async(&self, param_name: &str, gradient: Tensor) -> TorshResult<()> {
        if !self.overlap_config.enabled {
            return Ok(()); // Overlap not enabled, skip
        }

        // Mark parameter as used for unused parameter detection
        if self.overlap_config.detect_unused_parameters {
            let mut tracker = self
                .unused_param_tracker
                .lock()
                .expect("lock should not be poisoned");
            if tracker.enabled {
                tracker.used_parameters.insert(param_name.to_string());
            }
        }

        // Get bucket index for this parameter
        let bucket_index = self.param_to_bucket.get(param_name).copied().unwrap_or(0);

        // Send gradient to background worker
        if let Some(sender) = &self.gradient_sender {
            let message = GradientMessage {
                param_name: param_name.to_string(),
                gradient,
                bucket_index,
            };

            if let Err(e) = sender.send(message) {
                info!("  Failed to send gradient for {}: {}", param_name, e);
            }
        }

        Ok(())
    }

    /// Check for unused parameters and issue warnings
    pub fn check_unused_parameters(&self) -> TorshResult<Vec<String>> {
        if !self.overlap_config.detect_unused_parameters {
            return Ok(Vec::new());
        }

        let tracker = self
            .unused_param_tracker
            .lock()
            .expect("lock should not be poisoned");
        if !tracker.enabled {
            return Ok(Vec::new());
        }

        let unused: Vec<String> = tracker
            .all_parameters
            .difference(&tracker.used_parameters)
            .cloned()
            .collect();

        if !unused.is_empty() {
            info!(
                "  Found {} unused parameters in iteration {}:",
                unused.len(),
                tracker.iteration
            );
            for param in &unused {
                info!("    - {}", param);
            }
        }

        Ok(unused)
    }

    /// Start a new iteration (reset unused parameter tracking)
    pub fn start_iteration(&self) -> TorshResult<()> {
        if self.overlap_config.detect_unused_parameters {
            let mut tracker = self
                .unused_param_tracker
                .lock()
                .expect("lock should not be poisoned");
            if tracker.enabled {
                tracker.used_parameters.clear();
                tracker.iteration += 1;
            }
        }
        Ok(())
    }

    /// Get overlap computation statistics
    pub fn get_overlap_stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();

        stats.insert(
            "overlap_enabled".to_string(),
            serde_json::Value::Bool(self.overlap_config.enabled),
        );
        stats.insert(
            "max_pending_syncs".to_string(),
            serde_json::Value::Number(serde_json::Number::from(
                self.overlap_config.max_pending_syncs,
            )),
        );
        stats.insert(
            "sync_timeout_secs".to_string(),
            serde_json::Value::Number(serde_json::Number::from(
                self.overlap_config.sync_timeout_secs,
            )),
        );
        stats.insert(
            "unused_param_detection".to_string(),
            serde_json::Value::Bool(self.overlap_config.detect_unused_parameters),
        );

        if let Ok(tracker) = self.unused_param_tracker.lock() {
            stats.insert(
                "total_params".to_string(),
                serde_json::Value::Number(serde_json::Number::from(tracker.all_parameters.len())),
            );
            stats.insert(
                "used_params".to_string(),
                serde_json::Value::Number(serde_json::Number::from(tracker.used_parameters.len())),
            );
            stats.insert(
                "current_iteration".to_string(),
                serde_json::Value::Number(serde_json::Number::from(tracker.iteration)),
            );
        }

        // Semaphore availability
        let available_permits = self.sync_semaphore.available_permits();
        stats.insert(
            "available_sync_permits".to_string(),
            serde_json::Value::Number(serde_json::Number::from(available_permits)),
        );

        stats
    }

    /// Check if any parameters have gradients
    pub fn has_gradients(&self) -> bool {
        let parameters = self.module.parameters();

        for (_name, param) in parameters {
            let tensor = param.tensor();
            let tensor_guard = tensor.read();

            if tensor_guard.requires_grad() && tensor_guard.has_grad() {
                return true;
            }
        }

        false
    }

    /// Zero all gradients
    pub fn zero_grad(&mut self) -> TorshResult<()> {
        let parameters = self.module.parameters();

        for (_name, param) in parameters {
            let tensor = param.tensor();
            let tensor_guard = tensor.read();

            if tensor_guard.requires_grad() {
                tensor_guard.set_grad(None);
            }
        }

        Ok(())
    }

    /// Get gradient synchronization statistics
    pub fn get_sync_stats(&self) -> GradientSyncStats {
        let parameters = self.module.named_parameters();
        let mut total_parameters = 0;
        let mut parameters_with_grad = 0;
        let mut total_gradient_size = 0;

        for (_name, param) in parameters {
            let tensor = param.tensor();
            let tensor_guard = tensor.read();

            if tensor_guard.requires_grad() {
                total_parameters += 1;

                if tensor_guard.has_grad() {
                    parameters_with_grad += 1;
                    total_gradient_size += tensor_guard.numel() * std::mem::size_of::<f32>();
                }
            }
        }

        GradientSyncStats {
            total_parameters,
            parameters_with_grad,
            total_gradient_size_mb: total_gradient_size as f32 / (1024.0 * 1024.0),
            num_buckets: self.gradient_buckets.len(),
            world_size: self.process_group.world_size(),
        }
    }

    /// Enable/disable gradient bucketing at runtime
    pub fn set_bucketing_enabled(&mut self, enabled: bool) -> TorshResult<()> {
        self.bucket_config.enabled = enabled;

        if enabled && self.gradient_buckets.is_empty() {
            // Re-initialize buckets if they were disabled
            self.initialize_buckets()?;
        }

        Ok(())
    }

    /// Get bucket information for debugging
    pub fn get_bucket_info(&self) -> Vec<BucketInfo> {
        self.gradient_buckets
            .iter()
            .enumerate()
            .map(|(i, bucket)| BucketInfo {
                index: i,
                size_mb: bucket.size_mb(),
                num_parameters: bucket.parameters.len(),
                parameter_names: bucket.parameters.clone(),
            })
            .collect()
    }

    /// Perform a gradient consistency check across all processes
    /// This is useful for debugging distributed training issues
    pub async fn check_gradient_consistency(&self) -> TorshResult<bool> {
        // In a complete implementation, this would:
        // 1. Compute checksums of gradients on each process
        // 2. Use all_gather to collect checksums from all processes
        // 3. Compare checksums to detect inconsistencies
        // 4. Report which parameters have mismatched gradients

        // For now, just return true as a placeholder
        Ok(true)
    }
}

impl<M: Module> Module for DistributedDataParallel<M> {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Forward through underlying module
        self.module.forward(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.module.parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.module.named_parameters()
    }

    fn training(&self) -> bool {
        self.module.training()
    }

    fn train(&mut self) {
        self.module.train()
    }

    fn eval(&mut self) {
        self.module.eval()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.module.to_device(device)
    }
}
