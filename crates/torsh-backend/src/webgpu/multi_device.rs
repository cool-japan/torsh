//! WebGPU multi-device support and load balancing implementation
//!
//! This module provides advanced multi-device management for WebGPU backends,
//! including intelligent load balancing, work distribution, and device orchestration.

#[cfg(feature = "webgpu")]
use crate::webgpu::wgpu;
use crate::webgpu::{WebGpuDevice, WebGpuError, WebGpuResult};
use parking_lot::{Mutex, RwLock};
use scirs2_core::random::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Load balancing strategies for multi-device WebGPU
#[derive(Debug, Clone, PartialEq)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution across devices
    RoundRobin,
    /// Utilization-based selection (least loaded)
    UtilizationBased,
    /// Performance-based selection (fastest device)
    PerformanceBased,
    /// Memory-based selection (most available memory)
    MemoryBased,
    /// Adaptive strategy that learns optimal distribution
    Adaptive,
    /// Custom strategy with user-defined weights
    Custom(Vec<f32>),
}

/// Multi-device configuration for WebGPU
#[derive(Debug, Clone)]
pub struct MultiDeviceConfig {
    /// Load balancing strategy
    pub strategy: LoadBalancingStrategy,
    /// Maximum number of devices to use
    pub max_devices: usize,
    /// Enable automatic device discovery
    pub auto_discovery: bool,
    /// Device selection criteria
    pub device_filter: DeviceFilter,
    /// Work distribution granularity
    pub work_granularity: WorkGranularity,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    /// Rebalancing interval
    pub rebalance_interval: Duration,
    /// Minimum work size for multi-device distribution
    pub min_work_size: usize,
}

/// Device filtering criteria
#[derive(Debug, Clone)]
pub struct DeviceFilter {
    /// Preferred device types
    pub preferred_types: Vec<wgpu::DeviceType>,
    /// Minimum memory requirement (in bytes)
    pub min_memory: u64,
    /// Required features
    pub required_features: wgpu::Features,
    /// Excluded vendor IDs
    pub excluded_vendors: Vec<u32>,
    /// Maximum device count per type
    pub max_per_type: HashMap<wgpu::DeviceType, usize>,
}

/// Work distribution granularity
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WorkGranularity {
    /// Distribute at operation level
    Operation,
    /// Distribute at tensor level
    Tensor,
    /// Distribute at batch level
    Batch,
    /// Distribute at element level
    Element,
}

/// Device performance metrics
#[derive(Debug, Clone, Default)]
pub struct DeviceMetrics {
    /// Device utilization (0.0 to 1.0)
    pub utilization: f32,
    /// Available memory percentage
    pub memory_available: f32,
    /// Current active operations
    pub active_operations: usize,
    /// Completed operations count
    pub completed_operations: u64,
    /// Average operation latency (microseconds)
    pub avg_latency_us: f64,
    /// Throughput (operations per second)
    pub throughput_ops: f64,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f32,
    /// Temperature estimate (for throttling detection)
    pub temperature_estimate: f32,
    /// Power consumption estimate
    pub power_estimate: f32,
}

/// Device selection context for load balancing
#[derive(Debug, Clone)]
pub struct DeviceSelectionContext {
    /// Work size estimate
    pub work_size: usize,
    /// Operation type
    pub operation_type: String,
    /// Memory requirement
    pub memory_requirement: u64,
    /// Priority level
    pub priority: WorkPriority,
    /// Preferred device types
    pub preferred_devices: Vec<usize>,
    /// Maximum allowed latency
    pub max_latency: Option<Duration>,
}

/// Work priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WorkPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Work distribution plan
#[derive(Debug, Clone)]
pub struct WorkDistributionPlan {
    /// Device assignments
    pub assignments: Vec<DeviceAssignment>,
    /// Total estimated execution time
    pub estimated_time: Duration,
    /// Load balancing efficiency score
    pub efficiency_score: f32,
}

/// Individual device work assignment
#[derive(Debug, Clone)]
pub struct DeviceAssignment {
    /// Device ID
    pub device_id: usize,
    /// Work partition
    pub work_partition: WorkPartition,
    /// Estimated execution time
    pub estimated_time: Duration,
    /// Memory requirement
    pub memory_requirement: u64,
}

/// Work partition specification
#[derive(Debug, Clone)]
pub struct WorkPartition {
    /// Start offset
    pub start: usize,
    /// End offset
    pub end: usize,
    /// Partition weight
    pub weight: f32,
    /// Dependencies on other partitions
    pub dependencies: Vec<usize>,
}

/// Multi-device WebGPU manager
pub struct MultiDeviceWebGpuManager {
    /// Configuration
    config: MultiDeviceConfig,
    /// Available devices
    devices: RwLock<HashMap<usize, Arc<WebGpuDevice>>>,
    /// Device metrics
    device_metrics: Arc<RwLock<HashMap<usize, DeviceMetrics>>>,
    /// Load balancer
    load_balancer: Arc<Mutex<LoadBalancer>>,
    /// Work scheduler
    work_scheduler: Arc<Mutex<WorkScheduler>>,
    /// Performance monitor
    performance_monitor: Arc<PerformanceMonitor>,
    /// Next device ID for round-robin
    next_device_id: AtomicUsize,
    /// Manager statistics
    stats: Arc<RwLock<ManagerStats>>,
}

/// Load balancer implementation
#[derive(Debug)]
struct LoadBalancer {
    strategy: LoadBalancingStrategy,
    device_weights: HashMap<usize, f32>,
    last_selection: HashMap<String, usize>,
    selection_history: VecDeque<(Instant, usize, String)>,
    adaptation_data: AdaptationData,
}

/// Work scheduler for coordinating multi-device execution
#[derive(Debug)]
struct WorkScheduler {
    /// Pending work queue
    pending_work: VecDeque<WorkRequest>,
    /// Active work tracking
    active_work: HashMap<u64, ActiveWork>,
    /// Device queues
    device_queues: HashMap<usize, VecDeque<WorkRequest>>,
    /// Next work ID
    next_work_id: u64,
}

/// Performance monitor for collecting metrics
pub struct PerformanceMonitor {
    /// Device performance history
    device_history: RwLock<HashMap<usize, VecDeque<(Instant, DeviceMetrics)>>>,
    /// System-wide metrics
    system_metrics: RwLock<SystemMetrics>,
    /// Monitoring enabled flag
    enabled: bool,
}

/// System-wide performance metrics
#[derive(Debug, Default, Clone)]
pub struct SystemMetrics {
    /// Total operations executed
    pub total_operations: u64,
    /// Multi-device operations
    pub multi_device_operations: u64,
    /// Load balancing efficiency
    pub load_balance_efficiency: f32,
    /// Average device utilization
    pub avg_device_utilization: f32,
    /// Throughput improvement from multi-device
    pub throughput_improvement: f32,
    /// Memory efficiency across devices
    pub memory_efficiency: f32,
}

/// Manager statistics
#[derive(Debug, Default, Clone)]
pub struct ManagerStats {
    /// Total devices managed
    pub total_devices: usize,
    /// Active devices
    pub active_devices: usize,
    /// Total work distributed
    pub total_work_distributed: u64,
    /// Load balancing decisions
    pub load_balance_decisions: u64,
    /// Device selection accuracy
    pub selection_accuracy: f32,
    /// Rebalancing events
    pub rebalancing_events: u64,
}

/// Adaptation data for learning optimal strategies
#[derive(Debug, Default)]
struct AdaptationData {
    /// Performance history per strategy
    strategy_performance: HashMap<LoadBalancingStrategy, StrategyPerformance>,
    /// Device affinity scores
    device_affinity: HashMap<String, HashMap<usize, f32>>,
    /// Operation timing predictions
    timing_predictions: HashMap<String, TimingModel>,
}

/// Strategy performance tracking
#[derive(Debug, Default)]
struct StrategyPerformance {
    /// Total executions
    executions: u64,
    /// Average execution time
    avg_time: Duration,
    /// Success rate
    success_rate: f32,
    /// Efficiency score
    efficiency: f32,
}

/// Timing model for operation prediction
#[derive(Debug, Default)]
struct TimingModel {
    /// Base time per operation
    base_time: Duration,
    /// Time scaling factor
    scaling_factor: f64,
    /// Device-specific multipliers
    device_multipliers: HashMap<usize, f32>,
    /// Confidence level
    confidence: f32,
}

/// Work request for scheduling
#[derive(Debug, Clone)]
struct WorkRequest {
    /// Unique work ID
    id: u64,
    /// Operation context
    context: DeviceSelectionContext,
    /// Creation timestamp
    created_at: Instant,
    /// Assigned device (if any)
    assigned_device: Option<usize>,
    /// Retry count
    retry_count: usize,
}

/// Active work tracking
#[derive(Debug)]
struct ActiveWork {
    /// Work request
    request: WorkRequest,
    /// Start time
    started_at: Instant,
    /// Estimated completion time
    estimated_completion: Instant,
    /// Current status
    status: WorkStatus,
}

/// Work execution status
#[derive(Debug, Clone, Copy, PartialEq)]
enum WorkStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl Default for MultiDeviceConfig {
    fn default() -> Self {
        Self {
            strategy: LoadBalancingStrategy::UtilizationBased,
            max_devices: 8,
            auto_discovery: true,
            device_filter: DeviceFilter::default(),
            work_granularity: WorkGranularity::Operation,
            enable_monitoring: true,
            rebalance_interval: Duration::from_secs(30),
            min_work_size: 1024,
        }
    }
}

impl Default for DeviceFilter {
    fn default() -> Self {
        Self {
            preferred_types: vec![
                wgpu::DeviceType::DiscreteGpu,
                wgpu::DeviceType::IntegratedGpu,
            ],
            min_memory: 512 * 1024 * 1024, // 512 MB
            required_features: wgpu::Features::empty(),
            excluded_vendors: vec![],
            max_per_type: HashMap::new(),
        }
    }
}

impl MultiDeviceWebGpuManager {
    /// Create a new multi-device manager
    pub fn new(config: MultiDeviceConfig) -> Self {
        Self {
            config,
            devices: RwLock::new(HashMap::new()),
            device_metrics: Arc::new(RwLock::new(HashMap::new())),
            load_balancer: Arc::new(Mutex::new(LoadBalancer::new(
                LoadBalancingStrategy::UtilizationBased,
            ))),
            work_scheduler: Arc::new(Mutex::new(WorkScheduler::new())),
            performance_monitor: Arc::new(PerformanceMonitor::new(true)),
            next_device_id: AtomicUsize::new(0),
            stats: Arc::new(RwLock::new(ManagerStats::default())),
        }
    }

    /// Initialize the multi-device manager
    pub async fn initialize(&self) -> WebGpuResult<()> {
        if self.config.auto_discovery {
            self.discover_devices().await?;
        }

        // Start background monitoring and rebalancing
        if self.config.enable_monitoring {
            self.start_monitoring().await;
        }

        Ok(())
    }

    /// Discover and initialize available WebGPU devices
    async fn discover_devices(&self) -> WebGpuResult<()> {
        let adapters = crate::webgpu::enumerate_adapters().await?;
        let mut device_count = 0;

        for (adapter_index, adapter) in adapters.iter().enumerate() {
            if device_count >= self.config.max_devices {
                break;
            }

            let adapter_info = adapter.get_info();

            // Apply device filtering
            if !self.should_use_device(&adapter_info) {
                continue;
            }

            // Create device
            let device_id = device_count;
            match WebGpuDevice::from_adapter_index(adapter_index, device_id).await {
                Ok(device) => {
                    let device = Arc::new(device);

                    // Store device
                    self.devices.write().insert(device_id, device.clone());

                    // Initialize metrics
                    let initial_metrics = DeviceMetrics::default();
                    self.device_metrics
                        .write()
                        .insert(device_id, initial_metrics);

                    device_count += 1;

                    #[cfg(feature = "webgpu")]
                    log::info!(
                        "Initialized WebGPU device {}: {} ({:?})",
                        device_id,
                        adapter_info.name,
                        adapter_info.device_type
                    );
                    #[cfg(not(feature = "webgpu"))]
                    let _ = (&device_id, &adapter_info);
                }
                Err(e) => {
                    #[cfg(feature = "webgpu")]
                    log::warn!(
                        "Failed to initialize WebGPU device {}: {}",
                        adapter_index,
                        e
                    );
                    #[cfg(not(feature = "webgpu"))]
                    let _ = (&adapter_index, &e);
                }
            }
        }

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_devices = device_count;
            stats.active_devices = device_count;
        }

        if device_count == 0 {
            return Err(WebGpuError::DeviceCreation(
                "No suitable WebGPU devices found".to_string(),
            ));
        }

        #[cfg(feature = "webgpu")]
        log::info!("Initialized {} WebGPU devices", device_count);
        Ok(())
    }

    /// Check if a device should be used based on filtering criteria
    fn should_use_device(&self, adapter_info: &wgpu::AdapterInfo) -> bool {
        let filter = &self.config.device_filter;

        // Check device type preference
        if !filter.preferred_types.is_empty()
            && !filter.preferred_types.contains(&adapter_info.device_type)
        {
            return false;
        }

        // Check excluded vendors
        if filter.excluded_vendors.contains(&adapter_info.vendor) {
            return false;
        }

        // Check per-type limits
        if let Some(&max_count) = filter.max_per_type.get(&adapter_info.device_type) {
            let current_count = self
                .devices
                .read()
                .values()
                .filter(|d| d.adapter_info().device_type == adapter_info.device_type)
                .count();
            if current_count >= max_count {
                return false;
            }
        }

        true
    }

    /// Select optimal device for a given context
    pub fn select_device(&self, context: &DeviceSelectionContext) -> WebGpuResult<usize> {
        let mut load_balancer = self.load_balancer.lock();
        let device_metrics = self.device_metrics.read();
        let devices = self.devices.read();

        if devices.is_empty() {
            return Err(WebGpuError::ResourceNotFound(
                "No devices available".to_string(),
            ));
        }

        let selected_device = match self.config.strategy {
            LoadBalancingStrategy::RoundRobin => self.select_round_robin(&devices),
            LoadBalancingStrategy::UtilizationBased => {
                self.select_by_utilization(&devices, &device_metrics)
            }
            LoadBalancingStrategy::PerformanceBased => {
                self.select_by_performance(&devices, &device_metrics, context)
            }
            LoadBalancingStrategy::MemoryBased => {
                self.select_by_memory(&devices, &device_metrics, context)
            }
            LoadBalancingStrategy::Adaptive => {
                self.select_adaptive(&devices, &device_metrics, context, &mut load_balancer)
            }
            LoadBalancingStrategy::Custom(ref weights) => self.select_custom(&devices, weights),
        };

        // Update selection history
        load_balancer.record_selection(selected_device, &context.operation_type);

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.load_balance_decisions += 1;
        }

        Ok(selected_device)
    }

    /// Round-robin device selection
    fn select_round_robin(&self, devices: &HashMap<usize, Arc<WebGpuDevice>>) -> usize {
        let device_ids: Vec<_> = devices.keys().copied().collect();
        let index = self.next_device_id.fetch_add(1, Ordering::Relaxed) % device_ids.len();
        device_ids[index]
    }

    /// Utilization-based device selection (least loaded)
    fn select_by_utilization(
        &self,
        devices: &HashMap<usize, Arc<WebGpuDevice>>,
        metrics: &HashMap<usize, DeviceMetrics>,
    ) -> usize {
        devices
            .keys()
            .min_by(|&a, &b| {
                let util_a = metrics.get(a).map(|m| m.utilization).unwrap_or(1.0);
                let util_b = metrics.get(b).map(|m| m.utilization).unwrap_or(1.0);
                util_a
                    .partial_cmp(&util_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(0)
    }

    /// Performance-based device selection
    fn select_by_performance(
        &self,
        devices: &HashMap<usize, Arc<WebGpuDevice>>,
        metrics: &HashMap<usize, DeviceMetrics>,
        context: &DeviceSelectionContext,
    ) -> usize {
        devices
            .keys()
            .max_by(|&a, &b| {
                let score_a = self.calculate_performance_score(*a, metrics, context);
                let score_b = self.calculate_performance_score(*b, metrics, context);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(0)
    }

    /// Memory-based device selection
    fn select_by_memory(
        &self,
        devices: &HashMap<usize, Arc<WebGpuDevice>>,
        metrics: &HashMap<usize, DeviceMetrics>,
        context: &DeviceSelectionContext,
    ) -> usize {
        devices
            .keys()
            .filter(|&device_id| {
                // Check if device has enough memory
                if let Some(device) = devices.get(device_id) {
                    let (_used, free) = device.memory_info();
                    free >= context.memory_requirement
                } else {
                    false
                }
            })
            .max_by(|&a, &b| {
                let mem_a = metrics.get(a).map(|m| m.memory_available).unwrap_or(0.0);
                let mem_b = metrics.get(b).map(|m| m.memory_available).unwrap_or(0.0);
                mem_a
                    .partial_cmp(&mem_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(0)
    }

    /// Adaptive device selection with learning
    fn select_adaptive(
        &self,
        devices: &HashMap<usize, Arc<WebGpuDevice>>,
        metrics: &HashMap<usize, DeviceMetrics>,
        context: &DeviceSelectionContext,
        load_balancer: &mut LoadBalancer,
    ) -> usize {
        // Use adaptation data to make informed decisions
        if let Some(best_device) = load_balancer
            .predict_best_device(&context.operation_type, devices.keys().copied().collect())
        {
            best_device
        } else {
            // Fallback to utilization-based
            self.select_by_utilization(devices, metrics)
        }
    }

    /// Custom weighted device selection
    fn select_custom(&self, devices: &HashMap<usize, Arc<WebGpuDevice>>, weights: &[f32]) -> usize {
        let device_ids: Vec<_> = devices.keys().copied().collect();

        // Select based on weights (simple weighted random selection)
        let total_weight: f32 = weights.iter().sum();
        if total_weight <= 0.0 {
            return device_ids[0];
        }

        let mut rng = thread_rng();
        let mut random_val = rng.random::<f32>() * total_weight;
        for (i, &weight) in weights.iter().enumerate() {
            random_val -= weight;
            if random_val <= 0.0 && i < device_ids.len() {
                return device_ids[i];
            }
        }

        device_ids[0]
    }

    /// Calculate performance score for a device
    fn calculate_performance_score(
        &self,
        device_id: usize,
        metrics: &HashMap<usize, DeviceMetrics>,
        context: &DeviceSelectionContext,
    ) -> f32 {
        let metric = metrics.get(&device_id).cloned().unwrap_or_default();

        // Composite score based on multiple factors
        let utilization_score = 1.0 - metric.utilization;
        let throughput_score = (metric.throughput_ops / 1000.0).min(1.0) as f32;
        let latency_score = 1.0 / (1.0 + (metric.avg_latency_us / 1000.0) as f32);
        let memory_score = metric.memory_available;
        let reliability_score = 1.0 - metric.error_rate;

        // Priority-based weighting
        let (util_weight, perf_weight, mem_weight, rel_weight) = match context.priority {
            WorkPriority::Critical => (0.4, 0.3, 0.2, 0.1),
            WorkPriority::High => (0.3, 0.3, 0.2, 0.2),
            WorkPriority::Normal => (0.25, 0.25, 0.25, 0.25),
            WorkPriority::Low => (0.2, 0.2, 0.3, 0.3),
        };

        utilization_score * util_weight
            + throughput_score * perf_weight
            + latency_score * perf_weight
            + memory_score * mem_weight
            + reliability_score * rel_weight
    }

    /// Create work distribution plan for multi-device execution
    pub fn create_distribution_plan(
        &self,
        context: &DeviceSelectionContext,
        total_work_size: usize,
    ) -> WebGpuResult<WorkDistributionPlan> {
        if total_work_size < self.config.min_work_size {
            // Single device execution
            let device_id = self.select_device(context)?;
            return Ok(WorkDistributionPlan {
                assignments: vec![DeviceAssignment {
                    device_id,
                    work_partition: WorkPartition {
                        start: 0,
                        end: total_work_size,
                        weight: 1.0,
                        dependencies: vec![],
                    },
                    estimated_time: self.estimate_execution_time(device_id, total_work_size),
                    memory_requirement: context.memory_requirement,
                }],
                estimated_time: self.estimate_execution_time(device_id, total_work_size),
                efficiency_score: 1.0,
            });
        }

        // Multi-device distribution
        let devices = self.devices.read();
        let metrics = self.device_metrics.read();
        let _num_devices = devices.len().min(self.config.max_devices);

        let mut assignments = Vec::new();
        let mut total_estimated_time = Duration::from_secs(0);

        // Calculate device capabilities and weights
        let mut device_weights = Vec::new();
        let mut total_weight = 0.0f32;

        for &device_id in devices.keys() {
            let weight = self.calculate_device_weight(device_id, &metrics, context);
            device_weights.push((device_id, weight));
            total_weight += weight;
        }

        // Sort by weight (highest first)
        device_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Distribute work proportionally
        let mut remaining_work = total_work_size;
        let mut start_offset = 0;

        for (i, (device_id, weight)) in device_weights.iter().enumerate() {
            if remaining_work == 0 {
                break;
            }

            let work_fraction = if i == device_weights.len() - 1 {
                // Last device gets all remaining work
                1.0
            } else {
                weight / total_weight
            };

            let work_size = (remaining_work as f32 * work_fraction) as usize;
            let actual_work_size = work_size.min(remaining_work);

            if actual_work_size > 0 {
                let partition = WorkPartition {
                    start: start_offset,
                    end: start_offset + actual_work_size,
                    weight: *weight,
                    dependencies: vec![],
                };

                let estimated_time = self.estimate_execution_time(*device_id, actual_work_size);
                total_estimated_time = total_estimated_time.max(estimated_time);

                assignments.push(DeviceAssignment {
                    device_id: *device_id,
                    work_partition: partition,
                    estimated_time,
                    memory_requirement: (context.memory_requirement as f32 * work_fraction) as u64,
                });

                start_offset += actual_work_size;
                remaining_work -= actual_work_size;
            }
        }

        // Calculate efficiency score
        let efficiency_score = self.calculate_efficiency_score(&assignments, total_work_size);

        Ok(WorkDistributionPlan {
            assignments,
            estimated_time: total_estimated_time,
            efficiency_score,
        })
    }

    /// Calculate device weight for work distribution
    fn calculate_device_weight(
        &self,
        device_id: usize,
        metrics: &HashMap<usize, DeviceMetrics>,
        _context: &DeviceSelectionContext,
    ) -> f32 {
        let metric = metrics.get(&device_id).cloned().unwrap_or_default();

        // Base weight from performance characteristics
        let base_weight = match self.devices.read().get(&device_id) {
            Some(device) => {
                let adapter_info = device.adapter_info();
                match adapter_info.device_type {
                    wgpu::DeviceType::DiscreteGpu => 1.0,
                    wgpu::DeviceType::IntegratedGpu => 0.6,
                    wgpu::DeviceType::VirtualGpu => 0.3,
                    wgpu::DeviceType::Cpu => 0.1,
                    wgpu::DeviceType::Other => 0.2,
                }
            }
            None => 0.1,
        };

        // Adjust for current utilization
        let utilization_factor = 1.0 - metric.utilization;

        // Adjust for memory availability
        let memory_factor = metric.memory_available;

        // Adjust for reliability
        let reliability_factor = 1.0 - metric.error_rate;

        base_weight * utilization_factor * memory_factor * reliability_factor
    }

    /// Estimate execution time for a device and work size
    fn estimate_execution_time(&self, device_id: usize, work_size: usize) -> Duration {
        let metrics = self.device_metrics.read();
        let metric = metrics.get(&device_id).cloned().unwrap_or_default();

        // Base estimation from device type and current metrics
        let base_time_per_unit = if metric.throughput_ops > 0.0 {
            Duration::from_secs_f64(1.0 / metric.throughput_ops)
        } else {
            // Fallback estimation based on device type
            match self.devices.read().get(&device_id) {
                Some(device) => {
                    let adapter_info = device.adapter_info();
                    match adapter_info.device_type {
                        wgpu::DeviceType::DiscreteGpu => Duration::from_micros(1),
                        wgpu::DeviceType::IntegratedGpu => Duration::from_micros(3),
                        wgpu::DeviceType::VirtualGpu => Duration::from_micros(10),
                        wgpu::DeviceType::Cpu => Duration::from_micros(50),
                        wgpu::DeviceType::Other => Duration::from_micros(20),
                    }
                }
                None => Duration::from_micros(10),
            }
        };

        base_time_per_unit * work_size as u32
    }

    /// Calculate efficiency score for a distribution plan
    fn calculate_efficiency_score(
        &self,
        assignments: &[DeviceAssignment],
        total_work: usize,
    ) -> f32 {
        if assignments.is_empty() {
            return 0.0;
        }

        // Perfect efficiency would be equal execution time across all devices
        let max_time = assignments
            .iter()
            .map(|a| a.estimated_time)
            .max()
            .unwrap_or(Duration::from_secs(0));
        let min_time = assignments
            .iter()
            .map(|a| a.estimated_time)
            .min()
            .unwrap_or(Duration::from_secs(0));

        if max_time.is_zero() {
            return 1.0;
        }

        // Efficiency based on time balance
        let time_efficiency = min_time.as_secs_f32() / max_time.as_secs_f32();

        // Efficiency based on work distribution
        let ideal_work_per_device = total_work as f32 / assignments.len() as f32;
        let work_variance: f32 = assignments
            .iter()
            .map(|a| {
                let work_size = a.work_partition.end - a.work_partition.start;
                (work_size as f32 - ideal_work_per_device).powi(2)
            })
            .sum::<f32>()
            / assignments.len() as f32;

        let work_efficiency = 1.0 / (1.0 + work_variance / ideal_work_per_device.powi(2));

        // Combined efficiency score
        (time_efficiency + work_efficiency) / 2.0
    }

    /// Start background monitoring and rebalancing
    async fn start_monitoring(&self) {
        let performance_monitor = Arc::clone(&self.performance_monitor);
        let device_metrics = Arc::clone(&self.device_metrics);
        let rebalance_interval = self.config.rebalance_interval;
        // Clone Arc to devices for use in spawned task
        let devices = Arc::new(self.devices.read().clone());

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(rebalance_interval);
            loop {
                interval.tick().await;

                // Update device metrics
                let devices_lock = RwLock::new(devices.as_ref().clone());
                Self::update_device_metrics(&devices_lock, &device_metrics).await;

                // Collect performance data
                performance_monitor.collect_metrics(&device_metrics).await;
            }
        });
    }

    /// Update device metrics
    async fn update_device_metrics(
        devices: &RwLock<HashMap<usize, Arc<WebGpuDevice>>>,
        device_metrics: &RwLock<HashMap<usize, DeviceMetrics>>,
    ) {
        let devices_read = devices.read();
        let mut metrics_write = device_metrics.write();

        for (device_id, device) in devices_read.iter() {
            let (used_memory, free_memory) = device.memory_info();
            let total_memory = used_memory + free_memory;

            let memory_available = if total_memory > 0 {
                free_memory as f32 / total_memory as f32
            } else {
                0.0
            };

            // Estimate utilization based on memory usage and device activity
            let utilization = if total_memory > 0 {
                (used_memory as f32 / total_memory as f32).min(1.0)
            } else {
                0.0
            };

            let metric = metrics_write.entry(*device_id).or_default();
            metric.memory_available = memory_available;
            metric.utilization = utilization;

            // Update other metrics (simplified - real implementation would track more)
            metric.active_operations = 0; // Would track from actual operations
            metric.completed_operations += 1; // Increment for demo
        }
    }

    /// Get device by ID
    pub fn get_device(&self, device_id: usize) -> Option<Arc<WebGpuDevice>> {
        self.devices.read().get(&device_id).cloned()
    }

    /// Get all devices
    pub fn get_devices(&self) -> Vec<Arc<WebGpuDevice>> {
        self.devices.read().values().cloned().collect()
    }

    /// Get device metrics
    pub fn get_device_metrics(&self, device_id: usize) -> Option<DeviceMetrics> {
        self.device_metrics.read().get(&device_id).cloned()
    }

    /// Get system metrics
    pub fn get_system_metrics(&self) -> SystemMetrics {
        self.performance_monitor.system_metrics.read().clone()
    }

    /// Get manager statistics
    pub fn get_stats(&self) -> ManagerStats {
        self.stats.read().clone()
    }
}

impl LoadBalancer {
    fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            strategy,
            device_weights: HashMap::new(),
            last_selection: HashMap::new(),
            selection_history: VecDeque::new(),
            adaptation_data: AdaptationData::default(),
        }
    }

    fn record_selection(&mut self, device_id: usize, operation_type: &str) {
        self.last_selection
            .insert(operation_type.to_string(), device_id);
        self.selection_history
            .push_back((Instant::now(), device_id, operation_type.to_string()));

        // Keep only recent history
        while self.selection_history.len() > 1000 {
            self.selection_history.pop_front();
        }
    }

    fn predict_best_device(
        &self,
        operation_type: &str,
        available_devices: Vec<usize>,
    ) -> Option<usize> {
        // Simple prediction based on recent success
        if let Some(affinity_map) = self.adaptation_data.device_affinity.get(operation_type) {
            available_devices
                .iter()
                .max_by(|&a, &b| {
                    let score_a = affinity_map.get(a).copied().unwrap_or(0.0);
                    let score_b = affinity_map.get(b).copied().unwrap_or(0.0);
                    score_a
                        .partial_cmp(&score_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied()
        } else {
            None
        }
    }
}

impl WorkScheduler {
    fn new() -> Self {
        Self {
            pending_work: VecDeque::new(),
            active_work: HashMap::new(),
            device_queues: HashMap::new(),
            next_work_id: 0,
        }
    }
}

impl PerformanceMonitor {
    fn new(enabled: bool) -> Self {
        Self {
            device_history: RwLock::new(HashMap::new()),
            system_metrics: RwLock::new(SystemMetrics::default()),
            enabled,
        }
    }

    async fn collect_metrics(&self, device_metrics: &RwLock<HashMap<usize, DeviceMetrics>>) {
        if !self.enabled {
            return;
        }

        let metrics = device_metrics.read();
        let mut history = self.device_history.write();
        let now = Instant::now();

        for (device_id, metric) in metrics.iter() {
            let device_history = history.entry(*device_id).or_insert_with(VecDeque::new);
            device_history.push_back((now, metric.clone()));

            // Keep only recent history (last hour)
            while device_history.len() > 3600 {
                device_history.pop_front();
            }
        }

        // Update system metrics
        let mut system_metrics = self.system_metrics.write();
        if !metrics.is_empty() {
            system_metrics.avg_device_utilization =
                metrics.values().map(|m| m.utilization).sum::<f32>() / metrics.len() as f32;
            system_metrics.memory_efficiency =
                metrics.values().map(|m| m.memory_available).sum::<f32>() / metrics.len() as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_device_config_default() {
        let config = MultiDeviceConfig::default();
        assert_eq!(config.strategy, LoadBalancingStrategy::UtilizationBased);
        assert_eq!(config.max_devices, 8);
        assert!(config.auto_discovery);
        assert!(config.enable_monitoring);
    }

    #[test]
    fn test_device_filter_default() {
        let filter = DeviceFilter::default();
        assert!(filter
            .preferred_types
            .contains(&wgpu::DeviceType::DiscreteGpu));
        assert_eq!(filter.min_memory, 512 * 1024 * 1024);
        assert!(filter.excluded_vendors.is_empty());
    }

    #[test]
    fn test_work_priority_ordering() {
        assert!(WorkPriority::Critical > WorkPriority::High);
        assert!(WorkPriority::High > WorkPriority::Normal);
        assert!(WorkPriority::Normal > WorkPriority::Low);
    }

    #[test]
    fn test_device_metrics_default() {
        let metrics = DeviceMetrics::default();
        assert_eq!(metrics.utilization, 0.0);
        assert_eq!(metrics.active_operations, 0);
        assert_eq!(metrics.error_rate, 0.0);
    }

    #[tokio::test]
    async fn test_manager_creation() {
        let config = MultiDeviceConfig::default();
        let manager = MultiDeviceWebGpuManager::new(config);

        let stats = manager.get_stats();
        assert_eq!(stats.total_devices, 0);
        assert_eq!(stats.active_devices, 0);
    }

    #[test]
    fn test_load_balancer_creation() {
        let balancer = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);
        assert_eq!(balancer.strategy, LoadBalancingStrategy::RoundRobin);
        assert!(balancer.device_weights.is_empty());
    }

    #[test]
    fn test_work_scheduler_creation() {
        let scheduler = WorkScheduler::new();
        assert_eq!(scheduler.next_work_id, 0);
        assert!(scheduler.pending_work.is_empty());
        assert!(scheduler.active_work.is_empty());
    }

    #[test]
    fn test_performance_monitor_creation() {
        let monitor = PerformanceMonitor::new(true);
        assert!(monitor.enabled);

        let disabled_monitor = PerformanceMonitor::new(false);
        assert!(!disabled_monitor.enabled);
    }
}
