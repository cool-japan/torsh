//! Transmission scheduling and protocol optimization for communication-efficient distributed training.
//!
//! This module provides comprehensive transmission management capabilities including adaptive
//! scheduling, batching strategies, protocol switching, and quality-of-service guarantees
//! for distributed deep learning communication.
//!
//! # Key Components
//!
//! - **TransmissionScheduler**: Core transmission coordinator with adaptive scheduling algorithms
//! - **Protocol Management**: Dynamic protocol switching and performance optimization
//! - **Batching Strategies**: Intelligent batching for optimal throughput and latency
//! - **Priority Management**: QoS-aware priority queuing and resource allocation
//!
//! # Features
//!
//! - **Adaptive Scheduling**: Dynamic algorithms that adapt to network conditions
//! - **Protocol Switching**: Automatic selection of optimal communication protocols
//! - **Deadline Management**: Support for time-sensitive gradient transmission
//! - **Bandwidth Optimization**: Efficient bandwidth utilization and allocation
//! - **Fault Tolerance**: Robust handling of network failures and protocol degradation
//!
//! # Examples
//!
//! ## Basic Transmission Scheduling
//! ```rust,ignore
//! use torsh_autograd::communication_efficient::transmission::*;
//! use torsh_autograd::communication_efficient::config::*;
//!
//! let mut scheduler = TransmissionScheduler::new();
//!
//! // Configure adaptive scheduling
//! scheduler.enable_adaptive_scheduling(true);
//! scheduler.set_batch_size_limit(64);
//!
//! // Schedule gradient transmission
//! let gradient = CommunicationEfficientGradient::default();
//! scheduler.schedule(gradient).unwrap();
//!
//! // Get next batch for transmission
//! let batch = scheduler.get_next_batch().unwrap();
//! ```
//!
//! ## Protocol Management
//! ```rust,ignore
//! use torsh_autograd::communication_efficient::transmission::*;
//! use torsh_autograd::communication_efficient::config::ProtocolOptimization;
//!
//! let mut protocol_stack = ProtocolStack::new();
//!
//! // Enable automatic protocol switching
//! protocol_stack.enable_protocol_switching(true);
//!
//! // Switch to reliable protocol when needed
//! protocol_stack.switch_to_reliable_protocol().unwrap();
//!
//! // Monitor protocol performance
//! let metrics = protocol_stack.get_protocol_metrics(ProtocolOptimization::TCP);
//! ```
//!
//! ## Advanced Priority Management
//! ```rust,ignore
//! use torsh_autograd::communication_efficient::transmission::*;
//! use torsh_autograd::communication_efficient::config::CommunicationPriority;
//!
//! let mut scheduler = TransmissionScheduler::new();
//! scheduler.set_scheduling_strategy(SchedulingStrategy::WeightedFairQueuing);
//!
//! // Configure priority weights
//! scheduler.set_priority_weight(CommunicationPriority::High, 2.0);
//! scheduler.set_priority_weight(CommunicationPriority::Normal, 1.0);
//! scheduler.set_priority_weight(CommunicationPriority::Low, 0.5);
//!
//! // Enable dynamic priority adjustment
//! scheduler.enable_dynamic_priority_adjustment(true);
//! ```

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::communication_efficient::{
    config::{
        CommunicationConfig, CommunicationEfficientGradient, CommunicationPriority,
        ProtocolOptimization, QualityOfService,
    },
    CommunicationError,
};

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::net::SocketAddr;
use std::time::{Duration, Instant};

/// Main transmission scheduler responsible for coordinating gradient transmission
/// with adaptive scheduling algorithms and quality-of-service guarantees.
///
/// The scheduler supports multiple scheduling strategies, dynamic batching,
/// priority management, and deadline-aware transmission ordering.
#[derive(Debug)]
pub struct TransmissionScheduler {
    /// Current scheduling strategy for transmission ordering
    scheduling_strategy: SchedulingStrategy,
    /// Priority-ordered transmission queue
    transmission_queue: BTreeMap<SchedulingKey, CommunicationEfficientGradient>,
    /// Batch management component
    batch_manager: TransmissionBatchManager,
    /// Priority queue management component
    priority_manager: TransmissionPriorityManager,
    /// Deadline tracking for time-sensitive transmissions
    deadlines: HashMap<u64, Instant>,
    /// Whether to use adaptive scheduling algorithms
    adaptive_scheduling: bool,
    /// Network condition metrics for adaptive decisions
    network_metrics: NetworkMetrics,
    /// Quality of service requirements
    qos_requirements: QualityOfService,
}

/// Scheduling strategies for transmission ordering and prioritization.
///
/// Different strategies optimize for various objectives such as fairness,
/// latency, throughput, or deadline compliance.
#[derive(Debug, Clone, PartialEq)]
pub enum SchedulingStrategy {
    /// First-In-First-Out scheduling for simple fairness
    FIFO,
    /// Priority-based scheduling using communication priority levels
    Priority,
    /// Earliest Deadline First for time-sensitive communications
    EarliestDeadlineFirst,
    /// Shortest Job First for minimizing average completion time
    ShortestJobFirst,
    /// Weighted Fair Queuing for proportional fairness
    WeightedFairQueuing,
    /// Adaptive scheduling that dynamically selects optimal strategy
    AdaptiveScheduling,
}

/// Composite key for scheduling decisions in transmission queue.
///
/// The ordering of fields determines the priority: priority > deadline > size > timestamp.
/// This ensures high-priority, urgent, small transmissions are processed first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulingKey {
    /// Communication priority level (higher priorities processed first)
    pub priority: CommunicationPriority,
    /// Optional deadline for time-sensitive transmissions
    pub deadline: Option<Instant>,
    /// Size-based ordering (smaller transmissions first for SJF)
    pub size: u64,
    /// Timestamp for FIFO ordering among same-priority items
    pub timestamp: Instant,
}

impl PartialOrd for SchedulingKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SchedulingKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse priority comparison so higher priorities come first (are "less than")
        let priority_cmp = other.priority.cmp(&self.priority);
        if priority_cmp != std::cmp::Ordering::Equal {
            return priority_cmp;
        }

        // Earlier deadlines come first (are "less than")
        let deadline_cmp = match (self.deadline, other.deadline) {
            (Some(a), Some(b)) => a.cmp(&b),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        };
        if deadline_cmp != std::cmp::Ordering::Equal {
            return deadline_cmp;
        }

        // Smaller sizes come first (are "less than")
        let size_cmp = self.size.cmp(&other.size);
        if size_cmp != std::cmp::Ordering::Equal {
            return size_cmp;
        }

        // Earlier timestamps come first (are "less than")
        self.timestamp.cmp(&other.timestamp)
    }
}

/// Batch management for optimizing transmission efficiency through grouping.
///
/// Supports adaptive batching that adjusts batch sizes based on network conditions,
/// latency requirements, and throughput objectives.
#[derive(Debug)]
pub struct TransmissionBatchManager {
    /// Current batch being assembled
    pub current_batch: Vec<CommunicationEfficientGradient>,
    /// Maximum number of gradients per batch
    pub batch_size_limit: usize,
    /// Maximum time to wait before sending incomplete batch
    pub batch_time_limit: Duration,
    /// When current batch assembly started
    pub batch_start_time: Option<Instant>,
    /// Whether to use adaptive batching algorithms
    pub adaptive_batching: bool,
    /// Minimum batch size for efficiency
    pub min_batch_size: usize,
    /// Maximum delay tolerance for batching
    pub max_batching_delay: Duration,
}

/// Priority queue management with dynamic weighting and adjustment.
///
/// Manages separate queues for different priority levels and supports
/// dynamic priority adjustment based on network conditions and QoS requirements.
#[derive(Debug)]
pub struct TransmissionPriorityManager {
    /// Separate queues for each priority level
    pub priority_queues: HashMap<CommunicationPriority, VecDeque<CommunicationEfficientGradient>>,
    /// Weight assigned to each priority level for fair queuing
    pub priority_weights: HashMap<CommunicationPriority, f64>,
    /// Whether to dynamically adjust priorities
    pub dynamic_priority_adjustment: bool,
    /// Historical performance metrics per priority
    pub priority_metrics: HashMap<CommunicationPriority, PriorityMetrics>,
    /// Current round-robin state for weighted fair queuing
    pub round_robin_state: HashMap<CommunicationPriority, f64>,
}

/// Performance metrics for priority level monitoring.
#[derive(Debug, Clone, Default)]
pub struct PriorityMetrics {
    /// Average transmission latency for this priority
    pub avg_latency: Duration,
    /// Throughput achieved for this priority
    pub throughput: f64,
    /// Number of transmissions for this priority
    pub transmission_count: u64,
    /// Success rate for this priority
    pub success_rate: f64,
}

/// Network condition metrics for adaptive scheduling decisions.
#[derive(Debug, Clone, Default)]
pub struct NetworkMetrics {
    /// Current available bandwidth
    pub available_bandwidth: u64,
    /// Average network latency
    pub avg_latency: Duration,
    /// Packet loss rate
    pub packet_loss_rate: f64,
    /// Network congestion level (0.0 to 1.0)
    pub congestion_level: f64,
    /// Jitter measurement
    pub jitter: Duration,
}

/// Protocol stack for dynamic protocol selection and switching.
///
/// Manages multiple communication protocols and automatically selects
/// the optimal protocol based on network conditions and requirements.
#[derive(Debug)]
pub struct ProtocolStack {
    /// Available communication protocols
    protocols: HashMap<ProtocolOptimization, Box<dyn CommunicationProtocol>>,
    /// Currently active protocol
    active_protocol: ProtocolOptimization,
    /// Whether automatic protocol switching is enabled
    protocol_switching: bool,
    /// Performance metrics for each protocol
    protocol_metrics: HashMap<ProtocolOptimization, ProtocolMetrics>,
    /// Protocol selection history for learning
    selection_history: VecDeque<ProtocolSelection>,
    /// Switching threshold for protocol changes
    switching_threshold: f64,
}

/// Historical record of protocol selection decisions.
#[derive(Debug, Clone)]
pub struct ProtocolSelection {
    /// Protocol that was selected
    pub protocol: ProtocolOptimization,
    /// Network conditions at time of selection
    pub network_conditions: NetworkMetrics,
    /// Resulting performance metrics
    pub performance: ProtocolMetrics,
    /// Timestamp of selection
    pub timestamp: Instant,
}

/// Performance metrics for protocol evaluation and switching decisions.
///
/// These metrics are used to evaluate protocol performance and make
/// informed decisions about when to switch protocols.
#[derive(Debug, Default, Clone)]
pub struct ProtocolMetrics {
    /// Average latency for this protocol
    pub latency: Duration,
    /// Throughput achieved by this protocol
    pub throughput: f64,
    /// Reliability score (0.0 to 1.0)
    pub reliability: f64,
    /// Energy efficiency score (0.0 to 1.0)
    pub energy_efficiency: f64,
    /// CPU usage ratio (0.0 to 1.0)
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
}

/// Trait for communication protocol implementations.
///
/// Defines the interface that all communication protocols must implement
/// for integration with the protocol stack and transmission scheduler.
pub trait CommunicationProtocol: std::fmt::Debug + Send + Sync {
    /// Send data to specified destination
    ///
    /// # Arguments
    ///
    /// * `data` - Raw data bytes to transmit
    /// * `destination` - Target network address
    ///
    /// # Returns
    ///
    /// Result indicating success or failure of transmission
    fn send(&mut self, data: &[u8], destination: SocketAddr) -> Result<(), CommunicationError>;

    /// Receive data into provided buffer
    ///
    /// # Arguments
    ///
    /// * `buffer` - Buffer to store received data
    ///
    /// # Returns
    ///
    /// Number of bytes received or error
    fn receive(&mut self, buffer: &mut [u8]) -> Result<usize, CommunicationError>;

    /// Get current performance metrics for this protocol
    fn get_metrics(&self) -> ProtocolMetrics;

    /// Configure protocol with provided settings
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration parameters for the protocol
    fn configure(&mut self, config: &CommunicationConfig) -> Result<(), CommunicationError>;

    /// Check if protocol is suitable for current network conditions
    ///
    /// # Arguments
    ///
    /// * `network_metrics` - Current network condition measurements
    ///
    /// # Returns
    ///
    /// Suitability score (0.0 to 1.0, higher is better)
    fn evaluate_suitability(&self, network_metrics: &NetworkMetrics) -> f64;

    /// Get protocol-specific optimization parameters
    fn get_optimization_params(&self) -> HashMap<String, f64>;
}

impl Default for TransmissionScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl TransmissionScheduler {
    /// Create a new transmission scheduler with default configuration.
    ///
    /// Initializes the scheduler with priority-based scheduling, adaptive batching,
    /// and dynamic priority adjustment enabled.
    pub fn new() -> Self {
        Self {
            scheduling_strategy: SchedulingStrategy::Priority,
            transmission_queue: BTreeMap::new(),
            batch_manager: TransmissionBatchManager {
                current_batch: Vec::new(),
                batch_size_limit: 32,
                batch_time_limit: Duration::from_millis(100),
                batch_start_time: None,
                adaptive_batching: true,
                min_batch_size: 4,
                max_batching_delay: Duration::from_millis(50),
            },
            priority_manager: TransmissionPriorityManager {
                priority_queues: HashMap::new(),
                priority_weights: HashMap::new(),
                dynamic_priority_adjustment: true,
                priority_metrics: HashMap::new(),
                round_robin_state: HashMap::new(),
            },
            deadlines: HashMap::new(),
            adaptive_scheduling: true,
            network_metrics: NetworkMetrics::default(),
            qos_requirements: QualityOfService::default(),
        }
    }

    /// Create a new scheduler with specific configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Communication configuration parameters
    pub fn with_config(config: CommunicationConfig) -> Self {
        let mut scheduler = Self::new();

        // Configure batch manager based on configuration flags
        if config.latency_optimization {
            scheduler.batch_manager.batch_time_limit = Duration::from_millis(50);
            scheduler.batch_manager.max_batching_delay = Duration::from_millis(10);
            scheduler.batch_manager.batch_size_limit = 16; // Smaller batches for low latency
        }

        if let Some(budget) = config.communication_budget {
            // Adjust batch size based on bandwidth budget
            let budget_factor = (budget as f64 / 1_000_000.0).max(1.0);
            scheduler.batch_manager.batch_size_limit = ((budget_factor * 32.0) as usize).max(8);
        }

        scheduler
    }

    /// Schedule a gradient for transmission.
    ///
    /// Adds the gradient to the transmission queue according to the current
    /// scheduling strategy and updates relevant metrics.
    ///
    /// # Arguments
    ///
    /// * `gradient` - The gradient to schedule for transmission
    ///
    /// # Returns
    ///
    /// Result indicating success or failure of scheduling operation
    pub fn schedule(
        &mut self,
        gradient: CommunicationEfficientGradient,
    ) -> Result<(), CommunicationError> {
        // Create scheduling key based on current strategy
        let key = self.create_scheduling_key(&gradient)?;

        // Update priority queues if using priority-based scheduling
        if matches!(
            self.scheduling_strategy,
            SchedulingStrategy::Priority | SchedulingStrategy::WeightedFairQueuing
        ) {
            self.priority_manager
                .priority_queues
                .entry(gradient.priority.clone())
                .or_insert_with(VecDeque::new)
                .push_back(gradient.clone());
        }

        // Add to main transmission queue
        self.transmission_queue.insert(key, gradient);

        // Update network metrics if adaptive scheduling is enabled
        if self.adaptive_scheduling {
            self.update_network_metrics();
            self.adapt_scheduling_strategy();
        }

        Ok(())
    }

    /// Get the next batch of gradients for transmission.
    ///
    /// Assembles a batch of gradients according to the current batching strategy
    /// and scheduling algorithm, considering QoS requirements and network conditions.
    ///
    /// # Returns
    ///
    /// Vector of gradients ready for transmission or error
    pub fn get_next_batch(
        &mut self,
    ) -> Result<Vec<CommunicationEfficientGradient>, CommunicationError> {
        let mut batch = Vec::new();
        let mut keys_to_remove = Vec::new();

        // Determine effective batch size
        let effective_batch_size = if self.batch_manager.adaptive_batching {
            self.calculate_adaptive_batch_size()
        } else {
            self.batch_manager.batch_size_limit
        };

        // Check if we should force batch completion due to time constraints
        let _force_batch = self.should_force_batch_completion();

        // Collect gradients according to scheduling strategy
        match self.scheduling_strategy {
            SchedulingStrategy::FIFO => {
                self.collect_fifo_batch(&mut batch, &mut keys_to_remove, effective_batch_size)?;
            }
            SchedulingStrategy::Priority => {
                self.collect_priority_batch(&mut batch, &mut keys_to_remove, effective_batch_size)?;
            }
            SchedulingStrategy::EarliestDeadlineFirst => {
                self.collect_edf_batch(&mut batch, &mut keys_to_remove, effective_batch_size)?;
            }
            SchedulingStrategy::ShortestJobFirst => {
                self.collect_sjf_batch(&mut batch, &mut keys_to_remove, effective_batch_size)?;
            }
            SchedulingStrategy::WeightedFairQueuing => {
                self.collect_wfq_batch(&mut batch, &mut keys_to_remove, effective_batch_size)?;
            }
            SchedulingStrategy::AdaptiveScheduling => {
                self.collect_adaptive_batch(&mut batch, &mut keys_to_remove, effective_batch_size)?;
            }
        }

        // Remove scheduled gradients from queue
        for key in keys_to_remove {
            self.transmission_queue.remove(&key);
        }

        // Update batch timing
        if !batch.is_empty() {
            self.batch_manager.batch_start_time = Some(Instant::now());
        }

        // Update priority metrics
        self.update_priority_metrics(&batch);

        Ok(batch)
    }

    /// Set the scheduling strategy for transmission ordering.
    ///
    /// # Arguments
    ///
    /// * `strategy` - New scheduling strategy to use
    pub fn set_scheduling_strategy(&mut self, strategy: SchedulingStrategy) {
        self.scheduling_strategy = strategy;
    }

    /// Enable or disable adaptive scheduling.
    ///
    /// When enabled, the scheduler will automatically adjust its strategy
    /// based on network conditions and performance metrics.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to enable adaptive scheduling
    pub fn enable_adaptive_scheduling(&mut self, enabled: bool) {
        self.adaptive_scheduling = enabled;
    }

    /// Set the batch size limit for transmission batching.
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum number of gradients per batch
    pub fn set_batch_size_limit(&mut self, limit: usize) {
        self.batch_manager.batch_size_limit = limit;
    }

    /// Set the batch time limit for incomplete batches.
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum time to wait before sending incomplete batch
    pub fn set_batch_time_limit(&mut self, limit: Duration) {
        self.batch_manager.batch_time_limit = limit;
    }

    /// Set priority weight for weighted fair queuing.
    ///
    /// # Arguments
    ///
    /// * `priority` - Priority level to configure
    /// * `weight` - Weight value (higher values get more bandwidth)
    pub fn set_priority_weight(&mut self, priority: CommunicationPriority, weight: f64) {
        self.priority_manager
            .priority_weights
            .insert(priority, weight);
    }

    /// Enable or disable dynamic priority adjustment.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to enable dynamic priority adjustment
    pub fn enable_dynamic_priority_adjustment(&mut self, enabled: bool) {
        self.priority_manager.dynamic_priority_adjustment = enabled;
    }

    /// Get current transmission queue length.
    pub fn queue_length(&self) -> usize {
        self.transmission_queue.len()
    }

    /// Get current network metrics.
    pub fn get_network_metrics(&self) -> &NetworkMetrics {
        &self.network_metrics
    }

    /// Update quality of service requirements.
    ///
    /// # Arguments
    ///
    /// * `qos` - New QoS requirements
    pub fn update_qos_requirements(&mut self, qos: QualityOfService) {
        self.qos_requirements = qos;

        // Adjust batch management based on new QoS requirements
        self.batch_manager.batch_time_limit = self.qos_requirements.max_latency / 2;
        self.batch_manager.max_batching_delay = self.qos_requirements.max_latency / 4;

        let throughput_factor = (self.qos_requirements.min_bandwidth as f64 / 1_000_000.0).max(1.0);
        self.batch_manager.batch_size_limit = ((throughput_factor * 32.0) as usize).max(8);
    }

    // Private helper methods for scheduling implementation

    fn create_scheduling_key(
        &self,
        gradient: &CommunicationEfficientGradient,
    ) -> Result<SchedulingKey, CommunicationError> {
        let deadline = self.deadlines.get(&gradient.round).copied();

        Ok(SchedulingKey {
            priority: gradient.priority.clone(),
            deadline,
            size: gradient.bandwidth_requirement,
            timestamp: gradient.timestamp,
        })
    }

    fn calculate_adaptive_batch_size(&self) -> usize {
        let base_size = self.batch_manager.batch_size_limit;

        // Adjust based on network conditions
        let network_factor = if self.network_metrics.congestion_level > 0.7 {
            0.5 // Reduce batch size during congestion
        } else if self.network_metrics.congestion_level < 0.3 {
            1.5 // Increase batch size when network is clear
        } else {
            1.0
        };

        // Adjust based on latency requirements
        let latency_factor = if self.network_metrics.avg_latency > self.qos_requirements.max_latency
        {
            0.7 // Smaller batches for better latency
        } else {
            1.2
        };

        ((base_size as f64 * network_factor * latency_factor) as usize)
            .max(self.batch_manager.min_batch_size)
            .min(self.batch_manager.batch_size_limit * 2)
    }

    fn should_force_batch_completion(&self) -> bool {
        if let Some(start_time) = self.batch_manager.batch_start_time {
            start_time.elapsed() >= self.batch_manager.batch_time_limit
        } else {
            false
        }
    }

    fn collect_fifo_batch(
        &self,
        batch: &mut Vec<CommunicationEfficientGradient>,
        keys_to_remove: &mut Vec<SchedulingKey>,
        batch_size: usize,
    ) -> Result<(), CommunicationError> {
        for (key, gradient) in &self.transmission_queue {
            if batch.len() >= batch_size {
                break;
            }
            batch.push(gradient.clone());
            keys_to_remove.push(key.clone());
        }
        Ok(())
    }

    fn collect_priority_batch(
        &self,
        batch: &mut Vec<CommunicationEfficientGradient>,
        keys_to_remove: &mut Vec<SchedulingKey>,
        batch_size: usize,
    ) -> Result<(), CommunicationError> {
        // Priority-based collection (BTreeMap already sorts by priority)
        for (key, gradient) in &self.transmission_queue {
            if batch.len() >= batch_size {
                break;
            }
            batch.push(gradient.clone());
            keys_to_remove.push(key.clone());
        }
        Ok(())
    }

    fn collect_edf_batch(
        &self,
        batch: &mut Vec<CommunicationEfficientGradient>,
        keys_to_remove: &mut Vec<SchedulingKey>,
        batch_size: usize,
    ) -> Result<(), CommunicationError> {
        // Sort by deadline first, then by other criteria
        let mut deadline_sorted: Vec<_> = self.transmission_queue.iter().collect();
        deadline_sorted.sort_by(|a, b| match (a.0.deadline, b.0.deadline) {
            (Some(d1), Some(d2)) => d1.cmp(&d2),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => a.0.cmp(b.0),
        });

        for (key, gradient) in deadline_sorted.into_iter().take(batch_size) {
            batch.push(gradient.clone());
            keys_to_remove.push(key.clone());
        }
        Ok(())
    }

    fn collect_sjf_batch(
        &self,
        batch: &mut Vec<CommunicationEfficientGradient>,
        keys_to_remove: &mut Vec<SchedulingKey>,
        batch_size: usize,
    ) -> Result<(), CommunicationError> {
        // Sort by size (shortest job first)
        let mut size_sorted: Vec<_> = self.transmission_queue.iter().collect();
        size_sorted.sort_by_key(|(key, _)| key.size);

        for (key, gradient) in size_sorted.into_iter().take(batch_size) {
            batch.push(gradient.clone());
            keys_to_remove.push(key.clone());
        }
        Ok(())
    }

    fn collect_wfq_batch(
        &self,
        batch: &mut Vec<CommunicationEfficientGradient>,
        keys_to_remove: &mut Vec<SchedulingKey>,
        batch_size: usize,
    ) -> Result<(), CommunicationError> {
        // Weighted fair queuing implementation
        let mut selected_count = 0;
        let mut round_robin_state = self.priority_manager.round_robin_state.clone();

        while selected_count < batch_size && !self.transmission_queue.is_empty() {
            let mut selected_any = false;

            // Process each priority level according to its weight
            for (priority, weight) in &self.priority_manager.priority_weights {
                let current_credit = round_robin_state.entry(priority.clone()).or_insert(0.0);
                *current_credit += weight;

                if *current_credit >= 1.0 {
                    // Select gradient from this priority queue
                    if let Some((key, gradient)) = self.find_next_for_priority(priority) {
                        batch.push(gradient.clone());
                        keys_to_remove.push(key);
                        selected_count += 1;
                        *current_credit -= 1.0;
                        selected_any = true;

                        if selected_count >= batch_size {
                            break;
                        }
                    }
                }
            }

            if !selected_any {
                break; // No more gradients available
            }
        }

        Ok(())
    }

    fn collect_adaptive_batch(
        &self,
        batch: &mut Vec<CommunicationEfficientGradient>,
        keys_to_remove: &mut Vec<SchedulingKey>,
        batch_size: usize,
    ) -> Result<(), CommunicationError> {
        // Adaptive strategy selection based on current conditions
        let strategy = self.select_optimal_strategy();

        match strategy {
            SchedulingStrategy::Priority => {
                self.collect_priority_batch(batch, keys_to_remove, batch_size)
            }
            SchedulingStrategy::EarliestDeadlineFirst => {
                self.collect_edf_batch(batch, keys_to_remove, batch_size)
            }
            SchedulingStrategy::ShortestJobFirst => {
                self.collect_sjf_batch(batch, keys_to_remove, batch_size)
            }
            SchedulingStrategy::WeightedFairQueuing => {
                self.collect_wfq_batch(batch, keys_to_remove, batch_size)
            }
            _ => self.collect_fifo_batch(batch, keys_to_remove, batch_size),
        }
    }

    fn find_next_for_priority(
        &self,
        priority: &CommunicationPriority,
    ) -> Option<(SchedulingKey, &CommunicationEfficientGradient)> {
        self.transmission_queue
            .iter()
            .find(|(key, _)| key.priority == *priority)
            .map(|(key, gradient)| (key.clone(), gradient))
    }

    fn select_optimal_strategy(&self) -> SchedulingStrategy {
        // Simple heuristic for strategy selection
        if self.has_urgent_deadlines() {
            SchedulingStrategy::EarliestDeadlineFirst
        } else if self.network_metrics.congestion_level > 0.7 {
            SchedulingStrategy::ShortestJobFirst
        } else if self.has_mixed_priorities() {
            SchedulingStrategy::WeightedFairQueuing
        } else {
            SchedulingStrategy::Priority
        }
    }

    fn has_urgent_deadlines(&self) -> bool {
        let now = Instant::now();
        self.deadlines
            .values()
            .any(|&deadline| deadline.saturating_duration_since(now) < Duration::from_millis(100))
    }

    fn has_mixed_priorities(&self) -> bool {
        let priorities: std::collections::HashSet<_> = self
            .transmission_queue
            .keys()
            .map(|key| &key.priority)
            .collect();
        priorities.len() > 1
    }

    fn update_network_metrics(&mut self) {
        // Placeholder for network metrics collection
        // In a real implementation, this would gather actual network statistics
        // For now, we'll simulate some basic metrics

        // This would typically be implemented by:
        // - Measuring actual bandwidth usage
        // - Tracking latency measurements
        // - Monitoring packet loss
        // - Calculating congestion levels
    }

    fn adapt_scheduling_strategy(&mut self) {
        if !self.adaptive_scheduling {
            return;
        }

        // Adapt strategy based on current conditions
        let optimal_strategy = self.select_optimal_strategy();
        if optimal_strategy != self.scheduling_strategy {
            self.scheduling_strategy = optimal_strategy;
        }
    }

    fn update_priority_metrics(&mut self, batch: &[CommunicationEfficientGradient]) {
        let batch_time = Instant::now();

        for gradient in batch {
            let metrics = self
                .priority_manager
                .priority_metrics
                .entry(gradient.priority.clone())
                .or_insert_with(PriorityMetrics::default);

            // Update transmission count
            metrics.transmission_count += 1;

            // Update average latency (simplified calculation)
            let transmission_latency = batch_time.duration_since(gradient.timestamp);
            metrics.avg_latency = if metrics.transmission_count == 1 {
                transmission_latency
            } else {
                Duration::from_nanos(
                    (metrics.avg_latency.as_nanos() as f64 * 0.9
                        + transmission_latency.as_nanos() as f64 * 0.1) as u64,
                )
            };
        }
    }
}

impl Default for ProtocolStack {
    fn default() -> Self {
        Self::new()
    }
}

impl ProtocolStack {
    /// Create a new protocol stack with default configuration.
    pub fn new() -> Self {
        Self {
            protocols: HashMap::new(),
            active_protocol: ProtocolOptimization::TCP,
            protocol_switching: true,
            protocol_metrics: HashMap::new(),
            selection_history: VecDeque::with_capacity(100),
            switching_threshold: 0.1, // 10% performance improvement required for switching
        }
    }

    /// Register a new communication protocol with the stack.
    ///
    /// # Arguments
    ///
    /// * `protocol_type` - Type identifier for the protocol
    /// * `protocol` - Protocol implementation
    pub fn register_protocol(
        &mut self,
        protocol_type: ProtocolOptimization,
        protocol: Box<dyn CommunicationProtocol>,
    ) {
        self.protocols.insert(protocol_type, protocol);
        self.protocol_metrics
            .insert(protocol_type, ProtocolMetrics::default());
    }

    /// Switch to a reliable protocol for critical communications.
    ///
    /// This method switches to the most reliable protocol available,
    /// typically TCP for guaranteed delivery.
    pub fn switch_to_reliable_protocol(&mut self) -> Result<(), CommunicationError> {
        // Prefer TCP for reliability, fallback to UDP if needed
        let reliable_protocol = if self.protocols.contains_key(&ProtocolOptimization::TCP) {
            ProtocolOptimization::TCP
        } else if self.protocols.contains_key(&ProtocolOptimization::UDP) {
            ProtocolOptimization::UDP
        } else {
            return Err(CommunicationError::ProtocolError);
        };

        self.active_protocol = reliable_protocol;
        self.record_protocol_selection(reliable_protocol, NetworkMetrics::default());
        Ok(())
    }

    /// Switch to a high-throughput protocol for bulk data transfer.
    ///
    /// This method selects the protocol with the highest throughput
    /// characteristics for efficient bulk data transmission.
    pub fn switch_to_high_throughput_protocol(&mut self) -> Result<(), CommunicationError> {
        // Find protocol with highest throughput
        let best_protocol = self
            .protocol_metrics
            .iter()
            .max_by(|(_, a), (_, b)| {
                a.throughput
                    .partial_cmp(&b.throughput)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(proto, _)| *proto)
            .unwrap_or(ProtocolOptimization::UDP);

        if self.protocols.contains_key(&best_protocol) {
            self.active_protocol = best_protocol;
            self.record_protocol_selection(best_protocol, NetworkMetrics::default());
            Ok(())
        } else {
            Err(CommunicationError::ProtocolError)
        }
    }

    /// Enable or disable automatic protocol switching.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to enable automatic protocol switching
    pub fn enable_protocol_switching(&mut self, enabled: bool) {
        self.protocol_switching = enabled;
    }

    /// Get performance metrics for a specific protocol.
    ///
    /// # Arguments
    ///
    /// * `protocol` - Protocol to get metrics for
    ///
    /// # Returns
    ///
    /// Performance metrics or None if protocol not registered
    pub fn get_protocol_metrics(&self, protocol: ProtocolOptimization) -> Option<&ProtocolMetrics> {
        self.protocol_metrics.get(&protocol)
    }

    /// Get the currently active protocol.
    pub fn get_active_protocol(&self) -> ProtocolOptimization {
        self.active_protocol
    }

    /// Evaluate and potentially switch to optimal protocol based on conditions.
    ///
    /// # Arguments
    ///
    /// * `network_metrics` - Current network condition measurements
    pub fn optimize_protocol_selection(
        &mut self,
        network_metrics: &NetworkMetrics,
    ) -> Result<(), CommunicationError> {
        if !self.protocol_switching {
            return Ok(());
        }

        let current_score =
            self.evaluate_protocol_performance(self.active_protocol, network_metrics);
        let mut best_protocol = self.active_protocol;
        let mut best_score = current_score;

        // Evaluate all available protocols
        for protocol in self.protocols.keys().copied() {
            let score = self.evaluate_protocol_performance(protocol, network_metrics);
            if score > best_score + self.switching_threshold {
                best_score = score;
                best_protocol = protocol;
            }
        }

        // Switch if we found a significantly better protocol
        if best_protocol != self.active_protocol {
            self.active_protocol = best_protocol;
            self.record_protocol_selection(best_protocol, network_metrics.clone());
        }

        Ok(())
    }

    /// Update metrics for a protocol based on recent performance.
    ///
    /// # Arguments
    ///
    /// * `protocol` - Protocol to update metrics for
    /// * `metrics` - New performance measurements
    pub fn update_protocol_metrics(
        &mut self,
        protocol: ProtocolOptimization,
        metrics: ProtocolMetrics,
    ) {
        self.protocol_metrics.insert(protocol, metrics);
    }

    /// Set the threshold for protocol switching decisions.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum performance improvement required for switching
    pub fn set_switching_threshold(&mut self, threshold: f64) {
        self.switching_threshold = threshold.max(0.0).min(1.0);
    }

    /// Get protocol selection history for analysis.
    pub fn get_selection_history(&self) -> &VecDeque<ProtocolSelection> {
        &self.selection_history
    }

    // Private helper methods

    fn evaluate_protocol_performance(
        &self,
        protocol: ProtocolOptimization,
        network_metrics: &NetworkMetrics,
    ) -> f64 {
        if let Some(protocol_impl) = self.protocols.get(&protocol) {
            protocol_impl.evaluate_suitability(network_metrics)
        } else {
            0.0
        }
    }

    fn record_protocol_selection(
        &mut self,
        protocol: ProtocolOptimization,
        network_conditions: NetworkMetrics,
    ) {
        let selection = ProtocolSelection {
            protocol,
            network_conditions,
            performance: self
                .protocol_metrics
                .get(&protocol)
                .cloned()
                .unwrap_or_default(),
            timestamp: Instant::now(),
        };

        self.selection_history.push_back(selection);

        // Keep history bounded
        if self.selection_history.len() > 100 {
            self.selection_history.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transmission_scheduler_creation() {
        let scheduler = TransmissionScheduler::new();
        assert!(matches!(
            scheduler.scheduling_strategy,
            SchedulingStrategy::Priority
        ));
        assert!(scheduler.adaptive_scheduling);
        assert_eq!(scheduler.queue_length(), 0);
    }

    #[test]
    fn test_protocol_stack_creation() {
        let stack = ProtocolStack::new();
        assert!(matches!(stack.active_protocol, ProtocolOptimization::TCP));
        assert!(stack.protocol_switching);
    }

    #[test]
    fn test_scheduling_key_ordering() {
        let now = Instant::now();

        let key1 = SchedulingKey {
            priority: CommunicationPriority::High,
            deadline: Some(now),
            size: 100,
            timestamp: now,
        };

        let key2 = SchedulingKey {
            priority: CommunicationPriority::Low,
            deadline: Some(now),
            size: 50,
            timestamp: now,
        };

        // High priority should come before low priority
        assert!(key1 < key2);

        // Test priority ordering more thoroughly
        let high_priority = SchedulingKey {
            priority: CommunicationPriority::High,
            deadline: None,
            size: 100,
            timestamp: now,
        };

        let low_priority = SchedulingKey {
            priority: CommunicationPriority::Low,
            deadline: None,
            size: 100,
            timestamp: now,
        };

        assert!(
            high_priority < low_priority,
            "High priority should be less than low priority for ordering"
        );

        // Test size ordering with same priority
        let small_task = SchedulingKey {
            priority: CommunicationPriority::Normal,
            deadline: None,
            size: 10,
            timestamp: now,
        };

        let large_task = SchedulingKey {
            priority: CommunicationPriority::Normal,
            deadline: None,
            size: 100,
            timestamp: now,
        };

        assert!(
            small_task < large_task,
            "Smaller tasks should come before larger tasks"
        );
    }

    #[test]
    fn test_scheduler_configuration() {
        let mut scheduler = TransmissionScheduler::new();

        scheduler.set_scheduling_strategy(SchedulingStrategy::FIFO);
        assert!(matches!(
            scheduler.scheduling_strategy,
            SchedulingStrategy::FIFO
        ));

        scheduler.set_batch_size_limit(64);
        assert_eq!(scheduler.batch_manager.batch_size_limit, 64);

        scheduler.enable_adaptive_scheduling(false);
        assert!(!scheduler.adaptive_scheduling);
    }

    #[test]
    fn test_priority_weight_configuration() {
        let mut scheduler = TransmissionScheduler::new();

        scheduler.set_priority_weight(CommunicationPriority::High, 2.0);
        scheduler.set_priority_weight(CommunicationPriority::Normal, 1.0);

        assert_eq!(
            scheduler
                .priority_manager
                .priority_weights
                .get(&CommunicationPriority::High),
            Some(&2.0)
        );
        assert_eq!(
            scheduler
                .priority_manager
                .priority_weights
                .get(&CommunicationPriority::Normal),
            Some(&1.0)
        );
    }

    #[test]
    fn test_adaptive_batch_size_calculation() {
        let scheduler = TransmissionScheduler::new();
        let batch_size = scheduler.calculate_adaptive_batch_size();

        // Should be within reasonable bounds
        assert!(batch_size >= scheduler.batch_manager.min_batch_size);
        assert!(batch_size <= scheduler.batch_manager.batch_size_limit * 2);
    }

    #[test]
    fn test_protocol_stack_switching() {
        let mut stack = ProtocolStack::new();

        assert!(stack.switch_to_reliable_protocol().is_err()); // No protocols registered yet

        stack.enable_protocol_switching(false);
        assert!(!stack.protocol_switching);

        stack.set_switching_threshold(0.2);
        assert_eq!(stack.switching_threshold, 0.2);
    }
}
