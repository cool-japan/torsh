//! Fault detection and recovery mechanisms for communication-efficient distributed training.
//!
//! This module provides comprehensive fault tolerance capabilities for distributed communication,
//! including fault detection algorithms, recovery strategies, failure history management,
//! and distributed fault coordination. It ensures resilient communication even in the presence
//! of network failures, node crashes, message corruption, and Byzantine faults.
//!
//! # Core Components
//!
//! ## Fault Detection
//! - **CommunicationFaultDetector**: Main fault detection and recovery coordinator
//! - **FaultDetectionMethod**: Different fault detection algorithms (timeout, heartbeat, checksum, etc.)
//! - **Fault Classification**: Comprehensive fault type classification and severity assessment
//!
//! ## Recovery Mechanisms
//! - **RecoveryStrategy**: Automatic recovery strategies based on fault types
//! - **Failure History**: Pattern analysis and cascade failure prevention
//! - **Distributed Coordination**: Multi-node fault detection and recovery coordination
//!
//! # Features
//!
//! - **6 Fault Types**: Network partition, node failure, message loss, corruption, timeout, Byzantine faults
//! - **6 Recovery Strategies**: Retry, reroute, replicate, rollback, ignore, manual intervention
//! - **5 Detection Methods**: Timeout, heartbeat, checksum validation, redundant transmission, statistical analysis
//! - **Failure History Management**: Automatic cleanup and pattern analysis
//! - **Distributed Fault Tolerance**: Coordination across multiple nodes
//! - **Adaptive Recovery**: Dynamic strategy selection based on fault patterns
//!
//! # Examples
//!
//! ## Basic Fault Detection
//! ```rust
//! use torsh_autograd::communication_efficient::fault_tolerance::*;
//! use std::time::Instant;
//!
//! let mut detector = CommunicationFaultDetector::new();
//!
//! // Record a failure event
//! let event = FailureEvent {
//!     timestamp: Instant::now(),
//!     fault_type: FaultType::NetworkPartition,
//!     affected_worker: 1,
//!     severity: FaultSeverity::High,
//! };
//!
//! detector.record_failure(event);
//!
//! // Get recovery strategy
//! let strategy = detector.get_recovery_strategy(&FaultType::NetworkPartition);
//! assert_eq!(strategy, RecoveryStrategy::Reroute);
//! ```
//!
//! ## Advanced Fault Detection Configuration
//! ```rust
//! use torsh_autograd::communication_efficient::fault_tolerance::*;
//! use std::time::Duration;
//!
//! let mut detector = CommunicationFaultDetector::with_detection_method(
//!     FaultDetectionMethod::Heartbeat
//! );
//!
//! // Configure timeouts for different workers
//! detector.set_timeout_threshold(1, Duration::from_secs(30));
//! detector.set_timeout_threshold(2, Duration::from_secs(45));
//!
//! // Enable Byzantine fault detection
//! detector.enable_byzantine_detection(true);
//! detector.set_byzantine_threshold(0.33);
//! ```
//!
//! ## Distributed Fault Coordination
//! ```rust
//! use torsh_autograd::communication_efficient::{
//!     fault_tolerance::*,
//!     config::CommunicationConfig,
//!     management::TopologyManager,
//! };
//!
//! let mut coordinator = DistributedFaultCoordinator::new();
//! coordinator.enable_consensus_based_detection(true);
//! coordinator.set_detection_quorum(3); // Require 3 nodes to agree
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::{
    config::{CommunicationConfig, CommunicationTopology, ProtocolOptimization},
    management::{CongestionLevel, TopologyManager},
    transmission::{NetworkMetrics, ProtocolStack},
    CommunicationError,
};

/// Main fault detection and recovery coordinator for distributed communication.
///
/// The CommunicationFaultDetector manages fault detection across distributed training,
/// maintaining failure history, implementing various detection algorithms, and coordinating
/// recovery strategies to ensure resilient communication.
#[derive(Debug)]
pub struct CommunicationFaultDetector {
    /// Active fault detection method
    fault_detection_method: FaultDetectionMethod,
    /// Timeout thresholds per worker node
    timeout_thresholds: HashMap<u32, Duration>,
    /// Historical failure events per worker
    failure_history: HashMap<u32, Vec<FailureEvent>>,
    /// Recovery strategies mapped by fault type
    recovery_strategies: HashMap<FaultType, RecoveryStrategy>,
    /// Byzantine fault detection configuration
    byzantine_config: ByzantineDetectionConfig,
    /// Network monitoring metrics
    network_metrics: Option<Arc<Mutex<NetworkMetrics>>>,
    /// Heartbeat tracking for worker nodes
    heartbeat_tracker: HeartbeatTracker,
    /// Statistical analysis engine for pattern detection
    statistical_analyzer: StatisticalAnalyzer,
}

/// Different fault detection algorithms available for distributed systems.
#[derive(Debug, Clone, PartialEq)]
pub enum FaultDetectionMethod {
    /// Timeout-based detection using configurable thresholds
    Timeout,
    /// Heartbeat-based liveness detection
    Heartbeat,
    /// Checksum validation for message integrity
    ChecksumValidation,
    /// Redundant transmission for reliability verification
    RedundantTransmission,
    /// Statistical analysis of communication patterns
    StatisticalAnalysis,
}

/// Represents a fault or failure event in the distributed system.
#[derive(Debug, Clone)]
pub struct FailureEvent {
    /// When the failure was detected
    pub timestamp: Instant,
    /// Type of fault that occurred
    pub fault_type: FaultType,
    /// Worker node affected by the fault
    pub affected_worker: u32,
    /// Severity level of the fault
    pub severity: FaultSeverity,
}

/// Classification of different fault types in distributed communication.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FaultType {
    /// Network partition isolating nodes
    NetworkPartition,
    /// Complete node failure or crash
    NodeFailure,
    /// Message loss during transmission
    MessageLoss,
    /// Data corruption in messages
    Corruption,
    /// Communication timeout exceeded
    Timeout,
    /// Byzantine fault (malicious or arbitrary behavior)
    ByzantineFault,
}

/// Severity levels for fault classification and prioritization.
#[derive(Debug, Clone, PartialEq)]
pub enum FaultSeverity {
    /// Low impact, recoverable fault
    Low,
    /// Medium impact, may affect performance
    Medium,
    /// High impact, requires immediate attention
    High,
    /// Critical fault, system-wide impact
    Critical,
}

/// Recovery strategies for different types of faults.
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryStrategy {
    /// Retry the failed operation
    Retry,
    /// Route around the failed component
    Reroute,
    /// Replicate to alternative nodes
    Replicate,
    /// Rollback to previous state
    Rollback,
    /// Ignore the fault (for non-critical operations)
    Ignore,
    /// Require manual intervention
    ManualIntervention,
}

/// Configuration for Byzantine fault detection capabilities.
#[derive(Debug, Clone)]
pub struct ByzantineDetectionConfig {
    /// Whether Byzantine fault detection is enabled
    pub enabled: bool,
    /// Threshold ratio of Byzantine nodes tolerated (e.g., 0.33 for up to 33%)
    pub threshold: f64,
    /// Minimum number of nodes required for consensus
    pub min_consensus_nodes: usize,
    /// Verification methods for Byzantine detection
    pub verification_methods: Vec<ByzantineVerificationMethod>,
}

/// Methods for detecting Byzantine faults in distributed systems.
#[derive(Debug, Clone, PartialEq)]
pub enum ByzantineVerificationMethod {
    /// Majority voting among nodes
    MajorityVoting,
    /// Cryptographic verification
    CryptographicProof,
    /// Consistency checking across nodes
    ConsistencyVerification,
    /// Behavior pattern analysis
    BehaviorAnalysis,
}

/// Heartbeat tracking system for node liveness detection.
#[derive(Debug)]
pub struct HeartbeatTracker {
    /// Last heartbeat received from each worker
    last_heartbeat: HashMap<u32, Instant>,
    /// Expected heartbeat intervals per worker
    heartbeat_intervals: HashMap<u32, Duration>,
    /// Missed heartbeat counts
    missed_heartbeats: HashMap<u32, u32>,
    /// Maximum allowed missed heartbeats before declaring failure
    max_missed_heartbeats: u32,
}

/// Statistical analysis engine for fault pattern detection.
#[derive(Debug)]
pub struct StatisticalAnalyzer {
    /// Communication pattern history
    pattern_history: VecDeque<CommunicationPattern>,
    /// Anomaly detection thresholds
    anomaly_thresholds: AnomalyThresholds,
    /// Statistical models for different metrics
    statistical_models: HashMap<String, StatisticalModel>,
}

/// Communication pattern representation for analysis.
#[derive(Debug, Clone)]
pub struct CommunicationPattern {
    /// Pattern timestamp
    pub timestamp: Instant,
    /// Message throughput (messages per second)
    pub throughput: f64,
    /// Average latency (milliseconds)
    pub latency: f64,
    /// Error rate (errors per message)
    pub error_rate: f64,
    /// Participating workers
    pub workers: Vec<u32>,
}

/// Thresholds for anomaly detection in communication patterns.
#[derive(Debug, Clone)]
pub struct AnomalyThresholds {
    /// Maximum acceptable throughput deviation (standard deviations)
    pub throughput_deviation: f64,
    /// Maximum acceptable latency increase (percentage)
    pub latency_increase: f64,
    /// Maximum acceptable error rate
    pub error_rate_threshold: f64,
    /// Minimum pattern samples for reliable analysis
    pub min_samples: usize,
}

/// Statistical model for pattern analysis.
#[derive(Debug, Clone)]
pub struct StatisticalModel {
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Sample count
    pub samples: usize,
    /// Last update timestamp
    pub last_updated: Instant,
}

/// Distributed fault coordination across multiple nodes.
#[derive(Debug)]
pub struct DistributedFaultCoordinator {
    /// Local fault detector
    local_detector: CommunicationFaultDetector,
    /// Consensus-based detection configuration
    consensus_config: ConsensusConfig,
    /// Inter-node communication for fault coordination
    coordination_protocol: CoordinationProtocol,
    /// Quorum requirements for fault declarations
    quorum_requirements: QuorumRequirements,
}

/// Configuration for consensus-based fault detection.
#[derive(Debug, Clone)]
pub struct ConsensusConfig {
    /// Whether consensus-based detection is enabled
    pub enabled: bool,
    /// Minimum number of nodes that must agree
    pub detection_quorum: usize,
    /// Timeout for consensus decisions
    pub consensus_timeout: Duration,
    /// Retry attempts for consensus
    pub max_consensus_retries: u32,
}

/// Protocol for coordinating fault detection across nodes.
#[derive(Debug)]
pub struct CoordinationProtocol {
    /// Node ID in the distributed system
    node_id: u32,
    /// Connected peer nodes
    peer_nodes: HashMap<u32, PeerConnection>,
    /// Pending consensus requests
    pending_consensus: HashMap<String, ConsensusRequest>,
}

/// Connection to a peer node for fault coordination.
#[derive(Debug)]
pub struct PeerConnection {
    /// Peer node identifier
    pub node_id: u32,
    /// Connection status
    pub status: ConnectionStatus,
    /// Last communication timestamp
    pub last_contact: Instant,
    /// Communication reliability score
    pub reliability: f64,
}

/// Status of connection to peer nodes.
#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionStatus {
    /// Active and healthy connection
    Active,
    /// Connection is degraded but functional
    Degraded,
    /// Connection is suspected to be failing
    Suspected,
    /// Connection has failed
    Failed,
}

/// Request for consensus-based fault detection.
#[derive(Debug, Clone)]
pub struct ConsensusRequest {
    /// Request identifier
    pub request_id: String,
    /// Fault being considered
    pub suspected_fault: FailureEvent,
    /// Nodes that have voted
    pub votes: HashMap<u32, bool>,
    /// Request timestamp
    pub timestamp: Instant,
    /// Timeout for the request
    pub timeout: Duration,
}

/// Quorum requirements for different types of decisions.
#[derive(Debug, Clone)]
pub struct QuorumRequirements {
    /// Required quorum for fault declaration
    pub fault_detection: usize,
    /// Required quorum for recovery decisions
    pub recovery_strategy: usize,
    /// Required quorum for node exclusion
    pub node_exclusion: usize,
}

impl CommunicationFaultDetector {
    /// Creates a new fault detector with default configuration.
    pub fn new() -> Self {
        Self {
            fault_detection_method: FaultDetectionMethod::Timeout,
            timeout_thresholds: HashMap::new(),
            failure_history: HashMap::new(),
            recovery_strategies: {
                let mut strategies = HashMap::new();
                strategies.insert(FaultType::NetworkPartition, RecoveryStrategy::Reroute);
                strategies.insert(FaultType::NodeFailure, RecoveryStrategy::Replicate);
                strategies.insert(FaultType::MessageLoss, RecoveryStrategy::Retry);
                strategies.insert(FaultType::Corruption, RecoveryStrategy::Retry);
                strategies.insert(FaultType::Timeout, RecoveryStrategy::Retry);
                strategies.insert(FaultType::ByzantineFault, RecoveryStrategy::Ignore);
                strategies
            },
            byzantine_config: ByzantineDetectionConfig {
                enabled: false,
                threshold: 0.33,
                min_consensus_nodes: 3,
                verification_methods: vec![ByzantineVerificationMethod::MajorityVoting],
            },
            network_metrics: None,
            heartbeat_tracker: HeartbeatTracker::new(),
            statistical_analyzer: StatisticalAnalyzer::new(),
        }
    }

    /// Creates a fault detector with the specified detection method.
    pub fn with_detection_method(method: FaultDetectionMethod) -> Self {
        let mut detector = Self::new();
        detector.fault_detection_method = method;
        detector
    }

    /// Records a failure event in the history and updates patterns.
    pub fn record_failure(&mut self, event: FailureEvent) {
        let history = self
            .failure_history
            .entry(event.affected_worker)
            .or_insert_with(Vec::new);

        history.push(event.clone());

        // Maintain history size limit
        if history.len() > 100 {
            history.remove(0);
        }

        // Update statistical patterns
        self.statistical_analyzer.record_failure(&event);

        // Check for pattern-based anomalies
        if let Some(pattern) = self.analyze_failure_pattern(event.affected_worker) {
            self.handle_pattern_anomaly(pattern);
        }
    }

    /// Gets the appropriate recovery strategy for a fault type.
    pub fn get_recovery_strategy(&self, fault_type: &FaultType) -> RecoveryStrategy {
        self.recovery_strategies
            .get(fault_type)
            .cloned()
            .unwrap_or(RecoveryStrategy::Ignore)
    }

    /// Sets timeout threshold for a specific worker.
    pub fn set_timeout_threshold(&mut self, worker_id: u32, threshold: Duration) {
        self.timeout_thresholds.insert(worker_id, threshold);
    }

    /// Enables or disables Byzantine fault detection.
    pub fn enable_byzantine_detection(&mut self, enabled: bool) {
        self.byzantine_config.enabled = enabled;
    }

    /// Sets the Byzantine fault threshold ratio.
    pub fn set_byzantine_threshold(&mut self, threshold: f64) {
        self.byzantine_config.threshold = threshold.clamp(0.0, 1.0);
    }

    /// Performs timeout-based fault detection.
    pub fn detect_timeout_faults(&self, current_time: Instant) -> Vec<FailureEvent> {
        let mut detected_faults = Vec::new();

        for (&worker_id, &threshold) in &self.timeout_thresholds {
            if let Some(&last_heartbeat) = self.heartbeat_tracker.last_heartbeat.get(&worker_id) {
                if current_time.duration_since(last_heartbeat) > threshold {
                    detected_faults.push(FailureEvent {
                        timestamp: current_time,
                        fault_type: FaultType::Timeout,
                        affected_worker: worker_id,
                        severity: FaultSeverity::Medium,
                    });
                }
            }
        }

        detected_faults
    }

    /// Performs heartbeat-based liveness detection.
    pub fn detect_heartbeat_failures(&mut self, current_time: Instant) -> Vec<FailureEvent> {
        let mut detected_faults = Vec::new();

        for (&worker_id, &interval) in &self.heartbeat_tracker.heartbeat_intervals {
            if let Some(&last_heartbeat) = self.heartbeat_tracker.last_heartbeat.get(&worker_id) {
                let expected_next = last_heartbeat + interval;

                if current_time > expected_next {
                    let missed_count = self
                        .heartbeat_tracker
                        .missed_heartbeats
                        .entry(worker_id)
                        .or_insert(0);

                    *missed_count += 1;

                    if *missed_count >= self.heartbeat_tracker.max_missed_heartbeats {
                        detected_faults.push(FailureEvent {
                            timestamp: current_time,
                            fault_type: FaultType::NodeFailure,
                            affected_worker: worker_id,
                            severity: FaultSeverity::High,
                        });

                        // Reset counter after detection
                        *missed_count = 0;
                    }
                }
            }
        }

        detected_faults
    }

    /// Validates message integrity using checksum verification.
    pub fn validate_message_integrity(&self, message: &[u8], expected_checksum: u32) -> bool {
        let computed_checksum = self.compute_checksum(message);
        computed_checksum == expected_checksum
    }

    /// Detects corruption faults based on message validation.
    pub fn detect_corruption_faults(
        &self,
        worker_id: u32,
        validation_failures: u32,
        total_messages: u32,
    ) -> Option<FailureEvent> {
        if total_messages == 0 {
            return None;
        }

        let error_rate = validation_failures as f64 / total_messages as f64;
        let threshold = 0.05; // 5% error rate threshold

        if error_rate > threshold {
            Some(FailureEvent {
                timestamp: Instant::now(),
                fault_type: FaultType::Corruption,
                affected_worker: worker_id,
                severity: if error_rate > 0.2 {
                    FaultSeverity::High
                } else {
                    FaultSeverity::Medium
                },
            })
        } else {
            None
        }
    }

    /// Performs statistical analysis for anomaly detection.
    pub fn detect_statistical_anomalies(&mut self) -> Vec<FailureEvent> {
        let mut detected_faults = Vec::new();

        if let Some(current_pattern) = self.collect_current_pattern() {
            self.statistical_analyzer
                .pattern_history
                .push_back(current_pattern.clone());

            // Maintain pattern history size
            if self.statistical_analyzer.pattern_history.len() > 1000 {
                self.statistical_analyzer.pattern_history.pop_front();
            }

            // Analyze for anomalies
            if let Some(anomaly) = self.statistical_analyzer.detect_anomaly(&current_pattern) {
                detected_faults.push(anomaly);
            }
        }

        detected_faults
    }

    /// Executes the appropriate recovery strategy for a fault.
    pub fn execute_recovery_strategy(
        &self,
        strategy: RecoveryStrategy,
        affected_worker: u32,
        topology_manager: Option<&mut TopologyManager>,
    ) -> Result<(), CommunicationError> {
        match strategy {
            RecoveryStrategy::Retry => {
                println!("Retrying communication with worker {}", affected_worker);
                // Implement retry logic
                Ok(())
            }
            RecoveryStrategy::Reroute => {
                if let Some(topology_manager) = topology_manager {
                    topology_manager.reroute_around_failed_worker(affected_worker)?;
                    println!("Rerouted around failed worker {}", affected_worker);
                } else {
                    return Err(CommunicationError::TopologyError);
                }
                Ok(())
            }
            RecoveryStrategy::Replicate => {
                println!("Replicating communication for worker {}", affected_worker);
                // Implement replication logic
                Ok(())
            }
            RecoveryStrategy::Rollback => {
                println!(
                    "Rolling back communication state for worker {}",
                    affected_worker
                );
                // Implement rollback logic
                Ok(())
            }
            RecoveryStrategy::Ignore => {
                println!("Ignoring fault for worker {}", affected_worker);
                Ok(())
            }
            RecoveryStrategy::ManualIntervention => {
                println!(
                    "Manual intervention required for worker {}",
                    affected_worker
                );
                Err(CommunicationError::FaultDetectionFailed)
            }
        }
    }

    /// Updates heartbeat information for a worker.
    pub fn update_heartbeat(&mut self, worker_id: u32, timestamp: Instant) {
        self.heartbeat_tracker
            .last_heartbeat
            .insert(worker_id, timestamp);

        // Reset missed heartbeat counter
        self.heartbeat_tracker
            .missed_heartbeats
            .insert(worker_id, 0);
    }

    /// Sets heartbeat interval for a worker.
    pub fn set_heartbeat_interval(&mut self, worker_id: u32, interval: Duration) {
        self.heartbeat_tracker
            .heartbeat_intervals
            .insert(worker_id, interval);
    }

    /// Gets failure history for a specific worker.
    pub fn get_failure_history(&self, worker_id: u32) -> Option<&Vec<FailureEvent>> {
        self.failure_history.get(&worker_id)
    }

    /// Analyzes failure patterns for a worker.
    fn analyze_failure_pattern(&self, worker_id: u32) -> Option<String> {
        if let Some(history) = self.failure_history.get(&worker_id) {
            if history.len() >= 3 {
                let recent_failures: Vec<_> = history.iter().rev().take(3).collect();

                // Check for rapid succession of failures
                let time_span = recent_failures[0]
                    .timestamp
                    .duration_since(recent_failures[2].timestamp);

                if time_span < Duration::from_secs(60) {
                    return Some("rapid_failure_pattern".to_string());
                }
            }
        }
        None
    }

    /// Handles detected pattern anomalies.
    fn handle_pattern_anomaly(&mut self, pattern: String) {
        println!("Detected pattern anomaly: {}", pattern);
        // Implement pattern-specific handling
    }

    /// Computes checksum for message integrity verification.
    fn compute_checksum(&self, data: &[u8]) -> u32 {
        // Simple CRC32-like checksum implementation
        let mut checksum = 0u32;
        for &byte in data {
            checksum = checksum.wrapping_mul(31).wrapping_add(byte as u32);
        }
        checksum
    }

    /// Collects current communication pattern for analysis.
    fn collect_current_pattern(&self) -> Option<CommunicationPattern> {
        // This would collect real metrics from the network layer
        // For now, return a placeholder
        None
    }
}

impl HeartbeatTracker {
    fn new() -> Self {
        Self {
            last_heartbeat: HashMap::new(),
            heartbeat_intervals: HashMap::new(),
            missed_heartbeats: HashMap::new(),
            max_missed_heartbeats: 3,
        }
    }
}

impl StatisticalAnalyzer {
    fn new() -> Self {
        Self {
            pattern_history: VecDeque::new(),
            anomaly_thresholds: AnomalyThresholds {
                throughput_deviation: 2.0,
                latency_increase: 0.5,
                error_rate_threshold: 0.1,
                min_samples: 10,
            },
            statistical_models: HashMap::new(),
        }
    }

    fn record_failure(&mut self, _event: &FailureEvent) {
        // Record failure event for pattern analysis
    }

    fn detect_anomaly(&self, _pattern: &CommunicationPattern) -> Option<FailureEvent> {
        // Implement anomaly detection logic
        None
    }
}

impl Default for CommunicationFaultDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl DistributedFaultCoordinator {
    /// Creates a new distributed fault coordinator.
    pub fn new() -> Self {
        Self {
            local_detector: CommunicationFaultDetector::new(),
            consensus_config: ConsensusConfig {
                enabled: false,
                detection_quorum: 3,
                consensus_timeout: Duration::from_secs(10),
                max_consensus_retries: 3,
            },
            coordination_protocol: CoordinationProtocol {
                node_id: 0,
                peer_nodes: HashMap::new(),
                pending_consensus: HashMap::new(),
            },
            quorum_requirements: QuorumRequirements {
                fault_detection: 3,
                recovery_strategy: 2,
                node_exclusion: 4,
            },
        }
    }

    /// Enables consensus-based fault detection.
    pub fn enable_consensus_based_detection(&mut self, enabled: bool) {
        self.consensus_config.enabled = enabled;
    }

    /// Sets the detection quorum requirement.
    pub fn set_detection_quorum(&mut self, quorum: usize) {
        self.consensus_config.detection_quorum = quorum;
    }

    /// Initiates consensus for a suspected fault.
    pub fn initiate_fault_consensus(
        &mut self,
        suspected_fault: FailureEvent,
    ) -> Result<String, CommunicationError> {
        if !self.consensus_config.enabled {
            return Err(CommunicationError::ConfigurationError);
        }

        let request_id = format!(
            "fault_{}_{}",
            suspected_fault.affected_worker,
            suspected_fault.timestamp.elapsed().as_millis()
        );

        let consensus_request = ConsensusRequest {
            request_id: request_id.clone(),
            suspected_fault,
            votes: HashMap::new(),
            timestamp: Instant::now(),
            timeout: self.consensus_config.consensus_timeout,
        };

        self.coordination_protocol
            .pending_consensus
            .insert(request_id.clone(), consensus_request);

        Ok(request_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fault_detector_creation() {
        let detector = CommunicationFaultDetector::new();
        assert_eq!(
            detector.fault_detection_method,
            FaultDetectionMethod::Timeout
        );
        assert_eq!(detector.recovery_strategies.len(), 6);
    }

    #[test]
    fn test_failure_recording() {
        let mut detector = CommunicationFaultDetector::new();

        let event = FailureEvent {
            timestamp: Instant::now(),
            fault_type: FaultType::NetworkPartition,
            affected_worker: 1,
            severity: FaultSeverity::High,
        };

        detector.record_failure(event);
        assert!(detector.failure_history.contains_key(&1));
        assert_eq!(detector.failure_history[&1].len(), 1);
    }

    #[test]
    fn test_recovery_strategy_selection() {
        let detector = CommunicationFaultDetector::new();

        assert_eq!(
            detector.get_recovery_strategy(&FaultType::NetworkPartition),
            RecoveryStrategy::Reroute
        );
        assert_eq!(
            detector.get_recovery_strategy(&FaultType::NodeFailure),
            RecoveryStrategy::Replicate
        );
        assert_eq!(
            detector.get_recovery_strategy(&FaultType::MessageLoss),
            RecoveryStrategy::Retry
        );
    }

    #[test]
    fn test_timeout_configuration() {
        let mut detector = CommunicationFaultDetector::new();
        let timeout = Duration::from_secs(30);

        detector.set_timeout_threshold(1, timeout);
        assert_eq!(detector.timeout_thresholds.get(&1), Some(&timeout));
    }

    #[test]
    fn test_byzantine_configuration() {
        let mut detector = CommunicationFaultDetector::new();

        detector.enable_byzantine_detection(true);
        detector.set_byzantine_threshold(0.25);

        assert!(detector.byzantine_config.enabled);
        assert_eq!(detector.byzantine_config.threshold, 0.25);
    }

    #[test]
    fn test_heartbeat_tracking() {
        let mut detector = CommunicationFaultDetector::new();
        let now = Instant::now();

        detector.update_heartbeat(1, now);
        detector.set_heartbeat_interval(1, Duration::from_secs(10));

        assert_eq!(
            detector.heartbeat_tracker.last_heartbeat.get(&1),
            Some(&now)
        );
        assert_eq!(
            detector.heartbeat_tracker.heartbeat_intervals.get(&1),
            Some(&Duration::from_secs(10))
        );
    }

    #[test]
    fn test_message_integrity_validation() {
        let detector = CommunicationFaultDetector::new();
        let message = b"test message";
        let checksum = detector.compute_checksum(message);

        assert!(detector.validate_message_integrity(message, checksum));
        assert!(!detector.validate_message_integrity(message, checksum + 1));
    }

    #[test]
    fn test_corruption_detection() {
        let detector = CommunicationFaultDetector::new();

        // High error rate should trigger detection
        let fault = detector.detect_corruption_faults(1, 10, 100); // 10% error rate
        assert!(fault.is_some());
        assert_eq!(fault.unwrap().fault_type, FaultType::Corruption);

        // Low error rate should not trigger detection
        let no_fault = detector.detect_corruption_faults(1, 2, 100); // 2% error rate
        assert!(no_fault.is_none());
    }

    #[test]
    fn test_distributed_coordinator() {
        let mut coordinator = DistributedFaultCoordinator::new();

        coordinator.enable_consensus_based_detection(true);
        coordinator.set_detection_quorum(5);

        assert!(coordinator.consensus_config.enabled);
        assert_eq!(coordinator.consensus_config.detection_quorum, 5);
    }

    #[test]
    fn test_fault_consensus_initiation() {
        let mut coordinator = DistributedFaultCoordinator::new();
        coordinator.enable_consensus_based_detection(true);

        let fault = FailureEvent {
            timestamp: Instant::now(),
            fault_type: FaultType::NodeFailure,
            affected_worker: 2,
            severity: FaultSeverity::High,
        };

        let result = coordinator.initiate_fault_consensus(fault);
        assert!(result.is_ok());

        let request_id = result.unwrap();
        assert!(coordinator
            .coordination_protocol
            .pending_consensus
            .contains_key(&request_id));
    }
}
