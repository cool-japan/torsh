use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use crate::compression::GradientCompressor;

/// Lock ordering and timeout utilities to prevent deadlocks
mod lock_utilities {
    use super::*;

    /// Maximum time to wait for a lock before timing out
    #[allow(dead_code)]
    pub const LOCK_TIMEOUT: Duration = Duration::from_secs(5);

    /// Safely acquire a lock with timeout
    #[allow(dead_code)]
    pub fn acquire_with_timeout<T>(
        mutex: &Arc<Mutex<T>>,
    ) -> Result<std::sync::MutexGuard<T>, &'static str> {
        // For now, we use try_lock as std::sync::Mutex doesn't have timeout
        // In production code, consider using parking_lot::Mutex which has timeout
        mutex.try_lock().map_err(|_| "Lock acquisition timeout")
    }

    /// Safely acquire multiple locks with consistent ordering
    /// Prevents deadlocks by always acquiring locks in the same order
    #[allow(dead_code)]
    pub fn acquire_ordered_locks<'a, T1, T2>(
        lock1: &'a Arc<Mutex<T1>>,
        lock2: &'a Arc<Mutex<T2>>,
    ) -> Result<(std::sync::MutexGuard<'a, T1>, std::sync::MutexGuard<'a, T2>), &'static str> {
        // Acquire locks in consistent order based on pointer addresses
        let addr1 = Arc::as_ptr(lock1) as usize;
        let addr2 = Arc::as_ptr(lock2) as usize;

        if addr1 < addr2 {
            let guard1 = acquire_with_timeout(lock1)?;
            let guard2 = acquire_with_timeout(lock2)?;
            Ok((guard1, guard2))
        } else {
            let guard2 = acquire_with_timeout(lock2)?;
            let guard1 = acquire_with_timeout(lock1)?;
            Ok((guard1, guard2))
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParameterServerConfig {
    pub server_addresses: Vec<SocketAddr>,
    pub worker_id: u32,
    pub num_workers: u32,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub gradient_compression: Option<GradientCompressor<f32>>,
    pub staleness_threshold: u32,
    pub heartbeat_interval: Duration,
    pub timeout_duration: Duration,
    pub backup_frequency: Duration,
    pub fault_tolerance: bool,
}

impl Default for ParameterServerConfig {
    fn default() -> Self {
        Self {
            server_addresses: vec![],
            worker_id: 0,
            num_workers: 1,
            batch_size: 32,
            learning_rate: 0.01,
            momentum: 0.9,
            weight_decay: 0.0001,
            gradient_compression: None,
            staleness_threshold: 10,
            heartbeat_interval: Duration::from_secs(5),
            timeout_duration: Duration::from_secs(30),
            backup_frequency: Duration::from_secs(60),
            fault_tolerance: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParameterUpdate {
    pub parameter_id: String,
    pub gradient: Vec<f32>,
    pub worker_id: u32,
    pub version: u64,
    pub timestamp: Instant,
    pub priority: UpdatePriority,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum UpdatePriority {
    Low,
    Normal,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub struct ParameterState {
    pub parameter_id: String,
    pub values: Vec<f32>,
    pub version: u64,
    pub last_updated: Instant,
    pub gradient_accumulator: Vec<f32>,
    pub momentum_buffer: Vec<f32>,
    pub update_count: u64,
    pub staleness_count: u32,
}

#[derive(Debug)]
pub struct ParameterServer {
    config: ParameterServerConfig,
    parameters: Arc<RwLock<HashMap<String, ParameterState>>>,
    pending_updates: Arc<Mutex<VecDeque<ParameterUpdate>>>,
    worker_connections: Arc<Mutex<HashMap<u32, TcpStream>>>,
    server_running: Arc<Mutex<bool>>,
    update_scheduler: Arc<Mutex<UpdateScheduler>>,
    fault_detector: Arc<Mutex<FaultDetector>>,
    backup_manager: Arc<Mutex<BackupManager>>,
    metrics: Arc<Mutex<ParameterServerMetrics>>,
}

// ParameterServer is Send + Sync
unsafe impl Send for ParameterServer {}
unsafe impl Sync for ParameterServer {}

#[allow(dead_code)]
#[derive(Debug)]
pub struct UpdateScheduler {
    update_queue: VecDeque<ParameterUpdate>,
    processing_updates: HashMap<String, ParameterUpdate>,
    batch_updates: HashMap<String, Vec<ParameterUpdate>>,
    scheduler_strategy: SchedulerStrategy,
}

// UpdateScheduler is Send + Sync
unsafe impl Send for UpdateScheduler {}
unsafe impl Sync for UpdateScheduler {}

#[derive(Debug, Clone, PartialEq)]
pub enum SchedulerStrategy {
    FIFO,
    Priority,
    BatchedSynchronous,
    AsynchronousStale,
    AdaptiveSync,
}

#[derive(Debug)]
pub struct FaultDetector {
    worker_heartbeats: HashMap<u32, Instant>,
    failed_workers: HashSet<u32>,
    recovery_actions: Vec<RecoveryAction>,
    detection_threshold: Duration,
}

// FaultDetector is Send + Sync
unsafe impl Send for FaultDetector {}
unsafe impl Sync for FaultDetector {}

#[derive(Debug, Clone)]
pub enum RecoveryAction {
    RestartWorker(u32),
    RerouteTraffic(u32, u32),
    RestoreFromBackup(String),
    RebalanceLoad,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct BackupManager {
    backup_storage: HashMap<String, Vec<u8>>,
    backup_frequency: Duration,
    last_backup: Instant,
    backup_locations: Vec<String>,
}

// BackupManager is Send + Sync
unsafe impl Send for BackupManager {}
unsafe impl Sync for BackupManager {}

#[derive(Debug, Default, Clone)]
pub struct ParameterServerMetrics {
    pub total_updates: u64,
    pub successful_updates: u64,
    pub failed_updates: u64,
    pub average_update_time: Duration,
    pub total_parameters: usize,
    pub active_workers: u32,
    pub staleness_violations: u64,
    pub network_bytes_sent: u64,
    pub network_bytes_received: u64,
    pub backup_count: u64,
    pub fault_recoveries: u64,
}

impl ParameterServer {
    pub fn new(config: ParameterServerConfig) -> Self {
        Self {
            config: config.clone(),
            parameters: Arc::new(RwLock::new(HashMap::new())),
            pending_updates: Arc::new(Mutex::new(VecDeque::new())),
            worker_connections: Arc::new(Mutex::new(HashMap::new())),
            server_running: Arc::new(Mutex::new(false)),
            update_scheduler: Arc::new(Mutex::new(UpdateScheduler::new(
                SchedulerStrategy::AdaptiveSync,
            ))),
            fault_detector: Arc::new(Mutex::new(FaultDetector::new(config.timeout_duration))),
            backup_manager: Arc::new(Mutex::new(BackupManager::new(config.backup_frequency))),
            metrics: Arc::new(Mutex::new(ParameterServerMetrics::default())),
        }
    }

    pub fn start(&self) -> Result<(), ParameterServerError> {
        {
            let mut running = self.server_running.lock().unwrap();
            *running = true;
        }

        self.start_server_listener()?;
        self.start_update_processor()?;
        self.start_heartbeat_monitor()?;
        self.start_backup_manager()?;
        self.start_fault_detector()?;

        Ok(())
    }

    pub fn stop(&self) -> Result<(), ParameterServerError> {
        {
            let mut running = self.server_running.lock().unwrap();
            *running = false;
        }

        self.close_all_connections()?;
        self.save_final_backup()?;

        Ok(())
    }

    fn start_server_listener(&self) -> Result<(), ParameterServerError> {
        let config = self.config.clone();
        let parameters = self.parameters.clone();
        let pending_updates = self.pending_updates.clone();
        let worker_connections = self.worker_connections.clone();
        let server_running = self.server_running.clone();
        let metrics = self.metrics.clone();

        thread::spawn(move || {
            if let Some(addr) = config.server_addresses.first() {
                match TcpListener::bind(addr) {
                    Ok(listener) => {
                        println!("Parameter server listening on {}", addr);

                        for stream in listener.incoming() {
                            if !*server_running.lock().unwrap() {
                                break;
                            }

                            match stream {
                                Ok(stream) => {
                                    let config = config.clone();
                                    let parameters = parameters.clone();
                                    let pending_updates = pending_updates.clone();
                                    let worker_connections = worker_connections.clone();
                                    let metrics = metrics.clone();

                                    thread::spawn(move || {
                                        if let Err(e) = Self::handle_worker_connection(
                                            stream,
                                            config,
                                            parameters,
                                            pending_updates,
                                            worker_connections,
                                            metrics,
                                        ) {
                                            eprintln!("Error handling worker connection: {}", e);
                                        }
                                    });
                                }
                                Err(e) => {
                                    eprintln!("Error accepting connection: {}", e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Error binding to address: {}", e);
                    }
                }
            }
        });

        Ok(())
    }

    fn handle_worker_connection(
        mut stream: TcpStream,
        _config: ParameterServerConfig,
        parameters: Arc<RwLock<HashMap<String, ParameterState>>>,
        pending_updates: Arc<Mutex<VecDeque<ParameterUpdate>>>,
        _worker_connections: Arc<Mutex<HashMap<u32, TcpStream>>>,
        metrics: Arc<Mutex<ParameterServerMetrics>>,
    ) -> Result<(), ParameterServerError> {
        let mut buffer = [0; 1024];

        loop {
            match stream.read(&mut buffer) {
                Ok(0) => break,
                Ok(bytes_read) => {
                    metrics.lock().unwrap().network_bytes_received += bytes_read as u64;

                    let message = String::from_utf8_lossy(&buffer[..bytes_read]);
                    let response = Self::process_worker_message(
                        &message,
                        &_config,
                        &parameters,
                        &pending_updates,
                    )?;

                    stream.write_all(response.as_bytes())?;
                    stream.flush()?;

                    metrics.lock().unwrap().network_bytes_sent += response.len() as u64;
                }
                Err(e) => {
                    eprintln!("Error reading from worker: {}", e);
                    break;
                }
            }
        }

        Ok(())
    }

    fn process_worker_message(
        message: &str,
        _config: &ParameterServerConfig,
        parameters: &Arc<RwLock<HashMap<String, ParameterState>>>,
        pending_updates: &Arc<Mutex<VecDeque<ParameterUpdate>>>,
    ) -> Result<String, ParameterServerError> {
        let parts: Vec<&str> = message.trim().split('|').collect();

        match parts.get(0) {
            Some(&"GET_PARAMS") => {
                if let Some(param_id) = parts.get(1) {
                    let params = parameters.read().unwrap();
                    if let Some(param_state) = params.get(*param_id) {
                        let serialized = Self::serialize_parameter_state(param_state)?;
                        Ok(format!("PARAMS|{}|{}", param_id, serialized))
                    } else {
                        Ok("ERROR|Parameter not found".to_string())
                    }
                } else {
                    Ok("ERROR|Invalid GET_PARAMS format".to_string())
                }
            }
            Some(&"UPDATE_PARAMS") => {
                if parts.len() >= 4 {
                    let param_id = parts[1].to_string();
                    let worker_id: u32 = parts[2].parse().unwrap_or(0);
                    let gradient_data = parts[3];

                    let gradient = Self::deserialize_gradient(gradient_data)?;

                    let update = ParameterUpdate {
                        parameter_id: param_id,
                        gradient,
                        worker_id,
                        version: 0,
                        timestamp: Instant::now(),
                        priority: UpdatePriority::Normal,
                    };

                    pending_updates.lock().unwrap().push_back(update);
                    Ok("ACK|Update queued".to_string())
                } else {
                    Ok("ERROR|Invalid UPDATE_PARAMS format".to_string())
                }
            }
            Some(&"HEARTBEAT") => {
                if let Some(worker_id_str) = parts.get(1) {
                    let worker_id: u32 = worker_id_str.parse().unwrap_or(0);
                    Ok(format!("HEARTBEAT_ACK|{}", worker_id))
                } else {
                    Ok("ERROR|Invalid HEARTBEAT format".to_string())
                }
            }
            Some(&"REGISTER_WORKER") => {
                if let Some(worker_id_str) = parts.get(1) {
                    let worker_id: u32 = worker_id_str.parse().unwrap_or(0);
                    Ok(format!("REGISTERED|{}", worker_id))
                } else {
                    Ok("ERROR|Invalid REGISTER_WORKER format".to_string())
                }
            }
            _ => Ok("ERROR|Unknown command".to_string()),
        }
    }

    fn serialize_parameter_state(state: &ParameterState) -> Result<String, ParameterServerError> {
        let mut serialized = format!("{}|{}", state.version, state.values.len());
        for value in &state.values {
            serialized.push_str(&format!("|{}", value));
        }
        Ok(serialized)
    }

    fn deserialize_gradient(data: &str) -> Result<Vec<f32>, ParameterServerError> {
        let parts: Vec<&str> = data.split(',').collect();
        let mut gradient = Vec::new();

        for part in parts {
            match part.parse::<f32>() {
                Ok(value) => gradient.push(value),
                Err(_) => return Err(ParameterServerError::InvalidGradientFormat),
            }
        }

        Ok(gradient)
    }

    fn start_update_processor(&self) -> Result<(), ParameterServerError> {
        let parameters = self.parameters.clone();
        let pending_updates = self.pending_updates.clone();
        let _update_scheduler = self.update_scheduler.clone();
        let server_running = self.server_running.clone();
        let config = self.config.clone();
        let metrics = self.metrics.clone();

        thread::spawn(move || {
            while *server_running.lock().unwrap() {
                let update = {
                    let mut queue = pending_updates.lock().unwrap();
                    queue.pop_front()
                };

                if let Some(update) = update {
                    let start_time = Instant::now();

                    match Self::apply_parameter_update(&update, &parameters, &config) {
                        Ok(()) => {
                            let mut metrics_guard = metrics.lock().unwrap();
                            metrics_guard.successful_updates += 1;
                            metrics_guard.total_updates += 1;

                            let update_time = start_time.elapsed();
                            metrics_guard.average_update_time = Duration::from_nanos(
                                (metrics_guard.average_update_time.as_nanos() as f64 * 0.9
                                    + update_time.as_nanos() as f64 * 0.1)
                                    as u64,
                            );
                        }
                        Err(e) => {
                            eprintln!("Error applying update: {}", e);
                            let mut metrics_guard = metrics.lock().unwrap();
                            metrics_guard.failed_updates += 1;
                        }
                    }
                } else {
                    thread::sleep(Duration::from_millis(10));
                }
            }
        });

        Ok(())
    }

    fn apply_parameter_update(
        update: &ParameterUpdate,
        parameters: &Arc<RwLock<HashMap<String, ParameterState>>>,
        config: &ParameterServerConfig,
    ) -> Result<(), ParameterServerError> {
        // Validate inputs before acquiring lock
        if update.gradient.is_empty() {
            return Err(ParameterServerError::DimensionMismatch);
        }

        // Pre-compute values to minimize lock duration
        let momentum_factor = config.momentum as f32;
        let inv_momentum_factor = 1.0 - momentum_factor;
        let learning_rate = config.learning_rate as f32;
        let weight_decay = config.weight_decay as f32;
        let now = Instant::now();

        // Quick write operation with minimal computation in critical section
        let mut params = parameters.write().unwrap();

        let param_state = params
            .entry(update.parameter_id.clone())
            .or_insert_with(|| ParameterState {
                parameter_id: update.parameter_id.clone(),
                values: vec![0.0; update.gradient.len()],
                version: 0,
                last_updated: now,
                gradient_accumulator: vec![0.0; update.gradient.len()],
                momentum_buffer: vec![0.0; update.gradient.len()],
                update_count: 0,
                staleness_count: 0,
            });

        // Validate dimensions after getting or creating state
        if update.gradient.len() != param_state.values.len() {
            return Err(ParameterServerError::DimensionMismatch);
        }

        // Optimized vectorized updates to minimize time in critical section
        param_state
            .gradient_accumulator
            .iter_mut()
            .zip(update.gradient.iter())
            .for_each(|(acc, &grad)| *acc += grad);

        param_state
            .momentum_buffer
            .iter_mut()
            .zip(param_state.gradient_accumulator.iter())
            .for_each(|(momentum, &grad)| {
                *momentum = momentum_factor * *momentum + inv_momentum_factor * grad;
            });

        param_state
            .values
            .iter_mut()
            .zip(param_state.momentum_buffer.iter())
            .for_each(|(value, &momentum)| {
                *value -= learning_rate * momentum + weight_decay * *value;
            });

        // Reset and update metadata
        param_state.gradient_accumulator.fill(0.0);
        param_state.version += 1;
        param_state.last_updated = now;
        param_state.update_count += 1;

        Ok(())
    }

    fn start_heartbeat_monitor(&self) -> Result<(), ParameterServerError> {
        let fault_detector = self.fault_detector.clone();
        let server_running = self.server_running.clone();
        let heartbeat_interval = self.config.heartbeat_interval;

        thread::spawn(move || {
            while *server_running.lock().unwrap() {
                {
                    let mut detector = fault_detector.lock().unwrap();
                    detector.check_worker_health();
                }

                thread::sleep(heartbeat_interval);
            }
        });

        Ok(())
    }

    fn start_backup_manager(&self) -> Result<(), ParameterServerError> {
        let backup_manager = self.backup_manager.clone();
        let parameters = self.parameters.clone();
        let server_running = self.server_running.clone();
        let metrics = self.metrics.clone();

        thread::spawn(move || {
            while *server_running.lock().unwrap() {
                let should_backup = {
                    let manager = backup_manager.lock().unwrap();
                    manager.should_create_backup()
                };

                if should_backup {
                    // Create snapshot to avoid holding locks simultaneously
                    let params_snapshot = {
                        let params = parameters.read().unwrap();
                        params.clone()
                    };

                    // Now safely acquire backup manager lock
                    if let Ok(mut manager) = backup_manager.try_lock() {
                        if let Err(e) = manager.create_backup(&params_snapshot) {
                            eprintln!("Error creating backup: {}", e);
                        } else {
                            // Update metrics separately to avoid nested locks
                            if let Ok(mut metrics_guard) = metrics.try_lock() {
                                metrics_guard.backup_count += 1;
                            }
                        }
                    } else {
                        eprintln!("Warning: Backup manager busy, skipping backup");
                    }
                }

                thread::sleep(Duration::from_secs(10));
            }
        });

        Ok(())
    }

    fn start_fault_detector(&self) -> Result<(), ParameterServerError> {
        let fault_detector = self.fault_detector.clone();
        let server_running = self.server_running.clone();
        let metrics = self.metrics.clone();

        thread::spawn(move || {
            while *server_running.lock().unwrap() {
                let recovery_actions = {
                    let mut detector = fault_detector.lock().unwrap();
                    detector.detect_and_recover_faults()
                };

                for action in recovery_actions {
                    if let Err(e) = Self::execute_recovery_action(&action) {
                        eprintln!("Error executing recovery action: {}", e);
                    } else {
                        metrics.lock().unwrap().fault_recoveries += 1;
                    }
                }

                thread::sleep(Duration::from_secs(5));
            }
        });

        Ok(())
    }

    fn execute_recovery_action(action: &RecoveryAction) -> Result<(), ParameterServerError> {
        match action {
            RecoveryAction::RestartWorker(worker_id) => {
                println!("Restarting worker {}", worker_id);
                Ok(())
            }
            RecoveryAction::RerouteTraffic(from_worker, to_worker) => {
                println!(
                    "Rerouting traffic from worker {} to worker {}",
                    from_worker, to_worker
                );
                Ok(())
            }
            RecoveryAction::RestoreFromBackup(backup_id) => {
                println!("Restoring from backup {}", backup_id);
                Ok(())
            }
            RecoveryAction::RebalanceLoad => {
                println!("Rebalancing load across workers");
                Ok(())
            }
        }
    }

    fn close_all_connections(&self) -> Result<(), ParameterServerError> {
        let mut connections = self.worker_connections.lock().unwrap();

        for (worker_id, stream) in connections.drain() {
            if let Err(e) = stream.shutdown(std::net::Shutdown::Both) {
                eprintln!("Error closing connection for worker {}: {}", worker_id, e);
            }
        }

        Ok(())
    }

    fn save_final_backup(&self) -> Result<(), ParameterServerError> {
        // Create snapshot to avoid holding both locks simultaneously
        let params_snapshot = {
            let params = self.parameters.read().unwrap();
            params.clone()
        };

        // Now safely acquire backup manager lock
        let mut manager = self.backup_manager.lock().unwrap();
        manager.create_backup(&params_snapshot)?;
        manager.save_to_disk()?;

        Ok(())
    }

    pub fn get_parameters(
        &self,
        parameter_id: &str,
    ) -> Result<Option<ParameterState>, ParameterServerError> {
        let params = self.parameters.read().unwrap();
        Ok(params.get(parameter_id).cloned())
    }

    pub fn update_parameters(&self, update: ParameterUpdate) -> Result<(), ParameterServerError> {
        let mut queue = self.pending_updates.lock().unwrap();
        queue.push_back(update);
        Ok(())
    }

    pub fn get_metrics(&self) -> ParameterServerMetrics {
        (*self.metrics.lock().unwrap()).clone()
    }

    pub fn register_parameter(
        &self,
        parameter_id: String,
        initial_values: Vec<f32>,
    ) -> Result<(), ParameterServerError> {
        let mut params = self.parameters.write().unwrap();

        let param_state = ParameterState {
            parameter_id: parameter_id.clone(),
            values: initial_values.clone(),
            version: 0,
            last_updated: Instant::now(),
            gradient_accumulator: vec![0.0; initial_values.len()],
            momentum_buffer: vec![0.0; initial_values.len()],
            update_count: 0,
            staleness_count: 0,
        };

        params.insert(parameter_id, param_state);
        Ok(())
    }
}

impl UpdateScheduler {
    pub fn new(strategy: SchedulerStrategy) -> Self {
        Self {
            update_queue: VecDeque::new(),
            processing_updates: HashMap::new(),
            batch_updates: HashMap::new(),
            scheduler_strategy: strategy,
        }
    }

    pub fn schedule_update(&mut self, update: ParameterUpdate) -> Result<(), ParameterServerError> {
        match self.scheduler_strategy {
            SchedulerStrategy::FIFO => {
                self.update_queue.push_back(update);
            }
            SchedulerStrategy::Priority => {
                let mut inserted = false;
                let mut temp_queue = VecDeque::new();

                while let Some(queued_update) = self.update_queue.pop_front() {
                    if !inserted && update.priority > queued_update.priority {
                        temp_queue.push_back(update.clone());
                        inserted = true;
                    }
                    temp_queue.push_back(queued_update);
                }

                if !inserted {
                    temp_queue.push_back(update);
                }

                self.update_queue = temp_queue;
            }
            SchedulerStrategy::BatchedSynchronous => {
                let batch = self
                    .batch_updates
                    .entry(update.parameter_id.clone())
                    .or_insert_with(Vec::new);
                batch.push(update);
            }
            SchedulerStrategy::AsynchronousStale => {
                self.update_queue.push_back(update);
            }
            SchedulerStrategy::AdaptiveSync => {
                if self.should_batch_update(&update) {
                    let batch = self
                        .batch_updates
                        .entry(update.parameter_id.clone())
                        .or_insert_with(Vec::new);
                    batch.push(update);
                } else {
                    self.update_queue.push_back(update);
                }
            }
        }

        Ok(())
    }

    fn should_batch_update(&self, update: &ParameterUpdate) -> bool {
        let staleness = update.timestamp.elapsed();
        staleness.as_millis() < 100
    }

    pub fn get_next_update(&mut self) -> Option<ParameterUpdate> {
        match self.scheduler_strategy {
            SchedulerStrategy::BatchedSynchronous => self.get_next_batch_update(),
            _ => self.update_queue.pop_front(),
        }
    }

    fn get_next_batch_update(&mut self) -> Option<ParameterUpdate> {
        for (_param_id, batch) in &mut self.batch_updates {
            if batch.len() >= 4 {
                let combined_update = Self::combine_batch_updates_static(batch)?;
                batch.clear();
                return Some(combined_update);
            }
        }
        None
    }

    fn combine_batch_updates_static(batch: &[ParameterUpdate]) -> Option<ParameterUpdate> {
        if batch.is_empty() {
            return None;
        }

        let first_update = &batch[0];
        let mut combined_gradient = first_update.gradient.clone();

        for update in &batch[1..] {
            for (i, &grad) in update.gradient.iter().enumerate() {
                if i < combined_gradient.len() {
                    combined_gradient[i] += grad;
                }
            }
        }

        for grad in &mut combined_gradient {
            *grad /= batch.len() as f32;
        }

        Some(ParameterUpdate {
            parameter_id: first_update.parameter_id.clone(),
            gradient: combined_gradient,
            worker_id: first_update.worker_id,
            version: batch.iter().map(|u| u.version).max().unwrap_or(0),
            timestamp: Instant::now(),
            priority: batch
                .iter()
                .map(|u| u.priority.clone())
                .max()
                .unwrap_or(UpdatePriority::Normal),
        })
    }
}

impl FaultDetector {
    pub fn new(detection_threshold: Duration) -> Self {
        Self {
            worker_heartbeats: HashMap::new(),
            failed_workers: HashSet::new(),
            recovery_actions: Vec::new(),
            detection_threshold,
        }
    }

    pub fn update_worker_heartbeat(&mut self, worker_id: u32) {
        self.worker_heartbeats.insert(worker_id, Instant::now());
        self.failed_workers.remove(&worker_id);
    }

    pub fn check_worker_health(&mut self) {
        let now = Instant::now();
        let mut newly_failed = Vec::new();

        for (&worker_id, &last_heartbeat) in &self.worker_heartbeats {
            if now.duration_since(last_heartbeat) > self.detection_threshold {
                if !self.failed_workers.contains(&worker_id) {
                    newly_failed.push(worker_id);
                }
            }
        }

        for worker_id in newly_failed {
            self.failed_workers.insert(worker_id);
            self.recovery_actions
                .push(RecoveryAction::RestartWorker(worker_id));
        }
    }

    pub fn detect_and_recover_faults(&mut self) -> Vec<RecoveryAction> {
        let actions = self.recovery_actions.clone();
        self.recovery_actions.clear();
        actions
    }
}

impl BackupManager {
    pub fn new(backup_frequency: Duration) -> Self {
        Self {
            backup_storage: HashMap::new(),
            backup_frequency,
            last_backup: Instant::now(),
            backup_locations: Vec::new(),
        }
    }

    pub fn should_create_backup(&self) -> bool {
        self.last_backup.elapsed() >= self.backup_frequency
    }

    pub fn create_backup(
        &mut self,
        parameters: &HashMap<String, ParameterState>,
    ) -> Result<(), ParameterServerError> {
        let backup_id = format!("backup_{}", Instant::now().elapsed().as_secs());
        let mut backup_data = Vec::new();

        for (param_id, param_state) in parameters {
            let serialized = format!(
                "{}:{}",
                param_id,
                Self::serialize_parameter_state(param_state)?
            );
            backup_data.extend_from_slice(serialized.as_bytes());
            backup_data.push(b'\n');
        }

        self.backup_storage.insert(backup_id, backup_data);
        self.last_backup = Instant::now();

        Ok(())
    }

    fn serialize_parameter_state(state: &ParameterState) -> Result<String, ParameterServerError> {
        let mut serialized = format!("{}|{}", state.version, state.values.len());
        for value in &state.values {
            serialized.push_str(&format!("|{}", value));
        }
        Ok(serialized)
    }

    pub fn restore_from_backup(
        &mut self,
        backup_id: &str,
    ) -> Result<HashMap<String, ParameterState>, ParameterServerError> {
        if let Some(backup_data) = self.backup_storage.get(backup_id) {
            let backup_str = String::from_utf8(backup_data.clone())
                .map_err(|_| ParameterServerError::BackupCorruption)?;

            let mut parameters = HashMap::new();

            for line in backup_str.lines() {
                if let Some((param_id, param_data)) = line.split_once(':') {
                    let param_state = Self::deserialize_parameter_state(param_data)?;
                    parameters.insert(param_id.to_string(), param_state);
                }
            }

            Ok(parameters)
        } else {
            Err(ParameterServerError::BackupNotFound)
        }
    }

    fn deserialize_parameter_state(data: &str) -> Result<ParameterState, ParameterServerError> {
        let parts: Vec<&str> = data.split('|').collect();

        if parts.len() < 2 {
            return Err(ParameterServerError::InvalidBackupFormat);
        }

        let version = parts[0]
            .parse::<u64>()
            .map_err(|_| ParameterServerError::InvalidBackupFormat)?;

        let values_len = parts[1]
            .parse::<usize>()
            .map_err(|_| ParameterServerError::InvalidBackupFormat)?;

        if parts.len() != 2 + values_len {
            return Err(ParameterServerError::InvalidBackupFormat);
        }

        let mut values = Vec::new();
        for i in 2..2 + values_len {
            let value = parts[i]
                .parse::<f32>()
                .map_err(|_| ParameterServerError::InvalidBackupFormat)?;
            values.push(value);
        }

        Ok(ParameterState {
            parameter_id: String::new(),
            values,
            version,
            last_updated: Instant::now(),
            gradient_accumulator: vec![0.0; values_len],
            momentum_buffer: vec![0.0; values_len],
            update_count: 0,
            staleness_count: 0,
        })
    }

    pub fn save_to_disk(&self) -> Result<(), ParameterServerError> {
        println!("Saving {} backups to disk", self.backup_storage.len());
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum ParameterServerError {
    NetworkError(String),
    SerializationError(String),
    InvalidGradientFormat,
    DimensionMismatch,
    BackupCorruption,
    BackupNotFound,
    InvalidBackupFormat,
    WorkerTimeout,
    ServerNotRunning,
}

impl std::fmt::Display for ParameterServerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParameterServerError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            ParameterServerError::SerializationError(msg) => {
                write!(f, "Serialization error: {}", msg)
            }
            ParameterServerError::InvalidGradientFormat => write!(f, "Invalid gradient format"),
            ParameterServerError::DimensionMismatch => write!(f, "Dimension mismatch"),
            ParameterServerError::BackupCorruption => write!(f, "Backup corruption"),
            ParameterServerError::BackupNotFound => write!(f, "Backup not found"),
            ParameterServerError::InvalidBackupFormat => write!(f, "Invalid backup format"),
            ParameterServerError::WorkerTimeout => write!(f, "Worker timeout"),
            ParameterServerError::ServerNotRunning => write!(f, "Server not running"),
        }
    }
}

impl std::error::Error for ParameterServerError {}

impl From<std::io::Error> for ParameterServerError {
    fn from(error: std::io::Error) -> Self {
        ParameterServerError::NetworkError(error.to_string())
    }
}

#[allow(dead_code)]
pub struct ParameterServerClient {
    config: ParameterServerConfig,
    server_connections: HashMap<SocketAddr, TcpStream>,
    worker_id: u32,
}

impl ParameterServerClient {
    pub fn new(config: ParameterServerConfig) -> Self {
        Self {
            config,
            server_connections: HashMap::new(),
            worker_id: 0,
        }
    }

    pub fn connect(&mut self) -> Result<(), ParameterServerError> {
        let addresses = self.config.server_addresses.clone();
        for addr in addresses {
            match TcpStream::connect(addr) {
                Ok(stream) => {
                    self.server_connections.insert(addr, stream);
                    self.register_worker(addr)?;
                }
                Err(e) => {
                    eprintln!("Failed to connect to server {}: {}", addr, e);
                }
            }
        }

        if self.server_connections.is_empty() {
            return Err(ParameterServerError::NetworkError(
                "No servers available".to_string(),
            ));
        }

        Ok(())
    }

    fn register_worker(&mut self, addr: SocketAddr) -> Result<(), ParameterServerError> {
        if let Some(stream) = self.server_connections.get_mut(&addr) {
            let message = format!("REGISTER_WORKER|{}", self.config.worker_id);
            stream.write_all(message.as_bytes())?;
            stream.flush()?;

            let mut buffer = [0; 1024];
            let bytes_read = stream.read(&mut buffer)?;
            let response = String::from_utf8_lossy(&buffer[..bytes_read]);

            if response.starts_with("REGISTERED") {
                println!("Worker {} registered successfully", self.config.worker_id);
                Ok(())
            } else {
                Err(ParameterServerError::NetworkError(
                    "Registration failed".to_string(),
                ))
            }
        } else {
            Err(ParameterServerError::NetworkError(
                "Connection not found".to_string(),
            ))
        }
    }

    pub fn get_parameters(&mut self, parameter_id: &str) -> Result<Vec<f32>, ParameterServerError> {
        let addr =
            self.config
                .server_addresses
                .first()
                .ok_or(ParameterServerError::NetworkError(
                    "No servers configured".to_string(),
                ))?;

        if let Some(stream) = self.server_connections.get_mut(addr) {
            let message = format!("GET_PARAMS|{}", parameter_id);
            stream.write_all(message.as_bytes())?;
            stream.flush()?;

            let mut buffer = [0; 4096];
            let bytes_read = stream.read(&mut buffer)?;
            let response = String::from_utf8_lossy(&buffer[..bytes_read]);

            if response.starts_with("PARAMS") {
                let parts: Vec<&str> = response.split('|').collect();
                if parts.len() >= 3 {
                    return self.parse_parameter_response(parts[2]);
                }
            }

            Err(ParameterServerError::NetworkError(
                "Invalid response format".to_string(),
            ))
        } else {
            Err(ParameterServerError::NetworkError(
                "Connection not found".to_string(),
            ))
        }
    }

    fn parse_parameter_response(&self, data: &str) -> Result<Vec<f32>, ParameterServerError> {
        let parts: Vec<&str> = data.split('|').collect();

        if parts.len() < 2 {
            return Err(ParameterServerError::SerializationError(
                "Invalid parameter data".to_string(),
            ));
        }

        let _version = parts[0]
            .parse::<u64>()
            .map_err(|_| ParameterServerError::SerializationError("Invalid version".to_string()))?;

        let values_len = parts[1].parse::<usize>().map_err(|_| {
            ParameterServerError::SerializationError("Invalid values length".to_string())
        })?;

        if parts.len() != 2 + values_len {
            return Err(ParameterServerError::SerializationError(
                "Mismatched values count".to_string(),
            ));
        }

        let mut values = Vec::new();
        for i in 2..2 + values_len {
            let value = parts[i].parse::<f32>().map_err(|_| {
                ParameterServerError::SerializationError("Invalid value".to_string())
            })?;
            values.push(value);
        }

        Ok(values)
    }

    pub fn update_parameters(
        &mut self,
        parameter_id: &str,
        gradient: &[f32],
    ) -> Result<(), ParameterServerError> {
        let addr =
            self.config
                .server_addresses
                .first()
                .ok_or(ParameterServerError::NetworkError(
                    "No servers configured".to_string(),
                ))?;

        if let Some(stream) = self.server_connections.get_mut(addr) {
            let gradient_str = gradient
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");

            let message = format!(
                "UPDATE_PARAMS|{}|{}|{}",
                parameter_id, self.config.worker_id, gradient_str
            );
            stream.write_all(message.as_bytes())?;
            stream.flush()?;

            let mut buffer = [0; 1024];
            let bytes_read = stream.read(&mut buffer)?;
            let response = String::from_utf8_lossy(&buffer[..bytes_read]);

            if response.starts_with("ACK") {
                Ok(())
            } else {
                Err(ParameterServerError::NetworkError(
                    "Update failed".to_string(),
                ))
            }
        } else {
            Err(ParameterServerError::NetworkError(
                "Connection not found".to_string(),
            ))
        }
    }

    pub fn send_heartbeat(&mut self) -> Result<(), ParameterServerError> {
        let addr =
            self.config
                .server_addresses
                .first()
                .ok_or(ParameterServerError::NetworkError(
                    "No servers configured".to_string(),
                ))?;

        if let Some(stream) = self.server_connections.get_mut(addr) {
            let message = format!("HEARTBEAT|{}", self.config.worker_id);
            stream.write_all(message.as_bytes())?;
            stream.flush()?;

            let mut buffer = [0; 1024];
            let bytes_read = stream.read(&mut buffer)?;
            let response = String::from_utf8_lossy(&buffer[..bytes_read]);

            if response.starts_with("HEARTBEAT_ACK") {
                Ok(())
            } else {
                Err(ParameterServerError::NetworkError(
                    "Heartbeat failed".to_string(),
                ))
            }
        } else {
            Err(ParameterServerError::NetworkError(
                "Connection not found".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_server_config_default() {
        let config = ParameterServerConfig::default();
        assert_eq!(config.worker_id, 0);
        assert_eq!(config.num_workers, 1);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.learning_rate, 0.01);
    }

    #[test]
    fn test_parameter_update_creation() {
        let update = ParameterUpdate {
            parameter_id: "test_param".to_string(),
            gradient: vec![0.1, 0.2, 0.3],
            worker_id: 1,
            version: 1,
            timestamp: Instant::now(),
            priority: UpdatePriority::High,
        };

        assert_eq!(update.parameter_id, "test_param");
        assert_eq!(update.gradient.len(), 3);
        assert_eq!(update.worker_id, 1);
        assert_eq!(update.priority, UpdatePriority::High);
    }

    #[test]
    fn test_parameter_state_creation() {
        let state = ParameterState {
            parameter_id: "test_param".to_string(),
            values: vec![1.0, 2.0, 3.0],
            version: 1,
            last_updated: Instant::now(),
            gradient_accumulator: vec![0.0; 3],
            momentum_buffer: vec![0.0; 3],
            update_count: 0,
            staleness_count: 0,
        };

        assert_eq!(state.parameter_id, "test_param");
        assert_eq!(state.values.len(), 3);
        assert_eq!(state.version, 1);
    }

    #[test]
    fn test_update_scheduler_fifo() {
        let mut scheduler = UpdateScheduler::new(SchedulerStrategy::FIFO);

        let update1 = ParameterUpdate {
            parameter_id: "param1".to_string(),
            gradient: vec![0.1],
            worker_id: 1,
            version: 1,
            timestamp: Instant::now(),
            priority: UpdatePriority::Low,
        };

        let update2 = ParameterUpdate {
            parameter_id: "param2".to_string(),
            gradient: vec![0.2],
            worker_id: 2,
            version: 1,
            timestamp: Instant::now(),
            priority: UpdatePriority::High,
        };

        scheduler.schedule_update(update1.clone()).unwrap();
        scheduler.schedule_update(update2).unwrap();

        let next_update = scheduler.get_next_update().unwrap();
        assert_eq!(next_update.parameter_id, "param1");
    }

    #[test]
    fn test_update_scheduler_priority() {
        let mut scheduler = UpdateScheduler::new(SchedulerStrategy::Priority);

        let update1 = ParameterUpdate {
            parameter_id: "param1".to_string(),
            gradient: vec![0.1],
            worker_id: 1,
            version: 1,
            timestamp: Instant::now(),
            priority: UpdatePriority::Low,
        };

        let update2 = ParameterUpdate {
            parameter_id: "param2".to_string(),
            gradient: vec![0.2],
            worker_id: 2,
            version: 1,
            timestamp: Instant::now(),
            priority: UpdatePriority::High,
        };

        scheduler.schedule_update(update1).unwrap();
        scheduler.schedule_update(update2.clone()).unwrap();

        let next_update = scheduler.get_next_update().unwrap();
        assert_eq!(next_update.parameter_id, "param2");
    }

    #[test]
    fn test_fault_detector() {
        let mut detector = FaultDetector::new(Duration::from_secs(5));

        detector.update_worker_heartbeat(1);
        assert!(!detector.failed_workers.contains(&1));

        thread::sleep(Duration::from_millis(10));
        detector.check_worker_health();
        assert!(!detector.failed_workers.contains(&1));
    }

    #[test]
    fn test_backup_manager() {
        let mut manager = BackupManager::new(Duration::from_secs(10));
        let mut parameters = HashMap::new();

        let param_state = ParameterState {
            parameter_id: "test_param".to_string(),
            values: vec![1.0, 2.0, 3.0],
            version: 1,
            last_updated: Instant::now(),
            gradient_accumulator: vec![0.0; 3],
            momentum_buffer: vec![0.0; 3],
            update_count: 0,
            staleness_count: 0,
        };

        parameters.insert("test_param".to_string(), param_state);

        assert!(manager.create_backup(&parameters).is_ok());
        assert!(!manager.backup_storage.is_empty());
    }

    #[test]
    fn test_parameter_server_creation() {
        let config = ParameterServerConfig::default();
        let server = ParameterServer::new(config);

        assert!(server.parameters.read().unwrap().is_empty());
        assert!(server.pending_updates.lock().unwrap().is_empty());
    }

    #[test]
    fn test_parameter_server_register_parameter() {
        let config = ParameterServerConfig::default();
        let server = ParameterServer::new(config);

        let result = server.register_parameter("test_param".to_string(), vec![1.0, 2.0, 3.0]);
        assert!(result.is_ok());

        let params = server.parameters.read().unwrap();
        assert!(params.contains_key("test_param"));
        assert_eq!(params["test_param"].values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_parameter_server_client() {
        let config = ParameterServerConfig::default();
        let client = ParameterServerClient::new(config);

        assert!(client.server_connections.is_empty());
        assert_eq!(client.worker_id, 0);
    }

    #[test]
    fn test_gradient_serialization() {
        let gradient = vec![0.1, 0.2, 0.3, 0.4];
        let gradient_str = gradient
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");

        let deserialized = ParameterServer::deserialize_gradient(&gradient_str).unwrap();
        assert_eq!(deserialized, gradient);
    }

    #[test]
    fn test_parameter_state_serialization() {
        let state = ParameterState {
            parameter_id: "test_param".to_string(),
            values: vec![1.0, 2.0, 3.0],
            version: 1,
            last_updated: Instant::now(),
            gradient_accumulator: vec![0.0; 3],
            momentum_buffer: vec![0.0; 3],
            update_count: 0,
            staleness_count: 0,
        };

        let serialized = ParameterServer::serialize_parameter_state(&state).unwrap();
        assert!(serialized.contains("1|3|1|2|3"));
    }

    #[test]
    fn test_update_priority_ordering() {
        assert!(UpdatePriority::Critical > UpdatePriority::High);
        assert!(UpdatePriority::High > UpdatePriority::Normal);
        assert!(UpdatePriority::Normal > UpdatePriority::Low);
    }

    #[test]
    fn test_parameter_server_error_display() {
        let error = ParameterServerError::DimensionMismatch;
        assert_eq!(format!("{}", error), "Dimension mismatch");

        let error = ParameterServerError::NetworkError("Test error".to_string());
        assert_eq!(format!("{}", error), "Network error: Test error");
    }
}
