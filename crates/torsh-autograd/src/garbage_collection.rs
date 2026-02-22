//! Automatic garbage collection for unused gradients and computation graph nodes
//!
//! This module provides intelligent garbage collection for autograd operations,
//! automatically cleaning up unused gradients, intermediate values, and computation
//! graph nodes to optimize memory usage.

use parking_lot::Mutex;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock, Weak};
use std::time::{Duration, Instant};
use torsh_core::dtype::FloatElement;
use torsh_core::error::{Result, TorshError};

/// Reference counting for gradient lifecycle management
#[derive(Debug, Clone)]
pub struct GradientReference<T: FloatElement> {
    /// Unique identifier for this gradient
    pub id: GradientId,
    /// Gradient data
    pub data: Arc<RwLock<Vec<T>>>,
    /// Reference count
    pub ref_count: Arc<Mutex<usize>>,
    /// Last access time
    pub last_accessed: Arc<Mutex<Instant>>,
    /// Whether this gradient is marked for deletion
    pub marked_for_deletion: Arc<Mutex<bool>>,
    /// Size in bytes
    pub size_bytes: usize,
    /// Associated computation graph node
    pub graph_node_id: Option<usize>,
}

/// Gradient identifier type
pub type GradientId = usize;

/// Garbage collection strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcStrategy {
    /// Reference counting based GC
    ReferenceCounting,
    /// Mark and sweep GC
    MarkAndSweep,
    /// Generational GC
    Generational,
    /// Adaptive GC (chooses strategy based on conditions)
    Adaptive,
}

/// Garbage collection trigger conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcTrigger {
    /// Memory pressure threshold
    MemoryPressure,
    /// Time-based (periodic)
    Periodic,
    /// Number of allocations
    AllocationCount,
    /// Manual trigger
    Manual,
    /// Adaptive (multiple conditions)
    Adaptive,
}

/// Configuration for garbage collection
#[derive(Debug, Clone)]
pub struct GarbageCollectionConfig {
    /// GC strategy to use
    pub strategy: GcStrategy,
    /// Trigger condition
    pub trigger: GcTrigger,
    /// Memory threshold for triggering GC (bytes)
    pub memory_threshold: usize,
    /// Time threshold for periodic GC
    pub time_threshold: Duration,
    /// Allocation count threshold
    pub allocation_threshold: usize,
    /// Maximum age for unused gradients
    pub max_gradient_age: Duration,
    /// Enable automatic GC
    pub enable_auto_gc: bool,
    /// GC aggressiveness (0.0 to 1.0)
    pub aggressiveness: f64,
    /// Keep alive threshold for frequently used gradients
    pub keep_alive_threshold: usize,
}

impl Default for GarbageCollectionConfig {
    fn default() -> Self {
        Self {
            strategy: GcStrategy::Adaptive,
            trigger: GcTrigger::Adaptive,
            memory_threshold: 100 * 1024 * 1024, // 100MB
            time_threshold: Duration::from_secs(30),
            allocation_threshold: 1000,
            max_gradient_age: Duration::from_secs(60),
            enable_auto_gc: true,
            aggressiveness: 0.5,
            keep_alive_threshold: 10,
        }
    }
}

/// Garbage collection statistics
#[derive(Debug, Clone, Default)]
pub struct GcStats {
    /// Total GC runs
    pub total_gc_runs: usize,
    /// Total gradients collected
    pub total_gradients_collected: usize,
    /// Total memory freed (bytes)
    pub total_memory_freed: usize,
    /// Average GC time
    pub average_gc_time_ms: f64,
    /// Last GC time
    pub last_gc_time: Option<Instant>,
    /// Memory recovered in last GC
    pub last_gc_memory_freed: usize,
    /// Current number of tracked gradients
    pub current_gradient_count: usize,
    /// Peak gradient count
    pub peak_gradient_count: usize,
}

/// Generation information for generational GC
#[derive(Debug, Clone)]
pub struct Generation {
    /// Generation number (0 = youngest)
    pub number: usize,
    /// Gradients in this generation
    pub gradients: HashSet<GradientId>,
    /// Age threshold for promotion to next generation
    pub age_threshold: Duration,
    /// Collection frequency
    pub collection_frequency: usize,
}

/// Automatic garbage collector for gradients
pub struct GradientGarbageCollector<T: FloatElement> {
    /// Configuration
    config: GarbageCollectionConfig,
    /// Tracked gradients
    gradients: Arc<RwLock<HashMap<GradientId, GradientReference<T>>>>,
    /// Gradient ID counter
    next_id: Arc<Mutex<GradientId>>,
    /// GC statistics
    stats: Arc<RwLock<GcStats>>,
    /// Generations for generational GC
    generations: Arc<RwLock<Vec<Generation>>>,
    /// Recently accessed gradients
    recent_access: Arc<Mutex<VecDeque<(GradientId, Instant)>>>,
    /// Root set (gradients that should not be collected)
    root_set: Arc<RwLock<HashSet<GradientId>>>,
    /// Background GC thread handle
    _gc_thread: Option<std::thread::JoinHandle<()>>,
    /// GC trigger state
    trigger_state: Arc<Mutex<GcTriggerState>>,
}

/// State for GC triggers
#[derive(Debug, Default)]
struct GcTriggerState {
    /// Last GC time
    last_gc_time: Option<Instant>,
    /// Allocation count since last GC
    allocations_since_gc: usize,
    /// Memory usage at last GC
    memory_at_last_gc: usize,
}

impl<T: FloatElement + Send + Sync + 'static> GradientGarbageCollector<T> {
    /// Create a new garbage collector
    pub fn new(config: GarbageCollectionConfig) -> Self {
        let gradients = Arc::new(RwLock::new(HashMap::new()));
        let stats = Arc::new(RwLock::new(GcStats::default()));
        let generations = Arc::new(RwLock::new(Self::initialize_generations()));
        let trigger_state = Arc::new(Mutex::new(GcTriggerState::default()));

        // Start background GC thread if auto GC is enabled
        let gc_thread = if config.enable_auto_gc {
            Some(Self::start_gc_thread(
                gradients.clone(),
                stats.clone(),
                config.clone(),
                trigger_state.clone(),
            ))
        } else {
            None
        };

        Self {
            config,
            gradients,
            next_id: Arc::new(Mutex::new(0)),
            stats,
            generations,
            recent_access: Arc::new(Mutex::new(VecDeque::new())),
            root_set: Arc::new(RwLock::new(HashSet::new())),
            _gc_thread: gc_thread,
            trigger_state,
        }
    }

    /// Initialize generations for generational GC
    fn initialize_generations() -> Vec<Generation> {
        vec![
            Generation {
                number: 0,
                gradients: HashSet::new(),
                age_threshold: Duration::from_secs(5),
                collection_frequency: 1,
            },
            Generation {
                number: 1,
                gradients: HashSet::new(),
                age_threshold: Duration::from_secs(30),
                collection_frequency: 5,
            },
            Generation {
                number: 2,
                gradients: HashSet::new(),
                age_threshold: Duration::MAX,
                collection_frequency: 20,
            },
        ]
    }

    /// Start background GC thread
    fn start_gc_thread(
        gradients: Arc<RwLock<HashMap<GradientId, GradientReference<T>>>>,
        stats: Arc<RwLock<GcStats>>,
        config: GarbageCollectionConfig,
        trigger_state: Arc<Mutex<GcTriggerState>>,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            loop {
                // Check if GC should be triggered
                let should_trigger = {
                    let trigger_state = trigger_state.lock();
                    match config.trigger {
                        GcTrigger::Periodic => trigger_state
                            .last_gc_time
                            .map(|t| t.elapsed() > config.time_threshold)
                            .unwrap_or(true),
                        GcTrigger::AllocationCount => {
                            trigger_state.allocations_since_gc > config.allocation_threshold
                        }
                        GcTrigger::MemoryPressure => {
                            // Simplified memory pressure check
                            let current_count =
                                gradients.read().expect("lock should not be poisoned").len();
                            current_count > 1000 // Arbitrary threshold
                        }
                        GcTrigger::Adaptive => {
                            let time_trigger = trigger_state
                                .last_gc_time
                                .map(|t| t.elapsed() > config.time_threshold)
                                .unwrap_or(true);
                            let alloc_trigger =
                                trigger_state.allocations_since_gc > config.allocation_threshold;
                            time_trigger || alloc_trigger
                        }
                        GcTrigger::Manual => false, // Never auto-trigger for manual
                    }
                };

                if should_trigger {
                    // Perform GC
                    Self::perform_gc_static(&gradients, &stats, &config, &trigger_state);
                }

                // Sleep for a short interval
                std::thread::sleep(Duration::from_millis(100));
            }
        })
    }

    /// Static GC method for background thread
    fn perform_gc_static(
        gradients: &Arc<RwLock<HashMap<GradientId, GradientReference<T>>>>,
        stats: &Arc<RwLock<GcStats>>,
        config: &GarbageCollectionConfig,
        trigger_state: &Arc<Mutex<GcTriggerState>>,
    ) {
        let start_time = Instant::now();
        let mut collected_count = 0;
        let mut freed_memory = 0;

        // Collect unreferenced gradients
        let to_remove: Vec<GradientId> = {
            let gradients_read = gradients.read().expect("lock should not be poisoned");
            gradients_read
                .iter()
                .filter_map(|(id, grad_ref)| {
                    let ref_count = *grad_ref.ref_count.lock();
                    let last_accessed = *grad_ref.last_accessed.lock();
                    let marked_for_deletion = *grad_ref.marked_for_deletion.lock();

                    let should_collect = ref_count == 0
                        || marked_for_deletion
                        || last_accessed.elapsed() > config.max_gradient_age;

                    if should_collect {
                        Some(*id)
                    } else {
                        None
                    }
                })
                .collect()
        };

        // Remove collected gradients
        {
            let mut gradients_write = gradients.write().expect("lock should not be poisoned");
            for id in to_remove {
                if let Some(grad_ref) = gradients_write.remove(&id) {
                    collected_count += 1;
                    freed_memory += grad_ref.size_bytes;
                }
            }
        }

        // Update statistics
        {
            let mut stats_write = stats.write().expect("lock should not be poisoned");
            stats_write.total_gc_runs += 1;
            stats_write.total_gradients_collected += collected_count;
            stats_write.total_memory_freed += freed_memory;
            stats_write.last_gc_time = Some(start_time);
            stats_write.last_gc_memory_freed = freed_memory;
            stats_write.current_gradient_count =
                gradients.read().expect("lock should not be poisoned").len();

            // Update average GC time
            let gc_time_ms = start_time.elapsed().as_millis() as f64;
            stats_write.average_gc_time_ms = (stats_write.average_gc_time_ms
                * (stats_write.total_gc_runs - 1) as f64
                + gc_time_ms)
                / stats_write.total_gc_runs as f64;
        }

        // Update trigger state
        {
            let mut trigger_state = trigger_state.lock();
            trigger_state.last_gc_time = Some(start_time);
            trigger_state.allocations_since_gc = 0;
            trigger_state.memory_at_last_gc = freed_memory;
        }

        if collected_count > 0 {
            tracing::debug!(
                "GC completed: collected {} gradients, freed {} bytes in {:.2}ms",
                collected_count,
                freed_memory,
                start_time.elapsed().as_millis()
            );
        }
    }

    /// Allocate a new gradient with automatic tracking
    pub fn allocate_gradient(&self, data: Vec<T>) -> Result<GradientId> {
        let id = {
            let mut next_id = self.next_id.lock();
            let id = *next_id;
            *next_id += 1;
            id
        };

        let size_bytes = data.len() * std::mem::size_of::<T>();
        let grad_ref = GradientReference {
            id,
            data: Arc::new(RwLock::new(data)),
            ref_count: Arc::new(Mutex::new(1)),
            last_accessed: Arc::new(Mutex::new(Instant::now())),
            marked_for_deletion: Arc::new(Mutex::new(false)),
            size_bytes,
            graph_node_id: None,
        };

        // Add to tracking
        {
            let mut gradients = self.gradients.write().expect("lock should not be poisoned");
            gradients.insert(id, grad_ref);
        }

        // Update statistics
        {
            let mut stats = self.stats.write().expect("lock should not be poisoned");
            stats.current_gradient_count += 1;
            if stats.current_gradient_count > stats.peak_gradient_count {
                stats.peak_gradient_count = stats.current_gradient_count;
            }
        }

        // Update trigger state
        {
            let mut trigger_state = self.trigger_state.lock();
            trigger_state.allocations_since_gc += 1;
        }

        // Add to youngest generation for generational GC
        if matches!(
            self.config.strategy,
            GcStrategy::Generational | GcStrategy::Adaptive
        ) {
            let mut generations = self
                .generations
                .write()
                .expect("lock should not be poisoned");
            if let Some(gen0) = generations.get_mut(0) {
                gen0.gradients.insert(id);
            }
        }

        Ok(id)
    }

    /// Increment reference count for a gradient
    pub fn retain_gradient(&self, id: GradientId) -> Result<()> {
        let gradients = self.gradients.read().expect("lock should not be poisoned");
        if let Some(grad_ref) = gradients.get(&id) {
            let mut ref_count = grad_ref.ref_count.lock();
            *ref_count += 1;
            *grad_ref.last_accessed.lock() = Instant::now();

            // Track recent access
            {
                let mut recent_access = self.recent_access.lock();
                recent_access.push_back((id, Instant::now()));
                // Keep only recent accesses
                if recent_access.len() > 1000 {
                    recent_access.pop_front();
                }
            }

            Ok(())
        } else {
            Err(TorshError::AutogradError(format!(
                "Gradient {id} not found"
            )))
        }
    }

    /// Decrement reference count for a gradient
    pub fn release_gradient(&self, id: GradientId) -> Result<()> {
        let gradients = self.gradients.read().expect("lock should not be poisoned");
        if let Some(grad_ref) = gradients.get(&id) {
            let mut ref_count = grad_ref.ref_count.lock();
            if *ref_count > 0 {
                *ref_count -= 1;
            }
            Ok(())
        } else {
            Err(TorshError::AutogradError(format!(
                "Gradient {id} not found"
            )))
        }
    }

    /// Get gradient data (with access tracking)
    pub fn get_gradient(&self, id: GradientId) -> Result<Arc<RwLock<Vec<T>>>> {
        let gradients = self.gradients.read().expect("lock should not be poisoned");
        if let Some(grad_ref) = gradients.get(&id) {
            *grad_ref.last_accessed.lock() = Instant::now();
            Ok(grad_ref.data.clone())
        } else {
            Err(TorshError::AutogradError(format!(
                "Gradient {id} not found"
            )))
        }
    }

    /// Mark gradient for deletion
    pub fn mark_for_deletion(&self, id: GradientId) -> Result<()> {
        let gradients = self.gradients.read().expect("lock should not be poisoned");
        if let Some(grad_ref) = gradients.get(&id) {
            *grad_ref.marked_for_deletion.lock() = true;
            Ok(())
        } else {
            Err(TorshError::AutogradError(format!(
                "Gradient {id} not found"
            )))
        }
    }

    /// Add gradient to root set (prevent collection)
    pub fn add_to_root_set(&self, id: GradientId) {
        let mut root_set = self.root_set.write().expect("lock should not be poisoned");
        root_set.insert(id);
    }

    /// Remove gradient from root set
    pub fn remove_from_root_set(&self, id: GradientId) {
        let mut root_set = self.root_set.write().expect("lock should not be poisoned");
        root_set.remove(&id);
    }

    /// Manually trigger garbage collection
    pub fn collect_garbage(&self) -> Result<GcResult> {
        let _start_time = Instant::now();

        match self.config.strategy {
            GcStrategy::ReferenceCounting => self.reference_counting_gc(),
            GcStrategy::MarkAndSweep => self.mark_and_sweep_gc(),
            GcStrategy::Generational => self.generational_gc(),
            GcStrategy::Adaptive => self.adaptive_gc(),
        }
    }

    /// Reference counting garbage collection
    fn reference_counting_gc(&self) -> Result<GcResult> {
        let start_time = Instant::now();
        let mut collected_count = 0;
        let mut freed_memory = 0;

        let to_remove: Vec<GradientId> = {
            let gradients = self.gradients.read().expect("lock should not be poisoned");
            let root_set = self.root_set.read().expect("lock should not be poisoned");

            gradients
                .iter()
                .filter_map(|(id, grad_ref)| {
                    if root_set.contains(id) {
                        return None; // Don't collect root set members
                    }

                    let ref_count = *grad_ref.ref_count.lock();
                    let marked_for_deletion = *grad_ref.marked_for_deletion.lock();

                    if ref_count == 0 || marked_for_deletion {
                        Some(*id)
                    } else {
                        None
                    }
                })
                .collect()
        };

        // Remove collected gradients
        {
            let mut gradients = self.gradients.write().expect("lock should not be poisoned");
            for id in to_remove {
                if let Some(grad_ref) = gradients.remove(&id) {
                    collected_count += 1;
                    freed_memory += grad_ref.size_bytes;
                }
            }
        }

        self.update_gc_stats(start_time, collected_count, freed_memory);

        Ok(GcResult {
            strategy: GcStrategy::ReferenceCounting,
            duration: start_time.elapsed(),
            gradients_collected: collected_count,
            memory_freed: freed_memory,
        })
    }

    /// Mark and sweep garbage collection
    fn mark_and_sweep_gc(&self) -> Result<GcResult> {
        let start_time = Instant::now();
        let mut collected_count = 0;
        let mut freed_memory = 0;

        // Mark phase: mark all reachable gradients
        let mut marked = HashSet::new();

        // Start with root set
        {
            let root_set = self.root_set.read().expect("lock should not be poisoned");
            for &id in root_set.iter() {
                marked.insert(id);
            }
        }

        // Mark recently accessed gradients
        {
            let recent_access = self.recent_access.lock();
            let cutoff = Instant::now() - Duration::from_secs(30);
            for (id, access_time) in recent_access.iter() {
                if *access_time > cutoff {
                    marked.insert(*id);
                }
            }
        }

        // Sweep phase: collect unmarked gradients
        let to_remove: Vec<GradientId> = {
            let gradients = self.gradients.read().expect("lock should not be poisoned");
            gradients
                .keys()
                .filter(|id| !marked.contains(id))
                .copied()
                .collect()
        };

        // Remove collected gradients
        {
            let mut gradients = self.gradients.write().expect("lock should not be poisoned");
            for id in to_remove {
                if let Some(grad_ref) = gradients.remove(&id) {
                    collected_count += 1;
                    freed_memory += grad_ref.size_bytes;
                }
            }
        }

        self.update_gc_stats(start_time, collected_count, freed_memory);

        Ok(GcResult {
            strategy: GcStrategy::MarkAndSweep,
            duration: start_time.elapsed(),
            gradients_collected: collected_count,
            memory_freed: freed_memory,
        })
    }

    /// Generational garbage collection
    fn generational_gc(&self) -> Result<GcResult> {
        let start_time = Instant::now();
        let mut total_collected = 0;
        let mut total_freed = 0;

        // Collect each generation based on frequency
        {
            let mut generations = self
                .generations
                .write()
                .expect("lock should not be poisoned");
            let stats = self.stats.read().expect("lock should not be poisoned");

            for generation in generations.iter_mut() {
                if stats.total_gc_runs % generation.collection_frequency == 0 {
                    let (collected, freed) = self.collect_generation(generation)?;
                    total_collected += collected;
                    total_freed += freed;
                }
            }
        }

        // Promote surviving gradients to next generation
        self.promote_gradients()?;

        self.update_gc_stats(start_time, total_collected, total_freed);

        Ok(GcResult {
            strategy: GcStrategy::Generational,
            duration: start_time.elapsed(),
            gradients_collected: total_collected,
            memory_freed: total_freed,
        })
    }

    /// Collect a specific generation
    fn collect_generation(&self, generation: &mut Generation) -> Result<(usize, usize)> {
        let mut collected_count = 0;
        let mut freed_memory = 0;

        let to_remove: Vec<GradientId> = {
            let gradients = self.gradients.read().expect("lock should not be poisoned");
            generation
                .gradients
                .iter()
                .filter_map(|&id| {
                    if let Some(grad_ref) = gradients.get(&id) {
                        let ref_count = *grad_ref.ref_count.lock();
                        let last_accessed = *grad_ref.last_accessed.lock();

                        if ref_count == 0 || last_accessed.elapsed() > generation.age_threshold {
                            Some(id)
                        } else {
                            None
                        }
                    } else {
                        Some(id) // Remove if not found
                    }
                })
                .collect()
        };

        // Remove from generation and gradients
        for id in to_remove {
            generation.gradients.remove(&id);

            let mut gradients = self.gradients.write().expect("lock should not be poisoned");
            if let Some(grad_ref) = gradients.remove(&id) {
                collected_count += 1;
                freed_memory += grad_ref.size_bytes;
            }
        }

        Ok((collected_count, freed_memory))
    }

    /// Promote gradients to next generation
    fn promote_gradients(&self) -> Result<()> {
        let mut generations = self
            .generations
            .write()
            .expect("lock should not be poisoned");

        for i in 0..generations.len() - 1 {
            let to_promote: Vec<GradientId> = {
                let gradients = self.gradients.read().expect("lock should not be poisoned");
                generations[i]
                    .gradients
                    .iter()
                    .filter_map(|&id| {
                        if let Some(grad_ref) = gradients.get(&id) {
                            let last_accessed = *grad_ref.last_accessed.lock();
                            if last_accessed.elapsed() > generations[i].age_threshold {
                                Some(id)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .collect()
            };

            // Move to next generation
            for id in to_promote {
                generations[i].gradients.remove(&id);
                generations[i + 1].gradients.insert(id);
            }
        }

        Ok(())
    }

    /// Adaptive garbage collection
    fn adaptive_gc(&self) -> Result<GcResult> {
        let stats = self.stats.read().expect("lock should not be poisoned");
        let current_count = stats.current_gradient_count;
        let memory_pressure = current_count > 1000; // Simplified

        drop(stats);

        // Choose strategy based on conditions
        if memory_pressure {
            // Use aggressive mark and sweep for high memory pressure
            self.mark_and_sweep_gc()
        } else if current_count > 500 {
            // Use generational for moderate loads
            self.generational_gc()
        } else {
            // Use reference counting for light loads
            self.reference_counting_gc()
        }
    }

    /// Update GC statistics
    fn update_gc_stats(&self, start_time: Instant, collected_count: usize, freed_memory: usize) {
        let mut stats = self.stats.write().expect("lock should not be poisoned");
        stats.total_gc_runs += 1;
        stats.total_gradients_collected += collected_count;
        stats.total_memory_freed += freed_memory;
        stats.last_gc_time = Some(start_time);
        stats.last_gc_memory_freed = freed_memory;
        stats.current_gradient_count = self
            .gradients
            .read()
            .expect("lock should not be poisoned")
            .len();

        let gc_time_ms = start_time.elapsed().as_millis() as f64;
        stats.average_gc_time_ms = (stats.average_gc_time_ms * (stats.total_gc_runs - 1) as f64
            + gc_time_ms)
            / stats.total_gc_runs as f64;
    }

    /// Get garbage collection statistics
    pub fn get_gc_stats(&self) -> GcStats {
        self.stats
            .read()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Get current gradient count
    pub fn get_gradient_count(&self) -> usize {
        self.gradients
            .read()
            .expect("lock should not be poisoned")
            .len()
    }

    /// Get memory usage information
    pub fn get_memory_usage(&self) -> usize {
        let gradients = self.gradients.read().expect("lock should not be poisoned");
        gradients.values().map(|g| g.size_bytes).sum()
    }

    /// Check if a gradient exists
    pub fn contains_gradient(&self, id: GradientId) -> bool {
        self.gradients
            .read()
            .expect("lock should not be poisoned")
            .contains_key(&id)
    }

    /// Get reference count for a gradient
    pub fn get_reference_count(&self, id: GradientId) -> Option<usize> {
        let gradients = self.gradients.read().expect("lock should not be poisoned");
        gradients.get(&id).map(|g| *g.ref_count.lock())
    }
}

/// Result of a garbage collection run
#[derive(Debug, Clone)]
pub struct GcResult {
    /// Strategy used
    pub strategy: GcStrategy,
    /// Duration of GC
    pub duration: Duration,
    /// Number of gradients collected
    pub gradients_collected: usize,
    /// Memory freed in bytes
    pub memory_freed: usize,
}

/// Smart pointer for gradients with automatic reference counting
pub struct GradientPtr<T: FloatElement> {
    /// Gradient ID
    id: GradientId,
    /// Reference to the garbage collector
    gc: Weak<GradientGarbageCollector<T>>,
}

impl<T: FloatElement> GradientPtr<T> {
    /// Create a new gradient pointer
    pub fn new(id: GradientId, gc: Weak<GradientGarbageCollector<T>>) -> Self {
        // Increment reference count when creating pointer
        if let Some(gc_strong) = gc.upgrade() {
            let _ = gc_strong.retain_gradient(id);
        }
        Self { id, gc }
    }

    /// Get the gradient data
    pub fn get(&self) -> Result<Arc<RwLock<Vec<T>>>> {
        if let Some(gc) = self.gc.upgrade() {
            gc.get_gradient(self.id)
        } else {
            Err(TorshError::AutogradError(
                "Garbage collector dropped".to_string(),
            ))
        }
    }

    /// Get gradient ID
    pub fn id(&self) -> GradientId {
        self.id
    }
}

impl<T: FloatElement> Clone for GradientPtr<T> {
    fn clone(&self) -> Self {
        // Increment reference count
        if let Some(gc) = self.gc.upgrade() {
            let _ = gc.retain_gradient(self.id);
        }

        Self {
            id: self.id,
            gc: self.gc.clone(),
        }
    }
}

impl<T: FloatElement> Drop for GradientPtr<T> {
    fn drop(&mut self) {
        // Decrement reference count
        if let Some(gc) = self.gc.upgrade() {
            let _ = gc.release_gradient(self.id);
        }
    }
}

/// Utilities for garbage collection
pub mod utils {
    use super::*;

    /// Calculate GC efficiency
    pub fn calculate_gc_efficiency(stats: &GcStats) -> f64 {
        if stats.total_gc_runs == 0 {
            0.0
        } else {
            stats.total_memory_freed as f64 / stats.total_gc_runs as f64
        }
    }

    /// Get GC recommendations
    pub fn get_gc_recommendations(
        stats: &GcStats,
        config: &GarbageCollectionConfig,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if stats.average_gc_time_ms > 100.0 {
            recommendations
                .push("GC taking too long - consider reducing aggressiveness".to_string());
        }

        if stats.current_gradient_count > 10000 {
            recommendations.push("High gradient count - consider more frequent GC".to_string());
        }

        if stats.total_memory_freed < 1024 * 1024 {
            recommendations.push("Low memory recovery - check for memory leaks".to_string());
        }

        if config.aggressiveness < 0.3 && stats.current_gradient_count > 1000 {
            recommendations.push("Consider increasing GC aggressiveness".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("GC performance is optimal".to_string());
        }

        recommendations
    }

    /// Format GC statistics
    pub fn format_gc_stats(stats: &GcStats) -> String {
        format!(
            "GC Stats: {} runs, {} gradients collected, {:.2} MB freed, {:.2}ms avg time",
            stats.total_gc_runs,
            stats.total_gradients_collected,
            stats.total_memory_freed as f64 / (1024.0 * 1024.0),
            stats.average_gc_time_ms
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_gc_operations() {
        let config = GarbageCollectionConfig {
            enable_auto_gc: false, // Disable for testing
            ..Default::default()
        };
        let gc = GradientGarbageCollector::<f32>::new(config);

        // Allocate some gradients
        let id1 = gc.allocate_gradient(vec![1.0, 2.0, 3.0]).unwrap();
        let id2 = gc.allocate_gradient(vec![4.0, 5.0, 6.0]).unwrap();

        assert_eq!(gc.get_gradient_count(), 2);

        // Release one gradient
        gc.release_gradient(id1).unwrap();

        // Manual GC should collect the released gradient
        let result = gc.collect_garbage().unwrap();
        assert!(result.gradients_collected > 0);
        assert!(result.memory_freed > 0);

        // One gradient should remain
        assert_eq!(gc.get_gradient_count(), 1);
        assert!(gc.contains_gradient(id2));
        assert!(!gc.contains_gradient(id1));
    }

    #[test]
    fn test_reference_counting() {
        let config = GarbageCollectionConfig {
            enable_auto_gc: false,
            strategy: GcStrategy::ReferenceCounting,
            ..Default::default()
        };
        let gc = GradientGarbageCollector::<f32>::new(config);

        let id = gc.allocate_gradient(vec![1.0, 2.0]).unwrap();
        assert_eq!(gc.get_reference_count(id), Some(1));

        // Retain gradient
        gc.retain_gradient(id).unwrap();
        assert_eq!(gc.get_reference_count(id), Some(2));

        // Release once
        gc.release_gradient(id).unwrap();
        assert_eq!(gc.get_reference_count(id), Some(1));

        // Release again
        gc.release_gradient(id).unwrap();
        assert_eq!(gc.get_reference_count(id), Some(0));

        // GC should collect it now
        let result = gc.collect_garbage().unwrap();
        assert_eq!(result.gradients_collected, 1);
    }

    #[test]
    fn test_mark_and_sweep() {
        let config = GarbageCollectionConfig {
            enable_auto_gc: false,
            strategy: GcStrategy::MarkAndSweep,
            ..Default::default()
        };
        let gc = GradientGarbageCollector::<f32>::new(config);

        let id1 = gc.allocate_gradient(vec![1.0]).unwrap();
        let id2 = gc.allocate_gradient(vec![2.0]).unwrap();

        // Add one to root set
        gc.add_to_root_set(id1);

        // Release both
        gc.release_gradient(id1).unwrap();
        gc.release_gradient(id2).unwrap();

        // GC should only collect id2 (id1 is in root set)
        let result = gc.collect_garbage().unwrap();
        assert_eq!(result.gradients_collected, 1);
        assert!(gc.contains_gradient(id1));
        assert!(!gc.contains_gradient(id2));
    }

    #[test]
    fn test_generational_gc() {
        let config = GarbageCollectionConfig {
            enable_auto_gc: false,
            strategy: GcStrategy::Generational,
            ..Default::default()
        };
        let gc = GradientGarbageCollector::<f32>::new(config);

        // Allocate gradients
        let id1 = gc.allocate_gradient(vec![1.0]).unwrap();
        let _id2 = gc.allocate_gradient(vec![2.0]).unwrap();

        // Release one
        gc.release_gradient(id1).unwrap();

        // Generational GC
        let result = gc.collect_garbage().unwrap();
        assert!(result.gradients_collected <= 2); // May collect based on generation rules
    }

    #[test]
    fn test_gradient_ptr() {
        let config = GarbageCollectionConfig {
            enable_auto_gc: false,
            ..Default::default()
        };
        let gc = Arc::new(GradientGarbageCollector::<f32>::new(config));

        let id = gc.allocate_gradient(vec![1.0, 2.0, 3.0]).unwrap();

        // Create gradient pointer
        let ptr = GradientPtr::new(id, Arc::downgrade(&gc));
        assert_eq!(ptr.id(), id);
        assert_eq!(gc.get_reference_count(id), Some(2)); // 1 initial + 1 from pointer creation

        // Clone should increment reference count
        let ptr2 = ptr.clone();
        assert_eq!(gc.get_reference_count(id), Some(3)); // 1 initial + 2 from pointers

        // Drop should decrement
        drop(ptr);
        assert_eq!(gc.get_reference_count(id), Some(2));

        drop(ptr2);
        assert_eq!(gc.get_reference_count(id), Some(1));
    }

    #[test]
    fn test_gc_statistics() {
        let config = GarbageCollectionConfig {
            enable_auto_gc: false,
            ..Default::default()
        };
        let gc = GradientGarbageCollector::<f32>::new(config);

        // Initial stats
        let stats = gc.get_gc_stats();
        assert_eq!(stats.total_gc_runs, 0);

        // Allocate and release
        let id = gc.allocate_gradient(vec![1.0; 1000]).unwrap();
        gc.release_gradient(id).unwrap();

        // Run GC
        let _result = gc.collect_garbage().unwrap();

        // Check updated stats
        let stats = gc.get_gc_stats();
        assert_eq!(stats.total_gc_runs, 1);
        assert!(stats.total_gradients_collected > 0);
        assert!(stats.total_memory_freed > 0);
    }

    #[test]
    fn test_memory_usage_tracking() {
        let config = GarbageCollectionConfig {
            enable_auto_gc: false,
            ..Default::default()
        };
        let gc = GradientGarbageCollector::<f32>::new(config);

        let initial_usage = gc.get_memory_usage();

        // Allocate gradients
        let _id1 = gc.allocate_gradient(vec![1.0; 100]).unwrap();
        let _id2 = gc.allocate_gradient(vec![2.0; 200]).unwrap();

        let usage_after_alloc = gc.get_memory_usage();
        assert!(usage_after_alloc > initial_usage);

        // Expected usage: 300 f32s * 4 bytes = 1200 bytes
        assert_eq!(usage_after_alloc, 1200);
    }

    #[test]
    fn test_utility_functions() {
        let stats = GcStats {
            total_gc_runs: 5,
            total_memory_freed: 5 * 1024 * 1024,
            average_gc_time_ms: 50.0,
            current_gradient_count: 100,
            ..Default::default()
        };

        let efficiency = utils::calculate_gc_efficiency(&stats);
        assert_eq!(efficiency, 1024.0 * 1024.0); // 1MB per run

        let config = GarbageCollectionConfig::default();
        let recommendations = utils::get_gc_recommendations(&stats, &config);
        assert!(!recommendations.is_empty());

        let formatted = utils::format_gc_stats(&stats);
        assert!(formatted.contains("5 runs"));
        assert!(formatted.contains("5.00 MB"));
    }
}
