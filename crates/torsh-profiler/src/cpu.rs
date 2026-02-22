//! CPU profiling

use crate::{ProfileEvent, TorshResult};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use torsh_core::TorshError;

/// CPU profiler for tracking performance metrics
pub struct CpuProfiler {
    events: Arc<Mutex<Vec<ProfileEvent>>>,
    start_time: Instant,
    enabled: bool,
}

impl Default for CpuProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuProfiler {
    /// Create a new CPU profiler
    pub fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
            start_time: Instant::now(),
            enabled: false,
        }
    }

    /// Enable profiling
    pub fn enable(&mut self) {
        self.enabled = true;
        self.start_time = Instant::now();
        if let Ok(mut events) = self.events.lock() {
            events.clear();
        }
    }

    /// Disable profiling
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Record an event
    pub fn record_event(&self, name: &str, category: &str, duration: Duration) -> TorshResult<()> {
        if !self.enabled {
            return Ok(());
        }

        let mut events = self.events.lock().map_err(|_| {
            TorshError::InvalidArgument("Failed to acquire lock on events".to_string())
        })?;

        let thread_id = thread::current().id();
        let thread_id_num = format!("{thread_id:?}").parse::<usize>().unwrap_or(0);

        let start_us = self.start_time.elapsed().as_micros() as u64;
        let duration_us = duration.as_micros() as u64;

        events.push(ProfileEvent {
            name: name.to_string(),
            category: category.to_string(),
            start_us,
            duration_us,
            thread_id: thread_id_num,
            operation_count: None,
            flops: None,
            bytes_transferred: None,
            stack_trace: None,
        });

        Ok(())
    }

    /// Get all recorded events
    pub fn get_events(&self) -> TorshResult<Vec<ProfileEvent>> {
        let events = self.events.lock().map_err(|_| {
            TorshError::InvalidArgument("Failed to acquire lock on events".to_string())
        })?;
        Ok(events.clone())
    }

    /// Clear all events
    pub fn clear(&self) -> TorshResult<()> {
        let mut events = self.events.lock().map_err(|_| {
            TorshError::InvalidArgument("Failed to acquire lock on events".to_string())
        })?;
        events.clear();
        Ok(())
    }
}

/// RAII guard for automatic profiling scope
pub struct ProfileScope {
    profiler: Arc<CpuProfiler>,
    name: String,
    category: String,
    start: Instant,
}

impl ProfileScope {
    /// Create a new profile scope
    pub fn new(profiler: Arc<CpuProfiler>, name: &str, category: &str) -> Self {
        Self {
            profiler,
            name: name.to_string(),
            category: category.to_string(),
            start: Instant::now(),
        }
    }

    /// Create a new profile scope with a simple name and category (for backwards compatibility)
    pub fn simple(name: String, category: String) -> Self {
        // Create a default CpuProfiler for this scope
        let mut cpu_profiler = CpuProfiler::new();
        cpu_profiler.enable();
        let profiler = Arc::new(cpu_profiler);

        Self {
            profiler,
            name,
            category,
            start: Instant::now(),
        }
    }
}

impl Drop for ProfileScope {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        let _ = self
            .profiler
            .record_event(&self.name, &self.category, duration);
    }
}

/// Get CPU usage statistics
pub fn get_cpu_usage() -> TorshResult<f64> {
    // This is a simplified implementation
    // In a real implementation, you would read from /proc/stat or use platform-specific APIs
    Ok(0.0) // Placeholder
}

/// Get CPU frequency information
pub fn get_cpu_frequency() -> TorshResult<u64> {
    // Placeholder implementation
    Ok(2400000000) // 2.4 GHz
}

/// Profile CPU operations
pub fn profile_cpu() -> TorshResult<Vec<ProfileEvent>> {
    let mut profiler = CpuProfiler::new();
    profiler.enable();

    // Simulate some CPU operations
    let start = Instant::now();

    // Simulate matrix multiplication
    let _: Vec<i32> = (0..1000).map(|x| x * x).collect();
    let duration = start.elapsed();

    profiler.record_event("matrix_mul", "compute", duration)?;

    profiler.get_events()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_profiler_creation() {
        let profiler = CpuProfiler::new();
        assert!(!profiler.enabled);
    }

    #[test]
    fn test_cpu_profiler_enable_disable() {
        let mut profiler = CpuProfiler::new();
        profiler.enable();
        assert!(profiler.enabled);

        profiler.disable();
        assert!(!profiler.enabled);
    }

    #[test]
    fn test_cpu_profiler_record_event() {
        let mut profiler = CpuProfiler::new();
        profiler.enable();

        let duration = Duration::from_millis(10);
        profiler
            .record_event("test_event", "test_category", duration)
            .unwrap();

        let events = profiler.get_events().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "test_event");
        assert_eq!(events[0].category, "test_category");
    }

    #[test]
    fn test_profile_scope() {
        let mut profiler = CpuProfiler::new();
        profiler.enable();
        let profiler_arc = Arc::new(profiler);

        {
            let _scope = ProfileScope::new(profiler_arc.clone(), "test_scope", "test");
            // Simulate some work
            std::thread::sleep(Duration::from_millis(1));
        }

        let events = profiler_arc.get_events().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "test_scope");
    }
}
