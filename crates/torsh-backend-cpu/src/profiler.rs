//! CPU Profiler Implementation

use torsh_backends::profiler::{EventId, EventType};
use torsh_backends::{Profiler, ProfilerEvent, ProfilerStats};
use torsh_core::error::Result;

#[cfg(feature = "std")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "std")]
#[allow(unused_imports)]
use std::time::{Duration, Instant};

#[cfg(not(feature = "std"))]
use alloc::sync::Arc;
#[cfg(not(feature = "std"))]
use spin::Mutex;

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

/// CPU profiler implementation
#[derive(Debug)]
pub struct CpuProfiler {
    events: Arc<Mutex<Vec<ProfilerEvent>>>,
    stats: Arc<Mutex<ProfilerStats>>,
    enabled: Arc<Mutex<bool>>,
    next_event_id: Arc<Mutex<u64>>,
}

impl CpuProfiler {
    /// Create a new CPU profiler
    pub fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(Mutex::new(ProfilerStats::default())),
            enabled: Arc::new(Mutex::new(false)),
            next_event_id: Arc::new(Mutex::new(1)),
        }
    }

    fn next_id(&self) -> EventId {
        let mut id = self.next_event_id.lock().unwrap();
        let event_id = EventId(*id);
        *id += 1;
        event_id
    }
}

impl Profiler for CpuProfiler {
    fn start(&mut self) -> Result<()> {
        let mut enabled = self.enabled.lock().unwrap();
        *enabled = true;

        let mut events = self.events.lock().unwrap();
        events.clear();

        let mut stats = self.stats.lock().unwrap();
        *stats = ProfilerStats::default();

        Ok(())
    }

    fn stop(&mut self) -> Result<()> {
        let mut enabled = self.enabled.lock().unwrap();
        *enabled = false;
        Ok(())
    }

    fn is_enabled(&self) -> bool {
        *self.enabled.lock().unwrap()
    }

    fn begin_event(&mut self, name: &str) -> Result<EventId> {
        if !self.is_enabled() {
            return Ok(EventId(0));
        }

        let event_id = self.next_id();
        let event = ProfilerEvent::new(
            event_id,
            name.to_string(),
            EventType::Custom(name.to_string()),
        );

        let mut events = self.events.lock().unwrap();
        events.push(event);

        Ok(event_id)
    }

    fn end_event(&mut self, event_id: EventId) -> Result<()> {
        if !self.is_enabled() || event_id.0 == 0 {
            return Ok(());
        }

        let mut events = self.events.lock().unwrap();
        if let Some(event) = events.iter_mut().find(|e| e.id == event_id) {
            event.finish();
        }

        Ok(())
    }

    fn marker(&mut self, name: &str) -> Result<()> {
        if !self.is_enabled() {
            return Ok(());
        }

        let event_id = self.next_id();
        let mut event = ProfilerEvent::new(event_id, name.to_string(), EventType::Marker);
        event.finish();

        let mut events = self.events.lock().unwrap();
        events.push(event);

        Ok(())
    }

    fn stats(&self) -> ProfilerStats {
        self.stats.lock().unwrap().clone()
    }

    fn events(&self) -> &[ProfilerEvent] {
        // This is a limitation of the current design - we can't return a reference
        // to data behind a mutex. In practice, this method would need to be redesigned
        // or we'd need to use a different synchronization mechanism
        &[]
    }

    fn clear(&mut self) {
        let mut events = self.events.lock().unwrap();
        events.clear();

        let mut stats = self.stats.lock().unwrap();
        *stats = ProfilerStats::default();
    }

    fn report(&self) -> String {
        let events = self.events.lock().unwrap();
        let stats = self.stats.lock().unwrap();

        let mut report = String::new();
        report.push_str("=== CPU Profiler Report ===\n");
        report.push_str(&format!("Total Events: {}\n", stats.total_events));
        report.push_str(&format!(
            "Total Time: {:.2}ms\n",
            stats.total_time.as_secs_f64() * 1000.0
        ));

        report.push_str("\n=== Events ===\n");
        for event in events.iter() {
            if let Some(duration) = event.duration() {
                report.push_str(&format!(
                    "{}: {:.2}μs\n",
                    event.name,
                    duration.as_secs_f64() * 1_000_000.0
                ));
            } else {
                report.push_str(&format!("{}: (marker)\n", event.name));
            }
        }

        report
    }
}

impl Default for CpuProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for CpuProfiler {
    fn clone(&self) -> Self {
        Self {
            events: Arc::clone(&self.events),
            stats: Arc::clone(&self.stats),
            enabled: Arc::clone(&self.enabled),
            next_event_id: Arc::clone(&self.next_event_id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_profiler_basic() {
        let mut profiler = CpuProfiler::new();
        assert!(!profiler.is_enabled());

        profiler.start().unwrap();
        assert!(profiler.is_enabled());

        let event_id = profiler.begin_event("test_event").unwrap();
        profiler.end_event(event_id).unwrap();

        profiler.marker("test_marker").unwrap();

        profiler.stop().unwrap();
        assert!(!profiler.is_enabled());

        let report = profiler.report();
        assert!(report.contains("test_event"));
        assert!(report.contains("test_marker"));
    }

    #[test]
    fn test_cpu_profiler_disabled() {
        let mut profiler = CpuProfiler::new();

        // Operations should be no-ops when disabled
        let event_id = profiler.begin_event("test").unwrap();
        assert_eq!(event_id.0, 0);

        profiler.end_event(event_id).unwrap();
        profiler.marker("test_marker").unwrap();

        let report = profiler.report();
        assert!(!report.contains("test"));
    }

    #[test]
    fn test_cpu_profiler_clear() {
        let mut profiler = CpuProfiler::new();
        profiler.start().unwrap();

        profiler.begin_event("test").unwrap();
        profiler.clear();

        let report = profiler.report();
        assert!(!report.contains("test"));
    }
}
