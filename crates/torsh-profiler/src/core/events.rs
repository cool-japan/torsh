//! Core event tracking and management

use std::time::Instant;

/// Profiler context
#[derive(Debug, Clone)]
pub struct ProfilerContext {
    pub profiler_name: String,
    pub start_time: Instant,
    pub enabled: bool,
    pub overhead_tracking: bool,
    pub trace_collection: bool,
    pub memory_tracking: bool,
    pub thread_safe: bool,
}

impl Default for ProfilerContext {
    fn default() -> Self {
        Self {
            profiler_name: "default".to_string(),
            start_time: Instant::now(),
            enabled: true,
            overhead_tracking: false,
            trace_collection: false,
            memory_tracking: false,
            thread_safe: true,
        }
    }
}

/// Core profiling event
#[derive(Debug, Clone)]
pub struct ProfilingEvent {
    pub name: String,
    pub category: String,
    pub thread_id: usize,
    pub start_time: Instant,
    pub duration: Option<std::time::Duration>,
    pub metadata: std::collections::HashMap<String, String>,
}

impl ProfilingEvent {
    pub fn new(name: String, category: String) -> Self {
        Self {
            name,
            category,
            thread_id: 0, // Simplified for now - could use a thread-local counter
            start_time: Instant::now(),
            duration: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    pub fn finish(&mut self) {
        self.duration = Some(self.start_time.elapsed());
    }

    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
}

/// Event collection and management
pub struct EventCollector {
    events: Vec<ProfilingEvent>,
    enabled: bool,
    max_events: usize,
}

impl EventCollector {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            enabled: true,
            max_events: 100_000,
        }
    }

    pub fn add_event(&mut self, event: ProfilingEvent) {
        if !self.enabled {
            return;
        }

        if self.events.len() >= self.max_events {
            self.events.remove(0); // Remove oldest event
        }

        self.events.push(event);
    }

    pub fn get_events(&self) -> &[ProfilingEvent] {
        &self.events
    }

    pub fn clear(&mut self) {
        self.events.clear();
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn set_max_events(&mut self, max_events: usize) {
        self.max_events = max_events;
    }
}

impl Default for EventCollector {
    fn default() -> Self {
        Self::new()
    }
}
