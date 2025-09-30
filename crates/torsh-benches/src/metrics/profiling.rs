//! Performance profiling and analysis
//!
//! This module provides detailed performance profiling capabilities including
//! event tracking, Chrome tracing integration, and statistical analysis.

use std::time::{Duration, Instant};

/// Performance profiler for detailed analysis
pub struct PerformanceProfiler {
    events: Vec<ProfileEvent>,
    current_stack: Vec<String>,
    start_time: Instant,
}

impl PerformanceProfiler {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            current_stack: Vec::new(),
            start_time: Instant::now(),
        }
    }

    /// Begin a profiling event
    pub fn begin_event(&mut self, name: &str) {
        let event = ProfileEvent {
            name: name.to_string(),
            event_type: ProfileEventType::Begin,
            timestamp: self.start_time.elapsed(),
            thread_id: std::thread::current().id(),
            stack_depth: self.current_stack.len(),
        };

        self.current_stack.push(name.to_string());
        self.events.push(event);
    }

    /// End a profiling event
    pub fn end_event(&mut self, name: &str) {
        if let Some(stack_name) = self.current_stack.pop() {
            assert_eq!(stack_name, name, "Mismatched profiling events");
        }

        let event = ProfileEvent {
            name: name.to_string(),
            event_type: ProfileEventType::End,
            timestamp: self.start_time.elapsed(),
            thread_id: std::thread::current().id(),
            stack_depth: self.current_stack.len(),
        };

        self.events.push(event);
    }

    /// Add an instant marker
    pub fn marker(&mut self, name: &str) {
        let event = ProfileEvent {
            name: name.to_string(),
            event_type: ProfileEventType::Marker,
            timestamp: self.start_time.elapsed(),
            thread_id: std::thread::current().id(),
            stack_depth: self.current_stack.len(),
        };

        self.events.push(event);
    }

    /// Generate Chrome tracing format output
    pub fn export_chrome_trace(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        writeln!(file, "{{")?;
        writeln!(file, "  \"traceEvents\": [")?;

        for (i, event) in self.events.iter().enumerate() {
            let phase = match event.event_type {
                ProfileEventType::Begin => "B",
                ProfileEventType::End => "E",
                ProfileEventType::Marker => "i",
            };

            writeln!(
                file,
                "    {{\"name\": \"{}\", \"ph\": \"{}\", \"ts\": {}, \"pid\": 1, \"tid\": {:?}}}{}",
                event.name,
                phase,
                event.timestamp.as_micros(),
                event.thread_id,
                if i < self.events.len() - 1 { "," } else { "" }
            )?;
        }

        writeln!(file, "  ]")?;
        writeln!(file, "}}")?;

        Ok(())
    }

    /// Generate performance report
    pub fn generate_report(&self) -> PerformanceReport {
        let mut durations = std::collections::HashMap::new();
        let mut stack = Vec::new();

        for event in &self.events {
            match event.event_type {
                ProfileEventType::Begin => {
                    stack.push((event.name.clone(), event.timestamp));
                }
                ProfileEventType::End => {
                    if let Some((name, start_time)) = stack.pop() {
                        assert_eq!(name, event.name);
                        let duration = event.timestamp - start_time;
                        durations
                            .entry(name)
                            .or_insert_with(Vec::new)
                            .push(duration);
                    }
                }
                ProfileEventType::Marker => {} // Instant events don't have duration
            }
        }

        let mut function_stats = Vec::new();
        for (name, times) in durations {
            let total_time: Duration = times.iter().sum();
            let avg_time = total_time / times.len() as u32;
            let min_time = times.iter().min().copied().unwrap_or_default();
            let max_time = times.iter().max().copied().unwrap_or_default();

            function_stats.push(FunctionStats {
                name,
                call_count: times.len(),
                total_time,
                average_time: avg_time,
                min_time,
                max_time,
            });
        }

        // Sort by total time descending
        function_stats.sort_by(|a, b| b.total_time.cmp(&a.total_time));

        PerformanceReport {
            function_stats,
            total_events: self.events.len(),
        }
    }
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Profiling event
#[derive(Debug, Clone)]
pub struct ProfileEvent {
    pub name: String,
    pub event_type: ProfileEventType,
    pub timestamp: Duration,
    pub thread_id: std::thread::ThreadId,
    pub stack_depth: usize,
}

/// Profiling event type
#[derive(Debug, Clone, Copy)]
pub enum ProfileEventType {
    Begin,
    End,
    Marker,
}

/// Performance analysis report
#[derive(Debug)]
pub struct PerformanceReport {
    pub function_stats: Vec<FunctionStats>,
    pub total_events: usize,
}

/// Statistics for a profiled function
#[derive(Debug)]
pub struct FunctionStats {
    pub name: String,
    pub call_count: usize,
    pub total_time: Duration,
    pub average_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
}

/// Scoped profiler that automatically ends when dropped
pub struct ScopedProfiler<'a> {
    profiler: &'a mut PerformanceProfiler,
    name: String,
}

impl<'a> ScopedProfiler<'a> {
    pub fn new(profiler: &'a mut PerformanceProfiler, name: &str) -> Self {
        profiler.begin_event(name);
        Self {
            profiler,
            name: name.to_string(),
        }
    }
}

impl<'a> Drop for ScopedProfiler<'a> {
    fn drop(&mut self) {
        self.profiler.end_event(&self.name);
    }
}
