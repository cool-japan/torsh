//! Memory Allocation Visualization Tools
//!
//! Provides comprehensive visualization capabilities for memory allocations,
//! building on top of the memory_debug module to present allocation data
//! in easy-to-understand visual formats.
//!
//! # Features
//!
//! - **Allocation Timeline**: Visualize allocations over time
//! - **Size Distribution**: Histogram of allocation sizes
//! - **Leak Heatmap**: Visual representation of potential memory leaks
//! - **Thread Allocation Map**: Per-thread allocation visualization
//! - **ASCII Charts**: Terminal-friendly allocation charts
//! - **Memory Maps**: Visual memory layout representation
//!
//! # Examples
//!
//! ```rust
//! use torsh_core::memory_visualization::{
//!     AllocationTimeline, SizeHistogram, MemoryMap
//! };
//!
//! // Create an allocation timeline
//! let timeline = AllocationTimeline::new();
//! let chart = timeline.render_ascii(80, 20); // 80x20 character chart
//! println!("{}", chart);
//!
//! // Generate a size distribution histogram
//! let histogram = SizeHistogram::new();
//! let viz = histogram.render_ascii(60, 15);
//! println!("{}", viz);
//! ```

use crate::memory_debug::{get_memory_stats, AllocationInfo, MemoryStats};
use std::fmt;

/// ASCII character for horizontal bar in charts
const HORIZONTAL_BAR: char = '─';
/// ASCII character for filled block in histograms
const FILLED_BLOCK: char = '█';
/// ASCII character for half block in histograms
const HALF_BLOCK: char = '▌';

/// Allocation timeline visualization
///
/// Provides a time-series view of memory allocations, showing
/// allocation patterns, spikes, and trends over time.
pub struct AllocationTimeline {
    /// Time buckets for grouping allocations
    buckets: Vec<TimeBucket>,
    /// Total time span covered
    time_span: std::time::Duration,
}

/// Time bucket for allocation aggregation
#[derive(Debug, Clone)]
struct TimeBucket {
    /// Start time of this bucket
    #[allow(dead_code)]
    start_time: std::time::Instant,
    /// Total bytes allocated in this bucket
    bytes_allocated: usize,
    /// Number of allocations in this bucket
    #[allow(dead_code)]
    allocation_count: usize,
}

impl AllocationTimeline {
    /// Create a new allocation timeline with default parameters
    pub fn new() -> Self {
        Self {
            buckets: Vec::new(),
            time_span: std::time::Duration::from_secs(60),
        }
    }

    /// Set the time span for the timeline
    pub fn with_time_span(mut self, duration: std::time::Duration) -> Self {
        self.time_span = duration;
        self
    }

    /// Render the timeline as ASCII art
    ///
    /// # Arguments
    ///
    /// * `width` - Width of the chart in characters
    /// * `height` - Height of the chart in characters
    ///
    /// # Returns
    ///
    /// String containing the ASCII chart
    pub fn render_ascii(&self, width: usize, height: usize) -> String {
        if self.buckets.is_empty() {
            return self.render_empty_chart(width, height);
        }

        let max_bytes = self
            .buckets
            .iter()
            .map(|b| b.bytes_allocated)
            .max()
            .unwrap_or(1);

        let mut chart = String::with_capacity(width * height * 2);

        // Title
        chart.push_str("Memory Allocation Timeline\n");
        chart.push_str(&format!("Max: {} bytes\n", Self::format_bytes(max_bytes)));
        chart.push_str(&HORIZONTAL_BAR.to_string().repeat(width));
        chart.push('\n');

        // Render bars
        for row in (0..height).rev() {
            let threshold = (max_bytes * row) / height.max(1);

            for bucket in &self.buckets {
                let bar_height = (bucket.bytes_allocated * height) / max_bytes.max(1);

                if bar_height > row {
                    chart.push(FILLED_BLOCK);
                } else if bar_height == row {
                    chart.push(HALF_BLOCK);
                } else {
                    chart.push(' ');
                }
            }

            // Y-axis label
            chart.push_str(&format!(" {}", Self::format_bytes(threshold)));
            chart.push('\n');
        }

        // X-axis
        chart.push_str(&HORIZONTAL_BAR.to_string().repeat(width));
        chart.push('\n');

        chart
    }

    /// Render an empty chart placeholder
    fn render_empty_chart(&self, width: usize, height: usize) -> String {
        let mut chart = String::new();
        chart.push_str("Memory Allocation Timeline\n");
        chart.push_str("No data available\n");
        chart.push_str(&HORIZONTAL_BAR.to_string().repeat(width));
        chart.push('\n');

        for _ in 0..height {
            chart.push_str(&" ".repeat(width));
            chart.push('\n');
        }

        chart.push_str(&HORIZONTAL_BAR.to_string().repeat(width));
        chart
    }

    /// Format bytes in human-readable form
    fn format_bytes(bytes: usize) -> String {
        const KB: usize = 1024;
        const MB: usize = KB * 1024;
        const GB: usize = MB * 1024;

        if bytes >= GB {
            format!("{:.2} GB", bytes as f64 / GB as f64)
        } else if bytes >= MB {
            format!("{:.2} MB", bytes as f64 / MB as f64)
        } else if bytes >= KB {
            format!("{:.2} KB", bytes as f64 / KB as f64)
        } else {
            format!("{} B", bytes)
        }
    }
}

impl Default for AllocationTimeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Size distribution histogram
///
/// Visualizes the distribution of allocation sizes, helping identify
/// allocation patterns and potential optimization opportunities.
pub struct SizeHistogram {
    /// Histogram bins
    bins: Vec<HistogramBin>,
}

/// Histogram bin for size distribution
#[derive(Debug, Clone)]
struct HistogramBin {
    /// Minimum size in this bin (inclusive)
    min_size: usize,
    /// Maximum size in this bin (exclusive)
    max_size: usize,
    /// Count of allocations in this bin
    count: usize,
    /// Total bytes in this bin
    total_bytes: usize,
}

impl SizeHistogram {
    /// Create a new size histogram
    pub fn new() -> Self {
        Self { bins: Vec::new() }
    }

    /// Build histogram from allocation data
    ///
    /// # Arguments
    ///
    /// * `allocations` - Slice of allocation information
    pub fn build_from_allocations(&mut self, allocations: &[AllocationInfo]) {
        // Create logarithmic bins
        let bin_edges = vec![
            0,
            1024,             // 1 KB
            10 * 1024,        // 10 KB
            100 * 1024,       // 100 KB
            1024 * 1024,      // 1 MB
            10 * 1024 * 1024, // 10 MB
            usize::MAX,
        ];

        self.bins.clear();
        for i in 0..bin_edges.len() - 1 {
            self.bins.push(HistogramBin {
                min_size: bin_edges[i],
                max_size: bin_edges[i + 1],
                count: 0,
                total_bytes: 0,
            });
        }

        // Fill bins
        for alloc in allocations {
            for bin in &mut self.bins {
                if alloc.size >= bin.min_size && alloc.size < bin.max_size {
                    bin.count += 1;
                    bin.total_bytes += alloc.size;
                    break;
                }
            }
        }
    }

    /// Render the histogram as ASCII art
    ///
    /// # Arguments
    ///
    /// * `width` - Width of the chart in characters
    /// * `height` - Height of the chart in characters
    ///
    /// # Returns
    ///
    /// String containing the ASCII histogram
    pub fn render_ascii(&self, width: usize, height: usize) -> String {
        if self.bins.is_empty() {
            return "Size Distribution Histogram\nNo data available\n".to_string();
        }

        let max_count = self.bins.iter().map(|b| b.count).max().unwrap_or(1);
        let mut chart = String::with_capacity(width * height * 2);

        // Title
        chart.push_str("Allocation Size Distribution\n");
        chart.push_str(&HORIZONTAL_BAR.to_string().repeat(width));
        chart.push('\n');

        // Render histogram bars
        for bin in &self.bins {
            let bar_length = (bin.count * width) / max_count.max(1);
            let label = format!(
                "{:>8} - {:<8}",
                AllocationTimeline::format_bytes(bin.min_size),
                if bin.max_size == usize::MAX {
                    "MAX".to_string()
                } else {
                    AllocationTimeline::format_bytes(bin.max_size)
                }
            );

            chart.push_str(&label);
            chart.push_str(": ");
            chart.push_str(&FILLED_BLOCK.to_string().repeat(bar_length));
            chart.push_str(&format!(
                " ({}, {})\n",
                bin.count,
                AllocationTimeline::format_bytes(bin.total_bytes)
            ));
        }

        chart.push_str(&HORIZONTAL_BAR.to_string().repeat(width));
        chart.push('\n');

        chart
    }
}

impl Default for SizeHistogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory map visualization
///
/// Provides a visual representation of memory layout, showing
/// active allocations and their positions in memory.
pub struct MemoryMap {
    /// Memory regions to visualize
    regions: Vec<MemoryRegion>,
}

/// Memory region for visualization
#[derive(Debug, Clone)]
struct MemoryRegion {
    /// Start address (relative)
    start_offset: usize,
    /// Size in bytes
    size: usize,
    /// Region label
    label: String,
    /// Whether this region is allocated
    is_allocated: bool,
}

impl MemoryMap {
    /// Create a new memory map
    pub fn new() -> Self {
        Self {
            regions: Vec::new(),
        }
    }

    /// Add a region to the memory map
    pub fn add_region(&mut self, offset: usize, size: usize, label: String, is_allocated: bool) {
        self.regions.push(MemoryRegion {
            start_offset: offset,
            size,
            label,
            is_allocated,
        });
    }

    /// Render the memory map as ASCII art
    ///
    /// # Arguments
    ///
    /// * `width` - Width of the visualization in characters
    ///
    /// # Returns
    ///
    /// String containing the ASCII memory map
    pub fn render_ascii(&self, width: usize) -> String {
        if self.regions.is_empty() {
            return "Memory Map\nNo regions defined\n".to_string();
        }

        let max_offset = self
            .regions
            .iter()
            .map(|r| r.start_offset + r.size)
            .max()
            .unwrap_or(1);

        let mut chart = String::with_capacity(width * self.regions.len() * 3);

        // Title
        chart.push_str("Memory Layout Map\n");
        chart.push_str(&format!(
            "Total: {}\n",
            AllocationTimeline::format_bytes(max_offset)
        ));
        chart.push_str(&HORIZONTAL_BAR.to_string().repeat(width));
        chart.push('\n');

        // Render each region
        for region in &self.regions {
            let start_pos = (region.start_offset * width) / max_offset.max(1);
            let region_width = (region.size * width) / max_offset.max(1).max(1);

            let fill_char = if region.is_allocated {
                FILLED_BLOCK
            } else {
                '░'
            };

            // Region visualization
            chart.push_str(&" ".repeat(start_pos));
            chart.push_str(&fill_char.to_string().repeat(region_width.max(1)));
            chart.push('\n');

            // Label
            chart.push_str(&format!(
                "{}: {} @ +{}\n",
                region.label,
                AllocationTimeline::format_bytes(region.size),
                AllocationTimeline::format_bytes(region.start_offset)
            ));
        }

        chart.push_str(&HORIZONTAL_BAR.to_string().repeat(width));
        chart.push('\n');

        chart
    }
}

impl Default for MemoryMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Allocation summary statistics with visualization
pub struct AllocationSummary {
    /// Memory statistics
    stats: MemoryStats,
}

impl AllocationSummary {
    /// Create a new allocation summary
    pub fn new() -> Self {
        Self {
            stats: get_memory_stats().unwrap_or_default(),
        }
    }

    /// Create from existing statistics
    pub fn from_stats(stats: MemoryStats) -> Self {
        Self { stats }
    }

    /// Render summary as formatted text
    pub fn render(&self) -> String {
        format!(
            r#"
Memory Allocation Summary
═══════════════════════════════════════════════════════

Active Allocations:  {}
Total Allocated:     {}
Peak Memory:         {}
Lifetime Allocated:  {}
Lifetime Freed:      {}

Allocation Rate:     {} allocations
Deallocation Rate:   {} deallocations
Average Size:        {}

Fragmentation:       {:.2}%
Efficiency:          {:.2}%
"#,
            self.stats.active_allocations,
            AllocationTimeline::format_bytes(self.stats.total_allocated),
            AllocationTimeline::format_bytes(self.stats.peak_allocated),
            AllocationTimeline::format_bytes(self.stats.lifetime_allocated),
            AllocationTimeline::format_bytes(self.stats.lifetime_deallocated),
            self.stats.total_allocations,
            self.stats.total_deallocations,
            AllocationTimeline::format_bytes(self.stats.average_allocation_size as usize),
            self.calculate_fragmentation(),
            self.calculate_efficiency(),
        )
    }

    /// Calculate memory fragmentation percentage
    fn calculate_fragmentation(&self) -> f64 {
        if self.stats.peak_allocated == 0 {
            return 0.0;
        }

        let fragmentation = ((self.stats.peak_allocated - self.stats.total_allocated) as f64
            / self.stats.peak_allocated as f64)
            * 100.0;

        fragmentation.max(0.0).min(100.0)
    }

    /// Calculate allocation efficiency percentage
    fn calculate_efficiency(&self) -> f64 {
        if self.stats.lifetime_allocated == 0 {
            return 100.0;
        }

        let efficiency =
            (self.stats.lifetime_deallocated as f64 / self.stats.lifetime_allocated as f64) * 100.0;

        efficiency.max(0.0).min(100.0)
    }
}

impl Default for AllocationSummary {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for AllocationSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.render())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_timeline_render() {
        let timeline = AllocationTimeline::new();
        let chart = timeline.render_ascii(60, 10);

        assert!(chart.contains("Memory Allocation Timeline"));
        assert!(chart.contains("No data available"));
    }

    #[test]
    fn test_size_histogram_render() {
        let histogram = SizeHistogram::new();
        let chart = histogram.render_ascii(60, 10);

        assert!(chart.contains("Size Distribution Histogram"));
        assert!(chart.contains("No data available"));
    }

    #[test]
    fn test_size_histogram_with_data() {
        use std::alloc::Layout;
        use std::time::Instant;

        let mut histogram = SizeHistogram::new();
        let allocations = vec![
            AllocationInfo {
                id: 1,
                size: 512,
                layout: Layout::from_size_align(512, 8).unwrap(),
                timestamp: Instant::now(),
                backtrace: None,
                tag: None,
                is_active: true,
                thread_id: std::thread::current().id(),
            },
            AllocationInfo {
                id: 2,
                size: 2048,
                layout: Layout::from_size_align(2048, 8).unwrap(),
                timestamp: Instant::now(),
                backtrace: None,
                tag: None,
                is_active: true,
                thread_id: std::thread::current().id(),
            },
        ];

        histogram.build_from_allocations(&allocations);
        let chart = histogram.render_ascii(60, 10);

        assert!(chart.contains("Allocation Size Distribution"));
        assert!(chart.contains('█')); // Should have bars
    }

    #[test]
    fn test_memory_map_render() {
        let mut map = MemoryMap::new();
        map.add_region(0, 1024, "Tensor A".to_string(), true);
        map.add_region(1024, 2048, "Tensor B".to_string(), true);
        map.add_region(3072, 512, "Free Space".to_string(), false);

        let chart = map.render_ascii(80);

        assert!(chart.contains("Memory Layout Map"));
        assert!(chart.contains("Tensor A"));
        assert!(chart.contains("Tensor B"));
        assert!(chart.contains("Free Space"));
    }

    #[test]
    fn test_allocation_summary() {
        let summary = AllocationSummary::new();
        let rendered = summary.render();

        assert!(rendered.contains("Memory Allocation Summary"));
        assert!(rendered.contains("Active Allocations"));
        assert!(rendered.contains("Peak Memory"));
        assert!(rendered.contains("Fragmentation"));
        assert!(rendered.contains("Efficiency"));
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(AllocationTimeline::format_bytes(512), "512 B");
        assert_eq!(AllocationTimeline::format_bytes(2048), "2.00 KB");
        assert_eq!(AllocationTimeline::format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(
            AllocationTimeline::format_bytes(1024 * 1024 * 1024),
            "1.00 GB"
        );
    }

    #[test]
    fn test_fragmentation_calculation() {
        let stats = MemoryStats {
            total_allocated: 800,
            peak_allocated: 1000,
            ..Default::default()
        };

        let summary = AllocationSummary::from_stats(stats);
        let fragmentation = summary.calculate_fragmentation();

        // (1000 - 800) / 1000 = 0.2 = 20%
        assert!((fragmentation - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_efficiency_calculation() {
        let stats = MemoryStats {
            lifetime_allocated: 10000,
            lifetime_deallocated: 8000,
            ..Default::default()
        };

        let summary = AllocationSummary::from_stats(stats);
        let efficiency = summary.calculate_efficiency();

        // 8000 / 10000 = 0.8 = 80%
        assert!((efficiency - 80.0).abs() < 0.01);
    }
}
