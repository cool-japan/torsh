//! # DatasetProfileStats - Trait Implementations
//!
//! This module contains trait implementations for `DatasetProfileStats`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::DatasetProfileStats;

#[cfg(feature = "std")]
impl std::fmt::Display for DatasetProfileStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Dataset Profile Statistics:")?;
        writeln!(f, "  Total Accesses: {}", self.total_accesses)?;
        writeln!(
            f,
            "  Sequential Accesses: {} ({:.1}%)",
            self.sequential_accesses,
            self.sequential_ratio * 100.0
        )?;
        writeln!(
            f,
            "  Avg Access Time: {:.2} µs ({:.3} ms)",
            self.avg_access_time_us,
            self.avg_access_time_us / 1000.0
        )?;
        writeln!(
            f,
            "  Throughput: {:.1} accesses/sec",
            self.throughput_accesses_per_sec
        )?;
        writeln!(f, "  Elapsed Time: {:.2} seconds", self.elapsed_seconds)?;
        Ok(())
    }
}
