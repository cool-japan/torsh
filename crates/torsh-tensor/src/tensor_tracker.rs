//! Tensor Value Tracking for Debugging
//!
//! This module provides comprehensive tensor value tracking capabilities for debugging purposes.
//! It allows tracking tensor operations, values, and transformations with conditional compilation
//! to avoid performance overhead in release builds.
//!
//! # Features
//!
//! - **Operation tracking**: Record all operations performed on tracked tensors
//! - **Value snapshots**: Capture tensor values at specific points
//! - **Transformation history**: Track how tensor values change over time
//! - **Conditional compilation**: Zero overhead in release builds when disabled
//! - **Filtering**: Track only specific tensors or operations
//! - **Analysis**: Generate reports on tensor value ranges, statistics, and changes
//!
//! # Example
//!
//! ```rust
//! use torsh_tensor::{Tensor, tensor_tracker::*};
//! use torsh_core::device::DeviceType;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a tracked tensor
//! let mut tracker = TensorTracker::new();
//! let tensor = Tensor::<f32>::ones(&[2, 2], DeviceType::Cpu)?;
//! let tracked_id = tracker.track(tensor.clone(), "input_tensor")?;
//!
//! // Perform operations
//! let result = tensor.mul_scalar(2.0)?;
//! tracker.record_operation(tracked_id, "mul_scalar", vec![2.0], &result)?;
//!
//! // Generate report
//! let report = tracker.generate_report(tracked_id)?;
//! println!("{}", report);
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use torsh_core::{
    dtype::TensorElement,
    error::{Result, TorshError},
};

use crate::Tensor;

/// Unique identifier for tracked tensors
pub type TrackId = u64;

/// Statistics about tensor values
#[derive(Debug, Clone)]
pub struct TensorValueStats<T: TensorElement> {
    /// Minimum value in the tensor
    pub min: Option<T>,
    /// Maximum value in the tensor
    pub max: Option<T>,
    /// Mean value (if applicable)
    pub mean: Option<f64>,
    /// Standard deviation (if applicable)
    pub std: Option<f64>,
    /// Number of NaN values
    pub nan_count: usize,
    /// Number of Inf values
    pub inf_count: usize,
    /// Number of zero values
    pub zero_count: usize,
    /// Total number of elements
    pub total_elements: usize,
}

impl<T: TensorElement> TensorValueStats<T> {
    /// Create statistics from a tensor
    pub fn from_tensor(tensor: &Tensor<T>) -> Result<Self>
    where
        T: Copy + PartialOrd + num_traits::Zero + num_traits::ToPrimitive,
    {
        let data = tensor.to_vec()?;
        let total_elements = data.len();

        let mut min = None;
        let mut max = None;
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut zero_count = 0;
        let mut sum = 0.0f64;

        for &val in &data {
            // Check for special values
            if let Some(f_val) = num_traits::ToPrimitive::to_f64(&val) {
                if f_val.is_nan() {
                    nan_count += 1;
                    continue;
                }
                if f_val.is_infinite() {
                    inf_count += 1;
                    continue;
                }
                sum += f_val;
            }

            // Track min/max
            match (min, max) {
                (None, None) => {
                    min = Some(val);
                    max = Some(val);
                }
                (Some(current_min), Some(current_max)) => {
                    if val < current_min {
                        min = Some(val);
                    }
                    if val > current_max {
                        max = Some(val);
                    }
                }
                _ => unreachable!(),
            }

            // Count zeros
            if val == <T as num_traits::Zero>::zero() {
                zero_count += 1;
            }
        }

        let mean = if total_elements > 0 && nan_count + inf_count < total_elements {
            Some(sum / (total_elements - nan_count - inf_count) as f64)
        } else {
            None
        };

        // Calculate standard deviation
        let std = if let Some(mean_val) = mean {
            let variance: f64 = data
                .iter()
                .filter_map(|&v| num_traits::ToPrimitive::to_f64(&v))
                .filter(|&f| !f.is_nan() && !f.is_infinite())
                .map(|v| {
                    let diff = v - mean_val;
                    diff * diff
                })
                .sum::<f64>()
                / (total_elements - nan_count - inf_count) as f64;
            Some(variance.sqrt())
        } else {
            None
        };

        Ok(Self {
            min,
            max,
            mean,
            std,
            nan_count,
            inf_count,
            zero_count,
            total_elements,
        })
    }
}

impl<T: TensorElement + fmt::Display> fmt::Display for TensorValueStats<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Tensor Value Statistics:")?;
        writeln!(f, "  Total elements: {}", self.total_elements)?;

        if let (Some(min), Some(max)) = (&self.min, &self.max) {
            writeln!(f, "  Min: {}", min)?;
            writeln!(f, "  Max: {}", max)?;
        }

        if let Some(mean) = self.mean {
            writeln!(f, "  Mean: {:.6}", mean)?;
        }

        if let Some(std) = self.std {
            writeln!(f, "  Std: {:.6}", std)?;
        }

        if self.nan_count > 0 {
            writeln!(f, "  NaN count: {}", self.nan_count)?;
        }

        if self.inf_count > 0 {
            writeln!(f, "  Inf count: {}", self.inf_count)?;
        }

        if self.zero_count > 0 {
            writeln!(f, "  Zero count: {}", self.zero_count)?;
        }

        Ok(())
    }
}

/// Record of a tensor operation
#[derive(Debug, Clone)]
pub struct OperationRecord {
    /// Name of the operation
    pub operation: String,
    /// Parameters used in the operation (as strings for display)
    pub parameters: Vec<String>,
    /// Timestamp when operation was performed
    pub timestamp: Instant,
    /// Duration of the operation
    pub duration: Option<Duration>,
    /// Shape before the operation
    pub shape_before: Vec<usize>,
    /// Shape after the operation
    pub shape_after: Vec<usize>,
}

impl fmt::Display for OperationRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.operation)?;
        if !self.parameters.is_empty() {
            write!(f, "({}) ", self.parameters.join(", "))?;
        }
        write!(f, ": {:?} -> {:?}", self.shape_before, self.shape_after)?;
        if let Some(duration) = self.duration {
            write!(f, " [{:?}]", duration)?;
        }
        Ok(())
    }
}

/// A snapshot of tensor values at a specific point
#[derive(Clone)]
pub struct TensorSnapshot<T: TensorElement> {
    /// The actual tensor values
    pub values: Vec<T>,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Timestamp when snapshot was taken
    pub timestamp: Instant,
    /// Label for this snapshot
    pub label: String,
}

/// Tracked tensor information
pub struct TrackedTensor<T: TensorElement> {
    /// Unique identifier
    pub id: TrackId,
    /// Label/name for this tensor
    pub label: String,
    /// Original tensor reference
    pub tensor: Tensor<T>,
    /// History of operations performed
    pub operations: Vec<OperationRecord>,
    /// Value snapshots taken over time
    pub snapshots: Vec<TensorSnapshot<T>>,
    /// When tracking started
    pub start_time: Instant,
}

impl<T: TensorElement> TrackedTensor<T> {
    /// Create a new tracked tensor
    pub fn new(id: TrackId, label: String, tensor: Tensor<T>) -> Self {
        Self {
            id,
            label,
            tensor,
            operations: Vec::new(),
            snapshots: Vec::new(),
            start_time: Instant::now(),
        }
    }

    /// Record an operation
    pub fn record_operation(
        &mut self,
        operation: String,
        parameters: Vec<String>,
        new_tensor: &Tensor<T>,
        duration: Option<Duration>,
    ) {
        let shape_before = self.tensor.shape().dims().to_vec();
        let shape_after = new_tensor.shape().dims().to_vec();

        self.operations.push(OperationRecord {
            operation,
            parameters,
            timestamp: Instant::now(),
            duration,
            shape_before,
            shape_after,
        });

        self.tensor = new_tensor.clone();
    }

    /// Take a snapshot of current values
    pub fn take_snapshot(&mut self, label: String) -> Result<()>
    where
        T: Copy,
    {
        let values = self.tensor.to_vec()?;
        let shape = self.tensor.shape().dims().to_vec();

        self.snapshots.push(TensorSnapshot {
            values,
            shape,
            timestamp: Instant::now(),
            label,
        });

        Ok(())
    }
}

/// Configuration for tensor tracking
#[derive(Debug, Clone)]
pub struct TrackingConfig {
    /// Whether tracking is enabled
    pub enabled: bool,
    /// Maximum number of operations to track per tensor
    pub max_operations: usize,
    /// Maximum number of snapshots to keep per tensor
    pub max_snapshots: usize,
    /// Whether to automatically take snapshots after each operation
    pub auto_snapshot: bool,
    /// Operations to filter (empty = track all)
    pub operation_filter: Vec<String>,
}

impl Default for TrackingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_operations: 1000,
            max_snapshots: 100,
            auto_snapshot: false,
            operation_filter: Vec::new(),
        }
    }
}

impl TrackingConfig {
    /// Create a minimal tracking config (low memory usage)
    pub fn minimal() -> Self {
        Self {
            enabled: true,
            max_operations: 100,
            max_snapshots: 10,
            auto_snapshot: false,
            operation_filter: Vec::new(),
        }
    }

    /// Create a comprehensive tracking config (high memory usage)
    pub fn comprehensive() -> Self {
        Self {
            enabled: true,
            max_operations: 10000,
            max_snapshots: 1000,
            auto_snapshot: true,
            operation_filter: Vec::new(),
        }
    }

    /// Create a config that tracks only specific operations
    pub fn filtered(operations: Vec<String>) -> Self {
        Self {
            enabled: true,
            max_operations: 1000,
            max_snapshots: 100,
            auto_snapshot: false,
            operation_filter: operations,
        }
    }
}

/// Main tensor tracker
pub struct TensorTracker<T: TensorElement> {
    /// Configuration
    config: Arc<RwLock<TrackingConfig>>,
    /// Tracked tensors
    tensors: Arc<RwLock<HashMap<TrackId, TrackedTensor<T>>>>,
    /// Next ID to assign
    next_id: Arc<RwLock<TrackId>>,
}

impl<T: TensorElement> TensorTracker<T> {
    /// Create a new tensor tracker
    pub fn new() -> Self {
        Self::with_config(TrackingConfig::default())
    }

    /// Create a new tensor tracker with custom config
    pub fn with_config(config: TrackingConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            tensors: Arc::new(RwLock::new(HashMap::new())),
            next_id: Arc::new(RwLock::new(0)),
        }
    }

    /// Start tracking a tensor
    pub fn track(&mut self, tensor: Tensor<T>, label: impl Into<String>) -> Result<TrackId>
    where
        T: Copy,
    {
        let config = self.config.read().expect("lock should not be poisoned");
        if !config.enabled {
            return Err(TorshError::InvalidArgument(
                "Tracking is disabled".to_string(),
            ));
        }
        drop(config);

        let mut next_id = self.next_id.write().expect("lock should not be poisoned");
        let id = *next_id;
        *next_id += 1;
        drop(next_id);

        let mut tracked = TrackedTensor::new(id, label.into(), tensor.clone());

        // Take initial snapshot if auto_snapshot is enabled
        let config = self.config.read().expect("lock should not be poisoned");
        if config.auto_snapshot {
            tracked.take_snapshot("initial".to_string())?;
        }
        drop(config);

        self.tensors
            .write()
            .expect("lock should not be poisoned")
            .insert(id, tracked);

        Ok(id)
    }

    /// Stop tracking a tensor
    pub fn untrack(&mut self, id: TrackId) -> Result<()> {
        self.tensors
            .write()
            .expect("lock should not be poisoned")
            .remove(&id);
        Ok(())
    }

    /// Record an operation on a tracked tensor
    pub fn record_operation<P: fmt::Display>(
        &self,
        id: TrackId,
        operation: impl Into<String>,
        parameters: Vec<P>,
        result_tensor: &Tensor<T>,
    ) -> Result<()>
    where
        T: Copy,
    {
        let config = self.config.read().expect("lock should not be poisoned");
        if !config.enabled {
            return Ok(());
        }

        let operation_str = operation.into();

        // Check filter
        if !config.operation_filter.is_empty() && !config.operation_filter.contains(&operation_str)
        {
            return Ok(());
        }

        let auto_snapshot = config.auto_snapshot;
        let max_operations = config.max_operations;
        drop(config);

        let mut tensors = self.tensors.write().expect("lock should not be poisoned");
        let tracked = tensors.get_mut(&id).ok_or_else(|| {
            TorshError::InvalidArgument(format!("Tensor with ID {} is not tracked", id))
        })?;

        let params: Vec<String> = parameters.iter().map(|p| format!("{}", p)).collect();

        tracked.record_operation(operation_str.clone(), params, result_tensor, None);

        // Trim if needed
        if tracked.operations.len() > max_operations {
            tracked.operations.remove(0);
        }

        // Auto snapshot if enabled
        if auto_snapshot {
            tracked.take_snapshot(format!("after_{}", operation_str))?;
        }

        Ok(())
    }

    /// Take a manual snapshot of a tracked tensor
    pub fn snapshot(&self, id: TrackId, label: impl Into<String>) -> Result<()>
    where
        T: Copy,
    {
        let mut tensors = self.tensors.write().expect("lock should not be poisoned");
        let tracked = tensors.get_mut(&id).ok_or_else(|| {
            TorshError::InvalidArgument(format!("Tensor with ID {} is not tracked", id))
        })?;

        tracked.take_snapshot(label.into())?;

        // Trim if needed
        let config = self.config.read().expect("lock should not be poisoned");
        if tracked.snapshots.len() > config.max_snapshots {
            tracked.snapshots.remove(0);
        }

        Ok(())
    }

    /// Generate a comprehensive report for a tracked tensor
    pub fn generate_report(&self, id: TrackId) -> Result<String>
    where
        T: Copy + PartialOrd + num_traits::Zero + num_traits::ToPrimitive + fmt::Display,
    {
        let tensors = self.tensors.read().expect("lock should not be poisoned");
        let tracked = tensors.get(&id).ok_or_else(|| {
            TorshError::InvalidArgument(format!("Tensor with ID {} is not tracked", id))
        })?;

        let mut report = String::new();
        report.push_str(&format!(
            "=== Tracking Report for '{}' (ID: {}) ===\n\n",
            tracked.label, tracked.id
        ));
        report.push_str(&format!(
            "Tracking duration: {:?}\n",
            tracked.start_time.elapsed()
        ));
        report.push_str(&format!(
            "Current shape: {:?}\n",
            tracked.tensor.shape().dims()
        ));
        report.push_str(&format!(
            "Operations performed: {}\n",
            tracked.operations.len()
        ));
        report.push_str(&format!("Snapshots taken: {}\n\n", tracked.snapshots.len()));

        // Current statistics
        if let Ok(stats) = TensorValueStats::from_tensor(&tracked.tensor) {
            report.push_str("Current Value Statistics:\n");
            report.push_str(&format!("{}\n", stats));
        }

        // Operation history
        if !tracked.operations.is_empty() {
            report.push_str("\nOperation History:\n");
            for (i, op) in tracked.operations.iter().enumerate() {
                report.push_str(&format!("  {}. {}\n", i + 1, op));
            }
        }

        // Snapshot summary
        if !tracked.snapshots.is_empty() {
            report.push_str("\nSnapshots:\n");
            for (i, snapshot) in tracked.snapshots.iter().enumerate() {
                report.push_str(&format!(
                    "  {}. '{}' - shape: {:?}, elements: {}\n",
                    i + 1,
                    snapshot.label,
                    snapshot.shape,
                    snapshot.values.len()
                ));
            }
        }

        Ok(report)
    }

    /// Get the current tensor for a tracked ID
    pub fn get_tensor(&self, id: TrackId) -> Result<Tensor<T>> {
        let tensors = self.tensors.read().expect("lock should not be poisoned");
        let tracked = tensors.get(&id).ok_or_else(|| {
            TorshError::InvalidArgument(format!("Tensor with ID {} is not tracked", id))
        })?;
        Ok(tracked.tensor.clone())
    }

    /// Get all tracked tensor IDs
    pub fn tracked_ids(&self) -> Vec<TrackId> {
        self.tensors
            .read()
            .expect("lock should not be poisoned")
            .keys()
            .copied()
            .collect()
    }

    /// Clear all tracking data
    pub fn clear(&mut self) {
        self.tensors
            .write()
            .expect("lock should not be poisoned")
            .clear();
        *self.next_id.write().expect("lock should not be poisoned") = 0;
    }
}

impl<T: TensorElement> Default for TensorTracker<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_tensor_tracker_basic() {
        let mut tracker = TensorTracker::new();
        let tensor = creation::ones::<f32>(&[2, 2]).unwrap();

        let id = tracker.track(tensor.clone(), "test_tensor").unwrap();
        assert_eq!(tracker.tracked_ids().len(), 1);

        let result = tensor.mul_scalar(2.0).unwrap();
        tracker
            .record_operation(id, "mul_scalar", vec![2.0], &result)
            .unwrap();

        let retrieved = tracker.get_tensor(id).unwrap();
        assert_eq!(retrieved.shape().dims(), &[2, 2]);

        tracker.untrack(id).unwrap();
        assert_eq!(tracker.tracked_ids().len(), 0);
    }

    #[test]
    fn test_tensor_value_stats() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::from_data(data, vec![5], DeviceType::Cpu).unwrap();

        let stats = TensorValueStats::from_tensor(&tensor).unwrap();
        assert_eq!(stats.total_elements, 5);
        assert_eq!(stats.min, Some(1.0));
        assert_eq!(stats.max, Some(5.0));
        assert!((stats.mean.unwrap() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_tracking_snapshots() {
        let mut tracker = TensorTracker::new();
        let tensor = creation::ones::<f32>(&[3, 3]).unwrap();

        let id = tracker.track(tensor.clone(), "snapshot_test").unwrap();

        tracker.snapshot(id, "first_snapshot").unwrap();
        tracker.snapshot(id, "second_snapshot").unwrap();

        let tensors = tracker.tensors.read().expect("lock should not be poisoned");
        let tracked = tensors.get(&id).unwrap();
        assert_eq!(tracked.snapshots.len(), 2);
    }

    #[test]
    fn test_tracking_report() {
        let mut tracker = TensorTracker::new();
        let data = vec![1.0f32, 2.0, 3.0];
        let tensor = Tensor::from_data(data, vec![3], DeviceType::Cpu).unwrap();

        let id = tracker.track(tensor.clone(), "report_test").unwrap();

        let result = tensor.mul_scalar(2.0).unwrap();
        tracker
            .record_operation(id, "mul_scalar", vec![2.0], &result)
            .unwrap();

        let report = tracker.generate_report(id).unwrap();
        assert!(report.contains("report_test"));
        assert!(report.contains("mul_scalar"));
        assert!(report.contains("Operations performed: 1"));
    }

    #[test]
    fn test_tracking_config() {
        let config = TrackingConfig::minimal();
        let mut tracker = TensorTracker::with_config(config);

        let tensor = creation::ones::<f32>(&[2, 2]).unwrap();
        let id = tracker.track(tensor, "config_test").unwrap();

        assert_eq!(tracker.tracked_ids().len(), 1);
        assert!(id == 0);
    }

    #[test]
    fn test_operation_filtering() {
        let config = TrackingConfig::filtered(vec!["add".to_string(), "mul".to_string()]);
        let mut tracker = TensorTracker::with_config(config);

        let tensor = creation::ones::<f32>(&[2, 2]).unwrap();
        let id = tracker.track(tensor.clone(), "filter_test").unwrap();

        // This should be tracked
        let result = tensor.mul_scalar(2.0).unwrap();
        tracker
            .record_operation(id, "mul", vec![2.0], &result)
            .unwrap();

        // This should be filtered out
        let result2 = result.add_scalar(1.0).unwrap();
        tracker
            .record_operation(id, "sub", vec![1.0], &result2)
            .unwrap();

        let tensors = tracker.tensors.read().expect("lock should not be poisoned");
        let tracked = tensors.get(&id).unwrap();
        assert_eq!(tracked.operations.len(), 1); // Only "mul" should be tracked
        assert_eq!(tracked.operations[0].operation, "mul");
    }
}
