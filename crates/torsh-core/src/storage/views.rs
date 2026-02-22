//! Storage view system for zero-copy tensor operations
//!
//! This module provides efficient views into existing storage without copying data,
//! enabling zero-copy slicing, sub-tensors, and memory sharing operations.

use crate::error::Result;
use crate::storage::core::{SharedStorage, Storage};
use std::sync::Arc;

/// View into existing storage for zero-copy slicing operations
///
/// StorageView provides a window into a parent storage without copying the underlying data.
/// It tracks the offset and length within the parent storage and provides safe access
/// to the viewed portion.
///
/// # Examples
///
/// ```ignore
/// use torsh_core::storage::{SharedStorage, StorageView};
///
/// // Create a view into storage
/// let view = StorageView::new(shared_storage, 10, 20)?;
/// assert_eq!(view.offset(), 10);
/// assert_eq!(view.view_len(), 20);
///
/// // Create sub-views
/// let sub_view = view.slice(5, 10)?;
/// assert_eq!(sub_view.offset(), 15); // 10 + 5
/// assert_eq!(sub_view.view_len(), 10);
/// ```
#[derive(Debug)]
pub struct StorageView<S: Storage> {
    parent: SharedStorage<S>,
    offset: usize,
    len: usize,
}

impl<S: Storage> StorageView<S> {
    /// Create a new view into existing storage
    ///
    /// # Arguments
    /// * `parent` - The parent shared storage
    /// * `offset` - Starting offset in the parent storage
    /// * `len` - Length of the view in elements
    ///
    /// # Returns
    /// A new storage view or an error if the bounds are invalid
    pub fn new(parent: SharedStorage<S>, offset: usize, len: usize) -> Result<Self> {
        if offset + len > parent.get().len() {
            return Err(crate::error::TorshError::IndexOutOfBounds {
                index: offset + len,
                size: parent.get().len(),
            });
        }

        Ok(StorageView {
            parent,
            offset,
            len,
        })
    }

    /// Get the parent storage
    pub fn parent(&self) -> &SharedStorage<S> {
        &self.parent
    }

    /// Get the offset in the parent storage
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get the length of this view
    pub fn view_len(&self) -> usize {
        self.len
    }

    /// Check if this view is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the end offset (exclusive) in the parent storage
    pub fn end_offset(&self) -> usize {
        self.offset + self.len
    }

    /// Check if this view overlaps with another view
    ///
    /// Two views overlap if they reference the same parent storage and their
    /// offset ranges intersect.
    pub fn overlaps(&self, other: &StorageView<S>) -> bool {
        // Check if they reference the same parent storage
        if Arc::ptr_eq(self.parent.inner_arc(), other.parent.inner_arc()) {
            let self_end = self.offset + self.len;
            let other_end = other.offset + other.len;

            !(self_end <= other.offset || other_end <= self.offset)
        } else {
            false
        }
    }

    /// Check if this view completely contains another view
    pub fn contains(&self, other: &StorageView<S>) -> bool {
        if Arc::ptr_eq(self.parent.inner_arc(), other.parent.inner_arc()) {
            self.offset <= other.offset && other.end_offset() <= self.end_offset()
        } else {
            false
        }
    }

    /// Check if this view is adjacent to another view
    pub fn is_adjacent(&self, other: &StorageView<S>) -> bool {
        if Arc::ptr_eq(self.parent.inner_arc(), other.parent.inner_arc()) {
            self.end_offset() == other.offset || other.end_offset() == self.offset
        } else {
            false
        }
    }

    /// Create a sub-view of this view
    ///
    /// # Arguments
    /// * `start` - Starting offset within this view
    /// * `len` - Length of the sub-view
    ///
    /// # Returns
    /// A new storage view that represents a subset of this view
    pub fn slice(&self, start: usize, len: usize) -> Result<StorageView<S>> {
        if start + len > self.len {
            return Err(crate::error::TorshError::IndexOutOfBounds {
                index: start + len,
                size: self.len,
            });
        }

        StorageView::new(self.parent.clone(), self.offset + start, len)
    }

    /// Split this view at the given position
    ///
    /// # Arguments
    /// * `at` - Position to split at (relative to this view)
    ///
    /// # Returns
    /// A tuple of (left_view, right_view) or an error if the position is invalid
    pub fn split_at(&self, at: usize) -> Result<(StorageView<S>, StorageView<S>)> {
        if at > self.len {
            return Err(crate::error::TorshError::IndexOutOfBounds {
                index: at,
                size: self.len,
            });
        }

        let left = StorageView::new(self.parent.clone(), self.offset, at)?;
        let right = StorageView::new(self.parent.clone(), self.offset + at, self.len - at)?;

        Ok((left, right))
    }

    /// Create multiple non-overlapping views by splitting at given positions
    ///
    /// # Arguments
    /// * `positions` - Positions to split at (must be sorted)
    ///
    /// # Returns
    /// A vector of storage views or an error if any position is invalid
    pub fn split_at_positions(&self, positions: &[usize]) -> Result<Vec<StorageView<S>>> {
        if positions.is_empty() {
            return Ok(vec![self.clone()]);
        }

        // Verify positions are sorted and within bounds
        for (i, &pos) in positions.iter().enumerate() {
            if pos > self.len {
                return Err(crate::error::TorshError::IndexOutOfBounds {
                    index: pos,
                    size: self.len,
                });
            }
            if i > 0 && pos <= positions[i - 1] {
                return Err(crate::error::TorshError::InvalidArgument(
                    "Split positions must be sorted and unique".to_string(),
                ));
            }
        }

        let mut views = Vec::new();
        let mut last_pos = 0;

        for &pos in positions {
            if pos > last_pos {
                views.push(StorageView::new(
                    self.parent.clone(),
                    self.offset + last_pos,
                    pos - last_pos,
                )?);
            }
            last_pos = pos;
        }

        // Add final segment if there's remaining data
        if last_pos < self.len {
            views.push(StorageView::new(
                self.parent.clone(),
                self.offset + last_pos,
                self.len - last_pos,
            )?);
        }

        Ok(views)
    }

    /// Merge this view with an adjacent view
    ///
    /// # Arguments
    /// * `other` - The other view to merge with
    ///
    /// # Returns
    /// A new view that spans both views, or an error if they're not adjacent
    pub fn merge(&self, other: &StorageView<S>) -> Result<StorageView<S>> {
        if !Arc::ptr_eq(self.parent.inner_arc(), other.parent.inner_arc()) {
            return Err(crate::error::TorshError::InvalidArgument(
                "Cannot merge views from different parent storage".to_string(),
            ));
        }

        let (start_offset, total_len) = if self.end_offset() == other.offset {
            // self comes before other
            (self.offset, self.len + other.len)
        } else if other.end_offset() == self.offset {
            // other comes before self
            (other.offset, other.len + self.len)
        } else {
            return Err(crate::error::TorshError::InvalidArgument(
                "Views are not adjacent and cannot be merged".to_string(),
            ));
        };

        StorageView::new(self.parent.clone(), start_offset, total_len)
    }

    /// Check if this view can be safely extended
    ///
    /// # Arguments
    /// * `additional_len` - Additional length to extend by
    ///
    /// # Returns
    /// True if the extension would be within parent storage bounds
    pub fn can_extend(&self, additional_len: usize) -> bool {
        self.offset + self.len + additional_len <= self.parent.get().len()
    }

    /// Extend this view if possible
    ///
    /// # Arguments
    /// * `additional_len` - Additional length to extend by
    ///
    /// # Returns
    /// A new extended view or an error if extension is not possible
    pub fn extend(&self, additional_len: usize) -> Result<StorageView<S>> {
        if !self.can_extend(additional_len) {
            return Err(crate::error::TorshError::IndexOutOfBounds {
                index: self.offset + self.len + additional_len,
                size: self.parent.get().len(),
            });
        }

        StorageView::new(self.parent.clone(), self.offset, self.len + additional_len)
    }

    /// Get view statistics
    pub fn statistics(&self) -> ViewStatistics {
        ViewStatistics {
            offset: self.offset,
            length: self.len,
            parent_length: self.parent.get().len(),
            parent_ref_count: self.parent.strong_count(),
            coverage_ratio: self.len as f64 / self.parent.get().len() as f64,
        }
    }

    /// Convert this view to a new independent storage (copies data)
    pub fn to_owned_storage(&self) -> Result<S>
    where
        S: Clone,
    {
        // This would require access to the actual data, which depends on the storage implementation
        // For now, we provide the interface - actual implementation would depend on storage type
        self.parent.get().clone_storage()
    }
}

impl<S: Storage> Clone for StorageView<S> {
    fn clone(&self) -> Self {
        StorageView {
            parent: self.parent.clone(),
            offset: self.offset,
            len: self.len,
        }
    }
}

/// Statistics about a storage view
#[derive(Debug, Clone)]
pub struct ViewStatistics {
    /// Offset within parent storage
    pub offset: usize,
    /// Length of the view
    pub length: usize,
    /// Total length of parent storage
    pub parent_length: usize,
    /// Reference count of parent storage
    pub parent_ref_count: usize,
    /// Ratio of view length to parent length
    pub coverage_ratio: f64,
}

impl std::fmt::Display for ViewStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ViewStats(offset={}, len={}, parent_len={}, refs={}, coverage={:.2}%)",
            self.offset,
            self.length,
            self.parent_length,
            self.parent_ref_count,
            self.coverage_ratio * 100.0
        )
    }
}

/// Builder for creating complex view hierarchies
#[derive(Debug)]
pub struct ViewBuilder<S: Storage> {
    parent: SharedStorage<S>,
    operations: Vec<ViewOperation>,
}

#[derive(Debug, Clone)]
enum ViewOperation {
    Slice { start: usize, len: usize },
    SplitAt { position: usize },
    Extend { additional_len: usize },
}

impl<S: Storage> ViewBuilder<S> {
    /// Create a new view builder
    pub fn new(parent: SharedStorage<S>) -> Self {
        Self {
            parent,
            operations: Vec::new(),
        }
    }

    /// Add a slice operation
    pub fn slice(mut self, start: usize, len: usize) -> Self {
        self.operations.push(ViewOperation::Slice { start, len });
        self
    }

    /// Add a split operation
    pub fn split_at(mut self, position: usize) -> Self {
        self.operations.push(ViewOperation::SplitAt { position });
        self
    }

    /// Add an extend operation
    pub fn extend(mut self, additional_len: usize) -> Self {
        self.operations
            .push(ViewOperation::Extend { additional_len });
        self
    }

    /// Build the final view by applying all operations
    pub fn build(self) -> Result<StorageView<S>> {
        let parent_len = self.parent.get().len();
        let mut current_view = StorageView::new(self.parent, 0, parent_len)?;

        for operation in self.operations {
            current_view = match operation {
                ViewOperation::Slice { start, len } => current_view.slice(start, len)?,
                ViewOperation::SplitAt { position } => {
                    let (left, _right) = current_view.split_at(position)?;
                    left
                }
                ViewOperation::Extend { additional_len } => current_view.extend(additional_len)?,
            };
        }

        Ok(current_view)
    }

    /// Build and return multiple views if split operations were used
    pub fn build_multiple(self) -> Result<Vec<StorageView<S>>> {
        // Simplified implementation - in practice would handle multiple splits
        Ok(vec![self.build()?])
    }
}

/// Utility functions for working with storage views
pub mod utils {
    use super::*;

    /// Find all overlapping views in a collection
    pub fn find_overlapping_views<S: Storage>(views: &[StorageView<S>]) -> Vec<(usize, usize)> {
        let mut overlaps = Vec::new();

        for (i, view1) in views.iter().enumerate() {
            for (j, view2) in views.iter().enumerate().skip(i + 1) {
                if view1.overlaps(view2) {
                    overlaps.push((i, j));
                }
            }
        }

        overlaps
    }

    /// Merge all adjacent views in a collection
    pub fn merge_adjacent_views<S: Storage>(
        views: Vec<StorageView<S>>,
    ) -> Result<Vec<StorageView<S>>> {
        if views.is_empty() {
            return Ok(views);
        }

        let mut result = Vec::new();
        let mut views_iter = views.into_iter();
        let mut current = views_iter
            .next()
            .expect("views is non-empty after is_empty check");

        for view in views_iter {
            match current.merge(&view) {
                Ok(merged) => current = merged,
                Err(_) => {
                    result.push(current);
                    current = view;
                }
            }
        }
        result.push(current);

        Ok(result)
    }

    /// Calculate total memory coverage of a set of views
    pub fn calculate_coverage<S: Storage>(views: &[StorageView<S>]) -> f64 {
        if views.is_empty() {
            return 0.0;
        }

        let total_view_elements: usize = views.iter().map(|v| v.view_len()).sum();
        let parent_elements = views[0].parent().get().len();

        if parent_elements == 0 {
            return 0.0;
        }

        total_view_elements as f64 / parent_elements as f64
    }

    /// Check if a set of views completely covers the parent storage
    pub fn views_cover_parent<S: Storage>(views: &[StorageView<S>]) -> bool {
        if views.is_empty() {
            return false;
        }

        let mut coverage = vec![false; views[0].parent().get().len()];

        for view in views {
            for i in view.offset()..view.end_offset() {
                if i < coverage.len() {
                    coverage[i] = true;
                }
            }
        }

        coverage.iter().all(|&covered| covered)
    }
}
