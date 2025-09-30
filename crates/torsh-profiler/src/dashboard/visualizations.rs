//! Advanced visualizations for performance dashboard
//!
//! This module provides sophisticated visualization capabilities including 3D performance
//! landscapes, heatmaps, charts, and interactive visualizations for the ToRSh dashboard.

use crate::{ProfileEvent, Profiler, TorshResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_core::TorshError;

use super::types::{
    HeatmapCell, PerformancePoint3D, VisualizationColorScheme, VisualizationConfig,
};

// =============================================================================
// 3D Performance Landscape
// =============================================================================

/// Advanced 3D performance landscape generator
pub struct PerformanceLandscape {
    config: VisualizationConfig,
    points: Vec<PerformancePoint3D>,
    metadata: LandscapeMetadata,
}

impl PerformanceLandscape {
    /// Create a new performance landscape with configuration
    pub fn new(config: VisualizationConfig) -> Self {
        Self {
            config,
            points: Vec::new(),
            metadata: LandscapeMetadata::default(),
        }
    }

    /// Generate 3D landscape from profiling data
    pub fn generate_from_profiler(&mut self, profiler: &Profiler) -> TorshResult<()> {
        let events = profiler.events();
        self.points.clear();

        if events.is_empty() {
            return Ok(());
        }

        // Update metadata
        self.metadata.total_events = events.len();
        self.metadata.generation_timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Group events by thread and time windows
        let mut thread_buckets: HashMap<usize, Vec<&ProfileEvent>> = HashMap::new();

        for event in events {
            let thread_id = event.thread_id;
            thread_buckets.entry(thread_id).or_default().push(event);
        }

        // Generate landscape points
        for (thread_idx, (_thread_id, thread_events)) in thread_buckets.iter().enumerate() {
            // Sort events by start time
            let mut sorted_events = thread_events.clone();
            sorted_events.sort_by_key(|e| e.start_us);

            // Create time windows and calculate metrics
            let window_size = self.calculate_optimal_window_size(&sorted_events);
            let mut current_window_start = 0u64;
            let mut window_events = Vec::new();

            for event in sorted_events {
                if event.start_us >= current_window_start + window_size {
                    if !window_events.is_empty() {
                        let point = self.create_landscape_point(
                            current_window_start as f64 / 1_000_000.0, // Convert to seconds
                            thread_idx as f64,
                            &window_events,
                        );
                        self.points.push(point);
                    }

                    current_window_start = (event.start_us / window_size) * window_size;
                    window_events.clear();
                }
                window_events.push(event);
            }

            // Handle final window
            if !window_events.is_empty() {
                let point = self.create_landscape_point(
                    current_window_start as f64 / 1_000_000.0,
                    thread_idx as f64,
                    &window_events,
                );
                self.points.push(point);
            }
        }

        // Update final metadata
        self.metadata.point_count = self.points.len();
        self.metadata.x_range = self.get_x_range();
        self.metadata.y_range = self.get_y_range();
        self.metadata.z_range = self.get_z_range();

        Ok(())
    }

    /// Calculate optimal window size based on data distribution
    fn calculate_optimal_window_size(&self, events: &[&ProfileEvent]) -> u64 {
        if events.is_empty() {
            return 100_000; // 100ms default
        }

        let total_duration = events.last().unwrap().start_us - events.first().unwrap().start_us;
        let target_windows = self.config.grid_resolution;

        if total_duration > 0 && target_windows > 0 {
            (total_duration / target_windows as u64).max(10_000) // Min 10ms windows
        } else {
            100_000 // 100ms default
        }
    }

    /// Create a landscape point from events in a time window
    fn create_landscape_point(
        &self,
        time: f64,
        thread: f64,
        events: &[&ProfileEvent],
    ) -> PerformancePoint3D {
        let total_duration: f64 = events.iter().map(|e| e.duration_us as f64).sum();
        let total_flops: f64 = events
            .iter()
            .filter_map(|e| e.flops.map(|f| f as f64))
            .sum();
        let operation_count = events.len();

        // Calculate advanced performance metrics
        let avg_duration = if operation_count > 0 {
            total_duration / operation_count as f64
        } else {
            0.0
        };

        let intensity = if total_duration > 0.0 {
            operation_count as f64 / total_duration
        } else {
            0.0
        };

        // Calculate efficiency metric
        let efficiency = if total_flops > 0.0 && total_duration > 0.0 {
            total_flops / total_duration
        } else {
            0.0
        };

        PerformancePoint3D {
            x: time,
            y: thread,
            z: total_duration / 1000.0, // Convert to milliseconds for Z height
            intensity,
            metadata: format!(
                "Thread {}, {} ops, {:.2}ms avg, {:.0} FLOPS, efficiency: {:.2}",
                thread as u64,
                operation_count,
                avg_duration / 1000.0,
                total_flops,
                efficiency
            ),
        }
    }

    /// Export landscape as JSON for visualization libraries
    pub fn export_json(&self) -> TorshResult<String> {
        let json_data = serde_json::json!({
            "type": "performance_landscape_3d",
            "version": "1.0",
            "config": self.config,
            "points": self.points,
            "metadata": self.metadata
        });

        serde_json::to_string_pretty(&json_data).map_err(|e| {
            TorshError::SerializationError(format!("Failed to serialize landscape: {e}"))
        })
    }

    /// Export as Three.js compatible format
    pub fn export_threejs(&self) -> TorshResult<String> {
        let positions: Vec<f64> = self
            .points
            .iter()
            .flat_map(|p| vec![p.x, p.y, p.z])
            .collect();

        let colors: Vec<f64> = self
            .points
            .iter()
            .flat_map(|p| self.intensity_to_rgb(p.intensity))
            .collect();

        let json_data = serde_json::json!({
            "geometry": {
                "type": "BufferGeometry",
                "attributes": {
                    "position": {
                        "itemSize": 3,
                        "type": "Float32Array",
                        "array": positions
                    },
                    "color": {
                        "itemSize": 3,
                        "type": "Float32Array",
                        "array": colors
                    }
                }
            },
            "material": {
                "type": "PointsMaterial",
                "size": 2.0,
                "vertexColors": true
            },
            "metadata": self.metadata
        });

        serde_json::to_string_pretty(&json_data).map_err(|e| {
            TorshError::SerializationError(format!("Failed to serialize Three.js landscape: {e}"))
        })
    }

    /// Convert intensity to RGB color values
    fn intensity_to_rgb(&self, intensity: f64) -> Vec<f64> {
        match &self.config.color_scheme {
            VisualizationColorScheme::Thermal => {
                vec![intensity, 0.0, 1.0 - intensity]
            }
            VisualizationColorScheme::Viridis => {
                let (r, g, b) = viridis_color_map(intensity);
                vec![r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0]
            }
            VisualizationColorScheme::Plasma => {
                let (r, g, b) = plasma_color_map(intensity);
                vec![r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0]
            }
            VisualizationColorScheme::Custom { start, end } => {
                let r = start[0] as f64 + intensity * (end[0] as f64 - start[0] as f64);
                let g = start[1] as f64 + intensity * (end[1] as f64 - start[1] as f64);
                let b = start[2] as f64 + intensity * (end[2] as f64 - start[2] as f64);
                vec![r / 255.0, g / 255.0, b / 255.0]
            }
        }
    }

    /// Get coordinate ranges
    fn get_x_range(&self) -> (f64, f64) {
        if self.points.is_empty() {
            return (0.0, 0.0);
        }
        let min_x = self
            .points
            .iter()
            .map(|p| p.x)
            .fold(f64::INFINITY, f64::min);
        let max_x = self
            .points
            .iter()
            .map(|p| p.x)
            .fold(f64::NEG_INFINITY, f64::max);
        (min_x, max_x)
    }

    fn get_y_range(&self) -> (f64, f64) {
        if self.points.is_empty() {
            return (0.0, 0.0);
        }
        let min_y = self
            .points
            .iter()
            .map(|p| p.y)
            .fold(f64::INFINITY, f64::min);
        let max_y = self
            .points
            .iter()
            .map(|p| p.y)
            .fold(f64::NEG_INFINITY, f64::max);
        (min_y, max_y)
    }

    fn get_z_range(&self) -> (f64, f64) {
        if self.points.is_empty() {
            return (0.0, 0.0);
        }
        let min_z = self
            .points
            .iter()
            .map(|p| p.z)
            .fold(f64::INFINITY, f64::min);
        let max_z = self
            .points
            .iter()
            .map(|p| p.z)
            .fold(f64::NEG_INFINITY, f64::max);
        (min_z, max_z)
    }

    /// Get all points for external access
    pub fn get_points(&self) -> &[PerformancePoint3D] {
        &self.points
    }

    /// Get metadata
    pub fn get_metadata(&self) -> &LandscapeMetadata {
        &self.metadata
    }
}

// =============================================================================
// Performance Heatmap
// =============================================================================

/// Advanced performance heatmap generator
pub struct PerformanceHeatmap {
    config: VisualizationConfig,
    cells: Vec<HeatmapCell>,
    width: usize,
    height: usize,
    metadata: HeatmapMetadata,
}

impl PerformanceHeatmap {
    /// Create a new performance heatmap
    pub fn new(config: VisualizationConfig, width: usize, height: usize) -> Self {
        Self {
            config,
            cells: Vec::new(),
            width,
            height,
            metadata: HeatmapMetadata::default(),
        }
    }

    /// Generate operation-based heatmap from profiling data
    pub fn generate_operation_heatmap(&mut self, profiler: &Profiler) -> TorshResult<()> {
        let events = profiler.events();
        self.cells.clear();

        if events.is_empty() {
            return Ok(());
        }

        // Update metadata
        self.metadata.total_events = events.len();
        self.metadata.generation_timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Create operation mapping
        let mut operation_map: HashMap<String, usize> = HashMap::new();
        let mut operation_index = 0;

        // Map unique operations to indices
        for event in events {
            if !operation_map.contains_key(&event.name) {
                operation_map.insert(event.name.clone(), operation_index);
                operation_index += 1;
            }
        }

        // Limit to top operations if too many
        let max_operations = self.height;
        if operation_map.len() > max_operations {
            let event_refs: Vec<&ProfileEvent> = events.iter().collect();
            operation_map = self.get_top_operations(&operation_map, &event_refs, max_operations);
        }

        // Create time windows
        let event_refs: Vec<&ProfileEvent> = events.iter().collect();
        let time_window_us = self.calculate_time_window(&event_refs);
        let max_time = events.iter().map(|e| e.start_us).max().unwrap_or(0);
        let num_windows = ((max_time / time_window_us) + 1).min(self.width as u64) as usize;

        // Initialize grid
        let mut grid = vec![vec![0.0f64; num_windows]; operation_map.len()];

        // Fill grid with performance data
        for event in events {
            if let Some(&op_index) = operation_map.get(&event.name) {
                let time_index = ((event.start_us / time_window_us) as usize).min(num_windows - 1);
                grid[op_index][time_index] += event.duration_us as f64;
            }
        }

        // Normalize and create heatmap cells
        let max_value = grid
            .iter()
            .flat_map(|row| row.iter())
            .fold(0.0f64, |acc, &val| acc.max(val));

        for (row_idx, row) in grid.iter().enumerate() {
            for (col_idx, &value) in row.iter().enumerate() {
                let intensity = if max_value > 0.0 {
                    value / max_value
                } else {
                    0.0
                };
                let color = self.intensity_to_color(intensity);

                let cell = HeatmapCell {
                    row: row_idx,
                    col: col_idx,
                    intensity,
                    label: self.get_cell_label(row_idx, col_idx, &operation_map),
                    color,
                };
                self.cells.push(cell);
            }
        }

        // Update final metadata
        self.metadata.cell_count = self.cells.len();
        self.metadata.max_intensity = max_value;
        self.metadata.operation_count = operation_map.len();
        self.metadata.time_window_count = num_windows;

        Ok(())
    }

    /// Calculate optimal time window size
    fn calculate_time_window(&self, events: &[&ProfileEvent]) -> u64 {
        if events.is_empty() {
            return 100_000;
        }

        let total_duration = events.last().unwrap().start_us - events.first().unwrap().start_us;
        if total_duration > 0 {
            (total_duration / self.width as u64).max(10_000) // Min 10ms windows
        } else {
            100_000 // 100ms default
        }
    }

    /// Get top operations by frequency
    fn get_top_operations(
        &self,
        operation_map: &HashMap<String, usize>,
        events: &[&ProfileEvent],
        max_operations: usize,
    ) -> HashMap<String, usize> {
        let op_names: Vec<String> = operation_map.keys().cloned().collect();
        let mut op_counts: Vec<(String, usize)> = op_names
            .into_iter()
            .map(|name| {
                let count = events.iter().filter(|e| e.name == name).count();
                (name, count)
            })
            .collect();

        op_counts.sort_by(|a, b| b.1.cmp(&a.1));

        let mut new_map = HashMap::new();
        for (i, (op_name, _)) in op_counts.iter().take(max_operations).enumerate() {
            new_map.insert(op_name.clone(), i);
        }
        new_map
    }

    /// Get cell label for display
    fn get_cell_label(
        &self,
        row_idx: usize,
        col_idx: usize,
        operation_map: &HashMap<String, usize>,
    ) -> Option<String> {
        // Find operation name for this row
        let operation_name = operation_map
            .iter()
            .find(|(_, &idx)| idx == row_idx)
            .map(|(name, _)| name.clone());

        operation_name.map(|name| format!("{}@{}", name, col_idx))
    }

    /// Convert intensity to color based on color scheme
    fn intensity_to_color(&self, intensity: f64) -> String {
        match &self.config.color_scheme {
            VisualizationColorScheme::Thermal => {
                // Blue (cold) to red (hot)
                let r = (intensity * 255.0) as u8;
                let b = ((1.0 - intensity) * 255.0) as u8;
                format!("#{r:02x}00{b:02x}")
            }
            VisualizationColorScheme::Viridis => {
                let (r, g, b) = viridis_color_map(intensity);
                format!("#{r:02x}{g:02x}{b:02x}")
            }
            VisualizationColorScheme::Plasma => {
                let (r, g, b) = plasma_color_map(intensity);
                format!("#{r:02x}{g:02x}{b:02x}")
            }
            VisualizationColorScheme::Custom { start, end } => {
                let r = (start[0] as f64 + intensity * (end[0] as f64 - start[0] as f64)) as u8;
                let g = (start[1] as f64 + intensity * (end[1] as f64 - start[1] as f64)) as u8;
                let b = (start[2] as f64 + intensity * (end[2] as f64 - start[2] as f64)) as u8;
                format!("#{r:02x}{g:02x}{b:02x}")
            }
        }
    }

    /// Export heatmap as JSON
    pub fn export_json(&self) -> TorshResult<String> {
        let json_data = serde_json::json!({
            "type": "performance_heatmap",
            "version": "1.0",
            "config": self.config,
            "cells": self.cells,
            "dimensions": {
                "width": self.width,
                "height": self.height
            },
            "metadata": self.metadata
        });

        serde_json::to_string_pretty(&json_data).map_err(|e| {
            TorshError::SerializationError(format!("Failed to serialize heatmap: {e}"))
        })
    }

    /// Export as D3.js compatible format
    pub fn export_d3(&self) -> TorshResult<String> {
        let data: Vec<serde_json::Value> = self
            .cells
            .iter()
            .map(|cell| {
                serde_json::json!({
                    "x": cell.col,
                    "y": cell.row,
                    "value": cell.intensity,
                    "color": cell.color,
                    "label": cell.label
                })
            })
            .collect();

        let json_data = serde_json::json!({
            "format": "d3_heatmap",
            "data": data,
            "dimensions": {
                "width": self.width,
                "height": self.height
            },
            "metadata": self.metadata
        });

        serde_json::to_string_pretty(&json_data).map_err(|e| {
            TorshError::SerializationError(format!("Failed to serialize D3 heatmap: {e}"))
        })
    }

    /// Get all cells
    pub fn get_cells(&self) -> &[HeatmapCell] {
        &self.cells
    }

    /// Get metadata
    pub fn get_metadata(&self) -> &HeatmapMetadata {
        &self.metadata
    }

    /// Generate heatmap from profiling data (alias)
    pub fn generate_from_profiler(&mut self, profiler: &Profiler) -> TorshResult<()> {
        self.generate_operation_heatmap(profiler)
    }
}

// =============================================================================
// Color Map Functions
// =============================================================================

/// Viridis color map implementation
fn viridis_color_map(t: f64) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    // Simplified viridis approximation with better accuracy
    let r = if t < 0.5 {
        68.0 + t * 2.0 * (103.0 - 68.0)
    } else {
        103.0 + (t - 0.5) * 2.0 * (253.0 - 103.0)
    };
    let g = if t < 0.25 {
        1.0 + t * 4.0 * (50.0 - 1.0)
    } else if t < 0.75 {
        50.0 + (t - 0.25) * 2.0 * (180.0 - 50.0)
    } else {
        180.0 + (t - 0.75) * 4.0 * (231.0 - 180.0)
    };
    let b = if t < 0.5 {
        84.0 + t * 2.0 * (60.0 - 84.0)
    } else {
        60.0 + (t - 0.5) * 2.0 * (37.0 - 60.0)
    };
    (r as u8, g as u8, b as u8)
}

/// Plasma color map implementation
fn plasma_color_map(t: f64) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    // Simplified plasma approximation with better accuracy
    let r = if t < 0.5 {
        13.0 + t * 2.0 * (150.0 - 13.0)
    } else {
        150.0 + (t - 0.5) * 2.0 * (240.0 - 150.0)
    };
    let g = if t < 0.3 {
        8.0 + t * (100.0 - 8.0) / 0.3
    } else if t < 0.7 {
        100.0 + (t - 0.3) * (200.0 - 100.0) / 0.4
    } else {
        200.0 + (t - 0.7) * (249.0 - 200.0) / 0.3
    };
    let b = if t < 0.6 {
        135.0 + t * (80.0 - 135.0) / 0.6
    } else {
        80.0 + (t - 0.6) * (33.0 - 80.0) / 0.4
    };
    (r as u8, g as u8, b as u8)
}

/// Convert RGB to HSV for advanced color operations
pub fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f64, f64, f64) {
    let r = r as f64 / 255.0;
    let g = g as f64 / 255.0;
    let b = b as f64 / 255.0;

    let max = r.max(g.max(b));
    let min = r.min(g.min(b));
    let delta = max - min;

    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        60.0 * ((g - b) / delta) % 6.0
    } else if max == g {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };

    let s = if max == 0.0 { 0.0 } else { delta / max };
    let v = max;

    (h, s, v)
}

// =============================================================================
// Metadata Structures
// =============================================================================

/// Metadata for 3D landscape visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LandscapeMetadata {
    pub total_events: usize,
    pub point_count: usize,
    pub generation_timestamp: u64,
    pub x_range: (f64, f64),
    pub y_range: (f64, f64),
    pub z_range: (f64, f64),
}

impl Default for LandscapeMetadata {
    fn default() -> Self {
        Self {
            total_events: 0,
            point_count: 0,
            generation_timestamp: 0,
            x_range: (0.0, 0.0),
            y_range: (0.0, 0.0),
            z_range: (0.0, 0.0),
        }
    }
}

/// Metadata for heatmap visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatmapMetadata {
    pub total_events: usize,
    pub cell_count: usize,
    pub generation_timestamp: u64,
    pub max_intensity: f64,
    pub operation_count: usize,
    pub time_window_count: usize,
}

impl Default for HeatmapMetadata {
    fn default() -> Self {
        Self {
            total_events: 0,
            cell_count: 0,
            generation_timestamp: 0,
            max_intensity: 0.0,
            operation_count: 0,
            time_window_count: 0,
        }
    }
}

// =============================================================================
// High-Level Visualization Functions
// =============================================================================

/// Generate 3D landscape visualization
pub fn generate_3d_landscape(
    profiler: &Profiler,
    config: Option<VisualizationConfig>,
) -> TorshResult<String> {
    let config = config.unwrap_or_default();
    let mut landscape = PerformanceLandscape::new(config);
    landscape.generate_from_profiler(profiler)?;
    landscape.export_json()
}

/// Generate 3D landscape for Three.js
pub fn generate_3d_landscape_threejs(
    profiler: &Profiler,
    config: Option<VisualizationConfig>,
) -> TorshResult<String> {
    let config = config.unwrap_or_default();
    let mut landscape = PerformanceLandscape::new(config);
    landscape.generate_from_profiler(profiler)?;
    landscape.export_threejs()
}

/// Generate performance heatmap visualization
pub fn generate_performance_heatmap(
    profiler: &Profiler,
    width: usize,
    height: usize,
    config: Option<VisualizationConfig>,
) -> TorshResult<String> {
    let config = config.unwrap_or_default();
    let mut heatmap = PerformanceHeatmap::new(config, width, height);
    heatmap.generate_operation_heatmap(profiler)?;
    heatmap.export_json()
}

/// Generate heatmap for D3.js
pub fn generate_performance_heatmap_d3(
    profiler: &Profiler,
    width: usize,
    height: usize,
    config: Option<VisualizationConfig>,
) -> TorshResult<String> {
    let config = config.unwrap_or_default();
    let mut heatmap = PerformanceHeatmap::new(config, width, height);
    heatmap.generate_operation_heatmap(profiler)?;
    heatmap.export_d3()
}

/// Generate comprehensive visualization package
pub fn generate_visualization_package(
    profiler: &Profiler,
    config: Option<VisualizationConfig>,
) -> TorshResult<VisualizationPackage> {
    let config = config.unwrap_or_default();

    let landscape_json = generate_3d_landscape(profiler, Some(config.clone()))?;
    let landscape_threejs = generate_3d_landscape_threejs(profiler, Some(config.clone()))?;
    let heatmap_json = generate_performance_heatmap(profiler, 50, 30, Some(config.clone()))?;
    let heatmap_d3 = generate_performance_heatmap_d3(profiler, 50, 30, Some(config.clone()))?;

    Ok(VisualizationPackage {
        landscape_json,
        landscape_threejs,
        heatmap_json,
        heatmap_d3,
        config,
        generation_timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    })
}

// =============================================================================
// Supporting Types
// =============================================================================

/// Complete visualization package with multiple formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationPackage {
    pub landscape_json: String,
    pub landscape_threejs: String,
    pub heatmap_json: String,
    pub heatmap_d3: String,
    pub config: VisualizationConfig,
    pub generation_timestamp: u64,
}

/// Visualization export formats
#[derive(Debug, Clone)]
pub enum VisualizationFormat {
    Json,
    ThreeJs,
    D3,
    Svg,
    Canvas,
}

/// Visualization export options
#[derive(Debug, Clone)]
pub struct ExportOptions {
    pub format: VisualizationFormat,
    pub width: usize,
    pub height: usize,
    pub include_metadata: bool,
    pub compress: bool,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: VisualizationFormat::Json,
            width: 800,
            height: 600,
            include_metadata: true,
            compress: false,
        }
    }
}
