// Copyright (c) 2025 ToRSh Project
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Flamegraph Generation for Autograd Operations
//!
//! This module provides flamegraph visualization for autograd operations,
//! enabling visual performance analysis and bottleneck identification.
//!
//! # Features
//!
//! - **Hierarchical Visualization**: Shows operation call stacks
//! - **Time-based Coloring**: Color-codes operations by execution time
//! - **Interactive HTML**: Generates interactive flamegraphs
//! - **SVG Export**: High-quality SVG output
//! - **Differential Flamegraphs**: Compare performance across runs
//! - **Memory Flamegraphs**: Visualize memory allocations

use crate::error_handling::AutogradResult;
use crate::gradient_tracer::{GradientPath, TraceEvent};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Flamegraph node representing an operation in the call stack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlamegraphNode {
    /// Operation name
    pub name: String,

    /// Total time spent in this operation (including children)
    pub total_time: Duration,

    /// Self time (excluding children)
    pub self_time: Duration,

    /// Number of invocations
    pub invocation_count: usize,

    /// Total memory allocated
    pub total_memory: usize,

    /// Child nodes
    pub children: IndexMap<String, FlamegraphNode>,

    /// Stack depth
    pub depth: usize,
}

impl FlamegraphNode {
    /// Create a new node
    pub fn new(name: String, depth: usize) -> Self {
        Self {
            name,
            total_time: Duration::ZERO,
            self_time: Duration::ZERO,
            invocation_count: 0,
            total_memory: 0,
            children: IndexMap::new(),
            depth,
        }
    }

    /// Add time to this node
    pub fn add_time(&mut self, duration: Duration) {
        self.total_time += duration;
        self.self_time += duration;
        self.invocation_count += 1;
    }

    /// Add memory to this node
    pub fn add_memory(&mut self, bytes: usize) {
        self.total_memory += bytes;
    }

    /// Get or create a child node
    pub fn get_or_create_child(&mut self, name: String) -> &mut FlamegraphNode {
        let depth = self.depth + 1;
        self.children
            .entry(name.clone())
            .or_insert_with(|| FlamegraphNode::new(name, depth))
    }

    /// Calculate percentage of total time
    pub fn percentage(&self, total: Duration) -> f64 {
        if total.is_zero() {
            0.0
        } else {
            (self.total_time.as_secs_f64() / total.as_secs_f64()) * 100.0
        }
    }

    /// Get total node count (including children)
    pub fn node_count(&self) -> usize {
        1 + self
            .children
            .values()
            .map(|c| c.node_count())
            .sum::<usize>()
    }
}

/// Flamegraph configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlamegraphConfig {
    /// Minimum percentage to display a node (filter out small nodes)
    pub min_percentage: f64,

    /// Color scheme
    pub color_scheme: ColorScheme,

    /// Whether to include memory information
    pub include_memory: bool,

    /// Width of the flamegraph (in pixels)
    pub width: usize,

    /// Height per row (in pixels)
    pub row_height: usize,

    /// Font size
    pub font_size: usize,

    /// Whether to reverse the stack (icicle graph)
    pub reverse: bool,
}

impl Default for FlamegraphConfig {
    fn default() -> Self {
        Self {
            min_percentage: 0.1,
            color_scheme: ColorScheme::Time,
            include_memory: true,
            width: 1200,
            row_height: 20,
            font_size: 12,
            reverse: false,
        }
    }
}

/// Color scheme for flamegraph
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ColorScheme {
    /// Color by execution time
    Time,
    /// Color by memory usage
    Memory,
    /// Color by operation type
    OperationType,
    /// Gradient from blue to red
    Gradient,
}

/// Flamegraph builder
pub struct FlamegraphBuilder {
    /// Root node
    root: FlamegraphNode,

    /// Configuration
    config: FlamegraphConfig,

    /// Total execution time
    total_time: Duration,
}

impl FlamegraphBuilder {
    /// Create a new flamegraph builder
    pub fn new(config: FlamegraphConfig) -> Self {
        Self {
            root: FlamegraphNode::new("root".to_string(), 0),
            config,
            total_time: Duration::ZERO,
        }
    }

    /// Build flamegraph from a gradient path
    pub fn from_gradient_path(path: &GradientPath, config: FlamegraphConfig) -> Self {
        let mut builder = Self::new(config);
        builder.process_path(path);
        builder
    }

    /// Process a gradient path
    fn process_path(&mut self, path: &GradientPath) {
        // Build call stacks from trace events
        let mut call_stack: Vec<&TraceEvent> = Vec::new();
        let mut event_map: HashMap<u64, &TraceEvent> = HashMap::new();

        // Index events
        for event in &path.events {
            event_map.insert(event.id, event);
        }

        // Process events in chronological order
        for event in &path.events {
            // Build call stack for this event
            call_stack.clear();
            call_stack.push(event);

            let mut current_id = event.parent_id;
            while let Some(parent_id) = current_id {
                if let Some(parent) = event_map.get(&parent_id) {
                    call_stack.push(parent);
                    current_id = parent.parent_id;
                } else {
                    break;
                }
            }

            // Reverse to get root-to-leaf order
            call_stack.reverse();

            // Add to flamegraph
            if let Some(duration) = event.duration {
                self.total_time = self.total_time.max(duration);

                let mut current_node = &mut self.root;

                for stack_event in &call_stack {
                    let name = stack_event.operation.clone();
                    current_node = current_node.get_or_create_child(name);
                }

                current_node.add_time(duration);

                if let Some(mem) = event.memory_allocated {
                    current_node.add_memory(mem);
                }
            }
        }

        Self::calculate_self_time(&mut self.root);
    }

    /// Calculate self time (excluding children)
    fn calculate_self_time(node: &mut FlamegraphNode) {
        let children_time: Duration = node.children.values().map(|c| c.total_time).sum();

        node.self_time = node.total_time.saturating_sub(children_time);

        // Process children separately to avoid borrow checker issues
        let child_keys: Vec<_> = node.children.keys().cloned().collect();
        for key in child_keys {
            if let Some(child) = node.children.get_mut(&key) {
                Self::calculate_self_time(child);
            }
        }
    }

    /// Generate SVG flamegraph
    pub fn to_svg(&self) -> AutogradResult<String> {
        let mut svg = String::new();

        // Calculate total height
        let total_height = (self.root.node_count() * self.config.row_height) + 50;

        // SVG header
        svg.push_str(&format!(
            r#"<svg version="1.1" width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
            self.config.width, total_height
        ));
        svg.push_str("\n");

        // Title
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"20\" font-family=\"Verdana\" font-size=\"16\" fill=\"#000\">Autograd Flamegraph (Total: {:.3}ms)</text>",
            self.config.width / 2 - 150,
            self.total_time.as_secs_f64() * 1000.0
        ));
        svg.push_str("\n");

        // Render nodes
        let mut y_offset = 30;
        self.render_node_svg(
            &self.root,
            0.0,
            self.config.width as f64,
            &mut y_offset,
            &mut svg,
        );

        svg.push_str("</svg>\n");

        Ok(svg)
    }

    /// Render a node as SVG
    fn render_node_svg(
        &self,
        node: &FlamegraphNode,
        x: f64,
        width: f64,
        y_offset: &mut usize,
        svg: &mut String,
    ) {
        if node.name == "root" {
            // Skip root node, render children directly
            for (_, child) in &node.children {
                let child_width =
                    width * (child.total_time.as_secs_f64() / node.total_time.as_secs_f64());
                self.render_node_svg(child, x, child_width, y_offset, svg);
            }
            return;
        }

        let percentage = node.percentage(self.total_time);

        // Filter out small nodes
        if percentage < self.config.min_percentage {
            return;
        }

        let y = *y_offset;

        // Choose color based on scheme
        let color = self.get_color(node, percentage);

        // Draw rectangle
        svg.push_str(&format!(
            "<rect x=\"{:.2}\" y=\"{}\" width=\"{:.2}\" height=\"{}\" fill=\"{}\" stroke=\"#FFF\" stroke-width=\"0.5\">",
            x, y, width, self.config.row_height, color
        ));
        svg.push_str(&format!(
            "<title>{} ({:.2}%, {:.3}ms, {} calls{})</title>",
            node.name,
            percentage,
            node.total_time.as_secs_f64() * 1000.0,
            node.invocation_count,
            if self.config.include_memory {
                format!(", {} bytes", node.total_memory)
            } else {
                String::new()
            }
        ));
        svg.push_str("</rect>\n");

        // Draw text
        if width > 50.0 {
            // Truncate text if too long
            let max_chars = (width / 8.0) as usize;
            let text = if node.name.len() > max_chars {
                format!("{}...", &node.name[..max_chars.saturating_sub(3)])
            } else {
                node.name.clone()
            };

            svg.push_str(&format!(
                "<text x=\"{:.2}\" y=\"{}\" font-family=\"Verdana\" font-size=\"{}\" fill=\"#000\">{}</text>",
                x + 5.0,
                y + self.config.row_height / 2 + self.config.font_size / 3,
                self.config.font_size,
                text
            ));
            svg.push_str("\n");
        }

        // Render children
        *y_offset += self.config.row_height;

        let mut child_x = x;
        for (_, child) in &node.children {
            let child_width =
                width * (child.total_time.as_secs_f64() / node.total_time.as_secs_f64());
            self.render_node_svg(child, child_x, child_width, y_offset, svg);
            child_x += child_width;
        }
    }

    /// Get color for a node based on color scheme
    fn get_color(&self, node: &FlamegraphNode, percentage: f64) -> String {
        match self.config.color_scheme {
            ColorScheme::Time => {
                // Gradient from green (fast) to red (slow)
                let intensity = (percentage / 100.0).min(1.0);
                let r = (255.0 * intensity) as u8;
                let g = (255.0 * (1.0 - intensity)) as u8;
                format!("rgb({}, {}, 50)", r, g)
            }
            ColorScheme::Memory => {
                // Gradient based on memory usage
                let max_memory = 1_000_000.0; // 1MB
                let intensity = (node.total_memory as f64 / max_memory).min(1.0);
                let r = (255.0 * intensity) as u8;
                let b = (255.0 * (1.0 - intensity)) as u8;
                format!("rgb({}, 100, {})", r, b)
            }
            ColorScheme::OperationType => {
                // Different colors for different operation types
                let hash = node
                    .name
                    .bytes()
                    .fold(0u32, |acc, b| acc.wrapping_add(b as u32));
                let hue = (hash % 360) as f64;
                format!("hsl({}, 70%, 60%)", hue)
            }
            ColorScheme::Gradient => {
                // Blue to red gradient by depth
                let max_depth = 10.0;
                let intensity = (node.depth as f64 / max_depth).min(1.0);
                let r = (255.0 * intensity) as u8;
                let b = (255.0 * (1.0 - intensity)) as u8;
                format!("rgb({}, 100, {})", r, b)
            }
        }
    }

    /// Generate text-based flamegraph
    pub fn to_text(&self) -> String {
        let mut output = String::new();
        self.render_node_text(&self.root, 0, &mut output);
        output
    }

    /// Render a node as text
    fn render_node_text(&self, node: &FlamegraphNode, indent: usize, output: &mut String) {
        if node.name != "root" {
            let percentage = node.percentage(self.total_time);
            output.push_str(&format!(
                "{}{} ({:.2}%, {:.3}ms, {} calls",
                "  ".repeat(indent),
                node.name,
                percentage,
                node.total_time.as_secs_f64() * 1000.0,
                node.invocation_count
            ));

            if self.config.include_memory && node.total_memory > 0 {
                output.push_str(&format!(", {} bytes", node.total_memory));
            }

            output.push_str(")\n");
        }

        let mut children: Vec<_> = node.children.values().collect();
        children.sort_by(|a, b| b.total_time.cmp(&a.total_time));

        for child in children {
            self.render_node_text(child, indent + 1, output);
        }
    }

    /// Generate collapsed format (for external flamegraph tools)
    pub fn to_collapsed(&self) -> String {
        let mut output = String::new();
        let mut stack = Vec::new();
        self.render_collapsed(&self.root, &mut stack, &mut output);
        output
    }

    /// Render collapsed format
    fn render_collapsed(
        &self,
        node: &FlamegraphNode,
        stack: &mut Vec<String>,
        output: &mut String,
    ) {
        if node.name != "root" {
            stack.push(node.name.clone());
        }

        if node.children.is_empty() && !stack.is_empty() {
            output.push_str(&stack.join(";"));
            output.push_str(&format!(" {}\n", node.invocation_count));
        }

        for (_, child) in &node.children {
            self.render_collapsed(child, stack, output);
        }

        if node.name != "root" {
            stack.pop();
        }
    }

    /// Get root node
    pub fn root(&self) -> &FlamegraphNode {
        &self.root
    }
}

/// Per-frame difference between two flamegraphs (new minus old).
///
/// A "frame" is identified by its full root-to-frame stack path (the segments
/// joined by `";"`, matching the collapsed flamegraph format), so two boxes
/// that share an operation name but sit under different ancestors are treated
/// as distinct frames.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameDelta {
    /// Fully-qualified frame identity (root-to-frame stack path, `";"`-joined).
    pub frame: String,

    /// Change in self-time, new minus old, in nanoseconds (may be negative).
    pub self_time_delta_ns: i128,

    /// Change in total-time, new minus old, in nanoseconds (may be negative).
    pub total_time_delta_ns: i128,

    /// Self-time of the frame in the old (baseline) graph.
    pub old_self_time: Duration,

    /// Self-time of the frame in the new (comparison) graph.
    pub new_self_time: Duration,

    /// Total-time of the frame in the old (baseline) graph.
    pub old_total_time: Duration,

    /// Total-time of the frame in the new (comparison) graph.
    pub new_total_time: Duration,
}

/// Result of comparing two flamegraphs frame-by-frame.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FlamegraphComparison {
    /// Delta for every frame present in either graph, keyed by frame identity.
    ///
    /// Frames present in only one graph use [`Duration::ZERO`] for the missing
    /// side, so their delta carries the full magnitude with the correct sign.
    pub frame_deltas: IndexMap<String, FrameDelta>,

    /// Frames present only in the new (comparison) graph.
    pub appeared: Vec<String>,

    /// Frames present only in the old (baseline) graph.
    pub disappeared: Vec<String>,
}

/// Differential flamegraph comparing two runs
pub struct DifferentialFlamegraph {
    baseline: FlamegraphBuilder,
    comparison: FlamegraphBuilder,
}

impl DifferentialFlamegraph {
    /// Create a new differential flamegraph
    pub fn new(baseline: FlamegraphBuilder, comparison: FlamegraphBuilder) -> Self {
        Self {
            baseline,
            comparison,
        }
    }

    /// Generate differential report
    pub fn report(&self) -> String {
        let mut output = String::new();

        output.push_str("=== Differential Flamegraph Report ===\n\n");

        output.push_str(&format!(
            "Baseline total time: {:.3}ms\n",
            self.baseline.total_time.as_secs_f64() * 1000.0
        ));
        output.push_str(&format!(
            "Comparison total time: {:.3}ms\n",
            self.comparison.total_time.as_secs_f64() * 1000.0
        ));

        let diff_ms = (self.comparison.total_time.as_secs_f64()
            - self.baseline.total_time.as_secs_f64())
            * 1000.0;
        let diff_pct = if !self.baseline.total_time.is_zero() {
            (diff_ms / (self.baseline.total_time.as_secs_f64() * 1000.0)) * 100.0
        } else {
            0.0
        };

        output.push_str(&format!(
            "\nDifference: {:.3}ms ({:+.2}%)\n\n",
            diff_ms, diff_pct
        ));

        output.push_str("Top changed operations:\n");

        let comparison = self.compare();

        // Order frames by the magnitude of their total-time change so the most
        // significant regressions/improvements surface first.
        let mut ranked: Vec<&FrameDelta> = comparison.frame_deltas.values().collect();
        ranked.sort_by_key(|delta| std::cmp::Reverse(delta.total_time_delta_ns.abs()));

        for delta in ranked.iter().take(10) {
            let total_delta_ms = delta.total_time_delta_ns as f64 / 1_000_000.0;
            let self_delta_ms = delta.self_time_delta_ns as f64 / 1_000_000.0;
            let tag = if comparison.appeared.contains(&delta.frame) {
                " [APPEARED]"
            } else if comparison.disappeared.contains(&delta.frame) {
                " [DISAPPEARED]"
            } else {
                ""
            };
            output.push_str(&format!(
                "  {}{}: total {:+.3}ms, self {:+.3}ms\n",
                delta.frame, tag, total_delta_ms, self_delta_ms
            ));
        }

        if !comparison.appeared.is_empty() {
            output.push_str(&format!(
                "\nAppeared ({}): {}\n",
                comparison.appeared.len(),
                comparison.appeared.join(", ")
            ));
        }
        if !comparison.disappeared.is_empty() {
            output.push_str(&format!(
                "Disappeared ({}): {}\n",
                comparison.disappeared.len(),
                comparison.disappeared.join(", ")
            ));
        }

        output
    }

    /// Compare the baseline and comparison flamegraphs frame-by-frame.
    ///
    /// Every frame present in either graph receives a [`FrameDelta`] recording
    /// the change in self-time and total-time (new minus old). Frames present
    /// only in the new graph are reported in [`FlamegraphComparison::appeared`]
    /// and frames present only in the old graph in
    /// [`FlamegraphComparison::disappeared`].
    pub fn compare(&self) -> FlamegraphComparison {
        let old_frames = Self::flatten_frames(self.baseline.root());
        let new_frames = Self::flatten_frames(self.comparison.root());

        let mut frame_deltas = IndexMap::new();
        let mut appeared = Vec::new();
        let mut disappeared = Vec::new();

        // Walk the new graph first so its frames keep their traversal order,
        // then append any frames that exist only in the old graph.
        for (frame, &(new_self, new_total)) in &new_frames {
            let (old_self, old_total) = old_frames
                .get(frame)
                .copied()
                .unwrap_or((Duration::ZERO, Duration::ZERO));

            if !old_frames.contains_key(frame) {
                appeared.push(frame.clone());
            }

            frame_deltas.insert(
                frame.clone(),
                FrameDelta {
                    frame: frame.clone(),
                    self_time_delta_ns: new_self.as_nanos() as i128 - old_self.as_nanos() as i128,
                    total_time_delta_ns: new_total.as_nanos() as i128
                        - old_total.as_nanos() as i128,
                    old_self_time: old_self,
                    new_self_time: new_self,
                    old_total_time: old_total,
                    new_total_time: new_total,
                },
            );
        }

        for (frame, &(old_self, old_total)) in &old_frames {
            if new_frames.contains_key(frame) {
                continue;
            }

            disappeared.push(frame.clone());
            frame_deltas.insert(
                frame.clone(),
                FrameDelta {
                    frame: frame.clone(),
                    self_time_delta_ns: -(old_self.as_nanos() as i128),
                    total_time_delta_ns: -(old_total.as_nanos() as i128),
                    old_self_time: old_self,
                    new_self_time: Duration::ZERO,
                    old_total_time: old_total,
                    new_total_time: Duration::ZERO,
                },
            );
        }

        FlamegraphComparison {
            frame_deltas,
            appeared,
            disappeared,
        }
    }

    /// Flatten a flamegraph tree into `frame path -> (self_time, total_time)`.
    ///
    /// The root sentinel node is skipped; every other node contributes one
    /// entry keyed by its `";"`-joined stack path.
    fn flatten_frames(root: &FlamegraphNode) -> IndexMap<String, (Duration, Duration)> {
        let mut frames = IndexMap::new();
        let mut stack: Vec<String> = Vec::new();
        Self::collect_frames(root, &mut stack, &mut frames);
        frames
    }

    /// Depth-first helper that accumulates frame times into `frames`.
    fn collect_frames(
        node: &FlamegraphNode,
        stack: &mut Vec<String>,
        frames: &mut IndexMap<String, (Duration, Duration)>,
    ) {
        let is_root = node.name == "root" && node.depth == 0;

        if !is_root {
            stack.push(node.name.clone());
            let key = stack.join(";");
            let entry = frames
                .entry(key)
                .or_insert((Duration::ZERO, Duration::ZERO));
            entry.0 += node.self_time;
            entry.1 += node.total_time;
        }

        for child in node.children.values() {
            Self::collect_frames(child, stack, frames);
        }

        if !is_root {
            stack.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_node() -> FlamegraphNode {
        let mut root = FlamegraphNode::new("root".to_string(), 0);
        root.add_time(Duration::from_millis(100));

        let mut child1 = FlamegraphNode::new("matmul".to_string(), 1);
        child1.add_time(Duration::from_millis(60));

        let mut child2 = FlamegraphNode::new("add".to_string(), 1);
        child2.add_time(Duration::from_millis(40));

        root.children.insert("matmul".to_string(), child1);
        root.children.insert("add".to_string(), child2);

        root
    }

    #[test]
    fn test_node_creation() {
        let node = FlamegraphNode::new("test".to_string(), 0);
        assert_eq!(node.name, "test");
        assert_eq!(node.invocation_count, 0);
        assert_eq!(node.depth, 0);
    }

    #[test]
    fn test_percentage_calculation() {
        let node = create_test_node();
        let total = Duration::from_millis(100);

        assert_eq!(node.percentage(total), 100.0);

        let child = node.children.get("matmul").unwrap();
        assert_eq!(child.percentage(total), 60.0);
    }

    #[test]
    fn test_node_count() {
        let node = create_test_node();
        assert_eq!(node.node_count(), 3); // root + 2 children
    }

    #[test]
    fn test_text_output() {
        let mut builder = FlamegraphBuilder::new(FlamegraphConfig::default());
        builder.root = create_test_node();
        builder.total_time = Duration::from_millis(100);

        let text = builder.to_text();
        assert!(text.contains("matmul"));
        assert!(text.contains("add"));
    }

    #[test]
    fn test_collapsed_format() {
        let mut builder = FlamegraphBuilder::new(FlamegraphConfig::default());
        builder.root = create_test_node();

        let collapsed = builder.to_collapsed();
        assert!(collapsed.contains("matmul"));
    }

    #[test]
    fn test_svg_generation() {
        let mut builder = FlamegraphBuilder::new(FlamegraphConfig::default());
        builder.root = create_test_node();
        builder.total_time = Duration::from_millis(100);

        let svg = builder.to_svg();
        assert!(svg.is_ok());

        let svg_str = svg.unwrap();
        assert!(svg_str.contains("<svg"));
        assert!(svg_str.contains("</svg>"));
    }

    /// Build a single frame node with explicit self/total time (in ms).
    fn make_frame(name: &str, depth: usize, self_ms: u64, total_ms: u64) -> FlamegraphNode {
        let mut node = FlamegraphNode::new(name.to_string(), depth);
        node.self_time = Duration::from_millis(self_ms);
        node.total_time = Duration::from_millis(total_ms);
        node.invocation_count = 1;
        node
    }

    fn builder_with_root(root: FlamegraphNode, total_ms: u64) -> FlamegraphBuilder {
        let mut builder = FlamegraphBuilder::new(FlamegraphConfig::default());
        builder.root = root;
        builder.total_time = Duration::from_millis(total_ms);
        builder
    }

    #[test]
    fn test_compare_flamegraphs_frame_deltas() {
        // --- OLD (baseline) graph ----------------------------------------
        // root
        //   forward   self=20ms total=100ms
        //     matmul  self=60ms total=60ms
        //     add     self=20ms total=20ms   (DISAPPEARS)
        //   backward  self=50ms total=50ms   (DISAPPEARS)
        let mut old_forward = make_frame("forward", 1, 20, 100);
        old_forward
            .children
            .insert("matmul".to_string(), make_frame("matmul", 2, 60, 60));
        old_forward
            .children
            .insert("add".to_string(), make_frame("add", 2, 20, 20));

        let mut old_root = FlamegraphNode::new("root".to_string(), 0);
        old_root.children.insert("forward".to_string(), old_forward);
        old_root
            .children
            .insert("backward".to_string(), make_frame("backward", 1, 50, 50));

        // --- NEW (comparison) graph --------------------------------------
        // root
        //   forward   self=40ms total=120ms
        //     matmul  self=50ms total=50ms   (got faster: -10ms)
        //     relu    self=30ms total=30ms   (APPEARS)
        //   optimizer self=10ms total=10ms   (APPEARS)
        let mut new_forward = make_frame("forward", 1, 40, 120);
        new_forward
            .children
            .insert("matmul".to_string(), make_frame("matmul", 2, 50, 50));
        new_forward
            .children
            .insert("relu".to_string(), make_frame("relu", 2, 30, 30));

        let mut new_root = FlamegraphNode::new("root".to_string(), 0);
        new_root.children.insert("forward".to_string(), new_forward);
        new_root
            .children
            .insert("optimizer".to_string(), make_frame("optimizer", 1, 10, 10));

        let diff = DifferentialFlamegraph::new(
            builder_with_root(old_root, 150),
            builder_with_root(new_root, 130),
        );
        let result = diff.compare();

        const MS: i128 = 1_000_000; // nanoseconds per millisecond

        // Frame present in BOTH graphs: forward grew by +20ms (self and total).
        let forward = result
            .frame_deltas
            .get("forward")
            .expect("forward frame should be present");
        assert_eq!(forward.self_time_delta_ns, 20 * MS);
        assert_eq!(forward.total_time_delta_ns, 20 * MS);
        assert_eq!(forward.old_total_time, Duration::from_millis(100));
        assert_eq!(forward.new_total_time, Duration::from_millis(120));

        // Frame present in BOTH graphs that got FASTER: matmul (-10ms).
        let matmul = result
            .frame_deltas
            .get("forward;matmul")
            .expect("forward;matmul frame should be present");
        assert_eq!(matmul.self_time_delta_ns, -10 * MS);
        assert_eq!(matmul.total_time_delta_ns, -10 * MS);

        // APPEARED set must be exactly {forward;relu, optimizer}.
        let mut appeared = result.appeared.clone();
        appeared.sort();
        assert_eq!(
            appeared,
            vec!["forward;relu".to_string(), "optimizer".to_string()]
        );

        // DISAPPEARED set must be exactly {backward, forward;add}.
        let mut disappeared = result.disappeared.clone();
        disappeared.sort();
        assert_eq!(
            disappeared,
            vec!["backward".to_string(), "forward;add".to_string()]
        );

        // Appeared frames carry the full positive magnitude (old side is zero).
        let relu = result
            .frame_deltas
            .get("forward;relu")
            .expect("relu frame should be present");
        assert_eq!(relu.self_time_delta_ns, 30 * MS);
        assert_eq!(relu.total_time_delta_ns, 30 * MS);
        assert_eq!(relu.old_self_time, Duration::ZERO);
        assert_eq!(relu.new_self_time, Duration::from_millis(30));

        // Disappeared frames carry the full negative magnitude (new side zero).
        let backward = result
            .frame_deltas
            .get("backward")
            .expect("backward frame should be present");
        assert_eq!(backward.self_time_delta_ns, -50 * MS);
        assert_eq!(backward.total_time_delta_ns, -50 * MS);
        assert_eq!(backward.new_self_time, Duration::ZERO);
        assert_eq!(backward.old_self_time, Duration::from_millis(50));

        // Every frame from either graph is accounted for (4 shared/changed +
        // 2 appeared + 2 disappeared = 6 distinct frame paths).
        assert_eq!(result.frame_deltas.len(), 6);

        // The textual report should surface the appeared/disappeared frames.
        let report = diff.report();
        assert!(report.contains("[APPEARED]"));
        assert!(report.contains("[DISAPPEARED]"));
        assert!(report.contains("optimizer"));
        assert!(report.contains("backward"));
    }
}
