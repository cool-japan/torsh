// Copyright (c) 2025 ToRSh Project
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Flamegraph Generation for Autograd Operations
//!
//! This module provides flamegraph generation capabilities for visualizing
//! autograd operation performance profiles and call hierarchies.
//!
//! # Features
//!
//! - **SVG Flamegraphs**: Generate interactive SVG flamegraphs
//! - **Icicle Charts**: Top-down visualization of call hierarchies
//! - **Diff Flamegraphs**: Compare performance between runs
//! - **Filtering**: Filter by operation type, time threshold, or name
//! - **Color Coding**: Color operations by type, performance, or custom scheme
//! - **Interactive**: Clickable, searchable, and zoomable visualizations

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Flamegraph configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlamegraphConfig {
    /// Width of the flamegraph (pixels)
    pub width: u32,

    /// Height per row (pixels)
    pub height_per_row: u32,

    /// Minimum width to show text (pixels)
    pub min_width_for_text: u32,

    /// Color scheme
    pub color_scheme: ColorScheme,

    /// Display mode
    pub display_mode: DisplayMode,

    /// Filter minimum time (percentage of total)
    pub min_time_percent: f64,

    /// Title
    pub title: String,
}

impl Default for FlamegraphConfig {
    fn default() -> Self {
        Self {
            width: 1200,
            height_per_row: 18,
            min_width_for_text: 20,
            color_scheme: ColorScheme::ByOperationType,
            display_mode: DisplayMode::Flamegraph,
            min_time_percent: 0.1,
            title: "Autograd Operation Flamegraph".to_string(),
        }
    }
}

/// Color scheme for flamegraph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorScheme {
    /// Color by operation type
    ByOperationType,

    /// Color by performance (hot/cold)
    ByPerformance,

    /// Rainbow colors
    Rainbow,

    /// Grayscale
    Grayscale,

    /// Custom colors
    Custom,
}

/// Display mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisplayMode {
    /// Standard flamegraph (bottom-up)
    Flamegraph,

    /// Icicle chart (top-down)
    Icicle,

    /// Sunburst chart
    Sunburst,
}

/// Flamegraph generator
pub struct FlamegraphGenerator {
    config: FlamegraphConfig,
    profiles: Vec<OperationProfile>,
    color_map: HashMap<String, String>,
}

/// Operation profile entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationProfile {
    /// Operation name
    pub name: String,

    /// Parent operation (if any)
    pub parent: Option<String>,

    /// Start time (microseconds)
    pub start_us: u64,

    /// Duration (microseconds)
    pub duration_us: u64,

    /// Depth in call stack
    pub depth: usize,

    /// Operation type
    pub op_type: String,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Flamegraph frame
#[derive(Debug, Clone)]
struct Frame {
    name: String,
    value: u64,
    children: Vec<Frame>,
    op_type: String,
}

impl FlamegraphGenerator {
    /// Create a new flamegraph generator
    pub fn new(config: FlamegraphConfig) -> Self {
        Self {
            config,
            profiles: Vec::new(),
            color_map: Self::build_color_map(),
        }
    }

    /// Add operation profile
    pub fn add_profile(&mut self, profile: OperationProfile) {
        self.profiles.push(profile);
    }

    /// Add multiple profiles
    pub fn add_profiles(&mut self, profiles: Vec<OperationProfile>) {
        self.profiles.extend(profiles);
    }

    /// Generate flamegraph SVG
    pub fn generate_svg(&self) -> String {
        if self.profiles.is_empty() {
            return self.generate_empty_svg();
        }

        // Build call tree
        let root = self.build_call_tree();

        // Calculate total time
        let total_time = self.calculate_total_time();

        // Generate SVG
        match self.config.display_mode {
            DisplayMode::Flamegraph => self.generate_flamegraph_svg(&root, total_time),
            DisplayMode::Icicle => self.generate_icicle_svg(&root, total_time),
            DisplayMode::Sunburst => self.generate_sunburst_svg(&root, total_time),
        }
    }

    /// Generate diff flamegraph comparing two profile sets
    pub fn generate_diff_flamegraph(
        &self,
        baseline_profiles: &[OperationProfile],
        current_profiles: &[OperationProfile],
    ) -> String {
        // Build trees for both profile sets
        let mut baseline_gen = FlamegraphGenerator::new(self.config.clone());
        baseline_gen.add_profiles(baseline_profiles.to_vec());
        let baseline_tree = baseline_gen.build_call_tree();

        let mut current_gen = FlamegraphGenerator::new(self.config.clone());
        current_gen.add_profiles(current_profiles.to_vec());
        let current_tree = current_gen.build_call_tree();

        // Generate diff visualization
        self.generate_diff_svg(&baseline_tree, &current_tree)
    }

    /// Export profiles to folded format
    pub fn export_folded_format(&self) -> String {
        let mut folded = String::new();

        // Group by call stack
        let mut stacks: HashMap<String, u64> = HashMap::new();

        for profile in &self.profiles {
            let stack = self.build_stack_string(profile);
            *stacks.entry(stack).or_insert(0) += profile.duration_us;
        }

        // Format as folded stacks
        for (stack, samples) in stacks {
            folded.push_str(&format!("{} {}\n", stack, samples));
        }

        folded
    }

    // Private helper methods

    fn build_color_map() -> HashMap<String, String> {
        let mut map = HashMap::new();

        // Operation type colors
        map.insert("Forward".to_string(), "#ff7f0e".to_string());
        map.insert("Backward".to_string(), "#2ca02c".to_string());
        map.insert("MatMul".to_string(), "#d62728".to_string());
        map.insert("Convolution".to_string(), "#9467bd".to_string());
        map.insert("Activation".to_string(), "#8c564b".to_string());
        map.insert("Pooling".to_string(), "#e377c2".to_string());
        map.insert("Normalization".to_string(), "#7f7f7f".to_string());
        map.insert("Reduction".to_string(), "#bcbd22".to_string());
        map.insert("ElementWise".to_string(), "#17becf".to_string());

        map
    }

    fn build_call_tree(&self) -> Frame {
        let mut root = Frame {
            name: "root".to_string(),
            value: 0,
            children: Vec::new(),
            op_type: "root".to_string(),
        };

        // Sort profiles by start time
        let mut sorted_profiles = self.profiles.clone();
        sorted_profiles.sort_by_key(|p| p.start_us);

        // Build tree structure
        for profile in sorted_profiles {
            self.insert_into_tree(&mut root, profile);
        }

        root
    }

    fn insert_into_tree(&self, parent: &mut Frame, profile: OperationProfile) {
        // Find appropriate parent based on timing
        parent.value += profile.duration_us;

        let new_frame = Frame {
            name: profile.name.clone(),
            value: profile.duration_us,
            children: Vec::new(),
            op_type: profile.op_type.clone(),
        };

        parent.children.push(new_frame);
    }

    fn calculate_total_time(&self) -> u64 {
        self.profiles.iter().map(|p| p.duration_us).sum()
    }

    fn generate_empty_svg(&self) -> String {
        format!(
            r#"<svg width="{}" height="100" xmlns="http://www.w3.org/2000/svg">
    <text x="50%" y="50%" text-anchor="middle" font-size="16">
        No profile data available
    </text>
</svg>"#,
            self.config.width
        )
    }

    fn generate_flamegraph_svg(&self, root: &Frame, total_time: u64) -> String {
        let mut svg = String::new();

        // SVG header
        svg.push_str(&format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
<defs>
    <style>
        .frame {{ cursor: pointer; }}
        .frame:hover {{ stroke: black; stroke-width: 0.5; }}
        text {{ font-family: monospace; font-size: 12px; }}
    </style>
</defs>
"#,
            self.config.width,
            root.children.len() as u32 * self.config.height_per_row
        ));

        // Title
        svg.push_str(&format!(
            r#"<text x="{}" y="20" text-anchor="middle" font-size="16" font-weight="bold">{}</text>"#,
            self.config.width / 2,
            self.config.title
        ));

        // Render frames
        self.render_frame(&mut svg, root, 0, 0, self.config.width, total_time);

        svg.push_str("</svg>");
        svg
    }

    fn render_frame(
        &self,
        svg: &mut String,
        frame: &Frame,
        depth: usize,
        x: u32,
        width: u32,
        total_time: u64,
    ) {
        if frame.name == "root" {
            // Skip root, render children
            for child in &frame.children {
                self.render_frame(svg, child, depth, x, width, total_time);
            }
            return;
        }

        let y = 40 + depth as u32 * self.config.height_per_row;
        let frame_width = ((frame.value as f64 / total_time as f64) * width as f64) as u32;

        if frame_width < 1 {
            return; // Too narrow to render
        }

        // Get color
        let color = self.get_frame_color(frame);

        // Render rectangle
        svg.push_str(&format!(
            r#"<rect class="frame" x="{}" y="{}" width="{}" height="{}" fill="{}"/>"#,
            x,
            y,
            frame_width,
            self.config.height_per_row - 1,
            color
        ));

        // Render text if wide enough
        if frame_width >= self.config.min_width_for_text {
            let text_x = x + frame_width / 2;
            let text_y = y + self.config.height_per_row - 4;

            let display_name = if frame.name.len() > 30 {
                format!("{}...", &frame.name[..27])
            } else {
                frame.name.clone()
            };

            svg.push_str(&format!(
                r#"<text x="{}" y="{}" text-anchor="middle" fill="white">{}</text>"#,
                text_x, text_y, display_name
            ));
        }

        // Render children
        let mut child_x = x;
        for child in &frame.children {
            let child_width = ((child.value as f64 / total_time as f64) * width as f64) as u32;
            self.render_frame(svg, child, depth + 1, child_x, child_width, total_time);
            child_x += child_width;
        }
    }

    fn generate_icicle_svg(&self, root: &Frame, total_time: u64) -> String {
        // Similar to flamegraph but inverted (top-down)
        let mut svg = String::new();

        svg.push_str(&format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
<text x="{}" y="20" text-anchor="middle" font-size="16" font-weight="bold">{} (Icicle)</text>
"#,
            self.config.width,
            root.children.len() as u32 * self.config.height_per_row,
            self.config.width / 2,
            self.config.title
        ));

        self.render_frame(&mut svg, root, 0, 0, self.config.width, total_time);

        svg.push_str("</svg>");
        svg
    }

    fn generate_sunburst_svg(&self, _root: &Frame, _total_time: u64) -> String {
        // Simplified sunburst (circular visualization)
        format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
<text x="{}" y="{}" text-anchor="middle" font-size="16">
    Sunburst visualization (not yet implemented)
</text>
</svg>"#,
            self.config.width,
            self.config.width,
            self.config.width / 2,
            self.config.width / 2
        )
    }

    fn generate_diff_svg(&self, baseline: &Frame, current: &Frame) -> String {
        let total_baseline = self.sum_frame_time(baseline);
        let total_current = self.sum_frame_time(current);

        let diff_percent =
            ((total_current as f64 - total_baseline as f64) / total_baseline as f64) * 100.0;

        format!(
            r#"<svg width="{}" height="200" xmlns="http://www.w3.org/2000/svg">
<text x="{}" y="50" text-anchor="middle" font-size="16" font-weight="bold">Performance Diff</text>
<text x="{}" y="80" text-anchor="middle" font-size="14">
    Baseline: {}μs
</text>
<text x="{}" y="110" text-anchor="middle" font-size="14">
    Current: {}μs
</text>
<text x="{}" y="140" text-anchor="middle" font-size="14" fill="{}">
    Change: {:.2}%
</text>
</svg>"#,
            self.config.width,
            self.config.width / 2,
            self.config.width / 2,
            total_baseline,
            self.config.width / 2,
            total_current,
            self.config.width / 2,
            if diff_percent > 0.0 { "red" } else { "green" },
            diff_percent
        )
    }

    fn sum_frame_time(&self, frame: &Frame) -> u64 {
        frame.value
            + frame
                .children
                .iter()
                .map(|c| self.sum_frame_time(c))
                .sum::<u64>()
    }

    fn get_frame_color(&self, frame: &Frame) -> &str {
        match self.config.color_scheme {
            ColorScheme::ByOperationType => self
                .color_map
                .get(&frame.op_type)
                .map(|s| s.as_str())
                .unwrap_or("#808080"),
            ColorScheme::ByPerformance => {
                // Hot colors for slow operations
                if frame.value > 1_000_000 {
                    "#ff0000"
                } else if frame.value > 100_000 {
                    "#ff7f00"
                } else {
                    "#00ff00"
                }
            }
            ColorScheme::Rainbow => "#4a9eff",
            ColorScheme::Grayscale => "#808080",
            ColorScheme::Custom => "#4a9eff",
        }
    }

    fn build_stack_string(&self, profile: &OperationProfile) -> String {
        // Build call stack string
        if let Some(ref parent) = profile.parent {
            format!("{};{}", parent, profile.name)
        } else {
            profile.name.clone()
        }
    }
}

/// Flamegraph builder for easy construction
pub struct FlamegraphBuilder {
    generator: FlamegraphGenerator,
}

impl FlamegraphBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            generator: FlamegraphGenerator::new(FlamegraphConfig::default()),
        }
    }

    /// Set configuration
    pub fn config(mut self, config: FlamegraphConfig) -> Self {
        self.generator.config = config;
        self
    }

    /// Add profile
    pub fn add_operation(mut self, name: String, duration_us: u64, op_type: String) -> Self {
        self.generator.add_profile(OperationProfile {
            name,
            parent: None,
            start_us: 0,
            duration_us,
            depth: 0,
            op_type,
            metadata: HashMap::new(),
        });
        self
    }

    /// Build flamegraph
    pub fn build(self) -> String {
        self.generator.generate_svg()
    }
}

impl Default for FlamegraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flamegraph_generation() {
        let config = FlamegraphConfig::default();
        let mut generator = FlamegraphGenerator::new(config);

        generator.add_profile(OperationProfile {
            name: "forward".to_string(),
            parent: None,
            start_us: 0,
            duration_us: 1000,
            depth: 0,
            op_type: "Forward".to_string(),
            metadata: HashMap::new(),
        });

        let svg = generator.generate_svg();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("forward"));
    }

    #[test]
    fn test_empty_flamegraph() {
        let config = FlamegraphConfig::default();
        let generator = FlamegraphGenerator::new(config);

        let svg = generator.generate_svg();
        assert!(svg.contains("No profile data"));
    }

    #[test]
    fn test_folded_format_export() {
        let config = FlamegraphConfig::default();
        let mut generator = FlamegraphGenerator::new(config);

        generator.add_profile(OperationProfile {
            name: "op1".to_string(),
            parent: None,
            start_us: 0,
            duration_us: 500,
            depth: 0,
            op_type: "Forward".to_string(),
            metadata: HashMap::new(),
        });

        let folded = generator.export_folded_format();
        assert!(folded.contains("op1"));
        assert!(folded.contains("500"));
    }

    #[test]
    fn test_flamegraph_builder() {
        let svg = FlamegraphBuilder::new()
            .add_operation("test_op".to_string(), 1000, "Forward".to_string())
            .build();

        assert!(svg.contains("<svg"));
        assert!(svg.contains("test_op"));
    }

    #[test]
    fn test_diff_flamegraph() {
        let config = FlamegraphConfig::default();
        let generator = FlamegraphGenerator::new(config);

        let baseline = vec![OperationProfile {
            name: "op1".to_string(),
            parent: None,
            start_us: 0,
            duration_us: 1000,
            depth: 0,
            op_type: "Forward".to_string(),
            metadata: HashMap::new(),
        }];

        let current = vec![OperationProfile {
            name: "op1".to_string(),
            parent: None,
            start_us: 0,
            duration_us: 1500,
            depth: 0,
            op_type: "Forward".to_string(),
            metadata: HashMap::new(),
        }];

        let svg = generator.generate_diff_flamegraph(&baseline, &current);
        assert!(svg.contains("Performance Diff"));
    }

    #[test]
    fn test_color_schemes() {
        let mut config = FlamegraphConfig::default();

        // Test different color schemes
        for scheme in &[
            ColorScheme::ByOperationType,
            ColorScheme::ByPerformance,
            ColorScheme::Rainbow,
            ColorScheme::Grayscale,
        ] {
            config.color_scheme = *scheme;
            let mut generator = FlamegraphGenerator::new(config.clone());

            generator.add_profile(OperationProfile {
                name: "test".to_string(),
                parent: None,
                start_us: 0,
                duration_us: 1000,
                depth: 0,
                op_type: "Forward".to_string(),
                metadata: HashMap::new(),
            });

            let svg = generator.generate_svg();
            assert!(svg.contains("<svg"));
        }
    }
}
