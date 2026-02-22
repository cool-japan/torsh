//! Graph analysis, metrics, and validation functionality

use crate::fx::types::{GraphStats, MemoryEstimate, Node};
use crate::graph_analysis::{
    calculate_graph_metrics, DetectedPattern, GraphLinter, GraphMetrics, LintReport,
    PatternDetector,
};
use crate::memory_optimization::{MemoryAnalyzer, MemoryUsageReport};
use crate::{FxGraph, TorshResult};
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

impl FxGraph {
    /// Validate that the graph is well-formed
    pub fn validate(&self) -> TorshResult<()> {
        // Check that we have at least one input and one output
        if self.inputs.is_empty() {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Graph must have at least one input".to_string(),
            ));
        }

        if self.outputs.is_empty() {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Graph must have at least one output".to_string(),
            ));
        }

        // Check that all input/output indices are valid
        for &input_idx in &self.inputs {
            if self.graph.node_weight(input_idx).is_none() {
                return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                    "Invalid input node index: {input_idx:?}"
                )));
            }
        }

        for &output_idx in &self.outputs {
            if self.graph.node_weight(output_idx).is_none() {
                return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                    "Invalid output node index: {output_idx:?}"
                )));
            }
        }

        // Check for cycles in non-control-flow parts
        // (This is a simplified check - full cycle detection would need to handle control flow)
        if petgraph::algo::is_cyclic_directed(&self.graph) {
            // Allow cycles if they're part of explicit loop constructs
            let has_explicit_loops = !self.loop_nodes().is_empty();
            if !has_explicit_loops {
                return Err(torsh_core::error::TorshError::InvalidArgument(
                    "Graph contains cycles outside of explicit loop constructs".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Quick validation with detailed error reporting
    pub fn validate_detailed(&self) -> TorshResult<String> {
        // First run basic validation
        self.validate()?;

        let mut report = String::new();
        report.push_str("✅ Graph validation PASSED\n");

        // Additional detailed checks
        let metrics = self.metrics();
        report.push_str(&format!(
            "📊 Graph complexity score: {:.2}\n",
            metrics.complexity_score
        ));

        // Performance recommendations
        if self.node_count() > 1000 {
            report.push_str("💡 Large graph detected - consider using parallel traversal\n");
        }

        if self.get_depth() > 50 {
            report.push_str("💡 Deep graph detected - consider optimization passes\n");
        }

        let op_counts = self.operation_counts();
        if op_counts.len() > 20 {
            report.push_str("💡 Many unique operations - consider operation fusion\n");
        }

        Ok(report)
    }

    /// Quick graph inspection - returns a human-readable diagnostic report
    pub fn inspect(&self) -> String {
        let mut report = String::new();
        report.push_str("=== FX Graph Inspection Report ===\n");
        report.push_str(&format!("Graph ID: {self:p}\n"));
        let node_count = self.node_count();
        report.push_str(&format!("Node Count: {node_count}\n"));
        let edge_count = self.edge_count();
        report.push_str(&format!("Edge Count: {edge_count}\n"));
        let input_count = self.inputs().len();
        report.push_str(&format!("Input Nodes: {input_count}\n"));
        let output_count = self.outputs().len();
        report.push_str(&format!("Output Nodes: {output_count}\n"));

        // Validation status
        match self.validate() {
            Ok(_) => report.push_str("✅ Validation: PASSED\n"),
            Err(e) => report.push_str(&format!("❌ Validation: FAILED - {e}\n")),
        }

        // Operation analysis
        let op_counts = self.operation_counts();
        if !op_counts.is_empty() {
            report.push_str("\n--- Operation Distribution ---\n");
            for (op, count) in op_counts.iter() {
                report.push_str(&format!("  {op}: {count} instances\n"));
            }
        }

        // Graph properties
        report.push_str("\n--- Graph Properties ---\n");
        let is_linear = self.is_linear_chain();
        report.push_str(&format!("  Is Linear Chain: {is_linear}\n"));
        let has_cycles = self.has_cycles();
        report.push_str(&format!("  Has Cycles: {has_cycles}\n"));
        let max_depth = self.get_depth();
        report.push_str(&format!("  Max Depth: {max_depth}\n"));

        // Potential issues
        let orphaned = self.find_orphaned_nodes();
        let dead_ends = self.find_dead_end_nodes();
        if !orphaned.is_empty() || !dead_ends.is_empty() {
            report.push_str("\n--- Potential Issues ---\n");
            if !orphaned.is_empty() {
                report.push_str(&format!(
                    "  ⚠️  {} orphaned nodes detected\n",
                    orphaned.len()
                ));
            }
            if !dead_ends.is_empty() {
                report.push_str(&format!(
                    "  ⚠️  {} dead-end nodes detected\n",
                    dead_ends.len()
                ));
            }
        }

        report.push_str("================================\n");
        report
    }

    /// Export graph structure as a simple text table for debugging
    pub fn debug_table(&self) -> String {
        let mut table = String::new();
        table.push_str("Index | Type      | Name/Operation     | Inputs | Outputs\n");
        table.push_str("------|-----------|-------------------|---------|---------\n");

        // Build adjacency lists for counting
        let mut incoming: HashMap<NodeIndex, usize> = HashMap::new();
        let mut outgoing: HashMap<NodeIndex, usize> = HashMap::new();

        for (src, target, _) in self.edges() {
            *outgoing.entry(src).or_insert(0) += 1;
            *incoming.entry(target).or_insert(0) += 1;
        }

        for (idx, node) in self.nodes() {
            let node_type = match node {
                Node::Input(_) => "Input",
                Node::Call(_, _) => "Call",
                Node::Output => "Output",
                Node::Conditional { .. } => "Conditional",
                Node::Loop { .. } => "Loop",
                Node::Merge { .. } => "Merge",
                Node::GetAttr { .. } => "GetAttr",
            };

            let node_name = match node {
                Node::Input(name) => name.clone(),
                Node::Call(op, _) => op.clone(),
                Node::Output => "output".to_string(),
                Node::Conditional { condition, .. } => format!("if({condition})"),
                Node::Loop { condition, .. } => format!("while({condition})"),
                Node::Merge { .. } => "merge".to_string(),
                Node::GetAttr { target, attr } => format!("{target}.{attr}"),
            };

            let input_count = incoming.get(&idx).unwrap_or(&0);
            let output_count = outgoing.get(&idx).unwrap_or(&0);

            table.push_str(&format!(
                "{:5} | {:9} | {:17} | {:6} | {:7}\n",
                idx.index(),
                node_type,
                node_name,
                input_count,
                output_count
            ));
        }

        table
    }

    /// Get performance recommendations based on graph analysis
    pub fn performance_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check graph size
        if self.node_count() > 1000 {
            recommendations.push("Consider using parallel traversal for large graphs".to_string());
        }

        // Check depth
        if self.get_depth() > 50 {
            recommendations.push(
                "Deep graph detected - apply optimization passes to reduce depth".to_string(),
            );
        }

        // Check for repeated operations
        let op_counts = self.operation_counts();
        for (op, count) in op_counts.iter() {
            if *count > 10 {
                recommendations.push(format!(
                    "Operation '{op}' appears {count} times - consider fusion"
                ));
            }
        }

        // Check for orphaned nodes
        let orphaned = self.find_orphaned_nodes();
        if !orphaned.is_empty() {
            recommendations.push("Remove orphaned nodes to reduce memory usage".to_string());
        }

        // Check for dead ends
        let dead_ends = self.find_dead_end_nodes();
        if !dead_ends.is_empty() {
            recommendations
                .push("Remove dead-end nodes that don't contribute to outputs".to_string());
        }

        // Check for linear chains that could be optimized
        if self.is_linear_chain() && self.call_nodes().len() > 5 {
            recommendations
                .push("Linear chain detected - consider operation fusion passes".to_string());
        }

        if recommendations.is_empty() {
            recommendations
                .push("Graph is well-optimized, no immediate recommendations".to_string());
        }

        recommendations
    }

    /// Lint the graph for best practices and potential issues
    pub fn lint(&self) -> LintReport {
        let linter = GraphLinter::new();
        linter.lint_graph(self)
    }

    /// Analyze memory usage of the graph
    pub fn analyze_memory(&self) -> MemoryUsageReport {
        MemoryAnalyzer::analyze_memory_usage(self)
    }

    /// Calculate comprehensive metrics for the graph
    pub fn metrics(&self) -> GraphMetrics {
        calculate_graph_metrics(self)
    }

    /// Detect patterns in the graph
    pub fn detect_patterns(&self) -> Vec<DetectedPattern> {
        PatternDetector::detect_patterns(self)
    }

    /// Get a summary of graph statistics
    pub fn summary(&self) -> String {
        let input_count = self.input_nodes().len();
        let output_count = self.output_nodes().len();
        let op_count = self.call_nodes().len();
        let conditional_count = self.conditional_nodes().len();
        let loop_count = self.loop_nodes().len();

        format!(
            "FX Graph Summary:\n\
             Total Nodes: {}\n\
             Total Edges: {}\n\
             Inputs: {}\n\
             Outputs: {}\n\
             Operations: {}\n\
             Conditionals: {}\n\
             Loops: {}",
            self.node_count(),
            self.edge_count(),
            input_count,
            output_count,
            op_count,
            conditional_count,
            loop_count
        )
    }

    /// Count operations by type
    pub fn operation_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for (_, node) in self.call_nodes() {
            if let Node::Call(op_name, _) = node {
                *counts.entry(op_name.clone()).or_insert(0) += 1;
            }
        }
        counts
    }

    /// Check if the graph is a simple linear chain
    pub fn is_linear_chain(&self) -> bool {
        // A linear chain has exactly one path from inputs to outputs
        let inputs = self.input_nodes();
        let outputs = self.output_nodes();

        // Must have exactly one input and one output
        if inputs.len() != 1 || outputs.len() != 1 {
            return false;
        }

        // Check if there's a unique path from input to output
        // This is a simplified check - a complete implementation would do graph traversal
        let call_nodes = self.call_nodes();

        // For a linear chain, each node (except input/output) should have exactly one incoming and one outgoing edge
        // This is a basic heuristic check
        call_nodes.len() + 2 == self.node_count() && self.edge_count() == call_nodes.len() + 1
    }

    /// Get graph depth (longest path from input to output)
    pub fn get_depth(&self) -> usize {
        // Simplified depth calculation - counts the number of operation nodes in longest path
        // In a more sophisticated implementation, this would use topological sorting and dynamic programming
        self.call_nodes().len()
    }

    /// Check if the graph has any cycles (ignoring explicit loop constructs)
    pub fn has_cycles(&self) -> bool {
        petgraph::algo::is_cyclic_directed(&self.graph) && self.loop_nodes().is_empty()
    }

    /// Check if the graph represents a simple linear transformation pipeline
    pub fn is_pipeline(&self) -> bool {
        // A pipeline has exactly one input and one output with a linear chain between them
        if self.inputs.len() != 1 || self.outputs.len() != 1 {
            return false;
        }

        // Should be a linear chain without any branching
        self.is_linear_chain() && !self.has_cycles()
    }

    /// Check if the graph contains any control flow constructs
    pub fn has_control_flow(&self) -> bool {
        !self.conditional_nodes().is_empty() || !self.loop_nodes().is_empty()
    }

    /// Get the longest path length in the graph (for estimating execution time)
    pub fn critical_path_length(&self) -> usize {
        // This is a simplified implementation
        // A full implementation would use topological sort and dynamic programming
        self.get_depth()
    }

    /// Get graph complexity metrics as a simple numeric score
    pub fn complexity_score(&self) -> f64 {
        let node_count = self.node_count() as f64;
        let edge_count = self.edge_count() as f64;
        let depth = self.get_depth() as f64;
        let has_cycles = if self.has_cycles() { 1.0 } else { 0.0 };
        let is_linear = if self.is_linear_chain() { 0.5 } else { 1.0 };

        // Weight factors for different complexity components
        let node_weight = 0.1;
        let edge_weight = 0.15;
        let depth_weight = 0.2;
        let cycle_weight = 10.0;
        let structure_weight = 5.0;

        (node_count * node_weight)
            + (edge_count * edge_weight)
            + (depth * depth_weight)
            + (has_cycles * cycle_weight)
            + (is_linear * structure_weight)
    }

    /// Find nodes with no incoming edges (besides inputs)
    pub fn find_orphaned_nodes(&self) -> Vec<NodeIndex> {
        let input_indices: std::collections::HashSet<_> = self.inputs.iter().collect();
        let mut orphaned = Vec::new();

        for node_idx in self.graph.node_indices() {
            if !input_indices.contains(&node_idx) {
                let incoming_edges = self
                    .graph
                    .edges_directed(node_idx, petgraph::Direction::Incoming)
                    .count();
                if incoming_edges == 0 {
                    orphaned.push(node_idx);
                }
            }
        }

        orphaned
    }

    /// Find nodes with no outgoing edges (besides outputs)
    pub fn find_dead_end_nodes(&self) -> Vec<NodeIndex> {
        let output_indices: std::collections::HashSet<_> = self.outputs.iter().collect();
        let mut dead_ends = Vec::new();

        for node_idx in self.graph.node_indices() {
            if !output_indices.contains(&node_idx) {
                let outgoing_edges = self
                    .graph
                    .edges_directed(node_idx, petgraph::Direction::Outgoing)
                    .count();
                let incoming_edges = self
                    .graph
                    .edges_directed(node_idx, petgraph::Direction::Incoming)
                    .count();
                // Only consider nodes that have incoming edges but no outgoing edges
                // This excludes orphaned nodes (which have no incoming edges)
                if outgoing_edges == 0 && incoming_edges > 0 {
                    dead_ends.push(node_idx);
                }
            }
        }

        dead_ends
    }

    /// Get detailed statistics about the graph structure
    pub fn detailed_stats(&self) -> GraphStats {
        let nodes: Vec<_> = self.nodes().collect();
        let edges = self.edges();

        // Calculate various statistics
        let mut fanout_distribution: HashMap<usize, usize> = HashMap::new();
        let mut fanin_distribution: HashMap<usize, usize> = HashMap::new();
        let mut max_fanout = 0;
        let mut max_fanin = 0;

        for (idx, _) in &nodes {
            let fanout = self.node_fanout(*idx);
            let fanin = self.node_fanin(*idx);

            *fanout_distribution.entry(fanout).or_default() += 1;
            *fanin_distribution.entry(fanin).or_default() += 1;

            max_fanout = max_fanout.max(fanout);
            max_fanin = max_fanin.max(fanin);
        }

        let node_types = self.get_node_type_distribution();
        let operation_counts = self.operation_counts();

        GraphStats {
            total_nodes: self.node_count(),
            total_edges: self.edge_count(),
            input_count: self.inputs.len(),
            output_count: self.outputs.len(),
            max_fanout,
            max_fanin,
            average_fanout: if !nodes.is_empty() {
                edges.len() as f64 / nodes.len() as f64
            } else {
                0.0
            },
            depth: self.get_depth(),
            is_linear: self.is_linear_chain(),
            is_pipeline: self.is_pipeline(),
            has_cycles: self.has_cycles(),
            complexity_score: self.complexity_score(),
            node_type_distribution: node_types,
            operation_distribution: operation_counts,
            fanout_distribution,
            fanin_distribution,
        }
    }

    /// Get distribution of node types in the graph
    pub fn get_node_type_distribution(&self) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();

        for (_, node) in self.nodes() {
            let node_type = match node {
                Node::Input(_) => "Input",
                Node::Call(_, _) => "Call",
                Node::Output => "Output",
                Node::Conditional { .. } => "Conditional",
                Node::Loop { .. } => "Loop",
                Node::Merge { .. } => "Merge",
                Node::GetAttr { .. } => "GetAttr",
            };

            *distribution.entry(node_type.to_string()).or_default() += 1;
        }

        distribution
    }

    /// Calculate memory usage estimation for the graph
    pub fn estimate_memory_usage(&self) -> MemoryEstimate {
        let node_size = std::mem::size_of::<Node>();
        let edge_size = std::mem::size_of::<crate::fx::types::Edge>();
        let index_size = std::mem::size_of::<NodeIndex>();

        let nodes_memory = self.node_count() * node_size;
        let edges_memory = self.edge_count() * edge_size;
        let input_indices_memory = self.inputs.len() * index_size;
        let output_indices_memory = self.outputs.len() * index_size;

        // Estimate petgraph overhead (approximate)
        let graph_overhead = self.node_count() * 16 + self.edge_count() * 24;

        let total_memory = nodes_memory
            + edges_memory
            + input_indices_memory
            + output_indices_memory
            + graph_overhead;

        MemoryEstimate {
            total_bytes: total_memory,
            nodes_bytes: nodes_memory,
            edges_bytes: edges_memory,
            metadata_bytes: input_indices_memory + output_indices_memory,
            overhead_bytes: graph_overhead,
            estimated_peak_multiplier: 2.5, // Peak usage during operations
        }
    }
}
