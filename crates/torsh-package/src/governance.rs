//! Package Governance and Lineage Tracking
//!
//! This module provides comprehensive governance capabilities for ML model packages,
//! including lineage tracking, provenance recording, compliance management, and
//! audit trail generation for regulatory requirements (SOC 2, GDPR, HIPAA, etc.).
//!
//! # Features
//!
//! - **Lineage Tracking**: Record package derivation and transformation relationships
//! - **Provenance Recording**: Track package origins, creators, and sources
//! - **Compliance Management**: Tag packages with compliance requirements and certifications
//! - **Transformation History**: Record all operations performed on packages
//! - **Lineage Graphs**: Build and query directed acyclic graphs of package relationships
//! - **Compliance Reporting**: Generate audit reports for regulatory requirements
//! - **Lineage Visualization**: Export lineage to formats like Graphviz DOT
//!
//! # Examples
//!
//! ```rust
//! use torsh_package::governance::{LineageTracker, LineageRelation, ProvenanceInfo};
//! use chrono::Utc;
//!
//! // Create a lineage tracker
//! let mut tracker = LineageTracker::new();
//!
//! // Record provenance for a base package
//! let provenance = ProvenanceInfo {
//!     package_id: "base-model-v1".to_string(),
//!     creator: "alice@example.com".to_string(),
//!     creation_time: Utc::now(),
//!     source_url: Some("https://models.example.com/base".to_string()),
//!     source_commit: Some("abc123".to_string()),
//!     build_environment: vec![("python", "3.11"), ("torch", "2.0.0")]
//!         .into_iter()
//!         .map(|(k, v)| (k.to_string(), v.to_string()))
//!         .collect(),
//!     description: "Base model trained on public dataset".to_string(),
//! };
//! tracker.record_provenance(provenance);
//!
//! // Record a derived package
//! tracker.add_lineage(
//!     "fine-tuned-v1".to_string(),
//!     "base-model-v1".to_string(),
//!     LineageRelation::DerivedFrom,
//!     "Fine-tuned on custom dataset".to_string(),
//! ).unwrap();
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use torsh_core::error::TorshError;

/// Package lineage relationship types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LineageRelation {
    /// Package is derived from another (e.g., fine-tuned model)
    DerivedFrom,
    /// Package is trained from dataset or pre-trained model
    TrainedFrom,
    /// Package is quantized version of another
    QuantizedFrom,
    /// Package is pruned version of another
    PrunedFrom,
    /// Package is distilled from another (knowledge distillation)
    DistilledFrom,
    /// Package is converted from another format
    ConvertedFrom,
    /// Package merges multiple packages
    MergedFrom,
    /// Package is a checkpoint from training run
    CheckpointOf,
    /// Package uses another as dependency
    DependsOn,
    /// Custom relationship (with description)
    Custom(String),
}

impl LineageRelation {
    /// Get human-readable description of the relationship
    pub fn description(&self) -> String {
        match self {
            Self::DerivedFrom => "derived from".to_string(),
            Self::TrainedFrom => "trained from".to_string(),
            Self::QuantizedFrom => "quantized from".to_string(),
            Self::PrunedFrom => "pruned from".to_string(),
            Self::DistilledFrom => "distilled from".to_string(),
            Self::ConvertedFrom => "converted from".to_string(),
            Self::MergedFrom => "merged from".to_string(),
            Self::CheckpointOf => "checkpoint of".to_string(),
            Self::DependsOn => "depends on".to_string(),
            Self::Custom(desc) => desc.clone(),
        }
    }
}

/// Package provenance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceInfo {
    /// Unique package identifier
    pub package_id: String,
    /// Creator user/organization
    pub creator: String,
    /// Creation timestamp
    pub creation_time: DateTime<Utc>,
    /// Source repository URL
    pub source_url: Option<String>,
    /// Source code commit hash
    pub source_commit: Option<String>,
    /// Build environment details (tool versions, dependencies)
    pub build_environment: HashMap<String, String>,
    /// Human-readable description
    pub description: String,
}

/// Lineage edge connecting two packages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageEdge {
    /// Source package ID (parent)
    pub from: String,
    /// Target package ID (child)
    pub to: String,
    /// Relationship type
    pub relation: LineageRelation,
    /// Timestamp when relationship was recorded
    pub timestamp: DateTime<Utc>,
    /// Additional metadata about the transformation
    pub metadata: HashMap<String, String>,
    /// Human-readable description of the transformation
    pub description: String,
}

/// Transformation operation performed on a package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationRecord {
    /// Package ID
    pub package_id: String,
    /// Operation type (e.g., "quantization", "pruning", "fine-tuning")
    pub operation: String,
    /// Timestamp of operation
    pub timestamp: DateTime<Utc>,
    /// User who performed the operation
    pub performed_by: String,
    /// Operation parameters
    pub parameters: HashMap<String, String>,
    /// Operation result/status
    pub result: String,
    /// Duration in seconds
    pub duration_secs: f64,
}

/// Compliance requirement level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComplianceLevel {
    /// No specific compliance required
    None,
    /// Internal company policies
    Internal,
    /// Industry standards (e.g., ISO)
    Industry,
    /// Regulatory requirements (e.g., GDPR, HIPAA)
    Regulatory,
    /// Critical security requirements
    CriticalSecurity,
}

/// Compliance metadata for a package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceMetadata {
    /// Package ID
    pub package_id: String,
    /// Compliance level required
    pub level: ComplianceLevel,
    /// Compliance tags (e.g., "HIPAA", "SOC2", "GDPR")
    pub tags: Vec<String>,
    /// Certifications obtained
    pub certifications: Vec<String>,
    /// Data classification (e.g., "Public", "Internal", "Confidential", "Restricted")
    pub data_classification: String,
    /// Retention policy in days
    pub retention_days: Option<u32>,
    /// Access restrictions
    pub access_restrictions: Vec<String>,
    /// Audit requirements
    pub audit_required: bool,
    /// Last audit date
    pub last_audit: Option<DateTime<Utc>>,
    /// Next audit due date
    pub next_audit_due: Option<DateTime<Utc>>,
}

/// Lineage query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageQueryResult {
    /// Package IDs in the lineage
    pub packages: Vec<String>,
    /// Edges connecting the packages
    pub edges: Vec<LineageEdge>,
    /// Provenance information for each package
    pub provenance: HashMap<String, ProvenanceInfo>,
}

/// Compliance audit report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    /// Report generation timestamp
    pub generated_at: DateTime<Utc>,
    /// Total packages audited
    pub total_packages: usize,
    /// Compliant packages
    pub compliant_packages: usize,
    /// Non-compliant packages
    pub non_compliant_packages: usize,
    /// Packages needing review
    pub needs_review: Vec<String>,
    /// Compliance issues found
    pub issues: Vec<ComplianceIssue>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Compliance issue found during audit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceIssue {
    /// Package ID with issue
    pub package_id: String,
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue description
    pub description: String,
    /// Remediation steps
    pub remediation: String,
}

/// Issue severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Low severity - informational
    Low,
    /// Medium severity - should be addressed
    Medium,
    /// High severity - must be addressed soon
    High,
    /// Critical severity - immediate action required
    Critical,
}

/// Package lineage tracker
///
/// Tracks package provenance, relationships, transformations, and compliance metadata.
/// Builds directed acyclic graphs (DAGs) of package lineage for traceability and governance.
pub struct LineageTracker {
    /// Provenance information by package ID
    provenance: HashMap<String, ProvenanceInfo>,
    /// Lineage edges
    edges: Vec<LineageEdge>,
    /// Adjacency list for efficient graph traversal (forward edges)
    forward_graph: HashMap<String, Vec<usize>>,
    /// Adjacency list for reverse traversal (backward edges)
    backward_graph: HashMap<String, Vec<usize>>,
    /// Transformation records by package ID
    transformations: HashMap<String, Vec<TransformationRecord>>,
    /// Compliance metadata by package ID
    compliance: HashMap<String, ComplianceMetadata>,
}

impl LineageTracker {
    /// Create a new lineage tracker
    pub fn new() -> Self {
        Self {
            provenance: HashMap::new(),
            edges: Vec::new(),
            forward_graph: HashMap::new(),
            backward_graph: HashMap::new(),
            transformations: HashMap::new(),
            compliance: HashMap::new(),
        }
    }

    /// Record provenance information for a package
    pub fn record_provenance(&mut self, info: ProvenanceInfo) {
        let package_id = info.package_id.clone();
        self.provenance.insert(package_id, info);
    }

    /// Add a lineage relationship between two packages
    pub fn add_lineage(
        &mut self,
        from: String,
        to: String,
        relation: LineageRelation,
        description: String,
    ) -> Result<(), TorshError> {
        // Check for cycles
        if self.would_create_cycle(&from, &to) {
            return Err(TorshError::InvalidArgument(format!(
                "Adding edge from {} to {} would create a cycle",
                from, to
            )));
        }

        let edge = LineageEdge {
            from: from.clone(),
            to: to.clone(),
            relation,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
            description,
        };

        let edge_idx = self.edges.len();
        self.edges.push(edge);

        // Update adjacency lists
        self.forward_graph
            .entry(from.clone())
            .or_insert_with(Vec::new)
            .push(edge_idx);
        self.backward_graph
            .entry(to)
            .or_insert_with(Vec::new)
            .push(edge_idx);

        Ok(())
    }

    /// Add lineage with metadata
    pub fn add_lineage_with_metadata(
        &mut self,
        from: String,
        to: String,
        relation: LineageRelation,
        description: String,
        metadata: HashMap<String, String>,
    ) -> Result<(), TorshError> {
        self.add_lineage(from.clone(), to.clone(), relation, description)?;

        // Update metadata for the last edge
        if let Some(edge) = self.edges.last_mut() {
            edge.metadata = metadata;
        }

        Ok(())
    }

    /// Record a transformation operation
    pub fn record_transformation(&mut self, record: TransformationRecord) {
        let package_id = record.package_id.clone();
        self.transformations
            .entry(package_id)
            .or_insert_with(Vec::new)
            .push(record);
    }

    /// Set compliance metadata for a package
    pub fn set_compliance(&mut self, metadata: ComplianceMetadata) {
        let package_id = metadata.package_id.clone();
        self.compliance.insert(package_id, metadata);
    }

    /// Get provenance information for a package
    pub fn get_provenance(&self, package_id: &str) -> Option<&ProvenanceInfo> {
        self.provenance.get(package_id)
    }

    /// Get all ancestors of a package (packages it derives from)
    pub fn get_ancestors(&self, package_id: &str) -> Vec<String> {
        let mut ancestors = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(package_id.to_string());

        while let Some(current) = queue.pop_front() {
            if !visited.insert(current.clone()) {
                continue;
            }

            if let Some(edge_indices) = self.backward_graph.get(&current) {
                for &idx in edge_indices {
                    let edge = &self.edges[idx];
                    ancestors.push(edge.from.clone());
                    queue.push_back(edge.from.clone());
                }
            }
        }

        ancestors
    }

    /// Get all descendants of a package (packages derived from it)
    pub fn get_descendants(&self, package_id: &str) -> Vec<String> {
        let mut descendants = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(package_id.to_string());

        while let Some(current) = queue.pop_front() {
            if !visited.insert(current.clone()) {
                continue;
            }

            if let Some(edge_indices) = self.forward_graph.get(&current) {
                for &idx in edge_indices {
                    let edge = &self.edges[idx];
                    descendants.push(edge.to.clone());
                    queue.push_back(edge.to.clone());
                }
            }
        }

        descendants
    }

    /// Get full lineage (ancestors and descendants) of a package
    pub fn get_lineage(&self, package_id: &str) -> LineageQueryResult {
        let mut packages = HashSet::new();
        packages.insert(package_id.to_string());

        let ancestors = self.get_ancestors(package_id);
        let descendants = self.get_descendants(package_id);

        packages.extend(ancestors);
        packages.extend(descendants);

        let relevant_edges: Vec<LineageEdge> = self
            .edges
            .iter()
            .filter(|edge| packages.contains(&edge.from) || packages.contains(&edge.to))
            .cloned()
            .collect();

        let provenance: HashMap<String, ProvenanceInfo> = packages
            .iter()
            .filter_map(|id| self.provenance.get(id).map(|p| (id.clone(), p.clone())))
            .collect();

        LineageQueryResult {
            packages: packages.into_iter().collect(),
            edges: relevant_edges,
            provenance,
        }
    }

    /// Get transformation history for a package
    pub fn get_transformations(&self, package_id: &str) -> Vec<&TransformationRecord> {
        self.transformations
            .get(package_id)
            .map(|records| records.iter().collect())
            .unwrap_or_default()
    }

    /// Get compliance metadata for a package
    pub fn get_compliance(&self, package_id: &str) -> Option<&ComplianceMetadata> {
        self.compliance.get(package_id)
    }

    /// Generate compliance audit report
    pub fn generate_compliance_report(&self) -> ComplianceReport {
        let mut issues = Vec::new();
        let mut needs_review = Vec::new();
        let mut compliant = 0;
        let mut non_compliant = 0;

        for (package_id, metadata) in &self.compliance {
            // Check if audit is overdue
            if let Some(due_date) = metadata.next_audit_due {
                if due_date < Utc::now() {
                    issues.push(ComplianceIssue {
                        package_id: package_id.clone(),
                        severity: IssueSeverity::High,
                        description: "Compliance audit overdue".to_string(),
                        remediation: "Schedule and complete compliance audit".to_string(),
                    });
                    non_compliant += 1;
                } else {
                    compliant += 1;
                }
            } else if metadata.audit_required {
                needs_review.push(package_id.clone());
            }

            // Check for missing certifications for regulatory compliance
            if metadata.level == ComplianceLevel::Regulatory && metadata.certifications.is_empty() {
                issues.push(ComplianceIssue {
                    package_id: package_id.clone(),
                    severity: IssueSeverity::Critical,
                    description: "Regulatory compliance required but no certifications found"
                        .to_string(),
                    remediation: "Obtain required compliance certifications".to_string(),
                });
            }

            // Check for missing provenance
            if !self.provenance.contains_key(package_id) {
                issues.push(ComplianceIssue {
                    package_id: package_id.clone(),
                    severity: IssueSeverity::Medium,
                    description: "Missing provenance information".to_string(),
                    remediation: "Record complete provenance information for the package"
                        .to_string(),
                });
            }
        }

        // Sort issues by severity
        issues.sort_by(|a, b| b.severity.cmp(&a.severity));

        let recommendations = self.generate_recommendations(&issues);

        ComplianceReport {
            generated_at: Utc::now(),
            total_packages: self.compliance.len(),
            compliant_packages: compliant,
            non_compliant_packages: non_compliant,
            needs_review,
            issues,
            recommendations,
        }
    }

    /// Export lineage to Graphviz DOT format
    pub fn export_to_dot(&self, package_id: &str) -> String {
        let lineage = self.get_lineage(package_id);
        let mut dot = String::from("digraph PackageLineage {\n");
        dot.push_str("  rankdir=LR;\n");
        dot.push_str("  node [shape=box];\n\n");

        // Add nodes
        for pkg_id in &lineage.packages {
            let label = if let Some(prov) = lineage.provenance.get(pkg_id) {
                format!("{}\\n{}", pkg_id, prov.creator)
            } else {
                pkg_id.clone()
            };
            dot.push_str(&format!("  \"{}\" [label=\"{}\"];\n", pkg_id, label));
        }

        dot.push('\n');

        // Add edges
        for edge in &lineage.edges {
            let label = edge.relation.description();
            dot.push_str(&format!(
                "  \"{}\" -> \"{}\" [label=\"{}\"];\n",
                edge.from, edge.to, label
            ));
        }

        dot.push_str("}\n");
        dot
    }

    /// Export lineage to JSON
    pub fn export_to_json(&self, package_id: &str) -> Result<String, TorshError> {
        let lineage = self.get_lineage(package_id);
        serde_json::to_string_pretty(&lineage)
            .map_err(|e| TorshError::SerializationError(e.to_string()))
    }

    /// Get statistics about the lineage graph
    pub fn get_statistics(&self) -> LineageStatistics {
        let total_packages = self.provenance.len();
        let total_edges = self.edges.len();
        let total_transformations: usize = self.transformations.values().map(|v| v.len()).sum();
        let packages_with_compliance = self.compliance.len();

        // Find root packages (no incoming edges)
        let roots: Vec<String> = self
            .provenance
            .keys()
            .filter(|id| !self.backward_graph.contains_key(*id))
            .cloned()
            .collect();

        // Find leaf packages (no outgoing edges)
        let leaves: Vec<String> = self
            .provenance
            .keys()
            .filter(|id| !self.forward_graph.contains_key(*id))
            .cloned()
            .collect();

        // Calculate average depth
        let avg_depth = if !roots.is_empty() {
            let total_depth: usize = roots.iter().map(|r| self.calculate_depth(r)).sum();
            total_depth as f64 / roots.len() as f64
        } else {
            0.0
        };

        LineageStatistics {
            total_packages,
            total_edges,
            total_transformations,
            packages_with_compliance,
            root_packages: roots.len(),
            leaf_packages: leaves.len(),
            average_depth: avg_depth,
        }
    }

    // Private helper methods

    fn would_create_cycle(&self, from: &str, to: &str) -> bool {
        // Check if there's already a path from 'to' to 'from'
        // If so, adding an edge from 'from' to 'to' would create a cycle
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(to.to_string());

        while let Some(current) = queue.pop_front() {
            if current == from {
                return true;
            }

            if !visited.insert(current.clone()) {
                continue;
            }

            if let Some(edge_indices) = self.forward_graph.get(&current) {
                for &idx in edge_indices {
                    let edge = &self.edges[idx];
                    queue.push_back(edge.to.clone());
                }
            }
        }

        false
    }

    fn calculate_depth(&self, package_id: &str) -> usize {
        let mut max_depth = 0;
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back((package_id.to_string(), 0));

        while let Some((current, depth)) = queue.pop_front() {
            if !visited.insert(current.clone()) {
                continue;
            }

            max_depth = max_depth.max(depth);

            if let Some(edge_indices) = self.forward_graph.get(&current) {
                for &idx in edge_indices {
                    let edge = &self.edges[idx];
                    queue.push_back((edge.to.clone(), depth + 1));
                }
            }
        }

        max_depth
    }

    fn generate_recommendations(&self, issues: &[ComplianceIssue]) -> Vec<String> {
        let mut recommendations = Vec::new();

        let critical_count = issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Critical)
            .count();
        let high_count = issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::High)
            .count();

        if critical_count > 0 {
            recommendations.push(format!(
                "Address {} critical compliance issues immediately",
                critical_count
            ));
        }

        if high_count > 0 {
            recommendations.push(format!(
                "Schedule remediation for {} high-severity issues within 30 days",
                high_count
            ));
        }

        if self.compliance.len() < self.provenance.len() {
            recommendations
                .push("Define compliance metadata for all packages in the lineage".to_string());
        }

        // Check for packages without provenance
        let missing_provenance = self.provenance.len();
        let total_packages = self.forward_graph.len().max(self.backward_graph.len());
        if missing_provenance < total_packages {
            recommendations.push(
                "Complete provenance information for all packages to ensure full traceability"
                    .to_string(),
            );
        }

        if recommendations.is_empty() {
            recommendations
                .push("All packages are compliant. Continue regular audits.".to_string());
        }

        recommendations
    }
}

impl Default for LineageTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Lineage graph statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageStatistics {
    /// Total number of packages tracked
    pub total_packages: usize,
    /// Total number of lineage edges
    pub total_edges: usize,
    /// Total number of transformation records
    pub total_transformations: usize,
    /// Number of packages with compliance metadata
    pub packages_with_compliance: usize,
    /// Number of root packages (no parents)
    pub root_packages: usize,
    /// Number of leaf packages (no children)
    pub leaf_packages: usize,
    /// Average depth of lineage tree
    pub average_depth: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lineage_tracker_creation() {
        let tracker = LineageTracker::new();
        let stats = tracker.get_statistics();
        assert_eq!(stats.total_packages, 0);
        assert_eq!(stats.total_edges, 0);
    }

    #[test]
    fn test_record_provenance() {
        let mut tracker = LineageTracker::new();

        let provenance = ProvenanceInfo {
            package_id: "test-pkg".to_string(),
            creator: "alice@example.com".to_string(),
            creation_time: Utc::now(),
            source_url: Some("https://github.com/test/repo".to_string()),
            source_commit: Some("abc123".to_string()),
            build_environment: HashMap::new(),
            description: "Test package".to_string(),
        };

        tracker.record_provenance(provenance);

        let retrieved = tracker.get_provenance("test-pkg");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().creator, "alice@example.com");
    }

    #[test]
    fn test_add_lineage() {
        let mut tracker = LineageTracker::new();

        // Add provenance for packages
        let prov1 = ProvenanceInfo {
            package_id: "base".to_string(),
            creator: "alice".to_string(),
            creation_time: Utc::now(),
            source_url: None,
            source_commit: None,
            build_environment: HashMap::new(),
            description: "Base model".to_string(),
        };
        tracker.record_provenance(prov1);

        let prov2 = ProvenanceInfo {
            package_id: "derived".to_string(),
            creator: "bob".to_string(),
            creation_time: Utc::now(),
            source_url: None,
            source_commit: None,
            build_environment: HashMap::new(),
            description: "Derived model".to_string(),
        };
        tracker.record_provenance(prov2);

        // Add lineage relationship
        let result = tracker.add_lineage(
            "base".to_string(),
            "derived".to_string(),
            LineageRelation::DerivedFrom,
            "Fine-tuned version".to_string(),
        );

        assert!(result.is_ok());
        let stats = tracker.get_statistics();
        assert_eq!(stats.total_edges, 1);
    }

    #[test]
    fn test_cycle_detection() {
        let mut tracker = LineageTracker::new();

        // Add A -> B
        tracker
            .add_lineage(
                "A".to_string(),
                "B".to_string(),
                LineageRelation::DerivedFrom,
                "A to B".to_string(),
            )
            .unwrap();

        // Add B -> C
        tracker
            .add_lineage(
                "B".to_string(),
                "C".to_string(),
                LineageRelation::DerivedFrom,
                "B to C".to_string(),
            )
            .unwrap();

        // Try to add C -> A (would create cycle)
        let result = tracker.add_lineage(
            "C".to_string(),
            "A".to_string(),
            LineageRelation::DerivedFrom,
            "C to A".to_string(),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_get_ancestors() {
        let mut tracker = LineageTracker::new();

        tracker
            .add_lineage(
                "A".to_string(),
                "B".to_string(),
                LineageRelation::DerivedFrom,
                "".to_string(),
            )
            .unwrap();
        tracker
            .add_lineage(
                "B".to_string(),
                "C".to_string(),
                LineageRelation::DerivedFrom,
                "".to_string(),
            )
            .unwrap();

        let ancestors = tracker.get_ancestors("C");
        assert_eq!(ancestors.len(), 2);
        assert!(ancestors.contains(&"B".to_string()));
        assert!(ancestors.contains(&"A".to_string()));
    }

    #[test]
    fn test_get_descendants() {
        let mut tracker = LineageTracker::new();

        tracker
            .add_lineage(
                "A".to_string(),
                "B".to_string(),
                LineageRelation::DerivedFrom,
                "".to_string(),
            )
            .unwrap();
        tracker
            .add_lineage(
                "A".to_string(),
                "C".to_string(),
                LineageRelation::DerivedFrom,
                "".to_string(),
            )
            .unwrap();

        let descendants = tracker.get_descendants("A");
        assert_eq!(descendants.len(), 2);
        assert!(descendants.contains(&"B".to_string()));
        assert!(descendants.contains(&"C".to_string()));
    }

    #[test]
    fn test_transformation_recording() {
        let mut tracker = LineageTracker::new();

        let record = TransformationRecord {
            package_id: "test".to_string(),
            operation: "quantization".to_string(),
            timestamp: Utc::now(),
            performed_by: "alice".to_string(),
            parameters: [("bits".to_string(), "8".to_string())]
                .iter()
                .cloned()
                .collect(),
            result: "success".to_string(),
            duration_secs: 120.5,
        };

        tracker.record_transformation(record);

        let transformations = tracker.get_transformations("test");
        assert_eq!(transformations.len(), 1);
        assert_eq!(transformations[0].operation, "quantization");
    }

    #[test]
    fn test_compliance_metadata() {
        let mut tracker = LineageTracker::new();

        let metadata = ComplianceMetadata {
            package_id: "test".to_string(),
            level: ComplianceLevel::Regulatory,
            tags: vec!["HIPAA".to_string(), "SOC2".to_string()],
            certifications: vec!["ISO27001".to_string()],
            data_classification: "Confidential".to_string(),
            retention_days: Some(2555),
            access_restrictions: vec!["internal-only".to_string()],
            audit_required: true,
            last_audit: Some(Utc::now()),
            next_audit_due: Some(Utc::now() + chrono::Duration::days(90)),
        };

        tracker.set_compliance(metadata);

        let retrieved = tracker.get_compliance("test");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().level, ComplianceLevel::Regulatory);
        assert_eq!(retrieved.unwrap().tags.len(), 2);
    }

    #[test]
    fn test_compliance_report() {
        let mut tracker = LineageTracker::new();

        // Add package with overdue audit
        let metadata = ComplianceMetadata {
            package_id: "overdue".to_string(),
            level: ComplianceLevel::Regulatory,
            tags: vec!["GDPR".to_string()],
            certifications: vec![],
            data_classification: "Restricted".to_string(),
            retention_days: None,
            access_restrictions: vec![],
            audit_required: true,
            last_audit: Some(Utc::now() - chrono::Duration::days(180)),
            next_audit_due: Some(Utc::now() - chrono::Duration::days(1)),
        };
        tracker.set_compliance(metadata);

        let report = tracker.generate_compliance_report();
        assert_eq!(report.total_packages, 1);
        assert!(report.non_compliant_packages > 0);
        assert!(!report.issues.is_empty());
    }

    #[test]
    fn test_export_to_dot() {
        let mut tracker = LineageTracker::new();

        tracker
            .add_lineage(
                "A".to_string(),
                "B".to_string(),
                LineageRelation::DerivedFrom,
                "".to_string(),
            )
            .unwrap();

        let dot = tracker.export_to_dot("A");
        assert!(dot.contains("digraph PackageLineage"));
        assert!(dot.contains("\"A\""));
        assert!(dot.contains("\"B\""));
        assert!(dot.contains("derived from"));
    }

    #[test]
    fn test_export_to_json() {
        let mut tracker = LineageTracker::new();

        let prov = ProvenanceInfo {
            package_id: "A".to_string(),
            creator: "alice".to_string(),
            creation_time: Utc::now(),
            source_url: None,
            source_commit: None,
            build_environment: HashMap::new(),
            description: "Test".to_string(),
        };
        tracker.record_provenance(prov);

        tracker
            .add_lineage(
                "A".to_string(),
                "B".to_string(),
                LineageRelation::DerivedFrom,
                "".to_string(),
            )
            .unwrap();

        let json = tracker.export_to_json("A");
        assert!(json.is_ok());
        let json_str = json.unwrap();
        assert!(json_str.contains("\"packages\""));
        assert!(json_str.contains("\"edges\""));
    }

    #[test]
    fn test_lineage_statistics() {
        let mut tracker = LineageTracker::new();

        // Create a small lineage tree
        for i in 0..5 {
            let prov = ProvenanceInfo {
                package_id: format!("pkg-{}", i),
                creator: "alice".to_string(),
                creation_time: Utc::now(),
                source_url: None,
                source_commit: None,
                build_environment: HashMap::new(),
                description: format!("Package {}", i),
            };
            tracker.record_provenance(prov);
        }

        tracker
            .add_lineage(
                "pkg-0".to_string(),
                "pkg-1".to_string(),
                LineageRelation::DerivedFrom,
                "".to_string(),
            )
            .unwrap();
        tracker
            .add_lineage(
                "pkg-1".to_string(),
                "pkg-2".to_string(),
                LineageRelation::DerivedFrom,
                "".to_string(),
            )
            .unwrap();

        let stats = tracker.get_statistics();
        assert_eq!(stats.total_packages, 5);
        assert_eq!(stats.total_edges, 2);
        assert!(stats.root_packages > 0);
        assert!(stats.leaf_packages > 0);
    }
}
