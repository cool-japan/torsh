//! Automated metric reporting and visualization
//!
//! This module provides utilities for generating comprehensive metric reports
//! in various formats (Markdown, JSON, HTML).

use crate::{Metric, MetricCollection};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_tensor::Tensor;

/// Metric report in various formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricReport {
    /// Experiment name
    pub experiment_name: String,
    /// Timestamp of the evaluation
    pub timestamp: String,
    /// Individual metric results
    pub metrics: HashMap<String, f64>,
    /// Metadata (model name, dataset, hyperparameters, etc.)
    pub metadata: HashMap<String, String>,
    /// Summary statistics
    pub summary: ReportSummary,
}

/// Summary statistics for the report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    /// Number of metrics computed
    pub num_metrics: usize,
    /// Number of samples evaluated
    pub num_samples: usize,
    /// Best metric value
    pub best_metric: Option<(String, f64)>,
    /// Worst metric value
    pub worst_metric: Option<(String, f64)>,
}

impl MetricReport {
    /// Create a new metric report
    pub fn new(experiment_name: String) -> Self {
        Self {
            experiment_name,
            timestamp: chrono::Utc::now().to_rfc3339(),
            metrics: HashMap::new(),
            metadata: HashMap::new(),
            summary: ReportSummary {
                num_metrics: 0,
                num_samples: 0,
                best_metric: None,
                worst_metric: None,
            },
        }
    }

    /// Add a metric result
    pub fn add_metric(&mut self, name: String, value: f64) {
        self.metrics.insert(name, value);
        self.update_summary();
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Set number of samples
    pub fn set_num_samples(&mut self, n: usize) {
        self.summary.num_samples = n;
    }

    /// Update summary statistics
    fn update_summary(&mut self) {
        self.summary.num_metrics = self.metrics.len();

        if !self.metrics.is_empty() {
            let mut best = ("".to_string(), f64::NEG_INFINITY);
            let mut worst = ("".to_string(), f64::INFINITY);

            for (name, &value) in &self.metrics {
                if value > best.1 {
                    best = (name.clone(), value);
                }
                if value < worst.1 {
                    worst = (name.clone(), value);
                }
            }

            self.summary.best_metric = Some(best);
            self.summary.worst_metric = Some(worst);
        }
    }

    /// Generate Markdown report
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str(&format!("# Metric Report: {}\n\n", self.experiment_name));
        md.push_str(&format!("**Timestamp:** {}\n\n", self.timestamp));

        // Metadata
        if !self.metadata.is_empty() {
            md.push_str("## Metadata\n\n");
            for (key, value) in &self.metadata {
                md.push_str(&format!("- **{}:** {}\n", key, value));
            }
            md.push_str("\n");
        }

        // Summary
        md.push_str("## Summary\n\n");
        md.push_str(&format!(
            "- **Number of Metrics:** {}\n",
            self.summary.num_metrics
        ));
        md.push_str(&format!(
            "- **Number of Samples:** {}\n",
            self.summary.num_samples
        ));

        if let Some((name, value)) = &self.summary.best_metric {
            md.push_str(&format!("- **Best Metric:** {} = {:.4}\n", name, value));
        }
        if let Some((name, value)) = &self.summary.worst_metric {
            md.push_str(&format!("- **Worst Metric:** {} = {:.4}\n", name, value));
        }
        md.push_str("\n");

        // Metrics table
        md.push_str("## Metrics\n\n");
        md.push_str("| Metric | Value |\n");
        md.push_str("|--------|-------|\n");

        let mut metrics: Vec<_> = self.metrics.iter().collect();
        metrics.sort_by(|a, b| a.0.cmp(b.0));

        for (name, value) in metrics {
            md.push_str(&format!("| {} | {:.6} |\n", name, value));
        }

        md
    }

    /// Generate JSON report
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Generate compact JSON report
    pub fn to_json_compact(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Generate HTML report
    pub fn to_html(&self) -> String {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<meta charset=\"UTF-8\">\n");
        html.push_str(&format!(
            "<title>Metric Report: {}</title>\n",
            self.experiment_name
        ));
        html.push_str("<style>\n");
        html.push_str("body { font-family: Arial, sans-serif; margin: 20px; }\n");
        html.push_str("h1 { color: #333; }\n");
        html.push_str("table { border-collapse: collapse; width: 100%; margin-top: 20px; }\n");
        html.push_str("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n");
        html.push_str("th { background-color: #4CAF50; color: white; }\n");
        html.push_str("tr:nth-child(even) { background-color: #f2f2f2; }\n");
        html.push_str(
            ".metadata { background-color: #f9f9f9; padding: 10px; border-radius: 5px; }\n",
        );
        html.push_str("</style>\n</head>\n<body>\n");

        html.push_str(&format!(
            "<h1>Metric Report: {}</h1>\n",
            self.experiment_name
        ));
        html.push_str(&format!(
            "<p><strong>Timestamp:</strong> {}</p>\n",
            self.timestamp
        ));

        // Metadata
        if !self.metadata.is_empty() {
            html.push_str("<div class=\"metadata\">\n<h2>Metadata</h2>\n<ul>\n");
            for (key, value) in &self.metadata {
                html.push_str(&format!("<li><strong>{}:</strong> {}</li>\n", key, value));
            }
            html.push_str("</ul>\n</div>\n");
        }

        // Summary
        html.push_str("<h2>Summary</h2>\n<ul>\n");
        html.push_str(&format!(
            "<li><strong>Number of Metrics:</strong> {}</li>\n",
            self.summary.num_metrics
        ));
        html.push_str(&format!(
            "<li><strong>Number of Samples:</strong> {}</li>\n",
            self.summary.num_samples
        ));

        if let Some((name, value)) = &self.summary.best_metric {
            html.push_str(&format!(
                "<li><strong>Best Metric:</strong> {} = {:.4}</li>\n",
                name, value
            ));
        }
        if let Some((name, value)) = &self.summary.worst_metric {
            html.push_str(&format!(
                "<li><strong>Worst Metric:</strong> {} = {:.4}</li>\n",
                name, value
            ));
        }
        html.push_str("</ul>\n");

        // Metrics table
        html.push_str("<h2>Metrics</h2>\n<table>\n");
        html.push_str("<tr><th>Metric</th><th>Value</th></tr>\n");

        let mut metrics: Vec<_> = self.metrics.iter().collect();
        metrics.sort_by(|a, b| a.0.cmp(b.0));

        for (name, value) in metrics {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{:.6}</td></tr>\n",
                name, value
            ));
        }

        html.push_str("</table>\n</body>\n</html>");

        html
    }

    /// Save report to file
    pub fn save(&self, path: &str, format: ReportFormat) -> Result<(), std::io::Error> {
        use std::fs::File;
        use std::io::Write;

        let content = match format {
            ReportFormat::Markdown => self.to_markdown(),
            ReportFormat::Json => self
                .to_json()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?,
            ReportFormat::JsonCompact => self
                .to_json_compact()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?,
            ReportFormat::Html => self.to_html(),
        };

        let mut file = File::create(path)?;
        file.write_all(content.as_bytes())?;

        Ok(())
    }
}

/// Report output format
#[derive(Debug, Clone, Copy)]
pub enum ReportFormat {
    Markdown,
    Json,
    JsonCompact,
    Html,
}

/// Report builder for easy report creation
pub struct ReportBuilder {
    report: MetricReport,
}

impl ReportBuilder {
    /// Create a new report builder
    pub fn new(experiment_name: String) -> Self {
        Self {
            report: MetricReport::new(experiment_name),
        }
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.report.add_metadata(key, value);
        self
    }

    /// Set number of samples
    pub fn with_num_samples(mut self, n: usize) -> Self {
        self.report.set_num_samples(n);
        self
    }

    /// Evaluate and add metrics from a metric collection
    pub fn evaluate_collection(
        mut self,
        collection: &mut MetricCollection,
        predictions: &Tensor,
        targets: &Tensor,
    ) -> Self {
        let results = collection.compute(predictions, targets);
        for (name, value) in results {
            self.report.add_metric(name, value);
        }
        self
    }

    /// Evaluate and add a single metric
    pub fn evaluate_metric<M: Metric>(
        mut self,
        metric: &M,
        predictions: &Tensor,
        targets: &Tensor,
    ) -> Self {
        let value = metric.compute(predictions, targets);
        self.report.add_metric(metric.name().to_string(), value);
        self
    }

    /// Add a pre-computed metric
    pub fn add_metric(mut self, name: String, value: f64) -> Self {
        self.report.add_metric(name, value);
        self
    }

    /// Build the final report
    pub fn build(self) -> MetricReport {
        self.report
    }
}

/// Comparison report for multiple experiments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    /// Name of the comparison
    pub name: String,
    /// Individual experiment reports
    pub experiments: Vec<MetricReport>,
    /// Timestamp
    pub timestamp: String,
}

impl ComparisonReport {
    /// Create a new comparison report
    pub fn new(name: String) -> Self {
        Self {
            name,
            experiments: Vec::new(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Add an experiment report
    pub fn add_experiment(&mut self, report: MetricReport) {
        self.experiments.push(report);
    }

    /// Generate Markdown comparison table
    pub fn to_markdown(&self) -> String {
        if self.experiments.is_empty() {
            return "No experiments to compare".to_string();
        }

        let mut md = String::new();
        md.push_str(&format!("# Comparison Report: {}\n\n", self.name));
        md.push_str(&format!("**Timestamp:** {}\n\n", self.timestamp));

        // Collect all unique metrics
        let mut all_metrics = std::collections::HashSet::new();
        for exp in &self.experiments {
            for metric in exp.metrics.keys() {
                all_metrics.insert(metric.clone());
            }
        }

        let mut sorted_metrics: Vec<_> = all_metrics.into_iter().collect();
        sorted_metrics.sort();

        // Create header
        md.push_str("| Metric |");
        for exp in &self.experiments {
            md.push_str(&format!(" {} |", exp.experiment_name));
        }
        md.push_str("\n|--------|");
        for _ in &self.experiments {
            md.push_str("--------|");
        }
        md.push_str("\n");

        // Create rows
        for metric in sorted_metrics {
            md.push_str(&format!("| {} |", metric));
            for exp in &self.experiments {
                if let Some(&value) = exp.metrics.get(&metric) {
                    md.push_str(&format!(" {:.6} |", value));
                } else {
                    md.push_str(" N/A |");
                }
            }
            md.push_str("\n");
        }

        md
    }

    /// Generate JSON comparison
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

// Use chrono for timestamps (add to dependencies)
mod chrono {
    pub struct Utc;
    impl Utc {
        pub fn now() -> Self {
            Utc
        }
        pub fn to_rfc3339(&self) -> String {
            use std::time::{SystemTime, UNIX_EPOCH};
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
            format!("{}", now)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_report_creation() {
        let mut report = MetricReport::new("Test Experiment".to_string());
        report.add_metric("accuracy".to_string(), 0.95);
        report.add_metric("precision".to_string(), 0.92);
        report.add_metadata("model".to_string(), "ResNet50".to_string());

        assert_eq!(report.metrics.len(), 2);
        assert_eq!(report.metadata.len(), 1);
        assert_eq!(report.summary.num_metrics, 2);
    }

    #[test]
    fn test_markdown_generation() {
        let mut report = MetricReport::new("Test".to_string());
        report.add_metric("accuracy".to_string(), 0.95);

        let md = report.to_markdown();
        assert!(md.contains("# Metric Report: Test"));
        assert!(md.contains("accuracy"));
        assert!(md.contains("0.95"));
    }

    #[test]
    fn test_json_generation() {
        let mut report = MetricReport::new("Test".to_string());
        report.add_metric("accuracy".to_string(), 0.95);

        let json = report.to_json().unwrap();
        assert!(json.contains("Test"));
        assert!(json.contains("accuracy"));
    }

    #[test]
    fn test_html_generation() {
        let mut report = MetricReport::new("Test".to_string());
        report.add_metric("accuracy".to_string(), 0.95);

        let html = report.to_html();
        assert!(html.contains("<html>"));
        assert!(html.contains("Test"));
        assert!(html.contains("accuracy"));
    }

    #[test]
    fn test_report_builder() {
        let report = ReportBuilder::new("Test".to_string())
            .with_metadata("model".to_string(), "ResNet".to_string())
            .with_num_samples(1000)
            .add_metric("accuracy".to_string(), 0.95)
            .build();

        assert_eq!(report.summary.num_samples, 1000);
        assert_eq!(report.metrics.len(), 1);
    }

    #[test]
    fn test_comparison_report() {
        let mut comp = ComparisonReport::new("Model Comparison".to_string());

        let mut report1 = MetricReport::new("Exp1".to_string());
        report1.add_metric("accuracy".to_string(), 0.95);

        let mut report2 = MetricReport::new("Exp2".to_string());
        report2.add_metric("accuracy".to_string(), 0.92);

        comp.add_experiment(report1);
        comp.add_experiment(report2);

        let md = comp.to_markdown();
        assert!(md.contains("Exp1"));
        assert!(md.contains("Exp2"));
        assert!(md.contains("0.95"));
    }
}
