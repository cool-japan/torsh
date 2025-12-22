//! Metric visualization utilities
//!
//! This module provides utilities for generating plot-ready data from metrics.
//! The data can be used with plotting libraries like plotters, plotly, or exported
//! to formats consumable by Python visualization libraries.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Plot-ready data for confusion matrix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusionMatrixPlot {
    /// Matrix values (row-major order)
    pub values: Vec<Vec<f64>>,
    /// Class labels
    pub labels: Vec<String>,
    /// Whether values are normalized
    pub normalized: bool,
    /// Title for the plot
    pub title: String,
    /// Color map suggestion
    pub colormap: String,
}

impl ConfusionMatrixPlot {
    /// Create a new confusion matrix plot
    pub fn new(values: Vec<Vec<f64>>, labels: Vec<String>) -> Self {
        Self {
            values,
            labels,
            normalized: false,
            title: "Confusion Matrix".to_string(),
            colormap: "Blues".to_string(),
        }
    }

    /// Set whether values are normalized
    pub fn with_normalized(mut self, normalized: bool) -> Self {
        self.normalized = normalized;
        self
    }

    /// Set the title
    pub fn with_title(mut self, title: String) -> Self {
        self.title = title;
        self
    }

    /// Set the colormap
    pub fn with_colormap(mut self, colormap: String) -> Self {
        self.colormap = colormap;
        self
    }

    /// Export to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Export to CSV format
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();

        // Header
        csv.push_str(",");
        for label in &self.labels {
            csv.push_str(label);
            csv.push(',');
        }
        csv.push('\n');

        // Data rows
        for (i, row) in self.values.iter().enumerate() {
            csv.push_str(&self.labels[i]);
            csv.push(',');
            for value in row {
                csv.push_str(&format!("{:.4},", value));
            }
            csv.push('\n');
        }

        csv
    }

    /// Export to LaTeX format
    pub fn to_latex(&self) -> String {
        let mut latex = String::new();

        latex.push_str("\\begin{table}[h]\n");
        latex.push_str("\\centering\n");
        latex.push_str(&format!("\\caption{{{}}}\n", self.title));
        latex.push_str("\\begin{tabular}{|c|");
        for _ in &self.labels {
            latex.push_str("c|");
        }
        latex.push_str("}\n");
        latex.push_str("\\hline\n");

        // Header
        latex.push_str("& ");
        for (i, label) in self.labels.iter().enumerate() {
            latex.push_str(&format!("\\textbf{{{}}}", label));
            if i < self.labels.len() - 1 {
                latex.push_str(" & ");
            }
        }
        latex.push_str(" \\\\\n");
        latex.push_str("\\hline\n");

        // Data rows
        for (i, row) in self.values.iter().enumerate() {
            latex.push_str(&format!("\\textbf{{{}}} & ", self.labels[i]));
            for (j, value) in row.iter().enumerate() {
                latex.push_str(&format!("{:.2}", value));
                if j < row.len() - 1 {
                    latex.push_str(" & ");
                }
            }
            latex.push_str(" \\\\\n");
        }

        latex.push_str("\\hline\n");
        latex.push_str("\\end{tabular}\n");
        latex.push_str("\\end{table}\n");

        latex
    }

    /// Export to Markdown format
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str(&format!("# {}\n\n", self.title));
        md.push_str("|");
        for label in &self.labels {
            md.push_str(&format!(" {} |", label));
        }
        md.push('\n');

        // Separator
        md.push_str("|");
        for _ in &self.labels {
            md.push_str(" --- |");
        }
        md.push('\n');

        // Data rows
        for (i, row) in self.values.iter().enumerate() {
            md.push_str(&format!("| **{}** |", self.labels[i]));
            for value in row {
                md.push_str(&format!(" {:.2} |", value));
            }
            md.push('\n');
        }

        md
    }

    /// Export to HTML format
    pub fn to_html(&self) -> String {
        let mut html = String::new();

        html.push_str(&format!("<div class=\"confusion-matrix\">\n"));
        html.push_str(&format!("<h3>{}</h3>\n", self.title));
        html.push_str("<table class=\"matrix-table\">\n");
        html.push_str("<thead>\n<tr>\n<th></th>\n");

        // Header
        for label in &self.labels {
            html.push_str(&format!("<th>{}</th>\n", label));
        }
        html.push_str("</tr>\n</thead>\n<tbody>\n");

        // Data rows
        for (i, row) in self.values.iter().enumerate() {
            html.push_str("<tr>\n");
            html.push_str(&format!("<th>{}</th>\n", self.labels[i]));
            for value in row {
                html.push_str(&format!("<td>{:.2}</td>\n", value));
            }
            html.push_str("</tr>\n");
        }

        html.push_str("</tbody>\n</table>\n</div>\n");

        html
    }
}

/// Plot-ready data for ROC curve
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ROCCurvePlot {
    /// False positive rates
    pub fpr: Vec<f64>,
    /// True positive rates
    pub tpr: Vec<f64>,
    /// Area under the curve
    pub auc: f64,
    /// Class label (for multi-class)
    pub label: Option<String>,
    /// Title for the plot
    pub title: String,
}

impl ROCCurvePlot {
    /// Create a new ROC curve plot
    pub fn new(fpr: Vec<f64>, tpr: Vec<f64>, auc: f64) -> Self {
        Self {
            fpr,
            tpr,
            auc,
            label: None,
            title: format!("ROC Curve (AUC = {:.4})", auc),
        }
    }

    /// Set the class label
    pub fn with_label(mut self, label: String) -> Self {
        self.label = Some(label.clone());
        self.title = format!("ROC Curve - {} (AUC = {:.4})", label, self.auc);
        self
    }

    /// Export to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Export to CSV format
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();
        csv.push_str("fpr,tpr\n");

        for (fpr, tpr) in self.fpr.iter().zip(self.tpr.iter()) {
            csv.push_str(&format!("{:.6},{:.6}\n", fpr, tpr));
        }

        csv
    }
}

/// Plot-ready data for Precision-Recall curve
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PRCurvePlot {
    /// Precision values
    pub precision: Vec<f64>,
    /// Recall values
    pub recall: Vec<f64>,
    /// Average precision
    pub average_precision: f64,
    /// Class label (for multi-class)
    pub label: Option<String>,
    /// Title for the plot
    pub title: String,
}

impl PRCurvePlot {
    /// Create a new PR curve plot
    pub fn new(precision: Vec<f64>, recall: Vec<f64>, average_precision: f64) -> Self {
        Self {
            precision,
            recall,
            average_precision,
            label: None,
            title: format!("Precision-Recall Curve (AP = {:.4})", average_precision),
        }
    }

    /// Set the class label
    pub fn with_label(mut self, label: String) -> Self {
        self.label = Some(label.clone());
        self.title = format!(
            "Precision-Recall Curve - {} (AP = {:.4})",
            label, self.average_precision
        );
        self
    }

    /// Export to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Export to CSV format
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();
        csv.push_str("recall,precision\n");

        for (recall, precision) in self.recall.iter().zip(self.precision.iter()) {
            csv.push_str(&format!("{:.6},{:.6}\n", recall, precision));
        }

        csv
    }
}

/// Plot-ready data for learning curve
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningCurvePlot {
    /// Training sizes
    pub train_sizes: Vec<usize>,
    /// Training scores (mean)
    pub train_scores_mean: Vec<f64>,
    /// Training scores (std)
    pub train_scores_std: Vec<f64>,
    /// Validation scores (mean)
    pub val_scores_mean: Vec<f64>,
    /// Validation scores (std)
    pub val_scores_std: Vec<f64>,
    /// Metric name
    pub metric_name: String,
    /// Title for the plot
    pub title: String,
}

impl LearningCurvePlot {
    /// Create a new learning curve plot
    pub fn new(
        train_sizes: Vec<usize>,
        train_scores_mean: Vec<f64>,
        train_scores_std: Vec<f64>,
        val_scores_mean: Vec<f64>,
        val_scores_std: Vec<f64>,
        metric_name: String,
    ) -> Self {
        let title = format!("Learning Curve - {}", metric_name);
        Self {
            train_sizes,
            train_scores_mean,
            train_scores_std,
            val_scores_mean,
            val_scores_std,
            metric_name,
            title,
        }
    }

    /// Export to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Export to CSV format
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();
        csv.push_str("train_size,train_mean,train_std,val_mean,val_std\n");

        for i in 0..self.train_sizes.len() {
            csv.push_str(&format!(
                "{},{:.6},{:.6},{:.6},{:.6}\n",
                self.train_sizes[i],
                self.train_scores_mean[i],
                self.train_scores_std[i],
                self.val_scores_mean[i],
                self.val_scores_std[i]
            ));
        }

        csv
    }
}

/// Plot-ready data for calibration curve
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationCurvePlot {
    /// Mean predicted probabilities
    pub mean_predicted_probs: Vec<f64>,
    /// Fraction of positives
    pub fraction_of_positives: Vec<f64>,
    /// Sample counts per bin
    pub sample_counts: Vec<usize>,
    /// Expected calibration error
    pub ece: f64,
    /// Title for the plot
    pub title: String,
}

impl CalibrationCurvePlot {
    /// Create a new calibration curve plot
    pub fn new(
        mean_predicted_probs: Vec<f64>,
        fraction_of_positives: Vec<f64>,
        sample_counts: Vec<usize>,
        ece: f64,
    ) -> Self {
        Self {
            mean_predicted_probs,
            fraction_of_positives,
            sample_counts,
            ece,
            title: format!("Calibration Curve (ECE = {:.4})", ece),
        }
    }

    /// Export to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Export to CSV format
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();
        csv.push_str("mean_predicted_prob,fraction_of_positives,sample_count\n");

        for i in 0..self.mean_predicted_probs.len() {
            csv.push_str(&format!(
                "{:.6},{:.6},{}\n",
                self.mean_predicted_probs[i], self.fraction_of_positives[i], self.sample_counts[i]
            ));
        }

        csv
    }
}

/// Plot-ready data for feature importance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportancePlot {
    /// Feature names
    pub feature_names: Vec<String>,
    /// Importance values
    pub importances: Vec<f64>,
    /// Standard deviations (optional)
    pub std: Option<Vec<f64>>,
    /// Title for the plot
    pub title: String,
}

impl FeatureImportancePlot {
    /// Create a new feature importance plot
    pub fn new(feature_names: Vec<String>, importances: Vec<f64>) -> Self {
        Self {
            feature_names,
            importances,
            std: None,
            title: "Feature Importance".to_string(),
        }
    }

    /// Set standard deviations
    pub fn with_std(mut self, std: Vec<f64>) -> Self {
        self.std = Some(std);
        self
    }

    /// Sort by importance (descending)
    pub fn sorted(mut self) -> Self {
        let mut indices: Vec<usize> = (0..self.importances.len()).collect();
        indices.sort_by(|&a, &b| {
            self.importances[b]
                .partial_cmp(&self.importances[a])
                .unwrap()
        });

        let sorted_names: Vec<String> = indices
            .iter()
            .map(|&i| self.feature_names[i].clone())
            .collect();
        let sorted_importances: Vec<f64> = indices.iter().map(|&i| self.importances[i]).collect();
        let sorted_std = self
            .std
            .map(|std| indices.iter().map(|&i| std[i]).collect());

        self.feature_names = sorted_names;
        self.importances = sorted_importances;
        self.std = sorted_std;
        self
    }

    /// Export to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Export to CSV format
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();

        if self.std.is_some() {
            csv.push_str("feature,importance,std\n");
            let std = self.std.as_ref().unwrap();
            for i in 0..self.feature_names.len() {
                csv.push_str(&format!(
                    "{},{:.6},{:.6}\n",
                    self.feature_names[i], self.importances[i], std[i]
                ));
            }
        } else {
            csv.push_str("feature,importance\n");
            for i in 0..self.feature_names.len() {
                csv.push_str(&format!(
                    "{},{:.6}\n",
                    self.feature_names[i], self.importances[i]
                ));
            }
        }

        csv
    }
}

/// Plot-ready data for metric comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricComparisonPlot {
    /// Model names
    pub model_names: Vec<String>,
    /// Metrics per model
    pub metrics: HashMap<String, Vec<f64>>,
    /// Title for the plot
    pub title: String,
}

impl MetricComparisonPlot {
    /// Create a new metric comparison plot
    pub fn new(model_names: Vec<String>, metrics: HashMap<String, Vec<f64>>) -> Self {
        Self {
            model_names,
            metrics,
            title: "Model Comparison".to_string(),
        }
    }

    /// Set the title
    pub fn with_title(mut self, title: String) -> Self {
        self.title = title;
        self
    }

    /// Export to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Export to CSV format
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();

        // Header
        csv.push_str("model");
        for metric_name in self.metrics.keys() {
            csv.push(',');
            csv.push_str(metric_name);
        }
        csv.push('\n');

        // Data rows
        for (i, model_name) in self.model_names.iter().enumerate() {
            csv.push_str(model_name);
            for metric_values in self.metrics.values() {
                csv.push_str(&format!(",{:.6}", metric_values[i]));
            }
            csv.push('\n');
        }

        csv
    }
}

/// Visualization export format
#[derive(Debug, Clone, Copy)]
pub enum ExportFormat {
    Json,
    Csv,
    Latex,
    Markdown,
    Html,
}

/// Visualization data aggregator
pub struct VisualizationAggregator {
    confusion_matrices: Vec<ConfusionMatrixPlot>,
    roc_curves: Vec<ROCCurvePlot>,
    pr_curves: Vec<PRCurvePlot>,
    learning_curves: Vec<LearningCurvePlot>,
}

impl VisualizationAggregator {
    /// Create a new visualization aggregator
    pub fn new() -> Self {
        Self {
            confusion_matrices: Vec::new(),
            roc_curves: Vec::new(),
            pr_curves: Vec::new(),
            learning_curves: Vec::new(),
        }
    }

    /// Add a confusion matrix
    pub fn add_confusion_matrix(&mut self, plot: ConfusionMatrixPlot) {
        self.confusion_matrices.push(plot);
    }

    /// Add a ROC curve
    pub fn add_roc_curve(&mut self, plot: ROCCurvePlot) {
        self.roc_curves.push(plot);
    }

    /// Add a PR curve
    pub fn add_pr_curve(&mut self, plot: PRCurvePlot) {
        self.pr_curves.push(plot);
    }

    /// Add a learning curve
    pub fn add_learning_curve(&mut self, plot: LearningCurvePlot) {
        self.learning_curves.push(plot);
    }

    /// Export all visualizations
    pub fn export_all(&self, format: ExportFormat) -> HashMap<String, String> {
        let mut exports = HashMap::new();

        for (i, cm) in self.confusion_matrices.iter().enumerate() {
            let key = format!("confusion_matrix_{}", i);
            let data = match format {
                ExportFormat::Json => cm.to_json().unwrap_or_default(),
                ExportFormat::Csv => cm.to_csv(),
                ExportFormat::Latex => cm.to_latex(),
                ExportFormat::Markdown => cm.to_markdown(),
                ExportFormat::Html => cm.to_html(),
            };
            exports.insert(key, data);
        }

        for (i, roc) in self.roc_curves.iter().enumerate() {
            let key = format!("roc_curve_{}", i);
            let data = match format {
                ExportFormat::Json => roc.to_json().unwrap_or_default(),
                ExportFormat::Csv => roc.to_csv(),
                ExportFormat::Latex => format!("% ROC Curve data for {}", roc.title),
                ExportFormat::Markdown => format!("## {}\n\nAUC: {:.4}", roc.title, roc.auc),
                ExportFormat::Html => format!(
                    "<div><h3>{}</h3><p>AUC: {:.4}</p></div>",
                    roc.title, roc.auc
                ),
            };
            exports.insert(key, data);
        }

        for (i, pr) in self.pr_curves.iter().enumerate() {
            let key = format!("pr_curve_{}", i);
            let data = match format {
                ExportFormat::Json => pr.to_json().unwrap_or_default(),
                ExportFormat::Csv => pr.to_csv(),
                ExportFormat::Latex => format!("% PR Curve data for {}", pr.title),
                ExportFormat::Markdown => format!(
                    "## {}\n\nAverage Precision: {:.4}",
                    pr.title, pr.average_precision
                ),
                ExportFormat::Html => format!(
                    "<div><h3>{}</h3><p>Average Precision: {:.4}</p></div>",
                    pr.title, pr.average_precision
                ),
            };
            exports.insert(key, data);
        }

        exports
    }
}

impl Default for VisualizationAggregator {
    fn default() -> Self {
        Self::new()
    }
}

/// Interactive HTML dashboard builder
pub struct InteractiveDashboard {
    title: String,
    plots: Vec<(String, String)>, // (plot_title, html_content)
    styles: String,
    scripts: String,
}

impl InteractiveDashboard {
    /// Create a new interactive dashboard
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            plots: Vec::new(),
            styles: Self::default_styles(),
            scripts: Self::default_scripts(),
        }
    }

    /// Add a confusion matrix plot
    pub fn add_confusion_matrix(&mut self, plot: &ConfusionMatrixPlot) {
        self.plots.push((plot.title.clone(), plot.to_html()));
    }

    /// Add a custom HTML plot
    pub fn add_custom_plot(&mut self, title: impl Into<String>, html: impl Into<String>) {
        self.plots.push((title.into(), html.into()));
    }

    /// Set custom styles
    pub fn with_custom_styles(mut self, styles: impl Into<String>) -> Self {
        self.styles = styles.into();
        self
    }

    /// Set custom scripts
    pub fn with_custom_scripts(mut self, scripts: impl Into<String>) -> Self {
        self.scripts = scripts.into();
        self
    }

    /// Generate complete HTML dashboard
    pub fn to_html(&self) -> String {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n");
        html.push_str("<html lang=\"en\">\n");
        html.push_str("<head>\n");
        html.push_str("    <meta charset=\"UTF-8\">\n");
        html.push_str(
            "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n",
        );
        html.push_str(&format!("    <title>{}</title>\n", self.title));
        html.push_str("    <style>\n");
        html.push_str(&self.styles);
        html.push_str("    </style>\n");
        html.push_str("</head>\n");
        html.push_str("<body>\n");
        html.push_str(&format!(
            "    <h1 class=\"dashboard-title\">{}</h1>\n",
            self.title
        ));
        html.push_str("    <div class=\"dashboard-container\">\n");

        for (plot_title, plot_html) in &self.plots {
            html.push_str("        <div class=\"plot-section\">\n");
            html.push_str(&format!("            <h2>{}</h2>\n", plot_title));
            html.push_str(&format!("            {}\n", plot_html));
            html.push_str("        </div>\n");
        }

        html.push_str("    </div>\n");
        html.push_str("    <script>\n");
        html.push_str(&self.scripts);
        html.push_str("    </script>\n");
        html.push_str("</body>\n");
        html.push_str("</html>\n");

        html
    }

    /// Save dashboard to file
    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let html = self.to_html();
        let mut file = File::create(path)?;
        file.write_all(html.as_bytes())?;

        Ok(())
    }

    /// Default CSS styles for the dashboard
    fn default_styles() -> String {
        r#"
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .dashboard-title {
            text-align: center;
            color: white;
            font-size: 2.5rem;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }

        .dashboard-container {
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 25px;
        }

        .plot-section {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .plot-section:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.2);
        }

        .plot-section h2 {
            color: #333;
            font-size: 1.5rem;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }

        .matrix-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }

        .matrix-table th,
        .matrix-table td {
            padding: 12px;
            text-align: center;
            border: 1px solid #ddd;
        }

        .matrix-table th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
        }

        .matrix-table tbody tr:nth-child(even) {
            background-color: #f8f9fa;
        }

        .matrix-table tbody tr:hover {
            background-color: #e9ecef;
            transition: background-color 0.2s ease;
        }

        .confusion-matrix h3 {
            color: #555;
            font-size: 1.2rem;
            margin-bottom: 15px;
        }

        @media (max-width: 768px) {
            .dashboard-container {
                grid-template-columns: 1fr;
            }

            .dashboard-title {
                font-size: 1.8rem;
            }
        }
        "#.to_string()
    }

    /// Default JavaScript for interactivity
    fn default_scripts() -> String {
        r#"
        // Add interactivity to tables
        document.addEventListener('DOMContentLoaded', function() {
            const tables = document.querySelectorAll('.matrix-table');

            tables.forEach(table => {
                const cells = table.querySelectorAll('tbody td');

                cells.forEach(cell => {
                    cell.addEventListener('click', function() {
                        const value = this.textContent;
                        const row = this.parentElement.querySelector('th').textContent;
                        const col = this.cellIndex;
                        const colHeader = table.querySelectorAll('thead th')[col].textContent;

                        alert(`Value: ${value}\nRow: ${row}\nColumn: ${colHeader}`);
                    });
                });
            });

            // Add animation on scroll
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.style.opacity = '1';
                        entry.target.style.transform = 'translateY(0)';
                    }
                });
            });

            document.querySelectorAll('.plot-section').forEach(section => {
                section.style.opacity = '0';
                section.style.transform = 'translateY(20px)';
                section.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                observer.observe(section);
            });
        });
        "#
        .to_string()
    }
}

impl Default for InteractiveDashboard {
    fn default() -> Self {
        Self::new("Metrics Dashboard")
    }
}

/// LaTeX document builder for comprehensive reports
pub struct LatexReportBuilder {
    title: String,
    author: Option<String>,
    sections: Vec<(String, String)>, // (section_title, content)
    document_class: String,
}

impl LatexReportBuilder {
    /// Create a new LaTeX report builder
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            author: None,
            sections: Vec::new(),
            document_class: "article".to_string(),
        }
    }

    /// Set the author
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Set document class
    pub fn with_document_class(mut self, class: impl Into<String>) -> Self {
        self.document_class = class.into();
        self
    }

    /// Add a section with content
    pub fn add_section(&mut self, title: impl Into<String>, content: impl Into<String>) {
        self.sections.push((title.into(), content.into()));
    }

    /// Add a confusion matrix
    pub fn add_confusion_matrix(&mut self, plot: &ConfusionMatrixPlot) {
        self.add_section(plot.title.clone(), plot.to_latex());
    }

    /// Generate complete LaTeX document
    pub fn to_latex(&self) -> String {
        let mut latex = String::new();

        latex.push_str(&format!("\\documentclass{{{}}}\n", self.document_class));
        latex.push_str("\\usepackage[utf8]{inputenc}\n");
        latex.push_str("\\usepackage{graphicx}\n");
        latex.push_str("\\usepackage{booktabs}\n");
        latex.push_str("\\usepackage{float}\n");
        latex.push_str("\\usepackage{hyperref}\n");
        latex.push_str("\\usepackage{geometry}\n");
        latex.push_str("\\geometry{margin=1in}\n\n");

        latex.push_str(&format!("\\title{{{}}}\n", self.title));
        if let Some(ref author) = self.author {
            latex.push_str(&format!("\\author{{{}}}\n", author));
        }
        latex.push_str("\\date{\\today}\n\n");

        latex.push_str("\\begin{document}\n\n");
        latex.push_str("\\maketitle\n\n");

        for (section_title, content) in &self.sections {
            latex.push_str(&format!("\\section{{{}}}\n\n", section_title));
            latex.push_str(content);
            latex.push_str("\n\n");
        }

        latex.push_str("\\end{document}\n");

        latex
    }

    /// Save LaTeX document to file
    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let latex = self.to_latex();
        let mut file = File::create(path)?;
        file.write_all(latex.as_bytes())?;

        Ok(())
    }
}

impl Default for LatexReportBuilder {
    fn default() -> Self {
        Self::new("Metrics Report")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confusion_matrix_plot() {
        let values = vec![vec![10.0, 2.0], vec![3.0, 15.0]];
        let labels = vec!["Class 0".to_string(), "Class 1".to_string()];

        let plot = ConfusionMatrixPlot::new(values, labels);

        assert_eq!(plot.values.len(), 2);
        assert_eq!(plot.labels.len(), 2);

        let json = plot.to_json();
        assert!(json.is_ok());

        let csv = plot.to_csv();
        assert!(csv.contains("Class 0"));
    }

    #[test]
    fn test_roc_curve_plot() {
        let fpr = vec![0.0, 0.1, 0.2, 0.5, 1.0];
        let tpr = vec![0.0, 0.5, 0.8, 0.9, 1.0];
        let auc = 0.85;

        let plot = ROCCurvePlot::new(fpr, tpr, auc);

        assert_eq!(plot.fpr.len(), 5);
        assert_eq!(plot.auc, 0.85);

        let json = plot.to_json();
        assert!(json.is_ok());
    }

    #[test]
    fn test_feature_importance_sorted() {
        let names = vec![
            "feat1".to_string(),
            "feat2".to_string(),
            "feat3".to_string(),
        ];
        let importances = vec![0.3, 0.5, 0.2];

        let plot = FeatureImportancePlot::new(names, importances).sorted();

        assert_eq!(plot.feature_names[0], "feat2");
        assert_eq!(plot.importances[0], 0.5);
    }

    #[test]
    fn test_visualization_aggregator() {
        let mut agg = VisualizationAggregator::new();

        let cm = ConfusionMatrixPlot::new(
            vec![vec![10.0, 2.0], vec![3.0, 15.0]],
            vec!["A".to_string(), "B".to_string()],
        );

        agg.add_confusion_matrix(cm);

        let exports = agg.export_all(ExportFormat::Json);
        assert!(!exports.is_empty());
    }

    #[test]
    fn test_confusion_matrix_latex_export() {
        let values = vec![vec![10.0, 2.0], vec![3.0, 15.0]];
        let labels = vec!["Class 0".to_string(), "Class 1".to_string()];

        let plot = ConfusionMatrixPlot::new(values, labels);
        let latex = plot.to_latex();

        assert!(latex.contains("\\begin{table}"));
        assert!(latex.contains("\\begin{tabular}"));
        assert!(latex.contains("Class 0"));
        assert!(latex.contains("Class 1"));
        assert!(latex.contains("\\end{table}"));
    }

    #[test]
    fn test_confusion_matrix_markdown_export() {
        let values = vec![vec![10.0, 2.0], vec![3.0, 15.0]];
        let labels = vec!["Class 0".to_string(), "Class 1".to_string()];

        let plot = ConfusionMatrixPlot::new(values, labels);
        let md = plot.to_markdown();

        assert!(md.contains("# Confusion Matrix"));
        assert!(md.contains("Class 0"));
        assert!(md.contains("Class 1"));
        assert!(md.contains("|"));
    }

    #[test]
    fn test_confusion_matrix_html_export() {
        let values = vec![vec![10.0, 2.0], vec![3.0, 15.0]];
        let labels = vec!["Class 0".to_string(), "Class 1".to_string()];

        let plot = ConfusionMatrixPlot::new(values, labels);
        let html = plot.to_html();

        assert!(html.contains("<table"));
        assert!(html.contains("<th>Class 0</th>"));
        assert!(html.contains("<th>Class 1</th>"));
        assert!(html.contains("<td>"));
    }

    #[test]
    fn test_interactive_dashboard() {
        let mut dashboard = InteractiveDashboard::new("Test Dashboard");

        let cm = ConfusionMatrixPlot::new(
            vec![vec![10.0, 2.0], vec![3.0, 15.0]],
            vec!["A".to_string(), "B".to_string()],
        );

        dashboard.add_confusion_matrix(&cm);

        let html = dashboard.to_html();

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Test Dashboard"));
        assert!(html.contains("<table"));
        assert!(html.contains("<style>"));
        assert!(html.contains("<script>"));
    }

    #[test]
    fn test_interactive_dashboard_custom_plot() {
        let mut dashboard = InteractiveDashboard::new("Custom Dashboard");

        dashboard.add_custom_plot("My Plot", "<div>Custom content</div>");

        let html = dashboard.to_html();

        assert!(html.contains("My Plot"));
        assert!(html.contains("Custom content"));
    }

    #[test]
    fn test_latex_report_builder() {
        let mut report = LatexReportBuilder::new("Test Report")
            .with_author("Test Author")
            .with_document_class("article");

        report.add_section("Introduction", "This is a test introduction.");

        let cm = ConfusionMatrixPlot::new(
            vec![vec![10.0, 2.0], vec![3.0, 15.0]],
            vec!["A".to_string(), "B".to_string()],
        );

        report.add_confusion_matrix(&cm);

        let latex = report.to_latex();

        assert!(latex.contains("\\documentclass{article}"));
        assert!(latex.contains("\\title{Test Report}"));
        assert!(latex.contains("\\author{Test Author}"));
        assert!(latex.contains("\\section{Introduction}"));
        assert!(latex.contains("\\section{Confusion Matrix}"));
        assert!(latex.contains("\\begin{document}"));
        assert!(latex.contains("\\end{document}"));
    }

    #[test]
    fn test_latex_report_builder_multiple_sections() {
        let mut report = LatexReportBuilder::new("Multi-section Report");

        report.add_section("Section 1", "Content 1");
        report.add_section("Section 2", "Content 2");
        report.add_section("Section 3", "Content 3");

        let latex = report.to_latex();

        assert!(latex.contains("\\section{Section 1}"));
        assert!(latex.contains("\\section{Section 2}"));
        assert!(latex.contains("\\section{Section 3}"));
    }

    #[test]
    fn test_dashboard_default_styles() {
        let dashboard = InteractiveDashboard::default();
        let html = dashboard.to_html();

        assert!(html.contains("body"));
        assert!(html.contains("font-family"));
        assert!(html.contains(".dashboard-title"));
        assert!(html.contains(".plot-section"));
    }

    #[test]
    fn test_dashboard_default_scripts() {
        let dashboard = InteractiveDashboard::default();
        let html = dashboard.to_html();

        assert!(html.contains("document.addEventListener"));
        assert!(html.contains("DOMContentLoaded"));
    }
}
