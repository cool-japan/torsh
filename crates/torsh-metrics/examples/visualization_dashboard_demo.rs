//! Comprehensive demonstration of visualization capabilities in ToRSh Metrics
//!
//! This example shows how to:
//! - Export metrics to multiple formats (JSON, CSV, LaTeX, Markdown, HTML)
//! - Create interactive HTML dashboards
//! - Generate LaTeX reports for publications
//! - Build comprehensive metric visualizations
//!
//! Run with: cargo run --example visualization_dashboard_demo

use std::collections::HashMap;
use torsh_metrics::{
    ConfusionMatrixPlot, ExportFormat, InteractiveDashboard, LatexReportBuilder,
    MetricComparisonPlot, VisualizationAggregator,
};

fn main() {
    println!("==============================================");
    println!("ToRSh Metrics Visualization Demo");
    println!("==============================================\n");

    // Demonstrate confusion matrix exports
    confusion_matrix_export_demo();

    println!("\n");

    // Demonstrate interactive dashboard
    interactive_dashboard_demo();

    println!("\n");

    // Demonstrate LaTeX report generation
    latex_report_demo();

    println!("\n");

    // Demonstrate metric comparison
    metric_comparison_demo();

    println!("\n==============================================");
    println!("Demo completed successfully!");
    println!("Check the /tmp directory for generated files:");
    println!("  - /tmp/confusion_matrix.json");
    println!("  - /tmp/confusion_matrix.csv");
    println!("  - /tmp/confusion_matrix.tex");
    println!("  - /tmp/confusion_matrix.md");
    println!("  - /tmp/metrics_dashboard.html");
    println!("  - /tmp/metrics_report.tex");
    println!("==============================================");
}

/// Demonstrate confusion matrix export to multiple formats
fn confusion_matrix_export_demo() {
    println!("📊 Confusion Matrix Export Demo");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Create a confusion matrix for a 3-class problem
    let values = vec![
        vec![85.0, 10.0, 5.0], // True class 0
        vec![8.0, 88.0, 4.0],  // True class 1
        vec![3.0, 6.0, 91.0],  // True class 2
    ];

    let labels = vec!["Cat".to_string(), "Dog".to_string(), "Bird".to_string()];

    let cm_plot = ConfusionMatrixPlot::new(values, labels)
        .with_title("Animal Classification Results".to_string())
        .with_colormap("viridis".to_string());

    println!("\n1️⃣ JSON Export:");
    println!("   ─────────────");
    if let Ok(json) = cm_plot.to_json() {
        println!("   {}", &json[..200.min(json.len())]);
        println!("   ...");
        // Save to file
        if let Err(e) = std::fs::write("/tmp/confusion_matrix.json", json) {
            println!("   ⚠ Failed to save JSON: {}", e);
        } else {
            println!("   ✓ Saved to /tmp/confusion_matrix.json");
        }
    }

    println!("\n2️⃣ CSV Export:");
    println!("   ─────────────");
    let csv = cm_plot.to_csv();
    println!("{}", &csv[..150.min(csv.len())]);
    println!("   ...");
    if let Err(e) = std::fs::write("/tmp/confusion_matrix.csv", csv) {
        println!("   ⚠ Failed to save CSV: {}", e);
    } else {
        println!("   ✓ Saved to /tmp/confusion_matrix.csv");
    }

    println!("\n3️⃣ LaTeX Export:");
    println!("   ─────────────");
    let latex = cm_plot.to_latex();
    println!("{}", &latex[..200.min(latex.len())]);
    println!("   ...");
    if let Err(e) = std::fs::write("/tmp/confusion_matrix.tex", latex) {
        println!("   ⚠ Failed to save LaTeX: {}", e);
    } else {
        println!("   ✓ Saved to /tmp/confusion_matrix.tex");
    }

    println!("\n4️⃣ Markdown Export:");
    println!("   ─────────────");
    let markdown = cm_plot.to_markdown();
    println!("{}", &markdown[..150.min(markdown.len())]);
    println!("   ...");
    if let Err(e) = std::fs::write("/tmp/confusion_matrix.md", markdown) {
        println!("   ⚠ Failed to save Markdown: {}", e);
    } else {
        println!("   ✓ Saved to /tmp/confusion_matrix.md");
    }

    println!("\n5️⃣ HTML Export:");
    println!("   ─────────────");
    let html = cm_plot.to_html();
    println!("   Generated {} characters of HTML", html.len());
    println!("   ✓ HTML table with styling");
}

/// Demonstrate interactive HTML dashboard creation
fn interactive_dashboard_demo() {
    println!("📊 Interactive HTML Dashboard Demo");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let mut dashboard = InteractiveDashboard::new("Model Evaluation Dashboard");

    // Add confusion matrix
    let cm_values = vec![
        vec![92.0, 5.0, 3.0],
        vec![4.0, 94.0, 2.0],
        vec![2.0, 3.0, 95.0],
    ];

    let cm_labels = vec![
        "Class A".to_string(),
        "Class B".to_string(),
        "Class C".to_string(),
    ];

    let cm_plot = ConfusionMatrixPlot::new(cm_values, cm_labels)
        .with_title("Test Set Confusion Matrix".to_string());

    dashboard.add_confusion_matrix(&cm_plot);

    // Add custom metric summary
    let metrics_html = r#"
    <div class="metrics-summary">
        <div class="metric-card">
            <h4>Accuracy</h4>
            <p class="metric-value">93.67%</p>
        </div>
        <div class="metric-card">
            <h4>Precision (Macro)</h4>
            <p class="metric-value">93.33%</p>
        </div>
        <div class="metric-card">
            <h4>Recall (Macro)</h4>
            <p class="metric-value">93.67%</p>
        </div>
        <div class="metric-card">
            <h4>F1-Score (Macro)</h4>
            <p class="metric-value">93.50%</p>
        </div>
    </div>
    <style>
        .metrics-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .metric-card h4 {
            margin: 0 0 10px 0;
            font-size: 0.9rem;
            opacity: 0.9;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            margin: 0;
        }
    </style>
    "#;

    dashboard.add_custom_plot("Performance Metrics", metrics_html);

    // Save dashboard
    if let Err(e) = dashboard.save_to_file("/tmp/metrics_dashboard.html") {
        println!("⚠ Failed to save dashboard: {}", e);
    } else {
        println!("✓ Created interactive dashboard");
        println!("  • Responsive grid layout");
        println!("  • Hover effects and animations");
        println!("  • Click interactions on table cells");
        println!("  • Professional gradient styling");
        println!("  • Saved to /tmp/metrics_dashboard.html");
        println!("\n  💡 Open the file in a web browser to see the interactive dashboard!");
    }
}

/// Demonstrate LaTeX report generation for academic publications
fn latex_report_demo() {
    println!("📊 LaTeX Report Generation Demo");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let mut report = LatexReportBuilder::new("Deep Learning Model Evaluation")
        .with_author("ToRSh Metrics Framework")
        .with_document_class("article");

    // Add introduction section
    report.add_section(
        "Introduction",
        r"We present a comprehensive evaluation of our deep learning model on a multi-class classification task. The model was trained on a dataset consisting of three balanced classes with 1000 samples each.",
    );

    // Add methodology section
    report.add_section(
        "Methodology",
        r"The evaluation was performed using a stratified 80/20 train-test split. We report standard classification metrics including accuracy, precision, recall, and F1-score. All metrics are computed on the held-out test set of 600 samples.",
    );

    // Add results with confusion matrix
    let cm_values = vec![
        vec![175.0, 15.0, 10.0],
        vec![12.0, 180.0, 8.0],
        vec![8.0, 10.0, 182.0],
    ];

    let cm_labels = vec![
        "Class A".to_string(),
        "Class B".to_string(),
        "Class C".to_string(),
    ];

    let cm_plot =
        ConfusionMatrixPlot::new(cm_values, cm_labels).with_title("Test Set Results".to_string());

    report.add_confusion_matrix(&cm_plot);

    // Add performance table
    let performance_table = r"\begin{table}[h]
\centering
\caption{Classification Performance Metrics}
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Metric} & \textbf{Macro Avg} & \textbf{Weighted Avg} & \textbf{Accuracy} \\
\hline
Precision & 0.931 & 0.932 & - \\
Recall & 0.895 & 0.896 & - \\
F1-Score & 0.912 & 0.913 & - \\
Accuracy & - & - & 0.895 \\
\hline
\end{tabular}
\end{table}";

    report.add_section("Performance Metrics", performance_table);

    // Add discussion
    report.add_section(
        "Discussion",
        r"The model achieves strong performance across all classes with an overall accuracy of 89.5\%. The confusion matrix shows that misclassifications are relatively balanced across classes, suggesting no systematic bias. The F1-scores above 0.91 for all classes indicate good precision-recall trade-offs.",
    );

    // Save report
    if let Err(e) = report.save_to_file("/tmp/metrics_report.tex") {
        println!("⚠ Failed to save LaTeX report: {}", e);
    } else {
        println!("✓ Created LaTeX report");
        println!("  • Professional academic formatting");
        println!("  • Multiple sections with proper LaTeX structure");
        println!("  • Tables and figures ready for compilation");
        println!("  • Saved to /tmp/metrics_report.tex");
        println!("\n  💡 Compile with: pdflatex /tmp/metrics_report.tex");
    }
}

/// Demonstrate metric comparison across models
fn metric_comparison_demo() {
    println!("📊 Metric Comparison Demo");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let model_names = vec![
        "Baseline CNN".to_string(),
        "ResNet-18".to_string(),
        "EfficientNet-B0".to_string(),
        "Vision Transformer".to_string(),
    ];

    let mut metrics = HashMap::new();
    metrics.insert("Accuracy".to_string(), vec![0.82, 0.89, 0.91, 0.93]);
    metrics.insert("Precision".to_string(), vec![0.80, 0.88, 0.90, 0.92]);
    metrics.insert("Recall".to_string(), vec![0.81, 0.87, 0.89, 0.91]);
    metrics.insert("F1-Score".to_string(), vec![0.805, 0.875, 0.895, 0.915]);

    let comparison = MetricComparisonPlot::new(model_names, metrics)
        .with_title("Model Architecture Comparison".to_string());

    println!("\n📋 CSV Export:");
    let csv = comparison.to_csv();
    println!("{}", csv);

    println!("\n📄 JSON Export:");
    if let Ok(json) = comparison.to_json() {
        println!("Generated {} characters of JSON", json.len());
    }

    // Create aggregated export
    println!("\n📦 Aggregated Export Demo:");
    let mut agg = VisualizationAggregator::new();

    let cm = ConfusionMatrixPlot::new(
        vec![
            vec![45.0, 3.0, 2.0],
            vec![2.0, 47.0, 1.0],
            vec![1.0, 2.0, 47.0],
        ],
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
    );

    agg.add_confusion_matrix(cm);

    for format in &[
        ExportFormat::Json,
        ExportFormat::Csv,
        ExportFormat::Latex,
        ExportFormat::Markdown,
        ExportFormat::Html,
    ] {
        let exports = agg.export_all(*format);
        let format_name = match format {
            ExportFormat::Json => "JSON",
            ExportFormat::Csv => "CSV",
            ExportFormat::Latex => "LaTeX",
            ExportFormat::Markdown => "Markdown",
            ExportFormat::Html => "HTML",
        };
        println!("  ✓ Exported to {}: {} items", format_name, exports.len());
    }
}
