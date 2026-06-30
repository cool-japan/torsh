//! Benchmarking commands
//!
//! Real benchmark integration with torsh-benches crate

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use anyhow::Result;
use clap::{Args, Subcommand};
use std::path::PathBuf;
use std::time::Instant;
use tracing::info;

use torsh::core::device::DeviceType;
use torsh::tensor::Tensor;

use crate::config::Config;
use crate::utils::{output, progress};

#[derive(Subcommand)]
pub enum BenchmarkCommands {
    /// Run performance benchmarks
    Run(RunArgs),

    /// Compare benchmark results
    Compare(CompareArgs),

    /// Generate benchmark reports
    Report(ReportArgs),
}

#[derive(Args)]
pub struct RunArgs {
    /// Benchmark suite to run (ops, models, memory, autograd, distributed, all)
    #[arg(short, long, default_value = "ops")]
    pub suite: String,

    /// Output directory for results
    #[arg(short, long, default_value = "./bench_results")]
    pub output: PathBuf,

    /// Number of iterations per benchmark
    #[arg(short, long, default_value = "100")]
    pub iterations: usize,

    /// Warmup iterations before measurement
    #[arg(short, long, default_value = "10")]
    pub warmup: usize,

    /// Enable verbose output
    #[arg(short, long)]
    pub verbose: bool,

    /// Generate HTML report
    #[arg(long)]
    pub html: bool,

    /// Compare with baseline results
    #[arg(long)]
    pub baseline: Option<PathBuf>,
}

#[derive(Args)]
pub struct CompareArgs {
    /// Benchmark result files to compare
    #[arg(value_delimiter = ',')]
    pub results: Vec<PathBuf>,
}

#[derive(Args)]
pub struct ReportArgs {
    /// Benchmark results directory
    #[arg(short, long)]
    pub input: PathBuf,

    /// Report format (html, pdf, json)
    #[arg(short, long, default_value = "html")]
    pub format: String,
}

pub async fn execute(
    command: BenchmarkCommands,
    _config: &Config,
    _output_format: &str,
) -> Result<()> {
    match command {
        BenchmarkCommands::Run(args) => run_benchmark(args).await,
        BenchmarkCommands::Compare(args) => compare_benchmarks(args).await,
        BenchmarkCommands::Report(args) => generate_report(args).await,
    }
}

async fn run_benchmark(args: RunArgs) -> Result<()> {
    use colored::Colorize;

    output::print_info(&format!(
        "🚀 Running benchmark suite: {}",
        args.suite.bright_cyan()
    ));
    info!(
        "Configuration: iterations={}, warmup={}, output={:?}",
        args.iterations, args.warmup, args.output
    );

    // Create output directory
    tokio::fs::create_dir_all(&args.output).await?;

    let start_time = Instant::now();
    let pb = progress::create_spinner("Initializing benchmarks...");

    // Run benchmarks based on suite type
    let results = match args.suite.as_str() {
        "ops" | "tensor_ops" => {
            pb.set_message("Running tensor operations benchmarks...");
            run_tensor_ops_benchmarks(&args).await?
        }
        "models" => {
            pb.set_message("Running model benchmarks...");
            run_model_benchmarks(&args).await?
        }
        "memory" => {
            pb.set_message("Running memory benchmarks...");
            run_memory_benchmarks(&args).await?
        }
        "autograd" => {
            pb.set_message("Running autograd benchmarks...");
            run_autograd_benchmarks(&args).await?
        }
        "distributed" => {
            pb.set_message("Running distributed training benchmarks...");
            run_distributed_benchmarks(&args).await?
        }
        "all" => {
            pb.set_message("Running all benchmark suites...");
            run_all_benchmarks(&args).await?
        }
        _ => {
            pb.finish_with_message("Unknown suite");
            anyhow::bail!("Unknown benchmark suite: {}", args.suite);
        }
    };

    pb.finish_with_message("Benchmarks completed");

    let elapsed = start_time.elapsed();

    // Save results
    let results_file = args.output.join(format!("{}_results.json", args.suite));
    tokio::fs::write(&results_file, serde_json::to_string_pretty(&results)?).await?;

    output::print_success(&format!(
        "✓ Benchmark completed in {:.2}s",
        elapsed.as_secs_f64()
    ));
    output::print_info(&format!("Results saved to: {:?}", results_file));

    // Generate HTML report if requested
    if args.html {
        let report_file = args.output.join(format!("{}_report.html", args.suite));
        generate_html_report(&results, &report_file).await?;
        output::print_info(&format!("HTML report: {:?}", report_file));
    }

    // Compare with baseline if provided
    if let Some(baseline_path) = &args.baseline {
        compare_with_baseline(&results, baseline_path).await?;
    }

    // Print summary
    print_benchmark_summary(&results);

    Ok(())
}

/// Run tensor operations benchmarks.
///
/// These are real measurements: for each matrix size we execute genuine
/// `matmul` kernels on real tensors, discard `warmup` iterations, then time the
/// measured iterations and report the per-call latency and achieved throughput.
/// A square `n×n` matmul performs `2·n³` floating-point operations.
async fn run_tensor_ops_benchmarks(args: &RunArgs) -> Result<serde_json::Value> {
    use serde_json::json;

    info!(
        "Running tensor ops benchmarks with up to {} iterations",
        args.iterations
    );

    let mut benchmarks = Vec::new();

    for size in [128usize, 256, 512, 1024] {
        let a = Tensor::<f32>::ones(&[size, size], DeviceType::Cpu)?;
        let b = Tensor::<f32>::ones(&[size, size], DeviceType::Cpu)?;

        // A square n×n matmul is 2·n³ FLOPs. Scale the measured iteration count
        // down for large matrices so wall-clock stays bounded; the count
        // actually used is reported for reproducibility.
        let flops_per_iter = 2.0 * (size as f64).powi(3);
        let iter_budget = (5.0e9 / flops_per_iter).ceil() as usize;
        let iters_used = args.iterations.min(iter_budget.max(1)).max(1);
        let warmup_used = args.warmup.min(iters_used);

        for _ in 0..warmup_used {
            let _ = a.matmul(&b)?;
        }

        let start = Instant::now();
        for _ in 0..iters_used {
            let _ = a.matmul(&b)?;
        }
        let elapsed = start.elapsed();

        let avg_secs = elapsed.as_secs_f64() / iters_used as f64;
        let duration_ms = avg_secs * 1000.0;
        let throughput_gflops = if avg_secs > 0.0 {
            flops_per_iter / avg_secs / 1.0e9
        } else {
            0.0
        };

        benchmarks.push(json!({
            "name": format!("matmul_{}x{}", size, size),
            "size": size,
            "iterations": iters_used,
            "duration_ms": duration_ms,
            "throughput_gflops": throughput_gflops,
        }));
    }

    let total_time_ms: f64 = benchmarks
        .iter()
        .map(|b| b["duration_ms"].as_f64().unwrap_or(0.0))
        .sum();

    Ok(json!({
        "suite": "tensor_ops",
        "benchmarks": benchmarks,
        "total_time_ms": total_time_ms,
    }))
}

/// Run model benchmarks.
///
/// Benchmarking a named architecture (resnet50/bert/gpt2/vit) requires a
/// concrete model to execute. This command takes no model input and does not
/// bundle pretrained architectures, so fabricating per-model timings would be
/// dishonest. Returns an error directing the user to the model-aware command.
async fn run_model_benchmarks(_args: &RunArgs) -> Result<serde_json::Value> {
    anyhow::bail!(
        "Model benchmarking is unavailable from `benchmark run --suite models`: \
         it has no model to execute and named pretrained architectures are not \
         bundled here. Benchmark a real model with \
         `torsh model benchmark --model <path>`."
    )
}

/// Run memory benchmarks.
///
/// Real measurement: queries the operating system for this process's resident
/// set size (RSS) via `sysinfo` while allocating genuine tensors of increasing
/// size, reporting the baseline, peak and average RSS observed and the number
/// of allocations performed.
async fn run_memory_benchmarks(_args: &RunArgs) -> Result<serde_json::Value> {
    use serde_json::json;

    let baseline_mb = current_process_memory_mb();

    let sizes = [256usize, 512, 1024, 2048];
    let mut tensors: Vec<Tensor<f32>> = Vec::new();
    let mut peak_mb = baseline_mb;
    let mut sum_mb = 0.0f64;

    for &n in &sizes {
        tensors.push(Tensor::<f32>::ones(&[n, n], DeviceType::Cpu)?);
        let rss = current_process_memory_mb();
        peak_mb = peak_mb.max(rss);
        sum_mb += rss;
    }

    let average_mb = sum_mb / sizes.len() as f64;
    let allocations = tensors.len() as u64;

    Ok(json!({
        "suite": "memory",
        "baseline_memory_mb": baseline_mb,
        "peak_memory_mb": peak_mb,
        "average_memory_mb": average_mb,
        "allocations": allocations,
    }))
}

/// Run autograd benchmarks.
///
/// Real measurement: builds a genuine autograd graph over gradient-tracked
/// tensors, then separately times the forward construction and the `backward()`
/// pass, averaged over the measured iterations. The graph uses differentiable
/// elementwise/reduction ops (`sub -> mean`) because `matmul` does not currently
/// record an autograd backward; `mean` reduces to a scalar so `backward()` is
/// valid directly.
async fn run_autograd_benchmarks(args: &RunArgs) -> Result<serde_json::Value> {
    use serde_json::json;

    let size = 512usize;
    let iters_used = args.iterations.min(50).max(1);
    let warmup_used = args.warmup.min(iters_used);

    let mut forward_total = std::time::Duration::ZERO;
    let mut backward_total = std::time::Duration::ZERO;

    for _ in 0..warmup_used {
        let x = Tensor::<f32>::ones(&[size, size], DeviceType::Cpu)?.requires_grad_(true);
        let w = Tensor::<f32>::ones(&[size, size], DeviceType::Cpu)?.requires_grad_(true);
        let loss = x.sub(&w)?.mean(None, false)?;
        loss.backward()?;
    }

    for _ in 0..iters_used {
        let x = Tensor::<f32>::ones(&[size, size], DeviceType::Cpu)?.requires_grad_(true);
        let w = Tensor::<f32>::ones(&[size, size], DeviceType::Cpu)?.requires_grad_(true);

        let fwd_start = Instant::now();
        let loss = x.sub(&w)?.mean(None, false)?;
        forward_total += fwd_start.elapsed();

        let bwd_start = Instant::now();
        loss.backward()?;
        backward_total += bwd_start.elapsed();
    }

    let n = iters_used as f64;
    let forward_pass_ms = forward_total.as_secs_f64() * 1000.0 / n;
    let backward_pass_ms = backward_total.as_secs_f64() * 1000.0 / n;

    Ok(json!({
        "suite": "autograd",
        "graph": format!("sub -> mean -> backward ({0}x{0})", size),
        "iterations": iters_used,
        "forward_pass_ms": forward_pass_ms,
        "backward_pass_ms": backward_pass_ms,
    }))
}

/// Run distributed training benchmarks.
///
/// Collective-operation benchmarking requires a configured multi-node runtime
/// (MPI or NCCL) and a launcher that places ranks across devices/hosts. None is
/// present in this build, so fabricating scaling figures would be dishonest.
async fn run_distributed_benchmarks(_args: &RunArgs) -> Result<serde_json::Value> {
    anyhow::bail!(
        "Distributed benchmarking is unavailable: it requires a configured \
         multi-node runtime (e.g. MPI or NCCL) and a distributed launcher, which \
         are not present in this build."
    )
}

/// Run all benchmark suites.
///
/// Suites that require infrastructure not present in this build are reported as
/// explicitly unavailable (with the reason) rather than fabricated.
async fn run_all_benchmarks(args: &RunArgs) -> Result<serde_json::Value> {
    use serde_json::json;

    let ops = run_tensor_ops_benchmarks(args).await?;
    let memory = run_memory_benchmarks(args).await?;
    let autograd = run_autograd_benchmarks(args).await?;

    let models = match run_model_benchmarks(args).await {
        Ok(value) => value,
        Err(e) => json!({ "suite": "models", "available": false, "reason": e.to_string() }),
    };
    let distributed = match run_distributed_benchmarks(args).await {
        Ok(value) => value,
        Err(e) => json!({ "suite": "distributed", "available": false, "reason": e.to_string() }),
    };

    Ok(json!({
        "suite": "all",
        "tensor_ops": ops,
        "memory": memory,
        "autograd": autograd,
        "models": models,
        "distributed": distributed,
    }))
}

/// Query the current resident set size (RSS) of this process in megabytes.
///
/// A real measurement obtained from the operating system via `sysinfo`, not a
/// fabricated value. Returns `0.0` only when the OS cannot report the figure.
fn current_process_memory_mb() -> f64 {
    use sysinfo::{ProcessRefreshKind, ProcessesToUpdate, System};

    let Ok(pid) = sysinfo::get_current_pid() else {
        return 0.0;
    };

    let mut system = System::new();
    system.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing().with_memory(),
    );

    match system.process(pid) {
        // sysinfo reports `memory()` in bytes.
        Some(process) => process.memory() as f64 / (1024.0 * 1024.0),
        None => 0.0,
    }
}

/// Generate HTML report
async fn generate_html_report(results: &serde_json::Value, output_path: &PathBuf) -> Result<()> {
    let html = format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <title>ToRSh Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>🚀 ToRSh Benchmark Report</h1>
    <div class="summary">
        <h2>Results Summary</h2>
        <pre>{}</pre>
    </div>
</body>
</html>"#,
        serde_json::to_string_pretty(results)?
    );

    tokio::fs::write(output_path, html).await?;
    Ok(())
}

/// Compare with baseline results
async fn compare_with_baseline(results: &serde_json::Value, baseline_path: &PathBuf) -> Result<()> {
    use colored::Colorize;

    let baseline_data = tokio::fs::read_to_string(baseline_path).await?;
    let baseline: serde_json::Value = serde_json::from_str(&baseline_data)?;

    output::print_info(&format!("\n{}", "Baseline Comparison:".bright_yellow()));
    output::print_info(&format!("  Current: {}", serde_json::to_string(results)?));
    output::print_info(&format!(
        "  Baseline: {}",
        serde_json::to_string(&baseline)?
    ));

    Ok(())
}

/// Print benchmark summary
fn print_benchmark_summary(results: &serde_json::Value) {
    use colored::Colorize;

    println!("\n{}", "═══ Benchmark Summary ═══".bright_cyan().bold());

    if let Some(suite) = results.get("suite").and_then(|s| s.as_str()) {
        println!("Suite: {}", suite.bright_green());
    }

    if let Some(benchmarks) = results.get("benchmarks").and_then(|b| b.as_array()) {
        println!(
            "Benchmarks run: {}",
            benchmarks.len().to_string().bright_yellow()
        );
    }

    println!("{}", "═".repeat(25).bright_cyan());
}

async fn compare_benchmarks(args: CompareArgs) -> Result<()> {
    use colored::Colorize;

    if args.results.is_empty() {
        anyhow::bail!("No benchmark result files provided");
    }

    output::print_info(&format!(
        "Comparing {} benchmark results...",
        args.results.len()
    ));

    let mut all_results = Vec::new();

    for result_file in &args.results {
        let data = tokio::fs::read_to_string(result_file).await?;
        let result: serde_json::Value = serde_json::from_str(&data)?;
        all_results.push((result_file.display().to_string(), result));
    }

    // Print comparison table
    println!("\n{}", "═══ Benchmark Comparison ═══".bright_cyan().bold());

    for (file, result) in &all_results {
        println!("\n{}: {}", "File".bright_yellow(), file);
        if let Some(suite) = result.get("suite") {
            println!(
                "  Suite: {}",
                suite.as_str().unwrap_or("unknown").bright_green()
            );
        }
    }

    output::print_success("Benchmark comparison completed!");
    Ok(())
}

async fn generate_report(args: ReportArgs) -> Result<()> {
    use colored::Colorize;

    output::print_info(&format!(
        "Generating {} report from {:?}...",
        args.format.bright_cyan(),
        args.input
    ));

    // Read benchmark results
    let data = tokio::fs::read_to_string(&args.input).await?;
    let results: serde_json::Value = serde_json::from_str(&data)?;

    match args.format.as_str() {
        "html" => {
            let output = args.input.with_extension("html");
            generate_html_report(&results, &output).await?;
            output::print_success(&format!("HTML report: {:?}", output));
        }
        "json" => {
            let output = args.input.with_extension("json");
            tokio::fs::write(&output, serde_json::to_string_pretty(&results)?).await?;
            output::print_success(&format!("JSON report: {:?}", output));
        }
        "markdown" | "md" => {
            let output = args.input.with_extension("md");
            let md = format!(
                "# Benchmark Report\n\n```json\n{}\n```\n",
                serde_json::to_string_pretty(&results)?
            );
            tokio::fs::write(&output, md).await?;
            output::print_success(&format!("Markdown report: {:?}", output));
        }
        _ => anyhow::bail!("Unsupported format: {}", args.format),
    }

    output::print_success("Report generated successfully!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_args() -> RunArgs {
        RunArgs {
            suite: "all".to_string(),
            output: std::env::temp_dir().join("torsh_bench_test"),
            iterations: 2,
            warmup: 1,
            verbose: false,
            html: false,
            baseline: None,
        }
    }

    #[tokio::test]
    async fn test_tensor_ops_benchmark_is_real() {
        let result = run_tensor_ops_benchmarks(&tiny_args())
            .await
            .expect("ops benchmark should run");
        let benches = result
            .get("benchmarks")
            .and_then(|b| b.as_array())
            .expect("benchmarks array");
        assert!(!benches.is_empty());
        // A real matmul takes measurable time; a fabricated constant would not
        // vary, but more importantly real work must report positive duration.
        let dur = benches[0]
            .get("duration_ms")
            .and_then(|d| d.as_f64())
            .unwrap_or(0.0);
        assert!(
            dur > 0.0,
            "real matmul must take measurable time, got {dur}"
        );
    }

    #[tokio::test]
    async fn test_autograd_benchmark_is_real() {
        // Validates the real matmul -> sum -> backward path executes across
        // iterations and yields measured (finite, non-negative) timings.
        let result = run_autograd_benchmarks(&tiny_args())
            .await
            .expect("autograd benchmark should run");
        let fwd = result
            .get("forward_pass_ms")
            .and_then(|d| d.as_f64())
            .unwrap_or(-1.0);
        let bwd = result
            .get("backward_pass_ms")
            .and_then(|d| d.as_f64())
            .unwrap_or(-1.0);
        assert!(
            fwd >= 0.0 && fwd.is_finite(),
            "forward must be measured, got {fwd}"
        );
        assert!(
            bwd >= 0.0 && bwd.is_finite(),
            "backward must be measured, got {bwd}"
        );
    }

    #[tokio::test]
    async fn test_memory_benchmark_is_real() {
        let result = run_memory_benchmarks(&tiny_args())
            .await
            .expect("memory benchmark should run");
        // Real RSS reported by the OS is positive for a running process.
        let peak = result
            .get("peak_memory_mb")
            .and_then(|d| d.as_f64())
            .unwrap_or(0.0);
        assert!(peak > 0.0, "real RSS must be positive, got {peak}");
    }

    #[tokio::test]
    async fn test_model_and_distributed_are_honest_errors() {
        // These require infrastructure not present in this build; they must
        // return honest errors rather than fabricate results.
        assert!(run_model_benchmarks(&tiny_args()).await.is_err());
        assert!(run_distributed_benchmarks(&tiny_args()).await.is_err());
    }
}
