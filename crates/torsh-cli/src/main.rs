//! ToRSh CLI - Command-line tools for the ToRSh deep learning framework
//!
//! This CLI provides a comprehensive suite of tools for working with ToRSh models,
//! datasets, and the machine learning workflow.

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use colored::*;
use std::path::PathBuf;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod commands;
mod config;
mod utils;

use commands::*;

/// ToRSh CLI - Advanced deep learning framework command-line tools
#[derive(Parser)]
#[command(
    name = "torsh",
    author,
    version,
    about = "Command-line tools for the ToRSh deep learning framework",
    long_about = r#"
ToRSh CLI provides a comprehensive suite of command-line tools for machine learning workflows:

• Model operations: convert, optimize, quantize, and inspect models
• Training utilities: start training, resume from checkpoints, distributed training
• Dataset tools: download, preprocess, validate, and analyze datasets  
• Benchmarking: performance testing and profiling
• Hub integration: download and upload models to the ToRSh Hub
• Development tools: code generation, testing, and debugging

Examples:
  torsh model convert --input model.pth --output model.torsh --format torsh
  torsh train --config config.yaml --resume checkpoint.pth
  torsh benchmark --model resnet50 --batch-size 32
  torsh hub download microsoft/resnet50 --cache-dir ./models
"#
)]
struct Cli {
    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Enable quiet mode (minimal output)
    #[arg(short, long, global = true)]
    quiet: bool,

    /// Configuration file path
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,

    /// Output format (json, yaml, table)
    #[arg(long, global = true, default_value = "table")]
    output_format: String,

    /// Disable colored output
    #[arg(long, global = true)]
    no_color: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Model operations (convert, optimize, inspect)
    #[command(subcommand)]
    Model(ModelCommands),

    /// Training utilities
    #[command(subcommand)]
    Train(TrainCommands),

    /// Dataset operations
    #[command(subcommand)]
    Dataset(DatasetCommands),

    /// Benchmarking and profiling
    #[command(subcommand)]
    Benchmark(BenchmarkCommands),

    /// ToRSh Hub integration
    #[command(subcommand)]
    Hub(HubCommands),

    /// Development and debugging tools
    #[command(subcommand)]
    Dev(DevCommands),

    /// System information and diagnostics
    Info(InfoCommand),

    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: clap_complete::Shell,
    },

    /// Initialize new ToRSh project
    Init(InitCommand),

    /// Update ToRSh installation and models
    Update(UpdateCommand),
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    init_logging(cli.verbose, cli.quiet)?;

    // Disable colors if requested or if not in a terminal
    if cli.no_color || !console::Term::stdout().features().colors_supported() {
        colored::control::set_override(false);
    }

    // Load configuration
    let config = config::load_config(cli.config.as_deref()).await?;

    info!("ToRSh CLI starting");

    // Execute the command
    match cli.command {
        Commands::Model(cmd) => model::execute(cmd, &config, &cli.output_format).await,
        Commands::Train(cmd) => train::execute(cmd, &config, &cli.output_format).await,
        Commands::Dataset(cmd) => dataset::execute(cmd, &config, &cli.output_format).await,
        Commands::Benchmark(cmd) => benchmark::execute(cmd, &config, &cli.output_format).await,
        Commands::Hub(cmd) => hub::execute(cmd, &config, &cli.output_format).await,
        Commands::Dev(cmd) => dev::execute(cmd, &config, &cli.output_format).await,
        Commands::Info(cmd) => info::execute(cmd, &config, &cli.output_format).await,
        Commands::Completions { shell } => {
            generate_completions(shell);
            Ok(())
        }
        Commands::Init(cmd) => init::execute(cmd, &config, &cli.output_format).await,
        Commands::Update(cmd) => update::execute(cmd, &config, &cli.output_format).await,
    }
}

fn init_logging(verbose: bool, quiet: bool) -> Result<()> {
    let level = if quiet {
        tracing::Level::ERROR
    } else if verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| match level {
                tracing::Level::ERROR => tracing_subscriber::EnvFilter::new("error"),
                tracing::Level::WARN => tracing_subscriber::EnvFilter::new("warn"),
                tracing::Level::INFO => tracing_subscriber::EnvFilter::new("info"),
                tracing::Level::DEBUG => tracing_subscriber::EnvFilter::new("debug"),
                tracing::Level::TRACE => tracing_subscriber::EnvFilter::new("trace"),
            }),
        )
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .init();

    Ok(())
}

fn generate_completions(shell: clap_complete::Shell) {
    let mut app = Cli::command();
    let name = app.get_name().to_string();
    clap_complete::generate(shell, &mut app, name, &mut std::io::stdout());
}

/// Display the ToRSh banner
pub fn display_banner() {
    let banner = r#"
  ______         _____   _____ _     
 |__   _|       |  __ \ / ____| |    
    | | ___  _ _| |__) | (___ | |__  
    | |/ _ \| '__|  _  / \___ \| '_ \ 
   _| | (_) | |  | | \ \ ____) | | | |
  |_| \___/|_|  |_|  \_\_____/|_| |_|
                                     
"#;

    println!("{}", banner.bright_cyan().bold());
    println!(
        "{}",
        "ToRSh CLI - Advanced Deep Learning Framework Tools"
            .bright_white()
            .bold()
    );
    println!(
        "{}",
        format!("Version: {} | Build: {}", env!("CARGO_PKG_VERSION"), "dev").bright_black()
    );
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_cmd::Command;

    #[test]
    #[ignore = "Requires built torsh binary"]
    fn test_cli_help() {
        let mut cmd = Command::cargo_bin("torsh")
            .expect("Failed to find torsh binary - make sure to build it first");
        cmd.arg("--help").assert().success();
    }

    #[test]
    #[ignore = "Requires built torsh binary"]
    fn test_cli_version() {
        let mut cmd = Command::cargo_bin("torsh")
            .expect("Failed to find torsh binary - make sure to build it first");
        cmd.arg("--version").assert().success();
    }

    #[test]
    #[ignore = "Requires built torsh binary"]
    fn test_cli_info_command() {
        let mut cmd = Command::cargo_bin("torsh")
            .expect("Failed to find torsh binary - make sure to build it first");
        cmd.arg("info").assert().success();
    }

    #[test]
    fn test_version_check() {
        // Test that we can parse the version
        assert!(!env!("CARGO_PKG_VERSION").is_empty());
    }
}
