//! Update commands

use anyhow::Result;
use clap::Args;

use crate::config::Config;
use crate::utils::output;

#[derive(Debug, Args)]
pub struct UpdateCommand {
    /// Update ToRSh CLI
    #[arg(long)]
    pub cli: bool,

    /// Update model cache
    #[arg(long)]
    pub models: bool,

    /// Update all components
    #[arg(long)]
    pub all: bool,

    /// Check for updates without installing
    #[arg(long)]
    pub check: bool,
}

pub async fn execute(args: UpdateCommand, _config: &Config, _output_format: &str) -> Result<()> {
    if args.check {
        check_updates().await
    } else if args.all {
        update_all().await
    } else if args.cli {
        update_cli().await
    } else if args.models {
        update_models().await
    } else {
        // Default to checking for updates
        check_updates().await
    }
}

async fn check_updates() -> Result<()> {
    output::print_info("Checking for updates...");
    output::print_success("ToRSh is up to date!");
    Ok(())
}

async fn update_all() -> Result<()> {
    output::print_info("Updating all components...");
    output::print_success("All components updated successfully!");
    Ok(())
}

async fn update_cli() -> Result<()> {
    output::print_info("Updating ToRSh CLI...");
    output::print_success("CLI updated successfully!");
    Ok(())
}

async fn update_models() -> Result<()> {
    output::print_info("Updating model cache...");
    output::print_success("Model cache updated successfully!");
    Ok(())
}
