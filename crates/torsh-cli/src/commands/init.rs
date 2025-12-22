//! Project initialization commands

use anyhow::Result;
use clap::Args;
use std::path::{Path, PathBuf};

use crate::config::Config;
use crate::utils::output;

#[derive(Debug, Args)]
pub struct InitCommand {
    /// Project name
    #[arg(short, long)]
    pub name: Option<String>,

    /// Project directory
    #[arg(short, long)]
    pub directory: Option<PathBuf>,

    /// Project template (basic, vision, nlp, custom)
    #[arg(short, long, default_value = "basic")]
    pub template: String,

    /// Enable Git repository initialization
    #[arg(long)]
    pub git: bool,

    /// Use interactive mode
    #[arg(short, long)]
    pub interactive: bool,
}

pub async fn execute(args: InitCommand, _config: &Config, _output_format: &str) -> Result<()> {
    output::print_info("Initializing new ToRSh project...");

    let project_name = args.name.unwrap_or_else(|| "torsh-project".to_string());
    let project_dir = args
        .directory
        .unwrap_or_else(|| PathBuf::from(&project_name));

    // Create project directory
    tokio::fs::create_dir_all(&project_dir).await?;

    // Create basic project structure
    create_project_structure(&project_dir, &args.template).await?;

    output::print_success(&format!(
        "Project '{}' initialized successfully!",
        project_name
    ));
    output::print_info(&format!("Location: {}", project_dir.display()));

    Ok(())
}

async fn create_project_structure(dir: &Path, template: &str) -> Result<()> {
    // Create basic directories
    let src_dir = dir.join("src");
    tokio::fs::create_dir_all(&src_dir).await?;

    // Create main.rs
    let _main_content = match template {
        "vision" => {
            r#"//! Vision project template

use anyhow::Result;

fn main() -> Result<()> {
    println!("ToRSh Vision Project");

    // TODO: Add your vision model training code here
    // Example:
    // let model = create_vision_model();
    // let dataset = load_image_dataset()?;
    // train_model(model, dataset)?;

    Ok(())
}
"#
        }
        "nlp" => {
            r#"//! NLP project template

use anyhow::Result;

fn main() -> Result<()> {
    println!("ToRSh NLP Project");

    // TODO: Add your NLP model training code here
    // Example:
    // let model = create_text_model();
    // let dataset = load_text_dataset()?;
    // train_model(model, dataset)?;

    Ok(())
}
"#
        }
        _ => {
            r#"//! Basic ToRSh project template

use anyhow::Result;

fn main() -> Result<()> {
    println!("ToRSh Basic Project");

    // TODO: Add your machine learning code here
    // Example:
    // let model = create_model();
    // let dataset = load_dataset()?;
    // train_model(model, dataset)?;

    Ok(())
}
"#
        }
    };

    // Use placeholder content since we don't have actual template files
    let main_rs = src_dir.join("main.rs");
    tokio::fs::write(
        &main_rs,
        "// ToRSh project\nfn main() {\n    println!(\"Hello, ToRSh!\");\n}",
    )
    .await?;

    // Create Cargo.toml
    let cargo_toml = dir.join("Cargo.toml");
    let cargo_content = r#"[package]
name = "torsh-project"
version = "0.1.0"
edition = "2021"

[dependencies]
torsh = "0.1.0-alpha.2"
"#;
    tokio::fs::write(&cargo_toml, cargo_content).await?;

    Ok(())
}
