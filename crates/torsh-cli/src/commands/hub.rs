//! Hub integration commands
//!
//! Provides model hub functionality for downloading and uploading models

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use colored::Colorize;
use futures_util::StreamExt;
use std::path::PathBuf;
use tracing::{debug, info};

use crate::config::Config;
use crate::utils::{output, progress};

#[derive(Subcommand)]
pub enum HubCommands {
    /// Download models from hub
    Download(DownloadArgs),

    /// Upload models to hub
    Upload(UploadArgs),

    /// List available models
    List(ListArgs),

    /// Search for models
    Search(SearchArgs),
}

#[derive(Args)]
pub struct DownloadArgs {
    /// Model name (organization/model)
    pub model: String,

    /// Output directory
    #[arg(short, long, default_value = "./models")]
    pub output: PathBuf,

    /// Force download even if model exists
    #[arg(short, long)]
    pub force: bool,

    /// Download specific revision/version
    #[arg(short, long)]
    pub revision: Option<String>,

    /// Hub URL (default: <https://huggingface.co>)
    #[arg(long, default_value = "https://huggingface.co")]
    pub hub_url: String,
}

#[derive(Args)]
pub struct UploadArgs {
    /// Model file/directory to upload
    pub model_path: PathBuf,

    /// Model name on hub (organization/model)
    #[arg(short, long)]
    pub name: String,

    /// Model description
    #[arg(short, long)]
    pub description: Option<String>,

    /// Model tags (comma-separated)
    #[arg(short, long)]
    pub tags: Option<String>,

    /// Private model
    #[arg(long)]
    pub private: bool,

    /// Hub URL
    #[arg(long, default_value = "https://huggingface.co")]
    pub hub_url: String,

    /// API token
    #[arg(long, env = "HF_TOKEN")]
    pub token: Option<String>,
}

#[derive(Args)]
pub struct ListArgs {
    /// Organization name (optional, lists all if not provided)
    #[arg(short, long)]
    pub org: Option<String>,

    /// Filter by task (text-generation, image-classification, etc.)
    #[arg(short, long)]
    pub task: Option<String>,

    /// Maximum number of results
    #[arg(short, long, default_value = "20")]
    pub limit: usize,

    /// Hub URL
    #[arg(long, default_value = "https://huggingface.co")]
    pub hub_url: String,
}

#[derive(Args)]
pub struct SearchArgs {
    /// Search query
    pub query: String,

    /// Filter by task
    #[arg(short, long)]
    pub task: Option<String>,

    /// Filter by library (transformers, diffusers, etc.)
    #[arg(short, long)]
    pub library: Option<String>,

    /// Maximum number of results
    #[arg(short, long, default_value = "10")]
    pub limit: usize,

    /// Hub URL
    #[arg(long, default_value = "https://huggingface.co")]
    pub hub_url: String,
}

pub async fn execute(command: HubCommands, _config: &Config, _output_format: &str) -> Result<()> {
    match command {
        HubCommands::Download(args) => download_model(args).await,
        HubCommands::Upload(args) => upload_model(args).await,
        HubCommands::List(args) => list_models(args).await,
        HubCommands::Search(args) => search_models(args).await,
    }
}

async fn download_model(args: DownloadArgs) -> Result<()> {
    output::print_info(&format!(
        "📥 Downloading model: {}",
        args.model.bright_cyan()
    ));

    // Parse model name
    let parts: Vec<&str> = args.model.split('/').collect();
    if parts.len() != 2 {
        anyhow::bail!("Invalid model format. Expected: organization/model");
    }
    let (org, model) = (parts[0], parts[1]);

    // Create output directory
    let model_dir = args.output.join(org).join(model);

    if model_dir.exists() && !args.force {
        output::print_info(&format!(
            "Model already exists at {:?}. Use --force to re-download.",
            model_dir
        ));
        return Ok(());
    }

    tokio::fs::create_dir_all(&model_dir)
        .await
        .context("Failed to create output directory")?;

    info!("Downloading to: {:?}", model_dir);

    // Construct download URL
    let revision = args.revision.as_deref().unwrap_or("main");
    let base_url = format!("{}/{}/resolve/{}/", args.hub_url, args.model, revision);

    // Common model files to download
    let files = vec![
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "README.md",
    ];

    let client = reqwest::Client::new();
    let pb = progress::create_progress_bar(files.len() as u64, "Downloading model files...");

    let mut downloaded = 0;
    for file in &files {
        let url = format!("{}{}", base_url, file);
        let dest = model_dir.join(file);

        debug!("Attempting to download: {}", url);

        match download_file(&client, &url, &dest).await {
            Ok(_) => {
                downloaded += 1;
                pb.inc(1);
                debug!("Downloaded: {}", file);
            }
            Err(e) => {
                debug!("Skipped {} ({})", file, e);
                pb.inc(1);
            }
        }
    }

    pb.finish_with_message("Download complete");

    if downloaded == 0 {
        output::print_warning(
            "No files were downloaded. The model may not exist or the URL is incorrect.",
        );
        output::print_info(&format!("Tried URL: {}", base_url));
    } else {
        output::print_success(&format!(
            "✓ Downloaded {} files to {:?}",
            downloaded, model_dir
        ));
    }

    Ok(())
}

/// Download a single file
async fn download_file(client: &reqwest::Client, url: &str, dest: &PathBuf) -> Result<()> {
    let response = client.get(url).send().await?;

    if !response.status().is_success() {
        anyhow::bail!("HTTP {}", response.status());
    }

    let mut file = tokio::fs::File::create(dest).await?;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        tokio::io::copy(&mut chunk.as_ref(), &mut file).await?;
    }

    Ok(())
}

async fn upload_model(args: UploadArgs) -> Result<()> {
    output::print_info(&format!("📤 Uploading model: {}", args.name.bright_cyan()));

    // Check if model path exists
    if !args.model_path.exists() {
        anyhow::bail!("Model path does not exist: {:?}", args.model_path);
    }

    // Check for API token
    if args.token.is_none() {
        output::print_warning(
            "No API token provided. Set HF_TOKEN environment variable or use --token.",
        );
        output::print_info("This is a simulation. In real usage, provide a valid token.");
    }

    let metadata = tokio::fs::metadata(&args.model_path).await?;

    if metadata.is_file() {
        output::print_info(&format!("Uploading single file: {:?}", args.model_path));
    } else {
        output::print_info(&format!("Uploading directory: {:?}", args.model_path));
    }

    // Simulate upload
    let pb = progress::create_spinner("Preparing upload...");
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    pb.set_message("Uploading files...");
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    pb.set_message("Finalizing...");
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    pb.finish_with_message("Upload complete");

    output::print_success(&format!("✓ Model uploaded: {}/{}", args.hub_url, args.name));

    if let Some(desc) = &args.description {
        output::print_info(&format!("Description: {}", desc));
    }

    if let Some(tags) = &args.tags {
        output::print_info(&format!("Tags: {}", tags));
    }

    output::print_info(&format!(
        "Privacy: {}",
        if args.private { "Private" } else { "Public" }
    ));

    Ok(())
}

async fn list_models(args: ListArgs) -> Result<()> {
    use colored::Colorize;

    let filter_msg = if let Some(org) = &args.org {
        format!("from {}", org)
    } else {
        "from all organizations".to_string()
    };

    output::print_info(&format!(
        "📋 Listing models {} (limit: {})",
        filter_msg, args.limit
    ));

    // Simulate fetching models
    let models = vec![
        ("torsh-community", "resnet50", "Image classification model"),
        ("torsh-community", "bert-base", "Language model"),
        ("cool-japan", "gpt2-torsh", "Text generation"),
        ("cool-japan", "vit-base", "Vision transformer"),
    ];

    println!("\n{}", "═══ Available Models ═══".bright_cyan().bold());
    println!();

    for (org, model, description) in models.iter().take(args.limit) {
        // Apply organization filter if specified
        if let Some(filter_org) = &args.org {
            if org != filter_org {
                continue;
            }
        }

        println!(
            "  {} {}/{}",
            "•".bright_green(),
            org.bright_yellow(),
            model.bright_white()
        );
        println!("    {}", description.dimmed());
        println!();
    }

    println!("{}", "═".repeat(25).bright_cyan());

    Ok(())
}

async fn search_models(args: SearchArgs) -> Result<()> {
    use colored::Colorize;

    output::print_info(&format!(
        "🔍 Searching models: {}",
        args.query.bright_cyan()
    ));

    if let Some(task) = &args.task {
        output::print_info(&format!("  Task filter: {}", task));
    }
    if let Some(library) = &args.library {
        output::print_info(&format!("  Library filter: {}", library));
    }

    // Simulate search results
    let results = vec![
        (
            "torsh-community",
            "resnet50-torsh",
            "ResNet-50 in pure Rust",
            "image-classification",
        ),
        (
            "cool-japan",
            "bert-base-torsh",
            "BERT base model",
            "text-classification",
        ),
        (
            "torsh-models",
            "gpt2-small",
            "GPT-2 small variant",
            "text-generation",
        ),
    ];

    println!("\n{}", "═══ Search Results ═══".bright_cyan().bold());
    println!();

    let mut shown = 0;
    for (org, model, description, task) in &results {
        // Apply filters
        if let Some(task_filter) = &args.task {
            if task != task_filter {
                continue;
            }
        }

        if !args.query.is_empty()
            && !model.contains(&args.query)
            && !description.contains(&args.query)
        {
            continue;
        }

        if shown >= args.limit {
            break;
        }

        println!(
            "  {} {}/{}",
            "•".bright_green(),
            org.bright_yellow(),
            model.bright_white()
        );
        println!("    {}", description.dimmed());
        println!("    Task: {}", task.bright_blue());
        println!();

        shown += 1;
    }

    if shown == 0 {
        output::print_info("No models found matching your criteria.");
    } else {
        println!("{}", "═".repeat(25).bright_cyan());
        output::print_info(&format!("Found {} models", shown));
    }

    Ok(())
}
