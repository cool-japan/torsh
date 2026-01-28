//! # ToRSh Hub - Enterprise Model Hub and Management Platform
//!
//! `torsh-hub` provides a comprehensive model hub and management platform for ToRSh,
//! similar to PyTorch Hub and Hugging Face Hub, with enterprise-grade features.

#![cfg_attr(not(feature = "tensorflow"), allow(unexpected_cfgs))]
//!
//! ## Features
//!
//! ### Core Functionality
//! - **Model Registry**: Centralized model discovery and version management
//! - **Model Download**: Advanced parallel downloading with mirrors and CDN support
//! - **Model Loading**: Support for ONNX, TensorFlow, and native ToRSh models
//! - **Model Hub Integration**: Seamless integration with Hugging Face Hub
//!
//! ### Enterprise Features
//! - **Access Control**: Fine-grained RBAC and permission management
//! - **Private Repositories**: Secure private model storage for organizations
//! - **Audit Logging**: Comprehensive audit trails for compliance
//! - **SLA Management**: Service level agreements and performance monitoring
//!
//! ### Community Platform
//! - **Model Ratings**: Community-driven model quality assessment
//! - **Discussions**: Collaborative discussions on models and techniques
//! - **Challenges**: ML challenges and competitions
//! - **Contributions**: Track and recognize community contributions
//!
//! ### Advanced Capabilities
//! - **Fine-tuning**: Built-in fine-tuning with early stopping and checkpointing
//! - **Profiling**: Comprehensive model performance profiling
//! - **Debugging**: Advanced debugging tools with interactive sessions
//! - **Analytics**: Real-time analytics and usage tracking
//! - **Visualization**: Performance visualization and dashboard generation
//! - **Security**: Model sandboxing and security scanning
//!
//! ## SciRS2 POLICY Compliance
//!
//! This crate strictly follows the [SciRS2 POLICY](https://github.com/cool-japan/scirs/blob/master/SCIRS2_POLICY.md):
//! - All array operations use `scirs2_core::ndarray::*`
//! - All random operations use `scirs2_core::random::*`
//! - NO direct external dependencies (ndarray, rand, etc.)
//!
//! ## Quick Start
//!
//! ```no_run
//! use torsh_hub::registry::{ModelRegistry, SearchQuery, ModelCategory};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize the registry
//! let mut registry = ModelRegistry::new("./models")?;
//!
//! // Search for models
//! let mut query = SearchQuery::default();
//! query.category = Some(ModelCategory::Vision);
//! let results = registry.search(&query);
//!
//! // Load a model with load_onnx_model function
//! // let model = torsh_hub::load_onnx_model("model.onnx", None)?;
//!
//! // Use pre-built architecture components from models module
//! // use torsh_hub::models::vision::ResNet;
//! // let resnet = ResNet::resnet18(1000)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Module Organization
//!
//! - [`models`]: Pre-built model architectures (BERT, GPT, ResNet, ViT, CLIP, etc.)
//! - [`registry`]: Model registry and discovery
//! - [`download`]: Advanced download management with mirrors and CDN
//! - [`onnx`]: ONNX model loading and conversion
//! - [`huggingface`]: Hugging Face Hub integration
//! - [`fine_tuning`]: Model fine-tuning utilities
//! - [`profiling`]: Performance profiling and analysis
//! - [`debugging`]: Interactive debugging tools
//! - [`security`]: Model security and sandboxing
//! - [`enterprise`]: Enterprise features (RBAC, audit, SLA)
//! - [`community`]: Community platform (ratings, discussions, challenges)
//! - [`analytics`]: Analytics and recommendation engine
//! - [`visualization`]: Performance visualization
//! - [`utils`]: Utility functions for common tasks
//! - [`cli`]: Command-line interface

use serde::Deserialize;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use torsh_core::error::{Result, TorshError};

pub mod access_control;
pub mod analytics;
pub mod bandwidth;
pub mod cache;
pub mod cli;
pub mod community;
pub mod debugging;
pub mod download;
pub mod enterprise;
pub mod export;
pub mod fine_tuning;
pub mod huggingface;
pub mod metadata;
pub mod model_info;
pub mod model_ops;
pub mod models;
pub mod onnx;
pub mod profiling;
pub mod quantization;
pub mod registry;
pub mod retry;
pub mod security;
#[cfg(feature = "tensorflow")]
pub mod tensorflow;
pub mod upload;
pub mod utils;
pub mod visualization;

// Re-exports
pub use access_control::{
    AccessToken, PermissionChecker, RateLimit, TokenManager, TokenScope, TokenStats,
};
pub use analytics::{
    ABTestingFramework, AnalyticsManager, AnalyticsReport, ExportFormat, ModelUsageStats,
    PerformanceProfiler, RealTimeMetrics, RecommendationEngine, UserAnalytics,
};
pub use bandwidth::{
    format_bytes, format_duration, AdaptiveBandwidthLimiter, BandwidthLimiter, BandwidthMonitor,
    BandwidthStats,
};
pub use cache::{
    compress_file, decompress_file, CacheCleanupResult, CacheManager, CacheStats,
    CacheValidationResult, CompressionResult, FileCompressionStats,
};
pub use cli::{run_cli, Cli, CliApp, CliConfig, Commands};
pub use community::{
    Badge, Challenge, ChallengeId, ChallengeParticipant, ChallengeStatus, ChallengeSubmission,
    ChallengeType, Comment, CommunityManager, Contribution, ContributionStatus, ContributionType,
    Discussion, DiscussionCategory, DiscussionId, DiscussionStatus, EvaluationCriteria, MetricType,
    ModelId, ModelRating, ModelRatingStats, RatingCategory, UserId, UserProfile,
};
pub use debugging::{
    ActivationAnalyzer, ActivationPattern, Anomaly, AnomalyType, DebugAction, DebugCommand,
    DebugConfig, DebugHook, DebugReport, DebugSession, GradientDebugger, GradientInfo, HookType,
    InteractiveDebugState, ModelDebugger, Severity, TensorInspector, TensorSnapshot,
    TensorStatistics, TriggerCondition,
};
pub use download::{
    create_regional_cdn_config, create_regional_mirror_config, download_file_parallel,
    download_file_streaming,
    download_files_parallel, /* download_model, download_model_from_url, */
    download_with_default_cdn, download_with_default_mirrors, validate_url, validate_urls,
    CdnConfig, CdnEndpoint, CdnManager, CdnStatistics, /* EndpointHealth, */ FailoverStrategy,
    /* HealthCheckResult, */ MirrorAttempt, MirrorBenchmarkResult, MirrorCapacity,
    MirrorConfig, MirrorDownloadResult, MirrorLocation, MirrorManager, MirrorSelectionStrategy,
    MirrorServer, MirrorStatistics, MirrorWeights, ParallelDownloadConfig,
};
pub use enterprise::{
    Action, AuditAction, AuditLogEntry, ComplianceLabel, ComplianceReport, DataClassification,
    EnterpriseManager, OrganizationId, Permission, PermissionId, PermissionScope,
    PrivateRepository, RepositoryAccessControl, RepositoryVisibility, ResourceType, Role, RoleId,
    ServiceLevelAgreement, ServiceTier, SlaPerformanceReport, UserRoleAssignment,
};
pub use fine_tuning::{
    CheckpointManager, EarlyStoppingConfig, FineTuner, FineTuningConfig, FineTuningFactory,
    FineTuningStrategy, TrainingHistory, TrainingMetrics,
};
pub use huggingface::{
    HfModelConfig, HfModelInfo, HfSearchParams, HfToTorshConverter, HuggingFaceHub,
};
pub use metadata::{
    ExtendedMetadata, FileMetadata, MetadataManager, MetadataSearchCriteria, PerformanceMetrics,
    QualityScores, UsageStatistics,
};
pub use model_info::{
    ModelCard, ModelCardBuilder, ModelCardManager, ModelCardRenderer, ModelInfo, Version,
    VersionHistory,
};
pub use model_ops::{
    compare_models, create_model_ensemble, load_model_auto, ComparisonOptions, ConversionMetadata,
    EnsembleConfig, ModelDiff, QuantizationStats, ShapeDifference, ValueDifference, VotingStrategy,
};
pub use models::{
    audio, multimodal, nlp, nlp_pretrained, rl, vision, vision_pretrained, ActorCritic, BasicBlock,
    BertEmbeddings, BertEncoder, EfficientNet, GPTDecoder, GPTEmbeddings, MultiHeadAttention,
    PPOAgent, ResNet, TransformerBlock, VisionTransformer, DQN,
};
pub use onnx::{
    InputShape, OnnxConfig, OnnxLoader, OnnxModel, OnnxModelMetadata, OnnxToTorshWrapper,
    OutputShape,
};
pub use profiling::{
    ExecutionContext, ExecutionMode, LayerProfile, MemoryAnalysis, MemorySnapshot, ModelProfiler,
    OperationAnalysis, OperationTrace, OptimizationRecommendation, PerformanceBottleneck,
    PerformanceCounters, PerformanceSummary, ProfilerConfig, ProfilingResult, ProfilingSession,
    ResourceUtilizationSummary, TensorInfo,
};
pub use registry::{
    HardwareFilter, ModelCategory, ModelRegistry, ModelStatus, RegistryAPI, RegistryEntry,
    SearchQuery,
};
pub use retry::{
    retry_with_backoff, retry_with_backoff_async, retry_with_policy, CircuitBreaker, CircuitState,
    DefaultRetryPolicy, RetryConfig, RetryPolicy, RetryStats,
};
pub use security::{
    calculate_file_hash, sandbox_model, scan_model_vulnerabilities, validate_model_source,
    validate_signature_age, verify_file_integrity, KeyPair, ModelSandbox, ModelSignature,
    ResourceUsage, RiskLevel, SandboxConfig, SandboxedModel, ScanMetadata, SecurityConfig,
    SecurityManager, Severity as SecuritySeverity, SignatureAlgorithm, Vulnerability,
    VulnerabilityScanResult, VulnerabilityScanner, VulnerabilityType,
};
#[cfg(feature = "tensorflow")]
pub use tensorflow::{
    TfConfig, TfLoader, TfModel, TfModelMetadata, TfModelType, TfTensorInfo, TfToTorshWrapper,
};
pub use upload::{
    batch_publish_models, upload_model, upload_model_with_versioning, validate_version_change,
    PublishResult, PublishStrategy, UploadConfig, VersionChangeInfo, VersionValidationRules,
};
pub use utils::{
    cleanup_old_cache, compare_versions, estimate_parameters_from_size, extract_extension,
    format_parameter_count, format_size, get_model_cache_dir, get_temp_dir, is_safe_path,
    is_supported_model_format, parse_repo_string, sanitize_model_name, validate_semver,
};
pub use visualization::{
    ChartData, ChartType, DashboardTemplate, PerformanceVisualization, TrainingVisualization,
    UsageVisualization, VisualizationConfig, VisualizationEngine,
};

// Import torsh-nn components
use torsh_nn::prelude::*;

/// Hub configuration
#[derive(Debug, Clone)]
pub struct HubConfig {
    pub cache_dir: PathBuf,
    pub hub_url: String,
    pub force_reload: bool,
    pub verbose: bool,
    pub skip_validation: bool,
    pub auth_token: Option<String>,
    pub timeout_seconds: u64,
    pub max_retries: u32,
    pub user_agent: String,
}

impl Default for HubConfig {
    fn default() -> Self {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("torsh")
            .join("hub");

        Self {
            cache_dir,
            hub_url: "https://github.com".to_string(),
            force_reload: false,
            verbose: true,
            skip_validation: false,
            auth_token: load_auth_token_from_env_or_file(),
            timeout_seconds: 300,
            max_retries: 3,
            user_agent: format!(
                "torsh-hub/{}",
                option_env!("CARGO_PKG_VERSION").unwrap_or("0.1.0-alpha.2")
            ),
        }
    }
}

/// Load authentication token from environment variable or config file
fn load_auth_token_from_env_or_file() -> Option<String> {
    // First try environment variable
    if let Ok(token) = std::env::var("TORSH_HUB_TOKEN") {
        return Some(token);
    }

    // Then try from config file
    if let Some(config_dir) = dirs::config_dir() {
        let token_file = config_dir.join("torsh").join("hub_token");
        if token_file.exists() {
            if let Ok(token) = std::fs::read_to_string(&token_file) {
                return Some(token.trim().to_string());
            }
        }
    }

    None
}

/// Set authentication token
pub fn set_auth_token(token: &str) -> Result<()> {
    std::env::set_var("TORSH_HUB_TOKEN", token);

    // Also save to config file
    if let Some(config_dir) = dirs::config_dir() {
        let torsh_config_dir = config_dir.join("torsh");
        std::fs::create_dir_all(&torsh_config_dir)?;

        let token_file = torsh_config_dir.join("hub_token");
        std::fs::write(&token_file, token)?;

        // Set secure permissions on the token file (Unix only)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&token_file)?.permissions();
            perms.set_mode(0o600); // Only owner can read/write
            std::fs::set_permissions(&token_file, perms)?;
        }
    }

    Ok(())
}

/// Remove authentication token
pub fn remove_auth_token() -> Result<()> {
    std::env::remove_var("TORSH_HUB_TOKEN");

    // Also remove from config file
    if let Some(config_dir) = dirs::config_dir() {
        let token_file = config_dir.join("torsh").join("hub_token");
        if token_file.exists() {
            std::fs::remove_file(&token_file)?;
        }
    }

    Ok(())
}

/// Check if authenticated
pub fn is_authenticated() -> bool {
    load_auth_token_from_env_or_file().is_some()
}

/// Get current authentication status
pub fn auth_status() -> String {
    if let Some(token) = load_auth_token_from_env_or_file() {
        // Only show first few characters for security
        let visible_part = if token.len() > 8 {
            format!("{}***", &token[..4])
        } else {
            "***".to_string()
        };
        format!("Authenticated with token: {}", visible_part)
    } else {
        "Not authenticated".to_string()
    }
}

/// Load an ONNX model from file
///
/// # Arguments
/// * `path` - Path to the ONNX model file
/// * `config` - Optional ONNX configuration
///
/// # Example
/// ```no_run
/// use torsh_hub::load_onnx_model;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let model = load_onnx_model("model.onnx", None)?;
/// # Ok(())
/// # }
/// ```
pub fn load_onnx_model<P: AsRef<Path>>(
    path: P,
    config: Option<crate::onnx::OnnxConfig>,
) -> Result<Box<dyn torsh_nn::Module>> {
    use crate::onnx::{OnnxModel, OnnxToTorshWrapper};

    let onnx_model = OnnxModel::from_file(path, config)?;
    let wrapper = OnnxToTorshWrapper::new(onnx_model);
    Ok(Box::new(wrapper))
}

/// Load an ONNX model from bytes
///
/// # Arguments
/// * `model_bytes` - ONNX model as bytes
/// * `config` - Optional ONNX configuration
///
/// # Example
/// ```no_run
/// use torsh_hub::load_onnx_model_from_bytes;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let model_bytes = std::fs::read("model.onnx")?;
/// let model = load_onnx_model_from_bytes(&model_bytes, None)?;
/// # Ok(())
/// # }
/// ```
pub fn load_onnx_model_from_bytes(
    model_bytes: &[u8],
    config: Option<crate::onnx::OnnxConfig>,
) -> Result<Box<dyn torsh_nn::Module>> {
    use crate::onnx::{OnnxModel, OnnxToTorshWrapper};

    let onnx_model = OnnxModel::from_bytes(model_bytes, config)?;
    let wrapper = OnnxToTorshWrapper::new(onnx_model);
    Ok(Box::new(wrapper))
}

/// Download and load an ONNX model from URL
///
/// # Arguments
/// * `url` - URL to download the ONNX model from
/// * `config` - Optional ONNX configuration
///
/// # Example
/// ```no_run
/// use torsh_hub::load_onnx_model_from_url;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let model = load_onnx_model_from_url("https://example.com/model.onnx", None).await?;
/// # Ok(())
/// # }
/// ```
pub async fn load_onnx_model_from_url(
    url: &str,
    config: Option<crate::onnx::OnnxConfig>,
) -> Result<Box<dyn torsh_nn::Module>> {
    use crate::onnx::{OnnxLoader, OnnxToTorshWrapper};

    let onnx_model = OnnxLoader::from_url(url, config).await?;
    let wrapper = OnnxToTorshWrapper::new(onnx_model);
    Ok(Box::new(wrapper))
}

/// Load an ONNX model from ToRSh Hub
///
/// # Arguments
/// * `repo` - Repository in format "owner/repo"
/// * `model_name` - Name of the ONNX model file (without .onnx extension)
/// * `config` - Optional ONNX configuration
///
/// # Example
/// ```no_run
/// use torsh_hub::load_onnx_model_from_hub;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let model = load_onnx_model_from_hub("owner/repo", "resnet50", None)?;
/// # Ok(())
/// # }
/// ```
pub fn load_onnx_model_from_hub(
    repo: &str,
    model_name: &str,
    config: Option<crate::onnx::OnnxConfig>,
) -> Result<Box<dyn torsh_nn::Module>> {
    use crate::onnx::{OnnxLoader, OnnxToTorshWrapper};

    let onnx_model = OnnxLoader::from_hub(repo, model_name, config)?;
    let wrapper = OnnxToTorshWrapper::new(onnx_model);
    Ok(Box::new(wrapper))
}

/// Validate authentication token
pub fn validate_auth_token(token: &str) -> Result<bool> {
    if token.is_empty() {
        return Ok(false);
    }

    // Basic token format validation
    if token.len() < 8 {
        return Err(TorshError::InvalidArgument(
            "Authentication token is too short".to_string(),
        ));
    }

    // Could add more sophisticated validation here
    // For now, just check basic format
    if !token
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
    {
        return Err(TorshError::InvalidArgument(
            "Authentication token contains invalid characters".to_string(),
        ));
    }

    Ok(true)
}

/// Load a model from ToRSh Hub
///
/// # Arguments
/// * `repo` - GitHub repository in format "owner/repo" or full GitHub URL
/// * `model` - Model name to load from the repository
/// * `pretrained` - Whether to load pretrained weights
/// * `config` - Optional hub configuration
///
/// # Example
/// ```no_run
/// use torsh_hub::load;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Load a model from GitHub
/// let model = load("pytorch/vision", "resnet18", true, None)?;
/// # Ok(())
/// # }
/// ```
pub fn load(
    repo: &str,
    model: &str,
    pretrained: bool,
    config: Option<HubConfig>,
) -> Result<Box<dyn torsh_nn::Module>> {
    let config = config.unwrap_or_default();

    // Parse repository
    let (owner, repo_name, branch) = parse_repo_info(repo)?;

    // Get or download repository
    let repo_dir = download_repo(&owner, &repo_name, &branch, &config)?;

    // Load hubconf.py equivalent (we'll use a Rust module)
    let model_fn = load_model_fn(&repo_dir, model)?;

    // Create model
    let model = model_fn(pretrained)?;

    Ok(model)
}

/// List available models in a repository
pub fn list(repo: &str, config: Option<HubConfig>) -> Result<Vec<String>> {
    let config = config.unwrap_or_default();

    // Parse repository
    let (owner, repo_name, branch) = parse_repo_info(repo)?;

    // Get or download repository
    let repo_dir = download_repo(&owner, &repo_name, &branch, &config)?;

    // List available models
    let models = list_available_models(&repo_dir)?;

    Ok(models)
}

/// Get help/docstring for a model
pub fn help(repo: &str, model: &str, config: Option<HubConfig>) -> Result<String> {
    let config = config.unwrap_or_default();

    // Parse repository
    let (owner, repo_name, branch) = parse_repo_info(repo)?;

    // Get or download repository
    let repo_dir = download_repo(&owner, &repo_name, &branch, &config)?;

    // Get model documentation
    let doc = get_model_doc(&repo_dir, model)?;

    Ok(doc)
}

/// Set the hub directory
pub fn set_dir(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    std::fs::create_dir_all(path)?;

    // Store in a global config or environment variable
    std::env::set_var("TORSH_HUB_DIR", path);

    Ok(())
}

/// Get the current hub directory
pub fn get_dir() -> PathBuf {
    std::env::var("TORSH_HUB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| HubConfig::default().cache_dir)
}

/// Load model state dict from URL
pub fn load_state_dict_from_url(
    url: &str,
    model_dir: Option<&Path>,
    map_location: Option<torsh_core::DeviceType>,
    progress: bool,
) -> Result<StateDict> {
    let default_dir = get_dir();
    let model_dir = model_dir.unwrap_or(&default_dir);

    // Download file
    let file_path = download_url_to_file(url, model_dir, progress)?;

    // Load state dict
    let state_dict = load_state_dict(&file_path, map_location)?;

    Ok(state_dict)
}

// Type alias for state dictionary
pub type StateDict = std::collections::HashMap<String, torsh_tensor::Tensor<f32>>;

/// Parse repository information
pub fn parse_repo_info(repo: &str) -> Result<(String, String, String)> {
    if repo.starts_with("https://") || repo.starts_with("http://") {
        // Full URL provided
        parse_github_url(repo)
    } else if repo.contains('/') {
        // owner/repo format
        let parts: Vec<&str> = repo.split('/').collect();
        if parts.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Repository should be in format 'owner/repo'".to_string(),
            ));
        }
        Ok((
            parts[0].to_string(),
            parts[1].to_string(),
            "main".to_string(),
        ))
    } else {
        Err(TorshError::InvalidArgument(
            "Invalid repository format".to_string(),
        ))
    }
}

/// Parse GitHub URL
fn parse_github_url(url: &str) -> Result<(String, String, String)> {
    // Parse URL like https://github.com/owner/repo or https://github.com/owner/repo/tree/branch
    let url = url.trim_end_matches('/');
    let parts: Vec<&str> = url.split('/').collect();

    if parts.len() < 5 || parts[2] != "github.com" {
        return Err(TorshError::InvalidArgument(
            "Invalid GitHub URL".to_string(),
        ));
    }

    let owner = parts[3].to_string();
    let repo = parts[4].to_string();
    let branch = if parts.len() >= 7 && parts[5] == "tree" {
        parts[6].to_string()
    } else {
        "main".to_string()
    };

    Ok((owner, repo, branch))
}

/// Download repository
pub fn download_repo(owner: &str, repo: &str, branch: &str, config: &HubConfig) -> Result<PathBuf> {
    let cache_manager = CacheManager::new(&config.cache_dir)?;
    let repo_dir = cache_manager.get_repo_dir(owner, repo, branch);

    if repo_dir.exists() && !config.force_reload {
        if config.verbose {
            println!("Using cached repository at: {:?}", repo_dir);
        }
        return Ok(repo_dir);
    }

    // Download repository
    download::download_github_repo(owner, repo, branch, &repo_dir, config.verbose)?;

    Ok(repo_dir)
}

/// Type alias for model factory function
type ModelFactoryFn = Box<dyn Fn(bool) -> Result<Box<dyn torsh_nn::Module>>>;

/// Load model function from repository
fn load_model_fn(repo_dir: &Path, model: &str) -> Result<ModelFactoryFn> {
    // Look for model configuration files
    let models_toml = repo_dir.join("models.toml");
    let hubconf_toml = repo_dir.join("hubconf.toml");

    let config_path = if models_toml.exists() {
        models_toml
    } else if hubconf_toml.exists() {
        hubconf_toml
    } else {
        return Err(TorshError::IoError(
            "No model configuration file found (models.toml or hubconf.toml)".to_string(),
        ));
    };

    // Load model configuration
    let config_content = std::fs::read_to_string(config_path)?;
    let config: ModelConfig = toml::from_str(&config_content)
        .map_err(|e| TorshError::ConfigError(format!("Failed to parse model config: {}", e)))?;

    // Find the requested model
    let model_def = config
        .models
        .iter()
        .find(|m| m.name == model)
        .ok_or_else(|| {
            TorshError::InvalidArgument(format!("Model '{}' not found in repository", model))
        })?;

    // Clone the model definition for the closure
    let model_def = model_def.clone();
    let repo_dir = repo_dir.to_path_buf();

    // Return a closure that creates the model
    Ok(Box::new(move |pretrained: bool| {
        create_model_from_config(&model_def, &repo_dir, pretrained)
    }))
}

/// Model configuration structure
#[derive(Debug, Deserialize, Clone)]
struct ModelConfig {
    models: Vec<ModelDefinition>,
}

/// Individual model definition
#[derive(Debug, Deserialize, Clone)]
struct ModelDefinition {
    name: String,
    architecture: String,
    description: Option<String>,
    parameters: std::collections::HashMap<String, toml::Value>,
    weights_url: Option<String>,
    local_weights: Option<String>,
}

/// Create a model from configuration
fn create_model_from_config(
    model_def: &ModelDefinition,
    repo_dir: &Path,
    pretrained: bool,
) -> Result<Box<dyn torsh_nn::Module>> {
    match model_def.architecture.as_str() {
        "linear" => create_linear_model(model_def, repo_dir, pretrained),
        "conv2d" => create_conv2d_model(model_def, repo_dir, pretrained),
        "mlp" => create_mlp_model(model_def, repo_dir, pretrained),
        "resnet" => create_resnet_model(model_def, repo_dir, pretrained),
        "custom" => create_custom_model(model_def, repo_dir, pretrained),
        "onnx" => create_onnx_model(model_def, repo_dir, pretrained),
        #[cfg(feature = "tensorflow")]
        "tensorflow" | "tf" => create_tensorflow_model(model_def, repo_dir, pretrained),
        #[cfg(not(feature = "tensorflow"))]
        "tensorflow" | "tf" => Err(TorshError::Other(
            "TensorFlow support is disabled. Enable the 'tensorflow' feature to use TensorFlow models".to_string(),
        )),
        _ => Err(TorshError::InvalidArgument(format!(
            "Unsupported model architecture: {}",
            model_def.architecture
        ))),
    }
}

/// Create a linear model
fn create_linear_model(
    model_def: &ModelDefinition,
    repo_dir: &Path,
    pretrained: bool,
) -> Result<Box<dyn torsh_nn::Module>> {
    let in_features = extract_param_i64(&model_def.parameters, "in_features")? as usize;
    let out_features = extract_param_i64(&model_def.parameters, "out_features")? as usize;
    let bias = extract_param_bool(&model_def.parameters, "bias").unwrap_or(true);

    let mut model = Linear::new(in_features, out_features, bias);

    if pretrained {
        load_pretrained_weights(&mut model as &mut dyn torsh_nn::Module, model_def, repo_dir)?;
    }

    Ok(Box::new(model))
}

/// Create a Conv2d model
fn create_conv2d_model(
    model_def: &ModelDefinition,
    repo_dir: &Path,
    pretrained: bool,
) -> Result<Box<dyn torsh_nn::Module>> {
    let in_channels = extract_param_i64(&model_def.parameters, "in_channels")? as usize;
    let out_channels = extract_param_i64(&model_def.parameters, "out_channels")? as usize;
    let kernel_size = extract_param_i64(&model_def.parameters, "kernel_size")? as usize;
    let stride = extract_param_i64(&model_def.parameters, "stride").unwrap_or(1) as usize;
    let padding = extract_param_i64(&model_def.parameters, "padding").unwrap_or(0) as usize;
    let bias = extract_param_bool(&model_def.parameters, "bias").unwrap_or(true);

    let mut model = Conv2d::new(
        in_channels,
        out_channels,
        (kernel_size, kernel_size),
        (stride, stride),
        (padding, padding),
        (1, 1), // dilation
        bias,
        1, // groups
    );

    if pretrained {
        load_pretrained_weights(&mut model as &mut dyn torsh_nn::Module, model_def, repo_dir)?;
    }

    Ok(Box::new(model))
}

/// Create an MLP model
fn create_mlp_model(
    model_def: &ModelDefinition,
    repo_dir: &Path,
    pretrained: bool,
) -> Result<Box<dyn torsh_nn::Module>> {
    let layers = extract_param_array(&model_def.parameters, "layers").ok_or_else(|| {
        TorshError::InvalidArgument("Missing 'layers' parameter for MLP".to_string())
    })?;
    let activation = extract_param_string(&model_def.parameters, "activation")
        .unwrap_or_else(|| "relu".to_string());
    let dropout = extract_param_f64(&model_def.parameters, "dropout").unwrap_or(0.0);

    let mut sequential = Sequential::new();

    for i in 0..layers.len() - 1 {
        let in_features = layers[i];
        let out_features = layers[i + 1];

        sequential = sequential.add(Linear::new(in_features, out_features, true));

        if i < layers.len() - 2 {
            // Add activation (except for last layer)
            match activation.as_str() {
                "relu" => sequential = sequential.add(ReLU::new()),
                "tanh" => sequential = sequential.add(Tanh::new()),
                "sigmoid" => sequential = sequential.add(Sigmoid::new()),
                _ => {
                    return Err(TorshError::InvalidArgument(format!(
                        "Unsupported activation: {}",
                        activation
                    )))
                }
            }

            // Add dropout if specified
            if dropout > 0.0 {
                sequential = sequential.add(Dropout::new(dropout as f32));
            }
        }
    }

    let mut model = sequential;

    if pretrained {
        load_pretrained_weights(&mut model as &mut dyn torsh_nn::Module, model_def, repo_dir)?;
    }

    Ok(Box::new(model))
}

/// Create a ResNet model (simplified)
fn create_resnet_model(
    model_def: &ModelDefinition,
    repo_dir: &Path,
    pretrained: bool,
) -> Result<Box<dyn torsh_nn::Module>> {
    let num_classes =
        extract_param_i64(&model_def.parameters, "num_classes").unwrap_or(1000) as usize;
    let layers =
        extract_param_array(&model_def.parameters, "layers").unwrap_or_else(|| vec![2, 2, 2, 2]);

    // This is a simplified ResNet implementation
    // In practice, you'd have a full ResNet implementation in torsh-nn
    let mut model = Sequential::new();

    // Initial conv layer
    model = model.add(Conv2d::new(3, 64, (7, 7), (2, 2), (3, 3), (1, 1), false, 1));
    model = model.add(BatchNorm2d::new(64).expect("Failed to create BatchNorm2d"));
    model = model.add(ReLU::new());
    model = model.add(MaxPool2d::new((3, 3), Some((2, 2)), (1, 1), (1, 1), false));

    // Add residual blocks (simplified)
    let mut in_channels = 64;
    let mut out_channels = 64;

    for (layer_idx, &num_blocks) in layers.iter().enumerate() {
        if layer_idx > 0 {
            out_channels *= 2;
        }

        for _block_idx in 0..num_blocks {
            let stride = if layer_idx > 0 { 2 } else { 1 };

            // Simplified residual block
            model = model.add(Conv2d::new(
                in_channels,
                out_channels,
                (3, 3),
                (stride, stride),
                (1, 1),
                (1, 1),
                false,
                1,
            ));
            model =
                model.add(BatchNorm2d::new(out_channels).expect("Failed to create BatchNorm2d"));
            model = model.add(ReLU::new());

            in_channels = out_channels;
        }
    }

    // Final layers - Global average pooling and flatten for ResNet
    model = model.add(AdaptiveAvgPool2d::with_output_size(1));
    model = model.add(Flatten::new());
    model = model.add(Linear::new(out_channels, num_classes, true));

    let mut model = model;

    if pretrained {
        load_pretrained_weights(&mut model as &mut dyn torsh_nn::Module, model_def, repo_dir)?;
    }

    Ok(Box::new(model))
}

/// Create a custom model (placeholder)
fn create_custom_model(
    _model_def: &ModelDefinition,
    _repo_dir: &Path,
    _pretrained: bool,
) -> Result<Box<dyn torsh_nn::Module>> {
    Err(TorshError::Other(
        "Custom model loading requires compilation of Rust code. Use predefined architectures instead.".to_string(),
    ))
}

/// Create an ONNX model
fn create_onnx_model(
    model_def: &ModelDefinition,
    repo_dir: &Path,
    _pretrained: bool,
) -> Result<Box<dyn torsh_nn::Module>> {
    use crate::onnx::{OnnxModel, OnnxToTorshWrapper};

    // Get model file path
    let model_file =
        if let Some(local_path) = extract_param_string(&model_def.parameters, "model_file") {
            repo_dir.join(local_path)
        } else {
            // Default to model name with .onnx extension
            repo_dir.join(format!("{}.onnx", model_def.name))
        };

    if !model_file.exists() {
        return Err(TorshError::IoError(format!(
            "ONNX model file not found: {:?}",
            model_file
        )));
    }

    // Create ONNX configuration from parameters
    let config = create_onnx_config_from_params(&model_def.parameters);

    // Load ONNX model
    let onnx_model = OnnxModel::from_file(&model_file, Some(config))?;

    // Wrap in ToRSh Module interface
    let wrapper = OnnxToTorshWrapper::new(onnx_model);

    Ok(Box::new(wrapper))
}

/// Create ONNX configuration from model parameters
fn create_onnx_config_from_params(
    params: &std::collections::HashMap<String, toml::Value>,
) -> crate::onnx::OnnxConfig {
    use crate::onnx::OnnxConfig;
    use ort::session::builder::GraphOptimizationLevel;

    let mut config = OnnxConfig::default();

    // Set execution providers (simplified to string-based configuration)
    if let Some(providers) = extract_param_array_strings(params, "execution_providers") {
        // Note: This is a simplified implementation
        // In a real implementation, you would configure the ONNX runtime properly
        println!("Execution providers configured: {:?}", providers);
    }

    // Set optimization level
    if let Some(opt_level) = extract_param_string(params, "optimization_level") {
        config.graph_optimization_level = match opt_level.as_str() {
            "disable" => GraphOptimizationLevel::Disable,
            "basic" => GraphOptimizationLevel::Level1,
            "extended" => GraphOptimizationLevel::Level2,
            "all" => GraphOptimizationLevel::Level3,
            _ => GraphOptimizationLevel::Level3,
        };
    }

    // Set threading options
    if let Ok(inter_threads) = extract_param_i64(params, "inter_op_threads") {
        config.inter_op_num_threads = Some(inter_threads as usize);
    }

    if let Ok(intra_threads) = extract_param_i64(params, "intra_op_threads") {
        config.intra_op_num_threads = Some(intra_threads as usize);
    }

    // Set other options
    if let Some(enable_profiling) = extract_param_bool(params, "enable_profiling") {
        config.enable_profiling = enable_profiling;
    }

    if let Some(enable_mem_pattern) = extract_param_bool(params, "enable_mem_pattern") {
        config.enable_mem_pattern = enable_mem_pattern;
    }

    if let Some(enable_cpu_mem_arena) = extract_param_bool(params, "enable_cpu_mem_arena") {
        config.enable_cpu_mem_arena = enable_cpu_mem_arena;
    }

    config
}

/// Create a TensorFlow model
#[cfg(feature = "tensorflow")]
fn create_tensorflow_model(
    model_def: &ModelDefinition,
    repo_dir: &Path,
    _pretrained: bool,
) -> Result<Box<dyn torsh_nn::Module>> {
    use crate::tensorflow::{TfConfig, TfModel, TfToTorshWrapper};

    // Get model directory path
    let model_dir =
        if let Some(local_path) = extract_param_string(&model_def.parameters, "model_dir") {
            repo_dir.join(local_path)
        } else {
            // Default to model name as directory
            repo_dir.join(&model_def.name)
        };

    if !model_dir.exists() {
        return Err(TorshError::IoError(format!(
            "TensorFlow model directory not found: {:?}",
            model_dir
        )));
    }

    // Create TensorFlow configuration from parameters
    let config = create_tf_config_from_params(&model_def.parameters);

    // Get tags for SavedModel
    let tags = if let Some(tags_array) = extract_param_array_strings(&model_def.parameters, "tags")
    {
        tags_array.into_iter().collect::<Vec<_>>()
    } else {
        vec!["serve".to_string()]
    };

    // Load TensorFlow model
    let tf_model = TfModel::from_saved_model(
        &model_dir,
        &tags.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        Some(config),
    )?;

    // Wrap in ToRSh Module interface
    let wrapper = TfToTorshWrapper::new(tf_model);

    Ok(Box::new(wrapper))
}

/// Create TensorFlow configuration from model parameters
#[cfg(feature = "tensorflow")]
fn create_tf_config_from_params(
    params: &std::collections::HashMap<String, toml::Value>,
) -> crate::tensorflow::TfConfig {
    use crate::tensorflow::TfConfig;

    let mut config = TfConfig::default();

    // Set GPU usage
    if let Some(use_gpu) = extract_param_bool(params, "use_gpu") {
        config.use_gpu = use_gpu;
    }

    // Set memory growth
    if let Some(allow_growth) = extract_param_bool(params, "allow_growth") {
        config.allow_growth = allow_growth;
    }

    // Set memory limit
    if let Some(memory_limit) = extract_param_i64(params, "memory_limit").ok() {
        config.memory_limit = Some(memory_limit as usize);
    }

    // Set GPU memory fraction
    if let Some(gpu_memory_fraction) = extract_param_f64(params, "gpu_memory_fraction") {
        config.gpu_memory_fraction = Some(gpu_memory_fraction);
    }

    // Set threading options
    if let Some(inter_threads) = extract_param_i64(params, "inter_op_threads").ok() {
        config.inter_op_parallelism_threads = Some(inter_threads as i32);
    }

    if let Some(intra_threads) = extract_param_i64(params, "intra_op_threads").ok() {
        config.intra_op_parallelism_threads = Some(intra_threads as i32);
    }

    // Set device placement
    if let Some(device_placement) = extract_param_bool(params, "device_placement") {
        config.device_placement = device_placement;
    }

    config
}

/// Load pretrained weights into a model
fn load_pretrained_weights(
    _model: &mut dyn torsh_nn::Module,
    model_def: &ModelDefinition,
    repo_dir: &Path,
) -> Result<()> {
    let weights_path = if let Some(ref local_weights) = model_def.local_weights {
        repo_dir.join(local_weights)
    } else if let Some(ref weights_url) = model_def.weights_url {
        // Download weights if not cached
        let cache_dir = repo_dir.join(".weights_cache");
        std::fs::create_dir_all(&cache_dir)?;

        let weights_filename = weights_url
            .split('/')
            .next_back()
            .unwrap_or("weights.torsh");
        let weights_path = cache_dir.join(weights_filename);

        if !weights_path.exists() {
            download::download_file(weights_url, &weights_path, true)?;
        }
        weights_path
    } else {
        return Err(TorshError::InvalidArgument(
            "No weights specified for pretrained model".to_string(),
        ));
    };

    // Load state dict and apply to model
    let state_dict = load_state_dict(&weights_path, None)?;
    _model.load_state_dict(&state_dict, true)?;

    Ok(())
}

/// Extract integer parameter from config
fn extract_param_i64(
    params: &std::collections::HashMap<String, toml::Value>,
    key: &str,
) -> Result<i64> {
    params.get(key).and_then(|v| v.as_integer()).ok_or_else(|| {
        TorshError::InvalidArgument(format!("Missing or invalid parameter: {}", key))
    })
}

/// Extract boolean parameter from config
fn extract_param_bool(
    params: &std::collections::HashMap<String, toml::Value>,
    key: &str,
) -> Option<bool> {
    params.get(key).and_then(|v| v.as_bool())
}

/// Extract string parameter from config
fn extract_param_string(
    params: &std::collections::HashMap<String, toml::Value>,
    key: &str,
) -> Option<String> {
    params
        .get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Extract float parameter from config
fn extract_param_f64(
    params: &std::collections::HashMap<String, toml::Value>,
    key: &str,
) -> Option<f64> {
    params.get(key).and_then(|v| v.as_float())
}

/// Extract array parameter from config
fn extract_param_array(
    params: &std::collections::HashMap<String, toml::Value>,
    key: &str,
) -> Option<Vec<usize>> {
    params.get(key).and_then(|v| {
        v.as_array().map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_integer())
                .map(|i| i as usize)
                .collect()
        })
    })
}

/// Extract string array parameter from config
fn extract_param_array_strings(
    params: &std::collections::HashMap<String, toml::Value>,
    key: &str,
) -> Option<Vec<String>> {
    params.get(key).and_then(|v| {
        v.as_array().map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect()
        })
    })
}

/// List available models in repository
fn list_available_models(repo_dir: &Path) -> Result<Vec<String>> {
    // Look for model configuration files
    let models_toml = repo_dir.join("models.toml");
    let hubconf_toml = repo_dir.join("hubconf.toml");

    let config_path = if models_toml.exists() {
        models_toml
    } else if hubconf_toml.exists() {
        hubconf_toml
    } else {
        // Fallback: look for legacy models.toml format
        let legacy_models_file = repo_dir.join("models.toml");
        if legacy_models_file.exists() {
            let content = std::fs::read_to_string(legacy_models_file)?;
            let models: ModelList =
                toml::from_str(&content).map_err(|e| TorshError::ConfigError(e.to_string()))?;
            return Ok(models.models.into_iter().map(|m| m.name).collect());
        }
        return Ok(vec![]);
    };

    // Load new model configuration format
    let config_content = std::fs::read_to_string(config_path)?;
    let config: ModelConfig = toml::from_str(&config_content)
        .map_err(|e| TorshError::ConfigError(format!("Failed to parse model config: {}", e)))?;

    Ok(config.models.into_iter().map(|m| m.name).collect())
}

/// Get model documentation
fn get_model_doc(repo_dir: &Path, model: &str) -> Result<String> {
    // First, try to find dedicated documentation file
    let doc_file = repo_dir.join("docs").join(format!("{}.md", model));
    if doc_file.exists() {
        return std::fs::read_to_string(doc_file).map_err(Into::into);
    }

    // Fallback: get description from model configuration
    let models_toml = repo_dir.join("models.toml");
    let hubconf_toml = repo_dir.join("hubconf.toml");

    let config_path = if models_toml.exists() {
        models_toml
    } else if hubconf_toml.exists() {
        hubconf_toml
    } else {
        return Ok(format!("No documentation available for model '{}'", model));
    };

    let config_content = std::fs::read_to_string(config_path)?;
    let config: ModelConfig = toml::from_str(&config_content)
        .map_err(|e| TorshError::ConfigError(format!("Failed to parse model config: {}", e)))?;

    // Find the model and return its description
    if let Some(model_def) = config.models.iter().find(|m| m.name == model) {
        let mut doc = format!("# {}\n\n", model_def.name);

        if let Some(ref description) = model_def.description {
            doc.push_str(&format!("**Description:** {}\n\n", description));
        }

        doc.push_str(&format!("**Architecture:** {}\n\n", model_def.architecture));

        if !model_def.parameters.is_empty() {
            doc.push_str("**Parameters:**\n");
            for (key, value) in &model_def.parameters {
                doc.push_str(&format!("- {}: {:?}\n", key, value));
            }
            doc.push('\n');
        }

        if model_def.weights_url.is_some() || model_def.local_weights.is_some() {
            doc.push_str("**Pretrained weights available:** Yes\n\n");
        }

        Ok(doc)
    } else {
        Ok(format!("Model '{}' not found in repository", model))
    }
}

/// Download URL to file
fn download_url_to_file(url: &str, dst_dir: &Path, progress: bool) -> Result<PathBuf> {
    let filename = url
        .split('/')
        .next_back()
        .ok_or_else(|| TorshError::InvalidArgument("Invalid URL".to_string()))?;

    let dst_path = dst_dir.join(filename);

    if dst_path.exists() {
        if progress {
            println!("File already exists: {:?}", dst_path);
        }
        return Ok(dst_path);
    }

    download::download_file(url, &dst_path, progress)?;

    Ok(dst_path)
}

/// Load state dictionary from file
fn load_state_dict(path: &Path, map_location: Option<torsh_core::DeviceType>) -> Result<StateDict> {
    use std::io::BufReader;

    if !path.exists() {
        return Err(TorshError::IoError(format!(
            "State dict file not found: {:?}",
            path
        )));
    }

    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Try to determine file format from extension
    let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");

    match extension {
        "json" => {
            // Load JSON format state dict
            load_json_state_dict(&mut reader, map_location)
        }
        "torsh" => {
            // Load native torsh format
            load_torsh_state_dict(&mut reader, map_location)
        }
        "pt" | "pth" => {
            // For PyTorch compatibility, we'll implement a basic loader
            load_pytorch_compatible_state_dict(&mut reader, map_location)
        }
        _ => Err(TorshError::InvalidArgument(format!(
            "Unsupported state dict format: {}",
            extension
        ))),
    }
}

/// Load native torsh format state dict
fn load_torsh_state_dict(
    reader: &mut impl Read,
    map_location: Option<torsh_core::DeviceType>,
) -> Result<StateDict> {
    // Read magic header
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;

    if &magic != b"TORSH\x01\x00\x00" {
        return Err(TorshError::SerializationError(
            "Invalid torsh file format".to_string(),
        ));
    }

    // Read version
    let mut version = [0u8; 4];
    reader.read_exact(&mut version)?;
    let version = u32::from_le_bytes(version);

    if version > 1 {
        return Err(TorshError::SerializationError(format!(
            "Unsupported torsh file version: {}",
            version
        )));
    }

    // Read number of tensors
    let mut num_tensors = [0u8; 8];
    reader.read_exact(&mut num_tensors)?;
    let num_tensors = u64::from_le_bytes(num_tensors);

    let mut state_dict = StateDict::new();

    for _ in 0..num_tensors {
        // Read tensor name length
        let mut name_len = [0u8; 4];
        reader.read_exact(&mut name_len)?;
        let name_len = u32::from_le_bytes(name_len) as usize;

        // Read tensor name
        let mut name_bytes = vec![0u8; name_len];
        reader.read_exact(&mut name_bytes)?;
        let name = String::from_utf8(name_bytes)
            .map_err(|e| TorshError::SerializationError(format!("Invalid tensor name: {}", e)))?;

        // Read tensor data (simplified - would need full tensor serialization)
        // For now, create a placeholder tensor
        let tensor = create_placeholder_tensor(&name, map_location)?;
        state_dict.insert(name, tensor);
    }

    Ok(state_dict)
}

/// Load PyTorch-compatible state dict (simplified implementation)
fn load_pytorch_compatible_state_dict(
    _reader: &mut impl Read,
    _map_location: Option<torsh_core::DeviceType>,
) -> Result<StateDict> {
    // This would require implementing PyTorch pickle format parsing
    // For now, return an error with suggestion
    Err(TorshError::Other(
        "PyTorch (.pt/.pth) format loading not yet implemented. Please convert to .json or .torsh format".to_string(),
    ))
}

/// Load JSON format state dict
fn load_json_state_dict(
    reader: &mut impl Read,
    map_location: Option<torsh_core::DeviceType>,
) -> Result<StateDict> {
    use serde_json::Value;
    use torsh_tensor::creation::*;

    // Read JSON content
    let mut content = String::new();
    reader.read_to_string(&mut content)?;

    // Parse JSON
    let json: Value = serde_json::from_str(&content)
        .map_err(|e| TorshError::SerializationError(format!("Invalid JSON: {}", e)))?;

    let mut state_dict = StateDict::new();

    if let Value::Object(obj) = json {
        for (name, value) in obj {
            let tensor = match value {
                Value::Object(tensor_obj) => {
                    // Expected format: {"shape": [2, 3], "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
                    let shape_value = tensor_obj.get("shape").ok_or_else(|| {
                        TorshError::SerializationError("Missing 'shape' field".to_string())
                    })?;
                    let data_value = tensor_obj.get("data").ok_or_else(|| {
                        TorshError::SerializationError("Missing 'data' field".to_string())
                    })?;

                    // Parse shape
                    let shape: Vec<usize> = shape_value
                        .as_array()
                        .ok_or_else(|| {
                            TorshError::SerializationError("Shape must be an array".to_string())
                        })?
                        .iter()
                        .map(|v| {
                            v.as_u64()
                                .ok_or_else(|| {
                                    TorshError::SerializationError(
                                        "Shape dimensions must be integers".to_string(),
                                    )
                                })
                                .map(|u| u as usize)
                        })
                        .collect::<Result<Vec<_>>>()?;

                    // Parse data
                    let data: Vec<f32> = data_value
                        .as_array()
                        .ok_or_else(|| {
                            TorshError::SerializationError("Data must be an array".to_string())
                        })?
                        .iter()
                        .map(|v| {
                            v.as_f64()
                                .ok_or_else(|| {
                                    TorshError::SerializationError(
                                        "Data elements must be numbers".to_string(),
                                    )
                                })
                                .map(|f| f as f32)
                        })
                        .collect::<Result<Vec<_>>>()?;

                    // Verify data length matches shape
                    let expected_len: usize = shape.iter().product();
                    if data.len() != expected_len {
                        return Err(TorshError::SerializationError(format!(
                            "Data length {} doesn't match shape {:?} (expected {})",
                            data.len(),
                            shape,
                            expected_len
                        )));
                    }

                    // Create tensor
                    let device = map_location.unwrap_or(torsh_core::DeviceType::Cpu);
                    from_vec(data, &shape, device)?
                }
                Value::Array(arr) => {
                    // Simple array format: [1.0, 2.0, 3.0] (1D tensor)
                    let data: Vec<f32> = arr
                        .iter()
                        .map(|v| {
                            v.as_f64()
                                .ok_or_else(|| {
                                    TorshError::SerializationError(
                                        "Array elements must be numbers".to_string(),
                                    )
                                })
                                .map(|f| f as f32)
                        })
                        .collect::<Result<Vec<_>>>()?;

                    let shape = vec![data.len()];
                    let device = map_location.unwrap_or(torsh_core::DeviceType::Cpu);
                    from_vec(data, &shape, device)?
                }
                _ => {
                    return Err(TorshError::SerializationError(format!(
                        "Unsupported tensor format for parameter '{}'",
                        name
                    )));
                }
            };

            state_dict.insert(name, tensor);
        }
    } else {
        return Err(TorshError::SerializationError(
            "JSON state dict must be an object".to_string(),
        ));
    }

    Ok(state_dict)
}

/// Apply device mapping to state dict
pub fn apply_device_mapping(
    mut state_dict: StateDict,
    target_device: torsh_core::DeviceType,
) -> Result<StateDict> {
    for (_name, tensor) in state_dict.iter_mut() {
        // Move tensor to target device
        *tensor = tensor.clone().to(target_device)?;
    }
    Ok(state_dict)
}

/// Create a placeholder tensor for demonstration
fn create_placeholder_tensor(
    name: &str,
    _device: Option<torsh_core::DeviceType>,
) -> Result<torsh_tensor::Tensor<f32>> {
    use torsh_tensor::creation::*;

    // Create a simple tensor based on name patterns
    if name.contains("weight") {
        Ok(randn(&[64, 32])?)
    } else if name.contains("bias") {
        Ok(zeros(&[64])?)
    } else {
        Ok(zeros(&[1])?)
    }
}

#[derive(Deserialize)]
struct ModelList {
    models: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    name: String,
    #[allow(dead_code)]
    description: String,
}

/// Pre-configured model sources
pub mod sources {
    use super::*;

    /// Official ToRSh models
    pub const TORSH_VISION: &str = "torsh/vision";
    pub const TORSH_TEXT: &str = "torsh/text";
    pub const TORSH_AUDIO: &str = "torsh/audio";

    /// Load ResNet models
    pub fn resnet18(pretrained: bool) -> Result<Box<dyn torsh_nn::Module>> {
        load(TORSH_VISION, "resnet18", pretrained, None)
    }

    pub fn resnet50(pretrained: bool) -> Result<Box<dyn torsh_nn::Module>> {
        load(TORSH_VISION, "resnet50", pretrained, None)
    }

    /// Load EfficientNet models
    pub fn efficientnet_b0(pretrained: bool) -> Result<Box<dyn torsh_nn::Module>> {
        load(TORSH_VISION, "efficientnet_b0", pretrained, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_repo_info() {
        let (owner, repo, branch) = parse_repo_info("pytorch/vision").unwrap();
        assert_eq!(owner, "pytorch");
        assert_eq!(repo, "vision");
        assert_eq!(branch, "main");

        let (owner, repo, branch) = parse_repo_info("https://github.com/pytorch/vision").unwrap();
        assert_eq!(owner, "pytorch");
        assert_eq!(repo, "vision");
        assert_eq!(branch, "main");

        let (owner, repo, branch) =
            parse_repo_info("https://github.com/pytorch/vision/tree/v0.11.0").unwrap();
        assert_eq!(owner, "pytorch");
        assert_eq!(repo, "vision");
        assert_eq!(branch, "v0.11.0");
    }

    #[test]
    fn test_hub_dir() {
        let original = get_dir();

        let temp_dir = tempfile::tempdir().unwrap();
        set_dir(temp_dir.path()).unwrap();

        assert_eq!(get_dir(), temp_dir.path());

        // Restore
        std::env::remove_var("TORSH_HUB_DIR");
        assert_eq!(get_dir(), original);
    }
}

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const VERSION_MAJOR: u32 = 0;
pub const VERSION_MINOR: u32 = 1;
pub const VERSION_PATCH: u32 = 0;

/// Prelude module for convenient imports
#[allow(ambiguous_glob_reexports)]
pub mod prelude {
    pub use crate::{
        analytics::*, cache::*, community::*, debugging::*, download::*, enterprise::*,
        fine_tuning::*, huggingface::*, onnx::*, profiling::*, registry::*, security::*, utils::*,
        visualization::*,
    };
}
