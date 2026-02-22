# ToRSh Hub API Reference

## Core Functions

### Model Loading

#### `hub::load(repo_name: &str, version: &str) -> Result<Box<dyn Module>>`

Load a model from the ToRSh Hub.

**Parameters:**
- `repo_name`: Model repository name in format "username/model-name"
- `version`: Model version (e.g., "v1.0.0", "latest")

**Returns:**
- `Result<Box<dyn Module>>`: The loaded model implementing the Module trait

**Example:**
```rust
let model = hub::load("torsh/resnet50", "latest")?;
```

#### `hub::load_with_config(repo_name: &str, config: ModelConfig) -> Result<Box<dyn Module>>`

Load a model with custom configuration.

**Parameters:**
- `repo_name`: Model repository name
- `config`: Model configuration settings

**Example:**
```rust
let config = ModelConfig::default()
    .with_device(Device::Cuda(0))
    .with_precision(Precision::FP16);
let model = hub::load_with_config("torsh/bert-base", config)?;
```

### Model Information

#### `hub::info(repo_name: &str) -> Result<ModelInfo>`

Get model information and metadata.

#### `hub::model_card(repo_name: &str) -> Result<ModelCard>`

Get the model card with detailed documentation.

#### `hub::list_models() -> Result<Vec<ModelInfo>>`

List all available models.

## Core Types

### ModelConfig

Configuration for model loading.

```rust
pub struct ModelConfig {
    pub device: Option<Device>,
    pub precision: Option<Precision>,
    pub cache_enabled: bool,
    pub verify_signature: bool,
    pub download_config: Option<DownloadConfig>,
}
```

**Methods:**
- `new() -> Self`
- `with_device(device: Device) -> Self`
- `with_precision(precision: Precision) -> Self`
- `with_cache(enabled: bool) -> Self`
- `with_signature_verification(enabled: bool) -> Self`

### ModelInfo

Model metadata and information.

```rust
pub struct ModelInfo {
    pub name: String,
    pub description: String,
    pub version: String,
    pub architecture: String,
    pub framework: String,
    pub size_bytes: u64,
    pub accuracy: Option<f64>,
    pub created_at: u64,
    pub updated_at: u64,
    pub downloads: u64,
    pub license: String,
    pub tags: Vec<String>,
}
```

### ModelCard

Comprehensive model documentation.

```rust
pub struct ModelCard {
    pub model_info: ModelInfo,
    pub description: String,
    pub intended_use: String,
    pub limitations: String,
    pub training_data: Option<String>,
    pub evaluation_data: Option<String>,
    pub metrics: Vec<PerformanceMetric>,
    pub ethical_considerations: Option<String>,
    pub references: Vec<String>,
}
```

**Methods:**
- `render_markdown() -> Result<String>`
- `render_html() -> Result<String>`
- `export_json() -> Result<String>`

## Registry API

### ModelRegistry

Central registry for model discovery and management.

```rust
pub struct ModelRegistry {
    // Implementation details
}
```

**Methods:**

#### `new() -> Self`

Create a new model registry instance.

#### `search_models(query: SearchQuery) -> Result<Vec<RegistryEntry>>`

Search for models based on criteria.

**Parameters:**
- `query`: Search query with filters

**Example:**
```rust
let query = SearchQuery::new()
    .category(ModelCategory::Vision)
    .accuracy_min(0.8)
    .max_model_size(100_000_000);
let results = registry.search_models(query)?;
```

#### `get_model(repo_name: &str) -> Result<RegistryEntry>`

Get detailed information about a specific model.

#### `register_model(entry: RegistryEntry) -> Result<()>`

Register a new model in the registry.

### SearchQuery

Query builder for model search.

```rust
pub struct SearchQuery {
    pub category: Option<ModelCategory>,
    pub tags: Vec<String>,
    pub accuracy_min: Option<f64>,
    pub accuracy_max: Option<f64>,
    pub model_size_max: Option<u64>,
    pub hardware_filter: Option<HardwareFilter>,
    pub license_filter: Vec<String>,
    pub sort_by: SortBy,
    pub limit: Option<usize>,
}
```

**Methods:**
- `new() -> Self`
- `category(category: ModelCategory) -> Self`
- `tag(tag: impl Into<String>) -> Self`
- `accuracy_range(min: f64, max: f64) -> Self`
- `max_model_size(size: u64) -> Self`
- `hardware_requirements(filter: HardwareFilter) -> Self`
- `license(license: impl Into<String>) -> Self`
- `sort_by(sort: SortBy) -> Self`
- `limit(limit: usize) -> Self`

### ModelCategory

Enumeration of model categories.

```rust
pub enum ModelCategory {
    Vision,
    NLP,
    Audio,
    Multimodal,
    Reinforcement,
    Generative,
    Classification,
    Detection,
    Segmentation,
    Translation,
    Other(String),
}
```

## Community API

### CommunityManager

Manager for community features.

```rust
pub struct CommunityManager {
    // Implementation details
}
```

**Methods:**

#### `new() -> Self`

Create a new community manager.

#### `add_rating(rating: ModelRating) -> Result<()>`

Add a rating for a model.

#### `get_rating_stats(model_id: &str) -> Option<&ModelRatingStats>`

Get aggregated rating statistics.

#### `get_model_ratings(model_id: &str) -> Vec<&ModelRating>`

Get all ratings for a model.

#### `add_comment(comment: Comment) -> Result<String>`

Add a comment and return comment ID.

#### `create_discussion(discussion: Discussion) -> Result<DiscussionId>`

Create a new discussion thread.

#### `search_discussions(query: &str, tags: Option<&[String]>) -> Vec<&Discussion>`

Search discussions by query and tags.

### ModelRating

User rating for a model.

```rust
pub struct ModelRating {
    pub user_id: UserId,
    pub model_id: ModelId,
    pub rating: u8, // 1-5 stars
    pub review: Option<String>,
    pub timestamp: u64,
    pub helpful_votes: u32,
    pub categories: Vec<RatingCategory>,
}
```

### RatingCategory

Categories for detailed ratings.

```rust
pub enum RatingCategory {
    Accuracy,
    Performance,
    EaseOfUse,
    Documentation,
    Reliability,
    Novelty,
}
```

### Comment

Comment on a model or discussion.

```rust
pub struct Comment {
    pub id: String,
    pub author_id: UserId,
    pub content: String,
    pub timestamp: u64,
    pub parent_id: Option<String>,
    pub model_id: Option<ModelId>,
    pub discussion_id: Option<DiscussionId>,
    pub upvotes: u32,
    pub downvotes: u32,
    pub is_edited: bool,
    pub is_pinned: bool,
    pub is_moderator_comment: bool,
}
```

### Discussion

Discussion thread.

```rust
pub struct Discussion {
    pub id: DiscussionId,
    pub title: String,
    pub description: String,
    pub author_id: UserId,
    pub created_at: u64,
    pub updated_at: u64,
    pub category: DiscussionCategory,
    pub tags: Vec<String>,
    pub status: DiscussionStatus,
    pub views: u32,
    pub participants: Vec<UserId>,
    pub is_pinned: bool,
    pub is_locked: bool,
    pub related_models: Vec<ModelId>,
}
```

## Enterprise API

### EnterpriseManager

Manager for enterprise features.

```rust
pub struct EnterpriseManager {
    // Implementation details
}
```

**Methods:**

#### `new() -> Self`

Create a new enterprise manager with default permissions and roles.

#### `create_private_repository(repo: PrivateRepository) -> Result<RepositoryId>`

Create a private repository.

#### `check_repository_access(repo_id: &str, user_id: &str) -> Result<bool>`

Check if user has access to a repository.

#### `create_role(role: Role) -> Result<RoleId>`

Create a new role.

#### `assign_role(assignment: UserRoleAssignment) -> Result<()>`

Assign a role to a user.

#### `check_permission(user_id: &str, permission_id: &str, resource_id: Option<&str>) -> bool`

Check if user has a specific permission.

#### `generate_compliance_report(org_id: &str, framework: ComplianceLabel, period_start: u64, period_end: u64) -> Result<String>`

Generate a compliance report.

### PrivateRepository

Private repository configuration.

```rust
pub struct PrivateRepository {
    pub id: RepositoryId,
    pub name: String,
    pub description: Option<String>,
    pub organization_id: OrganizationId,
    pub owner_id: UserId,
    pub visibility: RepositoryVisibility,
    pub access_control: RepositoryAccessControl,
    pub storage_config: StorageConfig,
    pub backup_config: BackupConfig,
    pub compliance_labels: Vec<ComplianceLabel>,
    pub data_classification: DataClassification,
}
```

### Role

RBAC role definition.

```rust
pub struct Role {
    pub id: RoleId,
    pub name: String,
    pub description: String,
    pub organization_id: OrganizationId,
    pub permissions: HashSet<PermissionId>,
    pub created_at: u64,
    pub updated_at: u64,
    pub is_system_role: bool,
    pub inheritance: Vec<RoleId>,
}
```

### Permission

Permission definition.

```rust
pub struct Permission {
    pub id: PermissionId,
    pub name: String,
    pub description: String,
    pub resource_type: ResourceType,
    pub action: Action,
    pub scope: PermissionScope,
}
```

## Analytics API

### AnalyticsManager

Manager for analytics and performance tracking.

```rust
pub struct AnalyticsManager {
    // Implementation details
}
```

**Methods:**

#### `new() -> Self`

Create a new analytics manager.

#### `track_model_download(model_id: &str, user_id: &str) -> Result<()>`

Track a model download event.

#### `track_model_inference(model_id: &str, user_id: &str, latency_ms: f64) -> Result<()>`

Track model inference performance.

#### `get_model_usage_stats(model_id: &str) -> Result<ModelUsageStats>`

Get usage statistics for a model.

#### `generate_analytics_report(period: TimePeriod) -> Result<AnalyticsReport>`

Generate comprehensive analytics report.

### ModelUsageStats

Usage statistics for a model.

```rust
pub struct ModelUsageStats {
    pub model_id: String,
    pub total_downloads: u64,
    pub unique_users: u64,
    pub avg_inference_time_ms: f64,
    pub total_inferences: u64,
    pub error_rate: f64,
    pub popular_versions: Vec<(String, u64)>,
    pub geographic_distribution: HashMap<String, u64>,
}
```

## Download API

### Parallel Downloads

#### `download_file_parallel(url: &str, path: &Path, config: ParallelDownloadConfig) -> Result<()>`

Download a file using parallel connections.

#### `download_files_parallel(downloads: Vec<DownloadRequest>, config: ParallelDownloadConfig) -> Result<Vec<DownloadResult>>`

Download multiple files in parallel.

### CDN Management

#### `CdnManager`

Manager for CDN endpoints and failover.

```rust
pub struct CdnManager {
    // Implementation details
}
```

**Methods:**
- `new(config: CdnConfig) -> Self`
- `add_endpoint(endpoint: CdnEndpoint) -> Result<()>`
- `health_check() -> Result<Vec<HealthCheckResult>>`
- `get_best_endpoint(location: Option<&str>) -> Option<&CdnEndpoint>`

## Security API

### SecurityManager

Manager for security features.

```rust
pub struct SecurityManager {
    // Implementation details
}
```

**Methods:**

#### `new(config: SecurityConfig) -> Self`

Create security manager with configuration.

#### `sign_model(model_path: &Path, key_pair: &KeyPair) -> Result<ModelSignature>`

Sign a model with cryptographic signature.

#### `verify_signature(model_path: &Path, signature: &ModelSignature, public_key: &PublicKey) -> Result<bool>`

Verify model signature.

#### `scan_vulnerabilities(model_path: &Path) -> Result<VulnerabilityScanResult>`

Scan model for security vulnerabilities.

### ModelSignature

Cryptographic signature for models.

```rust
pub struct ModelSignature {
    pub algorithm: SignatureAlgorithm,
    pub signature: Vec<u8>,
    pub public_key: Vec<u8>,
    pub timestamp: u64,
    pub metadata: HashMap<String, String>,
}
```

## Cache API

### CacheManager

Manager for model caching.

```rust
pub struct CacheManager {
    // Implementation details
}
```

**Methods:**

#### `new(config: CacheConfig) -> Result<Self>`

Create cache manager with configuration.

#### `get_cached_model(repo_name: &str, version: &str) -> Option<PathBuf>`

Get cached model path if available.

#### `cache_model(repo_name: &str, version: &str, model_path: &Path) -> Result<()>`

Cache a model.

#### `clear_cache() -> Result<()>`

Clear all cached models.

#### `get_cache_info() -> Result<CacheInfo>`

Get cache usage information.

### CacheConfig

Cache configuration.

```rust
pub struct CacheConfig {
    pub cache_directory: Option<PathBuf>,
    pub max_cache_size_gb: f64,
    pub auto_cleanup: bool,
    pub compression_enabled: bool,
    pub encryption_enabled: bool,
}
```

## Error Types

### TorshError

Main error type for ToRSh Hub operations.

```rust
pub enum TorshError {
    NetworkError(String),
    InvalidInput(String),
    ModelNotFound(String),
    PermissionDenied(String),
    SignatureVerificationFailed(String),
    CacheError(String),
    ConfigurationError(String),
    SecurityError(String),
    ComplianceError(String),
}
```

## Constants and Enums

### Device

Target device for model execution.

```rust
pub enum Device {
    Cpu,
    Cuda(usize),
    Metal,
    WebGpu,
}
```

### Precision

Numerical precision for model weights.

```rust
pub enum Precision {
    FP32,
    FP16,
    BF16,
    INT8,
    INT4,
}
```

### SortBy

Sorting options for search results.

```rust
pub enum SortBy {
    Relevance,
    Downloads,
    Rating,
    Updated,
    Created,
    Name,
    Size,
}
```

## Utility Functions

#### `current_timestamp() -> u64`

Get current Unix timestamp.

#### `format_bytes(bytes: u64) -> String`

Format byte count as human-readable string.

#### `format_duration(duration: Duration) -> String`

Format duration as human-readable string.

#### `validate_repo_name(name: &str) -> Result<()>`

Validate repository name format.

#### `validate_version(version: &str) -> Result<()>`

Validate version string format.

## Configuration

### Environment Variables

- `TORSH_HUB_URL`: Hub API endpoint
- `TORSH_HUB_API_KEY`: API authentication key
- `TORSH_HUB_CACHE_DIR`: Cache directory path
- `TORSH_HUB_CACHE_SIZE_GB`: Maximum cache size
- `TORSH_HUB_PARALLEL_DOWNLOADS`: Number of parallel download connections
- `TORSH_HUB_DOWNLOAD_TIMEOUT`: Download timeout in seconds
- `TORSH_HUB_VERIFY_SIGNATURES`: Enable signature verification (true/false)

### Configuration File Format

Configuration files use TOML format and can be placed at:
- `~/.torsh/hub_config.toml` (user-specific)
- `/etc/torsh/hub_config.toml` (system-wide)

```toml
[hub]
url = "https://hub.torsh.ai"
api_key = "your-api-key"

[cache]
directory = "/path/to/cache"
max_size_gb = 10.0
auto_cleanup = true

[download]
parallel_downloads = 4
timeout_seconds = 300
retry_attempts = 3

[security]
verify_signatures = true
trusted_publishers = ["torsh", "huggingface"]
scan_vulnerabilities = true
```

## Examples

For comprehensive usage examples, see the [Hub Guide](hub_guide.md) and the examples directory in the repository.