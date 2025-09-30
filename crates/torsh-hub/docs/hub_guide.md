# ToRSh Hub User Guide

## Introduction

ToRSh Hub is a comprehensive model sharing and discovery platform for the ToRSh deep learning framework. It provides PyTorch Hub-compatible functionality while adding advanced features for model management, community collaboration, and enterprise deployment.

## Getting Started

### Installation

ToRSh Hub is included as part of the ToRSh framework:

```rust
use torsh_hub::*;

// Load a model from the hub
let model = hub::load("username/model-name", "v1.0.0")?;

// Or load with custom configuration
let config = ModelConfig::default()
    .with_device(Device::Cuda(0))
    .with_precision(Precision::FP16);
let model = hub::load_with_config("username/model-name", config)?;
```

### Basic Model Loading

```rust
use torsh_hub::{hub, ModelConfig};

// Load a pre-trained ResNet model
let resnet = hub::load("torsh/resnet50", "latest")?;

// Load with specific device placement
let config = ModelConfig::default().with_device(Device::Cuda(0));
let model = hub::load_with_config("torsh/bert-base", config)?;
```

## Core Features

### Model Discovery

```rust
use torsh_hub::{ModelRegistry, SearchQuery, ModelCategory};

let registry = ModelRegistry::new();

// Search for vision models
let query = SearchQuery::new()
    .category(ModelCategory::Vision)
    .accuracy_min(0.8)
    .max_model_size(100_000_000); // 100MB

let results = registry.search_models(query)?;
for model in results {
    println!("Found: {} (accuracy: {:.2})", model.name, model.accuracy);
}
```

### Model Information

```rust
use torsh_hub::{ModelCard, ModelInfo};

// Get detailed model information
let model_info = hub::info("username/model-name")?;
println!("Model: {}", model_info.name);
println!("Description: {}", model_info.description);
println!("Architecture: {}", model_info.architecture);

// Get model card
let model_card = hub::model_card("username/model-name")?;
println!("Model Card:\n{}", model_card.render_markdown()?);
```

## Community Features

### Model Ratings

```rust
use torsh_hub::{CommunityManager, ModelRating, RatingCategory};

let mut community = CommunityManager::new();

// Add a rating
let rating = ModelRating {
    user_id: "user123".to_string(),
    model_id: "username/model-name".to_string(),
    rating: 5,
    review: Some("Excellent model with great accuracy!".to_string()),
    categories: vec![RatingCategory::Accuracy, RatingCategory::Performance],
    ..Default::default()
};

community.add_rating(rating)?;

// Get rating statistics
let stats = community.get_rating_stats("username/model-name");
if let Some(stats) = stats {
    println!("Average rating: {:.1}/5.0 ({} reviews)", 
             stats.average_rating, stats.total_ratings);
}
```

### Discussions

```rust
use torsh_hub::{Discussion, DiscussionCategory, DiscussionStatus};

// Create a discussion
let discussion = Discussion {
    title: "Best practices for fine-tuning BERT".to_string(),
    description: "What are the recommended approaches for fine-tuning BERT models?".to_string(),
    author_id: "user123".to_string(),
    category: DiscussionCategory::Tutorials,
    tags: vec!["bert".to_string(), "fine-tuning".to_string()],
    status: DiscussionStatus::Open,
    ..Default::default()
};

let discussion_id = community.create_discussion(discussion)?;
```

### Challenges

```rust
use torsh_hub::{Challenge, ChallengeType, EvaluationCriteria, MetricType};

// Create a challenge
let challenge = Challenge {
    title: "Image Classification Accuracy Challenge".to_string(),
    description: "Build the most accurate image classifier for CIFAR-10".to_string(),
    organizer_id: "organizer123".to_string(),
    challenge_type: ChallengeType::ModelAccuracy,
    evaluation_criteria: vec![
        EvaluationCriteria {
            name: "Top-1 Accuracy".to_string(),
            description: "Classification accuracy on test set".to_string(),
            weight: 0.8,
            metric_type: MetricType::Accuracy,
        },
        EvaluationCriteria {
            name: "Model Efficiency".to_string(),
            description: "Inference speed (FPS)".to_string(),
            weight: 0.2,
            metric_type: MetricType::ThroughputOps,
        },
    ],
    ..Default::default()
};

let challenge_id = community.create_challenge(challenge)?;
```

## Advanced Features

### Fine-Tuning

```rust
use torsh_hub::{FineTuner, FineTuningConfig, FineTuningStrategy};

// Load a pre-trained model for fine-tuning
let base_model = hub::load("torsh/bert-base", "latest")?;

// Configure fine-tuning
let config = FineTuningConfig {
    strategy: FineTuningStrategy::LoRA { rank: 16, alpha: 32 },
    learning_rate: 2e-5,
    batch_size: 16,
    num_epochs: 3,
    warmup_steps: 500,
    ..Default::default()
};

// Create fine-tuner
let mut fine_tuner = FineTuner::new(base_model, config)?;

// Train on your dataset
fine_tuner.train(train_dataset, validation_dataset)?;

// Save the fine-tuned model
fine_tuner.save("my-username/my-finetuned-model")?;
```

### Model Analytics

```rust
use torsh_hub::{AnalyticsManager, ModelUsageStats};

let analytics = AnalyticsManager::new();

// Track model usage
analytics.track_model_download("username/model-name", "user123")?;
analytics.track_model_inference("username/model-name", "user123", 45.2)?; // 45.2ms latency

// Get usage statistics
let stats = analytics.get_model_usage_stats("username/model-name")?;
println!("Downloads: {}", stats.total_downloads);
println!("Average inference time: {:.1}ms", stats.avg_inference_time_ms);
```

### Caching

```rust
use torsh_hub::{CacheManager, CacheConfig};

// Configure caching
let cache_config = CacheConfig {
    max_cache_size_gb: 10.0,
    cache_directory: Some("/path/to/cache".into()),
    auto_cleanup: true,
    ..Default::default()
};

let cache_manager = CacheManager::new(cache_config)?;

// Models are automatically cached when loaded
let model = hub::load("username/large-model", "latest")?; // Downloads and caches
let model2 = hub::load("username/large-model", "latest")?; // Loads from cache

// Manual cache management
cache_manager.clear_cache()?;
let cache_info = cache_manager.get_cache_info()?;
println!("Cache size: {:.1} GB", cache_info.total_size_gb);
```

## Enterprise Features

### Private Repositories

```rust
use torsh_hub::{EnterpriseManager, PrivateRepository, RepositoryVisibility, 
                RepositoryAccessControl, StorageConfig, EncryptionAlgorithm};

let mut enterprise = EnterpriseManager::new();

// Create a private repository
let repo = PrivateRepository {
    name: "confidential-models".to_string(),
    description: Some("Private models for internal use".to_string()),
    organization_id: "acme-corp".to_string(),
    owner_id: "admin".to_string(),
    visibility: RepositoryVisibility::Private,
    access_control: RepositoryAccessControl {
        require_mfa: true,
        session_timeout_minutes: 30,
        ip_whitelist: vec!["192.168.1.0/24".to_string()],
        ..Default::default()
    },
    storage_config: StorageConfig {
        encryption_enabled: true,
        encryption_algorithm: EncryptionAlgorithm::AES256,
        compression_enabled: true,
        ..Default::default()
    },
    ..Default::default()
};

let repo_id = enterprise.create_private_repository(repo)?;
```

### Role-Based Access Control

```rust
use torsh_hub::{Role, Permission, UserRoleAssignment, ResourceType, Action, PermissionScope};

// Create a custom role
let role = Role {
    name: "Model Reviewer".to_string(),
    description: "Can review and approve models".to_string(),
    organization_id: "acme-corp".to_string(),
    permissions: vec![
        "model.read".to_string(),
        "model.review".to_string(),
        "model.approve".to_string(),
    ].into_iter().collect(),
    ..Default::default()
};

let role_id = enterprise.create_role(role)?;

// Assign role to user
let assignment = UserRoleAssignment {
    user_id: "reviewer123".to_string(),
    role_id: role_id.clone(),
    organization_id: "acme-corp".to_string(),
    assigned_by: "admin".to_string(),
    expires_at: Some(current_timestamp() + 86400 * 365), // 1 year
    ..Default::default()
};

enterprise.assign_role(assignment)?;

// Check permissions
let can_review = enterprise.check_permission("reviewer123", "model.review", Some("model123"));
```

### Audit Logging

```rust
use torsh_hub::{AuditLogEntry, AuditAction, ResourceType};

// Audit logs are automatically generated for all actions
let logs = enterprise.get_audit_logs(
    "acme-corp",
    Some(current_timestamp() - 86400), // Last 24 hours
    None
);

for log in logs {
    println!("{}: {} performed {} on {} {}",
             log.timestamp,
             log.user_id.as_deref().unwrap_or("system"),
             format!("{:?}", log.action),
             format!("{:?}", log.resource_type),
             log.resource_id);
}
```

### Compliance Reporting

```rust
use torsh_hub::{ComplianceLabel, ComplianceReport};

// Generate compliance report
let report_id = enterprise.generate_compliance_report(
    "acme-corp",
    ComplianceLabel::SOC2,
    current_timestamp() - 86400 * 90, // Last 90 days
    current_timestamp()
)?;

// Get the report
if let Some(report) = enterprise.compliance_reports.get(&report_id) {
    println!("Compliance Score: {:.1}%", report.compliance_score * 100.0);
    println!("Status: {:?}", report.status);
    
    for finding in &report.findings {
        println!("Finding: {} ({})", finding.description, finding.severity);
    }
}
```

## Configuration

### Environment Variables

```bash
# Cache configuration
export TORSH_HUB_CACHE_DIR="/path/to/cache"
export TORSH_HUB_CACHE_SIZE_GB="10"

# Hub endpoints
export TORSH_HUB_URL="https://hub.torsh.ai"
export TORSH_HUB_API_KEY="your-api-key"

# Performance tuning
export TORSH_HUB_PARALLEL_DOWNLOADS="4"
export TORSH_HUB_DOWNLOAD_TIMEOUT="300"
```

### Configuration File

Create `~/.torsh/hub_config.toml`:

```toml
[cache]
directory = "/path/to/cache"
max_size_gb = 10.0
auto_cleanup = true

[download]
parallel_downloads = 4
timeout_seconds = 300
retry_attempts = 3

[hub]
url = "https://hub.torsh.ai"
api_key = "your-api-key"

[security]
verify_signatures = true
trusted_publishers = ["torsh", "huggingface"]
```

## Best Practices

### Model Selection

1. **Check ratings and reviews** before downloading models
2. **Verify model signatures** for security
3. **Consider model size** vs. accuracy trade-offs
4. **Test on your specific use case** before production deployment

### Performance Optimization

1. **Use appropriate precision** (FP16 for inference, FP32 for training)
2. **Enable caching** for frequently used models
3. **Use parallel downloads** for large models
4. **Consider CDN mirrors** for faster downloads

### Security

1. **Verify model signatures** before use
2. **Use private repositories** for sensitive models
3. **Implement proper access controls** in enterprise environments
4. **Regular audit log reviews** for compliance

### Community Engagement

1. **Rate and review models** you use
2. **Participate in discussions** to share knowledge
3. **Contribute improvements** back to the community
4. **Join challenges** to advance the state of the art

## Troubleshooting

### Common Issues

**Model not found:**
```rust
// Check if model exists
let info = hub::info("username/model-name");
match info {
    Ok(info) => println!("Model found: {}", info.name),
    Err(e) => println!("Model not found: {}", e),
}
```

**Download failures:**
```rust
// Use retry mechanism
let mut attempts = 0;
let max_attempts = 3;

loop {
    match hub::load("username/model-name", "latest") {
        Ok(model) => break,
        Err(e) => {
            attempts += 1;
            if attempts >= max_attempts {
                return Err(e);
            }
            std::thread::sleep(std::time::Duration::from_secs(2));
        }
    }
}
```

**Cache issues:**
```rust
// Clear cache if corrupted
let cache_manager = CacheManager::new(CacheConfig::default())?;
cache_manager.clear_cache()?;
```

### Getting Help

- **Documentation**: [https://docs.torsh.ai/hub](https://docs.torsh.ai/hub)
- **Community Forums**: Join discussions on the ToRSh Hub
- **Issue Tracker**: Report bugs on GitHub
- **Enterprise Support**: Contact support for enterprise features

## API Reference

See the [API Reference](api_reference.md) for detailed documentation of all types and functions.

## Examples

See the [examples](examples/) directory for more comprehensive usage examples.