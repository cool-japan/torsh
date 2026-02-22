# Package Distribution and Deployment Guide

This guide covers best practices and workflows for distributing and deploying ToRSh packages in production environments.

## Table of Contents

1. [Distribution Strategies](#distribution-strategies)
2. [Package Registry](#package-registry)
3. [CDN Integration](#cdn-integration)
4. [Mirror Management](#mirror-management)
5. [Cloud Storage](#cloud-storage)
6. [High Availability](#high-availability)
7. [Security Considerations](#security-considerations)
8. [Monitoring & Analytics](#monitoring--analytics)
9. [Backup & Recovery](#backup--recovery)
10. [Production Deployment](#production-deployment)

## Distribution Strategies

### 1. Direct Distribution

Simple distribution for small-scale deployments:

```rust
use torsh_package::Package;

// Load package
let package = Package::load("model.torshpkg")?;

// Distribute via HTTP, FTP, or file sharing
// Simple but not scalable for large models
```

**Pros:**
- Simple setup
- No infrastructure required
- Good for internal teams

**Cons:**
- No redundancy
- Limited scalability
- No geographic optimization

### 2. Registry-Based Distribution

Centralized distribution through a package registry:

```rust
use torsh_package::{RegistryClient, RegistryConfig};

let config = RegistryConfig {
    base_url: "https://registry.example.com".to_string(),
    api_key: Some("your-api-key".to_string()),
    timeout_secs: 30,
    retry_attempts: 3,
};

let client = RegistryClient::new(config);

// Publish package
client.publish("my-model", "1.0.0", &package_data)?;

// Download package
let package = client.download("my-model", "1.0.0")?;
```

**Pros:**
- Centralized management
- Version control
- Access control
- Search and discovery

**Cons:**
- Single point of failure
- Requires infrastructure
- Potential bottleneck for large files

### 3. CDN-Based Distribution

Global distribution with edge caching:

```rust
use torsh_package::{CdnManager, CdnConfig, CdnProvider};

let config = CdnConfig {
    provider: CdnProvider::Cloudflare,
    account_id: "your-account".to_string(),
    api_key: "your-key".to_string(),
    zone_id: Some("your-zone".to_string()),
    edge_nodes: vec![/* edge node configs */],
};

let mut cdn = CdnManager::new(config);

// Upload to CDN
let url = cdn.upload_package("my-model", "1.0.0", &package_data)?;

// Users download from nearest edge node
// Fast global distribution
```

**Pros:**
- Global distribution
- Low latency
- High bandwidth
- DDoS protection
- Automatic caching

**Cons:**
- Cost
- CDN vendor lock-in
- Complex setup

### 4. Hybrid Distribution

Combine multiple strategies for optimal performance:

```rust
use torsh_package::{
    RegistryClient, CdnManager, MirrorManager, MirrorConfig
};

// Primary: Registry for metadata and small packages
let registry = RegistryClient::new(registry_config);

// Secondary: CDN for large model weights
let cdn = CdnManager::new(cdn_config);

// Tertiary: Mirrors for high availability
let mut mirrors = MirrorManager::new(mirror_config);

// Intelligent routing based on:
// - Package size
// - User location
// - Network conditions
// - Cost optimization
```

## Package Registry

### Setting Up a Registry

```rust
use torsh_package::{RegistryClient, RegistryConfig};

let config = RegistryConfig {
    base_url: "https://packages.company.com".to_string(),
    api_key: Some(std::env::var("REGISTRY_API_KEY").ok()),
    timeout_secs: 60,
    retry_attempts: 3,
};

let mut client = RegistryClient::new(config);
```

### Publishing Packages

```rust
// Publish with metadata
client.publish_with_metadata(
    "resnet50",
    "1.0.0",
    &package_data,
    PackageMetadata {
        description: Some("ResNet-50 for image classification".to_string()),
        author: Some("ML Team".to_string()),
        license: Some("MIT".to_string()),
        tags: vec!["computer-vision".to_string(), "classification".to_string()],
        ..Default::default()
    },
)?;
```

### Searching Packages

```rust
// Search by name
let results = client.search("resnet")?;
for result in results {
    println!("{} v{}", result.name, result.version);
}

// Get package metadata
let metadata = client.get_metadata("resnet50", "1.0.0")?;
```

### Package Caching

```rust
use torsh_package::PackageCache;

let cache = PackageCache::new(
    "/var/cache/torsh-packages",
    1024 * 1024 * 1024 * 10, // 10 GB cache
)?;

// Download with caching
let package = if let Some(cached) = cache.get("resnet50", "1.0.0")? {
    cached
} else {
    let pkg = client.download("resnet50", "1.0.0")?;
    cache.put("resnet50", "1.0.0", &pkg)?;
    pkg
};
```

## CDN Integration

### Multi-Provider CDN Setup

```rust
use torsh_package::{CdnManager, CdnConfig, CdnProvider, CdnRegion};

// Configure CDN
let config = CdnConfig {
    provider: CdnProvider::Cloudflare,
    account_id: "your-account".to_string(),
    api_key: "your-api-key".to_string(),
    zone_id: Some("your-zone".to_string()),
    edge_nodes: vec![
        EdgeNode {
            id: "us-east-1".to_string(),
            region: CdnRegion::NorthAmerica,
            endpoint: "https://us-east-1.cdn.example.com".to_string(),
            status: EdgeNodeStatus::Active,
            capacity_gbps: 100.0,
            current_load: 0.0,
            latency_ms: 0.0,
        },
        // Add more edge nodes for global coverage
    ],
};

let mut cdn = CdnManager::new(config);
```

### Uploading to CDN

```rust
// Upload with cache control
let url = cdn.upload_package_with_options(
    "my-model",
    "1.0.0",
    &package_data,
    CacheControl::Immutable,  // Cache forever (versioned content)
)?;

println!("Package available at: {}", url);
```

### Best Edge Node Selection

```rust
// CDN automatically selects best edge node based on:
// - Geographic proximity
// - Current load
// - Latency
// - Bandwidth availability

let best_url = cdn.get_best_url_for_location(
    "my-model",
    "1.0.0",
    "user-ip-address",
)?;
```

## Mirror Management

### Setting Up Mirrors

```rust
use torsh_package::{
    MirrorManager, MirrorConfig, Mirror, SelectionStrategy
};

let config = MirrorConfig {
    selection_strategy: SelectionStrategy::Geographic,
    health_check_interval_secs: 60,
    failover: Some(FailoverConfig {
        enabled: true,
        max_retries: 3,
        retry_delay_secs: 5,
        auto_fallback: true,
    }),
};

let mut manager = MirrorManager::new(config);

// Add mirrors
manager.add_mirror(Mirror {
    id: "mirror-us".to_string(),
    url: "https://us-mirror.example.com".to_string(),
    region: "us-east-1".to_string(),
    priority: 1,
    weight: 100,
    status: MirrorHealth::Healthy,
    ..Default::default()
})?;

manager.add_mirror(Mirror {
    id: "mirror-eu".to_string(),
    url: "https://eu-mirror.example.com".to_string(),
    region: "eu-central-1".to_string(),
    priority: 1,
    weight: 100,
    status: MirrorHealth::Healthy,
    ..Default::default()
})?;
```

### Automatic Failover

```rust
// Download with automatic failover
let package = manager.download_with_failover("my-model", "1.0.0")?;

// Health checks run automatically
manager.perform_health_checks()?;

// Get mirror statistics
let stats = manager.get_statistics();
println!("Total mirrors: {}", stats.total_mirrors);
println!("Healthy mirrors: {}", stats.healthy_mirrors);
```

## Cloud Storage

### S3-Compatible Storage

```rust
use torsh_package::{StorageManager, LocalStorage, S3Config};

// For production: use real S3 backend
let storage = MockS3Storage::new(S3Config {
    bucket: "my-models".to_string(),
    region: "us-east-1".to_string(),
    access_key: "your-access-key".to_string(),
    secret_key: "your-secret-key".to_string(),
    endpoint: None,  // Use AWS endpoint
});

let mut manager = StorageManager::new(Box::new(storage));

// Store package
manager.put("my-model/1.0.0/package.torshpkg", &package_data)?;

// Retrieve package
let data = manager.get("my-model/1.0.0/package.torshpkg")?;
```

### Multi-Cloud Strategy

```rust
// Primary: S3
let s3_storage = /* S3 backend */;

// Backup: GCS
let gcs_storage = /* GCS backend */;

// Backup: Azure Blob
let azure_storage = /* Azure backend */;

// Replicate across clouds for disaster recovery
for storage in [s3_storage, gcs_storage, azure_storage] {
    storage.put("my-model/1.0.0/package.torshpkg", &package_data)?;
}
```

## High Availability

### Multi-Region Replication

```rust
use torsh_package::{
    ReplicationManager, ReplicationConfig, ReplicationNode,
    ConsistencyLevel
};

let config = ReplicationConfig {
    consistency: ConsistencyLevel::Quorum,  // Majority must confirm
    replication_factor: 3,
    auto_failover: true,
    sync_interval_secs: 60,
};

let mut replication = ReplicationManager::new(config);

// Add nodes across regions
replication.add_node(ReplicationNode::new(
    "node-us-east".to_string(),
    "us-east-1".to_string(),
    "https://us-east.example.com".to_string(),
    1,  // High priority
    1024 * 1024 * 1024 * 1000,  // 1 TB capacity
))?;

replication.add_node(ReplicationNode::new(
    "node-eu-central".to_string(),
    "eu-central-1".to_string(),
    "https://eu-central.example.com".to_string(),
    1,
    1024 * 1024 * 1024 * 1000,
))?;

replication.add_node(ReplicationNode::new(
    "node-ap-southeast".to_string(),
    "ap-southeast-1".to_string(),
    "https://ap-southeast.example.com".to_string(),
    2,  // Lower priority
    1024 * 1024 * 1024 * 500,  // 500 GB capacity
))?;

// Replicate package
replication.replicate_package("my-model", "1.0.0", &package_data)?;

// Automatic health checks and failover
replication.health_check()?;
```

### Consistency Models

1. **Eventual Consistency**: Fastest, best for read-heavy workloads
2. **Quorum Consistency**: Balanced, good for most use cases
3. **Strong Consistency**: Slowest, required for critical data
4. **Causal Consistency**: Maintains causality relationships

## Security Considerations

### Package Signing

```rust
use torsh_package::{PackageSigner, SignatureAlgorithm};

let signer = PackageSigner::new(SignatureAlgorithm::Ed25519);

// Sign package before distribution
let signature = signer.sign(&package_data)?;

// Verify on download
signer.verify(&package_data, &signature)?;
```

### Encryption

```rust
use torsh_package::{PackageEncryptor, EncryptionAlgorithm};

let encryptor = PackageEncryptor::new(EncryptionAlgorithm::Aes256Gcm);

// Encrypt sensitive models
let encrypted = encryptor.encrypt(&package_data, "password")?;

// Decrypt on authorized systems
let decrypted = encryptor.decrypt(&encrypted, "password")?;
```

### Access Control

```rust
use torsh_package::{AccessControlManager, Role, Permission};

let mut acl = AccessControlManager::new();

// Define roles
acl.create_role("data-scientist", vec![
    Permission::DownloadPackage,
    Permission::ViewMetadata,
])?;

acl.create_role("ml-engineer", vec![
    Permission::DownloadPackage,
    Permission::UploadPackage,
    Permission::ViewMetadata,
    Permission::ModifyMetadata,
])?;

// Grant access
acl.grant_user_role("alice@company.com", "ml-engineer")?;
```

## Monitoring & Analytics

### Metrics Collection

```rust
use torsh_package::{MetricsCollector, MetricType, AlertThreshold};

let mut collector = MetricsCollector::new();

// Configure alerting
collector.set_alert_threshold(
    MetricType::DownloadTime,
    AlertThreshold::Maximum(Duration::from_secs(30)),
);

// Record operations
collector.record_download("my-model", "1.0.0", Duration::from_secs(5));
collector.record_upload("my-model", "2.0.0", 1024 * 1024 * 100);  // 100 MB

// Generate reports
let report = collector.generate_report();
println!("Total downloads: {}", report.total_downloads);
println!("Total bandwidth: {} GB", report.total_bandwidth_bytes / 1024 / 1024 / 1024);
```

### Dashboard Integration

```rust
// Export metrics to JSON for dashboards
let json = collector.export_to_json()?;

// Send to monitoring system (Prometheus, Grafana, Datadog, etc.)
send_to_monitoring_system(&json)?;
```

## Backup & Recovery

### Automated Backups

```rust
use torsh_package::{BackupManager, BackupConfig, BackupStrategy, RetentionPolicy};

let config = BackupConfig {
    destination: PathBuf::from("/backups"),
    strategy: BackupStrategy::Incremental,
    compression: true,
    encryption: true,
    retention: RetentionPolicy::KeepLast(30),  // Keep last 30 backups
};

let mut backup_mgr = BackupManager::new(config);

// Create backup
let backup_id = backup_mgr.create_backup("my-model", "1.0.0", &package_data)?;

// Verify backup
let verification = backup_mgr.verify_backup(&backup_id);
assert!(verification.success);

// Restore if needed
let restored = backup_mgr.restore_backup(&backup_id)?;
```

### Disaster Recovery

```rust
// Create recovery point before major updates
let recovery_point = backup_mgr.create_recovery_point(
    "my-model",
    "1.0.0",
    "Before production deployment".to_string(),
)?;

// If deployment fails, restore to recovery point
let restored = backup_mgr.restore_to_recovery_point(&recovery_point)?;
```

## Production Deployment

### Complete Production Setup

```rust
use torsh_package::*;

struct ProductionPackageDistribution {
    registry: RegistryClient,
    cdn: CdnManager,
    mirrors: MirrorManager,
    replication: ReplicationManager,
    backup: BackupManager,
    monitoring: MetricsCollector,
    acl: AccessControlManager,
}

impl ProductionPackageDistribution {
    pub fn new() -> Self {
        Self {
            registry: RegistryClient::new(registry_config()),
            cdn: CdnManager::new(cdn_config()),
            mirrors: MirrorManager::new(mirror_config()),
            replication: ReplicationManager::new(replication_config()),
            backup: BackupManager::new(backup_config()),
            monitoring: MetricsCollector::new(),
            acl: AccessControlManager::new(),
        }
    }

    pub fn deploy_package(&mut self, name: &str, version: &str, data: &[u8]) -> Result<(), TorshError> {
        // 1. Verify signature
        // 2. Create backup
        let backup_id = self.backup.create_backup(name, version, data)?;

        // 3. Publish to registry
        self.registry.publish(name, version, data)?;

        // 4. Upload to CDN
        self.cdn.upload_package(name, version, data)?;

        // 5. Replicate across regions
        self.replication.replicate_package(name, version, data)?;

        // 6. Update mirrors
        self.mirrors.synchronize(name, version, data)?;

        // 7. Record metrics
        self.monitoring.record_upload(name, version, data.len() as u64);

        Ok(())
    }
}
```

### Deployment Checklist

- [ ] Package signed with valid certificate
- [ ] Backup created before deployment
- [ ] Access controls configured
- [ ] Monitoring and alerting enabled
- [ ] CDN cache configured
- [ ] Mirrors synchronized
- [ ] Replication verified
- [ ] Health checks passing
- [ ] Rollback plan ready
- [ ] Documentation updated

## Best Practices

1. **Always sign packages** before distribution
2. **Use CDN** for large models (>100 MB)
3. **Enable replication** for critical packages
4. **Monitor downloads** and bandwidth usage
5. **Automate backups** before updates
6. **Test disaster recovery** procedures
7. **Use semantic versioning** consistently
8. **Document dependencies** clearly
9. **Set up alerting** for failures
10. **Review access logs** regularly

## Troubleshooting

### Slow Downloads

```rust
// Check CDN statistics
let stats = cdn.get_statistics();
println!("Cache hit rate: {:.1}%", stats.cache_hit_rate * 100.0);

// Check mirror health
manager.perform_health_checks()?;
let stats = manager.get_statistics();
println!("Healthy mirrors: {}/{}", stats.healthy_mirrors, stats.total_mirrors);
```

### Replication Issues

```rust
// Check replication lag
let stats = replication.get_statistics();
println!("Avg replication lag: {:.2}s", stats.avg_replication_lag_secs);

// Check for conflicts
let conflicts = replication.get_conflicts();
if !conflicts.is_empty() {
    replication.resolve_conflicts()?;
}
```

### Storage Full

```rust
// Check storage usage
let stats = storage_manager.get_stats();
println!("Storage used: {} / {} GB",
    stats.total_size / 1024 / 1024 / 1024,
    stats.capacity / 1024 / 1024 / 1024
);

// Clean up old backups
backup_mgr.apply_retention_policy()?;
```

## Conclusion

Effective package distribution requires:
- **Redundancy**: Multiple mirrors and replication
- **Performance**: CDN and caching
- **Security**: Signing and access control
- **Monitoring**: Analytics and alerting
- **Recovery**: Backups and disaster recovery

Choose the distribution strategy that best fits your:
- Scale (team size, user base)
- Budget (infrastructure costs)
- Requirements (latency, availability, security)
- Regulatory needs (compliance, data sovereignty)
