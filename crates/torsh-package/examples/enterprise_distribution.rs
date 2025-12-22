//! Enterprise Distribution Features Example
//!
//! This example demonstrates the enterprise-grade distribution features:
//! - CDN integration for fast package distribution
//! - Mirror management for high availability
//! - Audit logging for security compliance and tracking

use std::path::PathBuf;

use torsh_package::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Enterprise Distribution Features Demo ===\n");

    // 1. Demonstrate CDN integration
    demonstrate_cdn()?;

    // 2. Demonstrate mirror management
    demonstrate_mirrors()?;

    // 3. Demonstrate audit logging
    demonstrate_audit_logging()?;

    println!("\n=== All demonstrations completed successfully! ===");
    Ok(())
}

/// Demonstrate CDN integration for fast package distribution
fn demonstrate_cdn() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. CDN Integration");
    println!("------------------");

    // Create CDN configuration
    let config = CdnConfig::new(CdnProvider::Cloudflare, "https://cdn.torsh.rs".to_string())
        .with_cache_ttl(86400) // 24 hours
        .with_edge_compression(true)
        .add_region(CdnRegion::NorthAmerica)
        .add_region(CdnRegion::Europe)
        .add_region(CdnRegion::AsiaPacific);

    println!("CDN Configuration:");
    println!("  Provider: {:?}", config.provider);
    println!("  Endpoint: {}", config.endpoint);
    println!("  Cache TTL: {} seconds", config.cache_ttl);
    println!("  Edge compression: {}", config.edge_compression);
    println!("  Regions: {} configured", config.regions.len());

    // Create CDN manager
    let mut manager = CdnManager::new(config);

    // Add edge nodes
    let node1 = EdgeNode::new(
        "edge-us-east-1".to_string(),
        "New York, USA".to_string(),
        CdnRegion::NorthAmerica,
    );
    manager.add_edge_node(node1);

    let node2 = EdgeNode::new(
        "edge-eu-west-1".to_string(),
        "London, UK".to_string(),
        CdnRegion::Europe,
    );
    manager.add_edge_node(node2);

    let node3 = EdgeNode::new(
        "edge-ap-southeast-1".to_string(),
        "Singapore".to_string(),
        CdnRegion::AsiaPacific,
    );
    manager.add_edge_node(node3);

    println!("\n✓ Edge nodes configured:");
    for node in manager.get_healthy_nodes() {
        println!(
            "  - {} ({}) - Score: {:.2}",
            node.id,
            node.location,
            node.calculate_score()
        );
    }

    // Upload package to CDN
    let package_data = b"package content here";
    let url = manager.upload_package("ml-models", "1.0.0", package_data)?;
    println!("\n✓ Package uploaded to CDN:");
    println!("  URL: {}", url);

    // Get best edge node for a region
    if let Some(best_node) = manager.get_best_node(&CdnRegion::NorthAmerica) {
        println!("\n✓ Best edge node for North America:");
        println!("  Node: {}", best_node.id);
        println!("  Location: {}", best_node.location);
        println!("  Load: {}%", best_node.load);
        println!("  Latency: {}ms", best_node.latency_ms);
    }

    // Generate cache control headers
    let cache_control = manager.generate_cache_control("1.0.0");
    println!("\n✓ Cache-Control header:");
    println!("  {}", cache_control);

    // Get package URL
    if let Some(cached_url) = manager.get_package_url("ml-models", "1.0.0") {
        println!("\n✓ Package available in CDN cache:");
        println!("  Cached URL: {}", cached_url);
    }

    // Display CDN statistics
    let stats = manager.get_statistics();
    println!("\nCDN Statistics:");
    println!("  Total requests: {}", stats.total_requests);
    println!("  Cache hit rate: {:.1}%", stats.cache_hit_rate * 100.0);
    println!("  Bytes transferred: {} bytes", stats.bytes_transferred);

    println!();
    Ok(())
}

/// Demonstrate mirror management for high availability
fn demonstrate_mirrors() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Mirror Management");
    println!("--------------------");

    // Create mirror manager with geographic selection strategy
    let mut manager = MirrorManager::new(SelectionStrategy::Geographic);

    // Configure primary mirror (US)
    let us_mirror = Mirror::new(
        MirrorConfig::new(
            "us-mirror-1".to_string(),
            "https://us.mirrors.torsh.rs".to_string(),
            "us-east".to_string(),
        )
        .with_priority(10)
        .with_weight(100),
    );
    manager.add_mirror(us_mirror)?;

    // Configure European mirror
    let eu_mirror = Mirror::new(
        MirrorConfig::new(
            "eu-mirror-1".to_string(),
            "https://eu.mirrors.torsh.rs".to_string(),
            "europe".to_string(),
        )
        .with_priority(10)
        .with_weight(100),
    );
    manager.add_mirror(eu_mirror)?;

    // Configure Asian mirror
    let asia_mirror = Mirror::new(
        MirrorConfig::new(
            "asia-mirror-1".to_string(),
            "https://asia.mirrors.torsh.rs".to_string(),
            "asia-pacific".to_string(),
        )
        .with_priority(20)
        .with_weight(80),
    );
    manager.add_mirror(asia_mirror)?;

    println!("Mirror Configuration:");
    println!("  Total mirrors: {}", manager.get_available_mirrors().len());
    println!("  Healthy mirrors: {}", manager.get_healthy_mirrors().len());
    println!("  Selection strategy: Geographic");

    // Update mirror health status
    manager.update_mirror_health("us-mirror-1", MirrorHealth::Healthy)?;
    manager.update_mirror_health("eu-mirror-1", MirrorHealth::Healthy)?;
    manager.update_mirror_health("asia-mirror-1", MirrorHealth::Degraded)?;

    println!("\nMirror Health Status:");
    for mirror in manager.get_available_mirrors() {
        println!(
            "  - {} ({:?}) - Priority: {}, Weight: {}",
            mirror.config.id, mirror.health, mirror.config.priority, mirror.config.weight
        );
    }

    // Select best mirror for US region
    if let Some(selection) = manager.select_mirror(Some("us-east")) {
        println!("\n✓ Best mirror for US East:");
        println!("  Selected: {}", selection.mirror.config.url);
        println!("  Fallbacks available: {}", selection.fallbacks.len());

        if !selection.fallbacks.is_empty() {
            println!("\nFallback mirrors:");
            for (i, fallback) in selection.fallbacks.iter().take(2).enumerate() {
                println!(
                    "  {}. {} (Score: {:.2})",
                    i + 1,
                    fallback.config.url,
                    fallback.calculate_score()
                );
            }
        }
    }

    // Configure failover settings
    let failover_config = FailoverConfig {
        enabled: true,
        max_retries: 3,
        retry_delay_ms: 1000,
        auto_failback: true,
        min_healthy_mirrors: 2,
    };
    manager.set_failover_config(failover_config);

    println!("\n✓ Failover Configuration:");
    let config = manager.get_failover_config();
    println!("  Enabled: {}", config.enabled);
    println!("  Max retries: {}", config.max_retries);
    println!("  Retry delay: {}ms", config.retry_delay_ms);
    println!("  Auto failback: {}", config.auto_failback);

    // Check if sufficient mirrors are available
    if manager.has_sufficient_mirrors() {
        println!("\n✓ Sufficient healthy mirrors available for high availability");
    }

    // Display mirror statistics
    let stats = manager.get_statistics();
    println!("\nMirror Statistics:");
    println!("  Total requests: {}", stats.total_requests);
    println!("  Failed requests: {}", stats.failed_requests);
    println!("  Failover count: {}", stats.failover_count);

    println!();
    Ok(())
}

/// Demonstrate audit logging for security compliance
fn demonstrate_audit_logging() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Audit Logging");
    println!("----------------");

    // Create audit log configuration
    let config = AuditLogConfig::new(PathBuf::from("/tmp/torsh-audit.log"));
    config.validate()?;

    println!("Audit Log Configuration:");
    println!("  Enabled: {}", config.enabled);
    println!("  Log path: {:?}", config.log_path);
    println!("  Format: {:?}", config.format);
    println!("  Min severity: {:?}", config.min_severity);
    println!("  Buffer size: {} events", config.buffer_size);

    // Create audit logger
    let mut logger = AuditLogger::new(config)?;

    // Log various events
    println!("\nLogging events...");

    // Log package download
    logger.log_download("user123", "torch", "2.1.0")?;
    println!("  ✓ Logged package download");

    // Log package upload
    logger.log_upload("user456", "my-model", "1.0.0")?;
    println!("  ✓ Logged package upload");

    // Log access denied event
    logger.log_access_denied("user789", "private-package", "Insufficient permissions")?;
    println!("  ✓ Logged access denial");

    // Log security violation
    logger.log_security_violation(
        Some("suspicious-user"),
        "Multiple failed login attempts",
        "Possible brute force attack detected",
    )?;
    println!("  ✓ Logged security violation");

    // Log custom event
    let custom_event = AuditEvent::new(
        AuditEventType::ConfigurationChange,
        "Updated package cache settings".to_string(),
    )
    .with_user("admin".to_string())
    .with_severity(AuditSeverity::Info)
    .add_metadata("setting".to_string(), "cache_size".to_string())
    .add_metadata("old_value".to_string(), "1000".to_string())
    .add_metadata("new_value".to_string(), "2000".to_string());

    logger.log(custom_event)?;
    println!("  ✓ Logged configuration change");

    // Display statistics
    let stats = logger.get_statistics();
    println!("\nAudit Statistics:");
    println!("  Total events logged: {}", stats.total_events);
    println!("  Failed actions: {}", stats.failed_actions);
    println!("  Security violations: {}", stats.security_violations);

    println!("\nEvents by type:");
    for (event_type, count) in &stats.events_by_type {
        println!("  - {}: {}", event_type, count);
    }

    println!("\nEvents by severity:");
    for (severity, count) in &stats.events_by_severity {
        println!("  - {}: {}", severity, count);
    }

    // Query specific event types
    let download_count = logger.get_event_count(&AuditEventType::PackageDownload);
    let upload_count = logger.get_event_count(&AuditEventType::PackageUpload);
    let denied_count = logger.get_event_count(&AuditEventType::AccessDenied);

    println!("\nEvent counts:");
    println!("  Package downloads: {}", download_count);
    println!("  Package uploads: {}", upload_count);
    println!("  Access denied: {}", denied_count);

    // Demonstrate event formatting
    println!("\nSample audit event (JSON format):");
    let sample_event = AuditEvent::new(
        AuditEventType::PackageDownload,
        "Download sample-package".to_string(),
    )
    .with_user("demo-user".to_string())
    .with_ip("192.168.1.100".to_string())
    .with_resource("sample-package:1.0.0".to_string());

    println!("{}", sample_event.to_json()?);

    println!("\nSample audit event (Text format):");
    println!("{}", sample_event.to_text());

    // Flush audit log
    logger.flush()?;
    println!("\n✓ Audit log flushed to disk");

    println!();
    Ok(())
}
