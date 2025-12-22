//! Production Features Demonstration
//!
//! This example demonstrates the enterprise-grade production features
//! implemented in torsh-package:
//! - Package Lineage Tracking (ML Governance)
//! - Monitoring & Analytics (Observability)
//! - Backup & Recovery (Data Safety)
//! - High Availability & Replication (Distributed Systems)

use chrono::Utc;
use std::path::PathBuf;
use std::time::Duration;
use torsh_package::{
    // Monitoring
    AlertThreshold,
    // Backup
    BackupConfig,
    BackupManager,
    BackupStrategy,
    // Governance
    ComplianceLevel,
    ComplianceMetadata,
    // Replication
    ConsistencyLevel,
    LineageRelation,
    LineageTracker,
    MetricType,
    MetricsCollector,
    ProvenanceInfo,
    ReplicationConfig,
    ReplicationManager,
    ReplicationNode,
    RetentionPolicy,
    TransformationRecord,
};

fn main() {
    println!("=== ToRSh Package Production Features Demo ===\n");

    // 1. PACKAGE LINEAGE TRACKING & GOVERNANCE
    println!("1. Package Lineage Tracking & ML Governance");
    println!("-------------------------------------------");
    demonstrate_governance();

    // 2. MONITORING & ANALYTICS
    println!("\n2. Monitoring & Analytics");
    println!("-------------------------");
    demonstrate_monitoring();

    // 3. BACKUP & RECOVERY
    println!("\n3. Backup & Recovery");
    println!("--------------------");
    demonstrate_backup();

    // 4. HIGH AVAILABILITY & REPLICATION
    println!("\n4. High Availability & Replication");
    println!("----------------------------------");
    demonstrate_replication();

    println!("\n=== Demo Complete ===");
}

fn demonstrate_governance() {
    let mut tracker = LineageTracker::new();

    // Record provenance for base model
    println!("Recording provenance for base model...");
    let base_provenance = ProvenanceInfo {
        package_id: "resnet50-base".to_string(),
        creator: "ml-team@company.com".to_string(),
        creation_time: Utc::now(),
        source_url: Some("https://github.com/company/models".to_string()),
        source_commit: Some("abc123def456".to_string()),
        build_environment: [
            ("python".to_string(), "3.11.0".to_string()),
            ("pytorch".to_string(), "2.0.0".to_string()),
            ("cuda".to_string(), "11.8".to_string()),
        ]
        .iter()
        .cloned()
        .collect(),
        description: "ResNet-50 base model trained on ImageNet".to_string(),
    };
    tracker.record_provenance(base_provenance);

    // Record fine-tuned model
    println!("Recording fine-tuned model lineage...");
    let finetuned_provenance = ProvenanceInfo {
        package_id: "resnet50-medical".to_string(),
        creator: "medical-ai@company.com".to_string(),
        creation_time: Utc::now(),
        source_url: Some("https://github.com/company/medical-models".to_string()),
        source_commit: Some("def456ghi789".to_string()),
        build_environment: [("python".to_string(), "3.11.0".to_string())]
            .iter()
            .cloned()
            .collect(),
        description: "ResNet-50 fine-tuned for medical imaging".to_string(),
    };
    tracker.record_provenance(finetuned_provenance);

    // Add lineage relationship
    tracker
        .add_lineage(
            "resnet50-base".to_string(),
            "resnet50-medical".to_string(),
            LineageRelation::DerivedFrom,
            "Fine-tuned on medical imaging dataset".to_string(),
        )
        .unwrap();

    // Record transformation
    println!("Recording transformation operation...");
    let transformation = TransformationRecord {
        package_id: "resnet50-medical".to_string(),
        operation: "fine-tuning".to_string(),
        timestamp: Utc::now(),
        performed_by: "medical-ai@company.com".to_string(),
        parameters: [
            ("learning_rate".to_string(), "0.001".to_string()),
            ("epochs".to_string(), "50".to_string()),
            ("batch_size".to_string(), "32".to_string()),
        ]
        .iter()
        .cloned()
        .collect(),
        result: "success".to_string(),
        duration_secs: 14400.0, // 4 hours
    };
    tracker.record_transformation(transformation);

    // Set compliance metadata
    println!("Setting compliance requirements...");
    let compliance = ComplianceMetadata {
        package_id: "resnet50-medical".to_string(),
        level: ComplianceLevel::Regulatory,
        tags: vec!["HIPAA".to_string(), "FDA".to_string()],
        certifications: vec!["ISO13485".to_string()],
        data_classification: "Confidential-Medical".to_string(),
        retention_days: Some(2555), // 7 years
        access_restrictions: vec!["medical-team-only".to_string()],
        audit_required: true,
        last_audit: Some(Utc::now()),
        next_audit_due: Some(Utc::now() + chrono::Duration::days(90)),
    };
    tracker.set_compliance(compliance);

    // Query lineage
    let lineage = tracker.get_lineage("resnet50-medical");
    println!("Lineage query results:");
    println!("  - Packages in lineage: {}", lineage.packages.len());
    println!("  - Relationships: {}", lineage.edges.len());

    // Get statistics
    let stats = tracker.get_statistics();
    println!("Lineage statistics:");
    println!("  - Total packages: {}", stats.total_packages);
    println!("  - Total edges: {}", stats.total_edges);
    println!(
        "  - Packages with compliance: {}",
        stats.packages_with_compliance
    );

    // Generate compliance report
    let report = tracker.generate_compliance_report();
    println!("Compliance report:");
    println!("  - Total packages: {}", report.total_packages);
    println!("  - Compliant packages: {}", report.compliant_packages);
    println!("  - Issues found: {}", report.issues.len());

    // Export to DOT format for visualization
    let dot = tracker.export_to_dot("resnet50-medical");
    println!("Graphviz DOT export: {} bytes", dot.len());
}

fn demonstrate_monitoring() {
    let mut collector = MetricsCollector::new();

    // Configure alerting
    collector.set_alert_threshold(
        MetricType::DownloadTime,
        AlertThreshold::Maximum(Duration::from_secs(10)),
    );

    collector.set_alert_threshold(
        MetricType::ErrorCount,
        AlertThreshold::MaxCount {
            count: 5,
            window: chrono::Duration::minutes(5),
        },
    );

    // Simulate package operations
    println!("Simulating package operations...");
    for i in 1..=10 {
        collector.record_download(
            &format!("model-{}", i % 3 + 1),
            "1.0.0",
            Duration::from_secs(2 + i % 5),
        );
    }

    for i in 1..=5 {
        collector.record_upload(
            &format!("model-{}", i % 2 + 1),
            "2.0.0",
            1024 * 1024 * (50 + i * 10), // 50-100 MB
        );
    }

    // Record some accesses
    for i in 1..=20 {
        collector.record_access(
            &format!("model-{}", i % 3 + 1),
            "1.0.0",
            Some(&format!("user-{}", i % 5 + 1)),
        );
    }

    // Record resource usage
    collector.record_resource_usage(
        4 * 1024 * 1024 * 1024,   // 4 GB memory
        100 * 1024 * 1024 * 1024, // 100 GB storage
        45.5,                     // 45.5% CPU
    );

    // Get time-series data
    if let Some(ts) = collector.get_time_series(MetricType::Download) {
        println!("Download time-series statistics:");
        println!("  - Data points: {}", ts.stats.count);
        println!("  - Mean: {:.2}s", ts.stats.mean);
        println!("  - p95: {:.2}s", ts.stats.p95);
        println!("  - p99: {:.2}s", ts.stats.p99);
    }

    // Check alerts
    let alerts = collector.get_active_alerts();
    println!("Active alerts: {}", alerts.len());
    for alert in alerts {
        println!(
            "  - [{:?}] {} ({})",
            alert.severity, alert.message, alert.current_value
        );
    }

    // Generate analytics report
    let report = collector.generate_report();
    println!("Analytics report:");
    println!("  - Total downloads: {}", report.total_downloads);
    println!("  - Total uploads: {}", report.total_uploads);
    println!(
        "  - Total bandwidth: {:.2} MB",
        report.total_bandwidth_bytes as f64 / 1024.0 / 1024.0
    );
    println!(
        "  - Total storage: {:.2} GB",
        report.total_storage_bytes as f64 / 1024.0 / 1024.0 / 1024.0
    );
    println!(
        "  - Top packages: {:?}",
        report.top_packages.iter().take(3).collect::<Vec<_>>()
    );

    // Export to JSON
    if let Ok(json) = collector.export_to_json() {
        println!("JSON export: {} bytes", json.len());
    }
}

fn demonstrate_backup() {
    let config = BackupConfig {
        destination: PathBuf::from("/backups/torsh-packages"),
        strategy: BackupStrategy::Incremental,
        compression: true,
        encryption: false,
        retention: RetentionPolicy::KeepLast(7),
    };

    let mut manager = BackupManager::new(config);

    // Create full backup
    println!("Creating full backup...");
    let package_data = b"Package data for ResNet-50 model with weights and configuration";
    let backup_id = manager
        .create_backup("resnet50-medical", "1.0.0", package_data)
        .unwrap();
    println!("Backup created: {}", backup_id);

    // Create incremental backups
    println!("Creating incremental backups...");
    for i in 1..=3 {
        let incremental_data = format!("Incremental update {}", i).into_bytes();
        let inc_id = manager
            .create_backup("resnet50-medical", "1.0.0", &incremental_data)
            .unwrap();
        println!("Incremental backup {}: {}", i, inc_id);
    }

    // List backups
    let backups = manager.list_backups("resnet50-medical");
    println!("Total backups for resnet50-medical: {}", backups.len());

    // Verify backup integrity
    println!("Verifying backup integrity...");
    let verification = manager.verify_backup(&backup_id);
    println!("Verification result:");
    println!("  - Success: {}", verification.success);
    println!("  - Checksum valid: {}", verification.checksum_valid);
    println!("  - Readable: {}", verification.readable);

    // Get backup statistics
    let stats = manager.get_statistics();
    println!("Backup statistics:");
    println!("  - Total backups: {}", stats.total_backups);
    println!("  - Full backups: {}", stats.full_backups);
    println!("  - Incremental backups: {}", stats.incremental_backups);
    println!(
        "  - Total storage: {:.2} KB",
        stats.total_storage_bytes as f64 / 1024.0
    );
    if stats.compression_ratio > 0.0 {
        println!(
            "  - Compression ratio: {:.1}%",
            stats.compression_ratio * 100.0
        );
    }

    // Create recovery point
    println!("Creating recovery point...");
    let rp_id = manager
        .create_recovery_point(
            "resnet50-medical",
            "1.0.0",
            "Before production deployment".to_string(),
        )
        .unwrap();
    println!("Recovery point created: {}", rp_id);

    // Restore from backup
    println!("Restoring from backup...");
    match manager.restore_backup(&backup_id) {
        Ok(data) => {
            println!("Backup restored successfully: {} bytes", data.len());
        }
        Err(e) => {
            println!("Restore failed: {}", e);
        }
    }
}

fn demonstrate_replication() {
    let config = ReplicationConfig {
        consistency: ConsistencyLevel::Quorum,
        replication_factor: 3,
        auto_failover: true,
        sync_interval_secs: 60,
    };

    let mut manager = ReplicationManager::new(config);

    // Add replication nodes across regions
    println!("Adding replication nodes...");
    let nodes = vec![
        ("node-us-east", "us-east-1", 1, 1024 * 1024 * 1024 * 100),
        ("node-us-west", "us-west-2", 2, 1024 * 1024 * 1024 * 100),
        (
            "node-eu-central",
            "eu-central-1",
            1,
            1024 * 1024 * 1024 * 100,
        ),
        (
            "node-ap-southeast",
            "ap-southeast-1",
            3,
            1024 * 1024 * 1024 * 50,
        ),
    ];

    for (id, region, priority, capacity) in nodes {
        let node = ReplicationNode::new(
            id.to_string(),
            region.to_string(),
            format!("https://{}.packages.example.com", id),
            priority,
            capacity,
        );
        manager.add_node(node).unwrap();
        println!("  - Added node: {} ({})", id, region);
    }

    // Replicate a package
    println!("Replicating package across nodes...");
    let package_data = b"Model package data";
    match manager.replicate_package("resnet50-medical", "1.0.0", package_data) {
        Ok(_) => println!("Package replicated successfully"),
        Err(e) => println!("Replication failed: {}", e),
    }

    // Perform health check
    println!("Performing health checks...");
    manager.health_check().unwrap();

    // List nodes
    let nodes = manager.list_nodes();
    println!("Replication nodes:");
    for node in nodes {
        println!(
            "  - {}: {} (status: {:?}, priority: {})",
            node.id, node.region, node.status, node.priority
        );
    }

    // Get replication statistics
    let stats = manager.get_statistics();
    println!("Replication statistics:");
    println!("  - Total nodes: {}", stats.total_nodes);
    println!("  - Healthy nodes: {}", stats.healthy_nodes);
    println!("  - Total replicas: {}", stats.total_replicas);
    println!("  - Successful operations: {}", stats.successful_operations);
    println!("  - Failed operations: {}", stats.failed_operations);
    println!("  - Active conflicts: {}", stats.active_conflicts);
    println!(
        "  - Avg replication lag: {:.2}s",
        stats.avg_replication_lag_secs
    );
}
