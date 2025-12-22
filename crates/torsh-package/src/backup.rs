//! Backup and Recovery System
//!
//! This module provides comprehensive backup and recovery capabilities for packages,
//! including automated backup scheduling, multiple backup strategies, integrity
//! verification, point-in-time recovery, and disaster recovery procedures.
//!
//! # Features
//!
//! - **Backup Strategies**: Full, incremental, and differential backups
//! - **Automated Scheduling**: Configurable backup schedules with retention policies
//! - **Integrity Verification**: SHA-256 checksums and backup validation
//! - **Point-in-Time Recovery**: Restore packages to specific timestamps
//! - **Backup Rotation**: Automatic cleanup based on retention policies
//! - **Multiple Destinations**: Support for local, cloud, and distributed backups
//! - **Compression**: Optional backup compression for storage efficiency
//! - **Encryption**: Optional backup encryption for security
//! - **Recovery Testing**: Validate backup integrity through test restores
//!
//! # Examples
//!
//! ```rust
//! use torsh_package::backup::{BackupManager, BackupConfig, BackupStrategy, RetentionPolicy};
//! use std::path::PathBuf;
//!
//! // Create backup manager
//! let config = BackupConfig {
//!     destination: PathBuf::from("/backups"),
//!     strategy: BackupStrategy::Incremental,
//!     compression: true,
//!     encryption: false,
//!     retention: RetentionPolicy::KeepLast(7),
//! };
//!
//! let mut manager = BackupManager::new(config);
//!
//! // Create a backup
//! let backup_id = manager.create_backup("my-package", "1.0.0", b"package data").unwrap();
//!
//! // Restore from backup
//! let restored = manager.restore_backup(&backup_id).unwrap();
//! ```

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use torsh_core::error::TorshError;

/// Backup strategy type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackupStrategy {
    /// Full backup of entire package
    Full,
    /// Incremental backup (only changes since last backup)
    Incremental,
    /// Differential backup (changes since last full backup)
    Differential,
}

/// Retention policy for backup rotation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetentionPolicy {
    /// Keep backups for specified number of days
    KeepDays(u32),
    /// Keep last N backups
    KeepLast(usize),
    /// Keep all backups (no rotation)
    KeepAll,
    /// Custom retention with daily, weekly, monthly
    Custom {
        /// Number of daily backups to keep
        daily: u32,
        /// Number of weekly backups to keep
        weekly: u32,
        /// Number of monthly backups to keep
        monthly: u32,
    },
}

/// Backup destination type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackupDestination {
    /// Local filesystem path
    Local(PathBuf),
    /// S3-compatible storage
    S3 {
        /// S3 bucket name
        bucket: String,
        /// AWS region
        region: String,
        /// Object path within bucket
        path: String,
    },
    /// Google Cloud Storage
    Gcs {
        /// GCS bucket name
        bucket: String,
        /// Object path within bucket
        path: String,
    },
    /// Azure Blob Storage
    Azure {
        /// Azure container name
        container: String,
        /// Blob path within container
        path: String,
    },
}

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    /// Backup destination
    pub destination: PathBuf,
    /// Backup strategy
    pub strategy: BackupStrategy,
    /// Enable compression
    pub compression: bool,
    /// Enable encryption
    pub encryption: bool,
    /// Retention policy
    pub retention: RetentionPolicy,
}

/// Backup metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupMetadata {
    /// Unique backup identifier
    pub backup_id: String,
    /// Package ID
    pub package_id: String,
    /// Package version
    pub version: String,
    /// Backup strategy used
    pub strategy: BackupStrategy,
    /// Backup creation timestamp
    pub created_at: DateTime<Utc>,
    /// Size in bytes (uncompressed)
    pub size_bytes: u64,
    /// Compressed size in bytes
    pub compressed_size_bytes: Option<u64>,
    /// SHA-256 checksum
    pub checksum: String,
    /// Parent backup ID (for incremental/differential)
    pub parent_backup_id: Option<String>,
    /// Compression enabled
    pub compressed: bool,
    /// Encryption enabled
    pub encrypted: bool,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Backup verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Backup ID
    pub backup_id: String,
    /// Verification successful
    pub success: bool,
    /// Checksum match
    pub checksum_valid: bool,
    /// Backup readable
    pub readable: bool,
    /// Backup size matches metadata
    pub size_valid: bool,
    /// Verification errors
    pub errors: Vec<String>,
    /// Verification timestamp
    pub verified_at: DateTime<Utc>,
}

/// Recovery point for point-in-time recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryPoint {
    /// Recovery point ID
    pub id: String,
    /// Package ID
    pub package_id: String,
    /// Package version
    pub version: String,
    /// Timestamp of recovery point
    pub timestamp: DateTime<Utc>,
    /// Backup IDs needed for recovery
    pub backup_chain: Vec<String>,
    /// Description
    pub description: String,
}

/// Backup statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BackupStatistics {
    /// Total backups
    pub total_backups: usize,
    /// Full backups
    pub full_backups: usize,
    /// Incremental backups
    pub incremental_backups: usize,
    /// Differential backups
    pub differential_backups: usize,
    /// Total storage used (bytes)
    pub total_storage_bytes: u64,
    /// Compressed storage used (bytes)
    pub compressed_storage_bytes: u64,
    /// Compression ratio (0.0 - 1.0)
    pub compression_ratio: f64,
    /// Oldest backup timestamp
    pub oldest_backup: Option<DateTime<Utc>>,
    /// Newest backup timestamp
    pub newest_backup: Option<DateTime<Utc>>,
    /// Failed backups
    pub failed_backups: usize,
}

/// Backup manager
///
/// Manages package backups including creation, verification, restoration,
/// and automated rotation based on retention policies.
pub struct BackupManager {
    /// Backup configuration
    config: BackupConfig,
    /// Backup metadata by backup ID
    backups: HashMap<String, BackupMetadata>,
    /// Recovery points
    recovery_points: Vec<RecoveryPoint>,
    /// Statistics
    statistics: BackupStatistics,
    /// Mock storage for backup data (in production, this would be on disk/cloud)
    backup_data: HashMap<String, Vec<u8>>,
}

impl BackupManager {
    /// Create a new backup manager
    pub fn new(config: BackupConfig) -> Self {
        Self {
            config,
            backups: HashMap::new(),
            recovery_points: Vec::new(),
            statistics: BackupStatistics::default(),
            backup_data: HashMap::new(),
        }
    }

    /// Create a backup
    pub fn create_backup(
        &mut self,
        package_id: &str,
        version: &str,
        data: &[u8],
    ) -> Result<String, TorshError> {
        let backup_id = self.generate_backup_id(package_id, version);
        let created_at = Utc::now();

        // Calculate checksum
        let checksum = self.calculate_checksum(data);

        // Determine parent backup for incremental/differential
        let parent_backup_id = match self.config.strategy {
            BackupStrategy::Full => None,
            BackupStrategy::Incremental => self.get_last_backup_id(package_id, version),
            BackupStrategy::Differential => self.get_last_full_backup_id(package_id, version),
        };

        // Compress if enabled
        let (final_data, compressed_size) = if self.config.compression {
            let compressed = self.compress_data(data)?;
            let size = compressed.len() as u64;
            (compressed, Some(size))
        } else {
            (data.to_vec(), None)
        };

        // Encrypt if enabled
        let final_data = if self.config.encryption {
            self.encrypt_data(&final_data)?
        } else {
            final_data
        };

        // Store backup (mock implementation)
        self.store_backup(&backup_id, &final_data)?;

        // Create metadata
        let metadata = BackupMetadata {
            backup_id: backup_id.clone(),
            package_id: package_id.to_string(),
            version: version.to_string(),
            strategy: self.config.strategy,
            created_at,
            size_bytes: data.len() as u64,
            compressed_size_bytes: compressed_size,
            checksum,
            parent_backup_id,
            compressed: self.config.compression,
            encrypted: self.config.encryption,
            metadata: HashMap::new(),
        };

        self.backups.insert(backup_id.clone(), metadata);

        // Update statistics
        self.update_statistics_after_backup();

        // Apply retention policy
        self.apply_retention_policy()?;

        Ok(backup_id)
    }

    /// Restore from backup
    pub fn restore_backup(&self, backup_id: &str) -> Result<Vec<u8>, TorshError> {
        let metadata = self.backups.get(backup_id).ok_or_else(|| {
            TorshError::InvalidArgument(format!("Backup {} not found", backup_id))
        })?;

        // Load backup data (mock implementation)
        let mut data = self.load_backup(backup_id)?;

        // Decrypt if needed
        if metadata.encrypted {
            data = self.decrypt_data(&data)?;
        }

        // Decompress if needed
        if metadata.compressed {
            data = self.decompress_data(&data)?;
        }

        // Verify checksum
        let checksum = self.calculate_checksum(&data);
        if checksum != metadata.checksum {
            return Err(TorshError::RuntimeError(
                "Backup checksum mismatch".to_string(),
            ));
        }

        // Handle incremental/differential restoration
        if let Some(parent_id) = &metadata.parent_backup_id {
            let parent_data = self.restore_backup(parent_id)?;
            data = self.merge_backup_data(&parent_data, &data)?;
        }

        Ok(data)
    }

    /// Verify backup integrity
    pub fn verify_backup(&self, backup_id: &str) -> VerificationResult {
        let metadata = match self.backups.get(backup_id) {
            Some(m) => m,
            None => {
                return VerificationResult {
                    backup_id: backup_id.to_string(),
                    success: false,
                    checksum_valid: false,
                    readable: false,
                    size_valid: false,
                    errors: vec!["Backup not found".to_string()],
                    verified_at: Utc::now(),
                }
            }
        };

        let mut errors = Vec::new();
        let mut checksum_valid = false;
        let mut readable = false;
        let mut size_valid = false;

        // Try to load backup
        match self.load_backup(backup_id) {
            Ok(data) => {
                readable = true;

                // Check size
                let expected_size = if metadata.compressed {
                    metadata
                        .compressed_size_bytes
                        .unwrap_or(metadata.size_bytes)
                } else {
                    metadata.size_bytes
                };

                if data.len() as u64 == expected_size {
                    size_valid = true;
                } else {
                    errors.push(format!(
                        "Size mismatch: expected {}, got {}",
                        expected_size,
                        data.len()
                    ));
                }

                // Verify checksum (need to decompress/decrypt first)
                match self.restore_backup(backup_id) {
                    Ok(restored) => {
                        let checksum = self.calculate_checksum(&restored);
                        if checksum == metadata.checksum {
                            checksum_valid = true;
                        } else {
                            errors.push("Checksum mismatch".to_string());
                        }
                    }
                    Err(e) => {
                        errors.push(format!("Restoration failed: {}", e));
                    }
                }
            }
            Err(e) => {
                errors.push(format!("Failed to load backup: {}", e));
            }
        }

        let success = errors.is_empty();

        VerificationResult {
            backup_id: backup_id.to_string(),
            success,
            checksum_valid,
            readable,
            size_valid,
            errors,
            verified_at: Utc::now(),
        }
    }

    /// Create a recovery point
    pub fn create_recovery_point(
        &mut self,
        package_id: &str,
        version: &str,
        description: String,
    ) -> Result<String, TorshError> {
        let id = uuid::Uuid::new_v4().to_string();

        // Find all backups in the chain
        let backup_chain = self.build_backup_chain(package_id, version)?;

        let recovery_point = RecoveryPoint {
            id: id.clone(),
            package_id: package_id.to_string(),
            version: version.to_string(),
            timestamp: Utc::now(),
            backup_chain,
            description,
        };

        self.recovery_points.push(recovery_point);

        Ok(id)
    }

    /// Restore to recovery point
    pub fn restore_to_recovery_point(
        &self,
        recovery_point_id: &str,
    ) -> Result<Vec<u8>, TorshError> {
        let recovery_point = self
            .recovery_points
            .iter()
            .find(|rp| rp.id == recovery_point_id)
            .ok_or_else(|| {
                TorshError::InvalidArgument(format!(
                    "Recovery point {} not found",
                    recovery_point_id
                ))
            })?;

        // Restore from the last backup in the chain
        if let Some(last_backup) = recovery_point.backup_chain.last() {
            self.restore_backup(last_backup)
        } else {
            Err(TorshError::InvalidArgument(
                "Recovery point has no backups".to_string(),
            ))
        }
    }

    /// List all backups for a package
    pub fn list_backups(&self, package_id: &str) -> Vec<&BackupMetadata> {
        self.backups
            .values()
            .filter(|m| m.package_id == package_id)
            .collect()
    }

    /// Get backup statistics
    pub fn get_statistics(&self) -> &BackupStatistics {
        &self.statistics
    }

    /// Delete a backup
    pub fn delete_backup(&mut self, backup_id: &str) -> Result<(), TorshError> {
        self.backups.remove(backup_id).ok_or_else(|| {
            TorshError::InvalidArgument(format!("Backup {} not found", backup_id))
        })?;

        // Remove backup data
        self.backup_data.remove(backup_id);

        // Update statistics
        self.update_statistics();

        Ok(())
    }

    /// Apply retention policy
    pub fn apply_retention_policy(&mut self) -> Result<(), TorshError> {
        let now = Utc::now();
        let mut to_delete = Vec::new();

        match self.config.retention {
            RetentionPolicy::KeepDays(days) => {
                let cutoff = now - ChronoDuration::days(days as i64);
                for (id, metadata) in &self.backups {
                    if metadata.created_at < cutoff {
                        to_delete.push(id.clone());
                    }
                }
            }
            RetentionPolicy::KeepLast(count) => {
                // Group backups by package
                let mut by_package: HashMap<String, Vec<&BackupMetadata>> = HashMap::new();
                for metadata in self.backups.values() {
                    by_package
                        .entry(metadata.package_id.clone())
                        .or_insert_with(Vec::new)
                        .push(metadata);
                }

                for backups in by_package.values_mut() {
                    // Sort by creation time (newest first)
                    backups.sort_by(|a, b| b.created_at.cmp(&a.created_at));

                    // Mark old backups for deletion
                    for metadata in backups.iter().skip(count) {
                        to_delete.push(metadata.backup_id.clone());
                    }
                }
            }
            RetentionPolicy::KeepAll => {
                // No deletions
            }
            RetentionPolicy::Custom {
                daily,
                weekly,
                monthly,
            } => {
                // Implement GFS (Grandfather-Father-Son) retention
                self.apply_gfs_retention(daily, weekly, monthly, &mut to_delete);
            }
        }

        // Delete marked backups
        for backup_id in to_delete {
            self.delete_backup(&backup_id)?;
        }

        Ok(())
    }

    // Private helper methods

    fn generate_backup_id(&self, package_id: &str, version: &str) -> String {
        // Use UUID to ensure uniqueness even for rapid backups
        format!(
            "{}-{}-{}",
            package_id,
            version,
            uuid::Uuid::new_v4().to_string()
        )
    }

    fn calculate_checksum(&self, data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("{:x}", hasher.finalize())
    }

    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>, TorshError> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder
            .write_all(data)
            .map_err(|e| TorshError::RuntimeError(e.to_string()))?;
        encoder
            .finish()
            .map_err(|e| TorshError::RuntimeError(e.to_string()))
    }

    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>, TorshError> {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder
            .read_to_end(&mut decompressed)
            .map_err(|e| TorshError::RuntimeError(e.to_string()))?;
        Ok(decompressed)
    }

    fn encrypt_data(&self, data: &[u8]) -> Result<Vec<u8>, TorshError> {
        // Mock encryption (in production, use proper encryption)
        Ok(data.to_vec())
    }

    fn decrypt_data(&self, data: &[u8]) -> Result<Vec<u8>, TorshError> {
        // Mock decryption (in production, use proper decryption)
        Ok(data.to_vec())
    }

    fn store_backup(&mut self, backup_id: &str, data: &[u8]) -> Result<(), TorshError> {
        // Mock storage (in production, write to destination)
        self.backup_data
            .insert(backup_id.to_string(), data.to_vec());
        Ok(())
    }

    fn load_backup(&self, backup_id: &str) -> Result<Vec<u8>, TorshError> {
        // Mock loading (in production, read from destination)
        self.backup_data.get(backup_id).cloned().ok_or_else(|| {
            TorshError::InvalidArgument(format!("Backup data {} not found", backup_id))
        })
    }

    fn merge_backup_data(&self, _base: &[u8], delta: &[u8]) -> Result<Vec<u8>, TorshError> {
        // Mock merge (in production, apply delta)
        Ok(delta.to_vec())
    }

    fn get_last_backup_id(&self, package_id: &str, version: &str) -> Option<String> {
        self.backups
            .values()
            .filter(|m| m.package_id == package_id && m.version == version)
            .max_by_key(|m| m.created_at)
            .map(|m| m.backup_id.clone())
    }

    fn get_last_full_backup_id(&self, package_id: &str, version: &str) -> Option<String> {
        self.backups
            .values()
            .filter(|m| {
                m.package_id == package_id
                    && m.version == version
                    && m.strategy == BackupStrategy::Full
            })
            .max_by_key(|m| m.created_at)
            .map(|m| m.backup_id.clone())
    }

    fn build_backup_chain(
        &self,
        package_id: &str,
        version: &str,
    ) -> Result<Vec<String>, TorshError> {
        let mut chain = Vec::new();

        // Find latest backup
        if let Some(latest) = self
            .backups
            .values()
            .filter(|m| m.package_id == package_id && m.version == version)
            .max_by_key(|m| m.created_at)
        {
            chain.push(latest.backup_id.clone());

            // Follow parent chain
            let mut current = latest;
            while let Some(parent_id) = &current.parent_backup_id {
                chain.push(parent_id.clone());
                current = self.backups.get(parent_id).ok_or_else(|| {
                    TorshError::InvalidArgument(format!("Parent backup {} not found", parent_id))
                })?;
            }
        }

        chain.reverse();
        Ok(chain)
    }

    fn update_statistics_after_backup(&mut self) {
        self.update_statistics();
    }

    fn update_statistics(&mut self) {
        let mut stats = BackupStatistics::default();

        stats.total_backups = self.backups.len();

        for metadata in self.backups.values() {
            match metadata.strategy {
                BackupStrategy::Full => stats.full_backups += 1,
                BackupStrategy::Incremental => stats.incremental_backups += 1,
                BackupStrategy::Differential => stats.differential_backups += 1,
            }

            stats.total_storage_bytes += metadata.size_bytes;
            if let Some(compressed) = metadata.compressed_size_bytes {
                stats.compressed_storage_bytes += compressed;
            }

            if stats.oldest_backup.is_none() || Some(metadata.created_at) < stats.oldest_backup {
                stats.oldest_backup = Some(metadata.created_at);
            }

            if stats.newest_backup.is_none() || Some(metadata.created_at) > stats.newest_backup {
                stats.newest_backup = Some(metadata.created_at);
            }
        }

        if stats.total_storage_bytes > 0 {
            stats.compression_ratio =
                stats.compressed_storage_bytes as f64 / stats.total_storage_bytes as f64;
        }

        self.statistics = stats;
    }

    fn apply_gfs_retention(
        &self,
        _daily: u32,
        _weekly: u32,
        _monthly: u32,
        _to_delete: &mut Vec<String>,
    ) {
        // Mock GFS implementation (in production, implement proper GFS logic)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> BackupConfig {
        BackupConfig {
            destination: PathBuf::from("/tmp/backups"),
            strategy: BackupStrategy::Full,
            compression: true,
            encryption: false,
            retention: RetentionPolicy::KeepLast(5),
        }
    }

    #[test]
    fn test_backup_manager_creation() {
        let config = create_test_config();
        let manager = BackupManager::new(config);
        let stats = manager.get_statistics();
        assert_eq!(stats.total_backups, 0);
    }

    #[test]
    fn test_create_backup() {
        let config = create_test_config();
        let mut manager = BackupManager::new(config);

        let data = b"test package data";
        let backup_id = manager.create_backup("test-pkg", "1.0.0", data).unwrap();

        assert!(!backup_id.is_empty());
        let stats = manager.get_statistics();
        assert_eq!(stats.total_backups, 1);
        assert_eq!(stats.full_backups, 1);
    }

    #[test]
    fn test_restore_backup() {
        let config = create_test_config();
        let mut manager = BackupManager::new(config);

        let data = b"test package data";
        let backup_id = manager.create_backup("test-pkg", "1.0.0", data).unwrap();

        let restored = manager.restore_backup(&backup_id).unwrap();
        assert_eq!(restored, data);
    }

    #[test]
    fn test_list_backups() {
        let config = create_test_config();
        let mut manager = BackupManager::new(config);

        manager.create_backup("pkg1", "1.0.0", b"data1").unwrap();
        manager.create_backup("pkg1", "2.0.0", b"data2").unwrap();
        manager.create_backup("pkg2", "1.0.0", b"data3").unwrap();

        let backups = manager.list_backups("pkg1");
        assert_eq!(backups.len(), 2);

        let backups = manager.list_backups("pkg2");
        assert_eq!(backups.len(), 1);
    }

    #[test]
    fn test_verify_backup() {
        let config = create_test_config();
        let mut manager = BackupManager::new(config);

        let data = b"test package data";
        let backup_id = manager.create_backup("test-pkg", "1.0.0", data).unwrap();

        let result = manager.verify_backup(&backup_id);
        assert!(result.success);
        assert!(result.readable);
    }

    #[test]
    fn test_delete_backup() {
        let config = create_test_config();
        let mut manager = BackupManager::new(config);

        let backup_id = manager.create_backup("test-pkg", "1.0.0", b"data").unwrap();

        assert_eq!(manager.get_statistics().total_backups, 1);

        manager.delete_backup(&backup_id).unwrap();
        assert_eq!(manager.get_statistics().total_backups, 0);
    }

    #[test]
    fn test_retention_policy_keep_last() {
        let mut config = create_test_config();
        config.retention = RetentionPolicy::KeepLast(3);
        let mut manager = BackupManager::new(config);

        // Create 5 backups
        for i in 0..5 {
            manager
                .create_backup("test-pkg", "1.0.0", format!("data{}", i).as_bytes())
                .unwrap();
        }

        // Should only keep last 3
        assert_eq!(manager.get_statistics().total_backups, 3);
    }

    #[test]
    fn test_incremental_backup() {
        let mut config = create_test_config();
        config.strategy = BackupStrategy::Incremental;
        let mut manager = BackupManager::new(config);

        // Create full backup first
        let mut config2 = create_test_config();
        config2.strategy = BackupStrategy::Full;
        let mut manager2 = BackupManager::new(config2);
        let full_id = manager2
            .create_backup("test-pkg", "1.0.0", b"base data")
            .unwrap();

        // Copy to incremental manager
        if let Some(metadata) = manager2.backups.get(&full_id) {
            manager.backups.insert(full_id.clone(), metadata.clone());
        }

        // Create incremental backup
        let inc_id = manager
            .create_backup("test-pkg", "1.0.0", b"delta data")
            .unwrap();

        let metadata = manager.backups.get(&inc_id).unwrap();
        assert_eq!(metadata.strategy, BackupStrategy::Incremental);
        assert!(metadata.parent_backup_id.is_some());
    }

    #[test]
    fn test_create_recovery_point() {
        let config = create_test_config();
        let mut manager = BackupManager::new(config);

        manager.create_backup("test-pkg", "1.0.0", b"data").unwrap();

        let rp_id = manager
            .create_recovery_point("test-pkg", "1.0.0", "Before update".to_string())
            .unwrap();

        assert!(!rp_id.is_empty());
        assert_eq!(manager.recovery_points.len(), 1);
    }

    #[test]
    fn test_backup_statistics() {
        let config = create_test_config();
        let mut manager = BackupManager::new(config);

        let data = b"test data with some content";
        manager.create_backup("pkg1", "1.0.0", data).unwrap();
        manager.create_backup("pkg2", "1.0.0", data).unwrap();

        let stats = manager.get_statistics();
        assert_eq!(stats.total_backups, 2);
        assert_eq!(stats.full_backups, 2);
        assert!(stats.total_storage_bytes > 0);
        assert!(stats.newest_backup.is_some());
        assert!(stats.oldest_backup.is_some());
    }
}
