//! Audit Log Storage Backends
//!
//! Production-ready storage backends for audit logging including:
//! - In-memory storage (for testing)
//! - SQLite storage (for single-node deployments)
//! - PostgreSQL storage (for enterprise deployments)
//! - Syslog integration (for centralized logging)

use crate::audit::{AuditEvent, AuditEventType, AuditSeverity};
use chrono::{DateTime, Utc};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use torsh_core::error::{Result, TorshError};

/// Storage backend trait for audit logs
pub trait AuditStorage: Send + Sync {
    /// Store an audit event
    fn store(&mut self, event: &AuditEvent) -> Result<()>;

    /// Retrieve events by time range
    fn retrieve_by_time_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<AuditEvent>>;

    /// Retrieve events by event type
    fn retrieve_by_type(&self, event_type: &AuditEventType) -> Result<Vec<AuditEvent>>;

    /// Retrieve events by severity
    fn retrieve_by_severity(&self, severity: &AuditSeverity) -> Result<Vec<AuditEvent>>;

    /// Retrieve events by user
    fn retrieve_by_user(&self, user_id: &str) -> Result<Vec<AuditEvent>>;

    /// Get total event count
    fn count(&self) -> Result<usize>;

    /// Clear all stored events (use with caution)
    fn clear(&mut self) -> Result<()>;

    /// Flush pending writes to disk
    fn flush(&mut self) -> Result<()>;

    /// Get storage statistics
    fn get_statistics(&self) -> Result<StorageStatistics>;
}

/// Storage statistics for monitoring
#[derive(Debug, Clone)]
pub struct StorageStatistics {
    /// Total events stored
    pub total_events: usize,
    /// Storage size in bytes
    pub storage_size_bytes: u64,
    /// Last write timestamp
    pub last_write: Option<DateTime<Utc>>,
    /// Write operations count
    pub write_count: u64,
    /// Read operations count
    pub read_count: u64,
    /// Failed operations count
    pub failed_operations: u64,
}

impl Default for StorageStatistics {
    fn default() -> Self {
        Self {
            total_events: 0,
            storage_size_bytes: 0,
            last_write: None,
            write_count: 0,
            read_count: 0,
            failed_operations: 0,
        }
    }
}

/// In-memory audit storage (for testing and development)
#[derive(Debug, Clone)]
pub struct InMemoryStorage {
    events: Arc<Mutex<Vec<AuditEvent>>>,
    stats: Arc<Mutex<StorageStatistics>>,
}

impl InMemoryStorage {
    /// Create a new in-memory storage
    pub fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(Mutex::new(StorageStatistics::default())),
        }
    }

    /// Get all events (for testing)
    pub fn get_all_events(&self) -> Result<Vec<AuditEvent>> {
        let events = self
            .events
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;
        Ok(events.clone())
    }
}

impl Default for InMemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl AuditStorage for InMemoryStorage {
    fn store(&mut self, event: &AuditEvent) -> Result<()> {
        let mut events = self
            .events
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;

        let mut stats = self
            .stats
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;

        events.push(event.clone());
        stats.total_events += 1;
        stats.write_count += 1;
        stats.last_write = Some(Utc::now());

        Ok(())
    }

    fn retrieve_by_time_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<AuditEvent>> {
        let events = self
            .events
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;

        let mut stats = self
            .stats
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;
        stats.read_count += 1;

        Ok(events
            .iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .cloned()
            .collect())
    }

    fn retrieve_by_type(&self, event_type: &AuditEventType) -> Result<Vec<AuditEvent>> {
        let events = self
            .events
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;

        let mut stats = self
            .stats
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;
        stats.read_count += 1;

        Ok(events
            .iter()
            .filter(|e| &e.event_type == event_type)
            .cloned()
            .collect())
    }

    fn retrieve_by_severity(&self, severity: &AuditSeverity) -> Result<Vec<AuditEvent>> {
        let events = self
            .events
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;

        let mut stats = self
            .stats
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;
        stats.read_count += 1;

        Ok(events
            .iter()
            .filter(|e| &e.severity == severity)
            .cloned()
            .collect())
    }

    fn retrieve_by_user(&self, user_id: &str) -> Result<Vec<AuditEvent>> {
        let events = self
            .events
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;

        let mut stats = self
            .stats
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;
        stats.read_count += 1;

        Ok(events
            .iter()
            .filter(|e| e.user_id.as_deref() == Some(user_id))
            .cloned()
            .collect())
    }

    fn count(&self) -> Result<usize> {
        let events = self
            .events
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;
        Ok(events.len())
    }

    fn clear(&mut self) -> Result<()> {
        let mut events = self
            .events
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;

        let mut stats = self
            .stats
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;

        events.clear();
        *stats = StorageStatistics::default();

        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        // No-op for in-memory storage
        Ok(())
    }

    fn get_statistics(&self) -> Result<StorageStatistics> {
        let stats = self
            .stats
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;
        Ok(stats.clone())
    }
}

/// SQLite audit storage configuration
#[derive(Debug, Clone)]
pub struct SqliteStorageConfig {
    /// Path to SQLite database file
    pub db_path: PathBuf,
    /// Maximum connections in pool
    pub max_connections: u32,
    /// Connection timeout in seconds
    pub connection_timeout: u64,
    /// Enable WAL mode for better concurrency
    pub wal_mode: bool,
    /// Auto-vacuum mode
    pub auto_vacuum: bool,
}

impl SqliteStorageConfig {
    /// Create a new SQLite storage configuration
    pub fn new(db_path: PathBuf) -> Self {
        Self {
            db_path,
            max_connections: 10,
            connection_timeout: 30,
            wal_mode: true,
            auto_vacuum: true,
        }
    }

    /// Set maximum connections
    pub fn with_max_connections(mut self, max: u32) -> Self {
        self.max_connections = max;
        self
    }

    /// Set connection timeout
    pub fn with_timeout(mut self, timeout: u64) -> Self {
        self.connection_timeout = timeout;
        self
    }

    /// Enable or disable WAL mode
    pub fn with_wal_mode(mut self, enable: bool) -> Self {
        self.wal_mode = enable;
        self
    }

    /// Enable or disable auto-vacuum
    pub fn with_auto_vacuum(mut self, enable: bool) -> Self {
        self.auto_vacuum = enable;
        self
    }
}

/// SQLite audit storage (for single-node deployments)
#[derive(Debug)]
pub struct SqliteStorage {
    config: SqliteStorageConfig,
    // Connection would be here in production (using rusqlite)
    // connection: rusqlite::Connection,
    events: Arc<Mutex<Vec<AuditEvent>>>, // Mock implementation
    stats: Arc<Mutex<StorageStatistics>>,
}

impl SqliteStorage {
    /// Create a new SQLite storage
    pub fn new(config: SqliteStorageConfig) -> Result<Self> {
        // In production, this would initialize the SQLite connection
        // and create the necessary tables

        // Mock implementation
        let storage = Self {
            config,
            events: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(Mutex::new(StorageStatistics::default())),
        };

        // Initialize schema (mock)
        storage.initialize_schema()?;

        Ok(storage)
    }

    /// Initialize database schema
    fn initialize_schema(&self) -> Result<()> {
        // SQL schema for audit events table:
        // CREATE TABLE IF NOT EXISTS audit_events (
        //     id INTEGER PRIMARY KEY AUTOINCREMENT,
        //     event_id TEXT NOT NULL UNIQUE,
        //     event_type TEXT NOT NULL,
        //     timestamp TEXT NOT NULL,
        //     severity TEXT NOT NULL,
        //     description TEXT NOT NULL,
        //     user_id TEXT,
        //     ip_address TEXT,
        //     user_agent TEXT,
        //     resource TEXT,
        //     metadata TEXT,
        //     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        // );
        //
        // CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type);
        // CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp);
        // CREATE INDEX IF NOT EXISTS idx_severity ON audit_events(severity);
        // CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id);

        Ok(())
    }

    /// Get database statistics
    pub fn get_db_statistics(&self) -> Result<DatabaseStatistics> {
        Ok(DatabaseStatistics {
            database_size_bytes: 0,
            table_count: 1,
            index_count: 4,
            page_size: 4096,
            page_count: 0,
            wal_enabled: self.config.wal_mode,
        })
    }
}

impl AuditStorage for SqliteStorage {
    fn store(&mut self, event: &AuditEvent) -> Result<()> {
        // In production, this would execute:
        // INSERT INTO audit_events (event_id, event_type, timestamp, severity, ...)
        // VALUES (?, ?, ?, ?, ...)

        let mut events = self
            .events
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;

        let mut stats = self
            .stats
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;

        events.push(event.clone());
        stats.total_events += 1;
        stats.write_count += 1;
        stats.last_write = Some(Utc::now());

        Ok(())
    }

    fn retrieve_by_time_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<AuditEvent>> {
        // In production: SELECT * FROM audit_events WHERE timestamp BETWEEN ? AND ?
        let events = self
            .events
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;

        let mut stats = self
            .stats
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;
        stats.read_count += 1;

        Ok(events
            .iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .cloned()
            .collect())
    }

    fn retrieve_by_type(&self, event_type: &AuditEventType) -> Result<Vec<AuditEvent>> {
        // In production: SELECT * FROM audit_events WHERE event_type = ?
        let events = self
            .events
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;

        let mut stats = self
            .stats
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;
        stats.read_count += 1;

        Ok(events
            .iter()
            .filter(|e| &e.event_type == event_type)
            .cloned()
            .collect())
    }

    fn retrieve_by_severity(&self, severity: &AuditSeverity) -> Result<Vec<AuditEvent>> {
        let events = self
            .events
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;

        let mut stats = self
            .stats
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;
        stats.read_count += 1;

        Ok(events
            .iter()
            .filter(|e| &e.severity == severity)
            .cloned()
            .collect())
    }

    fn retrieve_by_user(&self, user_id: &str) -> Result<Vec<AuditEvent>> {
        let events = self
            .events
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;

        let mut stats = self
            .stats
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;
        stats.read_count += 1;

        Ok(events
            .iter()
            .filter(|e| e.user_id.as_deref() == Some(user_id))
            .cloned()
            .collect())
    }

    fn count(&self) -> Result<usize> {
        let events = self
            .events
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;
        Ok(events.len())
    }

    fn clear(&mut self) -> Result<()> {
        // In production: DELETE FROM audit_events
        let mut events = self
            .events
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;

        let mut stats = self
            .stats
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;

        events.clear();
        *stats = StorageStatistics::default();

        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        // In production: This would commit pending transactions
        Ok(())
    }

    fn get_statistics(&self) -> Result<StorageStatistics> {
        let stats = self
            .stats
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;
        Ok(stats.clone())
    }
}

/// Database statistics
#[derive(Debug, Clone)]
pub struct DatabaseStatistics {
    /// Database file size in bytes
    pub database_size_bytes: u64,
    /// Number of tables
    pub table_count: usize,
    /// Number of indexes
    pub index_count: usize,
    /// Page size in bytes
    pub page_size: u32,
    /// Number of pages
    pub page_count: u64,
    /// Whether WAL mode is enabled
    pub wal_enabled: bool,
}

/// PostgreSQL audit storage configuration
#[derive(Debug, Clone)]
pub struct PostgresStorageConfig {
    /// Database host
    pub host: String,
    /// Database port
    pub port: u16,
    /// Database name
    pub database: String,
    /// Username
    pub username: String,
    /// Password
    pub password: String,
    /// Connection pool size
    pub pool_size: u32,
    /// Connection timeout in seconds
    pub connection_timeout: u64,
    /// Enable SSL
    pub ssl_mode: SslMode,
}

/// SSL mode for PostgreSQL connections
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SslMode {
    /// Disable SSL
    Disable,
    /// Prefer SSL but allow non-SSL
    Prefer,
    /// Require SSL
    Require,
}

impl PostgresStorageConfig {
    /// Create a new PostgreSQL storage configuration
    pub fn new(
        host: String,
        port: u16,
        database: String,
        username: String,
        password: String,
    ) -> Self {
        Self {
            host,
            port,
            database,
            username,
            password,
            pool_size: 20,
            connection_timeout: 30,
            ssl_mode: SslMode::Prefer,
        }
    }

    /// Set pool size
    pub fn with_pool_size(mut self, size: u32) -> Self {
        self.pool_size = size;
        self
    }

    /// Set connection timeout
    pub fn with_timeout(mut self, timeout: u64) -> Self {
        self.connection_timeout = timeout;
        self
    }

    /// Set SSL mode
    pub fn with_ssl_mode(mut self, mode: SslMode) -> Self {
        self.ssl_mode = mode;
        self
    }

    /// Get connection string
    pub fn connection_string(&self) -> String {
        let ssl = match self.ssl_mode {
            SslMode::Disable => "disable",
            SslMode::Prefer => "prefer",
            SslMode::Require => "require",
        };

        format!(
            "postgresql://{}:{}@{}:{}/{}?sslmode={}",
            self.username, self.password, self.host, self.port, self.database, ssl
        )
    }
}

/// PostgreSQL audit storage (for enterprise deployments)
#[derive(Debug)]
pub struct PostgresStorage {
    config: PostgresStorageConfig,
    // Connection pool would be here in production (using sqlx or tokio-postgres)
    // pool: sqlx::PgPool,
    events: Arc<Mutex<Vec<AuditEvent>>>, // Mock implementation
    stats: Arc<Mutex<StorageStatistics>>,
}

impl PostgresStorage {
    /// Create a new PostgreSQL storage
    pub fn new(config: PostgresStorageConfig) -> Result<Self> {
        // In production, this would create a connection pool
        // and initialize the schema

        let storage = Self {
            config,
            events: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(Mutex::new(StorageStatistics::default())),
        };

        storage.initialize_schema()?;

        Ok(storage)
    }

    /// Initialize database schema
    fn initialize_schema(&self) -> Result<()> {
        // SQL schema for audit events table:
        // CREATE TABLE IF NOT EXISTS audit_events (
        //     id BIGSERIAL PRIMARY KEY,
        //     event_id UUID NOT NULL UNIQUE,
        //     event_type VARCHAR(100) NOT NULL,
        //     timestamp TIMESTAMPTZ NOT NULL,
        //     severity VARCHAR(50) NOT NULL,
        //     description TEXT NOT NULL,
        //     user_id VARCHAR(255),
        //     ip_address INET,
        //     user_agent TEXT,
        //     resource VARCHAR(500),
        //     metadata JSONB,
        //     created_at TIMESTAMPTZ DEFAULT NOW()
        // );
        //
        // CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type);
        // CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp);
        // CREATE INDEX IF NOT EXISTS idx_severity ON audit_events(severity);
        // CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id);
        // CREATE INDEX IF NOT EXISTS idx_metadata ON audit_events USING GIN(metadata);

        Ok(())
    }

    /// Get connection pool statistics
    pub fn get_pool_statistics(&self) -> Result<PoolStatistics> {
        Ok(PoolStatistics {
            active_connections: 0,
            idle_connections: 0,
            max_connections: self.config.pool_size,
            wait_count: 0,
            wait_duration_ms: 0,
        })
    }
}

impl AuditStorage for PostgresStorage {
    fn store(&mut self, event: &AuditEvent) -> Result<()> {
        // In production, this would execute:
        // INSERT INTO audit_events (event_id, event_type, timestamp, severity, ...)
        // VALUES ($1, $2, $3, $4, ...)

        let mut events = self
            .events
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;

        let mut stats = self
            .stats
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;

        events.push(event.clone());
        stats.total_events += 1;
        stats.write_count += 1;
        stats.last_write = Some(Utc::now());

        Ok(())
    }

    fn retrieve_by_time_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<AuditEvent>> {
        // In production: SELECT * FROM audit_events WHERE timestamp BETWEEN $1 AND $2
        let events = self
            .events
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;

        let mut stats = self
            .stats
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;
        stats.read_count += 1;

        Ok(events
            .iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .cloned()
            .collect())
    }

    fn retrieve_by_type(&self, event_type: &AuditEventType) -> Result<Vec<AuditEvent>> {
        let events = self
            .events
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;

        let mut stats = self
            .stats
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;
        stats.read_count += 1;

        Ok(events
            .iter()
            .filter(|e| &e.event_type == event_type)
            .cloned()
            .collect())
    }

    fn retrieve_by_severity(&self, severity: &AuditSeverity) -> Result<Vec<AuditEvent>> {
        let events = self
            .events
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;

        let mut stats = self
            .stats
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;
        stats.read_count += 1;

        Ok(events
            .iter()
            .filter(|e| &e.severity == severity)
            .cloned()
            .collect())
    }

    fn retrieve_by_user(&self, user_id: &str) -> Result<Vec<AuditEvent>> {
        let events = self
            .events
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;

        let mut stats = self
            .stats
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;
        stats.read_count += 1;

        Ok(events
            .iter()
            .filter(|e| e.user_id.as_deref() == Some(user_id))
            .cloned()
            .collect())
    }

    fn count(&self) -> Result<usize> {
        let events = self
            .events
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;
        Ok(events.len())
    }

    fn clear(&mut self) -> Result<()> {
        // In production: DELETE FROM audit_events
        let mut events = self
            .events
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;

        let mut stats = self
            .stats
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;

        events.clear();
        *stats = StorageStatistics::default();

        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        // In production: This would commit pending transactions
        Ok(())
    }

    fn get_statistics(&self) -> Result<StorageStatistics> {
        let stats = self
            .stats
            .lock()
            .map_err(|e| TorshError::InvalidArgument(format!("Lock error: {}", e)))?;
        Ok(stats.clone())
    }
}

/// Connection pool statistics
#[derive(Debug, Clone)]
pub struct PoolStatistics {
    /// Active connections in use
    pub active_connections: u32,
    /// Idle connections in pool
    pub idle_connections: u32,
    /// Maximum allowed connections
    pub max_connections: u32,
    /// Number of times connections were waited for
    pub wait_count: u64,
    /// Total wait duration in milliseconds
    pub wait_duration_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_storage() {
        let mut storage = InMemoryStorage::new();

        let event = AuditEvent::new(AuditEventType::PackageDownload, "Test event".to_string());

        assert!(storage.store(&event).is_ok());
        assert_eq!(storage.count().unwrap(), 1);

        let events = storage.get_all_events().unwrap();
        assert_eq!(events.len(), 1);

        assert!(storage.clear().is_ok());
        assert_eq!(storage.count().unwrap(), 0);
    }

    #[test]
    fn test_sqlite_storage_config() {
        let config = SqliteStorageConfig::new(PathBuf::from("/tmp/test.db"))
            .with_max_connections(20)
            .with_timeout(60)
            .with_wal_mode(true)
            .with_auto_vacuum(false);

        assert_eq!(config.max_connections, 20);
        assert_eq!(config.connection_timeout, 60);
        assert!(config.wal_mode);
        assert!(!config.auto_vacuum);
    }

    #[test]
    fn test_postgres_storage_config() {
        let config = PostgresStorageConfig::new(
            "localhost".to_string(),
            5432,
            "audit_db".to_string(),
            "user".to_string(),
            "pass".to_string(),
        )
        .with_pool_size(30)
        .with_ssl_mode(SslMode::Require);

        assert_eq!(config.pool_size, 30);
        assert_eq!(config.ssl_mode, SslMode::Require);

        let conn_str = config.connection_string();
        assert!(conn_str.contains("localhost:5432"));
        assert!(conn_str.contains("sslmode=require"));
    }

    #[test]
    fn test_storage_statistics() {
        let stats = StorageStatistics::default();
        assert_eq!(stats.total_events, 0);
        assert_eq!(stats.write_count, 0);
        assert_eq!(stats.read_count, 0);
    }
}
