//! Database integration for loading data from various database backends
//!
//! This module provides a unified interface for loading data from different
//! database systems such as SQLite, PostgreSQL, MySQL, etc.

use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

use crate::dataset::Dataset;
use crate::error::DataError;
use torsh_core::TensorElement;
use torsh_tensor::Tensor;

#[derive(Error, Debug)]
pub enum DatabaseError {
    #[error("Connection error: {0}")]
    ConnectionError(String),
    #[error("Query error: {0}")]
    QueryError(String),
    #[error("Type conversion error: {0}")]
    TypeConversionError(String),
    #[error("Configuration error: {0}")]
    ConfigError(String),
    #[error("Column not found: {0}")]
    ColumnNotFound(String),
}

impl From<DatabaseError> for DataError {
    fn from(err: DatabaseError) -> Self {
        DataError::Other(err.to_string())
    }
}

/// Supported database backends
#[derive(Debug, Clone, PartialEq)]
pub enum DatabaseBackend {
    SQLite,
    PostgreSQL,
    MySQL,
    Memory, // In-memory database for testing
}

impl fmt::Display for DatabaseBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DatabaseBackend::SQLite => write!(f, "SQLite"),
            DatabaseBackend::PostgreSQL => write!(f, "PostgreSQL"),
            DatabaseBackend::MySQL => write!(f, "MySQL"),
            DatabaseBackend::Memory => write!(f, "Memory"),
        }
    }
}

/// Database value types
#[derive(Debug, Clone)]
pub enum DatabaseValue {
    Integer(i64),
    Float(f64),
    Text(String),
    Blob(Vec<u8>),
    Null,
}

impl DatabaseValue {
    /// Convert to a tensor element if possible
    pub fn to_tensor_element<T: TensorElement>(&self) -> std::result::Result<T, DatabaseError> {
        match self {
            DatabaseValue::Integer(val) => T::from_f64(*val as f64).ok_or_else(|| {
                DatabaseError::TypeConversionError(format!(
                    "Cannot convert integer {val} to target type"
                ))
            }),
            DatabaseValue::Float(val) => T::from_f64(*val).ok_or_else(|| {
                DatabaseError::TypeConversionError(format!(
                    "Cannot convert float {val} to target type"
                ))
            }),
            DatabaseValue::Text(val) => {
                // Try to parse as number
                if let Ok(num) = val.parse::<f64>() {
                    T::from_f64(num).ok_or_else(|| {
                        DatabaseError::TypeConversionError(format!(
                            "Cannot convert parsed number {num} to target type"
                        ))
                    })
                } else {
                    Err(DatabaseError::TypeConversionError(format!(
                        "Cannot convert text '{val}' to numeric type"
                    )))
                }
            }
            DatabaseValue::Null => T::from_f64(0.0).ok_or_else(|| {
                DatabaseError::TypeConversionError("Cannot convert NULL to target type".to_string())
            }),
            DatabaseValue::Blob(_) => Err(DatabaseError::TypeConversionError(
                "Cannot convert BLOB to numeric type".to_string(),
            )),
        }
    }
}

/// A row of data from a database query result
#[derive(Debug, Clone)]
pub struct DatabaseRow {
    columns: HashMap<String, DatabaseValue>,
}

impl DatabaseRow {
    /// Create a new database row
    pub fn new() -> Self {
        Self {
            columns: HashMap::new(),
        }
    }

    /// Add a column value
    pub fn add_column(&mut self, name: String, value: DatabaseValue) {
        self.columns.insert(name, value);
    }

    /// Get a column value by name
    pub fn get_column(&self, name: &str) -> Option<&DatabaseValue> {
        self.columns.get(name)
    }

    /// Get all column names
    pub fn column_names(&self) -> Vec<&String> {
        self.columns.keys().collect()
    }

    /// Convert a column to a tensor element
    pub fn column_to_tensor_element<T: TensorElement>(
        &self,
        column_name: &str,
    ) -> std::result::Result<T, DatabaseError> {
        let value = self
            .get_column(column_name)
            .ok_or_else(|| DatabaseError::ColumnNotFound(column_name.to_string()))?;
        value.to_tensor_element()
    }

    /// Convert multiple columns to a tensor
    pub fn columns_to_tensor<T: TensorElement>(
        &self,
        column_names: &[&str],
    ) -> std::result::Result<Tensor<T>, DatabaseError> {
        let mut values = Vec::with_capacity(column_names.len());

        for &column_name in column_names {
            let tensor_value = self.column_to_tensor_element::<T>(column_name)?;
            values.push(tensor_value);
        }

        let shape = vec![values.len()];
        Tensor::from_vec(values, &shape)
            .map_err(|e| DatabaseError::TypeConversionError(e.to_string()))
    }
}

impl Default for DatabaseRow {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for database connections
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    pub backend: DatabaseBackend,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub database: String,
    pub username: Option<String>,
    pub password: Option<String>,
    pub connection_string: Option<String>,
}

impl DatabaseConfig {
    /// Create a new database config
    pub fn new(backend: DatabaseBackend, database: String) -> Self {
        Self {
            backend,
            host: None,
            port: None,
            database,
            username: None,
            password: None,
            connection_string: None,
        }
    }

    /// Set host and port
    pub fn with_host_port(mut self, host: String, port: u16) -> Self {
        self.host = Some(host);
        self.port = Some(port);
        self
    }

    /// Set credentials
    pub fn with_credentials(mut self, username: String, password: String) -> Self {
        self.username = Some(username);
        self.password = Some(password);
        self
    }

    /// Set custom connection string
    pub fn with_connection_string(mut self, connection_string: String) -> Self {
        self.connection_string = Some(connection_string);
        self
    }

    /// Build connection string based on backend
    pub fn build_connection_string(&self) -> String {
        if let Some(ref custom) = self.connection_string {
            return custom.clone();
        }

        match self.backend {
            DatabaseBackend::SQLite => {
                format!("sqlite:{}", self.database)
            }
            DatabaseBackend::PostgreSQL => {
                let host = self.host.as_deref().unwrap_or("localhost");
                let port = self.port.unwrap_or(5432);
                let username = self.username.as_deref().unwrap_or("postgres");
                let password = self.password.as_deref().unwrap_or("");
                format!(
                    "postgresql://{}:{}@{}:{}/{}",
                    username, password, host, port, self.database
                )
            }
            DatabaseBackend::MySQL => {
                let host = self.host.as_deref().unwrap_or("localhost");
                let port = self.port.unwrap_or(3306);
                let username = self.username.as_deref().unwrap_or("root");
                let password = self.password.as_deref().unwrap_or("");
                format!(
                    "mysql://{}:{}@{}:{}/{}",
                    username, password, host, port, self.database
                )
            }
            DatabaseBackend::Memory => ":memory:".to_string(),
        }
    }
}

/// Trait for database connections
pub trait DatabaseConnection: Send + Sync {
    /// Execute a query and return the results
    fn execute_query(
        &mut self,
        query: &str,
    ) -> std::result::Result<Vec<DatabaseRow>, DatabaseError>;

    /// Get table names
    fn get_table_names(&mut self) -> std::result::Result<Vec<String>, DatabaseError>;

    /// Get column names for a table
    fn get_column_names(
        &mut self,
        table_name: &str,
    ) -> std::result::Result<Vec<String>, DatabaseError>;

    /// Count rows in a table
    fn count_rows(&mut self, table_name: &str) -> std::result::Result<usize, DatabaseError>;

    /// Close the connection
    fn close(&mut self) -> std::result::Result<(), DatabaseError>;
}

/// Mock database connection for testing and demonstration
pub struct MockDatabaseConnection {
    _backend: DatabaseBackend,
    tables: HashMap<String, Vec<DatabaseRow>>,
}

impl MockDatabaseConnection {
    /// Create a new mock connection
    pub fn new(backend: DatabaseBackend) -> Self {
        let mut tables = HashMap::new();

        // Create some sample data
        let mut sample_rows = Vec::new();
        for i in 0..100 {
            let mut row = DatabaseRow::new();
            row.add_column("id".to_string(), DatabaseValue::Integer(i));
            row.add_column("value".to_string(), DatabaseValue::Float(i as f64 * 1.5));
            row.add_column("name".to_string(), DatabaseValue::Text(format!("item_{i}")));
            sample_rows.push(row);
        }
        tables.insert("sample_table".to_string(), sample_rows);

        Self {
            _backend: backend,
            tables,
        }
    }
}

impl DatabaseConnection for MockDatabaseConnection {
    fn execute_query(
        &mut self,
        query: &str,
    ) -> std::result::Result<Vec<DatabaseRow>, DatabaseError> {
        // Very simple query parsing for demo purposes
        let query_lower = query.to_lowercase();

        if query_lower.contains("select") && query_lower.contains("from") {
            // Extract table name (very simplified)
            if let Some(table_name) = query_lower.split("from").nth(1) {
                let table_name = table_name.split_whitespace().next().unwrap_or("").trim();

                if let Some(rows) = self.tables.get(table_name) {
                    // Apply LIMIT if present
                    if let Some(limit_part) = query_lower.split("limit").nth(1) {
                        if let Ok(limit) = limit_part.trim().parse::<usize>() {
                            return Ok(rows.iter().take(limit).cloned().collect());
                        }
                    }

                    return Ok(rows.clone());
                }
            }
        }

        Err(DatabaseError::QueryError(format!(
            "Query not supported: {query}"
        )))
    }

    fn get_table_names(&mut self) -> std::result::Result<Vec<String>, DatabaseError> {
        Ok(self.tables.keys().cloned().collect())
    }

    fn get_column_names(
        &mut self,
        table_name: &str,
    ) -> std::result::Result<Vec<String>, DatabaseError> {
        if let Some(rows) = self.tables.get(table_name) {
            if let Some(first_row) = rows.first() {
                return Ok(first_row
                    .column_names()
                    .iter()
                    .map(|s| (*s).clone())
                    .collect());
            }
        }
        Err(DatabaseError::QueryError(format!(
            "Table not found: {table_name}"
        )))
    }

    fn count_rows(&mut self, table_name: &str) -> std::result::Result<usize, DatabaseError> {
        if let Some(rows) = self.tables.get(table_name) {
            Ok(rows.len())
        } else {
            Err(DatabaseError::QueryError(format!(
                "Table not found: {table_name}"
            )))
        }
    }

    fn close(&mut self) -> std::result::Result<(), DatabaseError> {
        // Nothing to do for mock connection
        Ok(())
    }
}

/// Dataset that loads data from a database table
pub struct DatabaseDataset {
    connection: Box<dyn DatabaseConnection>,
    table_name: String,
    columns: Vec<String>,
    total_rows: usize,
    _batch_size: usize,
}

impl DatabaseDataset {
    /// Create a new database dataset
    pub fn new(
        mut connection: Box<dyn DatabaseConnection>,
        table_name: String,
        columns: Option<Vec<String>>,
        batch_size: Option<usize>,
    ) -> std::result::Result<Self, DatabaseError> {
        // Get column names if not specified
        let columns = match columns {
            Some(cols) => cols,
            None => connection.get_column_names(&table_name)?,
        };

        let total_rows = connection.count_rows(&table_name)?;
        let batch_size = batch_size.unwrap_or(1);

        Ok(Self {
            connection,
            table_name,
            columns,
            total_rows,
            _batch_size: batch_size,
        })
    }

    /// Get column names
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    /// Get table name
    pub fn table_name(&self) -> &str {
        &self.table_name
    }

    /// Read a batch of rows
    pub fn read_batch(
        &mut self,
        start_idx: usize,
        batch_size: usize,
    ) -> std::result::Result<Vec<DatabaseRow>, DatabaseError> {
        let query = format!(
            "SELECT {} FROM {} LIMIT {} OFFSET {}",
            self.columns.join(", "),
            self.table_name,
            batch_size,
            start_idx
        );

        self.connection.execute_query(&query)
    }

    /// Convert rows to tensors
    pub fn rows_to_tensors<T: TensorElement>(
        &self,
        rows: &[DatabaseRow],
    ) -> std::result::Result<Vec<Tensor<T>>, DatabaseError> {
        let mut column_tensors = Vec::new();

        for column_name in &self.columns {
            let mut column_values = Vec::with_capacity(rows.len());

            for row in rows {
                let value = row.column_to_tensor_element::<T>(column_name)?;
                column_values.push(value);
            }

            let shape = vec![column_values.len()];
            let tensor = Tensor::from_vec(column_values, &shape)
                .map_err(|e| DatabaseError::TypeConversionError(e.to_string()))?;
            column_tensors.push(tensor);
        }

        Ok(column_tensors)
    }
}

impl Dataset for DatabaseDataset {
    type Item = DatabaseRow;

    fn len(&self) -> usize {
        self.total_rows
    }

    fn get(&self, index: usize) -> torsh_core::error::Result<Self::Item> {
        if index >= self.total_rows {
            return Err(DataError::Other(format!(
                "Index {} out of bounds for dataset of size {}",
                index, self.total_rows
            ))
            .into());
        }

        // This is inefficient for individual row access but works for demonstration
        let _query = format!(
            "SELECT {} FROM {} LIMIT 1 OFFSET {}",
            self.columns.join(", "),
            self.table_name,
            index
        );

        // Since we need &mut self but trait requires &self, we'll create a simple workaround
        // In practice, you'd design this differently or use interior mutability
        Err(DataError::Other(
            "Individual row access not supported. Use batch operations instead.".to_string(),
        )
        .into())
    }
}

/// Builder for creating database datasets
pub struct DatabaseDatasetBuilder {
    config: DatabaseConfig,
    table_name: Option<String>,
    columns: Option<Vec<String>>,
    batch_size: Option<usize>,
    query: Option<String>,
}

impl DatabaseDatasetBuilder {
    /// Create a new builder
    pub fn new(config: DatabaseConfig) -> Self {
        Self {
            config,
            table_name: None,
            columns: None,
            batch_size: None,
            query: None,
        }
    }

    /// Set the table name
    pub fn table(mut self, table_name: String) -> Self {
        self.table_name = Some(table_name);
        self
    }

    /// Set the columns to select
    pub fn columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }

    /// Set the batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Set a custom query
    pub fn query(mut self, query: String) -> Self {
        self.query = Some(query);
        self
    }

    /// Build the database dataset
    pub fn build(self) -> std::result::Result<DatabaseDataset, DatabaseError> {
        let connection: Box<dyn DatabaseConnection> = match self.config.backend {
            DatabaseBackend::Memory => Box::new(MockDatabaseConnection::new(self.config.backend)),
            _ => {
                // For now, use mock connection for all backends
                // In a real implementation, you'd create actual database connections
                Box::new(MockDatabaseConnection::new(self.config.backend))
            }
        };

        let table_name = self
            .table_name
            .ok_or_else(|| DatabaseError::ConfigError("Table name is required".to_string()))?;

        DatabaseDataset::new(connection, table_name, self.columns, self.batch_size)
    }
}

/// Utility functions for database operations
pub mod database_utils {
    use super::*;

    /// Create a SQLite database configuration
    pub fn sqlite_config<P: AsRef<std::path::Path>>(database_path: P) -> DatabaseConfig {
        DatabaseConfig::new(
            DatabaseBackend::SQLite,
            database_path.as_ref().to_string_lossy().to_string(),
        )
    }

    /// Create a PostgreSQL database configuration
    pub fn postgresql_config(
        host: &str,
        port: u16,
        database: &str,
        username: &str,
        password: &str,
    ) -> DatabaseConfig {
        DatabaseConfig::new(DatabaseBackend::PostgreSQL, database.to_string())
            .with_host_port(host.to_string(), port)
            .with_credentials(username.to_string(), password.to_string())
    }

    /// Create a MySQL database configuration
    pub fn mysql_config(
        host: &str,
        port: u16,
        database: &str,
        username: &str,
        password: &str,
    ) -> DatabaseConfig {
        DatabaseConfig::new(DatabaseBackend::MySQL, database.to_string())
            .with_host_port(host.to_string(), port)
            .with_credentials(username.to_string(), password.to_string())
    }

    /// Create an in-memory database configuration for testing
    pub fn memory_config() -> DatabaseConfig {
        DatabaseConfig::new(DatabaseBackend::Memory, ":memory:".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_value_conversion() {
        let int_val = DatabaseValue::Integer(42);
        let float_val = DatabaseValue::Float(3.14);
        let text_val = DatabaseValue::Text("123.45".to_string());

        assert!(int_val.to_tensor_element::<f32>().is_ok());
        assert!(float_val.to_tensor_element::<f64>().is_ok());
        assert!(text_val.to_tensor_element::<f32>().is_ok());
    }

    #[test]
    fn test_database_row() {
        let mut row = DatabaseRow::new();
        row.add_column("id".to_string(), DatabaseValue::Integer(1));
        row.add_column("value".to_string(), DatabaseValue::Float(2.5));

        assert!(row.get_column("id").is_some());
        assert!(row.get_column("nonexistent").is_none());
        assert_eq!(row.column_names().len(), 2);
    }

    #[test]
    fn test_database_config() {
        let config = DatabaseConfig::new(DatabaseBackend::SQLite, "test.db".to_string());
        assert_eq!(config.build_connection_string(), "sqlite:test.db");

        let pg_config =
            database_utils::postgresql_config("localhost", 5432, "testdb", "user", "pass");
        assert!(pg_config
            .build_connection_string()
            .contains("postgresql://"));
    }

    #[test]
    fn test_mock_connection() {
        let mut conn = MockDatabaseConnection::new(DatabaseBackend::Memory);

        let tables = conn.get_table_names().unwrap();
        assert!(!tables.is_empty());

        let columns = conn.get_column_names("sample_table").unwrap();
        assert!(!columns.is_empty());

        let count = conn.count_rows("sample_table").unwrap();
        assert!(count > 0);
    }

    #[test]
    fn test_database_dataset_builder() {
        let config = database_utils::memory_config();
        let builder = DatabaseDatasetBuilder::new(config)
            .table("sample_table".to_string())
            .columns(vec!["id".to_string(), "value".to_string()])
            .batch_size(10);

        let dataset = builder.build();
        assert!(dataset.is_ok());
    }
}
