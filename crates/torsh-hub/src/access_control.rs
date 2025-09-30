//! Access Control and Token Management
//!
//! This module provides comprehensive access control mechanisms including
//! token-based authentication, role-based access control (RBAC), and
//! fine-grained permissions for models and operations.

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use hex;
use scirs2_core::random::{Random, Rng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use torsh_core::error::{GeneralError, Result, TorshError};

/// Access token for authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessToken {
    /// Unique token identifier
    pub token_id: String,
    /// Token value (hashed for storage)
    pub token_hash: String,
    /// User ID associated with this token
    pub user_id: String,
    /// Token name/description
    pub name: String,
    /// Token creation time
    pub created_at: DateTime<Utc>,
    /// Token expiration time (None for no expiration)
    pub expires_at: Option<DateTime<Utc>>,
    /// Token scopes/permissions
    pub scopes: HashSet<TokenScope>,
    /// Token metadata
    pub metadata: HashMap<String, String>,
    /// Whether token is active
    pub is_active: bool,
    /// Last used timestamp
    pub last_used: Option<DateTime<Utc>>,
    /// Usage count
    pub usage_count: u64,
    /// Rate limiting info
    pub rate_limit: Option<RateLimit>,
}

/// Token scopes defining permissions
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TokenScope {
    /// Read model information
    ModelRead,
    /// Download models
    ModelDownload,
    /// Upload models
    ModelUpload,
    /// Delete models
    ModelDelete,
    /// Manage model metadata
    ModelMetadata,
    /// List models
    ModelList,
    /// Search models
    ModelSearch,
    /// Registry read access
    RegistryRead,
    /// Registry write access
    RegistryWrite,
    /// Admin privileges
    Admin,
    /// Custom scope
    Custom(String),
}

impl TokenScope {
    /// Check if this scope includes another scope
    pub fn includes(&self, other: &TokenScope) -> bool {
        match (self, other) {
            (TokenScope::Admin, _) => true, // Admin has all permissions
            (TokenScope::ModelMetadata, TokenScope::ModelRead) => true,
            (TokenScope::RegistryWrite, TokenScope::RegistryRead) => true,
            (a, b) => a == b,
        }
    }
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    /// Maximum requests per time window
    pub max_requests: u32,
    /// Time window in seconds
    pub window_seconds: u32,
    /// Current request count in window
    pub current_count: u32,
    /// Window start time
    pub window_start: DateTime<Utc>,
}

impl RateLimit {
    /// Create a new rate limit
    pub fn new(max_requests: u32, window_seconds: u32) -> Self {
        Self {
            max_requests,
            window_seconds,
            current_count: 0,
            window_start: Utc::now(),
        }
    }

    /// Check if request is allowed and update counters
    pub fn check_and_update(&mut self) -> bool {
        let now = Utc::now();
        let window_duration = ChronoDuration::seconds(self.window_seconds as i64);

        // Reset window if expired
        if now - self.window_start > window_duration {
            self.current_count = 0;
            self.window_start = now;
        }

        // Check if under limit
        if self.current_count < self.max_requests {
            self.current_count += 1;
            true
        } else {
            false
        }
    }

    /// Get time until rate limit resets
    pub fn time_until_reset(&self) -> ChronoDuration {
        let window_duration = ChronoDuration::seconds(self.window_seconds as i64);
        let elapsed = Utc::now() - self.window_start;

        if elapsed < window_duration {
            window_duration - elapsed
        } else {
            ChronoDuration::zero()
        }
    }
}

impl AccessToken {
    /// Create a new access token
    pub fn new(
        user_id: String,
        name: String,
        scopes: HashSet<TokenScope>,
        expires_in: Option<ChronoDuration>,
    ) -> (Self, String) {
        let token_id = generate_token_id();
        let raw_token = generate_raw_token();
        let token_hash = hash_token(&raw_token);

        let expires_at = expires_in.map(|duration| Utc::now() + duration);

        let token = Self {
            token_id,
            token_hash,
            user_id,
            name,
            created_at: Utc::now(),
            expires_at,
            scopes,
            metadata: HashMap::new(),
            is_active: true,
            last_used: None,
            usage_count: 0,
            rate_limit: None,
        };

        (token, raw_token)
    }

    /// Check if token is valid
    pub fn is_valid(&self) -> bool {
        self.is_active && !self.is_expired()
    }

    /// Check if token is expired
    pub fn is_expired(&self) -> bool {
        self.expires_at.map(|exp| Utc::now() > exp).unwrap_or(false)
    }

    /// Check if token has specific scope
    pub fn has_scope(&self, scope: &TokenScope) -> bool {
        self.scopes.iter().any(|s| s.includes(scope))
    }

    /// Update last used timestamp
    pub fn mark_used(&mut self) {
        self.last_used = Some(Utc::now());
        self.usage_count += 1;
    }

    /// Check rate limit
    pub fn check_rate_limit(&mut self) -> bool {
        self.rate_limit
            .as_mut()
            .map(|rl| rl.check_and_update())
            .unwrap_or(true) // No rate limit means allowed
    }

    /// Set rate limit
    pub fn set_rate_limit(&mut self, max_requests: u32, window_seconds: u32) {
        self.rate_limit = Some(RateLimit::new(max_requests, window_seconds));
    }

    /// Verify raw token against stored hash
    pub fn verify_token(&self, raw_token: &str) -> bool {
        hash_token(raw_token) == self.token_hash
    }
}

/// Token manager for handling access tokens
#[derive(Debug)]
pub struct TokenManager {
    /// Active tokens by token ID
    tokens: HashMap<String, AccessToken>,
    /// Token lookup by hash
    token_by_hash: HashMap<String, String>, // hash -> token_id
    /// Tokens by user ID
    tokens_by_user: HashMap<String, HashSet<String>>, // user_id -> token_ids
}

impl TokenManager {
    /// Create a new token manager
    pub fn new() -> Self {
        Self {
            tokens: HashMap::new(),
            token_by_hash: HashMap::new(),
            tokens_by_user: HashMap::new(),
        }
    }

    /// Create a new access token
    pub fn create_token(
        &mut self,
        user_id: String,
        name: String,
        scopes: HashSet<TokenScope>,
        expires_in: Option<ChronoDuration>,
    ) -> Result<(String, String)> {
        let (token, raw_token) = AccessToken::new(user_id.clone(), name, scopes, expires_in);
        let token_id = token.token_id.clone();
        let token_hash = token.token_hash.clone();

        // Store token
        self.tokens.insert(token_id.clone(), token);
        self.token_by_hash.insert(token_hash, token_id.clone());

        // Update user mapping
        self.tokens_by_user
            .entry(user_id)
            .or_insert_with(HashSet::new)
            .insert(token_id.clone());

        Ok((token_id, raw_token))
    }

    /// Authenticate a raw token
    pub fn authenticate(&mut self, raw_token: &str) -> Result<&mut AccessToken> {
        let token_hash = hash_token(raw_token);

        let token_id = self.token_by_hash.get(&token_hash).ok_or_else(|| {
            TorshError::General(GeneralError::RuntimeError("Invalid token".to_string()))
        })?;

        let token = self.tokens.get_mut(token_id).ok_or_else(|| {
            TorshError::General(GeneralError::RuntimeError("Token not found".to_string()))
        })?;

        if !token.is_valid() {
            return Err(TorshError::General(GeneralError::RuntimeError(
                "Token is invalid or expired".to_string(),
            )));
        }

        // Check rate limit
        if !token.check_rate_limit() {
            return Err(TorshError::General(GeneralError::RuntimeError(
                "Rate limit exceeded".to_string(),
            )));
        }

        token.mark_used();
        Ok(token)
    }

    /// Revoke a token
    pub fn revoke_token(&mut self, token_id: &str) -> Result<()> {
        let token = self.tokens.get_mut(token_id).ok_or_else(|| {
            TorshError::General(GeneralError::RuntimeError("Token not found".to_string()))
        })?;

        token.is_active = false;
        Ok(())
    }

    /// Delete a token
    pub fn delete_token(&mut self, token_id: &str) -> Result<()> {
        let token = self.tokens.remove(token_id).ok_or_else(|| {
            TorshError::General(GeneralError::RuntimeError("Token not found".to_string()))
        })?;

        // Remove from hash lookup
        self.token_by_hash.remove(&token.token_hash);

        // Remove from user mapping
        if let Some(user_tokens) = self.tokens_by_user.get_mut(&token.user_id) {
            user_tokens.remove(token_id);
            if user_tokens.is_empty() {
                self.tokens_by_user.remove(&token.user_id);
            }
        }

        Ok(())
    }

    /// List tokens for a user
    pub fn list_user_tokens(&self, user_id: &str) -> Vec<&AccessToken> {
        self.tokens_by_user
            .get(user_id)
            .map(|token_ids| {
                token_ids
                    .iter()
                    .filter_map(|id| self.tokens.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Clean up expired tokens
    pub fn cleanup_expired(&mut self) -> usize {
        let expired_tokens: Vec<String> = self
            .tokens
            .iter()
            .filter(|(_, token)| token.is_expired())
            .map(|(id, _)| id.clone())
            .collect();

        let count = expired_tokens.len();

        for token_id in expired_tokens {
            let _ = self.delete_token(&token_id);
        }

        count
    }

    /// Get token statistics
    pub fn stats(&self) -> TokenStats {
        let total_tokens = self.tokens.len();
        let active_tokens = self.tokens.values().filter(|t| t.is_valid()).count();
        let expired_tokens = self.tokens.values().filter(|t| t.is_expired()).count();
        let inactive_tokens = self.tokens.values().filter(|t| !t.is_active).count();

        TokenStats {
            total_tokens,
            active_tokens,
            expired_tokens,
            inactive_tokens,
            unique_users: self.tokens_by_user.len(),
        }
    }
}

impl Default for TokenManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Token statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenStats {
    pub total_tokens: usize,
    pub active_tokens: usize,
    pub expired_tokens: usize,
    pub inactive_tokens: usize,
    pub unique_users: usize,
}

/// Permission checker for operations
#[derive(Debug)]
pub struct PermissionChecker {
    /// Required permissions for operations
    operation_permissions: HashMap<String, HashSet<TokenScope>>,
}

impl PermissionChecker {
    /// Create a new permission checker
    pub fn new() -> Self {
        let mut checker = Self {
            operation_permissions: HashMap::new(),
        };

        // Set up default permissions
        checker.setup_default_permissions();
        checker
    }

    /// Set up default operation permissions
    fn setup_default_permissions(&mut self) {
        // Model operations
        self.require_permission("model.read", TokenScope::ModelRead);
        self.require_permission("model.download", TokenScope::ModelDownload);
        self.require_permission("model.upload", TokenScope::ModelUpload);
        self.require_permission("model.delete", TokenScope::ModelDelete);
        self.require_permission("model.metadata", TokenScope::ModelMetadata);
        self.require_permission("model.list", TokenScope::ModelList);
        self.require_permission("model.search", TokenScope::ModelSearch);

        // Registry operations
        self.require_permission("registry.read", TokenScope::RegistryRead);
        self.require_permission("registry.write", TokenScope::RegistryWrite);

        // Admin operations
        self.require_permissions("admin.user_management", vec![TokenScope::Admin]);
        self.require_permissions("admin.token_management", vec![TokenScope::Admin]);
    }

    /// Require a specific permission for an operation
    pub fn require_permission(&mut self, operation: &str, scope: TokenScope) {
        self.operation_permissions
            .entry(operation.to_string())
            .or_insert_with(HashSet::new)
            .insert(scope);
    }

    /// Require multiple permissions for an operation
    pub fn require_permissions(&mut self, operation: &str, scopes: Vec<TokenScope>) {
        let perms = self
            .operation_permissions
            .entry(operation.to_string())
            .or_insert_with(HashSet::new);

        for scope in scopes {
            perms.insert(scope);
        }
    }

    /// Check if token has permission for operation
    pub fn check_permission(&self, token: &AccessToken, operation: &str) -> bool {
        if let Some(required_scopes) = self.operation_permissions.get(operation) {
            // Check if token has any of the required scopes
            required_scopes.iter().any(|scope| token.has_scope(scope))
        } else {
            // No specific permissions required
            true
        }
    }

    /// Authorize operation with detailed error
    pub fn authorize(&self, token: &AccessToken, operation: &str) -> Result<()> {
        if !token.is_valid() {
            return Err(TorshError::General(GeneralError::RuntimeError(
                "Token is invalid or expired".to_string(),
            )));
        }

        if !self.check_permission(token, operation) {
            return Err(TorshError::General(GeneralError::SecurityError(format!(
                "Insufficient permissions for operation: {}",
                operation
            ))));
        }

        Ok(())
    }
}

impl Default for PermissionChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate a unique token ID
fn generate_token_id() -> String {
    format!("tok_{}", generate_random_string(16))
}

/// Generate a raw token
fn generate_raw_token() -> String {
    generate_random_string(32)
}

/// Generate a random string
fn generate_random_string(length: usize) -> String {
    let mut rng = scirs2_core::random::thread_rng();
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

    (0..length)
        .map(|_| {
            let idx = rng.gen_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect()
}

/// Hash a token for storage
fn hash_token(token: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_creation() {
        let scopes = vec![TokenScope::ModelRead, TokenScope::ModelDownload]
            .into_iter()
            .collect();

        let (token, raw_token) = AccessToken::new(
            "user123".to_string(),
            "test-token".to_string(),
            scopes,
            Some(ChronoDuration::hours(24)),
        );

        assert_eq!(token.user_id, "user123");
        assert_eq!(token.name, "test-token");
        assert!(token.is_valid());
        assert!(token.verify_token(&raw_token));
    }

    #[test]
    fn test_token_scopes() {
        let token_scopes = vec![TokenScope::ModelRead, TokenScope::Admin]
            .into_iter()
            .collect();

        let (token, _) = AccessToken::new(
            "user123".to_string(),
            "test-token".to_string(),
            token_scopes,
            None,
        );

        assert!(token.has_scope(&TokenScope::ModelRead));
        assert!(token.has_scope(&TokenScope::ModelDownload)); // Admin includes all
        assert!(token.has_scope(&TokenScope::Admin));
    }

    #[test]
    fn test_rate_limiting() {
        let mut rate_limit = RateLimit::new(2, 60); // 2 requests per minute

        assert!(rate_limit.check_and_update()); // First request
        assert!(rate_limit.check_and_update()); // Second request
        assert!(!rate_limit.check_and_update()); // Third request should fail
    }

    #[test]
    fn test_token_manager() {
        let mut manager = TokenManager::new();

        let scopes = vec![TokenScope::ModelRead].into_iter().collect();
        let (token_id, raw_token) = manager
            .create_token(
                "user123".to_string(),
                "test-token".to_string(),
                scopes,
                None,
            )
            .unwrap();

        // Test authentication
        let token = manager.authenticate(&raw_token).unwrap();
        assert_eq!(token.user_id, "user123");

        // Test revocation
        manager.revoke_token(&token_id).unwrap();
        assert!(manager.authenticate(&raw_token).is_err());
    }

    #[test]
    fn test_permission_checker() {
        let checker = PermissionChecker::new();

        let scopes = vec![TokenScope::ModelRead].into_iter().collect();
        let (token, _) = AccessToken::new(
            "user123".to_string(),
            "test-token".to_string(),
            scopes,
            None,
        );

        assert!(checker.check_permission(&token, "model.read"));
        assert!(!checker.check_permission(&token, "model.upload"));

        assert!(checker.authorize(&token, "model.read").is_ok());
        assert!(checker.authorize(&token, "model.upload").is_err());
    }
}
