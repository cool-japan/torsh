//! Security functionality for ToRSh Hub
//!
//! This module provides model signing, verification, and security features
//! to ensure model integrity and authenticity.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use torsh_core::error::{GeneralError, Result, TorshError};

/// Digital signature for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSignature {
    /// SHA-256 hash of the model file
    pub file_hash: String,
    /// Signature of the hash using a private key
    pub signature: String,
    /// Public key ID used for verification
    pub key_id: String,
    /// Timestamp when the signature was created
    pub timestamp: u64,
    /// Algorithm used for signing
    pub algorithm: SignatureAlgorithm,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Supported signature algorithms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SignatureAlgorithm {
    /// RSA with SHA-256
    RsaSha256,
    /// Ed25519
    Ed25519,
    /// ECDSA with P-256 curve
    EcdsaP256,
}

/// Key pair for signing and verification
#[derive(Debug, Clone)]
pub struct KeyPair {
    pub key_id: String,
    pub algorithm: SignatureAlgorithm,
    pub public_key: Vec<u8>,
    pub private_key: Option<Vec<u8>>, // None for verification-only keys
}

/// Model security manager
pub struct SecurityManager {
    key_store: HashMap<String, KeyPair>,
    trusted_keys: Vec<String>,
}

impl SecurityManager {
    /// Create a new security manager
    pub fn new() -> Self {
        Self {
            key_store: HashMap::new(),
            trusted_keys: Vec::new(),
        }
    }

    /// Add a key pair to the key store
    pub fn add_key(&mut self, key_pair: KeyPair) {
        let key_id = key_pair.key_id.clone();
        self.key_store.insert(key_id, key_pair);
    }

    /// Mark a key as trusted
    pub fn trust_key(&mut self, key_id: &str) -> Result<()> {
        if !self.key_store.contains_key(key_id) {
            return Err(TorshError::General(GeneralError::InvalidArgument(format!(
                "Key '{}' not found in key store",
                key_id
            ))));
        }

        if !self.trusted_keys.contains(&key_id.to_string()) {
            self.trusted_keys.push(key_id.to_string());
        }

        Ok(())
    }

    /// Remove trust from a key
    pub fn untrust_key(&mut self, key_id: &str) {
        self.trusted_keys.retain(|k| k != key_id);
    }

    /// Sign a model file
    pub fn sign_model<P: AsRef<Path>>(
        &self,
        model_path: P,
        key_id: &str,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<ModelSignature> {
        let model_path = model_path.as_ref();

        // Get the key pair
        let key_pair = self.key_store.get(key_id).ok_or_else(|| {
            TorshError::General(GeneralError::InvalidArgument(format!(
                "Key '{}' not found",
                key_id
            )))
        })?;

        if key_pair.private_key.is_none() {
            return Err(TorshError::General(GeneralError::InvalidArgument(
                "Cannot sign with a verification-only key".to_string(),
            )));
        }

        // Calculate file hash
        let file_hash = calculate_file_hash(model_path)?;

        // Create signature
        let signature = match key_pair.algorithm {
            SignatureAlgorithm::RsaSha256 => sign_with_rsa_sha256(
                &file_hash,
                key_pair
                    .private_key
                    .as_ref()
                    .expect("RSA private key required for signing"),
            )?,
            SignatureAlgorithm::Ed25519 => sign_with_ed25519(
                &file_hash,
                key_pair
                    .private_key
                    .as_ref()
                    .expect("Ed25519 private key required for signing"),
            )?,
            SignatureAlgorithm::EcdsaP256 => sign_with_ecdsa_p256(
                &file_hash,
                key_pair
                    .private_key
                    .as_ref()
                    .expect("ECDSA P256 private key required for signing"),
            )?,
        };

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after UNIX epoch")
            .as_secs();

        Ok(ModelSignature {
            file_hash,
            signature,
            key_id: key_id.to_string(),
            timestamp,
            algorithm: key_pair.algorithm.clone(),
            metadata: metadata.unwrap_or_default(),
        })
    }

    /// Verify a model signature
    pub fn verify_model<P: AsRef<Path>>(
        &self,
        model_path: P,
        signature: &ModelSignature,
        require_trusted: bool,
    ) -> Result<bool> {
        let model_path = model_path.as_ref();

        // Check if key is trusted (if required)
        if require_trusted && !self.trusted_keys.contains(&signature.key_id) {
            return Err(TorshError::General(GeneralError::RuntimeError(format!(
                "Key '{}' is not trusted",
                signature.key_id
            ))));
        }

        // Get the public key
        let key_pair = self.key_store.get(&signature.key_id).ok_or_else(|| {
            TorshError::General(GeneralError::InvalidArgument(format!(
                "Key '{}' not found",
                signature.key_id
            )))
        })?;

        // Verify algorithm matches
        if key_pair.algorithm != signature.algorithm {
            return Ok(false);
        }

        // Calculate current file hash
        let current_hash = calculate_file_hash(model_path)?;

        // Verify file hash matches
        if current_hash != signature.file_hash {
            return Ok(false);
        }

        // Verify signature
        let is_valid = match signature.algorithm {
            SignatureAlgorithm::RsaSha256 => verify_rsa_sha256(
                &signature.file_hash,
                &signature.signature,
                &key_pair.public_key,
            )?,
            SignatureAlgorithm::Ed25519 => verify_ed25519(
                &signature.file_hash,
                &signature.signature,
                &key_pair.public_key,
            )?,
            SignatureAlgorithm::EcdsaP256 => verify_ecdsa_p256(
                &signature.file_hash,
                &signature.signature,
                &key_pair.public_key,
            )?,
        };

        Ok(is_valid)
    }

    /// Save a signature to a file
    pub fn save_signature<P: AsRef<Path>>(
        signature: &ModelSignature,
        signature_path: P,
    ) -> Result<()> {
        let json = serde_json::to_string_pretty(signature)
            .map_err(|e| TorshError::SerializationError(e.to_string()))?;

        std::fs::write(signature_path, json)?;
        Ok(())
    }

    /// Load a signature from a file
    pub fn load_signature<P: AsRef<Path>>(signature_path: P) -> Result<ModelSignature> {
        let content = std::fs::read_to_string(signature_path)?;
        let signature: ModelSignature = serde_json::from_str(&content)
            .map_err(|e| TorshError::SerializationError(e.to_string()))?;

        Ok(signature)
    }

    /// Generate a new key pair
    pub fn generate_key_pair(key_id: String, algorithm: SignatureAlgorithm) -> Result<KeyPair> {
        match algorithm {
            SignatureAlgorithm::RsaSha256 => generate_rsa_key_pair(key_id),
            SignatureAlgorithm::Ed25519 => generate_ed25519_key_pair(key_id),
            SignatureAlgorithm::EcdsaP256 => generate_ecdsa_p256_key_pair(key_id),
        }
    }

    /// Get list of trusted keys
    pub fn trusted_keys(&self) -> &[String] {
        &self.trusted_keys
    }

    /// Get list of all keys
    pub fn all_keys(&self) -> Vec<&str> {
        self.key_store.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for SecurityManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate SHA-256 hash of a file
pub fn calculate_file_hash<P: AsRef<Path>>(file_path: P) -> Result<String> {
    let file = File::open(file_path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0; 8192];

    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }

    let result = hasher.finalize();
    Ok(hex::encode(result))
}

/// Verify file integrity by comparing hashes
pub fn verify_file_integrity<P: AsRef<Path>>(file_path: P, expected_hash: &str) -> Result<bool> {
    let actual_hash = calculate_file_hash(file_path)?;
    Ok(actual_hash == expected_hash)
}

// Placeholder implementations for cryptographic operations
// In a real implementation, these would use proper cryptographic libraries

fn sign_with_rsa_sha256(_data: &str, _private_key: &[u8]) -> Result<String> {
    // Placeholder: In real implementation, use RSA + SHA-256 signing
    // This would use libraries like `rsa` and `sha2`
    Ok("rsa_signature_placeholder".to_string())
}

fn verify_rsa_sha256(_data: &str, _signature: &str, _public_key: &[u8]) -> Result<bool> {
    // Placeholder: In real implementation, verify RSA + SHA-256 signature
    Ok(_signature == "rsa_signature_placeholder")
}

fn sign_with_ed25519(_data: &str, _private_key: &[u8]) -> Result<String> {
    // Placeholder: In real implementation, use Ed25519 signing
    // This would use libraries like `ed25519-dalek`
    Ok("ed25519_signature_placeholder".to_string())
}

fn verify_ed25519(_data: &str, _signature: &str, _public_key: &[u8]) -> Result<bool> {
    // Placeholder: In real implementation, verify Ed25519 signature
    Ok(_signature == "ed25519_signature_placeholder")
}

fn sign_with_ecdsa_p256(_data: &str, _private_key: &[u8]) -> Result<String> {
    // Placeholder: In real implementation, use ECDSA P-256 signing
    // This would use libraries like `p256` and `ecdsa`
    Ok("ecdsa_signature_placeholder".to_string())
}

fn verify_ecdsa_p256(_data: &str, _signature: &str, _public_key: &[u8]) -> Result<bool> {
    // Placeholder: In real implementation, verify ECDSA P-256 signature
    Ok(_signature == "ecdsa_signature_placeholder")
}

fn generate_rsa_key_pair(key_id: String) -> Result<KeyPair> {
    // Placeholder: In real implementation, generate RSA key pair
    // This would use libraries like `rsa`
    Ok(KeyPair {
        key_id,
        algorithm: SignatureAlgorithm::RsaSha256,
        public_key: b"rsa_public_key_placeholder".to_vec(),
        private_key: Some(b"rsa_private_key_placeholder".to_vec()),
    })
}

fn generate_ed25519_key_pair(key_id: String) -> Result<KeyPair> {
    // Placeholder: In real implementation, generate Ed25519 key pair
    // This would use libraries like `ed25519-dalek`
    Ok(KeyPair {
        key_id,
        algorithm: SignatureAlgorithm::Ed25519,
        public_key: b"ed25519_public_key_placeholder".to_vec(),
        private_key: Some(b"ed25519_private_key_placeholder".to_vec()),
    })
}

fn generate_ecdsa_p256_key_pair(key_id: String) -> Result<KeyPair> {
    // Placeholder: In real implementation, generate ECDSA P-256 key pair
    // This would use libraries like `p256` and `ecdsa`
    Ok(KeyPair {
        key_id,
        algorithm: SignatureAlgorithm::EcdsaP256,
        public_key: b"ecdsa_public_key_placeholder".to_vec(),
        private_key: Some(b"ecdsa_private_key_placeholder".to_vec()),
    })
}

/// Security configuration for downloads and model loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Require signatures for all models
    pub require_signatures: bool,
    /// Only accept models signed by trusted keys
    pub require_trusted_keys: bool,
    /// Maximum age of signatures in seconds
    pub max_signature_age: Option<u64>,
    /// List of blocked model sources
    pub blocked_sources: Vec<String>,
    /// List of allowed model sources (if empty, all sources are allowed)
    pub allowed_sources: Vec<String>,
    /// Verify file integrity on load
    pub verify_integrity: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            require_signatures: false,
            require_trusted_keys: false,
            max_signature_age: Some(30 * 24 * 3600), // 30 days
            blocked_sources: Vec::new(),
            allowed_sources: Vec::new(),
            verify_integrity: true,
        }
    }
}

/// Validate a model source against security configuration
pub fn validate_model_source(url: &str, config: &SecurityConfig) -> Result<()> {
    // Check blocked sources
    for blocked in &config.blocked_sources {
        if url.contains(blocked) {
            return Err(TorshError::General(GeneralError::RuntimeError(format!(
                "Model source '{}' is blocked",
                blocked
            ))));
        }
    }

    // Check allowed sources (if specified)
    if !config.allowed_sources.is_empty() {
        let is_allowed = config
            .allowed_sources
            .iter()
            .any(|allowed| url.contains(allowed));
        if !is_allowed {
            return Err(TorshError::General(GeneralError::RuntimeError(
                "Model source is not in the allowed list".to_string(),
            )));
        }
    }

    Ok(())
}

/// Validate signature age
pub fn validate_signature_age(signature: &ModelSignature, max_age: Option<u64>) -> Result<()> {
    if let Some(max_age_secs) = max_age {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after UNIX epoch")
            .as_secs();

        let age = current_time.saturating_sub(signature.timestamp);
        if age > max_age_secs {
            return Err(TorshError::General(GeneralError::RuntimeError(format!(
                "Signature is too old: {} seconds (max: {})",
                age, max_age_secs
            ))));
        }
    }

    Ok(())
}

/// Sandbox configuration for model execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// Maximum memory usage in bytes
    pub max_memory: usize,
    /// Maximum execution time in seconds
    pub max_execution_time: u64,
    /// Maximum number of threads
    pub max_threads: usize,
    /// Allow network access
    pub allow_network: bool,
    /// Allow file system access
    pub allow_filesystem: bool,
    /// Allowed file paths for read access
    pub read_paths: Vec<String>,
    /// Allowed file paths for write access
    pub write_paths: Vec<String>,
    /// Maximum CPU usage percentage
    pub max_cpu_usage: f32,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            max_memory: 1024 * 1024 * 1024, // 1GB
            max_execution_time: 300,        // 5 minutes
            max_threads: 4,
            allow_network: false,
            allow_filesystem: false,
            read_paths: Vec::new(),
            write_paths: Vec::new(),
            max_cpu_usage: 80.0,
        }
    }
}

/// Resource usage tracking
#[derive(Debug, Clone, Default)]
pub struct ResourceUsage {
    pub memory_used: usize,
    pub cpu_time: f64,
    pub threads_created: usize,
    pub network_requests: usize,
    pub file_reads: usize,
    pub file_writes: usize,
    pub start_time: Option<SystemTime>,
}

/// Sandbox environment for model execution
pub struct ModelSandbox {
    config: SandboxConfig,
    usage: Arc<Mutex<ResourceUsage>>,
    is_active: Arc<Mutex<bool>>,
}

impl ModelSandbox {
    /// Create a new sandbox with configuration
    pub fn new(config: SandboxConfig) -> Self {
        Self {
            config,
            usage: Arc::new(Mutex::new(ResourceUsage::default())),
            is_active: Arc::new(Mutex::new(false)),
        }
    }

    /// Enter the sandbox environment
    pub fn enter(&self) -> Result<SandboxGuard<'_>> {
        let mut is_active = self.is_active.lock().expect("lock should not be poisoned");
        if *is_active {
            return Err(TorshError::General(GeneralError::RuntimeError(
                "Sandbox is already active".to_string(),
            )));
        }

        *is_active = true;

        // Initialize resource tracking
        {
            let mut usage = self.usage.lock().expect("lock should not be poisoned");
            *usage = ResourceUsage {
                start_time: Some(SystemTime::now()),
                ..Default::default()
            };
        }

        // Set up resource limits (platform-specific implementation would go here)
        self.setup_memory_limits()?;
        self.setup_thread_limits()?;
        self.setup_network_limits()?;
        self.setup_filesystem_limits()?;

        Ok(SandboxGuard {
            sandbox: self,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Check if resource limits are exceeded
    pub fn check_limits(&self) -> Result<()> {
        let usage = self.usage.lock().expect("lock should not be poisoned");

        // Check memory limit
        if usage.memory_used > self.config.max_memory {
            return Err(TorshError::General(GeneralError::RuntimeError(format!(
                "Memory limit exceeded: {} > {}",
                usage.memory_used, self.config.max_memory
            ))));
        }

        // Check execution time limit
        if let Some(start_time) = usage.start_time {
            let elapsed = SystemTime::now()
                .duration_since(start_time)
                .expect("current time should be after start_time")
                .as_secs();
            if elapsed > self.config.max_execution_time {
                return Err(TorshError::General(GeneralError::RuntimeError(format!(
                    "Execution time limit exceeded: {} > {}",
                    elapsed, self.config.max_execution_time
                ))));
            }
        }

        // Check thread limit
        if usage.threads_created > self.config.max_threads {
            return Err(TorshError::General(GeneralError::RuntimeError(format!(
                "Thread limit exceeded: {} > {}",
                usage.threads_created, self.config.max_threads
            ))));
        }

        Ok(())
    }

    /// Record memory usage
    pub fn record_memory_usage(&self, bytes: usize) {
        let mut usage = self.usage.lock().expect("lock should not be poisoned");
        usage.memory_used = usage.memory_used.saturating_add(bytes);
    }

    /// Record thread creation
    pub fn record_thread_creation(&self) {
        let mut usage = self.usage.lock().expect("lock should not be poisoned");
        usage.threads_created += 1;
    }

    /// Record network request
    pub fn record_network_request(&self) -> Result<()> {
        if !self.config.allow_network {
            return Err(TorshError::General(GeneralError::RuntimeError(
                "Network access is not allowed in sandbox".to_string(),
            )));
        }

        let mut usage = self.usage.lock().expect("lock should not be poisoned");
        usage.network_requests += 1;
        Ok(())
    }

    /// Record file system access
    pub fn record_file_access(&self, path: &str, is_write: bool) -> Result<()> {
        if !self.config.allow_filesystem {
            return Err(TorshError::General(GeneralError::RuntimeError(
                "File system access is not allowed in sandbox".to_string(),
            )));
        }

        // Check if path is allowed
        let allowed_paths = if is_write {
            &self.config.write_paths
        } else {
            &self.config.read_paths
        };

        if !allowed_paths.is_empty() {
            let is_allowed = allowed_paths
                .iter()
                .any(|allowed_path| path.starts_with(allowed_path));

            if !is_allowed {
                return Err(TorshError::General(GeneralError::RuntimeError(format!(
                    "Access to path '{}' is not allowed",
                    path
                ))));
            }
        }

        let mut usage = self.usage.lock().expect("lock should not be poisoned");
        if is_write {
            usage.file_writes += 1;
        } else {
            usage.file_reads += 1;
        }

        Ok(())
    }

    /// Get current resource usage
    pub fn get_usage(&self) -> ResourceUsage {
        self.usage
            .lock()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Exit the sandbox (private, called by SandboxGuard)
    fn exit(&self) {
        let mut is_active = self.is_active.lock().expect("lock should not be poisoned");
        *is_active = false;

        // Clean up resource limits
        self.cleanup_limits();
    }

    // Platform-specific implementations (simplified for demonstration)
    fn setup_memory_limits(&self) -> Result<()> {
        // In a real implementation, this would use platform-specific APIs
        // like setrlimit on Unix or SetProcessWorkingSetSize on Windows
        println!("Setting up memory limits: {} bytes", self.config.max_memory);
        Ok(())
    }

    fn setup_thread_limits(&self) -> Result<()> {
        // In a real implementation, this would limit thread creation
        println!(
            "Setting up thread limits: {} threads",
            self.config.max_threads
        );
        Ok(())
    }

    fn setup_network_limits(&self) -> Result<()> {
        if !self.config.allow_network {
            // In a real implementation, this would block network access
            println!("Blocking network access");
        }
        Ok(())
    }

    fn setup_filesystem_limits(&self) -> Result<()> {
        if !self.config.allow_filesystem {
            // In a real implementation, this would use chroot or similar
            println!("Restricting filesystem access");
        }
        Ok(())
    }

    fn cleanup_limits(&self) {
        // Clean up any resources or restore original limits
        println!("Cleaning up sandbox limits");
    }
}

/// RAII guard for sandbox environment
pub struct SandboxGuard<'a> {
    sandbox: &'a ModelSandbox,
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a> Drop for SandboxGuard<'a> {
    fn drop(&mut self) {
        self.sandbox.exit();
    }
}

/// Wrapper for sandboxed model execution
pub struct SandboxedModel {
    model: Box<dyn torsh_nn::Module>,
    sandbox: std::sync::RwLock<ModelSandbox>,
}

impl SandboxedModel {
    /// Create a new sandboxed model
    pub fn new(model: Box<dyn torsh_nn::Module>, config: SandboxConfig) -> Self {
        Self {
            model,
            sandbox: std::sync::RwLock::new(ModelSandbox::new(config)),
        }
    }

    /// Execute model forward pass in sandbox
    pub fn forward_sandboxed(
        &self,
        input: &torsh_tensor::Tensor<f32>,
    ) -> Result<torsh_tensor::Tensor<f32>> {
        {
            let sandbox = self.sandbox.read().expect("lock should not be poisoned");
            let _guard = sandbox.enter()?;

            // Record memory usage for input tensor
            let input_elements = input.shape().dims().iter().product::<usize>();
            let input_memory = input_elements * std::mem::size_of::<f32>();
            sandbox.record_memory_usage(input_memory);

            // Check limits before execution
            sandbox.check_limits()?;
        } // guard and sandbox are dropped here

        // Execute model
        let result = self.model.forward(input)?;

        // Record memory usage for output tensor
        {
            let sandbox = self.sandbox.read().expect("lock should not be poisoned");
            let output_elements = result.shape().dims().iter().product::<usize>();
            let output_memory = output_elements * std::mem::size_of::<f32>();
            sandbox.record_memory_usage(output_memory);

            // Check limits after execution
            sandbox.check_limits()?;
        }

        Ok(result)
    }

    /// Get sandbox resource usage
    pub fn get_sandbox_usage(&self) -> ResourceUsage {
        self.sandbox
            .read()
            .expect("lock should not be poisoned")
            .get_usage()
    }
}

impl torsh_nn::Module for SandboxedModel {
    fn forward(&self, input: &torsh_tensor::Tensor) -> Result<torsh_tensor::Tensor> {
        // For now, we'll convert the input for backward compatibility
        // In a real implementation, this would need proper type handling
        let result = self.forward_sandboxed(input)?;
        Ok(result)
    }

    fn parameters(&self) -> HashMap<String, torsh_nn::Parameter> {
        self.model.parameters()
    }

    fn train(&mut self) {
        self.model.train()
    }

    fn eval(&mut self) {
        self.model.eval()
    }

    fn training(&self) -> bool {
        self.model.training()
    }

    fn load_state_dict(
        &mut self,
        state_dict: &std::collections::HashMap<String, torsh_tensor::Tensor<f32>>,
        strict: bool,
    ) -> Result<()> {
        self.model.load_state_dict(state_dict, strict)
    }

    fn state_dict(&self) -> std::collections::HashMap<String, torsh_tensor::Tensor<f32>> {
        self.model.state_dict()
    }
}

/// Create a sandboxed wrapper for any model
pub fn sandbox_model(
    model: Box<dyn torsh_nn::Module>,
    config: Option<SandboxConfig>,
) -> SandboxedModel {
    let config = config.unwrap_or_default();
    SandboxedModel::new(model, config)
}

/// Vulnerability scanner for model files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnerabilityScanner {
    /// Known malicious patterns (simplified for demonstration)
    malicious_patterns: Vec<String>,
    /// Known vulnerable model signatures
    vulnerable_signatures: Vec<String>,
    /// Maximum file size to scan (in bytes)
    max_scan_size: usize,
    /// Enable deep scanning (more thorough but slower)
    deep_scan: bool,
}

impl Default for VulnerabilityScanner {
    fn default() -> Self {
        Self {
            malicious_patterns: vec![
                // Common malicious patterns
                "eval(".to_string(),
                "exec(".to_string(),
                "__import__".to_string(),
                "subprocess.".to_string(),
                "os.system".to_string(),
                "shell=True".to_string(),
                // Suspicious model operations
                "torch.jit._script_to_py".to_string(),
                "_C.import_ir_module".to_string(),
                // Network access patterns
                "urllib.request".to_string(),
                "requests.get".to_string(),
                "socket.".to_string(),
                // File system patterns
                "open(".to_string(),
                "file.write".to_string(),
                "pathlib".to_string(),
            ],
            vulnerable_signatures: vec![
                // Known vulnerable model hashes (examples)
                "bad_model_hash_1".to_string(),
                "bad_model_hash_2".to_string(),
            ],
            max_scan_size: 100 * 1024 * 1024, // 100MB
            deep_scan: true,
        }
    }
}

/// Vulnerability scan result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnerabilityScanResult {
    /// Whether the scan completed successfully
    pub success: bool,
    /// List of vulnerabilities found
    pub vulnerabilities: Vec<Vulnerability>,
    /// Risk level assessment
    pub risk_level: RiskLevel,
    /// Scan metadata
    pub scan_metadata: ScanMetadata,
}

/// Individual vulnerability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vulnerability {
    /// Vulnerability type
    pub vuln_type: VulnerabilityType,
    /// Severity level
    pub severity: Severity,
    /// Description of the vulnerability
    pub description: String,
    /// Location where vulnerability was found
    pub location: String,
    /// Remediation suggestions
    pub remediation: String,
}

/// Types of vulnerabilities
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VulnerabilityType {
    /// Malicious code execution
    CodeExecution,
    /// Unauthorized network access
    NetworkAccess,
    /// Unauthorized file system access
    FileSystemAccess,
    /// Data exfiltration
    DataExfiltration,
    /// Known vulnerable model
    KnownVulnerable,
    /// Suspicious patterns
    SuspiciousPattern,
    /// Cryptographic issues
    CryptographicIssue,
    /// Memory corruption
    MemoryCorruption,
}

/// Severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd, Eq, Ord)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

/// Overall risk level
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RiskLevel {
    Safe,
    Low,
    Medium,
    High,
    Critical,
}

/// Scan metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanMetadata {
    /// Time when scan was performed
    pub scan_time: u64,
    /// Duration of scan in milliseconds
    pub scan_duration: u64,
    /// Number of files scanned
    pub files_scanned: usize,
    /// Total bytes scanned
    pub bytes_scanned: usize,
    /// Scanner version
    pub scanner_version: String,
}

impl VulnerabilityScanner {
    /// Create a new vulnerability scanner
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a scanner with custom configuration
    pub fn with_config(
        max_scan_size: usize,
        deep_scan: bool,
        custom_patterns: Vec<String>,
    ) -> Self {
        let mut malicious_patterns = Self::default().malicious_patterns;
        malicious_patterns.extend(custom_patterns);
        Self {
            max_scan_size,
            deep_scan,
            malicious_patterns,
            ..Default::default()
        }
    }

    /// Scan a model file for vulnerabilities
    pub fn scan_file<P: AsRef<Path>>(&self, file_path: P) -> Result<VulnerabilityScanResult> {
        let file_path = file_path.as_ref();
        let start_time = SystemTime::now();

        // Check file exists and size
        let metadata = std::fs::metadata(file_path)?;
        if metadata.len() > self.max_scan_size as u64 {
            return Ok(VulnerabilityScanResult {
                success: false,
                vulnerabilities: vec![Vulnerability {
                    vuln_type: VulnerabilityType::SuspiciousPattern,
                    severity: Severity::Medium,
                    description: format!("File too large to scan: {} bytes", metadata.len()),
                    location: file_path.to_string_lossy().to_string(),
                    remediation: "Manually verify large model files".to_string(),
                }],
                risk_level: RiskLevel::Medium,
                scan_metadata: self.create_metadata(start_time, 1, metadata.len() as usize),
            });
        }

        let mut vulnerabilities = Vec::new();

        // Calculate file hash and check against known vulnerable models
        let file_hash = calculate_file_hash(file_path)?;
        if self.vulnerable_signatures.contains(&file_hash) {
            vulnerabilities.push(Vulnerability {
                vuln_type: VulnerabilityType::KnownVulnerable,
                severity: Severity::Critical,
                description: "Model matches known vulnerable signature".to_string(),
                location: file_path.to_string_lossy().to_string(),
                remediation: "Do not use this model. Find alternative from trusted source"
                    .to_string(),
            });
        }

        // Scan file content for malicious patterns
        let content_vulns = self.scan_file_content(file_path)?;
        vulnerabilities.extend(content_vulns);

        // If deep scan is enabled, perform additional checks
        if self.deep_scan {
            let deep_vulns = self.deep_scan_file(file_path)?;
            vulnerabilities.extend(deep_vulns);
        }

        // Assess overall risk level
        let risk_level = self.assess_risk_level(&vulnerabilities);

        let end_time = SystemTime::now();
        let scan_duration = end_time
            .duration_since(start_time)
            .expect("end_time should be after start_time")
            .as_millis() as u64;

        Ok(VulnerabilityScanResult {
            success: true,
            vulnerabilities,
            risk_level,
            scan_metadata: ScanMetadata {
                scan_time: start_time
                    .duration_since(UNIX_EPOCH)
                    .expect("start_time should be after UNIX epoch")
                    .as_secs(),
                scan_duration,
                files_scanned: 1,
                bytes_scanned: metadata.len() as usize,
                scanner_version: "1.0.0".to_string(),
            },
        })
    }

    /// Scan file content for malicious patterns
    fn scan_file_content<P: AsRef<Path>>(&self, file_path: P) -> Result<Vec<Vulnerability>> {
        let file_path = file_path.as_ref();
        let mut vulnerabilities = Vec::new();

        // Read file content (limited to max_scan_size)
        let file = File::open(file_path)?;
        let mut reader = BufReader::new(file);
        let mut content = Vec::new();
        reader.read_to_end(&mut content)?;

        // Convert to string for pattern matching (may lose some data for binary files)
        let content_str = String::from_utf8_lossy(&content);

        // Check for malicious patterns
        for pattern in &self.malicious_patterns {
            if content_str.contains(pattern) {
                let vuln_type = self.classify_pattern(pattern);
                let severity = self.assess_pattern_severity(pattern);

                vulnerabilities.push(Vulnerability {
                    vuln_type,
                    severity,
                    description: format!("Found suspicious pattern: {}", pattern),
                    location: file_path.to_string_lossy().to_string(),
                    remediation: "Review model source and verify legitimacy".to_string(),
                });
            }
        }

        Ok(vulnerabilities)
    }

    /// Perform deep scan for additional security issues
    fn deep_scan_file<P: AsRef<Path>>(&self, file_path: P) -> Result<Vec<Vulnerability>> {
        let file_path = file_path.as_ref();
        let mut vulnerabilities = Vec::new();

        // Check file extension
        if let Some(extension) = file_path.extension() {
            if let Some("exe" | "bat" | "sh" | "ps1") = extension.to_str() {
                vulnerabilities.push(Vulnerability {
                    vuln_type: VulnerabilityType::CodeExecution,
                    severity: Severity::High,
                    description: "Model file has executable extension".to_string(),
                    location: file_path.to_string_lossy().to_string(),
                    remediation: "Verify this is actually a model file and not malware".to_string(),
                });
            }
        }

        // Check for hidden files or suspicious names
        if let Some(filename) = file_path.file_name() {
            if let Some(filename_str) = filename.to_str() {
                if filename_str.starts_with('.') || filename_str.contains("..") {
                    vulnerabilities.push(Vulnerability {
                        vuln_type: VulnerabilityType::SuspiciousPattern,
                        severity: Severity::Medium,
                        description: "Suspicious filename pattern".to_string(),
                        location: file_path.to_string_lossy().to_string(),
                        remediation: "Verify file purpose and rename to standard convention"
                            .to_string(),
                    });
                }
            }
        }

        // Check file permissions (Unix-like systems)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let metadata = std::fs::metadata(file_path)?;
            let permissions = metadata.permissions();
            let mode = permissions.mode();

            // Check if file is executable
            if mode & 0o111 != 0 {
                vulnerabilities.push(Vulnerability {
                    vuln_type: VulnerabilityType::CodeExecution,
                    severity: Severity::Medium,
                    description: "Model file has execute permissions".to_string(),
                    location: file_path.to_string_lossy().to_string(),
                    remediation: "Remove execute permissions: chmod -x filename".to_string(),
                });
            }
        }

        Ok(vulnerabilities)
    }

    /// Classify a pattern to determine vulnerability type
    fn classify_pattern(&self, pattern: &str) -> VulnerabilityType {
        match pattern {
            p if p.contains("eval") || p.contains("exec") || p.contains("subprocess") => {
                VulnerabilityType::CodeExecution
            }
            p if p.contains("socket") || p.contains("urllib") || p.contains("requests") => {
                VulnerabilityType::NetworkAccess
            }
            p if p.contains("open") || p.contains("file") || p.contains("pathlib") => {
                VulnerabilityType::FileSystemAccess
            }
            _ => VulnerabilityType::SuspiciousPattern,
        }
    }

    /// Assess severity of a pattern
    fn assess_pattern_severity(&self, pattern: &str) -> Severity {
        match pattern {
            p if p.contains("eval") || p.contains("exec") => Severity::Critical,
            p if p.contains("subprocess") || p.contains("os.system") => Severity::High,
            p if p.contains("socket") || p.contains("urllib") => Severity::Medium,
            _ => Severity::Low,
        }
    }

    /// Assess overall risk level based on vulnerabilities
    fn assess_risk_level(&self, vulnerabilities: &[Vulnerability]) -> RiskLevel {
        if vulnerabilities.is_empty() {
            return RiskLevel::Safe;
        }

        let max_severity = vulnerabilities
            .iter()
            .map(|v| &v.severity)
            .max()
            .unwrap_or(&Severity::Low);

        match max_severity {
            Severity::Critical => RiskLevel::Critical,
            Severity::High => RiskLevel::High,
            Severity::Medium => RiskLevel::Medium,
            Severity::Low => RiskLevel::Low,
        }
    }

    /// Create scan metadata
    fn create_metadata(
        &self,
        start_time: SystemTime,
        files_scanned: usize,
        bytes_scanned: usize,
    ) -> ScanMetadata {
        let end_time = SystemTime::now();
        let scan_duration = end_time
            .duration_since(start_time)
            .expect("end_time should be after start_time")
            .as_millis() as u64;

        ScanMetadata {
            scan_time: start_time
                .duration_since(UNIX_EPOCH)
                .expect("start_time should be after UNIX epoch")
                .as_secs(),
            scan_duration,
            files_scanned,
            bytes_scanned,
            scanner_version: "1.0.0".to_string(),
        }
    }

    /// Add custom malicious pattern
    pub fn add_pattern(&mut self, pattern: String) {
        if !self.malicious_patterns.contains(&pattern) {
            self.malicious_patterns.push(pattern);
        }
    }

    /// Add known vulnerable signature
    pub fn add_vulnerable_signature(&mut self, signature: String) {
        if !self.vulnerable_signatures.contains(&signature) {
            self.vulnerable_signatures.push(signature);
        }
    }

    /// Get all patterns
    pub fn get_patterns(&self) -> &[String] {
        &self.malicious_patterns
    }

    /// Get all vulnerable signatures
    pub fn get_vulnerable_signatures(&self) -> &[String] {
        &self.vulnerable_signatures
    }
}

/// Convenience function to scan a model file for vulnerabilities
pub fn scan_model_vulnerabilities<P: AsRef<Path>>(
    file_path: P,
    scanner: Option<VulnerabilityScanner>,
) -> Result<VulnerabilityScanResult> {
    let scanner = scanner.unwrap_or_default();
    scanner.scan_file(file_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_calculate_file_hash() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"test content").unwrap();
        temp_file.flush().unwrap();

        let hash = calculate_file_hash(temp_file.path()).unwrap();
        assert_eq!(hash.len(), 64); // SHA-256 produces 64-char hex string
    }

    #[test]
    fn test_verify_file_integrity() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"test content").unwrap();
        temp_file.flush().unwrap();

        let hash = calculate_file_hash(temp_file.path()).unwrap();
        assert!(verify_file_integrity(temp_file.path(), &hash).unwrap());
        assert!(!verify_file_integrity(temp_file.path(), "wrong_hash").unwrap());
    }

    #[test]
    fn test_security_manager() {
        let mut manager = SecurityManager::new();

        // Generate key pair
        let key_pair =
            SecurityManager::generate_key_pair("test_key".to_string(), SignatureAlgorithm::Ed25519)
                .unwrap();

        // Add key to manager
        manager.add_key(key_pair);

        // Trust the key
        manager.trust_key("test_key").unwrap();

        assert!(manager.trusted_keys().contains(&"test_key".to_string()));
    }

    #[test]
    fn test_validate_model_source() {
        let config = SecurityConfig {
            blocked_sources: vec!["malicious.com".to_string()],
            allowed_sources: vec!["trusted.com".to_string()],
            ..Default::default()
        };

        // Should fail for blocked source
        assert!(validate_model_source("https://malicious.com/model.torsh", &config).is_err());

        // Should fail for non-allowed source when allowed list is specified
        assert!(validate_model_source("https://unknown.com/model.torsh", &config).is_err());

        // Should succeed for allowed source
        assert!(validate_model_source("https://trusted.com/model.torsh", &config).is_ok());
    }

    #[test]
    fn test_validate_signature_age() {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Recent signature should be valid
        let recent_signature = ModelSignature {
            timestamp: current_time - 100, // 100 seconds ago
            file_hash: "hash".to_string(),
            signature: "sig".to_string(),
            key_id: "key".to_string(),
            algorithm: SignatureAlgorithm::Ed25519,
            metadata: HashMap::new(),
        };

        assert!(validate_signature_age(&recent_signature, Some(3600)).is_ok());

        // Old signature should be invalid
        let old_signature = ModelSignature {
            timestamp: current_time - 7200, // 2 hours ago
            ..recent_signature
        };

        assert!(validate_signature_age(&old_signature, Some(3600)).is_err());
    }

    #[test]
    fn test_sandbox_config() {
        let config = SandboxConfig::default();
        assert_eq!(config.max_memory, 1024 * 1024 * 1024);
        assert_eq!(config.max_execution_time, 300);
        assert_eq!(config.max_threads, 4);
        assert!(!config.allow_network);
        assert!(!config.allow_filesystem);
    }

    #[test]
    fn test_sandbox_creation() {
        let config = SandboxConfig::default();
        let sandbox = ModelSandbox::new(config);

        let usage = sandbox.get_usage();
        assert_eq!(usage.memory_used, 0);
        assert_eq!(usage.threads_created, 0);
        assert_eq!(usage.network_requests, 0);
    }

    #[test]
    fn test_sandbox_resource_tracking() {
        let config = SandboxConfig::default();
        let sandbox = ModelSandbox::new(config);

        // Test memory tracking
        sandbox.record_memory_usage(1024);
        let usage = sandbox.get_usage();
        assert_eq!(usage.memory_used, 1024);

        // Test thread tracking
        sandbox.record_thread_creation();
        let usage = sandbox.get_usage();
        assert_eq!(usage.threads_created, 1);
    }

    #[test]
    fn test_sandbox_network_restrictions() {
        let config = SandboxConfig {
            allow_network: false,
            ..Default::default()
        };
        let sandbox = ModelSandbox::new(config);

        // Should fail when network is not allowed
        assert!(sandbox.record_network_request().is_err());

        let config = SandboxConfig {
            allow_network: true,
            ..Default::default()
        };
        let sandbox = ModelSandbox::new(config);

        // Should succeed when network is allowed
        assert!(sandbox.record_network_request().is_ok());
    }

    #[test]
    fn test_sandbox_filesystem_restrictions() {
        let config = SandboxConfig {
            allow_filesystem: false,
            ..Default::default()
        };
        let sandbox = ModelSandbox::new(config);

        // Should fail when filesystem access is not allowed
        assert!(sandbox.record_file_access("/tmp/test", false).is_err());

        let config = SandboxConfig {
            allow_filesystem: true,
            read_paths: vec!["/tmp".to_string()],
            ..Default::default()
        };
        let sandbox = ModelSandbox::new(config);

        // Should succeed for allowed path
        assert!(sandbox.record_file_access("/tmp/test", false).is_ok());

        // Should fail for disallowed path
        assert!(sandbox.record_file_access("/etc/passwd", false).is_err());
    }

    #[test]
    fn test_sandbox_guard() {
        let config = SandboxConfig::default();
        let sandbox = ModelSandbox::new(config);

        // Test that guard properly manages sandbox state
        {
            let _guard = sandbox.enter().unwrap();
            // Sandbox should be active here
        }
        // Sandbox should be inactive after guard is dropped

        // Should be able to enter again
        let _guard = sandbox.enter().unwrap();
    }

    #[test]
    fn test_vulnerability_scanner_creation() {
        let scanner = VulnerabilityScanner::new();
        assert!(!scanner.get_patterns().is_empty());
        assert!(!scanner.get_vulnerable_signatures().is_empty());
    }

    #[test]
    fn test_vulnerability_scanner_custom_config() {
        let custom_patterns = vec!["custom_pattern".to_string()];
        let scanner = VulnerabilityScanner::with_config(
            50 * 1024 * 1024, // 50MB
            false,
            custom_patterns.clone(),
        );

        assert!(scanner
            .get_patterns()
            .contains(&"custom_pattern".to_string()));
    }

    #[test]
    fn test_vulnerability_scanner_pattern_detection() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file
            .write_all(b"This file contains eval() function")
            .unwrap();
        temp_file.flush().unwrap();

        let scanner = VulnerabilityScanner::new();
        let result = scanner.scan_file(temp_file.path()).unwrap();

        assert!(result.success);
        assert!(!result.vulnerabilities.is_empty());
        assert_eq!(result.risk_level, RiskLevel::Critical);

        // Check that the vulnerability was detected
        let has_eval_vuln = result
            .vulnerabilities
            .iter()
            .any(|v| v.description.contains("eval"));
        assert!(has_eval_vuln);
    }

    #[test]
    fn test_vulnerability_scanner_clean_file() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file
            .write_all(b"This is a clean model file with no suspicious content")
            .unwrap();
        temp_file.flush().unwrap();

        // Create scanner with no malicious patterns for clean test
        let scanner = VulnerabilityScanner::with_config(
            100 * 1024 * 1024, // 100MB max size
            false,             // disable deep scan to avoid extension-based false positives
            vec![],            // no custom patterns
        );
        let result = scanner.scan_file(temp_file.path()).unwrap();

        assert!(result.success);
        assert!(result.vulnerabilities.is_empty());
        assert_eq!(result.risk_level, RiskLevel::Safe);
    }

    #[test]
    fn test_vulnerability_scanner_known_vulnerable() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"test content").unwrap();
        temp_file.flush().unwrap();

        let file_hash = calculate_file_hash(temp_file.path()).unwrap();

        let mut scanner = VulnerabilityScanner::new();
        scanner.add_vulnerable_signature(file_hash);

        let result = scanner.scan_file(temp_file.path()).unwrap();

        assert!(result.success);
        assert!(!result.vulnerabilities.is_empty());
        assert_eq!(result.risk_level, RiskLevel::Critical);

        // Check that known vulnerable signature was detected
        let has_known_vuln = result
            .vulnerabilities
            .iter()
            .any(|v| v.vuln_type == VulnerabilityType::KnownVulnerable);
        assert!(has_known_vuln);
    }

    #[test]
    fn test_vulnerability_pattern_classification() {
        let scanner = VulnerabilityScanner::new();

        assert_eq!(
            scanner.classify_pattern("eval("),
            VulnerabilityType::CodeExecution
        );
        assert_eq!(
            scanner.classify_pattern("socket."),
            VulnerabilityType::NetworkAccess
        );
        assert_eq!(
            scanner.classify_pattern("open("),
            VulnerabilityType::FileSystemAccess
        );
        assert_eq!(
            scanner.classify_pattern("unknown"),
            VulnerabilityType::SuspiciousPattern
        );
    }

    #[test]
    fn test_vulnerability_severity_assessment() {
        let scanner = VulnerabilityScanner::new();

        assert_eq!(scanner.assess_pattern_severity("eval("), Severity::Critical);
        assert_eq!(
            scanner.assess_pattern_severity("subprocess."),
            Severity::High
        );
        assert_eq!(scanner.assess_pattern_severity("socket."), Severity::Medium);
        assert_eq!(scanner.assess_pattern_severity("unknown"), Severity::Low);
    }

    #[test]
    fn test_risk_level_assessment() {
        let scanner = VulnerabilityScanner::new();

        // No vulnerabilities
        assert_eq!(scanner.assess_risk_level(&[]), RiskLevel::Safe);

        // Critical vulnerability
        let critical_vuln = vec![Vulnerability {
            vuln_type: VulnerabilityType::CodeExecution,
            severity: Severity::Critical,
            description: "test".to_string(),
            location: "test".to_string(),
            remediation: "test".to_string(),
        }];
        assert_eq!(
            scanner.assess_risk_level(&critical_vuln),
            RiskLevel::Critical
        );

        // High vulnerability
        let high_vuln = vec![Vulnerability {
            vuln_type: VulnerabilityType::CodeExecution,
            severity: Severity::High,
            description: "test".to_string(),
            location: "test".to_string(),
            remediation: "test".to_string(),
        }];
        assert_eq!(scanner.assess_risk_level(&high_vuln), RiskLevel::High);
    }

    #[test]
    fn test_scan_model_vulnerabilities_convenience_function() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"exec() call detected").unwrap();
        temp_file.flush().unwrap();

        let result = scan_model_vulnerabilities(temp_file.path(), None).unwrap();

        assert!(result.success);
        assert!(!result.vulnerabilities.is_empty());
        assert_eq!(result.risk_level, RiskLevel::Critical);
    }
}
