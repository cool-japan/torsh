//! Security features for package signing and encryption
//!
//! This module provides cryptographic security features including:
//! - Digital signatures for package integrity and authenticity
//! - Encryption for sensitive model packages
//! - Key management and verification

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use ring::aead;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use torsh_core::error::{Result, TorshError};

use crate::package::Package;

/// Package signature information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageSignature {
    /// Signature algorithm used
    pub algorithm: SignatureAlgorithm,
    /// Signature data
    pub signature: Vec<u8>,
    /// Public key for verification
    pub public_key: Vec<u8>,
    /// Timestamp when signature was created
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Additional signature metadata
    pub metadata: HashMap<String, String>,
}

/// Supported signature algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SignatureAlgorithm {
    /// Ed25519 digital signature
    Ed25519,
    /// RSA signature (future support)
    Rsa,
    /// ECDSA signature (future support)
    Ecdsa,
}

/// Encryption algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    /// AES-256-GCM authenticated encryption
    Aes256Gcm,
    /// ChaCha20-Poly1305 authenticated encryption
    ChaCha20Poly1305,
}

/// Encrypted package container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedPackage {
    /// Encryption algorithm used
    pub algorithm: EncryptionAlgorithm,
    /// Encrypted package data
    pub encrypted_data: Vec<u8>,
    /// Nonce/IV for decryption
    pub nonce: Vec<u8>,
    /// Additional authenticated data (AAD)
    pub aad: Option<Vec<u8>>,
    /// Key derivation metadata
    pub key_metadata: HashMap<String, String>,
    /// Timestamp of encryption
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Package signer for creating and verifying signatures
pub struct PackageSigner {
    signing_key: Option<SigningKey>,
    verifying_keys: Vec<VerifyingKey>,
    algorithm: SignatureAlgorithm,
}

/// Package encryptor for encrypting and decrypting packages
pub struct PackageEncryptor {
    algorithm: EncryptionAlgorithm,
}

/// Security error types
#[derive(Debug, thiserror::Error)]
pub enum SecurityError {
    /// Signature verification failed
    #[error("Signature verification failed: {0}")]
    InvalidSignature(String),
    /// Encryption/decryption error
    #[error("Encryption error: {0}")]
    EncryptionError(String),
    /// Key management error
    #[error("Key error: {0}")]
    KeyError(String),
}

impl PackageSigner {
    /// Create a new package signer with generated key pair
    pub fn new() -> Self {
        // Generate random bytes for the signing key
        use ring::rand::{SecureRandom, SystemRandom};
        let rng = SystemRandom::new();
        let mut secret_bytes = [0u8; 32];
        rng.fill(&mut secret_bytes)
            .expect("Failed to generate random key");

        let signing_key = SigningKey::from_bytes(&secret_bytes);

        Self {
            signing_key: Some(signing_key),
            verifying_keys: Vec::new(),
            algorithm: SignatureAlgorithm::Ed25519,
        }
    }

    /// Create a signer from an existing signing key
    pub fn from_signing_key(signing_key: SigningKey) -> Self {
        Self {
            signing_key: Some(signing_key),
            verifying_keys: Vec::new(),
            algorithm: SignatureAlgorithm::Ed25519,
        }
    }

    /// Create a verifier (no signing capability)
    pub fn verifier_only() -> Self {
        Self {
            signing_key: None,
            verifying_keys: Vec::new(),
            algorithm: SignatureAlgorithm::Ed25519,
        }
    }

    /// Add a trusted public key for verification
    pub fn add_trusted_key(&mut self, public_key: &[u8]) -> Result<()> {
        let verifying_key =
            VerifyingKey::from_bytes(public_key.try_into().map_err(|_| {
                TorshError::InvalidArgument("Invalid public key length".to_string())
            })?)
            .map_err(|e| TorshError::InvalidArgument(format!("Invalid public key: {}", e)))?;

        self.verifying_keys.push(verifying_key);
        Ok(())
    }

    /// Get public key for verification
    pub fn public_key(&self) -> Option<Vec<u8>> {
        self.signing_key
            .as_ref()
            .map(|sk| sk.verifying_key().to_bytes().to_vec())
    }

    /// Export signing key (use with caution!)
    pub fn export_signing_key(&self) -> Option<Vec<u8>> {
        self.signing_key.as_ref().map(|sk| sk.to_bytes().to_vec())
    }

    /// Sign a package
    pub fn sign_package(&self, package: &Package) -> Result<PackageSignature> {
        let signing_key = self
            .signing_key
            .as_ref()
            .ok_or_else(|| TorshError::InvalidArgument("No signing key available".to_string()))?;

        // Serialize package for signing
        let package_data = self.package_digest(package)?;

        // Create signature
        let signature = signing_key.sign(&package_data);

        let mut metadata = HashMap::new();
        metadata.insert("package_name".to_string(), package.name().to_string());
        metadata.insert(
            "package_version".to_string(),
            package.get_version().to_string(),
        );

        Ok(PackageSignature {
            algorithm: self.algorithm,
            signature: signature.to_bytes().to_vec(),
            public_key: signing_key.verifying_key().to_bytes().to_vec(),
            timestamp: chrono::Utc::now(),
            metadata,
        })
    }

    /// Verify package signature
    pub fn verify_package(&self, package: &Package, signature: &PackageSignature) -> Result<bool> {
        if signature.algorithm != self.algorithm {
            return Err(TorshError::InvalidArgument(format!(
                "Unsupported signature algorithm: {:?}",
                signature.algorithm
            )));
        }

        // Get verifying key from signature
        let verifying_key =
            VerifyingKey::from_bytes(signature.public_key.as_slice().try_into().map_err(|_| {
                TorshError::InvalidArgument("Invalid public key length".to_string())
            })?)
            .map_err(|e| TorshError::InvalidArgument(format!("Invalid public key: {}", e)))?;

        // Check if the key is trusted
        if !self.verifying_keys.is_empty()
            && !self
                .verifying_keys
                .iter()
                .any(|k| k.to_bytes() == verifying_key.to_bytes())
        {
            return Ok(false);
        }

        // Compute package digest
        let package_data = self.package_digest(package)?;

        // Parse signature
        let sig =
            Signature::from_bytes(signature.signature.as_slice().try_into().map_err(|_| {
                TorshError::InvalidArgument("Invalid signature length".to_string())
            })?);

        // Verify signature
        match verifying_key.verify(&package_data, &sig) {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Compute package digest for signing
    fn package_digest(&self, package: &Package) -> Result<Vec<u8>> {
        // Create a deterministic digest of the package
        let mut hasher = Sha256::new();

        // Hash package metadata
        hasher.update(package.name().as_bytes());
        hasher.update(package.get_version().as_bytes());

        // Hash resources in sorted order for determinism
        let mut resource_names: Vec<_> = package.resources().keys().collect();
        resource_names.sort();

        for name in resource_names {
            if let Some(resource) = package.resources().get(name) {
                hasher.update(name.as_bytes());
                hasher.update(&resource.data);
            }
        }

        Ok(hasher.finalize().to_vec())
    }

    /// Save signature to file
    pub fn save_signature<P: AsRef<Path>>(signature: &PackageSignature, path: P) -> Result<()> {
        let serialized = oxicode::serde::encode_to_vec(signature, oxicode::config::standard())
            .map_err(|e| TorshError::SerializationError(e.to_string()))?;

        fs::write(path, serialized).map_err(|e| TorshError::IoError(e.to_string()))?;
        Ok(())
    }

    /// Load signature from file
    pub fn load_signature<P: AsRef<Path>>(path: P) -> Result<PackageSignature> {
        let data = fs::read(path).map_err(|e| TorshError::IoError(e.to_string()))?;

        let (signature, _) = oxicode::serde::decode_from_slice(&data, oxicode::config::standard())
            .map_err(|e| TorshError::SerializationError(e.to_string()))?;

        Ok(signature)
    }
}

impl Default for PackageSigner {
    fn default() -> Self {
        Self::new()
    }
}

impl PackageEncryptor {
    /// Create a new package encryptor
    pub fn new(algorithm: EncryptionAlgorithm) -> Self {
        Self { algorithm }
    }

    /// Encrypt a package with a password
    pub fn encrypt_package_with_password(
        &self,
        package: &Package,
        password: &str,
    ) -> Result<EncryptedPackage> {
        // Serialize package
        let package_data = oxicode::serde::encode_to_vec(package, oxicode::config::standard())
            .map_err(|e| TorshError::SerializationError(e.to_string()))?;

        // Derive key from password using PBKDF2
        let salt = self.generate_salt();
        let key = self.derive_key_from_password(password, &salt)?;

        // Encrypt data
        let (encrypted_data, nonce) = self.encrypt_data(&package_data, &key)?;

        let mut key_metadata = HashMap::new();
        key_metadata.insert("kdf".to_string(), "pbkdf2".to_string());
        key_metadata.insert("salt".to_string(), hex::encode(&salt));
        key_metadata.insert("iterations".to_string(), "100000".to_string());

        Ok(EncryptedPackage {
            algorithm: self.algorithm,
            encrypted_data,
            nonce,
            aad: None,
            key_metadata,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Decrypt a package with a password
    pub fn decrypt_package_with_password(
        &self,
        encrypted: &EncryptedPackage,
        password: &str,
    ) -> Result<Package> {
        // Extract salt from metadata
        let salt_hex = encrypted
            .key_metadata
            .get("salt")
            .ok_or_else(|| TorshError::InvalidArgument("Missing salt in metadata".to_string()))?;

        let salt = hex::decode(salt_hex)
            .map_err(|e| TorshError::InvalidArgument(format!("Invalid salt: {}", e)))?;

        // Derive key from password
        let key = self.derive_key_from_password(password, &salt)?;

        // Decrypt data
        let decrypted_data =
            self.decrypt_data(&encrypted.encrypted_data, &encrypted.nonce, &key)?;

        // Deserialize package
        let (package, _) =
            oxicode::serde::decode_from_slice(&decrypted_data, oxicode::config::standard())
                .map_err(|e| TorshError::SerializationError(e.to_string()))?;

        Ok(package)
    }

    /// Encrypt data with AES-256-GCM
    fn encrypt_data(&self, data: &[u8], key: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
        match self.algorithm {
            EncryptionAlgorithm::Aes256Gcm => self.encrypt_aes_gcm(data, key),
            EncryptionAlgorithm::ChaCha20Poly1305 => self.encrypt_chacha20(data, key),
        }
    }

    /// Decrypt data
    fn decrypt_data(&self, encrypted: &[u8], nonce: &[u8], key: &[u8]) -> Result<Vec<u8>> {
        match self.algorithm {
            EncryptionAlgorithm::Aes256Gcm => self.decrypt_aes_gcm(encrypted, nonce, key),
            EncryptionAlgorithm::ChaCha20Poly1305 => self.decrypt_chacha20(encrypted, nonce, key),
        }
    }

    /// Encrypt with AES-256-GCM
    fn encrypt_aes_gcm(&self, data: &[u8], key: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
        let unbound_key = aead::UnboundKey::new(&aead::AES_256_GCM, key)
            .map_err(|_| TorshError::InvalidArgument("Invalid key for AES-256-GCM".to_string()))?;

        let nonce_bytes = self.generate_nonce(aead::NONCE_LEN);
        let nonce = aead::Nonce::try_assume_unique_for_key(&nonce_bytes)
            .map_err(|_| TorshError::InvalidArgument("Invalid nonce".to_string()))?;

        let sealing_key = aead::LessSafeKey::new(unbound_key);

        let mut in_out = data.to_vec();
        sealing_key
            .seal_in_place_append_tag(nonce, aead::Aad::empty(), &mut in_out)
            .map_err(|_| TorshError::InvalidArgument("Encryption failed".to_string()))?;

        Ok((in_out, nonce_bytes))
    }

    /// Decrypt with AES-256-GCM
    fn decrypt_aes_gcm(&self, encrypted: &[u8], nonce: &[u8], key: &[u8]) -> Result<Vec<u8>> {
        let unbound_key = aead::UnboundKey::new(&aead::AES_256_GCM, key)
            .map_err(|_| TorshError::InvalidArgument("Invalid key for AES-256-GCM".to_string()))?;

        let nonce = aead::Nonce::try_assume_unique_for_key(nonce)
            .map_err(|_| TorshError::InvalidArgument("Invalid nonce".to_string()))?;

        let opening_key = aead::LessSafeKey::new(unbound_key);

        let mut in_out = encrypted.to_vec();
        let decrypted = opening_key
            .open_in_place(nonce, aead::Aad::empty(), &mut in_out)
            .map_err(|_| TorshError::InvalidArgument("Decryption failed".to_string()))?;

        Ok(decrypted.to_vec())
    }

    /// Encrypt with ChaCha20-Poly1305
    fn encrypt_chacha20(&self, data: &[u8], key: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
        let unbound_key = aead::UnboundKey::new(&aead::CHACHA20_POLY1305, key).map_err(|_| {
            TorshError::InvalidArgument("Invalid key for ChaCha20-Poly1305".to_string())
        })?;

        let nonce_bytes = self.generate_nonce(aead::NONCE_LEN);
        let nonce = aead::Nonce::try_assume_unique_for_key(&nonce_bytes)
            .map_err(|_| TorshError::InvalidArgument("Invalid nonce".to_string()))?;

        let sealing_key = aead::LessSafeKey::new(unbound_key);

        let mut in_out = data.to_vec();
        sealing_key
            .seal_in_place_append_tag(nonce, aead::Aad::empty(), &mut in_out)
            .map_err(|_| TorshError::InvalidArgument("Encryption failed".to_string()))?;

        Ok((in_out, nonce_bytes))
    }

    /// Decrypt with ChaCha20-Poly1305
    fn decrypt_chacha20(&self, encrypted: &[u8], nonce: &[u8], key: &[u8]) -> Result<Vec<u8>> {
        let unbound_key = aead::UnboundKey::new(&aead::CHACHA20_POLY1305, key).map_err(|_| {
            TorshError::InvalidArgument("Invalid key for ChaCha20-Poly1305".to_string())
        })?;

        let nonce = aead::Nonce::try_assume_unique_for_key(nonce)
            .map_err(|_| TorshError::InvalidArgument("Invalid nonce".to_string()))?;

        let opening_key = aead::LessSafeKey::new(unbound_key);

        let mut in_out = encrypted.to_vec();
        let decrypted = opening_key
            .open_in_place(nonce, aead::Aad::empty(), &mut in_out)
            .map_err(|_| TorshError::InvalidArgument("Decryption failed".to_string()))?;

        Ok(decrypted.to_vec())
    }

    /// Derive encryption key from password using PBKDF2
    fn derive_key_from_password(&self, password: &str, salt: &[u8]) -> Result<Vec<u8>> {
        use ring::pbkdf2;

        let iterations =
            std::num::NonZeroU32::new(100_000).expect("100_000 is a valid non-zero u32");
        let mut key = vec![0u8; 32]; // 256-bit key

        pbkdf2::derive(
            pbkdf2::PBKDF2_HMAC_SHA256,
            iterations,
            salt,
            password.as_bytes(),
            &mut key,
        );

        Ok(key)
    }

    /// Generate random salt
    fn generate_salt(&self) -> Vec<u8> {
        use ring::rand::{SecureRandom, SystemRandom};

        let rng = SystemRandom::new();
        let mut salt = vec![0u8; 32];
        rng.fill(&mut salt).expect("Failed to generate salt");
        salt
    }

    /// Generate random nonce
    fn generate_nonce(&self, len: usize) -> Vec<u8> {
        use ring::rand::{SecureRandom, SystemRandom};

        let rng = SystemRandom::new();
        let mut nonce = vec![0u8; len];
        rng.fill(&mut nonce).expect("Failed to generate nonce");
        nonce
    }

    /// Save encrypted package to file
    pub fn save_encrypted<P: AsRef<Path>>(encrypted: &EncryptedPackage, path: P) -> Result<()> {
        let serialized = oxicode::serde::encode_to_vec(encrypted, oxicode::config::standard())
            .map_err(|e| TorshError::SerializationError(e.to_string()))?;

        fs::write(path, serialized).map_err(|e| TorshError::IoError(e.to_string()))?;
        Ok(())
    }

    /// Load encrypted package from file
    pub fn load_encrypted<P: AsRef<Path>>(path: P) -> Result<EncryptedPackage> {
        let data = fs::read(path).map_err(|e| TorshError::IoError(e.to_string()))?;

        let (encrypted, _) = oxicode::serde::decode_from_slice(&data, oxicode::config::standard())
            .map_err(|e| TorshError::SerializationError(e.to_string()))?;

        Ok(encrypted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_package_signer_creation() {
        let signer = PackageSigner::new();
        assert!(signer.public_key().is_some());
        assert!(signer.export_signing_key().is_some());
    }

    #[test]
    fn test_sign_and_verify_package() {
        let signer = PackageSigner::new();
        let package = Package::new("test".to_string(), "1.0.0".to_string());

        let signature = signer.sign_package(&package).unwrap();
        assert_eq!(signature.algorithm, SignatureAlgorithm::Ed25519);

        let is_valid = signer.verify_package(&package, &signature).unwrap();
        assert!(is_valid);
    }

    #[test]
    fn test_signature_fails_on_modified_package() {
        let signer = PackageSigner::new();
        let mut package = Package::new("test".to_string(), "1.0.0".to_string());

        let signature = signer.sign_package(&package).unwrap();

        // Modify package
        package.add_source_file("new", "new content").unwrap();

        let is_valid = signer.verify_package(&package, &signature).unwrap();
        assert!(!is_valid);
    }

    #[test]
    fn test_encrypt_decrypt_package() {
        let encryptor = PackageEncryptor::new(EncryptionAlgorithm::Aes256Gcm);
        let package = Package::new("secret".to_string(), "1.0.0".to_string());
        let password = "super_secret_password";

        let encrypted = encryptor
            .encrypt_package_with_password(&package, password)
            .unwrap();
        assert_eq!(encrypted.algorithm, EncryptionAlgorithm::Aes256Gcm);

        let decrypted = encryptor
            .decrypt_package_with_password(&encrypted, password)
            .unwrap();
        assert_eq!(decrypted.name(), package.name());
        assert_eq!(decrypted.get_version(), package.get_version());
    }

    #[test]
    fn test_decrypt_with_wrong_password_fails() {
        let encryptor = PackageEncryptor::new(EncryptionAlgorithm::Aes256Gcm);
        let package = Package::new("secret".to_string(), "1.0.0".to_string());

        let encrypted = encryptor
            .encrypt_package_with_password(&package, "correct_password")
            .unwrap();

        let result = encryptor.decrypt_package_with_password(&encrypted, "wrong_password");
        assert!(result.is_err());
    }

    #[test]
    fn test_chacha20_encryption() {
        let encryptor = PackageEncryptor::new(EncryptionAlgorithm::ChaCha20Poly1305);
        let package = Package::new("test".to_string(), "1.0.0".to_string());
        let password = "test_password";

        let encrypted = encryptor
            .encrypt_package_with_password(&package, password)
            .unwrap();
        assert_eq!(encrypted.algorithm, EncryptionAlgorithm::ChaCha20Poly1305);

        let decrypted = encryptor
            .decrypt_package_with_password(&encrypted, password)
            .unwrap();
        assert_eq!(decrypted.name(), package.name());
    }

    #[test]
    fn test_trusted_key_verification() {
        let signer = PackageSigner::new();
        let mut verifier = PackageSigner::verifier_only();

        // Add the signer's public key as trusted
        verifier
            .add_trusted_key(&signer.public_key().unwrap())
            .unwrap();

        let package = Package::new("test".to_string(), "1.0.0".to_string());
        let signature = signer.sign_package(&package).unwrap();

        let is_valid = verifier.verify_package(&package, &signature).unwrap();
        assert!(is_valid);
    }

    #[test]
    fn test_untrusted_key_verification_fails() {
        let signer = PackageSigner::new();
        let mut verifier = PackageSigner::verifier_only();

        // Add a different public key as trusted
        let other_signer = PackageSigner::new();
        verifier
            .add_trusted_key(&other_signer.public_key().unwrap())
            .unwrap();

        let package = Package::new("test".to_string(), "1.0.0".to_string());
        let signature = signer.sign_package(&package).unwrap();

        let is_valid = verifier.verify_package(&package, &signature).unwrap();
        assert!(!is_valid);
    }
}
