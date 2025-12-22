//! Comprehensive security tests

use tempfile::NamedTempFile;
use torsh_core::error::Result;
use torsh_package::security::*;
use torsh_package::Package;

#[test]
fn test_package_signing_and_verification() -> Result<()> {
    let signer = PackageSigner::new();
    let package = Package::new("secure_model".to_string(), "1.0.0".to_string());

    let signature = signer.sign_package(&package)?;

    assert_eq!(signature.algorithm, SignatureAlgorithm::Ed25519);
    assert!(!signature.signature.is_empty());
    assert!(!signature.public_key.is_empty());

    let is_valid = signer.verify_package(&package, &signature)?;
    assert!(is_valid);

    Ok(())
}

#[test]
fn test_signature_fails_on_tampered_package() -> Result<()> {
    let signer = PackageSigner::new();
    let mut package = Package::new("test".to_string(), "1.0.0".to_string());

    let signature = signer.sign_package(&package)?;

    // Tamper with the package
    package.add_source_file("malicious", "malicious code")?;

    let is_valid = signer.verify_package(&package, &signature)?;
    assert!(!is_valid);

    Ok(())
}

#[test]
fn test_trusted_key_verification() -> Result<()> {
    let signer = PackageSigner::new();
    let mut verifier = PackageSigner::verifier_only();

    let public_key = signer.public_key().unwrap();
    verifier.add_trusted_key(&public_key)?;

    let package = Package::new("test".to_string(), "1.0.0".to_string());
    let signature = signer.sign_package(&package)?;

    let is_valid = verifier.verify_package(&package, &signature)?;
    assert!(is_valid);

    Ok(())
}

#[test]
fn test_untrusted_key_rejection() -> Result<()> {
    let signer1 = PackageSigner::new();
    let signer2 = PackageSigner::new();
    let mut verifier = PackageSigner::verifier_only();

    // Add signer2's key as trusted
    verifier.add_trusted_key(&signer2.public_key().unwrap())?;

    let package = Package::new("test".to_string(), "1.0.0".to_string());
    let signature = signer1.sign_package(&package)?; // Signed by signer1

    let is_valid = verifier.verify_package(&package, &signature)?;
    assert!(!is_valid); // Should fail because signer1's key is not trusted

    Ok(())
}

#[test]
fn test_signature_persistence() -> Result<()> {
    let signer = PackageSigner::new();
    let package = Package::new("test".to_string(), "1.0.0".to_string());

    let signature = signer.sign_package(&package)?;

    let temp_file = NamedTempFile::new()?;
    PackageSigner::save_signature(&signature, temp_file.path())?;

    let loaded_signature = PackageSigner::load_signature(temp_file.path())?;

    assert_eq!(signature.signature, loaded_signature.signature);
    assert_eq!(signature.public_key, loaded_signature.public_key);
    assert_eq!(signature.algorithm, loaded_signature.algorithm);

    Ok(())
}

#[test]
fn test_aes_gcm_encryption() -> Result<()> {
    let encryptor = PackageEncryptor::new(EncryptionAlgorithm::Aes256Gcm);
    let package = Package::new("secret".to_string(), "1.0.0".to_string());
    let password = "super_secret_password_123";

    let encrypted = encryptor.encrypt_package_with_password(&package, password)?;

    assert_eq!(encrypted.algorithm, EncryptionAlgorithm::Aes256Gcm);
    assert!(!encrypted.encrypted_data.is_empty());
    assert!(!encrypted.nonce.is_empty());

    let decrypted = encryptor.decrypt_package_with_password(&encrypted, password)?;

    assert_eq!(decrypted.name(), package.name());
    assert_eq!(decrypted.get_version(), package.get_version());

    Ok(())
}

#[test]
fn test_chacha20_encryption() -> Result<()> {
    let encryptor = PackageEncryptor::new(EncryptionAlgorithm::ChaCha20Poly1305);
    let package = Package::new("secret".to_string(), "1.0.0".to_string());
    let password = "chacha_password";

    let encrypted = encryptor.encrypt_package_with_password(&package, password)?;

    assert_eq!(encrypted.algorithm, EncryptionAlgorithm::ChaCha20Poly1305);

    let decrypted = encryptor.decrypt_package_with_password(&encrypted, password)?;

    assert_eq!(decrypted.name(), package.name());

    Ok(())
}

#[test]
fn test_wrong_password_decryption_fails() {
    let encryptor = PackageEncryptor::new(EncryptionAlgorithm::Aes256Gcm);
    let package = Package::new("secret".to_string(), "1.0.0".to_string());

    let encrypted = encryptor
        .encrypt_package_with_password(&package, "correct_password")
        .unwrap();

    let result = encryptor.decrypt_package_with_password(&encrypted, "wrong_password");

    assert!(result.is_err());
}

#[test]
fn test_encrypted_package_persistence() -> Result<()> {
    let encryptor = PackageEncryptor::new(EncryptionAlgorithm::Aes256Gcm);
    let package = Package::new("secret".to_string(), "1.0.0".to_string());
    let password = "test_password";

    let encrypted = encryptor.encrypt_package_with_password(&package, password)?;

    let temp_file = NamedTempFile::new()?;
    PackageEncryptor::save_encrypted(&encrypted, temp_file.path())?;

    let loaded_encrypted = PackageEncryptor::load_encrypted(temp_file.path())?;

    let decrypted = encryptor.decrypt_package_with_password(&loaded_encrypted, password)?;

    assert_eq!(decrypted.name(), package.name());

    Ok(())
}

#[test]
fn test_key_export_import() -> Result<()> {
    let signer = PackageSigner::new();
    let exported_key = signer.export_signing_key().unwrap();

    let imported_signer = PackageSigner::from_signing_key(ed25519_dalek::SigningKey::from_bytes(
        &exported_key.try_into().unwrap(),
    ));

    let package = Package::new("test".to_string(), "1.0.0".to_string());

    let sig1 = signer.sign_package(&package)?;
    let sig2 = imported_signer.sign_package(&package)?;

    // Signatures should be identical (deterministic signing)
    assert_eq!(sig1.signature, sig2.signature);

    Ok(())
}

#[test]
fn test_signature_metadata() -> Result<()> {
    let signer = PackageSigner::new();
    let package = Package::new("my_model".to_string(), "2.1.0".to_string());

    let signature = signer.sign_package(&package)?;

    assert_eq!(signature.metadata.get("package_name").unwrap(), "my_model");
    assert_eq!(signature.metadata.get("package_version").unwrap(), "2.1.0");

    Ok(())
}

#[test]
fn test_encryption_with_large_package() -> Result<()> {
    let encryptor = PackageEncryptor::new(EncryptionAlgorithm::ChaCha20Poly1305);
    let mut package = Package::new("large".to_string(), "1.0.0".to_string());

    // Add some data to make it larger
    for i in 0..100 {
        package.add_source_file(&format!("file_{}", i), &"x".repeat(1000))?;
    }

    let password = "large_package_password";

    let encrypted = encryptor.encrypt_package_with_password(&package, password)?;
    let decrypted = encryptor.decrypt_package_with_password(&encrypted, password)?;

    assert_eq!(decrypted.name(), package.name());
    assert_eq!(decrypted.resources().len(), package.resources().len());

    Ok(())
}

#[test]
fn test_multiple_encryption_algorithms() -> Result<()> {
    let package = Package::new("test".to_string(), "1.0.0".to_string());
    let password = "test";

    let algorithms = vec![
        EncryptionAlgorithm::Aes256Gcm,
        EncryptionAlgorithm::ChaCha20Poly1305,
    ];

    for algorithm in algorithms {
        let encryptor = PackageEncryptor::new(algorithm);

        let encrypted = encryptor.encrypt_package_with_password(&package, password)?;
        let decrypted = encryptor.decrypt_package_with_password(&encrypted, password)?;

        assert_eq!(decrypted.name(), package.name());
    }

    Ok(())
}

#[test]
fn test_signature_timestamp() -> Result<()> {
    let signer = PackageSigner::new();
    let package = Package::new("test".to_string(), "1.0.0".to_string());

    let signature = signer.sign_package(&package)?;

    // Signature should have a recent timestamp
    let now = chrono::Utc::now();
    let time_diff = now.signed_duration_since(signature.timestamp);

    assert!(time_diff.num_seconds() < 5); // Should be created within 5 seconds

    Ok(())
}

#[test]
fn test_pbkdf2_key_derivation() -> Result<()> {
    let encryptor = PackageEncryptor::new(EncryptionAlgorithm::Aes256Gcm);
    let package = Package::new("test".to_string(), "1.0.0".to_string());
    let password = "pbkdf2_test";

    let encrypted = encryptor.encrypt_package_with_password(&package, password)?;

    // Check that KDF metadata is present
    assert!(encrypted.key_metadata.contains_key("kdf"));
    assert_eq!(encrypted.key_metadata.get("kdf").unwrap(), "pbkdf2");
    assert!(encrypted.key_metadata.contains_key("salt"));
    assert!(encrypted.key_metadata.contains_key("iterations"));

    Ok(())
}
