//! Common communication primitives to eliminate duplication across modules

use crate::{Backend, ProcessGroup, TorshDistributedError, TorshResult};

/// Execute a function with read access to the backend
///
/// This consolidates the common pattern of getting backend read access
/// and handling initialization checks.
pub fn with_backend_read<T, F>(process_group: &ProcessGroup, f: F) -> TorshResult<T>
where
    F: FnOnce(&dyn Backend) -> TorshResult<T>,
{
    let backend = process_group.backend();
    let backend_guard = backend.read();

    validate_backend_initialized(&**backend_guard)?;
    f(&**backend_guard)
}

/// Execute a function with write access to the backend
///
/// This consolidates the common pattern of getting backend write access
/// and handling initialization checks.
pub fn with_backend_write<T, F>(process_group: &ProcessGroup, f: F) -> TorshResult<T>
where
    F: FnOnce(&mut dyn Backend) -> TorshResult<T>,
{
    let backend = process_group.backend();
    let mut backend_guard = backend.write();

    validate_backend_initialized(&**backend_guard)?;
    f(&mut **backend_guard)
}

/// Validate that a rank is within bounds for the given world size
///
/// This consolidates the rank validation pattern used across modules.
pub fn validate_rank(rank: u32, world_size: u32) -> TorshResult<()> {
    if rank >= world_size {
        return Err(TorshDistributedError::RankOutOfBounds { rank, world_size });
    }
    Ok(())
}

/// Validate that the backend is initialized
///
/// This consolidates the backend initialization check used across modules.
pub fn validate_backend_initialized(backend: &dyn Backend) -> TorshResult<()> {
    if !backend.is_ready() {
        return Err(TorshDistributedError::BackendNotInitialized);
    }
    Ok(())
}

/// Validate that ranks are within bounds for a process group
///
/// This validates multiple ranks at once, useful for collective operations.
pub fn validate_ranks(ranks: &[u32], world_size: u32) -> TorshResult<()> {
    for &rank in ranks {
        validate_rank(rank, world_size)?;
    }
    Ok(())
}

/// Check if the current process is the root process
///
/// This consolidates the root checking pattern used in collective operations.
pub fn is_root_process(process_group: &ProcessGroup) -> bool {
    process_group.rank() == 0
}

/// Get the world size as a u32 for convenience
///
/// This provides a consistent way to get world size across modules.
pub fn get_world_size(process_group: &ProcessGroup) -> u32 {
    process_group.world_size()
}

/// Get the current rank as a u32 for convenience
///
/// This provides a consistent way to get rank across modules.
pub fn get_rank(process_group: &ProcessGroup) -> u32 {
    process_group.rank()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BackendType;

    #[test]
    fn test_validate_rank() {
        // Valid ranks
        assert!(validate_rank(0, 4).is_ok());
        assert!(validate_rank(3, 4).is_ok());

        // Invalid ranks
        assert!(validate_rank(4, 4).is_err());
        assert!(validate_rank(10, 4).is_err());
    }

    #[test]
    fn test_validate_ranks() {
        // Valid ranks
        assert!(validate_ranks(&[0, 1, 2], 4).is_ok());

        // Invalid ranks
        assert!(validate_ranks(&[0, 4, 2], 4).is_err());
        assert!(validate_ranks(&[0, 1, 10], 4).is_err());
    }

    #[tokio::test]
    async fn test_is_root_process() {
        let pg = ProcessGroup::new(BackendType::Gloo, 0, 4, "localhost", 8080)
            .await
            .unwrap();

        assert!(is_root_process(&pg));
    }
}
