//! Collective communication operations

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::backend::ReduceOp;
use crate::process_group::ProcessGroup;
use crate::TorshResult;
use log::info;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_core::dtype::FloatElement;
use torsh_tensor::Tensor;

/// Communication group for selective collective operations
#[derive(Debug, Clone)]
pub struct CommunicationGroup {
    /// Group identifier
    pub group_id: String,
    /// Ranks participating in this group
    pub ranks: Vec<u32>,
    /// Local rank within this group (0-indexed within the group)
    pub local_rank: u32,
    /// Size of this group
    pub group_size: u32,
    /// Global rank to local rank mapping
    pub global_to_local: HashMap<u32, u32>,
    /// Local rank to global rank mapping
    pub local_to_global: HashMap<u32, u32>,
}

impl CommunicationGroup {
    /// Create a new communication group
    pub fn new(group_id: String, ranks: Vec<u32>, current_global_rank: u32) -> TorshResult<Self> {
        if ranks.is_empty() {
            return Err(crate::TorshDistributedError::invalid_argument(
                "ranks",
                "Communication group cannot be empty",
                "non-empty vector of ranks",
            ));
        }

        if !ranks.contains(&current_global_rank) {
            return Err(crate::TorshDistributedError::invalid_argument(
                "current_global_rank",
                format!(
                    "Current rank {} not in group {:?}",
                    current_global_rank, ranks
                ),
                "rank that exists in the group",
            ));
        }

        let mut sorted_ranks = ranks.clone();
        sorted_ranks.sort_unstable();

        let mut global_to_local = HashMap::new();
        let mut local_to_global = HashMap::new();

        for (local_idx, &global_rank) in sorted_ranks.iter().enumerate() {
            global_to_local.insert(global_rank, local_idx as u32);
            local_to_global.insert(local_idx as u32, global_rank);
        }

        let local_rank = global_to_local[&current_global_rank];
        let group_size = sorted_ranks.len() as u32;

        Ok(Self {
            group_id,
            ranks: sorted_ranks,
            local_rank,
            group_size,
            global_to_local,
            local_to_global,
        })
    }

    /// Create a communication group for a range of ranks
    pub fn from_range(
        group_id: String,
        start_rank: u32,
        end_rank: u32,
        current_global_rank: u32,
    ) -> TorshResult<Self> {
        if start_rank >= end_rank {
            return Err(crate::TorshDistributedError::invalid_argument(
                "rank_range",
                "start_rank must be less than end_rank",
                "valid rank range where start < end",
            ));
        }

        let ranks: Vec<u32> = (start_rank..end_rank).collect();
        Self::new(group_id, ranks, current_global_rank)
    }

    /// Check if a global rank is in this group
    pub fn contains_rank(&self, global_rank: u32) -> bool {
        self.global_to_local.contains_key(&global_rank)
    }

    /// Get local rank for a global rank
    pub fn global_to_local_rank(&self, global_rank: u32) -> Option<u32> {
        self.global_to_local.get(&global_rank).copied()
    }

    /// Get global rank for a local rank
    pub fn local_to_global_rank(&self, local_rank: u32) -> Option<u32> {
        self.local_to_global.get(&local_rank).copied()
    }
}

/// Group manager for managing multiple communication groups
#[derive(Debug, Default)]
pub struct GroupManager {
    groups: HashMap<String, Arc<CommunicationGroup>>,
    current_global_rank: u32,
}

impl GroupManager {
    /// Create a new group manager
    pub fn new(current_global_rank: u32) -> Self {
        Self {
            groups: HashMap::new(),
            current_global_rank,
        }
    }

    /// Create and register a new communication group
    pub fn create_group(
        &mut self,
        group_id: String,
        ranks: Vec<u32>,
    ) -> TorshResult<Arc<CommunicationGroup>> {
        if self.groups.contains_key(&group_id) {
            return Err(crate::TorshDistributedError::invalid_argument(
                "group_id",
                format!("Group '{}' already exists", group_id),
                "unique group identifier",
            ));
        }

        let group = Arc::new(CommunicationGroup::new(
            group_id.clone(),
            ranks,
            self.current_global_rank,
        )?);
        self.groups.insert(group_id, Arc::clone(&group));
        Ok(group)
    }

    /// Create a communication group from a rank range
    pub fn create_group_from_range(
        &mut self,
        group_id: String,
        start_rank: u32,
        end_rank: u32,
    ) -> TorshResult<Arc<CommunicationGroup>> {
        if self.groups.contains_key(&group_id) {
            return Err(crate::TorshDistributedError::invalid_argument(
                "group_id",
                format!("Group '{}' already exists", group_id),
                "unique group identifier",
            ));
        }

        let group = Arc::new(CommunicationGroup::from_range(
            group_id.clone(),
            start_rank,
            end_rank,
            self.current_global_rank,
        )?);
        self.groups.insert(group_id, Arc::clone(&group));
        Ok(group)
    }

    /// Get a communication group by ID
    pub fn get_group(&self, group_id: &str) -> Option<Arc<CommunicationGroup>> {
        self.groups.get(group_id).cloned()
    }

    /// Remove a communication group
    pub fn remove_group(&mut self, group_id: &str) -> bool {
        self.groups.remove(group_id).is_some()
    }

    /// List all group IDs
    pub fn list_groups(&self) -> Vec<String> {
        self.groups.keys().cloned().collect()
    }

    /// Create predefined groups for common parallelism patterns
    pub fn create_standard_groups(
        &mut self,
        world_size: u32,
        data_parallel_size: u32,
        model_parallel_size: u32,
    ) -> TorshResult<()> {
        if data_parallel_size * model_parallel_size != world_size {
            return Err(crate::TorshDistributedError::invalid_argument(
                "parallelism_configuration",
                "data_parallel_size * model_parallel_size must equal world_size",
                format!(
                    "configuration where {} * {} = {}",
                    data_parallel_size, model_parallel_size, world_size
                ),
            ));
        }

        // Create data parallel groups (ranks that share the same model)
        for mp_rank in 0..model_parallel_size {
            let mut dp_ranks = Vec::new();
            for dp_rank in 0..data_parallel_size {
                dp_ranks.push(dp_rank * model_parallel_size + mp_rank);
            }
            let group_id = format!("data_parallel_{}", mp_rank);
            self.create_group(group_id, dp_ranks)?;
        }

        // Create model parallel groups (ranks that share the same data)
        for dp_rank in 0..data_parallel_size {
            let start_rank = dp_rank * model_parallel_size;
            let end_rank = start_rank + model_parallel_size;
            let group_id = format!("model_parallel_{}", dp_rank);
            self.create_group_from_range(group_id, start_rank, end_rank)?;
        }

        Ok(())
    }
}

/// All-reduce: reduce tensor across all processes and distribute result
pub async fn all_reduce<T>(
    _tensor: &mut Tensor<T>,
    op: ReduceOp,
    group: &ProcessGroup,
) -> TorshResult<()>
where
    T: FloatElement
        + Default
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    use crate::communication::with_backend_read;

    // For now, implement a mock version
    // In a real implementation, this would use the backend's communication primitives
    with_backend_read(group, |backend_guard| {
        // Mock implementation: for sum, divide by world size to simulate averaging
        if let ReduceOp::Sum = op {
            let world_size = backend_guard.world_size();
            // Create a scalar of type T from world_size
            // For mock implementation, we'll use a simple approach
            if world_size > 1 {
                // This is a mock implementation - in practice, we'd need proper type conversion
                // For now, we'll skip the averaging to avoid type issues
                // *tensor = tensor.div_scalar(T::from(world_size as f32))?;
            }
        }
        Ok(())
    })
}

/// All-gather: gather tensors from all processes
pub async fn all_gather<T: FloatElement>(
    output: &mut Vec<Tensor<T>>,
    input: &Tensor<T>,
    group: &ProcessGroup,
) -> TorshResult<()> {
    use crate::communication::validate_backend_initialized;

    let backend = group.backend();
    let backend_guard = backend.read();

    validate_backend_initialized(&**backend_guard)?;
    let world_size = backend_guard.world_size();

    // Mock implementation: duplicate input for each rank
    // In real implementation, this would call backend.all_gather with type conversion
    output.clear();
    for _ in 0..world_size {
        output.push(input.clone());
    }

    Ok(())
}

/// Broadcast: broadcast tensor from source rank to all processes
pub async fn broadcast<T: FloatElement>(
    _tensor: &mut Tensor<T>,
    src_rank: u32,
    group: &ProcessGroup,
) -> TorshResult<()> {
    use crate::communication::{validate_rank, with_backend_read};

    with_backend_read(group, |backend_guard| {
        validate_rank(src_rank, backend_guard.world_size())?;

        // Mock implementation: tensor remains unchanged
        // In real implementation, would receive from src_rank if we're not the source
        Ok(())
    })
}

/// Reduce: reduce tensor to destination rank
pub async fn reduce<T>(
    _tensor: &mut Tensor<T>,
    dst_rank: u32,
    op: ReduceOp,
    group: &ProcessGroup,
) -> TorshResult<()>
where
    T: FloatElement
        + Default
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    use crate::communication::{validate_rank, with_backend_read};

    with_backend_read(group, |backend_guard| {
        validate_rank(dst_rank, backend_guard.world_size())?;

        // Mock implementation
        if backend_guard.rank() == dst_rank && matches!(op, ReduceOp::Sum) {
            let world_size = backend_guard.world_size();
            // For mock implementation, skip the scaling to avoid type issues
            if world_size > 1 {
                // *tensor = tensor.mul_scalar(T::from(world_size as f32))?;
            }
        }
        Ok(())
    })
}

/// Scatter: scatter tensor chunks from source rank to all processes
pub async fn scatter<T: FloatElement>(
    output: &mut Tensor<T>,
    input: Option<&[Tensor<T>]>,
    src_rank: u32,
    group: &ProcessGroup,
) -> TorshResult<()> {
    use crate::communication::{validate_rank, with_backend_read};

    with_backend_read(group, |backend_guard| {
        validate_rank(src_rank, backend_guard.world_size())?;

        if backend_guard.rank() == src_rank {
            let tensors = input.ok_or_else(|| {
                crate::TorshDistributedError::invalid_argument(
                    "input_tensors",
                    "Input tensors required for source rank",
                    "non-empty vector of tensors for scatter operation",
                )
            })?;

            if tensors.len() != backend_guard.world_size() as usize {
                return Err(crate::TorshDistributedError::invalid_argument(
                    "tensors",
                    format!(
                        "Expected {} tensors, got {}",
                        backend_guard.world_size(),
                        tensors.len()
                    ),
                    format!("{} tensors (one per rank)", backend_guard.world_size()),
                ));
            }

            *output = tensors[backend_guard.rank() as usize].clone();
        }
        Ok(())
    })
}

/// Barrier synchronization across all processes
#[allow(clippy::await_holding_lock)]
pub async fn barrier(group: &ProcessGroup) -> TorshResult<()> {
    let backend = group.backend();
    let mut backend_guard = backend.write();

    if !backend_guard.is_ready() {
        return Err(crate::TorshDistributedError::BackendNotInitialized);
    }

    backend_guard.barrier().await
}

/// Send tensor to specified rank (point-to-point communication)
pub async fn send<T: FloatElement>(
    _tensor: &Tensor<T>,
    dst_rank: u32,
    tag: u32,
    group: &ProcessGroup,
) -> TorshResult<()> {
    use crate::communication::{validate_rank, with_backend_read};

    with_backend_read(group, |backend_guard| {
        validate_rank(dst_rank, backend_guard.world_size())?;

        // Mock implementation: store tensor in a global message queue
        // In a real implementation, this would use the backend's send primitives
        info!(
            "📤 Rank {} sending tensor with tag {} to rank {}",
            backend_guard.rank(),
            tag,
            dst_rank
        );
        Ok(())
    })
}

/// Receive tensor from specified rank (point-to-point communication)
pub async fn recv<T: FloatElement>(
    _tensor: &mut Tensor<T>,
    src_rank: u32,
    tag: u32,
    group: &ProcessGroup,
) -> TorshResult<()> {
    use crate::communication::{validate_rank, with_backend_read};

    with_backend_read(group, |backend_guard| {
        validate_rank(src_rank, backend_guard.world_size())?;

        // Mock implementation: tensor remains unchanged
        // In a real implementation, this would use the backend's recv primitives
        info!(
            "📥 Rank {} receiving tensor with tag {} from rank {}",
            backend_guard.rank(),
            tag,
            src_rank
        );
        Ok(())
    })
}

/// Non-blocking send (isend) - returns immediately without waiting for completion
pub async fn isend<T: FloatElement>(
    tensor: &Tensor<T>,
    dst_rank: u32,
    tag: u32,
    group: &ProcessGroup,
) -> TorshResult<()> {
    // For simplicity, use blocking send in mock implementation
    send(tensor, dst_rank, tag, group).await
}

/// Non-blocking receive (irecv) - returns immediately, tensor is filled when ready
pub async fn irecv<T: FloatElement>(
    tensor: &mut Tensor<T>,
    src_rank: u32,
    tag: u32,
    group: &ProcessGroup,
) -> TorshResult<()> {
    // For simplicity, use blocking recv in mock implementation
    recv(tensor, src_rank, tag, group).await
}

// ============================================================================
// Group-Aware Collective Operations
// ============================================================================

/// All-reduce within a communication group
pub async fn all_reduce_group<T>(
    _tensor: &mut Tensor<T>,
    op: ReduceOp,
    comm_group: &CommunicationGroup,
    process_group: &ProcessGroup,
) -> TorshResult<()>
where
    T: FloatElement
        + Default
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    let backend = process_group.backend();
    let backend_guard = backend.read();

    if !backend_guard.is_ready() {
        return Err(crate::TorshDistributedError::BackendNotInitialized);
    }

    let current_global_rank = backend_guard.rank();

    // Check if current rank is part of this group
    if !comm_group.contains_rank(current_global_rank) {
        return Ok(()); // Not part of this group, skip operation
    }

    // Mock implementation: for sum, divide by group size to simulate averaging
    if let ReduceOp::Sum = op {
        let group_size = comm_group.group_size;
        // For mock implementation, skip the scaling to avoid type issues
        if group_size > 1 {
            // *tensor = tensor.div_scalar(T::from(group_size as f32))?;
        }
    }

    info!(
        " All-reduce in group '{}': rank {} (local: {}) with {} participants",
        comm_group.group_id, current_global_rank, comm_group.local_rank, comm_group.group_size
    );

    Ok(())
}

/// Broadcast within a communication group
pub async fn broadcast_group<T: FloatElement>(
    _tensor: &mut Tensor<T>,
    src_local_rank: u32,
    comm_group: &CommunicationGroup,
    process_group: &ProcessGroup,
) -> TorshResult<()> {
    use crate::communication::{validate_rank, with_backend_read};

    with_backend_read(process_group, |backend_guard| {
        let current_global_rank = backend_guard.rank();

        // Check if current rank is part of this group
        if !comm_group.contains_rank(current_global_rank) {
            return Ok(()); // Not part of this group, skip operation
        }

        validate_rank(src_local_rank, comm_group.group_size)?;

        let src_global_rank = comm_group
            .local_to_global_rank(src_local_rank)
            .ok_or_else(|| {
                crate::TorshDistributedError::invalid_argument(
                    "src_local_rank",
                    format!(
                        "Invalid local rank {} in group '{}'",
                        src_local_rank, comm_group.group_id
                    ),
                    format!("valid local rank in range 0..{}", comm_group.group_size),
                )
            })?;

        info!(
            " Broadcast in group '{}': from local rank {} (global: {}) to {} participants",
            comm_group.group_id, src_local_rank, src_global_rank, comm_group.group_size
        );
        Ok(())
    })
}

/// All-gather within a communication group
pub async fn all_gather_group<T: FloatElement>(
    output: &mut Vec<Tensor<T>>,
    input: &Tensor<T>,
    comm_group: &CommunicationGroup,
    process_group: &ProcessGroup,
) -> TorshResult<()> {
    use crate::communication::with_backend_read;

    with_backend_read(process_group, |backend_guard| {
        let current_global_rank = backend_guard.rank();

        // Check if current rank is part of this group
        if !comm_group.contains_rank(current_global_rank) {
            return Ok(()); // Not part of this group, skip operation
        }

        // Mock implementation: duplicate input for each rank in the group
        output.clear();
        for _ in 0..comm_group.group_size {
            output.push(input.clone());
        }

        info!(
            "🔗 All-gather in group '{}': rank {} collecting from {} participants",
            comm_group.group_id, current_global_rank, comm_group.group_size
        );
        Ok(())
    })
}

/// Reduce within a communication group
pub async fn reduce_group<T>(
    _tensor: &mut Tensor<T>,
    dst_local_rank: u32,
    op: ReduceOp,
    comm_group: &CommunicationGroup,
    process_group: &ProcessGroup,
) -> TorshResult<()>
where
    T: FloatElement
        + Default
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    let backend = process_group.backend();
    let backend_guard = backend.read();

    if !backend_guard.is_ready() {
        return Err(crate::TorshDistributedError::BackendNotInitialized);
    }

    let current_global_rank = backend_guard.rank();

    // Check if current rank is part of this group
    if !comm_group.contains_rank(current_global_rank) {
        return Ok(()); // Not part of this group, skip operation
    }

    if dst_local_rank >= comm_group.group_size {
        return Err(crate::TorshDistributedError::RankOutOfBounds {
            rank: dst_local_rank,
            world_size: comm_group.group_size,
        });
    }

    let dst_global_rank = comm_group
        .local_to_global_rank(dst_local_rank)
        .ok_or_else(|| crate::TorshDistributedError::InvalidArgument {
            arg: "rank".to_string(),
            reason: format!(
                "Invalid local rank {} in group '{}'",
                dst_local_rank, comm_group.group_id
            ),
            expected: "valid local rank within the communication group".to_string(),
        })?;

    // Mock implementation
    if current_global_rank == dst_global_rank && matches!(op, ReduceOp::Sum) {
        let group_size = comm_group.group_size;
        // For mock implementation, skip the scaling to avoid type issues
        if group_size > 1 {
            // *tensor = tensor.mul_scalar(T::from(group_size as f32))?;
        }
    }

    info!(
        "⬇️  Reduce in group '{}': to local rank {} (global: {}) from {} participants",
        comm_group.group_id, dst_local_rank, dst_global_rank, comm_group.group_size
    );

    Ok(())
}

/// Barrier synchronization within a communication group
pub async fn barrier_group(
    comm_group: &CommunicationGroup,
    process_group: &ProcessGroup,
) -> TorshResult<()> {
    let backend = process_group.backend();
    let backend_guard = backend.read();

    if !backend_guard.is_ready() {
        return Err(crate::TorshDistributedError::BackendNotInitialized);
    }

    let current_global_rank = backend_guard.rank();

    // Check if current rank is part of this group
    if !comm_group.contains_rank(current_global_rank) {
        return Ok(()); // Not part of this group, skip operation
    }

    info!(
        "🚧 Barrier in group '{}': rank {} waiting for {} participants",
        comm_group.group_id, current_global_rank, comm_group.group_size
    );

    // Mock implementation: immediate return
    // In real implementation, would only synchronize with ranks in the group
    Ok(())
}

/// Point-to-point send within a communication group (using local ranks)
pub async fn send_group<T: FloatElement>(
    _tensor: &Tensor<T>,
    dst_local_rank: u32,
    tag: u32,
    comm_group: &CommunicationGroup,
    process_group: &ProcessGroup,
) -> TorshResult<()> {
    let backend = process_group.backend();
    let backend_guard = backend.read();

    if !backend_guard.is_ready() {
        return Err(crate::TorshDistributedError::BackendNotInitialized);
    }

    let current_global_rank = backend_guard.rank();

    // Check if current rank is part of this group
    if !comm_group.contains_rank(current_global_rank) {
        return Err(crate::TorshDistributedError::InvalidArgument {
            arg: "rank".to_string(),
            reason: format!(
                "Rank {} not in group '{}'",
                current_global_rank, comm_group.group_id
            ),
            expected: "rank must be member of the communication group".to_string(),
        });
    }

    if dst_local_rank >= comm_group.group_size {
        return Err(crate::TorshDistributedError::RankOutOfBounds {
            rank: dst_local_rank,
            world_size: comm_group.group_size,
        });
    }

    let dst_global_rank = comm_group
        .local_to_global_rank(dst_local_rank)
        .ok_or_else(|| crate::TorshDistributedError::InvalidArgument {
            arg: "rank".to_string(),
            reason: format!(
                "Invalid local rank {} in group '{}'",
                dst_local_rank, comm_group.group_id
            ),
            expected: "valid local rank within the communication group".to_string(),
        })?;

    info!(
        "📤 Group send in '{}': from rank {} to local rank {} (global: {}) with tag {}",
        comm_group.group_id, current_global_rank, dst_local_rank, dst_global_rank, tag
    );

    Ok(())
}

/// Point-to-point receive within a communication group (using local ranks)
pub async fn recv_group<T: FloatElement>(
    _tensor: &mut Tensor<T>,
    src_local_rank: u32,
    tag: u32,
    comm_group: &CommunicationGroup,
    process_group: &ProcessGroup,
) -> TorshResult<()> {
    let backend = process_group.backend();
    let backend_guard = backend.read();

    if !backend_guard.is_ready() {
        return Err(crate::TorshDistributedError::BackendNotInitialized);
    }

    let current_global_rank = backend_guard.rank();

    // Check if current rank is part of this group
    if !comm_group.contains_rank(current_global_rank) {
        return Err(crate::TorshDistributedError::InvalidArgument {
            arg: "rank".to_string(),
            reason: format!(
                "Rank {} not in group '{}'",
                current_global_rank, comm_group.group_id
            ),
            expected: "rank must be member of the communication group".to_string(),
        });
    }

    if src_local_rank >= comm_group.group_size {
        return Err(crate::TorshDistributedError::RankOutOfBounds {
            rank: src_local_rank,
            world_size: comm_group.group_size,
        });
    }

    let src_global_rank = comm_group
        .local_to_global_rank(src_local_rank)
        .ok_or_else(|| {
            crate::TorshDistributedError::invalid_argument(
                "src_local_rank",
                format!(
                    "Invalid local rank {} in group '{}'",
                    src_local_rank, comm_group.group_id
                ),
                format!("valid local rank in range 0..{}", comm_group.group_size),
            )
        })?;

    info!(
        "📥 Group recv in '{}': from local rank {} (global: {}) to rank {} with tag {}",
        comm_group.group_id, src_local_rank, src_global_rank, current_global_rank, tag
    );

    Ok(())
}

// ============================================================================
// Custom Collective Operations
// ============================================================================

/// Reduce-scatter: reduce tensors and scatter result chunks to all processes
/// Each rank gets a different chunk of the reduced result
pub async fn reduce_scatter<T>(
    output: &mut Tensor<T>,
    input: &Tensor<T>,
    op: ReduceOp,
    group: &ProcessGroup,
) -> TorshResult<()>
where
    T: FloatElement
        + Default
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    let backend = group.backend();
    let backend_guard = backend.read();

    if !backend_guard.is_ready() {
        return Err(crate::TorshDistributedError::BackendNotInitialized);
    }

    let rank = backend_guard.rank();
    let world_size = backend_guard.world_size();

    // Mock implementation: simulate reduce-scatter by copying a portion of input
    // In real implementation, would:
    // 1. All-reduce the full tensor
    // 2. Split result into world_size chunks
    // 3. Each rank gets its corresponding chunk

    // For simplicity, just copy the input and apply operation
    *output = input.clone();

    if let ReduceOp::Sum = op {
        let factor = world_size;
        // For mock implementation, skip the scaling to avoid type issues
        if factor > 1 {
            // *output = output.div_scalar(T::from(factor as f32))?;
        }
    }

    info!(
        " Reduce-scatter: rank {} processing chunk of reduced tensor",
        rank
    );

    Ok(())
}

/// All-to-all: each rank sends unique data to every other rank
/// output[i] receives data from rank i
pub async fn all_to_all<T: FloatElement>(
    output: &mut Vec<Tensor<T>>,
    input: &[Tensor<T>],
    group: &ProcessGroup,
) -> TorshResult<()> {
    let backend = group.backend();
    let backend_guard = backend.read();

    if !backend_guard.is_ready() {
        return Err(crate::TorshDistributedError::BackendNotInitialized);
    }

    let rank = backend_guard.rank();
    let world_size = backend_guard.world_size() as usize;

    if input.len() != world_size {
        return Err(crate::TorshDistributedError::InvalidArgument {
            arg: "rank".to_string(),
            reason: format!(
                "Input must have {} tensors for all-to-all, got {}",
                world_size,
                input.len()
            ),
            expected: format!("{} tensors (one per rank)", world_size),
        });
    }

    // Mock implementation: simulate all-to-all by copying appropriate input tensors
    // In real implementation, each rank would send input[i] to rank i
    // and receive from rank j into output[j]
    output.clear();

    for i in 0..world_size {
        // Simulate receiving from rank i
        if i < input.len() {
            output.push(input[i].clone());
        }
    }

    info!(
        " All-to-all: rank {} exchanging data with {} ranks",
        rank, world_size
    );

    Ok(())
}

/// Ring all-reduce: more bandwidth-efficient all-reduce for large tensors
/// Reduces communication volume by using ring topology
pub async fn ring_all_reduce<T>(
    _tensor: &mut Tensor<T>,
    op: ReduceOp,
    group: &ProcessGroup,
) -> TorshResult<()>
where
    T: FloatElement
        + Default
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    let backend = group.backend();
    let backend_guard = backend.read();

    if !backend_guard.is_ready() {
        return Err(crate::TorshDistributedError::BackendNotInitialized);
    }

    let rank = backend_guard.rank();
    let world_size = backend_guard.world_size();

    // Mock implementation: simulate ring all-reduce
    // In real implementation, would:
    // 1. Divide tensor into world_size chunks
    // 2. In reduce-scatter phase, reduce chunks in ring order
    // 3. In all-gather phase, gather reduced chunks in ring order

    if let ReduceOp::Sum = op {
        let world_size_f = world_size;
        // For mock implementation, skip the scaling to avoid type issues
        if world_size_f > 1 {
            // *tensor = tensor.div_scalar(T::from(world_size_f as f32))?;
        }
    }

    info!(
        " Ring all-reduce: rank {} in {}-node ring topology",
        rank, world_size
    );

    Ok(())
}

/// Hierarchical all-reduce: two-level all-reduce for multi-node scenarios
/// More efficient when there are multiple nodes with fast intra-node communication
pub async fn hierarchical_all_reduce<T>(
    _tensor: &mut Tensor<T>,
    op: ReduceOp,
    group: &ProcessGroup,
    ranks_per_node: u32,
) -> TorshResult<()>
where
    T: FloatElement
        + Default
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    let backend = group.backend();
    let backend_guard = backend.read();

    if !backend_guard.is_ready() {
        return Err(crate::TorshDistributedError::BackendNotInitialized);
    }

    let rank = backend_guard.rank();
    let world_size = backend_guard.world_size();

    if world_size % ranks_per_node != 0 {
        return Err(crate::TorshDistributedError::InvalidArgument {
            arg: "rank".to_string(),
            reason: "World size must be divisible by ranks_per_node for hierarchical all-reduce"
                .to_string(),
            expected: format!(
                "world_size divisible by ranks_per_node ({})",
                ranks_per_node
            ),
        });
    }

    let node_id = rank / ranks_per_node;
    let local_rank = rank % ranks_per_node;
    let num_nodes = world_size / ranks_per_node;

    // Mock implementation: simulate hierarchical all-reduce
    // In real implementation, would:
    // 1. Intra-node all-reduce (within each node)
    // 2. Inter-node all-reduce (between node representatives)
    // 3. Intra-node broadcast (from representatives to all ranks in node)

    if let ReduceOp::Sum = op {
        let world_size_f = world_size;
        // For mock implementation, skip the scaling to avoid type issues
        if world_size_f > 1 {
            // *tensor = tensor.div_scalar(T::from(world_size_f as f32))?;
        }
    }

    info!(
        " Hierarchical all-reduce: rank {} (node {}, local rank {}) with {} nodes × {} ranks/node",
        rank, node_id, local_rank, num_nodes, ranks_per_node
    );

    Ok(())
}

/// Bucket all-reduce: reduce multiple tensors efficiently by combining them
/// Useful for gradient synchronization in distributed training
pub async fn bucket_all_reduce<T>(
    tensors: &mut [Tensor<T>],
    op: ReduceOp,
    group: &ProcessGroup,
    max_bucket_size_mb: f32,
) -> TorshResult<()>
where
    T: FloatElement
        + Default
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    let backend = group.backend();
    let backend_guard = backend.read();

    if !backend_guard.is_ready() {
        return Err(crate::TorshDistributedError::BackendNotInitialized);
    }

    let rank = backend_guard.rank();
    let world_size = backend_guard.world_size();

    if tensors.is_empty() {
        return Ok(());
    }

    // Mock implementation: simulate bucketed all-reduce
    // In real implementation, would:
    // 1. Group tensors into buckets based on size limit
    // 2. Flatten each bucket into a single contiguous tensor
    // 3. Perform all-reduce on each flattened bucket
    // 4. Unflatten and distribute results back to original tensors

    let max_bucket_size_bytes = (max_bucket_size_mb * 1024.0 * 1024.0) as usize;
    let mut current_bucket_size = 0;
    let mut bucket_count = 0;

    for tensor in tensors.iter_mut() {
        let tensor_size = tensor.numel() * std::mem::size_of::<T>();

        if current_bucket_size + tensor_size > max_bucket_size_bytes && current_bucket_size > 0 {
            bucket_count += 1;
            current_bucket_size = tensor_size;
        } else {
            current_bucket_size += tensor_size;
        }

        // Apply operation to each tensor (simulate all-reduce)
        if let ReduceOp::Sum = op {
            let world_size_f = world_size;
            // For mock implementation, skip the scaling to avoid type issues
            if world_size_f > 1 {
                // *tensor = tensor.div_scalar(T::from(world_size_f as f32))?;
            }
        }
    }

    if current_bucket_size > 0 {
        bucket_count += 1;
    }

    info!(
        " Bucket all-reduce: rank {} processed {} tensors in {} buckets (max {:.1} MB/bucket)",
        rank,
        tensors.len(),
        bucket_count,
        max_bucket_size_mb
    );

    Ok(())
}

// ============================================================================
// Advanced Communication Primitives for Distributed Deep Learning
// ============================================================================

/// All-reduce with fusion: combines small tensors into larger buffers for efficiency
/// This is critical for gradient synchronization in distributed training
pub async fn fused_all_reduce<T>(
    tensors: &mut [Tensor<T>],
    op: ReduceOp,
    group: &ProcessGroup,
    fusion_threshold_bytes: usize,
) -> TorshResult<()>
where
    T: FloatElement
        + Default
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    let backend = group.backend();
    let backend_guard = backend.read();

    if !backend_guard.is_ready() {
        return Err(crate::TorshDistributedError::BackendNotInitialized);
    }

    let rank = backend_guard.rank();
    let world_size = backend_guard.world_size();

    if tensors.is_empty() {
        return Ok(());
    }

    // Group tensors into fusion groups based on size threshold
    let mut fusion_groups = Vec::new();
    let mut current_group = Vec::new();
    let mut current_size = 0;

    for (idx, tensor) in tensors.iter().enumerate() {
        let tensor_size = tensor.numel() * std::mem::size_of::<T>();

        if current_size + tensor_size > fusion_threshold_bytes && !current_group.is_empty() {
            fusion_groups.push(std::mem::take(&mut current_group));
            current_size = tensor_size;
            current_group.push(idx);
        } else {
            current_size += tensor_size;
            current_group.push(idx);
        }
    }

    if !current_group.is_empty() {
        fusion_groups.push(current_group);
    }

    // Process each fusion group
    for (group_idx, tensor_indices) in fusion_groups.iter().enumerate() {
        // In real implementation, would:
        // 1. Flatten tensors in group into contiguous buffer
        // 2. Perform single all-reduce on fused buffer
        // 3. Unflatten and distribute back to original tensors

        for &_tensor_idx in tensor_indices {
            if let ReduceOp::Sum = op {
                let world_size_f = world_size;
                // For mock implementation, skip the scaling to avoid type issues
                if world_size_f > 1 {
                    // tensors[tensor_idx] = tensors[tensor_idx].div_scalar(T::from(world_size_f as f32))?;
                }
            }
        }

        info!(
            "🔗 Fused all-reduce group {}: rank {} processed {} tensors",
            group_idx,
            rank,
            tensor_indices.len()
        );
    }

    info!(
        " Fused all-reduce complete: rank {} processed {} tensors in {} fusion groups",
        rank,
        tensors.len(),
        fusion_groups.len()
    );

    Ok(())
}

/// Variable-sized all-gather: gather tensors of different sizes from all ranks
/// Critical for dynamic neural networks where tensor sizes vary across ranks
pub async fn all_gather_varsize<T: FloatElement>(
    output: &mut Vec<Tensor<T>>,
    input: &Tensor<T>,
    group: &ProcessGroup,
) -> TorshResult<()> {
    let backend = group.backend();
    let backend_guard = backend.read();

    if !backend_guard.is_ready() {
        return Err(crate::TorshDistributedError::BackendNotInitialized);
    }

    let rank = backend_guard.rank();
    let world_size = backend_guard.world_size();

    // In real implementation, would:
    // 1. Exchange tensor sizes via all-gather of sizes
    // 2. Allocate output buffers based on received sizes
    // 3. Perform all-gather with appropriate offsets for each rank

    output.clear();

    // Mock implementation: simulate variable sizes
    for i in 0..world_size {
        // Simulate different tensor sizes from different ranks
        let _scale_factor = 1.0 + (i as f32 * 0.1);
        // For mock implementation, skip the scaling to avoid type issues
        // let scaled_tensor = input.mul_scalar(T::from(scale_factor))?;
        let scaled_tensor = input.clone();
        output.push(scaled_tensor);
    }

    info!(
        " Variable-size all-gather: rank {} collected from {} ranks with varying sizes",
        rank, world_size
    );

    Ok(())
}

/// Tree-based broadcast: more efficient for large world sizes
/// Uses binary tree topology to reduce latency compared to linear broadcast
pub async fn tree_broadcast<T: FloatElement>(
    _tensor: &mut Tensor<T>,
    src_rank: u32,
    group: &ProcessGroup,
) -> TorshResult<()> {
    let backend = group.backend();
    let backend_guard = backend.read();

    if !backend_guard.is_ready() {
        return Err(crate::TorshDistributedError::BackendNotInitialized);
    }

    let rank = backend_guard.rank();
    let world_size = backend_guard.world_size();

    if src_rank >= world_size {
        return Err(crate::TorshDistributedError::RankOutOfBounds {
            rank: src_rank,
            world_size,
        });
    }

    // In real implementation, would use binary tree topology:
    // - Root (src_rank) sends to its children
    // - Each receiving rank forwards to its children
    // - Continue until all ranks have received the data

    // Calculate tree position for logging
    let tree_depth = (world_size as f32).log2().ceil() as u32;
    let is_root = rank == src_rank;
    let parent_rank = if rank == src_rank {
        None
    } else {
        Some((rank - 1) / 2)
    };

    info!(
        "🌳 Tree broadcast: rank {} (root: {}, parent: {:?}) in {}-deep tree from root {}",
        rank, is_root, parent_rank, tree_depth, src_rank
    );

    Ok(())
}

/// Pipelined all-reduce: overlaps computation and communication
/// Useful for very large tensors that can be processed in chunks
pub async fn pipelined_all_reduce<T>(
    tensor: &mut Tensor<T>,
    op: ReduceOp,
    group: &ProcessGroup,
    pipeline_chunks: usize,
) -> TorshResult<()>
where
    T: FloatElement
        + Default
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    let backend = group.backend();
    let backend_guard = backend.read();

    if !backend_guard.is_ready() {
        return Err(crate::TorshDistributedError::BackendNotInitialized);
    }

    let rank = backend_guard.rank();
    let world_size = backend_guard.world_size();

    if pipeline_chunks == 0 {
        return Err(crate::TorshDistributedError::InvalidArgument {
            arg: "rank".to_string(),
            reason: "Pipeline chunks must be greater than 0".to_string(),
            expected: "pipeline_chunks > 0".to_string(),
        });
    }

    // In real implementation, would:
    // 1. Split tensor into pipeline_chunks
    // 2. Start all-reduce on chunk 0 while chunks 1+ are still being computed
    // 3. Pipeline the communication and computation for optimal overlap

    let chunk_size = tensor.numel().div_ceil(pipeline_chunks);

    // Mock implementation: apply operation to simulate pipelined processing
    if let ReduceOp::Sum = op {
        let world_size_f = world_size;
        // For mock implementation, skip the scaling to avoid type issues
        if world_size_f > 1 {
            // *tensor = tensor.div_scalar(T::from(world_size_f as f32))?;
        }
    }

    info!(
        "⚡ Pipelined all-reduce: rank {} processed tensor in {} chunks ({} elements/chunk)",
        rank, pipeline_chunks, chunk_size
    );

    Ok(())
}

/// Double-buffered all-reduce: uses double buffering to hide latency
/// Critical for overlapping gradient computation with communication
pub async fn double_buffered_all_reduce<T>(
    _current_buffer: &mut Tensor<T>,
    _next_buffer: &mut Tensor<T>,
    op: ReduceOp,
    group: &ProcessGroup,
) -> TorshResult<()>
where
    T: FloatElement
        + Default
        + Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    let backend = group.backend();
    let backend_guard = backend.read();

    if !backend_guard.is_ready() {
        return Err(crate::TorshDistributedError::BackendNotInitialized);
    }

    let rank = backend_guard.rank();
    let world_size = backend_guard.world_size();

    // In real implementation, would:
    // 1. Start all-reduce on current_buffer
    // 2. While current_buffer is being reduced, fill next_buffer with new data
    // 3. Swap buffers when current reduction completes
    // 4. Repeat for continuous pipelined operation

    // Mock implementation: process both buffers
    if let ReduceOp::Sum = op {
        let world_size_f = world_size;
        // For mock implementation, skip the scaling to avoid type issues
        if world_size_f > 1 {
            // *current_buffer = current_buffer.div_scalar(T::from(world_size_f as f32))?;
            // *next_buffer = next_buffer.div_scalar(T::from(world_size_f as f32))?;
        }
    }

    info!(
        " Double-buffered all-reduce: rank {} processed buffers with overlap",
        rank
    );

    Ok(())
}
