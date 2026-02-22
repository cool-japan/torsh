//! State management for JIT debugging
//!
//! This module provides comprehensive state management capabilities including
//! call stack management, memory state tracking, and execution state management.

use super::core::{CallFrame, ExecutionLocation};
use crate::{JitError, JitResult};
use std::collections::HashMap;

/// Call stack management
#[derive(Debug, Clone)]
pub struct CallStack {
    frames: Vec<CallFrame>,
    max_depth: usize,
}

impl CallStack {
    /// Create a new call stack
    pub fn new() -> Self {
        Self {
            frames: Vec::new(),
            max_depth: 1000, // Default maximum depth to prevent stack overflow
        }
    }

    /// Create a new call stack with specified maximum depth
    ///
    /// # Arguments
    /// * `max_depth` - Maximum allowed call stack depth
    pub fn with_max_depth(max_depth: usize) -> Self {
        Self {
            frames: Vec::new(),
            max_depth,
        }
    }

    /// Push a new frame onto the call stack
    ///
    /// # Arguments
    /// * `frame` - The call frame to push
    ///
    /// # Returns
    /// An error if the stack would exceed maximum depth
    pub fn push(&mut self, frame: CallFrame) -> JitResult<()> {
        if self.frames.len() >= self.max_depth {
            return Err(JitError::RuntimeError(format!(
                "Call stack overflow: maximum depth {} exceeded",
                self.max_depth
            )));
        }
        self.frames.push(frame);
        Ok(())
    }

    /// Pop the top frame from the call stack
    ///
    /// # Returns
    /// The return location from the popped frame, or Completed if stack is empty
    pub fn pop(&mut self) -> ExecutionLocation {
        if let Some(frame) = self.frames.pop() {
            frame.return_location
        } else {
            ExecutionLocation::Completed
        }
    }

    /// Get the current depth of the call stack
    pub fn depth(&self) -> usize {
        self.frames.len()
    }

    /// Check if the call stack is empty
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Get a reference to the current (top) frame
    pub fn current_frame(&self) -> Option<&CallFrame> {
        self.frames.last()
    }

    /// Get a mutable reference to the current (top) frame
    pub fn current_frame_mut(&mut self) -> Option<&mut CallFrame> {
        self.frames.last_mut()
    }

    /// Get a reference to a frame at a specific depth
    ///
    /// # Arguments
    /// * `depth` - The depth (0 is the top frame)
    pub fn frame_at(&self, depth: usize) -> Option<&CallFrame> {
        if depth < self.frames.len() {
            Some(&self.frames[self.frames.len() - 1 - depth])
        } else {
            None
        }
    }

    /// Get all frames in the call stack (top to bottom)
    pub fn frames(&self) -> &[CallFrame] {
        &self.frames
    }

    /// Clear the entire call stack
    pub fn clear(&mut self) {
        self.frames.clear();
    }

    /// Find frames by function name
    ///
    /// # Arguments
    /// * `function_name` - The name of the function to search for
    ///
    /// # Returns
    /// A vector of indices where the function appears in the stack
    pub fn find_frames_by_function(&self, function_name: &str) -> Vec<usize> {
        let mut depths: Vec<usize> = self
            .frames
            .iter()
            .enumerate()
            .filter_map(|(i, frame)| {
                if frame.function_name == function_name {
                    Some(self.frames.len() - 1 - i) // Convert to depth from top
                } else {
                    None
                }
            })
            .collect();
        depths.sort(); // Sort depths from top (0) to bottom (highest)
        depths
    }

    /// Get a summary of the call stack
    pub fn get_summary(&self) -> CallStackSummary {
        let function_calls: Vec<String> = self
            .frames
            .iter()
            .rev() // Top to bottom
            .map(|frame| frame.function_name.clone())
            .collect();

        CallStackSummary {
            depth: self.frames.len(),
            max_depth: self.max_depth,
            function_calls,
        }
    }

    /// Set the maximum depth for the call stack
    pub fn set_max_depth(&mut self, max_depth: usize) {
        self.max_depth = max_depth;
    }

    /// Get the maximum depth setting
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }
}

/// Call stack summary information
#[derive(Debug, Clone)]
pub struct CallStackSummary {
    pub depth: usize,
    pub max_depth: usize,
    pub function_calls: Vec<String>,
}

/// Memory state management
#[derive(Debug, Clone)]
pub struct MemoryState {
    memory: HashMap<u64, u8>,
    allocated_regions: HashMap<u64, MemoryRegion>,
    next_allocation_id: u64,
    total_allocated: usize,
    max_memory_usage: usize,
}

/// Information about an allocated memory region
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    pub id: u64,
    pub start_address: u64,
    pub size: usize,
    pub allocation_time: std::time::SystemTime,
    pub label: Option<String>,
}

impl MemoryState {
    /// Create a new memory state
    pub fn new() -> Self {
        Self {
            memory: HashMap::new(),
            allocated_regions: HashMap::new(),
            next_allocation_id: 1,
            total_allocated: 0,
            max_memory_usage: usize::MAX,
        }
    }

    /// Create a new memory state with memory limit
    ///
    /// # Arguments
    /// * `max_memory_usage` - Maximum allowed memory usage in bytes
    pub fn with_limit(max_memory_usage: usize) -> Self {
        Self {
            memory: HashMap::new(),
            allocated_regions: HashMap::new(),
            next_allocation_id: 1,
            total_allocated: 0,
            max_memory_usage,
        }
    }

    /// Read memory from a specific address
    ///
    /// # Arguments
    /// * `address` - Starting address to read from
    /// * `size` - Number of bytes to read
    ///
    /// # Returns
    /// A vector containing the memory contents (uninitialized memory returns 0)
    pub fn read_memory(&self, address: u64, size: usize) -> JitResult<Vec<u8>> {
        let mut data = Vec::new();
        for i in 0..size {
            let byte = self.memory.get(&(address + i as u64)).copied().unwrap_or(0);
            data.push(byte);
        }
        Ok(data)
    }

    /// Write memory to a specific address
    ///
    /// # Arguments
    /// * `address` - Starting address to write to
    /// * `data` - Data to write
    pub fn write_memory(&mut self, address: u64, data: &[u8]) -> JitResult<()> {
        for (i, &byte) in data.iter().enumerate() {
            self.memory.insert(address + i as u64, byte);
        }
        Ok(())
    }

    /// Allocate a memory region
    ///
    /// # Arguments
    /// * `address` - Starting address of the region
    /// * `size` - Size of the region in bytes
    /// * `label` - Optional label for the region
    ///
    /// # Returns
    /// The allocation ID
    pub fn allocate_region(
        &mut self,
        address: u64,
        size: usize,
        label: Option<String>,
    ) -> JitResult<u64> {
        // Check for memory limit
        if self.total_allocated + size > self.max_memory_usage {
            return Err(JitError::RuntimeError(format!(
                "Memory allocation would exceed limit: {} + {} > {}",
                self.total_allocated, size, self.max_memory_usage
            )));
        }

        // Check for overlapping regions
        for region in self.allocated_regions.values() {
            let region_end = region.start_address + region.size as u64;
            let new_end = address + size as u64;

            if !(new_end <= region.start_address || address >= region_end) {
                return Err(JitError::RuntimeError(format!(
                    "Memory region overlap: new region [0x{:x}, 0x{:x}) overlaps with existing region [0x{:x}, 0x{:x})",
                    address, new_end, region.start_address, region_end
                )));
            }
        }

        let allocation_id = self.next_allocation_id;
        self.next_allocation_id += 1;

        let region = MemoryRegion {
            id: allocation_id,
            start_address: address,
            size,
            allocation_time: std::time::SystemTime::now(),
            label,
        };

        self.allocated_regions.insert(allocation_id, region);
        self.total_allocated += size;

        Ok(allocation_id)
    }

    /// Deallocate a memory region
    ///
    /// # Arguments
    /// * `allocation_id` - The ID of the allocation to free
    pub fn deallocate_region(&mut self, allocation_id: u64) -> JitResult<()> {
        if let Some(region) = self.allocated_regions.remove(&allocation_id) {
            self.total_allocated -= region.size;

            // Clear the memory in this region
            for i in 0..region.size {
                self.memory.remove(&(region.start_address + i as u64));
            }

            Ok(())
        } else {
            Err(JitError::RuntimeError(format!(
                "Allocation ID {} not found",
                allocation_id
            )))
        }
    }

    /// Get information about all allocated regions
    pub fn get_allocated_regions(&self) -> Vec<&MemoryRegion> {
        self.allocated_regions.values().collect()
    }

    /// Find the region containing a specific address
    ///
    /// # Arguments
    /// * `address` - The address to search for
    ///
    /// # Returns
    /// Information about the region containing the address, if any
    pub fn find_region_containing(&self, address: u64) -> Option<&MemoryRegion> {
        self.allocated_regions.values().find(|region| {
            address >= region.start_address && address < region.start_address + region.size as u64
        })
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        MemoryStats {
            total_allocated: self.total_allocated,
            max_memory_usage: self.max_memory_usage,
            allocated_regions_count: self.allocated_regions.len(),
            memory_utilization: if self.max_memory_usage > 0 {
                self.total_allocated as f64 / self.max_memory_usage as f64
            } else {
                0.0
            },
        }
    }

    /// Clear all memory and allocations
    pub fn clear(&mut self) {
        self.memory.clear();
        self.allocated_regions.clear();
        self.total_allocated = 0;
    }

    /// Read a specific type from memory
    ///
    /// # Arguments
    /// * `address` - Address to read from
    ///
    /// # Returns
    /// The value read from memory
    pub fn read_u32(&self, address: u64) -> JitResult<u32> {
        let bytes = self.read_memory(address, 4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Read a u64 from memory
    pub fn read_u64(&self, address: u64) -> JitResult<u64> {
        let bytes = self.read_memory(address, 8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    /// Read an f32 from memory
    pub fn read_f32(&self, address: u64) -> JitResult<f32> {
        let bytes = self.read_memory(address, 4)?;
        Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Read an f64 from memory
    pub fn read_f64(&self, address: u64) -> JitResult<f64> {
        let bytes = self.read_memory(address, 8)?;
        Ok(f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    /// Write a u32 to memory
    pub fn write_u32(&mut self, address: u64, value: u32) -> JitResult<()> {
        let bytes = value.to_le_bytes();
        self.write_memory(address, &bytes)
    }

    /// Write a u64 to memory
    pub fn write_u64(&mut self, address: u64, value: u64) -> JitResult<()> {
        let bytes = value.to_le_bytes();
        self.write_memory(address, &bytes)
    }

    /// Write an f32 to memory
    pub fn write_f32(&mut self, address: u64, value: f32) -> JitResult<()> {
        let bytes = value.to_le_bytes();
        self.write_memory(address, &bytes)
    }

    /// Write an f64 to memory
    pub fn write_f64(&mut self, address: u64, value: f64) -> JitResult<()> {
        let bytes = value.to_le_bytes();
        self.write_memory(address, &bytes)
    }

    /// Set memory limit
    pub fn set_memory_limit(&mut self, limit: usize) {
        self.max_memory_usage = limit;
    }

    /// Get memory limit
    pub fn memory_limit(&self) -> usize {
        self.max_memory_usage
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_allocated: usize,
    pub max_memory_usage: usize,
    pub allocated_regions_count: usize,
    pub memory_utilization: f64,
}

impl Default for CallStack {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for MemoryState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_call_stack_operations() {
        let mut stack = CallStack::new();
        assert!(stack.is_empty());
        assert_eq!(stack.depth(), 0);

        let frame = CallFrame {
            function_name: "test_function".to_string(),
            location: ExecutionLocation::GraphNode(crate::NodeId::new(0)),
            return_location: ExecutionLocation::GraphNode(crate::NodeId::new(1)),
            local_variables: HashMap::new(),
        };

        stack.push(frame.clone()).unwrap();
        assert_eq!(stack.depth(), 1);
        assert!(!stack.is_empty());

        let current = stack.current_frame().unwrap();
        assert_eq!(current.function_name, "test_function");

        let return_location = stack.pop();
        assert_eq!(stack.depth(), 0);
        assert!(matches!(return_location, ExecutionLocation::GraphNode(_)));
    }

    #[test]
    fn test_call_stack_max_depth() {
        let mut stack = CallStack::with_max_depth(2);

        let frame1 = CallFrame {
            function_name: "func1".to_string(),
            location: ExecutionLocation::GraphNode(crate::NodeId::new(0)),
            return_location: ExecutionLocation::Completed,
            local_variables: HashMap::new(),
        };

        let frame2 = CallFrame {
            function_name: "func2".to_string(),
            location: ExecutionLocation::GraphNode(crate::NodeId::new(1)),
            return_location: ExecutionLocation::Completed,
            local_variables: HashMap::new(),
        };

        let frame3 = CallFrame {
            function_name: "func3".to_string(),
            location: ExecutionLocation::GraphNode(crate::NodeId::new(2)),
            return_location: ExecutionLocation::Completed,
            local_variables: HashMap::new(),
        };

        assert!(stack.push(frame1).is_ok());
        assert!(stack.push(frame2).is_ok());
        assert!(stack.push(frame3).is_err()); // Should exceed max depth
    }

    #[test]
    fn test_memory_state_basic_operations() {
        let mut memory = MemoryState::new();

        let data = vec![1, 2, 3, 4, 5];
        memory.write_memory(0x1000, &data).unwrap();

        let read_data = memory.read_memory(0x1000, 5).unwrap();
        assert_eq!(read_data, data);

        let partial_data = memory.read_memory(0x1002, 2).unwrap();
        assert_eq!(partial_data, vec![3, 4]);
    }

    #[test]
    fn test_memory_allocation() {
        let mut memory = MemoryState::new();

        let id1 = memory
            .allocate_region(0x1000, 100, Some("test_region".to_string()))
            .unwrap();
        assert_eq!(memory.get_allocated_regions().len(), 1);

        let region = memory.find_region_containing(0x1050).unwrap();
        assert_eq!(region.id, id1);
        assert_eq!(region.size, 100);

        memory.deallocate_region(id1).unwrap();
        assert_eq!(memory.get_allocated_regions().len(), 0);
    }

    #[test]
    fn test_memory_overlap_detection() {
        let mut memory = MemoryState::new();

        memory.allocate_region(0x1000, 100, None).unwrap();

        // This should fail due to overlap
        let result = memory.allocate_region(0x1050, 100, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_typed_access() {
        let mut memory = MemoryState::new();

        memory.write_u32(0x1000, 0x12345678).unwrap();
        memory.write_f64(0x1004, 3.14159).unwrap();

        let u32_val = memory.read_u32(0x1000).unwrap();
        assert_eq!(u32_val, 0x12345678);

        let f64_val = memory.read_f64(0x1004).unwrap();
        assert!((f64_val - 3.14159).abs() < 1e-10);
    }

    #[test]
    fn test_memory_limit() {
        let mut memory = MemoryState::with_limit(50);

        // This should succeed
        memory.allocate_region(0x1000, 40, None).unwrap();

        // This should fail due to exceeding limit
        let result = memory.allocate_region(0x2000, 20, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_call_stack_find_functions() {
        let mut stack = CallStack::new();

        let frame1 = CallFrame {
            function_name: "main".to_string(),
            location: ExecutionLocation::GraphNode(crate::NodeId::new(0)),
            return_location: ExecutionLocation::Completed,
            local_variables: HashMap::new(),
        };

        let frame2 = CallFrame {
            function_name: "helper".to_string(),
            location: ExecutionLocation::GraphNode(crate::NodeId::new(1)),
            return_location: ExecutionLocation::Completed,
            local_variables: HashMap::new(),
        };

        let frame3 = CallFrame {
            function_name: "main".to_string(),
            location: ExecutionLocation::GraphNode(crate::NodeId::new(2)),
            return_location: ExecutionLocation::Completed,
            local_variables: HashMap::new(),
        };

        stack.push(frame1).unwrap();
        stack.push(frame2).unwrap();
        stack.push(frame3).unwrap();

        let main_indices = stack.find_frames_by_function("main");
        assert_eq!(main_indices.len(), 2);
        assert_eq!(main_indices, vec![0, 2]); // depths from top

        let helper_indices = stack.find_frames_by_function("helper");
        assert_eq!(helper_indices.len(), 1);
        assert_eq!(helper_indices[0], 1);
    }

    #[test]
    fn test_memory_stats() {
        let mut memory = MemoryState::with_limit(1000);

        memory.allocate_region(0x1000, 100, None).unwrap();
        memory.allocate_region(0x2000, 200, None).unwrap();

        let stats = memory.get_memory_stats();
        assert_eq!(stats.total_allocated, 300);
        assert_eq!(stats.max_memory_usage, 1000);
        assert_eq!(stats.allocated_regions_count, 2);
        assert_eq!(stats.memory_utilization, 0.3);
    }
}
