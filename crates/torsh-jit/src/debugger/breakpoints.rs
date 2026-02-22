//! Breakpoint management for JIT debugging
//!
//! This module provides comprehensive breakpoint management capabilities including
//! setting, removing, and checking breakpoints at various execution locations.

use super::core::{Breakpoint, BreakpointId, BreakpointLocation, ExecutionLocation};
use crate::{JitError, JitResult};
use std::collections::HashMap;

/// Breakpoint manager
pub struct BreakpointManager {
    breakpoints: HashMap<BreakpointId, Breakpoint>,
    next_id: BreakpointId,
}

impl BreakpointManager {
    /// Create a new breakpoint manager
    pub fn new() -> Self {
        Self {
            breakpoints: HashMap::new(),
            next_id: BreakpointId(0),
        }
    }

    /// Set a breakpoint at the specified location
    ///
    /// # Arguments
    /// * `location` - The location where the breakpoint should be set
    ///
    /// # Returns
    /// The ID of the newly created breakpoint
    ///
    /// # Examples
    /// ```rust
    /// use torsh_jit::debugger::{BreakpointManager, BreakpointLocation};
    /// use torsh_jit::NodeId;
    ///
    /// let mut manager = BreakpointManager::new();
    /// let location = BreakpointLocation::GraphNode(NodeId::new(0));
    /// let id = manager.set_breakpoint(location).unwrap();
    /// ```
    pub fn set_breakpoint(&mut self, location: BreakpointLocation) -> JitResult<BreakpointId> {
        let id = self.next_id;
        self.next_id = BreakpointId(self.next_id.0 + 1);

        let breakpoint = Breakpoint {
            id,
            location,
            condition: None,
            enabled: true,
            hit_count: 0,
        };

        self.breakpoints.insert(id, breakpoint);
        Ok(id)
    }

    /// Set a conditional breakpoint at the specified location
    ///
    /// # Arguments
    /// * `location` - The location where the breakpoint should be set
    /// * `condition` - The condition that must be true for the breakpoint to trigger
    ///
    /// # Returns
    /// The ID of the newly created breakpoint
    pub fn set_conditional_breakpoint(
        &mut self,
        location: BreakpointLocation,
        condition: String,
    ) -> JitResult<BreakpointId> {
        let id = self.next_id;
        self.next_id = BreakpointId(self.next_id.0 + 1);

        let breakpoint = Breakpoint {
            id,
            location,
            condition: Some(condition),
            enabled: true,
            hit_count: 0,
        };

        self.breakpoints.insert(id, breakpoint);
        Ok(id)
    }

    /// Remove a breakpoint by ID
    ///
    /// # Arguments
    /// * `id` - The ID of the breakpoint to remove
    ///
    /// # Returns
    /// `Ok(())` if the breakpoint was removed, error if not found
    pub fn remove_breakpoint(&mut self, id: BreakpointId) -> JitResult<()> {
        if self.breakpoints.remove(&id).is_some() {
            Ok(())
        } else {
            Err(JitError::RuntimeError(format!(
                "Breakpoint {} not found",
                id.0
            )))
        }
    }

    /// Enable a breakpoint
    ///
    /// # Arguments
    /// * `id` - The ID of the breakpoint to enable
    pub fn enable_breakpoint(&mut self, id: BreakpointId) -> JitResult<()> {
        if let Some(breakpoint) = self.breakpoints.get_mut(&id) {
            breakpoint.enabled = true;
            Ok(())
        } else {
            Err(JitError::RuntimeError(format!(
                "Breakpoint {} not found",
                id.0
            )))
        }
    }

    /// Disable a breakpoint
    ///
    /// # Arguments
    /// * `id` - The ID of the breakpoint to disable
    pub fn disable_breakpoint(&mut self, id: BreakpointId) -> JitResult<()> {
        if let Some(breakpoint) = self.breakpoints.get_mut(&id) {
            breakpoint.enabled = false;
            Ok(())
        } else {
            Err(JitError::RuntimeError(format!(
                "Breakpoint {} not found",
                id.0
            )))
        }
    }

    /// Get a list of all breakpoints
    ///
    /// # Returns
    /// A vector of references to all breakpoints
    pub fn list_breakpoints(&self) -> Vec<&Breakpoint> {
        self.breakpoints.values().collect()
    }

    /// Get a specific breakpoint by ID
    ///
    /// # Arguments
    /// * `id` - The ID of the breakpoint to retrieve
    ///
    /// # Returns
    /// An optional reference to the breakpoint
    pub fn get_breakpoint(&self, id: BreakpointId) -> Option<&Breakpoint> {
        self.breakpoints.get(&id)
    }

    /// Check if there is a breakpoint at the specified location
    ///
    /// # Arguments
    /// * `location` - The execution location to check
    ///
    /// # Returns
    /// `true` if there is an enabled breakpoint at the location, `false` otherwise
    pub fn is_breakpoint_at(&self, location: &ExecutionLocation) -> bool {
        self.breakpoints.values().any(|bp| {
            bp.enabled
                && match (&bp.location, location) {
                    (
                        BreakpointLocation::GraphNode(bp_node),
                        ExecutionLocation::GraphNode(loc_node),
                    ) => bp_node == loc_node,
                    (
                        BreakpointLocation::Instruction {
                            function: bp_func,
                            instruction: bp_inst,
                        },
                        ExecutionLocation::Instruction {
                            function: loc_func,
                            instruction_index: loc_inst,
                        },
                    ) => bp_func == loc_func && *bp_inst == *loc_inst,
                    _ => false,
                }
        })
    }

    /// Increment hit count for breakpoints at the specified location
    ///
    /// # Arguments
    /// * `location` - The execution location where a hit occurred
    ///
    /// # Returns
    /// The number of breakpoints hit at this location
    pub fn hit_breakpoints_at(&mut self, location: &ExecutionLocation) -> usize {
        let mut hit_count = 0;
        for breakpoint in self.breakpoints.values_mut() {
            if breakpoint.enabled
                && match (&breakpoint.location, location) {
                    (
                        BreakpointLocation::GraphNode(bp_node),
                        ExecutionLocation::GraphNode(loc_node),
                    ) => bp_node == loc_node,
                    (
                        BreakpointLocation::Instruction {
                            function: bp_func,
                            instruction: bp_inst,
                        },
                        ExecutionLocation::Instruction {
                            function: loc_func,
                            instruction_index: loc_inst,
                        },
                    ) => bp_func == loc_func && *bp_inst == *loc_inst,
                    _ => false,
                }
            {
                breakpoint.hit_count += 1;
                hit_count += 1;
            }
        }
        hit_count
    }

    /// Clear all breakpoints
    pub fn clear_all_breakpoints(&mut self) {
        self.breakpoints.clear();
    }

    /// Get the number of breakpoints
    pub fn count(&self) -> usize {
        self.breakpoints.len()
    }

    /// Get the number of enabled breakpoints
    pub fn enabled_count(&self) -> usize {
        self.breakpoints.values().filter(|bp| bp.enabled).count()
    }

    /// Get all breakpoints at a specific location
    ///
    /// # Arguments
    /// * `location` - The location to search for breakpoints
    ///
    /// # Returns
    /// A vector of references to breakpoints at the specified location
    pub fn get_breakpoints_at(&self, location: &BreakpointLocation) -> Vec<&Breakpoint> {
        self.breakpoints
            .values()
            .filter(|bp| match (&bp.location, location) {
                (
                    BreakpointLocation::GraphNode(bp_node),
                    BreakpointLocation::GraphNode(loc_node),
                ) => bp_node == loc_node,
                (
                    BreakpointLocation::Instruction {
                        function: bp_func,
                        instruction: bp_inst,
                    },
                    BreakpointLocation::Instruction {
                        function: loc_func,
                        instruction: loc_inst,
                    },
                ) => bp_func == loc_func && bp_inst == loc_inst,
                _ => false,
            })
            .collect()
    }
}

impl Default for BreakpointManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NodeId;

    #[test]
    fn test_breakpoint_manager_creation() {
        let manager = BreakpointManager::new();
        assert_eq!(manager.count(), 0);
        assert_eq!(manager.enabled_count(), 0);
    }

    #[test]
    fn test_set_and_remove_breakpoint() {
        let mut manager = BreakpointManager::new();

        let location = BreakpointLocation::GraphNode(NodeId::new(0));
        let id = manager.set_breakpoint(location).unwrap();

        assert_eq!(manager.count(), 1);
        assert_eq!(manager.enabled_count(), 1);
        assert!(manager.remove_breakpoint(id).is_ok());
        assert_eq!(manager.count(), 0);
    }

    #[test]
    fn test_conditional_breakpoint() {
        let mut manager = BreakpointManager::new();

        let location = BreakpointLocation::GraphNode(NodeId::new(0));
        let condition = "x > 10".to_string();
        let id = manager
            .set_conditional_breakpoint(location, condition.clone())
            .unwrap();

        let breakpoint = manager.get_breakpoint(id).unwrap();
        assert_eq!(breakpoint.condition, Some(condition));
    }

    #[test]
    fn test_enable_disable_breakpoint() {
        let mut manager = BreakpointManager::new();

        let location = BreakpointLocation::GraphNode(NodeId::new(0));
        let id = manager.set_breakpoint(location).unwrap();

        assert_eq!(manager.enabled_count(), 1);

        manager.disable_breakpoint(id).unwrap();
        assert_eq!(manager.enabled_count(), 0);

        manager.enable_breakpoint(id).unwrap();
        assert_eq!(manager.enabled_count(), 1);
    }

    #[test]
    fn test_is_breakpoint_at() {
        let mut manager = BreakpointManager::new();

        let node_id = NodeId::new(0);
        let location = BreakpointLocation::GraphNode(node_id);
        manager.set_breakpoint(location).unwrap();

        let exec_location = ExecutionLocation::GraphNode(node_id);
        assert!(manager.is_breakpoint_at(&exec_location));

        let other_exec_location = ExecutionLocation::GraphNode(NodeId::new(1));
        assert!(!manager.is_breakpoint_at(&other_exec_location));
    }

    #[test]
    fn test_hit_breakpoints() {
        let mut manager = BreakpointManager::new();

        let node_id = NodeId::new(0);
        let location = BreakpointLocation::GraphNode(node_id);
        let id = manager.set_breakpoint(location).unwrap();

        let exec_location = ExecutionLocation::GraphNode(node_id);
        let hit_count = manager.hit_breakpoints_at(&exec_location);
        assert_eq!(hit_count, 1);

        let breakpoint = manager.get_breakpoint(id).unwrap();
        assert_eq!(breakpoint.hit_count, 1);
    }

    #[test]
    fn test_clear_all_breakpoints() {
        let mut manager = BreakpointManager::new();

        manager
            .set_breakpoint(BreakpointLocation::GraphNode(NodeId::new(0)))
            .unwrap();
        manager
            .set_breakpoint(BreakpointLocation::GraphNode(NodeId::new(1)))
            .unwrap();

        assert_eq!(manager.count(), 2);
        manager.clear_all_breakpoints();
        assert_eq!(manager.count(), 0);
    }
}
