//! CPU cache hierarchy information
//!
//! This module provides structures to represent CPU cache information
//! including L1/L2/L3 cache sizes, associativity, and TLB information.

/// Cache hierarchy information
#[derive(Debug, Clone)]
pub struct CacheInfo {
    /// L1 instruction cache size in bytes
    pub l1i_size: usize,
    /// L1 data cache size in bytes
    pub l1d_size: usize,
    /// L1 cache associativity
    pub l1_associativity: usize,
    /// L1 cache line size in bytes
    pub l1_line_size: usize,
    /// L2 cache size in bytes
    pub l2_size: usize,
    /// L2 cache associativity
    pub l2_associativity: usize,
    /// L2 cache line size in bytes
    pub l2_line_size: usize,
    /// L3 cache size in bytes
    pub l3_size: usize,
    /// L3 cache associativity
    pub l3_associativity: usize,
    /// L3 cache line size in bytes
    pub l3_line_size: usize,
    /// Number of cores sharing L3 cache
    pub l3_sharing: usize,
    /// TLB entries for 4KB pages
    pub tlb_4kb_entries: usize,
    /// TLB entries for 2MB pages
    pub tlb_2mb_entries: usize,
    /// TLB entries for 1GB pages
    pub tlb_1gb_entries: usize,
}

impl Default for CacheInfo {
    fn default() -> Self {
        Self {
            l1i_size: 32 * 1024, // 32KB
            l1d_size: 32 * 1024, // 32KB
            l1_associativity: 8,
            l1_line_size: 64,
            l2_size: 256 * 1024, // 256KB
            l2_associativity: 8,
            l2_line_size: 64,
            l3_size: 8 * 1024 * 1024, // 8MB
            l3_associativity: 16,
            l3_line_size: 64,
            l3_sharing: 8,
            tlb_4kb_entries: 64,
            tlb_2mb_entries: 32,
            tlb_1gb_entries: 4,
        }
    }
}
