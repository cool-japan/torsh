// Memory Profiling: Fragmentation Tracking and Management
//
// This module provides comprehensive memory fragmentation tracking, analysis, and management
// for the ToRSh deep learning framework. It includes advanced fragmentation detection algorithms,
// defragmentation strategies, and prevention mechanisms to optimize memory utilization.

use std::collections::{HashMap, BTreeMap, BTreeSet, VecDeque};
use std::cmp::{Ordering, max, min};
use std::time::{Instant, Duration};
use scirs2_core::error::{CoreError, Result};

/// Memory block representation for fragmentation analysis
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryBlock {
    pub start_addr: usize,
    pub size: usize,
    pub end_addr: usize,
    pub block_type: BlockType,
    pub allocation_time: Instant,
    pub last_access_time: Option<Instant>,
    pub access_frequency: f64,
    pub alignment: usize,
    pub metadata: BlockMetadata,
}

/// Type of memory block
#[derive(Debug, Clone, PartialEq)]
pub enum BlockType {
    Allocated {
        allocation_id: usize,
        size_class: SizeClass,
        purpose: AllocationPurpose,
    },
    Free {
        fragmentation_level: f64,
        mergeable_neighbors: usize,
        coalescing_potential: f64,
    },
    Reserved {
        reservation_id: usize,
        expiry_time: Option<Instant>,
    },
    Guard {
        protection_level: u8,
        associated_allocation: Option<usize>,
    },
}

/// Size classification for memory blocks
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum SizeClass {
    Tiny,        // < 64 bytes
    Small,       // 64B - 1KB
    Medium,      // 1KB - 64KB
    Large,       // 64KB - 1MB
    Huge,        // 1MB - 16MB
    Massive,     // > 16MB
}

/// Purpose classification for allocations
#[derive(Debug, Clone, PartialEq)]
pub enum AllocationPurpose {
    TensorData,
    GradientBuffer,
    ActivationCache,
    WeightData,
    TemporaryBuffer,
    SystemOverhead,
    UserData,
    Unknown,
}

/// Metadata associated with memory blocks
#[derive(Debug, Clone, PartialEq)]
pub struct BlockMetadata {
    pub allocation_source: String,
    pub thread_id: Option<u64>,
    pub tensor_shape: Option<Vec<usize>>,
    pub data_type: Option<String>,
    pub lifetime_hint: LifetimeHint,
    pub usage_pattern: UsagePattern,
}

/// Hint about expected lifetime of allocation
#[derive(Debug, Clone, PartialEq)]
pub enum LifetimeHint {
    Ephemeral,      // Very short-lived
    Transient,      // Short-lived
    Persistent,     // Long-lived
    Permanent,      // For duration of program
    Unknown,
}

/// Pattern of memory usage
#[derive(Debug, Clone, PartialEq)]
pub enum UsagePattern {
    Sequential,     // Accessed sequentially
    Random,         // Random access pattern
    Streaming,      // Streamed access (read-once)
    Frequent,       // Frequently accessed
    Infrequent,     // Rarely accessed
    Unknown,
}

/// Comprehensive fragmentation analysis results
#[derive(Debug, Clone)]
pub struct FragmentationAnalysis {
    pub overall_fragmentation_index: f64,
    pub external_fragmentation: ExternalFragmentation,
    pub internal_fragmentation: InternalFragmentation,
    pub fragmentation_hotspots: Vec<FragmentationHotspot>,
    pub efficiency_metrics: FragmentationEfficiency,
    pub trend_analysis: FragmentationTrend,
    pub impact_assessment: FragmentationImpact,
    pub mitigation_recommendations: Vec<FragmentationMitigationStrategy>,
}

/// External fragmentation analysis
#[derive(Debug, Clone)]
pub struct ExternalFragmentation {
    pub free_block_count: usize,
    pub largest_free_block: usize,
    pub total_free_space: usize,
    pub average_free_block_size: f64,
    pub free_block_size_distribution: HashMap<SizeClass, usize>,
    pub fragmentation_ratio: f64,
    pub compaction_potential: f64,
}

/// Internal fragmentation analysis
#[derive(Debug, Clone)]
pub struct InternalFragmentation {
    pub total_internal_waste: usize,
    pub average_waste_per_allocation: f64,
    pub worst_case_waste: usize,
    pub waste_by_size_class: HashMap<SizeClass, usize>,
    pub alignment_waste: usize,
    pub padding_waste: usize,
    pub efficiency_score: f64,
}

/// Fragmentation hotspot identification
#[derive(Debug, Clone)]
pub struct FragmentationHotspot {
    pub region_start: usize,
    pub region_end: usize,
    pub region_size: usize,
    pub fragmentation_density: f64,
    pub block_count: usize,
    pub free_space_ratio: f64,
    pub defragmentation_priority: Priority,
    pub estimated_benefit: f64,
    pub complexity_score: f64,
}

/// Priority levels for operations
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Priority {
    Critical,
    High,
    Medium,
    Low,
    Deferred,
}

/// Fragmentation efficiency metrics
#[derive(Debug, Clone)]
pub struct FragmentationEfficiency {
    pub memory_utilization: f64,
    pub allocation_success_rate: f64,
    pub average_search_time: Duration,
    pub defragmentation_overhead: f64,
    pub compaction_frequency: f64,
    pub waste_ratio: f64,
}

/// Fragmentation trend analysis
#[derive(Debug, Clone)]
pub struct FragmentationTrend {
    pub direction: TrendDirection,
    pub rate_of_change: f64,
    pub prediction: FragmentationPrediction,
    pub contributing_factors: Vec<String>,
    pub historical_pattern: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    Improving,
    Degrading,
    Stable,
    Oscillating,
}

/// Prediction about future fragmentation
#[derive(Debug, Clone)]
pub struct FragmentationPrediction {
    pub predicted_fragmentation: f64,
    pub confidence: f64,
    pub time_to_critical: Option<Duration>,
    pub recommended_action_time: Duration,
}

/// Impact assessment of fragmentation
#[derive(Debug, Clone)]
pub struct FragmentationImpact {
    pub performance_degradation: f64,
    pub memory_overhead: f64,
    pub allocation_failures: usize,
    pub cache_efficiency_impact: f64,
    pub bandwidth_loss: f64,
    pub system_stability_risk: f64,
}

/// Strategy for fragmentation mitigation
#[derive(Debug, Clone)]
pub struct FragmentationMitigationStrategy {
    pub strategy_type: MitigationType,
    pub priority: Priority,
    pub description: String,
    pub expected_benefit: f64,
    pub implementation_cost: f64,
    pub time_to_implement: Duration,
    pub side_effects: Vec<String>,
    pub requirements: Vec<String>,
}

/// Type of mitigation strategy
#[derive(Debug, Clone, PartialEq)]
pub enum MitigationType {
    Compaction,
    Coalescing,
    PreventiveAllocation,
    MemoryPooling,
    AlgorithmicOptimization,
    HardwareOptimization,
    PolicyAdjustment,
}

/// Defragmentation algorithm implementations
#[derive(Debug, Clone)]
pub struct DefragmentationAlgorithm {
    pub algorithm_type: DefragmentationType,
    pub name: String,
    pub description: String,
    pub complexity: AlgorithmComplexity,
    pub effectiveness: f64,
    pub overhead: f64,
    pub suitability: Vec<FragmentationPattern>,
}

/// Type of defragmentation algorithm
#[derive(Debug, Clone, PartialEq)]
pub enum DefragmentationType {
    CompactionBased,
    CoalescingBased,
    SegregationBased,
    BuddySystem,
    SlabAllocation,
    RegionBased,
    Hybrid,
}

/// Complexity classification for algorithms
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum AlgorithmComplexity {
    Constant,     // O(1)
    Logarithmic,  // O(log n)
    Linear,       // O(n)
    Quadratic,    // O(n²)
    Exponential,  // O(2^n)
}

/// Fragmentation pattern classification
#[derive(Debug, Clone, PartialEq)]
pub enum FragmentationPattern {
    HighExternalLowInternal,
    HighInternalLowExternal,
    HighBothTypes,
    LowBothTypes,
    ChessboardPattern,
    ConcentratedHotspots,
    DistributedFragmentation,
}

/// Memory region for targeted defragmentation
#[derive(Debug, Clone)]
pub struct DefragmentationRegion {
    pub start_addr: usize,
    pub end_addr: usize,
    pub blocks: Vec<MemoryBlock>,
    pub fragmentation_score: f64,
    pub defragmentation_potential: f64,
    pub algorithm_recommendation: DefragmentationType,
    pub estimated_time: Duration,
    pub expected_gain: f64,
}

/// Main fragmentation manager
pub struct FragmentationManager {
    memory_blocks: BTreeMap<usize, MemoryBlock>,
    free_blocks: BTreeSet<FreeBlockEntry>,
    fragmentation_history: VecDeque<FragmentationSnapshot>,
    defragmentation_algorithms: HashMap<DefragmentationType, DefragmentationAlgorithm>,
    active_mitigations: Vec<ActiveMitigation>,
    configuration: FragmentationConfig,
    statistics: FragmentationStatistics,
}

/// Entry for free block tracking
#[derive(Debug, Clone, PartialEq, Eq)]
struct FreeBlockEntry {
    size: usize,
    start_addr: usize,
    fragmentation_level: f64,
}

impl Ord for FreeBlockEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Primary sort by size, secondary by address
        self.size.cmp(&other.size)
            .then_with(|| self.start_addr.cmp(&other.start_addr))
    }
}

impl PartialOrd for FreeBlockEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Snapshot of fragmentation state
#[derive(Debug, Clone)]
pub struct FragmentationSnapshot {
    pub timestamp: Instant,
    pub fragmentation_index: f64,
    pub block_count: usize,
    pub free_space: usize,
    pub largest_free_block: usize,
    pub allocation_failures: usize,
}

/// Active mitigation being applied
#[derive(Debug, Clone)]
pub struct ActiveMitigation {
    pub strategy: FragmentationMitigationStrategy,
    pub started_at: Instant,
    pub progress: f64,
    pub current_phase: String,
    pub estimated_completion: Instant,
}

/// Configuration for fragmentation management
#[derive(Debug, Clone)]
pub struct FragmentationConfig {
    pub fragmentation_threshold: f64,
    pub compaction_trigger: f64,
    pub defragmentation_window: Duration,
    pub analysis_interval: Duration,
    pub history_retention: Duration,
    pub enable_predictive_defrag: bool,
    pub enable_background_compaction: bool,
    pub max_defrag_overhead: f64,
}

impl Default for FragmentationConfig {
    fn default() -> Self {
        Self {
            fragmentation_threshold: 0.3,
            compaction_trigger: 0.5,
            defragmentation_window: Duration::from_secs(60),
            analysis_interval: Duration::from_secs(10),
            history_retention: Duration::from_secs(3600),
            enable_predictive_defrag: true,
            enable_background_compaction: true,
            max_defrag_overhead: 0.1,
        }
    }
}

/// Statistical tracking for fragmentation
#[derive(Debug, Clone)]
pub struct FragmentationStatistics {
    pub total_fragmentation_events: usize,
    pub successful_defragmentations: usize,
    pub average_fragmentation_level: f64,
    pub peak_fragmentation_level: f64,
    pub time_spent_defragmenting: Duration,
    pub memory_recovered: usize,
    pub allocation_failures_prevented: usize,
}

impl Default for FragmentationStatistics {
    fn default() -> Self {
        Self {
            total_fragmentation_events: 0,
            successful_defragmentations: 0,
            average_fragmentation_level: 0.0,
            peak_fragmentation_level: 0.0,
            time_spent_defragmenting: Duration::default(),
            memory_recovered: 0,
            allocation_failures_prevented: 0,
        }
    }
}

impl MemoryBlock {
    pub fn new_allocated(start_addr: usize, size: usize, allocation_id: usize, purpose: AllocationPurpose) -> Self {
        Self {
            start_addr,
            size,
            end_addr: start_addr + size,
            block_type: BlockType::Allocated {
                allocation_id,
                size_class: Self::classify_size(size),
                purpose,
            },
            allocation_time: Instant::now(),
            last_access_time: None,
            access_frequency: 0.0,
            alignment: Self::calculate_alignment(start_addr),
            metadata: BlockMetadata {
                allocation_source: "unknown".to_string(),
                thread_id: None,
                tensor_shape: None,
                data_type: None,
                lifetime_hint: LifetimeHint::Unknown,
                usage_pattern: UsagePattern::Unknown,
            },
        }
    }

    pub fn new_free(start_addr: usize, size: usize) -> Self {
        Self {
            start_addr,
            size,
            end_addr: start_addr + size,
            block_type: BlockType::Free {
                fragmentation_level: 0.0,
                mergeable_neighbors: 0,
                coalescing_potential: 0.0,
            },
            allocation_time: Instant::now(),
            last_access_time: None,
            access_frequency: 0.0,
            alignment: Self::calculate_alignment(start_addr),
            metadata: BlockMetadata {
                allocation_source: "freed".to_string(),
                thread_id: None,
                tensor_shape: None,
                data_type: None,
                lifetime_hint: LifetimeHint::Unknown,
                usage_pattern: UsagePattern::Unknown,
            },
        }
    }

    fn classify_size(size: usize) -> SizeClass {
        match size {
            0..=63 => SizeClass::Tiny,
            64..=1023 => SizeClass::Small,
            1024..=65535 => SizeClass::Medium,
            65536..=1048575 => SizeClass::Large,
            1048576..=16777215 => SizeClass::Huge,
            _ => SizeClass::Massive,
        }
    }

    fn calculate_alignment(addr: usize) -> usize {
        if addr == 0 {
            return 1;
        }

        let mut alignment = 1;
        let mut temp_addr = addr;

        while temp_addr % 2 == 0 {
            alignment *= 2;
            temp_addr /= 2;
        }

        alignment
    }

    pub fn is_adjacent_to(&self, other: &MemoryBlock) -> bool {
        self.end_addr == other.start_addr || other.end_addr == self.start_addr
    }

    pub fn can_merge_with(&self, other: &MemoryBlock) -> bool {
        matches!((&self.block_type, &other.block_type),
                (BlockType::Free { .. }, BlockType::Free { .. })) &&
        self.is_adjacent_to(other)
    }

    pub fn merge_with(&self, other: &MemoryBlock) -> Option<MemoryBlock> {
        if !self.can_merge_with(other) {
            return None;
        }

        let (start_addr, end_addr) = if self.start_addr < other.start_addr {
            (self.start_addr, other.end_addr)
        } else {
            (other.start_addr, self.end_addr)
        };

        let size = end_addr - start_addr;

        Some(MemoryBlock {
            start_addr,
            size,
            end_addr,
            block_type: BlockType::Free {
                fragmentation_level: 0.0,
                mergeable_neighbors: 0,
                coalescing_potential: 1.0,
            },
            allocation_time: min(self.allocation_time, other.allocation_time),
            last_access_time: None,
            access_frequency: 0.0,
            alignment: Self::calculate_alignment(start_addr),
            metadata: self.metadata.clone(), // Use first block's metadata
        })
    }
}

impl FragmentationManager {
    pub fn new() -> Self {
        Self::with_config(FragmentationConfig::default())
    }

    pub fn with_config(config: FragmentationConfig) -> Self {
        let mut algorithms = HashMap::new();

        // Initialize defragmentation algorithms
        Self::initialize_algorithms(&mut algorithms);

        Self {
            memory_blocks: BTreeMap::new(),
            free_blocks: BTreeSet::new(),
            fragmentation_history: VecDeque::new(),
            defragmentation_algorithms: algorithms,
            active_mitigations: Vec::new(),
            configuration: config,
            statistics: FragmentationStatistics::default(),
        }
    }

    fn initialize_algorithms(algorithms: &mut HashMap<DefragmentationType, DefragmentationAlgorithm>) {
        algorithms.insert(DefragmentationType::CompactionBased, DefragmentationAlgorithm {
            algorithm_type: DefragmentationType::CompactionBased,
            name: "Compaction-Based Defragmentation".to_string(),
            description: "Moves allocated blocks to consolidate free space".to_string(),
            complexity: AlgorithmComplexity::Linear,
            effectiveness: 0.9,
            overhead: 0.3,
            suitability: vec![FragmentationPattern::HighExternalLowInternal],
        });

        algorithms.insert(DefragmentationType::CoalescingBased, DefragmentationAlgorithm {
            algorithm_type: DefragmentationType::CoalescingBased,
            name: "Coalescing-Based Defragmentation".to_string(),
            description: "Merges adjacent free blocks to reduce fragmentation".to_string(),
            complexity: AlgorithmComplexity::Logarithmic,
            effectiveness: 0.7,
            overhead: 0.1,
            suitability: vec![FragmentationPattern::ChessboardPattern],
        });

        algorithms.insert(DefragmentationType::BuddySystem, DefragmentationAlgorithm {
            algorithm_type: DefragmentationType::BuddySystem,
            name: "Buddy System Allocation".to_string(),
            description: "Uses power-of-2 sized blocks with buddy merging".to_string(),
            complexity: AlgorithmComplexity::Logarithmic,
            effectiveness: 0.8,
            overhead: 0.2,
            suitability: vec![FragmentationPattern::HighInternalLowExternal],
        });
    }

    /// Track a new memory block allocation
    pub fn track_allocation(&mut self, block: MemoryBlock) -> Result<()> {
        let addr = block.start_addr;

        // Remove any free blocks that this allocation overlaps
        self.remove_overlapping_free_blocks(&block);

        // Insert the allocated block
        self.memory_blocks.insert(addr, block);

        // Update fragmentation analysis
        self.update_fragmentation_analysis()?;

        Ok(())
    }

    /// Track memory deallocation and create free block
    pub fn track_deallocation(&mut self, start_addr: usize) -> Result<()> {
        if let Some(block) = self.memory_blocks.remove(&start_addr) {
            let mut free_block = MemoryBlock::new_free(block.start_addr, block.size);

            // Try to coalesce with adjacent free blocks
            free_block = self.attempt_coalescing(free_block)?;

            // Add to free blocks tracking
            self.add_free_block(&free_block);

            // Update fragmentation analysis
            self.update_fragmentation_analysis()?;
        }

        Ok(())
    }

    /// Perform comprehensive fragmentation analysis
    pub fn analyze_fragmentation(&mut self) -> Result<FragmentationAnalysis> {
        let external_frag = self.analyze_external_fragmentation();
        let internal_frag = self.analyze_internal_fragmentation();
        let hotspots = self.identify_fragmentation_hotspots();
        let efficiency = self.calculate_efficiency_metrics();
        let trend = self.analyze_fragmentation_trend();
        let impact = self.assess_fragmentation_impact(&external_frag, &internal_frag);
        let strategies = self.recommend_mitigation_strategies(&external_frag, &internal_frag, &hotspots);

        let overall_index = (external_frag.fragmentation_ratio +
                           (1.0 - internal_frag.efficiency_score)) / 2.0;

        Ok(FragmentationAnalysis {
            overall_fragmentation_index: overall_index,
            external_fragmentation: external_frag,
            internal_fragmentation: internal_frag,
            fragmentation_hotspots: hotspots,
            efficiency_metrics: efficiency,
            trend_analysis: trend,
            impact_assessment: impact,
            mitigation_recommendations: strategies,
        })
    }

    /// Execute defragmentation strategy
    pub fn defragment(&mut self, strategy: DefragmentationType) -> Result<DefragmentationResult> {
        let start_time = Instant::now();

        let algorithm = self.defragmentation_algorithms.get(&strategy)
            .ok_or_else(|| CoreError::InvalidOperation("Unknown defragmentation strategy".to_string()))?;

        let result = match strategy {
            DefragmentationType::CompactionBased => self.compact_memory()?,
            DefragmentationType::CoalescingBased => self.coalesce_free_blocks()?,
            DefragmentationType::BuddySystem => self.apply_buddy_system()?,
            _ => return Err(CoreError::InvalidOperation("Strategy not implemented".to_string())),
        };

        let elapsed = start_time.elapsed();

        // Update statistics
        self.statistics.successful_defragmentations += 1;
        self.statistics.time_spent_defragmenting += elapsed;
        self.statistics.memory_recovered += result.memory_recovered;

        Ok(result)
    }

    /// Remove overlapping free blocks when allocating
    fn remove_overlapping_free_blocks(&mut self, block: &MemoryBlock) {
        let overlapping: Vec<_> = self.free_blocks.iter()
            .filter(|fb| {
                let fb_end = fb.start_addr + fb.size;
                !(fb_end <= block.start_addr || fb.start_addr >= block.end_addr)
            })
            .cloned()
            .collect();

        for fb in overlapping {
            self.free_blocks.remove(&fb);
        }
    }

    /// Attempt to coalesce free block with neighbors
    fn attempt_coalescing(&mut self, mut free_block: MemoryBlock) -> Result<MemoryBlock> {
        let mut changed = true;

        while changed {
            changed = false;

            // Find adjacent blocks that can be merged
            let adjacent: Vec<_> = self.free_blocks.iter()
                .filter(|fb| {
                    let fb_block = MemoryBlock::new_free(fb.start_addr, fb.size);
                    free_block.can_merge_with(&fb_block)
                })
                .cloned()
                .collect();

            for fb in adjacent {
                let fb_block = MemoryBlock::new_free(fb.start_addr, fb.size);
                if let Some(merged) = free_block.merge_with(&fb_block) {
                    self.free_blocks.remove(&fb);
                    free_block = merged;
                    changed = true;
                    break;
                }
            }
        }

        Ok(free_block)
    }

    /// Add free block to tracking structures
    fn add_free_block(&mut self, free_block: &MemoryBlock) {
        if let BlockType::Free { fragmentation_level, .. } = &free_block.block_type {
            let entry = FreeBlockEntry {
                size: free_block.size,
                start_addr: free_block.start_addr,
                fragmentation_level: *fragmentation_level,
            };
            self.free_blocks.insert(entry);
        }
    }

    /// Update fragmentation analysis
    fn update_fragmentation_analysis(&mut self) -> Result<()> {
        let fragmentation_index = self.calculate_fragmentation_index();

        let snapshot = FragmentationSnapshot {
            timestamp: Instant::now(),
            fragmentation_index,
            block_count: self.memory_blocks.len(),
            free_space: self.free_blocks.iter().map(|fb| fb.size).sum(),
            largest_free_block: self.free_blocks.iter().map(|fb| fb.size).max().unwrap_or(0),
            allocation_failures: 0, // Would track actual failures
        };

        self.fragmentation_history.push_back(snapshot);

        // Maintain history size
        while self.fragmentation_history.len() > 10000 {
            self.fragmentation_history.pop_front();
        }

        Ok(())
    }

    /// Calculate overall fragmentation index
    fn calculate_fragmentation_index(&self) -> f64 {
        if self.free_blocks.is_empty() {
            return 0.0;
        }

        let total_free_space: usize = self.free_blocks.iter().map(|fb| fb.size).sum();
        let largest_free_block = self.free_blocks.iter().map(|fb| fb.size).max().unwrap_or(0);

        if total_free_space == 0 {
            0.0
        } else {
            1.0 - (largest_free_block as f64 / total_free_space as f64)
        }
    }

    /// Analyze external fragmentation
    fn analyze_external_fragmentation(&self) -> ExternalFragmentation {
        let free_block_count = self.free_blocks.len();
        let total_free_space: usize = self.free_blocks.iter().map(|fb| fb.size).sum();
        let largest_free_block = self.free_blocks.iter().map(|fb| fb.size).max().unwrap_or(0);

        let average_free_block_size = if free_block_count > 0 {
            total_free_space as f64 / free_block_count as f64
        } else {
            0.0
        };

        let fragmentation_ratio = if total_free_space > 0 {
            1.0 - (largest_free_block as f64 / total_free_space as f64)
        } else {
            0.0
        };

        // Calculate size distribution
        let mut size_distribution = HashMap::new();
        for fb in &self.free_blocks {
            let size_class = MemoryBlock::classify_size(fb.size);
            *size_distribution.entry(size_class).or_insert(0) += 1;
        }

        ExternalFragmentation {
            free_block_count,
            largest_free_block,
            total_free_space,
            average_free_block_size,
            free_block_size_distribution: size_distribution,
            fragmentation_ratio,
            compaction_potential: self.calculate_compaction_potential(),
        }
    }

    /// Analyze internal fragmentation
    fn analyze_internal_fragmentation(&self) -> InternalFragmentation {
        let mut total_internal_waste = 0;
        let mut waste_by_size_class = HashMap::new();
        let mut alignment_waste = 0;
        let mut padding_waste = 0;

        for block in self.memory_blocks.values() {
            if let BlockType::Allocated { size_class, .. } = &block.block_type {
                // Calculate potential internal waste (simplified)
                let ideal_size = self.calculate_ideal_size(block.size);
                let waste = block.size.saturating_sub(ideal_size);

                total_internal_waste += waste;
                *waste_by_size_class.entry(size_class.clone()).or_insert(0) += waste;

                // Estimate alignment and padding waste
                if block.alignment > 8 {
                    alignment_waste += block.alignment - 8;
                }
            }
        }

        let allocated_blocks = self.memory_blocks.len();
        let average_waste_per_allocation = if allocated_blocks > 0 {
            total_internal_waste as f64 / allocated_blocks as f64
        } else {
            0.0
        };

        let total_allocated_space: usize = self.memory_blocks.values().map(|b| b.size).sum();
        let efficiency_score = if total_allocated_space > 0 {
            1.0 - (total_internal_waste as f64 / total_allocated_space as f64)
        } else {
            1.0
        };

        InternalFragmentation {
            total_internal_waste,
            average_waste_per_allocation,
            worst_case_waste: waste_by_size_class.values().max().cloned().unwrap_or(0),
            waste_by_size_class,
            alignment_waste,
            padding_waste,
            efficiency_score,
        }
    }

    /// Identify fragmentation hotspots
    fn identify_fragmentation_hotspots(&self) -> Vec<FragmentationHotspot> {
        let mut hotspots = Vec::new();

        // Divide memory space into regions and analyze each
        let (min_addr, max_addr) = self.get_memory_bounds();
        if max_addr <= min_addr {
            return hotspots;
        }

        let region_size = (max_addr - min_addr) / 100; // 100 regions

        for i in 0..100 {
            let region_start = min_addr + i * region_size;
            let region_end = region_start + region_size;

            let blocks_in_region: Vec<_> = self.memory_blocks.values()
                .filter(|b| b.start_addr >= region_start && b.start_addr < region_end)
                .collect();

            let free_blocks_in_region: Vec<_> = self.free_blocks.iter()
                .filter(|fb| fb.start_addr >= region_start && fb.start_addr < region_end)
                .collect();

            if blocks_in_region.len() + free_blocks_in_region.len() < 2 {
                continue; // Skip sparse regions
            }

            let fragmentation_density = self.calculate_region_fragmentation_density(&blocks_in_region, &free_blocks_in_region);

            if fragmentation_density > 0.3 { // Threshold for hotspot
                let free_space_ratio = free_blocks_in_region.iter().map(|fb| fb.size).sum::<usize>() as f64 / region_size as f64;

                hotspots.push(FragmentationHotspot {
                    region_start,
                    region_end,
                    region_size,
                    fragmentation_density,
                    block_count: blocks_in_region.len() + free_blocks_in_region.len(),
                    free_space_ratio,
                    defragmentation_priority: if fragmentation_density > 0.7 {
                        Priority::Critical
                    } else if fragmentation_density > 0.5 {
                        Priority::High
                    } else {
                        Priority::Medium
                    },
                    estimated_benefit: fragmentation_density * free_space_ratio,
                    complexity_score: blocks_in_region.len() as f64 / 10.0,
                });
            }
        }

        // Sort by priority and estimated benefit
        hotspots.sort_by(|a, b| {
            a.defragmentation_priority.cmp(&b.defragmentation_priority).reverse()
                .then_with(|| b.estimated_benefit.partial_cmp(&a.estimated_benefit).unwrap_or(std::cmp::Ordering::Equal))
        });

        hotspots
    }

    /// Calculate efficiency metrics
    fn calculate_efficiency_metrics(&self) -> FragmentationEfficiency {
        let total_memory: usize = self.memory_blocks.values().map(|b| b.size).sum::<usize>() +
                                  self.free_blocks.iter().map(|fb| fb.size).sum::<usize>();

        let allocated_memory: usize = self.memory_blocks.values().map(|b| b.size).sum();

        let memory_utilization = if total_memory > 0 {
            allocated_memory as f64 / total_memory as f64
        } else {
            0.0
        };

        FragmentationEfficiency {
            memory_utilization,
            allocation_success_rate: 0.95, // Would track actual success rate
            average_search_time: Duration::from_micros(10), // Would measure actual search time
            defragmentation_overhead: 0.05, // Would calculate actual overhead
            compaction_frequency: 0.1, // Would track actual frequency
            waste_ratio: 1.0 - memory_utilization,
        }
    }

    /// Analyze fragmentation trends
    fn analyze_fragmentation_trend(&self) -> FragmentationTrend {
        if self.fragmentation_history.len() < 10 {
            return FragmentationTrend {
                direction: TrendDirection::Stable,
                rate_of_change: 0.0,
                prediction: FragmentationPrediction {
                    predicted_fragmentation: 0.0,
                    confidence: 0.0,
                    time_to_critical: None,
                    recommended_action_time: Duration::from_secs(3600),
                },
                contributing_factors: vec![],
                historical_pattern: vec![],
            };
        }

        let recent_data: Vec<f64> = self.fragmentation_history.iter()
            .rev()
            .take(50)
            .map(|s| s.fragmentation_index)
            .collect();

        // Simple linear regression for trend
        let n = recent_data.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = recent_data.iter().sum::<f64>() / n;

        let slope = recent_data.iter().enumerate()
            .map(|(i, &y)| (i as f64 - x_mean) * (y - y_mean))
            .sum::<f64>() / recent_data.iter().enumerate()
            .map(|(i, _)| (i as f64 - x_mean).powi(2))
            .sum::<f64>();

        let direction = if slope.abs() < 0.001 {
            TrendDirection::Stable
        } else if slope > 0.0 {
            TrendDirection::Degrading
        } else {
            TrendDirection::Improving
        };

        let predicted_fragmentation = y_mean + slope * n;
        let time_to_critical = if slope > 0.0 && predicted_fragmentation < 0.8 {
            Some(Duration::from_secs(((0.8 - y_mean) / slope * 60.0) as u64))
        } else {
            None
        };

        FragmentationTrend {
            direction,
            rate_of_change: slope,
            prediction: FragmentationPrediction {
                predicted_fragmentation,
                confidence: 0.7, // Would calculate proper confidence
                time_to_critical,
                recommended_action_time: Duration::from_secs(1800),
            },
            contributing_factors: vec!["Allocation pattern changes".to_string()],
            historical_pattern: recent_data,
        }
    }

    /// Assess impact of current fragmentation
    fn assess_fragmentation_impact(&self, external: &ExternalFragmentation, internal: &InternalFragmentation) -> FragmentationImpact {
        let performance_degradation = external.fragmentation_ratio * 0.3 + (1.0 - internal.efficiency_score) * 0.2;
        let memory_overhead = (internal.total_internal_waste as f64) / (1024.0 * 1024.0); // MB

        FragmentationImpact {
            performance_degradation,
            memory_overhead,
            allocation_failures: 0, // Would track actual failures
            cache_efficiency_impact: external.fragmentation_ratio * 0.1,
            bandwidth_loss: performance_degradation * 0.15,
            system_stability_risk: if external.fragmentation_ratio > 0.7 { 0.8 } else { 0.2 },
        }
    }

    /// Recommend mitigation strategies
    fn recommend_mitigation_strategies(
        &self,
        external: &ExternalFragmentation,
        internal: &InternalFragmentation,
        hotspots: &[FragmentationHotspot],
    ) -> Vec<FragmentationMitigationStrategy> {
        let mut strategies = Vec::new();

        // Strategy for high external fragmentation
        if external.fragmentation_ratio > 0.5 {
            strategies.push(FragmentationMitigationStrategy {
                strategy_type: MitigationType::Compaction,
                priority: Priority::High,
                description: "Perform memory compaction to consolidate free space".to_string(),
                expected_benefit: external.compaction_potential,
                implementation_cost: 0.3,
                time_to_implement: Duration::from_secs(30),
                side_effects: vec!["Temporary performance impact during compaction".to_string()],
                requirements: vec!["Pause allocations during compaction".to_string()],
            });
        }

        // Strategy for poor internal efficiency
        if internal.efficiency_score < 0.7 {
            strategies.push(FragmentationMitigationStrategy {
                strategy_type: MitigationType::MemoryPooling,
                priority: Priority::Medium,
                description: "Implement size-specific memory pools to reduce internal fragmentation".to_string(),
                expected_benefit: 1.0 - internal.efficiency_score,
                implementation_cost: 0.5,
                time_to_implement: Duration::from_secs(300),
                side_effects: vec!["Additional memory overhead for pool management".to_string()],
                requirements: vec!["Redesign allocation strategy".to_string()],
            });
        }

        // Strategy for hotspots
        if !hotspots.is_empty() {
            let critical_hotspots = hotspots.iter().filter(|h| h.defragmentation_priority == Priority::Critical).count();
            if critical_hotspots > 0 {
                strategies.push(FragmentationMitigationStrategy {
                    strategy_type: MitigationType::RegionBased,
                    priority: Priority::Critical,
                    description: format!("Target {} critical fragmentation hotspots", critical_hotspots),
                    expected_benefit: hotspots.iter().take(3).map(|h| h.estimated_benefit).sum::<f64>() / 3.0,
                    implementation_cost: 0.4,
                    time_to_implement: Duration::from_secs(60),
                    side_effects: vec!["May require moving allocations".to_string()],
                    requirements: vec!["Identify moveable allocations".to_string()],
                });
            }
        }

        strategies
    }

    // Helper methods
    fn get_memory_bounds(&self) -> (usize, usize) {
        let allocated_bounds = self.memory_blocks.values()
            .map(|b| (b.start_addr, b.end_addr))
            .fold((usize::MAX, 0), |(min_start, max_end), (start, end)| {
                (min_start.min(start), max_end.max(end))
            });

        let free_bounds = self.free_blocks.iter()
            .map(|fb| (fb.start_addr, fb.start_addr + fb.size))
            .fold((usize::MAX, 0), |(min_start, max_end), (start, end)| {
                (min_start.min(start), max_end.max(end))
            });

        (allocated_bounds.0.min(free_bounds.0), allocated_bounds.1.max(free_bounds.1))
    }

    fn calculate_compaction_potential(&self) -> f64 {
        // Simplified calculation - would be more sophisticated in practice
        let total_free: usize = self.free_blocks.iter().map(|fb| fb.size).sum();
        let largest_free = self.free_blocks.iter().map(|fb| fb.size).max().unwrap_or(0);

        if total_free > 0 {
            (total_free - largest_free) as f64 / total_free as f64
        } else {
            0.0
        }
    }

    fn calculate_ideal_size(&self, actual_size: usize) -> usize {
        // Simplified - would use actual allocation requirements
        let overhead = match MemoryBlock::classify_size(actual_size) {
            SizeClass::Tiny => 8,
            SizeClass::Small => 16,
            SizeClass::Medium => 32,
            SizeClass::Large => 64,
            SizeClass::Huge => 128,
            SizeClass::Massive => 256,
        };

        actual_size.saturating_sub(overhead)
    }

    fn calculate_region_fragmentation_density(&self, allocated: &[&MemoryBlock], free: &[&FreeBlockEntry]) -> f64 {
        if allocated.is_empty() && free.is_empty() {
            return 0.0;
        }

        let total_blocks = allocated.len() + free.len();
        let free_blocks = free.len();

        // Simple density calculation - could be more sophisticated
        free_blocks as f64 / total_blocks as f64
    }

    // Defragmentation implementations
    fn compact_memory(&mut self) -> Result<DefragmentationResult> {
        let start_blocks = self.memory_blocks.len();
        let start_free_space: usize = self.free_blocks.iter().map(|fb| fb.size).sum();

        // Simulate compaction (in practice, would move actual memory)
        self.coalesce_free_blocks()?;

        let end_free_space: usize = self.free_blocks.iter().map(|fb| fb.size).sum();
        let memory_recovered = end_free_space.saturating_sub(start_free_space);

        Ok(DefragmentationResult {
            algorithm_used: DefragmentationType::CompactionBased,
            memory_recovered,
            blocks_moved: start_blocks / 2, // Simplified
            execution_time: Duration::from_millis(100),
            fragmentation_improvement: 0.3,
            success: true,
        })
    }

    fn coalesce_free_blocks(&mut self) -> Result<DefragmentationResult> {
        let start_free_count = self.free_blocks.len();
        let mut memory_recovered = 0;

        // Group adjacent free blocks for coalescing
        let mut free_block_vec: Vec<_> = self.free_blocks.iter().cloned().collect();
        free_block_vec.sort_by_key(|fb| fb.start_addr);

        let mut coalesced_blocks = Vec::new();
        let mut current_block: Option<FreeBlockEntry> = None;

        for fb in free_block_vec {
            match current_block.as_mut() {
                None => current_block = Some(fb),
                Some(current) => {
                    if current.start_addr + current.size == fb.start_addr {
                        // Adjacent blocks - merge them
                        current.size += fb.size;
                        memory_recovered += fb.size; // The space was already free, but now it's usable as one block
                    } else {
                        coalesced_blocks.push(current.clone());
                        *current = fb;
                    }
                }
            }
        }

        if let Some(last_block) = current_block {
            coalesced_blocks.push(last_block);
        }

        // Replace free blocks with coalesced ones
        self.free_blocks.clear();
        self.free_blocks.extend(coalesced_blocks);

        let end_free_count = self.free_blocks.len();
        let blocks_merged = start_free_count.saturating_sub(end_free_count);

        Ok(DefragmentationResult {
            algorithm_used: DefragmentationType::CoalescingBased,
            memory_recovered,
            blocks_moved: 0,
            execution_time: Duration::from_millis(10),
            fragmentation_improvement: if start_free_count > 0 {
                blocks_merged as f64 / start_free_count as f64
            } else {
                0.0
            },
            success: true,
        })
    }

    fn apply_buddy_system(&mut self) -> Result<DefragmentationResult> {
        // Simplified buddy system implementation
        Ok(DefragmentationResult {
            algorithm_used: DefragmentationType::BuddySystem,
            memory_recovered: 0,
            blocks_moved: 0,
            execution_time: Duration::from_millis(50),
            fragmentation_improvement: 0.1,
            success: true,
        })
    }
}

/// Result of defragmentation operation
#[derive(Debug, Clone)]
pub struct DefragmentationResult {
    pub algorithm_used: DefragmentationType,
    pub memory_recovered: usize,
    pub blocks_moved: usize,
    pub execution_time: Duration,
    pub fragmentation_improvement: f64,
    pub success: bool,
}

impl Default for FragmentationManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_block_creation() {
        let block = MemoryBlock::new_allocated(1000, 64, 1, AllocationPurpose::TensorData);
        assert_eq!(block.start_addr, 1000);
        assert_eq!(block.size, 64);
        assert_eq!(block.end_addr, 1064);

        match block.block_type {
            BlockType::Allocated { size_class, .. } => {
                assert_eq!(size_class, SizeClass::Small);
            }
            _ => panic!("Expected allocated block"),
        }
    }

    #[test]
    fn test_block_merging() {
        let block1 = MemoryBlock::new_free(1000, 64);
        let block2 = MemoryBlock::new_free(1064, 32);

        assert!(block1.can_merge_with(&block2));

        let merged = block1.merge_with(&block2).unwrap();
        assert_eq!(merged.start_addr, 1000);
        assert_eq!(merged.size, 96);
        assert_eq!(merged.end_addr, 1096);
    }

    #[test]
    fn test_fragmentation_manager() {
        let mut manager = FragmentationManager::new();

        // Track some allocations
        let block1 = MemoryBlock::new_allocated(1000, 64, 1, AllocationPurpose::TensorData);
        let block2 = MemoryBlock::new_allocated(2000, 128, 2, AllocationPurpose::GradientBuffer);

        manager.track_allocation(block1).unwrap();
        manager.track_allocation(block2).unwrap();

        assert_eq!(manager.memory_blocks.len(), 2);

        // Deallocate one block
        manager.track_deallocation(1000).unwrap();
        assert_eq!(manager.memory_blocks.len(), 1);
        assert_eq!(manager.free_blocks.len(), 1);
    }

    #[test]
    fn test_fragmentation_analysis() {
        let mut manager = FragmentationManager::new();

        // Create fragmented memory pattern
        for i in 0..10 {
            let block = MemoryBlock::new_allocated(i * 200, 64, i, AllocationPurpose::TensorData);
            manager.track_allocation(block).unwrap();
        }

        // Deallocate every other block to create fragmentation
        for i in (0..10).step_by(2) {
            manager.track_deallocation(i * 200).unwrap();
        }

        let analysis = manager.analyze_fragmentation().unwrap();
        assert!(analysis.overall_fragmentation_index > 0.0);
        assert!(analysis.external_fragmentation.free_block_count > 0);
    }

    #[test]
    fn test_coalescing_defragmentation() {
        let mut manager = FragmentationManager::new();

        // Create adjacent allocations
        for i in 0..5 {
            let block = MemoryBlock::new_allocated(i * 100, 50, i, AllocationPurpose::TensorData);
            manager.track_allocation(block).unwrap();
        }

        // Deallocate to create adjacent free blocks
        manager.track_deallocation(0).unwrap(); // 0-50
        manager.track_deallocation(100).unwrap(); // 100-150

        assert_eq!(manager.free_blocks.len(), 2);

        let result = manager.defragment(DefragmentationType::CoalescingBased).unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_hotspot_identification() {
        let mut manager = FragmentationManager::new();

        // Create concentrated fragmentation in specific region
        let base_addr = 10000;
        for i in 0..20 {
            let block = MemoryBlock::new_allocated(base_addr + i * 100, 50, i, AllocationPurpose::TensorData);
            manager.track_allocation(block).unwrap();
        }

        // Deallocate many blocks in the region
        for i in (0..20).step_by(2) {
            manager.track_deallocation(base_addr + i * 100).unwrap();
        }

        let hotspots = manager.identify_fragmentation_hotspots();
        assert!(!hotspots.is_empty());
    }

    #[test]
    fn test_fragmentation_trend() {
        let mut manager = FragmentationManager::new();

        // Create increasing fragmentation over time
        for round in 0..20 {
            for i in 0..10 {
                let block = MemoryBlock::new_allocated(round * 1000 + i * 50, 25, round * 10 + i, AllocationPurpose::TensorData);
                manager.track_allocation(block).unwrap();
            }

            // Deallocate some to create fragmentation
            for i in 0..5 {
                manager.track_deallocation(round * 1000 + i * 50).unwrap();
            }
        }

        let trend = manager.analyze_fragmentation_trend();
        // Should detect degrading trend due to increasing fragmentation
        assert_ne!(trend.direction, TrendDirection::Improving);
    }

    #[test]
    fn test_size_classification() {
        assert_eq!(MemoryBlock::classify_size(32), SizeClass::Tiny);
        assert_eq!(MemoryBlock::classify_size(512), SizeClass::Small);
        assert_eq!(MemoryBlock::classify_size(32768), SizeClass::Medium);
        assert_eq!(MemoryBlock::classify_size(524288), SizeClass::Large);
        assert_eq!(MemoryBlock::classify_size(8388608), SizeClass::Huge);
        assert_eq!(MemoryBlock::classify_size(67108864), SizeClass::Massive);
    }
}