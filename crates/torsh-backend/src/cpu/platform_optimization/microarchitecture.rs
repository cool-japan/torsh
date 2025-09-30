//! CPU microarchitecture definitions for x86_64 and ARM64
//!
//! This module defines the various CPU microarchitectures we can detect and optimize for.

/// CPU microarchitecture types for x86_64
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum X86Microarchitecture {
    /// Intel Core 2 / Penryn (SSE4.1)
    Core2,
    /// Intel Nehalem (SSE4.2)
    Nehalem,
    /// Intel Sandy Bridge (AVX)
    SandyBridge,
    /// Intel Ivy Bridge (enhanced AVX)
    IvyBridge,
    /// Intel Haswell (AVX2, FMA)
    Haswell,
    /// Intel Broadwell (enhanced AVX2)
    Broadwell,
    /// Intel Skylake (AVX-512 foundation)
    Skylake,
    /// Intel Kaby Lake
    KabyLake,
    /// Intel Coffee Lake
    CoffeeLake,
    /// Intel Ice Lake (enhanced AVX-512)
    IceLake,
    /// Intel Tiger Lake
    TigerLake,
    /// Intel Alder Lake (hybrid architecture)
    AlderLake,
    /// Intel Raptor Lake
    RaptorLake,
    /// Intel Meteor Lake
    MeteorLake,
    /// AMD K8
    K8,
    /// AMD K10
    K10,
    /// AMD Bulldozer
    Bulldozer,
    /// AMD Piledriver
    Piledriver,
    /// AMD Steamroller
    Steamroller,
    /// AMD Excavator
    Excavator,
    /// AMD Zen
    Zen,
    /// AMD Zen+
    ZenPlus,
    /// AMD Zen 2
    Zen2,
    /// AMD Zen 3
    Zen3,
    /// AMD Zen 4
    Zen4,
    /// Unknown/Generic x86_64
    Unknown,
}

/// ARM microarchitecture types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArmMicroarchitecture {
    /// Apple A7 (Cyclone)
    Cyclone,
    /// Apple A8 (Typhoon)
    Typhoon,
    /// Apple A9 (Twister)
    Twister,
    /// Apple A10 (Hurricane)
    Hurricane,
    /// Apple A11 (Monsoon/Mistral)
    Bionic,
    /// Apple A12 (Vortex/Tempest)
    A12,
    /// Apple A13 (Lightning/Thunder)
    A13,
    /// Apple A14 (Firestorm/Icestorm)
    A14,
    /// Apple A15 (Avalanche/Blizzard)
    A15,
    /// Apple A16 (Everest/Sawtooth)
    A16,
    /// Apple M1 (Firestorm/Icestorm)
    M1,
    /// Apple M2 (Avalanche/Blizzard)
    M2,
    /// Apple M3 (Enhanced Avalanche/Blizzard)
    M3,
    /// ARM Cortex-A53
    CortexA53,
    /// ARM Cortex-A55
    CortexA55,
    /// ARM Cortex-A57
    CortexA57,
    /// ARM Cortex-A72
    CortexA72,
    /// ARM Cortex-A73
    CortexA73,
    /// ARM Cortex-A75
    CortexA75,
    /// ARM Cortex-A76
    CortexA76,
    /// ARM Cortex-A77
    CortexA77,
    /// ARM Cortex-A78
    CortexA78,
    /// ARM Cortex-X1
    CortexX1,
    /// ARM Cortex-A510
    CortexA510,
    /// ARM Cortex-A710
    CortexA710,
    /// ARM Cortex-X2
    CortexX2,
    /// ARM Cortex-A715
    CortexA715,
    /// ARM Cortex-X3
    CortexX3,
    /// ARM Neoverse V1
    NeoverseV1,
    /// ARM Neoverse N1
    NeoverseN1,
    /// ARM Neoverse N2
    NeoverseN2,
    /// Unknown/Generic ARM64
    Unknown,
}
