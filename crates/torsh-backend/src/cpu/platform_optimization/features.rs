//! CPU feature flags for runtime detection
//!
//! This module defines CPU feature flags that can be detected at runtime
//! for both x86_64 and ARM64 architectures.

use crate::error::BackendResult;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// CPU feature flags detected at runtime
#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    // x86_64 features
    pub sse: bool,
    pub sse2: bool,
    pub sse3: bool,
    pub ssse3: bool,
    pub sse4_1: bool,
    pub sse4_2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub avx512dq: bool,
    pub avx512cd: bool,
    pub avx512bw: bool,
    pub avx512vl: bool,
    pub avx512vnni: bool,
    pub avx512bf16: bool,
    pub avx512vp2intersect: bool,
    pub fma: bool,
    pub fma4: bool,
    pub bmi1: bool,
    pub bmi2: bool,
    pub lzcnt: bool,
    pub popcnt: bool,
    pub f16c: bool,
    pub rdrand: bool,
    pub rdseed: bool,
    pub aes: bool,
    pub pclmul: bool,
    pub sha: bool,
    pub adx: bool,
    pub prefetchw: bool,
    pub clflushopt: bool,
    pub clwb: bool,
    pub movbe: bool,
    pub rtm: bool,
    pub hle: bool,
    pub mpx: bool,
    pub xsave: bool,
    pub xsaveopt: bool,
    pub xgetbv: bool,
    pub invariant_tsc: bool,
    pub rdtscp: bool,

    // ARM64 features
    pub neon: bool,
    pub fp: bool,
    pub asimd: bool,
    pub aes_arm: bool,
    pub pmull: bool,
    pub sha1: bool,
    pub sha256: bool,
    pub crc32: bool,
    pub atomics: bool,
    pub fphp: bool,
    pub asimdhp: bool,
    pub cpuid: bool,
    pub asimdrdm: bool,
    pub jscvt: bool,
    pub fcma: bool,
    pub lrcpc: bool,
    pub dcpop: bool,
    pub sha3: bool,
    pub sm3: bool,
    pub sm4: bool,
    pub asimddp: bool,
    pub sha512: bool,
    pub sve: bool,
    pub sve2: bool,
    pub sveaes: bool,
    pub svepmull: bool,
    pub svebitperm: bool,
    pub svesha3: bool,
    pub svesm4: bool,
    pub flagm: bool,
    pub ssbs: bool,
    pub sb: bool,
    pub paca: bool,
    pub pacg: bool,
    pub dgh: bool,
    pub bf16: bool,
    pub i8mm: bool,
    pub rng: bool,
    pub bti: bool,
    pub mte: bool,
    pub ecv: bool,
    pub afp: bool,
    pub rpres: bool,
}

impl Default for CpuFeatures {
    fn default() -> Self {
        Self {
            // Initialize all features as false - will be detected at runtime
            sse: false,
            sse2: false,
            sse3: false,
            ssse3: false,
            sse4_1: false,
            sse4_2: false,
            avx: false,
            avx2: false,
            avx512f: false,
            avx512dq: false,
            avx512cd: false,
            avx512bw: false,
            avx512vl: false,
            avx512vnni: false,
            avx512bf16: false,
            avx512vp2intersect: false,
            fma: false,
            fma4: false,
            bmi1: false,
            bmi2: false,
            lzcnt: false,
            popcnt: false,
            f16c: false,
            rdrand: false,
            rdseed: false,
            aes: false,
            pclmul: false,
            sha: false,
            adx: false,
            prefetchw: false,
            clflushopt: false,
            clwb: false,
            movbe: false,
            rtm: false,
            hle: false,
            mpx: false,
            xsave: false,
            xsaveopt: false,
            xgetbv: false,
            invariant_tsc: false,
            rdtscp: false,
            neon: false,
            fp: false,
            asimd: false,
            aes_arm: false,
            pmull: false,
            sha1: false,
            sha256: false,
            crc32: false,
            atomics: false,
            fphp: false,
            asimdhp: false,
            cpuid: false,
            asimdrdm: false,
            jscvt: false,
            fcma: false,
            lrcpc: false,
            dcpop: false,
            sha3: false,
            sm3: false,
            sm4: false,
            asimddp: false,
            sha512: false,
            sve: false,
            sve2: false,
            sveaes: false,
            svepmull: false,
            svebitperm: false,
            svesha3: false,
            svesm4: false,
            flagm: false,
            ssbs: false,
            sb: false,
            paca: false,
            pacg: false,
            dgh: false,
            bf16: false,
            i8mm: false,
            rng: false,
            bti: false,
            mte: false,
            ecv: false,
            afp: false,
            rpres: false,
        }
    }
}

// CPUID helper function for x86_64
#[cfg(target_arch = "x86_64")]
fn has_cpuid() -> bool {
    true // CPUID is always available on x86_64
}

#[cfg(not(target_arch = "x86_64"))]
fn has_cpuid() -> bool {
    false
}

/// Detect CPU features for the current platform
pub fn detect_cpu_features() -> BackendResult<CpuFeatures> {
    #[cfg(target_arch = "x86_64")]
    {
        if !has_cpuid() {
            return Ok(CpuFeatures::default());
        }

        unsafe {
            let cpuid = __cpuid(1);
            let sse = (cpuid.edx & (1 << 25)) != 0;
            let sse2 = (cpuid.edx & (1 << 26)) != 0;
            let sse3 = (cpuid.ecx & (1 << 0)) != 0;
            let ssse3 = (cpuid.ecx & (1 << 9)) != 0;
            let sse4_1 = (cpuid.ecx & (1 << 19)) != 0;
            let sse4_2 = (cpuid.ecx & (1 << 20)) != 0;

            let extended_features = __cpuid(7);
            let avx = (cpuid.ecx & (1 << 28)) != 0;
            let avx2 = (extended_features.ebx & (1 << 5)) != 0;
            let avx512f = (extended_features.ebx & (1 << 16)) != 0;
            let avx512cd = (extended_features.ebx & (1 << 28)) != 0;
            let fma = (cpuid.ecx & (1 << 12)) != 0;

            Ok(CpuFeatures {
                sse,
                sse2,
                sse3,
                ssse3,
                sse4_1,
                sse4_2,
                avx,
                avx2,
                avx512f,
                avx512cd,
                fma,
                neon: false,
                ..Default::default()
            })
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        Ok(CpuFeatures {
            neon: true, // NEON is standard on AArch64
            ..Default::default()
        })
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        Ok(CpuFeatures::default())
    }
}
