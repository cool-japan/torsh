//! dhat-based allocation tracking benchmark for torsh-tensor.
//!
//! Proves that `GlobalMemoryPool::acquire_uninit<T>` genuinely reduces the
//! number of heap blocks compared with the naive "allocate a new Vec every
//! iteration" pattern.
//!
//! Run with:
//!   cargo bench --bench alloc_tracking -p torsh-tensor --all-features
//!
//! The benchmark writes a JSON summary to /tmp/alloc_tracking_result.json
//! for CI pickup and asserts that the pooled path uses ≥ 50 % fewer blocks
//! than the naive path.

use dhat;
use std::fs::File;
use std::io::Write as IoWrite;
use torsh_tensor::memory_pool::{global_acquire_uninit, init_memory_pool};

/// Size of each f32 buffer exercised in the hot loop.
const BUF_SIZE: usize = 4096;
/// Number of iterations per measurement phase.
const ITERATIONS: usize = 10_000;

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

/// Inner logic; returns `(naive_delta, pooled_delta)`.
fn run() -> Result<(u64, u64), Box<dyn std::error::Error>> {
    // ── Phase 1: naive Vec allocation ───────────────────────────────────────
    let stats_naive_before = dhat::HeapStats::get();

    for i in 0..ITERATIONS {
        // Mimic the old copy-on-hit pattern: fresh allocation + initialise +
        // let the Vec drop at end of iteration.
        let mut buf: Vec<f32> = Vec::with_capacity(BUF_SIZE);
        // Initialise so the compiler cannot optimise the allocation away.
        for j in 0..BUF_SIZE {
            buf.push((i * BUF_SIZE + j) as f32 * 0.001_f32);
        }
        // Simulate a small consumer that reads the last element so the whole
        // buffer is kept live.
        let _ = std::hint::black_box(buf.last().copied());
        // buf drops here → heap block freed
    }

    let stats_naive_after = dhat::HeapStats::get();
    let naive_delta = stats_naive_after.total_blocks - stats_naive_before.total_blocks;

    // ── Phase 2: pooled allocation via GlobalMemoryPool ─────────────────────
    // Warm the pool once before measuring to exclude the very first miss.
    {
        let mut warm = global_acquire_uninit::<f32>(BUF_SIZE);
        let slice = warm.as_uninit_slice_mut();
        for v in slice.iter_mut() {
            v.write(0.0_f32);
        }
        warm.release_to_pool();
    }

    let stats_pooled_before = dhat::HeapStats::get();

    for i in 0..ITERATIONS {
        let mut buf = global_acquire_uninit::<f32>(BUF_SIZE);

        // Initialise every element (satisfies safety contract of acquire_uninit).
        let uninit_slice = buf.as_uninit_slice_mut();
        for (j, slot) in uninit_slice.iter_mut().enumerate().take(BUF_SIZE) {
            slot.write((i * BUF_SIZE + j) as f32 * 0.001_f32);
        }

        // Consume the last element to keep the buffer live.
        let _ = std::hint::black_box(
            // SAFETY: we just initialised all BUF_SIZE elements above.
            unsafe { buf.as_uninit_slice_mut()[BUF_SIZE - 1].assume_init() },
        );

        // Return buffer to pool instead of freeing.
        buf.release_to_pool();
    }

    let stats_pooled_after = dhat::HeapStats::get();
    let pooled_delta = stats_pooled_after.total_blocks - stats_pooled_before.total_blocks;

    Ok((naive_delta, pooled_delta))
}

fn main() {
    // Initialise the global memory pool before the dhat profiler so that pool
    // infrastructure allocations do not pollute the measurement phases.
    init_memory_pool();

    let _profiler = dhat::Profiler::new_heap();

    let (naive_delta, pooled_delta) =
        run().expect("benchmark run should complete without I/O errors");

    println!("alloc blocks delta (naive):  {naive_delta}");
    println!("alloc blocks delta (pooled): {pooled_delta}");

    // ── Write JSON for CI pickup ─────────────────────────────────────────────
    let json = format!(
        "{{\n  \"naive_blocks\": {naive_delta},\n  \"pooled_blocks\": {pooled_delta},\n  \"iterations\": {ITERATIONS},\n  \"buf_size\": {BUF_SIZE}\n}}\n"
    );

    let json_path = std::env::temp_dir().join("alloc_tracking_result.json");
    let mut file = File::create(&json_path)
        .expect("should be able to create alloc_tracking_result.json in temp dir");
    file.write_all(json.as_bytes())
        .expect("should be able to write JSON result");
    println!("result written to {}", json_path.display());

    // ── Assertions ───────────────────────────────────────────────────────────
    // The pool should stay well under 1000 total new blocks for 10 K iterations
    // (ideally just 1: the single warm-up miss before the loop).
    assert!(
        pooled_delta < 1000,
        "Pool not reducing allocations: got {pooled_delta} pooled blocks (expected < 1000)"
    );

    // The pooled path must use ≥ 50 % fewer blocks than the naive path.
    let threshold = naive_delta / 2;
    assert!(
        pooled_delta <= threshold,
        "Pool did not achieve ≥ 50 % allocation reduction: \
         naive={naive_delta} pooled={pooled_delta} (need pooled ≤ {threshold})"
    );
}
