# ToRSh Profiler Troubleshooting Guide

This guide helps you diagnose and fix common issues when using the ToRSh Profiler.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Compilation Errors](#compilation-errors)
3. [Runtime Issues](#runtime-issues)
4. [Performance Issues](#performance-issues)
5. [Memory Issues](#memory-issues)
6. [Dashboard Issues](#dashboard-issues)
7. [Export/Report Issues](#exportreport-issues)
8. [Platform-Specific Issues](#platform-specific-issues)

## Installation Issues

### Issue: Cargo build fails with missing dependencies

**Symptoms:**
```
error: failed to resolve dependencies
```

**Solution:**
Ensure you have the correct version in your `Cargo.toml`:
```toml
[dependencies]
torsh-profiler = "0.1.0-alpha.1"
torsh-core = "0.1.0-alpha.1"
```

Update your dependencies:
```bash
cargo update
```

### Issue: Feature compilation errors

**Symptoms:**
```
error: feature `xyz` is not available
```

**Solution:**
Check available features in `Cargo.toml`:
```toml
[dependencies]
torsh-profiler = { version = "0.1.0-alpha.1", features = ["full"] }
```

Available features:
- `full` - All features (default)
- `dashboard` - Web dashboard support
- `ml-analysis` - Machine learning analytics
- `gpu-profiling` - GPU profiling support

## Compilation Errors

### Issue: ProfileScope not found

**Symptoms:**
```rust
error[E0433]: failed to resolve: use of undeclared type `ProfileScope`
```

**Solution:**
Add the correct import:
```rust
use torsh_profiler::ProfileScope;
// or
use torsh_profiler::*;
```

### Issue: Borrow checker errors with global profiler

**Symptoms:**
```rust
error[E0502]: cannot borrow as mutable because it is also borrowed as immutable
```

**Solution:**
Use proper scoping:
```rust
// ❌ Bad: Holding lock too long
let profiler = global_profiler();
let lock = profiler.lock().unwrap();
// ... long operations ...

// ✅ Good: Limit lock scope
{
    let profiler = global_profiler();
    let events = profiler.lock().unwrap().get_events();
    // Process events quickly
}
```

### Issue: Feature gate errors

**Symptoms:**
```rust
error[E0554]: `#![feature(...)]` may not be used on the stable release channel
```

**Solution:**
Remove unstable features or use nightly Rust:
```rust
// Remove this line:
// #![feature(unstable_feature)]

// Or use nightly:
rustup default nightly
```

## Runtime Issues

### Issue: Profiler not recording events

**Symptoms:**
- No profiling data in exports
- Empty dashboard
- Zero events in reports

**Solution:**

1. Ensure profiling is enabled:
```rust
start_profiling();

// Or manually enable
let mut profiler = global_profiler();
profiler.lock().unwrap().enable();
```

2. Check ProfileScope usage:
```rust
// ❌ Wrong: Scope dropped immediately
ProfileScope::simple("operation".to_string(), "category".to_string());

// ✅ Correct: Store scope in variable
let _scope = ProfileScope::simple("operation".to_string(), "category".to_string());
```

3. Verify event recording:
```rust
// Check if events are being recorded
let profiler = global_profiler();
let event_count = profiler.lock().unwrap().get_events().len();
println!("Recorded events: {}", event_count);
```

### Issue: High profiling overhead

**Symptoms:**
- Significant performance degradation
- Application becomes slow

**Solution:**

1. Monitor overhead:
```rust
let overhead_stats = get_optimization_stats().unwrap();
if overhead_stats.overhead_percentage > 5.0 {
    println!("High overhead: {:.2}%", overhead_stats.overhead_percentage);
}
```

2. Use sampling:
```rust
static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn high_frequency_function() {
    let count = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    
    // Only profile every 100th call
    if count % 100 == 0 {
        let _scope = ProfileScope::simple("sampled_op".to_string(), "compute".to_string());
        // Your operation here
    } else {
        // Your operation here without profiling
    }
}
```

3. Use optimized profiling:
```rust
init_optimized_profiling().unwrap();

let config = SamplingConfig {
    sampling_rate: 0.1, // 10% sampling
    adaptive_sampling: true,
    min_duration_threshold: Duration::from_micros(100),
};

let _optimized_profiler = create_optimized_profiler_with_config(config);
```

### Issue: Thread safety errors

**Symptoms:**
```
thread panicked at 'called `Result::unwrap()` on an `Err` value: "mutex poisoned"'
```

**Solution:**

1. Handle lock poisoning:
```rust
match global_profiler().lock() {
    Ok(profiler) => {
        // Use profiler
    }
    Err(poisoned) => {
        // Recover from poisoned mutex
        let profiler = poisoned.into_inner();
        // Use recovered profiler
    }
}
```

2. Use timeout locks:
```rust
use std::time::Duration;

if let Ok(profiler) = global_profiler().try_lock_for(Duration::from_millis(100)) {
    // Use profiler
} else {
    // Handle timeout
    eprintln!("Failed to acquire profiler lock");
}
```

## Performance Issues

### Issue: Memory growth over time

**Symptoms:**
- Increasing memory usage
- Eventual out-of-memory errors

**Solution:**

1. Limit event storage:
```rust
// Clear old events periodically
let profiler = global_profiler();
let mut prof = profiler.lock().unwrap();
if prof.get_events().len() > 10000 {
    prof.clear_events();
}
```

2. Use circular buffers:
```rust
let config = OptimizationConfig {
    max_events: 5000,
    use_circular_buffer: true,
    auto_cleanup: true,
};

init_optimized_profiling_with_config(config).unwrap();
```

3. Disable features you don't need:
```rust
let mut memory_profiler = MemoryProfiler::new();
memory_profiler.set_timeline_enabled(false); // Disable if not needed
memory_profiler.set_leak_detection_enabled(false); // Disable if not needed
```

### Issue: Slow dashboard loading

**Symptoms:**
- Dashboard takes long time to load
- Browser becomes unresponsive

**Solution:**

1. Limit data points:
```rust
let config = DashboardConfig {
    max_data_points: 500, // Reduce from default 1000
    refresh_interval: 30, // Increase refresh interval
    real_time_updates: false, // Disable real-time updates
    ..Default::default()
};
```

2. Disable expensive features:
```rust
let config = DashboardConfig {
    enable_stack_traces: false, // Expensive feature
    ..Default::default()
};
```

## Memory Issues

### Issue: Memory leak detection false positives

**Symptoms:**
- Leak detection reports leaks that aren't real
- High number of reported leaks

**Solution:**

1. Adjust leak detection sensitivity:
```rust
let mut memory_profiler = MemoryProfiler::new();
memory_profiler.set_leak_detection_enabled(true);

// Set minimum size threshold
memory_profiler.set_leak_threshold(1024 * 1024); // Only report leaks > 1MB

// Set time threshold
memory_profiler.set_leak_time_threshold(Duration::from_secs(300)); // 5 minutes
```

2. Filter system allocations:
```rust
// Implement custom filtering
fn is_system_allocation(ptr: usize, stack_trace: &Option<String>) -> bool {
    if let Some(trace) = stack_trace {
        // Filter known system allocation patterns
        trace.contains("std::") || trace.contains("libc::")
    } else {
        false
    }
}
```

### Issue: Stack trace capture overhead

**Symptoms:**
- High CPU usage
- Slow profiling performance

**Solution:**

1. Disable stack traces:
```rust
let config = DashboardConfig {
    enable_stack_traces: false,
    ..Default::default()
};
```

2. Use sampling for stack traces:
```rust
let mut memory_profiler = MemoryProfiler::new();
memory_profiler.set_stack_trace_sampling_rate(0.1); // 10% sampling
```

## Dashboard Issues

### Issue: Dashboard not accessible

**Symptoms:**
- Cannot connect to dashboard URL
- Connection refused errors

**Solution:**

1. Check if dashboard is running:
```rust
let dashboard = create_dashboard();
match dashboard.start(profiler, memory_profiler) {
    Ok(_) => println!("Dashboard started successfully"),
    Err(e) => eprintln!("Failed to start dashboard: {}", e),
}
```

2. Check port availability:
```rust
let config = DashboardConfig {
    port: 8081, // Try different port
    ..Default::default()
};
```

3. Check firewall/network settings:
```bash
# Check if port is open
netstat -ln | grep 8080

# Test local connection
curl http://localhost:8080
```

### Issue: Dashboard shows no data

**Symptoms:**
- Dashboard loads but shows zero metrics
- Empty charts and tables

**Solution:**

1. Verify profiler is recording data:
```rust
let profiler = global_profiler();
let events = profiler.lock().unwrap().get_events();
println!("Events recorded: {}", events.len());

if events.is_empty() {
    println!("No events recorded - check profiling setup");
}
```

2. Check dashboard data collection:
```rust
let dashboard = create_dashboard();
if let Some(data) = dashboard.get_current_data().unwrap() {
    println!("Dashboard has data: {} operations", 
             data.performance_metrics.total_operations);
} else {
    println!("Dashboard has no data");
}
```

### Issue: Dashboard alerts not working

**Symptoms:**
- Alerts configured but not triggering
- No alert notifications

**Solution:**

1. Verify alert configuration:
```rust
let alert_config = AlertConfig {
    duration_threshold: Duration::from_millis(100),
    memory_threshold: 1024 * 1024 * 1024, // 1GB
    enable_anomaly_detection: true,
    ..Default::default()
};

// Test alert manually
let mut alert_manager = create_alert_manager_with_config(alert_config);
alert_manager.check_duration_threshold("test_op", Duration::from_millis(150)).unwrap();
```

2. Check notification channels:
```rust
let alert_config = AlertConfig {
    notification_channels: vec![
        NotificationChannel::Console, // Simple console output for testing
    ],
    ..Default::default()
};
```

## Export/Report Issues

### Issue: Export files are empty or corrupted

**Symptoms:**
- Zero-byte export files
- Cannot open exported files
- Malformed JSON/CSV data

**Solution:**

1. Check if profiler has data:
```rust
let profiler = global_profiler();
let events = profiler.lock().unwrap().get_events();

if events.is_empty() {
    println!("No data to export");
    return;
}
```

2. Verify export permissions:
```rust
use std::fs::OpenOptions;

// Test write permissions
match OpenOptions::new().write(true).create(true).open("/tmp/test_write") {
    Ok(_) => println!("Write permissions OK"),
    Err(e) => eprintln!("Write permission error: {}", e),
}
```

3. Handle export errors:
```rust
match export(&profiler.lock().unwrap(), "/tmp/profile_data.json") {
    Ok(_) => println!("Export successful"),
    Err(e) => {
        eprintln!("Export failed: {}", e);
        // Try alternative location
        export(&profiler.lock().unwrap(), "./profile_data.json").unwrap();
    }
}
```

### Issue: Large export files

**Symptoms:**
- Export takes very long time
- Files are too large to open
- Out of disk space errors

**Solution:**

1. Use compression:
```rust
let export_config = CustomExportFormat {
    compression: Some(CompressionType::Gzip),
    ..Default::default()
};

let exporter = CustomExporter::new(export_config);
exporter.export(&profiler.lock().unwrap(), "/tmp/compressed_data.csv.gz").unwrap();
```

2. Export in batches:
```rust
let events = profiler.lock().unwrap().get_events();
let batch_size = 1000;

for (i, batch) in events.chunks(batch_size).enumerate() {
    let filename = format!("/tmp/profile_batch_{}.json", i);
    export_batch(batch, &filename).unwrap();
}
```

3. Filter data before export:
```rust
// Only export events longer than threshold
let filtered_events: Vec<_> = events.iter()
    .filter(|e| e.duration_us > 1000) // Only events > 1ms
    .collect();

export_filtered(&filtered_events, "/tmp/filtered_data.json").unwrap();
```

## Platform-Specific Issues

### Linux Issues

**Issue: Permission denied errors**
```bash
# Fix file permissions
chmod +x your_binary
sudo chown $USER:$USER /tmp/profile_data.json
```

**Issue: Missing system libraries**
```bash
# Install required packages
sudo apt-get install build-essential pkg-config
```

### macOS Issues

**Issue: Code signing issues**
```bash
# Disable Gatekeeper temporarily for testing
sudo spctl --master-disable
```

**Issue: Missing Xcode command line tools**
```bash
xcode-select --install
```

### Windows Issues

**Issue: Path length limitations**
Use shorter paths or enable long path support:
```powershell
# Enable long paths (requires admin)
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

**Issue: DLL loading errors**
Ensure all required DLLs are in PATH or next to executable.

## Debug Mode

Enable debug logging to diagnose issues:

```rust
// Set environment variable
std::env::set_var("TORSH_PROFILER_LOG", "debug");

// Or use tracing
use tracing::{debug, info, warn, error};

debug!("Starting profiling for operation: {}", operation_name);
```

Enable debug output in Cargo.toml:
```toml
[dependencies]
torsh-profiler = { version = "0.1.0-alpha.1", features = ["debug"] }
```

## Getting Help

If you're still experiencing issues:

1. **Check the logs**: Enable debug logging and check for error messages
2. **Minimal reproduction**: Create a minimal example that reproduces the issue
3. **System information**: Include OS, Rust version, and dependency versions
4. **File an issue**: Report the issue on the GitHub repository with:
   - Description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - System information
   - Relevant logs or error messages

## Common Error Codes

| Error Code | Description | Solution |
|------------|-------------|----------|
| `E001` | Profiler not initialized | Call `start_profiling()` |
| `E002` | Lock poisoned | Handle mutex poisoning gracefully |
| `E003` | Invalid configuration | Check configuration parameters |
| `E004` | Export failed | Check file permissions and disk space |
| `E005` | Dashboard port in use | Change port or stop conflicting service |
| `E006` | Memory allocation failed | Reduce memory usage or increase limits |
| `E007` | Invalid format | Check export format specification |
| `E008` | Timeout | Increase timeout or reduce data size |
| `E009` | Network error | Check network connectivity |
| `E010` | Feature not enabled | Enable required feature in Cargo.toml |

Remember: Most issues can be resolved by checking the logs, verifying configuration, and ensuring proper resource management.