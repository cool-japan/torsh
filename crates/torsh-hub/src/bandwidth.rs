//! Bandwidth Management and Rate Limiting
//!
//! This module provides bandwidth throttling capabilities for downloads,
//! uploads, and other network operations to ensure fair resource usage
//! and prevent network congestion.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use torsh_core::error::Result;

/// Bandwidth limiter using token bucket algorithm
#[derive(Debug, Clone)]
pub struct BandwidthLimiter {
    /// Maximum bytes per second
    max_bytes_per_sec: u64,
    /// Current number of tokens (bytes available)
    tokens: Arc<Mutex<f64>>,
    /// Maximum number of tokens in bucket
    bucket_capacity: f64,
    /// Last time tokens were added
    last_refill: Arc<Mutex<Instant>>,
}

impl BandwidthLimiter {
    /// Create a new bandwidth limiter
    ///
    /// # Arguments
    /// * `bytes_per_sec` - Maximum bytes per second allowed
    ///
    /// # Example
    /// ```
    /// use torsh_hub::bandwidth::BandwidthLimiter;
    ///
    /// // Limit to 1 MB/s
    /// let limiter = BandwidthLimiter::new(1024 * 1024);
    /// ```
    pub fn new(bytes_per_sec: u64) -> Self {
        let capacity = bytes_per_sec as f64 * 2.0; // 2 second burst capacity

        Self {
            max_bytes_per_sec: bytes_per_sec,
            tokens: Arc::new(Mutex::new(capacity)),
            bucket_capacity: capacity,
            last_refill: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Create unlimited bandwidth limiter (no throttling)
    pub fn unlimited() -> Self {
        Self::new(u64::MAX)
    }

    /// Wait for permission to send/receive specified number of bytes
    ///
    /// # Arguments
    /// * `bytes` - Number of bytes to send/receive
    ///
    /// # Returns
    /// * `Ok(())` when permission is granted
    /// * `Err(...)` if there's an error
    pub async fn acquire(&self, bytes: u64) -> Result<()> {
        if self.max_bytes_per_sec == u64::MAX {
            return Ok(()); // Unlimited
        }

        let bytes_f64 = bytes as f64;

        loop {
            // Refill tokens based on time elapsed
            self.refill_tokens();

            // Try to consume tokens
            {
                let mut tokens = self.tokens.lock().expect("lock should not be poisoned");
                if *tokens >= bytes_f64 {
                    *tokens -= bytes_f64;
                    return Ok(());
                }
            }

            // Calculate wait time needed
            let wait_time = self.calculate_wait_time(bytes_f64);
            if wait_time > Duration::ZERO {
                sleep(wait_time).await;
            }
        }
    }

    /// Refill tokens based on elapsed time
    fn refill_tokens(&self) {
        let now = Instant::now();
        let mut last_refill = self
            .last_refill
            .lock()
            .expect("lock should not be poisoned");
        let elapsed = now.duration_since(*last_refill);

        if elapsed >= Duration::from_millis(10) {
            // Minimum refill interval
            let tokens_to_add = elapsed.as_secs_f64() * self.max_bytes_per_sec as f64;

            let mut tokens = self.tokens.lock().expect("lock should not be poisoned");
            *tokens = (*tokens + tokens_to_add).min(self.bucket_capacity);
            *last_refill = now;
        }
    }

    /// Calculate how long to wait for specified bytes
    fn calculate_wait_time(&self, bytes: f64) -> Duration {
        let tokens = *self.tokens.lock().expect("lock should not be poisoned");
        let deficit = bytes - tokens;

        if deficit <= 0.0 {
            return Duration::ZERO;
        }

        let wait_seconds = deficit / self.max_bytes_per_sec as f64;
        Duration::from_secs_f64(wait_seconds.max(0.001)) // Minimum 1ms wait
    }

    /// Get current bandwidth utilization (0.0 to 1.0)
    pub fn utilization(&self) -> f64 {
        let tokens = *self.tokens.lock().expect("lock should not be poisoned");
        1.0 - (tokens / self.bucket_capacity)
    }

    /// Get available bytes without waiting
    pub fn available_bytes(&self) -> u64 {
        self.refill_tokens();
        *self.tokens.lock().expect("lock should not be poisoned") as u64
    }

    /// Update bandwidth limit
    pub fn set_limit(&mut self, bytes_per_sec: u64) {
        self.max_bytes_per_sec = bytes_per_sec;
        self.bucket_capacity = bytes_per_sec as f64 * 2.0;

        // Reset tokens to not exceed new capacity
        let mut tokens = self.tokens.lock().expect("lock should not be poisoned");
        *tokens = (*tokens).min(self.bucket_capacity);
    }
}

/// Bandwidth monitoring and statistics
#[derive(Debug, Clone)]
pub struct BandwidthMonitor {
    /// Total bytes transferred
    total_bytes: Arc<Mutex<u64>>,
    /// Transfer start time
    start_time: Instant,
    /// Recent transfer samples for rate calculation
    samples: Arc<Mutex<Vec<(Instant, u64)>>>,
    /// Maximum samples to keep
    max_samples: usize,
}

impl BandwidthMonitor {
    /// Create a new bandwidth monitor
    pub fn new() -> Self {
        Self {
            total_bytes: Arc::new(Mutex::new(0)),
            start_time: Instant::now(),
            samples: Arc::new(Mutex::new(Vec::new())),
            max_samples: 100, // Keep last 100 samples
        }
    }

    /// Record bytes transferred
    pub fn record_bytes(&self, bytes: u64) {
        let now = Instant::now();

        // Update total
        {
            let mut total = self
                .total_bytes
                .lock()
                .expect("lock should not be poisoned");
            *total += bytes;
        }

        // Add sample
        {
            let mut samples = self.samples.lock().expect("lock should not be poisoned");
            samples.push((now, bytes));

            // Keep only recent samples
            let samples_len = samples.len();
            if samples_len > self.max_samples {
                samples.drain(0..samples_len - self.max_samples);
            }
        }
    }

    /// Get average transfer rate in bytes per second
    pub fn average_rate(&self) -> f64 {
        let total = *self
            .total_bytes
            .lock()
            .expect("lock should not be poisoned");
        let elapsed = self.start_time.elapsed().as_secs_f64();

        if elapsed > 0.0 {
            total as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Get current transfer rate (last few seconds)
    pub fn current_rate(&self) -> f64 {
        let samples = self.samples.lock().expect("lock should not be poisoned");
        let now = Instant::now();
        let cutoff = now - Duration::from_secs(5); // Last 5 seconds

        let recent_bytes: u64 = samples
            .iter()
            .filter(|(time, _)| *time >= cutoff)
            .map(|(_, bytes)| *bytes)
            .sum();

        let recent_duration =
            if let Some((first_time, _)) = samples.iter().find(|(time, _)| *time >= cutoff) {
                now.duration_since(*first_time).as_secs_f64()
            } else {
                1.0
            };

        if recent_duration > 0.0 {
            recent_bytes as f64 / recent_duration
        } else {
            0.0
        }
    }

    /// Get total bytes transferred
    pub fn total_bytes(&self) -> u64 {
        *self
            .total_bytes
            .lock()
            .expect("lock should not be poisoned")
    }

    /// Get elapsed time since monitoring started
    pub fn elapsed_time(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get estimated time remaining for a given total size
    pub fn eta(&self, total_size: u64) -> Option<Duration> {
        let transferred = self.total_bytes();
        let rate = self.current_rate();

        if rate > 0.0 && transferred < total_size {
            let remaining = total_size - transferred;
            let seconds_remaining = remaining as f64 / rate;
            Some(Duration::from_secs_f64(seconds_remaining))
        } else {
            None
        }
    }
}

impl Default for BandwidthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Adaptive bandwidth limiter that adjusts based on network conditions
#[derive(Debug)]
pub struct AdaptiveBandwidthLimiter {
    /// Base bandwidth limiter
    limiter: BandwidthLimiter,
    /// Monitor for tracking performance
    monitor: BandwidthMonitor,
    /// Target utilization (0.0 to 1.0)
    target_utilization: f64,
    /// Minimum bandwidth limit
    min_limit: u64,
    /// Maximum bandwidth limit
    max_limit: u64,
    /// Last adjustment time
    last_adjustment: Instant,
    /// Adjustment interval
    adjustment_interval: Duration,
}

impl AdaptiveBandwidthLimiter {
    /// Create a new adaptive bandwidth limiter
    pub fn new(
        initial_limit: u64,
        min_limit: u64,
        max_limit: u64,
        target_utilization: f64,
    ) -> Self {
        Self {
            limiter: BandwidthLimiter::new(initial_limit),
            monitor: BandwidthMonitor::new(),
            target_utilization: target_utilization.clamp(0.1, 0.9),
            min_limit,
            max_limit,
            last_adjustment: Instant::now(),
            adjustment_interval: Duration::from_secs(30), // Adjust every 30 seconds
        }
    }

    /// Acquire bandwidth permission and adapt limits
    pub async fn acquire(&mut self, bytes: u64) -> Result<()> {
        // Record bytes for monitoring
        self.monitor.record_bytes(bytes);

        // Adapt limits if needed
        self.adapt_limits();

        // Acquire permission from base limiter
        self.limiter.acquire(bytes).await
    }

    /// Adapt bandwidth limits based on performance
    fn adapt_limits(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_adjustment) < self.adjustment_interval {
            return;
        }

        let utilization = self.limiter.utilization();
        let current_limit = self.limiter.max_bytes_per_sec;

        let new_limit = if utilization < self.target_utilization * 0.8 {
            // Low utilization, increase limit
            (current_limit as f64 * 1.1) as u64
        } else if utilization > self.target_utilization * 1.2 {
            // High utilization, decrease limit
            (current_limit as f64 * 0.9) as u64
        } else {
            current_limit // No change needed
        };

        let clamped_limit = new_limit.clamp(self.min_limit, self.max_limit);

        if clamped_limit != current_limit {
            self.limiter.set_limit(clamped_limit);
        }

        self.last_adjustment = now;
    }

    /// Get current statistics
    pub fn stats(&self) -> BandwidthStats {
        BandwidthStats {
            current_limit: self.limiter.max_bytes_per_sec,
            utilization: self.limiter.utilization(),
            average_rate: self.monitor.average_rate(),
            current_rate: self.monitor.current_rate(),
            total_bytes: self.monitor.total_bytes(),
            elapsed_time: self.monitor.elapsed_time(),
        }
    }
}

/// Bandwidth statistics
#[derive(Debug, Clone)]
pub struct BandwidthStats {
    pub current_limit: u64,
    pub utilization: f64,
    pub average_rate: f64,
    pub current_rate: f64,
    pub total_bytes: u64,
    pub elapsed_time: Duration,
}

impl BandwidthStats {
    /// Format as human-readable string
    pub fn format(&self) -> String {
        format!(
            "Limit: {}/s, Util: {:.1}%, Avg: {}/s, Current: {}/s, Total: {}, Time: {}",
            format_bytes(self.current_limit),
            self.utilization * 100.0,
            format_bytes(self.average_rate as u64),
            format_bytes(self.current_rate as u64),
            format_bytes(self.total_bytes),
            format_duration(self.elapsed_time)
        )
    }
}

/// Format bytes in human-readable format
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    const THRESHOLD: u64 = 1024;

    if bytes < THRESHOLD {
        return format!("{} B", bytes);
    }

    let mut value = bytes as f64;
    let mut unit_index = 0;

    while value >= THRESHOLD as f64 && unit_index < UNITS.len() - 1 {
        value /= THRESHOLD as f64;
        unit_index += 1;
    }

    format!("{:.1} {}", value, UNITS[unit_index])
}

/// Format duration in human-readable format
pub fn format_duration(duration: Duration) -> String {
    let total_seconds = duration.as_secs();
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;

    if hours > 0 {
        format!("{}:{:02}:{:02}", hours, minutes, seconds)
    } else if minutes > 0 {
        format!("{}:{:02}", minutes, seconds)
    } else {
        format!("{}s", seconds)
    }
}

/// Bandwidth-limited reader wrapper
pub struct BandwidthLimitedReader<R> {
    inner: R,
    limiter: Arc<BandwidthLimiter>,
    monitor: Arc<BandwidthMonitor>,
    acquire_future: Option<std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>>>,
    pending_bytes: u64,
}

impl<R> BandwidthLimitedReader<R> {
    /// Create a new bandwidth-limited reader
    pub fn new(reader: R, limiter: Arc<BandwidthLimiter>, monitor: Arc<BandwidthMonitor>) -> Self {
        Self {
            inner: reader,
            limiter,
            monitor,
            acquire_future: None,
            pending_bytes: 0,
        }
    }
}

impl<R: tokio::io::AsyncRead + Unpin> tokio::io::AsyncRead for BandwidthLimitedReader<R> {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        // If we have a pending acquire future, poll it first
        if let Some(mut future) = self.acquire_future.take() {
            match future.as_mut().poll(cx) {
                std::task::Poll::Ready(Ok(())) => {
                    // Acquire completed, we can proceed with the read
                    self.pending_bytes = 0;
                }
                std::task::Poll::Ready(Err(e)) => {
                    return std::task::Poll::Ready(Err(std::io::Error::other(e)));
                }
                std::task::Poll::Pending => {
                    // Still waiting for acquire, put the future back and return Pending
                    self.acquire_future = Some(future);
                    return std::task::Poll::Pending;
                }
            }
        }

        let initial_filled = buf.filled().len();

        match std::pin::Pin::new(&mut self.inner).poll_read(cx, buf) {
            std::task::Poll::Ready(Ok(())) => {
                let bytes_read = buf.filled().len() - initial_filled;
                if bytes_read > 0 {
                    self.monitor.record_bytes(bytes_read as u64);

                    // Create acquire future for the bytes we just read
                    let limiter = Arc::clone(&self.limiter);
                    let acquire_fut =
                        Box::pin(async move { limiter.acquire(bytes_read as u64).await });

                    self.acquire_future = Some(acquire_fut);
                    self.pending_bytes = bytes_read as u64;

                    // Try to poll the acquire future immediately
                    if let Some(mut future) = self.acquire_future.take() {
                        match future.as_mut().poll(cx) {
                            std::task::Poll::Ready(Ok(())) => {
                                // Acquire completed immediately
                                self.pending_bytes = 0;
                                std::task::Poll::Ready(Ok(()))
                            }
                            std::task::Poll::Ready(Err(e)) => {
                                std::task::Poll::Ready(Err(std::io::Error::other(e)))
                            }
                            std::task::Poll::Pending => {
                                // Need to wait for acquire, put future back
                                self.acquire_future = Some(future);
                                std::task::Poll::Pending
                            }
                        }
                    } else {
                        std::task::Poll::Ready(Ok(()))
                    }
                } else {
                    std::task::Poll::Ready(Ok(()))
                }
            }
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::Duration;

    #[test]
    fn test_bandwidth_limiter_creation() {
        let limiter = BandwidthLimiter::new(1024 * 1024); // 1 MB/s
        assert_eq!(limiter.max_bytes_per_sec, 1024 * 1024);
    }

    #[test]
    fn test_unlimited_limiter() {
        let limiter = BandwidthLimiter::unlimited();
        assert_eq!(limiter.max_bytes_per_sec, u64::MAX);
    }

    #[tokio::test]
    async fn test_bandwidth_limiting() {
        let limiter = BandwidthLimiter::new(1000000); // 1MB/s (high rate to avoid long waits)

        // Test that basic limiting works without hanging
        let result = limiter.acquire(1000).await;
        assert!(result.is_ok());

        // Test multiple small requests work
        let result = limiter.acquire(500).await;
        assert!(result.is_ok());

        let result = limiter.acquire(200).await;
        assert!(result.is_ok());

        // Test unlimited limiter
        let unlimited = BandwidthLimiter::unlimited();
        let result = unlimited.acquire(u64::MAX / 2).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_bandwidth_monitor() {
        let monitor = BandwidthMonitor::new();

        monitor.record_bytes(1000);
        monitor.record_bytes(500);

        assert_eq!(monitor.total_bytes(), 1500);
    }

    #[test]
    fn test_adaptive_limiter() {
        let limiter = AdaptiveBandwidthLimiter::new(
            1024 * 1024,     // 1 MB/s initial
            512 * 1024,      // 512 KB/s min
            2 * 1024 * 1024, // 2 MB/s max
            0.8,             // 80% target utilization
        );

        let stats = limiter.stats();
        assert_eq!(stats.current_limit, 1024 * 1024);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1048576), "1.0 MB");
        assert_eq!(format_bytes(1073741824), "1.0 GB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(30)), "30s");
        assert_eq!(format_duration(Duration::from_secs(90)), "1:30");
        assert_eq!(format_duration(Duration::from_secs(3665)), "1:01:05");
    }
}
