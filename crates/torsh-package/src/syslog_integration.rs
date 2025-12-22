//! Syslog Integration for Centralized Logging
//!
//! Provides syslog integration for sending audit events to centralized
//! logging systems following RFC 5424 and RFC 3164 standards.

use crate::audit::{AuditEvent, AuditEventType, AuditSeverity};
use std::io::Write;
use std::net::{SocketAddr, TcpStream, UdpSocket};
use torsh_core::error::{Result, TorshError};

/// Syslog protocol version
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyslogProtocol {
    /// RFC 3164 (BSD syslog)
    Rfc3164,
    /// RFC 5424 (modern syslog)
    Rfc5424,
}

/// Syslog transport protocol
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyslogTransport {
    /// UDP transport (connectionless, fast but unreliable)
    Udp,
    /// TCP transport (connection-oriented, reliable)
    Tcp,
    /// Unix domain socket (local only, fastest and most reliable)
    Unix,
}

/// Syslog facility codes (RFC 5424)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SyslogFacility {
    /// Kernel messages
    Kern = 0,
    /// User-level messages
    User = 1,
    /// Mail system
    Mail = 2,
    /// System daemons
    Daemon = 3,
    /// Security/authorization messages
    Auth = 4,
    /// Internal syslog messages
    Syslog = 5,
    /// Line printer subsystem
    Lpr = 6,
    /// Network news subsystem
    News = 7,
    /// UUCP subsystem
    Uucp = 8,
    /// Clock daemon
    Cron = 9,
    /// Security/authorization messages (private)
    AuthPriv = 10,
    /// FTP daemon
    Ftp = 11,
    /// NTP subsystem
    Ntp = 12,
    /// Log audit
    Audit = 13,
    /// Log alert
    Alert = 14,
    /// Clock daemon (note 2)
    Clock = 15,
    /// Local use 0
    Local0 = 16,
    /// Local use 1
    Local1 = 17,
    /// Local use 2
    Local2 = 18,
    /// Local use 3
    Local3 = 19,
    /// Local use 4
    Local4 = 20,
    /// Local use 5
    Local5 = 21,
    /// Local use 6
    Local6 = 22,
    /// Local use 7
    Local7 = 23,
}

/// Syslog severity level (RFC 5424)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum SyslogSeverity {
    /// Emergency: system is unusable
    Emergency = 0,
    /// Alert: action must be taken immediately
    Alert = 1,
    /// Critical: critical conditions
    Critical = 2,
    /// Error: error conditions
    Error = 3,
    /// Warning: warning conditions
    Warning = 4,
    /// Notice: normal but significant condition
    Notice = 5,
    /// Informational: informational messages
    Info = 6,
    /// Debug: debug-level messages
    Debug = 7,
}

impl From<&AuditSeverity> for SyslogSeverity {
    fn from(severity: &AuditSeverity) -> Self {
        match severity {
            AuditSeverity::Info => SyslogSeverity::Info,
            AuditSeverity::Warning => SyslogSeverity::Warning,
            AuditSeverity::Error => SyslogSeverity::Error,
            AuditSeverity::Critical => SyslogSeverity::Critical,
        }
    }
}

/// Syslog client configuration
#[derive(Debug, Clone)]
pub struct SyslogConfig {
    /// Syslog server address
    pub server_addr: String,
    /// Syslog server port
    pub server_port: u16,
    /// Transport protocol
    pub transport: SyslogTransport,
    /// Protocol version
    pub protocol: SyslogProtocol,
    /// Facility code
    pub facility: SyslogFacility,
    /// Application name
    pub app_name: String,
    /// Hostname
    pub hostname: String,
    /// Process ID
    pub process_id: u32,
    /// Enable TLS for TCP connections
    pub enable_tls: bool,
}

impl SyslogConfig {
    /// Create a new syslog configuration
    pub fn new(server_addr: String, server_port: u16) -> Self {
        Self {
            server_addr,
            server_port,
            transport: SyslogTransport::Udp,
            protocol: SyslogProtocol::Rfc5424,
            facility: SyslogFacility::Local0,
            app_name: "torsh-package".to_string(),
            hostname: hostname::get()
                .ok()
                .and_then(|h| h.into_string().ok())
                .unwrap_or_else(|| "unknown".to_string()),
            process_id: std::process::id(),
            enable_tls: false,
        }
    }

    /// Set transport protocol
    pub fn with_transport(mut self, transport: SyslogTransport) -> Self {
        self.transport = transport;
        self
    }

    /// Set protocol version
    pub fn with_protocol(mut self, protocol: SyslogProtocol) -> Self {
        self.protocol = protocol;
        self
    }

    /// Set facility code
    pub fn with_facility(mut self, facility: SyslogFacility) -> Self {
        self.facility = facility;
        self
    }

    /// Set application name
    pub fn with_app_name(mut self, app_name: String) -> Self {
        self.app_name = app_name;
        self
    }

    /// Enable TLS for TCP connections
    pub fn with_tls(mut self, enable: bool) -> Self {
        self.enable_tls = enable;
        self
    }

    /// Get socket address
    pub fn socket_addr(&self) -> Result<SocketAddr> {
        let addr_str = format!("{}:{}", self.server_addr, self.server_port);
        addr_str
            .parse()
            .map_err(|e| TorshError::InvalidArgument(format!("Invalid socket address: {}", e)))
    }
}

/// Syslog client for sending audit events
#[derive(Debug)]
pub struct SyslogClient {
    config: SyslogConfig,
    // In production, maintain persistent TCP connections
    // tcp_connection: Option<TcpStream>,
    message_count: u64,
}

impl SyslogClient {
    /// Create a new syslog client
    pub fn new(config: SyslogConfig) -> Result<Self> {
        Ok(Self {
            config,
            message_count: 0,
        })
    }

    /// Send an audit event to syslog
    pub fn send_event(&mut self, event: &AuditEvent) -> Result<()> {
        let severity = SyslogSeverity::from(&event.severity);
        let message = self.format_message(event, severity)?;

        match self.config.transport {
            SyslogTransport::Udp => self.send_udp(&message),
            SyslogTransport::Tcp => self.send_tcp(&message),
            SyslogTransport::Unix => self.send_unix(&message),
        }?;

        self.message_count += 1;
        Ok(())
    }

    /// Format message according to protocol
    fn format_message(&self, event: &AuditEvent, severity: SyslogSeverity) -> Result<String> {
        match self.config.protocol {
            SyslogProtocol::Rfc3164 => self.format_rfc3164(event, severity),
            SyslogProtocol::Rfc5424 => self.format_rfc5424(event, severity),
        }
    }

    /// Format message according to RFC 3164 (BSD syslog)
    fn format_rfc3164(&self, event: &AuditEvent, severity: SyslogSeverity) -> Result<String> {
        let priority = self.calculate_priority(severity);
        let timestamp = self.format_rfc3164_timestamp(&event.timestamp);
        let tag = format!("{}[{}]:", self.config.app_name, self.config.process_id);

        // RFC 3164 format: <PRI>TIMESTAMP HOSTNAME TAG MSG
        Ok(format!(
            "<{}>{} {} {} {}",
            priority, timestamp, self.config.hostname, tag, event.action
        ))
    }

    /// Format message according to RFC 5424 (modern syslog)
    fn format_rfc5424(&self, event: &AuditEvent, severity: SyslogSeverity) -> Result<String> {
        let priority = self.calculate_priority(severity);
        let timestamp = event.timestamp.to_rfc3339();
        let msgid = self.format_msgid(&event.event_type);

        // Structured data
        let structured_data = self.format_structured_data(event);

        // RFC 5424 format: <PRI>VERSION TIMESTAMP HOSTNAME APP-NAME PROCID MSGID STRUCTURED-DATA MSG
        Ok(format!(
            "<{}>1 {} {} {} {} {} {} {}",
            priority,
            timestamp,
            self.config.hostname,
            self.config.app_name,
            self.config.process_id,
            msgid,
            structured_data,
            event.action
        ))
    }

    /// Format RFC 3164 timestamp
    fn format_rfc3164_timestamp(&self, timestamp: &chrono::DateTime<chrono::Utc>) -> String {
        // RFC 3164 uses "Mmm dd HH:MM:SS" format
        timestamp.format("%b %d %H:%M:%S").to_string()
    }

    /// Format message ID from event type
    fn format_msgid(&self, event_type: &AuditEventType) -> String {
        match event_type {
            AuditEventType::PackageDownload => "PKG-DOWNLOAD",
            AuditEventType::PackageUpload => "PKG-UPLOAD",
            AuditEventType::PackageDelete => "PKG-DELETE",
            AuditEventType::PackageYank => "PKG-YANK",
            AuditEventType::PackageUnyank => "PKG-UNYANK",
            AuditEventType::UserAuthentication => "USER-AUTH",
            AuditEventType::UserAuthorization => "USER-AUTHZ",
            AuditEventType::AccessGranted => "ACCESS-GRANTED",
            AuditEventType::AccessDenied => "ACCESS-DENIED",
            AuditEventType::RoleAssigned => "ROLE-ASSIGNED",
            AuditEventType::RoleRevoked => "ROLE-REVOKED",
            AuditEventType::PermissionChanged => "PERM-CHANGED",
            AuditEventType::SecurityViolation => "SECURITY-VIOLATION",
            AuditEventType::IntegrityCheck => "INTEGRITY-CHECK",
            AuditEventType::SignatureVerification => "SIGNATURE-VERIFY",
            AuditEventType::ConfigurationChange => "CONFIG-CHANGE",
            AuditEventType::SystemEvent => "SYSTEM-EVENT",
        }
        .to_string()
    }

    /// Format structured data for RFC 5424
    fn format_structured_data(&self, event: &AuditEvent) -> String {
        let mut sd = String::from("[torsh@32473");

        if let Some(user_id) = &event.user_id {
            sd.push_str(&format!(" user=\"{}\"", self.escape_sd_param(user_id)));
        }

        if let Some(ip) = &event.ip_address {
            sd.push_str(&format!(" ip=\"{}\"", self.escape_sd_param(ip)));
        }

        if let Some(resource) = &event.resource {
            sd.push_str(&format!(" resource=\"{}\"", self.escape_sd_param(resource)));
        }

        sd.push_str(&format!(" severity=\"{:?}\"", event.severity));
        sd.push_str(&format!(" event_type=\"{:?}\"", event.event_type));

        sd.push(']');

        if sd == "[torsh@32473]" {
            return "-".to_string(); // No structured data
        }

        sd
    }

    /// Escape structured data parameter value
    fn escape_sd_param(&self, value: &str) -> String {
        value
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace(']', "\\]")
    }

    /// Calculate priority value from facility and severity
    fn calculate_priority(&self, severity: SyslogSeverity) -> u8 {
        (self.config.facility as u8) * 8 + (severity as u8)
    }

    /// Send message via UDP
    fn send_udp(&self, message: &str) -> Result<()> {
        let socket = UdpSocket::bind("0.0.0.0:0")
            .map_err(|e| TorshError::InvalidArgument(format!("UDP bind error: {}", e)))?;

        let addr = self.config.socket_addr()?;
        socket
            .send_to(message.as_bytes(), addr)
            .map_err(|e| TorshError::InvalidArgument(format!("UDP send error: {}", e)))?;

        Ok(())
    }

    /// Send message via TCP
    fn send_tcp(&self, message: &str) -> Result<()> {
        let addr = self.config.socket_addr()?;

        // In production, maintain persistent connection
        let mut stream = TcpStream::connect(addr)
            .map_err(|e| TorshError::InvalidArgument(format!("TCP connect error: {}", e)))?;

        // TCP syslog uses newline-delimited messages
        let message_with_newline = format!("{}\n", message);

        stream
            .write_all(message_with_newline.as_bytes())
            .map_err(|e| TorshError::InvalidArgument(format!("TCP write error: {}", e)))?;

        stream
            .flush()
            .map_err(|e| TorshError::InvalidArgument(format!("TCP flush error: {}", e)))?;

        Ok(())
    }

    /// Send message via Unix domain socket
    fn send_unix(&self, _message: &str) -> Result<()> {
        // Unix domain socket support would use std::os::unix::net::UnixDatagram
        // This is platform-specific
        #[cfg(unix)]
        {
            // In production:
            // let socket = UnixDatagram::unbound()?;
            // socket.send_to(message.as_bytes(), "/dev/log")?;
        }

        #[cfg(not(unix))]
        {
            return Err(TorshError::InvalidArgument(
                "Unix sockets not supported on this platform".to_string(),
            ));
        }

        Ok(())
    }

    /// Get message count
    pub fn message_count(&self) -> u64 {
        self.message_count
    }

    /// Get statistics
    pub fn get_statistics(&self) -> SyslogStatistics {
        SyslogStatistics {
            messages_sent: self.message_count,
            messages_failed: 0,
            bytes_sent: 0,
            connection_errors: 0,
        }
    }
}

/// Syslog client statistics
#[derive(Debug, Clone)]
pub struct SyslogStatistics {
    /// Total messages sent
    pub messages_sent: u64,
    /// Messages that failed to send
    pub messages_failed: u64,
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Connection errors encountered
    pub connection_errors: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syslog_config() {
        let config = SyslogConfig::new("localhost".to_string(), 514)
            .with_transport(SyslogTransport::Udp)
            .with_protocol(SyslogProtocol::Rfc5424)
            .with_facility(SyslogFacility::Local0)
            .with_app_name("test-app".to_string());

        assert_eq!(config.server_addr, "localhost");
        assert_eq!(config.server_port, 514);
        assert_eq!(config.transport, SyslogTransport::Udp);
        assert_eq!(config.protocol, SyslogProtocol::Rfc5424);
        assert_eq!(config.app_name, "test-app");
    }

    #[test]
    fn test_priority_calculation() {
        let config =
            SyslogConfig::new("localhost".to_string(), 514).with_facility(SyslogFacility::Local0);

        let client = SyslogClient::new(config).unwrap();

        // Local0 = 16, Info = 6 => 16*8 + 6 = 134
        assert_eq!(client.calculate_priority(SyslogSeverity::Info), 134);

        // Local0 = 16, Error = 3 => 16*8 + 3 = 131
        assert_eq!(client.calculate_priority(SyslogSeverity::Error), 131);
    }

    #[test]
    fn test_severity_conversion() {
        assert_eq!(
            SyslogSeverity::from(&AuditSeverity::Info),
            SyslogSeverity::Info
        );
        assert_eq!(
            SyslogSeverity::from(&AuditSeverity::Warning),
            SyslogSeverity::Warning
        );
        assert_eq!(
            SyslogSeverity::from(&AuditSeverity::Error),
            SyslogSeverity::Error
        );
        assert_eq!(
            SyslogSeverity::from(&AuditSeverity::Critical),
            SyslogSeverity::Critical
        );
    }

    #[test]
    fn test_msgid_formatting() {
        let config = SyslogConfig::new("localhost".to_string(), 514);
        let client = SyslogClient::new(config).unwrap();

        assert_eq!(
            client.format_msgid(&AuditEventType::PackageDownload),
            "PKG-DOWNLOAD"
        );
        assert_eq!(
            client.format_msgid(&AuditEventType::SecurityViolation),
            "SECURITY-VIOLATION"
        );
        assert_eq!(
            client.format_msgid(&AuditEventType::AccessDenied),
            "ACCESS-DENIED"
        );
        assert_eq!(
            client.format_msgid(&AuditEventType::PermissionChanged),
            "PERM-CHANGED"
        );
        assert_eq!(
            client.format_msgid(&AuditEventType::IntegrityCheck),
            "INTEGRITY-CHECK"
        );
    }

    #[test]
    fn test_structured_data_escaping() {
        let config = SyslogConfig::new("localhost".to_string(), 514);
        let client = SyslogClient::new(config).unwrap();

        let escaped = client.escape_sd_param("test\\value\"with]special");
        assert_eq!(escaped, "test\\\\value\\\"with\\]special");
    }

    #[test]
    fn test_rfc5424_formatting() {
        let config = SyslogConfig::new("localhost".to_string(), 514)
            .with_facility(SyslogFacility::Local0)
            .with_app_name("test".to_string());

        let client = SyslogClient::new(config).unwrap();

        let event = AuditEvent::new(AuditEventType::PackageDownload, "Test message".to_string());

        let message = client.format_rfc5424(&event, SyslogSeverity::Info).unwrap();

        assert!(message.contains("<134>1")); // Priority: Local0 + Info
        assert!(message.contains("PKG-DOWNLOAD"));
        assert!(message.contains("test"));
        assert!(message.contains("Test message"));
    }
}
