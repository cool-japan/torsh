//! Alert notification channels
//!
//! This module handles various notification channels for sending alerts,
//! including email, Slack, Discord, webhooks, and more.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Alert notification channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannel {
    Email {
        recipients: Vec<String>,
        smtp_config: SmtpConfig,
    },
    Slack {
        webhook_url: String,
        channel: String,
    },
    Discord {
        webhook_url: String,
    },
    PagerDuty {
        integration_key: String,
    },
    Webhook {
        url: String,
        headers: HashMap<String, String>,
    },
    Log {
        level: String,
        format: String,
    },
    Console,
}

/// SMTP configuration for email alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmtpConfig {
    pub server: String,
    pub port: u16,
    pub username: String,
    pub password: String,
    pub use_tls: bool,
}
