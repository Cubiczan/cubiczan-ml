//! Telegram & Discord Alert Bot — Push notifications when consensus crosses threshold.
//!
//! Provides a message-brokering system that monitors swarm consensus signals
//! and dispatches alerts to Telegram and Discord when predefined thresholds
//! are breached (confidence spike, signal flip, volatility regime change).
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────┐
//! │ Consensus Engine │ → produces ConsensusResult
//! └────────┬─────────┘
//!          │
//!          ▼
//! ┌──────────────────┐
//! │ AlertManager     │ ← evaluates threshold rules
//! └────────┬─────────┘
//!          │ Alert (if triggered)
//!          ▼
//! ┌──────────────────────────────┐
//! │ TelegramDispatcher           │
//! │ DiscordDispatcher            │
//! │ WebhookDispatcher            │
//! └──────────────────────────────┘
//! ```
//!
//! Each dispatcher formats messages for its platform and sends via HTTP API.
//! Rate limiting and deduplication prevent alert storms.

use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, ToSocketAddrs};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

/// Platform to send alerts to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlertPlatform {
    Telegram,
    Discord,
    Webhook,
}

impl AlertPlatform {
    pub fn as_str(&self) -> &'static str {
        match self {
            AlertPlatform::Telegram => "TELEGRAM",
            AlertPlatform::Discord => "DISCORD",
            AlertPlatform::Webhook => "WEBHOOK",
        }
    }
}

/// Severity level for an alert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

impl AlertSeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            AlertSeverity::Info => "INFO",
            AlertSeverity::Warning => "WARNING",
            AlertSeverity::Critical => "CRITICAL",
        }
    }

    /// Telegram emoji for severity.
    pub fn emoji(&self) -> &'static str {
        match self {
            AlertSeverity::Info => "\u{2139}\u{FE0F}",
            AlertSeverity::Warning => "\u{26A0}\u{FE0F}",
            AlertSeverity::Critical => "\u{1F534}",
        }
    }

    /// Discord color for embed.
    pub fn discord_color(&self) -> u32 {
        match self {
            AlertSeverity::Info => 0x3498DB,
            AlertSeverity::Warning => 0xF39C12,
            AlertSeverity::Critical => 0xE74C3C,
        }
    }
}

/// Reason for the alert being triggered.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertReason {
    /// Confidence crossed above a high threshold.
    HighConfidence { value: f64, threshold: f64 },
    /// Confidence dropped below a low threshold.
    LowConfidence { value: f64, threshold: f64 },
    /// Signal flipped from one direction to another.
    SignalFlip {
        from: String,
        to: String,
        market: String,
    },
    /// Volatility regime changed.
    VolatilityRegimeChange {
        from: String,
        to: String,
    },
    /// Liquidation risk level changed.
    LiquidationRiskChange {
        from: String,
        to: String,
    },
    /// Custom alert reason.
    Custom { message: String },
}

/// An alert ready for dispatch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub severity: AlertSeverity,
    pub reason: AlertReason,
    pub market: String,
    pub signal: Signal,
    pub confidence: f64,
    pub timestamp_ms: i64,
    pub platforms: Vec<AlertPlatform>,
}

impl Alert {
    /// Create a new alert with a unique ID.
    pub fn new(severity: AlertSeverity, reason: AlertReason, market: &str, signal: Signal, confidence: f64) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            severity,
            reason,
            market: market.to_string(),
            signal,
            confidence,
            timestamp_ms: chrono::Utc::now().timestamp_millis(),
            platforms: vec![],
        }
    }

    /// Add target platforms.
    pub fn to_platforms(mut self, platforms: &[AlertPlatform]) -> Self {
        self.platforms = platforms.to_vec();
        self
    }

    /// Format as a human-readable text message (Telegram-compatible).
    pub fn to_text(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("{} *{}*", self.severity.emoji(), self.severity.as_str()));
        lines.push(format!("Market: {}", self.market));
        lines.push(format!("Signal: {} ({:.1}%)", self.signal.as_str(), self.confidence));
        lines.push(format!("Reason: {}", self.format_reason()));
        lines.join("\n")
    }

    /// Format as a Discord embed-compatible JSON payload.
    pub fn to_discord_embed(&self) -> serde_json::Value {
        serde_json::json!({
            "embeds": [{
                "title": format!("{} {}", self.severity.emoji(), self.severity.as_str()),
                "color": self.severity.discord_color(),
                "fields": [
                    { "name": "Market", "value": self.market, "inline": true },
                    { "name": "Signal", "value": format!("{} ({:.1}%)", self.signal.as_str(), self.confidence), "inline": true },
                    { "name": "Reason", "value": self.format_reason() }
                ],
                "timestamp": chrono::DateTime::from_timestamp_millis(self.timestamp_ms)
                    .map(|dt| dt.to_rfc3339())
                    .unwrap_or_default(),
            }]
        })
    }

    fn format_reason(&self) -> String {
        match &self.reason {
            AlertReason::HighConfidence { value, threshold } => {
                format!("Confidence {:.1}% crossed high threshold {:.1}%", value, threshold)
            }
            AlertReason::LowConfidence { value, threshold } => {
                format!("Confidence {:.1}% dropped below {:.1}%", value, threshold)
            }
            AlertReason::SignalFlip { from, to, market } => {
                format!("Signal flipped {} → {} on {}", from, to, market)
            }
            AlertReason::VolatilityRegimeChange { from, to } => {
                format!("Volatility regime changed: {} → {}", from, to)
            }
            AlertReason::LiquidationRiskChange { from, to } => {
                format!("Liquidation risk changed: {} → {}", from, to)
            }
            AlertReason::Custom { message } => message.clone(),
        }
    }
}

/// Threshold rules for alert generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRules {
    /// Fire alert when confidence exceeds this (0-100).
    pub high_confidence_threshold: f64,
    /// Fire alert when confidence drops below this (0-100).
    pub low_confidence_threshold: f64,
    /// Fire alert on any signal flip.
    pub alert_on_signal_flip: bool,
    /// Fire alert on volatility regime change.
    pub alert_on_volatility_change: bool,
    /// Fire alert on liquidation risk change.
    pub alert_on_liquidation_risk_change: bool,
    /// Minimum time between alerts for the same market (seconds).
    pub cooldown_secs: u64,
    /// Target platforms for all alerts.
    pub default_platforms: Vec<AlertPlatform>,
}

impl Default for AlertRules {
    fn default() -> Self {
        Self {
            high_confidence_threshold: 75.0,
            low_confidence_threshold: 20.0,
            alert_on_signal_flip: true,
            alert_on_volatility_change: true,
            alert_on_liquidation_risk_change: true,
            cooldown_secs: 300, // 5 minutes
            default_platforms: vec![AlertPlatform::Telegram, AlertPlatform::Discord],
        }
    }
}

/// Configuration for Telegram bot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelegramConfig {
    pub bot_token: String,
    pub chat_id: String,
    pub parse_mode: String,
}

impl Default for TelegramConfig {
    fn default() -> Self {
        Self {
            bot_token: String::new(),
            chat_id: String::new(),
            parse_mode: "Markdown".into(),
        }
    }
}

/// Configuration for Discord webhook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscordConfig {
    pub webhook_url: String,
    pub username: String,
}

impl Default for DiscordConfig {
    fn default() -> Self {
        Self {
            webhook_url: String::new(),
            username: "SwarmFi Alerts".into(),
        }
    }
}

/// Generic webhook configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookConfig {
    pub url: String,
    pub headers: HashMap<String, String>,
    /// Additional hostname allowlist entries for this webhook (exact match, lowercase).
    #[serde(default)]
    pub allowed_hosts: Vec<String>,
}

/// Hosts that outbound alert HTTP may target by default.
const DEFAULT_ALLOWED_ALERT_HOSTS: &[&str] = &[
    "api.telegram.org",
    "discord.com",
    "discordapp.com",
    "canary.discord.com",
];

const BLOCKED_ALERT_HOSTS: &[&str] = &[
    "localhost",
    "localhost.localdomain",
    "metadata",
    "metadata.google.internal",
    "metadata.google.com",
    "metadata.goog",
    "kubernetes.default",
    "kubernetes.default.svc",
    "kubernetes.default.svc.cluster.local",
];

/// Whether `ip` is loopback, link-local, private, metadata, or otherwise unroutable.
pub fn is_blocked_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_blocked_ipv4(v4),
        IpAddr::V6(v6) => is_blocked_ipv6(v6),
    }
}

fn is_blocked_ipv4(v4: Ipv4Addr) -> bool {
    let o = v4.octets();
    v4.is_unspecified()
        || v4.is_loopback()
        || v4.is_private()
        || v4.is_link_local()
        || v4.is_broadcast()
        || v4.is_documentation()
        || (o[0] == 0)
        // CGNAT 100.64.0.0/10
        || (o[0] == 100 && o[1] >= 64 && o[1] <= 127)
        // Benchmarking 198.18.0.0/15
        || (o[0] == 198 && (o[1] == 18 || o[1] == 19))
        // IETF protocol assignments / reserved
        || (o[0] == 192 && o[1] == 0 && o[2] == 0)
        // Alibaba / some cloud metadata
        || (o == [100, 100, 100, 200])
        || (o == [169, 254, 169, 254])
}

fn is_blocked_ipv6(v6: Ipv6Addr) -> bool {
    if v6.is_unspecified() || v6.is_loopback() {
        return true;
    }
    let s = v6.segments();
    // Unique local fc00::/7
    if s[0] & 0xfe00 == 0xfc00 {
        return true;
    }
    // Link-local fe80::/10
    if s[0] & 0xffc0 == 0xfe80 {
        return true;
    }
    // IPv4-mapped
    if let Some(v4) = v6.to_ipv4_mapped() {
        return is_blocked_ipv4(v4);
    }
    // Deprecated IPv4-compatible
    if let Some(v4) = v6.to_ipv4() {
        if s[0] == 0 && s[1] == 0 && s[2] == 0 && s[3] == 0 && s[4] == 0 && s[5] == 0 {
            return is_blocked_ipv4(v4);
        }
    }
    false
}

fn host_matches_allowlist(host: &str, extra: &[String]) -> bool {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    DEFAULT_ALLOWED_ALERT_HOSTS
        .iter()
        .any(|allowed| host == *allowed || host.ends_with(&format!(".{allowed}")))
        || extra
            .iter()
            .any(|allowed| {
                let allowed = allowed.trim_end_matches('.').to_ascii_lowercase();
                !allowed.is_empty() && (host == allowed || host.ends_with(&format!(".{allowed}")))
            })
}

fn is_blocked_hostname(host: &str) -> bool {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    BLOCKED_ALERT_HOSTS.contains(&host.as_str())
        || host.ends_with(".localhost")
        || host.ends_with(".internal")
        || host.ends_with(".local")
}

/// Validate an outbound alert URL: https only, host allowlist, no credentials,
/// and no link-local / metadata / private destination (including DNS results).
pub fn validate_alert_url(raw: &str, extra_hosts: &[String]) -> anyhow::Result<reqwest::Url> {
    validate_alert_url_with_resolve(raw, extra_hosts, true)
}

fn validate_alert_url_with_resolve(
    raw: &str,
    extra_hosts: &[String],
    resolve_dns: bool,
) -> anyhow::Result<reqwest::Url> {
    let url = reqwest::Url::parse(raw).map_err(|e| anyhow::anyhow!("invalid alert URL: {e}"))?;
    if url.scheme() != "https" {
        anyhow::bail!("alert URL scheme must be https");
    }
    if !url.username().is_empty() || url.password().is_some() {
        anyhow::bail!("alert URLs with credentials are not allowed");
    }
    let host = url
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("alert URL is missing a host"))?;

    if is_blocked_hostname(host) {
        anyhow::bail!("alert URL host is blocked: {host}");
    }

    if let Ok(ip) = host.parse::<IpAddr>() {
        if is_blocked_ip(ip) {
            anyhow::bail!("alert URL target IP is blocked: {ip}");
        }
        if !host_matches_allowlist(host, extra_hosts) {
            anyhow::bail!("alert URL host is not in the allowlist: {host}");
        }
    } else if !host_matches_allowlist(host, extra_hosts) {
        anyhow::bail!("alert URL host is not in the allowlist: {host}");
    }

    if resolve_dns {
        let port = url.port_or_known_default().unwrap_or(443);
        let addrs = (host, port)
            .to_socket_addrs()
            .map_err(|e| anyhow::anyhow!("failed to resolve alert URL host {host}: {e}"))?;
        let mut saw_any = false;
        for addr in addrs {
            saw_any = true;
            if is_blocked_ip(addr.ip()) {
                anyhow::bail!(
                    "alert URL host {host} resolved to blocked address {}",
                    addr.ip()
                );
            }
        }
        if !saw_any {
            anyhow::bail!("alert URL host {host} resolved to no addresses");
        }
    }

    Ok(url)
}

fn telegram_token_is_safe(token: &str) -> bool {
    !token.is_empty()
        && token
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == ':' || c == '_' || c == '-')
}

fn alert_http_client() -> anyhow::Result<reqwest::Client> {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| anyhow::anyhow!("failed to build alert HTTP client: {e}"))
}

/// Stateful alert manager with rate limiting and deduplication.
pub struct AlertManager {
    rules: AlertRules,
    /// Last alert time per market.
    last_alert_time: Arc<Mutex<HashMap<String, Instant>>>,
    /// Previous signals per market (for flip detection).
    previous_signals: Arc<Mutex<HashMap<String, Signal>>>,
    /// Previous volatility regime per market.
    previous_volatility: Arc<Mutex<HashMap<String, VolatilityRegime>>>,
    /// Previous liquidation risk per market.
    previous_liquidation_risk: Arc<Mutex<HashMap<String, RiskLevel>>>,
}

impl AlertManager {
    pub fn new(rules: AlertRules) -> Self {
        Self {
            rules,
            last_alert_time: Arc::new(Mutex::new(HashMap::new())),
            previous_signals: Arc::new(Mutex::new(HashMap::new())),
            previous_volatility: Arc::new(Mutex::new(HashMap::new())),
            previous_liquidation_risk: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Evaluate a consensus result against alert rules.
    ///
    /// Returns a list of alerts that should be dispatched (may be empty).
    pub async fn evaluate(&self, result: &ConsensusResult) -> Vec<Alert> {
        let mut alerts = Vec::new();
        let market = &result.market;

        // Check cooldown
        {
            let last_times = self.last_alert_time.lock().await;
            if let Some(last) = last_times.get(market) {
                let elapsed = last.elapsed().as_secs();
                if elapsed < self.rules.cooldown_secs {
                    return alerts;
                }
            }
        }

        // Check confidence thresholds
        if result.confidence >= self.rules.high_confidence_threshold {
            alerts.push(Alert::new(
                AlertSeverity::Critical,
                AlertReason::HighConfidence {
                    value: result.confidence,
                    threshold: self.rules.high_confidence_threshold,
                },
                market,
                result.signal,
                result.confidence,
            ).to_platforms(&self.rules.default_platforms));
        }

        if result.confidence <= self.rules.low_confidence_threshold {
            alerts.push(Alert::new(
                AlertSeverity::Warning,
                AlertReason::LowConfidence {
                    value: result.confidence,
                    threshold: self.rules.low_confidence_threshold,
                },
                market,
                result.signal,
                result.confidence,
            ).to_platforms(&self.rules.default_platforms));
        }

        // Check signal flip
        if self.rules.alert_on_signal_flip {
            let mut prev_signals = self.previous_signals.lock().await;
            if let Some(prev) = prev_signals.get(market) {
                if *prev != result.signal && *prev != Signal::Neutral && result.signal != Signal::Neutral {
                    alerts.push(Alert::new(
                        AlertSeverity::Warning,
                        AlertReason::SignalFlip {
                            from: prev.as_str().to_string(),
                            to: result.signal.as_str().to_string(),
                            market: market.clone(),
                        },
                        market,
                        result.signal,
                        result.confidence,
                    ).to_platforms(&self.rules.default_platforms));
                }
            }
            prev_signals.insert(market.clone(), result.signal);
        }

        // Check volatility regime change
        if self.rules.alert_on_volatility_change {
            let mut prev_vol = self.previous_volatility.lock().await;
            if let Some(prev) = prev_vol.get(market) {
                if *prev != result.stigmergy_board.volatility_regime {
                    alerts.push(Alert::new(
                        AlertSeverity::Info,
                        AlertReason::VolatilityRegimeChange {
                            from: format!("{:?}", prev),
                            to: format!("{:?}", result.stigmergy_board.volatility_regime),
                        },
                        market,
                        result.signal,
                        result.confidence,
                    ).to_platforms(&self.rules.default_platforms));
                }
            }
            prev_vol.insert(market.clone(), result.stigmergy_board.volatility_regime);
        }

        // Check liquidation risk change
        if self.rules.alert_on_liquidation_risk_change {
            let mut prev_liq = self.previous_liquidation_risk.lock().await;
            if let Some(prev) = prev_liq.get(market) {
                if *prev != result.stigmergy_board.liquidation_risk_level {
                    alerts.push(Alert::new(
                        AlertSeverity::Warning,
                        AlertReason::LiquidationRiskChange {
                            from: format!("{:?}", prev),
                            to: format!("{:?}", result.stigmergy_board.liquidation_risk_level),
                        },
                        market,
                        result.signal,
                        result.confidence,
                    ).to_platforms(&self.rules.default_platforms));
                }
            }
            prev_liq.insert(market.clone(), result.stigmergy_board.liquidation_risk_level);
        }

        // Update last alert time
        if !alerts.is_empty() {
            self.last_alert_time.lock().await.insert(market.clone(), Instant::now());
        }

        alerts
    }

    /// Reset state for a market (for testing).
    pub async fn reset(&self, market: &str) {
        self.last_alert_time.lock().await.remove(market);
        self.previous_signals.lock().await.remove(market);
        self.previous_volatility.lock().await.remove(market);
        self.previous_liquidation_risk.lock().await.remove(market);
    }
}

/// Send an alert to Telegram.
pub async fn send_telegram_alert(config: &TelegramConfig, alert: &Alert) -> anyhow::Result<()> {
    if config.bot_token.is_empty() || config.chat_id.is_empty() {
        anyhow::bail!("Telegram config incomplete: bot_token or chat_id is empty");
    }
    if !telegram_token_is_safe(&config.bot_token) {
        anyhow::bail!("Telegram bot token contains invalid characters");
    }

    let url = format!(
        "https://api.telegram.org/bot{}/sendMessage",
        config.bot_token
    );
    let url = validate_alert_url(&url, &[])?;
    let payload = serde_json::json!({
        "chat_id": config.chat_id,
        "text": alert.to_text(),
        "parse_mode": config.parse_mode,
    });

    let client = alert_http_client()?;
    let resp = client.post(url).json(&payload).send().await?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await?;
        anyhow::bail!("Telegram API error {}: {}", status, body);
    }

    Ok(())
}

/// Send an alert to Discord via webhook.
pub async fn send_discord_alert(config: &DiscordConfig, alert: &Alert) -> anyhow::Result<()> {
    if config.webhook_url.is_empty() {
        anyhow::bail!("Discord webhook URL is empty");
    }
    let url = validate_alert_url(&config.webhook_url, &[])?;

    let mut payload = alert.to_discord_embed();
    payload["username"] = serde_json::json!(config.username);

    let client = alert_http_client()?;
    let resp = client.post(url).json(&payload).send().await?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await?;
        anyhow::bail!("Discord webhook error {}: {}", status, body);
    }

    Ok(())
}

/// Send an alert to a generic webhook.
pub async fn send_webhook_alert(config: &WebhookConfig, alert: &Alert) -> anyhow::Result<()> {
    if config.url.is_empty() {
        anyhow::bail!("Webhook URL is empty");
    }
    let url = validate_alert_url(&config.url, &config.allowed_hosts)?;

    let body = serde_json::to_string(alert)?;
    let mut builder = alert_http_client()?.post(url);
    builder = builder.header("Content-Type", "application/json");
    for (key, value) in &config.headers {
        builder = builder.header(key.as_str(), value.as_str());
    }

    let resp = builder.body(body).send().await?;
    let status = resp.status();
    if !status.is_success() {
        let resp_body = resp.text().await?;
        anyhow::bail!("Webhook error {}: {}", status, resp_body);
    }

    Ok(())
}

/// Format a consensus result as a summary message for chat platforms.
pub fn format_consensus_summary(result: &ConsensusResult) -> String {
    let mut lines = Vec::new();
    lines.push(format!("\u{1F4CA} *SwarmFi Consensus Update*"));
    lines.push(format!("Market: *{}*", result.market));
    lines.push(format!("Signal: *{}* ({:.1}%)", result.signal.as_str(), result.confidence));
    lines.push(String::new());
    lines.push("_Agent Votes:_".to_string());
    for vote in &result.agent_votes {
        lines.push(format!(
            "  {} {} ({:.0}%)",
            vote.signal.as_str(),
            vote.agent_type,
            vote.confidence
        ));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_consensus(market: &str, signal: Signal, confidence: f64) -> ConsensusResult {
        ConsensusResult {
            market: market.to_string(),
            signal,
            confidence,
            agent_votes: vec![AgentVote {
                agent_type: "TestAgent".into(),
                signal,
                confidence,
                reasoning: "test".into(),
            }],
            timestamp: chrono::Utc::now().timestamp_millis(),
            stigmergy_board: StigmergyBoard::default(),
        }
    }

    fn make_consensus_with_volatility(
        market: &str,
        signal: Signal,
        confidence: f64,
        vol: VolatilityRegime,
        liq: RiskLevel,
    ) -> ConsensusResult {
        let mut board = StigmergyBoard::default();
        board.volatility_regime = vol;
        board.liquidation_risk_level = liq;
        ConsensusResult {
            market: market.to_string(),
            signal,
            confidence,
            agent_votes: vec![],
            timestamp: chrono::Utc::now().timestamp_millis(),
            stigmergy_board: board,
        }
    }

    #[tokio::test]
    async fn test_high_confidence_alert() {
        let rules = AlertRules {
            high_confidence_threshold: 70.0,
            cooldown_secs: 0,
            ..Default::default()
        };
        let manager = AlertManager::new(rules);
        let result = make_consensus("BTC-USD", Signal::Long, 85.0);

        let alerts = manager.evaluate(&result).await;
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, AlertSeverity::Critical);
    }

    #[tokio::test]
    async fn test_low_confidence_alert() {
        let rules = AlertRules {
            low_confidence_threshold: 25.0,
            cooldown_secs: 0,
            ..Default::default()
        };
        let manager = AlertManager::new(rules);
        let result = make_consensus("BTC-USD", Signal::Neutral, 15.0);

        let alerts = manager.evaluate(&result).await;
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, AlertSeverity::Warning);
    }

    #[tokio::test]
    async fn test_signal_flip_alert() {
        let rules = AlertRules {
            alert_on_signal_flip: true,
            cooldown_secs: 0,
            ..Default::default()
        };
        let manager = AlertManager::new(rules);

        // First: LONG
        let r1 = make_consensus("BTC-USD", Signal::Long, 50.0);
        let a1 = manager.evaluate(&r1).await;
        assert!(a1.is_empty()); // No flip on first signal

        // Second: SHORT (flip from LONG)
        let r2 = make_consensus("BTC-USD", Signal::Short, 50.0);
        let a2 = manager.evaluate(&r2).await;
        assert_eq!(a2.len(), 1);
        match &a2[0].reason {
            AlertReason::SignalFlip { from, to, .. } => {
                assert_eq!(from, "LONG");
                assert_eq!(to, "SHORT");
            }
            _ => panic!("Expected SignalFlip"),
        }
    }

    #[tokio::test]
    async fn test_volatility_change_alert() {
        let rules = AlertRules {
            alert_on_volatility_change: true,
            cooldown_secs: 0,
            ..Default::default()
        };
        let manager = AlertManager::new(rules);

        // First: NORMAL
        let r1 = make_consensus_with_volatility(
            "ETH-USD", Signal::Neutral, 50.0,
            VolatilityRegime::Normal, RiskLevel::Low,
        );
        let _ = manager.evaluate(&r1).await;

        // Second: HIGH (change detected)
        let r2 = make_consensus_with_volatility(
            "ETH-USD", Signal::Neutral, 50.0,
            VolatilityRegime::High, RiskLevel::Low,
        );
        let alerts = manager.evaluate(&r2).await;
        assert_eq!(alerts.len(), 1);
        match &alerts[0].reason {
            AlertReason::VolatilityRegimeChange { from, to } => {
                assert!(from.contains("Normal"));
                assert!(to.contains("High"));
            }
            _ => panic!("Expected VolatilityRegimeChange"),
        }
    }

    #[tokio::test]
    async fn test_cooldown_prevents_spam() {
        let rules = AlertRules {
            high_confidence_threshold: 70.0,
            cooldown_secs: 300,
            ..Default::default()
        };
        let manager = AlertManager::new(rules);

        let result = make_consensus("BTC-USD", Signal::Long, 85.0);
        let a1 = manager.evaluate(&result).await;
        assert_eq!(a1.len(), 1);

        // Second call within cooldown should return empty
        let a2 = manager.evaluate(&result).await;
        assert!(a2.is_empty());
    }

    #[tokio::test]
    async fn test_cooldown_different_markets() {
        let rules = AlertRules {
            high_confidence_threshold: 70.0,
            cooldown_secs: 300,
            ..Default::default()
        };
        let manager = AlertManager::new(rules);

        let r1 = make_consensus("BTC-USD", Signal::Long, 85.0);
        let r2 = make_consensus("ETH-USD", Signal::Short, 85.0);

        let a1 = manager.evaluate(&r1).await;
        let a2 = manager.evaluate(&r2).await;

        assert_eq!(a1.len(), 1);
        assert_eq!(a2.len(), 1); // Different market, no cooldown
    }

    #[test]
    fn test_alert_text_formatting() {
        let alert = Alert::new(
            AlertSeverity::Critical,
            AlertReason::HighConfidence { value: 80.0, threshold: 75.0 },
            "BTC-USD",
            Signal::Long,
            80.0,
        );
        let text = alert.to_text();
        assert!(text.contains("BTC-USD"));
        assert!(text.contains("LONG"));
        assert!(text.contains("80.0%"));
    }

    #[test]
    fn test_alert_discord_embed() {
        let alert = Alert::new(
            AlertSeverity::Warning,
            AlertReason::Custom { message: "Test alert".into() },
            "ETH-USD",
            Signal::Short,
            60.0,
        );
        let embed = alert.to_discord_embed();
        assert!(embed["embeds"][0]["title"].as_str().unwrap().contains("WARNING"));
        assert_eq!(embed["embeds"][0]["color"], AlertSeverity::Warning.discord_color());
    }

    #[test]
    fn test_alert_serde() {
        let alert = Alert::new(
            AlertSeverity::Info,
            AlertReason::VolatilityRegimeChange {
                from: "NORMAL".into(),
                to: "HIGH".into(),
            },
            "SOL-USD",
            Signal::Neutral,
            40.0,
        );
        let json = serde_json::to_string(&alert).unwrap();
        let restored: Alert = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.market, "SOL-USD");
        assert_eq!(restored.severity, AlertSeverity::Info);
    }

    #[test]
    fn test_telegram_config_missing_throws() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = TelegramConfig::default(); // empty token
        let alert = Alert::new(
            AlertSeverity::Info,
            AlertReason::Custom { message: "test".into() },
            "BTC", Signal::Long, 50.0,
        );
        let result = rt.block_on(send_telegram_alert(&config, &alert));
        assert!(result.is_err());
    }

    #[test]
    fn test_discord_config_missing_throws() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = DiscordConfig::default(); // empty URL
        let alert = Alert::new(
            AlertSeverity::Info,
            AlertReason::Custom { message: "test".into() },
            "BTC", Signal::Long, 50.0,
        );
        let result = rt.block_on(send_discord_alert(&config, &alert));
        assert!(result.is_err());
    }

    #[test]
    fn test_format_consensus_summary() {
        let result = make_consensus("BTC-USD", Signal::Long, 72.0);
        let summary = format_consensus_summary(&result);
        assert!(summary.contains("SwarmFi"));
        assert!(summary.contains("BTC-USD"));
        assert!(summary.contains("LONG"));
    }

    #[test]
    fn test_alert_severity_helpers() {
        assert_eq!(AlertSeverity::Info.as_str(), "INFO");
        assert_eq!(AlertSeverity::Critical.discord_color(), 0xE74C3C);
    }

    #[test]
    fn test_alert_platform_as_str() {
        assert_eq!(AlertPlatform::Telegram.as_str(), "TELEGRAM");
        assert_eq!(AlertPlatform::Discord.as_str(), "DISCORD");
    }

    #[test]
    fn test_ssrf_rejects_non_https() {
        for raw in [
            "http://discord.com/api/webhooks/1/abc",
            "file:///etc/passwd",
            "ftp://discord.com/x",
        ] {
            let err = validate_alert_url_with_resolve(raw, &[], false).unwrap_err();
            assert!(
                err.to_string().contains("https") || err.to_string().contains("invalid"),
                "{raw}: {err}"
            );
        }
    }

    #[test]
    fn test_ssrf_rejects_metadata_and_link_local() {
        for raw in [
            "https://169.254.169.254/latest/meta-data",
            "https://metadata.google.internal/",
            "https://localhost/hook",
            "https://127.0.0.1/hook",
            "https://[::1]/hook",
            "https://10.0.0.5/hook",
            "https://192.168.1.10/hook",
            "https://172.16.0.2/hook",
            "https://[fe80::1]/hook",
            "https://100.100.100.200/latest/meta-data",
        ] {
            let err = validate_alert_url_with_resolve(raw, &[], false).unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains("blocked") || msg.contains("allowlist"),
                "{raw}: {msg}"
            );
        }
    }

    #[test]
    fn test_ssrf_rejects_unknown_host() {
        let err = validate_alert_url_with_resolve(
            "https://evil.example/steal",
            &[],
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("allowlist"));
    }

    #[test]
    fn test_ssrf_allows_discord_and_telegram_hosts() {
        for raw in [
            "https://discord.com/api/webhooks/1/abc",
            "https://discordapp.com/api/webhooks/1/abc",
            "https://api.telegram.org/bot123:ABC/sendMessage",
        ] {
            let url = validate_alert_url_with_resolve(raw, &[], false).unwrap();
            assert_eq!(url.scheme(), "https");
        }
    }

    #[test]
    fn test_ssrf_extra_allowlist() {
        let extra = vec!["hooks.example.com".into()];
        let url = validate_alert_url_with_resolve(
            "https://hooks.example.com/alert",
            &extra,
            false,
        )
        .unwrap();
        assert_eq!(url.host_str(), Some("hooks.example.com"));
        let err = validate_alert_url_with_resolve(
            "https://hooks.example.com/alert",
            &[],
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("allowlist"));
    }

    #[test]
    fn test_ssrf_rejects_credentials() {
        let err = validate_alert_url_with_resolve(
            "https://user:pass@discord.com/api/webhooks/1/abc",
            &[],
            false,
        )
        .unwrap_err();
        assert!(err.to_string().contains("credentials"));
    }

    #[test]
    fn test_blocked_ip_helpers() {
        assert!(is_blocked_ip("127.0.0.1".parse().unwrap()));
        assert!(is_blocked_ip("169.254.169.254".parse().unwrap()));
        assert!(is_blocked_ip("10.1.2.3".parse().unwrap()));
        assert!(is_blocked_ip("192.168.0.1".parse().unwrap()));
        assert!(is_blocked_ip("::1".parse().unwrap()));
        assert!(is_blocked_ip("fe80::1".parse().unwrap()));
        assert!(!is_blocked_ip("8.8.8.8".parse().unwrap()));
    }

    #[test]
    fn test_discord_rejects_ssrf_url() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = DiscordConfig {
            webhook_url: "https://169.254.169.254/latest/meta-data".into(),
            username: "test".into(),
        };
        let alert = Alert::new(
            AlertSeverity::Info,
            AlertReason::Custom { message: "test".into() },
            "BTC", Signal::Long, 50.0,
        );
        let result = rt.block_on(send_discord_alert(&config, &alert));
        assert!(result.is_err());
    }

    #[test]
    fn test_webhook_rejects_ssrf_url() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = WebhookConfig {
            url: "http://127.0.0.1/hook".into(),
            headers: HashMap::new(),
            allowed_hosts: vec![],
        };
        let alert = Alert::new(
            AlertSeverity::Info,
            AlertReason::Custom { message: "test".into() },
            "BTC", Signal::Long, 50.0,
        );
        let result = rt.block_on(send_webhook_alert(&config, &alert));
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_liquidation_risk_change_alert() {
        let rules = AlertRules {
            alert_on_liquidation_risk_change: true,
            cooldown_secs: 0,
            ..Default::default()
        };
        let manager = AlertManager::new(rules);

        let r1 = make_consensus_with_volatility(
            "BTC-USD", Signal::Long, 50.0,
            VolatilityRegime::Normal, RiskLevel::Low,
        );
        let _ = manager.evaluate(&r1).await;

        let r2 = make_consensus_with_volatility(
            "BTC-USD", Signal::Long, 50.0,
            VolatilityRegime::Normal, RiskLevel::High,
        );
        let alerts = manager.evaluate(&r2).await;
        assert_eq!(alerts.len(), 1);
        match &alerts[0].reason {
            AlertReason::LiquidationRiskChange { from, to } => {
                assert!(from.contains("Low"));
                assert!(to.contains("High"));
            }
            _ => panic!("Expected LiquidationRiskChange"),
        }
    }
}
