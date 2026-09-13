/// Configuration system for FinFlowRL experiments.
///
/// Provides a JSON-based configuration with dot-separated key access,
/// matching the original Python `Config` class.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use thiserror::Error;

use crate::safe_path::{resolve_under_allowed_bases, PathError};

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Key not found: {0}")]
    KeyNotFound(String),
    #[error("path is not allowed: {0}")]
    PathNotAllowed(String),
}

impl From<PathError> for ConfigError {
    fn from(e: PathError) -> Self {
        ConfigError::PathNotAllowed(e.0)
    }
}

/// A nested configuration value that can be a leaf (f64, string, etc.) or a sub-tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConfigValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Map(HashMap<String, ConfigValue>),
    Array(Vec<ConfigValue>),
}

impl ConfigValue {
    /// Get a nested value by dot-separated key path.
    pub fn get(&self, key: &str) -> Option<&ConfigValue> {
        let parts: Vec<&str> = key.split('.').collect();
        let mut current = self;
        for part in &parts {
            match current {
                ConfigValue::Map(map) => {
                    current = map.get(*part)?;
                }
                _ => return None,
            }
        }
        Some(current)
    }

    /// Set a nested value by dot-separated key path.
    pub fn set(&mut self, key: &str, value: ConfigValue) {
        let parts: Vec<&str> = key.split('.').collect();
        let mut current = self;
        for (i, part) in parts.iter().enumerate() {
            if i == parts.len() - 1 {
                // Last part — set the value
                if let ConfigValue::Map(map) = current {
                    map.insert(part.to_string(), value.clone());
                }
            } else {
                // Intermediate — ensure map exists
                if let ConfigValue::Map(map) = current {
                    if !map.contains_key(*part) {
                        map.insert(part.to_string(), ConfigValue::Map(HashMap::new()));
                    }
                    current = map.get_mut(*part).unwrap();
                }
            }
        }
    }

    /// Get as f64 if this is a number.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ConfigValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Get as i64 if this is a number.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            ConfigValue::Number(n) => Some(*n as i64),
            _ => None,
        }
    }

    /// Get as usize if this is a number.
    pub fn as_usize(&self) -> Option<usize> {
        self.as_i64().map(|v| v as usize)
    }

    /// Get as string reference if this is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ConfigValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Get as bool if this is a bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ConfigValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Get as array of f64 if this is an array of numbers.
    pub fn as_f64_vec(&self) -> Option<Vec<f64>> {
        match self {
            ConfigValue::Array(arr) => {
                let mut result = Vec::new();
                for v in arr {
                    if let ConfigValue::Number(n) = v {
                        result.push(*n);
                    } else {
                        return None;
                    }
                }
                Some(result)
            }
            _ => None,
        }
    }
}

/// YAML/JSON-based configuration for FinFlowRL experiments.
///
/// Mirrors the Python `Config` class with `get()`, `set()`, `save()`.
#[derive(Debug, Clone)]
pub struct Config {
    root: ConfigValue,
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

impl Config {
    /// Create a new config with default values matching the Python `_DEFAULT_CONFIG`.
    pub fn new() -> Self {
        let mut root = HashMap::new();

        // simulator
        let mut sim = HashMap::new();
        sim.insert("seed".to_string(), ConfigValue::Number(42.0));
        sim.insert("S0".to_string(), ConfigValue::Number(100.0));
        sim.insert("mu".to_string(), ConfigValue::Number(0.0));
        sim.insert("sigma".to_string(), ConfigValue::Number(0.02));
        sim.insert("jump_intensity".to_string(), ConfigValue::Number(0.1));
        sim.insert("jump_mean".to_string(), ConfigValue::Number(0.0));
        sim.insert("jump_std".to_string(), ConfigValue::Number(0.01));
        sim.insert("half_spread".to_string(), ConfigValue::Number(0.01));
        sim.insert("hawkes_mu".to_string(), ConfigValue::Number(5.0));
        sim.insert("hawkes_alpha".to_string(), ConfigValue::Number(2.0));
        sim.insert("hawkes_beta".to_string(), ConfigValue::Number(10.0));
        sim.insert("dt".to_string(), ConfigValue::Number(1.0));
        root.insert("simulator".to_string(), ConfigValue::Map(sim));

        // env
        let mut env = HashMap::new();
        env.insert("max_steps".to_string(), ConfigValue::Number(1000.0));
        env.insert("max_position".to_string(), ConfigValue::Number(10.0));
        env.insert("transaction_cost".to_string(), ConfigValue::Number(0.001));
        env.insert("inventory_penalty".to_string(), ConfigValue::Number(0.01));
        env.insert("reward_scale".to_string(), ConfigValue::Number(1.0));
        env.insert("obs_dim".to_string(), ConfigValue::Number(6.0));
        env.insert("act_dim".to_string(), ConfigValue::Number(1.0));
        root.insert("env".to_string(), ConfigValue::Map(env));

        // policy
        let mut pol = HashMap::new();
        pol.insert("obs_dim".to_string(), ConfigValue::Number(6.0));
        pol.insert("act_dim".to_string(), ConfigValue::Number(1.0));
        pol.insert("n_flow_steps".to_string(), ConfigValue::Number(10.0));
        pol.insert(
            "hidden_sizes".to_string(),
            ConfigValue::Array(vec![
                ConfigValue::Number(128.0),
                ConfigValue::Number(128.0),
                ConfigValue::Number(64.0),
            ]),
        );
        root.insert("policy".to_string(), ConfigValue::Map(pol));

        // pretrain
        let mut pt = HashMap::new();
        pt.insert("n_episodes".to_string(), ConfigValue::Number(100.0));
        pt.insert("steps_per_episode".to_string(), ConfigValue::Number(200.0));
        pt.insert("learning_rate".to_string(), ConfigValue::Number(1e-3));
        pt.insert("batch_size".to_string(), ConfigValue::Number(32.0));
        pt.insert("n_iterations".to_string(), ConfigValue::Number(1000.0));
        root.insert("pretrain".to_string(), ConfigValue::Map(pt));

        // finetune
        let mut ft = HashMap::new();
        ft.insert("n_episodes".to_string(), ConfigValue::Number(50.0));
        ft.insert("steps_per_episode".to_string(), ConfigValue::Number(500.0));
        ft.insert("n_epochs".to_string(), ConfigValue::Number(10.0));
        root.insert("finetune".to_string(), ConfigValue::Map(ft));

        // ppo
        let mut ppo = HashMap::new();
        ppo.insert(
            "hidden_sizes".to_string(),
            ConfigValue::Array(vec![
                ConfigValue::Number(64.0),
                ConfigValue::Number(64.0),
            ]),
        );
        ppo.insert("lr".to_string(), ConfigValue::Number(3e-4));
        ppo.insert("clip_ratio".to_string(), ConfigValue::Number(0.2));
        ppo.insert("gamma".to_string(), ConfigValue::Number(0.99));
        ppo.insert("lam".to_string(), ConfigValue::Number(0.95));
        root.insert("ppo".to_string(), ConfigValue::Map(ppo));

        // expert
        let mut expert = HashMap::new();
        expert.insert("type".to_string(), ConfigValue::String("glft".to_string()));
        expert.insert("gamma".to_string(), ConfigValue::Number(0.1));
        expert.insert("sigma".to_string(), ConfigValue::Number(0.02));
        root.insert("expert".to_string(), ConfigValue::Map(expert));

        Self {
            root: ConfigValue::Map(root),
        }
    }

    /// Load configuration from a JSON file.
    pub fn load(path: &str) -> Result<Self, ConfigError> {
        let path = resolve_under_allowed_bases(Path::new(path))?;
        let content = fs::read_to_string(&path)?;
        let root: ConfigValue = serde_json::from_str(&content)?;
        Ok(Self { root })
    }

    /// Save configuration to a JSON file.
    pub fn save(&self, path: &str) -> Result<(), ConfigError> {
        let path = resolve_under_allowed_bases(Path::new(path))?;
        let content = serde_json::to_string_pretty(&self.root)?;
        fs::write(&path, content)?;
        Ok(())
    }

    /// Get a configuration value by dot-separated key.
    /// Returns a reference to the `ConfigValue`.
    pub fn get(&self, key: &str) -> Option<&ConfigValue> {
        self.root.get(key)
    }

    /// Get a configuration value as f64, with a default.
    pub fn get_f64(&self, key: &str, default: f64) -> f64 {
        self.root
            .get(key)
            .and_then(|v| v.as_f64())
            .unwrap_or(default)
    }

    /// Get a configuration value as usize, with a default.
    pub fn get_usize(&self, key: &str, default: usize) -> usize {
        self.root
            .get(key)
            .and_then(|v| v.as_usize())
            .unwrap_or(default)
    }

    /// Set a configuration value by dot-separated key.
    pub fn set(&mut self, key: &str, value: ConfigValue) {
        self.root.set(key, value);
    }

    /// Return a reference to the root configuration value.
    pub fn raw(&self) -> &ConfigValue {
        &self.root
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default_creation() {
        let config = Config::new();
        assert_eq!(config.get_f64("simulator.seed", 0.0), 42.0);
        assert_eq!(config.get_f64("simulator.S0", 0.0), 100.0);
        assert_eq!(config.get_f64("simulator.sigma", 0.0), 0.02);
        assert_eq!(config.get_f64("env.max_steps", 0.0), 1000.0);
        assert_eq!(config.get_f64("env.obs_dim", 0.0), 6.0);
    }

    #[test]
    fn test_config_get_set() {
        let mut config = Config::new();
        assert_eq!(config.get_f64("simulator.seed", 0.0), 42.0);
        config.set("simulator.seed", ConfigValue::Number(99.0));
        assert_eq!(config.get_f64("simulator.seed", 0.0), 99.0);
    }

    #[test]
    fn test_config_missing_key() {
        let config = Config::new();
        assert!(config.get("nonexistent.key").is_none());
        assert_eq!(config.get_f64("nonexistent.key", 42.0), 42.0);
    }

    #[test]
    fn test_config_save_load_roundtrip() {
        let config = Config::new();
        let dir = std::env::temp_dir();
        let path = dir.join("finflowrl_test_config.json");
        let path_str = path.to_str().unwrap();

        config.save(path_str).unwrap();
        let loaded = Config::load(path_str).unwrap();
        assert_eq!(
            loaded.get_f64("simulator.seed", 0.0),
            config.get_f64("simulator.seed", 0.0)
        );
        assert_eq!(
            loaded.get_f64("ppo.gamma", 0.0),
            config.get_f64("ppo.gamma", 0.0)
        );

        let _ = fs::remove_file(path_str);
    }

    #[test]
    fn test_config_save_rejects_traversal() {
        let config = Config::new();
        let err = config.save("../escaped_finflowrl.json").unwrap_err();
        match err {
            ConfigError::PathNotAllowed(msg) => {
                assert!(msg.contains("traversal") || msg.contains("escapes") || msg.contains("outside"));
            }
            other => panic!("expected PathNotAllowed, got {other:?}"),
        }
        assert!(!Path::new("../escaped_finflowrl.json").exists());
    }

    #[test]
    fn test_config_load_rejects_traversal() {
        let err = Config::load("../../etc/passwd").unwrap_err();
        match err {
            ConfigError::PathNotAllowed(msg) => {
                assert!(msg.contains("traversal") || msg.contains("escapes") || msg.contains("outside"));
            }
            other => panic!("expected PathNotAllowed, got {other:?}"),
        }
    }

    #[test]
    fn test_config_load_rejects_etc_passwd() {
        let err = Config::load("/etc/passwd").unwrap_err();
        match err {
            ConfigError::PathNotAllowed(_) => {}
            other => panic!("expected PathNotAllowed, got {other:?}"),
        }
    }

    #[test]
    fn test_config_nested_access() {
        let config = Config::new();
        let ppo = config.get("ppo").unwrap();
        assert!(ppo.as_f64().is_none()); // It's a map
        assert!(config.get("ppo.gamma").unwrap().as_f64().is_some());
    }
}
