//! Registry for CHP decision cases.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use crate::models::*;
use crate::safe_path::resolve_under_allowed_bases;

fn meaningful_tokens(text: &str) -> HashSet<String> {
    let stop: HashSet<&str> = [
        "the", "and", "for", "with", "this", "that", "from", "into",
        "should", "would", "could", "team", "quarter", "new",
    ].iter().cloned().collect();

    let cleaned: String = text
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();

    cleaned
        .split_whitespace()
        .map(|chunk| chunk.trim_matches('-').trim_matches('_').to_lowercase())
        .filter(|t| t.len() >= 4 && !stop.contains(t.as_str()))
        .collect()
}

pub struct DecisionRegistry {
    cases: HashMap<String, DecisionCase>,
}

impl DecisionRegistry {
    pub fn new() -> Self {
        Self { cases: HashMap::new() }
    }

    pub fn add(&mut self, case: DecisionCase) {
        self.cases.insert(case.decision_id.clone(), case);
    }

    pub fn get(&self, decision_id: &str) -> Option<&DecisionCase> {
        self.cases.get(decision_id)
    }

    pub fn get_mut(&mut self, decision_id: &str) -> Option<&mut DecisionCase> {
        self.cases.get_mut(decision_id)
    }

    pub fn find_related(&self, text: &str) -> Vec<&DecisionCase> {
        let query = text.to_lowercase();
        let query_tokens = meaningful_tokens(&query);
        let mut hits = Vec::new();

        for case in self.cases.values() {
            // Direct match on title or domain
            if case.title.to_lowercase().contains(&query) || case.domain.to_lowercase().contains(&query) {
                hits.push(case);
                continue;
            }
            // Direct match on dossier core_problem
            if let Some(ref dossier) = case.dossier {
                if !dossier.core_problem.is_empty() && dossier.core_problem.to_lowercase().contains(&query) {
                    hits.push(case);
                    continue;
                }
            }
            // Token matching
            let haystacks: Vec<String> = {
                let mut h = vec![case.title.to_lowercase(), case.domain.to_lowercase()];
                if let Some(ref dossier) = case.dossier {
                    if !dossier.core_problem.is_empty() {
                        h.push(dossier.core_problem.to_lowercase());
                    }
                }
                h
            };
            if !query_tokens.is_empty() {
                let has_match = haystacks.iter().any(|haystack| {
                    let hay_tokens = meaningful_tokens(haystack);
                    hay_tokens.intersection(&query_tokens).count() >= 3
                });
                if has_match {
                    hits.push(case);
                }
            }
        }
        hits
    }

    pub fn locked(&self) -> Vec<&DecisionCase> {
        self.cases.values().filter(|c| c.status == SessionStatus::LOCKED).collect()
    }

    pub fn all(&self) -> Vec<&DecisionCase> {
        self.cases.values().collect()
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let path = resolve_under_allowed_bases(path)?;
        let mut data = HashMap::new();
        for (id, case) in &self.cases {
            let val = serde_json::to_value(case)?;
            data.insert(id.clone(), val);
        }
        let json = serde_json::to_string_pretty(&data)?;
        std::fs::write(&path, json)?;
        Ok(())
    }

    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let path = resolve_under_allowed_bases(path)?;
        if !path.exists() {
            return Ok(Self::new());
        }
        let raw = std::fs::read_to_string(&path)?;
        let map: HashMap<String, serde_json::Value> = serde_json::from_str(&raw)?;
        let mut registry = Self::new();
        for (id, case_data) in map {
            let case: DecisionCase = serde_json::from_value(case_data)?;
            registry.cases.insert(id, case);
        }
        Ok(registry)
    }
}

impl Default for DecisionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_get() {
        let mut reg = DecisionRegistry::new();
        let case = DecisionCase::new("dc-1", "Test Decision", "finance", "alice");
        reg.add(case);
        assert!(reg.get("dc-1").is_some());
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn test_find_related() {
        let mut reg = DecisionRegistry::new();
        let mut case = DecisionCase::new("dc-1", "Capital Allocation Plan", "capital_allocation", "alice");
        case.dossier = Some(Dossier {
            core_problem: "How to allocate capital across business units".into(),
            goal_state: vec!["g".into()],
            current_state: vec!["c".into()],
            constraints: vec!["x".into()],
            ..Default::default()
        });
        reg.add(case);
        let hits = reg.find_related("capital allocation");
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn test_locked() {
        let mut reg = DecisionRegistry::new();
        let mut case = DecisionCase::new("dc-1", "Test", "finance", "alice");
        case.status = SessionStatus::LOCKED;
        reg.add(case);
        assert_eq!(reg.locked().len(), 1);
    }

    #[test]
    fn test_save_load_round_trip() {
        let dir = std::env::temp_dir().join("chp_test_registry");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("test.json");

        let mut reg = DecisionRegistry::new();
        reg.add(DecisionCase::new("dc-1", "Test", "finance", "alice"));
        reg.save(&path).unwrap();

        let loaded = DecisionRegistry::load(&path).unwrap();
        assert!(loaded.get("dc-1").is_some());
        assert_eq!(loaded.get("dc-1").unwrap().title, "Test");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_save_rejects_traversal() {
        let mut reg = DecisionRegistry::new();
        reg.add(DecisionCase::new("dc-1", "Test", "finance", "alice"));
        let err = reg.save(Path::new("../escaped_registry.json")).unwrap_err();
        assert!(err.to_string().contains("traversal") || err.to_string().contains("escapes") || err.to_string().contains("outside"));
        assert!(!Path::new("../escaped_registry.json").exists());
    }

    #[test]
    fn test_load_rejects_traversal() {
        match DecisionRegistry::load(Path::new("../../etc/passwd")) {
            Ok(_) => panic!("expected path traversal to be rejected"),
            Err(err) => {
                let msg = err.to_string();
                assert!(msg.contains("traversal") || msg.contains("escapes") || msg.contains("outside"));
            }
        }
    }

    #[test]
    fn test_load_rejects_etc_passwd() {
        match DecisionRegistry::load(Path::new("/etc/passwd")) {
            Ok(_) => panic!("expected /etc/passwd to be rejected"),
            Err(err) => {
                let msg = err.to_string();
                assert!(msg.contains("outside") || msg.contains("escapes") || msg.contains("traversal"));
            }
        }
    }

    #[test]
    fn test_meaningful_tokens() {
        let tokens = meaningful_tokens("the quick brown fox jumps over the lazy dog");
        assert!(tokens.contains("quick"));
        assert!(tokens.contains("brown"));
        assert!(tokens.contains("jumps"));
        assert!(!tokens.contains("the")); // stop word
        assert!(!tokens.contains("fox")); // too short
    }
}
