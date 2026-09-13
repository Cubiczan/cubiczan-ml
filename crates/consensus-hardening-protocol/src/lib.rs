//! Consensus Hardening Protocol (CHP) — Rust Port
//!
//! Decision governance layer for multi-agent AI systems.
//! Ported from the Python `consensus-hardening-protocol` package (13,853 LOC).
//!
//! This crate provides the core CHP state machine, gate logic, foundation
//! validation, devil's advocate construction, packet contracts, registry,
//! and orchestration — all memory-safe and zero-dependency-on-Python.

pub mod models;
pub mod gates;
pub mod foundation;
pub mod payloads;
pub mod rounds;
pub mod parity;
pub mod devil;
pub mod validators;
pub mod dossier;
pub mod registry;
pub(crate) mod safe_path;
pub mod contracts;
pub mod orchestrator;
pub mod context;
pub mod parser;

pub use models::*;
pub use gates::*;
pub use foundation::*;
pub use payloads::*;
pub use rounds::*;
pub use parity::*;
pub use devil::*;
pub use validators::*;
pub use dossier::*;
pub use registry::*;
pub use contracts::*;
pub use orchestrator::*;
pub use context::*;
pub use parser::*;

#[cfg(feature = "superserve")]
pub mod superserve_bridge;
