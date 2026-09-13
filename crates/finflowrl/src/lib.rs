pub mod config;
pub(crate) mod safe_path;
pub mod experts;
pub mod models;
pub mod envs;
pub mod simulator;
pub mod agents;
pub mod training;
pub mod evaluation;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
