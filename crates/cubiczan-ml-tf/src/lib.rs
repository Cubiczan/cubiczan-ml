//! # CubicZan ML — TensorFlow Rust Bindings
//!
//! Provides TensorFlow integration for loading and running Python-trained models
//! in Rust, enabling seamless migration of existing TF/Keras pipelines into the
//! Cubiczan ML ecosystem.
//!
//! ## Modules
//!
//! - [`session`] — TF session management, model loading, batch inference
//! - [`bridge`] — Python-TF bridge for importing trained models
//! - [`models`] — Pre-built model interfaces for financial ML tasks

pub mod session;
pub mod bridge;
pub mod models;
pub(crate) mod safe_path;

pub use session::{TfSession, SessionConfig, SessionPool};
pub use bridge::{PyTfBridge, OnnxImporter};
pub use models::{TfLSTM, TfTransformer, TfClassifier, TfRiskModel};
