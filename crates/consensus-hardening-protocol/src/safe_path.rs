//! Path sanitization for filesystem open/read/write.
//!
//! Rejects `..` and null bytes, then resolves the path (canonicalizing when
//! the target exists, otherwise canonicalizing the nearest existing ancestor)
//! and requires the result to sit under an allowed base directory.

use std::path::{Component, Path, PathBuf};

/// Extra allowed filesystem root (absolute path). Relative paths use this
/// instead of the process current directory when the variable is set and
/// points at an existing directory.
pub const FS_BASE_ENV: &str = "CUBICZAN_FS_BASE";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathError(pub String);

impl std::fmt::Display for PathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for PathError {}

/// Reject `..` components / substrings and embedded NUL bytes.
pub fn reject_traversal(path: &Path) -> Result<(), PathError> {
    let raw = path.to_string_lossy();
    if raw.contains('\0') {
        return Err(PathError("path contains null byte".into()));
    }
    if raw.contains("..") {
        return Err(PathError(format!("path traversal rejected: {raw}")));
    }
    if path.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(PathError(format!("path traversal rejected: {raw}")));
    }
    Ok(())
}

/// Allowed roots: `CUBICZAN_FS_BASE` (if set), current directory, and temp dir.
pub fn allowed_bases() -> Result<Vec<PathBuf>, PathError> {
    let mut bases = Vec::new();
    if let Ok(extra) = std::env::var(FS_BASE_ENV) {
        let p = PathBuf::from(extra);
        if p.is_absolute() {
            bases.push(p);
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        bases.push(cwd);
    }
    bases.push(std::env::temp_dir());
    Ok(bases)
}

fn primary_base() -> Result<PathBuf, PathError> {
    if let Ok(extra) = std::env::var(FS_BASE_ENV) {
        let p = PathBuf::from(extra);
        if p.is_absolute() && p.is_dir() {
            return Ok(p);
        }
    }
    std::env::current_dir().map_err(|e| PathError(e.to_string()))
}

fn resolve_existing_or_new(path: &Path) -> Result<PathBuf, PathError> {
    if path.exists() {
        return path
            .canonicalize()
            .map_err(|e| PathError(format!("canonicalize {}: {e}", path.display())));
    }

    let mut suffix: Vec<std::ffi::OsString> = Vec::new();
    let mut current = path.to_path_buf();
    loop {
        match current.file_name() {
            Some(name) => suffix.push(name.to_os_string()),
            None => {
                return Err(PathError(format!(
                    "cannot resolve path {}",
                    path.display()
                )))
            }
        }
        match current.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => {
                if parent.exists() {
                    let mut resolved = parent.canonicalize().map_err(|e| {
                        PathError(format!("canonicalize {}: {e}", parent.display()))
                    })?;
                    for part in suffix.iter().rev() {
                        resolved.push(part);
                    }
                    return Ok(resolved);
                }
                current = parent.to_path_buf();
            }
            _ => {
                return Err(PathError(format!(
                    "cannot resolve path {}",
                    path.display()
                )))
            }
        }
    }
}

fn is_under_base(resolved: &Path, base: &Path) -> bool {
    let Ok(base_canon) = base.canonicalize() else {
        return false;
    };
    resolved.starts_with(&base_canon)
}

/// Resolve `path` under `base`: reject `..`, canonicalize, require `starts_with(base)`.
pub fn resolve_under_base(path: &Path, base: &Path) -> Result<PathBuf, PathError> {
    reject_traversal(path)?;
    reject_traversal(base)?;

    if !base.exists() {
        return Err(PathError(format!(
            "base directory does not exist: {}",
            base.display()
        )));
    }
    let base_canon = base
        .canonicalize()
        .map_err(|e| PathError(format!("canonicalize {}: {e}", base.display())))?;

    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_canon.join(path)
    };
    reject_traversal(&candidate)?;

    let resolved = resolve_existing_or_new(&candidate)?;
    if !resolved.starts_with(&base_canon) {
        return Err(PathError(format!(
            "path {} escapes allowed base {}",
            resolved.display(),
            base_canon.display()
        )));
    }
    Ok(resolved)
}

/// Resolve `path` so it cannot escape the process-allowed filesystem roots.
///
/// Relative paths are joined onto the primary base (`CUBICZAN_FS_BASE` or cwd).
/// Absolute paths must canonicalize under cwd, the temp directory, or
/// `CUBICZAN_FS_BASE`.
pub fn resolve_under_allowed_bases(path: &Path) -> Result<PathBuf, PathError> {
    reject_traversal(path)?;

    if path.is_absolute() {
        let resolved = resolve_existing_or_new(path)?;
        let bases = allowed_bases()?;
        if bases.iter().any(|base| is_under_base(&resolved, base)) {
            return Ok(resolved);
        }
        return Err(PathError(format!(
            "path {} is outside allowed directories",
            resolved.display()
        )));
    }

    resolve_under_base(path, &primary_base()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::os::unix::fs::symlink;

    fn jail() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "cz_safe_path_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn rejects_parent_dir_components() {
        assert!(reject_traversal(Path::new("../etc/passwd")).is_err());
        assert!(reject_traversal(Path::new("foo/../../etc/passwd")).is_err());
        assert!(reject_traversal(Path::new("/tmp/foo/../secret")).is_err());
        assert!(reject_traversal(Path::new("..")).is_err());
    }

    #[test]
    fn resolve_rejects_relative_traversal() {
        let dir = jail();
        assert!(resolve_under_base(Path::new("../outside.json"), &dir).is_err());
        assert!(resolve_under_base(Path::new("a/../../etc/passwd"), &dir).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_rejects_absolute_escape() {
        let dir = jail();
        assert!(resolve_under_base(Path::new("/etc/passwd"), &dir).is_err());
        assert!(resolve_under_allowed_bases(Path::new("/etc/passwd")).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_allows_path_under_base() {
        let dir = jail();
        let target = dir.join("ok.json");
        fs::write(&target, "{}").unwrap();
        let resolved = resolve_under_base(Path::new("ok.json"), &dir).unwrap();
        assert!(resolved.starts_with(dir.canonicalize().unwrap()));
        assert_eq!(resolved, target.canonicalize().unwrap());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_allows_new_file_under_temp() {
        let dir = jail();
        let target = dir.join("new_file.json");
        let resolved = resolve_under_allowed_bases(&target).unwrap();
        assert!(resolved.starts_with(dir.canonicalize().unwrap()));
        assert!(!resolved.exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolve_rejects_symlink_escape() {
        let dir = jail();
        let link = dir.join("escape");
        let _ = symlink("/etc/passwd", &link);
        if link.exists() {
            assert!(resolve_under_base(&link, &dir).is_err());
            assert!(resolve_under_allowed_bases(&link).is_err());
        }
        let _ = fs::remove_dir_all(&dir);
    }
}
