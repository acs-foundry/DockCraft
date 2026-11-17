#![forbid(unsafe_code)]

use std::fs::File;
use std::io::{Read, Write};
use std::os::fd::{AsFd, OwnedFd};
use std::path::{Component, Path, PathBuf};

use rustix::fs::{Mode, OFlags, ResolveFlags, fstat, mkdirat, openat, openat2};
use rustix::io::Errno;
use thiserror::Error;

pub const DEFAULT_MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;
pub const MAX_ZIP_DECOMPRESSED_BYTES: u64 = 32 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum SandboxError {
    #[error("path rejected: {0}")]
    RejectedPath(String),
    #[error("root not configured for {0}")]
    RootUnconfigured(&'static str),
    #[error("file exceeds byte cap ({size} > {cap})")]
    SizeLimitExceeded { size: u64, cap: u64 },
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("OS confinement error: {0}")]
    Os(String),
}

#[derive(Debug, Clone)]
pub struct PathSandbox {
    read_root: Option<PathBuf>,
    write_root: Option<PathBuf>,
    max_read_bytes: u64,
}

impl PathSandbox {
    pub fn new(
        read_root: Option<PathBuf>,
        write_root: Option<PathBuf>,
        max_read_bytes: u64,
    ) -> Result<Self, SandboxError> {
        let rr = read_root.map(|p| std::fs::canonicalize(&p)).transpose()?;
        let wr = write_root.map(|p| std::fs::canonicalize(&p)).transpose()?;
        Ok(Self {
            read_root: rr,
            write_root: wr,
            max_read_bytes,
        })
    }

    #[must_use]
    pub fn read_root(&self) -> Option<&Path> {
        self.read_root.as_deref()
    }

    #[must_use]
    pub fn write_root(&self) -> Option<&Path> {
        self.write_root.as_deref()
    }

    fn relative_segments<'a>(
        &self,
        root: &Path,
        candidate: &'a Path,
    ) -> Result<Vec<&'a std::ffi::OsStr>, SandboxError> {
        let rel = if candidate.is_absolute() {
            candidate.strip_prefix(root).map_err(|_| {
                SandboxError::RejectedPath(format!(
                    "path '{}' is outside sandbox root '{}'",
                    candidate.display(),
                    root.display()
                ))
            })?
        } else {
            candidate
        };

        let mut segs = Vec::new();
        for comp in rel.components() {
            match comp {
                Component::Normal(s) => {
                    if s.to_string_lossy().contains('\0') {
                        return Err(SandboxError::RejectedPath("NUL byte in path".to_string()));
                    }
                    segs.push(s);
                }
                Component::CurDir => {}
                Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                    return Err(SandboxError::RejectedPath(format!(
                        "disallowed component in '{}'",
                        candidate.display()
                    )));
                }
            }
        }
        if segs.is_empty() {
            return Err(SandboxError::RejectedPath(
                "empty relative target".to_string(),
            ));
        }
        Ok(segs)
    }

    pub fn read_bytes(&self, candidate: &Path) -> Result<Vec<u8>, SandboxError> {
        self.read_bytes_impl(candidate, false)
    }

    pub fn read_string(&self, candidate: &Path) -> Result<String, SandboxError> {
        let bytes = self.read_bytes(candidate)?;
        String::from_utf8(bytes)
            .map_err(|e| SandboxError::RejectedPath(format!("invalid UTF-8 file: {e}")))
    }

    pub fn write_bytes(&self, candidate: &Path, data: &[u8]) -> Result<(), SandboxError> {
        self.write_bytes_impl(candidate, data, false)
    }

    pub fn write_string(&self, candidate: &Path, text: &str) -> Result<(), SandboxError> {
        self.write_bytes(candidate, text.as_bytes())
    }

    fn open_root_fd(root: &Path) -> Result<OwnedFd, SandboxError> {
        openat(
            rustix::fs::CWD,
            root,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(|e| SandboxError::Os(format!("open root '{}': {e}", root.display())))
    }

    pub(crate) fn read_bytes_impl(
        &self,
        candidate: &Path,
        force_fallback: bool,
    ) -> Result<Vec<u8>, SandboxError> {
        let root = self
            .read_root
            .as_deref()
            .ok_or(SandboxError::RootUnconfigured("read"))?;
        let segs = self.relative_segments(root, candidate)?;
        let root_fd = Self::open_root_fd(root)?;

        let rel_joined: PathBuf = segs.iter().collect();
        let oflags = OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW;

        let fd = if !force_fallback {
            match openat2(
                root_fd.as_fd(),
                &rel_joined,
                oflags,
                Mode::empty(),
                ResolveFlags::BENEATH | ResolveFlags::NO_SYMLINKS,
            ) {
                Ok(fd) => fd,
                Err(Errno::NOSYS) => Self::open_read_dirfd_descent(root_fd, &segs)?,
                Err(e) => return Err(SandboxError::Os(format!("openat2 read failed: {e}"))),
            }
        } else {
            Self::open_read_dirfd_descent(root_fd, &segs)?
        };

        let st = fstat(fd.as_fd()).map_err(|e| SandboxError::Os(format!("fstat failed: {e}")))?;
        let size = st.st_size.max(0) as u64;
        if size > self.max_read_bytes {
            return Err(SandboxError::SizeLimitExceeded {
                size,
                cap: self.max_read_bytes,
            });
        }

        let file = File::from(fd);
        let mut buf = Vec::with_capacity(size.min(self.max_read_bytes) as usize);
        file.take(self.max_read_bytes + 1).read_to_end(&mut buf)?;
        if buf.len() as u64 > self.max_read_bytes {
            return Err(SandboxError::SizeLimitExceeded {
                size: buf.len() as u64,
                cap: self.max_read_bytes,
            });
        }
        Ok(buf)
    }

    fn open_read_dirfd_descent(
        mut dir_fd: OwnedFd,
        segs: &[&std::ffi::OsStr],
    ) -> Result<OwnedFd, SandboxError> {
        let last = segs.len() - 1;
        for seg in &segs[..last] {
            dir_fd = openat(
                dir_fd.as_fd(),
                *seg,
                OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .map_err(|e| SandboxError::Os(format!("dirfd descent failed: {e}")))?;
        }
        openat(
            dir_fd.as_fd(),
            segs[last],
            OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(|e| SandboxError::Os(format!("leaf openat failed: {e}")))
    }

    pub(crate) fn write_bytes_impl(
        &self,
        candidate: &Path,
        data: &[u8],
        force_fallback: bool,
    ) -> Result<(), SandboxError> {
        let root = self
            .write_root
            .as_deref()
            .ok_or(SandboxError::RootUnconfigured("write"))?;
        let segs = self.relative_segments(root, candidate)?;
        let root_fd = Self::open_root_fd(root)?;
        let last = segs.len() - 1;

        let leaf_fd = if !force_fallback && segs.len() == 1 {
            match openat2(
                root_fd.as_fd(),
                segs[0],
                OFlags::WRONLY
                    | OFlags::CREATE
                    | OFlags::TRUNC
                    | OFlags::NOFOLLOW
                    | OFlags::CLOEXEC,
                Mode::from_raw_mode(0o644),
                ResolveFlags::BENEATH | ResolveFlags::NO_SYMLINKS,
            ) {
                Ok(fd) => fd,
                Err(Errno::NOSYS) => {
                    Self::open_write_dirfd_descent(root_fd, &segs[..last], segs[last])?
                }
                Err(e) => return Err(SandboxError::Os(format!("openat2 write failed: {e}"))),
            }
        } else if !force_fallback {
            // Ensure intermediate directories exist safely with pinned dirfds first, then openat2 leaf or dirfd leaf
            let parent_fd = Self::ensure_dir_chain(root_fd, &segs[..last])?;
            match openat2(
                parent_fd.as_fd(),
                segs[last],
                OFlags::WRONLY
                    | OFlags::CREATE
                    | OFlags::TRUNC
                    | OFlags::NOFOLLOW
                    | OFlags::CLOEXEC,
                Mode::from_raw_mode(0o644),
                ResolveFlags::BENEATH | ResolveFlags::NO_SYMLINKS,
            ) {
                Ok(fd) => fd,
                Err(Errno::NOSYS) => openat(
                    parent_fd.as_fd(),
                    segs[last],
                    OFlags::WRONLY
                        | OFlags::CREATE
                        | OFlags::TRUNC
                        | OFlags::NOFOLLOW
                        | OFlags::CLOEXEC,
                    Mode::from_raw_mode(0o644),
                )
                .map_err(|e| SandboxError::Os(format!("write openat failed: {e}")))?,
                Err(e) => return Err(SandboxError::Os(format!("openat2 leaf write failed: {e}"))),
            }
        } else {
            Self::open_write_dirfd_descent(root_fd, &segs[..last], segs[last])?
        };

        let mut file = File::from(leaf_fd);
        file.write_all(data)?;
        file.flush()?;
        Ok(())
    }

    fn ensure_dir_chain(
        mut dir_fd: OwnedFd,
        parents: &[&std::ffi::OsStr],
    ) -> Result<OwnedFd, SandboxError> {
        for seg in parents {
            match mkdirat(dir_fd.as_fd(), *seg, Mode::from_raw_mode(0o755)) {
                Ok(()) | Err(Errno::EXIST) => {}
                Err(e) => return Err(SandboxError::Os(format!("mkdirat failed: {e}"))),
            }
            dir_fd = openat(
                dir_fd.as_fd(),
                *seg,
                OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .map_err(|e| SandboxError::Os(format!("dirfd descend failed: {e}")))?;
        }
        Ok(dir_fd)
    }

    fn open_write_dirfd_descent(
        root_fd: OwnedFd,
        parents: &[&std::ffi::OsStr],
        leaf: &std::ffi::OsStr,
    ) -> Result<OwnedFd, SandboxError> {
        let dir_fd = Self::ensure_dir_chain(root_fd, parents)?;
        openat(
            dir_fd.as_fd(),
            leaf,
            OFlags::WRONLY | OFlags::CREATE | OFlags::TRUNC | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::from_raw_mode(0o644),
        )
        .map_err(|e| SandboxError::Os(format!("leaf write openat failed: {e}")))
    }
}

/// Pure-Rust ZIP + JSON reader with hard `MAX_ZIP_DECOMPRESSED_BYTES` (32 MiB) cap against zip bombs.
pub fn read_bounded_chemcraft_entries(raw: &[u8]) -> Result<Vec<(String, Vec<u8>)>, SandboxError> {
    if raw.len() >= 4 && &raw[..4] == b"PK\x03\x04" {
        parse_bounded_zip(raw, MAX_ZIP_DECOMPRESSED_BYTES)
    } else if raw.len() >= 2 && raw[0] == 0x1f && raw[1] == 0x8b {
        let dec = flate2::read::GzDecoder::new(raw);
        let mut out = Vec::new();
        dec.take(MAX_ZIP_DECOMPRESSED_BYTES + 1)
            .read_to_end(&mut out)?;
        if out.len() as u64 > MAX_ZIP_DECOMPRESSED_BYTES {
            return Err(SandboxError::SizeLimitExceeded {
                size: out.len() as u64,
                cap: MAX_ZIP_DECOMPRESSED_BYTES,
            });
        }
        Ok(vec![("payload.json".to_string(), out)])
    } else {
        if raw.len() as u64 > MAX_ZIP_DECOMPRESSED_BYTES {
            return Err(SandboxError::SizeLimitExceeded {
                size: raw.len() as u64,
                cap: MAX_ZIP_DECOMPRESSED_BYTES,
            });
        }
        Ok(vec![("payload.json".to_string(), raw.to_vec())])
    }
}

fn parse_bounded_zip(data: &[u8], cap: u64) -> Result<Vec<(String, Vec<u8>)>, SandboxError> {
    let mut pos = 0usize;
    let mut total_unpacked = 0u64;
    let mut entries = Vec::new();

    while pos + 30 <= data.len() {
        if &data[pos..pos + 4] != b"PK\x03\x04" {
            break;
        }
        let flags = u16::from_le_bytes([data[pos + 6], data[pos + 7]]);
        let method = u16::from_le_bytes([data[pos + 8], data[pos + 9]]);
        let comp_size = u32::from_le_bytes([
            data[pos + 18],
            data[pos + 19],
            data[pos + 20],
            data[pos + 21],
        ]) as usize;
        let uncomp_size = u32::from_le_bytes([
            data[pos + 22],
            data[pos + 23],
            data[pos + 24],
            data[pos + 25],
        ]) as u64;
        let name_len = u16::from_le_bytes([data[pos + 26], data[pos + 27]]) as usize;
        let extra_len = u16::from_le_bytes([data[pos + 28], data[pos + 29]]) as usize;

        if (flags & 0x08) != 0 {
            return Err(SandboxError::RejectedPath(
                "streaming data-descriptor zip entries rejected for bounded safety".to_string(),
            ));
        }
        let header_end = pos + 30 + name_len + extra_len;
        if header_end + comp_size > data.len() {
            return Err(SandboxError::RejectedPath(
                "truncated zip entry".to_string(),
            ));
        }
        if total_unpacked.saturating_add(uncomp_size) > cap {
            return Err(SandboxError::SizeLimitExceeded {
                size: total_unpacked.saturating_add(uncomp_size),
                cap,
            });
        }

        let name = String::from_utf8_lossy(&data[pos + 30..pos + 30 + name_len]).to_string();
        let compressed = &data[header_end..header_end + comp_size];

        let remaining = cap - total_unpacked;
        let mut unpacked = Vec::new();
        match method {
            0 => {
                if comp_size as u64 > remaining {
                    return Err(SandboxError::SizeLimitExceeded {
                        size: total_unpacked + comp_size as u64,
                        cap,
                    });
                }
                unpacked.extend_from_slice(compressed);
            }
            8 => {
                let mut dec = flate2::read::DeflateDecoder::new(compressed);
                dec.by_ref()
                    .take(remaining + 1)
                    .read_to_end(&mut unpacked)?;
                if unpacked.len() as u64 > remaining {
                    return Err(SandboxError::SizeLimitExceeded {
                        size: total_unpacked + unpacked.len() as u64,
                        cap,
                    });
                }
            }
            other => {
                return Err(SandboxError::RejectedPath(format!(
                    "unsupported zip compression method {other}"
                )));
            }
        }
        total_unpacked += unpacked.len() as u64;
        entries.push((name, unpacked));
        pos = header_end + comp_size;
    }

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn rejects_zip_bomb_exceeding_32_mib_header_or_stream() -> Result<(), Box<dyn std::error::Error>>
    {
        // Compress 40 MiB of zeroes via raw DEFLATE wrapped in a minimal PK\x03\x04 local header claiming 100 bytes
        let zeros = vec![0u8; 40 * 1024 * 1024];
        let mut encoder =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::fast());
        encoder.write_all(&zeros)?;
        let deflated = encoder.finish()?;

        let name = b"project.json";
        let mut zip_bytes = Vec::new();
        zip_bytes.extend_from_slice(b"PK\x03\x04");
        zip_bytes.extend_from_slice(&20u16.to_le_bytes()); // version
        zip_bytes.extend_from_slice(&0u16.to_le_bytes()); // flags
        zip_bytes.extend_from_slice(&8u16.to_le_bytes()); // deflate
        zip_bytes.extend_from_slice(&0u32.to_le_bytes()); // time/date
        zip_bytes.extend_from_slice(&0u32.to_le_bytes()); // crc
        zip_bytes.extend_from_slice(&(deflated.len() as u32).to_le_bytes());
        zip_bytes.extend_from_slice(&100u32.to_le_bytes()); // deceptive uncompressed size!
        zip_bytes.extend_from_slice(&(name.len() as u16).to_le_bytes());
        zip_bytes.extend_from_slice(&0u16.to_le_bytes());
        zip_bytes.extend_from_slice(name);
        zip_bytes.extend_from_slice(&deflated);

        let err = read_bounded_chemcraft_entries(&zip_bytes)
            .err()
            .ok_or("expected zip bomb rejection")?;
        assert!(matches!(err, SandboxError::SizeLimitExceeded { .. }));
        Ok(())
    }

    #[test]
    fn openat2_and_dirfd_fallback_resist_4000_symlink_race_attempts()
    -> Result<(), Box<dyn std::error::Error>> {
        let base = std::env::temp_dir().join(format!("dockcraft_toctou_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let root = base.join("sandbox_root");
        let outside = base.join("outside_secret");
        std::fs::create_dir_all(root.join("sub"))?;
        std::fs::create_dir_all(&outside)?;
        std::fs::write(root.join("sub/safe.txt"), b"SAFE")?;
        std::fs::write(outside.join("secret.txt"), b"OUTSIDE_SECRET")?;

        let sandbox = PathSandbox::new(Some(root.clone()), Some(root.clone()), 1024 * 1024)?;
        let stop = Arc::new(AtomicBool::new(false));
        let stop_t = Arc::clone(&stop);
        let root_t = root.clone();
        let outside_t = outside.clone();

        // Racer flips root/race between a real directory and a symlink to outside_secret
        let racer = std::thread::spawn(move || {
            let link_path = root_t.join("race");
            let mut i = 0usize;
            while !stop_t.load(Ordering::Relaxed) {
                let _ = std::fs::remove_dir_all(&link_path);
                let _ = std::fs::remove_file(&link_path);
                if i.is_multiple_of(2) {
                    let _ = std::fs::create_dir_all(&link_path);
                    let _ = std::fs::write(link_path.join("secret.txt"), b"INSIDE");
                } else {
                    let _ = symlink(&outside_t, &link_path);
                }
                i += 1;
            }
        });

        for attempt in 0..4_000usize {
            let use_fallback = attempt % 2 == 1;
            if let Ok(bytes) = sandbox.read_bytes_impl(Path::new("race/secret.txt"), use_fallback) {
                assert_ne!(
                    bytes, b"OUTSIDE_SECRET",
                    "sandbox read escaped outside root on attempt {attempt} (fallback={use_fallback})"
                );
            }
            if sandbox
                .write_bytes_impl(Path::new("race/pwn.txt"), b"BAD", use_fallback)
                .is_ok()
            {
                assert!(
                    !outside.join("pwn.txt").exists(),
                    "sandbox write escaped outside root on attempt {attempt}"
                );
            }
        }

        stop.store(true, Ordering::Relaxed);
        let _ = racer.join();
        assert!(!outside.join("pwn.txt").exists());
        let _ = std::fs::remove_dir_all(&base);
        Ok(())
    }
}
