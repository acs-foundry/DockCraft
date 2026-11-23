#![forbid(unsafe_code)]

//! Bounded `.chemcraft` archive reading, on top of the shared confined filesystem access.
//!
//! The `PathSandbox` used here is the one published by `craftbio-sandbox`; this crate used to
//! carry a byte-for-byte equivalent copy of it, which meant a fix or a hardening on either side
//! did not reach the other. The copy is gone; what stays here is the archive reader, which is
//! domain logic rather than path confinement.

use std::io::Read;

pub use craftbio_sandbox::{PathSandbox, SandboxError};

/// Per-file read cap for this application.
///
/// **Deliberately not `craftbio_sandbox::DEFAULT_MAX_FILE_BYTES`**, which is 64 MiB. Receptor and
/// ligand inputs here are small, and the tighter cap is part of what the MCP server relies on;
/// taking the library default would silently relax it by a factor of 32.
pub const DEFAULT_MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;

pub const MAX_ZIP_DECOMPRESSED_BYTES: u64 = 32 * 1024 * 1024;

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
    use std::io::Write;
    use std::os::unix::fs::symlink;
    use std::path::Path;
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
    /// Note: this exercises the default resolution path only. The dirfd-descent fallback is
    /// internal to `craftbio-sandbox` now, and the same race is tested there against both paths.
    fn openat2_resists_4000_symlink_race_attempts() -> Result<(), Box<dyn std::error::Error>> {
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
            if let Ok(bytes) = sandbox.read_bytes(Path::new("race/secret.txt")) {
                assert_ne!(
                    bytes, b"OUTSIDE_SECRET",
                    "sandbox read escaped outside root on attempt {attempt}"
                );
            }
            if sandbox
                .write_bytes(Path::new("race/pwn.txt"), b"BAD")
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
