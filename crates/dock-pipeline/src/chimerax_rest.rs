//! Live channel to a running UCSF ChimeraX, over its own REST remote control.
//!
//! ChimeraX exposes a loopback HTTP endpoint once `remotecontrol rest start port 60958 json true`
//! has been run inside it. This is a minimal HTTP/1.1 client for that endpoint and nothing else:
//! no TLS (the socket never leaves 127.0.0.1), no HTTP library, no redirects, no keep-alive.
//!
//! # Why the command vocabulary is closed
//!
//! A ChimeraX command is arbitrary code in ChimeraX's process: `runscript` executes Python, and
//! `save` writes wherever ChimeraX's own sandbox allows. Accepting a command string from a caller
//! — an MCP client, a GUI field, a library user — would therefore turn this crate into a remote
//! code execution primitive wearing a visualization hat.
//!
//! So there is no `run(&str)` in the public surface. [`ChimeraXCommand`] enumerates the commands
//! DockCraft needs, each one rendered here from typed parts, and every path it can name is checked
//! to live under the export root before the request is sent. Adding a capability means adding a
//! variant and thinking about it once, which is the point.

use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddrV4, TcpStream};
use std::path::{Path, PathBuf};
use std::time::Duration;

use thiserror::Error;

/// Port ChimeraX's `remotecontrol rest start` is expected on. Matches the Flow of Work's
/// `fw.json` chimerax section.
pub const DEFAULT_REST_PORT: u16 = 60958;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);
const IO_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum RestError {
    #[error(
        "ChimeraX is not answering on 127.0.0.1:{port}. Open it and run inside: \
         remotecontrol rest start port {port} json true"
    )]
    NotRunning { port: u16 },
    #[error("ChimeraX transport error: {0}")]
    Io(String),
    #[error("ChimeraX returned a malformed response: {0}")]
    Malformed(String),
    #[error("ChimeraX rejected the command: {0}")]
    Command(String),
    #[error("refusing to name a path outside the export root: {0}")]
    PathOutsideRoot(String),
}

/// What DockCraft is allowed to ask a live ChimeraX to do.
///
/// Deliberately small. Every variant renders to one command, built from typed parts here.
#[derive(Debug, Clone)]
pub enum ChimeraXCommand {
    /// `version` — liveness and version probe.
    Version,
    /// `open <scene>` — a `.cxc` this crate wrote, inside the export root.
    OpenScene(PathBuf),
    /// `close <spec>` for a model number, e.g. `close #2`.
    CloseModel(u32),
    /// `view <spec>` for a model number.
    ViewModel(u32),
    /// `save <path> format pdb selectedOnly true` — writes the current selection, and only it,
    /// into a file under the export root.
    SaveSelectionPdb(PathBuf),
    /// `info selection level residue` — reads back what the person has picked on screen.
    InfoSelection,
}

impl ChimeraXCommand {
    /// The path this command names, if any. Checked against the export root before sending.
    fn path(&self) -> Option<&Path> {
        match self {
            Self::OpenScene(p) | Self::SaveSelectionPdb(p) => Some(p.as_path()),
            _ => None,
        }
    }

    fn render(&self) -> String {
        match self {
            Self::Version => "version".to_string(),
            Self::OpenScene(p) => format!("open {}", p.display()),
            Self::CloseModel(id) => format!("close #{id}"),
            Self::ViewModel(id) => format!("view #{id}"),
            Self::SaveSelectionPdb(p) => {
                format!("save {} format pdb selectedOnly true", p.display())
            }
            Self::InfoSelection => "info selection level residue".to_string(),
        }
    }
}

/// A reply from ChimeraX, with its log split the way ChimeraX splits it.
#[derive(Debug, Clone, Default)]
pub struct ChimeraXReply {
    pub notes: Vec<String>,
    pub warnings: Vec<String>,
    pub json_values: Vec<String>,
}

/// Client for a ChimeraX REST endpoint on loopback.
#[derive(Debug, Clone)]
pub struct ChimeraXRest {
    port: u16,
    export_root: PathBuf,
}

impl ChimeraXRest {
    /// The host is not a parameter. It is fixed to 127.0.0.1 because the endpoint has no
    /// authentication of any kind: anything that can reach it drives ChimeraX.
    #[must_use]
    pub fn new(port: u16, export_root: PathBuf) -> Self {
        Self { port, export_root }
    }

    #[must_use]
    pub fn with_default_port(export_root: PathBuf) -> Self {
        Self::new(DEFAULT_REST_PORT, export_root)
    }

    #[must_use]
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Is ChimeraX answering?
    pub fn is_live(&self) -> bool {
        self.send(&ChimeraXCommand::Version).is_ok()
    }

    /// Version string reported by ChimeraX, if it is running.
    pub fn version(&self) -> Result<String, RestError> {
        let reply = self.send(&ChimeraXCommand::Version)?;
        Ok(reply
            .notes
            .iter()
            .find(|n| n.contains("ChimeraX version"))
            .cloned()
            .unwrap_or_else(|| reply.notes.join(" ").trim().to_string()))
    }

    /// Sends one command from the closed vocabulary.
    pub fn send(&self, command: &ChimeraXCommand) -> Result<ChimeraXReply, RestError> {
        if let Some(path) = command.path() {
            self.check_path(path)?;
        }
        let body = self.http_get(&command.render())?;
        parse_reply(&body)
    }

    /// Paths handed to ChimeraX must live under the export root. ChimeraX will happily read and
    /// write anywhere its own Flatpak allow-list permits, so the narrowing has to happen here.
    fn check_path(&self, candidate: &Path) -> Result<(), RestError> {
        let root = self
            .export_root
            .canonicalize()
            .unwrap_or_else(|_| self.export_root.clone());
        // The file may not exist yet (a save target), so canonicalize its parent instead.
        let resolved = if candidate.exists() {
            candidate
                .canonicalize()
                .unwrap_or_else(|_| candidate.to_path_buf())
        } else {
            match candidate.parent() {
                Some(parent) => parent
                    .canonicalize()
                    .map(|p| {
                        candidate
                            .file_name()
                            .map_or_else(|| p.clone(), |name| p.join(name))
                    })
                    .unwrap_or_else(|_| candidate.to_path_buf()),
                None => candidate.to_path_buf(),
            }
        };
        if resolved.starts_with(&root) {
            Ok(())
        } else {
            Err(RestError::PathOutsideRoot(resolved.display().to_string()))
        }
    }

    fn http_get(&self, command: &str) -> Result<String, RestError> {
        let addr = SocketAddrV4::new(Ipv4Addr::LOCALHOST, self.port);
        let mut stream = TcpStream::connect_timeout(&addr.into(), CONNECT_TIMEOUT)
            .map_err(|_| RestError::NotRunning { port: self.port })?;
        stream
            .set_read_timeout(Some(IO_TIMEOUT))
            .and_then(|()| stream.set_write_timeout(Some(IO_TIMEOUT)))
            .map_err(|e| RestError::Io(e.to_string()))?;

        let request = format!(
            "GET /run?command={} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nAccept: application/json\r\n\
             User-Agent: DockCraft\r\nConnection: close\r\n\r\n",
            percent_encode(command),
            self.port
        );
        stream
            .write_all(request.as_bytes())
            .and_then(|()| stream.flush())
            .map_err(|e| RestError::Io(e.to_string()))?;

        let mut raw = Vec::new();
        stream
            .take(MAX_RESPONSE_BYTES as u64)
            .read_to_end(&mut raw)
            .map_err(|e| RestError::Io(e.to_string()))?;

        let text = String::from_utf8_lossy(&raw).into_owned();
        let (head, body) = text
            .split_once("\r\n\r\n")
            .ok_or_else(|| RestError::Malformed("no header/body separator".to_string()))?;
        let status = head
            .lines()
            .next()
            .and_then(|l| l.split_whitespace().nth(1))
            .unwrap_or("?");
        if status != "200" {
            return Err(RestError::Command(format!(
                "HTTP {status} from ChimeraX: {}",
                body.trim().chars().take(400).collect::<String>()
            )));
        }
        Ok(body.to_string())
    }
}

/// Percent-encodes everything that is not unreserved, so a command never breaks out of the query.
fn percent_encode(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() * 2);
    for byte in raw.as_bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            out.push(*byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

/// ChimeraX with `json true` answers with its log split into note / warning / error, plus a
/// top-level `error`. Both have to be honoured: a command can fail while the HTTP status is 200.
fn parse_reply(body: &str) -> Result<ChimeraXReply, RestError> {
    let value: serde_json::Value = serde_json::from_str(body.trim())
        .map_err(|e| RestError::Malformed(format!("{e}; body was: {}", body.trim())))?;

    if let Some(err) = value.get("error").and_then(|v| v.as_str())
        && !err.trim().is_empty()
    {
        return Err(RestError::Command(err.to_string()));
    }

    let pull = |key: &str| -> Vec<String> {
        value
            .get("log messages")
            .and_then(|m| m.get(key))
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    };

    let errors = pull("error");
    if !errors.is_empty() {
        return Err(RestError::Command(errors.join("; ")));
    }

    Ok(ChimeraXReply {
        notes: pull("note"),
        warnings: pull("warning"),
        json_values: value
            .get("json values")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_render_from_typed_parts_only() {
        assert_eq!(ChimeraXCommand::Version.render(), "version");
        assert_eq!(ChimeraXCommand::CloseModel(2).render(), "close #2");
        assert_eq!(ChimeraXCommand::ViewModel(1).render(), "view #1");
        assert_eq!(
            ChimeraXCommand::InfoSelection.render(),
            "info selection level residue"
        );
        assert_eq!(
            ChimeraXCommand::SaveSelectionPdb(PathBuf::from("/tmp/a/sel.pdb")).render(),
            "save /tmp/a/sel.pdb format pdb selectedOnly true"
        );
    }

    #[test]
    fn percent_encoding_cannot_break_out_of_the_query() {
        assert_eq!(percent_encode("open a b"), "open%20a%20b");
        // A semicolon would otherwise chain a second ChimeraX command.
        assert!(percent_encode("open x; runscript evil.py").contains("%3B"));
        assert!(!percent_encode("open x; runscript evil.py").contains(';'));
        assert!(percent_encode("a&b=c").contains("%26"));
    }

    #[test]
    fn a_path_outside_the_export_root_is_refused() -> Result<(), Box<dyn std::error::Error>> {
        let root = std::env::temp_dir().join(format!("dockcraft_rest_root_{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        let client = ChimeraXRest::new(DEFAULT_REST_PORT, root.clone());

        assert!(client.check_path(&root.join("scene.cxc")).is_ok());
        let outside = client.check_path(Path::new("/etc/passwd"));
        assert!(matches!(outside, Err(RestError::PathOutsideRoot(_))));
        let escape = client.check_path(&root.join("..").join("elsewhere.pdb"));
        assert!(matches!(escape, Err(RestError::PathOutsideRoot(_))));

        let _ = std::fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn a_failing_command_is_an_error_even_at_http_200() {
        let body = r#"{"json values": [], "python values": [], "log messages":
            {"note": [], "warning": [], "error": ["No such file or directory"], "bug": []},
            "error": null}"#;
        assert!(matches!(parse_reply(body), Err(RestError::Command(_))));

        let ok = r#"{"json values": [], "python values": [], "log messages":
            {"note": ["executed scene.cxc"], "warning": [], "error": [], "bug": []},
            "error": null}"#;
        let reply = parse_reply(ok).unwrap_or_default();
        assert_eq!(reply.notes, vec!["executed scene.cxc".to_string()]);
    }
}
