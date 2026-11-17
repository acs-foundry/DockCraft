# DockCraft

**100% Pure-Rust Virtual Screening & Structure-Based Drug Design Workstation** (`#![forbid(unsafe_code)]`, clean-room public algorithms).

DockCraft unifies two complementary workflows in a single portable Cargo workspace:

1. **Mass Virtual Screening Engine (PyRx / AutoDock Vina / Vinardo):**
   - Spatial-hash + 14-ray cavity enclosure pocket finder (`pocket-detect`) with 42-point Fibonacci Shrake-Rupley SASA and druggability ranking.
   - Precomputed 3D trilinear affinity grids + analytical spatial gradients, SO(3) + dihedral Iterated Local Search & L-BFGS (`dock-engine`) parallelized with `rayon`, plus optional external `vina` / `smina` / `gnina` execution when installed.
   - Multi-Parameter Optimization (`MPO`), pure-Rust ADMET / Lipinski / Veber / PAINS / QED filtering, `ROC AUC`, `EF1%`, and `EF5%` enrichment metrics (`dock-pipeline`).
2. **Interactive 3D Pocket & HYDE Affinity Optimizer (SeeSAR-style):**
   - Hardware-accelerated `wgpu` 24 WGSL 3D viewport (`dock-viewport`) inside `eframe`/`egui` (`dockcraft`) with receptor ribbon/pocket sticks, translucent cavity clouds, dashed pharmacophoric interactions, corner-drag `GridBox`, and click-to-select 3D atom picking.
   - Per-atom clean-room HYDE decomposition (`hyde-scorer`): desolvation penalty vs. satisfied H-bond/salt-bridge recovery vs. lipophilic burial & steric clash rendered as **green/red Fresnel coronas**.
   - Real-time **In-Pocket Inspirator**: mutate any clicked atom (`-H`, `-F`, `-Cl`, `-OH`, `-NH2`, `-CH3`, `-CF3`, `-CN`, `-OCH3`) with instant relaxation and $\Delta\Delta G$ feedback.

---

## Architecture (8-Crate Portable Workspace)

- `crates/struct-core` — PDB, mmCIF, MOL2, SDF (V2000/V3000), PDBQT parsers/writers, Gasteiger partial charges, pH 7.4 protonation, AutoDock4 typing, rotatable bond graph, and 3D SMILES coordinate builder.
- `crates/pocket-detect` — 1.2 Å cavity grid scanner, 14-direction ray enclosure, 42-pt Fibonacci SASA, and `GridBox` extraction.
- `crates/dock-engine` — Trilinear affinity grids, smooth Vina/Vinardo scoring + electrostatics, analytical SO(3)+torsion L-BFGS, `rayon` Iterated Local Search, RMSD clustering, and external engine adapter.
- `crates/hyde-scorer` — Per-atom HYDE $\Delta G$ & $K_i$, H-bond/salt-bridge/$\pi-\pi$/cation-$\pi$/halogen-bond detector, and interactive substituent mutator.
- `crates/dock-viewport` — Pure-Rust `wgpu` 24 WGSL instanced sphere/cylinder/corona/line pipelines integrated via `egui_wgpu::CallbackTrait`.
- `crates/dock-pipeline` — `openat2(RESOLVE_BENEATH | NO_SYMLINKS)` sandbox (`PathSandbox`), bounded 32 MiB `.chemcraft` reader, standalone ADMET/QED/MPO/ROC engine, and `MolCraft` (`.pml`/`.molcraft.json`) + `Chymera` (`.cxc`/`.chymera.json`) exporters.
- `apps/dockcraft-cli` — Headless CLI (`prepare-receptor`, `screen`, `score-pose`, `mcp`) and hardened JSON-RPC 2.0 MCP server (`stdio` and Bearer-token `tcp`).
- `apps/dockcraft` — Native desktop GUI workstation (`eframe` + `egui` + `wgpu`).

---

## Build, Test & Verify

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo build --release --workspace
```

Clean clones require no external sibling checkouts and compile cleanly out of the box.

## Security & Hardening Baseline

- `#![forbid(unsafe_code)]`, `unwrap_used = "deny"`, `expect_used = "deny"`, and `panic = "deny"` enforced workspace-wide.
- Overflow checks enabled in release profile (`overflow-checks = true`).
- MCP server and `PathSandbox` confine all reads/writes with Linux `openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)` and an equivalent `dirfd`-pinned `openat(..., O_NOFOLLOW)` fallback on `ENOSYS` kernels (verified under 4,000 concurrent symlink-race attempts with 0 escapes).
- Bounded 1 MiB MCP message frame caps, `initialize` lifecycle gate, constant-time Bearer token comparison, 32-connection TCP concurrency limit, `catch_unwind` isolation, 32 MiB `.chemcraft`/ZIP decompression limit, and 1,000-record MCP library cap.

## CLI & MCP Usage

```bash
# Prepare receptor
dockcraft-cli prepare-receptor --input receptor.pdb --output receptor.pdbqt

# Run virtual screening
dockcraft-cli screen \
  --receptor tests/fixtures/docking_benchmark/1stp_biot_pocket.pdb \
  --library library.smi \
  --center 2.0,0.0,0.5 --size 16,16,16 \
  --out results.csv

# Start hardened MCP server (stdio)
dockcraft-cli mcp --read-root ./inputs --write-root ./outputs --transport stdio
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
