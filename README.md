# DockCraft

**100% Pure-Rust Virtual Screening & Structure-Based Drug Design Workstation** (`#![forbid(unsafe_code)]`, clean-room public algorithms).

DockCraft unifies two complementary workflows in a single portable Cargo workspace:

1. **Mass Virtual Screening Engine (PyRx / AutoDock Vina / Vinardo):**
   - Spatial-hash + 14-ray cavity enclosure pocket finder (`pocket-detect`) with 42-point Fibonacci Shrake-Rupley SASA, druggability ranking, and bounded `GridBox` allocation (`<= 60 Å` axis / `1.5M` voxels).
   - Precomputed 3D trilinear affinity grids + analytical spatial gradients, rigid-core-anchored SO(3) + root-to-leaf dihedral L-BFGS & two-stage Iterated Local Search (`dock-engine`) parallelized with `rayon`, plus optional external `vina` / `smina` / `gnina` execution when installed.
   - Pure literature `ScoringFunction::Vina` and `ScoringFunction::Vinardo` (`electrostatic = 0.0`), plus opt-in `ScoringFunction::DockCraftHybrid` when adding screened electrostatics.
   - 8-descriptor Bickerton et al. (2012) `QED`, Wildman-Crippen `cLogP`, topological `TPSA`, Lipinski/Veber/hERG/BBB, core medicinal-chemistry PAINS filters (`rhodanine`, `thiazolidinone`, `quinone`, `catechol`, `michael`, `nitro`, `azo/hydrazone`, `isothiocyanate`, `strained-3ring`, `polyhalide`), `MPO`, `ROC AUC`, `EF1%`, and `EF5%` (`dock-pipeline`).
2. **Interactive 3D Pocket & HYDE-Inspired Affinity Optimizer:**
   - Hardware-accelerated `wgpu` 24 WGSL 3D viewport (`dock-viewport`) inside `eframe`/`egui` (`dockcraft`) with receptor ribbon/pocket sticks, translucent cavity clouds, dashed pharmacophoric interactions, `GridBox` gizmo, and click-to-select 3D atom picking.
   - Clean-room HYDE-inspired empirical SASA-desolvation, H-bond/salt-bridge/halogen-bond recovery, lipophilic burial, and steric clash decomposition (`hyde-scorer`) rendered as **green/red Fresnel coronas**.
   - Real-time **In-Pocket Inspirator**: mutate any clicked atom (`-H`, `-F`, `-Cl`, `-OH`, `-NH2`, `-CH3`, `-CF3`, `-CN`, `-OCH3`) with instant relaxation and $\Delta\Delta G$ feedback.

---

## Architecture (8-Crate Portable Workspace)

- `crates/struct-core` — PDB, mmCIF, MOL2, SDF (V2000/V3000), PDBQT parsers/writers (bounded atom limits), Gasteiger-Marsili PEOE partial charges (`20.02` H denominator), pH 7.4 protonation (including carboxylate, amine, and amidinium/guanidinium), AutoDock4 typing + hydrophobic flags, implicit-H molecular weight, root-to-leaf rotatable bond graph, and 3D SMILES conformer builder.
- `crates/pocket-detect` — 1.2 Å cavity grid scanner, 14-direction ray enclosure, 42-pt Fibonacci SASA, and bounded `GridBox` extraction.
- `crates/dock-engine` — Trilinear affinity grids, pure Vina / Vinardo / opt-in Hybrid scoring, rigid-core-anchored SO(3)+torsion L-BFGS (`m = 8`), `rayon` two-stage global search + ILS, RMSD clustering, and external engine adapter.
- `crates/hyde-scorer` — Clean-room HYDE-inspired per-atom $\Delta G$ & $K_i$, H-bond/salt-bridge/$\pi-\pi$/cation-$\pi$/halogen-bond detector, and interactive substituent mutator.
- `crates/dock-viewport` — Pure-Rust `wgpu` 24 WGSL instanced sphere/cylinder/corona/line pipelines integrated via `egui_wgpu::CallbackTrait`.
- `crates/dock-pipeline` — `openat2(RESOLVE_BENEATH | NO_SYMLINKS)` sandbox (`PathSandbox`), bounded 32 MiB `.chemcraft` reader, standalone 8-descriptor QED/PAINS/ADMET/MPO/ROC engine, and self-contained `MolCraft` (`.pml` + `receptor.pdb` + `ligand.sdf` + `.molcraft.json`) & `Chymera` (`.cxc` + `receptor.pdb` + `ligand.sdf` + `.chymera.json`) exporters.
- `apps/dockcraft-cli` — Headless CLI (`prepare-receptor`, `screen`, `score-pose`, `mcp`) and hardened JSON-RPC 2.0 MCP server (`stdio` and loopback-only Bearer-token `tcp`).
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

- `#![forbid(unsafe_code)]`, `unwrap_used = "deny"`, `expect_used = "deny"`, and `panic = "deny"` enforced workspace-wide, with `overflow-checks = true` in release profile.
- `PathSandbox` confines all reads/writes with Linux `openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)` and an equivalent `dirfd`-pinned `openat(..., O_NOFOLLOW)` fallback on `ENOSYS` kernels (verified under 4,000 concurrent symlink-race attempts with 0 escapes).
- TCP MCP server mandates `--auth-token`, rejects non-loopback `--bind` addresses (`0.0.0.0`, etc.), uses non-blocking multiplexed I/O with 2 s unauthenticated idle eviction so silent sockets cannot starve authenticated workers, enforces strict `initialize` gating (pre-init `notifications/initialized` rejected), bounds message frames at 1 MiB, caps `.chemcraft`/ZIP decompression at 32 MiB, and bounds `GridBox` voxels (`1.5M`) and parser atom counts (`MAX_RECEPTOR_ATOMS = 60,000`, `MAX_LIGAND_ATOMS = 512`).

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
