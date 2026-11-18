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

- `crates/struct-core` — PDB, mmCIF, MOL2, SDF (V2000/V3000), PDBQT parsers/writers (bounded atom limits), Gasteiger-Marsili PEOE partial charges (`20.02` H denominator), pH 7.4 protonation (carboxylate, amine, and **one delocalised charge per amidinium/guanidinium group**, so benzamidine is +1 and metformin +1), smallest-cycle ring perception from 3 to 8 members, aromaticity that does not leak across a fused bond, AutoDock4 typing + hydrophobic flags, implicit-H molecular weight, largest-rigid-fragment root with a root-to-leaf rotatable bond graph, and a 3D SMILES builder with element-aware bond lengths, VSEPR bond angles, puckered saturated rings and `@`/`@@` tetrahedral stereochemistry.
- `crates/pocket-detect` — 1.2 Å cavity grid scanner, 14-direction ray enclosure, 42-pt Fibonacci SASA, and bounded `GridBox` extraction.
- `crates/dock-engine` — Trilinear affinity grids, pure Vina / Vinardo / opt-in Hybrid scoring, rigid-core-anchored SO(3)+torsion L-BFGS (`m = 8`), `rayon` two-stage global search + ILS with a full-circle conformer bank, stall restarts and a walker/step budget that scales with the ligand's torsional degrees of freedom, RMSD clustering, and external engine adapter.
- `crates/hyde-scorer` — Clean-room HYDE-inspired per-atom $\Delta G$ & $K_i$, H-bond/salt-bridge/$\pi-\pi$/cation-$\pi$/halogen-bond detector, and interactive substituent mutator.
- `crates/dock-viewport` — Pure-Rust `wgpu` 24 WGSL instanced sphere/cylinder/corona/line pipelines integrated via `egui_wgpu::CallbackTrait`.
- `crates/dock-pipeline` — `openat2(RESOLVE_BENEATH | NO_SYMLINKS)` sandbox (`PathSandbox`), bounded 32 MiB `.chemcraft` reader, standalone 8-descriptor QED/PAINS/ADMET/MPO/ROC engine, and self-contained `MolCraft` (`.pml` + `receptor.pdb` + `ligand.sdf` + `.molcraft.json`) & `Chymera` (`.cxc` + `receptor.pdb` + `ligand.sdf` + `.chymera.json`) exporters.
- `apps/dockcraft-cli` — Headless CLI (`prepare-receptor`, `screen`, `score-pose`, `mcp`) and hardened JSON-RPC 2.0 MCP server (`stdio` and loopback-only Bearer-token `tcp`).
- `apps/dockcraft` — Native desktop GUI workstation (`eframe` + `egui` + `wgpu`).

---

## Validation status (measured, not claimed)

Every figure below is produced by a test in this repository. Run `cargo test --release --workspace -- --nocapture` to reproduce them.

### Re-docking from a randomized conformer

`crates/dock-pipeline/tests/randomized_conformer_redock.rs`. The crystal ligand is displaced 9.0 A **and every torsion is redrawn uniformly over the full circle**, which is the state a ligand arrives in from a SMILES embedding. The box follows the standard re-docking protocol (centred on the native ligand); finding the right *site* is a separate problem, covered by the pocket-detection test.

| Complex | Rotatable bonds | Mean RMSD | Worst | Under 2 A | Evaluations | Wall time (grid build included) |
|---|---|---|---|---|---|---|
| 3PTB / benzamidine | 1 | 1.20 A | 1.93 A | 12/12 | 22.7 k | 1.06 s |
| 1STP / biotin | 5 | 0.54 A | 0.94 A | 12/12 | 118 k | 1.86 s |
| 1HSG / indinavir | 13 | 0.97 A | 1.38 A | 12/12 | 933 k | 5.40 s |
| 3ERT / 4-hydroxytamoxifen | 8 | 1.48 A | 1.80 A | 12/12 | 294 k | 3.41 s |

Measured over 12 seeds per complex; the test in the repository runs 3 of them to keep the suite quick. **The wall times include building the affinity grid**, which is most of the cost for a single ligand and is amortised across a library in a campaign.

### Enrichment against null models

`crates/dock-pipeline/tests/enrichment_null_models.rs`. Six aryl amidines against fourteen decoys, **all twenty carrying a net +1 at pH 7.4**, so no model can win on charge alone.

Docking ROC AUC **0.976**, EF5% **3.33** (the attainable maximum for 6 actives in 20 compounds). Strongest null model: hydrogen-bond donor count at 0.750; TPSA 0.714; nitrogen count 0.607; formal charge exactly **0.500**; molecular weight 0.417; QED and the ADMET score 0.476.

Twenty hand-picked compounds are a sanity check, not a validation: a real one needs a property-matched decoy set with measured affinities. What this rules out is the failure mode it replaced, where the actives were the only cations in the set and ranking by formal charge alone scored a perfect 1.000.

### Grid accuracy

`crates/dock-engine/tests/grid_accuracy.rs`. The trilinear interpolant against direct evaluation of the scoring function, at 0.375 A spacing: exact at the grid nodes (4.2e-8 kcal/mol), mean absolute error 0.033 kcal/mol at cell centres, 0.088 kcal/mol worst case within the pose-relevant band (|E| <= 2 kcal/mol), and up to 1.0 kcal/mol deep inside the repulsive wall, where it does not affect ranking.

### Absolute energies

`crates/dock-pipeline/tests/affinity_accuracy.rs`. **The reported kcal/mol values are scores, not predicted affinities, and `estimated_ki_nm` is derived from one of them rather than measured.** Against published affinities for the four benchmark complexes the mean absolute error is 4.25 kcal/mol for Vina (worst 10.69) and 3.60 for HYDE (worst 6.98), dominated by streptavidin-biotin, which is outside the applicability domain of every empirical scoring function. Excluding it, both are around 2-2.5 kcal/mol, which is still three orders of magnitude in Ki. Both rank the micromolar complex last. Four complexes cannot establish a correlation and none is claimed.

### Taking the docking box from a ChimeraX selection

Open the receptor, select the binding-site residues on screen, and let DockCraft read the selection back over ChimeraX's own REST remote control:

```bash
dockcraft-cli chimerax --receptor receptor.pdb --ligand pose.sdf --name MySite --launch
# inside ChimeraX, once per session:
#   remotecontrol rest start port 60958 json true
#   select /A:346,347,350,353,387,388,391,394,404,421,424,428,521,524,525,528
dockcraft-cli screen --receptor receptor.pdb --library lib.sdf --out out.csv \
                     --box-from-chimerax --box-padding 5
```

Measured on 2026-11-18 with 3 randomized-torsion seeds per box:

| | Selection box centre vs native | Docking into the selection box | Docking into a box centred on the answer |
|---|---|---|---|
| 3PTB | 4.56 A | 3/3 under 2 A, mean **1.120 A** | 3/3, mean 1.129 A |
| 3ERT | 2.48 A | 3/3 under 2 A, mean **1.521 A** | 3/3, mean 1.491 A |

The human-chosen box costs nothing in accuracy against a box centred on the answer, and unlike that box it does not need to know the answer in advance. `crates/dock-pipeline/tests/live_chimerax_selection.rs` reproduces it.

**The command vocabulary sent to ChimeraX is closed.** A ChimeraX command is arbitrary code in ChimeraX's process -- `runscript` executes Python -- so `ChimeraXCommand` is an enum rendered from typed parts, there is no `run(&str)` in the public surface, the host is fixed to 127.0.0.1 in code, and every path a command can name is checked to live under the export root before the request is sent.

### Sending results back to ChemCraft

`dockcraft-cli screen --sidecar` also writes a versioned `*.dockcraft.json` next to the docked poses, under `$XDG_DATA_HOME/ChemCraft/dockcraft_results`, mirroring the `$XDG_DATA_HOME/DockCraft/chemcraft_handoff` directory ChemCraft uses for the other direction. `chemcraft-cli dock-results` folds it back into a project's SAR table under `DockCraft_*` keys, joining on the SAR row id and falling back to the canonical SMILES, so matched molecular pairs and activity cliffs can be computed on a docking score beside the measured one.

DockCraft deliberately does not write a `.chemcraft` project: that would mean fabricating ChemCraft's internal types and the two schemas would drift apart. Each tool writes only its own format, and the merge belongs to whoever owns the document.

### Known limitations

- Pocket detection ranks by druggability, not by the site you want. Measured against the native ligand's centroid, the best automatic pocket is 7.9 A off on 3PTB and 11.2 A off on 3ERT. Either give the engine a `GridBox` you have chosen, or point at the site in ChimeraX and let DockCraft take the box from your selection (see below).
- Ring perception covers 3 to 8 members. Macrocycles are not perceived as rings.
- Double-bond (`/`, `\\`) stereochemistry in SMILES is parsed and ignored; only tetrahedral `@`/`@@` is honoured.
- HYDE is a per-atom interaction decomposition for inspection. Do not read its total as an affinity.

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
