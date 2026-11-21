# DockCraft v0.4.0

**100% Pure-Rust Commercial-Grade Receptor Preparation, Active-Site Blueprinting, Virtual Screening & Structure-Based Drug Design Workstation** (`#![forbid(unsafe_code)]`, clean-room public algorithms).

DockCraft unifies rigorous receptor preparation & structural auditing (`prot-prep`), active-site MIF pharmacophore blueprinting (`pharm-core`), ultra-fast streaming 3D pharmacophore pre-screening (`pharm-screen`), high-throughput & multi-conformation ensemble virtual screening (AutoDock Vina / Vinardo / AutoDock4Zn-class metal coordination / pharmacophore / core tether / covalent / flexible sidechains / Boltzmann ensemble consensus), and interactive SeeSAR-inspired 3D HYDE affinity & torsion optimization in a single portable 10-crate Cargo workspace:

1. **Rigorous Receptor Preparation, Structural Audit & Ensemble Docking (`prot-prep`):**
   - **Automated Crystallographic & AlphaFold Audit:** Parses PDB (`HEADER`, `EXPDTA`, `REMARK 2/3/465/470`, `MODRES`, `SSBOND`, `HETNAM`) and mmCIF headers, auto-detects AlphaFold/ESMFold pLDDT models, resolves alternate conformations by **highest occupancy**, flags missing loops/chain breaks, verifies every expected heavy sidechain atom across all 20 amino acids, separates cocrystallized organic ligands from retained cofactors/metals/modified residues (`TPO`/`SEP`/`PTR`), audits active-site B-factors/pLDDT & protein steric clashes, and assigns a **Quality Grade (`A–D`)**.
   - **PROPKA-Inspired Local Pocket pKa & Protonation:** Models burial/desolvation, local H-bonds, Coulombic salt-bridges/acid-dyads, disulfide bridges, and catalytic metal coordination to assign exact Amber/PROPKA protonation states (`HID`, `HIE`, `HIP`, `ASH`, `GLH`, `LYN`, `CYM`, `CYX`) honored end-to-end by `struct-core`, `dock-engine`, `pharm-core`, and `hyde-scorer`.
   - **Iterative H-Bond Network Optimization:** Evaluates 180° sidechain flips (`ASN`, `GLN`, `HIS` + tautomer selection) and 12-angle polar `-OH`/`-SH` hydrogen torsion optimization (`SER`, `THR`, `TYR`, `CYS`).
   - **Multi-Conformation Ensemble Docking (`ensemble-dock`):** Docks ligands/libraries across 2..N receptor conformations (supports unaligned structures via per-receptor `--center "x1,y1,z1;x2,y2,z2"` or blind auto-pocket) and aggregates via **Boltzmann thermodynamic consensus** (`-RT ln((1/N) Σ exp(-ΔG_i / RT))`), Best Conformation, Mean, or Reciprocal Rank Fusion.
2. **Active-Site Blueprinting & 3D Pharmacophore Funnel (`pharm-core`):**
   - **7-probe Rayon-parallel Molecular Interaction Fields (MIFs):** `HDonor`, `HAcceptor`, `Hydrophobic`, `Aromatic`, `PositiveIonizable`, `NegativeIonizable`, and `MetalChelate` with pH 7.4 protein atom typing, same-sign electrostatic opposition, and 6-connected main-cavity isolation.
   - **PCA Pocket Dimensions & Optimal Ligand Envelope:** Extracts physical pocket dimensions ($L \times W \times D$), accessible volume ($\text{Å}^3$), hydrophobic/polar surface ratio, Mecozzi-Rebek optimal MW & heavy-atom envelope, optimal inter-feature 3D distance matrix with allowed ranges & vector angles, receptor wall exclusion spheres, and crystallographic water classification (`ConservedBridge` vs `Displaceable`).
   - **Hybrid Apo + Holo Consensus & Ultra-Fast Streaming Funnel (`pharm-screen`):** Bounded distance-clique DFS + exact 4×4 Horn unit-quaternion Kabsch 3D superposition ($\det R = +1$) + directional vector alignment + pocket shape containment + continuous steric wall penalty. Streams `.smi` / `.smi.gz` / `.sdf` / `.sdf.gz` / `.chemcraft` libraries at **`~0.09 ms/ligand` (`>10,000 ligands/s` on 24 threads)** and feeds pre-aligned 3D hits + soft `--pharmacophore-file` restraints straight into `dock` / `screen`.
   - **Centralized Bio Data Store (`CRAFT_DATA_DIR`):** Defaults to `/home/acs/Datos/CraftBio/{libraries,models,building_blocks,envs,campaigns}` (configurable via `CRAFT_DATA_DIR`) so multi-gigabyte libraries, ML models, and campaigns stay off the root partition.
2. **Mass Virtual Screening & Docking Engine (`pocket-detect` + `dock-engine`):**
   - **Parallel LIGSITE-csc 26-ray / 13-PSP blind pocket detector** (`pocket-detect`) with 1.0 Å occupancy grid, 6-connected subpocket separation, catalytic metal bonus, deep-core burial centroid, 42-point Fibonacci SASA, and SiteMap/DoGSiteScorer-class druggability ranking (`86.4%` Top-1 / `90.9%` Top-3 DCA ≤ 4 Å across 22 RCSB proteins in ~64 ms).
   - **Rayon-parallel 3D trilinear affinity grids** + analytical spatial gradients, rigid-core-anchored SO(3) + root-to-leaf dihedral L-BFGS (`m = 8`), Vina-style single-DOF Metropolis ILS, anchor+conformer-stratified seeding, and WL-refined symmetry-aware heavy-atom RMSD deduplication (`dock-engine`), plus optional external `vina` / `smina` / `gnina` execution when installed.
   - **Commercial docking modes:** AutoDock4Zn-inspired catalytic metal coordination (`Zn/Mg/Fe/Mn/Ca` with Tier-2 warhead typing for sulfonamides, hydroxamates, carboxylates/phosphonates, thiols), structural bridging water detection & retention (`detect-waters`, `--keep-waters`, `--auto-waters`), pharmacophore sphere restraints (`--pharmacophore`, `--pharmacophore-file`), template core tethering, covalent warhead attachment (`--covalent`), flexible receptor sidechain rotamers (`--flex-residues`, `chi1`/`chi2`), and Vina+HYDE consensus reranking (`--rerank-hyde`).
   - **Pure literature scoring & virtual screening metrics:** Exact `ScoringFunction::Vina` and `ScoringFunction::Vinardo` parity (`r = 0.9875`, MAE `0.288 kcal/mol` vs Official AutoDock Vina 1.2.7), opt-in `ScoringFunction::DockCraftHybrid`, plus 8-descriptor Bickerton `QED`, Wildman-Crippen `cLogP`, `TPSA`, Lipinski/Veber/hERG/BBB, PAINS filters, `MPO`, `ROC AUC`, `BEDROC(α=20)`, `EF1%`, and `EF5%` (`dock-pipeline`).
3. **Interactive 3D Pocket & SeeSAR-Style HYDE v2 Workstation (`hyde-scorer` + `dock-viewport` + `dockcraft`):**
   - Hardware-accelerated `wgpu` 24 WGSL 3D viewport (`dock-viewport`) inside `eframe`/`egui` (`dockcraft`) following unified **Craft UI** conventions (`File | Edit | View | Docking | Apps`, visible `v0.3.0`, interactive **Site Blueprint (MIFs & Distances)** panel with live hotspot toggles & 3D directional vectors + exclusion spheres, native Open/Save/Export dialogs, on-demand **ChemCraft**, **AdmetCraft**, and **ChimeraX** handoff).
   - **HYDE v2 empirical scoring & PoseBusters audit:** Pocket-shell SASA desolvation, 100% directional H-bond geometry (`D–H···A` angles), salt bridges, metal coordination, water bridges, $\pi\text{-}\pi$, cation-$\pi$, halogen bonds, honest $K_i$ bracket (`±1.2 kcal/mol`), built-in **PoseBusters 5-check geometric validator** (`validate-pose`), and **SeeSAR-style torsion strain traffic-lights** (`Green / Yellow / Red`).
   - **Real-time 32-Group In-Pocket Inspirator:** Mutate any clicked atom across 32 medicinal-chemistry substituents & bioisosteres (Halogens, Polar/H-Bond, Lipophilic/Small Rings, Bioisosteres/Heterocycles incl. Tetrazole, Oxetane, Cyclopropyl, Pyridine, Morpholine, Sulfonamide) with instant in-pocket relaxation and $\Delta\Delta G$ feedback.

---

## Architecture (10-Crate Portable Workspace)

- `crates/struct-core` — PDB, mmCIF, MOL2, SDF (V2000/V3000), PDBQT parsers/writers (bounded atom limits), Gasteiger-Marsili PEOE partial charges, pH 7.4 & PROPKA-state protonation (`HID`/`HIE`/`HIP`/`ASH`/`GLH`/`LYN`/`CYM`/`CYX`, delocalized amidinium/guanidinium `+1`, catalytic metal `+2`), ring perception from **3 to 24 members (macrocycles)**, non-leaking aromaticity, exact Vina X-Score AD4 typing + metal chelation tiers, structural bridging water perception (`detect_bridging_waters`), WL-refined symmetry-aware RMSD (`rmsd_heavy_symmetric`), largest-rigid-fragment root-to-leaf rotatable bond graph, and 3D SMILES builder with multi-ring embedding, puckered rings, `@`/`@@` tetrahedral chirality, and planar `E/Z` (`/`/`\`) alkene stereochemistry.
- `crates/prot-prep` — Raw PDB/mmCIF/AlphaFold structural audit, highest-occupancy `altloc` resolution, HETATM classification & organic ligand extraction, PROPKA-inspired local pKa protonation assignment, iterative 180° `ASN`/`GLN`/`HIS` H-bond flip optimization + 12-angle `SER`/`THR`/`TYR`/`CYS` polar-H orientation, and Rayon-parallel Boltzmann multi-conformation ensemble docking.
- `crates/pocket-detect` — Parallel LIGSITE-csc 26-ray / 13-PSP 1.0 Å cavity scanner, 6-connected subpocket separation, catalytic metal bonus, 42-pt Fibonacci SASA, and bounded `GridBox` extraction (`<= 60 Å` axis / `1.5M` voxels).
- `crates/pharm-core` — 7-probe parallel MIF active-site blueprinting (`analyze_binding_site`), 3×3 PCA pocket dimensions & Mecozzi-Rebek ligand envelope, Hybrid Apo+Holo consensus & optimal distance matrix, wall exclusion spheres, water assessment, 4×4 Horn unit-quaternion Kabsch 3D superposition, streaming `pharm-screen` funnel, and `CRAFT_DATA_DIR` manager.
- `crates/dock-engine` — Parallel trilinear affinity grids, Vina / Vinardo / DockCraftHybrid + catalytic metal coordination, 7-class pharmacophore/core-tether/covalent restraints + flexible sidechain rotamers (`constraints.rs`), rigid-core-anchored SO(3)+torsion L-BFGS (`m = 8`), `rayon` two-stage global search + single-DOF ILS, symmetry-aware clustering, and external engine adapter.
- `crates/hyde-scorer` — HYDE v2 per-atom $\Delta G$ & honest $K_i$ bracket, 100% directional H-bond/metal/water-bridge/salt-bridge/$\pi$/halogen interaction detector, built-in PoseBusters validator (`posebusters.rs`), torsion traffic-lights (`torsion.rs`), and 32-group interactive Inspirator.
- `crates/dock-viewport` — Pure-Rust `wgpu` 24 WGSL instanced sphere/cylinder/corona/line pipelines with torsion traffic-light beads, catalytic metal & bridging water markers, translucent pharmacophore spheres + directional vectors + toggleable site exclusion spheres, and ray atom picking.
- `crates/dock-pipeline` — `openat2(RESOLVE_BENEATH | NO_SYMLINKS)` sandbox (`PathSandbox`), bounded 32 MiB `.chemcraft` reader, standalone QED/PAINS/ADMET/MPO/ROC/BEDROC20 engine, ChimeraX REST live selection, and self-contained PyMOL (`.pml`) & ChimeraX (`.cxc`) scene exporters.
- `apps/dockcraft-cli` — Headless CLI (`prepare-receptor`, `prot-prep`, `detect-pockets`, `detect-waters`, `analyze-site`, `pharm-screen`, `dock`, `ensemble-dock`, `screen`, `score-pose`, `validate-pose`, `chimerax`, `mcp`) and hardened JSON-RPC 2.0 MCP server (`stdio` and loopback-only Bearer-token `tcp`, including `prepare_protein`, `ensemble_dock`, `analyze_binding_site`, and `pharmacophore_screen`).
- `apps/dockcraft` — Native desktop GUI workstation (`eframe` + `egui` + `wgpu`) with unified Craft menu bar, collapsible **Receptor Prep & Audit (pKa / Flips)** and **Site Blueprint** panels, native OS file dialogs, `.dockcraft` / `.pharm.json` persistence, and on-demand ChemCraft / AdmetCraft / ChimeraX handoff.

---

## Validation Status (Measured Head-to-Head vs Official AutoDock Vina 1.2.7)

### 6-Track Commercial Benchmark (22 Diverse RCSB Complexes + 4 DUD-E Property-Matched Targets, 600 Ligands)

All 22 RCSB crystal ligands (`3PTB, 1HSG, 3ERT, 1STP, 3HS4, 4TLN, 5TLN, 1M17, 1H1S, 1IEP, 1XKK, 1YWR, 1ERE, 1SQN, 1Z95, 1EVE, 1GPK, 3MXF, 1UDT, 2BSM, 1LPZ, 1X70` spanning proteases, kinases, nuclear receptors, Zn metalloenzymes, hydrolases, bromodomain, and chaperone) had official RCSB CCD bond orders/stereochemistry verified, and docking started from **unbiased RDKit ETKDGv3 conformers centered at `(0, 0, 0)`** evaluated with **in-place symmetry-aware heavy-atom RMSD** (no superposition) and **PoseBusters geometric validity**.

| Track / Metric | Baseline v0.1.0 | Official AutoDock Vina 1.2.7 | **DockCraft v0.2.0** |
|---|---:|---:|---:|
| **T1 Re-dock Top-1 ≤ 2.0 Å** (22 RCSB) | 11/22 (50.0%) | 11/22 (50.0%) | **14/22 (63.6%)** |
| **T1 Re-dock Top-3 ≤ 2.0 Å** (22 RCSB) | 16/22 (72.7%) | 15/22 (68.2%) | **20/22 (90.9%)** |
| **T1 PoseBusters Valid & ≤ 2.0 Å** | 11/22 (50.0%) | 11/22 (50.0%) | **14/22 (63.6%)** (22/22 PB valid) |
| **T1 Median Top-1 RMSD** | 2.27 Å | 1.71 Å | **1.61 Å** (`1.55 Å` w/ `--rerank-hyde`) |
| **T1 Median Wall Time** (incl. grid build) | 5,097 ms | 27,222 ms | **1,675 ms** (**16.3× faster**) |
| **T2 Blind Pocket Top-1 / Top-3 DCA ≤ 4 Å** | 27.3% / 31.8% | N/A | **86.4% (19/22) / 90.9% (20/22)** |
| **T2 Blind Pocket Top-1 DCC ≤ 4 Å** (med DCA) | 9.1% (6.37 Å) | N/A | **72.7% (16/22) (1.50 Å)** |
| **T3 Vina 1.2.7 Parity** ($r$ / $\rho$ / MAE) | 0.975 / 0.958 / 0.903 | 1.000 / 1.000 / 0.000 | **0.988 / 0.971 / 0.288 kcal/mol** |
| **T3 Vinardo 1.2.7 Parity** ($r$ / $\rho$ / MAE) | 0.937 / 0.918 / 0.568 | 1.000 / 1.000 / 0.000 | **0.956 / 0.948 / 0.345 kcal/mol** |
| **T3 vs Exp. $\Delta G$** (21 cplx excl. 1STP) | HYDE $\rho = +0.170$ | Vina $\rho = +0.317$, $r=+0.617$ | **HYDE $\rho = +0.411$**; **Consensus $\rho = +0.385$, $r=+0.610$** |
| **T4 DUD-E Mean ROC AUC** (4 tgts, 600 ligs) | 0.6065 | 0.7201 | **0.7753** |
| **T4 DUD-E Mean BEDROC(α=20)** | 0.4275 | 0.3563 | **0.5528** |
| **T4 DUD-E Mean EF1% / EF5%** | 3.75 / 2.62 | 3.75 / 2.44 | **3.75 / 3.38** |
| **T4 Virtual Screening Speed** | 155.7 ms/lig | 9,855.7 ms/lig | **136.2 ms/lig** (**72.4× faster**) |
| **T5 Metalloenzymes** (`3HS4 / 4TLN / 5TLN` Top-1) | 5.54 / 8.79 / 8.99 Å | 5.32 / 2.83 / 6.51 Å | **1.42 / 1.90 / 2.68 Å** (`1.64 Å` Top-3 & `--rerank-hyde`) |
| **T5 HYDE Directional H-Bonds** (with angle) | 0/119 (0%) | N/A | **77/77 (100%)** |

Per-target DUD-E virtual screening (`25` actives + `125` property-matched decoys each):
- `cah2` (Zn metalloenzyme): DockCraft AUC **0.793**, BEDROC20 **0.598**, EF1% **6.0** vs Vina 1.2.7 AUC 0.561, BEDROC20 0.086, EF1% 0.0 (max null descriptor 0.639).
- `cdk2` (kinase): DockCraft AUC **0.726**, BEDROC20 **0.670**, EF5% **4.5** vs Vina 1.2.7 AUC 0.709, BEDROC20 0.441, EF5% 3.0 (max null 0.669).
- `esr1` (nuclear receptor): DockCraft AUC **0.828**, BEDROC20 **0.544** vs Vina 1.2.7 AUC 0.866, BEDROC20 0.508 (max null 0.699).
- `try1` (protease): DockCraft AUC **0.754**, BEDROC20 **0.398**, EF5% **3.0** vs Vina 1.2.7 AUC 0.745, BEDROC20 0.390, EF5% 2.2.

### Active-Site Blueprinting & Streaming 3D Pharmacophore Funnel (`pharm-core` on DUD-E CDK2)

Benchmarked on human **CDK2** (`PDB 1KE6`, `1H1S`, and DUD-E `cdk2` with `474` ChEMBL actives + `27,850` property-matched ZINC decoys = `28,324` ligands, plus an apples-to-apples `N = 2,000` stratified subset of `200` actives + `1,800` decoys):

| Pipeline Mode (DUD-E CDK2) | Wall Time | Speedup | ROC AUC | BEDROC(α=20) | EF1% | EF5% | EF10% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Best 1D Null Descriptor (`cLogP`, `N=2,000`) | — | — | `0.665` | — | — | — | — |
| Unfiltered `DockCraft Vina` (`N=2,000`, exh=4) | `277.3 s` | `1.0×` | `0.715` | `0.360` | `4.00×` (8/20) | `3.30×` (33/100) | `3.15×` (63/200) |
| Stage 1 `pharm-screen` only (`N=2,000`) | **`0.20 s`** | `1,387×` | **`0.793`** | `0.432` | `4.50×` (9/20) | `4.50×` (45/100) | `3.60×` (72/200) |
| **2-Stage Funnel (`pharm-screen` Top 400 → Vina)** | **`35.6 s`** | **`7.8×`** | **`0.797`** | **`0.484`** | **`6.50×` (13/20)** | **`4.80×` (48/100)** | **`3.90×` (78/200)** |
| **2-Stage Consensus (`0.65·Vina + 0.35·Pharm`)** | **`35.6 s`** | **`7.8×`** | **`0.799`** | **`0.500`** | **`6.50×` (13/20)** | **`5.10×` (51/100)** | **`4.10×` (82/200)** |
| Full-Library Streaming `pharm-screen` (`N=28,324`) | **`2.65 s`** (`0.093 ms/lig`) | `~1,480×` | **`0.801`** | `0.294` *(1.67% prev.)* | **`7.60×` (36/283)** | **`6.12×` (145/1416)** | **`4.52×` (214/2832)** |

### Built-In Repository Regression Suite

Run `cargo test --release --workspace` to reproduce all workspace tests (all 9 crates passing, including `crates/pharm-core/tests/real_pockets.rs` verifying `>= 80%` crystallographic feature recovery across `3PTB`, `3ERT`, `1HSG`, `1STP`):
- **Randomized-conformer re-docking** (`crates/dock-pipeline/tests/randomized_conformer_redock.rs`): 48/48 poses under 2.0 Å across 3PTB, 1STP, 1HSG (13 rotatable bonds), and 3ERT.
- **Commercial modes verification** (`crates/dock-engine/tests/commercial_modes.rs`): symmetry-aware RMSD, 14-membered macrocycle perception + `E/Z` stereochemistry, bridging water detection/retention, Zn-sulfonamide coordination, pharmacophore sphere, core tether, covalent attachment, and flexible sidechain rotamers.
- **Grid accuracy** (`crates/dock-engine/tests/grid_accuracy.rs`): exact at nodes (`4.2e-8 kcal/mol`), `0.033 kcal/mol` MAE at cell centers.
- **Charge-matched null-model enrichment** (`crates/dock-pipeline/tests/enrichment_null_models.rs`): ROC AUC `0.976`, EF5% `3.33` vs best null descriptor `0.750` (formal charge `0.500`).
- **ChemCraft & ChimeraX interoperability** (`crates/dock-pipeline/tests/interop_chemcraft_chimerax.rs`): closed-vocabulary ChimeraX REST selection (`--box-from-chimerax`) and `*.dockcraft.json` sidecar round-trip to `chemcraft-cli dock-results`.

### Honest Scope & Remaining Limitations

- **Empirical scoring vs free energy:** As with all fast docking scoring functions (Vina, Glide SP, GOLD, HYDE), reported kcal/mol values are empirical scores — absolute $\Delta G$ RMSE on diverse complexes is ~2.0–2.6 kcal/mol (excluding extreme outliers like streptavidin–biotin), so `ki_range_nm` is displayed as an explicit `±1.2 kcal/mol` uncertainty bracket rather than a point prediction. Rigorous relative free energies (`< 1 kcal/mol`) still require alchemical FEP/TI MD.
- **Receptor backbone flexibility:** `--flex-residues` samples sidechain `chi1`/`chi2` rotamers; large loop or backbone induced-fit rearrangements (e.g., DFG-out kinase shifts or `1H1S` CDK2 loop) require ensemble docking across multiple receptor conformations.
- **Covalent bond-order perception:** `--covalent` enforces the covalent attachment distance/orientation and excludes 1-2/1-3 clashes across the tether, assuming the ligand is supplied in its post-reaction (ring-opened/adduct) state.

---

## Build, Test & Install

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --release --workspace
cargo build --release --workspace
./packaging/install.sh
```

Clean clones require no external sibling checkouts and compile cleanly out of the box.

## Security & Hardening Baseline

- `#![forbid(unsafe_code)]`, `unwrap_used = "deny"`, `expect_used = "deny"`, and `panic = "deny"` enforced workspace-wide, with `overflow-checks = true` in release profile.
- `PathSandbox` confines all reads/writes with Linux `openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)` and an equivalent `dirfd`-pinned `openat(..., O_NOFOLLOW)` fallback on `ENOSYS` kernels (verified under 4,000 concurrent symlink-race attempts with 0 escapes).
- TCP MCP server mandates `--auth-token`, rejects non-loopback `--bind` addresses (`0.0.0.0`, etc.), uses non-blocking multiplexed I/O with 2 s unauthenticated idle eviction, enforces strict `initialize` gating, bounds message frames at 1 MiB, caps `.chemcraft`/ZIP decompression at 32 MiB, and bounds `GridBox` voxels (`1.5M`) and parser atom counts (`MAX_RECEPTOR_ATOMS = 60,000`, `MAX_LIGAND_ATOMS = 512`).

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
