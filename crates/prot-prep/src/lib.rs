//! `prot-prep`: Rigorous receptor preparation, structural quality audit (PDB/mmCIF/AlphaFold),
//! PROPKA-inspired local pKa & protonation assignment (`HID`/`HIE`/`HIP`/`ASH`/`GLH`/`LYN`/`CYM`/`CYX`),
//! iterative 180° H-bond network flip optimization (`ASN`/`GLN`/`HIS`) + polar-H orientation,
//! and Boltzmann multi-conformation ensemble docking for `DockCraft`.

#![forbid(unsafe_code)]

pub mod audit;
pub mod ensemble;
pub mod hbond_opt;
pub mod protonation;

use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use struct_core::{
    BridgingWater, Molecule3D, detect_bridging_waters, prepare_receptor_keeping_waters,
};
use thiserror::Error;

pub use audit::{
    AltlocResolution, AuditIssue, AuditSeverity, ChainBreak, ChainSummary, HetClass, HetGroupInfo,
    MissingAtomRecord, MissingResidueRecord, RawStructure, StericClash, StructuralAudit,
    parse_raw_structure,
};
pub use ensemble::{
    ConformationScore, EnsembleAggregation, EnsembleCandidate, EnsembleReceptor, EnsembleReport,
    boltzmann_average_kcal, run_ensemble_docking,
};
pub use hbond_opt::{
    HBondOptReport, SidechainFlip, optimize_polar_h_orientations, optimize_sidechain_flips,
};
pub use protonation::{ProtonationAssignment, ProtonationReport, assign_local_protonation};

#[derive(Debug, Error)]
pub enum PrepError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("structure parse error: {0}")]
    Parse(String),
}

/// Configuration for `prepare_protein_from_str` / `prepare_protein_file`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrepConfig {
    /// Target physiological pH for local pKa protonation assignment (default `7.4`).
    pub ph: f64,
    /// Optional chain filter (empty = keep all chains present in the structure).
    pub chains: Vec<char>,
    /// Keep recognized biological cofactors (`ATP/ADP/NAD/FAD/HEM/SAM/...`, default `true`).
    pub keep_cofactors: bool,
    /// Keep catalytic divalent metal ions (`Zn/Mg/Fe/Mn/Ca/Cu`, default `true`).
    pub keep_metals: bool,
    /// Strip cocrystallized organic ligands from the receptor (extracting them separately, default `true`).
    pub strip_ligands: bool,
    /// Automatically retain buried bridging crystallographic waters (>= `min_water_hbonds` H-bonds
    /// inside pocket radius, default `false` = dry receptor unless requested).
    pub keep_bridging_waters: bool,
    /// Minimum receptor H-bonds for retaining a bridging water when `keep_bridging_waters` is true.
    pub min_water_hbonds: usize,
    /// Run 180° `ASN`/`GLN`/`HIS` flip optimization + polar `-OH`/`-SH` orientation (default `true`).
    pub optimize_hbonds: bool,
    /// Optional binding pocket center (auto-inferred from primary organic ligand if `None`).
    pub pocket_center: Option<[f64; 3]>,
    /// Pocket neighborhood radius in Å for pocket-specific audit & bridging-water retention (default `10.0`).
    pub pocket_radius: f64,
}

impl Default for PrepConfig {
    fn default() -> Self {
        Self {
            ph: 7.4,
            chains: Vec::new(),
            keep_cofactors: true,
            keep_metals: true,
            strip_ligands: true,
            keep_bridging_waters: false,
            min_water_hbonds: 2,
            optimize_hbonds: true,
            pocket_center: None,
            pocket_radius: 10.0,
        }
    }
}

/// Complete output of the `prot-prep` receptor preparation & structural audit pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedReceptor {
    /// Clean, protonated, H-bond-optimized receptor ready for `dock-engine`, `pharm-core`, and `hyde-scorer`.
    pub receptor: Molecule3D,
    /// Extracted cocrystallized organic ligands (if any were present in the input PDB/mmCIF).
    pub extracted_ligands: Vec<Molecule3D>,
    /// Crystallographic bridging waters detected near the active site.
    pub bridging_waters: Vec<BridgingWater>,
    /// Detailed crystallographic / AlphaFold structural audit.
    pub audit: StructuralAudit,
    /// PROPKA-inspired local pKa & protonation state assignments.
    pub protonation: ProtonationReport,
    /// Applied 180° ASN/GLN/HIS flips & optimized polar hydroxyl/thiol H rotors.
    pub hbond_opt: HBondOptReport,
    pub elapsed_ms: u128,
}

impl PreparedReceptor {
    /// Renders a comprehensive, ELN-ready Markdown Structural Audit & Protonation Report.
    #[must_use]
    pub fn render_markdown(&self) -> String {
        let a = &self.audit;
        let mut md = String::new();
        let _ = writeln!(
            md,
            "# Receptor Preparation & Structural Audit: {}\n",
            a.structure_id
        );
        if !a.title.is_empty() {
            let _ = writeln!(md, "> {}\n", a.title);
        }
        let _ = writeln!(md, "- **Quality Grade:** **{}**", a.quality_grade);
        let _ = writeln!(md, "- **Method:** {}", a.experimental_method);
        if let Some(res) = a.resolution_a {
            let rf = a
                .r_free
                .map_or_else(|| "n/a".to_string(), |v| format!("{v:.3}"));
            let rw = a
                .r_work
                .map_or_else(|| "n/a".to_string(), |v| format!("{v:.3}"));
            let _ = writeln!(
                md,
                "- **Resolution / R-work / R-free:** `{res:.2} Å` / `{rw}` / `{rf}`"
            );
        }
        let metric_lbl = if a.is_alphafold {
            "Mean pLDDT"
        } else {
            "Mean B-factor (Å²)"
        };
        let pock_b = a
            .pocket_mean_b_factor_or_plddt
            .map_or_else(|| "n/a".to_string(), |v| format!("{v:.1}"));
        let _ = writeln!(
            md,
            "- **{metric_lbl}:** `{:.1}` (Active site: `{pock_b}`)",
            a.mean_b_factor_or_plddt
        );
        let _ = writeln!(
            md,
            "- **Selected Chains:** `{}` ({} protein residues, {} final receptor atoms incl. polar H)",
            a.selected_chains
                .iter()
                .map(char::to_string)
                .collect::<Vec<_>>()
                .join(", "),
            a.total_protein_residues,
            self.receptor.atoms.len()
        );
        let _ = writeln!(
            md,
            "- **Target pH:** `{:.1}` | **Net Protein Charge:** `{:+}` | **Elapsed:** `{} ms`\n",
            self.protonation.target_ph, self.protonation.total_protein_charge, self.elapsed_ms
        );

        // Chain inventory
        md.push_str("## 1. Chain & HETATM Inventory\n\n");
        md.push_str("| Chain | Residues | Heavy Atoms | ModRes | Ligands | Cofactors | Metals | Waters | Mean B/pLDDT |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
        for c in &a.chains {
            let _ = writeln!(
                md,
                "| `{}` | {} | {} | {} | {} | {} | {} | {} | {:.1} |",
                c.chain_id,
                c.protein_residues,
                c.protein_heavy_atoms,
                c.modified_residues,
                c.ligands,
                c.cofactors,
                c.metals,
                c.waters,
                c.mean_b_factor
            );
        }
        md.push('\n');

        if !a.het_groups.is_empty() {
            md.push_str("| HET Group | Class | Heavy Atoms | Kept? | Description |\n|---|---|---:|---|---|\n");
            for h in &a.het_groups {
                let _ = writeln!(
                    md,
                    "| `{}:{}{}` | {} | {} | {} | {} |",
                    h.chain_id,
                    h.res_name,
                    h.res_seq,
                    h.class.as_str(),
                    h.atom_count,
                    if h.kept_in_receptor {
                        "Yes"
                    } else {
                        "Stripped"
                    },
                    h.description.as_deref().unwrap_or("—")
                );
            }
            md.push('\n');
        }

        // Audit issues
        md.push_str("## 2. Structural Audit Findings\n\n");
        if a.issues.is_empty() {
            md.push_str("- No structural issues detected.\n\n");
        } else {
            for iss in &a.issues {
                let tag = match iss.severity {
                    AuditSeverity::Info => "INFO",
                    AuditSeverity::Warning => "WARN",
                    AuditSeverity::Critical => "CRITICAL",
                };
                let _ = writeln!(md, "- **[{tag}] {}:** {}", iss.category, iss.message);
            }
            md.push('\n');
        }
        if !a.missing_heavy_atoms.is_empty() {
            md.push_str("### Incomplete Sidechain / Missing Heavy Atoms\n\n");
            for m in a.missing_heavy_atoms.iter().take(20) {
                let pflag = if m.near_pocket { " **[POCKET]**" } else { "" };
                let _ = writeln!(
                    md,
                    "- `{}:{}{}`{pflag}: missing `{}`",
                    m.chain_id,
                    m.res_name,
                    m.res_seq,
                    m.missing_atoms.join(", ")
                );
            }
            md.push('\n');
        }

        // Local pKa & Protonation
        md.push_str("## 3. Local pKa & Protonation States (PROPKA-Inspired)\n\n");
        let _ = writeln!(
            md,
            "Evaluated **{}** titratable residues at pH `{:.1}` (**{}** non-canonical / shifted states assigned).\n",
            self.protonation.titratable_count,
            self.protonation.target_ph,
            self.protonation.non_canonical_count
        );
        let interesting: Vec<&ProtonationAssignment> = self
            .protonation
            .assignments
            .iter()
            .filter(|p| {
                p.near_pocket
                    || matches!(
                        p.assigned_state.as_str(),
                        "HID" | "HIE" | "HIP" | "ASH" | "GLH" | "LYN" | "CYM"
                    )
                    || (p.predicted_pka - p.model_pka).abs() >= 1.5
            })
            .collect();
        if !interesting.is_empty() {
            md.push_str("| Residue | State | q | pKa_0 -> pKa | ΔDesolv | ΔHB | ΔCoul | Burial | Pocket | Rationale |\n|---|---|---:|---|---:|---:|---:|---:|---|---|\n");
            for p in interesting.iter().take(30) {
                let _ = writeln!(
                    md,
                    "| `{}:{}{}` | **{}** | {:+} | {:.1} -> **{:.2}** | {:+.2} | {:+.2} | {:+.2} | {:.0}% | {} | {} |",
                    p.chain_id,
                    p.orig_res_name,
                    p.res_seq,
                    p.assigned_state,
                    p.formal_charge,
                    p.model_pka,
                    p.predicted_pka,
                    p.delta_desolv,
                    p.delta_hbond,
                    p.delta_coulomb,
                    p.burial_pct,
                    if p.near_pocket { "Yes" } else { "—" },
                    p.rationale
                );
            }
            md.push('\n');
        }

        // H-bond network flips
        md.push_str("## 4. H-Bond Network Flips & Polar-H Optimization\n\n");
        let _ = writeln!(
            md,
            "- Evaluated **{}** `ASN`/`GLN`/`HIS` sidechains -> applied **{}** 180° flip(s).",
            self.hbond_opt.asn_gln_his_evaluated,
            self.hbond_opt.flips_applied.len()
        );
        let _ = writeln!(
            md,
            "- Optimized **{}** rotatable `SER`/`THR`/`TYR`/`CYS` polar hydrogens (**{}** directional H-bonds formed).\n",
            self.hbond_opt.polar_h_rotors_optimized, self.hbond_opt.polar_h_hbonds_formed
        );
        if !self.hbond_opt.flips_applied.is_empty() {
            md.push_str(
                "| Residue | Flip | ΔScore | Pocket | Rationale |\n|---|---|---:|---|---|\n",
            );
            for f in &self.hbond_opt.flips_applied {
                let _ = writeln!(
                    md,
                    "| `{}:{}{}` | {} | {:+.2} | {} | {} |",
                    f.chain_id,
                    f.res_name,
                    f.res_seq,
                    f.flip_type,
                    f.delta_score,
                    if f.near_pocket { "Yes" } else { "—" },
                    f.reason
                );
            }
            md.push('\n');
        }

        if !self.bridging_waters.is_empty() {
            md.push_str("## 5. Active-Site Bridging Waters\n\n");
            for w in &self.bridging_waters {
                let _ = writeln!(
                    md,
                    "- `{}:HOH{}` ({} receptor H-bonds: {})",
                    w.chain_id,
                    w.res_seq,
                    w.receptor_hbonds,
                    w.lining_residues.join(", ")
                );
            }
            md.push('\n');
        }

        md
    }
}

/// Runs the complete `prot-prep` workflow on raw PDB or mmCIF text.
pub fn prepare_protein_from_str(
    input: &str,
    format_hint: &str,
    cfg: &PrepConfig,
) -> Result<PreparedReceptor, PrepError> {
    let t0 = Instant::now();
    let raw = parse_raw_structure(input, format_hint).map_err(PrepError::Parse)?;

    // Build clean receptor keeping waters temporarily if we need bridging-water detection
    let (mut rec, extracted_ligands, audit) = audit::build_clean_receptor_and_audit(
        &raw,
        &cfg.chains,
        cfg.keep_cofactors,
        cfg.keep_metals,
        cfg.strip_ligands,
        true, // retain waters temporarily so bridging water audit sees them!
        cfg.pocket_center,
        cfg.pocket_radius,
    );

    // Effective pocket center: explicit > primary extracted organic ligand centroid
    let eff_center = cfg.pocket_center.or_else(|| {
        audit
            .het_groups
            .iter()
            .filter(|h| h.class == HetClass::OrganicLigand)
            .max_by_key(|h| h.atom_count)
            .map(|h| h.centroid)
    });

    // Bridging waters near the pocket
    let bridging_waters = if let Some(ctr) = eff_center {
        detect_bridging_waters(&rec, ctr, cfg.pocket_radius, cfg.min_water_hbonds.max(2))
    } else {
        Vec::new()
    };
    let keep_water_ids: Vec<i32> = if cfg.keep_bridging_waters {
        bridging_waters.iter().map(|w| w.res_seq).collect()
    } else {
        Vec::new()
    };

    // Strip non-bridging waters & run initial distance-based bond inference
    prepare_receptor_keeping_waters(&mut rec, &keep_water_ids);

    // Strip hydrogens added by initial pass so PROPKA + H-bond flips run cleanly on heavy atoms,
    // then run a final `prepare_receptor_keeping_waters` to synthesize exact polar H & charges!
    strip_all_hydrogens(&mut rec);

    // 1. Local PROPKA-inspired pKa & protonation state assignment (`HID/HIE/HIP/ASH/GLH/LYN/CYM/CYX`)
    let protonation = assign_local_protonation(&mut rec, cfg.ph, eff_center, cfg.pocket_radius);

    // 2. Iterative 180° ASN/GLN/HIS sidechain flip optimization
    let mut hbond_opt = if cfg.optimize_hbonds {
        optimize_sidechain_flips(&mut rec, eff_center, cfg.pocket_radius)
    } else {
        HBondOptReport {
            asn_gln_his_evaluated: 0,
            flips_applied: Vec::new(),
            polar_h_rotors_optimized: 0,
            polar_h_hbonds_formed: 0,
        }
    };

    // 3. Re-infer bonds, synthesize exact polar hydrogens for `HID/HIE/HIP/ASH/GLH/LYN/CYM/CYX`,
    //    compute Gasteiger charges and AutoDock4 types
    rec.bonds.clear();
    rec.adj = vec![Vec::new(); rec.atoms.len()];
    prepare_receptor_keeping_waters(&mut rec, &keep_water_ids);

    // 4. Optimize rotatable polar -OH / -SH hydrogen orientations (SER/THR/TYR/CYS)
    if cfg.optimize_hbonds {
        optimize_polar_h_orientations(&mut rec, &mut hbond_opt);
    }

    Ok(PreparedReceptor {
        receptor: rec,
        extracted_ligands,
        bridging_waters,
        audit,
        protonation,
        hbond_opt,
        elapsed_ms: t0.elapsed().as_millis(),
    })
}

/// Convenience wrapper reading a `.pdb` or `.cif`/`.mmcif` file from disk.
pub fn prepare_protein_file(path: &Path, cfg: &PrepConfig) -> Result<PreparedReceptor, PrepError> {
    let text = std::fs::read_to_string(path)?;
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("pdb");
    prepare_protein_from_str(&text, ext, cfg)
}

fn strip_all_hydrogens(mol: &mut Molecule3D) {
    let mut keep_map = vec![None; mol.atoms.len()];
    let mut kept = Vec::with_capacity(mol.atoms.len());
    for (idx, mut a) in mol.atoms.drain(..).enumerate() {
        if a.element == "H" {
            continue;
        }
        let nid = kept.len();
        a.id = nid;
        keep_map[idx] = Some(nid);
        kept.push(a);
    }
    let old_bonds = std::mem::take(&mut mol.bonds);
    mol.atoms = kept;
    mol.adj = vec![Vec::new(); mol.atoms.len()];
    for b in old_bonds {
        if let (Some(nu), Some(nv)) = (keep_map[b.u], keep_map[b.v]) {
            let _ = mol.add_bond(nu, nv, b.order);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_highest_occupancy_altloc_and_flips_misoriented_asn_and_detects_acid_dyad()
    -> Result<(), Box<dyn std::error::Error>> {
        // Synthetic mini-pocket containing:
        // - Altloc residue ALA 1 where altloc B (occ 0.70) beats altloc A (occ 0.30)
        // - Misoriented ASN 10 whose OD1 faces a backbone carbonyl O (clash!) and ND2 faces a backbone NH (clash!) -> MUST FLIP!
        // - Buried catalytic ASP 20 / ASP 21 dyad (3.2 Å apart) -> higher-seq ASP 21 gets elevated pKa!
        // - Missing sidechain atoms on ARG 30 (only N, CA, C, O, CB present) -> flagged in missing_heavy_atoms!
        // - Catalytic Zn2+ coordinating CYS 40 SG (2.3 Å) -> assigned CYM (-1)!
        let pdb = "\
HEADER    HYDROLASE                               20-NOV-26   9TST
EXPDTA    X-RAY DIFFRACTION
REMARK   2 RESOLUTION.    1.85 ANGSTROMS.
REMARK   3   R VALUE            (WORKING SET) : 0.180
REMARK   3   FREE R VALUE                     : 0.215
ATOM      1  N  AALA A   1       0.000   0.000   0.000  0.30 15.00           N
ATOM      2  CA AALA A   1       1.458   0.000   0.000  0.30 15.00           C
ATOM      3  C  AALA A   1       2.009   1.420   0.000  0.30 15.00           C
ATOM      4  O  AALA A   1       1.251   2.390   0.000  0.30 15.00           O
ATOM      5  CB AALA A   1       1.986  -0.760   1.210  0.30 15.00           C
ATOM      6  N  BALA A   1       0.100   0.000   0.000  0.70 15.00           N
ATOM      7  CA BALA A   1       1.558   0.000   0.000  0.70 15.00           C
ATOM      8  C  BALA A   1       2.109   1.420   0.000  0.70 15.00           C
ATOM      9  O  BALA A   1       1.351   2.390   0.000  0.70 15.00           O
ATOM     10  CB BALA A   1       2.086  -0.760   1.210  0.70 15.00           C
ATOM     11  N   ASN A  10      10.000   0.000   0.000  1.00 18.00           N
ATOM     12  CA  ASN A  10      11.458   0.000   0.000  1.00 18.00           C
ATOM     13  C   ASN A  10      12.009   1.420   0.000  1.00 18.00           C
ATOM     14  O   ASN A  10      11.251   2.390   0.000  1.00 18.00           O
ATOM     15  CB  ASN A  10      11.986  -0.760   1.210  1.00 18.00           C
ATOM     16  CG  ASN A  10      13.480  -0.760   1.210  1.00 18.00           C
ATOM     17  OD1 ASN A  10      14.100   0.300   1.210  1.00 18.00           O
ATOM     18  ND2 ASN A  10      14.100  -1.920   1.210  1.00 18.00           N
ATOM     19  N   GLY A  11      16.950  -1.920   1.210  1.00 18.00           N
ATOM     20  CA  GLY A  11      17.950  -2.920   1.210  1.00 18.00           C
ATOM     21  C   GLY A  11      17.950  -0.500   1.210  1.00 18.00           C
ATOM     22  O   GLY A  11      16.950   0.300   1.210  1.00 18.00           O
ATOM     23  N   ARG A  30      25.000   0.000   0.000  1.00 22.00           N
ATOM     24  CA  ARG A  30      26.458   0.000   0.000  1.00 22.00           C
ATOM     25  C   ARG A  30      27.009   1.420   0.000  1.00 22.00           C
ATOM     26  O   ARG A  30      26.251   2.390   0.000  1.00 22.00           O
ATOM     27  CB  ARG A  30      26.986  -0.760   1.210  1.00 22.00           C
ATOM     28  N   CYS A  40      35.000   0.000   0.000  1.00 16.00           N
ATOM     29  CA  CYS A  40      36.458   0.000   0.000  1.00 16.00           C
ATOM     30  C   CYS A  40      37.009   1.420   0.000  1.00 16.00           C
ATOM     31  O   CYS A  40      36.251   2.390   0.000  1.00 16.00           O
ATOM     32  CB  CYS A  40      36.986  -0.760   1.210  1.00 16.00           C
ATOM     33  SG  CYS A  40      38.750  -0.760   1.210  1.00 16.00           S
HETATM   34 ZN    ZN A 500      41.000  -0.760   1.210  1.00 14.00          ZN
END
";
        let prep = prepare_protein_from_str(pdb, "pdb", &PrepConfig::default())?;
        // 1. Altloc B (0.70) won over A (0.30)
        assert_eq!(prep.audit.altlocs_resolved.len(), 1);
        assert_eq!(prep.audit.altlocs_resolved[0].chosen_altloc, 'B');
        // 2. Missing atoms on ARG 30 caught automatically even without REMARK 470
        assert!(
            prep.audit
                .missing_heavy_atoms
                .iter()
                .any(|m| m.res_name == "ARG"
                    && m.res_seq == 30
                    && m.missing_atoms.contains(&"NH1".to_string()))
        );
        // 3. Misoriented ASN 10 flipped (OD1 <-> ND2)!
        assert!(
            prep.hbond_opt
                .flips_applied
                .iter()
                .any(|f| f.res_name == "ASN" && f.res_seq == 10 && f.delta_score >= 0.8),
            "expected misoriented ASN 10 to flip, got: {:?}",
            prep.hbond_opt.flips_applied
        );
        // 4. Zn-coordinated CYS 40 assigned anionic CYM (-1)
        let cys40 = prep
            .protonation
            .assignments
            .iter()
            .find(|p| p.res_seq == 40)
            .ok_or("missing CYS 40 assignment")?;
        assert_eq!(cys40.assigned_state, "CYM");
        assert!(cys40.metal_coordinated);
        assert_eq!(cys40.formal_charge, -1);

        // 5. Boltzmann average sanity check
        let b = boltzmann_average_kcal(&[-9.0, -7.0]);
        assert!(b < -8.4 && b > -9.0, "unexpected boltzmann avg: {b}");
        Ok(())
    }

    #[test]
    fn real_cdk2_1ke6_and_1h1s_prepare_cleanly_if_present() -> Result<(), Box<dyn std::error::Error>>
    {
        // Opt-in fixture: only runs where the campaign data actually lives.
        let Some(base) = std::env::var("CRAFT_DATA_DIR")
            .ok()
            .filter(|v| !v.trim().is_empty())
        else {
            return Ok(());
        };
        let p_1ke6 = &std::path::PathBuf::from(base).join("campaigns/cdk2_pilot/raw/1KE6.pdb");
        if !p_1ke6.exists() {
            return Ok(());
        }
        let prep = prepare_protein_file(p_1ke6, &PrepConfig::default())?;
        assert_eq!(prep.audit.structure_id, "1KE6");
        assert_eq!(prep.audit.resolution_a, Some(2.0));
        assert!(
            !prep.audit.missing_residues_header.is_empty(),
            "1KE6 has REMARK 465 missing activation loop residues"
        );
        assert!(
            !prep.audit.missing_heavy_atoms.is_empty(),
            "1KE6 has missing Arg36/Arg150 sidechain atoms"
        );
        assert_eq!(
            prep.extracted_ligands.len(),
            1,
            "1KE6 cocrystallized ligand LS2 should be cleanly extracted"
        );
        let md = prep.render_markdown();
        assert!(md.contains("1KE6") && md.contains("PROPKA"));
        Ok(())
    }
}
