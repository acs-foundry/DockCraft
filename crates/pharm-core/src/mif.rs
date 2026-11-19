#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet, VecDeque};

use dock_engine::{DockingConstraint, PharmacophoreKind};
use pocket_detect::GridBox;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use struct_core::{Molecule3D, cross3};

fn is_water_res_name(rn: &str) -> bool {
    matches!(
        rn.trim().to_ascii_uppercase().as_str(),
        "HOH" | "WAT" | "DOD" | "SOL" | "TIP3" | "H2O"
    )
}

use crate::kabsch::symmetric_eigen_3x3;
use crate::screen::extract_ligand_features;

/// Provenance of a binding-site pharmacophore feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureSource {
    /// Derived purely from apo receptor 3D Molecular Interaction Fields (MIFs).
    ApoMif,
    /// Apo MIF hotspot confirmed and refined by a cocrystallized holo ligand interaction.
    HybridApoHolo,
    /// Crystallographic holo interaction promoted directly from a reference ligand.
    HoloValidated,
    /// Manually added or edited by the user in GUI/CLI.
    UserDefined,
}

/// A single actionable pharmacophore hotspot inside the active site.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteFeature {
    pub id: String,
    pub kind: PharmacophoreKind,
    pub center: [f64; 3],
    pub radius_a: f64,
    /// Preferred unit interaction vector pointing from the ligand feature toward the receptor partner
    /// (for H-bonds, metal bonds, or aromatic ring normal).
    pub direction: Option<[f64; 3]>,
    /// Negative kcal/mol strength (more negative = stronger hotspot).
    pub energy_kcal: f64,
    /// Primary receptor residue responsible (e.g. `A:LEU83 O`).
    pub key_residue: String,
    /// Relative weight used during pharmacophore screening and soft-guided docking.
    pub weight: f64,
    pub enabled: bool,
    pub required: bool,
    pub source: FeatureSource,
}

/// Optimal inter-feature 3D distance (and optional vector angle) defining binding-site geometry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureDistanceConstraint {
    pub from_id: String,
    pub to_id: String,
    pub from_kind: PharmacophoreKind,
    pub to_kind: PharmacophoreKind,
    pub ideal_distance_a: f64,
    pub min_distance_a: f64,
    pub max_distance_a: f64,
    /// Angle (degrees) between directional vectors when both features are directional.
    pub vector_angle_deg: Option<f64>,
}

/// Receptor wall sphere used to reject steric clashes during 3D pharmacophore alignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExclusionSphere {
    pub center: [f64; 3],
    pub radius_a: f64,
    pub residue: String,
}

/// Classification of a crystallographic water inside the binding site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WaterRole {
    /// Forms >= 2 H-bonds to receptor atoms: structural bridge, recommended to retain.
    ConservedBridge,
    /// Weakly tethered (<= 1 H-bond) in pocket: candidate to displace for entropic gain.
    Displaceable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaterAssessment {
    pub chain_id: char,
    pub res_seq: i32,
    pub coords: [f64; 3],
    pub receptor_hbonds: usize,
    pub lining_residues: Vec<String>,
    pub role: WaterRole,
}

/// Physical dimensions and ligand-sizing recommendations derived from accessible cavity voxels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteGeometry {
    pub center: [f64; 3],
    pub accessible_volume_a3: f64,
    /// PCA principal spans `[Length, Width, Depth]` (Å, 2nd–98th percentile along principal axes).
    pub principal_dims_a: [f64; 3],
    /// Unit principal axes (rows: major, medium, minor).
    pub principal_axes: [[f64; 3]; 3],
    pub hydrophobic_fraction: f64,
    pub polar_fraction: f64,
    /// Mecozzi-Rebek optimal cavity occupancy (~45–70%) translated to ligand MW range (Da).
    pub recommended_mw_range: [f64; 2],
    pub recommended_heavy_atoms: [usize; 2],
    pub recommended_hbd_max: usize,
    pub recommended_hba_max: usize,
    pub has_catalytic_metal: bool,
}

/// Complete active-site interaction blueprint (`.pharm.json`), bridging apo/holo site analysis,
/// 3D pharmacophore screening (`pharm-screen`), `DockCraft` restraints, and `ChimeraX` overlays.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteBlueprint {
    pub format: String,
    pub version: String,
    pub target_name: String,
    pub grid_box: GridBox,
    pub geometry: SiteGeometry,
    pub features: Vec<SiteFeature>,
    pub distance_matrix: Vec<FeatureDistanceConstraint>,
    pub exclusion_spheres: Vec<ExclusionSphere>,
    pub waters: Vec<WaterAssessment>,
    pub mif_spacing_a: f64,
    pub cavity_voxel_count: usize,
    pub elapsed_ms: u128,
}

impl SiteBlueprint {
    /// Converts all currently enabled blueprint features into soft `DockingConstraint::PharmacophoreSphere`
    /// restraints ready to pass directly into `dock_engine::DockingConfig::constraints`.
    #[must_use]
    pub fn to_docking_constraints(
        &self,
        max_restraints: usize,
        force_scale: f64,
    ) -> Vec<DockingConstraint> {
        let mut active: Vec<&SiteFeature> = self.features.iter().filter(|f| f.enabled).collect();
        active.sort_by(|a, b| {
            b.required
                .cmp(&a.required)
                .then_with(|| a.energy_kcal.total_cmp(&b.energy_kcal))
        });
        active
            .into_iter()
            .take(max_restraints.max(1))
            .map(|f| DockingConstraint::PharmacophoreSphere {
                center: f.center,
                radius: (f.radius_a * 1.15).clamp(1.2, 2.6),
                kind: f.kind,
                weight: (f.weight * force_scale).clamp(0.4, 4.0),
            })
            .collect()
    }

    /// Rebuilds the optimal inter-feature distance matrix for all currently enabled features.
    pub fn rebuild_distance_matrix(&mut self) {
        self.distance_matrix = build_distance_matrix(&self.features);
    }

    /// Serializes this blueprint to formatted `.pharm.json`.
    pub fn to_json_pretty(&self) -> Result<String, crate::PharmError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| crate::PharmError::Serialization(e.to_string()))
    }

    /// Parses a `.pharm.json` blueprint from JSON text.
    pub fn from_json_str(raw: &str) -> Result<Self, crate::PharmError> {
        serde_json::from_str(raw).map_err(|e| crate::PharmError::Serialization(e.to_string()))
    }

    /// Renders a clean, human-readable Markdown *Target Binding Blueprint* summarizing dimensions,
    /// optimal pharmacophore features, inter-feature distances, and water/metal strategy.
    #[must_use]
    pub fn render_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str(&format!(
            "# Target Binding Blueprint: {}\n\n",
            self.target_name
        ));
        let g = &self.geometry;
        md.push_str("## 1. Active-Site Physical Dimensions & Ligand Sizing\n\n");
        md.push_str(&format!(
            "- **Pocket Center:** `({:.2}, {:.2}, {:.2}) Å`\n",
            g.center[0], g.center[1], g.center[2]
        ));
        md.push_str(&format!(
            "- **Accessible Volume:** `{:.0} Å³` ({} cavity voxels at {:.2} Å spacing)\n",
            g.accessible_volume_a3, self.cavity_voxel_count, self.mif_spacing_a
        ));
        md.push_str(&format!(
            "- **Principal Dimensions (L × W × D):** `{:.1} × {:.1} × {:.1} Å`\n",
            g.principal_dims_a[0], g.principal_dims_a[1], g.principal_dims_a[2]
        ));
        md.push_str(&format!(
            "- **Surface Character:** `{:.0}%` hydrophobic / `{:.0}%` polar{}\n",
            g.hydrophobic_fraction * 100.0,
            g.polar_fraction * 100.0,
            if g.has_catalytic_metal {
                " (**catalytic metal present**)"
            } else {
                ""
            }
        ));
        md.push_str(&format!(
            "- **Optimal Ligand Envelope (Mecozzi-Rebek 45–70% fill):** MW `{:.0}–{:.0} Da` ({}–{} heavy atoms), HBD ≤ `{}`, HBA ≤ `{}`\n\n",
            g.recommended_mw_range[0],
            g.recommended_mw_range[1],
            g.recommended_heavy_atoms[0],
            g.recommended_heavy_atoms[1],
            g.recommended_hbd_max,
            g.recommended_hba_max
        ));

        md.push_str("## 2. Optimal Pharmacophore Features (Hotspots)\n\n");
        md.push_str("| ID | Type | Active | Source | Key Residue | Center (Å) | Radius | Score (kcal/mol) |\n");
        md.push_str("|---|---|---:|---|---|---|---:|---:|\n");
        for f in &self.features {
            md.push_str(&format!(
                "| `{}` | `{:?}` | {} | `{:?}` | `{}` | `({:.2}, {:.2}, {:.2})` | `{:.2} Å` | `{:.2}` |\n",
                f.id,
                f.kind,
                if f.enabled { "Yes" } else { "No" },
                f.source,
                f.key_residue,
                f.center[0],
                f.center[1],
                f.center[2],
                f.radius_a,
                f.energy_kcal
            ));
        }

        md.push_str("\n## 3. Optimal Inter-Feature Distance Matrix\n\n");
        md.push_str("| Pair | Types | Ideal Distance | Allowed Range | Vector Angle |\n");
        md.push_str("|---|---|---:|---:|---:|\n");
        for d in &self.distance_matrix {
            let ang = d
                .vector_angle_deg
                .map_or_else(|| "—".to_string(), |a| format!("{a:.0}°"));
            md.push_str(&format!(
                "| `{}` ↔ `{}` | `{:?}` ↔ `{:?}` | **{:.2} Å** | `{:.2}–{:.2} Å` | {} |\n",
                d.from_id,
                d.to_id,
                d.from_kind,
                d.to_kind,
                d.ideal_distance_a,
                d.min_distance_a,
                d.max_distance_a,
                ang
            ));
        }

        if !self.waters.is_empty() {
            md.push_str("\n## 4. Active-Site Water Assessment\n\n");
            for w in &self.waters {
                md.push_str(&format!(
                    "- `HOH {}:{}` — **{:?}** ({} receptor H-bonds: `{}`)\n",
                    w.chain_id,
                    w.res_seq,
                    w.role,
                    w.receptor_hbonds,
                    w.lining_residues.join(", ")
                ));
            }
        }
        md.push_str(&format!(
            "\n*Generated by DockCraft `pharm-core` in {} ms ({} exclusion spheres).*\n",
            self.elapsed_ms,
            self.exclusion_spheres.len()
        ));
        md
    }

    /// Generates UCSF ChimeraX commands (`shape sphere` / `shape cone`) to visualize the active-site
    /// blueprint features directly inside ChimeraX.
    #[must_use]
    pub fn chimerax_overlay_commands(&self) -> String {
        let mut out = String::from("# DockCraft pharm-core active-site pharmacophore overlay\n");
        for f in self.features.iter().filter(|f| f.enabled) {
            let color = match f.kind {
                PharmacophoreKind::HDonor => "dodgerblue",
                PharmacophoreKind::HAcceptor => "crimson",
                PharmacophoreKind::Hydrophobic => "gold",
                PharmacophoreKind::Aromatic => "darkorange",
                PharmacophoreKind::PositiveIonizable => "royalblue",
                PharmacophoreKind::NegativeIonizable => "deeppink",
                PharmacophoreKind::MetalChelate => "mediumpurple",
                PharmacophoreKind::AnyHeavy => "lightgray",
            };
            out.push_str(&format!(
                "shape sphere center {:.3},{:.3},{:.3} radius {:.2} color {} transparency 55 name pharm_{}\n",
                f.center[0], f.center[1], f.center[2], f.radius_a, color, f.id
            ));
        }
        out
    }
}

/// Configuration for 3D active-site MIF profiling & pharmacophore blueprint extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteAnalysisConfig {
    /// MIF grid spacing in Å (default `0.65` Å; capped automatically if box is huge).
    pub spacing_a: f64,
    /// Maximum total pharmacophore features emitted in the blueprint (default `10`).
    pub max_features: usize,
    /// Maximum features per probe kind before global ranking (default `3`).
    pub max_per_kind: usize,
    /// Maximum receptor wall exclusion spheres (default `120`).
    pub max_exclusion_spheres: usize,
}

impl Default for SiteAnalysisConfig {
    fn default() -> Self {
        Self {
            spacing_a: 0.65,
            max_features: 10,
            max_per_kind: 3,
            max_exclusion_spheres: 120,
        }
    }
}

#[inline]
fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[inline]
fn norm3(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l > 1e-8 {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        [0.0, 0.0, 0.0]
    }
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vdw_radius_wall(el: &str) -> f64 {
    match el {
        "C" => 1.70,
        "N" => 1.55,
        "O" => 1.52,
        "S" => 1.80,
        "P" => 1.80,
        "F" => 1.47,
        "Cl" => 1.75,
        "Br" => 1.85,
        "I" => 1.98,
        "Zn" => 1.39,
        "Mg" => 1.73,
        "Fe" | "Mn" => 1.40,
        "Ca" => 1.94,
        _ => 1.70,
    }
}

struct RecAtomInfo {
    coords: [f64; 3],
    label: String,
    h_donor: bool,
    h_acceptor: bool,
    hydrophobic: bool,
    formal_charge: i8,
    is_metal: bool,
    metal_r_opt: f64,
    is_backbone_polar: bool,
    vdw_r: f64,
    /// Unit donor H-direction(s) (D -> H) for H-bond acceptor probe directionality.
    donor_dirs: Vec<[f64; 3]>,
    /// Unit lone-pair extension axis (away from bonded heavy neighbors) for H-bond donor probe.
    acceptor_dir: [f64; 3],
}

struct RecAromRing {
    center: [f64; 3],
    normal: [f64; 3],
    label: String,
}

struct SpatialHash {
    cell: f64,
    buckets: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl SpatialHash {
    fn new(points: &[[f64; 3]], cell: f64) -> Self {
        let mut buckets: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for (i, &p) in points.iter().enumerate() {
            buckets.entry(Self::key(p, cell)).or_default().push(i);
        }
        Self { cell, buckets }
    }

    #[inline]
    fn key(p: [f64; 3], s: f64) -> (i32, i32, i32) {
        (
            (p[0] / s).floor() as i32,
            (p[1] / s).floor() as i32,
            (p[2] / s).floor() as i32,
        )
    }

    fn query_into(&self, points: &[[f64; 3]], pos: [f64; 3], radius: f64, out: &mut Vec<usize>) {
        out.clear();
        let span = (radius / self.cell).ceil() as i32;
        let (cx, cy, cz) = Self::key(pos, self.cell);
        let r2 = radius * radius;
        for dx in -span..=span {
            for dy in -span..=span {
                for dz in -span..=span {
                    if let Some(bucket) = self.buckets.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &idx in bucket {
                            let p = points[idx];
                            let ddx = p[0] - pos[0];
                            let ddy = p[1] - pos[1];
                            let ddz = p[2] - pos[2];
                            if ddx * ddx + ddy * ddy + ddz * ddz <= r2 {
                                out.push(idx);
                            }
                        }
                    }
                }
            }
        }
    }
}

fn prepare_receptor_context(
    receptor: &Molecule3D,
    grid_box: &GridBox,
) -> (
    Vec<RecAtomInfo>,
    Vec<[f64; 3]>,
    Vec<RecAromRing>,
    Vec<WaterAssessment>,
) {
    let lo = grid_box.min_corner();
    let hi = grid_box.max_corner();
    let margin = 9.0;

    let mut rec_infos = Vec::new();
    let mut rec_coords = Vec::new();
    let mut water_candidates: Vec<(char, i32, [f64; 3])> = Vec::new();

    for (idx, a) in receptor.atoms.iter().enumerate() {
        if !a.is_heavy() {
            continue;
        }
        let p = a.coords;
        if p[0] < lo[0] - margin
            || p[0] > hi[0] + margin
            || p[1] < lo[1] - margin
            || p[1] > hi[1] + margin
            || p[2] < lo[2] - margin
            || p[2] > hi[2] + margin
        {
            continue;
        }
        if a.element == "O" && is_water_res_name(&a.res_name) {
            if grid_box.contains(p) {
                water_candidates.push((a.chain_id, a.res_seq, p));
            }
            continue;
        }

        // Precompute donor D->H directions and acceptor lone-pair extension axis
        let mut donor_dirs = Vec::new();
        let mut heavy_sum = [0.0; 3];
        let mut heavy_cnt = 0usize;
        if let Some(adj) = receptor.adj.get(idx) {
            for &(nb, _) in adj {
                let Some(na) = receptor.atoms.get(nb) else {
                    continue;
                };
                if na.element == "H" {
                    let d = norm3([
                        na.coords[0] - p[0],
                        na.coords[1] - p[1],
                        na.coords[2] - p[2],
                    ]);
                    if dot3(d, d) > 0.5 {
                        donor_dirs.push(d);
                    }
                } else {
                    let v = norm3([
                        na.coords[0] - p[0],
                        na.coords[1] - p[1],
                        na.coords[2] - p[2],
                    ]);
                    heavy_sum[0] += v[0];
                    heavy_sum[1] += v[1];
                    heavy_sum[2] += v[2];
                    heavy_cnt += 1;
                }
            }
        }
        let away_dir = if heavy_cnt > 0 {
            norm3([-heavy_sum[0], -heavy_sum[1], -heavy_sum[2]])
        } else {
            [0.0, 0.0, 0.0]
        };
        if a.h_donor && donor_dirs.is_empty() && dot3(away_dir, away_dir) > 0.5 {
            donor_dirs.push(away_dir);
        }

        let is_metal = a.ad4_type.is_metal();
        let metal_r_opt = match a.ad4_type {
            struct_core::Ad4Type::Ca => 2.35,
            struct_core::Ad4Type::Mg => 2.15,
            _ => 2.05,
        };
        let aname = a.name.trim().to_ascii_uppercase();
        let rname = a.res_name.trim().to_ascii_uppercase();
        let is_backbone_polar = a.is_backbone && (aname == "N" || aname == "O") && !a.is_hetatm;

        // Enforce exact protein pH 7.4 donor/acceptor/charge chemistry so PDB bond-order
        // ambiguities never turn backbone C=O or ASP/GLU carboxylates into spurious H-donors!
        let (eff_donor, eff_acceptor, eff_fc) = if !a.is_hetatm {
            match (rname.as_str(), aname.as_str()) {
                (_, "O" | "OXT") => (false, true, if aname == "OXT" { -1 } else { 0 }),
                ("PRO", "N") => (false, false, 0),
                (_, "N") => (true, false, 0),
                ("ASP", "OD1" | "OD2") | ("GLU", "OE1" | "OE2") => (false, true, -1),
                ("LYS", "NZ") | ("ARG", "NE" | "NH1" | "NH2") => (true, false, 1),
                ("ASN", "OD1") | ("GLN", "OE1") => (false, true, 0),
                ("ASN", "ND2") | ("GLN", "NE2") | ("TRP", "NE1") => (true, false, 0),
                ("SER", "OG") | ("THR", "OG1") | ("TYR", "OH") => (true, true, 0),
                ("HIS" | "HID" | "HIE", "ND1" | "NE2") => (true, true, 0),
                ("HIP", "ND1" | "NE2") => (true, false, 1),
                _ => (a.h_donor, a.h_acceptor, a.formal_charge),
            }
        } else {
            (a.h_donor, a.h_acceptor, a.formal_charge)
        };
        if eff_donor && donor_dirs.is_empty() && dot3(away_dir, away_dir) > 0.5 {
            donor_dirs.push(away_dir);
        }
        if !eff_donor {
            donor_dirs.clear();
        }

        let label = format!("{}:{}{} {}", a.chain_id, rname, a.res_seq, aname);
        rec_infos.push(RecAtomInfo {
            coords: p,
            label,
            h_donor: eff_donor,
            h_acceptor: eff_acceptor,
            hydrophobic: a.hydrophobic && !eff_donor && !eff_acceptor && eff_fc == 0,
            formal_charge: eff_fc,
            is_metal,
            metal_r_opt,
            is_backbone_polar,
            vdw_r: vdw_radius_wall(&a.element),
            donor_dirs,
            acceptor_dir: away_dir,
        });
        rec_coords.push(p);
    }

    // Extract aromatic sidechain rings (PHE/TYR/TRP/HIS or aromatic rings inside box+margin)
    let mut arom_rings = Vec::new();
    for ring in &receptor.rings {
        if !(5..=6).contains(&ring.len()) {
            continue;
        }
        if !ring.iter().all(|&i| {
            receptor.atoms.get(i).is_some_and(|a| {
                a.aromatic
                    || matches!(
                        a.res_name.trim(),
                        "PHE" | "TYR" | "TRP" | "HIS" | "HID" | "HIE" | "HIP"
                    )
            })
        }) {
            continue;
        }
        let k = ring.len() as f64;
        let mut cen = [0.0; 3];
        for &idx in ring {
            let c = receptor.atoms[idx].coords;
            cen[0] += c[0] / k;
            cen[1] += c[1] / k;
            cen[2] += c[2] / k;
        }
        if cen[0] < lo[0] - 7.0
            || cen[0] > hi[0] + 7.0
            || cen[1] < lo[1] - 7.0
            || cen[1] > hi[1] + 7.0
            || cen[2] < lo[2] - 7.0
            || cen[2] > hi[2] + 7.0
        {
            continue;
        }
        let mut nrm = [0.0; 3];
        for i in 0..ring.len() {
            let pa = receptor.atoms[ring[i]].coords;
            let pb = receptor.atoms[ring[(i + 1) % ring.len()]].coords;
            let cr = cross3(
                [pa[0] - cen[0], pa[1] - cen[1], pa[2] - cen[2]],
                [pb[0] - cen[0], pb[1] - cen[1], pb[2] - cen[2]],
            );
            nrm[0] += cr[0];
            nrm[1] += cr[1];
            nrm[2] += cr[2];
        }
        let normal = norm3(nrm);
        if dot3(normal, normal) > 0.5 {
            let a0 = &receptor.atoms[ring[0]];
            arom_rings.push(RecAromRing {
                center: cen,
                normal,
                label: format!("{}:{}{}", a0.chain_id, a0.res_name.trim(), a0.res_seq),
            });
        }
    }

    // Assess crystallographic waters inside the box (ConservedBridge vs Displaceable)
    let mut waters = Vec::new();
    for (chain_id, res_seq, wpos) in water_candidates {
        let mut hb = 0usize;
        let mut lining = Vec::new();
        for ra in &rec_infos {
            if !ra.h_donor && !ra.h_acceptor && !ra.is_metal {
                continue;
            }
            let d = dist3(wpos, ra.coords);
            if (2.1..=3.45).contains(&d) {
                hb += 1;
                if !lining.contains(&ra.label) {
                    lining.push(ra.label.clone());
                }
            }
        }
        let role = if hb >= 2 {
            WaterRole::ConservedBridge
        } else {
            WaterRole::Displaceable
        };
        waters.push(WaterAssessment {
            chain_id,
            res_seq,
            coords: wpos,
            receptor_hbonds: hb,
            lining_residues: lining,
            role,
        });
    }
    waters.sort_by_key(|a| std::cmp::Reverse(a.receptor_hbonds));

    (rec_infos, rec_coords, arom_rings, waters)
}

#[derive(Clone)]
struct RawProbePeak {
    kind: PharmacophoreKind,
    center: [f64; 3],
    direction: Option<[f64; 3]>,
    energy_kcal: f64,
    key_residue: String,
}

/// Evaluates 7-probe 3D Molecular Interaction Fields across the binding pocket, computes PCA
/// pocket geometry, extracts apo/holo pharmacophore hotspots, builds the optimal inter-feature
/// distance matrix, and generates receptor wall exclusion spheres.
pub fn analyze_binding_site(
    receptor: &Molecule3D,
    grid_box: &GridBox,
    holo_ligands: &[&Molecule3D],
    config: &SiteAnalysisConfig,
) -> Result<SiteBlueprint, crate::PharmError> {
    let t0 = std::time::Instant::now();
    if receptor.heavy_atom_count() == 0 {
        return Err(crate::PharmError::EmptyStructure);
    }

    let (rec_infos, rec_coords, arom_rings, waters) = prepare_receptor_context(receptor, grid_box);
    if rec_infos.is_empty() {
        return Err(crate::PharmError::NoPocketAtoms);
    }
    let has_catalytic_metal = rec_infos.iter().any(|r| r.is_metal);
    let hash = SpatialHash::new(&rec_coords, 3.5);

    // Choose safe MIF spacing so voxel count stays bounded (<= 220k voxels)
    let mut h = config.spacing_a.clamp(0.45, 1.25);
    loop {
        let nx = ((grid_box.size[0] / h).floor() as usize) + 1;
        let ny = ((grid_box.size[1] / h).floor() as usize) + 1;
        let nz = ((grid_box.size[2] / h).floor() as usize) + 1;
        if nx.saturating_mul(ny).saturating_mul(nz) <= 220_000 || h >= 1.5 {
            break;
        }
        h *= 1.12;
    }

    let lo = grid_box.min_corner();
    let nx = ((grid_box.size[0] / h).floor() as usize) + 1;
    let ny = ((grid_box.size[1] / h).floor() as usize) + 1;
    let nz = ((grid_box.size[2] / h).floor() as usize) + 1;

    // 6 cardinal + 8 diagonal directions for burial check
    const BURIAL_DIRS: [[f64; 3]; 14] = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [0.577, 0.577, 0.577],
        [-0.577, 0.577, 0.577],
        [0.577, -0.577, 0.577],
        [-0.577, -0.577, 0.577],
        [0.577, 0.577, -0.577],
        [-0.577, 0.577, -0.577],
        [0.577, -0.577, -0.577],
        [-0.577, -0.577, -0.577],
    ];

    struct SliceResult {
        cavity_pts: Vec<([usize; 3], [f64; 3], bool)>, // (grid_ijk, coord, is_hydrophobic_env)
        peaks: Vec<([usize; 3], RawProbePeak)>,
    }

    // Effective radial envelope around the active site so huge auto-boxes do not leak into
    // unrelated surface pockets or distant buried internal voids.
    let holo_atoms: Vec<[f64; 3]> = holo_ligands
        .iter()
        .flat_map(|l| l.atoms.iter().filter(|a| a.is_heavy()).map(|a| a.coords))
        .collect();
    let max_apo_r =
        (0.5 * grid_box.size[0].min(grid_box.size[1]).min(grid_box.size[2])).clamp(7.5, 11.0);

    let slices: Vec<SliceResult> = (0..nx)
        .into_par_iter()
        .map(|ix| {
            let x = lo[0] + ix as f64 * h;
            let mut near_6 = Vec::with_capacity(64);
            let mut near_9 = Vec::with_capacity(160);
            let mut cavity_pts = Vec::new();
            let mut peaks = Vec::new();

            for iy in 0..ny {
                let y = lo[1] + iy as f64 * h;
                for iz in 0..nz {
                    let z = lo[2] + iz as f64 * h;
                    let pos = [x, y, z];
                    if !holo_atoms.is_empty() {
                        if !holo_atoms.iter().any(|&hp| dist3(pos, hp) <= 4.5) {
                            continue;
                        }
                    } else if dist3(pos, grid_box.center) > max_apo_r {
                        continue;
                    }
                    hash.query_into(&rec_coords, pos, 6.0, &mut near_6);
                    if near_6.len() < 3 {
                        continue;
                    }

                    // Check steric clearance
                    let mut clash = false;
                    let mut min_d = 99.0_f64;
                    let mut near_hyd = 0usize;
                    let mut near_pol = 0usize;
                    for &ri in &near_6 {
                        let rinfo = &rec_infos[ri];
                        let d = dist3(pos, rinfo.coords);
                        let min_allow = if rinfo.is_metal { 1.80 } else { 2.45 };
                        if d < min_allow {
                            clash = true;
                            break;
                        }
                        if d < min_d {
                            min_d = d;
                        }
                        if d <= 4.8 {
                            if rinfo.hydrophobic {
                                near_hyd += 1;
                            }
                            if rinfo.h_donor || rinfo.h_acceptor || rinfo.formal_charge != 0 {
                                near_pol += 1;
                            }
                        }
                    }
                    if clash || min_d > 4.8 {
                        continue;
                    }

                    // Check pocket enclosure (>= 5 of 14 cones contain a receptor atom within 9 A)
                    hash.query_into(&rec_coords, pos, 9.0, &mut near_9);
                    let mut dirs_hit = 0u32;
                    for &ri in &near_9 {
                        let rc = rec_infos[ri].coords;
                        let v = norm3([rc[0] - x, rc[1] - y, rc[2] - z]);
                        for (di, &bdir) in BURIAL_DIRS.iter().enumerate() {
                            if (dirs_hit & (1u32 << di)) == 0 && dot3(v, bdir) > 0.72 {
                                dirs_hit |= 1u32 << di;
                            }
                        }
                    }
                    let n_cones = dirs_hit.count_ones();
                    if n_cones < 5 && near_6.len() < 8 {
                        continue;
                    }

                    let burial_factor = 1.0 + 0.04 * (n_cones as f64 - 5.0).clamp(0.0, 7.0);
                    cavity_pts.push(([ix, iy, iz], pos, near_hyd >= near_pol));

                    // --- Probe 1: HBondDonor (ligand donates H to receptor acceptor) ---
                    let mut hbd_top1 = 0.0_f64;
                    let mut hbd_top2 = 0.0_f64;
                    let mut hbd_dir = [0.0; 3];
                    let mut hbd_res = String::new();

                    // --- Probe 2: HBondAcceptor (ligand accepts H from receptor donor) ---
                    let mut hba_top1 = 0.0_f64;
                    let mut hba_top2 = 0.0_f64;
                    let mut hba_dir = [0.0; 3];
                    let mut hba_res = String::new();

                    // --- Probe 3: Hydrophobic ---
                    let mut hyd_e = 0.0_f64;
                    let mut hyd_res = String::new();
                    let mut hyd_best_c = 0.0_f64;

                    // --- Probe 5 & 6: Positive / Negative Ionizable (with same-sign repulsion) ---
                    let mut pos_top1 = 0.0_f64;
                    let mut pos_top2 = 0.0_f64;
                    let mut pos_rep = 0.0_f64;
                    let mut pos_res = String::new();
                    let mut neg_top1 = 0.0_f64;
                    let mut neg_top2 = 0.0_f64;
                    let mut neg_rep = 0.0_f64;
                    let mut neg_res = String::new();

                    // --- Probe 7: MetalChelator ---
                    let mut met_e = 0.0_f64;
                    let mut met_dir = [0.0; 3];
                    let mut met_res = String::new();

                    for &ri in &near_6 {
                        let ra = &rec_infos[ri];
                        let d = dist3(pos, ra.coords);
                        let to_rec = norm3([ra.coords[0] - x, ra.coords[1] - y, ra.coords[2] - z]);
                        let from_rec = [-to_rec[0], -to_rec[1], -to_rec[2]];

                        // HBD probe (physically capped to top-2 simultaneous receptor acceptors)
                        if ra.h_acceptor && !ra.is_metal && (2.50..=3.45).contains(&d) {
                            let t = (d - 2.88) / 0.34;
                            let g = (-t * t).exp();
                            let dir_w = if dot3(ra.acceptor_dir, ra.acceptor_dir) > 0.5 {
                                let c = dot3(from_rec, ra.acceptor_dir);
                                0.30 + 0.70 * ((c + 0.15) / 1.15).clamp(0.0, 1.0)
                            } else {
                                0.85
                            };
                            let mult = if ra.formal_charge < 0 {
                                1.35
                            } else if ra.is_backbone_polar {
                                1.18
                            } else {
                                1.0
                            };
                            let pair_e = -1.65 * g * dir_w * mult * burial_factor;
                            if pair_e < hbd_top1 {
                                hbd_top2 = hbd_top1;
                                hbd_top1 = pair_e;
                                hbd_dir = to_rec;
                                hbd_res = ra.label.clone();
                            } else if pair_e < hbd_top2 {
                                hbd_top2 = pair_e;
                            }
                        }

                        // HBA probe (physically capped to top-2 simultaneous receptor donors)
                        if ra.h_donor && !ra.is_metal && (2.50..=3.45).contains(&d) {
                            let t = (d - 2.88) / 0.34;
                            let g = (-t * t).exp();
                            let mut best_cos = 0.5_f64;
                            if !ra.donor_dirs.is_empty() {
                                best_cos = -1.0;
                                for &dh in &ra.donor_dirs {
                                    let c = dot3(from_rec, dh);
                                    if c > best_cos {
                                        best_cos = c;
                                    }
                                }
                            }
                            if best_cos >= 0.35 {
                                let dir_w =
                                    0.25 + 0.75 * ((best_cos - 0.35) / 0.65).clamp(0.0, 1.0);
                                let mult = if ra.formal_charge > 0 {
                                    1.35
                                } else if ra.is_backbone_polar {
                                    1.18
                                } else {
                                    1.0
                                };
                                let pair_e = -1.65 * g * dir_w * mult * burial_factor;
                                if pair_e < hba_top1 {
                                    hba_top2 = hba_top1;
                                    hba_top1 = pair_e;
                                    hba_dir = to_rec;
                                    hba_res = ra.label.clone();
                                } else if pair_e < hba_top2 {
                                    hba_top2 = pair_e;
                                }
                            }
                        }

                        // HYD probe
                        if ra.hydrophobic && (3.15..=5.40).contains(&d) {
                            let t = (d - 3.90) / 0.75;
                            let pair_e = -0.36 * (-t * t).exp() * burial_factor;
                            hyd_e += pair_e;
                            if pair_e < hyd_best_c {
                                hyd_best_c = pair_e;
                                hyd_res = ra.label.clone();
                            }
                        } else if (ra.h_donor || ra.h_acceptor || ra.formal_charge != 0) && d < 3.60
                        {
                            hyd_e += 0.35 * (3.60 - d);
                        }

                        // POS probe (attracted to receptor negative charge, repelled by receptor positive charge)
                        if ra.formal_charge < 0 && (2.60..=4.70).contains(&d) {
                            let t = (d - 3.15) / 0.80;
                            let pair_e = -2.25 * (-t * t).exp() * burial_factor;
                            if pair_e < pos_top1 {
                                pos_top2 = pos_top1;
                                pos_top1 = pair_e;
                                pos_res = ra.label.clone();
                            } else if pair_e < pos_top2 {
                                pos_top2 = pair_e;
                            }
                        } else if ra.formal_charge > 0 && !ra.is_metal && d <= 5.20 {
                            let t = (d - 3.00) / 1.15;
                            pos_rep += 2.30 * (-t * t).exp();
                        }

                        // NEG probe (attracted to receptor positive charge, repelled by receptor negative charge)
                        if ra.formal_charge > 0 && !ra.is_metal && (2.60..=4.70).contains(&d) {
                            let t = (d - 3.15) / 0.80;
                            let pair_e = -2.25 * (-t * t).exp() * burial_factor;
                            if pair_e < neg_top1 {
                                neg_top2 = neg_top1;
                                neg_top1 = pair_e;
                                neg_res = ra.label.clone();
                            } else if pair_e < neg_top2 {
                                neg_top2 = pair_e;
                            }
                        } else if ra.formal_charge < 0 && d <= 5.20 {
                            let t = (d - 3.00) / 1.15;
                            neg_rep += 2.30 * (-t * t).exp();
                        }

                        // MET probe
                        if ra.is_metal && (1.80..=2.70).contains(&d) {
                            let t = (d - ra.metal_r_opt) / 0.30;
                            let pair_e = -3.40 * (-t * t).exp();
                            if pair_e < met_e {
                                met_e = pair_e;
                                met_dir = to_rec;
                                met_res = ra.label.clone();
                            }
                        }
                    }

                    let hbd_e = (hbd_top1 + 0.55 * hbd_top2).max(-3.60);
                    let hba_e = (hba_top1 + 0.55 * hba_top2).max(-3.60);
                    let hyd_e = hyd_e.max(-3.20);
                    let pos_e = ((pos_top1 + 0.30 * pos_top2).max(-2.95) + pos_rep).min(0.0);
                    let neg_e = ((neg_top1 + 0.30 * neg_top2).max(-2.95) + neg_rep).min(0.0);

                    // --- Probe 4: Aromatic (pi-pi / T-shaped / cation-pi + hydrophobic enclosure) ---
                    let mut aro_e = 0.0_f64;
                    let mut aro_dir = None;
                    let mut aro_res = String::new();
                    for ring in &arom_rings {
                        let d = dist3(pos, ring.center);
                        if (3.20..=5.80).contains(&d) {
                            let v = norm3([
                                pos[0] - ring.center[0],
                                pos[1] - ring.center[1],
                                pos[2] - ring.center[2],
                            ]);
                            let cos_n = dot3(v, ring.normal).abs();
                            let pair_e = if cos_n >= 0.65 && d <= 4.5 {
                                let t = (d - 3.75) / 0.45;
                                -1.65 * (-t * t).exp() * burial_factor
                            } else if (4.1..=5.6).contains(&d) {
                                let t = (d - 4.85) / 0.55;
                                -1.25 * (-t * t).exp() * burial_factor
                            } else {
                                0.0
                            };
                            if pair_e < aro_e {
                                aro_e = pair_e;
                                aro_dir = Some(ring.normal);
                                aro_res = ring.label.clone();
                            }
                        }
                    }
                    if aro_e > -0.75 {
                        for &ri in &near_6 {
                            let ra = &rec_infos[ri];
                            if ra.formal_charge > 0 && !ra.is_metal {
                                let d = dist3(pos, ra.coords);
                                if (3.40..=5.20).contains(&d) {
                                    let t = (d - 4.10) / 0.65;
                                    let pair_e = -1.35 * (-t * t).exp() * burial_factor;
                                    if pair_e < aro_e {
                                        aro_e = pair_e;
                                        aro_dir = Some(norm3([
                                            ra.coords[0] - x,
                                            ra.coords[1] - y,
                                            ra.coords[2] - z,
                                        ]));
                                        aro_res = ra.label.clone();
                                    }
                                }
                            }
                        }
                    }
                    if aro_e < -0.60 && hyd_e < -0.40 {
                        aro_e = (aro_e + 0.30 * hyd_e).max(-3.40);
                    }

                    let ijk = [ix, iy, iz];
                    if hbd_e < -1.15 && !hbd_res.is_empty() {
                        peaks.push((
                            ijk,
                            RawProbePeak {
                                kind: PharmacophoreKind::HDonor,
                                center: pos,
                                direction: Some(hbd_dir),
                                energy_kcal: hbd_e,
                                key_residue: hbd_res,
                            },
                        ));
                    }
                    if hba_e < -1.15 && !hba_res.is_empty() {
                        peaks.push((
                            ijk,
                            RawProbePeak {
                                kind: PharmacophoreKind::HAcceptor,
                                center: pos,
                                direction: Some(hba_dir),
                                energy_kcal: hba_e,
                                key_residue: hba_res,
                            },
                        ));
                    }
                    if hyd_e < -1.10 && !hyd_res.is_empty() {
                        peaks.push((
                            ijk,
                            RawProbePeak {
                                kind: PharmacophoreKind::Hydrophobic,
                                center: pos,
                                direction: None,
                                energy_kcal: hyd_e,
                                key_residue: hyd_res,
                            },
                        ));
                    }
                    if aro_e < -0.90 && !aro_res.is_empty() {
                        peaks.push((
                            ijk,
                            RawProbePeak {
                                kind: PharmacophoreKind::Aromatic,
                                center: pos,
                                direction: aro_dir,
                                energy_kcal: aro_e,
                                key_residue: aro_res,
                            },
                        ));
                    }
                    if pos_e < -1.60 && !pos_res.is_empty() {
                        peaks.push((
                            ijk,
                            RawProbePeak {
                                kind: PharmacophoreKind::PositiveIonizable,
                                center: pos,
                                direction: None,
                                energy_kcal: pos_e,
                                key_residue: pos_res,
                            },
                        ));
                    }
                    if neg_e < -1.60 && !neg_res.is_empty() {
                        peaks.push((
                            ijk,
                            RawProbePeak {
                                kind: PharmacophoreKind::NegativeIonizable,
                                center: pos,
                                direction: None,
                                energy_kcal: neg_e,
                                key_residue: neg_res,
                            },
                        ));
                    }
                    if met_e < -2.00 && !met_res.is_empty() {
                        peaks.push((
                            ijk,
                            RawProbePeak {
                                kind: PharmacophoreKind::MetalChelate,
                                center: pos,
                                direction: Some(met_dir),
                                energy_kcal: met_e,
                                key_residue: met_res,
                            },
                        ));
                    }
                }
            }
            SliceResult { cavity_pts, peaks }
        })
        .collect();

    // Filter strictly to the 6-connected main binding cavity component anchored at grid_box.center
    // (eliminates sealed internal protein voids such as trypsin's ILE16-ASP194 salt-bridge cavity!).
    let mut occ_map: HashMap<[usize; 3], ([f64; 3], bool)> = HashMap::new();
    let mut raw_peaks_ijk: Vec<([usize; 3], RawProbePeak)> = Vec::new();
    for sl in slices {
        for (ijk, p, is_hyd) in sl.cavity_pts {
            occ_map.insert(ijk, (p, is_hyd));
        }
        raw_peaks_ijk.extend(sl.peaks);
    }

    let seed_ref = if !holo_atoms.is_empty() {
        let k = holo_atoms.len() as f64;
        let mut s = [0.0; 3];
        for p in &holo_atoms {
            s[0] += p[0] / k;
            s[1] += p[1] / k;
            s[2] += p[2] / k;
        }
        s
    } else {
        grid_box.center
    };
    let main_component = extract_main_connected_cavity(&occ_map, seed_ref);

    let mut all_cavity: Vec<[f64; 3]> = Vec::with_capacity(main_component.len());
    let mut hyd_vox = 0usize;
    for ijk in &main_component {
        if let Some(&(p, is_hyd)) = occ_map.get(ijk) {
            all_cavity.push(p);
            if is_hyd {
                hyd_vox += 1;
            }
        }
    }
    let all_peaks: Vec<RawProbePeak> = raw_peaks_ijk
        .into_iter()
        .filter(|(ijk, _)| main_component.contains(ijk))
        .map(|(_, pk)| pk)
        .collect();

    // 1. PCA Pocket Geometry & Optimal Ligand Sizing
    let geometry = compute_site_geometry(
        &all_cavity,
        hyd_vox,
        h,
        grid_box.center,
        has_catalytic_metal,
    );

    // 2. Non-maximum suppression (NMS) per probe class to extract spatially distinct Apo MIF hotspots
    let kinds = [
        PharmacophoreKind::MetalChelate,
        PharmacophoreKind::HDonor,
        PharmacophoreKind::HAcceptor,
        PharmacophoreKind::PositiveIonizable,
        PharmacophoreKind::NegativeIonizable,
        PharmacophoreKind::Aromatic,
        PharmacophoreKind::Hydrophobic,
    ];
    let mut apo_hotspots: Vec<RawProbePeak> = Vec::new();
    for kind in kinds {
        let mut subset: Vec<&RawProbePeak> = all_peaks.iter().filter(|p| p.kind == kind).collect();
        subset.sort_by(|a, b| a.energy_kcal.total_cmp(&b.energy_kcal));
        let sep = match kind {
            PharmacophoreKind::Hydrophobic | PharmacophoreKind::Aromatic => 3.3,
            PharmacophoreKind::PositiveIonizable | PharmacophoreKind::NegativeIonizable => 3.6,
            PharmacophoreKind::MetalChelate => 2.5,
            _ => 2.5,
        };
        let per_kind_cap = match kind {
            PharmacophoreKind::PositiveIonizable | PharmacophoreKind::NegativeIonizable => {
                config.max_per_kind.min(2)
            }
            _ => config.max_per_kind,
        };
        let unique_res_per_kind = matches!(
            kind,
            PharmacophoreKind::HDonor
                | PharmacophoreKind::HAcceptor
                | PharmacophoreKind::PositiveIonizable
                | PharmacophoreKind::NegativeIonizable
                | PharmacophoreKind::MetalChelate
        );
        let mut kept: Vec<RawProbePeak> = Vec::new();
        for pk in subset {
            let pk_res = residue_key(&pk.key_residue);
            let far = kept.iter().all(|k| {
                dist3(k.center, pk.center) >= sep
                    && (!unique_res_per_kind || residue_key(&k.key_residue) != pk_res)
            });
            if far {
                kept.push(pk.clone());
                if kept.len() >= per_kind_cap {
                    break;
                }
            }
        }
        apo_hotspots.extend(kept);
    }

    // 3. Hybrid Apo + Holo consensus refinement when holo ligand(s) are supplied
    let mut features =
        build_hybrid_features(apo_hotspots, holo_ligands, &rec_infos, config.max_features);

    // Update recommended HBD/HBA counts from detected polar hotspots
    let mut geometry = geometry;
    let n_hbd = features
        .iter()
        .filter(|f| f.enabled && f.kind == PharmacophoreKind::HDonor)
        .count();
    let n_hba = features
        .iter()
        .filter(|f| f.enabled && f.kind == PharmacophoreKind::HAcceptor)
        .count();
    geometry.recommended_hbd_max = (n_hbd + 2).clamp(2, 5);
    geometry.recommended_hba_max = (n_hba + 4).clamp(4, 10);

    // Assign clean sequential IDs (F1, F2, ...)
    for (i, f) in features.iter_mut().enumerate() {
        f.id = format!("F{}", i + 1);
    }

    // 4. Optimal inter-feature 3D distance matrix
    let distance_matrix = build_distance_matrix(&features);

    // 5. Compact shell of receptor wall exclusion spheres around the cavity
    let exclusion_spheres = build_exclusion_spheres(
        &rec_infos,
        &rec_coords,
        &all_cavity,
        config.max_exclusion_spheres,
    );

    Ok(SiteBlueprint {
        format: "dockcraft-site-blueprint".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        target_name: receptor.name.clone(),
        grid_box: grid_box.clone(),
        geometry,
        features,
        distance_matrix,
        exclusion_spheres,
        waters,
        mif_spacing_a: h,
        cavity_voxel_count: all_cavity.len(),
        elapsed_ms: t0.elapsed().as_millis(),
    })
}

fn extract_main_connected_cavity(
    occ_map: &HashMap<[usize; 3], ([f64; 3], bool)>,
    seed_ref: [f64; 3],
) -> HashSet<[usize; 3]> {
    let Some((&seed_ijk, _)) = occ_map
        .iter()
        .min_by(|(_, (pa, _)), (_, (pb, _))| dist3(*pa, seed_ref).total_cmp(&dist3(*pb, seed_ref)))
    else {
        return HashSet::new();
    };
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    visited.insert(seed_ijk);
    queue.push_back(seed_ijk);
    while let Some([ix, iy, iz]) = queue.pop_front() {
        let nbrs = [
            [ix.wrapping_sub(1), iy, iz],
            [ix + 1, iy, iz],
            [ix, iy.wrapping_sub(1), iz],
            [ix, iy + 1, iz],
            [ix, iy, iz.wrapping_sub(1)],
            [ix, iy, iz + 1],
        ];
        for nb in nbrs {
            if occ_map.contains_key(&nb) && visited.insert(nb) {
                queue.push_back(nb);
            }
        }
    }
    visited
}

fn compute_site_geometry(
    cavity_pts: &[[f64; 3]],
    hyd_vox: usize,
    spacing: f64,
    fallback_center: [f64; 3],
    has_catalytic_metal: bool,
) -> SiteGeometry {
    if cavity_pts.is_empty() {
        return SiteGeometry {
            center: fallback_center,
            accessible_volume_a3: 0.0,
            principal_dims_a: [0.0; 3],
            principal_axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            hydrophobic_fraction: 0.5,
            polar_fraction: 0.5,
            recommended_mw_range: [250.0, 500.0],
            recommended_heavy_atoms: [18, 36],
            recommended_hbd_max: 3,
            recommended_hba_max: 7,
            has_catalytic_metal,
        };
    }

    let n = cavity_pts.len() as f64;
    let mut center = [0.0; 3];
    for p in cavity_pts {
        center[0] += p[0] / n;
        center[1] += p[1] / n;
        center[2] += p[2] / n;
    }

    let mut cov = [[0.0_f64; 3]; 3];
    for p in cavity_pts {
        let d = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
        for r in 0..3 {
            for c in 0..3 {
                cov[r][c] += d[r] * d[c] / n;
            }
        }
    }
    let (_, axes) = symmetric_eigen_3x3(cov);

    // 2nd–98th percentile spans along each PCA axis for robust [L, W, D] dimensions
    let mut dims = [0.0_f64; 3];
    for (ax_i, &axis) in axes.iter().enumerate() {
        let mut projs: Vec<f64> = cavity_pts
            .iter()
            .map(|p| {
                (p[0] - center[0]) * axis[0]
                    + (p[1] - center[1]) * axis[1]
                    + (p[2] - center[2]) * axis[2]
            })
            .collect();
        projs.sort_by(f64::total_cmp);
        let lo_idx = ((projs.len() as f64) * 0.02) as usize;
        let hi_idx = (((projs.len() as f64) * 0.98) as usize).min(projs.len().saturating_sub(1));
        dims[ax_i] = (projs[hi_idx] - projs[lo_idx] + spacing).max(spacing);
    }

    let vol = n * spacing * spacing * spacing;
    let hyd_frac = (hyd_vox as f64 / n).clamp(0.0, 1.0);
    // Mecozzi-Rebek optimal binding occupancy (~45%–70% of accessible subpocket volume; ~18.5 A^3/heavy atom)
    let eff_vol = vol.clamp(220.0, 950.0);
    let min_ha = ((eff_vol * 0.42) / 18.5).round().clamp(12.0, 32.0) as usize;
    let max_ha = ((eff_vol * 0.68) / 18.5)
        .round()
        .clamp((min_ha + 6) as f64, 42.0) as usize;

    SiteGeometry {
        center,
        accessible_volume_a3: (vol * 10.0).round() / 10.0,
        principal_dims_a: dims,
        principal_axes: axes,
        hydrophobic_fraction: hyd_frac,
        polar_fraction: 1.0 - hyd_frac,
        recommended_mw_range: [(min_ha as f64) * 13.8, (max_ha as f64) * 14.0],
        recommended_heavy_atoms: [min_ha, max_ha],
        recommended_hbd_max: 3,
        recommended_hba_max: 7,
        has_catalytic_metal,
    }
}

fn default_radius_for(kind: PharmacophoreKind) -> f64 {
    match kind {
        PharmacophoreKind::MetalChelate => 1.15,
        PharmacophoreKind::HDonor | PharmacophoreKind::HAcceptor => 1.35,
        PharmacophoreKind::PositiveIonizable | PharmacophoreKind::NegativeIonizable => 1.50,
        PharmacophoreKind::Aromatic => 1.60,
        PharmacophoreKind::Hydrophobic | PharmacophoreKind::AnyHeavy => 1.75,
    }
}

fn residue_key(label: &str) -> &str {
    label.split_whitespace().next().unwrap_or(label)
}

fn build_hybrid_features(
    apo_hotspots: Vec<RawProbePeak>,
    holo_ligands: &[&Molecule3D],
    rec_infos: &[RecAtomInfo],
    max_features: usize,
) -> Vec<SiteFeature> {
    let mut features: Vec<SiteFeature> = apo_hotspots
        .into_iter()
        .map(|pk| SiteFeature {
            id: String::new(),
            kind: pk.kind,
            center: pk.center,
            radius_a: default_radius_for(pk.kind),
            direction: pk.direction,
            energy_kcal: pk.energy_kcal,
            key_residue: pk.key_residue,
            weight: 1.0,
            enabled: true,
            required: pk.kind == PharmacophoreKind::MetalChelate,
            source: FeatureSource::ApoMif,
        })
        .collect();

    if !holo_ligands.is_empty() {
        for &lig in holo_ligands {
            let lig_feats = extract_ligand_features(lig);
            for lf in lig_feats {
                // Verify this holo feature actually interacts with the receptor pocket
                let (interacts, nearest_res, to_rec) =
                    check_holo_feature_contact(lf.kind, lf.center, rec_infos);
                if !interacts {
                    continue;
                }
                // Check if an existing Apo hotspot matches within 2.4 A
                if let Some(existing) =
                    features
                        .iter_mut()
                        .filter(|f| f.kind == lf.kind)
                        .min_by(|a, b| {
                            dist3(a.center, lf.center).total_cmp(&dist3(b.center, lf.center))
                        })
                    && dist3(existing.center, lf.center) <= 2.4
                {
                    // Refine toward crystallographic consensus and mark HybridApoHolo
                    for k in 0..3 {
                        existing.center[k] = 0.30 * existing.center[k] + 0.70 * lf.center[k];
                    }
                    existing.energy_kcal = (existing.energy_kcal - 0.90).min(-2.10);
                    existing.weight = 1.45;
                    existing.source = FeatureSource::HybridApoHolo;
                    if !nearest_res.is_empty() {
                        existing.key_residue = nearest_res;
                    }
                } else {
                    // Promote validated crystallographic interaction not already in Apo top-K
                    let base_e = match lf.kind {
                        PharmacophoreKind::MetalChelate => -3.6,
                        PharmacophoreKind::HDonor | PharmacophoreKind::HAcceptor => -2.75,
                        PharmacophoreKind::PositiveIonizable
                        | PharmacophoreKind::NegativeIonizable => -2.80,
                        PharmacophoreKind::Aromatic => -2.40,
                        _ => -2.25,
                    };
                    features.push(SiteFeature {
                        id: String::new(),
                        kind: lf.kind,
                        center: lf.center,
                        radius_a: default_radius_for(lf.kind),
                        direction: lf.direction.or(to_rec),
                        energy_kcal: base_e,
                        key_residue: nearest_res,
                        weight: 1.35,
                        enabled: true,
                        required: lf.kind == PharmacophoreKind::MetalChelate,
                        source: FeatureSource::HoloValidated,
                    });
                }
            }
        }
    }

    // Rank by experimental holo confirmation + MIF energy strength
    features.sort_by(|a, b| {
        let rank_a = a.energy_kcal
            - match a.source {
                FeatureSource::HybridApoHolo => 2.2,
                FeatureSource::HoloValidated => 1.6,
                _ => 0.0,
            };
        let rank_b = b.energy_kcal
            - match b.source {
                FeatureSource::HybridApoHolo => 2.2,
                FeatureSource::HoloValidated => 1.6,
                _ => 0.0,
            };
        rank_a.total_cmp(&rank_b)
    });

    // Deduplicate same-type features (< 2.2 A), overlapping Apo HBD/HBA (< 1.5 A), and cap
    // unconfirmed ApoMif duplicates of the same class at 2 so polar/aromatic/hydrophobic balance holds.
    let mut selected: Vec<SiteFeature> = Vec::with_capacity(max_features);
    for f in features {
        let dup = selected.iter().any(|s| {
            let d = dist3(s.center, f.center);
            (s.kind == f.kind && d < 2.2)
                || (matches!(
                    (s.kind, f.kind),
                    (PharmacophoreKind::Aromatic, PharmacophoreKind::Hydrophobic)
                        | (PharmacophoreKind::Hydrophobic, PharmacophoreKind::Aromatic)
                ) && d < 1.5)
                || (matches!(
                    (s.kind, f.kind),
                    (PharmacophoreKind::HDonor, PharmacophoreKind::HAcceptor)
                        | (PharmacophoreKind::HAcceptor, PharmacophoreKind::HDonor)
                ) && d < 1.5
                    && s.source == FeatureSource::ApoMif
                    && f.source == FeatureSource::ApoMif)
        });
        if dup {
            continue;
        }
        if f.source == FeatureSource::ApoMif {
            let same_kind_apo = selected
                .iter()
                .filter(|s| s.kind == f.kind && s.source == FeatureSource::ApoMif)
                .count();
            if same_kind_apo >= 2 {
                continue;
            }
        }
        selected.push(f);
        if selected.len() >= max_features {
            break;
        }
    }

    // Enable top 7 features by default (preserving all HybridApoHolo up to 8)
    for (i, f) in selected.iter_mut().enumerate() {
        f.enabled = i < 7 || (i < 8 && f.source == FeatureSource::HybridApoHolo);
    }
    selected
}

fn check_holo_feature_contact(
    kind: PharmacophoreKind,
    center: [f64; 3],
    rec_infos: &[RecAtomInfo],
) -> (bool, String, Option<[f64; 3]>) {
    let mut best_d = 99.0_f64;
    let mut best_res = String::new();
    let mut best_dir = None;
    for ra in rec_infos {
        let d = dist3(center, ra.coords);
        let ok = match kind {
            PharmacophoreKind::HDonor => ra.h_acceptor && !ra.is_metal && d <= 3.55,
            PharmacophoreKind::HAcceptor => ra.h_donor && !ra.is_metal && d <= 3.55,
            PharmacophoreKind::MetalChelate => ra.is_metal && d <= 2.65,
            PharmacophoreKind::PositiveIonizable => {
                (ra.formal_charge < 0 || ra.h_acceptor) && d <= 4.6
            }
            PharmacophoreKind::NegativeIonizable => {
                (ra.formal_charge > 0 || ra.h_donor) && d <= 4.6
            }
            PharmacophoreKind::Aromatic | PharmacophoreKind::Hydrophobic => {
                ra.hydrophobic && d <= 4.8
            }
            PharmacophoreKind::AnyHeavy => d <= 4.5,
        };
        if ok && d < best_d {
            best_d = d;
            best_res = ra.label.clone();
            best_dir = Some(norm3([
                ra.coords[0] - center[0],
                ra.coords[1] - center[1],
                ra.coords[2] - center[2],
            ]));
        }
    }
    (best_d < 90.0, best_res, best_dir)
}

#[must_use]
pub fn build_distance_matrix(features: &[SiteFeature]) -> Vec<FeatureDistanceConstraint> {
    let active: Vec<&SiteFeature> = features.iter().filter(|f| f.enabled).collect();
    let mut out = Vec::new();
    for i in 0..active.len() {
        for j in (i + 1)..active.len() {
            let a = active[i];
            let b = active[j];
            let d = dist3(a.center, b.center);
            let tol = (a.radius_a + b.radius_a) * 0.55;
            let angle = match (a.direction, b.direction) {
                (Some(da), Some(db)) if dot3(da, da) > 0.5 && dot3(db, db) > 0.5 => {
                    Some(dot3(da, db).clamp(-1.0, 1.0).acos().to_degrees())
                }
                _ => None,
            };
            out.push(FeatureDistanceConstraint {
                from_id: a.id.clone(),
                to_id: b.id.clone(),
                from_kind: a.kind,
                to_kind: b.kind,
                ideal_distance_a: (d * 100.0).round() / 100.0,
                min_distance_a: ((d - tol).max(0.0) * 100.0).round() / 100.0,
                max_distance_a: ((d + tol) * 100.0).round() / 100.0,
                vector_angle_deg: angle,
            });
        }
    }
    out
}

fn build_exclusion_spheres(
    rec_infos: &[RecAtomInfo],
    rec_coords: &[[f64; 3]],
    cavity_pts: &[[f64; 3]],
    max_spheres: usize,
) -> Vec<ExclusionSphere> {
    if cavity_pts.is_empty() || max_spheres == 0 {
        return Vec::new();
    }
    let cav_hash = SpatialHash::new(cavity_pts, 4.0);
    let mut near_cav = Vec::new();
    let mut wall_cands: Vec<(f64, &RecAtomInfo)> = Vec::new();
    for (ri, rinfo) in rec_infos.iter().enumerate() {
        if rinfo.is_metal {
            continue; // Never mark catalytic metal ions as steric exclusion walls
        }
        cav_hash.query_into(cavity_pts, rec_coords[ri], 4.5, &mut near_cav);
        if !near_cav.is_empty() {
            let min_d = near_cav
                .iter()
                .map(|&ci| dist3(rec_coords[ri], cavity_pts[ci]))
                .fold(99.0_f64, f64::min);
            wall_cands.push((min_d, rinfo));
        }
    }
    wall_cands.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut out: Vec<ExclusionSphere> = Vec::with_capacity(max_spheres);
    for (_, rinfo) in wall_cands {
        if out.iter().all(|s| dist3(s.center, rinfo.coords) >= 1.95) {
            out.push(ExclusionSphere {
                center: rinfo.coords,
                // Center-to-center steric wall clearance against ligand heavy-atom centers:
                // H-bonds sit at 2.65–3.2 A and VdW contacts at 3.2–3.8 A; < 1.95 A is a real wall clash.
                radius_a: (rinfo.vdw_r * 1.15).clamp(1.75, 2.05),
                residue: rinfo.label.clone(),
            });
            if out.len() >= max_spheres {
                break;
            }
        }
    }
    out
}

/// Helper to test how many crystal ligand features are recovered within `tol_a` of a blueprint's
/// hotspots (used for scientific validation across the 22 RCSB benchmark complexes).
#[must_use]
pub fn evaluate_holo_feature_recovery(
    blueprint: &SiteBlueprint,
    receptor: &Molecule3D,
    crystal_ligand: &Molecule3D,
    tol_a: f64,
) -> (usize, usize) {
    let (rec_infos, _, _, _) = prepare_receptor_context(receptor, &blueprint.grid_box);
    let lig_feats = extract_ligand_features(crystal_ligand);
    let mut total = 0usize;
    let mut recovered = 0usize;
    for lf in lig_feats {
        let (interacts, _, _) = check_holo_feature_contact(lf.kind, lf.center, &rec_infos);
        if !interacts {
            continue;
        }
        total += 1;
        let hit = blueprint.features.iter().any(|sf| {
            let kind_ok = sf.kind == lf.kind
                || (sf.kind == PharmacophoreKind::Hydrophobic
                    && lf.kind == PharmacophoreKind::Aromatic)
                || (sf.kind == PharmacophoreKind::Aromatic
                    && lf.kind == PharmacophoreKind::Hydrophobic);
            kind_ok && dist3(sf.center, lf.center) <= tol_a
        });
        if hit {
            recovered += 1;
        }
    }
    (recovered, total)
}
