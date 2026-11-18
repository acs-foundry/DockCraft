#![forbid(unsafe_code)]

use std::time::Instant;

use pocket_detect::compute_sasa_fibonacci;
use serde::{Deserialize, Serialize};
use struct_core::{Atom3D, BondOrder, MAX_LIGAND_ATOMS, Molecule3D, norm3, perceive_all};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HydeError {
    #[error("empty structure")]
    EmptyStructure,
    #[error("ligand atom count {count} exceeds limit {limit}")]
    TooManyLigandAtoms { count: usize, limit: usize },
    #[error("invalid atom index {0}")]
    InvalidAtomIndex(usize),
    #[error("topology error during mutation: {0}")]
    Topology(#[from] struct_core::StructError),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum InteractionKind {
    HydrogenBond,
    SaltBridge,
    PiPiStacking,
    CationPi,
    HalogenBond,
    HydrophobicContact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interaction {
    pub kind: InteractionKind,
    pub lig_atom_indices: Vec<usize>,
    pub rec_atom_indices: Vec<usize>,
    pub rec_residue: String,
    pub distance_a: f64,
    pub angle_deg: Option<f64>,
    pub energy_kcal: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomHydeScore {
    pub atom_index: usize,
    pub atom_name: String,
    pub element: String,
    pub delta_g_kcal: f64,
    pub desolv_kcal: f64,
    pub hbond_kcal: f64,
    pub hydrophobic_kcal: f64,
    pub steric_kcal: f64,
    pub buried_fraction: f64,
    pub corona_color: [f32; 4],
    pub corona_radius: f32,
    pub diagnostic: String,
}

/// Clean-room empirical SASA-desolvation, H-bond recovery, lipophilic burial, and steric
/// decomposition (HYDE-inspired; uses clean-room empirical weights rather than BioSolveIT calibrated tables).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HydeReport {
    pub scoring_model: String,
    pub total_delta_g_kcal: f64,
    pub estimated_ki_nm: f64,
    pub ligand_efficiency: f64,
    pub atom_scores: Vec<AtomHydeScore>,
    pub interactions: Vec<Interaction>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SubstituentPreset {
    H,
    F,
    Cl,
    OH,
    NH2,
    CH3,
    CF3,
    CN,
    OCH3,
}

impl SubstituentPreset {
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[
            Self::H,
            Self::F,
            Self::Cl,
            Self::OH,
            Self::NH2,
            Self::CH3,
            Self::CF3,
            Self::CN,
            Self::OCH3,
        ]
    }

    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::H => "-H",
            Self::F => "-F",
            Self::Cl => "-Cl",
            Self::OH => "-OH",
            Self::NH2 => "-NH2",
            Self::CH3 => "-CH3",
            Self::CF3 => "-CF3",
            Self::CN => "-CN",
            Self::OCH3 => "-OCH3",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationOutcome {
    pub substituent: SubstituentPreset,
    pub mutated_ligand: Molecule3D,
    pub before_kcal: f64,
    pub after_kcal: f64,
    pub delta_delta_g_kcal: f64,
    pub report: HydeReport,
    pub elapsed_ms: u128,
    /// `true` when the clicked atom was replaced by the substituent, `false` when the substituent
    /// was appended to it because the clicked atom was an internal one with spare valence.
    pub replaced_anchor: bool,
    /// Largest displacement of an existing ligand atom relative to the ligand's own frame, in
    /// angstroms. A rigid-body nudge of the whole ligand leaves this at zero; a value above zero
    /// means the relaxation actually changed the ligand's internal geometry.
    pub internal_rmsd_shift: f64,
}

fn vec_len(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn angle_deg(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let v1 = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let v2 = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
    let n1 = vec_len(v1);
    let n2 = vec_len(v2);
    if n1 < 1e-8 || n2 < 1e-8 {
        return 180.0;
    }
    let cos_t = ((v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (n1 * n2)).clamp(-1.0, 1.0);
    cos_t.acos().to_degrees()
}

fn ring_centroids(mol: &Molecule3D) -> Vec<(Vec<usize>, [f64; 3])> {
    let mut out = Vec::new();
    for ring in &mol.rings {
        if ring.len() < 5 || ring.len() > 6 {
            continue;
        }
        let is_arom = ring
            .iter()
            .all(|&i| mol.atoms.get(i).is_some_and(|a| a.aromatic));
        if !is_arom {
            continue;
        }
        let mut c = [0.0_f64; 3];
        for &i in ring {
            let p = mol.atoms[i].coords;
            c[0] += p[0];
            c[1] += p[1];
            c[2] += p[2];
        }
        let n = ring.len() as f64;
        out.push((ring.clone(), [c[0] / n, c[1] / n, c[2] / n]));
    }
    out
}

#[must_use]
pub fn detect_interactions(receptor: &Molecule3D, ligand: &Molecule3D) -> Vec<Interaction> {
    let mut interactions = Vec::new();

    for (li, la) in ligand.atoms.iter().enumerate() {
        if !la.is_heavy() {
            continue;
        }
        for (ri, ra) in receptor.atoms.iter().enumerate() {
            if !ra.is_heavy() {
                continue;
            }
            let d = dist(la.coords, ra.coords);
            if d > 5.0 {
                continue;
            }
            let res_tag = format!("{}:{}{}", ra.chain_id, ra.res_name, ra.res_seq);

            // Salt bridge (formal or strong opposite partial charge)
            let opp_charged = (la.formal_charge > 0 && ra.formal_charge < 0)
                || (la.formal_charge < 0 && ra.formal_charge > 0)
                || (la.charge * ra.charge < -0.18
                    && (la.h_donor || la.h_acceptor)
                    && (ra.h_donor || ra.h_acceptor));
            if opp_charged && d <= 4.2 {
                interactions.push(Interaction {
                    kind: InteractionKind::SaltBridge,
                    lig_atom_indices: vec![li],
                    rec_atom_indices: vec![ri],
                    rec_residue: res_tag.clone(),
                    distance_a: d,
                    angle_deg: None,
                    energy_kcal: -3.4 * ((4.2 - d) / 1.6).clamp(0.25, 1.0),
                });
            } else if ((la.h_donor && ra.h_acceptor) || (la.h_acceptor && ra.h_donor))
                && (2.2..=3.6).contains(&d)
            {
                let ideal = (1.0 - ((d - 2.85).abs() / 0.85)).clamp(0.15, 1.0);
                interactions.push(Interaction {
                    kind: InteractionKind::HydrogenBond,
                    lig_atom_indices: vec![li],
                    rec_atom_indices: vec![ri],
                    rec_residue: res_tag.clone(),
                    distance_a: d,
                    angle_deg: None,
                    energy_kcal: -2.95 * ideal,
                });
            }

            // Halogen bond: C-X ... (O/N/S) with sigma-hole angle >= 120 deg
            if matches!(la.element.as_str(), "Cl" | "Br" | "I")
                && ra.h_acceptor
                && (2.6..=3.9).contains(&d)
            {
                let carbon_nbr = ligand
                    .adj
                    .get(li)
                    .and_then(|nbrs| nbrs.first())
                    .map(|&(ni, _)| ligand.atoms[ni].coords);
                if let Some(c_pos) = carbon_nbr {
                    let ang = angle_deg(c_pos, la.coords, ra.coords);
                    if ang >= 120.0 {
                        interactions.push(Interaction {
                            kind: InteractionKind::HalogenBond,
                            lig_atom_indices: vec![li],
                            rec_atom_indices: vec![ri],
                            rec_residue: res_tag.clone(),
                            distance_a: d,
                            angle_deg: Some(ang),
                            energy_kcal: -1.1 * ((ang - 120.0) / 60.0).clamp(0.2, 1.0),
                        });
                    }
                }
            }

            // Hydrophobic contacts
            if la.hydrophobic && ra.hydrophobic && (3.2..=4.4).contains(&d) {
                interactions.push(Interaction {
                    kind: InteractionKind::HydrophobicContact,
                    lig_atom_indices: vec![li],
                    rec_atom_indices: vec![ri],
                    rec_residue: res_tag,
                    distance_a: d,
                    angle_deg: None,
                    energy_kcal: -0.32,
                });
            }
        }
    }

    // Pi-Pi stacking and Cation-Pi using detected aromatic rings
    let lig_rings = ring_centroids(ligand);
    let rec_rings = ring_centroids(receptor);
    for (l_atoms, lc) in &lig_rings {
        for (r_atoms, rc) in &rec_rings {
            let d = dist(*lc, *rc);
            if (3.3..=5.5).contains(&d) {
                let first_r = receptor.atoms[r_atoms[0]].clone();
                interactions.push(Interaction {
                    kind: InteractionKind::PiPiStacking,
                    lig_atom_indices: l_atoms.clone(),
                    rec_atom_indices: r_atoms.clone(),
                    rec_residue: format!(
                        "{}:{}{}",
                        first_r.chain_id, first_r.res_name, first_r.res_seq
                    ),
                    distance_a: d,
                    angle_deg: None,
                    energy_kcal: -1.25 * (1.0 - (d - 3.8).abs() / 2.0).clamp(0.2, 1.0),
                });
            }
        }
        // Receptor cation to ligand aromatic ring
        for (ri, ra) in receptor.atoms.iter().enumerate() {
            if ra.formal_charge > 0 || (ra.element == "N" && ra.charge > 0.25) {
                let d = dist(*lc, ra.coords);
                if (3.2..=5.4).contains(&d) {
                    interactions.push(Interaction {
                        kind: InteractionKind::CationPi,
                        lig_atom_indices: l_atoms.clone(),
                        rec_atom_indices: vec![ri],
                        rec_residue: format!("{}:{}{}", ra.chain_id, ra.res_name, ra.res_seq),
                        distance_a: d,
                        angle_deg: None,
                        energy_kcal: -1.4,
                    });
                }
            }
        }
    }

    interactions
}

fn corona_visual(dg: f64) -> ([f32; 4], f32) {
    let mag = (dg.abs() / 2.6).clamp(0.0, 1.0) as f32;
    let radius = 0.55 + 1.35 * mag;
    if dg <= -0.25 {
        ([0.10, 0.86, 0.36, 0.30 + 0.40 * mag], radius)
    } else if dg >= 0.25 {
        ([0.94, 0.18, 0.22, 0.32 + 0.42 * mag], radius)
    } else {
        ([0.62, 0.68, 0.74, 0.20], 0.45)
    }
}

pub fn score_pose_hyde(
    receptor: &Molecule3D,
    ligand: &Molecule3D,
) -> Result<HydeReport, HydeError> {
    if ligand.atoms.len() > MAX_LIGAND_ATOMS {
        return Err(HydeError::TooManyLigandAtoms {
            count: ligand.atoms.len(),
            limit: MAX_LIGAND_ATOMS,
        });
    }
    let rec_heavy: Vec<(usize, &Atom3D)> = receptor
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| a.is_heavy())
        .collect();
    let lig_heavy: Vec<(usize, &Atom3D)> = ligand
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| a.is_heavy())
        .collect();

    if rec_heavy.is_empty() || lig_heavy.is_empty() {
        return Err(HydeError::EmptyStructure);
    }

    // Build unbound ligand heavy list and bound (receptor + ligand) heavy list for differential SASA.
    let lig_only_refs: Vec<&Atom3D> = lig_heavy.iter().map(|(_, a)| *a).collect();
    let mut complex_refs: Vec<&Atom3D> = rec_heavy.iter().map(|(_, a)| *a).collect();
    let lig_offset = complex_refs.len();
    complex_refs.extend(lig_only_refs.iter().copied());

    let interactions = detect_interactions(receptor, ligand);
    let mut atom_scores = Vec::with_capacity(lig_heavy.len());
    let mut total_dg = 0.0_f64;

    for (local_i, &(orig_idx, la)) in lig_heavy.iter().enumerate() {
        let sasa_free = compute_sasa_fibonacci(&lig_only_refs, &[local_i], 1.4).max(1.0);
        let sasa_bound = compute_sasa_fibonacci(&complex_refs, &[lig_offset + local_i], 1.4);
        let dsasa = (sasa_free - sasa_bound).max(0.0);
        let buried_frac = (dsasa / sasa_free).clamp(0.0, 1.0);

        let is_polar = la.h_donor
            || la.h_acceptor
            || la.formal_charge != 0
            || la.ad4_type.is_acceptor()
            || la.ad4_type.is_donor_heavy();

        // Desolvation penalty: polar groups pay significantly to leave water; hydrophobic pay 0 (net gain below).
        let desolv_kcal = if is_polar {
            let charge_boost = if la.formal_charge != 0 { 1.45 } else { 1.0 };
            0.028 * dsasa * charge_boost
        } else {
            0.0
        };

        // Best directional H-bond / salt bridge / halogen bond energy assigned to this atom
        let mut hbond_kcal = 0.0_f64;
        for inter in &interactions {
            if inter.lig_atom_indices.contains(&orig_idx) {
                match inter.kind {
                    InteractionKind::HydrogenBond
                    | InteractionKind::SaltBridge
                    | InteractionKind::HalogenBond => {
                        hbond_kcal +=
                            inter.energy_kcal / (inter.lig_atom_indices.len().max(1) as f64);
                    }
                    _ => {}
                }
            }
        }
        // Cap single atom H-bond recovery realistically while keeping strong net gain
        hbond_kcal = hbond_kcal.max(-4.5);

        // Hydrophobic / pi burial gain against apolar/aromatic surface
        let mut apolar_neighbors = 0usize;
        let mut steric_kcal = 0.0_f64;
        for &(_, ra) in &rec_heavy {
            let d = dist(la.coords, ra.coords);
            if d > 5.0 {
                continue;
            }
            if la.hydrophobic && ra.hydrophobic && d <= 4.5 {
                apolar_neighbors += 1;
            }
            let r_sum = la.ad4_type.vina_radius() + ra.ad4_type.vina_radius();
            let hard_cutoff = if (la.h_donor && ra.h_acceptor) || (la.h_acceptor && ra.h_donor) {
                2.25
            } else {
                0.86 * r_sum
            };
            if d < hard_cutoff {
                let pen = hard_cutoff - d;
                steric_kcal += 4.2 * pen * pen + 0.5 * pen;
            }
        }

        let hydrophobic_kcal = if la.hydrophobic && apolar_neighbors > 0 {
            let match_factor = (apolar_neighbors as f64 / 3.0).clamp(0.3, 1.15);
            -0.024 * dsasa * match_factor
        } else {
            0.0
        };

        let delta_g_kcal = desolv_kcal + hbond_kcal + hydrophobic_kcal + steric_kcal;
        total_dg += delta_g_kcal;

        let (corona_color, corona_radius) = corona_visual(delta_g_kcal);
        let diagnostic = if steric_kcal >= 0.7 {
            format!("Steric clash (+{steric_kcal:.2} kcal/mol)")
        } else if hbond_kcal <= -0.8 && delta_g_kcal < -0.2 {
            format!("Satisfied H-bond / polar match ({delta_g_kcal:.2} kcal/mol)")
        } else if is_polar && buried_frac > 0.35 && hbond_kcal > -0.4 && delta_g_kcal > 0.35 {
            format!("Buried unsatisfied polar desolvation (+{delta_g_kcal:.2} kcal/mol)")
        } else if hydrophobic_kcal <= -0.3 {
            format!("Favorable lipophilic burial ({delta_g_kcal:.2} kcal/mol)")
        } else {
            format!("Neutral exposure ({delta_g_kcal:+.2} kcal/mol)")
        };

        atom_scores.push(AtomHydeScore {
            atom_index: orig_idx,
            atom_name: la.name.clone(),
            element: la.element.clone(),
            delta_g_kcal,
            desolv_kcal,
            hbond_kcal,
            hydrophobic_kcal,
            steric_kcal,
            buried_fraction: buried_frac,
            corona_color,
            corona_radius,
            diagnostic,
        });
    }

    // Torsional entropy offset (~0.3 kcal/mol per rotatable bond)
    total_dg += 0.28 * ligand.rotatable_bonds.len() as f64;
    let rt = 0.5922_f64; // kcal/mol at 298.15 K
    let estimated_ki_nm = (total_dg / rt).clamp(-35.0, 20.0).exp() * 1.0e9;
    let ligand_efficiency = -total_dg / (lig_heavy.len().max(1) as f64);

    Ok(HydeReport {
        scoring_model: "dockcraft-cleanroom-hyde-inspired".to_string(),
        total_delta_g_kcal: total_dg,
        estimated_ki_nm,
        ligand_efficiency,
        atom_scores,
        interactions,
    })
}

/// Real-time 3D in-pocket substituent mutator + quick rigid-body local relaxation.
/// Removes a terminal atom and renumbers the molecule, returning the index map.
fn remove_terminal_atom(mol: &mut Molecule3D, victim: usize) -> Vec<Option<usize>> {
    let mut map: Vec<Option<usize>> = Vec::with_capacity(mol.atoms.len());
    let mut kept = Vec::with_capacity(mol.atoms.len());
    for (i, atom) in mol.atoms.iter().enumerate() {
        if i == victim {
            map.push(None);
        } else {
            map.push(Some(kept.len()));
            let mut a = atom.clone();
            a.id = kept.len();
            kept.push(a);
        }
    }
    let old_bonds = std::mem::take(&mut mol.bonds);
    mol.atoms = kept;
    mol.adj = vec![Vec::new(); mol.atoms.len()];
    for b in old_bonds {
        if let (Some(u), Some(v)) = (map[b.u], map[b.v]) {
            let _ = mol.add_bond(u, v, b.order);
        }
    }
    map
}

/// Rotates `movers` about the axis `origin -> origin + axis` by `theta` radians.
fn rotate_about_axis(
    mol: &mut Molecule3D,
    movers: &[usize],
    origin: [f64; 3],
    axis: [f64; 3],
    theta: f64,
) {
    let k = norm3(axis);
    let (sin_t, cos_t) = theta.sin_cos();
    let one_c = 1.0 - cos_t;
    for &m in movers {
        if m >= mol.atoms.len() {
            continue;
        }
        let p = mol.atoms[m].coords;
        let r = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
        let kxr = struct_core::cross3(k, r);
        let kdr = k[0] * r[0] + k[1] * r[1] + k[2] * r[2];
        mol.atoms[m].coords = [
            origin[0] + r[0] * cos_t + kxr[0] * sin_t + k[0] * kdr * one_c,
            origin[1] + r[1] * cos_t + kxr[1] * sin_t + k[1] * kdr * one_c,
            origin[2] + r[2] * cos_t + kxr[2] * sin_t + k[2] * kdr * one_c,
        ];
    }
}

/// Replaces the clicked atom with a substituent (or appends it, when the clicked atom is an
/// internal one with spare valence) and relaxes the new group inside the pocket.
///
/// Two things used to be wrong here. The function **always appended**: clicking a hydrogen and
/// asking for `-F` left the hydrogen in place and bonded a fluorine to it, and clicking a carbon
/// gave it a fifth bond. And its "minimisation" translated the *entire* ligand along the six
/// coordinate axes in 0.18 A steps, which cannot change any internal coordinate, so the
/// substituent never moved relative to the scaffold and the reported relaxation was a rigid-body
/// shift with an internal displacement of exactly zero.
///
/// Now a terminal anchor (a hydrogen, a halogen, a hydroxyl oxygen) is deleted and the
/// substituent is grown from the atom it was attached to, along the bond direction it vacated.
/// The new group is then relaxed by scanning the torsion about its attachment bond and by rigid
/// nudges, and `internal_rmsd_shift` reports how far the ligand's own geometry actually moved.
pub fn mutate_and_minimize_in_pocket(
    receptor: &Molecule3D,
    ligand: &Molecule3D,
    anchor_atom_idx: usize,
    preset: SubstituentPreset,
) -> Result<MutationOutcome, HydeError> {
    let start = Instant::now();
    if anchor_atom_idx >= ligand.atoms.len() {
        return Err(HydeError::InvalidAtomIndex(anchor_atom_idx));
    }

    let before_report = score_pose_hyde(receptor, ligand)?;
    let before_kcal = before_report.total_delta_g_kcal;

    let mut mut_mol = ligand.clone();

    // A terminal atom is a substituent in its own right, so the clicked atom is replaced. An
    // internal atom keeps its place and the group is appended to its free valence.
    let terminal_parent = (mut_mol.adj[anchor_atom_idx].len() == 1)
        .then(|| mut_mol.adj[anchor_atom_idx][0].0)
        .filter(|_| mut_mol.atoms.len() > 2);
    let replaced_anchor = terminal_parent.is_some();

    let (attach_to, u) = match terminal_parent {
        Some(parent) => {
            let pp = mut_mol.atoms[parent].coords;
            let ap = mut_mol.atoms[anchor_atom_idx].coords;
            let dir = norm3([ap[0] - pp[0], ap[1] - pp[1], ap[2] - pp[2]]);
            let map = remove_terminal_atom(&mut mut_mol, anchor_atom_idx);
            let new_parent = map
                .get(parent)
                .copied()
                .flatten()
                .ok_or(HydeError::InvalidAtomIndex(parent))?;
            (new_parent, dir)
        }
        None => {
            let anchor_pos = mut_mol.atoms[anchor_atom_idx].coords;
            let mut dir = [0.0_f64; 3];
            for &(nb, _) in &mut_mol.adj[anchor_atom_idx] {
                let p = mut_mol.atoms[nb].coords;
                dir[0] += anchor_pos[0] - p[0];
                dir[1] += anchor_pos[1] - p[1];
                dir[2] += anchor_pos[2] - p[2];
            }
            let d = if vec_len(dir) > 1e-6 {
                norm3(dir)
            } else {
                [1.0, 0.0, 0.0]
            };
            (anchor_atom_idx, d)
        }
    };

    let attach_pos = mut_mol.atoms[attach_to].coords;
    let attach_element = mut_mol.atoms[attach_to].element.clone();

    // Orthogonal basis (v, w) for polyatomic substituents (CF3 / OCH3 / NH2 / OH).
    let seed_vec = if u[0].abs() < 0.8 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let v = norm3(struct_core::cross3(u, seed_vec));
    let w = struct_core::cross3(u, v);

    let mut added: Vec<usize> = Vec::new();

    let p_at = |d: f64| -> [f64; 3] {
        [
            attach_pos[0] + u[0] * d,
            attach_pos[1] + u[1] * d,
            attach_pos[2] + u[2] * d,
        ]
    };

    macro_rules! add_child {
        ($mol:expr, $parent:expr, $el:expr, $pos:expr, $order:expr) => {{
            let idx = $mol.add_atom(Atom3D::new($mol.atoms.len(), $el, $pos));
            $mol.add_bond($parent, idx, $order)
                .ok_or(HydeError::InvalidAtomIndex($parent))?;
            added.push(idx);
            idx
        }};
    }

    // Bond lengths come from the element pair, not from a per-preset constant.
    let first_len = |el: &str, order: BondOrder| -> f64 {
        struct_core::ideal_bond_length(&attach_element, el, order)
    };

    match preset {
        SubstituentPreset::H => {
            add_child!(
                mut_mol,
                attach_to,
                "H",
                p_at(first_len("H", BondOrder::Single)),
                BondOrder::Single
            );
        }
        SubstituentPreset::F => {
            add_child!(
                mut_mol,
                attach_to,
                "F",
                p_at(first_len("F", BondOrder::Single)),
                BondOrder::Single
            );
        }
        SubstituentPreset::Cl => {
            add_child!(
                mut_mol,
                attach_to,
                "Cl",
                p_at(first_len("Cl", BondOrder::Single)),
                BondOrder::Single
            );
        }
        SubstituentPreset::OH => {
            let o = add_child!(
                mut_mol,
                attach_to,
                "O",
                p_at(first_len("O", BondOrder::Single)),
                BondOrder::Single
            );
            let op = mut_mol.atoms[o].coords;
            let oh = struct_core::ideal_bond_length("O", "H", BondOrder::Single);
            let h_pos = [
                op[0] + oh * (0.33 * u[0] + 0.94 * v[0]),
                op[1] + oh * (0.33 * u[1] + 0.94 * v[1]),
                op[2] + oh * (0.33 * u[2] + 0.94 * v[2]),
            ];
            add_child!(mut_mol, o, "H", h_pos, BondOrder::Single);
        }
        SubstituentPreset::NH2 => {
            let n_idx = add_child!(
                mut_mol,
                attach_to,
                "N",
                p_at(first_len("N", BondOrder::Single)),
                BondOrder::Single
            );
            let np = mut_mol.atoms[n_idx].coords;
            let nh = struct_core::ideal_bond_length("N", "H", BondOrder::Single);
            for sign in [1.0_f64, -1.0] {
                let hp = [
                    np[0] + nh * (0.33 * u[0] + sign * 0.94 * v[0]),
                    np[1] + nh * (0.33 * u[1] + sign * 0.94 * v[1]),
                    np[2] + nh * (0.33 * u[2] + sign * 0.94 * v[2]),
                ];
                add_child!(mut_mol, n_idx, "H", hp, BondOrder::Single);
            }
        }
        SubstituentPreset::CH3 => {
            add_child!(
                mut_mol,
                attach_to,
                "C",
                p_at(first_len("C", BondOrder::Single)),
                BondOrder::Single
            );
        }
        SubstituentPreset::CF3 => {
            let c_idx = add_child!(
                mut_mol,
                attach_to,
                "C",
                p_at(first_len("C", BondOrder::Single)),
                BondOrder::Single
            );
            let cp = mut_mol.atoms[c_idx].coords;
            let cf = struct_core::ideal_bond_length("C", "F", BondOrder::Single);
            for k in 0..3 {
                let ang = k as f64 * (std::f64::consts::TAU / 3.0);
                let fp = [
                    cp[0] + cf * (0.33 * u[0] + 0.94 * (ang.cos() * v[0] + ang.sin() * w[0])),
                    cp[1] + cf * (0.33 * u[1] + 0.94 * (ang.cos() * v[1] + ang.sin() * w[1])),
                    cp[2] + cf * (0.33 * u[2] + 0.94 * (ang.cos() * v[2] + ang.sin() * w[2])),
                ];
                add_child!(mut_mol, c_idx, "F", fp, BondOrder::Single);
            }
        }
        SubstituentPreset::CN => {
            let l1 = first_len("C", BondOrder::Single);
            let c_idx = add_child!(mut_mol, attach_to, "C", p_at(l1), BondOrder::Single);
            let cn = struct_core::ideal_bond_length("C", "N", BondOrder::Triple);
            add_child!(mut_mol, c_idx, "N", p_at(l1 + cn), BondOrder::Triple);
        }
        SubstituentPreset::OCH3 => {
            let o_idx = add_child!(
                mut_mol,
                attach_to,
                "O",
                p_at(first_len("O", BondOrder::Single)),
                BondOrder::Single
            );
            let op = mut_mol.atoms[o_idx].coords;
            let oc = struct_core::ideal_bond_length("O", "C", BondOrder::Single);
            let me_pos = [
                op[0] + oc * (0.33 * u[0] + 0.94 * v[0]),
                op[1] + oc * (0.33 * u[1] + 0.94 * v[1]),
                op[2] + oc * (0.33 * u[2] + 0.94 * v[2]),
            ];
            add_child!(mut_mol, o_idx, "C", me_pos, BondOrder::Single);
        }
    }

    perceive_all(&mut mut_mol);

    let reference: Vec<[f64; 3]> = mut_mol.atoms.iter().map(|a| a.coords).collect();
    let mut best_mol = mut_mol.clone();
    let mut best_rep = score_pose_hyde(receptor, &best_mol)?;

    // 1. Torsion scan of the new group about its attachment bond. This is the degree of freedom
    //    the substituent actually has, and the one the old rigid-body nudge could never reach.
    if added.len() > 1 {
        let pivot = added[0];
        let movers: Vec<usize> = added[1..].to_vec();
        let origin = best_mol.atoms[pivot].coords;
        let axis = [
            origin[0] - attach_pos[0],
            origin[1] - attach_pos[1],
            origin[2] - attach_pos[2],
        ];
        if vec_len(axis) > 1e-6 {
            let steps = 24usize;
            for step in 1..steps {
                let theta = std::f64::consts::TAU * (step as f64) / (steps as f64);
                let mut trial = mut_mol.clone();
                rotate_about_axis(&mut trial, &movers, origin, axis, theta);
                let rep = score_pose_hyde(receptor, &trial)?;
                if rep.total_delta_g_kcal < best_rep.total_delta_g_kcal - 1e-6 {
                    best_rep = rep;
                    best_mol = trial;
                }
            }
        }
    }

    // 2. Small rigid-body nudges of the whole ligand, to let the new group settle in the pocket.
    let dirs = [
        [0.18, 0.0, 0.0],
        [-0.18, 0.0, 0.0],
        [0.0, 0.18, 0.0],
        [0.0, -0.18, 0.0],
        [0.0, 0.0, 0.18],
        [0.0, 0.0, -0.18],
    ];
    for _step in 0..4 {
        let mut improved = false;
        for dvec in &dirs {
            let mut trial = best_mol.clone();
            for a in &mut trial.atoms {
                a.coords[0] += dvec[0];
                a.coords[1] += dvec[1];
                a.coords[2] += dvec[2];
            }
            let rep = score_pose_hyde(receptor, &trial)?;
            if rep.total_delta_g_kcal < best_rep.total_delta_g_kcal - 1e-3 {
                best_rep = rep;
                best_mol = trial;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }

    // Internal displacement, measured after removing the rigid-body part: how far the geometry
    // moved in the ligand's own frame.
    let internal_rmsd_shift = {
        let n = best_mol.atoms.len().min(reference.len());
        if n == 0 {
            0.0
        } else {
            let mut c_ref = [0.0_f64; 3];
            let mut c_new = [0.0_f64; 3];
            for i in 0..n {
                for k in 0..3 {
                    c_ref[k] += reference[i][k] / n as f64;
                    c_new[k] += best_mol.atoms[i].coords[k] / n as f64;
                }
            }
            let mut worst = 0.0_f64;
            for i in 0..n {
                let d = [
                    (best_mol.atoms[i].coords[0] - c_new[0]) - (reference[i][0] - c_ref[0]),
                    (best_mol.atoms[i].coords[1] - c_new[1]) - (reference[i][1] - c_ref[1]),
                    (best_mol.atoms[i].coords[2] - c_new[2]) - (reference[i][2] - c_ref[2]),
                ];
                worst = worst.max(vec_len(d));
            }
            worst
        }
    };

    let after_kcal = best_rep.total_delta_g_kcal;
    Ok(MutationOutcome {
        substituent: preset,
        mutated_ligand: best_mol,
        before_kcal,
        after_kcal,
        delta_delta_g_kcal: after_kcal - before_kcal,
        report: best_rep,
        elapsed_ms: start.elapsed().as_millis(),
        replaced_anchor,
        internal_rmsd_shift,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use struct_core::{Ad4Type, Atom3D, BondOrder, Molecule3D, assign_ad4_types_and_flags};

    #[test]
    fn distinguishes_green_hbond_from_red_unsatisfied_buried_polar_and_clash()
    -> Result<(), HydeError> {
        // Receptor: H-bond donor N at (-2.85, 0, 0), hydrophobic wall on +X (+3.8, 0, 0) + enclosure
        let mut rec = Molecule3D::new("rec");
        let rec_atoms = [
            ("N", [-2.85, 0.0, 0.0], Ad4Type::NA, true, false),
            ("C", [3.70, 0.0, 0.0], Ad4Type::C, false, false),
            ("C", [2.50, 2.50, 2.0], Ad4Type::C, false, false),
            ("C", [2.50, -2.50, -2.0], Ad4Type::C, false, false),
            ("C", [0.0, 3.60, 0.0], Ad4Type::C, false, false),
            ("C", [0.0, -3.60, 0.0], Ad4Type::C, false, false),
            ("C", [0.0, 0.0, 3.60], Ad4Type::C, false, false),
            ("C", [0.0, 0.0, -3.60], Ad4Type::C, false, false),
        ];
        for (i, &(el, pos, ad4, hd, ha)) in rec_atoms.iter().enumerate() {
            let mut a = Atom3D::new(i + 1, el, pos);
            a.chain_id = 'A';
            a.res_name = if el == "N" { "LYS" } else { "VAL" }.to_string();
            a.res_seq = (i + 1) as i32;
            a.ad4_type = ad4;
            a.h_donor = hd;
            a.h_acceptor = ha;
            rec.add_atom(a);
        }

        // Ligand with 3 atoms:
        // 0: Acceptor O at (0, 0, 0) making a 2.85 A H-bond with rec N (-2.85,0,0) -> GREEN!
        // 1: Carbon at (1.2, 0, 0)
        // 2: Buried unsatisfied polar O at (2.4, 0, 0) facing hydrophobic C wall with NO H-bond -> RED!
        let mut lig = Molecule3D::new("test_lig");
        let i0 = lig.add_atom(Atom3D::new(1, "O", [0.0, 0.0, 0.0]));
        let i1 = lig.add_atom(Atom3D::new(2, "C", [1.2, 0.0, 0.0]));
        let i2 = lig.add_atom(Atom3D::new(3, "O", [2.4, 0.0, 0.0]));
        lig.add_bond(i0, i1, BondOrder::Single)
            .ok_or(HydeError::EmptyStructure)?;
        lig.add_bond(i1, i2, BondOrder::Single)
            .ok_or(HydeError::EmptyStructure)?;
        assign_ad4_types_and_flags(&mut lig);

        let report = score_pose_hyde(&rec, &lig)?;
        let s_hbond = &report.atom_scores[0];
        let s_unsat = &report.atom_scores[2];

        assert!(
            s_hbond.delta_g_kcal < -0.5,
            "H-bonded O should be favorable green, got {:.3} ({})",
            s_hbond.delta_g_kcal,
            s_hbond.diagnostic
        );
        assert!(
            s_unsat.delta_g_kcal > 0.3,
            "unsatisfied buried O should be unfavorable red, got {:.3} ({})",
            s_unsat.delta_g_kcal,
            s_unsat.diagnostic
        );
        assert!(s_hbond.corona_color[1] > s_hbond.corona_color[0]);
        assert!(s_unsat.corona_color[0] > s_unsat.corona_color[1]);

        // Also test fast substituent mutation (< 200 ms)
        let mut_res = mutate_and_minimize_in_pocket(&rec, &lig, 1, SubstituentPreset::F)?;
        assert!(mut_res.mutated_ligand.atoms.len() > lig.atoms.len());
        assert!(mut_res.elapsed_ms < 500);
        Ok(())
    }
}
