use std::collections::{BTreeSet, HashMap, VecDeque};
use std::f64::consts::PI;

use crate::error::StructError;
use crate::types::{Ad4Type, Atom3D, BondOrder, Molecule3D, RotatableBond};

/// Maximum atoms permitted in a single SMILES input parsed to 3D.
pub const MAX_SMILES_3D_ATOMS: usize = 512;

/// Clean a protein receptor: strip non-structural waters (`HOH`/`WAT`/`DOD`) when requested,
/// infer missing covalent bonds, perceive rings/aromaticity, protonate at pH 7.4, add missing
/// polar hydrogens, compute Gasteiger partial charges, and assign AutoDock4 atom types.
pub fn prepare_receptor(mol: &mut Molecule3D, strip_waters: bool) {
    if strip_waters {
        keep_non_water_atoms(mol);
    }
    mol.infer_bonds_by_distance();
    perceive_all(mol);
}

fn keep_non_water_atoms(mol: &mut Molecule3D) {
    let mut keep_map = vec![None; mol.atoms.len()];
    let mut kept = Vec::with_capacity(mol.atoms.len());
    for (idx, mut a) in mol.atoms.drain(..).enumerate() {
        let rn = a.res_name.trim().to_ascii_uppercase();
        if matches!(rn.as_str(), "HOH" | "WAT" | "DOD" | "TIP" | "TIP3") {
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

/// Full chemical & AutoDock4 perception: rings, aromaticity, pH 7.4 protonation + polar H,
/// Gasteiger partial charges, AD4 types, donor/acceptor/hydrophobic flags, and rotatable bonds.
pub fn perceive_all(mol: &mut Molecule3D) {
    mol.detect_small_rings();
    perceive_aromatic_residues_and_rings(mol);
    apply_ph74_protonation_and_polar_h(mol);
    assign_gasteiger_charges(mol);
    assign_ad4_types_and_flags(mol);
    build_rotatable_bonds(mol);
}

fn perceive_aromatic_residues_and_rings(mol: &mut Molecule3D) {
    // 1. Protein canonical aromatic side chains by residue + atom name
    for a in &mut mol.atoms {
        let rn = a.res_name.trim().to_ascii_uppercase();
        let an = a.name.trim().to_ascii_uppercase();
        let is_prot_arom = matches!(
            (rn.as_str(), an.as_str()),
            ("PHE" | "TYR", "CG" | "CD1" | "CD2" | "CE1" | "CE2" | "CZ")
                | (
                    "TRP",
                    "CG" | "CD1" | "CD2" | "NE1" | "CE2" | "CE3" | "CZ2" | "CZ3" | "CH2"
                )
                | (
                    "HIS" | "HID" | "HIE" | "HIP",
                    "CG" | "ND1" | "CD2" | "CE1" | "NE2"
                )
        );
        if is_prot_arom {
            a.aromatic = true;
        }
    }

    // 2. Planarity/unsaturated 5/6-membered ring aromaticity (for ligands & non-standard PDB hetatms)
    for ring in &mol.rings {
        if !(5..=6).contains(&ring.len()) {
            continue;
        }
        let all_marked = ring.iter().all(|&v| mol.atoms[v].aromatic);
        let has_arom_bond = ring.windows(2).any(|w| {
            mol.adj[w[0]]
                .iter()
                .find(|&&(nb, _)| nb == w[1])
                .is_some_and(|&(_, bid)| mol.bonds[bid].order == BondOrder::Aromatic)
        });
        let planar = ring_is_planar(mol, ring);
        let has_unsat = ring.iter().any(|&v| {
            mol.adj[v].iter().any(|&(_, bid)| {
                matches!(
                    mol.bonds[bid].order,
                    BondOrder::Double | BondOrder::Aromatic
                )
            })
        });
        let valid_elems = ring
            .iter()
            .all(|&v| matches!(mol.atoms[v].element.as_str(), "C" | "N" | "O" | "S"));
        if all_marked
            || (valid_elems
                && (has_arom_bond || (planar && (has_unsat || deg_at_most_3(mol, ring)))))
        {
            for &v in ring {
                if !mol.atoms[v].is_backbone {
                    mol.atoms[v].aromatic = true;
                }
            }
        }
    }

    for b in &mut mol.bonds {
        if mol.atoms[b.u].aromatic && mol.atoms[b.v].aromatic && !mol.atoms[b.u].is_backbone {
            // Confirm edge belongs to same ring
            let same_ring = mol
                .rings
                .iter()
                .any(|r| r.contains(&b.u) && r.contains(&b.v));
            if same_ring {
                b.order = BondOrder::Aromatic;
            }
        }
    }
}

fn deg_at_most_3(mol: &Molecule3D, ring: &[usize]) -> bool {
    // Only consider inferred single-bond SDF/PDB 6-rings aromatic if heavy degree <= 3 AND ring is flat
    // AND has C-C lengths <= 1.43 A (distinguishing benzene 1.39 A from cyclohexane 1.54 A!)
    let mut sum_len = 0.0;
    let mut cnt = 0.0;
    for w in 0..ring.len() {
        let u = ring[w];
        let v = ring[(w + 1) % ring.len()];
        let pu = mol.atoms[u].coords;
        let pv = mol.atoms[v].coords;
        sum_len +=
            ((pu[0] - pv[0]).powi(2) + (pu[1] - pv[1]).powi(2) + (pu[2] - pv[2]).powi(2)).sqrt();
        cnt += 1.0;
        let hdeg = mol.adj[u]
            .iter()
            .filter(|&&(nb, _)| mol.atoms[nb].is_heavy())
            .count();
        if hdeg > 3 {
            return false;
        }
    }
    cnt > 0.0 && (sum_len / cnt) <= 1.44
}

fn ring_is_planar(mol: &Molecule3D, ring: &[usize]) -> bool {
    if ring.len() < 4 {
        return true;
    }
    let p0 = mol.atoms[ring[0]].coords;
    let p1 = mol.atoms[ring[1]].coords;
    let p2 = mol.atoms[ring[2]].coords;
    let a = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let b = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let nx = a[1] * b[2] - a[2] * b[1];
    let ny = a[2] * b[0] - a[0] * b[2];
    let nz = a[0] * b[1] - a[1] * b[0];
    let nlen = (nx * nx + ny * ny + nz * nz).sqrt();
    if nlen < 1e-4 {
        return false;
    }
    for &idx in &ring[3..] {
        let p = mol.atoms[idx].coords;
        let d = ((p[0] - p0[0]) * nx + (p[1] - p0[1]) * ny + (p[2] - p0[2]) * nz).abs() / nlen;
        if d > 0.12 {
            return false;
        }
    }
    true
}

/// Protonate acidic/basic centers at physiological pH 7.4 and synthesize missing polar H atoms
/// so H-bond donors (`HD`) have realistic 3D vectors for Vina/HYDE geometric scoring.
fn apply_ph74_protonation_and_polar_h(mol: &mut Molecule3D) {
    let n_orig = mol.atoms.len();
    // Pre-mark formal charges for pH 7.4 ionizable protein/ligand groups
    for i in 0..n_orig {
        let el = mol.atoms[i].element.clone();
        let an = mol.atoms[i].name.trim().to_ascii_uppercase();
        let rn = mol.atoms[i].res_name.trim().to_ascii_uppercase();

        if !mol.atoms[i].is_hetatm {
            // Protein standard residues at pH 7.4
            if matches!(
                (rn.as_str(), an.as_str()),
                ("ASP", "OD1" | "OD2") | ("GLU", "OE1" | "OE2")
            ) {
                mol.atoms[i].formal_charge = if an.ends_with('2') { -1 } else { 0 };
            } else if matches!((rn.as_str(), an.as_str()), ("LYS", "NZ") | ("ARG", "NH1")) {
                mol.atoms[i].formal_charge = 1;
            }
            continue;
        }

        // Ligand carboxylic / sulfonic / phosphonic deprotonation & aliphatic amine protonation
        if el == "O" && mol.adj[i].len() == 1 {
            let c_nbr = mol.adj[i][0].0;
            if matches!(mol.atoms[c_nbr].element.as_str(), "C" | "S" | "P") {
                let double_o = mol.adj[c_nbr].iter().any(|&(nb, bid)| {
                    nb != i
                        && mol.atoms[nb].element == "O"
                        && matches!(
                            mol.bonds[bid].order,
                            BondOrder::Double | BondOrder::Aromatic
                        )
                });
                let single_oh = mol.adj[i]
                    .iter()
                    .filter(|&&(nb, _)| mol.atoms[nb].element != "H")
                    .count()
                    == 1;
                if double_o && single_oh && mol.bonds[mol.adj[i][0].1].order == BondOrder::Single {
                    mol.atoms[i].formal_charge = -1;
                }
            }
        } else if el == "N" && !mol.atoms[i].aromatic && mol.atoms[i].formal_charge == 0 {
            let heavy_nbrs: Vec<(usize, BondOrder)> = mol.adj[i]
                .iter()
                .filter(|&&(nb, _)| mol.atoms[nb].is_heavy())
                .map(|&(nb, bid)| (nb, mol.bonds[bid].order))
                .collect();
            let all_single = heavy_nbrs.iter().all(|&(_, bo)| bo == BondOrder::Single);
            let near_ewg = heavy_nbrs.iter().any(|&(c, _)| {
                mol.atoms[c].aromatic
                    || mol.adj[c].iter().any(|&(x, bid)| {
                        x != i
                            && matches!(mol.atoms[x].element.as_str(), "O" | "S")
                            && mol.bonds[bid].order == BondOrder::Double
                    })
            });
            // Aliphatic amine without carbonyl/aryl conjugation -> protonated at pH 7.4 (+1)
            if all_single && !heavy_nbrs.is_empty() && heavy_nbrs.len() <= 3 && !near_ewg {
                mol.atoms[i].formal_charge = 1;
            }
        }
    }

    // Add missing polar H atoms on donors (`N`, `O`, `S`) that currently lack explicit bonded H
    let mut to_add: Vec<(usize, [f64; 3], String, char, i32, bool)> = Vec::new();
    for i in 0..n_orig {
        let a = &mol.atoms[i];
        let el = a.element.as_str();
        if !matches!(el, "N" | "O" | "S") {
            continue;
        }
        let existing_h = mol.adj[i]
            .iter()
            .filter(|&&(nb, _)| mol.atoms[nb].element == "H")
            .count();
        let bond_order_sum: usize = mol.adj[i]
            .iter()
            .map(|&(nb, bid)| {
                if mol.atoms[nb].element == "H" {
                    0
                } else {
                    match mol.bonds[bid].order {
                        BondOrder::Double => 2,
                        BondOrder::Triple => 3,
                        BondOrder::Single | BondOrder::Aromatic => 1,
                    }
                }
            })
            .sum();

        let target_h = match el {
            "N" => {
                if a.aromatic {
                    // Aromatic pyridine/imidazole nitrogen without substituent and with H if pyrrole/Trp NE1
                    let rn = a.res_name.trim().to_ascii_uppercase();
                    let an = a.name.trim().to_ascii_uppercase();
                    if (rn == "TRP" && an == "NE1")
                        || ((rn == "HIS" || rn == "HID") && an == "ND1")
                        || (rn == "HIE" && an == "NE2")
                    {
                        1
                    } else {
                        0
                    }
                } else {
                    let target_val: usize = if a.formal_charge > 0 {
                        4
                    } else if a.formal_charge < 0 {
                        2
                    } else {
                        3
                    };
                    target_val.saturating_sub(bond_order_sum)
                }
            }
            "O" => {
                if a.formal_charge < 0 {
                    0
                } else {
                    2usize.saturating_sub(bond_order_sum)
                }
            }
            "S" => {
                if a.formal_charge < 0 || a.aromatic || bond_order_sum >= 2 {
                    0
                } else {
                    2usize.saturating_sub(bond_order_sum)
                }
            }
            _ => 0,
        };

        let missing_h = target_h.saturating_sub(existing_h);
        if missing_h == 0 || mol.adj[i].is_empty() {
            continue;
        }
        let dirs = ideal_h_directions(mol, i, missing_h);
        let bl = if el == "S" { 1.33 } else { 1.01 };
        for d in dirs {
            let h_pos = [
                a.coords[0] + d[0] * bl,
                a.coords[1] + d[1] * bl,
                a.coords[2] + d[2] * bl,
            ];
            to_add.push((
                i,
                h_pos,
                a.res_name.clone(),
                a.chain_id,
                a.res_seq,
                a.is_hetatm,
            ));
        }
    }

    for (parent, pos, res_name, chain_id, res_seq, is_hetatm) in to_add {
        let mut h = Atom3D::new(0, "H", pos);
        h.name = "H".to_string();
        h.res_name = res_name;
        h.chain_id = chain_id;
        h.res_seq = res_seq;
        h.is_hetatm = is_hetatm;
        h.ad4_type = Ad4Type::HD;
        let hid = mol.add_atom(h);
        let _ = mol.add_bond(parent, hid, BondOrder::Single);
    }
}

fn ideal_h_directions(mol: &Molecule3D, center: usize, count: usize) -> Vec<[f64; 3]> {
    let pc = mol.atoms[center].coords;
    let mut sum = [0.0; 3];
    let mut nbr_vecs = Vec::new();
    for &(nb, _) in &mol.adj[center] {
        let pn = mol.atoms[nb].coords;
        let v = norm3([pn[0] - pc[0], pn[1] - pc[1], pn[2] - pc[2]]);
        sum[0] += v[0];
        sum[1] += v[1];
        sum[2] += v[2];
        nbr_vecs.push(v);
    }
    let away = norm3([-sum[0], -sum[1], -sum[2]]);
    let ref_v = if away[0].hypot(away[1]).hypot(away[2]) < 1e-4 {
        [1.0, 0.0, 0.0]
    } else {
        away
    };
    if count == 1 {
        return vec![ref_v];
    }
    // Build orthonormal basis around ref_v to fan out 2 or 3 NH2/NH3+ hydrogens cleanly
    let arb = if ref_v[0].abs() < 0.8 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u = norm3(cross3(ref_v, arb));
    let w = cross3(ref_v, u);
    let cone_cos = (54.7_f64.to_radians()).cos();
    let cone_sin = (54.7_f64.to_radians()).sin();
    let mut out = Vec::with_capacity(count);
    for k in 0..count {
        let phi = 2.0 * PI * (k as f64) / (count as f64);
        let d = [
            ref_v[0] * cone_cos + (u[0] * phi.cos() + w[0] * phi.sin()) * cone_sin,
            ref_v[1] * cone_cos + (u[1] * phi.cos() + w[1] * phi.sin()) * cone_sin,
            ref_v[2] * cone_cos + (u[2] * phi.cos() + w[2] * phi.sin()) * cone_sin,
        ];
        out.push(norm3(d));
    }
    out
}

/// 6-iteration Gasteiger-Marsili PEOE partial charges (damping 0.5^k).
pub fn assign_gasteiger_charges(mol: &mut Molecule3D) {
    let n = mol.atoms.len();
    if n == 0 {
        return;
    }
    let mut q: Vec<f64> = mol
        .atoms
        .iter()
        .map(|a| f64::from(a.formal_charge) * 0.5)
        .collect();
    let mut params: Vec<(f64, f64, f64)> = Vec::with_capacity(n);
    for (i, a) in mol.atoms.iter().enumerate() {
        let has_dbl = mol.adj[i]
            .iter()
            .any(|&(_, bid)| mol.bonds[bid].order == BondOrder::Double);
        let has_trp = mol.adj[i]
            .iter()
            .any(|&(_, bid)| mol.bonds[bid].order == BondOrder::Triple);
        params.push(gasteiger_abc(&a.element, has_dbl || a.aromatic, has_trp));
    }

    let mut damp = 0.5_f64;
    for _ in 0..6 {
        let chi: Vec<f64> = params
            .iter()
            .zip(q.iter())
            .map(|(&(a, b, c), &qi)| a + b * qi + c * qi * qi)
            .collect();
        let mut dq = vec![0.0; n];
        for bond in &mol.bonds {
            let i = bond.u;
            let j = bond.v;
            let (donor, acceptor, d_chi) = if chi[j] > chi[i] {
                (i, j, chi[j] - chi[i])
            } else {
                (j, i, chi[i] - chi[j])
            };
            let denom = (params[donor].0 + params[donor].1 + params[donor].2).max(4.0);
            let flow = (d_chi / denom) * damp;
            dq[donor] += flow;
            dq[acceptor] -= flow;
        }
        for k in 0..n {
            q[k] += dq[k];
        }
        damp *= 0.5;
    }

    for (a, qi) in mol.atoms.iter_mut().zip(q) {
        a.charge = qi.clamp(-2.0, 2.0);
    }
}

fn gasteiger_abc(el: &str, sp2: bool, sp: bool) -> (f64, f64, f64) {
    match el {
        "H" => (7.17, 6.24, -0.56),
        "C" if sp => (10.39, 9.45, 0.73),
        "C" if sp2 => (8.79, 9.32, 1.51),
        "C" => (7.98, 9.18, 1.88),
        "N" if sp => (15.68, 11.70, -0.27),
        "N" if sp2 => (12.87, 11.15, 0.85),
        "N" => (11.54, 10.82, 1.36),
        "O" if sp2 => (17.07, 13.79, 0.47),
        "O" => (14.18, 12.92, 1.39),
        "F" => (14.66, 13.85, 2.31),
        "Cl" => (11.00, 9.69, 1.35),
        "Br" => (10.08, 8.47, 1.16),
        "I" => (9.90, 7.96, 0.96),
        "S" if sp2 => (10.88, 9.49, 1.33),
        "S" => (10.14, 9.13, 1.38),
        "P" => (8.90, 8.24, 0.96),
        _ => (7.98, 9.18, 1.88),
    }
}

/// Assign AutoDock4 atom types (`C, A, N, NA, OA, SA, S, HD, H, F, Cl, Br, I, P`) and
/// H-bond donor/acceptor/hydrophobic flags.
pub fn assign_ad4_types_and_flags(mol: &mut Molecule3D) {
    for i in 0..mol.atoms.len() {
        let el = mol.atoms[i].element.clone();
        let arom = mol.atoms[i].aromatic;
        let bonded_h = mol.adj[i]
            .iter()
            .filter(|&&(nb, _)| mol.atoms[nb].element == "H")
            .count();
        let heavy_deg = mol.adj[i]
            .iter()
            .filter(|&&(nb, _)| mol.atoms[nb].is_heavy())
            .count();
        let has_polar_parent = mol.adj[i]
            .iter()
            .any(|&(nb, _)| matches!(mol.atoms[nb].element.as_str(), "N" | "O" | "S"));

        let ad4 = match el.as_str() {
            "C" => {
                if arom {
                    Ad4Type::A
                } else {
                    Ad4Type::C
                }
            }
            "N" => {
                if mol.atoms[i].formal_charge > 0 || bonded_h >= 2 || is_amide_nitrogen(mol, i) {
                    Ad4Type::N
                } else if bonded_h == 1 || heavy_deg <= 3 {
                    Ad4Type::NA
                } else {
                    Ad4Type::N
                }
            }
            "O" => Ad4Type::OA,
            "S" => {
                if heavy_deg <= 2 {
                    Ad4Type::SA
                } else {
                    Ad4Type::S
                }
            }
            "H" => {
                if has_polar_parent {
                    Ad4Type::HD
                } else {
                    Ad4Type::H
                }
            }
            "F" => Ad4Type::F,
            "Cl" => Ad4Type::Cl,
            "Br" => Ad4Type::Br,
            "I" => Ad4Type::I,
            "P" => Ad4Type::P,
            "Zn" => Ad4Type::Zn,
            "Mg" => Ad4Type::Mg,
            "Fe" => Ad4Type::Fe,
            "Ca" => Ad4Type::Ca,
            "Mn" => Ad4Type::Mn,
            _ => Ad4Type::X,
        };

        let has_on_neighbor = mol.adj[i]
            .iter()
            .any(|&(nb, _)| matches!(mol.atoms[nb].element.as_str(), "O" | "N"));
        let a = &mut mol.atoms[i];
        a.ad4_type = ad4;
        a.h_donor = bonded_h > 0 && matches!(el.as_str(), "N" | "O" | "S");
        a.h_acceptor = ad4.is_acceptor() && a.formal_charge <= 0;
        a.hydrophobic = ad4.is_hydrophobic() && !(el == "C" && has_on_neighbor);
    }
}

fn is_amide_nitrogen(mol: &Molecule3D, idx: usize) -> bool {
    mol.adj[idx].iter().any(|&(c, _)| {
        mol.atoms[c].element == "C"
            && mol.adj[c].iter().any(|&(o, bid)| {
                o != idx
                    && matches!(mol.atoms[o].element.as_str(), "O" | "S")
                    && mol.bonds[bid].order == BondOrder::Double
            })
    })
}

/// Extract non-ring, non-terminal rotatable single bonds anchored from the largest rigid component.
pub fn build_rotatable_bonds(mol: &mut Molecule3D) {
    mol.rotatable_bonds.clear();
    if mol.atoms.len() > 250 || mol.bonds.is_empty() {
        // Large receptors stay rigid; rotatable bonds are built for ligands/peptides <= 250 atoms
        return;
    }

    let mut ring_edges: BTreeSet<usize> = BTreeSet::new();
    for r in &mol.rings {
        for w in 0..r.len() {
            let u = r[w];
            let v = r[(w + 1) % r.len()];
            if let Some(&(_, bid)) = mol.adj[u].iter().find(|&&(nb, _)| nb == v) {
                ring_edges.insert(bid);
            }
        }
    }

    // Choose rigid root anchor: atom in most rings or closest to heavy centroid
    let root = (0..mol.atoms.len())
        .filter(|&i| mol.atoms[i].is_heavy())
        .max_by_key(|&i| {
            let r_cnt = mol.rings.iter().filter(|r| r.contains(&i)).count();
            (r_cnt, mol.adj[i].len())
        })
        .unwrap_or(0);

    for b in &mol.bonds {
        if b.order != BondOrder::Single || ring_edges.contains(&b.id) {
            continue;
        }
        if !mol.atoms[b.u].is_heavy() || !mol.atoms[b.v].is_heavy() {
            continue;
        }
        let h_deg_u = mol.adj[b.u]
            .iter()
            .filter(|&&(nb, _)| mol.atoms[nb].is_heavy())
            .count();
        let h_deg_v = mol.adj[b.v]
            .iter()
            .filter(|&&(nb, _)| mol.atoms[nb].is_heavy())
            .count();
        if h_deg_u <= 1 || h_deg_v <= 1 {
            continue; // Terminal bond (CH3, OH, halogen, NH3+)
        }
        // Exclude symmetric `-CX3` / `-NH3+` rotation caps
        if is_symmetric_cap(mol, b.u, b.v) || is_symmetric_cap(mol, b.v, b.u) {
            continue;
        }
        // Check component split removing bond `b.id`: keep root stationary, move the other half!
        let side_u = bfs_component_without_bond(mol, b.u, b.id);
        if side_u.contains(&b.v) {
            continue; // Large non-SSSR ring cycle
        }
        let side_v = bfs_component_without_bond(mol, b.v, b.id);
        let (anchor, tip, moving) = if side_u.contains(&root) {
            (b.u, b.v, side_v)
        } else {
            (b.v, b.u, side_u)
        };
        if !moving.is_empty() {
            mol.rotatable_bonds.push(RotatableBond {
                bond_id: b.id,
                u: anchor,
                v: tip,
                moving_atoms: moving,
            });
        }
    }
}

fn is_symmetric_cap(mol: &Molecule3D, node: usize, parent: usize) -> bool {
    let others: Vec<usize> = mol.adj[node]
        .iter()
        .map(|&(nb, _)| nb)
        .filter(|&nb| nb != parent)
        .collect();
    if others.len() != 3 {
        return false;
    }
    let e0 = &mol.atoms[others[0]].element;
    others.iter().all(|&o| &mol.atoms[o].element == e0)
}

fn bfs_component_without_bond(mol: &Molecule3D, start: usize, skip_bid: usize) -> Vec<usize> {
    let mut visited = vec![false; mol.atoms.len()];
    let mut q = VecDeque::new();
    let mut out = Vec::new();
    visited[start] = true;
    q.push_back(start);
    while let Some(u) = q.pop_front() {
        out.push(u);
        for &(v, bid) in &mol.adj[u] {
            if bid != skip_bid && !visited[v] {
                visited[v] = true;
                q.push_back(v);
            }
        }
    }
    out
}

/// Clean-room 3D conformer builder from a SMILES string (supports rings, branches, aromaticity,
/// double/triple bonds, halogens, charges, and pH 7.4 preparation + 3D force-field relaxation).
pub fn parse_smiles_to_3d(input: &str, name: Option<&str>) -> Result<Molecule3D, StructError> {
    let smi = input.trim();
    if smi.is_empty() {
        return Err(StructError::EmptyInput);
    }
    let bytes = smi.as_bytes();
    let mut mol = Molecule3D::new(name.unwrap_or(smi));
    mol.source_smiles = Some(smi.to_string());

    let mut stack: Vec<usize> = Vec::new();
    let mut current: Option<usize> = None;
    let mut pending_bond: Option<BondOrder> = None;
    let mut rings: HashMap<u16, (usize, Option<BondOrder>, usize)> = HashMap::new();
    let mut bracket_hs: Vec<(usize, u8)> = Vec::new();
    let mut i = 0usize;

    while i < bytes.len() {
        let ch = bytes[i];
        match ch {
            b'.' => {
                current = None;
                pending_bond = None;
                i += 1;
            }
            b'(' => {
                let cur = current.ok_or(StructError::InvalidSmiles {
                    pos: i,
                    msg: "branch '(' without atom".to_string(),
                })?;
                stack.push(cur);
                i += 1;
            }
            b')' => {
                current = Some(stack.pop().ok_or(StructError::InvalidSmiles {
                    pos: i,
                    msg: "unmatched ')'".to_string(),
                })?);
                i += 1;
            }
            b'-' => {
                pending_bond = Some(BondOrder::Single);
                i += 1;
            }
            b'=' => {
                pending_bond = Some(BondOrder::Double);
                i += 1;
            }
            b'#' => {
                pending_bond = Some(BondOrder::Triple);
                i += 1;
            }
            b':' => {
                pending_bond = Some(BondOrder::Aromatic);
                i += 1;
            }
            b'/' | b'\\' => {
                i += 1;
            }
            b'%' | b'0'..=b'9' => {
                let (rnum, next_i) = if ch == b'%' {
                    if i + 2 >= bytes.len()
                        || !bytes[i + 1].is_ascii_digit()
                        || !bytes[i + 2].is_ascii_digit()
                    {
                        return Err(StructError::InvalidSmiles {
                            pos: i,
                            msg: "invalid '%' ring tag".to_string(),
                        });
                    }
                    let d = ((bytes[i + 1] - b'0') * 10 + (bytes[i + 2] - b'0')) as u16;
                    (d, i + 3)
                } else {
                    (u16::from(ch - b'0'), i + 1)
                };
                let cur = current.ok_or(StructError::InvalidSmiles {
                    pos: i,
                    msg: "ring closure without atom".to_string(),
                })?;
                if let Some((other, bo1, _)) = rings.remove(&rnum) {
                    let bo = bo1.or(pending_bond).unwrap_or_else(|| {
                        if mol.atoms[other].aromatic && mol.atoms[cur].aromatic {
                            BondOrder::Aromatic
                        } else {
                            BondOrder::Single
                        }
                    });
                    let _ = mol.add_bond(other, cur, bo);
                } else {
                    rings.insert(rnum, (cur, pending_bond, i));
                }
                pending_bond = None;
                i = next_i;
            }
            b'[' => {
                let (atom, exp_h, next_i) = parse_smiles_bracket(bytes, i, mol.atoms.len())?;
                if mol.atoms.len() >= MAX_SMILES_3D_ATOMS {
                    return Err(StructError::TooManyAtoms {
                        count: mol.atoms.len() + 1,
                        limit: MAX_SMILES_3D_ATOMS,
                    });
                }
                let nid = mol.add_atom(atom);
                if exp_h > 0 {
                    bracket_hs.push((nid, exp_h));
                }
                if let Some(prev) = current {
                    let bo = pending_bond.unwrap_or_else(|| {
                        if mol.atoms[prev].aromatic && mol.atoms[nid].aromatic {
                            BondOrder::Aromatic
                        } else {
                            BondOrder::Single
                        }
                    });
                    let _ = mol.add_bond(prev, nid, bo);
                }
                current = Some(nid);
                pending_bond = None;
                i = next_i;
            }
            _ => {
                let (atom, next_i) = parse_smiles_organic(bytes, i, mol.atoms.len())?;
                if mol.atoms.len() >= MAX_SMILES_3D_ATOMS {
                    return Err(StructError::TooManyAtoms {
                        count: mol.atoms.len() + 1,
                        limit: MAX_SMILES_3D_ATOMS,
                    });
                }
                let nid = mol.add_atom(atom);
                if let Some(prev) = current {
                    let bo = pending_bond.unwrap_or_else(|| {
                        if mol.atoms[prev].aromatic && mol.atoms[nid].aromatic {
                            BondOrder::Aromatic
                        } else {
                            BondOrder::Single
                        }
                    });
                    let _ = mol.add_bond(prev, nid, bo);
                }
                current = Some(nid);
                pending_bond = None;
                i = next_i;
            }
        }
    }

    if !stack.is_empty() {
        return Err(StructError::InvalidSmiles {
            pos: bytes.len(),
            msg: "unclosed '('".to_string(),
        });
    }
    if let Some((_, (_, _, pos))) = rings.into_iter().next() {
        return Err(StructError::InvalidSmiles {
            pos,
            msg: "unclosed ring closure".to_string(),
        });
    }
    if mol.atoms.is_empty() {
        return Err(StructError::EmptyInput);
    }

    embed_3d_coordinates(&mut mol);
    for (parent, cnt) in bracket_hs {
        let dirs = ideal_h_directions(&mol, parent, usize::from(cnt));
        let ppos = mol.atoms[parent].coords;
        for d in dirs {
            let mut h = Atom3D::new(
                mol.atoms.len(),
                "H",
                [
                    ppos[0] + d[0] * 1.01,
                    ppos[1] + d[1] * 1.01,
                    ppos[2] + d[2] * 1.01,
                ],
            );
            h.ad4_type = Ad4Type::HD;
            let hid = mol.add_atom(h);
            let _ = mol.add_bond(parent, hid, BondOrder::Single);
        }
    }
    perceive_all(&mut mol);
    Ok(mol)
}

fn parse_smiles_organic(
    bytes: &[u8],
    pos: usize,
    id: usize,
) -> Result<(Atom3D, usize), StructError> {
    let b = bytes[pos];
    let (sym, arom, len) = if b == b'C' && bytes.get(pos + 1) == Some(&b'l') {
        ("Cl", false, 2)
    } else if b == b'B' && bytes.get(pos + 1) == Some(&b'r') {
        ("Br", false, 2)
    } else {
        let s = match b {
            b'B' => ("B", false),
            b'C' => ("C", false),
            b'N' => ("N", false),
            b'O' => ("O", false),
            b'P' => ("P", false),
            b'S' => ("S", false),
            b'F' => ("F", false),
            b'I' => ("I", false),
            b'c' => ("C", true),
            b'n' => ("N", true),
            b'o' => ("O", true),
            b's' => ("S", true),
            b'p' => ("P", true),
            b'b' => ("B", true),
            _ => {
                return Err(StructError::InvalidSmiles {
                    pos,
                    msg: format!("unexpected SMILES token '{}'", b as char),
                });
            }
        };
        (s.0, s.1, 1)
    };
    let mut atom = Atom3D::new(id, sym, [0.0; 3]);
    atom.aromatic = arom;
    Ok((atom, pos + len))
}

fn parse_smiles_bracket(
    bytes: &[u8],
    start: usize,
    id: usize,
) -> Result<(Atom3D, u8, usize), StructError> {
    let mut i = start + 1;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i >= bytes.len() || !bytes[i].is_ascii_alphabetic() {
        return Err(StructError::InvalidSmiles {
            pos: i,
            msg: "expected element symbol inside []".to_string(),
        });
    }
    let first = bytes[i];
    let arom = first.is_ascii_lowercase();
    let sym = if let Some(&sec) = bytes.get(i + 1)
        && sec.is_ascii_lowercase()
        && sec != b'h'
    {
        i += 2;
        format!("{}{}", first.to_ascii_uppercase() as char, sec as char)
    } else {
        i += 1;
        (first.to_ascii_uppercase() as char).to_string()
    };
    let mut chg: i8 = 0;
    let mut exp_h: u8 = 0;
    while i < bytes.len() && bytes[i] != b']' {
        match bytes[i] {
            b'H' => {
                if let Some(&d) = bytes.get(i + 1)
                    && d.is_ascii_digit()
                {
                    exp_h = (d - b'0').max(1);
                    i += 1;
                } else {
                    exp_h = 1;
                }
            }
            b'+' => chg = chg.saturating_add(1),
            b'-' => chg = chg.saturating_sub(1),
            b'2'..=b'9' if chg != 0 => {
                let mag = (bytes[i] - b'0') as i8;
                chg = if chg > 0 { mag } else { -mag };
            }
            _ => {}
        }
        i += 1;
    }
    if i >= bytes.len() {
        return Err(StructError::InvalidSmiles {
            pos: start,
            msg: "unclosed '['".to_string(),
        });
    }
    let mut atom = Atom3D::new(id, &sym, [0.0; 3]);
    atom.aromatic = arom;
    atom.formal_charge = chg;
    Ok((atom, exp_h, i + 1))
}

/// Generate clean 3D Cartesian coordinates (rings + tetrahedral/trigonal BFS + UFF-lite relaxation).
pub fn embed_3d_coordinates(mol: &mut Molecule3D) {
    mol.detect_small_rings();
    let n = mol.atoms.len();
    let mut coords: Vec<Option<[f64; 3]>> = vec![None; n];
    let mut placed_rings = vec![false; mol.rings.len()];

    if !mol.rings.is_empty() {
        let mut sorted_ri: Vec<usize> = (0..mol.rings.len()).collect();
        sorted_ri.sort_by_key(|&ri| std::cmp::Reverse(mol.rings[ri].len()));
        if let Some(&first_ri) = sorted_ri.first() {
            place_3d_ring(&mol.rings[first_ri], [0.0, 0.0, 0.0], 0.0, &mut coords);
            placed_rings[first_ri] = true;
            loop {
                let mut prog = false;
                for &ri in &sorted_ri {
                    if placed_rings[ri] {
                        continue;
                    }
                    let ring = &mol.rings[ri];
                    let placed: Vec<usize> = ring
                        .iter()
                        .copied()
                        .filter(|&v| coords[v].is_some())
                        .collect();
                    if placed.is_empty() {
                        continue;
                    }
                    if let Some((k1, k2)) = (0..ring.len()).find_map(|i| {
                        let j = (i + 1) % ring.len();
                        (coords[ring[i]].is_some() && coords[ring[j]].is_some()).then_some((i, j))
                    }) {
                        let p1 = coords[ring[k1]].unwrap_or([0.0; 3]);
                        let p2 = coords[ring[k2]].unwrap_or([1.4, 0.0, 0.0]);
                        let mid = [(p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5, 0.0];
                        let dx = p2[0] - p1[0];
                        let dy = p2[1] - p1[1];
                        let elen = dx.hypot(dy).max(1e-5);
                        let n1 = [-dy / elen, dx / elen, 0.0];
                        let n2 = [dy / elen, -dx / elen, 0.0];
                        let cen = centroid_opt3(&coords);
                        let normal = if (mid[0] + n1[0] - cen[0]).hypot(mid[1] + n1[1] - cen[1])
                            >= (mid[0] + n2[0] - cen[0]).hypot(mid[1] + n2[1] - cen[1])
                        {
                            n1
                        } else {
                            n2
                        };
                        let sides = ring.len() as f64;
                        let rad = elen / (2.0 * (PI / sides).sin());
                        let apo = (rad * rad - (elen * 0.5).powi(2)).max(0.0).sqrt();
                        let center = [mid[0] + normal[0] * apo, mid[1] + normal[1] * apo, 0.0];
                        let a1 = (p1[1] - center[1]).atan2(p1[0] - center[0]);
                        let a2 = (p2[1] - center[1]).atan2(p2[0] - center[0]);
                        let mut diff = a2 - a1;
                        while diff > PI {
                            diff -= 2.0 * PI;
                        }
                        while diff < -PI {
                            diff += 2.0 * PI;
                        }
                        let step = if diff < 0.0 {
                            -2.0 * PI / sides
                        } else {
                            2.0 * PI / sides
                        };
                        for (idx, &v) in ring.iter().enumerate() {
                            if coords[v].is_none() {
                                let d = (idx as isize - k1 as isize).rem_euclid(ring.len() as isize)
                                    as f64;
                                let ang = a1 + d * step;
                                coords[v] = Some([
                                    center[0] + rad * ang.cos(),
                                    center[1] + rad * ang.sin(),
                                    if !mol.atoms[v].aromatic && ring.len() == 6 && idx % 2 == 1 {
                                        0.25
                                    } else {
                                        0.0
                                    },
                                ]);
                            }
                        }
                    } else {
                        place_3d_ring(ring, [3.8, 0.5, 0.2], 0.3, &mut coords);
                    }
                    placed_rings[ri] = true;
                    prog = true;
                }
                if !prog {
                    break;
                }
            }
        }
    }

    // BFS place remaining acyclic atoms & secondary rings with 3D tetrahedral z-puckering
    let mut q = VecDeque::new();
    if coords.iter().all(Option::is_none) && n > 0 {
        coords[0] = Some([0.0, 0.0, 0.0]);
    }
    for (i, c) in coords.iter().enumerate() {
        if c.is_some() {
            q.push_back(i);
        }
    }
    while let Some(u) = q.pop_front() {
        let pu = coords[u].unwrap_or([0.0; 3]);
        let mut sum = [0.0; 3];
        let mut p_cnt = 0usize;
        let mut unplaced = Vec::new();
        for &(v, bid) in &mol.adj[u] {
            if let Some(pv) = coords[v] {
                let d = norm3([pv[0] - pu[0], pv[1] - pu[1], pv[2] - pu[2]]);
                sum[0] += d[0];
                sum[1] += d[1];
                sum[2] += d[2];
                p_cnt += 1;
            } else {
                unplaced.push((v, bid));
            }
        }
        let dir = if p_cnt == 0 {
            [1.0, 0.0, 0.0]
        } else {
            let n_dir = norm3([-sum[0], -sum[1], -sum[2]]);
            if n_dir[0].hypot(n_dir[1]).hypot(n_dir[2]) < 1e-4 {
                [0.0, 1.0, 0.4]
            } else {
                n_dir
            }
        };
        let arb = if dir[2].abs() < 0.8 {
            [0.0, 0.0, 1.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let t1 = norm3(cross3(dir, arb));
        let t2 = cross3(dir, t1);
        let k = unplaced.len();
        for (idx, (v, bid)) in unplaced.into_iter().enumerate() {
            let bl = match mol.bonds[bid].order {
                BondOrder::Double => 1.33,
                BondOrder::Triple => 1.20,
                BondOrder::Aromatic => 1.39,
                BondOrder::Single => {
                    if mol.atoms[u].element == "S" || mol.atoms[v].element == "S" {
                        1.78
                    } else if matches!(mol.atoms[u].element.as_str(), "Cl" | "Br" | "I")
                        || matches!(mol.atoms[v].element.as_str(), "Cl" | "Br" | "I")
                    {
                        1.76
                    } else {
                        1.50
                    }
                }
            };
            let spread: f64 = if mol.atoms[u].aromatic { 0.0 } else { 0.48 };
            let phi = 2.0 * PI * (idx as f64) / (k.max(1) as f64) + (u as f64) * 0.7;
            let v_dir = if k == 1 {
                norm3([
                    dir[0] + t1[0] * 0.35,
                    dir[1] + t1[1] * 0.35,
                    dir[2] + t2[2] * 0.25,
                ])
            } else {
                norm3([
                    dir[0] * 0.75 + (t1[0] * phi.cos() + t2[0] * phi.sin()) * spread.max(0.66),
                    dir[1] * 0.75 + (t1[1] * phi.cos() + t2[1] * phi.sin()) * spread.max(0.66),
                    dir[2] * 0.75 + (t1[2] * phi.cos() + t2[2] * phi.sin()) * spread.max(0.66),
                ])
            };
            coords[v] = Some([
                pu[0] + bl * v_dir[0],
                pu[1] + bl * v_dir[1],
                pu[2] + bl * v_dir[2],
            ]);
            q.push_back(v);
        }
    }

    // 40-step 3D bond-length + non-bonded clash relaxation
    let mut pos: Vec<[f64; 3]> = coords.into_iter().map(|c| c.unwrap_or([0.0; 3])).collect();
    for _ in 0..40 {
        let mut disp = vec![[0.0; 3]; n];
        for b in &mol.bonds {
            let target = match b.order {
                BondOrder::Double => 1.34,
                BondOrder::Triple => 1.20,
                BondOrder::Aromatic => 1.39,
                BondOrder::Single => 1.50,
            };
            let d = [
                pos[b.v][0] - pos[b.u][0],
                pos[b.v][1] - pos[b.u][1],
                pos[b.v][2] - pos[b.u][2],
            ];
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-4);
            let k = (len - target) / len * 0.3;
            for c in 0..3 {
                disp[b.u][c] += d[c] * k;
                disp[b.v][c] -= d[c] * k;
            }
        }
        for i in 0..n {
            for j in (i + 1)..n {
                if mol.adj[i].iter().any(|&(nb, _)| nb == j) {
                    continue;
                }
                let d = [
                    pos[j][0] - pos[i][0],
                    pos[j][1] - pos[i][1],
                    pos[j][2] - pos[i][2],
                ];
                let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-4);
                let min_d = 2.15;
                if len < min_d {
                    let push = (min_d - len) / len * 0.2;
                    for c in 0..3 {
                        disp[i][c] -= d[c] * push;
                        disp[j][c] += d[c] * push;
                    }
                }
            }
        }
        for (p, dp) in pos.iter_mut().zip(disp.iter()) {
            for c in 0..3 {
                p[c] += dp[c].clamp(-0.25, 0.25);
            }
        }
    }

    for (a, p) in mol.atoms.iter_mut().zip(pos) {
        a.coords = p;
    }
}

fn place_3d_ring(ring: &[usize], center: [f64; 3], rot: f64, coords: &mut [Option<[f64; 3]>]) {
    let k = ring.len() as f64;
    let rad = 1.40 / (2.0 * (PI / k).sin());
    for (i, &v) in ring.iter().enumerate() {
        let ang = rot + 2.0 * PI * (i as f64) / k;
        coords[v] = Some([
            center[0] + rad * ang.cos(),
            center[1] + rad * ang.sin(),
            center[2],
        ]);
    }
}

fn centroid_opt3(coords: &[Option<[f64; 3]>]) -> [f64; 3] {
    let mut s = [0.0; 3];
    let mut n = 0.0;
    for p in coords.iter().flatten() {
        s[0] += p[0];
        s[1] += p[1];
        s[2] += p[2];
        n += 1.0;
    }
    if n > 0.0 {
        [s[0] / n, s[1] / n, s[2] / n]
    } else {
        [0.0; 3]
    }
}

pub fn norm3(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l < 1e-8 {
        [1.0, 0.0, 0.0]
    } else {
        [v[0] / l, v[1] / l, v[2] / l]
    }
}

pub fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
