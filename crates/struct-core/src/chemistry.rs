use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::f64::consts::PI;

use crate::error::StructError;
use crate::types::{
    Ad4Type, Atom3D, BondOrder, Molecule3D, RotatableBond, element_covalent_radius,
    normalize_element,
};

/// Maximum atoms permitted in a single SMILES input parsed to 3D.
pub const MAX_SMILES_3D_ATOMS: usize = 512;

/// A crystallographic water molecule inside a binding pocket that forms >= 2 hydrogen bonds
/// with receptor polar atoms (e.g. HIV-1 protease flap water HOH301 in 1HSG).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgingWater {
    pub res_seq: i32,
    pub chain_id: char,
    pub atom_index: usize,
    pub coords: [f64; 3],
    pub receptor_hbonds: usize,
    pub lining_residues: Vec<String>,
}

fn is_water_res(res_name: &str) -> bool {
    matches!(
        res_name.trim().to_ascii_uppercase().as_str(),
        "HOH" | "WAT" | "DOD" | "TIP" | "TIP3"
    )
}

/// Detect buried structural/bridging waters within `radius` Å of `center` that make at least
/// `min_hbonds` polar contacts (2.3–3.5 Å) with receptor N/O atoms.
#[must_use]
pub fn detect_bridging_waters(
    raw_receptor: &Molecule3D,
    center: [f64; 3],
    radius: f64,
    min_hbonds: usize,
) -> Vec<BridgingWater> {
    let r2_box = radius * radius;
    let mut out = Vec::new();
    for (idx, a) in raw_receptor.atoms.iter().enumerate() {
        if a.element != "O" || !is_water_res(&a.res_name) {
            continue;
        }
        let d2 = (a.coords[0] - center[0]).powi(2)
            + (a.coords[1] - center[1]).powi(2)
            + (a.coords[2] - center[2]).powi(2);
        if d2 > r2_box {
            continue;
        }
        let mut hb = 0usize;
        let mut res_set: BTreeSet<String> = BTreeSet::new();
        for b in &raw_receptor.atoms {
            if is_water_res(&b.res_name) || !matches!(b.element.as_str(), "N" | "O") {
                continue;
            }
            let db2 = (a.coords[0] - b.coords[0]).powi(2)
                + (a.coords[1] - b.coords[1]).powi(2)
                + (a.coords[2] - b.coords[2]).powi(2);
            if (2.25 * 2.25..=3.55 * 3.55).contains(&db2) {
                hb += 1;
                res_set.insert(format!("{}:{}{}", b.chain_id, b.res_name, b.res_seq));
            }
        }
        if hb >= min_hbonds {
            out.push(BridgingWater {
                res_seq: a.res_seq,
                chain_id: a.chain_id,
                atom_index: idx,
                coords: a.coords,
                receptor_hbonds: hb,
                lining_residues: res_set.into_iter().collect(),
            });
        }
    }
    out.sort_by_key(|w| std::cmp::Reverse(w.receptor_hbonds));
    out
}

/// Clean a protein receptor: strip non-structural waters (`HOH`/`WAT`/`DOD`) when requested,
/// infer missing covalent bonds, perceive rings/aromaticity, protonate at pH 7.4, add missing
/// polar hydrogens, compute Gasteiger partial charges, and assign AutoDock4 atom types.
pub fn prepare_receptor(mol: &mut Molecule3D, strip_waters: bool) {
    if strip_waters {
        keep_non_water_atoms(mol, &[]);
    }
    mol.infer_bonds_by_distance();
    perceive_all(mol);
}

/// Prepare a receptor while selectively retaining specific crystallographic water residue numbers
/// (e.g. `&[301]` for HIV-1 protease bridging water HOH301).
pub fn prepare_receptor_keeping_waters(mol: &mut Molecule3D, keep_water_resseqs: &[i32]) {
    keep_non_water_atoms(mol, keep_water_resseqs);
    mol.infer_bonds_by_distance();
    perceive_all(mol);
}

fn keep_non_water_atoms(mol: &mut Molecule3D, keep_water_resseqs: &[i32]) {
    let mut keep_map = vec![None; mol.atoms.len()];
    let mut kept = Vec::with_capacity(mol.atoms.len());
    for (idx, mut a) in mol.atoms.drain(..).enumerate() {
        if is_water_res(&a.res_name) && !keep_water_resseqs.contains(&a.res_seq) {
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

    // 2. Ring aromaticity for ligands and non-standard PDB hetatms.
    //
    // The test is topological wherever bond orders are available: every atom of the ring must
    // contribute a pi electron through a double or aromatic bond **inside that same ring**, or be
    // a lone-pair-donating heteroatom of a five-membered ring (pyrrole, furan, thiophene,
    // imidazole). Asking only whether a ring atom has *any* unsaturated bond makes the saturated
    // ring of a fused system inherit its partner's aromaticity through the two fusion atoms,
    // which marked all 10 heavy atoms of tetralin, indane, indoline and
    // tetrahydroisoquinoline aromatic instead of 6, and then suppressed the pH 7.4 protonation
    // of the ring nitrogen because aromatic nitrogens are skipped.
    //
    // The geometric fallback (flat ring, short bonds, degree <= 3) is kept only for structures
    // that carry no bond-order information at all, such as a PDB hetatm whose bonds were inferred
    // from distances: there, geometry is the only evidence there is.
    let bond_orders_known = mol.bonds.iter().any(|b| b.order != BondOrder::Single);
    for ring in &mol.rings {
        if !(5..=6).contains(&ring.len()) {
            continue;
        }
        let all_marked = ring.iter().all(|&v| mol.atoms[v].aromatic);
        let ring_bond_order = |u: usize, v: usize| -> Option<BondOrder> {
            mol.adj[u]
                .iter()
                .find(|&&(nb, _)| nb == v)
                .map(|&(_, bid)| mol.bonds[bid].order)
        };
        let n = ring.len();
        // Every ring bond, not merely some bond: in a fused system the shared bond is aromatic
        // because it belongs to the aromatic partner ring, and accepting that one bond was what
        // made tetralin's saturated ring aromatic.
        let all_ring_bonds_aromatic = (0..n)
            .all(|w| ring_bond_order(ring[w], ring[(w + 1) % n]) == Some(BondOrder::Aromatic));
        let every_atom_contributes_pi = (0..n).all(|w| {
            let v = ring[w];
            let prev = ring[(w + n - 1) % n];
            let next = ring[(w + 1) % n];
            let multi = |o: Option<BondOrder>| {
                matches!(o, Some(BondOrder::Double) | Some(BondOrder::Aromatic))
            };
            if multi(ring_bond_order(v, prev)) || multi(ring_bond_order(v, next)) {
                return true;
            }
            // Lone-pair donor completing a 5-membered aromatic sextet.
            n == 5 && matches!(mol.atoms[v].element.as_str(), "N" | "O" | "S")
        });
        let valid_elems = ring
            .iter()
            .all(|&v| matches!(mol.atoms[v].element.as_str(), "C" | "N" | "O" | "S"));
        let geometric_fallback =
            !bond_orders_known && ring_is_planar(mol, ring) && deg_at_most_3(mol, ring);
        if (all_marked && bond_orders_known && all_ring_bonds_aromatic)
            || (all_marked && !bond_orders_known)
            || (valid_elems
                && (all_ring_bonds_aromatic || every_atom_contributes_pi || geometric_fallback))
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
    let (amidine_group_ns, amidine_sites) = amidine_guanidine_groups(mol);
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

        // Preserve +2 formal charge on catalytic divalent metal ions
        if matches!(el.as_str(), "Zn" | "Mg" | "Fe" | "Ca" | "Mn" | "Cu")
            && mol.atoms[i].formal_charge == 0
        {
            mol.atoms[i].formal_charge = 2;
            continue;
        }
        // Structural water oxygen: neutral, both donor and acceptor (2 hydrogens if unbonded)
        if el == "O" && is_water_res(&rn) {
            mol.atoms[i].formal_charge = 0;
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
            if amidine_group_ns.contains(&i) {
                // Delocalised amidinium/guanidinium: exactly one charge for the whole group.
                mol.atoms[i].formal_charge = i8::from(amidine_sites.contains(&i));
                continue;
            }
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
            // Aliphatic amine without carbonyl/aryl conjugation -> protonated at pH 7.4 (+1).
            // Amidines and guanidines are handled above, once per group.
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
                    // Aromatic nitrogen: protein Trp/His or Kekulé pyrrole/indole/indazole/pyrimidone NH
                    let rn = a.res_name.trim().to_ascii_uppercase();
                    let an = a.name.trim().to_ascii_uppercase();
                    if (rn == "TRP" && an == "NE1")
                        || ((rn == "HIS" || rn == "HID") && an == "ND1")
                        || (rn == "HIE" && an == "NE2")
                    {
                        1
                    } else if existing_h > 0 {
                        existing_h
                    } else if a.is_hetatm && a.formal_charge == 0 {
                        // In Kekulé SDF/MOL2, pyrrole/indole/indazole/pyrimidone [nH] has 2 Single heavy bonds
                        // and 0 Double bonds (bond_order_sum == 2), whereas pyridine =N- has a Double bond (sum == 3).
                        let heavy_bonds: Vec<BondOrder> = mol.adj[i]
                            .iter()
                            .filter(|&&(nb, _)| mol.atoms[nb].is_heavy())
                            .map(|&(_, bid)| mol.bonds[bid].order)
                            .collect();
                        if heavy_bonds.len() == 2
                            && heavy_bonds.iter().all(|&o| o == BondOrder::Single)
                        {
                            1
                        } else {
                            0
                        }
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
                if is_water_res(&a.res_name) {
                    2
                } else if a.formal_charge < 0 {
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
        .map(|a| f64::from(a.formal_charge))
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
            let denom = if mol.atoms[donor].element == "H" {
                20.02
            } else {
                (params[donor].0 + params[donor].1 + params[donor].2).max(4.0)
            };
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
                if mol.atoms[i].formal_charge > 0
                    || bonded_h >= 2
                    || is_amide_nitrogen(mol, i)
                    || (arom && bonded_h > 0)
                {
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

        let is_water = el == "O" && is_water_res(&mol.atoms[i].res_name);
        let has_hetero_neighbor = mol.adj[i].iter().any(|&(nb, _)| {
            matches!(
                mol.atoms[nb].element.as_str(),
                "O" | "N" | "S" | "F" | "Cl" | "Br" | "I" | "P"
            )
        });
        let a = &mut mol.atoms[i];
        a.ad4_type = ad4;
        // Exact AutoDock Vina X-Score H-bond / hydrophobic typing:
        // - Metals act as H-bond donors (XS_TYPE_Met_D); sulfur is neither donor nor acceptor;
        // - Structural waters are both donor and acceptor;
        // - Carbons are hydrophobic (XS_TYPE_C_H) iff bonded only to C/H; halogens are hydrophobic.
        a.h_donor =
            is_water || ad4.is_metal() || (bonded_h > 0 && matches!(el.as_str(), "N" | "O"));
        a.h_acceptor = is_water || (ad4.is_acceptor() && a.formal_charge <= 0);
        a.hydrophobic = ad4.is_hydrophobic() && !(el == "C" && has_hetero_neighbor);
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

/// Classify a heavy atom's metal-coordination strength (AutoDock4Zn / Glide / GOLD coordination rules):
/// - `2` (Strong): primary/secondary sulfonamide N, hydroxamate O/N (`-C(=O)N-O`), anionic O-/S-
///   (carboxylate/phosphonate/thiolate), free thiol `-SH`, or 5-ring aromatic azole lone-pair N.
/// - `1` (Moderate): neutral `OA` carbonyl/hydroxyl, 6-ring aromatic `NA` (pyridine), neutral amine `NA`.
/// - `0` (None): carbons, halogens, cationic `N+`, bulk amide N, thioether/sulfone S.
#[must_use]
pub fn metal_chelation_tier(mol: &Molecule3D, idx: usize) -> u8 {
    let Some(a) = mol.atoms.get(idx) else {
        return 0;
    };
    if !a.is_heavy() || a.ad4_type.is_metal() || a.formal_charge > 0 {
        return 0;
    }
    let heavy_deg = mol.adj[idx]
        .iter()
        .filter(|&&(nb, _)| mol.atoms[nb].is_heavy())
        .count();

    match a.element.as_str() {
        "N" => {
            // Primary/secondary sulfonamide/sulfamate N: bonded to S(=O)(=O) and <= 2 heavy neighbors
            let is_sulfonamide_n = heavy_deg <= 2
                && mol.adj[idx].iter().any(|&(s, _)| {
                    mol.atoms[s].element == "S"
                        && mol.adj[s]
                            .iter()
                            .filter(|&&(o, bid)| {
                                mol.atoms[o].element == "O"
                                    && mol.bonds[bid].order == BondOrder::Double
                            })
                            .count()
                            >= 2
                });
            if is_sulfonamide_n {
                return 2;
            }
            // Hydroxamate N: bonded to both C(=O) and O
            let has_o = mol.adj[idx]
                .iter()
                .any(|&(nb, _)| mol.atoms[nb].element == "O");
            if has_o && is_amide_nitrogen(mol, idx) {
                return 2;
            }
            // Neutral sp2/amine N acceptors (pyridine, imidazole, thiadiazole) get gentle tier 1
            if a.h_acceptor && !is_amide_nitrogen(mol, idx) {
                1
            } else {
                0
            }
        }
        "O" => {
            if a.formal_charge < 0 {
                return 2; // Anionic carboxylate, phosphonate, phenolate
            }
            // Hydroxamate hydroxyl O (bonded to N that is bonded to C=O)
            let is_hydroxamate_oh = mol.adj[idx]
                .iter()
                .any(|&(n, _)| mol.atoms[n].element == "N" && is_amide_nitrogen(mol, n));
            if is_hydroxamate_oh {
                return 2;
            }
            // Hydroxamate carbonyl O (C=O where C is bonded to N-O)
            let is_hydroxamate_co = mol.adj[idx].iter().any(|&(c, bid)| {
                mol.atoms[c].element == "C"
                    && mol.bonds[bid].order == BondOrder::Double
                    && mol.adj[c].iter().any(|&(n, _)| {
                        mol.atoms[n].element == "N"
                            && mol.adj[n]
                                .iter()
                                .any(|&(o2, _)| mol.atoms[o2].element == "O")
                    })
            });
            if is_hydroxamate_co {
                return 2;
            }
            // Carboxylate carbonyl O (C=O where C also has an anionic O-)
            let is_carboxylate_co = mol.adj[idx].iter().any(|&(c, bid)| {
                mol.atoms[c].element == "C"
                    && mol.bonds[bid].order == BondOrder::Double
                    && mol.adj[c].iter().any(|&(o2, _)| {
                        o2 != idx && mol.atoms[o2].element == "O" && mol.atoms[o2].formal_charge < 0
                    })
            });
            if is_carboxylate_co {
                return 2;
            }
            1
        }
        "S" => {
            // Free thiol (-SH) or thiolate (-S-)
            if heavy_deg <= 1 { 2 } else { 0 }
        }
        _ => 0,
    }
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

    // Pass 1: collect the bonds that qualify as rotatable, independently of where the root ends
    // up. The root can only be chosen once these are known, because it is defined relative to
    // them.
    let mut candidates: Vec<(usize, usize, usize)> = Vec::new(); // (bond_id, u, v)
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
        if is_symmetric_cap(mol, b.u, b.v) || is_symmetric_cap(mol, b.v, b.u) {
            continue;
        }
        if bfs_component_without_bond(mol, b.u, b.id).contains(&b.v) {
            continue; // Large non-SSSR ring cycle: cutting it does not split the molecule
        }
        candidates.push((b.id, b.u, b.v));
    }

    // Choose the rigid root anchor the way AutoDock/Vina does: the *largest rigid fragment*, i.e.
    // the biggest connected component left once every rotatable bond is cut. Picking the graph
    // centre instead (the previous rule) put the root in the middle of the flexible chain of, for
    // example, 4-phenylbutan-1-ol, which left the benzene ring on the moving side of a torsion:
    // the bulkiest rigid part of the ligand then swung around the docking position instead of
    // defining it. Ties are broken by ring membership, then by heavy-atom degree.
    let cut: BTreeSet<usize> = candidates.iter().map(|&(bid, _, _)| bid).collect();
    let mut fragment_of = vec![usize::MAX; mol.atoms.len()];
    let mut fragment_size: Vec<usize> = Vec::new();
    for start in 0..mol.atoms.len() {
        if !mol.atoms[start].is_heavy() || fragment_of[start] != usize::MAX {
            continue;
        }
        let fid = fragment_size.len();
        let mut size = 0usize;
        let mut q = VecDeque::new();
        fragment_of[start] = fid;
        q.push_back(start);
        while let Some(cur) = q.pop_front() {
            size += 1;
            for &(nb, bid) in &mol.adj[cur] {
                if mol.atoms[nb].is_heavy() && !cut.contains(&bid) && fragment_of[nb] == usize::MAX
                {
                    fragment_of[nb] = fid;
                    q.push_back(nb);
                }
            }
        }
        fragment_size.push(size);
    }

    let heavy_ids: Vec<usize> = (0..mol.atoms.len())
        .filter(|&i| mol.atoms[i].is_heavy())
        .collect();
    let root = heavy_ids
        .iter()
        .copied()
        .max_by_key(|&i| {
            let frag = fragment_of[i];
            let size = fragment_size.get(frag).copied().unwrap_or(0);
            let r_cnt = mol.rings.iter().filter(|r| r.contains(&i)).count();
            (size, r_cnt, mol.adj[i].len(), usize::MAX - i)
        })
        .unwrap_or(0);

    // BFS topological distance from root so nested rotatable bonds apply root-to-leaf.
    let mut depth_from_root = vec![usize::MAX; mol.atoms.len()];
    if root < mol.atoms.len() {
        let mut q = VecDeque::new();
        depth_from_root[root] = 0;
        q.push_back(root);
        while let Some(cur) = q.pop_front() {
            for &(nb, _) in &mol.adj[cur] {
                if depth_from_root[nb] == usize::MAX {
                    depth_from_root[nb] = depth_from_root[cur] + 1;
                    q.push_back(nb);
                }
            }
        }
    }

    // Pass 2: for each rotatable bond, keep the root side stationary and move the other half.
    for &(bid, bu, bv) in &candidates {
        let side_u = bfs_component_without_bond(mol, bu, bid);
        let side_v = bfs_component_without_bond(mol, bv, bid);
        let (anchor, tip, moving) = if side_u.contains(&root) {
            (bu, bv, side_v)
        } else {
            (bv, bu, side_u)
        };
        if !moving.is_empty() {
            mol.rotatable_bonds.push(RotatableBond {
                bond_id: bid,
                u: anchor,
                v: tip,
                moving_atoms: moving,
            });
        }
    }
    mol.rotatable_bonds
        .sort_by_key(|rb| (depth_from_root[rb.u], rb.moving_atoms.len()));
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
    // Ring-bond bookkeeping carries the slot reserved in the opening atom's neighbour list, so a
    // ring bond counts at the position where its digit was written rather than where it closes.
    let mut rings: HashMap<u16, (usize, Option<BondOrder>, usize, usize)> = HashMap::new();
    let mut bracket_hs: Vec<(usize, u8)> = Vec::new();
    // Tetrahedral chirality from the SMILES, and the neighbour order it refers to.
    const PENDING_RING: usize = usize::MAX;
    const PENDING_H: usize = usize::MAX - 1;
    let mut chiral: Vec<(usize, i8)> = Vec::new();
    let mut nbr_order: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut pending_dir: i8 = 0; // +1 for '/', -1 for '\'
    let mut ez_dirs: Vec<(usize, usize, i8)> = Vec::new();
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
            b'/' => {
                pending_dir = 1;
                i += 1;
            }
            b'\\' => {
                pending_dir = -1;
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
                if let Some((other, bo1, _, slot)) = rings.remove(&rnum) {
                    let bo = bo1.or(pending_bond).unwrap_or_else(|| {
                        if mol.atoms[other].aromatic && mol.atoms[cur].aromatic {
                            BondOrder::Aromatic
                        } else {
                            BondOrder::Single
                        }
                    });
                    let _ = mol.add_bond(other, cur, bo);
                    if let Some(list) = nbr_order.get_mut(&other)
                        && let Some(entry) = list.get_mut(slot)
                    {
                        *entry = cur;
                    }
                    nbr_order.entry(cur).or_default().push(other);
                } else {
                    let list = nbr_order.entry(cur).or_default();
                    let slot = list.len();
                    list.push(PENDING_RING);
                    rings.insert(rnum, (cur, pending_bond, i, slot));
                }
                pending_bond = None;
                i = next_i;
            }
            b'[' => {
                let (atom, exp_h, chirality, next_i) =
                    parse_smiles_bracket(bytes, i, mol.atoms.len())?;
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
                    if pending_dir != 0 {
                        ez_dirs.push((prev, nid, pending_dir));
                    }
                    nbr_order.entry(prev).or_default().push(nid);
                    nbr_order.entry(nid).or_default().push(prev);
                }
                // An implicit hydrogen written inside the brackets occupies the neighbour slot
                // immediately after the preceding atom (or the first slot when there is none).
                if exp_h == 1 {
                    nbr_order.entry(nid).or_default().push(PENDING_H);
                }
                if chirality != 0 {
                    chiral.push((nid, chirality));
                }
                current = Some(nid);
                pending_bond = None;
                pending_dir = 0;
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
                    if pending_dir != 0 {
                        ez_dirs.push((prev, nid, pending_dir));
                    }
                    nbr_order.entry(prev).or_default().push(nid);
                    nbr_order.entry(nid).or_default().push(prev);
                }
                current = Some(nid);
                pending_bond = None;
                pending_dir = 0;
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
    if let Some((_, (_, _, pos, _))) = rings.into_iter().next() {
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
        let first_h_id = mol.atoms.len();
        if let Some(list) = nbr_order.get_mut(&parent) {
            for slot in list.iter_mut() {
                if *slot == PENDING_H {
                    *slot = first_h_id;
                }
            }
        }
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
    enforce_tetrahedral_chirality(&mut mol, &chiral, &nbr_order);
    enforce_ez_stereo(&mut mol, &ez_dirs);
    perceive_all(&mut mol);
    Ok(mol)
}

/// Enforce SMILES `E/Z` (`/` and `\`) stereochemistry across non-ring double bonds by rigidly
/// rotating one substituent branch by 180° around the double-bond axis when its cis/trans
/// parity disagrees with the directional SMILES markers.
fn enforce_ez_stereo(mol: &mut Molecule3D, ez_dirs: &[(usize, usize, i8)]) {
    if ez_dirs.is_empty() {
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
    // Map directed pair (from, to) -> sign (+1 for '/', -1 for '\'); reverse bond has opposite sign
    let mut dir_map: HashMap<(usize, usize), i8> = HashMap::new();
    for &(u, v, s) in ez_dirs {
        dir_map.insert((u, v), s);
        dir_map.insert((v, u), -s);
    }

    for b in mol.bonds.clone() {
        if b.order != BondOrder::Double || ring_edges.contains(&b.id) {
            continue;
        }
        // Find one directional substituent on u (other than v) and one on v (other than u)
        let sub_u = mol.adj[b.u]
            .iter()
            .filter(|&&(nb, _)| nb != b.v)
            .find_map(|&(nb, _)| dir_map.get(&(b.u, nb)).map(|&s| (nb, s)));
        let sub_v = mol.adj[b.v]
            .iter()
            .filter(|&&(nb, _)| nb != b.u)
            .find_map(|&(nb, _)| dir_map.get(&(b.v, nb)).map(|&s| (nb, s)));
        let (Some((a, s_ua)), Some((d, s_vd))) = (sub_u, sub_v) else {
            continue;
        };
        // From a->u=v->d: dir(a->u) = -s_ua, dir(v->d) = s_vd.
        // Same SMILES slash (e.g. F/C=C/F => dir(a->u) == dir(v->d)) means TRANS (anti);
        // opposite SMILES slash (F/C=C\F => dir(a->u) == -dir(v->d)) means CIS (syn).
        let want_trans = (-s_ua) == s_vd;

        let pu = mol.atoms[b.u].coords;
        let pv = mol.atoms[b.v].coords;
        let pa = mol.atoms[a].coords;
        let pd = mol.atoms[d].coords;
        let axis = norm3([pv[0] - pu[0], pv[1] - pu[1], pv[2] - pu[2]]);
        let va = [pa[0] - pu[0], pa[1] - pu[1], pa[2] - pu[2]];
        let vd = [pd[0] - pv[0], pd[1] - pv[1], pd[2] - pv[2]];
        let da_ax = va[0] * axis[0] + va[1] * axis[1] + va[2] * axis[2];
        let dd_ax = vd[0] * axis[0] + vd[1] * axis[1] + vd[2] * axis[2];
        let pa_perp = [
            va[0] - da_ax * axis[0],
            va[1] - da_ax * axis[1],
            va[2] - da_ax * axis[2],
        ];
        let pd_perp = [
            vd[0] - dd_ax * axis[0],
            vd[1] - dd_ax * axis[1],
            vd[2] - dd_ax * axis[2],
        ];
        let pa_len =
            (pa_perp[0] * pa_perp[0] + pa_perp[1] * pa_perp[1] + pa_perp[2] * pa_perp[2]).sqrt();
        let pd_len =
            (pd_perp[0] * pd_perp[0] + pd_perp[1] * pd_perp[1] + pd_perp[2] * pd_perp[2]).sqrt();
        if pa_len < 1e-6 || pd_len < 1e-6 {
            continue;
        }
        let u1 = [
            pa_perp[0] / pa_len,
            pa_perp[1] / pa_len,
            pa_perp[2] / pa_len,
        ];
        let u2 = cross3(axis, u1);
        let cx = pd_perp[0] * u1[0] + pd_perp[1] * u1[1] + pd_perp[2] * u1[2];
        let cy = pd_perp[0] * u2[0] + pd_perp[1] * u2[1] + pd_perp[2] * u2[2];
        let cur_phi = cy.atan2(cx);
        let target_phi = if want_trans { PI } else { 0.0 };
        let delta = target_phi - cur_phi;
        if delta.sin().abs() < 1e-4 && delta.cos() > 0.999 {
            continue;
        }
        // Rotate the v-side branch around the u->v axis to snap exactly to planar E (180°) or Z (0°)
        let branch = branch_from(mol, b.v, b.u);
        if branch.contains(&b.u) {
            continue;
        }
        let cos_d = delta.cos();
        let sin_d = delta.sin();
        for atom_id in branch {
            if atom_id == b.v {
                continue;
            }
            let q = mol.atoms[atom_id].coords;
            let r = [q[0] - pv[0], q[1] - pv[1], q[2] - pv[2]];
            let k_dot_r = axis[0] * r[0] + axis[1] * r[1] + axis[2] * r[2];
            let k_cross_r = cross3(axis, r);
            mol.atoms[atom_id].coords = [
                pv[0] + r[0] * cos_d + k_cross_r[0] * sin_d + axis[0] * k_dot_r * (1.0 - cos_d),
                pv[1] + r[1] * cos_d + k_cross_r[1] * sin_d + axis[1] * k_dot_r * (1.0 - cos_d),
                pv[2] + r[2] * cos_d + k_cross_r[2] * sin_d + axis[2] * k_dot_r * (1.0 - cos_d),
            ];
        }
    }
}

/// Signed volume of the tetrahedron `(a, b, c, d)`.
///
/// OpenSMILES defines `@` as: viewed from the first neighbour towards the centre, the remaining
/// three appear anticlockwise. Working that out for an ideal tetrahedron gives a negative signed
/// volume for `@` and a positive one for `@@`.
fn signed_volume(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let w = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
    let cross = cross3(v, w);
    u[0] * cross[0] + u[1] * cross[1] + u[2] * cross[2]
}

/// Every atom reachable from `start` without passing through `blocked`.
fn branch_from(mol: &Molecule3D, start: usize, blocked: usize) -> Vec<usize> {
    let mut seen = vec![false; mol.atoms.len()];
    seen[blocked] = true;
    seen[start] = true;
    let mut out = vec![start];
    let mut q = VecDeque::from(vec![start]);
    while let Some(cur) = q.pop_front() {
        for &(nb, _) in &mol.adj[cur] {
            if !seen[nb] {
                seen[nb] = true;
                out.push(nb);
                q.push_back(nb);
            }
        }
    }
    out
}

/// Imposes the tetrahedral chirality written in the SMILES on the embedded 3D structure.
///
/// Without this the builder produced *identical* coordinates for `C[C@@H](N)C(=O)O` and
/// `C[C@H](N)C(=O)O`: the `@`/`@@` token was parsed and discarded, so one of the two enantiomers
/// was silently docked in place of the other, and nothing downstream could tell.
///
/// When the embedded centre has the wrong handedness, two of its four branches are exchanged by a
/// 180 degree rotation about the bisector of their bond directions. That operation is rigid about
/// the stereocentre, so every bond length and every internal geometry inside the two branches is
/// preserved exactly; only the two attachment directions swap. The pair chosen is the one with
/// disjoint branches (so a ring is never torn) and the fewest atoms to move.
fn enforce_tetrahedral_chirality(
    mol: &mut Molecule3D,
    chiral: &[(usize, i8)],
    nbr_order: &HashMap<usize, Vec<usize>>,
) {
    for &(centre, tag) in chiral {
        let Some(nbrs) = nbr_order.get(&centre) else {
            continue;
        };
        if nbrs.len() != 4 || nbrs.iter().any(|&x| x >= mol.atoms.len()) {
            continue;
        }
        let pts: Vec<[f64; 3]> = nbrs.iter().map(|&n| mol.atoms[n].coords).collect();
        let vol = signed_volume(pts[0], pts[1], pts[2], pts[3]);
        if vol.abs() < 1e-9 {
            continue;
        }
        let wanted_positive = tag > 0;
        if (vol > 0.0) == wanted_positive {
            continue;
        }

        let branches: Vec<Vec<usize>> = nbrs.iter().map(|&n| branch_from(mol, n, centre)).collect();
        let mut best: Option<(usize, usize, usize)> = None; // (cost, i, j)
        for i in 0..4 {
            for j in (i + 1)..4 {
                if branches[i].iter().any(|a| branches[j].contains(a)) {
                    continue; // same ring: exchanging these would tear it
                }
                let cost = branches[i].len() + branches[j].len();
                if best.is_none_or(|(c, _, _)| cost < c) {
                    best = Some((cost, i, j));
                }
            }
        }
        let Some((_, i, j)) = best else {
            continue;
        };

        let c = mol.atoms[centre].coords;
        let u = norm3([pts[i][0] - c[0], pts[i][1] - c[1], pts[i][2] - c[2]]);
        let v = norm3([pts[j][0] - c[0], pts[j][1] - c[1], pts[j][2] - c[2]]);
        let sum = [u[0] + v[0], u[1] + v[1], u[2] + v[2]];
        if sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2] < 1e-12 {
            continue; // collinear and opposite: not a tetrahedral centre
        }
        let axis = norm3(sum);
        // Rotation by pi about a unit axis: x -> 2 (axis . x) axis - x.
        for &atom_id in branches[i].iter().chain(branches[j].iter()) {
            let q = mol.atoms[atom_id].coords;
            let r = [q[0] - c[0], q[1] - c[1], q[2] - c[2]];
            let dot = axis[0] * r[0] + axis[1] * r[1] + axis[2] * r[2];
            mol.atoms[atom_id].coords = [
                c[0] + 2.0 * dot * axis[0] - r[0],
                c[1] + 2.0 * dot * axis[1] - r[1],
                c[2] + 2.0 * dot * axis[2] - r[2],
            ];
        }
    }
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

/// Parses a bracket atom. The third element of the returned tuple is the tetrahedral chirality
/// tag: `-1` for `@` (the remaining neighbours appear anticlockwise when viewed from the first
/// one), `+1` for `@@`, and `0` when none was written.
fn parse_smiles_bracket(
    bytes: &[u8],
    start: usize,
    id: usize,
) -> Result<(Atom3D, u8, i8, usize), StructError> {
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
    let mut chirality: i8 = 0;
    while i < bytes.len() && bytes[i] != b']' {
        match bytes[i] {
            b'@' => {
                if bytes.get(i + 1) == Some(&b'@') {
                    chirality = 1;
                    i += 1;
                } else {
                    chirality = -1;
                }
            }
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
    Ok((atom, exp_h, chirality, i + 1))
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
            let first_ring = &mol.rings[first_ri];
            place_3d_ring(
                first_ring,
                [0.0, 0.0, 0.0],
                0.0,
                ring_bond_length(mol, first_ring),
                ring_pucker(mol, first_ring),
                &mut coords,
            );
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
                                    if idx.is_multiple_of(2) {
                                        ring_pucker(mol, ring)
                                    } else {
                                        -ring_pucker(mol, ring)
                                    },
                                ]);
                            }
                        }
                    } else {
                        place_3d_ring(
                            ring,
                            [3.8, 0.5, 0.2],
                            0.3,
                            ring_bond_length(mol, ring),
                            ring_pucker(mol, ring),
                            &mut coords,
                        );
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
        let mut known: Vec<[f64; 3]> = Vec::with_capacity(4);
        let mut unplaced = Vec::new();
        for &(v, bid) in &mol.adj[u] {
            if let Some(pv) = coords[v] {
                known.push(norm3([pv[0] - pu[0], pv[1] - pu[1], pv[2] - pu[2]]));
            } else {
                unplaced.push((v, bid));
            }
        }
        let theta = ideal_bond_angle(mol, u);
        let phase = (u as f64) * 0.9;
        for (v, bid) in unplaced {
            if coords[v].is_some() {
                continue;
            }
            let bl = ideal_bond_length(
                &mol.atoms[u].element,
                &mol.atoms[v].element,
                mol.bonds[bid].order,
            );
            let v_dir = next_bond_direction(&known, theta, phase);
            known.push(v_dir);
            coords[v] = Some([
                pu[0] + bl * v_dir[0],
                pu[1] + bl * v_dir[1],
                pu[2] + bl * v_dir[2],
            ]);
            q.push_back(v);

            // If `v` enters an unplaced secondary ring system (e.g. multi-ring drugs like
            // tamoxifen, raloxifene, imatinib, indinavir), place that ring and any rings fused
            // to it as proper 3D polygons right now rather than growing them as a torn tree.
            if let Some(ri) = (0..mol.rings.len())
                .filter(|&r_idx| !placed_rings[r_idx] && mol.rings[r_idx].contains(&v))
                .max_by_key(|&r_idx| mol.rings[r_idx].len())
            {
                place_attached_ring_3d(mol, &mol.rings[ri], v, v_dir, &mut coords);
                placed_rings[ri] = true;
                for &rv in &mol.rings[ri] {
                    if rv != v {
                        q.push_back(rv);
                    }
                }
                // Propagate any fused rings sharing an edge with this newly placed ring
                loop {
                    let mut fused_prog = false;
                    for rj in 0..mol.rings.len() {
                        if placed_rings[rj] {
                            continue;
                        }
                        let ring = &mol.rings[rj];
                        if let Some((k1, k2)) = (0..ring.len()).find_map(|idx| {
                            let jdx = (idx + 1) % ring.len();
                            (coords[ring[idx]].is_some() && coords[ring[jdx]].is_some())
                                .then_some((idx, jdx))
                        }) {
                            place_fused_ring_3d(mol, ring, k1, k2, &mut coords);
                            placed_rings[rj] = true;
                            for &rv in ring {
                                q.push_back(rv);
                            }
                            fused_prog = true;
                        }
                    }
                    if !fused_prog {
                        break;
                    }
                }
            }
        }
    }

    // 1-3 restraints: an angle at atom j expressed as the distance its two neighbours should keep,
    // by the law of cosines. Without an angle term the relaxation has nothing to stop a strained
    // fused system from collapsing -- imipramine's seven-membered ring closed to a 2.09 A contact
    // between two heavy atoms -- and nothing to hold an sp2 centre at 120 degrees.
    let mut angle_pairs: Vec<(usize, usize, f64)> = Vec::new();
    let mut one_three: BTreeSet<(usize, usize)> = BTreeSet::new();
    for j in 0..n {
        let theta = ideal_bond_angle(mol, j);
        let nbrs: Vec<(usize, usize)> = mol.adj[j].iter().map(|&(v, bid)| (v, bid)).collect();
        for a in 0..nbrs.len() {
            for b in (a + 1)..nbrs.len() {
                let (i, bid_i) = nbrs[a];
                let (k, bid_k) = nbrs[b];
                // Inside a small ring the internal angle is fixed by the ring, not by
                // hybridisation, so leave those to the ring geometry and the bond terms.
                if mol
                    .rings
                    .iter()
                    .any(|r| r.contains(&i) && r.contains(&j) && r.contains(&k) && r.len() <= 6)
                {
                    continue;
                }
                let li = ideal_bond_length(
                    &mol.atoms[i].element,
                    &mol.atoms[j].element,
                    mol.bonds[bid_i].order,
                );
                let lk = ideal_bond_length(
                    &mol.atoms[k].element,
                    &mol.atoms[j].element,
                    mol.bonds[bid_k].order,
                );
                let target = (li * li + lk * lk - 2.0 * li * lk * theta.cos())
                    .max(0.25)
                    .sqrt();
                angle_pairs.push((i, k, target));
                one_three.insert((i.min(k), i.max(k)));
            }
        }
    }

    // 3D bond-length + angle + non-bonded clash relaxation, in two blocks: the saturated rings get
    // their pucker re-imposed between them, because the first block flattens the ones fused to an
    // aromatic partner.
    let mut pos: Vec<[f64; 3]> = coords.into_iter().map(|c| c.unwrap_or([0.0; 3])).collect();
    for step in 0..72 {
        if step == 44 {
            pucker_saturated_rings(mol, &mut pos);
        }
        if step % 8 == 7 || step == 71 {
            flatten_aromatic_rings(mol, &mut pos);
        }
        let mut disp = vec![[0.0; 3]; n];
        for b in &mol.bonds {
            let target =
                ideal_bond_length(&mol.atoms[b.u].element, &mol.atoms[b.v].element, b.order);
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
        for &(i, k, target) in &angle_pairs {
            let d = [
                pos[k][0] - pos[i][0],
                pos[k][1] - pos[i][1],
                pos[k][2] - pos[i][2],
            ];
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-4);
            let f = (len - target) / len * 0.18;
            for c in 0..3 {
                disp[i][c] += d[c] * f;
                disp[k][c] -= d[c] * f;
            }
        }
        for i in 0..n {
            for j in (i + 1)..n {
                if mol.adj[i].iter().any(|&(nb, _)| nb == j) {
                    continue;
                }
                if one_three.contains(&(i, j)) {
                    continue; // governed by its angle restraint above
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

/// Places a ring around `center`.
///
/// `bond_len` is the ring's own mean bond length, so a cyclohexane comes out at 1.52 A per bond
/// rather than at the benzene value. `pucker` is the out-of-plane amplitude: zero for an aromatic
/// or unsaturated ring, which really is flat, and non-zero for a saturated one, which is not. A
/// perfectly planar cyclohexane is eclipsed in every bond and about 6 kcal/mol above the chair,
/// and it also made the ring look flat to any downstream planarity test.
fn place_3d_ring(
    ring: &[usize],
    center: [f64; 3],
    rot: f64,
    bond_len: f64,
    pucker: f64,
    coords: &mut [Option<[f64; 3]>],
) {
    let n = ring.len();
    let k = n as f64;
    // For an even ring the alternating +-pucker turns the planar chord into the hypotenuse of a
    // right triangle, so the in-plane radius shrinks to keep the bond length right.
    let planar_chord = if pucker > 0.0 && n.is_multiple_of(2) {
        (bond_len * bond_len - 4.0 * pucker * pucker)
            .max(0.04)
            .sqrt()
    } else {
        bond_len
    };
    let rad = planar_chord / (2.0 * (PI / k).sin());
    for (i, &v) in ring.iter().enumerate() {
        let ang = rot + 2.0 * PI * (i as f64) / k;
        let z = if pucker <= 0.0 {
            0.0
        } else if n.is_multiple_of(2) {
            // Chair: alternating above and below the mean plane.
            if i.is_multiple_of(2) { pucker } else { -pucker }
        } else if i == 0 {
            // Envelope: a single flap atom out of the plane of the other four.
            pucker * 1.6
        } else {
            0.0
        };
        coords[v] = Some([
            center[0] + rad * ang.cos(),
            center[1] + rad * ang.sin(),
            center[2] + z,
        ]);
    }
}

/// Place a ring in 3D anchored at an already-placed attachment atom `anchor` so that the ring
/// extends cleanly outward along `out_dir` (e.g. a secondary phenyl or piperidine ring reached via BFS).
fn place_attached_ring_3d(
    mol: &Molecule3D,
    ring: &[usize],
    anchor: usize,
    out_dir: [f64; 3],
    coords: &mut [Option<[f64; 3]>],
) {
    let Some(pos_in_ring) = ring.iter().position(|&v| v == anchor) else {
        return;
    };
    let Some(p_anchor) = coords[anchor] else {
        return;
    };
    let n = ring.len();
    let k = n as f64;
    let bond_len = ring_bond_length(mol, ring);
    let pucker = ring_pucker(mol, ring);
    let planar_chord = if pucker > 0.0 && n.is_multiple_of(2) {
        (bond_len * bond_len - 4.0 * pucker * pucker)
            .max(0.04)
            .sqrt()
    } else {
        bond_len
    };
    let rad = planar_chord / (2.0 * (PI / k).sin());

    let e1 = norm3(out_dir);
    let arb = if e1[2].abs() < 0.85 {
        [0.0, 0.0, 1.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let e2 = norm3(cross3(e1, arb));
    let e3 = cross3(e1, e2);

    // Center lies at p_anchor + rad * e1, so anchor is at angle = PI in the (e1, e2) plane
    let center = [
        p_anchor[0] + rad * e1[0],
        p_anchor[1] + rad * e1[1],
        p_anchor[2] + rad * e1[2],
    ];
    for (idx, &v) in ring.iter().enumerate() {
        if coords[v].is_some() {
            continue;
        }
        let step = (idx as isize - pos_in_ring as isize).rem_euclid(n as isize) as f64;
        let ang = PI + 2.0 * PI * step / k;
        let z = if pucker <= 0.0 {
            0.0
        } else if n.is_multiple_of(2) {
            if idx.is_multiple_of(2) {
                pucker
            } else {
                -pucker
            }
        } else if idx == 0 {
            pucker * 1.6
        } else {
            0.0
        };
        let ca = ang.cos();
        let sa = ang.sin();
        coords[v] = Some([
            center[0] + rad * (ca * e1[0] + sa * e2[0]) + z * e3[0],
            center[1] + rad * (ca * e1[1] + sa * e2[1]) + z * e3[1],
            center[2] + rad * (ca * e1[2] + sa * e2[2]) + z * e3[2],
        ]);
    }
}

/// Place a ring in 3D fused across an already-placed shared edge `(ring[k1], ring[k2])`.
fn place_fused_ring_3d(
    mol: &Molecule3D,
    ring: &[usize],
    k1: usize,
    k2: usize,
    coords: &mut [Option<[f64; 3]>],
) {
    let (Some(p1), Some(p2)) = (coords[ring[k1]], coords[ring[k2]]) else {
        return;
    };
    let edge = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
    let elen = (edge[0] * edge[0] + edge[1] * edge[1] + edge[2] * edge[2])
        .sqrt()
        .max(1e-4);
    let e_edge = [edge[0] / elen, edge[1] / elen, edge[2] / elen];
    let mid = [
        0.5 * (p1[0] + p2[0]),
        0.5 * (p1[1] + p2[1]),
        0.5 * (p1[2] + p2[2]),
    ];
    // Choose in-plane outward normal pointing away from already-placed neighbours of ring[k1]/ring[k2]
    let cen = centroid_opt3(coords);
    let away = [mid[0] - cen[0], mid[1] - cen[1], mid[2] - cen[2]];
    let proj = away[0] * e_edge[0] + away[1] * e_edge[1] + away[2] * e_edge[2];
    let perp_raw = [
        away[0] - proj * e_edge[0],
        away[1] - proj * e_edge[1],
        away[2] - proj * e_edge[2],
    ];
    let perp_len =
        (perp_raw[0] * perp_raw[0] + perp_raw[1] * perp_raw[1] + perp_raw[2] * perp_raw[2]).sqrt();
    let e_out = if perp_len > 1e-4 {
        [
            perp_raw[0] / perp_len,
            perp_raw[1] / perp_len,
            perp_raw[2] / perp_len,
        ]
    } else {
        let arb = if e_edge[2].abs() < 0.85 {
            [0.0, 0.0, 1.0]
        } else {
            [1.0, 0.0, 0.0]
        };
        norm3(cross3(e_edge, arb))
    };
    let e_norm = norm3(cross3(e_edge, e_out));

    let n = ring.len();
    let sides = n as f64;
    let rad = elen / (2.0 * (PI / sides).sin());
    let apo = (rad * rad - (0.5 * elen).powi(2)).max(0.0).sqrt();
    let center = [
        mid[0] + apo * e_out[0],
        mid[1] + apo * e_out[1],
        mid[2] + apo * e_out[2],
    ];

    // Local 2D coordinates in (e_edge, e_out) frame
    let pucker = ring_pucker(mol, ring);
    let v1 = [p1[0] - center[0], p1[1] - center[1], p1[2] - center[2]];
    let v2 = [p2[0] - center[0], p2[1] - center[1], p2[2] - center[2]];
    let a1 = (v1[0] * e_out[0] + v1[1] * e_out[1] + v1[2] * e_out[2])
        .atan2(v1[0] * e_edge[0] + v1[1] * e_edge[1] + v1[2] * e_edge[2]);
    let a2 = (v2[0] * e_out[0] + v2[1] * e_out[1] + v2[2] * e_out[2])
        .atan2(v2[0] * e_edge[0] + v2[1] * e_edge[1] + v2[2] * e_edge[2]);
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
            let d = (idx as isize - k1 as isize).rem_euclid(n as isize) as f64;
            let ang = a1 + d * step;
            let z = if pucker <= 0.0 {
                0.0
            } else if idx.is_multiple_of(2) {
                pucker
            } else {
                -pucker
            };
            let ca = ang.cos();
            let sa = ang.sin();
            coords[v] = Some([
                center[0] + rad * (ca * e_edge[0] + sa * e_out[0]) + z * e_norm[0],
                center[1] + rad * (ca * e_edge[1] + sa * e_out[1]) + z * e_norm[1],
                center[2] + rad * (ca * e_edge[2] + sa * e_out[2]) + z * e_norm[2],
            ]);
        }
    }
}

/// Project every planar aromatic 5/6-ring onto its best-fit plane during 3D coordinate relaxation
/// so aromatic rings stay cleanly planar (~0.00 Å RMSD out-of-plane).
fn flatten_aromatic_rings(mol: &Molecule3D, pos: &mut [[f64; 3]]) {
    for ring in &mol.rings {
        if !(5..=6).contains(&ring.len()) {
            continue;
        }
        let is_arom = ring.iter().all(|&v| mol.atoms[v].aromatic);
        if !is_arom {
            continue;
        }
        let k = ring.len() as f64;
        let mut c = [0.0; 3];
        for &v in ring {
            for j in 0..3 {
                c[j] += pos[v][j] / k;
            }
        }
        // Compute mean ring normal via Newell's method (robust for any 5/6-gon orientation)
        let mut nrm = [0.0; 3];
        for i in 0..ring.len() {
            let a = pos[ring[i]];
            let b = pos[ring[(i + 1) % ring.len()]];
            let ra = [a[0] - c[0], a[1] - c[1], a[2] - c[2]];
            let rb = [b[0] - c[0], b[1] - c[1], b[2] - c[2]];
            let cr = cross3(ra, rb);
            for j in 0..3 {
                nrm[j] += cr[j];
            }
        }
        let nl = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt();
        if nl < 1e-6 {
            continue;
        }
        let u = [nrm[0] / nl, nrm[1] / nl, nrm[2] / nl];
        for &v in ring {
            let r = [pos[v][0] - c[0], pos[v][1] - c[1], pos[v][2] - c[2]];
            let h = r[0] * u[0] + r[1] * u[1] + r[2] * u[2];
            pos[v][0] -= 0.75 * h * u[0];
            pos[v][1] -= 0.75 * h * u[1];
            pos[v][2] -= 0.75 * h * u[2];
        }
    }
}

/// Out-of-plane amplitude for a ring: 0 when it is flat by chemistry, otherwise the standard
/// chair / envelope displacement.
fn ring_pucker(mol: &Molecule3D, ring: &[usize]) -> f64 {
    if !ring_is_saturated(mol, ring) || ring.len() < 5 {
        0.0
    } else if ring.len().is_multiple_of(2) {
        0.25
    } else {
        0.28
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

/// Locates amidine / guanidine / biguanide groups and decides where their single pH 7.4 charge
/// sits.
///
/// Returns `(group_nitrogens, protonation_sites)`. An amidinium or guanidinium is a *delocalised*
/// cation: the whole group carries one positive charge, shared across its nitrogens. Charging each
/// nitrogen separately gives benzamidine +2, guanidine +3 and metformin +5, which then propagates
/// into molecular weight, cLogP, hydrogen-bond counts, Gasteiger charges and every downstream
/// score. Centres that share a nitrogen (as the two amidine carbons of a biguanide do) are merged
/// into one group so metformin ends up +1, not +2.
fn amidine_guanidine_groups(mol: &Molecule3D) -> (BTreeSet<usize>, BTreeSet<usize>) {
    let mut group_ns = BTreeSet::new();
    let mut sites = BTreeSet::new();

    // 1. Amidine-like centres: a non-aromatic carbon bearing >= 2 nitrogens, at least one of them
    //    doubly bonded, and no doubly bonded O/S (which would make it an amide, urea or
    //    carbamate, none of them basic at pH 7.4).
    let mut centres: Vec<(usize, Vec<usize>, Vec<usize>)> = Vec::new(); // (c, all N, double-bonded N)
    for c in 0..mol.atoms.len() {
        if mol.atoms[c].element != "C" || mol.atoms[c].aromatic || !mol.atoms[c].is_hetatm {
            continue;
        }
        let mut ns = Vec::new();
        let mut double_ns = Vec::new();
        let mut has_double_chalcogen = false;
        for &(nb, bid) in &mol.adj[c] {
            let order = mol.bonds[bid].order;
            match mol.atoms[nb].element.as_str() {
                "N" if !mol.atoms[nb].aromatic => {
                    ns.push(nb);
                    if order == BondOrder::Double {
                        double_ns.push(nb);
                    }
                }
                "O" | "S" if order == BondOrder::Double => has_double_chalcogen = true,
                _ => {}
            }
        }
        if ns.len() >= 2 && !double_ns.is_empty() && !has_double_chalcogen {
            centres.push((c, ns, double_ns));
        }
    }
    if centres.is_empty() {
        return (group_ns, sites);
    }

    // 2. Merge centres that share a nitrogen (biguanides) into connected groups.
    let n_centres = centres.len();
    let mut parent: Vec<usize> = (0..n_centres).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while parent[r] != r {
            r = parent[r];
        }
        let mut cur = x;
        while parent[cur] != r {
            let next = parent[cur];
            parent[cur] = r;
            cur = next;
        }
        r
    }
    for a in 0..n_centres {
        for b in (a + 1)..n_centres {
            if centres[a].1.iter().any(|n| centres[b].1.contains(n)) {
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                if ra != rb {
                    parent[ra] = rb;
                }
            }
        }
    }

    // 3. One charge per group, placed on the lowest-indexed doubly bonded nitrogen.
    let mut by_root: HashMap<usize, (BTreeSet<usize>, BTreeSet<usize>)> = HashMap::new();
    for (i, centre) in centres.iter().enumerate() {
        let root = find(&mut parent, i);
        let entry = by_root.entry(root).or_default();
        entry.0.extend(centre.1.iter().copied());
        entry.1.extend(centre.2.iter().copied());
    }
    for (all_n, double_n) in by_root.into_values() {
        if let Some(&site) = double_n.iter().next().or_else(|| all_n.iter().next()) {
            sites.insert(site);
        }
        group_ns.extend(all_n);
    }
    (group_ns, sites)
}

/// Ideal bond angle at an atom, from its bonding: 180 degrees with a triple bond or two double
/// bonds, 120 degrees when it is aromatic or carries one double bond, 109.47 degrees otherwise.
fn ideal_bond_angle(mol: &Molecule3D, u: usize) -> f64 {
    let mut doubles = 0usize;
    let mut has_triple = false;
    for &(_, bid) in &mol.adj[u] {
        match mol.bonds[bid].order {
            BondOrder::Double => doubles += 1,
            BondOrder::Triple => has_triple = true,
            _ => {}
        }
    }
    if has_triple || doubles >= 2 {
        PI
    } else if doubles == 1 || mol.atoms[u].aromatic {
        2.0 * PI / 3.0
    } else {
        1.910_633_236_249_019 // acos(-1/3)
    }
}

/// A unit vector perpendicular to `e`, chosen deterministically and rotated by `phase` about `e`
/// so that successive centres along a chain do not all land in the same plane.
fn perpendicular_to(e: [f64; 3], phase: f64) -> [f64; 3] {
    let arb = if e[2].abs() < 0.8 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let t1 = norm3(cross3(e, arb));
    let t2 = cross3(e, t1);
    norm3([
        t1[0] * phase.cos() + t2[0] * phase.sin(),
        t1[1] * phase.cos() + t2[1] * phase.sin(),
        t1[2] * phase.cos() + t2[2] * phase.sin(),
    ])
}

/// The next bond direction at a centre that already points along `known`, such that it makes the
/// ideal angle `theta` with the existing bonds.
///
/// The builder used to spread every unplaced neighbour inside a fixed 41 degree cone around the
/// average opposite direction, whatever the geometry. That left sp3 carbons close to planar: the
/// signed volume of the four neighbours of alanine's alpha carbon came out at 0.025 A^3 where an
/// ideal tetrahedron gives 10.4, so the centre had almost no handedness to speak of and bond
/// angles were far from 109.5 degrees.
fn next_bond_direction(known: &[[f64; 3]], theta: f64, phase: f64) -> [f64; 3] {
    match known.len() {
        0 => [1.0, 0.0, 0.0],
        1 => {
            let e = known[0];
            let t = perpendicular_to(e, phase);
            let (s, c) = theta.sin_cos();
            norm3([
                e[0] * c + t[0] * s,
                e[1] * c + t[1] * s,
                e[2] * c + t[2] * s,
            ])
        }
        2 => {
            let (e0, e1) = (known[0], known[1]);
            let sum = [e0[0] + e1[0], e0[1] + e1[1], e0[2] + e1[2]];
            let mag2 = sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2];
            if mag2 < 1e-12 {
                // The two existing bonds are collinear and opposite: any perpendicular will do.
                return perpendicular_to(e0, phase);
            }
            let bis = norm3([-sum[0], -sum[1], -sum[2]]);
            let perp = norm3(cross3(e0, e1));
            let c0 = bis[0] * e0[0] + bis[1] * e0[1] + bis[2] * e0[2];
            if c0.abs() < 1e-9 {
                return bis;
            }
            let cb = (theta.cos() / c0).clamp(-1.0, 1.0);
            let sb = (1.0 - cb * cb).max(0.0).sqrt();
            norm3([
                bis[0] * cb + perp[0] * sb,
                bis[1] * cb + perp[1] * sb,
                bis[2] * cb + perp[2] * sb,
            ])
        }
        _ => {
            let mut sum = [0.0_f64; 3];
            for e in known {
                for k in 0..3 {
                    sum[k] += e[k];
                }
            }
            norm3([-sum[0], -sum[1], -sum[2]])
        }
    }
}

/// Whether a ring has an sp3 portion that should pucker out of plane.
///
/// Looser than [`ring_is_saturated`], which a fused ring can never satisfy: the two atoms a
/// tetralin shares with its benzene are aromatic and the bond between them is an aromatic bond,
/// so the saturated half would be treated as flat. What matters for geometry is whether the ring
/// atoms that are *not* part of an aromatic system are sp3, and whether there are enough of them
/// to pucker.
fn ring_has_sp3_portion(mol: &Molecule3D, ring: &[usize]) -> bool {
    let free: Vec<usize> = ring
        .iter()
        .copied()
        .filter(|&v| !mol.atoms[v].aromatic)
        .collect();
    if free.len() < 2 {
        return false;
    }
    free.iter().all(|&v| {
        !mol.adj[v]
            .iter()
            .any(|&(_, bid)| mol.bonds[bid].order == BondOrder::Double)
    })
}

/// Restores the pucker of saturated rings after the relaxation pass.
///
/// An isolated cyclohexane comes out of ring placement as a chair and stays one, but in a fused
/// system (tetralin, indane, tetrahydroisoquinoline, chromane) the two atoms shared with the
/// aromatic partner are pinned in its plane, and the bond-length relaxation then flattens the
/// saturated half as well. Flat is the wrong answer: a saturated six-ring fused to a benzene
/// adopts a half-chair, and the two carbons furthest from the fusion sit ~0.25 A out of the
/// aromatic plane.
///
/// The displacement alternates around the ring, which is what a chair is, and is damped to zero
/// at atoms that belong to an aromatic ring, which really are coplanar. Bond lengths are restored
/// by the relaxation steps that follow this call.
fn pucker_saturated_rings(mol: &Molecule3D, pos: &mut [[f64; 3]]) {
    for ring in &mol.rings {
        let n = ring.len();
        if !(5..=8).contains(&n) || !ring_has_sp3_portion(mol, ring) {
            continue;
        }
        let amplitude = if n.is_multiple_of(2) { 0.25 } else { 0.28 };
        // Mean plane of the ring: centroid plus the Newell normal.
        let mut cen = [0.0_f64; 3];
        for &v in ring {
            for k in 0..3 {
                cen[k] += pos[v][k] / n as f64;
            }
        }
        let mut normal = [0.0_f64; 3];
        for i in 0..n {
            let a = pos[ring[i]];
            let b = pos[ring[(i + 1) % n]];
            normal[0] += (a[1] - cen[1]) * (b[2] - cen[2]) - (a[2] - cen[2]) * (b[1] - cen[1]);
            normal[1] += (a[2] - cen[2]) * (b[0] - cen[0]) - (a[0] - cen[0]) * (b[2] - cen[2]);
            normal[2] += (a[0] - cen[0]) * (b[1] - cen[1]) - (a[1] - cen[1]) * (b[0] - cen[0]);
        }
        let nl = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if nl < 1e-6 {
            continue;
        }
        for component in normal.iter_mut() {
            *component /= nl;
        }

        // Ring distance from each position to the nearest atom pinned flat by an aromatic ring.
        let pinned: Vec<bool> = ring.iter().map(|&v| mol.atoms[v].aromatic).collect();
        let any_pinned = pinned.iter().any(|&b| b);
        let damping: Vec<f64> = (0..n)
            .map(|i| {
                if !any_pinned {
                    return 1.0;
                }
                let d = (0..n)
                    .filter(|&j| pinned[j])
                    .map(|j| {
                        let f = (i as isize - j as isize).unsigned_abs();
                        f.min(n - f)
                    })
                    .min()
                    .unwrap_or(0);
                (d as f64 / 2.0).min(1.0)
            })
            .collect();

        // An odd ring cannot alternate consistently all the way round, so it takes an envelope:
        // one flap atom out of the plane of the rest.
        let flap = (0..n)
            .filter(|&i| !pinned[i])
            .max_by(|&a, &b| damping[a].total_cmp(&damping[b]));
        for i in 0..n {
            let v = ring[i];
            if pinned[i] {
                continue;
            }
            let target = if n.is_multiple_of(2) {
                let sign = if i.is_multiple_of(2) { 1.0 } else { -1.0 };
                sign * amplitude * damping[i]
            } else if Some(i) == flap {
                amplitude * 1.6
            } else {
                0.0
            };
            let current = (pos[v][0] - cen[0]) * normal[0]
                + (pos[v][1] - cen[1]) * normal[1]
                + (pos[v][2] - cen[2]) * normal[2];
            let delta = target - current;
            for k in 0..3 {
                pos[v][k] += delta * normal[k];
            }
        }
    }
}

/// Ideal length of a bond between two elements at a given bond order, in angstroms.
///
/// Built from the Cordero/Pyykko single-bond covalent radii already in `element_covalent_radius`
/// (their sum reproduces C-C 1.52, C-N 1.47, C-O 1.42, C-S 1.81, C-Cl 1.78 and C-I 2.15), minus a
/// bond-order shortening: -0.19 A for a double bond, -0.32 A for a triple bond and -0.12 A for an
/// aromatic bond, with explicit values for the two common cases those corrections get wrong.
///
/// The 3D builder previously used a single constant per bond order -- 1.50 A for *every* single
/// bond regardless of the elements, and 1.34 A for every double bond -- so C-O came out 0.07 A too
/// long, C-N 0.03 A too long and every carbonyl C=O 0.11 A too long. Those errors go straight into
/// the docked pose, the van der Waals term and the H-bond geometry.
pub fn ideal_bond_length(el_a: &str, el_b: &str, order: BondOrder) -> f64 {
    let a = normalize_element(el_a);
    let b = normalize_element(el_b);
    let (lo, hi) = if a <= b {
        (a.as_str(), b.as_str())
    } else {
        (b.as_str(), a.as_str())
    };
    match (lo, hi, order) {
        ("O", "P", BondOrder::Double) => return 1.48,
        ("O", "S", BondOrder::Double) => return 1.44,
        ("N", "O", BondOrder::Double) => return 1.21,
        _ => {}
    }
    let base = element_covalent_radius(lo) + element_covalent_radius(hi);
    let shortening = match order {
        BondOrder::Single => 0.0,
        BondOrder::Aromatic => 0.12,
        BondOrder::Double => 0.19,
        BondOrder::Triple => 0.32,
    };
    (base - shortening).max(0.7)
}

/// A ring is saturated (and therefore puckered, not planar) when no atom of it is aromatic and no
/// bond *within* the ring is double or aromatic.
fn ring_is_saturated(mol: &Molecule3D, ring: &[usize]) -> bool {
    let n = ring.len();
    if ring.iter().any(|&v| mol.atoms[v].aromatic) {
        return false;
    }
    !(0..n).any(|w| {
        let (u, v) = (ring[w], ring[(w + 1) % n]);
        mol.adj[u]
            .iter()
            .find(|&&(nb, _)| nb == v)
            .is_some_and(|&(_, bid)| {
                matches!(
                    mol.bonds[bid].order,
                    BondOrder::Double | BondOrder::Aromatic
                )
            })
    })
}

/// Mean ideal length of the bonds that make up a ring.
fn ring_bond_length(mol: &Molecule3D, ring: &[usize]) -> f64 {
    let n = ring.len();
    let mut sum = 0.0;
    for w in 0..n {
        let (u, v) = (ring[w], ring[(w + 1) % n]);
        let order = mol.adj[u]
            .iter()
            .find(|&&(nb, _)| nb == v)
            .map_or(BondOrder::Single, |&(_, bid)| mol.bonds[bid].order);
        sum += ideal_bond_length(&mol.atoms[u].element, &mol.atoms[v].element, order);
    }
    sum / n as f64
}

/// Rotate every rotatable bond of `mol` by the given angles (radians), applied root-to-leaf so
/// that a parent torsion carries its whole distal branch. `angles` is indexed in the same order
/// as `mol.rotatable_bonds`; a shorter slice is cycled and extra entries are ignored.
///
/// The anchor side of each bond never moves, so the rigid core keeps its crystallographic
/// coordinates and only the distal branches are re-posed.
pub fn apply_torsion_angles(mol: &mut Molecule3D, angles: &[f64]) {
    if angles.is_empty() || mol.rotatable_bonds.is_empty() {
        return;
    }
    let bonds = mol.rotatable_bonds.clone();
    for (k, rb) in bonds.iter().enumerate() {
        let theta = angles[k % angles.len()];
        if !theta.is_finite() || theta == 0.0 {
            continue;
        }
        if rb.u >= mol.atoms.len() || rb.v >= mol.atoms.len() {
            continue;
        }
        let a = mol.atoms[rb.u].coords;
        let b = mol.atoms[rb.v].coords;
        let axis = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        if axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2] < 1e-18 {
            continue;
        }
        let k_axis = norm3(axis);
        let (s, c) = theta.sin_cos();
        let one_c = 1.0 - c;
        for &m in &rb.moving_atoms {
            if m >= mol.atoms.len() || m == rb.v {
                continue;
            }
            let p = mol.atoms[m].coords;
            let r = [p[0] - b[0], p[1] - b[1], p[2] - b[2]];
            let kxr = cross3(k_axis, r);
            let kdr = k_axis[0] * r[0] + k_axis[1] * r[1] + k_axis[2] * r[2];
            mol.atoms[m].coords = [
                b[0] + r[0] * c + kxr[0] * s + k_axis[0] * kdr * one_c,
                b[1] + r[1] * c + kxr[1] * s + k_axis[1] * kdr * one_c,
                b[2] + r[2] * c + kxr[2] * s + k_axis[2] * kdr * one_c,
            ];
        }
    }
}

/// Deterministic splitmix64, used so torsion randomization is reproducible without pulling an
/// RNG dependency into this layer.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Replace every torsion of `mol` by an angle drawn uniformly from `[-PI, PI)`, as happens when a
/// ligand arrives from a SMILES embedding rather than from its crystal pose. Returns the angles
/// applied. Deterministic for a given `seed`.
pub fn randomize_torsions(mol: &mut Molecule3D, seed: u64) -> Vec<f64> {
    let mut st = seed ^ 0x5DEE_CE66_D1FA_B0C5;
    let angles: Vec<f64> = (0..mol.rotatable_bonds.len())
        .map(|_| {
            let u = (splitmix64(&mut st) >> 11) as f64 / ((1u64 << 53) as f64);
            (u - 0.5) * 2.0 * PI
        })
        .collect();
    apply_torsion_angles(mol, &angles);
    angles
}
