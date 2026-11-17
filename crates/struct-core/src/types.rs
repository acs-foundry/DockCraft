use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};

/// AutoDock4 / Vina atom types.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize,
)]
pub enum Ad4Type {
    #[default]
    C,
    A,
    N,
    NA,
    NS,
    OA,
    OS,
    SA,
    S,
    H,
    HD,
    F,
    Cl,
    Br,
    I,
    P,
    Zn,
    Mg,
    Fe,
    Ca,
    Mn,
    X,
}

impl Ad4Type {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::C => "C",
            Self::A => "A",
            Self::N => "N",
            Self::NA => "NA",
            Self::NS => "NS",
            Self::OA => "OA",
            Self::OS => "OS",
            Self::SA => "SA",
            Self::S => "S",
            Self::H => "H",
            Self::HD => "HD",
            Self::F => "F",
            Self::Cl => "Cl",
            Self::Br => "Br",
            Self::I => "I",
            Self::P => "P",
            Self::Zn => "Zn",
            Self::Mg => "Mg",
            Self::Fe => "Fe",
            Self::Ca => "Ca",
            Self::Mn => "Mn",
            Self::X => "X",
        }
    }

    pub fn from_ad4_str(s: &str) -> Self {
        match s.trim() {
            "C" => Self::C,
            "A" => Self::A,
            "N" => Self::N,
            "NA" => Self::NA,
            "NS" => Self::NS,
            "OA" => Self::OA,
            "OS" => Self::OS,
            "SA" => Self::SA,
            "S" => Self::S,
            "H" => Self::H,
            "HD" => Self::HD,
            "F" => Self::F,
            "Cl" | "CL" => Self::Cl,
            "Br" | "BR" => Self::Br,
            "I" => Self::I,
            "P" => Self::P,
            "Zn" | "ZN" => Self::Zn,
            "Mg" | "MG" => Self::Mg,
            "Fe" | "FE" => Self::Fe,
            "Ca" | "CA" => Self::Ca,
            "Mn" | "MN" => Self::Mn,
            _ => Self::X,
        }
    }

    /// AutoDock Vina X-score van der Waals radii (Å).
    pub fn vina_radius(self) -> f64 {
        match self {
            Self::C | Self::A => 1.90,
            Self::N | Self::NA | Self::NS => 1.80,
            Self::OA | Self::OS => 1.70,
            Self::SA | Self::S => 2.00,
            Self::P => 2.10,
            Self::F => 1.50,
            Self::Cl => 1.80,
            Self::Br => 2.00,
            Self::I => 2.20,
            Self::H | Self::HD => 1.00,
            Self::Zn | Self::Mg | Self::Fe | Self::Ca | Self::Mn => 1.20,
            Self::X => 1.80,
        }
    }

    pub fn is_hydrophobic(self) -> bool {
        matches!(
            self,
            Self::C | Self::A | Self::F | Self::Cl | Self::Br | Self::I
        )
    }

    pub fn is_donor_heavy(self) -> bool {
        matches!(self, Self::N | Self::NA | Self::NS | Self::OA | Self::SA)
    }

    pub fn is_acceptor(self) -> bool {
        matches!(self, Self::NA | Self::NS | Self::OA | Self::OS | Self::SA)
    }

    pub fn is_heavy(self) -> bool {
        !matches!(self, Self::H | Self::HD)
    }
}

pub fn normalize_element(raw: &str) -> String {
    let t = raw.trim();
    if t.is_empty() {
        return "C".to_string();
    }
    let upper = t.to_ascii_uppercase();
    match upper.as_str() {
        "CL" => "Cl".to_string(),
        "BR" => "Br".to_string(),
        "ZN" => "Zn".to_string(),
        "MG" => "Mg".to_string(),
        "FE" => "Fe".to_string(),
        "CA" => "Ca".to_string(),
        "MN" => "Mn".to_string(),
        "SI" => "Si".to_string(),
        "SE" => "Se".to_string(),
        _ => {
            let mut chars = upper.chars();
            match (chars.next(), chars.next()) {
                (Some(c0), Some(c1)) if c1.is_ascii_alphabetic() => {
                    format!("{c0}{}", c1.to_ascii_lowercase())
                }
                (Some(c0), _) => c0.to_string(),
                _ => "C".to_string(),
            }
        }
    }
}

pub fn element_atomic_number(sym: &str) -> u8 {
    match normalize_element(sym).as_str() {
        "H" => 1,
        "B" => 5,
        "C" => 6,
        "N" => 7,
        "O" => 8,
        "F" => 9,
        "Mg" => 12,
        "P" => 15,
        "S" => 16,
        "Cl" => 17,
        "Ca" => 20,
        "Mn" => 25,
        "Fe" => 26,
        "Zn" => 30,
        "Br" => 35,
        "I" => 53,
        _ => 6,
    }
}

pub fn element_weight(sym: &str) -> f64 {
    match normalize_element(sym).as_str() {
        "H" => 1.008,
        "B" => 10.81,
        "C" => 12.011,
        "N" => 14.007,
        "O" => 15.999,
        "F" => 18.998,
        "Mg" => 24.305,
        "P" => 30.974,
        "S" => 32.06,
        "Cl" => 35.45,
        "Ca" => 40.078,
        "Mn" => 54.938,
        "Fe" => 55.845,
        "Zn" => 65.38,
        "Br" => 79.904,
        "I" => 126.904,
        _ => 12.011,
    }
}

pub fn element_covalent_radius(sym: &str) -> f64 {
    match normalize_element(sym).as_str() {
        "H" => 0.31,
        "C" => 0.76,
        "N" => 0.71,
        "O" => 0.66,
        "F" => 0.57,
        "P" => 1.07,
        "S" => 1.05,
        "Cl" => 1.02,
        "Br" => 1.20,
        "I" => 1.39,
        _ => 0.80,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum BondOrder {
    #[default]
    Single,
    Double,
    Triple,
    Aromatic,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Atom3D {
    pub id: usize,
    pub name: String,
    pub res_name: String,
    pub chain_id: char,
    pub res_seq: i32,
    pub is_hetatm: bool,
    pub element: String,
    pub coords: [f64; 3],
    pub charge: f64,
    pub formal_charge: i8,
    pub ad4_type: Ad4Type,
    pub aromatic: bool,
    pub h_donor: bool,
    pub h_acceptor: bool,
    pub hydrophobic: bool,
    pub b_factor: f64,
    pub is_backbone: bool,
}

impl Atom3D {
    pub fn new(id: usize, element: &str, coords: [f64; 3]) -> Self {
        let el = normalize_element(element);
        let ad4_type = match el.as_str() {
            "H" => Ad4Type::H,
            "N" => Ad4Type::N,
            "O" => Ad4Type::OA,
            "S" => Ad4Type::SA,
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
            _ => Ad4Type::C,
        };
        Self {
            id,
            name: el.clone(),
            res_name: "LIG".to_string(),
            chain_id: 'A',
            res_seq: 1,
            is_hetatm: true,
            element: el,
            coords,
            charge: 0.0,
            formal_charge: 0,
            ad4_type,
            aromatic: false,
            h_donor: false,
            h_acceptor: false,
            hydrophobic: ad4_type.is_hydrophobic(),
            b_factor: 20.0,
            is_backbone: false,
        }
    }

    pub fn is_heavy(&self) -> bool {
        self.element != "H"
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bond3D {
    pub id: usize,
    pub u: usize,
    pub v: usize,
    pub order: BondOrder,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RotatableBond {
    pub bond_id: usize,
    pub u: usize,
    pub v: usize,
    pub moving_atoms: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Molecule3D {
    pub name: String,
    pub atoms: Vec<Atom3D>,
    pub bonds: Vec<Bond3D>,
    pub adj: Vec<Vec<(usize, usize)>>,
    pub source_smiles: Option<String>,
    pub rotatable_bonds: Vec<RotatableBond>,
    pub rings: Vec<Vec<usize>>,
    pub properties: BTreeMap<String, String>,
}

impl Molecule3D {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    pub fn add_atom(&mut self, mut atom: Atom3D) -> usize {
        let id = self.atoms.len();
        atom.id = id;
        self.atoms.push(atom);
        self.adj.push(Vec::new());
        id
    }

    pub fn add_bond(&mut self, u: usize, v: usize, order: BondOrder) -> Option<usize> {
        if u >= self.atoms.len() || v >= self.atoms.len() || u == v {
            return None;
        }
        if let Some(&(_, existing)) = self.adj[u].iter().find(|&&(nb, _)| nb == v) {
            if self.bonds[existing].order == BondOrder::Single && order != BondOrder::Single {
                self.bonds[existing].order = order;
            }
            return Some(existing);
        }
        let id = self.bonds.len();
        self.bonds.push(Bond3D { id, u, v, order });
        self.adj[u].push((v, id));
        self.adj[v].push((u, id));
        Some(id)
    }

    pub fn heavy_atom_count(&self) -> usize {
        self.atoms.iter().filter(|a| a.is_heavy()).count()
    }

    pub fn implicit_hydrogens_at(&self, idx: usize) -> usize {
        let Some(a) = self.atoms.get(idx) else {
            return 0;
        };
        if !a.is_heavy() {
            return 0;
        }
        let mut bo_sum = 0.0_f64;
        if let Some(nbrs) = self.adj.get(idx) {
            for &(_, bid) in nbrs {
                bo_sum += match self.bonds[bid].order {
                    BondOrder::Single => 1.0,
                    BondOrder::Double => 2.0,
                    BondOrder::Triple => 3.0,
                    BondOrder::Aromatic => 1.5,
                };
            }
        }
        if bo_sum == 0.0 && !self.bonds.is_empty() {
            return 0;
        }
        let fc = i32::from(a.formal_charge);
        let target: i32 = match a.element.as_str() {
            "C" => (4 - fc.abs()).max(0),
            "N" => (3 + fc).clamp(0, 4),
            "O" => (2 + fc).clamp(0, 3),
            "S" => {
                if bo_sum <= 2.1 {
                    (2 + fc).clamp(0, 3)
                } else if bo_sum <= 4.1 {
                    4
                } else {
                    6
                }
            }
            "P" => {
                if bo_sum <= 3.1 {
                    (3 + fc).clamp(0, 4)
                } else {
                    5
                }
            }
            "F" | "Cl" | "Br" | "I" => (1 + fc).clamp(0, 1),
            _ => 0,
        };
        let used = bo_sum.ceil() as i32;
        (target - used).max(0) as usize
    }

    pub fn molecular_weight(&self) -> f64 {
        let explicit: f64 = self.atoms.iter().map(|a| element_weight(&a.element)).sum();
        let implicit_h: usize = if self.bonds.is_empty() {
            0
        } else {
            (0..self.atoms.len())
                .map(|i| self.implicit_hydrogens_at(i))
                .sum()
        };
        explicit + (implicit_h as f64) * element_weight("H")
    }

    /// Indices of the stationary rigid core (all heavy atoms not belonging to any distal moving branch).
    pub fn rigid_core_indices(&self) -> Vec<usize> {
        let mut moving = vec![false; self.atoms.len()];
        for rb in &self.rotatable_bonds {
            for &m in &rb.moving_atoms {
                if m < moving.len() {
                    moving[m] = true;
                }
            }
        }
        let core: Vec<usize> = self
            .atoms
            .iter()
            .enumerate()
            .filter(|(i, a)| a.is_heavy() && !moving[*i])
            .map(|(i, _)| i)
            .collect();
        if core.is_empty() {
            self.atoms
                .iter()
                .enumerate()
                .filter(|(_, a)| a.is_heavy())
                .map(|(i, _)| i)
                .collect()
        } else {
            core
        }
    }

    pub fn formula(&self) -> String {
        let mut counts: BTreeMap<&str, u32> = BTreeMap::new();
        for a in &self.atoms {
            *counts.entry(a.element.as_str()).or_default() += 1;
        }
        if !self.bonds.is_empty() {
            let imp_h: u32 = (0..self.atoms.len())
                .map(|i| self.implicit_hydrogens_at(i) as u32)
                .sum();
            if imp_h > 0 {
                *counts.entry("H").or_default() += imp_h;
            }
        }
        let mut out = String::new();
        if let Some(c) = counts.remove("C") {
            out.push('C');
            if c > 1 {
                out.push_str(&c.to_string());
            }
            if let Some(h) = counts.remove("H") {
                out.push('H');
                if h > 1 {
                    out.push_str(&h.to_string());
                }
            }
        }
        for (el, cnt) in counts {
            out.push_str(el);
            if cnt > 1 {
                out.push_str(&cnt.to_string());
            }
        }
        out
    }

    pub fn centroid_heavy(&self) -> [f64; 3] {
        let mut s = [0.0; 3];
        let mut n = 0.0;
        for a in &self.atoms {
            if a.is_heavy() {
                s[0] += a.coords[0];
                s[1] += a.coords[1];
                s[2] += a.coords[2];
                n += 1.0;
            }
        }
        if n > 0.0 {
            [s[0] / n, s[1] / n, s[2] / n]
        } else {
            [0.0; 3]
        }
    }

    pub fn bounding_box(&self) -> ([f64; 3], [f64; 3]) {
        if self.atoms.is_empty() {
            return ([0.0; 3], [0.0; 3]);
        }
        let mut min_p = [f64::INFINITY; 3];
        let mut max_p = [f64::NEG_INFINITY; 3];
        for a in &self.atoms {
            for k in 0..3 {
                min_p[k] = min_p[k].min(a.coords[k]);
                max_p[k] = max_p[k].max(a.coords[k]);
            }
        }
        (min_p, max_p)
    }

    /// Heavy-atom root-mean-square deviation (Å) assuming corresponding atom ordering.
    pub fn rmsd_heavy(&self, other: &Self) -> Option<f64> {
        let h1: Vec<[f64; 3]> = self
            .atoms
            .iter()
            .filter(|a| a.is_heavy())
            .map(|a| a.coords)
            .collect();
        let h2: Vec<[f64; 3]> = other
            .atoms
            .iter()
            .filter(|a| a.is_heavy())
            .map(|a| a.coords)
            .collect();
        if h1.is_empty() || h1.len() != h2.len() {
            return None;
        }
        let sum_sq: f64 = h1
            .iter()
            .zip(h2.iter())
            .map(|(a, b)| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2))
            .sum();
        Some((sum_sq / (h1.len() as f64)).sqrt())
    }

    /// Distance-based covalent bond inference via spatial grid (`O(N)` for large proteins).
    pub fn infer_bonds_by_distance(&mut self) {
        if !self.bonds.is_empty() || self.atoms.len() <= 1 {
            return;
        }
        let cell_size = 2.4_f64;
        let mut grid: BTreeMap<(i32, i32, i32), Vec<usize>> = BTreeMap::new();
        for (i, a) in self.atoms.iter().enumerate() {
            let key = (
                (a.coords[0] / cell_size).floor() as i32,
                (a.coords[1] / cell_size).floor() as i32,
                (a.coords[2] / cell_size).floor() as i32,
            );
            grid.entry(key).or_default().push(i);
        }

        let mut new_edges: Vec<(usize, usize)> = Vec::new();
        for (i, a) in self.atoms.iter().enumerate() {
            let ri = element_covalent_radius(&a.element);
            let gx = (a.coords[0] / cell_size).floor() as i32;
            let gy = (a.coords[1] / cell_size).floor() as i32;
            let gz = (a.coords[2] / cell_size).floor() as i32;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let Some(bucket) = grid.get(&(gx + dx, gy + dy, gz + dz)) else {
                            continue;
                        };
                        for &j in bucket {
                            if j <= i {
                                continue;
                            }
                            let b = &self.atoms[j];
                            if a.element == "H" && b.element == "H" {
                                continue;
                            }
                            let dist_sq = (a.coords[0] - b.coords[0]).powi(2)
                                + (a.coords[1] - b.coords[1]).powi(2)
                                + (a.coords[2] - b.coords[2]).powi(2);
                            let cutoff = (ri + element_covalent_radius(&b.element)) * 1.24 + 0.15;
                            if dist_sq <= cutoff * cutoff && dist_sq >= 0.16 {
                                new_edges.push((i, j));
                            }
                        }
                    }
                }
            }
        }
        for (u, v) in new_edges {
            let _ = self.add_bond(u, v, BondOrder::Single);
        }
    }

    /// Detect small 5/6-membered rings via shortest-path BFS per bond (fast even for large proteins).
    pub fn detect_small_rings(&mut self) {
        self.rings.clear();
        let n = self.atoms.len();
        if n < 3 || self.bonds.len() < 3 {
            return;
        }
        let mut seen_sets: std::collections::BTreeSet<Vec<usize>> =
            std::collections::BTreeSet::new();
        for (bid, b) in self.bonds.iter().enumerate() {
            if self.atoms[b.u].element == "H" || self.atoms[b.v].element == "H" {
                continue;
            }
            let mut visited: BTreeMap<usize, (usize, u8)> = BTreeMap::new();
            let mut q = VecDeque::new();
            visited.insert(b.u, (usize::MAX, 0));
            q.push_back(b.u);
            while let Some(cur) = q.pop_front() {
                let d_cur = visited.get(&cur).map_or(5, |&(_, d)| d);
                if cur == b.v || d_cur >= 5 {
                    continue;
                }
                for &(nbr, e) in &self.adj[cur] {
                    if e == bid || self.atoms[nbr].element == "H" {
                        continue;
                    }
                    if let std::collections::btree_map::Entry::Vacant(e) = visited.entry(nbr) {
                        e.insert((cur, d_cur + 1));
                        q.push_back(nbr);
                    }
                }
            }
            if let Some(&(mut prev_node, dv)) = visited.get(&b.v)
                && (2..=5).contains(&dv)
            {
                let mut ring = vec![b.v];
                while prev_node != usize::MAX {
                    ring.push(prev_node);
                    prev_node = visited.get(&prev_node).map_or(usize::MAX, |&(p, _)| p);
                }
                if (3..=6).contains(&ring.len()) {
                    let mut sorted = ring.clone();
                    sorted.sort_unstable();
                    if seen_sets.insert(sorted) {
                        self.rings.push(ring);
                    }
                }
            }
        }
    }
}
