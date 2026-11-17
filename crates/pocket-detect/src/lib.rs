#![forbid(unsafe_code)]

use std::collections::{BTreeSet, HashMap, VecDeque};

use serde::{Deserialize, Serialize};
use struct_core::{Atom3D, Molecule3D};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PocketError {
    #[error("receptor contains no heavy atoms")]
    EmptyReceptor,
    #[error("ligand contains no heavy atoms")]
    NoLigandAtoms,
    #[error("invalid grid box dimensions: {0}")]
    InvalidBox(String),
}

pub const MAX_GRID_BOX_AXIS_A: f64 = 60.0;
pub const MAX_GRID_VOXELS: usize = 1_500_000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GridBox {
    pub center: [f64; 3],
    pub size: [f64; 3],
    pub spacing: f64,
}

impl GridBox {
    pub fn from_center_size(
        center: [f64; 3],
        size: [f64; 3],
        spacing: f64,
    ) -> Result<Self, PocketError> {
        if !spacing.is_finite() || spacing < 0.15 || spacing > 2.0 {
            return Err(PocketError::InvalidBox(format!(
                "grid spacing {spacing} out of [0.15, 2.0]"
            )));
        }
        for (axis, (&c, &s)) in center.iter().zip(size.iter()).enumerate() {
            if !c.is_finite() || !s.is_finite() || s <= 1.0 || s > MAX_GRID_BOX_AXIS_A {
                return Err(PocketError::InvalidBox(format!(
                    "axis {axis} center/size ({c}, {s}) must have 1.0 < size <= {MAX_GRID_BOX_AXIS_A} A"
                )));
            }
        }
        let gbox = Self {
            center,
            size,
            spacing,
        };
        let dims = gbox.dimensions();
        let voxels = dims[0]
            .checked_mul(dims[1])
            .and_then(|v| v.checked_mul(dims[2]))
            .unwrap_or(usize::MAX);
        if voxels > MAX_GRID_VOXELS {
            return Err(PocketError::InvalidBox(format!(
                "grid voxel count {voxels} exceeds limit {MAX_GRID_VOXELS}"
            )));
        }
        Ok(gbox)
    }

    pub fn from_ligand(ligand: &Molecule3D, padding: f64) -> Result<Self, PocketError> {
        let heavy: Vec<[f64; 3]> = ligand
            .atoms
            .iter()
            .filter(|a| a.is_heavy())
            .map(|a| a.coords)
            .collect();
        if heavy.is_empty() {
            return Err(PocketError::NoLigandAtoms);
        }
        let mut min_c = [f64::INFINITY; 3];
        let mut max_c = [f64::NEG_INFINITY; 3];
        for c in &heavy {
            for k in 0..3 {
                min_c[k] = min_c[k].min(c[k]);
                max_c[k] = max_c[k].max(c[k]);
            }
        }
        let pad = padding.max(4.0);
        let center = [
            0.5 * (min_c[0] + max_c[0]),
            0.5 * (min_c[1] + max_c[1]),
            0.5 * (min_c[2] + max_c[2]),
        ];
        let size = [
            ((max_c[0] - min_c[0]) + 2.0 * pad).max(14.0),
            ((max_c[1] - min_c[1]) + 2.0 * pad).max(14.0),
            ((max_c[2] - min_c[2]) + 2.0 * pad).max(14.0),
        ];
        Self::from_center_size(center, size, 0.375)
    }

    #[must_use]
    pub fn min_corner(&self) -> [f64; 3] {
        [
            self.center[0] - 0.5 * self.size[0],
            self.center[1] - 0.5 * self.size[1],
            self.center[2] - 0.5 * self.size[2],
        ]
    }

    #[must_use]
    pub fn max_corner(&self) -> [f64; 3] {
        [
            self.center[0] + 0.5 * self.size[0],
            self.center[1] + 0.5 * self.size[1],
            self.center[2] + 0.5 * self.size[2],
        ]
    }

    #[must_use]
    pub fn dimensions(&self) -> [usize; 3] {
        let nx = ((self.size[0] / self.spacing).ceil() as usize).max(2) + 1;
        let ny = ((self.size[1] / self.spacing).ceil() as usize).max(2) + 1;
        let nz = ((self.size[2] / self.spacing).ceil() as usize).max(2) + 1;
        [nx, ny, nz]
    }

    #[must_use]
    pub fn contains(&self, p: [f64; 3]) -> bool {
        let min_c = self.min_corner();
        let max_c = self.max_corner();
        (0..3).all(|k| p[k] >= min_c[k] && p[k] <= max_c[k])
    }

    #[must_use]
    pub fn volume(&self) -> f64 {
        self.size[0] * self.size[1] * self.size[2]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ResidueTag {
    pub chain_id: String,
    pub res_seq: i32,
    pub res_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingPocket {
    pub id: usize,
    pub rank: usize,
    pub center: [f64; 3],
    pub grid_box: GridBox,
    pub volume_a3: f64,
    pub sasa_a2: f64,
    pub buriedness: f64,
    pub hydrophobicity: f64,
    pub hbond_capacity: f64,
    pub druggability_score: f64,
    pub probe_points: Vec<[f64; 3]>,
    pub lining_residues: Vec<ResidueTag>,
    pub lining_atom_indices: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketDetectConfig {
    pub grid_step: f64,
    pub probe_radius: f64,
    pub max_receptor_dist: f64,
    pub min_enclosed_rays: usize,
    pub min_pocket_points: usize,
    pub box_padding: f64,
    pub box_spacing: f64,
}

impl Default for PocketDetectConfig {
    fn default() -> Self {
        Self {
            grid_step: 1.2,
            probe_radius: 1.5,
            max_receptor_dist: 7.5,
            min_enclosed_rays: 9,
            min_pocket_points: 10,
            box_padding: 6.0,
            box_spacing: 0.375,
        }
    }
}

struct SpatialHash {
    cell_size: f64,
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl SpatialHash {
    fn new(atoms: &[&Atom3D], cell_size: f64) -> Self {
        let mut cells: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for (idx, atom) in atoms.iter().enumerate() {
            let key = Self::cell_of(atom.coords, cell_size);
            cells.entry(key).or_default().push(idx);
        }
        Self { cell_size, cells }
    }

    fn cell_of(p: [f64; 3], cell_size: f64) -> (i32, i32, i32) {
        (
            (p[0] / cell_size).floor() as i32,
            (p[1] / cell_size).floor() as i32,
            (p[2] / cell_size).floor() as i32,
        )
    }

    fn query_radius(&self, atoms: &[&Atom3D], center: [f64; 3], radius: f64) -> Vec<usize> {
        let span = (radius / self.cell_size).ceil() as i32 + 1;
        let (cx, cy, cz) = Self::cell_of(center, self.cell_size);
        let r2 = radius * radius;
        let mut out = Vec::new();
        for dx in -span..=span {
            for dy in -span..=span {
                for dz in -span..=span {
                    if let Some(bucket) = self.cells.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &idx in bucket {
                            let c = atoms[idx].coords;
                            let d2 = dist2(center, c);
                            if d2 <= r2 {
                                out.push(idx);
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// 14 unit directions: 6 cardinal axes + 8 cube diagonals.
fn enclosure_directions() -> [[f64; 3]; 14] {
    let inv_sqrt3 = 1.0 / 3.0_f64.sqrt();
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [inv_sqrt3, inv_sqrt3, inv_sqrt3],
        [inv_sqrt3, inv_sqrt3, -inv_sqrt3],
        [inv_sqrt3, -inv_sqrt3, inv_sqrt3],
        [inv_sqrt3, -inv_sqrt3, -inv_sqrt3],
        [-inv_sqrt3, inv_sqrt3, inv_sqrt3],
        [-inv_sqrt3, inv_sqrt3, -inv_sqrt3],
        [-inv_sqrt3, -inv_sqrt3, inv_sqrt3],
        [-inv_sqrt3, -inv_sqrt3, -inv_sqrt3],
    ]
}

fn ray_hits_receptor(
    origin: [f64; 3],
    dir: [f64; 3],
    max_dist: f64,
    nearby: &[usize],
    heavy_atoms: &[&Atom3D],
) -> bool {
    for &idx in nearby {
        let a = heavy_atoms[idx];
        let oc = [
            a.coords[0] - origin[0],
            a.coords[1] - origin[1],
            a.coords[2] - origin[2],
        ];
        let proj = oc[0] * dir[0] + oc[1] * dir[1] + oc[2] * dir[2];
        if proj <= 0.5 || proj > max_dist {
            continue;
        }
        let perp2 = (oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2]) - proj * proj;
        let hit_r = a.ad4_type.vina_radius() + 0.9;
        if perp2 <= hit_r * hit_r {
            return true;
        }
    }
    false
}

/// 42-point Fibonacci sphere Shrake-Rupley SASA for `subset` atoms in context of `all_heavy`.
#[must_use]
pub fn compute_sasa_fibonacci(
    all_heavy: &[&Atom3D],
    subset_indices: &[usize],
    probe_radius: f64,
) -> f64 {
    const N_PTS: usize = 42;
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let mut unit_pts = [[0.0_f64; 3]; N_PTS];
    for (i, pt) in unit_pts.iter_mut().enumerate() {
        let y = 1.0 - (i as f64 / (N_PTS - 1) as f64) * 2.0;
        let r = (1.0 - y * y).max(0.0).sqrt();
        let theta = golden_angle * i as f64;
        *pt = [theta.cos() * r, y, theta.sin() * r];
    }

    let hash = SpatialHash::new(all_heavy, 4.5);
    let mut total_sasa = 0.0;

    for &idx in subset_indices {
        let Some(atom) = all_heavy.get(idx) else {
            continue;
        };
        let r_i = atom.ad4_type.vina_radius() + probe_radius;
        let neighbors = hash.query_radius(all_heavy, atom.coords, r_i + 2.4 + probe_radius);
        let mut exposed = 0usize;

        for up in &unit_pts {
            let p = [
                atom.coords[0] + r_i * up[0],
                atom.coords[1] + r_i * up[1],
                atom.coords[2] + r_i * up[2],
            ];
            let mut occluded = false;
            for &n_idx in &neighbors {
                if n_idx == idx {
                    continue;
                }
                let n_atom = all_heavy[n_idx];
                let r_j = n_atom.ad4_type.vina_radius() + probe_radius;
                if dist2(p, n_atom.coords) < r_j * r_j {
                    occluded = true;
                    break;
                }
            }
            if !occluded {
                exposed += 1;
            }
        }

        let sphere_area = 4.0 * std::f64::consts::PI * r_i * r_i;
        total_sasa += sphere_area * (exposed as f64 / N_PTS as f64);
    }

    total_sasa
}

pub fn detect_pockets(
    receptor: &Molecule3D,
    config: &PocketDetectConfig,
) -> Result<Vec<BindingPocket>, PocketError> {
    let mut heavy_orig_idx = Vec::new();
    let mut heavy_atoms = Vec::new();
    for (idx, atom) in receptor.atoms.iter().enumerate() {
        if atom.is_heavy() {
            heavy_orig_idx.push(idx);
            heavy_atoms.push(atom);
        }
    }
    if heavy_atoms.is_empty() {
        return Err(PocketError::EmptyReceptor);
    }

    let mut min_c = [f64::INFINITY; 3];
    let mut max_c = [f64::NEG_INFINITY; 3];
    for atom in &heavy_atoms {
        for k in 0..3 {
            min_c[k] = min_c[k].min(atom.coords[k]);
            max_c[k] = max_c[k].max(atom.coords[k]);
        }
    }

    let step = config.grid_step.max(0.6);
    let ray_horizon = 11.0;
    let hash = SpatialHash::new(&heavy_atoms, 4.0);
    let dirs = enclosure_directions();

    type CavityVoxel = ((i32, i32, i32), [f64; 3], usize);
    let mut cavity_voxels: Vec<CavityVoxel> = Vec::new();
    let nx = ((max_c[0] - min_c[0]) / step).ceil() as i32;
    let ny = ((max_c[1] - min_c[1]) / step).ceil() as i32;
    let nz = ((max_c[2] - min_c[2]) / step).ceil() as i32;

    for ix in 0..=nx {
        let x = min_c[0] + ix as f64 * step;
        for iy in 0..=ny {
            let y = min_c[1] + iy as f64 * step;
            for iz in 0..=nz {
                let z = min_c[2] + iz as f64 * step;
                let p = [x, y, z];

                let nearby = hash.query_radius(&heavy_atoms, p, ray_horizon);
                if nearby.is_empty() {
                    continue;
                }

                let mut clashes = false;
                let mut min_surface_dist = f64::INFINITY;
                for &idx in &nearby {
                    let a = heavy_atoms[idx];
                    let d = dist2(p, a.coords).sqrt();
                    let min_allowed = a.ad4_type.vina_radius() + config.probe_radius * 0.65;
                    if d < min_allowed {
                        clashes = true;
                        break;
                    }
                    min_surface_dist = min_surface_dist.min(d);
                }
                if clashes || min_surface_dist > config.max_receptor_dist {
                    continue;
                }

                let enclosed = dirs
                    .iter()
                    .filter(|&&dir| ray_hits_receptor(p, dir, ray_horizon, &nearby, &heavy_atoms))
                    .count();
                if enclosed >= config.min_enclosed_rays {
                    cavity_voxels.push(((ix, iy, iz), p, enclosed));
                }
            }
        }
    }

    let mut voxel_map: HashMap<(i32, i32, i32), usize> = HashMap::new();
    for (idx, &(ijk, _, _)) in cavity_voxels.iter().enumerate() {
        voxel_map.insert(ijk, idx);
    }

    let mut visited = vec![false; cavity_voxels.len()];
    let mut clusters: Vec<Vec<usize>> = Vec::new();

    for start in 0..cavity_voxels.len() {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut queue = VecDeque::from([start]);
        let mut cluster = Vec::new();
        while let Some(curr) = queue.pop_front() {
            cluster.push(curr);
            let (cx, cy, cz) = cavity_voxels[curr].0;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if dx == 0 && dy == 0 && dz == 0 {
                            continue;
                        }
                        if let Some(&nbr) = voxel_map.get(&(cx + dx, cy + dy, cz + dz))
                            && !visited[nbr]
                        {
                            visited[nbr] = true;
                            queue.push_back(nbr);
                        }
                    }
                }
            }
        }
        if cluster.len() >= config.min_pocket_points {
            clusters.push(cluster);
        }
    }

    let voxel_vol = step * step * step;
    let mut pockets = Vec::with_capacity(clusters.len());

    for (cid, cluster) in clusters.into_iter().enumerate() {
        let mut pts = Vec::with_capacity(cluster.len());
        let mut sum = [0.0_f64; 3];
        let mut sum_enclosure = 0usize;
        let mut min_p = [f64::INFINITY; 3];
        let mut max_p = [f64::NEG_INFINITY; 3];
        let mut lining_heavy_set = BTreeSet::new();

        for &vidx in &cluster {
            let (_, p, enc) = cavity_voxels[vidx];
            pts.push(p);
            sum_enclosure += enc;
            for k in 0..3 {
                sum[k] += p[k];
                min_p[k] = min_p[k].min(p[k]);
                max_p[k] = max_p[k].max(p[k]);
            }
            for n_idx in hash.query_radius(&heavy_atoms, p, 4.6) {
                lining_heavy_set.insert(n_idx);
            }
        }

        let n = pts.len() as f64;
        let center = [sum[0] / n, sum[1] / n, sum[2] / n];
        let volume_a3 = n * voxel_vol;
        let buriedness = (sum_enclosure as f64 / (n * 14.0)).clamp(0.0, 1.0);

        let lining_heavy_vec: Vec<usize> = lining_heavy_set.iter().copied().collect();
        let sasa_a2 = compute_sasa_fibonacci(&heavy_atoms, &lining_heavy_vec, config.probe_radius);

        let mut apolar_count = 0usize;
        let mut polar_count = 0usize;
        let mut residues = BTreeSet::new();
        let mut lining_atom_indices = Vec::with_capacity(lining_heavy_vec.len());

        for &h_idx in &lining_heavy_vec {
            let atom = heavy_atoms[h_idx];
            lining_atom_indices.push(heavy_orig_idx[h_idx]);
            residues.insert(ResidueTag {
                chain_id: atom.chain_id.to_string(),
                res_seq: atom.res_seq,
                res_name: atom.res_name.clone(),
            });
            if atom.hydrophobic {
                apolar_count += 1;
            }
            if atom.ad4_type.is_acceptor() || atom.ad4_type.is_donor_heavy() {
                polar_count += 1;
            }
        }

        let total_lining = lining_heavy_vec.len().max(1) as f64;
        let hydrophobicity = apolar_count as f64 / total_lining;
        let hbond_capacity = polar_count as f64 / total_lining;

        let vol_score = (volume_a3 / 450.0).clamp(0.0, 1.0);
        let polarity_balance = 1.0 - (hydrophobicity - 0.62).abs();
        let druggability_score = (0.38 * buriedness
            + 0.30 * vol_score
            + 0.22 * polarity_balance.clamp(0.0, 1.0)
            + 0.10 * hbond_capacity.min(0.45) / 0.45)
            .clamp(0.0, 1.0);

        let box_size = [
            ((max_p[0] - min_p[0]) + 2.0 * config.box_padding).clamp(16.0, 42.0),
            ((max_p[1] - min_p[1]) + 2.0 * config.box_padding).clamp(16.0, 42.0),
            ((max_p[2] - min_p[2]) + 2.0 * config.box_padding).clamp(16.0, 42.0),
        ];
        let grid_box = GridBox::from_center_size(center, box_size, config.box_spacing)?;

        pockets.push(BindingPocket {
            id: cid + 1,
            rank: 0,
            center,
            grid_box,
            volume_a3,
            sasa_a2,
            buriedness,
            hydrophobicity,
            hbond_capacity,
            druggability_score,
            probe_points: pts,
            lining_residues: residues.into_iter().collect(),
            lining_atom_indices,
        });
    }

    pockets.sort_by(|a, b| {
        let sa = a.druggability_score * a.volume_a3.cbrt();
        let sb = b.druggability_score * b.volume_a3.cbrt();
        sb.total_cmp(&sa)
    });
    for (rank, p) in pockets.iter_mut().enumerate() {
        p.rank = rank + 1;
    }

    Ok(pockets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use struct_core::{Ad4Type, Atom3D, Molecule3D, assign_ad4_types_and_flags};

    fn make_synthetic_pocket_receptor() -> Molecule3D {
        let mut rec = Molecule3D::new("synthetic_barrel");
        let mut id = 1usize;
        // Build a hollow C-shaped cavity centered near (0, 0, 0) with radius ~5.5 A, open on +Z.
        for x in [-7, -4, 0, 4, 7] {
            for y in [-7, -4, 0, 4, 7] {
                for z in [-7, -4, 0, 4] {
                    let xf = x as f64;
                    let yf = y as f64;
                    let zf = z as f64;
                    let r = (xf * xf + yf * yf + zf * zf).sqrt();
                    if r < 4.5 || r > 8.2 {
                        continue;
                    }
                    let elem = if (x == 4 && y == 0 && z == 0) || (x == -4 && y == 0 && z == -4) {
                        "N"
                    } else if x == 0 && y == 4 && z == 0 {
                        "O"
                    } else {
                        "C"
                    };
                    let mut atom = Atom3D::new(id, elem, [xf, yf, zf]);
                    atom.res_name = match elem {
                        "N" => "ASN".to_string(),
                        "O" => "ASP".to_string(),
                        _ => "LEU".to_string(),
                    };
                    atom.chain_id = 'A';
                    atom.res_seq = id as i32;
                    atom.ad4_type = match elem {
                        "N" => Ad4Type::NA,
                        "O" => Ad4Type::OA,
                        _ => Ad4Type::C,
                    };
                    rec.add_atom(atom);
                    id += 1;
                }
            }
        }
        assign_ad4_types_and_flags(&mut rec);
        rec
    }

    #[test]
    fn detects_synthetic_cavity_and_scores_druggability() -> Result<(), PocketError> {
        let receptor = make_synthetic_pocket_receptor();
        let pockets = detect_pockets(&receptor, &PocketDetectConfig::default())?;
        assert!(
            !pockets.is_empty(),
            "expected at least one binding pocket in synthetic cavity"
        );
        let top = &pockets[0];
        let dist_to_origin = dist2(top.center, [0.0, 0.0, 0.0]).sqrt();
        assert!(
            dist_to_origin < 3.0,
            "pocket center {:?} should be near origin (dist={dist_to_origin:.2})",
            top.center
        );
        assert!(top.volume_a3 > 40.0);
        assert!(top.sasa_a2 > 10.0);
        assert!(top.druggability_score > 0.25 && top.druggability_score <= 1.0);
        assert!(!top.lining_residues.is_empty());
        Ok(())
    }

    #[test]
    fn grid_box_from_ligand_wraps_coordinates_with_padding() -> Result<(), PocketError> {
        let mut lig = Molecule3D::new("lig");
        lig.add_atom(Atom3D::new(1, "C", [10.0, 2.0, -3.0]));
        lig.add_atom(Atom3D::new(2, "O", [14.0, 6.0, 1.0]));
        let gbox = GridBox::from_ligand(&lig, 5.0)?;
        assert!((gbox.center[0] - 12.0).abs() < 1e-6);
        assert!((gbox.center[1] - 4.0).abs() < 1e-6);
        assert!((gbox.center[2] - (-1.0)).abs() < 1e-6);
        assert!(gbox.contains([10.0, 2.0, -3.0]));
        assert!(gbox.contains([14.0, 6.0, 1.0]));
        assert_eq!(
            gbox.dimensions()[0],
            ((gbox.size[0] / 0.375).ceil() as usize) + 1
        );
        // Reject oversized 200^3 A box DoS immediately
        assert!(GridBox::from_center_size([0.0, 0.0, 0.0], [200.0, 200.0, 200.0], 0.375).is_err());
        Ok(())
    }
}
