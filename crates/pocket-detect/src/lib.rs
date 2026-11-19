#![forbid(unsafe_code)]

use std::collections::{BTreeSet, HashMap, VecDeque};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use struct_core::{Ad4Type, Atom3D, Molecule3D};
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
            grid_step: 1.0,
            probe_radius: 1.4,
            max_receptor_dist: 6.2,
            min_enclosed_rays: 15,
            min_pocket_points: 12,
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

/// 13 opposing integer lattice ray pairs (3 cardinal + 6 face-diagonals + 4 body-diagonals = 26 rays)
/// implementing LIGSITE-csc Protein-Solvent-Protein (PSP) burial events on the 3D occupancy grid.
const PSP_AXIS_PAIRS: [[i32; 3]; 13] = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, -1, 0],
    [1, 0, 1],
    [1, 0, -1],
    [0, 1, 1],
    [0, 1, -1],
    [1, 1, 1],
    [1, 1, -1],
    [1, -1, 1],
    [1, -1, -1],
];

fn is_catalytic_metal(a: &Atom3D) -> bool {
    matches!(
        a.ad4_type,
        Ad4Type::Zn | Ad4Type::Mg | Ad4Type::Fe | Ad4Type::Ca | Ad4Type::Mn
    ) || matches!(a.element.as_str(), "Zn" | "Mg" | "Fe" | "Ca" | "Mn" | "Cu")
}

#[inline]
fn ray_hits_solid(
    occ: &[u8],
    dims: [usize; 3],
    start: [i32; 3],
    dir: [i32; 3],
    max_steps: i32,
) -> bool {
    let [nx, ny, nz] = [dims[0] as i32, dims[1] as i32, dims[2] as i32];
    let mut x = start[0];
    let mut y = start[1];
    let mut z = start[2];
    for _ in 0..max_steps {
        x += dir[0];
        y += dir[1];
        z += dir[2];
        if x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz {
            return false;
        }
        let idx = ((x as usize) * dims[1] + (y as usize)) * dims[2] + (z as usize);
        if occ[idx] != 0 {
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
    let mut metal_coords: Vec<[f64; 3]> = Vec::new();
    for (idx, atom) in receptor.atoms.iter().enumerate() {
        if atom.is_heavy() {
            if is_catalytic_metal(atom) {
                metal_coords.push(atom.coords);
            }
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

    let step = config.grid_step.clamp(0.6, 2.0);
    // Keep grid bounded for very large multi-domain assemblies while staying at 1.0 A for normal receptors
    let span = [
        max_c[0] - min_c[0] + 4.0,
        max_c[1] - min_c[1] + 4.0,
        max_c[2] - min_c[2] + 4.0,
    ];
    let eff_step = {
        let raw_voxels = (span[0] / step) * (span[1] / step) * (span[2] / step);
        if raw_voxels > 1_800_000.0 {
            step * (raw_voxels / 1_800_000.0).cbrt()
        } else {
            step
        }
    };
    let origin = [min_c[0] - 2.0, min_c[1] - 2.0, min_c[2] - 2.0];
    let nx = ((span[0] / eff_step).ceil() as usize).max(3) + 1;
    let ny = ((span[1] / eff_step).ceil() as usize).max(3) + 1;
    let nz = ((span[2] / eff_step).ceil() as usize).max(3) + 1;
    let dims = [nx, ny, nz];
    let n_vox = nx * ny * nz;

    // 1. Build 3D protein occupancy grid (`occ = 1` solid protein, `near_prot = 1` within max_receptor_dist)
    let mut occ = vec![0u8; n_vox];
    let mut near_prot = vec![0u8; n_vox];
    let probe_pad = (config.probe_radius * 0.62).clamp(0.65, 1.15);
    let max_rec_d = config.max_receptor_dist.clamp(4.0, 8.5);
    let max_rec_d2 = max_rec_d * max_rec_d;

    for atom in &heavy_atoms {
        let solid_r = if is_catalytic_metal(atom) {
            1.65
        } else {
            atom.ad4_type.vina_radius() + probe_pad
        };
        let solid_r2 = solid_r * solid_r;
        let reach_r = max_rec_d.max(solid_r);
        let r_vox = (reach_r / eff_step).ceil() as isize;
        let cx = ((atom.coords[0] - origin[0]) / eff_step).round() as isize;
        let cy = ((atom.coords[1] - origin[1]) / eff_step).round() as isize;
        let cz = ((atom.coords[2] - origin[2]) / eff_step).round() as isize;

        let x0 = (cx - r_vox).max(0) as usize;
        let x1 = (cx + r_vox).min(nx as isize - 1) as usize;
        let y0 = (cy - r_vox).max(0) as usize;
        let y1 = (cy + r_vox).min(ny as isize - 1) as usize;
        let z0 = (cz - r_vox).max(0) as usize;
        let z1 = (cz + r_vox).min(nz as isize - 1) as usize;

        for ix in x0..=x1 {
            let dx = origin[0] + ix as f64 * eff_step - atom.coords[0];
            let dx2 = dx * dx;
            if dx2 > max_rec_d2 {
                continue;
            }
            for iy in y0..=y1 {
                let dy = origin[1] + iy as f64 * eff_step - atom.coords[1];
                let dxy2 = dx2 + dy * dy;
                if dxy2 > max_rec_d2 {
                    continue;
                }
                let row = (ix * ny + iy) * nz;
                for iz in z0..=z1 {
                    let dz = origin[2] + iz as f64 * eff_step - atom.coords[2];
                    let d2 = dxy2 + dz * dz;
                    if d2 <= max_rec_d2 {
                        let vidx = row + iz;
                        near_prot[vidx] = 1;
                        if d2 <= solid_r2 {
                            occ[vidx] = 1;
                        }
                    }
                }
            }
        }
    }

    // 2. Parallel 26-ray / 13-PSP burial evaluation across empty voxels adjacent to protein
    let ray_horizon_a = 13.5_f64;
    let max_steps_cardinal = (ray_horizon_a / eff_step).ceil() as i32;
    let max_steps_diag2 = (ray_horizon_a / (eff_step * std::f64::consts::SQRT_2)).ceil() as i32;
    let max_steps_diag3 = (ray_horizon_a / (eff_step * 3.0_f64.sqrt())).ceil() as i32;
    let min_rays = config.min_enclosed_rays.clamp(10, 24);
    let min_psp = 4usize;

    #[derive(Clone, Copy)]
    struct CavityVoxel {
        ijk: (i32, i32, i32),
        pos: [f64; 3],
        psp: u8,
        rays: u8,
        near_metal: bool,
    }

    let cavity_voxels: Vec<CavityVoxel> = (1..nx.saturating_sub(1))
        .into_par_iter()
        .flat_map_iter(|ix| {
            let mut slice_out = Vec::new();
            let x = origin[0] + ix as f64 * eff_step;
            for iy in 1..ny.saturating_sub(1) {
                let y = origin[1] + iy as f64 * eff_step;
                let row = (ix * ny + iy) * nz;
                for iz in 1..nz.saturating_sub(1) {
                    let vidx = row + iz;
                    if occ[vidx] != 0 || near_prot[vidx] == 0 {
                        continue;
                    }
                    let start = [ix as i32, iy as i32, iz as i32];
                    let mut psp = 0u8;
                    let mut rays = 0u8;
                    for &d in &PSP_AXIS_PAIRS {
                        let norm_sq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                        let steps = match norm_sq {
                            1 => max_steps_cardinal,
                            2 => max_steps_diag2,
                            _ => max_steps_diag3,
                        };
                        let h_pos = ray_hits_solid(&occ, dims, start, d, steps);
                        let h_neg = ray_hits_solid(&occ, dims, start, [-d[0], -d[1], -d[2]], steps);
                        if h_pos {
                            rays += 1;
                        }
                        if h_neg {
                            rays += 1;
                        }
                        if h_pos && h_neg {
                            psp += 1;
                        }
                    }
                    let z = origin[2] + iz as f64 * eff_step;
                    let pos = [x, y, z];
                    let near_metal = metal_coords.iter().any(|&m| dist2(pos, m) <= 5.5 * 5.5);

                    let passes = (usize::from(psp) >= min_psp && usize::from(rays) >= min_rays)
                        || (near_metal && psp >= 3 && rays >= 12);
                    if passes {
                        slice_out.push(CavityVoxel {
                            ijk: (ix as i32, iy as i32, iz as i32),
                            pos,
                            psp,
                            rays,
                            near_metal,
                        });
                    }
                }
            }
            slice_out
        })
        .collect();

    if cavity_voxels.is_empty() {
        return Ok(Vec::new());
    }

    // 3. Core-seeded cavity separation (prevents merging separate pockets through thin surface necks)
    let mut voxel_map: HashMap<(i32, i32, i32), usize> =
        HashMap::with_capacity(cavity_voxels.len() * 2);
    for (idx, v) in cavity_voxels.iter().enumerate() {
        voxel_map.insert(v.ijk, idx);
    }

    // Core voxels have strong multi-axis PSP burial (psp >= 6 and rays >= 18) or catalytic metal proximity
    let is_core: Vec<bool> = cavity_voxels
        .iter()
        .map(|v| (v.psp >= 6 && v.rays >= 18) || (v.near_metal && v.psp >= 4 && v.rays >= 14))
        .collect();

    let mut label = vec![usize::MAX; cavity_voxels.len()];
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    const FACE6: [(i32, i32, i32); 6] = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ];

    // Step 3a: 6-connected components of deep cores
    let mut core_visited = vec![false; cavity_voxels.len()];
    for start in 0..cavity_voxels.len() {
        if !is_core[start] || core_visited[start] {
            continue;
        }
        core_visited[start] = true;
        let mut q = VecDeque::from([start]);
        let mut comp = Vec::new();
        let mut has_metal = false;
        while let Some(cur) = q.pop_front() {
            comp.push(cur);
            has_metal |= cavity_voxels[cur].near_metal;
            let (cx, cy, cz) = cavity_voxels[cur].ijk;
            for &(dx, dy, dz) in &FACE6 {
                if let Some(&nbr) = voxel_map.get(&(cx + dx, cy + dy, cz + dz))
                    && is_core[nbr]
                    && !core_visited[nbr]
                {
                    core_visited[nbr] = true;
                    q.push_back(nbr);
                }
            }
        }
        let min_core_pts = if has_metal { 4 } else { 6 };
        if comp.len() >= min_core_pts {
            let cid = clusters.len();
            for &vidx in &comp {
                label[vidx] = cid;
            }
            clusters.push(comp);
        }
    }

    // Step 3b: Multi-source BFS growth from deep cores up to 3 shells (~3 A) into surrounding cavity voxels
    if !clusters.is_empty() {
        let mut grow_q: VecDeque<(usize, usize, u8)> = VecDeque::new();
        for (cid, comp) in clusters.iter().enumerate() {
            for &vidx in comp {
                grow_q.push_back((vidx, cid, 0));
            }
        }
        while let Some((cur, cid, shell)) = grow_q.pop_front() {
            if shell >= 2 {
                continue;
            }
            let (cx, cy, cz) = cavity_voxels[cur].ijk;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if dx == 0 && dy == 0 && dz == 0 {
                            continue;
                        }
                        if let Some(&nbr) = voxel_map.get(&(cx + dx, cy + dy, cz + dz))
                            && label[nbr] == usize::MAX
                        {
                            label[nbr] = cid;
                            clusters[cid].push(nbr);
                            grow_q.push_back((nbr, cid, shell + 1));
                        }
                    }
                }
            }
        }
    }

    // Step 3c: Any remaining unassigned cavity clusters (handles shallower open clefts / bromodomains)
    for start in 0..cavity_voxels.len() {
        if label[start] != usize::MAX {
            continue;
        }
        let cid = clusters.len();
        label[start] = cid;
        let mut q = VecDeque::from([start]);
        let mut comp = Vec::new();
        while let Some(cur) = q.pop_front() {
            comp.push(cur);
            let (cx, cy, cz) = cavity_voxels[cur].ijk;
            for &(dx, dy, dz) in &FACE6 {
                if let Some(&nbr) = voxel_map.get(&(cx + dx, cy + dy, cz + dz))
                    && label[nbr] == usize::MAX
                {
                    label[nbr] = cid;
                    q.push_back(nbr);
                }
            }
        }
        if comp.len() >= config.min_pocket_points {
            clusters.push(comp);
        }
    }

    // Filter clusters below min_pocket_points (keep metal-bearing sites down to 6 voxels)
    clusters.retain(|comp| {
        let has_metal = comp.iter().any(|&v| cavity_voxels[v].near_metal);
        comp.len()
            >= if has_metal {
                6
            } else {
                config.min_pocket_points
            }
    });

    let hash = SpatialHash::new(&heavy_atoms, 4.0);
    let voxel_vol = eff_step * eff_step * eff_step;
    let mut scored_pockets: Vec<(f64, BindingPocket)> = Vec::with_capacity(clusters.len());

    for (cid, cluster) in clusters.into_iter().enumerate() {
        let mut pts = Vec::with_capacity(cluster.len());
        let mut sum_rays = 0.0_f64;
        let mut psp_vals = Vec::with_capacity(cluster.len());
        let mut min_p = [f64::INFINITY; 3];
        let mut max_p = [f64::NEG_INFINITY; 3];
        let mut lining_heavy_set = BTreeSet::new();
        let mut has_metal = false;
        let mut core_voxel_count = 0usize;

        for &vidx in &cluster {
            let v = cavity_voxels[vidx];
            pts.push(v.pos);
            let psp_f = f64::from(v.psp);
            let rays_f = f64::from(v.rays);
            psp_vals.push(psp_f);
            sum_rays += rays_f;
            has_metal |= v.near_metal;
            if is_core[vidx] {
                core_voxel_count += 1;
            }
            for k in 0..3 {
                min_p[k] = min_p[k].min(v.pos[k]);
                max_p[k] = max_p[k].max(v.pos[k]);
            }
            for n_idx in hash.query_radius(&heavy_atoms, v.pos, 4.5) {
                lining_heavy_set.insert(n_idx);
            }
        }

        let n = pts.len() as f64;
        let volume_a3 = n * voxel_vol;

        psp_vals.sort_by(f64::total_cmp);
        let q65_idx = ((psp_vals.len() * 65) / 100).min(psp_vals.len().saturating_sub(1));
        let q85_idx = ((psp_vals.len() * 85) / 100).min(psp_vals.len().saturating_sub(1));
        let psp_cut = psp_vals[q65_idx];
        let core_depth = (psp_vals[q85_idx] / 13.0).clamp(0.0, 1.0);
        let mean_psp = (psp_vals.iter().sum::<f64>() / (n * 13.0)).clamp(0.0, 1.0);
        let mean_rays = (sum_rays / (n * 26.0)).clamp(0.0, 1.0);
        let buriedness = (0.55 * mean_rays + 0.45 * mean_psp).clamp(0.0, 1.0);

        // Anchor pocket center on the deepest top-35% buried core voxels (plus metal proximity)
        // so wide kinase/protease mouths never pull the centroid away from the catalytic core.
        let mut w_sum = 0.0_f64;
        let mut w_pos = [0.0_f64; 3];
        for &vidx in &cluster {
            let v = cavity_voxels[vidx];
            let psp_f = f64::from(v.psp);
            if psp_f < psp_cut && !v.near_metal {
                continue;
            }
            let w = (psp_f * psp_f * psp_f).max(1.0) * if v.near_metal { 4.0 } else { 1.0 };
            w_sum += w;
            for k in 0..3 {
                w_pos[k] += v.pos[k] * w;
            }
        }
        let center = if w_sum > 1e-6 {
            [w_pos[0] / w_sum, w_pos[1] / w_sum, w_pos[2] / w_sum]
        } else {
            let mut s = [0.0; 3];
            for p in &pts {
                for k in 0..3 {
                    s[k] += p[k];
                }
            }
            [s[0] / n, s[1] / n, s[2] / n]
        };

        let lining_heavy_vec: Vec<usize> = lining_heavy_set.iter().copied().collect();
        let sasa_a2 = compute_sasa_fibonacci(&heavy_atoms, &lining_heavy_vec, config.probe_radius);

        let mut apolar_count = 0usize;
        let mut polar_count = 0usize;
        let mut residues = BTreeSet::new();
        let mut lining_atom_indices = Vec::with_capacity(lining_heavy_vec.len());

        for &h_idx in &lining_heavy_vec {
            let atom = heavy_atoms[h_idx];
            lining_atom_indices.push(heavy_orig_idx[h_idx]);
            if is_catalytic_metal(atom) {
                has_metal = true;
            }
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

        // SiteMap / DoGSiteScorer-inspired druggability & active-site ranking:
        // Deeply buried large active sites (150-3000 A3) outrank small incidental packing voids;
        // only shallow surface grooves (buriedness < 0.58) are penalized for excess size.
        let vol_rise = ((volume_a3 - 40.0) / 260.0).clamp(0.0, 1.0);
        let shallow_pen = if buriedness < 0.58 && volume_a3 > 1600.0 {
            1.0 / (1.0 + (volume_a3 - 1600.0) / 1000.0)
        } else {
            1.0 / (1.0 + ((volume_a3 - 3800.0).max(0.0) / 2500.0))
        };
        let vol_drug = (vol_rise * shallow_pen).clamp(0.0, 1.0);

        // Protein active-site lining (including backbone N/C/O) typically shows 0.28-0.55 apolar fraction
        let polarity_balance = (1.0 - (hydrophobicity - 0.42).abs() * 1.6).clamp(0.0, 1.0);
        let hbond_term = (hbond_capacity / 0.35).clamp(0.0, 1.0);
        let metal_flag = if has_metal { 1.0 } else { 0.0 };

        let druggability_score = (0.30 * buriedness
            + 0.24 * core_depth
            + 0.22 * vol_drug
            + 0.14 * polarity_balance
            + 0.06 * hbond_term
            + 0.04 * metal_flag)
            .clamp(0.0, 1.0);

        // SiteMap SiteScore-style ranking: sqrt(effective core size) * burial^1.5 + druggability + metal
        let eff_core_vol = ((core_voxel_count.max(1) as f64) * voxel_vol).min(2400.0);
        let site_size = (eff_core_vol.sqrt() / 32.0).clamp(0.15, 1.55);
        let burial_weight = buriedness.powf(1.4) * (0.55 + 0.45 * core_depth);
        let rank_score = 0.55 * druggability_score
            + 0.95 * site_size * burial_weight
            + 0.12 * polarity_balance
            + if has_metal { 0.40 } else { 0.0 };

        let box_size = [
            ((max_p[0] - min_p[0]) + 2.0 * config.box_padding).clamp(16.0, 42.0),
            ((max_p[1] - min_p[1]) + 2.0 * config.box_padding).clamp(16.0, 42.0),
            ((max_p[2] - min_p[2]) + 2.0 * config.box_padding).clamp(16.0, 42.0),
        ];
        let grid_box = GridBox::from_center_size(center, box_size, config.box_spacing)?;

        scored_pockets.push((
            rank_score,
            BindingPocket {
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
            },
        ));
    }

    scored_pockets.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut pockets: Vec<BindingPocket> = scored_pockets.into_iter().map(|(_, p)| p).collect();
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
