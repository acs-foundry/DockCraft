#![forbid(unsafe_code)]

use std::collections::HashMap;

use pocket_detect::GridBox;
use struct_core::{Ad4Type, Atom3D, Molecule3D};

use crate::DockError;
use crate::vina::{ScoringFunction, pairwise_energy_and_deriv, smooth_down};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AtomChannelKey {
    pub ad4: Ad4Type,
    pub h_donor: bool,
    pub h_acceptor: bool,
}

impl AtomChannelKey {
    #[must_use]
    pub fn from_atom(atom: &Atom3D) -> Self {
        Self {
            ad4: atom.ad4_type,
            h_donor: atom.h_donor,
            h_acceptor: atom.h_acceptor,
        }
    }
}

pub struct AffinityGridSet {
    pub grid_box: GridBox,
    pub dims: [usize; 3],
    pub origin: [f64; 3],
    pub spacing: f64,
    pub scoring: ScoringFunction,
    channels: Vec<AtomChannelKey>,
    /// Steric/Vina non-electrostatic grid per channel.
    channel_grids: Vec<Vec<f32>>,
    /// Charge-normalized electrostatic potential grid (`E_elec / q_lig`).
    elec_grid: Vec<f32>,
    lookup: HashMap<AtomChannelKey, usize>,
}

struct RecSpatialHash {
    cell_size: f64,
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl RecSpatialHash {
    fn new(atoms: &[&Atom3D], cell_size: f64) -> Self {
        let mut cells: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for (i, a) in atoms.iter().enumerate() {
            cells
                .entry(Self::key(a.coords, cell_size))
                .or_default()
                .push(i);
        }
        Self { cell_size, cells }
    }

    #[inline]
    fn key(c: [f64; 3], s: f64) -> (i32, i32, i32) {
        (
            (c[0] / s).floor() as i32,
            (c[1] / s).floor() as i32,
            (c[2] / s).floor() as i32,
        )
    }

    fn query_box_neighbors(&self, atoms: &[&Atom3D], pos: [f64; 3], cutoff: f64) -> Vec<usize> {
        let span = (cutoff / self.cell_size).ceil() as i32 + 1;
        let (cx, cy, cz) = Self::key(pos, self.cell_size);
        let cut2 = cutoff * cutoff;
        let mut out = Vec::new();
        for dx in -span..=span {
            for dy in -span..=span {
                for dz in -span..=span {
                    if let Some(bucket) = self.cells.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &idx in bucket {
                            let c = atoms[idx].coords;
                            let ddx = c[0] - pos[0];
                            let ddy = c[1] - pos[1];
                            let ddz = c[2] - pos[2];
                            if ddx * ddx + ddy * ddy + ddz * ddz <= cut2 {
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

impl AffinityGridSet {
    pub fn build(
        receptor: &Molecule3D,
        grid_box: &GridBox,
        channels_needed: &[AtomChannelKey],
        scoring: ScoringFunction,
    ) -> Result<Self, DockError> {
        let rec_heavy: Vec<&Atom3D> = receptor.atoms.iter().filter(|a| a.is_heavy()).collect();
        if rec_heavy.is_empty() {
            return Err(DockError::EmptyStructure);
        }

        let dims = grid_box.dimensions();
        let origin = grid_box.min_corner();
        let spacing = grid_box.spacing;
        let n_vox = dims[0] * dims[1] * dims[2];

        let mut unique_channels = Vec::new();
        let mut lookup = HashMap::new();
        for &ch in channels_needed {
            if let std::collections::hash_map::Entry::Vacant(e) = lookup.entry(ch) {
                e.insert(unique_channels.len());
                unique_channels.push(ch);
            }
        }

        let mut channel_grids = vec![vec![0.0_f32; n_vox]; unique_channels.len()];
        let mut elec_grid = vec![0.0_f32; n_vox];

        let hash = RecSpatialHash::new(&rec_heavy, 4.0);
        let mut probe_atoms: Vec<Atom3D> = unique_channels
            .iter()
            .enumerate()
            .map(|(i, ch)| {
                let mut a = Atom3D::new(i + 1, "X", [0.0, 0.0, 0.0]);
                a.ad4_type = ch.ad4;
                a.h_donor = ch.h_donor;
                a.h_acceptor = ch.h_acceptor;
                a.charge = 0.0; // electrostatics stored separately in elec_grid
                a
            })
            .collect();

        // Precompute grids in z-major or x-major consistent stride: idx = (ix * ny + iy) * nz + iz
        for ix in 0..dims[0] {
            let x = origin[0] + ix as f64 * spacing;
            for iy in 0..dims[1] {
                let y = origin[1] + iy as f64 * spacing;
                for iz in 0..dims[2] {
                    let z = origin[2] + iz as f64 * spacing;
                    let pos = [x, y, z];
                    let vox = (ix * dims[1] + iy) * dims[2] + iz;
                    let neighbors = hash.query_box_neighbors(&rec_heavy, pos, 8.0);
                    if neighbors.is_empty() {
                        continue;
                    }

                    // Electrostatic unit probe potential
                    let mut phi = 0.0_f64;
                    for &r_idx in &neighbors {
                        let ra = rec_heavy[r_idx];
                        let dx = pos[0] - ra.coords[0];
                        let dy = pos[1] - ra.coords[1];
                        let dz = pos[2] - ra.coords[2];
                        let r2 = dx * dx + dy * dy + dz * dz;
                        if r2 >= 64.0 {
                            continue;
                        }
                        let r = r2.sqrt();
                        if ra.charge.abs() > 1e-4 {
                            let r_eff = r.max(2.2);
                            let (env, _) = if r > 7.2 {
                                smooth_down(r, 7.2, 8.0)
                            } else {
                                (1.0, 0.0)
                            };
                            phi += 3.2 * ra.charge / (r_eff * r_eff) * env;
                        }
                    }
                    elec_grid[vox] = phi as f32;

                    for (c_idx, probe) in probe_atoms.iter_mut().enumerate() {
                        probe.coords = pos;
                        let mut sum_e = 0.0_f64;
                        for &r_idx in &neighbors {
                            let ra = rec_heavy[r_idx];
                            let dx = pos[0] - ra.coords[0];
                            let dy = pos[1] - ra.coords[1];
                            let dz = pos[2] - ra.coords[2];
                            let r = (dx * dx + dy * dy + dz * dz).sqrt();
                            if r < 8.0 {
                                let (e, _, _) = pairwise_energy_and_deriv(ra, probe, r, scoring);
                                sum_e += e;
                            }
                        }
                        channel_grids[c_idx][vox] = sum_e.min(250.0) as f32;
                    }
                }
            }
        }

        Ok(Self {
            grid_box: grid_box.clone(),
            dims,
            origin,
            spacing,
            scoring,
            channels: unique_channels,
            channel_grids,
            elec_grid,
            lookup,
        })
    }

    #[must_use]
    pub fn channel_index(&self, key: AtomChannelKey) -> Option<usize> {
        self.lookup.get(&key).copied()
    }

    #[must_use]
    pub fn channels(&self) -> &[AtomChannelKey] {
        &self.channels
    }

    /// Trilinear interpolation + analytical spatial gradient of channel energy + electrostatic term.
    #[inline]
    pub fn sample_atom(&self, channel_idx: usize, charge: f64, pos: [f64; 3]) -> (f64, [f64; 3]) {
        let mut clamped = pos;
        let mut wall_e = 0.0_f64;
        let mut wall_g = [0.0_f64; 3];
        let max_c = [
            self.origin[0] + (self.dims[0] - 1) as f64 * self.spacing,
            self.origin[1] + (self.dims[1] - 1) as f64 * self.spacing,
            self.origin[2] + (self.dims[2] - 1) as f64 * self.spacing,
        ];
        for k in 0..3 {
            if pos[k] < self.origin[k] {
                let d = self.origin[k] - pos[k];
                wall_e += 8.0 * d * d;
                wall_g[k] = -16.0 * d;
                clamped[k] = self.origin[k];
            } else if pos[k] > max_c[k] {
                let d = pos[k] - max_c[k];
                wall_e += 8.0 * d * d;
                wall_g[k] = 16.0 * d;
                clamped[k] = max_c[k];
            }
        }

        let inv_h = 1.0 / self.spacing;
        let fx =
            ((clamped[0] - self.origin[0]) * inv_h).clamp(0.0, (self.dims[0] - 1) as f64 - 1e-6);
        let fy =
            ((clamped[1] - self.origin[1]) * inv_h).clamp(0.0, (self.dims[1] - 1) as f64 - 1e-6);
        let fz =
            ((clamped[2] - self.origin[2]) * inv_h).clamp(0.0, (self.dims[2] - 1) as f64 - 1e-6);

        let ix = fx.floor() as usize;
        let iy = fy.floor() as usize;
        let iz = fz.floor() as usize;
        let tx = fx - ix as f64;
        let ty = fy - iy as f64;
        let tz = fz - iz as f64;

        let ny = self.dims[1];
        let nz = self.dims[2];
        let idx = |a: usize, b: usize, c: usize| -> usize { (a * ny + b) * nz + c };

        let i000 = idx(ix, iy, iz);
        let i100 = idx(ix + 1, iy, iz);
        let i010 = idx(ix, iy + 1, iz);
        let i110 = idx(ix + 1, iy + 1, iz);
        let i001 = idx(ix, iy, iz + 1);
        let i101 = idx(ix + 1, iy, iz + 1);
        let i011 = idx(ix, iy + 1, iz + 1);
        let i111 = idx(ix + 1, iy + 1, iz + 1);

        let (e_c, g_c) = trilinear(
            &self.channel_grids[channel_idx],
            [i000, i100, i010, i110, i001, i101, i011, i111],
            tx,
            ty,
            tz,
            inv_h,
        );

        let (e_q, g_q) = if charge.abs() > 1e-4 {
            let (phi, dphi) = trilinear(
                &self.elec_grid,
                [i000, i100, i010, i110, i001, i101, i011, i111],
                tx,
                ty,
                tz,
                inv_h,
            );
            (
                charge * phi,
                [charge * dphi[0], charge * dphi[1], charge * dphi[2]],
            )
        } else {
            (0.0, [0.0; 3])
        };

        (
            e_c + e_q + wall_e,
            [
                g_c[0] + g_q[0] + wall_g[0],
                g_c[1] + g_q[1] + wall_g[1],
                g_c[2] + g_q[2] + wall_g[2],
            ],
        )
    }
}

#[inline]
fn trilinear(
    grid: &[f32],
    c: [usize; 8],
    tx: f64,
    ty: f64,
    tz: f64,
    inv_h: f64,
) -> (f64, [f64; 3]) {
    let v000 = grid[c[0]] as f64;
    let v100 = grid[c[1]] as f64;
    let v010 = grid[c[2]] as f64;
    let v110 = grid[c[3]] as f64;
    let v001 = grid[c[4]] as f64;
    let v101 = grid[c[5]] as f64;
    let v011 = grid[c[6]] as f64;
    let v111 = grid[c[7]] as f64;

    let ux = 1.0 - tx;
    let uy = 1.0 - ty;
    let uz = 1.0 - tz;

    let e = v000 * ux * uy * uz
        + v100 * tx * uy * uz
        + v010 * ux * ty * uz
        + v110 * tx * ty * uz
        + v001 * ux * uy * tz
        + v101 * tx * uy * tz
        + v011 * ux * ty * tz
        + v111 * tx * ty * tz;

    let gx = ((v100 - v000) * uy * uz
        + (v110 - v010) * ty * uz
        + (v101 - v001) * uy * tz
        + (v111 - v011) * ty * tz)
        * inv_h;
    let gy = ((v010 - v000) * ux * uz
        + (v110 - v100) * tx * uz
        + (v011 - v001) * ux * tz
        + (v111 - v101) * tx * tz)
        * inv_h;
    let gz = ((v001 - v000) * ux * uy
        + (v101 - v100) * tx * uy
        + (v011 - v010) * ux * ty
        + (v111 - v110) * tx * ty)
        * inv_h;

    (e, [gx, gy, gz])
}
