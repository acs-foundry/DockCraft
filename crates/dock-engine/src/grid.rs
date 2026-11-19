#![forbid(unsafe_code)]

use std::collections::HashMap;

use pocket_detect::GridBox;
use rayon::prelude::*;
use struct_core::{Ad4Type, Atom3D, Molecule3D, metal_chelation_tier};

use crate::DockError;
use crate::vina::{ScoringFunction, pairwise_energy_and_deriv_full, smooth_down};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AtomChannelKey {
    pub ad4: Ad4Type,
    pub h_donor: bool,
    pub h_acceptor: bool,
    pub hydrophobic: bool,
    pub chelator_tier: u8,
}

impl AtomChannelKey {
    #[must_use]
    pub fn from_atom(atom: &Atom3D) -> Self {
        Self {
            ad4: atom.ad4_type,
            h_donor: atom.h_donor,
            h_acceptor: atom.h_acceptor,
            hydrophobic: atom.hydrophobic,
            chelator_tier: 0,
        }
    }

    #[must_use]
    pub fn from_mol_atom(mol: &Molecule3D, idx: usize, enable_metal: bool) -> Self {
        let atom = &mol.atoms[idx];
        Self {
            ad4: atom.ad4_type,
            h_donor: atom.h_donor,
            h_acceptor: atom.h_acceptor,
            hydrophobic: atom.hydrophobic,
            chelator_tier: if enable_metal && metal_chelation_tier(mol, idx) >= 2 {
                2
            } else {
                0
            },
        }
    }
}

pub struct AffinityGridSet {
    pub grid_box: GridBox,
    pub dims: [usize; 3],
    pub origin: [f64; 3],
    pub spacing: f64,
    pub scoring: ScoringFunction,
    pub metal_coordination: bool,
    channels: Vec<AtomChannelKey>,
    /// Steric/Vina non-electrostatic grid per channel.
    channel_grids: Vec<Vec<f32>>,
    /// Charge-normalized electrostatic potential grid (`E_elec / q_lig`).
    elec_grid: Vec<f32>,
    /// Catalytic metal coordinates inside or near the box (used to seed metalloenzyme anchors).
    metal_centers: Vec<[f64; 3]>,
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
        let enable_metal = scoring == ScoringFunction::DockCraftHybrid
            || channels_needed.iter().any(|c| c.chelator_tier > 0);
        Self::build_with_options(
            receptor,
            grid_box,
            channels_needed,
            scoring,
            enable_metal,
            None,
        )
    }

    /// Parallel receptor affinity grid builder supporting catalytic metal coordination and
    /// optional excluded receptor atom masks (for flexible sidechains or covalent attachment).
    pub fn build_with_options(
        receptor: &Molecule3D,
        grid_box: &GridBox,
        channels_needed: &[AtomChannelKey],
        scoring: ScoringFunction,
        metal_coordination: bool,
        excluded_rec_mask: Option<&[bool]>,
    ) -> Result<Self, DockError> {
        let rec_heavy: Vec<&Atom3D> = receptor
            .atoms
            .iter()
            .enumerate()
            .filter(|(idx, a)| {
                a.is_heavy()
                    && excluded_rec_mask
                        .and_then(|m| m.get(*idx).copied())
                        .is_none_or(|ex| !ex)
            })
            .map(|(_, a)| a)
            .collect();
        if rec_heavy.is_empty() {
            return Err(DockError::EmptyStructure);
        }

        let dims = grid_box.dimensions();
        let origin = grid_box.min_corner();
        let spacing = grid_box.spacing;
        let [nx, ny, nz] = dims;
        let n_vox = nx * ny * nz;
        let slice_len = ny * nz;

        let mut unique_channels = Vec::new();
        let mut lookup = HashMap::new();
        for &ch in channels_needed {
            let eff_ch = if metal_coordination {
                ch
            } else {
                AtomChannelKey {
                    chelator_tier: 0,
                    ..ch
                }
            };
            if let std::collections::hash_map::Entry::Vacant(e) = lookup.entry(eff_ch) {
                e.insert(unique_channels.len());
                unique_channels.push(eff_ch);
            }
        }
        let n_ch = unique_channels.len();

        let hash = RecSpatialHash::new(&rec_heavy, 4.0);
        let probe_templates: Vec<(Atom3D, u8)> = unique_channels
            .iter()
            .enumerate()
            .map(|(i, ch)| {
                let mut a = Atom3D::new(i + 1, "X", [0.0, 0.0, 0.0]);
                a.ad4_type = ch.ad4;
                a.h_donor = ch.h_donor;
                a.h_acceptor = ch.h_acceptor;
                a.hydrophobic = ch.hydrophobic;
                a.charge = 0.0;
                (a, ch.chelator_tier)
            })
            .collect();

        let need_elec = scoring == ScoringFunction::DockCraftHybrid;

        // Parallel X-slice computation via Rayon (~6-10x speedup on multi-core CPUs)
        let x_slices: Vec<(Vec<Vec<f32>>, Vec<f32>)> = (0..nx)
            .into_par_iter()
            .map(|ix| {
                let x = origin[0] + ix as f64 * spacing;
                let mut ch_bufs = vec![vec![0.0_f32; slice_len]; n_ch];
                let mut el_buf = if need_elec {
                    vec![0.0_f32; slice_len]
                } else {
                    Vec::new()
                };
                let mut local_probes = probe_templates.clone();

                for iy in 0..ny {
                    let y = origin[1] + iy as f64 * spacing;
                    let row = iy * nz;
                    for iz in 0..nz {
                        let z = origin[2] + iz as f64 * spacing;
                        let pos = [x, y, z];
                        let neighbors = hash.query_box_neighbors(&rec_heavy, pos, 8.0);
                        if neighbors.is_empty() {
                            continue;
                        }
                        let local_vox = row + iz;

                        if need_elec {
                            let mut phi = 0.0_f64;
                            for &r_idx in &neighbors {
                                let ra = rec_heavy[r_idx];
                                if ra.charge.abs() <= 1e-4 {
                                    continue;
                                }
                                let dx = pos[0] - ra.coords[0];
                                let dy = pos[1] - ra.coords[1];
                                let dz = pos[2] - ra.coords[2];
                                let r2 = dx * dx + dy * dy + dz * dz;
                                if r2 >= 64.0 {
                                    continue;
                                }
                                let r = r2.sqrt();
                                let r_eff = r.max(2.2);
                                let (env, _) = if r > 7.92 {
                                    smooth_down(r, 7.92, 8.0)
                                } else {
                                    (1.0, 0.0)
                                };
                                phi += 3.2 * ra.charge / (r_eff * r_eff) * env;
                            }
                            el_buf[local_vox] = phi as f32;
                        }

                        for (c_idx, (probe, tier)) in local_probes.iter_mut().enumerate() {
                            probe.coords = pos;
                            let mut sum_e = 0.0_f64;
                            for &r_idx in &neighbors {
                                let ra = rec_heavy[r_idx];
                                let dx = pos[0] - ra.coords[0];
                                let dy = pos[1] - ra.coords[1];
                                let dz = pos[2] - ra.coords[2];
                                let r2 = dx * dx + dy * dy + dz * dz;
                                if r2 < 64.0 {
                                    let r = r2.sqrt();
                                    let (e, _, _) = pairwise_energy_and_deriv_full(
                                        ra,
                                        probe,
                                        *tier,
                                        r,
                                        scoring,
                                        metal_coordination,
                                    );
                                    sum_e += e;
                                }
                            }
                            ch_bufs[c_idx][local_vox] = sum_e.min(250.0) as f32;
                        }
                    }
                }
                (ch_bufs, el_buf)
            })
            .collect();

        let mut channel_grids = vec![vec![0.0_f32; n_vox]; n_ch];
        let mut elec_grid = if need_elec {
            vec![0.0_f32; n_vox]
        } else {
            Vec::new()
        };
        for (ix, (ch_bufs, el_buf)) in x_slices.into_iter().enumerate() {
            let offset = ix * slice_len;
            for (c_idx, buf) in ch_bufs.into_iter().enumerate() {
                channel_grids[c_idx][offset..offset + slice_len].copy_from_slice(&buf);
            }
            if need_elec {
                elec_grid[offset..offset + slice_len].copy_from_slice(&el_buf);
            }
        }

        let metal_centers: Vec<[f64; 3]> = rec_heavy
            .iter()
            .filter(|a| a.ad4_type.is_metal() && grid_box.contains(a.coords))
            .map(|a| a.coords)
            .collect();

        Ok(Self {
            grid_box: grid_box.clone(),
            dims,
            origin,
            spacing,
            scoring,
            metal_coordination,
            channels: unique_channels,
            channel_grids,
            elec_grid,
            metal_centers,
            lookup,
        })
    }

    #[must_use]
    pub fn channel_index(&self, key: AtomChannelKey) -> Option<usize> {
        self.lookup.get(&key).copied().or_else(|| {
            self.lookup
                .get(&AtomChannelKey {
                    chelator_tier: 0,
                    ..key
                })
                .copied()
        })
    }

    #[must_use]
    pub fn metal_centers(&self) -> &[[f64; 3]] {
        &self.metal_centers
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

        let (e_q, g_q) = if self.scoring == ScoringFunction::DockCraftHybrid
            && !self.elec_grid.is_empty()
            && charge.abs() > 1e-4
        {
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

    /// Extracts sterically favorable, spatially diverse cavity points inside the GridBox.
    #[must_use]
    pub fn cavity_anchor_points(&self, max_points: usize) -> Vec<[f64; 3]> {
        let base_idx = self
            .channels
            .iter()
            .position(|k| matches!(k.ad4, Ad4Type::C | Ad4Type::A) && k.hydrophobic)
            .or_else(|| {
                self.channels
                    .iter()
                    .position(|k| matches!(k.ad4, Ad4Type::C | Ad4Type::A))
            })
            .unwrap_or(0);
        let Some(base_grid) = self.channel_grids.get(base_idx) else {
            return vec![self.grid_box.center];
        };
        let step = ((2.0 / self.spacing).round() as usize).max(2);
        let [nx, ny, nz] = self.dims;
        let mut scored: Vec<(f32, [f64; 3])> = Vec::new();
        for ix in (1..nx.saturating_sub(1)).step_by(step) {
            let x = self.origin[0] + ix as f64 * self.spacing;
            for iy in (1..ny.saturating_sub(1)).step_by(step) {
                let y = self.origin[1] + iy as f64 * self.spacing;
                for iz in (1..nz.saturating_sub(1)).step_by(step) {
                    let vox = (ix * ny + iy) * nz + iz;
                    let e = base_grid[vox];
                    if e < -0.01 {
                        let z = self.origin[2] + iz as f64 * self.spacing;
                        scored.push((e, [x, y, z]));
                    }
                }
            }
        }
        scored.sort_by(|a, b| a.0.total_cmp(&b.0));

        // Trim only ultra-shallow open-solvent corners (< 20% of deepest pocket well) when enough cavity voxels exist
        if let Some(&(best_e, _)) = scored.first() {
            let cut = (best_e * 0.20).min(-0.02);
            let n_above = scored.iter().filter(|&&(e, _)| e <= cut).count();
            if n_above >= 2 {
                scored.retain(|&(e, _)| e <= cut);
            }
        }

        let mut chosen = Vec::with_capacity(max_points.max(1));
        // Seed near any catalytic metal ion inside the pocket (offset 2.2-3.5 A into empty space)
        for &mc in &self.metal_centers {
            let mut best_near: Option<(f32, [f64; 3])> = None;
            for &(e, p) in &scored {
                let d2 = (p[0] - mc[0]).powi(2) + (p[1] - mc[1]).powi(2) + (p[2] - mc[2]).powi(2);
                if (2.0 * 2.0..=5.5 * 5.5).contains(&d2) && best_near.is_none_or(|(be, _)| e < be) {
                    best_near = Some((e, p));
                }
            }
            if let Some((_, p)) = best_near {
                chosen.push(p);
            }
        }
        for &(_, p) in &scored {
            let far = chosen.iter().all(|c: &[f64; 3]| {
                let dx = c[0] - p[0];
                let dy = c[1] - p[1];
                let dz = c[2] - p[2];
                dx * dx + dy * dy + dz * dz >= 2.0 * 2.0
            });
            if far {
                chosen.push(p);
                if chosen.len() >= max_points {
                    break;
                }
            }
        }
        if chosen.is_empty() {
            chosen.push(self.grid_box.center);
        }
        chosen
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
