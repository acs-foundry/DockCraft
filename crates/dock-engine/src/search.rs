#![forbid(unsafe_code)]

use std::collections::VecDeque;
use std::time::Instant;

use pocket_detect::GridBox;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use struct_core::{MAX_LIGAND_ATOMS, Molecule3D, metal_chelation_tier};

use crate::DockError;
use crate::constraints::{
    DockingConstraint, ResolvedConstraint, eval_constraints_and_grad, extract_flexible_sidechains,
    resolve_constraints,
};
use crate::grid::{AffinityGridSet, AtomChannelKey};
use crate::vina::{
    ScoreBreakdown, ScoringFunction, are_topologically_close, rotatable_penalty_divisor,
    score_complex_with_options, vina_vdw_radius,
};

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockingConfig {
    pub scoring: ScoringFunction,
    pub exhaustiveness: usize,
    pub num_modes: usize,
    pub ils_steps: usize,
    pub bfgs_iters: usize,
    pub min_rmsd_cluster: f64,
    pub seed: u64,
    /// Enable AutoDock4Zn-inspired catalytic metal coordination when receptor has Zn/Mg/Fe/Mn/Ca.
    #[serde(default = "default_true")]
    pub metal_coordination: bool,
    /// Optional pharmacophore spheres, template core tethers, or covalent warhead attachment.
    #[serde(default)]
    pub constraints: Vec<DockingConstraint>,
    /// Optional flexible receptor sidechain residue specs (e.g. `"A:ASP189"`, `"A:TRP79"`).
    #[serde(default)]
    pub flex_residues: Vec<String>,
}

impl Default for DockingConfig {
    fn default() -> Self {
        Self {
            scoring: ScoringFunction::Vina,
            exhaustiveness: 8,
            num_modes: 9,
            ils_steps: 24,
            bfgs_iters: 48,
            min_rmsd_cluster: 1.5,
            seed: 20_261_117,
            metal_coordination: true,
            constraints: Vec::new(),
            flex_residues: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoseTerms {
    pub breakdown: ScoreBreakdown,
    pub rmsd_to_best: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockingPose {
    pub mode: usize,
    pub affinity_kcal_mol: f64,
    pub molecule: Molecule3D,
    pub terms: PoseTerms,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockingReport {
    pub poses: Vec<DockingPose>,
    pub elapsed_ms: u128,
    pub total_evaluations: usize,
    /// Adapted receptor coordinates when `flex_residues` is non-empty.
    #[serde(default)]
    pub flex_receptor: Option<Molecule3D>,
}

#[derive(Debug, Clone)]
struct LigandTemplate {
    local_coords: Vec<[f64; 3]>,
    heavy_indices: Vec<usize>,
    heavy_channels: Vec<usize>,
    heavy_charges: Vec<f64>,
    heavy_tiers: Vec<u8>,
    intra_pairs: Vec<(usize, usize, f64)>, // (heavy_i, heavy_j, r_opt)
    torsions: Vec<(usize, usize, Vec<usize>)>, // (u, v, moving_atoms) sorted root-to-leaf
    rot_div: f64,
    constraints: Vec<ResolvedConstraint>,
}

#[derive(Debug, Clone)]
struct RigState {
    translation: [f64; 3],
    rot_mat: [[f64; 3]; 3],
    torsions: Vec<f64>,
}

fn mat3_identity() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn mat3_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

fn mat3_vec(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn rot_from_axis_angle(w: [f64; 3]) -> [[f64; 3]; 3] {
    let theta2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    if theta2 < 1e-18 {
        return [[1.0, -w[2], w[1]], [w[2], 1.0, -w[0]], [-w[1], w[0], 1.0]];
    }
    let theta = theta2.sqrt();
    let kx = w[0] / theta;
    let ky = w[1] / theta;
    let kz = w[2] / theta;
    let c = theta.cos();
    let s = theta.sin();
    let one_c = 1.0 - c;
    [
        [
            c + kx * kx * one_c,
            kx * ky * one_c - kz * s,
            kx * kz * one_c + ky * s,
        ],
        [
            ky * kx * one_c + kz * s,
            c + ky * ky * one_c,
            ky * kz * one_c - kx * s,
        ],
        [
            kz * kx * one_c - ky * s,
            kz * ky * one_c + kx * s,
            c + kz * kz * one_c,
        ],
    ]
}

#[inline]
fn rotate_point_around_axis(
    p: [f64; 3],
    origin: [f64; 3],
    axis_unit: [f64; 3],
    angle: f64,
) -> [f64; 3] {
    if angle.abs() < 1e-12 {
        return p;
    }
    let v = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
    let c = angle.cos();
    let s = angle.sin();
    let k_dot_v = axis_unit[0] * v[0] + axis_unit[1] * v[1] + axis_unit[2] * v[2];
    let k_cross_v = [
        axis_unit[1] * v[2] - axis_unit[2] * v[1],
        axis_unit[2] * v[0] - axis_unit[0] * v[2],
        axis_unit[0] * v[1] - axis_unit[1] * v[0],
    ];
    let one_c = 1.0 - c;
    [
        origin[0] + v[0] * c + k_cross_v[0] * s + axis_unit[0] * k_dot_v * one_c,
        origin[1] + v[1] * c + k_cross_v[1] * s + axis_unit[1] * k_dot_v * one_c,
        origin[2] + v[2] * c + k_cross_v[2] * s + axis_unit[2] * k_dot_v * one_c,
    ]
}

/// 24 rotational orientations spanning SO(3) uniformly (chiral octahedral group).
fn stratified_so3_rotations() -> Vec<[[f64; 3]; 3]> {
    let mut rots = Vec::with_capacity(24);
    let basis_axes: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for (p_idx, &ax) in basis_axes.iter().enumerate() {
        for &sgn in &[-1.0_f64, 1.0_f64] {
            let z = [ax[0] * sgn, ax[1] * sgn, ax[2] * sgn];
            let next_a = basis_axes[(p_idx + 1) % 3];
            let next_b = basis_axes[(p_idx + 2) % 3];
            let cross_z = [
                z[1] * next_a[2] - z[2] * next_a[1],
                z[2] * next_a[0] - z[0] * next_a[2],
                z[0] * next_a[1] - z[1] * next_a[0],
            ];
            let in_plane = [
                next_a,
                [-next_a[0], -next_a[1], -next_a[2]],
                if cross_z[0] + cross_z[1] + cross_z[2] != 0.0 {
                    next_b
                } else {
                    next_a
                },
                [-next_b[0], -next_b[1], -next_b[2]],
            ];
            for &x in &in_plane {
                let y = [
                    z[1] * x[2] - z[2] * x[1],
                    z[2] * x[0] - z[0] * x[2],
                    z[0] * x[1] - z[1] * x[0],
                ];
                let det = x[0] * (y[1] * z[2] - y[2] * z[1]) - x[1] * (y[0] * z[2] - y[2] * z[0])
                    + x[2] * (y[0] * z[1] - y[1] * z[0]);
                if (det - 1.0).abs() < 1e-4 {
                    rots.push([[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]);
                }
            }
        }
    }
    if rots.is_empty() {
        rots.push(mat3_identity());
    }
    rots
}

fn random_rotation(rng: &mut StdRng) -> [[f64; 3]; 3] {
    let u1: f64 = rng.r#gen::<f64>();
    let u2: f64 = rng.r#gen::<f64>() * std::f64::consts::TAU;
    let u3: f64 = rng.r#gen::<f64>() * std::f64::consts::TAU;
    let s1 = (1.0 - u1).sqrt();
    let s2 = u1.sqrt();
    let qx = s1 * u2.sin();
    let qy = s1 * u2.cos();
    let qz = s2 * u3.sin();
    let qw = s2 * u3.cos();
    [
        [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ],
        [
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ],
        [
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
    ]
}

impl LigandTemplate {
    #[allow(dead_code)]
    fn from_ligand(
        ligand: &Molecule3D,
        grids: &AffinityGridSet,
        sf: ScoringFunction,
    ) -> Option<Self> {
        Self::from_ligand_with_constraints(ligand, grids, sf, Vec::new())
    }

    fn from_ligand_with_constraints(
        ligand: &Molecule3D,
        grids: &AffinityGridSet,
        sf: ScoringFunction,
        constraints: Vec<ResolvedConstraint>,
    ) -> Option<Self> {
        let core_ids = ligand.rigid_core_indices();
        if core_ids.is_empty() {
            return None;
        }
        // Anchor local coordinates at stationary rigid core centroid so distal torsions never translate the core.
        let mut core_sum = [0.0_f64; 3];
        for &cid in &core_ids {
            let p = ligand.atoms[cid].coords;
            core_sum[0] += p[0];
            core_sum[1] += p[1];
            core_sum[2] += p[2];
        }
        let inv_n = 1.0 / (core_ids.len() as f64);
        let anchor_center = [
            core_sum[0] * inv_n,
            core_sum[1] * inv_n,
            core_sum[2] * inv_n,
        ];

        let local_coords: Vec<[f64; 3]> = ligand
            .atoms
            .iter()
            .map(|a| {
                [
                    a.coords[0] - anchor_center[0],
                    a.coords[1] - anchor_center[1],
                    a.coords[2] - anchor_center[2],
                ]
            })
            .collect();

        let mut heavy_indices = Vec::new();
        let mut heavy_channels = Vec::new();
        let mut heavy_charges = Vec::new();
        let mut heavy_tiers = Vec::new();
        let mut heavy_radii = Vec::new();
        for (i, a) in ligand.atoms.iter().enumerate() {
            if a.is_heavy() {
                let tier = if grids.metal_coordination {
                    metal_chelation_tier(ligand, i)
                } else {
                    0
                };
                let key = AtomChannelKey::from_mol_atom(ligand, i, grids.metal_coordination);
                let ch = grids.channel_index(key)?;
                heavy_indices.push(i);
                heavy_channels.push(ch);
                heavy_charges.push(a.charge);
                heavy_tiers.push(tier);
                heavy_radii.push(vina_vdw_radius(a.ad4_type, sf));
            }
        }
        if heavy_indices.is_empty() {
            return None;
        }

        let mut intra_pairs = Vec::new();
        for i in 0..heavy_indices.len() {
            let ui = heavy_indices[i];
            for j in (i + 1)..heavy_indices.len() {
                let uj = heavy_indices[j];
                if !are_topologically_close(ligand, ui, uj) {
                    intra_pairs.push((i, j, heavy_radii[i] + heavy_radii[j]));
                }
            }
        }

        // Preserve topological root-to-leaf order
        let torsions: Vec<(usize, usize, Vec<usize>)> = ligand
            .rotatable_bonds
            .iter()
            .map(|rb| (rb.u, rb.v, rb.moving_atoms.clone()))
            .collect();
        let rot_div = rotatable_penalty_divisor(torsions.len());

        Some(Self {
            local_coords,
            heavy_indices,
            heavy_channels,
            heavy_charges,
            heavy_tiers,
            intra_pairs,
            torsions,
            rot_div,
            constraints,
        })
    }

    fn apply_state(&self, state: &RigState, out_coords: &mut [[f64; 3]]) {
        out_coords.copy_from_slice(&self.local_coords);

        for (t_idx, (u, v, moving)) in self.torsions.iter().enumerate() {
            let angle = state.torsions[t_idx];
            if angle.abs() < 1e-10 {
                continue;
            }
            let p_u = out_coords[*u];
            let p_v = out_coords[*v];
            let dx = p_v[0] - p_u[0];
            let dy = p_v[1] - p_u[1];
            let dz = p_v[2] - p_u[2];
            let len = (dx * dx + dy * dy + dz * dz).sqrt();
            if len < 1e-8 {
                continue;
            }
            let axis = [dx / len, dy / len, dz / len];
            for &m in moving {
                out_coords[m] = rotate_point_around_axis(out_coords[m], p_u, axis, angle);
            }
        }

        for p in out_coords.iter_mut() {
            let r = mat3_vec(state.rot_mat, *p);
            *p = [
                r[0] + state.translation[0],
                r[1] + state.translation[1],
                r[2] + state.translation[2],
            ];
        }
    }

    fn eval_intra_energy(&self, coords: &[[f64; 3]]) -> f64 {
        let mut e_intra = 0.0_f64;
        for &(hi, hj, r_opt) in &self.intra_pairs {
            let ai = self.heavy_indices[hi];
            let aj = self.heavy_indices[hj];
            let dx = coords[ai][0] - coords[aj][0];
            let dy = coords[ai][1] - coords[aj][1];
            let dz = coords[ai][2] - coords[aj][2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < 1e-10 {
                e_intra += 50.0;
                continue;
            }
            let r = r2.sqrt();
            let d = r - r_opt;
            if d < -0.2 {
                let pen = d + 0.2;
                e_intra += 0.35 * 0.840_245 * pen * pen;
            }
        }
        e_intra
    }

    /// Evaluates `(energy, grad_trans, grad_omega, grad_torsions)` in the SO(3) local tangent around `state`.
    fn eval_and_grad(
        &self,
        grids: &AffinityGridSet,
        state: &RigState,
        delta_omega: [f64; 3],
        world_coords: &mut [[f64; 3]],
        atom_grads: &mut [[f64; 3]],
    ) -> (f64, [f64; 3], [f64; 3], Vec<f64>) {
        let rot_mat = if delta_omega[0] != 0.0 || delta_omega[1] != 0.0 || delta_omega[2] != 0.0 {
            mat3_mul(rot_from_axis_angle(delta_omega), state.rot_mat)
        } else {
            state.rot_mat
        };
        let eff = RigState {
            translation: state.translation,
            rot_mat,
            torsions: state.torsions.clone(),
        };
        self.apply_state(&eff, world_coords);
        atom_grads.fill([0.0; 3]);

        let mut e_inter = 0.0_f64;
        for (h_pos, &atom_idx) in self.heavy_indices.iter().enumerate() {
            let ch = self.heavy_channels[h_pos];
            let q = self.heavy_charges[h_pos];
            let (e, g) = grids.sample_atom(ch, q, world_coords[atom_idx]);
            e_inter += e;
            atom_grads[atom_idx][0] += g[0];
            atom_grads[atom_idx][1] += g[1];
            atom_grads[atom_idx][2] += g[2];
        }

        let mut e_intra = 0.0_f64;
        for &(hi, hj, r_opt) in &self.intra_pairs {
            let ai = self.heavy_indices[hi];
            let aj = self.heavy_indices[hj];
            let dx = world_coords[ai][0] - world_coords[aj][0];
            let dy = world_coords[ai][1] - world_coords[aj][1];
            let dz = world_coords[ai][2] - world_coords[aj][2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < 1e-10 {
                continue;
            }
            let r = r2.sqrt();
            let d = r - r_opt;
            if d < -0.2 {
                let pen = d + 0.2;
                let w = 0.35 * 0.840_245;
                e_intra += w * pen * pen;
                let df = (2.0 * w * pen) / r;
                atom_grads[ai][0] += df * dx;
                atom_grads[ai][1] += df * dy;
                atom_grads[ai][2] += df * dz;
                atom_grads[aj][0] -= df * dx;
                atom_grads[aj][1] -= df * dy;
                atom_grads[aj][2] -= df * dz;
            }
        }

        let e_cons = eval_constraints_and_grad(&self.constraints, world_coords, Some(atom_grads));

        let inv_div = 1.0 / self.rot_div;
        let energy = (e_inter + e_intra + e_cons) * inv_div;

        let mut g_trans = [0.0_f64; 3];
        let mut g_omega = [0.0_f64; 3];
        for &atom_idx in &self.heavy_indices {
            let g = atom_grads[atom_idx];
            g_trans[0] += g[0];
            g_trans[1] += g[1];
            g_trans[2] += g[2];
            let rx = world_coords[atom_idx][0] - eff.translation[0];
            let ry = world_coords[atom_idx][1] - eff.translation[1];
            let rz = world_coords[atom_idx][2] - eff.translation[2];
            g_omega[0] += ry * g[2] - rz * g[1];
            g_omega[1] += rz * g[0] - rx * g[2];
            g_omega[2] += rx * g[1] - ry * g[0];
        }
        for k in 0..3 {
            g_trans[k] *= inv_div;
            g_omega[k] *= inv_div;
        }

        let mut g_tors = vec![0.0_f64; self.torsions.len()];
        for (t_idx, (u, v, moving)) in self.torsions.iter().enumerate() {
            let p_u = world_coords[*u];
            let p_v = world_coords[*v];
            let dx = p_v[0] - p_u[0];
            let dy = p_v[1] - p_u[1];
            let dz = p_v[2] - p_u[2];
            let len = (dx * dx + dy * dy + dz * dz).sqrt();
            if len < 1e-8 {
                continue;
            }
            let ax = [dx / len, dy / len, dz / len];
            let mut sum_t = 0.0_f64;
            for &m in moving {
                let g = atom_grads[m];
                if g[0] == 0.0 && g[1] == 0.0 && g[2] == 0.0 {
                    continue;
                }
                let rx = world_coords[m][0] - p_u[0];
                let ry = world_coords[m][1] - p_u[1];
                let rz = world_coords[m][2] - p_u[2];
                let vel = [
                    ax[1] * rz - ax[2] * ry,
                    ax[2] * rx - ax[0] * rz,
                    ax[0] * ry - ax[1] * rx,
                ];
                sum_t += g[0] * vel[0] + g[1] * vel[1] + g[2] * vel[2];
            }
            g_tors[t_idx] = sum_t * inv_div;
        }

        (energy, g_trans, g_omega, g_tors)
    }
}

/// True Limited-Memory BFGS (m = 8 two-loop recursion) in SO(3) tangent + translation + torsion space.
fn local_minimize(
    tpl: &LigandTemplate,
    grids: &AffinityGridSet,
    state: &mut RigState,
    max_iters: usize,
    lock_torsions: bool,
    eval_counter: &mut usize,
) -> f64 {
    const M_HIST: usize = 8;
    let n_t = tpl.torsions.len();
    let n_dim = if lock_torsions { 6 } else { 6 + n_t };
    let mut world = vec![[0.0_f64; 3]; tpl.local_coords.len()];
    let mut atom_grads = vec![[0.0_f64; 3]; tpl.local_coords.len()];

    let pack_grad = |gt: [f64; 3], gw: [f64; 3], gtors: &[f64]| -> Vec<f64> {
        let mut g = Vec::with_capacity(n_dim);
        g.extend_from_slice(&gt);
        g.extend_from_slice(&gw);
        if !lock_torsions {
            g.extend_from_slice(gtors);
        }
        g
    };

    let (mut f_curr, gt, gw, gtors) =
        tpl.eval_and_grad(grids, state, [0.0; 3], &mut world, &mut atom_grads);
    *eval_counter += 1;
    let mut grad = pack_grad(gt, gw, &gtors);

    let mut s_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(M_HIST);
    let mut y_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(M_HIST);
    let mut rho_hist: VecDeque<f64> = VecDeque::with_capacity(M_HIST);

    for _ in 0..max_iters {
        let g_norm2: f64 = grad.iter().map(|v| v * v).sum();
        if g_norm2 < 1e-6 {
            break;
        }

        // Two-loop L-BFGS direction
        let mut q = grad.clone();
        let k_hist = s_hist.len();
        let mut alphas = vec![0.0_f64; k_hist];
        for i in (0..k_hist).rev() {
            let dot_sq: f64 = s_hist[i].iter().zip(q.iter()).map(|(a, b)| a * b).sum();
            let a_i = rho_hist[i] * dot_sq;
            alphas[i] = a_i;
            for (qk, &yk) in q.iter_mut().zip(y_hist[i].iter()) {
                *qk -= a_i * yk;
            }
        }
        let gamma = if let (Some(sl), Some(yl)) = (s_hist.back(), y_hist.back()) {
            let sy: f64 = sl.iter().zip(yl.iter()).map(|(a, b)| a * b).sum();
            let yy: f64 = yl.iter().map(|v| v * v).sum();
            if yy > 1e-12 && sy > 1e-12 {
                (sy / yy).clamp(0.02, 2.0)
            } else {
                0.35
            }
        } else {
            0.35
        };
        let mut z: Vec<f64> = q.iter().map(|&v| v * gamma).collect();
        for i in 0..k_hist {
            let dot_yz: f64 = y_hist[i].iter().zip(z.iter()).map(|(a, b)| a * b).sum();
            let b_i = rho_hist[i] * dot_yz;
            for (zk, &sk) in z.iter_mut().zip(s_hist[i].iter()) {
                *zk += (alphas[i] - b_i) * sk;
            }
        }

        let mut dir: Vec<f64> = z.into_iter().map(|v| -v).collect();
        let mut slope: f64 = grad.iter().zip(dir.iter()).map(|(a, b)| a * b).sum();
        if slope >= -1e-9 || !slope.is_finite() {
            // Reset L-BFGS history and fall back to steepest descent
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
            dir = grad.iter().map(|&g| -0.3 * g).collect();
            slope = grad.iter().zip(dir.iter()).map(|(a, b)| a * b).sum();
            if slope >= -1e-9 {
                break;
            }
        }

        // Guard maximum step norm so rotational/translational jumps stay within local trust region
        let max_trans = dir[0].abs().max(dir[1].abs()).max(dir[2].abs());
        if max_trans > 1.8 {
            let s = 1.8 / max_trans;
            for v in &mut dir {
                *v *= s;
            }
            slope *= s;
        }
        let max_rot = dir[3..].iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        if max_rot > 0.6 {
            let s = 0.6 / max_rot;
            for v in &mut dir {
                *v *= s;
            }
            slope *= s;
        }

        // Armijo backtracking
        let mut alpha = 1.0_f64;
        let mut accepted = false;
        for _ls in 0..10 {
            let mut cand = state.clone();
            cand.translation[0] += alpha * dir[0];
            cand.translation[1] += alpha * dir[1];
            cand.translation[2] += alpha * dir[2];
            let dw = [alpha * dir[3], alpha * dir[4], alpha * dir[5]];
            if !lock_torsions {
                for t in 0..n_t {
                    cand.torsions[t] += alpha * dir[6 + t];
                }
            }

            let (f_try, gt_try, gw_try, gtors_try) =
                tpl.eval_and_grad(grids, &cand, dw, &mut world, &mut atom_grads);
            *eval_counter += 1;

            if f_try <= f_curr + 1e-4 * alpha * slope {
                cand.rot_mat = mat3_mul(rot_from_axis_angle(dw), cand.rot_mat);
                let new_grad = pack_grad(gt_try, gw_try, &gtors_try);
                let step_vec: Vec<f64> = dir.iter().map(|&d| alpha * d).collect();
                let y_vec: Vec<f64> = new_grad
                    .iter()
                    .zip(grad.iter())
                    .map(|(ng, og)| ng - og)
                    .collect();
                let sy: f64 = step_vec.iter().zip(y_vec.iter()).map(|(a, b)| a * b).sum();
                if sy > 1e-8 {
                    if s_hist.len() == M_HIST {
                        s_hist.pop_front();
                        y_hist.pop_front();
                        rho_hist.pop_front();
                    }
                    s_hist.push_back(step_vec);
                    y_hist.push_back(y_vec);
                    rho_hist.push_back(1.0 / sy);
                }
                *state = cand;
                f_curr = f_try;
                grad = new_grad;
                accepted = true;
                break;
            }
            alpha *= 0.45;
        }

        if !accepted {
            break;
        }
    }

    f_curr
}

/// Smallest absolute difference between two angles, wrapped onto `[-PI, PI]`.
fn wrap_angle_diff(a: f64, b: f64) -> f64 {
    let mut d = (a - b) % std::f64::consts::TAU;
    if d > std::f64::consts::PI {
        d -= std::f64::consts::TAU;
    } else if d < -std::f64::consts::PI {
        d += std::f64::consts::TAU;
    }
    d.abs()
}

/// Root-mean-square torsion difference (radians) between two torsion vectors.
fn torsion_rms_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = wrap_angle_diff(x, y);
            d * d
        })
        .sum();
    (sum / a.len() as f64).sqrt()
}

/// Relaxes a torsion vector against the ligand's own steric term by cyclic coordinate descent,
/// so a randomly drawn conformer is strain-relieved before it enters the bank. Only the intra
/// term is used here: the receptor is not involved, exactly as in a standalone 3D conformer
/// generator.
fn relax_intra(
    tpl: &LigandTemplate,
    st: &mut RigState,
    coords: &mut [[f64; 3]],
    passes: usize,
) -> f64 {
    const TRIALS: [f64; 7] = [
        std::f64::consts::FRAC_PI_4,
        std::f64::consts::FRAC_PI_2,
        2.356_194_490_192_345,
        std::f64::consts::PI,
        -2.356_194_490_192_345,
        -std::f64::consts::FRAC_PI_2,
        -std::f64::consts::FRAC_PI_4,
    ];
    tpl.apply_state(st, coords);
    let mut best = tpl.eval_intra_energy(coords);
    for _ in 0..passes {
        let mut improved = false;
        for t in 0..st.torsions.len() {
            let orig = st.torsions[t];
            let mut best_angle = orig;
            let mut best_e = best;
            for &delta in &TRIALS {
                st.torsions[t] = orig + delta;
                tpl.apply_state(st, coords);
                let e = tpl.eval_intra_energy(coords);
                if e < best_e - 1e-9 {
                    best_e = e;
                    best_angle = st.torsions[t];
                }
            }
            st.torsions[t] = best_angle;
            if best_e < best - 1e-9 {
                best = best_e;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    tpl.apply_state(st, coords);
    best
}

/// Builds a conformer bank that actually spans the ligand's torsional space.
///
/// Entry 0 is the input conformation, but every other entry is drawn with **all** torsions
/// sampled independently over the full circle, strain-relaxed against the intra-ligand steric
/// term and then kept only if it is both clash-free and torsionally distinct from everything
/// already in the bank. Without this the search can only explore a neighbourhood of whatever
/// conformer it was handed, which makes re-docking succeed on crystal input and fail on the
/// SMILES-derived input the tool actually receives in a screening run.
fn generate_clash_free_conformers(tpl: &LigandTemplate, count: usize, seed: u64) -> Vec<Vec<f64>> {
    let n_t = tpl.torsions.len();
    let mut bank = vec![vec![0.0_f64; n_t]];
    if n_t == 0 || count <= 1 {
        return bank;
    }

    let mut st = RigState {
        translation: [0.0; 3],
        rot_mat: mat3_identity(),
        torsions: vec![0.0; n_t],
    };
    let mut coords = vec![[0.0; 3]; tpl.local_coords.len()];
    tpl.apply_state(&st, &mut coords);
    let baseline_intra = tpl.eval_intra_energy(&coords);
    let mut rng = StdRng::seed_from_u64(seed ^ 0xA5A5_5A5A);

    // Distinctness threshold in torsion space (~60 deg RMS), relaxed progressively if the space
    // is too crowded to fill the bank at the strict setting.
    let mut min_sep = 1.05_f64;
    let max_attempts = (count * 240).min(30_000);
    let relax_every = (max_attempts / 6).max(1);
    let mut attempts = 0usize;
    while bank.len() < count && attempts < max_attempts {
        attempts += 1;
        if attempts.is_multiple_of(relax_every) {
            min_sep *= 0.6;
        }
        for angle in st.torsions.iter_mut() {
            *angle = (rng.r#gen::<f64>() - 0.5) * std::f64::consts::TAU;
        }
        let e = relax_intra(tpl, &mut st, &mut coords, 3);
        if !e.is_finite() || e > baseline_intra + 2.5 {
            continue;
        }
        if bank
            .iter()
            .any(|b| torsion_rms_diff(b, &st.torsions) < min_sep)
        {
            continue;
        }
        bank.push(st.torsions.clone());
    }
    bank
}

fn heavy_rmsd(coords_a: &[[f64; 3]], coords_b: &[[f64; 3]], heavy_indices: &[usize]) -> f64 {
    if heavy_indices.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    for &idx in heavy_indices {
        let a = coords_a[idx];
        let b = coords_b[idx];
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        sum += dx * dx + dy * dy + dz * dz;
    }
    (sum / heavy_indices.len() as f64).sqrt()
}

pub fn dock_ligand(
    receptor: &Molecule3D,
    ligand: &Molecule3D,
    grid_box: &GridBox,
    config: &DockingConfig,
) -> Result<DockingReport, DockError> {
    if ligand.atoms.len() > MAX_LIGAND_ATOMS {
        return Err(DockError::TooManyLigandAtoms {
            count: ligand.atoms.len(),
            limit: MAX_LIGAND_ATOMS,
        });
    }
    let enable_metal =
        config.metal_coordination || config.scoring == ScoringFunction::DockCraftHybrid;
    let mut channels = Vec::with_capacity(ligand.atoms.len());
    for (i, a) in ligand.atoms.iter().enumerate() {
        if a.is_heavy() {
            channels.push(AtomChannelKey::from_mol_atom(ligand, i, enable_metal));
        }
    }
    let (_, mut excluded) = extract_flexible_sidechains(receptor, &config.flex_residues);
    let mut has_exclusions = excluded.iter().any(|&x| x);
    for c in &config.constraints {
        if let Some((cov_r, _)) = c.covalent_pair()
            && cov_r < excluded.len()
        {
            excluded[cov_r] = true;
            has_exclusions = true;
        }
    }
    let t_total = Instant::now();
    let grids = AffinityGridSet::build_with_options(
        receptor,
        grid_box,
        &channels,
        config.scoring,
        enable_metal,
        if has_exclusions {
            Some(excluded.as_slice())
        } else {
            None
        },
    )?;
    let mut report = dock_ligand_with_grids(receptor, ligand, &grids, config)?;
    // `dock_ligand_with_grids` times the search only, which is the right figure when a campaign
    // amortises one grid over a whole library. A single `dock_ligand` call pays for the grid too,
    // so the reported cost must include it.
    report.elapsed_ms = t_total.elapsed().as_millis();
    Ok(report)
}

pub fn dock_ligand_with_grids(
    receptor: &Molecule3D,
    ligand: &Molecule3D,
    grids: &AffinityGridSet,
    config: &DockingConfig,
) -> Result<DockingReport, DockError> {
    if ligand.atoms.len() > MAX_LIGAND_ATOMS {
        return Err(DockError::TooManyLigandAtoms {
            count: ligand.atoms.len(),
            limit: MAX_LIGAND_ATOMS,
        });
    }
    let start = Instant::now();
    let enable_metal = grids.metal_coordination
        || config.metal_coordination
        || config.scoring == ScoringFunction::DockCraftHybrid;
    let resolved_constraints = resolve_constraints(receptor, ligand, &config.constraints);
    let covalent_pair = config
        .constraints
        .iter()
        .find_map(DockingConstraint::covalent_pair);
    let (flex_sidechains, _) = extract_flexible_sidechains(receptor, &config.flex_residues);

    let tpl = LigandTemplate::from_ligand_with_constraints(
        ligand,
        grids,
        config.scoring,
        resolved_constraints,
    )
    .ok_or(DockError::EmptyStructure)?;

    let min_c = grids.grid_box.min_corner();
    let max_c = grids.grid_box.max_corner();
    let center = grids.grid_box.center;
    let n_t = tpl.torsions.len();

    // Stage 1: fast rigid+conformer coarse grid scan across spatially diverse anchors (up to 16,
    // plus priority constraint/covalent/metal targets) x 24 SO(3) x conformer bank.
    let mut anchors = grids.cavity_anchor_points(16);
    for rc in &tpl.constraints {
        match rc {
            ResolvedConstraint::Pharmacophore { center: c, .. } => anchors.insert(0, *c),
            ResolvedConstraint::Covalent { rec_coords, .. } => anchors.insert(0, *rec_coords),
            ResolvedConstraint::CoreTether { atom_targets, .. } if !atom_targets.is_empty() => {
                let mut s = [0.0; 3];
                for &(_, p) in atom_targets {
                    for k in 0..3 {
                        s[k] += p[k] / (atom_targets.len() as f64);
                    }
                }
                anchors.insert(0, s);
            }
            _ => {}
        }
    }
    if !anchors.iter().any(|p| {
        (p[0] - center[0]).powi(2) + (p[1] - center[1]).powi(2) + (p[2] - center[2]).powi(2) < 1.5
    }) {
        anchors.insert(0, center);
    }
    let rots = stratified_so3_rotations();
    // One bank entry per ~1/3 torsional degree of freedom, so a 13-rotatable-bond ligand gets a
    // genuinely multi-basin bank rather than a neighbourhood of its input conformer.
    let bank_size = (6 + 3 * n_t).clamp(6, 40);
    let conf_bank = generate_clash_free_conformers(&tpl, bank_size, config.seed);

    let mut raw_probes: Vec<(f64, usize, usize, RigState)> =
        Vec::with_capacity(anchors.len() * rots.len() * conf_bank.len());
    let mut coords_tmp = vec![[0.0; 3]; tpl.local_coords.len()];
    for (ai_idx, &tr) in anchors.iter().enumerate() {
        for &rm in &rots {
            for (ci, tors) in conf_bank.iter().enumerate() {
                let st = RigState {
                    translation: tr,
                    rot_mat: rm,
                    torsions: tors.clone(),
                };
                tpl.apply_state(&st, &mut coords_tmp);
                let mut e = 0.0_f64;
                for (hp, &ai) in tpl.heavy_indices.iter().enumerate() {
                    let (ae, _) = grids.sample_atom(
                        tpl.heavy_channels[hp],
                        tpl.heavy_charges[hp],
                        coords_tmp[ai],
                    );
                    e += ae;
                }
                let e_cons = eval_constraints_and_grad(&tpl.constraints, &coords_tmp, None);
                e = (e + tpl.eval_intra_energy(&coords_tmp) + e_cons) / tpl.rot_div;
                if e.is_finite() {
                    raw_probes.push((e, ai_idx, ci, st));
                }
            }
        }
    }
    raw_probes.sort_by(|a, b| a.0.total_cmp(&b.0));

    let n_walkers = config.exhaustiveness.max(1).max((6 + 2 * n_t).min(32));
    let n_stage1 = (n_walkers * 5).clamp(32, 80).max(2 * conf_bank.len());
    let mut rng_s1 = StdRng::seed_from_u64(config.seed ^ 0x5151_5151);
    // Stratify Stage-1 seeds across conformer bank members + top-3 orientations from each of
    // the top-4 deepest cavity/constraint anchors so tight buried pockets are never starved
    // while broad multi-subpocket clefts stay uniformly covered.
    let top_anchor_slots = anchors.len().min(4);
    let mut per_anchor: Vec<Vec<RigState>> = vec![Vec::new(); top_anchor_slots];
    let mut per_conf: Vec<Vec<RigState>> = vec![Vec::new(); conf_bank.len()];
    for (_, ai_idx, ci, st) in raw_probes {
        if ai_idx < top_anchor_slots && per_anchor[ai_idx].len() < 3 {
            per_anchor[ai_idx].push(st.clone());
        }
        if ci < per_conf.len() && per_conf[ci].len() < 4 {
            per_conf[ci].push(st);
        }
    }
    let mut stage1_seeds: Vec<RigState> = Vec::with_capacity(n_stage1);
    for round in 0..3 {
        for slots in &per_anchor {
            if stage1_seeds.len() >= n_stage1 / 4 {
                break;
            }
            if let Some(st) = slots.get(round) {
                stage1_seeds.push(st.clone());
            }
        }
    }
    for round in 0..4 {
        for slots in &per_conf {
            if stage1_seeds.len() >= (n_stage1 * 2) / 3 {
                break;
            }
            if let Some(st) = slots.get(round) {
                stage1_seeds.push(st.clone());
            }
        }
    }
    // Fill remaining Stage-1 slots uniformly across all cavity anchors so broad clefts are fully explored
    while stage1_seeds.len() < n_stage1 {
        let tr = anchors[rng_s1.gen_range(0..anchors.len())];
        let tors = conf_bank[rng_s1.gen_range(0..conf_bank.len())].clone();
        stage1_seeds.push(RigState {
            translation: [
                (tr[0] + (rng_s1.r#gen::<f64>() - 0.5) * 2.4).clamp(min_c[0] + 1.0, max_c[0] - 1.0),
                (tr[1] + (rng_s1.r#gen::<f64>() - 0.5) * 2.4).clamp(min_c[1] + 1.0, max_c[1] - 1.0),
                (tr[2] + (rng_s1.r#gen::<f64>() - 0.5) * 2.4).clamp(min_c[2] + 1.0, max_c[2] - 1.0),
            ],
            rot_mat: random_rotation(&mut rng_s1),
            torsions: tors,
        });
    }

    let stage1_eval: Vec<(f64, RigState, Vec<[f64; 3]>, usize)> = stage1_seeds
        .into_par_iter()
        .map(|mut st| {
            let mut evals = 0usize;
            // Quick rigid alignment then joint flexible descent
            let _ = local_minimize(&tpl, grids, &mut st, 16, true, &mut evals);
            let e = local_minimize(&tpl, grids, &mut st, 22, false, &mut evals);
            let mut cb = vec![[0.0; 3]; tpl.local_coords.len()];
            tpl.apply_state(&st, &mut cb);
            (e, st, cb, evals)
        })
        .collect();

    let mut stage1_evals = 0usize;
    let mut ranked_s1 = Vec::with_capacity(stage1_eval.len());
    for (e, st, cb, ev) in stage1_eval {
        stage1_evals += ev;
        if e.is_finite() {
            ranked_s1.push((e, st, cb));
        }
    }
    ranked_s1.sort_by(|a, b| a.0.total_cmp(&b.0));

    let basin_sep = 1.25 + 0.06 * n_t as f64;
    let mut basin_seeds: Vec<(f64, RigState)> = Vec::new();
    for (e, st, cb) in &ranked_s1 {
        let distinct = basin_seeds.iter().all(|(_, kept)| {
            let mut kcb = vec![[0.0; 3]; tpl.local_coords.len()];
            tpl.apply_state(kept, &mut kcb);
            heavy_rmsd(&kcb, cb, &tpl.heavy_indices) >= basin_sep
        });
        if distinct {
            basin_seeds.push((*e, st.clone()));
            if basin_seeds.len() >= n_walkers {
                break;
            }
        }
    }
    if basin_seeds.is_empty() {
        basin_seeds.push((
            0.0,
            RigState {
                translation: center,
                rot_mat: mat3_identity(),
                torsions: vec![0.0; n_t],
            },
        ));
    }

    // Stage 2: Parallel Vina-style Single-DOF Iterated Local Search starting from diverse Stage-1 basins.
    // Mutating ONE degree of freedom at a time (translation OR rotation OR a single torsion) lets
    // elongated multi-ring ligands (e.g. Imatinib 1IEP, Erlotinib 1M17, ESR1 modulators) thread
    // smoothly down narrow kinase/steroid channels without simultaneously slamming into pocket walls.
    let eff_ils = config.ils_steps.max(36 + 11 * n_t);
    let eff_bfgs = config.bfgs_iters.max(48 + 3 * n_t);

    type WalkerCand = (f64, RigState, Vec<[f64; 3]>);
    let walker_results: Vec<(Vec<WalkerCand>, usize)> = (0..n_walkers)
        .into_par_iter()
        .map(|w_idx| {
            let mut rng = StdRng::seed_from_u64(
                config.seed ^ (w_idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            let mut evals = 0usize;
            let (init_e, init_state) = &basin_seeds[w_idx % basin_seeds.len()];
            let mut curr_state = init_state.clone();
            let mut curr_e = *init_e;

            let mut candidates = Vec::new();
            let mut coords_buf = vec![[0.0; 3]; tpl.local_coords.len()];
            tpl.apply_state(&curr_state, &mut coords_buf);
            candidates.push((curr_e, curr_state.clone(), coords_buf.clone()));

            let temp_kt = 1.25_f64;
            let stall_limit = (12 + n_t).min(28);
            let mut since_improve = 0usize;
            let mut best_e = curr_e;
            for step in 0..eff_ils {
                if since_improve >= stall_limit {
                    since_improve = 0;
                    let anchor = anchors[rng.gen_range(0..anchors.len())];
                    let mut fresh = RigState {
                        translation: [
                            (anchor[0] + (rng.r#gen::<f64>() - 0.5) * 3.0)
                                .clamp(min_c[0] + 1.0, max_c[0] - 1.0),
                            (anchor[1] + (rng.r#gen::<f64>() - 0.5) * 3.0)
                                .clamp(min_c[1] + 1.0, max_c[1] - 1.0),
                            (anchor[2] + (rng.r#gen::<f64>() - 0.5) * 3.0)
                                .clamp(min_c[2] + 1.0, max_c[2] - 1.0),
                        ],
                        rot_mat: random_rotation(&mut rng),
                        torsions: conf_bank[rng.gen_range(0..conf_bank.len())].clone(),
                    };
                    let fe = local_minimize(&tpl, grids, &mut fresh, eff_bfgs, false, &mut evals);
                    tpl.apply_state(&fresh, &mut coords_buf);
                    candidates.push((fe, fresh.clone(), coords_buf.clone()));
                    curr_e = fe;
                    curr_state = fresh;
                    continue;
                }
                let mut mutated = curr_state.clone();
                let u_move: f64 = rng.r#gen();
                if step > 0 && step % 20 == 0 {
                    // Periodic pocket anchor jump without interrupting short multi-torsion descents
                    let anchor = anchors[rng.gen_range(0..anchors.len())];
                    mutated.translation = [
                        (anchor[0] + (rng.r#gen::<f64>() - 0.5) * 2.4)
                            .clamp(min_c[0] + 1.0, max_c[0] - 1.0),
                        (anchor[1] + (rng.r#gen::<f64>() - 0.5) * 2.4)
                            .clamp(min_c[1] + 1.0, max_c[1] - 1.0),
                        (anchor[2] + (rng.r#gen::<f64>() - 0.5) * 2.4)
                            .clamp(min_c[2] + 1.0, max_c[2] - 1.0),
                    ];
                    mutated.rot_mat = random_rotation(&mut rng);
                    if n_t > 0 {
                        mutated.torsions = conf_bank[rng.gen_range(0..conf_bank.len())].clone();
                    }
                } else if u_move < 0.72 {
                    // Official AutoDock Vina single-DOF move: mutate ONLY translation OR rotation OR ONE torsion
                    let which = rng.gen_range(0..(2 + n_t));
                    if which == 0 {
                        let amp = 2.0_f64;
                        for k in 0..3 {
                            mutated.translation[k] = (mutated.translation[k]
                                + (rng.r#gen::<f64>() - 0.5) * 2.0 * amp)
                                .clamp(min_c[k] + 1.0, max_c[k] - 1.0);
                        }
                    } else if which == 1 {
                        let rot_kick = [
                            (rng.r#gen::<f64>() - 0.5) * 1.6,
                            (rng.r#gen::<f64>() - 0.5) * 1.6,
                            (rng.r#gen::<f64>() - 0.5) * 1.6,
                        ];
                        mutated.rot_mat = mat3_mul(rot_from_axis_angle(rot_kick), mutated.rot_mat);
                    } else {
                        let tidx = which - 2;
                        if rng.r#gen::<f64>() < 0.5 {
                            mutated.torsions[tidx] =
                                (rng.r#gen::<f64>() - 0.5) * std::f64::consts::TAU;
                        } else {
                            mutated.torsions[tidx] +=
                                (rng.r#gen::<f64>() - 0.5) * std::f64::consts::PI;
                        }
                    }
                } else {
                    // Gentle coupled local move for concerted pocket settlement
                    for k in 0..3 {
                        mutated.translation[k] = (mutated.translation[k]
                            + (rng.r#gen::<f64>() - 0.5) * 1.4)
                            .clamp(min_c[k] + 1.0, max_c[k] - 1.0);
                    }
                    let rot_kick = [
                        (rng.r#gen::<f64>() - 0.5) * 0.65,
                        (rng.r#gen::<f64>() - 0.5) * 0.65,
                        (rng.r#gen::<f64>() - 0.5) * 0.65,
                    ];
                    mutated.rot_mat = mat3_mul(rot_from_axis_angle(rot_kick), mutated.rot_mat);
                    if n_t > 0 {
                        let tidx = rng.gen_range(0..n_t);
                        mutated.torsions[tidx] += (rng.r#gen::<f64>() - 0.5) * 1.2;
                    }
                }

                let mut_e = local_minimize(&tpl, grids, &mut mutated, eff_bfgs, false, &mut evals);
                tpl.apply_state(&mutated, &mut coords_buf);
                candidates.push((mut_e, mutated.clone(), coords_buf.clone()));

                if mut_e < best_e - 1e-6 {
                    best_e = mut_e;
                    since_improve = 0;
                } else {
                    since_improve += 1;
                }

                let de = mut_e - curr_e;
                if de <= 0.0 || rng.r#gen::<f64>() < (-de / temp_kt).exp() {
                    curr_e = mut_e;
                    curr_state = mutated;
                }
            }

            (candidates, evals)
        })
        .collect();

    let mut total_evals = stage1_evals;
    let mut all_candidates: Vec<(f64, RigState, Vec<[f64; 3]>)> = Vec::new();
    for (cands, ev) in walker_results {
        total_evals += ev;
        for (e, st, coords) in cands {
            if e.is_finite() {
                all_candidates.push((e, st, coords));
            }
        }
    }

    all_candidates.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Stage 3: RMSD clustering + deep L-BFGS polish + optional flexible sidechain rotamer optimization
    // + exact pairwise re-scoring + symmetry-aware deduplication.
    let mut cluster_states: Vec<(RigState, Vec<[f64; 3]>)> = Vec::new();
    for (_grid_e, mut st, mut coords) in all_candidates {
        let distinct = cluster_states.iter().all(|(_, kept)| {
            heavy_rmsd(kept, &coords, &tpl.heavy_indices) >= config.min_rmsd_cluster
        });
        if distinct {
            let mut polish_ev = 0usize;
            let _ = local_minimize(&tpl, grids, &mut st, 50, false, &mut polish_ev);
            total_evals += polish_ev;
            tpl.apply_state(&st, &mut coords);
            cluster_states.push((st, coords));
            if cluster_states.len() >= config.num_modes * 2 {
                break;
            }
        }
    }

    let mut exact_poses: Vec<(f64, f64, Molecule3D, ScoreBreakdown, Option<Molecule3D>)> =
        Vec::with_capacity(cluster_states.len());
    for (_, coords) in cluster_states {
        let mut mol = ligand.clone();
        for (atom, c) in mol.atoms.iter_mut().zip(coords.iter()) {
            atom.coords = *c;
        }
        let adapted_rec = if !flex_sidechains.is_empty() {
            let mut rec_copy = receptor.clone();
            for sc in &flex_sidechains {
                let best_angs = sc.optimize_rotamer(
                    &rec_copy,
                    &mol,
                    &tpl.heavy_indices,
                    &tpl.heavy_tiers,
                    &coords,
                    config.scoring,
                    enable_metal,
                );
                let moved = sc.posed_distal_atoms(&best_angs);
                for (li, moved_atom) in sc.rec_atom_indices.iter().skip(2).zip(moved.iter()) {
                    if let Some(target_atom) = rec_copy.atoms.get_mut(*li) {
                        target_atom.coords = moved_atom.coords;
                    }
                }
            }
            Some(rec_copy)
        } else {
            None
        };
        let eval_rec = adapted_rec.as_ref().unwrap_or(receptor);
        let bd =
            score_complex_with_options(eval_rec, &mol, config.scoring, enable_metal, covalent_pair);
        let e_cons = eval_constraints_and_grad(&tpl.constraints, &coords, None);
        // Rank by exact affinity + mild internal steric penalty + any constraint violation penalty
        let sort_key = bd.total_affinity_kcal + (0.35 * bd.intra_kcal + e_cons) / tpl.rot_div;
        exact_poses.push((sort_key, bd.total_affinity_kcal, mol, bd, adapted_rec));
    }

    exact_poses.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Final symmetry-aware deduplication so symmetric phenyl/carboxylate flips collapse cleanly
    let mut deduped: Vec<(f64, Molecule3D, ScoreBreakdown, Option<Molecule3D>)> =
        Vec::with_capacity(config.num_modes);
    for (_, aff, mol, bd, rec_opt) in exact_poses {
        let is_distinct = deduped.iter().all(|(_, kept, _, _)| {
            kept.rmsd_heavy_symmetric(&mol).unwrap_or(99.0) >= config.min_rmsd_cluster * 0.75
        });
        if is_distinct {
            deduped.push((aff, mol, bd, rec_opt));
            if deduped.len() >= config.num_modes {
                break;
            }
        }
    }

    let best_mol = deduped.first().map(|p| p.1.clone());
    let best_flex_rec = deduped.first().and_then(|p| p.3.clone());

    let poses = deduped
        .into_iter()
        .enumerate()
        .map(|(i, (aff, mut mol, bd, _))| {
            let rmsd_to_best = if let Some(ref best) = best_mol {
                best.rmsd_heavy_symmetric(&mol).unwrap_or(0.0)
            } else {
                0.0
            };
            mol.properties
                .insert("dockcraft_affinity_kcal".to_string(), format!("{aff:.3}"));
            DockingPose {
                mode: i + 1,
                affinity_kcal_mol: aff,
                molecule: mol,
                terms: PoseTerms {
                    breakdown: bd,
                    rmsd_to_best,
                },
            }
        })
        .collect();

    Ok(DockingReport {
        poses,
        elapsed_ms: start.elapsed().as_millis(),
        total_evaluations: total_evals,
        flex_receptor: best_flex_rec,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use struct_core::{
        Ad4Type, Atom3D, BondOrder, Molecule3D, assign_ad4_types_and_flags, build_rotatable_bonds,
        parse_smiles_to_3d,
    };

    #[test]
    fn analytical_gradients_match_finite_differences_with_nested_torsions()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut rec = Molecule3D::new("rec");
        let mut r1 = Atom3D::new(0, "N", [-3.2, 0.2, 0.1]);
        r1.ad4_type = Ad4Type::NA;
        r1.h_donor = true;
        rec.add_atom(r1);
        let mut r2 = Atom3D::new(1, "C", [2.8, 3.1, -0.5]);
        r2.ad4_type = Ad4Type::C;
        r2.hydrophobic = true;
        rec.add_atom(r2);

        let lig = parse_smiles_to_3d("c1ccccc1CCCCO", Some("Phenbutanol"))?;
        assert!(lig.rotatable_bonds.len() >= 3);
        let gbox = GridBox::from_center_size([0.0, 0.0, 0.0], [18.0, 18.0, 18.0], 0.375)?;
        let keys: Vec<AtomChannelKey> = lig
            .atoms
            .iter()
            .filter(|a| a.is_heavy())
            .map(AtomChannelKey::from_atom)
            .collect();
        let grids = AffinityGridSet::build(&rec, &gbox, &keys, ScoringFunction::Vina)?;
        let tpl = LigandTemplate::from_ligand(&lig, &grids, ScoringFunction::Vina)
            .ok_or(DockError::EmptyStructure)?;

        // Anchored-frame check.
        //
        // Asserting that the atoms of `rigid_core_indices()` do not move when a torsion changes
        // would be vacuous: that set is *defined* as the complement of every `moving_atoms` set,
        // and `apply_state` moves exactly those, so no choice of anchor could make it fail.
        //
        // The falsifiable property is the one the anchoring was introduced for: the world-frame
        // centroid of the rigid core must equal `state.translation` exactly, for every torsion
        // vector. Anchoring the local frame anywhere else (the whole-molecule centroid, say)
        // breaks this, and the requested docking position then no longer means what the search
        // thinks it means.
        let core = lig.rigid_core_indices();
        assert!(!core.is_empty());
        // Independent chemical check: the benzene ring is part of the rigid core by chemistry,
        // not by definition of the data structure.
        let aromatic: Vec<usize> = lig
            .atoms
            .iter()
            .enumerate()
            .filter(|(_, a)| a.aromatic && a.is_heavy())
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            aromatic.len(),
            6,
            "expected a benzene ring in the probe ligand"
        );
        for ai in &aromatic {
            assert!(
                core.contains(ai),
                "aromatic atom {ai} left out of the rigid core"
            );
        }

        let mut c0 = vec![[0.0; 3]; tpl.local_coords.len()];
        let translation = [0.2, -0.1, 0.3];
        for (case, torsions) in [
            ("zero", vec![0.0; tpl.torsions.len()]),
            (
                "mixed",
                (0..tpl.torsions.len())
                    .map(|i| 0.9 * (i as f64 + 1.0))
                    .collect::<Vec<f64>>(),
            ),
        ] {
            let st = RigState {
                translation,
                rot_mat: rot_from_axis_angle([0.3, -0.2, 0.45]),
                torsions,
            };
            tpl.apply_state(&st, &mut c0);
            let mut centroid = [0.0_f64; 3];
            for &cid in &core {
                for k in 0..3 {
                    centroid[k] += c0[cid][k];
                }
            }
            for k in 0..3 {
                centroid[k] /= core.len() as f64;
                assert!(
                    (centroid[k] - translation[k]).abs() < 1e-12,
                    "{case}: rigid-core centroid[{k}] = {} but the state asked for {}",
                    centroid[k],
                    translation[k]
                );
            }
        }

        // Compare analytical gradient against central finite differences (eps = 1e-4)
        let mut world = vec![[0.0; 3]; tpl.local_coords.len()];
        let mut agr = vec![[0.0; 3]; tpl.local_coords.len()];
        let state = RigState {
            translation: [0.35, -0.25, 0.15],
            rot_mat: rot_from_axis_angle([0.2, -0.15, 0.3]),
            torsions: (0..tpl.torsions.len())
                .map(|i| 0.25 * (i as f64 + 1.0))
                .collect(),
        };
        let (_, gt, gw, gtors) = tpl.eval_and_grad(&grids, &state, [0.0; 3], &mut world, &mut agr);
        let eps = 1e-4;
        for k in 0..3 {
            let mut sp = state.clone();
            let mut sm = state.clone();
            sp.translation[k] += eps;
            sm.translation[k] -= eps;
            let (ep, _, _, _) = tpl.eval_and_grad(&grids, &sp, [0.0; 3], &mut world, &mut agr);
            let (em, _, _, _) = tpl.eval_and_grad(&grids, &sm, [0.0; 3], &mut world, &mut agr);
            let fd = (ep - em) / (2.0 * eps);
            assert!(
                (fd - gt[k]).abs() < 1e-3,
                "trans grad[{k}] fd={fd} vs {0}",
                gt[k]
            );

            let mut dwp = [0.0; 3];
            let mut dwm = [0.0; 3];
            dwp[k] = eps;
            dwm[k] = -eps;
            let (epw, _, _, _) = tpl.eval_and_grad(&grids, &state, dwp, &mut world, &mut agr);
            let (emw, _, _, _) = tpl.eval_and_grad(&grids, &state, dwm, &mut world, &mut agr);
            let fdw = (epw - emw) / (2.0 * eps);
            assert!(
                (fdw - gw[k]).abs() < 1e-3,
                "rot grad[{k}] fd={fdw} vs {0}",
                gw[k]
            );
        }
        for t in 0..tpl.torsions.len() {
            let mut sp = state.clone();
            let mut sm = state.clone();
            sp.torsions[t] += eps;
            sm.torsions[t] -= eps;
            let (ep, _, _, _) = tpl.eval_and_grad(&grids, &sp, [0.0; 3], &mut world, &mut agr);
            let (em, _, _, _) = tpl.eval_and_grad(&grids, &sm, [0.0; 3], &mut world, &mut agr);
            let fdt = (ep - em) / (2.0 * eps);
            assert!(
                (fdt - gtors[t]).abs() < 1e-3,
                "tors grad[{t}] fd={fdt} vs {}",
                gtors[t]
            );
        }
        let _ = (
            Ad4Type::C,
            Atom3D::new(0, "C", [0.0; 3]),
            BondOrder::Single,
            assign_ad4_types_and_flags,
            build_rotatable_bonds,
        );
        Ok(())
    }

    #[test]
    fn redocks_complementary_ligand_from_9_angstrom_offset_under_2a_rmsd() -> Result<(), DockError>
    {
        let mut native = Molecule3D::new("native_probe");
        let a0 = native.add_atom({
            let mut a = Atom3D::new(1, "O", [-1.6, 0.0, 0.0]);
            a.charge = -0.55;
            a.ad4_type = Ad4Type::OA;
            a.h_acceptor = true;
            a.hydrophobic = false;
            a
        });
        let a1 = native.add_atom({
            let mut a = Atom3D::new(2, "C", [-0.2, 0.0, 0.0]);
            a.charge = 0.15;
            a.hydrophobic = false;
            a
        });
        let a2 = native.add_atom({
            let mut a = Atom3D::new(3, "C", [1.2, 0.4, 0.0]);
            a.charge = 0.0;
            a.hydrophobic = true;
            a
        });
        let a3 = native.add_atom({
            let mut a = Atom3D::new(4, "C", [2.4, -0.3, 0.4]);
            a.charge = 0.0;
            a.hydrophobic = true;
            a
        });
        let a4 = native.add_atom({
            let mut a = Atom3D::new(5, "N", [-1.0, 1.2, 0.0]);
            a.charge = 0.35;
            a.hydrophobic = false;
            a
        });
        let h4 = native.add_atom(Atom3D::new(6, "H", [-1.0, 2.2, 0.0]));
        native
            .add_bond(a0, a1, BondOrder::Single)
            .ok_or(DockError::EmptyStructure)?;
        native
            .add_bond(a1, a2, BondOrder::Single)
            .ok_or(DockError::EmptyStructure)?;
        native
            .add_bond(a2, a3, BondOrder::Single)
            .ok_or(DockError::EmptyStructure)?;
        native
            .add_bond(a1, a4, BondOrder::Single)
            .ok_or(DockError::EmptyStructure)?;
        native
            .add_bond(a4, h4, BondOrder::Single)
            .ok_or(DockError::EmptyStructure)?;
        assign_ad4_types_and_flags(&mut native);
        build_rotatable_bonds(&mut native);

        let mut receptor = Molecule3D::new("rec");
        let rec_specs = [
            ("N", [-4.45, 0.0, 0.0], 0.45, Ad4Type::N, true, false, false),
            (
                "O",
                [-1.0, 4.10, 0.0],
                -0.45,
                Ad4Type::OA,
                false,
                true,
                false,
            ),
            ("C", [1.5, 4.2, 0.0], 0.0, Ad4Type::C, false, false, true),
            ("C", [1.5, -3.5, 0.0], 0.0, Ad4Type::C, false, false, true),
            ("C", [2.5, 0.2, 4.2], 0.0, Ad4Type::C, false, false, true),
            ("C", [2.5, 0.2, -3.5], 0.0, Ad4Type::C, false, false, true),
            ("C", [6.2, -0.3, 0.4], 0.0, Ad4Type::C, false, false, true),
            ("C", [-2.5, -3.7, 0.0], 0.0, Ad4Type::C, false, false, true),
            ("C", [-1.0, 0.0, 3.8], 0.0, Ad4Type::C, false, false, true),
            ("C", [-1.0, 0.0, -3.8], 0.0, Ad4Type::C, false, false, true),
            ("C", [4.8, 2.8, 0.2], 0.0, Ad4Type::C, false, false, true),
            ("C", [4.8, -2.9, 0.4], 0.0, Ad4Type::C, false, false, true),
            ("C", [-3.8, 2.9, -2.1], 0.0, Ad4Type::C, false, false, true),
            ("C", [-3.8, -2.8, 2.1], 0.0, Ad4Type::C, false, false, true),
        ];
        for (i, &(el, pos, q, ad4, hd, ha, hyd)) in rec_specs.iter().enumerate() {
            let mut a = Atom3D::new(i + 1, el, pos);
            a.charge = q;
            a.ad4_type = ad4;
            a.h_donor = hd;
            a.h_acceptor = ha;
            a.hydrophobic = hyd;
            receptor.add_atom(a);
        }

        // Start 9.2 A away + rotated 120 deg!
        let mut scrambled = native.clone();
        for a in &mut scrambled.atoms {
            a.coords = [a.coords[2] + 6.5, -a.coords[0] - 5.2, a.coords[1] + 4.0];
        }

        let gbox = GridBox::from_center_size([1.5, 0.0, 0.0], [20.0, 18.0, 18.0], 0.375)?;
        let cfg = DockingConfig {
            exhaustiveness: 6,
            ils_steps: 24,
            bfgs_iters: 45,
            ..Default::default()
        };
        let report = dock_ligand(&receptor, &scrambled, &gbox, &cfg)?;
        let top = report.poses.first().ok_or(DockError::EmptyStructure)?;
        let rmsd = native
            .rmsd_heavy(&top.molecule)
            .ok_or(DockError::EmptyStructure)?;
        assert!(
            rmsd < 2.0,
            "expected global redocking from >9A offset RMSD < 2.0 A, got {rmsd:.3} A (score {:.3})",
            top.affinity_kcal_mol
        );
        Ok(())
    }
}
