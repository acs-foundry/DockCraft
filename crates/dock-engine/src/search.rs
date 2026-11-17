#![forbid(unsafe_code)]

use std::collections::VecDeque;
use std::time::Instant;

use pocket_detect::GridBox;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use struct_core::{MAX_LIGAND_ATOMS, Molecule3D};

use crate::DockError;
use crate::grid::{AffinityGridSet, AtomChannelKey};
use crate::vina::{
    ScoreBreakdown, ScoringFunction, are_topologically_close, rotatable_penalty_divisor,
    score_complex, vina_vdw_radius,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockingConfig {
    pub scoring: ScoringFunction,
    pub exhaustiveness: usize,
    pub num_modes: usize,
    pub ils_steps: usize,
    pub bfgs_iters: usize,
    pub min_rmsd_cluster: f64,
    pub seed: u64,
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
}

#[derive(Debug, Clone)]
struct LigandTemplate {
    local_coords: Vec<[f64; 3]>,
    heavy_indices: Vec<usize>,
    heavy_channels: Vec<usize>,
    heavy_charges: Vec<f64>,
    intra_pairs: Vec<(usize, usize, f64)>, // (heavy_i, heavy_j, r_opt)
    torsions: Vec<(usize, usize, Vec<usize>)>, // (u, v, moving_atoms) sorted root-to-leaf
    rot_div: f64,
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
    fn from_ligand(
        ligand: &Molecule3D,
        grids: &AffinityGridSet,
        sf: ScoringFunction,
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
        let mut heavy_radii = Vec::new();
        for (i, a) in ligand.atoms.iter().enumerate() {
            if a.is_heavy() {
                let ch = grids.channel_index(AtomChannelKey::from_atom(a))?;
                heavy_indices.push(i);
                heavy_channels.push(ch);
                heavy_charges.push(a.charge);
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
            intra_pairs,
            torsions,
            rot_div,
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

        let inv_div = 1.0 / self.rot_div;
        let energy = (e_inter + e_intra) * inv_div;

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

    for _attempt in 0..(count * 12) {
        if bank.len() >= count {
            break;
        }
        let mut tors = vec![0.0_f64; n_t];
        // Perturb subset of torsions incrementally while checking for intra clash
        let n_mut = rng.gen_range(1..=n_t.min(4));
        for _ in 0..n_mut {
            let idx = rng.gen_range(0..n_t);
            tors[idx] = (rng.r#gen::<f64>() - 0.5) * std::f64::consts::TAU;
        }
        st.torsions = tors.clone();
        tpl.apply_state(&st, &mut coords);
        if tpl.eval_intra_energy(&coords) <= baseline_intra + 2.5 {
            bank.push(tors);
        }
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
    let mut channels = Vec::with_capacity(ligand.atoms.len());
    for a in &ligand.atoms {
        if a.is_heavy() {
            channels.push(AtomChannelKey::from_atom(a));
        }
    }
    let grids = AffinityGridSet::build(receptor, grid_box, &channels, config.scoring)?;
    dock_ligand_with_grids(receptor, ligand, &grids, config)
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
    let tpl = LigandTemplate::from_ligand(ligand, grids, config.scoring)
        .ok_or(DockError::EmptyStructure)?;

    let min_c = grids.grid_box.min_corner();
    let max_c = grids.grid_box.max_corner();
    let center = grids.grid_box.center;
    let n_t = tpl.torsions.len();

    // Stage 1: fast rigid+conformer coarse grid scan across ALL (anchor x 24 SO(3) x conformer) combos,
    // then L-BFGS minimize top diverse candidates in parallel.
    let mut anchors = grids.cavity_anchor_points(8);
    if !anchors.iter().any(|p| {
        (p[0] - center[0]).powi(2) + (p[1] - center[1]).powi(2) + (p[2] - center[2]).powi(2) < 1.5
    }) {
        anchors.insert(0, center);
    }
    let rots = stratified_so3_rotations();
    let conf_bank = generate_clash_free_conformers(&tpl, (4 + n_t).min(8), config.seed);

    let mut raw_probes: Vec<(f64, RigState)> =
        Vec::with_capacity(anchors.len() * rots.len() * conf_bank.len().min(4));
    let mut coords_tmp = vec![[0.0; 3]; tpl.local_coords.len()];
    for &tr in &anchors {
        for &rm in &rots {
            for tors in conf_bank.iter().take(4) {
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
                e = (e + tpl.eval_intra_energy(&coords_tmp)) / tpl.rot_div;
                if e.is_finite() {
                    raw_probes.push((e, st));
                }
            }
        }
    }
    raw_probes.sort_by(|a, b| a.0.total_cmp(&b.0));

    let n_walkers = config.exhaustiveness.max(1);
    let n_stage1 = (n_walkers * 5).clamp(24, 64);
    let mut rng_s1 = StdRng::seed_from_u64(config.seed ^ 0x5151_5151);
    let mut stage1_seeds: Vec<RigState> = raw_probes
        .into_iter()
        .take(n_stage1 / 2)
        .map(|(_, st)| st)
        .collect();
    // Fill remaining Stage-1 slots with random pocket explorations so novel basins are always sampled
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

    let mut basin_seeds: Vec<(f64, RigState)> = Vec::new();
    for (e, st, cb) in &ranked_s1 {
        let distinct = basin_seeds.iter().all(|(_, kept)| {
            let mut kcb = vec![[0.0; 3]; tpl.local_coords.len()];
            tpl.apply_state(kept, &mut kcb);
            heavy_rmsd(&kcb, cb, &tpl.heavy_indices) >= 1.25
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

    // Stage 2: Parallel Iterated Local Search starting from diverse Stage-1 basins + pocket exploration
    let eff_ils = config.ils_steps.max(18 + n_t);
    let eff_bfgs = config.bfgs_iters.max(45 + 3 * n_t);

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

            let temp_kt = 1.4_f64;
            for step in 0..eff_ils {
                let mut mutated = curr_state.clone();
                if step % 5 == 0 {
                    // Occasional pocket anchor jump / large reorientation
                    let anchor = anchors[rng.gen_range(0..anchors.len())];
                    mutated.translation = [
                        (anchor[0] + (rng.r#gen::<f64>() - 0.5) * 2.2)
                            .clamp(min_c[0] + 1.0, max_c[0] - 1.0),
                        (anchor[1] + (rng.r#gen::<f64>() - 0.5) * 2.2)
                            .clamp(min_c[1] + 1.0, max_c[1] - 1.0),
                        (anchor[2] + (rng.r#gen::<f64>() - 0.5) * 2.2)
                            .clamp(min_c[2] + 1.0, max_c[2] - 1.0),
                    ];
                    mutated.rot_mat = random_rotation(&mut rng);
                    if n_t > 0 {
                        mutated.torsions = conf_bank[rng.gen_range(0..conf_bank.len())].clone();
                    }
                } else {
                    let amp = if step % 3 == 0 { 1.8 } else { 0.9 };
                    for k in 0..3 {
                        mutated.translation[k] = (mutated.translation[k]
                            + (rng.r#gen::<f64>() - 0.5) * 2.0 * amp)
                            .clamp(min_c[k] + 1.0, max_c[k] - 1.0);
                    }
                    let rot_kick = [
                        (rng.r#gen::<f64>() - 0.5) * 1.1,
                        (rng.r#gen::<f64>() - 0.5) * 1.1,
                        (rng.r#gen::<f64>() - 0.5) * 1.1,
                    ];
                    mutated.rot_mat = mat3_mul(rot_from_axis_angle(rot_kick), mutated.rot_mat);
                    if n_t > 0 {
                        let k_tors = rng.gen_range(1..=n_t.min(3));
                        for _ in 0..k_tors {
                            let tidx = rng.gen_range(0..n_t);
                            mutated.torsions[tidx] +=
                                (rng.r#gen::<f64>() - 0.5) * std::f64::consts::PI;
                        }
                    }
                }

                let mut_e = local_minimize(&tpl, grids, &mut mutated, eff_bfgs, false, &mut evals);
                tpl.apply_state(&mutated, &mut coords_buf);
                candidates.push((mut_e, mutated.clone(), coords_buf.clone()));

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

    // RMSD clustering + deep Stage-3 L-BFGS polish + exact pairwise re-scoring
    let mut cluster_states: Vec<(RigState, Vec<[f64; 3]>)> = Vec::new();
    for (_grid_e, mut st, mut coords) in all_candidates {
        let distinct = cluster_states.iter().all(|(_, kept)| {
            heavy_rmsd(kept, &coords, &tpl.heavy_indices) >= config.min_rmsd_cluster
        });
        if distinct {
            let mut polish_ev = 0usize;
            let _ = local_minimize(&tpl, grids, &mut st, 45, false, &mut polish_ev);
            total_evals += polish_ev;
            tpl.apply_state(&st, &mut coords);
            cluster_states.push((st, coords));
            if cluster_states.len() >= config.num_modes * 2 {
                break;
            }
        }
    }

    let mut exact_poses: Vec<(f64, Molecule3D, ScoreBreakdown)> =
        Vec::with_capacity(cluster_states.len());
    for (_, coords) in cluster_states {
        let mut mol = ligand.clone();
        for (atom, c) in mol.atoms.iter_mut().zip(coords.iter()) {
            atom.coords = *c;
        }
        let bd = score_complex(receptor, &mol, config.scoring);
        exact_poses.push((bd.total_affinity_kcal, mol, bd));
    }

    exact_poses.sort_by(|a, b| a.0.total_cmp(&b.0));
    exact_poses.truncate(config.num_modes);

    let best_coords: Option<Vec<[f64; 3]>> = exact_poses
        .first()
        .map(|p| p.1.atoms.iter().map(|a| a.coords).collect());

    let poses = exact_poses
        .into_iter()
        .enumerate()
        .map(|(i, (aff, mut mol, bd))| {
            let rmsd_to_best = if let Some(ref ref_c) = best_coords {
                let c: Vec<[f64; 3]> = mol.atoms.iter().map(|a| a.coords).collect();
                heavy_rmsd(ref_c, &c, &tpl.heavy_indices)
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

        // Check rotating torsion by 17 degrees does NOT translate rigid benzene core
        let core = lig.rigid_core_indices();
        let state0 = RigState {
            translation: [0.2, -0.1, 0.3],
            rot_mat: mat3_identity(),
            torsions: vec![0.0; tpl.torsions.len()],
        };
        let mut c0 = vec![[0.0; 3]; tpl.local_coords.len()];
        let mut c1 = vec![[0.0; 3]; tpl.local_coords.len()];
        tpl.apply_state(&state0, &mut c0);
        let mut state_rot = state0.clone();
        state_rot.torsions[0] = 17.0_f64.to_radians();
        tpl.apply_state(&state_rot, &mut c1);
        let mut core_drift = 0.0_f64;
        for &cid in &core {
            let d = (c0[cid][0] - c1[cid][0])
                .hypot(c0[cid][1] - c1[cid][1])
                .hypot(c0[cid][2] - c1[cid][2]);
            core_drift = core_drift.max(d);
        }
        assert!(core_drift < 1e-12, "rigid core drifted {core_drift} A");

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
