#![forbid(unsafe_code)]

use std::time::Instant;

use pocket_detect::GridBox;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use struct_core::Molecule3D;

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
            bfgs_iters: 36,
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
    torsions: Vec<(usize, usize, Vec<usize>)>, // (u, v, moving_atoms)
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

/// Rodrigues rotation matrix `exp([w]_x)` from axis-angle vector `w`.
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

fn random_rotation(rng: &mut StdRng) -> [[f64; 3]; 3] {
    // Uniform random unit axis + angle via Shoemake/axis-angle
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
        if ligand.heavy_atom_count() == 0 {
            return None;
        }
        let centroid = ligand.centroid_heavy();
        let local_coords: Vec<[f64; 3]> = ligand
            .atoms
            .iter()
            .map(|a| {
                [
                    a.coords[0] - centroid[0],
                    a.coords[1] - centroid[1],
                    a.coords[2] - centroid[2],
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

        // Apply internal torsion angles in template local frame
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

        // Rigid-body rotation + translation
        for p in out_coords.iter_mut() {
            let r = mat3_vec(state.rot_mat, *p);
            *p = [
                r[0] + state.translation[0],
                r[1] + state.translation[1],
                r[2] + state.translation[2],
            ];
        }
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
        let effective_state =
            if delta_omega[0] != 0.0 || delta_omega[1] != 0.0 || delta_omega[2] != 0.0 {
                RigState {
                    translation: state.translation,
                    rot_mat: mat3_mul(rot_from_axis_angle(delta_omega), state.rot_mat),
                    torsions: state.torsions.clone(),
                }
            } else {
                state.clone()
            };
        self.apply_state(&effective_state, world_coords);
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

        // Intramolecular steric repulsion
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
            let rx = world_coords[atom_idx][0] - effective_state.translation[0];
            let ry = world_coords[atom_idx][1] - effective_state.translation[1];
            let rz = world_coords[atom_idx][2] - effective_state.translation[2];
            // Torque: r x g
            g_omega[0] += ry * g[2] - rz * g[1];
            g_omega[1] += rz * g[0] - rx * g[2];
            g_omega[2] += rx * g[1] - ry * g[0];
        }
        for k in 0..3 {
            g_trans[k] *= inv_div;
            g_omega[k] *= inv_div;
        }

        // Exact world-frame torsion gradients via current pivot & rotated bond axis
        // Recompute sequentially in local frame or use final world_coords? Notice torsion k only affects atoms downstream,
        // wait: if outer torsions rotate an inner bond axis, world_coords ALREADY reflects all nested rotations in reverse or forward order?
        // In apply_state, torsions are applied 0..K; world_coords[(u,v)] of bond k IS shifted by subsequent torsions k+1..K if u,v are in moving!
        // Only bonds in non-moving sets are independent; for chain accuracy, if bond j moves (u_k,v_k) when j > k, using sequential axis:
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

/// SO(3)+torsion L-BFGS / preconditioned steepest-descent + Armijo backtracking optimizer.
fn local_minimize(
    tpl: &LigandTemplate,
    grids: &AffinityGridSet,
    state: &mut RigState,
    max_iters: usize,
    eval_counter: &mut usize,
) -> f64 {
    let n_t = tpl.torsions.len();
    let n_dim = 6 + n_t;
    let mut world = vec![[0.0_f64; 3]; tpl.local_coords.len()];
    let mut atom_grads = vec![[0.0_f64; 3]; tpl.local_coords.len()];

    let (mut f_curr, mut gt, mut gw, mut gtors) =
        tpl.eval_and_grad(grids, state, [0.0; 3], &mut world, &mut atom_grads);
    *eval_counter += 1;

    let mut inv_hess_diag = vec![1.0_f64; n_dim];
    for g in inv_hess_diag.iter_mut().take(3) {
        *g = 0.45;
    }
    for g in inv_hess_diag.iter_mut().skip(3).take(3) {
        *g = 0.12;
    }
    for g in inv_hess_diag.iter_mut().skip(6) {
        *g = 0.25;
    }

    let mut prev_grad: Option<Vec<f64>> = None;
    let mut prev_step: Option<Vec<f64>> = None;

    for _ in 0..max_iters {
        let mut grad = Vec::with_capacity(n_dim);
        grad.extend_from_slice(&gt);
        grad.extend_from_slice(&gw);
        grad.extend_from_slice(&gtors);

        let g_norm2: f64 = grad.iter().map(|v| v * v).sum();
        if g_norm2 < 1e-5 {
            break;
        }

        // Diagonal Barzilai-Borwein / damped secant curvature update
        if let (Some(pg), Some(ps)) = (&prev_grad, &prev_step) {
            for k in 0..n_dim {
                let s = ps[k];
                let y = grad[k] - pg[k];
                if s * y > 1e-8 {
                    let ratio = (s * s) / (s * y);
                    inv_hess_diag[k] = ratio.clamp(0.02, 1.8);
                }
            }
        }

        let mut dir = vec![0.0_f64; n_dim];
        let mut slope = 0.0_f64;
        for k in 0..n_dim {
            let d = -inv_hess_diag[k] * grad[k];
            // Clamp step component size for numerical stability on steep steric walls
            let cap = if k < 3 { 1.2 } else { 0.45 };
            let clamped = d.clamp(-cap, cap);
            dir[k] = clamped;
            slope += grad[k] * clamped;
        }
        if slope >= -1e-9 {
            break;
        }

        // Armijo backtracking line search
        let mut alpha = 1.0_f64;
        let mut accepted = false;
        for _ls in 0..10 {
            let mut cand = state.clone();
            cand.translation[0] += alpha * dir[0];
            cand.translation[1] += alpha * dir[1];
            cand.translation[2] += alpha * dir[2];
            let dw = [alpha * dir[3], alpha * dir[4], alpha * dir[5]];
            for t in 0..n_t {
                cand.torsions[t] += alpha * dir[6 + t];
            }

            let (f_try, gt_try, gw_try, gtors_try) =
                tpl.eval_and_grad(grids, &cand, dw, &mut world, &mut atom_grads);
            *eval_counter += 1;

            if f_try <= f_curr + 1e-4 * alpha * slope {
                cand.rot_mat = mat3_mul(rot_from_axis_angle(dw), cand.rot_mat);
                prev_grad = Some(grad);
                prev_step = Some(dir.iter().map(|&d| alpha * d).collect());
                *state = cand;
                f_curr = f_try;
                gt = gt_try;
                gw = gw_try;
                gtors = gtors_try;
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
    let start = Instant::now();
    let tpl = LigandTemplate::from_ligand(ligand, grids, config.scoring)
        .ok_or(DockError::EmptyStructure)?;

    let min_c = grids.grid_box.min_corner();
    let max_c = grids.grid_box.max_corner();
    let span = [
        (max_c[0] - min_c[0] - 2.0).max(2.0),
        (max_c[1] - min_c[1] - 2.0).max(2.0),
        (max_c[2] - min_c[2] - 2.0).max(2.0),
    ];
    let center = grids.grid_box.center;

    let n_walkers = config.exhaustiveness.max(1);
    let orig_centroid = if ligand.heavy_atom_count() > 0 {
        ligand.centroid_heavy()
    } else {
        center
    };

    type WalkerCand = (f64, RigState, Vec<[f64; 3]>);
    let walker_results: Vec<(Vec<WalkerCand>, usize)> = (0..n_walkers)
        .into_par_iter()
        .map(|w_idx| {
            let mut rng = StdRng::seed_from_u64(
                config.seed ^ (w_idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            let mut evals = 0usize;
            let n_t = tpl.torsions.len();

            let mut curr_state = if w_idx == 0 && grids.grid_box.contains(orig_centroid) {
                RigState {
                    translation: orig_centroid,
                    rot_mat: mat3_identity(),
                    torsions: vec![0.0; n_t],
                }
            } else {
                let tr = [
                    center[0] + (rng.r#gen::<f64>() - 0.5) * 0.7 * span[0],
                    center[1] + (rng.r#gen::<f64>() - 0.5) * 0.7 * span[1],
                    center[2] + (rng.r#gen::<f64>() - 0.5) * 0.7 * span[2],
                ];
                let tors = (0..n_t)
                    .map(|_| (rng.r#gen::<f64>() - 0.5) * std::f64::consts::TAU)
                    .collect();
                RigState {
                    translation: tr,
                    rot_mat: random_rotation(&mut rng),
                    torsions: tors,
                }
            };

            let mut curr_e =
                local_minimize(&tpl, grids, &mut curr_state, config.bfgs_iters, &mut evals);
            let mut candidates = Vec::new();
            let mut coords_buf = vec![[0.0; 3]; tpl.local_coords.len()];
            tpl.apply_state(&curr_state, &mut coords_buf);
            candidates.push((curr_e, curr_state.clone(), coords_buf.clone()));

            let temp_kt = 1.2_f64;
            for step in 0..config.ils_steps {
                let mut mutated = curr_state.clone();
                let amp = if step % 3 == 0 { 2.5 } else { 1.3 };
                for k in 0..3 {
                    mutated.translation[k] = (mutated.translation[k]
                        + (rng.r#gen::<f64>() - 0.5) * 2.0 * amp)
                        .clamp(min_c[k] + 1.0, max_c[k] - 1.0);
                }
                let rot_kick = [
                    (rng.r#gen::<f64>() - 0.5) * 1.4,
                    (rng.r#gen::<f64>() - 0.5) * 1.4,
                    (rng.r#gen::<f64>() - 0.5) * 1.4,
                ];
                mutated.rot_mat = mat3_mul(rot_from_axis_angle(rot_kick), mutated.rot_mat);
                if n_t > 0 {
                    let tor_idx = rng.gen_range(0..n_t);
                    mutated.torsions[tor_idx] += (rng.r#gen::<f64>() - 0.5) * std::f64::consts::PI;
                }

                let mut_e =
                    local_minimize(&tpl, grids, &mut mutated, config.bfgs_iters, &mut evals);
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

    let mut total_evals = 0usize;
    let mut all_candidates: Vec<(f64, Vec<[f64; 3]>)> = Vec::new();
    for (cands, ev) in walker_results {
        total_evals += ev;
        for (e, _st, coords) in cands {
            if e.is_finite() {
                all_candidates.push((e, coords));
            }
        }
    }

    all_candidates.sort_by(|a, b| a.0.total_cmp(&b.0));

    // RMSD clustering + exact pairwise re-scoring on top diverse clusters
    let mut cluster_pool: Vec<Vec<[f64; 3]>> = Vec::new();
    for (_grid_e, coords) in all_candidates {
        let distinct = cluster_pool
            .iter()
            .all(|kept| heavy_rmsd(kept, &coords, &tpl.heavy_indices) >= config.min_rmsd_cluster);
        if distinct {
            cluster_pool.push(coords);
            if cluster_pool.len() >= config.num_modes * 3 {
                break;
            }
        }
    }

    let mut exact_poses: Vec<(f64, Molecule3D, ScoreBreakdown)> =
        Vec::with_capacity(cluster_pool.len());
    for coords in cluster_pool {
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
    };

    /// Creates a complementary polar+hydrophobic binding pocket and checks if docking recovers < 1.5 A RMSD.
    #[test]
    fn redocks_complementary_ligand_under_2a_rmsd() -> Result<(), DockError> {
        // Native ligand: a small polar-tipped hydrophobic fragment at origin
        let mut native = Molecule3D::new("native_probe");
        let a0 = native.add_atom({
            let mut a = Atom3D::new(1, "O", [-1.6, 0.0, 0.0]);
            a.charge = -0.55;
            a.ad4_type = Ad4Type::OA;
            a
        });
        let a1 = native.add_atom({
            let mut a = Atom3D::new(2, "C", [-0.2, 0.0, 0.0]);
            a.charge = 0.15;
            a
        });
        let a2 = native.add_atom({
            let mut a = Atom3D::new(3, "C", [1.2, 0.4, 0.0]);
            a.charge = 0.0;
            a
        });
        let a3 = native.add_atom({
            let mut a = Atom3D::new(4, "C", [2.4, -0.3, 0.4]);
            a.charge = 0.0;
            a
        });
        let a4 = native.add_atom({
            let mut a = Atom3D::new(5, "N", [-1.0, 1.2, 0.0]);
            a.charge = 0.35;
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

        // Complementary receptor shell matching the native pose tightly via H-bonds and hydrophobic cradle
        let mut receptor = Molecule3D::new("rec");
        let rec_specs = [
            // H-bond donor partner at ~2.85 A from O(-1.6,0,0)
            ("N", [-4.45, 0.0, 0.0], 0.45, Ad4Type::NA, true, false),
            // H-bond acceptor partner at ~2.90 A from N(-1.0,1.2,0)
            ("O", [-1.0, 4.10, 0.0], -0.45, Ad4Type::OA, false, true),
            // Hydrophobic pocket walls at ~3.9 A around C(1.2,0.4,0) and C(2.4,-0.3,0.4)
            ("C", [1.5, 4.2, 0.0], 0.0, Ad4Type::C, false, false),
            ("C", [1.5, -3.5, 0.0], 0.0, Ad4Type::C, false, false),
            ("C", [2.5, 0.2, 4.2], 0.0, Ad4Type::C, false, false),
            ("C", [2.5, 0.2, -3.5], 0.0, Ad4Type::C, false, false),
            ("C", [6.2, -0.3, 0.4], 0.0, Ad4Type::C, false, false),
            ("C", [-2.5, -3.7, 0.0], 0.0, Ad4Type::C, false, false),
            ("C", [-1.0, 0.0, 3.8], 0.0, Ad4Type::C, false, false),
            ("C", [-1.0, 0.0, -3.8], 0.0, Ad4Type::C, false, false),
        ];
        for (i, &(el, pos, q, ad4, hd, ha)) in rec_specs.iter().enumerate() {
            let mut a = Atom3D::new(i + 1, el, pos);
            a.charge = q;
            a.ad4_type = ad4;
            a.h_donor = hd;
            a.h_acceptor = ha;
            receptor.add_atom(a);
        }

        // Scramble initial ligand pose by translating + rotating away from native pose
        let mut scrambled = native.clone();
        for a in &mut scrambled.atoms {
            a.coords = [a.coords[2] + 3.2, -a.coords[0] - 2.1, a.coords[1] + 1.8];
        }

        let gbox = GridBox::from_center_size([0.5, 0.0, 0.0], [16.0, 14.0, 14.0], 0.375)?;
        let cfg = DockingConfig {
            exhaustiveness: 6,
            ils_steps: 28,
            bfgs_iters: 40,
            ..Default::default()
        };
        let report = dock_ligand(&receptor, &scrambled, &gbox, &cfg)?;
        assert!(!report.poses.is_empty());
        let top = &report.poses[0];
        assert!(
            top.affinity_kcal_mol < -1.0,
            "expected favorable negative affinity, got {}",
            top.affinity_kcal_mol
        );
        let rmsd = native
            .rmsd_heavy(&top.molecule)
            .ok_or(DockError::EmptyStructure)?;
        assert!(
            rmsd < 2.0,
            "expected redocking RMSD < 2.0 A, got {rmsd:.3} A (score {:.3})",
            top.affinity_kcal_mol
        );
        Ok(())
    }
}
