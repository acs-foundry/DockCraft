#![forbid(unsafe_code)]

use std::sync::{Arc, RwLock};

use bytemuck::{Pod, Zeroable};
use egui_wgpu::{Callback, CallbackResources, CallbackTrait, ScreenDescriptor, wgpu};
use hyde_scorer::{HydeReport, InteractionKind, TorsionStatus};
use pocket_detect::{BindingPocket, GridBox};
use serde::{Deserialize, Serialize};
use struct_core::Molecule3D;
use wgpu::util::DeviceExt;

fn default_true_vp() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewportOptions {
    pub show_receptor_backbone: bool,
    pub show_pocket_residues: bool,
    pub show_pocket_cloud: bool,
    pub show_hyde_coronas: bool,
    pub show_interactions: bool,
    pub show_grid_box: bool,
    #[serde(default = "default_true_vp")]
    pub show_torsion_lights: bool,
    #[serde(default = "default_true_vp")]
    pub show_waters_metals: bool,
    /// Optional 3D pharmacophore constraint spheres `(center, radius, rgba)` rendered in the viewport.
    #[serde(default)]
    pub pharmacophore_spheres: Vec<([f64; 3], f64, [f32; 4])>,
    /// Optional 3D pharmacophore directional vectors `(origin, tip, rgba)` rendered as lines.
    #[serde(default)]
    pub pharmacophore_vectors: Vec<([f64; 3], [f64; 3], [f32; 4])>,
    /// Optional receptor wall exclusion spheres `(center, radius)` from `pharm-core`.
    #[serde(default)]
    pub exclusion_spheres: Vec<([f64; 3], f64)>,
    #[serde(default)]
    pub show_exclusion_spheres: bool,
    pub background_rgba: [f32; 4],
}

impl Default for ViewportOptions {
    fn default() -> Self {
        Self {
            show_receptor_backbone: true,
            show_pocket_residues: true,
            show_pocket_cloud: true,
            show_hyde_coronas: true,
            show_interactions: true,
            show_grid_box: true,
            show_torsion_lights: true,
            show_waters_metals: true,
            pharmacophore_spheres: Vec::new(),
            pharmacophore_vectors: Vec::new(),
            exclusion_spheres: Vec::new(),
            show_exclusion_spheres: false,
            background_rgba: [0.075, 0.09, 0.12, 1.0],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Camera3D {
    pub target: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub fov_y_rad: f32,
}

impl Default for Camera3D {
    fn default() -> Self {
        Self {
            target: [0.0, 0.0, 0.0],
            yaw: 0.65,
            pitch: 0.35,
            distance: 24.0,
            fov_y_rad: 45.0_f32.to_radians(),
        }
    }
}

impl Camera3D {
    #[must_use]
    pub fn eye_pos(&self) -> [f32; 3] {
        let cp = self.pitch.cos();
        [
            self.target[0] + self.distance * cp * self.yaw.cos(),
            self.target[1] + self.distance * self.pitch.sin(),
            self.target[2] + self.distance * cp * self.yaw.sin(),
        ]
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * 0.008;
        self.pitch = (self.pitch + dy * 0.008).clamp(-1.48, 1.48);
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        let (right, up, _) = self.basis();
        let scale = self.distance * 0.0018;
        for k in 0..3 {
            self.target[k] -= right[k] * dx * scale;
            self.target[k] += up[k] * dy * scale;
        }
    }

    pub fn zoom(&mut self, scroll_delta: f32) {
        let factor = (1.0 - scroll_delta * 0.0015).clamp(0.65, 1.45);
        self.distance = (self.distance * factor).clamp(4.0, 160.0);
    }

    pub fn focus_on(&mut self, center: [f32; 3], span: f32) {
        self.target = center;
        self.distance = (span * 1.45).clamp(12.0, 110.0);
    }

    #[must_use]
    pub fn basis(&self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let eye = self.eye_pos();
        let fwd = normalize3([
            self.target[0] - eye[0],
            self.target[1] - eye[1],
            self.target[2] - eye[2],
        ]);
        let world_up = [0.0_f32, 1.0, 0.0];
        let right = normalize3(cross3f(fwd, world_up));
        let up = normalize3(cross3f(right, fwd));
        (right, up, fwd)
    }

    #[must_use]
    pub fn view_proj_col_major(&self, aspect: f32) -> [[f32; 4]; 4] {
        let eye = self.eye_pos();
        let (right, up, fwd) = self.basis();
        let view = [
            [right[0], up[0], -fwd[0], 0.0],
            [right[1], up[1], -fwd[1], 0.0],
            [right[2], up[2], -fwd[2], 0.0],
            [-dot3f(right, eye), -dot3f(up, eye), dot3f(fwd, eye), 1.0],
        ];
        let f = 1.0 / (0.5 * self.fov_y_rad).tan();
        let near = 0.2_f32;
        let far = 500.0_f32;
        let nf = 1.0 / (near - far);
        let proj = [
            [f / aspect.max(0.1), 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, far * nf, -1.0],
            [0.0, 0.0, (near * far) * nf, 0.0],
        ];
        mul_mat4(proj, view)
    }

    /// Computes a world-space picking ray `(origin, unit_dir)` from normalized screen coords `nx, ny in [-1, 1]`.
    #[must_use]
    pub fn screen_ray(&self, nx: f32, ny: f32, aspect: f32) -> ([f32; 3], [f32; 3]) {
        let eye = self.eye_pos();
        let (right, up, fwd) = self.basis();
        let tan_half = (0.5 * self.fov_y_rad).tan();
        let dir = normalize3([
            fwd[0] + right[0] * (nx * aspect * tan_half) + up[0] * (ny * tan_half),
            fwd[1] + right[1] * (nx * aspect * tan_half) + up[1] * (ny * tan_half),
            fwd[2] + right[2] * (nx * aspect * tan_half) + up[2] * (ny * tan_half),
        ]);
        (eye, dir)
    }
}

fn dot3f(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3f(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let n = dot3f(v, v).sqrt();
    if n < 1e-6 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / n, v[1] / n, v[2] / n]
    }
}

fn mul_mat4(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0_f32; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            out[c][r] =
                a[0][r] * b[c][0] + a[1][r] * b[c][1] + a[2][r] * b[c][2] + a[3][r] * b[c][3];
        }
    }
    out
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub eye_pos: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SphereVertex {
    pub local_pos: [f32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SphereInstance {
    pub pos_rad: [f32; 4],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CylinderInstance {
    pub a_rad: [f32; 4],
    pub b_pad: [f32; 4],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LineVertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
}

#[derive(Debug, Clone, Default)]
pub struct DockSceneData {
    pub opaque_spheres: Vec<SphereInstance>,
    pub translucent_coronas: Vec<SphereInstance>,
    pub cylinders: Vec<CylinderInstance>,
    pub lines: Vec<LineVertex>,
    pub pickable_ligand_atoms: Vec<(usize, [f32; 3], f32)>,
}

fn element_color(el: &str) -> [f32; 4] {
    match el {
        "C" => [0.35, 0.78, 0.68, 1.0],
        "N" => [0.24, 0.52, 0.96, 1.0],
        "O" => [0.92, 0.25, 0.25, 1.0],
        "S" => [0.92, 0.80, 0.22, 1.0],
        "P" => [0.95, 0.52, 0.16, 1.0],
        "F" => [0.55, 0.90, 0.35, 1.0],
        "Cl" => [0.22, 0.85, 0.28, 1.0],
        "Br" => [0.65, 0.22, 0.18, 1.0],
        "I" => [0.58, 0.22, 0.72, 1.0],
        "H" => [0.88, 0.90, 0.94, 1.0],
        _ => [0.70, 0.74, 0.80, 1.0],
    }
}

fn f64_to_f32_3(c: [f64; 3]) -> [f32; 3] {
    [c[0] as f32, c[1] as f32, c[2] as f32]
}

fn push_dashed_line(
    out: &mut Vec<LineVertex>,
    a: [f32; 3],
    b: [f32; 3],
    color: [f32; 4],
    segments: usize,
) {
    let n = segments.max(1);
    for i in (0..n).step_by(2) {
        let t0 = i as f32 / n as f32;
        let t1 = ((i + 1).min(n)) as f32 / n as f32;
        let p0 = [
            a[0] + (b[0] - a[0]) * t0,
            a[1] + (b[1] - a[1]) * t0,
            a[2] + (b[2] - a[2]) * t0,
        ];
        let p1 = [
            a[0] + (b[0] - a[0]) * t1,
            a[1] + (b[1] - a[1]) * t1,
            a[2] + (b[2] - a[2]) * t1,
        ];
        out.push(LineVertex { pos: p0, color });
        out.push(LineVertex { pos: p1, color });
    }
}

fn push_grid_box_lines(out: &mut Vec<LineVertex>, gbox: &GridBox) {
    let mn = f64_to_f32_3(gbox.min_corner());
    let mx = f64_to_f32_3(gbox.max_corner());
    let col = [0.22, 0.86, 0.95, 0.95];
    let corners = [
        [mn[0], mn[1], mn[2]],
        [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]],
        [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]],
        [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]],
        [mn[0], mx[1], mx[2]],
    ];
    let edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];
    for (u, v) in edges {
        out.push(LineVertex {
            pos: corners[u],
            color: col,
        });
        out.push(LineVertex {
            pos: corners[v],
            color: col,
        });
    }
}

#[must_use]
pub fn build_dock_scene(
    receptor: Option<&Molecule3D>,
    ligand: Option<&Molecule3D>,
    pocket: Option<&BindingPocket>,
    hyde: Option<&HydeReport>,
    grid_box: Option<&GridBox>,
    selected_atom: Option<usize>,
    opts: &ViewportOptions,
) -> DockSceneData {
    let mut scene = DockSceneData::default();
    let focus = grid_box
        .map(|g| g.center)
        .or_else(|| pocket.map(|p| p.center))
        .or_else(|| {
            ligand
                .filter(|l| l.heavy_atom_count() > 0)
                .map(Molecule3D::centroid_heavy)
        })
        .unwrap_or([0.0, 0.0, 0.0]);

    if let Some(rec) = receptor {
        // C-alpha / backbone ribbon tube + nearby pocket sticks
        if opts.show_receptor_backbone {
            let ca_atoms: Vec<&struct_core::Atom3D> = rec
                .atoms
                .iter()
                .filter(|a| a.name == "CA" || (rec.atoms.len() < 120 && a.is_heavy()))
                .collect();
            for win in ca_atoms.windows(2) {
                let a = win[0];
                let b = win[1];
                let p0 = f64_to_f32_3(a.coords);
                let p1 = f64_to_f32_3(b.coords);
                let dx = p1[0] - p0[0];
                let dy = p1[1] - p0[1];
                let dz = p1[2] - p0[2];
                if a.chain_id == b.chain_id && (dx * dx + dy * dy + dz * dz) < 25.0 {
                    scene.cylinders.push(CylinderInstance {
                        a_rad: [p0[0], p0[1], p0[2], 0.24],
                        b_pad: [p1[0], p1[1], p1[2], 0.0],
                        color: [0.34, 0.42, 0.54, 1.0],
                    });
                }
            }
        }

        if opts.show_pocket_residues {
            for b in &rec.bonds {
                let ua = &rec.atoms[b.u];
                let va = &rec.atoms[b.v];
                if !ua.is_heavy() || !va.is_heavy() {
                    continue;
                }
                let du2 = (ua.coords[0] - focus[0]).powi(2)
                    + (ua.coords[1] - focus[1]).powi(2)
                    + (ua.coords[2] - focus[2]).powi(2);
                if du2 > 11.0 * 11.0 {
                    continue;
                }
                let p0 = f64_to_f32_3(ua.coords);
                let p1 = f64_to_f32_3(va.coords);
                let mid = [
                    0.5 * (p0[0] + p1[0]),
                    0.5 * (p0[1] + p1[1]),
                    0.5 * (p0[2] + p1[2]),
                ];
                let mut c0 = element_color(&ua.element);
                let mut c1 = element_color(&va.element);
                if ua.element == "C" {
                    c0 = [0.62, 0.65, 0.72, 1.0];
                }
                if va.element == "C" {
                    c1 = [0.62, 0.65, 0.72, 1.0];
                }
                scene.cylinders.push(CylinderInstance {
                    a_rad: [p0[0], p0[1], p0[2], 0.12],
                    b_pad: [mid[0], mid[1], mid[2], 0.0],
                    color: c0,
                });
                scene.cylinders.push(CylinderInstance {
                    a_rad: [mid[0], mid[1], mid[2], 0.12],
                    b_pad: [p1[0], p1[1], p1[2], 0.0],
                    color: c1,
                });
            }
            for a in &rec.atoms {
                if !a.is_heavy() {
                    continue;
                }
                let d2 = (a.coords[0] - focus[0]).powi(2)
                    + (a.coords[1] - focus[1]).powi(2)
                    + (a.coords[2] - focus[2]).powi(2);
                if d2 <= 9.5 * 9.5 {
                    let p = f64_to_f32_3(a.coords);
                    let col = if a.element == "C" {
                        [0.62, 0.65, 0.72, 1.0]
                    } else {
                        element_color(&a.element)
                    };
                    scene.opaque_spheres.push(SphereInstance {
                        pos_rad: [p[0], p[1], p[2], 0.22],
                        color: col,
                    });
                }
            }
        }
        if opts.show_waters_metals {
            for a in &rec.atoms {
                if !a.is_heavy() {
                    continue;
                }
                let p = f64_to_f32_3(a.coords);
                if a.ad4_type.is_metal() {
                    scene.opaque_spheres.push(SphereInstance {
                        pos_rad: [p[0], p[1], p[2], 0.68],
                        color: [0.76, 0.64, 0.96, 1.0],
                    });
                } else if a.element == "O"
                    && matches!(
                        a.res_name.trim().to_ascii_uppercase().as_str(),
                        "HOH" | "WAT" | "DOD"
                    )
                {
                    scene.opaque_spheres.push(SphereInstance {
                        pos_rad: [p[0], p[1], p[2], 0.40],
                        color: [0.22, 0.78, 0.98, 1.0],
                    });
                }
            }
        }
    }

    if let Some(lig) = ligand {
        for (idx, a) in lig.atoms.iter().enumerate() {
            let p = f64_to_f32_3(a.coords);
            let rad = if a.is_heavy() { 0.34 } else { 0.20 };
            let col = if selected_atom == Some(idx) {
                [1.0, 0.88, 0.25, 1.0]
            } else {
                element_color(&a.element)
            };
            scene.opaque_spheres.push(SphereInstance {
                pos_rad: [p[0], p[1], p[2], rad],
                color: col,
            });
            if a.is_heavy() {
                scene.pickable_ligand_atoms.push((idx, p, 0.55));
            }
        }
        for b in &lig.bonds {
            let p0 = f64_to_f32_3(lig.atoms[b.u].coords);
            let p1 = f64_to_f32_3(lig.atoms[b.v].coords);
            let mid = [
                0.5 * (p0[0] + p1[0]),
                0.5 * (p0[1] + p1[1]),
                0.5 * (p0[2] + p1[2]),
            ];
            let c0 = element_color(&lig.atoms[b.u].element);
            let c1 = element_color(&lig.atoms[b.v].element);
            scene.cylinders.push(CylinderInstance {
                a_rad: [p0[0], p0[1], p0[2], 0.16],
                b_pad: [mid[0], mid[1], mid[2], 0.0],
                color: c0,
            });
            scene.cylinders.push(CylinderInstance {
                a_rad: [mid[0], mid[1], mid[2], 0.16],
                b_pad: [p1[0], p1[1], p1[2], 0.0],
                color: c1,
            });
        }
    }

    if opts.show_pocket_cloud
        && let Some(pock) = pocket
    {
        for &pt in pock.probe_points.iter().step_by(2) {
            scene.translucent_coronas.push(SphereInstance {
                pos_rad: [pt[0] as f32, pt[1] as f32, pt[2] as f32, 0.72],
                color: [0.18, 0.72, 0.88, 0.18],
            });
        }
    }

    if let Some(rep) = hyde {
        if opts.show_hyde_coronas
            && let Some(lig) = ligand
        {
            for sc in &rep.atom_scores {
                if let Some(a) = lig.atoms.get(sc.atom_index) {
                    let p = f64_to_f32_3(a.coords);
                    scene.translucent_coronas.push(SphereInstance {
                        pos_rad: [p[0], p[1], p[2], sc.corona_radius],
                        color: sc.corona_color,
                    });
                }
            }
        }
        if opts.show_torsion_lights
            && let (Some(lig), Some(tors)) = (ligand, rep.torsion_report.as_ref())
        {
            for alert in &tors.torsions {
                if let (Some(au), Some(av)) =
                    (lig.atoms.get(alert.atom_u), lig.atoms.get(alert.atom_v))
                {
                    let p0 = f64_to_f32_3(au.coords);
                    let p1 = f64_to_f32_3(av.coords);
                    let mid = [
                        0.5 * (p0[0] + p1[0]),
                        0.5 * (p0[1] + p1[1]),
                        0.5 * (p0[2] + p1[2]),
                    ];
                    let col = match alert.status {
                        TorsionStatus::Green => [0.14, 0.88, 0.36, 0.95],
                        TorsionStatus::Yellow => [0.98, 0.80, 0.14, 0.95],
                        TorsionStatus::Red => [0.96, 0.18, 0.22, 0.98],
                    };
                    scene.opaque_spheres.push(SphereInstance {
                        pos_rad: [mid[0], mid[1], mid[2], 0.27],
                        color: col,
                    });
                }
            }
        }
        if opts.show_interactions
            && let (Some(rec), Some(lig)) = (receptor, ligand)
        {
            for inter in &rep.interactions {
                if let (Some(&li), Some(&ri)) = (
                    inter.lig_atom_indices.first(),
                    inter.rec_atom_indices.first(),
                ) {
                    let a = f64_to_f32_3(lig.atoms[li].coords);
                    let b = f64_to_f32_3(rec.atoms[ri].coords);
                    let col = match inter.kind {
                        InteractionKind::HydrogenBond => [0.20, 0.92, 0.95, 0.95],
                        InteractionKind::SaltBridge => [1.0, 0.80, 0.18, 0.95],
                        InteractionKind::PiPiStacking => [0.76, 0.42, 0.96, 0.90],
                        InteractionKind::CationPi => [0.95, 0.55, 0.22, 0.90],
                        InteractionKind::HalogenBond => [0.45, 0.95, 0.35, 0.95],
                        InteractionKind::MetalCoordination => [0.98, 0.38, 0.92, 0.98],
                        InteractionKind::WaterBridge => [0.35, 0.78, 1.0, 0.95],
                        InteractionKind::HydrophobicContact => continue,
                    };
                    push_dashed_line(&mut scene.lines, a, b, col, 9);
                }
            }
        }
    }

    for &(center, radius, color) in &opts.pharmacophore_spheres {
        scene.translucent_coronas.push(SphereInstance {
            pos_rad: [
                center[0] as f32,
                center[1] as f32,
                center[2] as f32,
                radius as f32,
            ],
            color,
        });
    }
    for &(origin, tip, color) in &opts.pharmacophore_vectors {
        let a = f64_to_f32_3(origin);
        let b = f64_to_f32_3(tip);
        scene.lines.push(LineVertex { pos: a, color });
        scene.lines.push(LineVertex { pos: b, color });
        scene.opaque_spheres.push(SphereInstance {
            pos_rad: [b[0], b[1], b[2], 0.18],
            color,
        });
    }
    if opts.show_exclusion_spheres {
        for &(center, radius) in &opts.exclusion_spheres {
            scene.translucent_coronas.push(SphereInstance {
                pos_rad: [
                    center[0] as f32,
                    center[1] as f32,
                    center[2] as f32,
                    radius as f32,
                ],
                color: [0.45, 0.48, 0.54, 0.12],
            });
        }
    }

    if opts.show_grid_box
        && let Some(gb) = grid_box
    {
        push_grid_box_lines(&mut scene.lines, gb);
    }

    scene
}

#[must_use]
pub fn pick_atom_ray(
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    pickables: &[(usize, [f32; 3], f32)],
) -> Option<usize> {
    let mut best_t = f32::INFINITY;
    let mut best_idx = None;
    for &(idx, center, radius) in pickables {
        let oc = [
            ray_origin[0] - center[0],
            ray_origin[1] - center[1],
            ray_origin[2] - center[2],
        ];
        let b = dot3f(oc, ray_dir);
        let c = dot3f(oc, oc) - radius * radius;
        let disc = b * b - c;
        if disc >= 0.0 {
            let t = -b - disc.sqrt();
            if t > 0.1 && t < best_t {
                best_t = t;
                best_idx = Some(idx);
            }
        }
    }
    best_idx
}

#[derive(Default)]
pub struct SharedViewportPayload {
    pub scene: DockSceneData,
    pub camera: Camera3D,
    pub background: [f32; 4],
}

pub struct DockWgpuResources {
    cam_buf: wgpu::Buffer,
    cam_bg: wgpu::BindGroup,
    sphere_vb: wgpu::Buffer,
    sphere_ib: wgpu::Buffer,
    sphere_idx_count: u32,
    cyl_vb: wgpu::Buffer,
    cyl_ib: wgpu::Buffer,
    cyl_idx_count: u32,
    opaque_sphere_pipe: wgpu::RenderPipeline,
    corona_sphere_pipe: wgpu::RenderPipeline,
    cylinder_pipe: wgpu::RenderPipeline,
    line_pipe: wgpu::RenderPipeline,
    quad_pipe: wgpu::RenderPipeline,
    quad_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    offscreen_size: [u32; 2],
    color_view: Option<wgpu::TextureView>,
    depth_view: Option<wgpu::TextureView>,
    quad_bg: Option<wgpu::BindGroup>,
    sphere_inst_buf: Option<(wgpu::Buffer, u32)>,
    corona_inst_buf: Option<(wgpu::Buffer, u32)>,
    cyl_inst_buf: Option<(wgpu::Buffer, u32)>,
    line_vb: Option<(wgpu::Buffer, u32)>,
}

fn generate_uv_sphere(stacks: u32, slices: u32) -> (Vec<SphereVertex>, Vec<u16>) {
    let mut verts = Vec::new();
    let mut indices = Vec::new();
    for i in 0..=stacks {
        let v = i as f32 / stacks as f32;
        let phi = v * std::f32::consts::PI;
        for j in 0..=slices {
            let u = j as f32 / slices as f32;
            let theta = u * std::f32::consts::TAU;
            verts.push(SphereVertex {
                local_pos: [phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin()],
            });
        }
    }
    for i in 0..stacks {
        for j in 0..slices {
            let row1 = (i * (slices + 1) + j) as u16;
            let row2 = ((i + 1) * (slices + 1) + j) as u16;
            indices.extend_from_slice(&[row1, row2, row1 + 1, row1 + 1, row2, row2 + 1]);
        }
    }
    (verts, indices)
}

fn generate_unit_cylinder(slices: u32) -> (Vec<SphereVertex>, Vec<u16>) {
    let mut verts = Vec::new();
    let mut indices = Vec::new();
    for ring in 0..=1 {
        let z = ring as f32;
        for j in 0..=slices {
            let ang = (j as f32 / slices as f32) * std::f32::consts::TAU;
            verts.push(SphereVertex {
                local_pos: [ang.cos(), ang.sin(), z],
            });
        }
    }
    let stride = (slices + 1) as u16;
    for j in 0..slices as u16 {
        indices.extend_from_slice(&[j, j + stride, j + 1, j + 1, j + stride, j + stride + 1]);
    }
    (verts, indices)
}

impl DockWgpuResources {
    pub fn ensure_registered(render_state: &egui_wgpu::RenderState) {
        let mut res = render_state.renderer.write();
        if res.callback_resources.get::<Self>().is_none() {
            let created = Self::new(&render_state.device, render_state.target_format);
            res.callback_resources.insert(created);
        }
    }

    fn new(device: &wgpu::Device, egui_target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dockcraft_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let cam_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dock_cam_ub"),
            contents: bytemuck::bytes_of(&CameraUniform {
                view_proj: Camera3D::default().view_proj_col_major(1.33),
                eye_pos: [0.0, 0.0, 24.0, 1.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let cam_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dock_cam_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dock_cam_bg"),
            layout: &cam_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: cam_buf.as_entire_binding(),
            }],
        });

        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dock_3d_pl"),
            bind_group_layouts: &[&cam_bgl],
            push_constant_ranges: &[],
        });

        let offscreen_fmt = wgpu::TextureFormat::Rgba8Unorm;
        let depth_fmt = wgpu::TextureFormat::Depth32Float;

        let sphere_vbl = [
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<SphereVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                }],
            },
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<SphereInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 0,
                        shader_location: 1,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 16,
                        shader_location: 2,
                    },
                ],
            },
        ];

        let opaque_sphere_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dock_opaque_sphere"),
            layout: Some(&pipe_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_sphere"),
                buffers: &sphere_vbl,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_solid"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: offscreen_fmt,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_fmt,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let corona_sphere_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dock_corona_sphere"),
            layout: Some(&pipe_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_sphere"),
                buffers: &sphere_vbl,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_corona"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: offscreen_fmt,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_fmt,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let cyl_vbl = [
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<SphereVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                }],
            },
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<CylinderInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 0,
                        shader_location: 1,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 16,
                        shader_location: 2,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 32,
                        shader_location: 3,
                    },
                ],
            },
        ];

        let cylinder_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dock_cylinder"),
            layout: Some(&pipe_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_cylinder"),
                buffers: &cyl_vbl,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_solid"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: offscreen_fmt,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_fmt,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let line_vbl = [wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LineVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 12,
                    shader_location: 1,
                },
            ],
        }];

        let line_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dock_lines"),
            layout: Some(&pipe_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_line"),
                buffers: &line_vbl,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_line"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: offscreen_fmt,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_fmt,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let quad_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dock_quad_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let quad_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dock_quad_pl"),
            bind_group_layouts: &[&quad_bgl],
            push_constant_ranges: &[],
        });

        let quad_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dock_quad"),
            layout: Some(&quad_pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_quad"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_quad"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: egui_target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let (sv, si) = generate_uv_sphere(12, 16);
        let sphere_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dock_sphere_vb"),
            contents: bytemuck::cast_slice(&sv),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let sphere_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dock_sphere_ib"),
            contents: bytemuck::cast_slice(&si),
            usage: wgpu::BufferUsages::INDEX,
        });

        let (cv, ci) = generate_unit_cylinder(10);
        let cyl_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dock_cyl_vb"),
            contents: bytemuck::cast_slice(&cv),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let cyl_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dock_cyl_ib"),
            contents: bytemuck::cast_slice(&ci),
            usage: wgpu::BufferUsages::INDEX,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            cam_buf,
            cam_bg,
            sphere_vb,
            sphere_ib,
            sphere_idx_count: si.len() as u32,
            cyl_vb,
            cyl_ib,
            cyl_idx_count: ci.len() as u32,
            opaque_sphere_pipe,
            corona_sphere_pipe,
            cylinder_pipe,
            line_pipe,
            quad_pipe,
            quad_bgl,
            sampler,
            offscreen_size: [0, 0],
            color_view: None,
            depth_view: None,
            quad_bg: None,
            sphere_inst_buf: None,
            corona_inst_buf: None,
            cyl_inst_buf: None,
            line_vb: None,
        }
    }

    fn ensure_targets(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        let w = w.max(64);
        let h = h.max(64);
        if self.offscreen_size == [w, h] && self.color_view.is_some() {
            return;
        }
        let color_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dock_offscreen_color"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dock_offscreen_depth"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let cview = color_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let dview = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let qbg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dock_quad_bg"),
            layout: &self.quad_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&cview),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.offscreen_size = [w, h];
        self.color_view = Some(cview);
        self.depth_view = Some(dview);
        self.quad_bg = Some(qbg);
    }
}

pub struct DockPaintCallback {
    pub viewport_px: [u32; 2],
    pub payload: Arc<RwLock<SharedViewportPayload>>,
}

impl DockPaintCallback {
    #[must_use]
    pub fn into_paint_callback(
        rect: egui::Rect,
        viewport_px: [u32; 2],
        payload: Arc<RwLock<SharedViewportPayload>>,
    ) -> egui::PaintCallback {
        Callback::new_paint_callback(
            rect,
            Self {
                viewport_px,
                payload,
            },
        )
    }
}

impl CallbackTrait for DockPaintCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &ScreenDescriptor,
        egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(res) = callback_resources.get_mut::<DockWgpuResources>() else {
            return Vec::new();
        };
        let Ok(guard) = self.payload.read() else {
            return Vec::new();
        };
        res.ensure_targets(device, self.viewport_px[0], self.viewport_px[1]);

        let aspect = (self.viewport_px[0].max(1) as f32) / (self.viewport_px[1].max(1) as f32);
        let eye = guard.camera.eye_pos();
        let cam_u = CameraUniform {
            view_proj: guard.camera.view_proj_col_major(aspect),
            eye_pos: [eye[0], eye[1], eye[2], 1.0],
        };
        queue.write_buffer(&res.cam_buf, 0, bytemuck::bytes_of(&cam_u));

        res.sphere_inst_buf = (!guard.scene.opaque_spheres.is_empty()).then(|| {
            (
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("inst_sp"),
                    contents: bytemuck::cast_slice(&guard.scene.opaque_spheres),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
                guard.scene.opaque_spheres.len() as u32,
            )
        });
        res.corona_inst_buf = (!guard.scene.translucent_coronas.is_empty()).then(|| {
            (
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("inst_cor"),
                    contents: bytemuck::cast_slice(&guard.scene.translucent_coronas),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
                guard.scene.translucent_coronas.len() as u32,
            )
        });
        res.cyl_inst_buf = (!guard.scene.cylinders.is_empty()).then(|| {
            (
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("inst_cyl"),
                    contents: bytemuck::cast_slice(&guard.scene.cylinders),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
                guard.scene.cylinders.len() as u32,
            )
        });
        res.line_vb = (!guard.scene.lines.is_empty()).then(|| {
            (
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("vb_lines"),
                    contents: bytemuck::cast_slice(&guard.scene.lines),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
                guard.scene.lines.len() as u32,
            )
        });

        let (Some(cview), Some(dview)) = (&res.color_view, &res.depth_view) else {
            return Vec::new();
        };
        let bg = guard.background;
        let mut rpass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dock_offscreen_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: cview,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: bg[0] as f64,
                        g: bg[1] as f64,
                        b: bg[2] as f64,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: dview,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        rpass.set_bind_group(0, &res.cam_bg, &[]);
        if let Some((buf, count)) = &res.cyl_inst_buf {
            rpass.set_pipeline(&res.cylinder_pipe);
            rpass.set_vertex_buffer(0, res.cyl_vb.slice(..));
            rpass.set_vertex_buffer(1, buf.slice(..));
            rpass.set_index_buffer(res.cyl_ib.slice(..), wgpu::IndexFormat::Uint16);
            rpass.draw_indexed(0..res.cyl_idx_count, 0, 0..*count);
        }
        if let Some((buf, count)) = &res.sphere_inst_buf {
            rpass.set_pipeline(&res.opaque_sphere_pipe);
            rpass.set_vertex_buffer(0, res.sphere_vb.slice(..));
            rpass.set_vertex_buffer(1, buf.slice(..));
            rpass.set_index_buffer(res.sphere_ib.slice(..), wgpu::IndexFormat::Uint16);
            rpass.draw_indexed(0..res.sphere_idx_count, 0, 0..*count);
        }
        if let Some((buf, count)) = &res.line_vb {
            rpass.set_pipeline(&res.line_pipe);
            rpass.set_vertex_buffer(0, buf.slice(..));
            rpass.draw(0..*count, 0..1);
        }
        if let Some((buf, count)) = &res.corona_inst_buf {
            rpass.set_pipeline(&res.corona_sphere_pipe);
            rpass.set_vertex_buffer(0, res.sphere_vb.slice(..));
            rpass.set_vertex_buffer(1, buf.slice(..));
            rpass.set_index_buffer(res.sphere_ib.slice(..), wgpu::IndexFormat::Uint16);
            rpass.draw_indexed(0..res.sphere_idx_count, 0, 0..*count);
        }

        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &CallbackResources,
    ) {
        if let Some(res) = callback_resources.get::<DockWgpuResources>()
            && let Some(qbg) = &res.quad_bg
        {
            render_pass.set_pipeline(&res.quad_pipe);
            render_pass.set_bind_group(0, qbg, &[]);
            render_pass.draw(0..6, 0..1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use struct_core::{Atom3D, BondOrder};

    #[test]
    fn builds_scene_geometry_and_picks_ligand_atom_via_ray() {
        let mut lig = Molecule3D::new("lig");
        let a0 = lig.add_atom(Atom3D::new(1, "C", [0.0, 0.0, 0.0]));
        let a1 = lig.add_atom(Atom3D::new(2, "N", [1.5, 0.0, 0.0]));
        let _ = lig.add_bond(a0, a1, BondOrder::Single);

        let gbox = GridBox::from_center_size([0.5, 0.0, 0.0], [12.0, 12.0, 12.0], 0.375).ok();
        let scene = build_dock_scene(
            None,
            Some(&lig),
            None,
            None,
            gbox.as_ref(),
            Some(1),
            &ViewportOptions::default(),
        );
        assert_eq!(scene.opaque_spheres.len(), 2);
        assert_eq!(scene.cylinders.len(), 2);
        assert_eq!(scene.lines.len(), 24); // 12 cube edges * 2 vertices

        let hit = pick_atom_ray(
            [0.0, 0.0, 10.0],
            [0.0, 0.0, -1.0],
            &scene.pickable_ligand_atoms,
        );
        assert_eq!(hit, Some(0));

        let cam = Camera3D::default();
        let vp = cam.view_proj_col_major(1.5);
        assert!(vp[0][0].is_finite() && vp[3][3].is_finite());
    }
}
