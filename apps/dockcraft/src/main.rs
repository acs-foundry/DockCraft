#![forbid(unsafe_code)]

mod dialogs;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use dialogs::{
    DockCraftProject, launch_admetcraft_handoff, launch_chemcraft_with_path, pick_open_file,
    pick_save_file,
};
use dock_engine::{
    DockingConfig, DockingConstraint, DockingPose, PharmacophoreKind, ScoringFunction, dock_ligand,
    score_complex,
};
use dock_pipeline::{
    AdmetProfile, CampaignReport, ChimeraXRest, LigandInputRecord, build_chimerax_export,
    build_pymol_export, campaign_report_to_csv, default_chemcraft_results_dir,
    default_chimerax_export_dir, evaluate_admet, grid_box_from_chimerax_selection,
    launch_chimerax_for_scene, load_ligand_library_bytes, run_screening_campaign,
    write_results_sidecar,
};
use dock_viewport::{
    Camera3D, DockPaintCallback, DockWgpuResources, SharedViewportPayload, ViewportOptions,
    build_dock_scene, pick_atom_ray,
};
use eframe::egui::{self, Color32, RichText};
use hyde_scorer::{
    HydeReport, SubstituentPreset, TorsionStatus, consensus_affinity_kcal,
    mutate_and_minimize_in_pocket, score_pose_hyde,
};
use pharm_core::{FeatureSource, SiteAnalysisConfig, SiteBlueprint, analyze_binding_site};
use pocket_detect::{BindingPocket, GridBox, PocketDetectConfig, detect_pockets};
use prot_prep::{AuditSeverity, PrepConfig, PreparedReceptor, prepare_protein_from_str};
use struct_core::{
    Atom3D, BondOrder, BridgingWater, Molecule3D, assign_ad4_types_and_flags,
    detect_bridging_waters, parse_mmcif, parse_mol2, parse_pdb, parse_pdbqt,
    parse_sdf_3d_with_limit, parse_smiles_to_3d, perceive_all, prepare_receptor,
    prepare_receptor_keeping_waters, write_pdb, write_sdf_3d,
};

const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InspiratorCategory {
    HalogensSmall,
    AlkylFluoro,
    PolarHBond,
    RingsBioisosteres,
}

impl InspiratorCategory {
    fn label(self) -> &'static str {
        match self {
            Self::HalogensSmall => "Small/Hal",
            Self::AlkylFluoro => "Alkyl/F",
            Self::PolarHBond => "Polar/HB",
            Self::RingsBioisosteres => "Rings/Bio",
        }
    }

    fn presets(self) -> &'static [SubstituentPreset] {
        match self {
            Self::HalogensSmall => &[
                SubstituentPreset::H,
                SubstituentPreset::F,
                SubstituentPreset::Cl,
                SubstituentPreset::Br,
                SubstituentPreset::I,
                SubstituentPreset::CH3,
                SubstituentPreset::CN,
                SubstituentPreset::SH,
            ],
            Self::AlkylFluoro => &[
                SubstituentPreset::Et,
                SubstituentPreset::IPr,
                SubstituentPreset::TBu,
                SubstituentPreset::CPr,
                SubstituentPreset::CF3,
                SubstituentPreset::CHF2,
                SubstituentPreset::OCF3,
                SubstituentPreset::OEt,
            ],
            Self::PolarHBond => &[
                SubstituentPreset::OH,
                SubstituentPreset::OCH3,
                SubstituentPreset::NH2,
                SubstituentPreset::NHMe,
                SubstituentPreset::NMe2,
                SubstituentPreset::COOH,
                SubstituentPreset::CONH2,
                SubstituentPreset::CONHMe,
                SubstituentPreset::COOMe,
                SubstituentPreset::SO2Me,
                SubstituentPreset::SO2NH2,
            ],
            Self::RingsBioisosteres => &[
                SubstituentPreset::Phenyl,
                SubstituentPreset::Pyridyl,
                SubstituentPreset::Morpholino,
                SubstituentPreset::Piperazino,
                SubstituentPreset::Tetrazole,
            ],
        }
    }
}

fn build_demo_showcase() -> (Molecule3D, Molecule3D) {
    let mut rec = Molecule3D::new("1STP_Streptavidin_Pocket");
    let res = [
        ("N", "ASN", 23, [-3.20, 0.20, 0.00]),
        ("O", "SER", 27, [0.50, -3.50, 0.40]),
        ("O", "TYR", 43, [0.80, 4.10, 0.00]),
        ("O", "SER", 45, [-0.80, 2.75, -1.40]),
        ("N", "ASN", 49, [7.80, -3.20, -2.90]),
        ("C", "TRP", 79, [4.50, -0.20, 4.30]),
        ("C", "TRP", 92, [2.40, -4.40, -2.50]),
        ("C", "TRP", 108, [4.10, 1.40, -3.60]),
        ("C", "LEU", 110, [0.80, 0.20, 3.80]),
        ("O", "ASP", 128, [-3.00, -1.80, -1.60]),
        ("C", "VAL", 47, [2.80, 4.30, 1.60]),
        ("C", "PHE", 130, [7.50, -0.80, 1.90]),
    ];
    for (i, &(el, rn, seq, pos)) in res.iter().enumerate() {
        let mut a = Atom3D::new(i + 1, el, pos);
        a.chain_id = 'A';
        a.res_name = rn.to_string();
        a.res_seq = seq;
        a.name = if el == "C" { "CB" } else { el }.to_string();
        rec.add_atom(a);
    }
    prepare_receptor(&mut rec, false);

    let mut lig = Molecule3D::new("Biotin_Probe");
    lig.source_smiles = Some("O=C(O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@@H]21".to_string());
    let coords = [
        ("O", [-0.40, 0.20, 0.00]),
        ("C", [0.82, 0.20, 0.00]),
        ("N", [1.52, 1.38, 0.00]),
        ("N", [1.52, -0.98, 0.00]),
        ("C", [2.95, 0.92, 0.00]),
        ("C", [2.95, -0.58, 0.00]),
        ("S", [4.32, 0.17, 0.85]),
        ("C", [3.70, -1.65, -0.80]),
        ("C", [5.12, -1.75, -0.25]),
        ("O", [5.88, -2.65, -1.05]),
    ];
    let mut ids = Vec::new();
    for (i, &(el, pos)) in coords.iter().enumerate() {
        ids.push(lig.add_atom(Atom3D::new(i + 1, el, pos)));
    }
    let bonds = [
        (0, 1, BondOrder::Double),
        (1, 2, BondOrder::Single),
        (1, 3, BondOrder::Single),
        (2, 4, BondOrder::Single),
        (3, 5, BondOrder::Single),
        (4, 5, BondOrder::Single),
        (4, 6, BondOrder::Single),
        (5, 7, BondOrder::Single),
        (7, 8, BondOrder::Single),
        (8, 9, BondOrder::Single),
    ];
    for &(u, v, ord) in &bonds {
        let _ = lig.add_bond(ids[u], ids[v], ord);
    }
    assign_ad4_types_and_flags(&mut lig);
    perceive_all(&mut lig);
    (rec, lig)
}

fn format_ki_nm(ki: f64) -> String {
    if !ki.is_finite() {
        "n/a".to_string()
    } else if ki < 1.0 {
        format!("{:.2} pM", ki * 1000.0)
    } else if ki < 1000.0 {
        format!("{ki:.1} nM")
    } else if ki < 1_000_000.0 {
        format!("{:.2} uM", ki / 1000.0)
    } else {
        format!("{:.2} mM", ki / 1_000_000.0)
    }
}

struct DockCraftApp {
    current_project_path: Option<PathBuf>,
    receptor_path_input: String,
    ligand_path_input: String,
    raw_receptor_with_waters: Option<Molecule3D>,
    raw_receptor_source: Option<(String, String)>,
    receptor: Option<Molecule3D>,
    prep_ph: f64,
    prep_chains_input: String,
    prep_keep_metals: bool,
    prep_keep_cofactors: bool,
    prep_keep_bridging_waters: bool,
    prep_optimize_hbonds: bool,
    prep_result: Option<PreparedReceptor>,
    ligand: Option<Molecule3D>,
    pose_history: Vec<Molecule3D>,
    pockets: Vec<BindingPocket>,
    selected_pocket_idx: Option<usize>,
    bridging_waters: Vec<BridgingWater>,
    kept_water_resseqs: BTreeSet<i32>,
    grid_box: GridBox,
    scoring: ScoringFunction,
    exhaustiveness: usize,
    metal_coordination: bool,
    hyde_rerank: bool,
    constraints: Vec<DockingConstraint>,
    site_blueprint: Option<SiteBlueprint>,
    blueprint_use_holo: bool,
    new_pharm_kind: PharmacophoreKind,
    new_pharm_radius: f64,
    flex_residues_input: String,
    dock_poses: Vec<DockingPose>,
    selected_pose_idx: usize,
    campaign_report: Option<CampaignReport>,
    hyde_report: Option<HydeReport>,
    vina_score_kcal: Option<f64>,
    admet_preview: Option<AdmetProfile>,
    selected_atom_idx: Option<usize>,
    inspirator_cat: InspiratorCategory,
    last_mutation_delta: Option<(String, f64)>,
    status_line: String,
    palette_open: bool,
    palette_query: String,
    viewport_opts: ViewportOptions,
    camera: Camera3D,
    shared_payload: Arc<RwLock<SharedViewportPayload>>,
    scene_dirty: bool,
}

impl DockCraftApp {
    fn new(cc: &eframe::CreationContext<'_>, startup_file: Option<PathBuf>) -> Self {
        if let Some(rs) = cc.wgpu_render_state.as_ref() {
            DockWgpuResources::ensure_registered(rs);
        }
        let (rec, lig) = build_demo_showcase();
        let gbox = GridBox::from_ligand(&lig, 6.0).unwrap_or(GridBox {
            center: [1.5, 0.0, 0.0],
            size: [18.0, 18.0, 18.0],
            spacing: 0.375,
        });
        let hyde = score_pose_hyde(&rec, &lig).ok();
        let vina_e = Some(score_complex(&rec, &lig, ScoringFunction::Vina).total_affinity_kcal);
        let mut cam = Camera3D::default();
        cam.focus_on([1.5, 0.0, 0.0], 12.0);

        let mut app = Self {
            current_project_path: None,
            receptor_path_input: String::new(),
            ligand_path_input: String::new(),
            raw_receptor_with_waters: Some(rec.clone()),
            raw_receptor_source: None,
            receptor: Some(rec),
            prep_ph: 7.4,
            prep_chains_input: String::new(),
            prep_keep_metals: true,
            prep_keep_cofactors: true,
            prep_keep_bridging_waters: false,
            prep_optimize_hbonds: true,
            prep_result: None,
            ligand: Some(lig),
            pose_history: Vec::new(),
            pockets: Vec::new(),
            selected_pocket_idx: None,
            bridging_waters: Vec::new(),
            kept_water_resseqs: BTreeSet::new(),
            grid_box: gbox,
            scoring: ScoringFunction::Vina,
            exhaustiveness: 8,
            metal_coordination: true,
            hyde_rerank: true,
            constraints: Vec::new(),
            site_blueprint: None,
            blueprint_use_holo: true,
            new_pharm_kind: PharmacophoreKind::HAcceptor,
            new_pharm_radius: 2.0,
            flex_residues_input: String::new(),
            dock_poses: Vec::new(),
            selected_pose_idx: 0,
            campaign_report: None,
            hyde_report: hyde,
            vina_score_kcal: vina_e,
            admet_preview: None,
            selected_atom_idx: Some(2),
            inspirator_cat: InspiratorCategory::HalogensSmall,
            last_mutation_delta: None,
            status_line: format!(
                "Ready — DockCraft v{APP_VERSION} (AutoDock4Zn Metal + SeeSAR HYDE v2 + PoseBusters + Craft Suite)"
            ),
            palette_open: false,
            palette_query: String::new(),
            viewport_opts: ViewportOptions::default(),
            camera: cam,
            shared_payload: Arc::new(RwLock::new(SharedViewportPayload::default())),
            scene_dirty: true,
        };
        app.detect_pockets_now();
        if let Some(path) = startup_file {
            app.open_startup_path(&path);
        }
        app
    }

    fn open_startup_path(&mut self, path: &Path) {
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let s = path.display().to_string();
        match ext.as_str() {
            "dockcraft" => self.load_project_file(path),
            "pdb" | "cif" | "mmcif" | "pdbqt" => {
                self.receptor_path_input = s;
                self.load_receptor_from_path();
            }
            _ => {
                self.ligand_path_input = s;
                self.load_ligand_from_input();
            }
        }
    }

    fn rescore_active_complex(&mut self) {
        if let (Some(rec), Some(lig)) = (&self.receptor, &self.ligand) {
            self.hyde_report = score_pose_hyde(rec, lig).ok();
            self.vina_score_kcal = Some(score_complex(rec, lig, self.scoring).total_affinity_kcal);
        }
    }

    fn rebuild_receptor_with_kept_waters(&mut self) {
        let Some(raw) = &self.raw_receptor_with_waters else {
            return;
        };
        let mut rec = raw.clone();
        let ids: Vec<i32> = self.kept_water_resseqs.iter().copied().collect();
        prepare_receptor_keeping_waters(&mut rec, &ids);
        self.receptor = Some(rec);
        self.rescore_active_complex();
        self.scene_dirty = true;
    }

    fn detect_waters_now(&mut self) {
        let Some(raw) = &self.raw_receptor_with_waters else {
            self.status_line =
                "Load a PDB/mmCIF receptor first to scan bridging waters.".to_string();
            return;
        };
        let rad = 0.5
            * self.grid_box.size[0]
                .max(self.grid_box.size[1])
                .max(self.grid_box.size[2]);
        self.bridging_waters = detect_bridging_waters(raw, self.grid_box.center, rad.max(10.0), 2);
        self.status_line = format!(
            "Detected {} structural bridging water(s) (>= 2 receptor H-bonds) inside GridBox.",
            self.bridging_waters.len()
        );
    }

    fn detect_pockets_now(&mut self) {
        let Some(rec) = &self.receptor else {
            self.status_line = "Load a receptor first.".to_string();
            return;
        };
        match detect_pockets(rec, &PocketDetectConfig::default()) {
            Ok(p) => {
                let count = p.len();
                self.pockets = p;
                if let Some(top) = self.pockets.first() {
                    self.selected_pocket_idx = Some(0);
                    self.grid_box = top.grid_box.clone();
                }
                self.detect_waters_now();
                self.status_line = format!(
                    "Detected {count} druggable pocket(s) (LIGSITE-csc 26-ray + metal-aware)."
                );
                self.scene_dirty = true;
            }
            Err(e) => self.status_line = format!("Pocket detection error: {e}"),
        }
    }

    fn parsed_flex_residues(&self) -> Vec<String> {
        self.flex_residues_input
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToString::to_string)
            .collect()
    }

    fn pharm_kind_rgba(kind: PharmacophoreKind) -> [f32; 4] {
        match kind {
            PharmacophoreKind::HDonor => [0.20, 0.65, 1.0, 0.28],
            PharmacophoreKind::HAcceptor => [1.0, 0.30, 0.30, 0.28],
            PharmacophoreKind::Aromatic => [0.95, 0.55, 0.18, 0.28],
            PharmacophoreKind::Hydrophobic => [0.95, 0.85, 0.25, 0.25],
            PharmacophoreKind::PositiveIonizable => [0.28, 0.45, 1.0, 0.30],
            PharmacophoreKind::NegativeIonizable => [1.0, 0.25, 0.65, 0.30],
            PharmacophoreKind::MetalChelate => [0.88, 0.35, 0.98, 0.32],
            PharmacophoreKind::AnyHeavy => [0.65, 0.90, 0.70, 0.22],
        }
    }

    fn sync_pharmacophore_visuals(&mut self) {
        let mut spheres: Vec<([f64; 3], f64, [f32; 4])> = self
            .constraints
            .iter()
            .filter_map(|c| match c {
                DockingConstraint::PharmacophoreSphere {
                    center,
                    radius,
                    kind,
                    ..
                } => Some((*center, *radius, Self::pharm_kind_rgba(*kind))),
                _ => None,
            })
            .collect();
        let mut vectors: Vec<([f64; 3], [f64; 3], [f32; 4])> = Vec::new();
        let mut excl: Vec<([f64; 3], f64)> = Vec::new();
        if let Some(bp) = &self.site_blueprint {
            if spheres.is_empty() {
                for f in bp.features.iter().filter(|f| f.enabled) {
                    spheres.push((f.center, f.radius_a, Self::pharm_kind_rgba(f.kind)));
                }
            }
            for f in bp.features.iter().filter(|f| f.enabled) {
                if let Some(dir) = f.direction {
                    let tip = [
                        f.center[0] + dir[0] * 1.85,
                        f.center[1] + dir[1] * 1.85,
                        f.center[2] + dir[2] * 1.85,
                    ];
                    let mut col = Self::pharm_kind_rgba(f.kind);
                    col[3] = 0.95;
                    vectors.push((f.center, tip, col));
                }
            }
            excl = bp
                .exclusion_spheres
                .iter()
                .map(|e| (e.center, e.radius_a))
                .collect();
        }
        self.viewport_opts.pharmacophore_spheres = spheres;
        self.viewport_opts.pharmacophore_vectors = vectors;
        self.viewport_opts.exclusion_spheres = excl;
        self.scene_dirty = true;
    }

    fn analyze_active_site_now(&mut self) {
        let Some(rec) = self
            .raw_receptor_with_waters
            .as_ref()
            .or(self.receptor.as_ref())
        else {
            self.status_line = "Load a receptor before analyzing the active site.".to_string();
            return;
        };
        let holo_vec: Vec<&Molecule3D> = if self.blueprint_use_holo {
            self.ligand.iter().collect()
        } else {
            Vec::new()
        };
        match analyze_binding_site(
            rec,
            &self.grid_box,
            &holo_vec,
            &SiteAnalysisConfig::default(),
        ) {
            Ok(bp) => {
                self.status_line = format!(
                    "Site Blueprint: {:.0} A3 ({:.1}x{:.1}x{:.1} A) | {} hotspots | {} optimal distances ({} ms)",
                    bp.geometry.accessible_volume_a3,
                    bp.geometry.principal_dims_a[0],
                    bp.geometry.principal_dims_a[1],
                    bp.geometry.principal_dims_a[2],
                    bp.features.len(),
                    bp.distance_matrix.len(),
                    bp.elapsed_ms
                );
                self.site_blueprint = Some(bp);
                self.sync_pharmacophore_visuals();
            }
            Err(e) => self.status_line = format!("Active-site MIF analysis error: {e}"),
        }
    }

    fn apply_blueprint_to_docking(&mut self) {
        let Some(bp) = &self.site_blueprint else {
            self.status_line = "Run 'Analyze Active Site (MIFs)' first.".to_string();
            return;
        };
        self.constraints = bp.to_docking_constraints(6, 0.9);
        self.sync_pharmacophore_visuals();
        self.status_line = format!(
            "Applied {} enabled blueprint features as soft pharmacophore docking restraints.",
            self.constraints.len()
        );
    }

    fn load_blueprint_dialog(&mut self) {
        let Some(path) = pick_open_file(
            "Load Active-Site Blueprint (.pharm.json)",
            "Site Blueprints",
            "*.pharm.json *.json",
        ) else {
            return;
        };
        let Ok(raw) = std::fs::read_to_string(&path) else {
            self.status_line = format!("Cannot read blueprint '{path}'");
            return;
        };
        match SiteBlueprint::from_json_str(&raw) {
            Ok(bp) => {
                self.grid_box = bp.grid_box.clone();
                self.status_line = format!(
                    "Loaded blueprint '{}' ({} features, {} distances).",
                    bp.target_name,
                    bp.features.len(),
                    bp.distance_matrix.len()
                );
                self.site_blueprint = Some(bp);
                self.sync_pharmacophore_visuals();
            }
            Err(e) => self.status_line = format!("Invalid blueprint JSON: {e}"),
        }
    }

    fn export_blueprint_dialog(&mut self) {
        let Some(bp) = &self.site_blueprint else {
            self.status_line = "Run 'Analyze Active Site (MIFs)' before exporting.".to_string();
            return;
        };
        let Some(path_str) = pick_save_file(
            "Export Active-Site Blueprint",
            "site_blueprint.pharm.json",
            "Site Blueprint (.pharm.json / .md)",
            "*.pharm.json *.md",
        ) else {
            return;
        };
        let path = PathBuf::from(&path_str);
        let write_res = if path_str.ends_with(".md") {
            std::fs::write(&path, bp.render_markdown())
        } else {
            match bp.to_json_pretty() {
                Ok(js) => {
                    let md_path = path.with_extension("md");
                    let _ = std::fs::write(&md_path, bp.render_markdown());
                    std::fs::write(&path, js)
                }
                Err(e) => {
                    self.status_line = format!("Blueprint serialize error: {e}");
                    return;
                }
            }
        };
        match write_res {
            Ok(()) => self.status_line = format!("Exported Site Blueprint -> {}", path.display()),
            Err(e) => self.status_line = format!("Blueprint export failed: {e}"),
        }
    }

    fn run_docking_now(&mut self) {
        let (Some(rec), Some(lig)) = (&self.receptor, &self.ligand) else {
            self.status_line = "Need both receptor and ligand loaded.".to_string();
            return;
        };
        let cfg = DockingConfig {
            scoring: self.scoring,
            exhaustiveness: self.exhaustiveness,
            metal_coordination: self.metal_coordination,
            constraints: self.constraints.clone(),
            flex_residues: self.parsed_flex_residues(),
            ..Default::default()
        };
        match dock_ligand(rec, lig, &self.grid_box, &cfg) {
            Ok(mut rep) => {
                let eval_rec = rep.flex_receptor.clone().unwrap_or_else(|| rec.clone());
                if self.hyde_rerank && rep.poses.len() > 1 {
                    let mut scored: Vec<(f64, DockingPose)> = rep
                        .poses
                        .into_iter()
                        .map(|p| {
                            let hyde_dg = score_pose_hyde(&eval_rec, &p.molecule)
                                .map_or(p.affinity_kcal_mol, |h| h.total_delta_g_kcal);
                            let cons = consensus_affinity_kcal(
                                p.affinity_kcal_mol,
                                p.affinity_kcal_mol,
                                hyde_dg,
                            );
                            (cons, p)
                        })
                        .collect();
                    scored.sort_by(|a, b| a.0.total_cmp(&b.0));
                    rep.poses = scored
                        .into_iter()
                        .enumerate()
                        .map(|(i, (_, mut p))| {
                            p.mode = i + 1;
                            p
                        })
                        .collect();
                }
                if let Some(flex_rec) = rep.flex_receptor {
                    self.receptor = Some(flex_rec);
                }
                if let Some(best) = rep.poses.first() {
                    self.ligand = Some(best.molecule.clone());
                    self.vina_score_kcal = Some(best.affinity_kcal_mol);
                    if let Some(active_rec) = &self.receptor {
                        self.hyde_report = score_pose_hyde(active_rec, &best.molecule).ok();
                    }
                    self.admet_preview = None;
                }
                let pb_tag = self
                    .hyde_report
                    .as_ref()
                    .and_then(|h| h.posebusters.as_ref())
                    .map_or("", |pb| if pb.valid { " | PB PASS" } else { " | PB WARN" });
                self.status_line = format!(
                    "Docked {} modes in {} ms (best: {:.2} kcal/mol{pb_tag}).",
                    rep.poses.len(),
                    rep.elapsed_ms,
                    rep.poses.first().map_or(0.0, |p| p.affinity_kcal_mol)
                );
                self.dock_poses = rep.poses;
                self.selected_pose_idx = 0;
                self.scene_dirty = true;
            }
            Err(e) => self.status_line = format!("Docking error: {e}"),
        }
    }

    fn run_campaign_now(&mut self) {
        let Some(rec) = &self.receptor else {
            return;
        };
        let path = self.ligand_path_input.trim();
        let records = if !path.is_empty() && Path::new(path).exists() {
            match std::fs::read(path) {
                Ok(bytes) => {
                    let ext = Path::new(path)
                        .extension()
                        .and_then(|s| s.to_str())
                        .unwrap_or("smi");
                    match load_ligand_library_bytes(ext, &bytes, 250) {
                        Ok(r) => r,
                        Err(e) => {
                            self.status_line = format!("Library error: {e}");
                            return;
                        }
                    }
                }
                Err(e) => {
                    self.status_line = format!("File read error: {e}");
                    return;
                }
            }
        } else if let Some(lig) = &self.ligand {
            vec![LigandInputRecord {
                name: lig.name.clone(),
                molecule: lig.clone(),
                is_active: Some(true),
            }]
        } else {
            return;
        };

        let cfg = DockingConfig {
            scoring: self.scoring,
            exhaustiveness: self.exhaustiveness.min(4),
            metal_coordination: self.metal_coordination,
            constraints: self.constraints.clone(),
            ..Default::default()
        };
        match run_screening_campaign(rec, &self.grid_box, &records, &cfg) {
            Ok(rep) => {
                self.status_line = format!(
                    "Campaign finished {} compounds in {} ms.",
                    rep.results.len(),
                    rep.elapsed_ms
                );
                if let Some(top) = rep.results.first() {
                    self.ligand = Some(top.best_pose.molecule.clone());
                    self.vina_score_kcal = Some(top.best_affinity_kcal);
                    self.hyde_report = Some(top.hyde_report.clone());
                }
                self.campaign_report = Some(rep);
                self.scene_dirty = true;
            }
            Err(e) => self.status_line = format!("Campaign failed: {e}"),
        }
    }

    fn open_receptor_dialog(&mut self) {
        if let Some(path) = pick_open_file(
            "Open Protein Receptor",
            "Receptors",
            "*.pdb *.cif *.mmcif *.pdbqt *.mol2",
        ) {
            self.receptor_path_input = path;
            self.load_receptor_from_path();
        }
    }

    fn open_ligand_dialog(&mut self) {
        if let Some(path) = pick_open_file(
            "Open Ligand or Library",
            "Ligands & Libraries",
            "*.sdf *.mol2 *.pdbqt *.smi *.ism *.csv *.chemcraft *.pdb",
        ) {
            self.ligand_path_input = path;
            self.load_ligand_from_input();
        }
    }

    fn load_receptor_from_path(&mut self) {
        let p = self.receptor_path_input.trim();
        let Ok(raw) = std::fs::read_to_string(p) else {
            self.status_line = format!("Cannot read receptor '{p}'");
            return;
        };
        let ext = Path::new(p)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("pdb")
            .to_ascii_lowercase();
        let parsed_raw = match ext.as_str() {
            "cif" | "mmcif" => parse_mmcif(&raw, false),
            "pdbqt" => parse_pdbqt(&raw),
            "mol2" => parse_mol2(&raw),
            _ => parse_pdb(&raw, false),
        };
        match parsed_raw {
            Ok(raw_rec) => {
                let mut rec = raw_rec.clone();
                prepare_receptor(&mut rec, true);
                if rec.heavy_atom_count() > 0 {
                    let c = rec.centroid_heavy();
                    self.camera
                        .focus_on([c[0] as f32, c[1] as f32, c[2] as f32], 18.0);
                }
                self.raw_receptor_with_waters = Some(raw_rec);
                self.raw_receptor_source = Some((raw, ext));
                self.prep_result = None;
                self.kept_water_resseqs.clear();
                self.receptor = Some(rec);
                self.detect_pockets_now();
                self.rescore_active_complex();
                self.scene_dirty = true;
            }
            Err(e) => self.status_line = format!("Receptor parse error: {e}"),
        }
    }

    fn run_prot_prep_now(&mut self) {
        let (raw_txt, fmt) = if let Some((txt, ext)) = &self.raw_receptor_source {
            (txt.clone(), ext.clone())
        } else if let Some(rec) = self
            .raw_receptor_with_waters
            .as_ref()
            .or(self.receptor.as_ref())
        {
            (write_pdb(rec), "pdb".to_string())
        } else {
            self.status_line = "Load a PDB/mmCIF receptor first to run ProtPrep.".to_string();
            return;
        };
        let chains: Vec<char> = self
            .prep_chains_input
            .split(',')
            .filter_map(|s| s.trim().chars().next())
            .collect();
        let cfg = PrepConfig {
            ph: self.prep_ph,
            chains,
            keep_cofactors: self.prep_keep_cofactors,
            keep_metals: self.prep_keep_metals,
            strip_ligands: true,
            keep_bridging_waters: self.prep_keep_bridging_waters,
            optimize_hbonds: self.prep_optimize_hbonds,
            pocket_center: Some(self.grid_box.center),
            pocket_radius: 10.0,
            ..Default::default()
        };
        match prepare_protein_from_str(&raw_txt, &fmt, &cfg) {
            Ok(prep) => {
                self.receptor = Some(prep.receptor.clone());
                self.bridging_waters = prep.bridging_waters.clone();
                if self.ligand.is_none()
                    && let Some(ext_lig) = prep.extracted_ligands.first()
                {
                    self.ligand = Some(ext_lig.clone());
                }
                self.status_line = format!(
                    "ProtPrep {}: Grade {} | pH {:.1} (q={:+}) | {} flips | {} polar-H rotors ({} ms)",
                    prep.audit.structure_id,
                    prep.audit.quality_grade,
                    prep.protonation.target_ph,
                    prep.protonation.total_protein_charge,
                    prep.hbond_opt.flips_applied.len(),
                    prep.hbond_opt.polar_h_rotors_optimized,
                    prep.elapsed_ms
                );
                self.prep_result = Some(prep);
                self.rescore_active_complex();
                self.scene_dirty = true;
            }
            Err(e) => self.status_line = format!("ProtPrep error: {e}"),
        }
    }

    fn export_prep_audit_dialog(&mut self) {
        let Some(prep) = &self.prep_result else {
            self.status_line = "Run ProtPrep before exporting the audit report.".to_string();
            return;
        };
        let Some(path) = pick_save_file(
            "Export Structural Audit Report (Markdown)",
            "receptor_audit.md",
            "Markdown Report",
            "*.md",
        ) else {
            return;
        };
        match std::fs::write(&path, prep.render_markdown()) {
            Ok(()) => self.status_line = format!("Exported receptor audit -> {path}"),
            Err(e) => self.status_line = format!("Audit export failed: {e}"),
        }
    }

    fn load_ligand_from_input(&mut self) {
        let inp = self.ligand_path_input.trim();
        if inp.is_empty() {
            return;
        }
        let res = if Path::new(inp).exists() {
            match std::fs::read_to_string(inp) {
                Ok(raw) => {
                    let ext = Path::new(inp)
                        .extension()
                        .and_then(|s| s.to_str())
                        .unwrap_or("sdf")
                        .to_ascii_lowercase();
                    match ext.as_str() {
                        "mol2" => parse_mol2(&raw),
                        "pdbqt" => parse_pdbqt(&raw),
                        "pdb" => parse_pdb(&raw, false),
                        "smi" | "ism" => {
                            let first = raw
                                .lines()
                                .find(|l| !l.trim().is_empty())
                                .and_then(|l| l.split_whitespace().next())
                                .unwrap_or("");
                            parse_smiles_to_3d(first, Some("smiles_ligand"))
                        }
                        "chemcraft" | "csv" => {
                            let bytes = std::fs::read(inp).unwrap_or_default();
                            match load_ligand_library_bytes(&ext, &bytes, 1) {
                                Ok(mut list) if !list.is_empty() => Ok(list.remove(0).molecule),
                                _ => Err(struct_core::StructError::EmptyInput),
                            }
                        }
                        _ => parse_sdf_3d_with_limit(&raw, 1)
                            .pop()
                            .unwrap_or(Err(struct_core::StructError::EmptyInput)),
                    }
                }
                Err(e) => {
                    self.status_line = format!("Read error: {e}");
                    return;
                }
            }
        } else {
            parse_smiles_to_3d(inp, Some("smiles_ligand"))
        };

        match res {
            Ok(mut lig) => {
                perceive_all(&mut lig);
                if self.pockets.is_empty()
                    && let Ok(gb) = GridBox::from_ligand(&lig, 6.0)
                {
                    self.grid_box = gb;
                }
                self.ligand = Some(lig);
                self.admet_preview = None;
                self.rescore_active_complex();
                self.scene_dirty = true;
                self.status_line = "Ligand loaded.".to_string();
            }
            Err(e) => self.status_line = format!("Ligand error: {e}"),
        }
    }

    fn save_project(&mut self, force_dialog: bool) {
        let target = if force_dialog || self.current_project_path.is_none() {
            pick_save_file(
                "Save DockCraft Project",
                "workspace.dockcraft",
                "DockCraft Project",
                "*.dockcraft",
            )
            .map(PathBuf::from)
        } else {
            self.current_project_path.clone()
        };
        let Some(path) = target else {
            return;
        };
        let proj = DockCraftProject {
            format: "dockcraft-project".to_string(),
            version: APP_VERSION.to_string(),
            receptor: self.receptor.clone(),
            ligand: self.ligand.clone(),
            grid_box: self.grid_box.clone(),
            scoring: self.scoring,
            exhaustiveness: self.exhaustiveness,
            metal_coordination: self.metal_coordination,
            hyde_rerank: self.hyde_rerank,
            constraints: self.constraints.clone(),
            flex_residues: self.parsed_flex_residues(),
            kept_water_resseqs: self.kept_water_resseqs.iter().copied().collect(),
        };
        match serde_json::to_string_pretty(&proj)
            .and_then(|s| std::fs::write(&path, s).map_err(serde_json::Error::io))
        {
            Ok(()) => {
                self.current_project_path = Some(path.clone());
                self.status_line = format!("Saved project -> {}", path.display());
            }
            Err(e) => self.status_line = format!("Save failed: {e}"),
        }
    }

    fn load_project_file(&mut self, path: &Path) {
        let Ok(raw) = std::fs::read_to_string(path) else {
            self.status_line = format!("Cannot read project '{}'", path.display());
            return;
        };
        match serde_json::from_str::<DockCraftProject>(&raw) {
            Ok(proj) => {
                self.receptor = proj.receptor;
                self.raw_receptor_with_waters = self.receptor.clone();
                self.ligand = proj.ligand;
                self.grid_box = proj.grid_box;
                self.scoring = proj.scoring;
                self.exhaustiveness = proj.exhaustiveness;
                self.metal_coordination = proj.metal_coordination;
                self.hyde_rerank = proj.hyde_rerank;
                self.constraints = proj.constraints;
                self.flex_residues_input = proj.flex_residues.join(", ");
                self.kept_water_resseqs = proj.kept_water_resseqs.into_iter().collect();
                self.current_project_path = Some(path.to_path_buf());
                self.sync_pharmacophore_visuals();
                self.rescore_active_complex();
                self.status_line = format!("Loaded project {}", path.display());
            }
            Err(e) => self.status_line = format!("Invalid .dockcraft project: {e}"),
        }
    }

    fn export_poses_sdf(&mut self) {
        let Some(path) = pick_save_file(
            "Export Docked Poses (SDF)",
            "docked_poses.sdf",
            "SDF 3D Structures",
            "*.sdf",
        ) else {
            return;
        };
        let refs: Vec<&Molecule3D> = if !self.dock_poses.is_empty() {
            self.dock_poses.iter().map(|p| &p.molecule).collect()
        } else if let Some(lig) = &self.ligand {
            vec![lig]
        } else {
            self.status_line = "No ligand or docked poses to export.".to_string();
            return;
        };
        match std::fs::write(&path, write_sdf_3d(&refs)) {
            Ok(()) => self.status_line = format!("Exported {} pose(s) -> {path}", refs.len()),
            Err(e) => self.status_line = format!("SDF export failed: {e}"),
        }
    }

    fn export_receptor_pdb(&mut self) {
        let Some(rec) = &self.receptor else {
            self.status_line = "No receptor loaded.".to_string();
            return;
        };
        let Some(path) = pick_save_file(
            "Export Prepared Receptor (PDB)",
            "receptor_prepared.pdb",
            "PDB Structure",
            "*.pdb",
        ) else {
            return;
        };
        match std::fs::write(&path, write_pdb(rec)) {
            Ok(()) => self.status_line = format!("Exported receptor -> {path}"),
            Err(e) => self.status_line = format!("PDB export failed: {e}"),
        }
    }

    fn export_campaign_csv(&mut self) {
        let Some(rep) = &self.campaign_report else {
            self.status_line = "Run a virtual screening campaign first.".to_string();
            return;
        };
        let Some(path) = pick_save_file(
            "Export Screening Results (CSV)",
            "screening_results.csv",
            "CSV Table",
            "*.csv",
        ) else {
            return;
        };
        match std::fs::write(&path, campaign_report_to_csv(rep)) {
            Ok(()) => self.status_line = format!("Exported campaign CSV -> {path}"),
            Err(e) => self.status_line = format!("CSV export failed: {e}"),
        }
    }

    fn open_in_admetcraft(&mut self) {
        let mut entries: Vec<(String, String)> = Vec::new();
        if let Some(rep) = &self.campaign_report {
            for cand in rep.results.iter().take(100) {
                if let Some(smi) = cand
                    .source_properties
                    .get("ChemCraft_canonical_smiles")
                    .or(cand.best_pose.molecule.source_smiles.as_ref())
                {
                    entries.push((smi.clone(), cand.name.clone()));
                }
            }
        }
        if entries.is_empty()
            && let Some(lig) = &self.ligand
            && let Some(smi) = &lig.source_smiles
        {
            entries.push((smi.clone(), lig.name.clone()));
        }
        match launch_admetcraft_handoff(&entries) {
            Ok(msg) => self.status_line = msg,
            Err(err) => self.status_line = err,
        }
    }

    fn open_in_chemcraft(&mut self) {
        let dir = default_chemcraft_results_dir();
        if std::fs::create_dir_all(&dir).is_err() {
            self.status_line = format!("Cannot create ChemCraft handoff dir {}", dir.display());
            return;
        }
        if let Some(rep) = &self.campaign_report {
            let rec_name = self
                .receptor
                .as_ref()
                .map_or("receptor", |r| r.name.as_str());
            match write_results_sidecar(
                rep,
                rec_name,
                &self.grid_box,
                &format!("{:?}", self.scoring),
                self.exhaustiveness,
                &dir,
                "dockcraft_gui",
            ) {
                Ok(path) => match launch_chemcraft_with_path(&path) {
                    Ok(msg) => self.status_line = msg,
                    Err(e) => self.status_line = e,
                },
                Err(e) => self.status_line = format!("ChemCraft sidecar error: {e}"),
            }
        } else if let Some(lig) = &self.ligand {
            let sdf_path = dir.join("dockcraft_pose.sdf");
            if std::fs::write(&sdf_path, write_sdf_3d(&[lig])).is_ok() {
                match launch_chemcraft_with_path(&sdf_path) {
                    Ok(msg) => self.status_line = msg,
                    Err(e) => self.status_line = e,
                }
            }
        } else {
            self.status_line = "No ligand or screening results to send to ChemCraft.".to_string();
        }
    }

    fn launch_chimerax_now(&mut self) {
        let (Some(rec), Some(lig), Some(hyde)) = (&self.receptor, &self.ligand, &self.hyde_report)
        else {
            self.status_line = "Load and score a complex before launching ChimeraX.".to_string();
            return;
        };
        let cname = if lig.name.trim().is_empty() {
            "DockCraftComplex"
        } else {
            lig.name.as_str()
        };
        match launch_chimerax_for_scene(cname, rec, lig, hyde, None) {
            Ok(b) => {
                self.status_line = format!(
                    "ChimeraX opened with REST (127.0.0.1:60958) -> {} | Select residues & click 'Box <- ChimeraX'",
                    b.script_path.display()
                );
            }
            Err(e) => self.status_line = format!("ChimeraX error: {e}"),
        }
    }

    fn grab_box_from_chimerax(&mut self) {
        let root = default_chimerax_export_dir();
        let _ = std::fs::create_dir_all(&root);
        let client = ChimeraXRest::with_default_port(root.clone());
        match grid_box_from_chimerax_selection(&client, &root, 5.0) {
            Ok(sel) => {
                self.grid_box = sel.grid_box;
                self.camera.focus_on(
                    [
                        self.grid_box.center[0] as f32,
                        self.grid_box.center[1] as f32,
                        self.grid_box.center[2] as f32,
                    ],
                    16.0,
                );
                self.detect_waters_now();
                self.scene_dirty = true;
                self.status_line = format!(
                    "GridBox synced from ChimeraX ({} residues, {} atoms).",
                    sel.residues.len(),
                    sel.heavy_atoms
                );
            }
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("not answering") {
                    self.status_line = "ChimeraX REST (127.0.0.1:60958) not answering — launch via 'ChimeraX' button or run in ChimeraX: remotecontrol rest start port 60958 json true".to_string();
                } else {
                    self.status_line = format!("ChimeraX REST box error: {msg}");
                }
            }
        }
    }

    fn apply_mutation(&mut self, preset: SubstituentPreset) {
        let (Some(rec), Some(lig), Some(anchor)) =
            (&self.receptor, &self.ligand, self.selected_atom_idx)
        else {
            self.status_line = "Click a ligand atom first to apply in-pocket mutation.".to_string();
            return;
        };
        match mutate_and_minimize_in_pocket(rec, lig, anchor, preset) {
            Ok(out) => {
                self.pose_history.push(lig.clone());
                self.vina_score_kcal =
                    Some(score_complex(rec, &out.mutated_ligand, self.scoring).total_affinity_kcal);
                self.ligand = Some(out.mutated_ligand);
                self.admet_preview = None;
                self.last_mutation_delta =
                    Some((preset.label().to_string(), out.delta_delta_g_kcal));
                self.hyde_report = Some(out.report);
                self.status_line = format!(
                    "Mutated atom #{anchor} -> {}: dG {:+.2} kcal/mol ({} ms)",
                    preset.label(),
                    out.delta_delta_g_kcal,
                    out.elapsed_ms
                );
                self.scene_dirty = true;
            }
            Err(e) => self.status_line = format!("Mutation failed: {e}"),
        }
    }

    fn export_interop_dialog(&mut self, target_pymol: bool) {
        let (Some(rec), Some(lig), Some(hyde)) = (&self.receptor, &self.ligand, &self.hyde_report)
        else {
            self.status_line = "Score a complex before exporting.".to_string();
            return;
        };
        let (title, def, label, glob) = if target_pymol {
            (
                "Export PyMOL Scene",
                "dockcraft_scene.pml",
                "PyMOL Script",
                "*.pml",
            )
        } else {
            (
                "Export ChimeraX Scene",
                "dockcraft_scene.cxc",
                "ChimeraX Script",
                "*.cxc",
            )
        };
        let Some(script_str) = pick_save_file(title, def, label, glob) else {
            return;
        };
        let script_path = PathBuf::from(&script_str);
        let dir = script_path.parent().unwrap_or_else(|| Path::new("."));
        let (script_txt, bundle, json_ext) = if target_pymol {
            let (s, b) = build_pymol_export(&lig.name, rec, lig, hyde);
            (s, b, "pymol.json")
        } else {
            let (s, b) = build_chimerax_export(&lig.name, rec, lig, hyde);
            (s, b, "chimerax.json")
        };
        let p_rec = dir.join("receptor.pdb");
        let p_lig = dir.join("ligand.sdf");
        let p_json = dir.join(format!("dockcraft_scene.{json_ext}"));
        let ok = std::fs::write(&p_rec, &bundle.receptor_pdb).is_ok()
            && std::fs::write(&p_lig, &bundle.ligand_sdf).is_ok()
            && std::fs::write(&script_path, script_txt).is_ok()
            && serde_json::to_string_pretty(&bundle)
                .map(|js| std::fs::write(&p_json, js).is_ok())
                .unwrap_or(false);
        self.status_line = if ok {
            format!("Exported scene -> {}", script_path.display())
        } else {
            "Scene export failed.".to_string()
        };
    }

    fn sync_viewport_payload(&mut self) {
        let sel_pocket = self.selected_pocket_idx.and_then(|i| self.pockets.get(i));
        let scene = if self.scene_dirty {
            self.scene_dirty = false;
            Some(build_dock_scene(
                self.receptor.as_ref(),
                self.ligand.as_ref(),
                sel_pocket,
                self.hyde_report.as_ref(),
                Some(&self.grid_box),
                self.selected_atom_idx,
                &self.viewport_opts,
            ))
        } else {
            None
        };
        if let Ok(mut guard) = self.shared_payload.write() {
            if let Some(s) = scene {
                guard.scene = s;
            }
            guard.camera = self.camera.clone();
            guard.background = self.viewport_opts.background_rgba;
        }
    }
}

impl eframe::App for DockCraftApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Craft keyboard shortcuts
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::K)) {
            self.palette_open = !self.palette_open;
        }
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::O)) {
            self.open_receptor_dialog();
        }
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::S)) {
            let shift = ctx.input(|i| i.modifiers.shift);
            self.save_project(shift);
        }
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::D)) {
            self.run_docking_now();
        }
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::Z))
            && let Some(prev) = self.pose_history.pop()
        {
            self.ligand = Some(prev);
            self.rescore_active_complex();
            self.scene_dirty = true;
        }

        self.sync_viewport_payload();

        // 1. Unified Craft Menu Bar (matches ChemCraft v0.2.1 & AdmetCraft v0.3.1)
        egui::TopBottomPanel::top("craft_menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.strong("DockCraft");
                ui.label(RichText::new(format!("v{APP_VERSION}")).small().weak());
                ui.separator();

                ui.menu_button("File", |ui| {
                    if ui.button("Open Receptor (.pdb/.cif)... (Ctrl+O)").clicked() {
                        self.open_receptor_dialog();
                        ui.close_menu();
                    }
                    if ui
                        .button("Open Ligand / Library (.sdf/.smi/.chemcraft)...")
                        .clicked()
                    {
                        self.open_ligand_dialog();
                        ui.close_menu();
                    }
                    if ui.button("Open Project (.dockcraft)...").clicked() {
                        if let Some(p) = pick_open_file(
                            "Open DockCraft Project",
                            "DockCraft Project",
                            "*.dockcraft",
                        ) {
                            self.load_project_file(Path::new(&p));
                        }
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Save Project (Ctrl+S)").clicked() {
                        self.save_project(false);
                        ui.close_menu();
                    }
                    if ui.button("Save Project As... (Ctrl+Shift+S)").clicked() {
                        self.save_project(true);
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.menu_button("Export Structures / Data", |ui| {
                        if ui.button("Docked Poses (.sdf)...").clicked() {
                            self.export_poses_sdf();
                            ui.close_menu();
                        }
                        if ui.button("Prepared Receptor (.pdb)...").clicked() {
                            self.export_receptor_pdb();
                            ui.close_menu();
                        }
                        if ui.button("Screening Results (.csv)...").clicked() {
                            self.export_campaign_csv();
                            ui.close_menu();
                        }
                        ui.separator();
                        if ui.button("PyMOL Scene (.pml)...").clicked() {
                            self.export_interop_dialog(true);
                            ui.close_menu();
                        }
                        if ui.button("ChimeraX Scene (.cxc)...").clicked() {
                            self.export_interop_dialog(false);
                            ui.close_menu();
                        }
                    });
                    ui.separator();
                    if ui.button("Load Demo Showcase (1STP)").clicked() {
                        let (rec, lig) = build_demo_showcase();
                        self.raw_receptor_with_waters = Some(rec.clone());
                        self.receptor = Some(rec);
                        self.ligand = Some(lig);
                        self.kept_water_resseqs.clear();
                        self.constraints.clear();
                        self.detect_pockets_now();
                        self.rescore_active_complex();
                        self.scene_dirty = true;
                        ui.close_menu();
                    }
                });

                ui.menu_button("Edit", |ui| {
                    if ui
                        .add_enabled(
                            !self.pose_history.is_empty(),
                            egui::Button::new("Undo Mutation (Ctrl+Z)"),
                        )
                        .clicked()
                    {
                        if let Some(prev) = self.pose_history.pop() {
                            self.ligand = Some(prev);
                            self.rescore_active_complex();
                            self.scene_dirty = true;
                        }
                        ui.close_menu();
                    }
                    if ui.button("Center GridBox on Ligand").clicked() {
                        if let Some(lig) = &self.ligand
                            && let Ok(gb) = GridBox::from_ligand(lig, 6.0)
                        {
                            self.grid_box = gb;
                            self.scene_dirty = true;
                        }
                        ui.close_menu();
                    }
                    if ui.button("Clear Pharmacophore Constraints").clicked() {
                        self.constraints.clear();
                        self.sync_pharmacophore_visuals();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Command Palette (Ctrl+K)").clicked() {
                        self.palette_open = true;
                        ui.close_menu();
                    }
                });

                ui.menu_button("View", |ui| {
                    let mut changed = false;
                    changed |= ui
                        .checkbox(
                            &mut self.viewport_opts.show_receptor_backbone,
                            "Receptor Backbone",
                        )
                        .changed();
                    changed |= ui
                        .checkbox(
                            &mut self.viewport_opts.show_pocket_residues,
                            "Pocket Residues",
                        )
                        .changed();
                    changed |= ui
                        .checkbox(&mut self.viewport_opts.show_pocket_cloud, "Pocket Cloud")
                        .changed();
                    changed |= ui
                        .checkbox(&mut self.viewport_opts.show_hyde_coronas, "HYDE Coronas")
                        .changed();
                    changed |= ui
                        .checkbox(
                            &mut self.viewport_opts.show_torsion_lights,
                            "Torsion Traffic-Lights",
                        )
                        .changed();
                    changed |= ui
                        .checkbox(
                            &mut self.viewport_opts.show_waters_metals,
                            "Waters & Metals",
                        )
                        .changed();
                    changed |= ui
                        .checkbox(&mut self.viewport_opts.show_interactions, "3D Interactions")
                        .changed();
                    changed |= ui
                        .checkbox(&mut self.viewport_opts.show_grid_box, "Docking GridBox")
                        .changed();
                    changed |= ui
                        .checkbox(
                            &mut self.viewport_opts.show_exclusion_spheres,
                            "Site Exclusion Spheres",
                        )
                        .changed();
                    if changed {
                        self.scene_dirty = true;
                    }
                    ui.separator();
                    if ui.button("Reset 3D Camera").clicked() {
                        self.camera = Camera3D::default();
                        let c = self.grid_box.center;
                        self.camera
                            .focus_on([c[0] as f32, c[1] as f32, c[2] as f32], 16.0);
                        ui.close_menu();
                    }
                });

                ui.menu_button("Docking", |ui| {
                    if ui
                        .button("Prepare & Audit Receptor (pKa / H-Bond Flips)")
                        .clicked()
                    {
                        self.run_prot_prep_now();
                        ui.close_menu();
                    }
                    if ui.button("Detect Binding Pockets").clicked() {
                        self.detect_pockets_now();
                        ui.close_menu();
                    }
                    if ui
                        .button("Analyze Active Site (MIFs & Distances)")
                        .clicked()
                    {
                        self.analyze_active_site_now();
                        ui.close_menu();
                    }
                    if ui.button("Load Site Blueprint (.pharm.json)...").clicked() {
                        self.load_blueprint_dialog();
                        ui.close_menu();
                    }
                    if ui
                        .button("Export Site Blueprint (.pharm.json / .md)...")
                        .clicked()
                    {
                        self.export_blueprint_dialog();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Detect Bridging Waters").clicked() {
                        self.detect_waters_now();
                        ui.close_menu();
                    }
                    if ui.button("Run Docking (Ctrl+D)").clicked() {
                        self.run_docking_now();
                        ui.close_menu();
                    }
                    if ui.button("Run Virtual Screening").clicked() {
                        self.run_campaign_now();
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.checkbox(
                        &mut self.metal_coordination,
                        "AutoDock4Zn Metal Coordination",
                    );
                    ui.checkbox(&mut self.hyde_rerank, "HYDE v2 + Torsion Consensus Re-rank");
                });

                ui.menu_button("Apps", |ui| {
                    if ui.button("Abrir en AdmetCraft (ADMET & PK)").clicked() {
                        self.open_in_admetcraft();
                        ui.close_menu();
                    }
                    if ui.button("Enviar a ChemCraft (2D/3D SAR)").clicked() {
                        self.open_in_chemcraft();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Visualizar en ChimeraX").clicked() {
                        self.launch_chimerax_now();
                        ui.close_menu();
                    }
                    if ui.button("Tomar caja desde ChimeraX").clicked() {
                        self.grab_box_from_chimerax();
                        ui.close_menu();
                    }
                });
            });
        });

        // 2. Single-Row Quick Toolbar
        egui::TopBottomPanel::top("quick_toolbar").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui.button("Open Rec").clicked() {
                    self.open_receptor_dialog();
                }
                if ui.button("Open Lig").clicked() {
                    self.open_ligand_dialog();
                }
                if ui.button("Save").clicked() {
                    self.save_project(false);
                }
                ui.separator();
                if ui.button("Pockets").clicked() {
                    self.detect_pockets_now();
                }
                if ui.button(RichText::new("Dock").strong()).clicked() {
                    self.run_docking_now();
                }
                if ui.button("Screen").clicked() {
                    self.run_campaign_now();
                }
                egui::ComboBox::from_id_salt("sf_combo")
                    .selected_text(match self.scoring {
                        ScoringFunction::Vina => "Vina",
                        ScoringFunction::Vinardo => "Vinardo",
                        ScoringFunction::DockCraftHybrid => "Hybrid",
                    })
                    .width(74.0)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.scoring, ScoringFunction::Vina, "Vina");
                        ui.selectable_value(&mut self.scoring, ScoringFunction::Vinardo, "Vinardo");
                        ui.selectable_value(
                            &mut self.scoring,
                            ScoringFunction::DockCraftHybrid,
                            "Hybrid",
                        );
                    });
                ui.add(egui::Slider::new(&mut self.exhaustiveness, 1..=16).text("Exh"));
                ui.checkbox(&mut self.hyde_rerank, "HYDE");
                ui.separator();
                if ui.button(RichText::new("ChemCraft").strong()).clicked() {
                    self.open_in_chemcraft();
                }
                if ui.button(RichText::new("AdmetCraft").strong()).clicked() {
                    self.open_in_admetcraft();
                }
                if ui.button("ChimeraX").clicked() {
                    self.launch_chimerax_now();
                }
                if ui.button("Box <- ChimeraX").clicked() {
                    self.grab_box_from_chimerax();
                }
            });
        });

        // 3. Bottom Screening Table + Status Bar with Version
        egui::TopBottomPanel::bottom("bottom_campaign")
            .resizable(true)
            .default_height(150.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.strong(format!("DockCraft v{APP_VERSION}"));
                    ui.separator();
                    ui.label(&self.status_line);
                    if let Some(rep) = &self.campaign_report
                        && let Some(enr) = &rep.enrichment
                    {
                        ui.separator();
                        ui.label(format!(
                            "ROC AUC: {:.3} | BEDROC20: {:.3} | EF1%: {:.2} | EF5%: {:.2}",
                            enr.roc_auc, enr.bedroc_20, enr.ef_1_percent, enr.ef_5_percent
                        ));
                    }
                });
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    egui::Grid::new("mpo_table").striped(true).show(ui, |ui| {
                        for h in [
                            "Rank", "Compound", "Vina dG", "HYDE dG", "Ki (nM)", "LE", "MPO",
                        ] {
                            ui.strong(h);
                        }
                        ui.end_row();
                        if let Some(rep) = &self.campaign_report {
                            for cand in &rep.results {
                                if ui
                                    .selectable_label(false, format!("#{}", cand.rank))
                                    .clicked()
                                {
                                    self.ligand = Some(cand.best_pose.molecule.clone());
                                    self.vina_score_kcal = Some(cand.best_affinity_kcal);
                                    self.hyde_report = Some(cand.hyde_report.clone());
                                    self.scene_dirty = true;
                                }
                                ui.label(&cand.name);
                                ui.label(format!("{:.2}", cand.best_affinity_kcal));
                                ui.label(format!("{:.2}", cand.hyde_delta_g_kcal));
                                ui.label(format_ki_nm(cand.estimated_ki_nm));
                                ui.label(format!("{:.3}", cand.ligand_efficiency));
                                ui.label(format!("{:.3}", cand.mpo_score));
                                ui.end_row();
                            }
                        }
                    });
                });
            });

        // 4. Left Panel: Pockets, Waters, Advanced Modes, GridBox, Docked Modes
        egui::SidePanel::left("left_pyrx")
            .resizable(true)
            .default_width(265.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    // Quick SMILES / Path input
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut self.ligand_path_input)
                                .desired_width(175.0)
                                .hint_text("SMILES or ligand path..."),
                        );
                        if ui.button("Load").clicked() {
                            self.load_ligand_from_input();
                        }
                    });
                    ui.separator();

                    ui.heading("Binding Pockets");
                    let mut clicked_pocket: Option<usize> = None;
                    for (idx, p) in self.pockets.iter().take(6).enumerate() {
                        let sel = self.selected_pocket_idx == Some(idx);
                        if ui
                            .selectable_label(
                                sel,
                                format!(
                                    "#{} Drug {:.2} | {:.0} A3 | {:.0} A2",
                                    p.rank, p.druggability_score, p.volume_a3, p.sasa_a2
                                ),
                            )
                            .clicked()
                        {
                            clicked_pocket = Some(idx);
                        }
                    }
                    if let Some(idx) = clicked_pocket
                        && let Some(p) = self.pockets.get(idx).cloned()
                    {
                        self.selected_pocket_idx = Some(idx);
                        self.grid_box = p.grid_box;
                        self.camera.focus_on(
                            [p.center[0] as f32, p.center[1] as f32, p.center[2] as f32],
                            14.0,
                        );
                        self.detect_waters_now();
                        self.scene_dirty = true;
                    }

                    ui.separator();
                    egui::CollapsingHeader::new("Receptor Prep & Audit (pKa / Flips)")
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label("pH:");
                                ui.add(
                                    egui::DragValue::new(&mut self.prep_ph)
                                        .speed(0.1)
                                        .range(4.0..=10.0),
                                );
                                ui.label("Chains:");
                                ui.add(
                                    egui::TextEdit::singleline(&mut self.prep_chains_input)
                                        .desired_width(52.0)
                                        .hint_text("All"),
                                );
                            });
                            ui.horizontal_wrapped(|ui| {
                                ui.checkbox(&mut self.prep_keep_metals, "Metals");
                                ui.checkbox(&mut self.prep_keep_cofactors, "Cofactors");
                                ui.checkbox(&mut self.prep_optimize_hbonds, "H-Bond Flips");
                                ui.checkbox(&mut self.prep_keep_bridging_waters, "Bridge HOH");
                            });
                            ui.horizontal(|ui| {
                                if ui.button("Prepare & Audit").clicked() {
                                    self.run_prot_prep_now();
                                }
                                if self.prep_result.is_some()
                                    && ui.small_button("Export Audit (.md)").clicked()
                                {
                                    self.export_prep_audit_dialog();
                                }
                            });
                            if let Some(prep) = &self.prep_result {
                                let a = &prep.audit;
                                let g_col = if a.quality_grade.starts_with('A') {
                                    Color32::from_rgb(46, 204, 113)
                                } else if a.quality_grade.starts_with('B') {
                                    Color32::from_rgb(241, 196, 15)
                                } else {
                                    Color32::from_rgb(235, 77, 75)
                                };
                                ui.colored_label(g_col, RichText::new(&a.quality_grade).strong());
                                if let Some(res) = a.resolution_a {
                                    ui.small(format!(
                                        "Res: {res:.2} A | R-free: {} | Net q: {:+}",
                                        a.r_free.map_or_else(
                                            || "n/a".to_string(),
                                            |v| format!("{v:.3}")
                                        ),
                                        prep.protonation.total_protein_charge
                                    ));
                                } else if a.is_alphafold {
                                    ui.small(format!(
                                        "AlphaFold mean pLDDT: {:.1} | Net q: {:+}",
                                        a.mean_b_factor_or_plddt,
                                        prep.protonation.total_protein_charge
                                    ));
                                }
                                ui.small(format!(
                                    "Altlocs: {} | Missing atoms: {} | Flips: {} | OH rotors: {}",
                                    a.altlocs_resolved.len(),
                                    a.missing_heavy_atoms.len(),
                                    prep.hbond_opt.flips_applied.len(),
                                    prep.hbond_opt.polar_h_rotors_optimized
                                ));
                                for iss in a.issues.iter().take(4) {
                                    let c = match iss.severity {
                                        AuditSeverity::Critical => Color32::from_rgb(235, 77, 75),
                                        AuditSeverity::Warning => Color32::from_rgb(241, 196, 15),
                                        AuditSeverity::Info => Color32::from_rgb(130, 180, 240),
                                    };
                                    ui.colored_label(
                                        c,
                                        RichText::new(format!(
                                            "• {}: {}",
                                            iss.category, iss.message
                                        ))
                                        .small(),
                                    );
                                }
                            }
                        });

                    egui::CollapsingHeader::new("Site Blueprint (MIFs & Distances)")
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.checkbox(
                                &mut self.blueprint_use_holo,
                                "Use loaded ligand (Holo consensus)",
                            );
                            ui.horizontal(|ui| {
                                if ui.button("Analyze Site").clicked() {
                                    self.analyze_active_site_now();
                                }
                                if ui.button("Apply -> Restraints").clicked() {
                                    self.apply_blueprint_to_docking();
                                }
                            });
                            let mut bp_changed = false;
                            if let Some(bp) = &mut self.site_blueprint {
                                let g = &bp.geometry;
                                ui.small(format!(
                                    "Vol: {:.0} A3 | LxWxD: {:.1}x{:.1}x{:.1} A",
                                    g.accessible_volume_a3,
                                    g.principal_dims_a[0],
                                    g.principal_dims_a[1],
                                    g.principal_dims_a[2]
                                ));
                                ui.small(format!(
                                    "Hyd {:.0}% / Pol {:.0}% | Opt MW {:.0}-{:.0} Da",
                                    g.hydrophobic_fraction * 100.0,
                                    g.polar_fraction * 100.0,
                                    g.recommended_mw_range[0],
                                    g.recommended_mw_range[1]
                                ));
                                ui.separator();
                                ui.strong("Hotspots (toggle to edit matrix):");
                                for f in &mut bp.features {
                                    let src_tag = match f.source {
                                        FeatureSource::ApoMif => "Apo",
                                        FeatureSource::HybridApoHolo => "Hyb",
                                        FeatureSource::HoloValidated => "Holo",
                                        FeatureSource::UserDefined => "User",
                                    };
                                    ui.horizontal(|ui| {
                                        if ui
                                            .checkbox(
                                                &mut f.enabled,
                                                format!("{} {:?} [{}]", f.id, f.kind, src_tag),
                                            )
                                            .changed()
                                        {
                                            bp_changed = true;
                                        }
                                        ui.small(format!(
                                            "{} ({:.1})",
                                            f.key_residue, f.energy_kcal
                                        ));
                                    });
                                }
                                if bp_changed {
                                    bp.rebuild_distance_matrix();
                                }
                                if !bp.distance_matrix.is_empty() {
                                    ui.separator();
                                    ui.strong(format!(
                                        "Optimal Distances ({} pairs):",
                                        bp.distance_matrix.len()
                                    ));
                                    for d in bp.distance_matrix.iter().take(10) {
                                        ui.small(format!(
                                            "{}<->{}: {:.2} A ({:.1}-{:.1})",
                                            d.from_id,
                                            d.to_id,
                                            d.ideal_distance_a,
                                            d.min_distance_a,
                                            d.max_distance_a
                                        ));
                                    }
                                }
                            }
                            if bp_changed {
                                self.sync_pharmacophore_visuals();
                            }
                        });

                    egui::CollapsingHeader::new("Bridging Waters & Metals")
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.checkbox(&mut self.metal_coordination, "AutoDock4Zn Metal Coord");
                            if ui.button("Scan Bridging Waters").clicked() {
                                self.detect_waters_now();
                            }
                            let mut changed = false;
                            for bw in &self.bridging_waters {
                                let mut kept = self.kept_water_resseqs.contains(&bw.res_seq);
                                let lbl = format!(
                                    "HOH {} ({} HB: {})",
                                    bw.res_seq,
                                    bw.receptor_hbonds,
                                    bw.lining_residues
                                        .iter()
                                        .take(2)
                                        .cloned()
                                        .collect::<Vec<_>>()
                                        .join(",")
                                );
                                if ui.checkbox(&mut kept, lbl).changed() {
                                    if kept {
                                        self.kept_water_resseqs.insert(bw.res_seq);
                                    } else {
                                        self.kept_water_resseqs.remove(&bw.res_seq);
                                    }
                                    changed = true;
                                }
                            }
                            if changed {
                                self.rebuild_receptor_with_kept_waters();
                            }
                        });

                    egui::CollapsingHeader::new("Constraints & Flex Sidechains")
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.label("Flexible Residues (e.g. A:ASP189,A:TRP79):");
                            ui.text_edit_singleline(&mut self.flex_residues_input);
                            ui.separator();
                            ui.horizontal(|ui| {
                                egui::ComboBox::from_id_salt("pharm_kind")
                                    .selected_text(format!("{:?}", self.new_pharm_kind))
                                    .width(96.0)
                                    .show_ui(ui, |ui| {
                                        for k in [
                                            PharmacophoreKind::HDonor,
                                            PharmacophoreKind::HAcceptor,
                                            PharmacophoreKind::Aromatic,
                                            PharmacophoreKind::Hydrophobic,
                                            PharmacophoreKind::PositiveIonizable,
                                            PharmacophoreKind::NegativeIonizable,
                                            PharmacophoreKind::MetalChelate,
                                        ] {
                                            ui.selectable_value(
                                                &mut self.new_pharm_kind,
                                                k,
                                                format!("{k:?}"),
                                            );
                                        }
                                    });
                                ui.add(
                                    egui::DragValue::new(&mut self.new_pharm_radius)
                                        .speed(0.1)
                                        .range(0.5..=5.0)
                                        .prefix("R="),
                                );
                            });
                            ui.horizontal(|ui| {
                                if ui.button("+ Sphere at Atom/Center").clicked() {
                                    let ctr = self
                                        .selected_atom_idx
                                        .and_then(|i| {
                                            self.ligand.as_ref().and_then(|l| l.atoms.get(i))
                                        })
                                        .map_or(self.grid_box.center, |a| a.coords);
                                    self.constraints
                                        .push(DockingConstraint::PharmacophoreSphere {
                                            center: ctr,
                                            radius: self.new_pharm_radius,
                                            kind: self.new_pharm_kind,
                                            weight: 4.0,
                                        });
                                    self.sync_pharmacophore_visuals();
                                }
                                if !self.constraints.is_empty() && ui.button("Clear").clicked() {
                                    self.constraints.clear();
                                    self.sync_pharmacophore_visuals();
                                }
                            });
                            ui.small(format!("Active constraints: {}", self.constraints.len()));
                        });

                    ui.separator();
                    ui.heading("Grid Box (A)");
                    let mut box_changed = false;
                    for (axis, lbl) in ["X", "Y", "Z"].iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(format!("C{lbl}"));
                            box_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut self.grid_box.center[axis])
                                        .speed(0.2),
                                )
                                .changed();
                            ui.label(format!("S{lbl}"));
                            box_changed |= ui
                                .add(
                                    egui::DragValue::new(&mut self.grid_box.size[axis])
                                        .speed(0.2)
                                        .range(8.0..=60.0),
                                )
                                .changed();
                        });
                    }
                    if box_changed {
                        self.scene_dirty = true;
                    }

                    ui.separator();
                    ui.heading("Docked Modes");
                    for (i, pose) in self.dock_poses.iter().enumerate() {
                        let sel = self.selected_pose_idx == i;
                        if ui
                            .selectable_label(
                                sel,
                                format!(
                                    "Mode {} | {:.2} kcal/mol | RMSD {:.2}",
                                    pose.mode, pose.affinity_kcal_mol, pose.terms.rmsd_to_best
                                ),
                            )
                            .clicked()
                        {
                            self.selected_pose_idx = i;
                            self.ligand = Some(pose.molecule.clone());
                            self.vina_score_kcal = Some(pose.affinity_kcal_mol);
                            if let Some(rec) = &self.receptor {
                                self.hyde_report = score_pose_hyde(rec, &pose.molecule).ok();
                            }
                            self.scene_dirty = true;
                        }
                    }
                });
            });

        // 5. Right Panel: SeeSAR HYDE v2, PoseBusters, Torsion Traffic-Lights, 32-Group Inspirator, On-Demand ADMET
        egui::SidePanel::right("right_hyde")
            .resizable(true)
            .default_width(325.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("SeeSAR HYDE v2 & Pose Audit");
                    if let Some(rep) = &self.hyde_report {
                        if let Some(pb) = &rep.posebusters {
                            let (col, badge) = if pb.valid {
                                (Color32::from_rgb(46, 204, 113), "PASS PoseBusters (5/5)")
                            } else {
                                (Color32::from_rgb(235, 77, 75), "WARN PoseBusters")
                            };
                            ui.colored_label(col, RichText::new(badge).strong())
                                .on_hover_text(&pb.summary);
                        }
                        let vina_v = self.vina_score_kcal.unwrap_or(rep.total_delta_g_kcal);
                        let cons = consensus_affinity_kcal(vina_v, vina_v, rep.total_delta_g_kcal);
                        ui.label(format!(
                            "Vina: {:.2} | HYDE: {:.2} | Consensus: {:.2} kcal/mol",
                            vina_v, rep.total_delta_g_kcal, cons
                        ));
                        ui.label(format!(
                            "Est. Ki: {} [{} – {}] | LE: {:.3}",
                            format_ki_nm(rep.estimated_ki_nm),
                            format_ki_nm(rep.ki_range_nm.0),
                            format_ki_nm(rep.ki_range_nm.1),
                            rep.ligand_efficiency
                        ));

                        if let Some(tr) = &rep.torsion_report {
                            ui.horizontal(|ui| {
                                ui.strong("Torsions:");
                                ui.colored_label(
                                    Color32::from_rgb(46, 204, 113),
                                    format!("{} G", tr.green_count),
                                );
                                ui.colored_label(
                                    Color32::from_rgb(241, 196, 15),
                                    format!("{} Y", tr.yellow_count),
                                );
                                ui.colored_label(
                                    Color32::from_rgb(235, 77, 75),
                                    format!("{} R", tr.red_count),
                                );
                                ui.small(format!("(+{:.2} kcal)", tr.total_strain_kcal));
                            });
                            if !tr.torsions.is_empty() {
                                egui::CollapsingHeader::new("Rotatable Dihedrals")
                                    .default_open(false)
                                    .show(ui, |ui| {
                                        for t in &tr.torsions {
                                            let c = match t.status {
                                                TorsionStatus::Green => {
                                                    Color32::from_rgb(46, 204, 113)
                                                }
                                                TorsionStatus::Yellow => {
                                                    Color32::from_rgb(241, 196, 15)
                                                }
                                                TorsionStatus::Red => {
                                                    Color32::from_rgb(235, 77, 75)
                                                }
                                            };
                                            ui.horizontal(|ui| {
                                                ui.colored_label(c, "●");
                                                ui.small(format!(
                                                    "{}-{} | {} | {:.0}° (+{:.2})",
                                                    t.atom_u,
                                                    t.atom_v,
                                                    t.pattern,
                                                    t.dihedral_deg,
                                                    t.strain_kcal
                                                ));
                                            });
                                        }
                                    });
                            }
                        }
                    }

                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.strong("ADMET (On-Demand):");
                        if ui.button("Abrir en AdmetCraft").clicked() {
                            self.open_in_admetcraft();
                        }
                        if ui.small_button("Preview").clicked()
                            && let Some(lig) = &self.ligand
                        {
                            self.admet_preview = Some(evaluate_admet(lig));
                        }
                    });
                    if let Some(adm) = &self.admet_preview {
                        ui.small(format!(
                            "MW {:.1} | cLogP {:.2} | TPSA {:.1} | QED {:.2} | Lipinski {}",
                            adm.mw, adm.clogp, adm.tpsa, adm.qed_score, adm.lipinski_violations
                        ));
                    }

                    ui.separator();
                    ui.heading("In-Pocket Inspirator (32 Groups)");
                    ui.horizontal(|ui| {
                        ui.label(format!(
                            "Atom: {}",
                            self.selected_atom_idx
                                .map_or_else(|| "None".to_string(), |i| format!("#{i}"))
                        ));
                        if !self.pose_history.is_empty()
                            && ui.button("Undo").clicked()
                            && let Some(prev) = self.pose_history.pop()
                        {
                            self.ligand = Some(prev);
                            self.rescore_active_complex();
                            self.scene_dirty = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        for cat in [
                            InspiratorCategory::HalogensSmall,
                            InspiratorCategory::AlkylFluoro,
                            InspiratorCategory::PolarHBond,
                            InspiratorCategory::RingsBioisosteres,
                        ] {
                            ui.selectable_value(&mut self.inspirator_cat, cat, cat.label());
                        }
                    });
                    ui.horizontal_wrapped(|ui| {
                        for &preset in self.inspirator_cat.presets() {
                            if ui.button(preset.label()).clicked() {
                                self.apply_mutation(preset);
                            }
                        }
                    });
                    if let Some((lbl, ddg)) = &self.last_mutation_delta {
                        let col = if *ddg <= 0.0 {
                            Color32::LIGHT_GREEN
                        } else {
                            Color32::LIGHT_RED
                        };
                        ui.colored_label(col, format!("Last {lbl}: ddG = {ddg:+.2} kcal/mol"));
                    }

                    ui.separator();
                    ui.heading("Per-Atom HYDE Breakdown");
                    egui::ScrollArea::vertical()
                        .id_salt("atom_hyde_scroll")
                        .max_height(180.0)
                        .show(ui, |ui| {
                            if let Some(rep) = &self.hyde_report {
                                for sc in &rep.atom_scores {
                                    let col = Color32::from_rgb(
                                        (sc.corona_color[0] * 255.0) as u8,
                                        (sc.corona_color[1] * 255.0) as u8,
                                        (sc.corona_color[2] * 255.0) as u8,
                                    );
                                    ui.horizontal(|ui| {
                                        ui.colored_label(col, "●");
                                        let sel = self.selected_atom_idx == Some(sc.atom_index);
                                        if ui
                                            .selectable_label(
                                                sel,
                                                format!(
                                                    "#{} {} ({:+.2})",
                                                    sc.atom_index, sc.element, sc.delta_g_kcal
                                                ),
                                            )
                                            .clicked()
                                        {
                                            self.selected_atom_idx = Some(sc.atom_index);
                                            self.scene_dirty = true;
                                        }
                                        ui.small(&sc.diagnostic);
                                    });
                                }
                            }
                        });

                    ui.separator();
                    ui.heading("3D Directional Interactions");
                    egui::ScrollArea::vertical()
                        .id_salt("inter_scroll")
                        .max_height(130.0)
                        .show(ui, |ui| {
                            if let Some(rep) = &self.hyde_report {
                                for inter in &rep.interactions {
                                    let ang_str = inter
                                        .angle_deg
                                        .map_or_else(String::new, |a| format!(" | {a:.0}°"));
                                    ui.small(format!(
                                        "{:?} | {} | {:.2} A{ang_str} ({:+.2})",
                                        inter.kind,
                                        inter.rec_residue,
                                        inter.distance_a,
                                        inter.energy_kcal
                                    ));
                                }
                            }
                        });
                });
            });

        // 6. Central 3D wgpu Viewport
        egui::CentralPanel::default().show(ctx, |ui| {
            let avail = ui.available_size();
            let (rect, resp) = ui.allocate_exact_size(avail, egui::Sense::click_and_drag());

            if resp.dragged_by(egui::PointerButton::Primary) {
                let d = resp.drag_delta();
                self.camera.orbit(d.x, d.y);
            }
            if resp.dragged_by(egui::PointerButton::Secondary)
                || resp.dragged_by(egui::PointerButton::Middle)
            {
                let d = resp.drag_delta();
                self.camera.pan(d.x, d.y);
            }
            if resp.hovered() {
                let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
                if scroll.abs() > 0.01 {
                    self.camera.zoom(scroll);
                }
            }
            if resp.clicked()
                && let Some(pos) = resp.interact_pointer_pos()
            {
                let nx = ((pos.x - rect.left()) / rect.width().max(1.0)) * 2.0 - 1.0;
                let ny = 1.0 - ((pos.y - rect.top()) / rect.height().max(1.0)) * 2.0;
                let aspect = rect.width() / rect.height().max(1.0);
                let (ro, rd) = self.camera.screen_ray(nx, ny, aspect);
                if let Ok(guard) = self.shared_payload.read()
                    && let Some(hit) = pick_atom_ray(ro, rd, &guard.scene.pickable_ligand_atoms)
                {
                    self.selected_atom_idx = Some(hit);
                    self.scene_dirty = true;
                    self.status_line = format!("Picked ligand atom #{hit} in 3D.");
                }
            }

            let ppp = ctx.pixels_per_point();
            let px_w = ((rect.width() * ppp).round() as u32).max(64);
            let px_h = ((rect.height() * ppp).round() as u32).max(64);
            let cb = DockPaintCallback::into_paint_callback(
                rect,
                [px_w, px_h],
                Arc::clone(&self.shared_payload),
            );
            ui.painter().add(cb);
        });

        // 7. Command Palette (Ctrl+K) — clean PyMOL / ChimeraX / Craft Suite actions
        if self.palette_open {
            egui::Window::new("Command Palette (Ctrl+K)")
                .collapsible(false)
                .anchor(egui::Align2::CENTER_TOP, [0.0, 48.0])
                .show(ctx, |ui| {
                    ui.text_edit_singleline(&mut self.palette_query);
                    let q = self.palette_query.to_ascii_lowercase();
                    let cmds = [
                        ("Detect Binding Pockets", 0usize),
                        ("Detect Structural Bridging Waters", 1),
                        ("Run Docking (Vina + Metal + HYDE Consensus)", 2),
                        ("Toggle HYDE Coronas", 3),
                        ("Toggle Torsion Traffic-Lights", 4),
                        ("Open in AdmetCraft (ADMET & PK)", 5),
                        ("Send to ChemCraft (SAR)", 6),
                        ("Launch ChimeraX Scene", 7),
                        ("Grab GridBox from ChimeraX", 8),
                        ("Export PyMOL (.pml)", 9),
                        ("Export ChimeraX (.cxc)", 10),
                        ("Reset 3D Camera", 11),
                    ];
                    for (label, code) in cmds {
                        if !q.is_empty() && !label.to_ascii_lowercase().contains(&q) {
                            continue;
                        }
                        if ui.button(label).clicked() {
                            match code {
                                0 => self.detect_pockets_now(),
                                1 => self.detect_waters_now(),
                                2 => self.run_docking_now(),
                                3 => {
                                    self.viewport_opts.show_hyde_coronas =
                                        !self.viewport_opts.show_hyde_coronas;
                                    self.scene_dirty = true;
                                }
                                4 => {
                                    self.viewport_opts.show_torsion_lights =
                                        !self.viewport_opts.show_torsion_lights;
                                    self.scene_dirty = true;
                                }
                                5 => self.open_in_admetcraft(),
                                6 => self.open_in_chemcraft(),
                                7 => self.launch_chimerax_now(),
                                8 => self.grab_box_from_chimerax(),
                                9 => self.export_interop_dialog(true),
                                10 => self.export_interop_dialog(false),
                                _ => {
                                    self.camera = Camera3D::default();
                                }
                            }
                            self.palette_open = false;
                        }
                    }
                });
        }
    }
}

fn main() -> eframe::Result<()> {
    let startup_file = std::env::args().nth(1).map(PathBuf::from);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 880.0])
            .with_title(format!(
                "DockCraft v{APP_VERSION} — Virtual Screening & SeeSAR HYDE SBDD"
            )),
        renderer: eframe::Renderer::Wgpu,
        depth_buffer: 0,
        ..Default::default()
    };
    eframe::run_native(
        "DockCraft",
        options,
        Box::new(move |cc| Ok(Box::new(DockCraftApp::new(cc, startup_file)))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn showcase_complex_scores_hyde_and_mutates() -> Result<(), Box<dyn std::error::Error>> {
        let (rec, lig) = build_demo_showcase();
        let _pockets = detect_pockets(&rec, &PocketDetectConfig::default())?;
        let rep = score_pose_hyde(&rec, &lig)?;
        assert_eq!(rep.atom_scores.len(), lig.heavy_atom_count());
        assert!(
            !rep.interactions.is_empty(),
            "expected 3D interactions in showcase"
        );
        let pb = rep.posebusters.as_ref().ok_or("posebusters")?;
        assert!(pb.valid, "showcase PoseBusters failed: {}", pb.summary);
        assert!(rep.torsion_report.is_some());
        assert!(evaluate_admet(&lig).mw > 100.0);
        let mut_out = mutate_and_minimize_in_pocket(&rec, &lig, 2, SubstituentPreset::Tetrazole)?;
        assert!(mut_out.mutated_ligand.atoms.len() > lig.atoms.len());
        Ok(())
    }
}
