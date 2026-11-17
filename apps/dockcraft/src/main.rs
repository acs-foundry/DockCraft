#![forbid(unsafe_code)]

use std::path::Path;
use std::sync::{Arc, RwLock};

use dock_engine::{DockingConfig, DockingPose, ScoringFunction, dock_ligand};
use dock_pipeline::{
    AdmetProfile, CampaignReport, LigandInputRecord, build_chymera_export, build_molcraft_export,
    evaluate_admet, load_ligand_library_bytes, run_screening_campaign,
};
use dock_viewport::{
    Camera3D, DockPaintCallback, DockWgpuResources, SharedViewportPayload, ViewportOptions,
    build_dock_scene, pick_atom_ray,
};
use eframe::egui;
use hyde_scorer::{HydeReport, SubstituentPreset, mutate_and_minimize_in_pocket, score_pose_hyde};
use pocket_detect::{BindingPocket, GridBox, PocketDetectConfig, detect_pockets};
use struct_core::{
    Atom3D, BondOrder, Molecule3D, assign_ad4_types_and_flags, parse_mmcif, parse_mol2, parse_pdb,
    parse_pdbqt, parse_sdf_3d_with_limit, parse_smiles_to_3d, perceive_all, prepare_receptor,
};

fn build_demo_showcase() -> (Molecule3D, Molecule3D) {
    let mut rec = Molecule3D::new("1STP_Streptavidin_Pocket");
    let res = [
        ("N", "ASN", 23, [-3.0, 0.2, 0.0]),
        ("O", "SER", 27, [-1.5, -2.9, 0.4]),
        ("O", "TYR", 43, [0.4, 3.0, 0.0]),
        ("O", "SER", 45, [2.8, 2.2, -1.8]),
        ("N", "ASN", 49, [-0.6, 0.5, -3.1]),
        ("C", "TRP", 79, [3.6, -0.5, 1.5]),
        ("C", "TRP", 92, [1.5, -3.5, -1.2]),
        ("C", "TRP", 108, [3.8, 0.2, -2.2]),
        ("C", "LEU", 110, [0.5, -0.4, 3.6]),
        ("N", "ASP", 128, [-3.8, -1.8, -0.6]),
        ("C", "VAL", 47, [2.2, 3.3, 1.8]),
        ("C", "PHE", 130, [4.6, -2.1, -0.4]),
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
    let coords = [
        ("O", [-0.4, 0.2, 0.0]),
        ("C", [0.8, 0.2, 0.0]),
        ("N", [1.5, 1.3, 0.0]),
        ("N", [1.5, -0.9, 0.0]),
        ("C", [2.9, 0.8, 0.0]),
        ("C", [2.9, -0.5, 0.0]),
        ("S", [4.1, 0.2, 0.7]),
        ("C", [3.6, -1.5, -0.8]),
        ("C", [5.0, -1.7, -0.3]),
        ("O", [-1.6, -1.6, 0.2]),
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
        (3, 9, BondOrder::Single),
    ];
    for &(u, v, ord) in &bonds {
        let _ = lig.add_bond(ids[u], ids[v], ord);
    }
    assign_ad4_types_and_flags(&mut lig);
    perceive_all(&mut lig);
    (rec, lig)
}

struct DockCraftApp {
    receptor_path_input: String,
    ligand_path_input: String,
    export_dir_input: String,
    receptor: Option<Molecule3D>,
    ligand: Option<Molecule3D>,
    pose_history: Vec<Molecule3D>,
    pockets: Vec<BindingPocket>,
    selected_pocket_idx: Option<usize>,
    grid_box: GridBox,
    scoring: ScoringFunction,
    exhaustiveness: usize,
    dock_poses: Vec<DockingPose>,
    selected_pose_idx: usize,
    campaign_report: Option<CampaignReport>,
    hyde_report: Option<HydeReport>,
    admet_profile: Option<AdmetProfile>,
    selected_atom_idx: Option<usize>,
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
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
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
        let admet = Some(evaluate_admet(&lig));
        let mut cam = Camera3D::default();
        cam.focus_on([1.5, 0.0, 0.0], 12.0);

        let mut app = Self {
            receptor_path_input: String::new(),
            ligand_path_input: String::new(),
            export_dir_input: ".".to_string(),
            receptor: Some(rec),
            ligand: Some(lig),
            pose_history: Vec::new(),
            pockets: Vec::new(),
            selected_pocket_idx: None,
            grid_box: gbox,
            scoring: ScoringFunction::Vina,
            exhaustiveness: 6,
            dock_poses: Vec::new(),
            selected_pose_idx: 0,
            campaign_report: None,
            hyde_report: hyde,
            admet_profile: admet,
            selected_atom_idx: Some(2),
            last_mutation_delta: None,
            status_line:
                "Showcase complex loaded. Click any ligand atom in 3D or mutate on the right."
                    .to_string(),
            palette_open: false,
            palette_query: String::new(),
            viewport_opts: ViewportOptions::default(),
            camera: cam,
            shared_payload: Arc::new(RwLock::new(SharedViewportPayload::default())),
            scene_dirty: true,
        };
        app.detect_pockets_now();
        app
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
                self.status_line = format!("Detected {count} binding pocket(s).");
                self.scene_dirty = true;
            }
            Err(e) => self.status_line = format!("Pocket detection error: {e}"),
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
            ..Default::default()
        };
        match dock_ligand(rec, lig, &self.grid_box, &cfg) {
            Ok(rep) => {
                if let Some(best) = rep.poses.first() {
                    self.ligand = Some(best.molecule.clone());
                    self.hyde_report = score_pose_hyde(rec, &best.molecule).ok();
                    self.admet_profile = Some(evaluate_admet(&best.molecule));
                }
                self.status_line = format!(
                    "Docked {} modes in {} ms (best: {:.2} kcal/mol).",
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
                    self.hyde_report = Some(top.hyde_report.clone());
                    self.admet_profile = Some(top.admet.clone());
                }
                self.campaign_report = Some(rep);
                self.scene_dirty = true;
            }
            Err(e) => self.status_line = format!("Campaign failed: {e}"),
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
        let parsed = match ext.as_str() {
            "cif" | "mmcif" => parse_mmcif(&raw, true),
            "pdbqt" => parse_pdbqt(&raw),
            "mol2" => parse_mol2(&raw),
            _ => parse_pdb(&raw, true),
        };
        match parsed {
            Ok(mut rec) => {
                prepare_receptor(&mut rec, true);
                if rec.heavy_atom_count() > 0 {
                    let c = rec.centroid_heavy();
                    self.camera
                        .focus_on([c[0] as f32, c[1] as f32, c[2] as f32], 18.0);
                }
                self.receptor = Some(rec);
                self.detect_pockets_now();
                self.scene_dirty = true;
            }
            Err(e) => self.status_line = format!("Receptor parse error: {e}"),
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
                if let Ok(gb) = GridBox::from_ligand(&lig, 6.0) {
                    self.grid_box = gb;
                }
                if let Some(rec) = &self.receptor {
                    self.hyde_report = score_pose_hyde(rec, &lig).ok();
                }
                self.admet_profile = Some(evaluate_admet(&lig));
                self.ligand = Some(lig);
                self.scene_dirty = true;
                self.status_line = "Ligand loaded.".to_string();
            }
            Err(e) => self.status_line = format!("Ligand error: {e}"),
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
                self.admet_profile = Some(evaluate_admet(&out.mutated_ligand));
                self.ligand = Some(out.mutated_ligand);
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

    fn export_interop(&mut self, target_molcraft: bool) {
        let (Some(rec), Some(lig), Some(hyde)) = (&self.receptor, &self.ligand, &self.hyde_report)
        else {
            self.status_line = "Score a complex before exporting.".to_string();
            return;
        };
        let dir = Path::new(self.export_dir_input.trim());
        if target_molcraft {
            let (pml, bundle) = build_molcraft_export(&lig.name, rec, lig, hyde);
            let p1 = dir.join("dockcraft_scene.pml");
            let p2 = dir.join("dockcraft_scene.molcraft.json");
            let ok = std::fs::write(&p1, pml).is_ok()
                && serde_json::to_string_pretty(&bundle)
                    .map(|js| std::fs::write(&p2, js).is_ok())
                    .unwrap_or(false);
            self.status_line = if ok {
                format!("Exported MolCraft bundle to {}", dir.display())
            } else {
                "MolCraft export failed".to_string()
            };
        } else {
            let (cxc, bundle) = build_chymera_export(&lig.name, rec, lig, hyde);
            let p1 = dir.join("dockcraft_scene.cxc");
            let p2 = dir.join("dockcraft_scene.chymera.json");
            let ok = std::fs::write(&p1, cxc).is_ok()
                && serde_json::to_string_pretty(&bundle)
                    .map(|js| std::fs::write(&p2, js).is_ok())
                    .unwrap_or(false);
            self.status_line = if ok {
                format!("Exported Chymera bundle to {}", dir.display())
            } else {
                "Chymera export failed".to_string()
            };
        }
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
        if ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::K)) {
            self.palette_open = !self.palette_open;
        }
        self.sync_viewport_payload();

        egui::TopBottomPanel::top("top_ribbon").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading("DockCraft");
                ui.separator();
                if ui.button("Showcase (1STP)").clicked() {
                    let (rec, lig) = build_demo_showcase();
                    self.hyde_report = score_pose_hyde(&rec, &lig).ok();
                    self.admet_profile = Some(evaluate_admet(&lig));
                    self.receptor = Some(rec);
                    self.ligand = Some(lig);
                    self.detect_pockets_now();
                    self.scene_dirty = true;
                }
                if ui.button("Detect Pockets").clicked() {
                    self.detect_pockets_now();
                }
                if ui.button("Run Docking").clicked() {
                    self.run_docking_now();
                }
                if ui.button("Screen Library").clicked() {
                    self.run_campaign_now();
                }
                ui.separator();
                egui::ComboBox::from_id_salt("sf_combo")
                    .selected_text(match self.scoring {
                        ScoringFunction::Vina => "Vina",
                        ScoringFunction::Vinardo => "Vinardo",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.scoring, ScoringFunction::Vina, "Vina");
                        ui.selectable_value(&mut self.scoring, ScoringFunction::Vinardo, "Vinardo");
                    });
                ui.add(egui::Slider::new(&mut self.exhaustiveness, 1..=16).text("Exh"));
                ui.separator();
                if ui
                    .checkbox(&mut self.viewport_opts.show_hyde_coronas, "Coronas")
                    .changed()
                    || ui
                        .checkbox(&mut self.viewport_opts.show_pocket_cloud, "Pocket")
                        .changed()
                    || ui
                        .checkbox(&mut self.viewport_opts.show_interactions, "H-Bonds")
                        .changed()
                    || ui
                        .checkbox(&mut self.viewport_opts.show_grid_box, "GridBox")
                        .changed()
                {
                    self.scene_dirty = true;
                }
                ui.separator();
                if ui.button("-> MolCraft").clicked() {
                    self.export_interop(true);
                }
                if ui.button("-> Chymera").clicked() {
                    self.export_interop(false);
                }
                if ui.button("Cmd (Ctrl+K)").clicked() {
                    self.palette_open = true;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Receptor:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.receptor_path_input)
                        .desired_width(210.0)
                        .hint_text("/path/to/receptor.pdb|.cif"),
                );
                if ui.button("Load Rec").clicked() {
                    self.load_receptor_from_path();
                }
                ui.label("Ligand/Lib:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.ligand_path_input)
                        .desired_width(210.0)
                        .hint_text("SMILES or .sdf/.csv/.chemcraft"),
                );
                if ui.button("Load Lig").clicked() {
                    self.load_ligand_from_input();
                }
            });
        });

        egui::TopBottomPanel::bottom("bottom_campaign")
            .resizable(true)
            .default_height(155.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.strong("Status:");
                    ui.label(&self.status_line);
                    if let Some(rep) = &self.campaign_report
                        && let Some(enr) = &rep.enrichment
                    {
                        ui.separator();
                        ui.label(format!(
                            "ROC AUC: {:.3} | EF1%: {:.2} | EF5%: {:.2}",
                            enr.roc_auc, enr.ef_1_percent, enr.ef_5_percent
                        ));
                    }
                });
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    egui::Grid::new("mpo_table").striped(true).show(ui, |ui| {
                        for h in [
                            "Rank", "Compound", "Vina dG", "HYDE dG", "Ki (nM)", "LE", "MPO",
                            "ADMET",
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
                                    self.hyde_report = Some(cand.hyde_report.clone());
                                    self.admet_profile = Some(cand.admet.clone());
                                    self.scene_dirty = true;
                                }
                                ui.label(&cand.name);
                                ui.label(format!("{:.2}", cand.best_affinity_kcal));
                                ui.label(format!("{:.2}", cand.hyde_delta_g_kcal));
                                ui.label(format!("{:.1}", cand.estimated_ki_nm));
                                ui.label(format!("{:.3}", cand.ligand_efficiency));
                                ui.label(format!("{:.3}", cand.mpo_score));
                                ui.label(format!("{:.1}", cand.admet.admet_score));
                                ui.end_row();
                            }
                        }
                    });
                });
            });

        egui::SidePanel::left("left_pyrx")
            .resizable(true)
            .default_width(250.0)
            .show(ctx, |ui| {
                ui.heading("Binding Pockets");
                for (idx, p) in self.pockets.iter().enumerate() {
                    let sel = self.selected_pocket_idx == Some(idx);
                    if ui
                        .selectable_label(
                            sel,
                            format!(
                                "#{} Drugg {:.2} | {:.0} A3 | {:.0} A2",
                                p.rank, p.druggability_score, p.volume_a3, p.sasa_a2
                            ),
                        )
                        .clicked()
                    {
                        self.selected_pocket_idx = Some(idx);
                        self.grid_box = p.grid_box.clone();
                        self.camera.focus_on(
                            [p.center[0] as f32, p.center[1] as f32, p.center[2] as f32],
                            14.0,
                        );
                        self.scene_dirty = true;
                    }
                }
                ui.separator();
                ui.heading("Grid Box (A)");
                let mut box_changed = false;
                for (axis, lbl) in ["X", "Y", "Z"].iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(format!("C{lbl}"));
                        box_changed |= ui
                            .add(egui::DragValue::new(&mut self.grid_box.center[axis]).speed(0.2))
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
                        if let Some(rec) = &self.receptor {
                            self.hyde_report = score_pose_hyde(rec, &pose.molecule).ok();
                        }
                        self.scene_dirty = true;
                    }
                }
            });

        egui::SidePanel::right("right_hyde")
            .resizable(true)
            .default_width(305.0)
            .show(ctx, |ui| {
                ui.heading("SeeSAR HYDE Inspector");
                if let Some(rep) = &self.hyde_report {
                    ui.label(format!("HYDE dG: {:.2} kcal/mol", rep.total_delta_g_kcal));
                    ui.label(format!(
                        "Est. Ki: {:.2} nM | LE: {:.3}",
                        rep.estimated_ki_nm, rep.ligand_efficiency
                    ));
                }
                if let Some(adm) = &self.admet_profile {
                    ui.label(format!(
                        "MW {:.1} | cLogP {:.2} | TPSA {:.1} | QED {:.2} | Lipinski {}",
                        adm.mw, adm.clogp, adm.tpsa, adm.qed_score, adm.lipinski_violations
                    ));
                }

                ui.separator();
                ui.heading("In-Pocket Inspirator");
                ui.horizontal(|ui| {
                    ui.label(format!(
                        "Selected Atom: {}",
                        self.selected_atom_idx
                            .map_or_else(|| "None".to_string(), |i| format!("#{i}"))
                    ));
                    if !self.pose_history.is_empty()
                        && ui.button("Undo").clicked()
                        && let Some(prev) = self.pose_history.pop()
                    {
                        if let Some(rec) = &self.receptor {
                            self.hyde_report = score_pose_hyde(rec, &prev).ok();
                        }
                        self.admet_profile = Some(evaluate_admet(&prev));
                        self.ligand = Some(prev);
                        self.scene_dirty = true;
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    for &preset in SubstituentPreset::all() {
                        if ui.button(preset.label()).clicked() {
                            self.apply_mutation(preset);
                        }
                    }
                });
                if let Some((lbl, ddg)) = &self.last_mutation_delta {
                    let col = if *ddg <= 0.0 {
                        egui::Color32::LIGHT_GREEN
                    } else {
                        egui::Color32::LIGHT_RED
                    };
                    ui.colored_label(col, format!("Last {lbl}: ddG = {ddg:+.2} kcal/mol"));
                }

                ui.separator();
                ui.heading("Per-Atom HYDE Breakdown");
                egui::ScrollArea::vertical()
                    .max_height(240.0)
                    .show(ui, |ui| {
                        if let Some(rep) = &self.hyde_report {
                            for sc in &rep.atom_scores {
                                let col = egui::Color32::from_rgb(
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
                ui.heading("3D Interactions");
                egui::ScrollArea::vertical()
                    .max_height(130.0)
                    .show(ui, |ui| {
                        if let Some(rep) = &self.hyde_report {
                            for inter in &rep.interactions {
                                ui.small(format!(
                                    "{:?} | {} | {:.2} A ({:+.2})",
                                    inter.kind,
                                    inter.rec_residue,
                                    inter.distance_a,
                                    inter.energy_kcal
                                ));
                            }
                        }
                    });
            });

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

        if self.palette_open {
            egui::Window::new("Command Palette (Ctrl+K)")
                .collapsible(false)
                .anchor(egui::Align2::CENTER_TOP, [0.0, 48.0])
                .show(ctx, |ui| {
                    ui.text_edit_singleline(&mut self.palette_query);
                    let q = self.palette_query.to_ascii_lowercase();
                    let cmds = [
                        ("Detect Binding Pockets", 0usize),
                        ("Run Vina Docking", 1),
                        ("Toggle HYDE Coronas", 2),
                        ("Export MolCraft (.pml)", 3),
                        ("Export Chymera (.cxc)", 4),
                        ("Reset Camera", 5),
                    ];
                    for (label, code) in cmds {
                        if !q.is_empty() && !label.to_ascii_lowercase().contains(&q) {
                            continue;
                        }
                        if ui.button(label).clicked() {
                            match code {
                                0 => self.detect_pockets_now(),
                                1 => self.run_docking_now(),
                                2 => {
                                    self.viewport_opts.show_hyde_coronas =
                                        !self.viewport_opts.show_hyde_coronas;
                                    self.scene_dirty = true;
                                }
                                3 => self.export_interop(true),
                                4 => self.export_interop(false),
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
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1360.0, 860.0])
            .with_title("DockCraft - Virtual Screening & HYDE SBDD"),
        renderer: eframe::Renderer::Wgpu,
        depth_buffer: 0,
        ..Default::default()
    };
    eframe::run_native(
        "DockCraft",
        options,
        Box::new(|cc| Ok(Box::new(DockCraftApp::new(cc)))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn showcase_complex_scores_hyde_and_mutates() -> Result<(), Box<dyn std::error::Error>> {
        let (rec, lig) = build_demo_showcase();
        let pockets = detect_pockets(&rec, &PocketDetectConfig::default())?;
        assert!(!pockets.is_empty() || rec.atoms.len() == 12);
        let rep = score_pose_hyde(&rec, &lig)?;
        assert_eq!(rep.atom_scores.len(), lig.heavy_atom_count());
        let mut_out = mutate_and_minimize_in_pocket(&rec, &lig, 2, SubstituentPreset::F)?;
        assert!(mut_out.mutated_ligand.atoms.len() > lig.atoms.len());
        Ok(())
    }
}
