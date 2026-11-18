#![forbid(unsafe_code)]
//! Interoperability with ChemCraft (upstream, the library) and ChimeraX (downstream, the viewer).
//!
//! Both directions were nominal rather than real:
//!
//! * A genuine `.chemcraft` project is a ZIP whose `manifest.json` holds a `ChemCraftDocument`:
//!   `{format, version, canvas_mol, reaction_arrows, sar_table: {rows: [...]}, sar_filter}`. The
//!   loader looked only for a top-level `molecules` / `rows` / `compounds` key, so a real project
//!   failed with "no valid ligands in input library" and only a hand-written JSON that ChemCraft
//!   never writes would load.
//! * The `.cxc` scene was written next to the DockCraft project. ChimeraX runs as a Flatpak with
//!   `--sandbox` and an explicit allow-list that does not include it, so it could not be opened
//!   however correct the script was.
//!
//! The fixture here was produced by ChemCraft's own `save_chemcraft_atomic`, not hand-written, so
//! it stays honest about the schema it is meant to pin.

use std::path::Path;

use dock_pipeline::{
    build_chimerax_export, canonical_export_target, default_chimerax_export_dir,
    load_ligand_library_bytes, measured_pic50_of, write_chimerax_session_bundle,
};
use hyde_scorer::score_pose_hyde;
use struct_core::{parse_pdb, parse_sdf_3d_with_limit, perceive_all, prepare_receptor};

fn fixture(rel: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(rel)
}

#[test]
fn reads_a_real_chemcraft_project_with_its_measured_potencies()
-> Result<(), Box<dyn std::error::Error>> {
    let raw = std::fs::read(fixture("interop/trypsin_s1_series.chemcraft"))?;
    // Byte-level guard: this really is the ZIP that ChemCraft writes, not a loose JSON.
    assert_eq!(
        &raw[..4],
        b"PK\x03\x04",
        "fixture should be a .chemcraft ZIP"
    );

    let lib = load_ligand_library_bytes("chemcraft", &raw, 100)?;
    assert_eq!(lib.len(), 5, "expected the five rows of the SAR table");

    let names: Vec<&str> = lib.iter().map(|r| r.name.as_str()).collect();
    assert!(names.contains(&"Benzamidine"), "names were {names:?}");
    assert!(names.contains(&"2-Naphthamidine"), "names were {names:?}");

    // The measured potency has to survive the handoff: it is the only thing that lets a screening
    // run be checked against reality rather than against a guess.
    for record in &lib {
        let pic50 = measured_pic50_of(&record.molecule)
            .ok_or_else(|| format!("{}: pIC50 was dropped in the handoff", record.name))?;
        assert!(
            (3.0..=7.0).contains(&pic50),
            "{}: implausible pIC50 {pic50}",
            record.name
        );
        assert_eq!(
            record.molecule.properties.get("Series").map(String::as_str),
            Some("trypsin_S1"),
            "{}: the series column was dropped",
            record.name
        );
    }

    let benz = lib
        .iter()
        .find(|r| r.name == "Benzamidine")
        .ok_or("missing benzamidine")?;
    assert!((measured_pic50_of(&benz.molecule).unwrap_or(0.0) - 4.74).abs() < 1e-6);
    // ChemCraft writes its own canonical SMILES, which is not the input spelling; the geometry
    // still has to come out right.
    assert_eq!(benz.molecule.heavy_atom_count(), 9);
    assert_eq!(
        benz.molecule
            .atoms
            .iter()
            .map(|a| a.formal_charge as i32)
            .sum::<i32>(),
        1,
        "the amidine should carry one delocalised positive charge at pH 7.4"
    );
    Ok(())
}

#[test]
fn export_target_names_accept_the_legacy_spellings() {
    assert_eq!(canonical_export_target("chimerax"), Some("chimerax"));
    assert_eq!(canonical_export_target("chymera"), Some("chimerax"));
    assert_eq!(canonical_export_target("ChimeraX"), Some("chimerax"));
    assert_eq!(canonical_export_target("pymol"), Some("pymol"));
    assert_eq!(canonical_export_target("molcraft"), Some("pymol"));
    assert_eq!(canonical_export_target("pdb"), None);
}

#[test]
fn chimerax_scene_lands_in_a_directory_chimerax_can_read() -> Result<(), Box<dyn std::error::Error>>
{
    // The default root must be under ChimeraX's own XDG data directory, which is on the Flatpak
    // allow-list. Anywhere else and ChimeraX cannot open the file.
    let default_dir = default_chimerax_export_dir();
    let as_text = default_dir.display().to_string();
    assert!(
        as_text.contains("ChimeraX") && as_text.ends_with("dockcraft_export"),
        "default export dir {as_text} is not inside ChimeraX's own data directory"
    );

    let rec_raw = std::fs::read_to_string(fixture("docking_benchmark/3ptb_trypsin.pdb"))?;
    let mut rec = parse_pdb(&rec_raw, true)?;
    prepare_receptor(&mut rec, true);
    let lig_raw = std::fs::read_to_string(fixture("docking_benchmark/3ptb_benzamidine.sdf"))?;
    let mut lig = parse_sdf_3d_with_limit(&lig_raw, 1)
        .pop()
        .ok_or("missing ligand")??;
    perceive_all(&mut lig);
    let hyde = score_pose_hyde(&rec, &lig)?;

    let tmp = std::env::temp_dir().join(format!("dockcraft_cxc_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp)?;

    let bundle = write_chimerax_session_bundle("3PTB benzamidine", &rec, &lig, &hyde, Some(&tmp))?;

    // One directory per complex, so several scenes can coexist without their relative
    // `receptor.pdb` / `ligand.sdf` names colliding.
    assert!(bundle.scene_dir.starts_with(tmp.canonicalize()?));
    assert_eq!(
        bundle.scene_dir.file_name().and_then(|s| s.to_str()),
        Some("3PTB_benzamidine")
    );
    for path in [
        &bundle.script_path,
        &bundle.receptor_path,
        &bundle.ligand_path,
        &bundle.bundle_path,
    ] {
        assert!(path.is_file(), "{} was not written", path.display());
    }

    let cxc = std::fs::read_to_string(&bundle.script_path)?;
    assert!(cxc.contains("open receptor.pdb"), "cxc was:\n{cxc}");
    assert!(cxc.contains("open ligand.sdf"), "cxc was:\n{cxc}");
    assert!(
        !cxc.contains("Chymera"),
        "the script still calls ChimeraX by the old misspelling"
    );

    // The command handed to ChimeraX is built here from a fixed vocabulary, never assembled from
    // caller text: a ChimeraX command can run Python through `runscript`.
    assert!(bundle.open_command.starts_with("open /"));
    assert!(bundle.open_command.ends_with("scene.cxc"));

    // The annotations the viewer needs are in the sidecar.
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&bundle.bundle_path)?)?;
    assert_eq!(json["format_target"], "ChimeraX");
    assert!(
        json["atom_annotations"]
            .as_array()
            .is_some_and(|a| !a.is_empty())
    );

    let _ = std::fs::remove_dir_all(&tmp);
    Ok(())
}

#[test]
fn pymol_and_chimerax_scripts_are_each_in_their_own_language()
-> Result<(), Box<dyn std::error::Error>> {
    let rec_raw = std::fs::read_to_string(fixture("docking_benchmark/3ptb_trypsin.pdb"))?;
    let mut rec = parse_pdb(&rec_raw, true)?;
    prepare_receptor(&mut rec, true);
    let lig_raw = std::fs::read_to_string(fixture("docking_benchmark/3ptb_benzamidine.sdf"))?;
    let mut lig = parse_sdf_3d_with_limit(&lig_raw, 1)
        .pop()
        .ok_or("missing ligand")??;
    perceive_all(&mut lig);
    let hyde = score_pose_hyde(&rec, &lig)?;

    let (cxc, cxc_bundle) = build_chimerax_export("x", &rec, &lig, &hyde);
    assert!(cxc.contains("cartoon #1"), "ChimeraX syntax expected");
    assert_eq!(cxc_bundle.format_target, "ChimeraX");

    let (pml, pml_bundle) = dock_pipeline::build_pymol_export("x", &rec, &lig, &hyde);
    assert!(
        pml.contains("load receptor.pdb, receptor"),
        "PyMOL syntax expected"
    );
    assert_eq!(
        pml_bundle.format_target, "PyMOL",
        "the PyMOL exporter used to claim a target called MolCraft, which is not a program"
    );
    Ok(())
}

#[test]
fn results_sidecar_round_trips_the_join_keys_back_to_chemcraft()
-> Result<(), Box<dyn std::error::Error>> {
    use dock_pipeline::{
        ResultsSidecar, SIDECAR_FORMAT, default_chemcraft_results_dir, run_screening_campaign,
        write_results_sidecar,
    };

    // The results directory mirrors ChemCraft's own handoff directory, so the pair is symmetric.
    let default_dir = default_chemcraft_results_dir().display().to_string();
    assert!(
        default_dir.contains("ChemCraft") && default_dir.ends_with("dockcraft_results"),
        "unexpected results directory {default_dir}"
    );

    let rec_raw = std::fs::read_to_string(fixture("docking_benchmark/3ptb_trypsin.pdb"))?;
    let mut rec = parse_pdb(&rec_raw, true)?;
    prepare_receptor(&mut rec, true);
    let lig_raw = std::fs::read_to_string(fixture("docking_benchmark/3ptb_benzamidine.sdf"))?;
    let mut ben = parse_sdf_3d_with_limit(&lig_raw, 1)
        .pop()
        .ok_or("missing ligand")??;
    perceive_all(&mut ben);
    let gbox = pocket_detect::GridBox::from_ligand(&ben, 5.0)?;

    let raw = std::fs::read(fixture("interop/trypsin_s1_series.chemcraft"))?;
    let lib = load_ligand_library_bytes("chemcraft", &raw, 100)?;

    let cfg = dock_engine::DockingConfig {
        scoring: dock_engine::ScoringFunction::Vina,
        exhaustiveness: 4,
        ils_steps: 10,
        bfgs_iters: 20,
        ..Default::default()
    };
    let report = run_screening_campaign(&rec, &gbox, &lib, &cfg)?;

    let tmp = std::env::temp_dir().join(format!("dockcraft_sidecar_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp)?;
    let path = write_results_sidecar(&report, "3PTB", &gbox, "vina", 4, &tmp, "trypsin_s1")?;

    let sidecar: ResultsSidecar = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    assert_eq!(sidecar.format, SIDECAR_FORMAT);
    assert_eq!(sidecar.rows.len(), 5);

    for row in &sidecar.rows {
        // The join key has to be ChemCraft's own canonical spelling, not one DockCraft invented:
        // a merge that matched on anything else would silently drop rows.
        assert!(
            !row.canonical_smiles.is_empty(),
            "{}: no canonical SMILES to join on",
            row.name
        );
        assert!(
            row.source_row_id.is_some(),
            "{}: the SAR row id was not carried through",
            row.name
        );
        // The measured potency goes back out untouched, so a reader can line computed against
        // measured without a second lookup.
        assert!(
            row.measured_pic50.is_some(),
            "{}: the measured pIC50 was lost on the return leg",
            row.name
        );
        // Poses are referenced relatively, so the directory can be moved or copied whole.
        assert!(!row.pose_sdf.starts_with('/'), "pose path is absolute");
        let pose = tmp.join(&row.pose_sdf);
        assert!(pose.is_file(), "{} was not written", pose.display());
        let text = std::fs::read_to_string(&pose)?;
        assert!(text.contains("$$$$"), "{}: not an SDF", pose.display());
    }

    let ids: Vec<u64> = sidecar
        .rows
        .iter()
        .filter_map(|r| r.source_row_id)
        .collect();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), ids.len(), "row ids are not unique: {ids:?}");

    let _ = std::fs::remove_dir_all(&tmp);
    Ok(())
}
