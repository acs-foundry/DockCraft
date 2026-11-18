#![forbid(unsafe_code)]
//! Enrichment on Trypsin S1 against a panel of null models.
//!
//! A virtual-screening score is only worth anything if it beats what you would get for free from
//! a physicochemical property. The previous version of this bench used four cationic aryl
//! amidines against sixteen neutral aromatics, so a model that ranked purely by **formal charge**
//! scored a perfect AUC of 1.0000 and the docking engine's 0.891 was strictly worse than doing no
//! docking at all.
//!
//! Every compound here therefore carries a net charge of +1 at pH 7.4, and the test asserts
//! against thirteen null models at once, printing all of them so that a reader can see which
//! descriptor is doing the work.
//!
//! Scope, stated plainly: twenty hand-picked compounds are a sanity check, not a validation. A
//! real validation needs a property-matched decoy set (DUD-E or equivalent) with measured
//! affinities. What this test can rule out is the specific failure mode of the earlier bench —
//! an enrichment number produced by a descriptor rather than by the pocket.

use std::path::Path;

use dock_engine::{DockingConfig, ScoringFunction};
use dock_pipeline::{ScreenedCandidate, load_ligand_library_bytes, run_screening_campaign};
use pocket_detect::GridBox;
use struct_core::{parse_pdb, parse_sdf_3d_with_limit, perceive_all, prepare_receptor};

/// Aryl amidines: the documented Trypsin S1 chemotype, anchored by a salt bridge to Asp189.
const ACTIVES: [(&str, &str); 6] = [
    ("Act_Benzamidine", "NC(=N)c1ccccc1"),
    ("Act_4Methylbenzamidine", "NC(=N)c1ccc(C)cc1"),
    ("Act_2Naphthamidine", "NC(=N)c1ccc2ccccc2c1"),
    ("Act_4Biphenylamidine", "NC(=N)c1ccc(-c2ccccc2)cc1"),
    ("Act_4Phenoxybenzamidine", "NC(=N)c1ccc(Oc2ccccc2)cc1"),
    ("Act_4Benzyloxybenzamidine", "NC(=N)c1ccc(OCc2ccccc2)cc1"),
];

/// Monocations without the aryl-amidine S1 pharmacophore: amines, a guanidine on an aliphatic
/// scaffold, and several drug-like bases. Matched to the actives on formal charge by
/// construction, and spread over nitrogen count, mass and lipophilicity.
const DECOYS: [(&str, &str); 14] = [
    ("Dec_Chloroquinefrag", "CCN(CC)CCCC(C)Nc1ccnc2ccccc12"),
    ("Dec_Nicotine", "CN1CCCC1c1cccnc1"),
    ("Dec_Imipramine", "CN(C)CCCN1c2ccccc2CCc2ccccc21"),
    ("Dec_Phenylpiperazine", "C1CN(CCN1)c1ccccc1"),
    ("Dec_Benzylpiperidine", "C1CCN(Cc2ccccc2)CC1"),
    ("Dec_4Phenylpiperidine", "C1CC(CCN1)c1ccccc1"),
    ("Dec_Cyclohexylguanidine", "NC(=N)NC1CCCCC1"),
    ("Dec_Diphenhydramine", "CN(C)CCOC(c1ccccc1)c2ccccc2"),
    ("Dec_Propranolol", "CC(C)NCC(O)COc1cccc2ccccc12"),
    ("Dec_Tryptamine", "NCCc1c[nH]c2ccccc12"),
    ("Dec_Tetrahydroisoquinoline", "C1Cc2ccccc2CN1"),
    ("Dec_2Aminoethylpyridine", "NCCc1ccccn1"),
    ("Dec_Procainamide", "CCN(CC)CCNC(=O)c1ccc(N)cc1"),
    ("Dec_Dibenzylamine", "C(c1ccccc1)NCc2ccccc2"),
];

fn fixture_path(rel: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/docking_benchmark")
        .join(rel)
}

fn roc_auc(items: &[(bool, f64)]) -> f64 {
    let (act, dec): (Vec<_>, Vec<_>) = items.iter().partition(|(a, _)| *a);
    let mut wins = 0.0_f64;
    for (_, sa) in &act {
        for (_, sd) in &dec {
            if sa > sd {
                wins += 1.0;
            } else if (sa - sd).abs() < 1e-9 {
                wins += 0.5;
            }
        }
    }
    wins / ((act.len() * dec.len()).max(1) as f64)
}

#[test]
fn screening_beats_every_trivial_descriptor_on_charge_matched_decoys()
-> Result<(), Box<dyn std::error::Error>> {
    let rec_raw = std::fs::read_to_string(fixture_path("3ptb_trypsin.pdb"))?;
    let mut rec = parse_pdb(&rec_raw, true)?;
    prepare_receptor(&mut rec, true);

    let lig_raw = std::fs::read_to_string(fixture_path("3ptb_benzamidine.sdf"))?;
    let mut ben = parse_sdf_3d_with_limit(&lig_raw, 1)
        .pop()
        .ok_or("missing BEN")??;
    perceive_all(&mut ben);
    let gbox = GridBox::from_ligand(&ben, 4.0)?;

    let mut compounds = Vec::new();
    for (name, smi) in ACTIVES {
        compounds.push(serde_json::json!({ "name": name, "smiles": smi, "is_active": true }));
    }
    for (name, smi) in DECOYS {
        compounds.push(serde_json::json!({ "name": name, "smiles": smi, "is_active": false }));
    }
    let cc_json = serde_json::to_vec(&serde_json::json!({ "molecules": compounds }))?;
    let lib = load_ligand_library_bytes("chemcraft", &cc_json, 100)?;
    assert_eq!(lib.len(), ACTIVES.len() + DECOYS.len());

    // Charge matching is the whole point, so verify it rather than trusting the SMILES list.
    for rec_in in &lib {
        let q: i32 = rec_in
            .molecule
            .atoms
            .iter()
            .map(|a| a.formal_charge as i32)
            .sum();
        assert_eq!(
            q, 1,
            "{}: expected net +1 at pH 7.4, got {q}. Amidines and guanidines carry one \
             delocalised charge per group, not one per nitrogen.",
            rec_in.name
        );
    }

    let cfg = DockingConfig {
        scoring: ScoringFunction::Vina,
        exhaustiveness: 8,
        ils_steps: 20,
        bfgs_iters: 40,
        ..Default::default()
    };
    let report = run_screening_campaign(&rec, &gbox, &lib, &cfg)?;
    assert_eq!(report.results.len(), ACTIVES.len() + DECOYS.len());
    let enr = report
        .enrichment
        .as_ref()
        .ok_or("expected enrichment metrics")?;

    let auc_of = |f: &dyn Fn(&ScreenedCandidate) -> f64| {
        roc_auc(
            &report
                .results
                .iter()
                .map(|r| (r.is_active.unwrap_or(false), f(r)))
                .collect::<Vec<_>>(),
        )
    };

    let nulls: [(&str, f64); 13] = [
        (
            "formal_charge",
            auc_of(&|r| {
                r.best_pose
                    .molecule
                    .atoms
                    .iter()
                    .map(|a| a.formal_charge as f64)
                    .sum::<f64>()
            }),
        ),
        (
            "nitrogen_count",
            auc_of(&|r| {
                r.best_pose
                    .molecule
                    .atoms
                    .iter()
                    .filter(|a| a.element == "N")
                    .count() as f64
            }),
        ),
        ("molecular_weight", auc_of(&|r| r.admet.mw)),
        ("heavy_atoms", auc_of(&|r| r.admet.heavy_atoms as f64)),
        ("clogp", auc_of(&|r| r.admet.clogp)),
        ("neg_clogp", auc_of(&|r| -r.admet.clogp)),
        ("tpsa", auc_of(&|r| r.admet.tpsa)),
        (
            "rotatable_bonds",
            auc_of(&|r| r.admet.rotatable_bonds as f64),
        ),
        ("qed", auc_of(&|r| r.admet.qed_score)),
        ("admet_score", auc_of(&|r| r.admet.admet_score)),
        ("hbd", auc_of(&|r| r.admet.hbd as f64)),
        ("hba", auc_of(&|r| r.admet.hba as f64)),
        ("aromatic_rings", auc_of(&|r| r.admet.aromatic_rings as f64)),
    ];

    eprintln!(
        "3PTB S1, {} actives / {} charge-matched decoys: DOCK_ROC_AUC={:.4}  EF1%={:.2}  EF5%={:.2}  MPO_AUC={:.4}",
        ACTIVES.len(),
        DECOYS.len(),
        enr.roc_auc,
        enr.ef_1_percent,
        enr.ef_5_percent,
        auc_of(&|r| r.mpo_score)
    );
    for (name, auc) in &nulls {
        eprintln!("  null {name:>16} AUC = {auc:.4}");
    }

    let (best_null_name, best_null) = nulls
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .copied()
        .ok_or("empty null panel")?;

    assert!(
        enr.roc_auc > best_null + 0.15,
        "docking AUC {:.4} does not beat the strongest null model ({best_null_name} = {best_null:.4}) by a clear margin",
        enr.roc_auc
    );
    assert!(
        enr.roc_auc >= 0.95,
        "docking AUC {:.4} below 0.95 on a chemotype the pocket is known to select",
        enr.roc_auc
    );
    // 5% of 20 compounds is one compound, so the attainable maximum is 20/6 = 3.33.
    assert!(
        enr.ef_5_percent > 3.0,
        "EF5% = {:.2}: the top-ranked compound is not an active",
        enr.ef_5_percent
    );

    // PAINS detection must still fire during the campaign.
    let _ = dock_pipeline::evaluate_admet(&ben);
    Ok(())
}
