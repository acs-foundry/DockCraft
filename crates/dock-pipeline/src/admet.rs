#![forbid(unsafe_code)]

use std::process::Command;

use serde::{Deserialize, Serialize};
use struct_core::{BondOrder, Molecule3D};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmetProfile {
    pub mw: f64,
    pub clogp: f64,
    pub hbd: usize,
    pub hba: usize,
    pub tpsa: f64,
    pub rotatable_bonds: usize,
    pub heavy_atoms: usize,
    pub lipinski_violations: usize,
    pub veber_pass: bool,
    pub bbb_penetrant: bool,
    pub herg_risk: bool,
    pub pains_alerts: Vec<String>,
    pub qed_score: f64,
    pub admet_score: f64,
    pub backend_used: String,
}

#[must_use]
pub fn evaluate_admet(mol: &Molecule3D) -> AdmetProfile {
    let mw = mol.molecular_weight();
    let heavy_atoms = mol.heavy_atom_count();
    let rotatable_bonds = mol.rotatable_bonds.len();

    let mut hbd = 0usize;
    let mut hba = 0usize;
    let mut tpsa = 0.0_f64;
    let mut clogp = 0.0_f64;

    for (idx, a) in mol.atoms.iter().enumerate() {
        if !a.is_heavy() {
            continue;
        }
        if a.h_donor {
            hbd += 1;
        }
        if a.h_acceptor {
            hba += 1;
        }

        let deg = mol.adj.get(idx).map_or(0, |v| v.len());
        let has_double = mol.adj.get(idx).is_some_and(|v| {
            v.iter()
                .any(|&(_, bid)| mol.bonds[bid].order == BondOrder::Double)
        });

        match a.element.as_str() {
            "C" => clogp += if a.aromatic { 0.29 } else { 0.14 },
            "N" => {
                clogp -= if a.formal_charge > 0 { 1.1 } else { 0.35 };
                tpsa += if a.h_donor {
                    26.0
                } else if a.aromatic {
                    12.9
                } else {
                    15.2
                };
            }
            "O" => {
                clogp -= if a.formal_charge < 0 { 1.3 } else { 0.42 };
                tpsa += if a.h_donor {
                    20.2
                } else if has_double {
                    17.1
                } else {
                    9.2
                };
            }
            "S" => {
                clogp += 0.65;
                if a.h_donor || deg <= 1 {
                    tpsa += 25.3;
                }
            }
            "P" => {
                clogp += 0.2;
                tpsa += 13.6;
            }
            "F" => clogp += 0.42,
            "CL" => clogp += 0.89,
            "BR" => clogp += 1.09,
            "I" => clogp += 1.35,
            _ => {}
        }
    }

    let mut pains_alerts = Vec::new();
    // Simple topological PAINS-like alerts: N-O/N=O nitro-aromatic or Michael-like strained carbonyl/quinone or excess halides
    let halogen_count = mol
        .atoms
        .iter()
        .filter(|a| matches!(a.element.as_str(), "CL" | "BR" | "I"))
        .count();
    if halogen_count >= 5 {
        pains_alerts.push("polyhalogenated_lipophilic_burden".to_string());
    }
    for (i, a) in mol.atoms.iter().enumerate() {
        if a.element == "N"
            && let Some(nbrs) = mol.adj.get(i)
        {
            let o_double = nbrs
                .iter()
                .filter(|&&(nb, bid)| {
                    mol.atoms[nb].element == "O" && mol.bonds[bid].order == BondOrder::Double
                })
                .count();
            if o_double >= 2 {
                pains_alerts.push("nitro_alert".to_string());
            }
            if nbrs
                .iter()
                .filter(|&&(nb, bid)| {
                    mol.atoms[nb].element == "N" && mol.bonds[bid].order == BondOrder::Double
                })
                .count()
                >= 1
            {
                pains_alerts.push("azo_hydrazone_alert".to_string());
            }
        }
    }
    pains_alerts.sort();
    pains_alerts.dedup();

    let mut lipinski_violations = 0usize;
    if mw > 500.0 {
        lipinski_violations += 1;
    }
    if clogp > 5.0 {
        lipinski_violations += 1;
    }
    if hbd > 5 {
        lipinski_violations += 1;
    }
    if hba > 10 {
        lipinski_violations += 1;
    }

    let veber_pass = rotatable_bonds <= 10 && tpsa <= 140.0;
    let bbb_penetrant = mw <= 450.0 && tpsa <= 90.0 && hbd <= 3 && (1.0..=4.8).contains(&clogp);
    let herg_risk = clogp > 4.2
        && mol
            .atoms
            .iter()
            .any(|a| a.element == "N" && (a.formal_charge > 0 || a.charge > 0.1));

    // Simplified Bickerton QED-style desirability in [0, 1]
    let d_mw = (1.0 - ((mw - 340.0).abs() / 260.0)).clamp(0.1, 1.0);
    let d_logp = (1.0 - ((clogp - 2.6).abs() / 3.8)).clamp(0.1, 1.0);
    let d_tpsa = (1.0 - ((tpsa - 68.0).abs() / 95.0)).clamp(0.1, 1.0);
    let d_rot = (1.0 - (rotatable_bonds as f64 / 12.0)).clamp(0.1, 1.0);
    let d_alert = if pains_alerts.is_empty() { 1.0 } else { 0.45 };
    let qed_score = (d_mw * d_logp * d_tpsa * d_rot * d_alert).powf(0.2);

    let mut admet_score = qed_score * 100.0;
    admet_score -= lipinski_violations as f64 * 10.0;
    if !veber_pass {
        admet_score -= 10.0;
    }
    if herg_risk {
        admet_score -= 12.0;
    }
    admet_score = admet_score.clamp(0.0, 100.0);

    AdmetProfile {
        mw,
        clogp,
        hbd,
        hba,
        tpsa,
        rotatable_bonds,
        heavy_atoms,
        lipinski_violations,
        veber_pass,
        bbb_penetrant,
        herg_risk,
        pains_alerts,
        qed_score,
        admet_score,
        backend_used: "dockcraft-standalone-admet".to_string(),
    }
}

/// Optional discovery hook for sibling `admetcraft-cli` binary if installed on `$PATH`.
#[must_use]
pub fn detect_sibling_clis() -> (bool, bool) {
    let has_admet = Command::new("admetcraft-cli")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    let has_chem = Command::new("chemcraft-cli")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    (has_admet, has_chem)
}
