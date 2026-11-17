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
    pub aromatic_rings: usize,
    pub heavy_atoms: usize,
    pub lipinski_violations: usize,
    pub veber_pass: bool,
    pub bbb_penetrant: bool,
    pub herg_risk: bool,
    pub pains_alerts: Vec<String>,
    pub pains_scope: String,
    pub qed_score: f64,
    pub admet_score: f64,
    pub backend_used: String,
}

/// Bickerton et al. (Nature Chemistry 2012) Asymmetric Double Sigmoidal (ADS) desirability function.
#[allow(clippy::too_many_arguments)]
fn ads(x: f64, a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, dx_max: f64) -> f64 {
    let t1 = 1.0 / (1.0 + (-(x - c + 0.5 * d) / e).exp());
    let t2 = 1.0 / (1.0 + (-(x - c - 0.5 * d) / f).exp());
    ((a + b * t1 * (1.0 - t2)) / dx_max).clamp(1e-6, 1.0)
}

/// Exact 8-descriptor Bickerton et al. (2012) QED score using published ADS parameters & weights.
/// Expects `[mw, alogp, hba, hbd, psa, rotb, arom, alerts]`.
fn compute_bickerton_qed_8(d: [f64; 8]) -> f64 {
    let [mw, alogp, hba, hbd, psa, rotb, arom, alerts] = d;
    let vals = [
        ads(
            mw,
            2.817_065_973,
            392.575_495_3,
            290.748_976_4,
            2.419_764_353,
            49.223_256_77,
            65.370_517_07,
            104.980_556_1,
        ),
        ads(
            alogp,
            3.172_690_585,
            137.862_475_1,
            2.534_937_431,
            4.581_497_897,
            0.822_739_154,
            0.576_295_591,
            131.318_660_4,
        ),
        ads(
            hba,
            2.948_620_388,
            160.460_597_2,
            3.615_294_657,
            4.435_986_202,
            0.290_141_953,
            1.300_669_958,
            148.776_304_6,
        ),
        ads(
            hbd,
            1.618_662_227,
            1_010.051_101,
            0.985_094_388,
            0.000_000_001,
            0.713_820_843,
            0.920_922_555,
            258.163_261_6,
        ),
        ads(
            psa,
            1.876_861_559,
            125.223_265_7,
            62.907_735_54,
            87.833_666_14,
            12.019_998_24,
            28.513_247_32,
            104.568_616_7,
        ),
        ads(
            rotb,
            0.010_000_000,
            272.412_142_7,
            2.558_379_970,
            1.565_547_684,
            1.271_567_166,
            2.758_063_707,
            105.442_040_3,
        ),
        ads(
            arom,
            3.217_788_970,
            957.737_410_8,
            2.274_627_939,
            0.000_000_001,
            1.317_690_384,
            0.375_760_881,
            312.337_261_0,
        ),
        ads(
            alerts,
            0.010_000_000,
            1_199.094_025,
            -0.090_028_83,
            0.000_000_001,
            0.185_904_477,
            0.875_193_782,
            417.725_314_0,
        ),
    ];
    let weights = [0.66_f64, 0.46, 0.05, 0.61, 0.06, 0.65, 0.48, 0.95];
    let w_sum: f64 = weights.iter().sum();
    let weighted_ln: f64 = vals
        .iter()
        .zip(weights.iter())
        .map(|(&d, &w)| w * d.ln())
        .sum();
    (weighted_ln / w_sum).exp().clamp(0.0, 1.0)
}

fn has_exocyclic_double(
    mol: &Molecule3D,
    ring_atom: usize,
    target_el: &str,
    ring: &[usize],
) -> bool {
    mol.adj.get(ring_atom).is_some_and(|nbrs| {
        nbrs.iter().any(|&(nb, bid)| {
            !ring.contains(&nb)
                && mol.atoms[nb].element == target_el
                && mol.bonds[bid].order == BondOrder::Double
        })
    })
}

fn detect_pains_alerts(mol: &Molecule3D) -> Vec<String> {
    let mut alerts = Vec::new();

    // 1. Polyhalogenated lipophilic frequent hitters
    let halogen_count = mol
        .atoms
        .iter()
        .filter(|a| matches!(a.element.as_str(), "Cl" | "Br" | "I"))
        .count();
    if halogen_count >= 5 {
        alerts.push("polyhalogenated_lipophilic_burden".to_string());
    }

    // 2. Nitrogen-based alerts: nitro, azo/hydrazone, acylhydrazide, isothiocyanate
    for (i, a) in mol.atoms.iter().enumerate() {
        let Some(nbrs) = mol.adj.get(i) else {
            continue;
        };
        if a.element == "N" {
            let o_double = nbrs
                .iter()
                .filter(|&&(nb, bid)| {
                    mol.atoms[nb].element == "O" && mol.bonds[bid].order == BondOrder::Double
                })
                .count();
            if o_double >= 2
                || (o_double == 1
                    && nbrs.iter().any(|&(nb, _)| {
                        mol.atoms[nb].element == "O" && mol.atoms[nb].formal_charge < 0
                    }))
            {
                alerts.push("nitro_group".to_string());
            }
            if nbrs.iter().any(|&(nb, bid)| {
                mol.atoms[nb].element == "N" && mol.bonds[bid].order == BondOrder::Double
            }) {
                alerts.push("azo_group".to_string());
            }
            // Hydrazone C=N-N or Acylhydrazide C(=O)-N-N
            let has_n_single = nbrs.iter().any(|&(nb, bid)| {
                mol.atoms[nb].element == "N" && mol.bonds[bid].order == BondOrder::Single
            });
            if has_n_single {
                let has_c_double = nbrs.iter().any(|&(nb, bid)| {
                    mol.atoms[nb].element == "C" && mol.bonds[bid].order == BondOrder::Double
                });
                let has_acyl_c = nbrs.iter().any(|&(nb, _)| {
                    mol.atoms[nb].element == "C"
                        && mol.adj.get(nb).is_some_and(|cn| {
                            cn.iter().any(|&(o, ob)| {
                                matches!(mol.atoms[o].element.as_str(), "O" | "S")
                                    && mol.bonds[ob].order == BondOrder::Double
                            })
                        })
                });
                if has_c_double || has_acyl_c {
                    alerts.push("hydrazone_or_acylhydrazide".to_string());
                }
            }
            // Isothiocyanate -N=C=S
            if nbrs.iter().any(|&(c, bid)| {
                mol.atoms[c].element == "C"
                    && mol.bonds[bid].order == BondOrder::Double
                    && mol.adj.get(c).is_some_and(|cn| {
                        cn.iter().any(|&(s, sb)| {
                            mol.atoms[s].element == "S" && mol.bonds[sb].order == BondOrder::Double
                        })
                    })
            }) {
                alerts.push("isothiocyanate".to_string());
            }
        }
    }

    // 3. Michael acceptor: non-aromatic C=C adjacent to C(=O), S(=O)2, or C#N
    for b in &mol.bonds {
        if b.order != BondOrder::Double {
            continue;
        }
        let u = b.u;
        let v = b.v;
        if mol.atoms[u].element != "C"
            || mol.atoms[v].element != "C"
            || mol.atoms[u].aromatic
            || mol.atoms[v].aromatic
        {
            continue;
        }
        let has_ewg = |c_idx: usize, other: usize| -> bool {
            mol.adj.get(c_idx).is_some_and(|nbrs| {
                nbrs.iter().any(|&(nb, _)| {
                    nb != other
                        && ((mol.atoms[nb].element == "C"
                            && mol.adj.get(nb).is_some_and(|n2| {
                                n2.iter().any(|&(o, ob)| {
                                    (mol.atoms[o].element == "O"
                                        && mol.bonds[ob].order == BondOrder::Double)
                                        || (mol.atoms[o].element == "N"
                                            && mol.bonds[ob].order == BondOrder::Triple)
                                })
                            }))
                            || (mol.atoms[nb].element == "S"
                                && mol.adj.get(nb).is_some_and(|n2| {
                                    n2.iter()
                                        .filter(|&&(o, ob)| {
                                            mol.atoms[o].element == "O"
                                                && mol.bonds[ob].order == BondOrder::Double
                                        })
                                        .count()
                                        >= 2
                                })))
                })
            })
        };
        if has_ewg(u, v) || has_ewg(v, u) {
            alerts.push("michael_acceptor".to_string());
        }
    }

    // 4. Ring-based PAINS: strained 3-rings (epoxide/aziridine), rhodanine/thiohydantoin, quinone, catechol
    for ring in &mol.rings {
        if ring.len() == 3
            && ring
                .iter()
                .any(|&i| matches!(mol.atoms[i].element.as_str(), "O" | "N" | "S"))
        {
            alerts.push("strained_3_heterocycle".to_string());
        }
        if ring.len() == 5 {
            let has_n = ring.iter().any(|&i| mol.atoms[i].element == "N");
            let has_s = ring.iter().any(|&i| mol.atoms[i].element == "S");
            let has_thione = ring
                .iter()
                .any(|&i| mol.atoms[i].element == "C" && has_exocyclic_double(mol, i, "S", ring));
            let has_carbonyl = ring
                .iter()
                .any(|&i| mol.atoms[i].element == "C" && has_exocyclic_double(mol, i, "O", ring));
            if has_n && (has_s || has_thione) && (has_thione || has_carbonyl) {
                if has_s && has_thione {
                    alerts.push("rhodanine_core".to_string());
                } else if has_thione || has_carbonyl {
                    alerts.push("thiohydantoin_or_thiazolidinone".to_string());
                }
            }
        }
        if ring.len() == 6 {
            let n_exocyclic_o = ring
                .iter()
                .filter(|&&i| {
                    mol.atoms[i].element == "C" && has_exocyclic_double(mol, i, "O", ring)
                })
                .count();
            if n_exocyclic_o >= 2 {
                alerts.push("quinone_core".to_string());
            }
            if ring.iter().all(|&i| mol.atoms[i].aromatic) {
                // Catechol: two adjacent ring carbons bearing single-bonded OH
                let is_oh_subst = |ri: usize| -> bool {
                    mol.adj.get(ri).is_some_and(|nbrs| {
                        nbrs.iter().any(|&(nb, bid)| {
                            !ring.contains(&nb)
                                && mol.atoms[nb].element == "O"
                                && mol.bonds[bid].order == BondOrder::Single
                                && mol.adj.get(nb).is_some_and(|on| {
                                    on.iter()
                                        .all(|&(x, _)| x == ri || mol.atoms[x].element == "H")
                                })
                        })
                    })
                };
                for w in 0..ring.len() {
                    let u = ring[w];
                    let v = ring[(w + 1) % ring.len()];
                    if is_oh_subst(u) && is_oh_subst(v) {
                        alerts.push("catechol_motif".to_string());
                    }
                }
            }
        }
    }

    alerts.sort();
    alerts.dedup();
    alerts
}

#[must_use]
pub fn evaluate_admet(mol: &Molecule3D) -> AdmetProfile {
    let mw = mol.molecular_weight();
    let heavy_atoms = mol.heavy_atom_count();
    let rotatable_bonds = mol.rotatable_bonds.len();
    let aromatic_rings = mol
        .rings
        .iter()
        .filter(|r| !r.is_empty() && r.iter().all(|&i| mol.atoms[i].aromatic))
        .count();

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
            "Cl" => clogp += 0.89,
            "Br" => clogp += 1.09,
            "I" => clogp += 1.35,
            _ => {}
        }
    }

    let pains_alerts = detect_pains_alerts(mol);

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

    let qed_score = compute_bickerton_qed_8([
        mw,
        clogp,
        hba as f64,
        hbd as f64,
        tpsa,
        rotatable_bonds as f64,
        aromatic_rings as f64,
        pains_alerts.len() as f64,
    ]);

    let mut admet_score = qed_score * 100.0;
    admet_score -= lipinski_violations as f64 * 10.0;
    admet_score -= (pains_alerts.len() as f64) * 18.0;
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
        aromatic_rings,
        heavy_atoms,
        lipinski_violations,
        veber_pass,
        bbb_penetrant,
        herg_risk,
        pains_alerts,
        pains_scope: "core-medchem-filters (rhodanine, thiazolidinone, quinone, catechol, michael, nitro, azo/hydrazone, isothiocyanate, strained-3ring, polyhalide)".to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use struct_core::parse_smiles_to_3d;

    #[test]
    fn bickerton_qed_and_rhodanine_quinone_halide_pains_filters_work()
    -> Result<(), struct_core::StructError> {
        let rhodanine = parse_smiles_to_3d("O=C1NC(=S)SC1=Cc2ccccc2", Some("Rhodanine_PAINS"))?;
        let pr_rho = evaluate_admet(&rhodanine);
        assert!(
            pr_rho.pains_alerts.iter().any(|a| a.contains("rhodanine")),
            "rhodanine must trigger PAINS alert, got {:?}",
            pr_rho.pains_alerts
        );

        let quinone = parse_smiles_to_3d("O=C1C=CC(=O)C=C1", Some("Benzoquinone"))?;
        let pr_q = evaluate_admet(&quinone);
        assert!(pr_q.pains_alerts.contains(&"quinone_core".to_string()));

        let halobenzene = parse_smiles_to_3d("Clc1c(Cl)c(Cl)c(Cl)c(Cl)c1Br", Some("Hexahalo"))?;
        let pr_h = evaluate_admet(&halobenzene);
        assert!(
            pr_h.pains_alerts
                .contains(&"polyhalogenated_lipophilic_burden".to_string())
        );

        // Clean drug-like scaffold (ibuprofen-like) should score well on 8-descriptor QED with 0 PAINS
        let clean = parse_smiles_to_3d("CC(C)Cc1ccc(C(C)C(=O)O)cc1", Some("Ibuprofen"))?;
        let pr_clean = evaluate_admet(&clean);
        assert!(pr_clean.pains_alerts.is_empty());
        assert!((pr_clean.mw - 205.28).abs() < 1.5, "mw was {}", pr_clean.mw);
        assert!(pr_clean.qed_score > 0.65 && pr_clean.qed_score < 0.95);
        assert!(pr_clean.admet_score > pr_rho.admet_score);
        Ok(())
    }
}
