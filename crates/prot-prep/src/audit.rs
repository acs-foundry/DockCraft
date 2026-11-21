#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::{Deserialize, Serialize};
use struct_core::{Atom3D, BondOrder, Molecule3D, normalize_element};

/// Classification of non-standard / `HETATM` residues encountered in PDB/mmCIF files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HetClass {
    Water,
    BufferOrSolvent,
    CatalyticMetal,
    Cofactor,
    ModifiedResidue,
    OrganicLigand,
}

impl HetClass {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Water => "Water",
            Self::BufferOrSolvent => "Buffer/Solvent",
            Self::CatalyticMetal => "CatalyticMetal",
            Self::Cofactor => "Cofactor",
            Self::ModifiedResidue => "ModifiedResidue",
            Self::OrganicLigand => "OrganicLigand",
        }
    }
}

/// Severity of an automated structural audit finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditSeverity {
    Info,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditIssue {
    pub severity: AuditSeverity,
    pub category: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingResidueRecord {
    pub chain_id: char,
    pub res_name: String,
    pub res_seq: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingAtomRecord {
    pub chain_id: char,
    pub res_name: String,
    pub res_seq: i32,
    pub missing_atoms: Vec<String>,
    pub near_pocket: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AltlocResolution {
    pub chain_id: char,
    pub res_name: String,
    pub res_seq: i32,
    pub chosen_altloc: char,
    pub chosen_occupancy: f64,
    pub discarded_altlocs: Vec<char>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainBreak {
    pub chain_id: char,
    pub res_before: String,
    pub res_after: String,
    pub distance_a: f64,
    pub near_pocket: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HetGroupInfo {
    pub chain_id: char,
    pub res_name: String,
    pub res_seq: i32,
    pub atom_count: usize,
    pub class: HetClass,
    pub description: Option<String>,
    pub kept_in_receptor: bool,
    pub centroid: [f64; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainSummary {
    pub chain_id: char,
    pub protein_residues: usize,
    pub protein_heavy_atoms: usize,
    pub modified_residues: usize,
    pub ligands: usize,
    pub cofactors: usize,
    pub metals: usize,
    pub waters: usize,
    pub mean_b_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StericClash {
    pub atom_a: String,
    pub atom_b: String,
    pub distance_a: f64,
}

/// Complete structural quality report produced before docking / MIF site analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralAudit {
    pub structure_id: String,
    pub title: String,
    pub experimental_method: String,
    pub resolution_a: Option<f64>,
    pub r_work: Option<f64>,
    pub r_free: Option<f64>,
    pub is_alphafold: bool,
    pub quality_grade: String,
    pub chains: Vec<ChainSummary>,
    pub selected_chains: Vec<char>,
    pub total_protein_residues: usize,
    pub mean_b_factor_or_plddt: f64,
    pub pocket_mean_b_factor_or_plddt: Option<f64>,
    pub high_b_or_low_plddt_pocket_residues: Vec<String>,
    pub missing_residues_header: Vec<MissingResidueRecord>,
    pub missing_heavy_atoms: Vec<MissingAtomRecord>,
    pub chain_breaks: Vec<ChainBreak>,
    pub altlocs_resolved: Vec<AltlocResolution>,
    pub disulfide_bonds: Vec<String>,
    pub het_groups: Vec<HetGroupInfo>,
    pub steric_clashes: Vec<StericClash>,
    pub issues: Vec<AuditIssue>,
}

/// Raw parsed atom before altloc resolution and HETATM filtering.
#[derive(Debug, Clone)]
pub struct RawAtom {
    pub serial: i32,
    pub name: String,
    pub alt_loc: char,
    pub res_name: String,
    pub chain_id: char,
    pub res_seq: i32,
    pub is_hetatm: bool,
    pub coords: [f64; 3],
    pub occupancy: f64,
    pub b_factor: f64,
    pub element: String,
    pub formal_charge: i8,
}

/// Raw parsed structure with crystallographic / AlphaFold metadata preserved.
#[derive(Debug, Clone)]
pub struct RawStructure {
    pub structure_id: String,
    pub title: String,
    pub experimental_method: String,
    pub resolution_a: Option<f64>,
    pub r_work: Option<f64>,
    pub r_free: Option<f64>,
    pub is_alphafold_header: bool,
    pub missing_residues: Vec<MissingResidueRecord>,
    pub missing_atoms_header: Vec<(char, String, i32, Vec<String>)>,
    pub modres_keys: BTreeSet<(char, String, i32)>,
    pub hetnam: HashMap<String, String>,
    pub ssbonds_header: Vec<String>,
    pub atoms: Vec<RawAtom>,
    pub conects: Vec<(i32, i32)>,
}

/// Returns `true` if `res_name` is a standard 20 amino acid or Amber/PROPKA protonation alias.
#[must_use]
pub fn is_standard_protein_res(res_name: &str) -> bool {
    matches!(
        res_name.trim().to_ascii_uppercase().as_str(),
        "ALA"
            | "ARG"
            | "ASN"
            | "ASP"
            | "ASH"
            | "CYS"
            | "CYX"
            | "CYM"
            | "GLN"
            | "GLU"
            | "GLH"
            | "GLY"
            | "HIS"
            | "HID"
            | "HIE"
            | "HIP"
            | "ILE"
            | "LEU"
            | "LYS"
            | "LYN"
            | "MET"
            | "PHE"
            | "PRO"
            | "SER"
            | "THR"
            | "TRP"
            | "TYR"
            | "VAL"
    )
}

#[must_use]
pub fn is_water_res(res_name: &str) -> bool {
    matches!(
        res_name.trim().to_ascii_uppercase().as_str(),
        "HOH" | "WAT" | "DOD" | "TIP" | "TIP3" | "SOL"
    )
}

#[must_use]
pub fn is_modified_protein_res(res_name: &str) -> bool {
    matches!(
        res_name.trim().to_ascii_uppercase().as_str(),
        "TPO"
            | "SEP"
            | "PTR"
            | "MSE"
            | "CSO"
            | "CSD"
            | "CME"
            | "OCS"
            | "MLY"
            | "KCX"
            | "PCA"
            | "ACE"
            | "NME"
    )
}

#[must_use]
pub fn is_catalytic_metal_res(res_name: &str, atoms: &[&RawAtom]) -> bool {
    if atoms.len() == 1 {
        let el = atoms[0].element.as_str();
        if matches!(el, "Zn" | "Mg" | "Fe" | "Mn" | "Ca" | "Cu" | "Co" | "Ni") {
            return true;
        }
    }
    matches!(
        res_name.trim().to_ascii_uppercase().as_str(),
        "ZN" | "MG" | "FE" | "FE2" | "MN" | "CA" | "CU" | "CU1" | "CO" | "NI" | "HEM" | "HEC"
    ) && atoms.len() <= 2
}

#[must_use]
pub fn is_cofactor_res(res_name: &str) -> bool {
    matches!(
        res_name.trim().to_ascii_uppercase().as_str(),
        "ATP"
            | "ADP"
            | "AMP"
            | "ANP"
            | "ACP"
            | "GTP"
            | "GDP"
            | "GNP"
            | "NAD"
            | "NAI"
            | "NAP"
            | "NDP"
            | "FAD"
            | "FMN"
            | "HEM"
            | "HEC"
            | "SAH"
            | "SAM"
            | "PLP"
            | "COA"
            | "ACO"
            | "TPP"
            | "THF"
            | "BTN"
            | "SF4"
            | "FES"
    )
}

#[must_use]
pub fn is_buffer_or_solvent_res(res_name: &str) -> bool {
    matches!(
        res_name.trim().to_ascii_uppercase().as_str(),
        "EDO"
            | "GOL"
            | "PEG"
            | "PGE"
            | "PG4"
            | "1PE"
            | "2PE"
            | "DMS"
            | "DMSO"
            | "ACT"
            | "ACY"
            | "SO4"
            | "PO4"
            | "NO3"
            | "CO3"
            | "CIT"
            | "TRS"
            | "MES"
            | "EPE"
            | "MPD"
            | "FMT"
            | "CL"
            | "BR"
            | "IOD"
            | "NA"
            | "K"
            | "NH4"
            | "IPA"
            | "EOH"
            | "MOH"
            | "BME"
            | "DTT"
            | "UNX"
            | "UNL"
    )
}

/// Classifies a `HETATM` residue group into Water, Buffer, Metal, Cofactor, ModifiedResidue, or OrganicLigand.
#[must_use]
pub fn classify_het_group(res_name: &str, atoms: &[&RawAtom], is_modres: bool) -> HetClass {
    let rn = res_name.trim().to_ascii_uppercase();
    if is_water_res(&rn) {
        return HetClass::Water;
    }
    if is_modres || is_modified_protein_res(&rn) {
        return HetClass::ModifiedResidue;
    }
    if is_catalytic_metal_res(&rn, atoms) {
        return HetClass::CatalyticMetal;
    }
    if is_cofactor_res(&rn) {
        return HetClass::Cofactor;
    }
    if is_buffer_or_solvent_res(&rn) {
        return HetClass::BufferOrSolvent;
    }
    // Single-atom non-metal ion (e.g. Cl-, Na+, Br-)
    if atoms.len() <= 2 && !atoms.iter().any(|a| a.element == "C") {
        return HetClass::BufferOrSolvent;
    }
    HetClass::OrganicLigand
}

fn infer_element(name: &str) -> String {
    let t = name.trim_start_matches(|c: char| c.is_ascii_digit());
    let up = t.to_ascii_uppercase();
    if up.starts_with("CL") {
        "Cl".to_string()
    } else if up.starts_with("BR") {
        "Br".to_string()
    } else if up.starts_with("ZN") {
        "Zn".to_string()
    } else if up.starts_with("MG") {
        "Mg".to_string()
    } else if up.starts_with("FE") {
        "Fe".to_string()
    } else if up.starts_with("MN") {
        "Mn".to_string()
    } else if up.starts_with("CA") && t.len() == 2 {
        "Ca".to_string()
    } else {
        t.chars()
            .next()
            .filter(|c| c.is_ascii_alphabetic())
            .map_or_else(|| "C".to_string(), |c| c.to_ascii_uppercase().to_string())
    }
}

/// Parses raw PDB or mmCIF text preserving headers, altlocs, occupancies, MODRES, and REMARKs.
pub fn parse_raw_structure(input: &str, format_hint: &str) -> Result<RawStructure, String> {
    if input.trim().is_empty() {
        return Err("empty structure input".to_string());
    }
    let is_cif = format_hint.eq_ignore_ascii_case("cif")
        || format_hint.eq_ignore_ascii_case("mmcif")
        || input.contains("_atom_site.");
    if is_cif {
        parse_raw_mmcif(input)
    } else {
        parse_raw_pdb(input)
    }
}

fn parse_raw_pdb(input: &str) -> Result<RawStructure, String> {
    let mut rs = RawStructure {
        structure_id: "PDB".to_string(),
        title: String::new(),
        experimental_method: String::new(),
        resolution_a: None,
        r_work: None,
        r_free: None,
        is_alphafold_header: false,
        missing_residues: Vec::new(),
        missing_atoms_header: Vec::new(),
        modres_keys: BTreeSet::new(),
        hetnam: HashMap::new(),
        ssbonds_header: Vec::new(),
        atoms: Vec::new(),
        conects: Vec::new(),
    };

    let mut first_model_done = false;
    let mut in_remark465_table = false;
    let mut in_remark470_table = false;

    for raw in input.lines() {
        if raw.starts_with("ENDMDL") {
            first_model_done = true;
            continue;
        }
        if first_model_done && (raw.starts_with("ATOM  ") || raw.starts_with("HETATM")) {
            break;
        }
        if let Some(rest) = raw.strip_prefix("HEADER") {
            let trimmed = rest.trim();
            if let Some(id) = trimmed.split_whitespace().last() {
                rs.structure_id = id.to_string();
            }
            if trimmed.to_ascii_uppercase().contains("ALPHAFOLD") {
                rs.is_alphafold_header = true;
            }
            continue;
        }
        if let Some(rest) = raw.strip_prefix("TITLE ") {
            let t = rest.get(4..).unwrap_or(rest).trim();
            if !t.is_empty() {
                if !rs.title.is_empty() {
                    rs.title.push(' ');
                }
                rs.title.push_str(t);
                if t.to_ascii_uppercase().contains("ALPHAFOLD")
                    || t.to_ascii_uppercase().contains("ESMFOLD")
                {
                    rs.is_alphafold_header = true;
                }
            }
            continue;
        }
        if let Some(rest) = raw.strip_prefix("EXPDTA") {
            rs.experimental_method = rest.trim().to_string();
            continue;
        }
        if let Some(rest) = raw.strip_prefix("REMARK   2 RESOLUTION.") {
            for tok in rest.split_whitespace() {
                if let Ok(v) = tok.parse::<f64>()
                    && v.is_finite()
                    && v > 0.0
                    && v < 30.0
                {
                    rs.resolution_a = Some(v);
                    break;
                }
            }
            continue;
        }
        if raw.starts_with("REMARK   3") {
            let up = raw.to_ascii_uppercase();
            if up.contains("FREE R VALUE")
                && !up.contains("SELECTION")
                && !up.contains("SIZE")
                && !up.contains("COUNT")
                && !up.contains("BIN")
                && let Some(val) = raw
                    .split(':')
                    .nth(1)
                    .and_then(|s| s.split_whitespace().next())
                    .and_then(|s| s.parse::<f64>().ok())
                && (0.0..1.0).contains(&val)
                && rs.r_free.is_none()
            {
                rs.r_free = Some(val);
            } else if up.contains("R VALUE")
                && up.contains("WORKING SET")
                && let Some(val) = raw
                    .split(':')
                    .nth(1)
                    .and_then(|s| s.split_whitespace().next())
                    .and_then(|s| s.parse::<f64>().ok())
                && (0.0..1.0).contains(&val)
                && rs.r_work.is_none()
            {
                rs.r_work = Some(val);
            }
            continue;
        }
        if raw.starts_with("REMARK 465") {
            let rest = raw.get(10..).unwrap_or("").trim();
            if rest.starts_with("M RES C SSSEQI") {
                in_remark465_table = true;
                continue;
            }
            if in_remark465_table && !rest.is_empty() {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                // Format: [M] RES C SSSEQI
                let (res_tok, ch_tok, seq_tok) = if toks.len() == 3 {
                    (toks[0], toks[1], toks[2])
                } else if toks.len() >= 4 {
                    (toks[1], toks[2], toks[3])
                } else {
                    continue;
                };
                if let Some(ch) = ch_tok.chars().next()
                    && let Ok(seq) = seq_tok
                        .trim_end_matches(|c: char| c.is_ascii_alphabetic())
                        .parse::<i32>()
                {
                    rs.missing_residues.push(MissingResidueRecord {
                        chain_id: ch,
                        res_name: res_tok.to_string(),
                        res_seq: seq,
                    });
                }
            }
            continue;
        }
        if raw.starts_with("REMARK 470") {
            let rest = raw.get(10..).unwrap_or("").trim();
            if rest.starts_with("M RES CSSEQI") {
                in_remark470_table = true;
                continue;
            }
            if in_remark470_table && !rest.is_empty() {
                let toks: Vec<&str> = rest.split_whitespace().collect();
                if toks.len() >= 4 {
                    let res_tok = toks[0];
                    let ch_tok = toks[1];
                    let seq_tok = toks[2];
                    if let Some(ch) = ch_tok.chars().next()
                        && let Ok(seq) = seq_tok
                            .trim_end_matches(|c: char| c.is_ascii_alphabetic())
                            .parse::<i32>()
                    {
                        let atoms: Vec<String> =
                            toks[3..].iter().map(|s| (*s).to_string()).collect();
                        rs.missing_atoms_header
                            .push((ch, res_tok.to_string(), seq, atoms));
                    }
                }
            }
            continue;
        }
        if raw.starts_with("MODRES") {
            let res_name = raw.get(12..15).unwrap_or("").trim().to_string();
            let chain_id = raw
                .get(16..17)
                .unwrap_or("A")
                .trim()
                .chars()
                .next()
                .unwrap_or('A');
            if let Ok(seq) = raw.get(18..22).unwrap_or("").trim().parse::<i32>()
                && !res_name.is_empty()
            {
                rs.modres_keys.insert((chain_id, res_name, seq));
            }
            continue;
        }
        if raw.starts_with("HETNAM") {
            let het_id = raw.get(11..14).unwrap_or("").trim().to_string();
            let desc = raw.get(15..).unwrap_or("").trim();
            if !het_id.is_empty() && !desc.is_empty() {
                let entry = rs.hetnam.entry(het_id).or_default();
                if !entry.is_empty() {
                    entry.push(' ');
                }
                entry.push_str(desc);
            }
            continue;
        }
        if raw.starts_with("SSBOND") {
            let c1 = raw.get(15..16).unwrap_or("A").trim();
            let s1 = raw.get(17..21).unwrap_or("").trim();
            let c2 = raw.get(29..30).unwrap_or("A").trim();
            let s2 = raw.get(31..35).unwrap_or("").trim();
            if !s1.is_empty() && !s2.is_empty() {
                rs.ssbonds_header.push(format!("{c1}:CYS{s1}-{c2}:CYS{s2}"));
            }
            continue;
        }
        if raw.starts_with("REMARK") && raw.to_ascii_uppercase().contains("ALPHAFOLD") {
            rs.is_alphafold_header = true;
            continue;
        }
        if raw.starts_with("ATOM  ") || raw.starts_with("HETATM") {
            let is_hetatm = raw.starts_with("HETATM");
            let serial: i32 = raw
                .get(6..11)
                .unwrap_or("")
                .trim()
                .parse()
                .unwrap_or((rs.atoms.len() + 1) as i32);
            let name = raw.get(12..16).unwrap_or("C").trim().to_string();
            let alt_loc = raw
                .get(16..17)
                .and_then(|s| s.chars().next())
                .unwrap_or(' ');
            let res_name = raw.get(17..20).unwrap_or("UNK").trim().to_string();
            let chain_id = raw
                .get(21..22)
                .unwrap_or("A")
                .trim()
                .chars()
                .next()
                .unwrap_or('A');
            let res_seq: i32 = raw.get(22..26).unwrap_or("1").trim().parse().unwrap_or(1);
            let Ok(x) = raw.get(30..38).unwrap_or("").trim().parse::<f64>() else {
                continue;
            };
            let Ok(y) = raw.get(38..46).unwrap_or("").trim().parse::<f64>() else {
                continue;
            };
            let Ok(z) = raw.get(46..54).unwrap_or("").trim().parse::<f64>() else {
                continue;
            };
            if !x.is_finite() || !y.is_finite() || !z.is_finite() {
                continue;
            }
            let occupancy: f64 = raw
                .get(54..60)
                .unwrap_or("1.0")
                .trim()
                .parse()
                .unwrap_or(1.0);
            let b_factor: f64 = raw
                .get(60..66)
                .unwrap_or("20.0")
                .trim()
                .parse()
                .unwrap_or(20.0);
            let raw_el = raw.get(76..78).unwrap_or("").trim();
            let element = if raw_el.is_empty() {
                infer_element(&name)
            } else {
                normalize_element(raw_el)
            };
            let mut formal_charge = 0i8;
            if let Some(chg_str) = raw.get(78..80).map(str::trim)
                && chg_str.len() == 2
            {
                let bs = chg_str.as_bytes();
                if bs[0].is_ascii_digit() && matches!(bs[1], b'+' | b'-') {
                    let mag = (bs[0] - b'0') as i8;
                    formal_charge = if bs[1] == b'+' { mag } else { -mag };
                }
            }
            rs.atoms.push(RawAtom {
                serial,
                name,
                alt_loc,
                res_name,
                chain_id,
                res_seq,
                is_hetatm,
                coords: [x, y, z],
                occupancy,
                b_factor,
                element,
                formal_charge,
            });
            continue;
        }
        if let Some(rest) = raw.strip_prefix("CONECT") {
            let mut nums = Vec::new();
            let mut p = 0;
            while p + 5 <= rest.len() {
                if let Ok(v) = rest[p..p + 5].trim().parse::<i32>() {
                    nums.push(v);
                }
                p += 5;
            }
            if let Some(&src) = nums.first() {
                for &dst in &nums[1..] {
                    rs.conects.push((src, dst));
                }
            }
        }
    }

    if rs.atoms.is_empty() {
        return Err("no ATOM/HETATM records found in PDB".to_string());
    }
    Ok(rs)
}

fn parse_raw_mmcif(input: &str) -> Result<RawStructure, String> {
    let mut rs = RawStructure {
        structure_id: "mmCIF".to_string(),
        title: String::new(),
        experimental_method: String::new(),
        resolution_a: None,
        r_work: None,
        r_free: None,
        is_alphafold_header: input.to_ascii_uppercase().contains("ALPHAFOLD"),
        missing_residues: Vec::new(),
        missing_atoms_header: Vec::new(),
        modres_keys: BTreeSet::new(),
        hetnam: HashMap::new(),
        ssbonds_header: Vec::new(),
        atoms: Vec::new(),
        conects: Vec::new(),
    };

    let mut in_atom_site = false;
    let mut cols: Vec<String> = Vec::new();
    let mut col_idx: HashMap<String, usize> = HashMap::new();

    for raw in input.lines() {
        let t = raw.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if let Some(dn) = t.strip_prefix("data_") {
            rs.structure_id = dn.trim().to_string();
            continue;
        }
        if t.starts_with("_refine.ls_d_res_high")
            && let Some(v) = t
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse::<f64>().ok())
        {
            rs.resolution_a = Some(v);
            continue;
        }
        if t.starts_with("_refine.ls_R_factor_R_free")
            && let Some(v) = t
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse::<f64>().ok())
        {
            rs.r_free = Some(v);
            continue;
        }
        if t.starts_with("_refine.ls_R_factor_R_work")
            && let Some(v) = t
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse::<f64>().ok())
        {
            rs.r_work = Some(v);
            continue;
        }
        if t.starts_with("_exptl.method") {
            rs.experimental_method = t
                .trim_start_matches("_exptl.method")
                .trim()
                .trim_matches('\'')
                .trim_matches('"')
                .to_string();
            continue;
        }
        if t == "loop_" {
            in_atom_site = false;
            cols.clear();
            col_idx.clear();
            continue;
        }
        if t.starts_with("_atom_site.") {
            in_atom_site = true;
            let key = t.split_whitespace().next().unwrap_or(t).to_string();
            col_idx.insert(key.clone(), cols.len());
            cols.push(key);
            continue;
        }
        if t.starts_with('_') {
            if in_atom_site {
                in_atom_site = false;
            }
            continue;
        }
        if !in_atom_site {
            continue;
        }
        let toks = tokenize_cif_line(t);
        if toks.len() < cols.len() {
            continue;
        }
        let get = |k: &str| {
            col_idx
                .get(k)
                .and_then(|&i| toks.get(i))
                .map(String::as_str)
        };
        // Only model 1
        if let Some(m) = get("_atom_site.pdbx_PDB_model_num")
            && m != "1"
            && m != "."
        {
            continue;
        }
        let Ok(x) = get("_atom_site.Cartn_x").unwrap_or("").parse::<f64>() else {
            continue;
        };
        let Ok(y) = get("_atom_site.Cartn_y").unwrap_or("").parse::<f64>() else {
            continue;
        };
        let Ok(z) = get("_atom_site.Cartn_z").unwrap_or("").parse::<f64>() else {
            continue;
        };
        let name = get("_atom_site.auth_atom_id")
            .or_else(|| get("_atom_site.label_atom_id"))
            .unwrap_or("C")
            .to_string();
        let alt_loc = get("_atom_site.label_alt_id")
            .and_then(|s| {
                if s == "." || s == "?" {
                    None
                } else {
                    s.chars().next()
                }
            })
            .unwrap_or(' ');
        let res_name = get("_atom_site.auth_comp_id")
            .or_else(|| get("_atom_site.label_comp_id"))
            .unwrap_or("UNK")
            .to_string();
        let chain_id = get("_atom_site.auth_asym_id")
            .or_else(|| get("_atom_site.label_asym_id"))
            .and_then(|s| s.chars().next())
            .unwrap_or('A');
        let res_seq: i32 = get("_atom_site.auth_seq_id")
            .or_else(|| get("_atom_site.label_seq_id"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let is_hetatm =
            get("_atom_site.group_PDB").is_some_and(|g| g.eq_ignore_ascii_case("HETATM"));
        let occupancy: f64 = get("_atom_site.occupancy")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        let b_factor: f64 = get("_atom_site.B_iso_or_equiv")
            .and_then(|s| s.parse().ok())
            .unwrap_or(20.0);
        let element = get("_atom_site.type_symbol")
            .map(normalize_element)
            .unwrap_or_else(|| infer_element(&name));
        let serial = get("_atom_site.id")
            .and_then(|s| s.parse().ok())
            .unwrap_or((rs.atoms.len() + 1) as i32);

        rs.atoms.push(RawAtom {
            serial,
            name,
            alt_loc,
            res_name,
            chain_id,
            res_seq,
            is_hetatm,
            coords: [x, y, z],
            occupancy,
            b_factor,
            element,
            formal_charge: 0,
        });
    }

    if rs.atoms.is_empty() {
        return Err("no _atom_site records found in mmCIF".to_string());
    }
    Ok(rs)
}

fn tokenize_cif_line(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    for c in line.chars() {
        if let Some(q) = quote {
            if c == q {
                quote = None;
                out.push(std::mem::take(&mut cur));
            } else {
                cur.push(c);
            }
        } else if matches!(c, '\'' | '"') && cur.is_empty() {
            quote = Some(c);
        } else if c.is_whitespace() {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        } else {
            cur.push(c);
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Canonical expected heavy atoms for every standard protein residue (excluding C-terminal `OXT`).
#[must_use]
pub fn expected_heavy_atoms(res_name: &str) -> Option<&'static [&'static str]> {
    match res_name.trim().to_ascii_uppercase().as_str() {
        "GLY" => Some(&["N", "CA", "C", "O"]),
        "ALA" => Some(&["N", "CA", "C", "O", "CB"]),
        "VAL" => Some(&["N", "CA", "C", "O", "CB", "CG1", "CG2"]),
        "LEU" => Some(&["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"]),
        "ILE" => Some(&["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"]),
        "SER" => Some(&["N", "CA", "C", "O", "CB", "OG"]),
        "THR" => Some(&["N", "CA", "C", "O", "CB", "OG1", "CG2"]),
        "CYS" | "CYX" | "CYM" => Some(&["N", "CA", "C", "O", "CB", "SG"]),
        "MET" => Some(&["N", "CA", "C", "O", "CB", "CG", "SD", "CE"]),
        "PRO" => Some(&["N", "CA", "C", "O", "CB", "CG", "CD"]),
        "ASP" | "ASH" => Some(&["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"]),
        "ASN" => Some(&["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"]),
        "GLU" | "GLH" => Some(&["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"]),
        "GLN" => Some(&["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"]),
        "LYS" | "LYN" => Some(&["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"]),
        "ARG" => Some(&[
            "N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2",
        ]),
        "HIS" | "HID" | "HIE" | "HIP" => {
            Some(&["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"])
        }
        "PHE" => Some(&[
            "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ",
        ]),
        "TYR" => Some(&[
            "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH",
        ]),
        "TRP" => Some(&[
            "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2",
        ]),
        _ => None,
    }
}

/// Filters `RawStructure` by chains, resolves alternate conformations by highest occupancy,
/// separates HETATM groups, extracts organic ligands, and builds the clean `Molecule3D` + `StructuralAudit`.
#[allow(clippy::too_many_arguments)]
pub fn build_clean_receptor_and_audit(
    raw: &RawStructure,
    selected_chains_filter: &[char],
    keep_cofactors: bool,
    keep_metals: bool,
    strip_ligands: bool,
    keep_all_waters: bool,
    pocket_center: Option<[f64; 3]>,
    pocket_radius: f64,
) -> (Molecule3D, Vec<Molecule3D>, StructuralAudit) {
    // 1. Discover all chains & compute per-residue altloc resolution (highest mean occupancy wins!)
    let mut all_chains: BTreeSet<char> = BTreeSet::new();
    let mut res_altlocs: BTreeMap<(char, i32, String), BTreeMap<char, Vec<f64>>> = BTreeMap::new();
    for a in &raw.atoms {
        all_chains.insert(a.chain_id);
        if a.alt_loc != ' ' {
            res_altlocs
                .entry((a.chain_id, a.res_seq, a.res_name.clone()))
                .or_default()
                .entry(a.alt_loc)
                .or_default()
                .push(a.occupancy);
        }
    }

    let mut chosen_altloc_map: HashMap<(char, i32, String), char> = HashMap::new();
    let mut altlocs_resolved: Vec<AltlocResolution> = Vec::new();
    for ((ch, seq, rname), alt_map) in &res_altlocs {
        if alt_map.is_empty() {
            continue;
        }
        let mut ranked: Vec<(char, f64)> = alt_map
            .iter()
            .map(|(&ac, occs)| {
                let mean = occs.iter().sum::<f64>() / (occs.len().max(1) as f64);
                (ac, mean)
            })
            .collect();
        // Sort highest occupancy first; tie-break alphabetically ('A' before 'B')
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let (best_c, best_occ) = ranked[0];
        chosen_altloc_map.insert((*ch, *seq, rname.clone()), best_c);
        if ranked.len() > 1 {
            altlocs_resolved.push(AltlocResolution {
                chain_id: *ch,
                res_name: rname.clone(),
                res_seq: *seq,
                chosen_altloc: best_c,
                chosen_occupancy: (best_occ * 100.0).round() / 100.0,
                discarded_altlocs: ranked[1..].iter().map(|&(c, _)| c).collect(),
            });
        }
    }

    let selected_chains: Vec<char> = if selected_chains_filter.is_empty() {
        all_chains.iter().copied().collect()
    } else {
        selected_chains_filter
            .iter()
            .copied()
            .filter(|c| all_chains.contains(c))
            .collect()
    };
    let chain_set: BTreeSet<char> = selected_chains.iter().copied().collect();

    // 2. Group atoms after altloc resolution (across ALL chains for summary, then filter for clean receptor)
    let resolved_atoms: Vec<&RawAtom> = raw
        .atoms
        .iter()
        .filter(|a| {
            if a.alt_loc == ' ' {
                return true;
            }
            chosen_altloc_map
                .get(&(a.chain_id, a.res_seq, a.res_name.clone()))
                .is_none_or(|&best| best == a.alt_loc)
        })
        .collect();

    // Group HETATM residues
    let mut het_by_res: BTreeMap<(char, i32, String), Vec<&RawAtom>> = BTreeMap::new();
    let mut prot_by_res: BTreeMap<(char, i32, String), Vec<&RawAtom>> = BTreeMap::new();
    for &a in &resolved_atoms {
        let is_mod = raw
            .modres_keys
            .contains(&(a.chain_id, a.res_name.clone(), a.res_seq))
            || is_modified_protein_res(&a.res_name);
        if (!a.is_hetatm && is_standard_protein_res(&a.res_name)) || is_mod {
            prot_by_res
                .entry((a.chain_id, a.res_seq, a.res_name.clone()))
                .or_default()
                .push(a);
        } else {
            het_by_res
                .entry((a.chain_id, a.res_seq, a.res_name.clone()))
                .or_default()
                .push(a);
        }
    }

    // 3. Per-chain summaries across the raw structure
    let mut chain_summaries = Vec::new();
    for &ch in &all_chains {
        let mut prot_res = 0usize;
        let mut prot_ha = 0usize;
        let mut mod_res = 0usize;
        let mut b_sum = 0.0;
        let mut b_cnt = 0usize;
        for ((c, _, rn), atoms) in &prot_by_res {
            if *c != ch {
                continue;
            }
            prot_res += 1;
            if is_modified_protein_res(rn) {
                mod_res += 1;
            }
            for a in atoms {
                if a.element != "H" {
                    prot_ha += 1;
                    b_sum += a.b_factor;
                    b_cnt += 1;
                }
            }
        }
        let mut ligs = 0usize;
        let mut cofs = 0usize;
        let mut mets = 0usize;
        let mut wats = 0usize;
        for ((c, seq, rn), atoms) in &het_by_res {
            if *c != ch {
                continue;
            }
            let cls =
                classify_het_group(rn, atoms, raw.modres_keys.contains(&(*c, rn.clone(), *seq)));
            match cls {
                HetClass::OrganicLigand => ligs += 1,
                HetClass::Cofactor => cofs += 1,
                HetClass::CatalyticMetal => mets += 1,
                HetClass::Water => wats += 1,
                HetClass::ModifiedResidue => mod_res += 1,
                HetClass::BufferOrSolvent => {}
            }
        }
        chain_summaries.push(ChainSummary {
            chain_id: ch,
            protein_residues: prot_res,
            protein_heavy_atoms: prot_ha,
            modified_residues: mod_res,
            ligands: ligs,
            cofactors: cofs,
            metals: mets,
            waters: wats,
            mean_b_factor: if b_cnt > 0 {
                (b_sum / (b_cnt as f64) * 100.0).round() / 100.0
            } else {
                0.0
            },
        });
    }

    // 4. Classify HETATMs in selected chains, decide retention, and extract organic ligands
    let mut het_groups = Vec::new();
    let mut extracted_ligands = Vec::new();
    let mut keep_het_keys: BTreeSet<(char, i32, String)> = BTreeSet::new();

    for ((ch, seq, rn), atoms) in &het_by_res {
        if !chain_set.contains(ch) {
            continue;
        }
        let is_mod = raw.modres_keys.contains(&(*ch, rn.clone(), *seq));
        let cls = classify_het_group(rn, atoms, is_mod);
        let heavy: Vec<&&RawAtom> = atoms.iter().filter(|a| a.element != "H").collect();
        let n_h = heavy.len().max(1) as f64;
        let centroid = [
            heavy.iter().map(|a| a.coords[0]).sum::<f64>() / n_h,
            heavy.iter().map(|a| a.coords[1]).sum::<f64>() / n_h,
            heavy.iter().map(|a| a.coords[2]).sum::<f64>() / n_h,
        ];
        let keep = match cls {
            HetClass::Water => keep_all_waters,
            HetClass::BufferOrSolvent => false,
            HetClass::CatalyticMetal => keep_metals,
            HetClass::Cofactor => keep_cofactors,
            HetClass::ModifiedResidue => true,
            HetClass::OrganicLigand => !strip_ligands,
        };
        if keep {
            keep_het_keys.insert((*ch, *seq, rn.clone()));
        }
        if cls != HetClass::Water {
            het_groups.push(HetGroupInfo {
                chain_id: *ch,
                res_name: rn.clone(),
                res_seq: *seq,
                atom_count: heavy.len(),
                class: cls,
                description: raw.hetnam.get(rn).cloned(),
                kept_in_receptor: keep,
                centroid: [
                    (centroid[0] * 100.0).round() / 100.0,
                    (centroid[1] * 100.0).round() / 100.0,
                    (centroid[2] * 100.0).round() / 100.0,
                ],
            });
        }
        if cls == HetClass::OrganicLigand && heavy.len() >= 5 {
            let mut lig = Molecule3D::new(format!("{}_{}_{}", raw.structure_id, rn, seq));
            let mut ser_map = HashMap::new();
            for a in atoms {
                let mut at = Atom3D::new(lig.atoms.len(), &a.element, a.coords);
                at.name = a.name.clone();
                at.res_name = a.res_name.clone();
                at.chain_id = a.chain_id;
                at.res_seq = a.res_seq;
                at.is_hetatm = true;
                at.b_factor = a.b_factor;
                at.formal_charge = a.formal_charge;
                let id = lig.add_atom(at);
                ser_map.insert(a.serial, id);
            }
            for &(u, v) in &raw.conects {
                if let (Some(&iu), Some(&iv)) = (ser_map.get(&u), ser_map.get(&v)) {
                    let _ = lig.add_bond(iu, iv, BondOrder::Single);
                }
            }
            lig.infer_bonds_by_distance();
            struct_core::perceive_all(&mut lig);
            extracted_ligands.push(lig);
        }
    }

    // Infer pocket center from primary organic ligand if caller did not supply one
    let effective_pocket_center = pocket_center.or_else(|| {
        het_groups
            .iter()
            .filter(|h| h.class == HetClass::OrganicLigand)
            .max_by_key(|h| h.atom_count)
            .map(|h| h.centroid)
    });
    let near_pocket = |p: [f64; 3]| -> bool {
        effective_pocket_center.is_some_and(|c| {
            let d2 = (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2);
            d2 <= pocket_radius * pocket_radius
        })
    };

    // 5. Build clean receptor Molecule3D (excluding raw H so protonation/H-bond opt builds clean H!)
    let mut rec = Molecule3D::new(raw.structure_id.clone());
    let mut serial_to_idx: HashMap<i32, usize> = HashMap::new();
    for &a in &resolved_atoms {
        if !chain_set.contains(&a.chain_id) {
            continue;
        }
        // Strip raw hydrogens so pH-aware protonation and H-bond orientation regenerate consistent H
        if a.element == "H" {
            continue;
        }
        let is_mod = raw
            .modres_keys
            .contains(&(a.chain_id, a.res_name.clone(), a.res_seq))
            || is_modified_protein_res(&a.res_name);
        let is_prot = (!a.is_hetatm && is_standard_protein_res(&a.res_name)) || is_mod;
        let keep_het = keep_het_keys.contains(&(a.chain_id, a.res_seq, a.res_name.clone()));
        if !is_prot && !keep_het {
            continue;
        }
        let mut atom = Atom3D::new(rec.atoms.len(), &a.element, a.coords);
        atom.is_backbone = is_prot && matches!(a.name.as_str(), "N" | "CA" | "C" | "O");
        atom.name = a.name.clone();
        atom.res_name = a.res_name.clone();
        atom.chain_id = a.chain_id;
        atom.res_seq = a.res_seq;
        atom.is_hetatm = a.is_hetatm && !is_mod;
        atom.b_factor = a.b_factor;
        atom.formal_charge = a.formal_charge;
        let idx = rec.add_atom(atom);
        serial_to_idx.insert(a.serial, idx);
    }
    for &(u, v) in &raw.conects {
        if let (Some(&iu), Some(&iv)) = (serial_to_idx.get(&u), serial_to_idx.get(&v)) {
            let _ = rec.add_bond(iu, iv, BondOrder::Single);
        }
    }

    // 6. Geometric missing-atom check & chain-break audit on selected protein residues
    let mut missing_heavy_atoms = Vec::new();
    let mut chain_breaks = Vec::new();
    let mut b_all_sum = 0.0;
    let mut b_all_cnt = 0usize;
    let mut b_pock_sum = 0.0;
    let mut b_pock_cnt = 0usize;
    let mut high_b_pocket = Vec::new();
    let mut prev_c_by_chain: HashMap<char, (i32, String, [f64; 3])> = HashMap::new();
    let mut total_protein_residues = 0usize;
    let mut cys_sg_atoms: Vec<(char, i32, [f64; 3])> = Vec::new();

    for ((ch, seq, rn), atoms) in &prot_by_res {
        if !chain_set.contains(ch) {
            continue;
        }
        total_protein_residues += 1;
        let present: BTreeSet<&str> = atoms.iter().map(|a| a.name.as_str()).collect();
        let heavy: Vec<&&RawAtom> = atoms.iter().filter(|a| a.element != "H").collect();
        let mean_b = if heavy.is_empty() {
            0.0
        } else {
            heavy.iter().map(|a| a.b_factor).sum::<f64>() / (heavy.len() as f64)
        };
        let res_center = if heavy.is_empty() {
            [0.0; 3]
        } else {
            let n = heavy.len() as f64;
            [
                heavy.iter().map(|a| a.coords[0]).sum::<f64>() / n,
                heavy.iter().map(|a| a.coords[1]).sum::<f64>() / n,
                heavy.iter().map(|a| a.coords[2]).sum::<f64>() / n,
            ]
        };
        let in_pock = near_pocket(res_center);
        b_all_sum += mean_b;
        b_all_cnt += 1;
        if in_pock {
            b_pock_sum += mean_b;
            b_pock_cnt += 1;
        }

        if let Some(exp) = expected_heavy_atoms(rn) {
            let missing: Vec<String> = exp
                .iter()
                .filter(|&&ea| !present.contains(ea))
                .map(|s| (*s).to_string())
                .collect();
            if !missing.is_empty() {
                missing_heavy_atoms.push(MissingAtomRecord {
                    chain_id: *ch,
                    res_name: rn.clone(),
                    res_seq: *seq,
                    missing_atoms: missing,
                    near_pocket: in_pock,
                });
            }
        }

        // Peptide backbone continuity (C(i) -> N(i+1) should be ~1.33 Å; > 2.0 Å is a chain break!)
        let cur_n = atoms.iter().find(|a| a.name == "N").map(|a| a.coords);
        let cur_c = atoms.iter().find(|a| a.name == "C").map(|a| a.coords);
        if let (Some((prev_seq, prev_rn, prev_c)), Some(n_pos)) = (prev_c_by_chain.get(ch), cur_n) {
            let d = dist3(*prev_c, n_pos);
            if d > 2.05 || (*seq - *prev_seq) > 1 {
                chain_breaks.push(ChainBreak {
                    chain_id: *ch,
                    res_before: format!("{ch}:{prev_rn}{prev_seq}"),
                    res_after: format!("{ch}:{rn}{seq}"),
                    distance_a: (d * 100.0).round() / 100.0,
                    near_pocket: in_pock || near_pocket(*prev_c),
                });
            }
        }
        if let Some(c_pos) = cur_c {
            prev_c_by_chain.insert(*ch, (*seq, rn.clone(), c_pos));
        }

        if matches!(rn.as_str(), "CYS" | "CYX")
            && let Some(sg) = atoms.iter().find(|a| a.name == "SG")
        {
            cys_sg_atoms.push((*ch, *seq, sg.coords));
        }
    }

    // Disulfide bridges (geometric SG-SG < 2.35 Å)
    let mut disulfide_bonds: Vec<String> = Vec::new();
    for i in 0..cys_sg_atoms.len() {
        for j in (i + 1)..cys_sg_atoms.len() {
            let d = dist3(cys_sg_atoms[i].2, cys_sg_atoms[j].2);
            if d < 2.35 {
                disulfide_bonds.push(format!(
                    "{}:CYS{}-{}:CYS{} ({:.2} A)",
                    cys_sg_atoms[i].0, cys_sg_atoms[i].1, cys_sg_atoms[j].0, cys_sg_atoms[j].1, d
                ));
            }
        }
    }

    let mean_b_or_plddt = if b_all_cnt > 0 {
        (b_all_sum / (b_all_cnt as f64) * 100.0).round() / 100.0
    } else {
        0.0
    };
    let is_alphafold = raw.is_alphafold_header
        || (raw.resolution_a.is_none()
            && raw.experimental_method.is_empty()
            && mean_b_or_plddt > 55.0
            && mean_b_or_plddt <= 100.0);

    // Flag high B-factor (> 50 A^2) or low AlphaFold pLDDT (< 70) residues inside the pocket
    for ((ch, seq, rn), atoms) in &prot_by_res {
        if !chain_set.contains(ch) {
            continue;
        }
        let heavy: Vec<&&RawAtom> = atoms.iter().filter(|a| a.element != "H").collect();
        if heavy.is_empty() {
            continue;
        }
        let n = heavy.len() as f64;
        let center = [
            heavy.iter().map(|a| a.coords[0]).sum::<f64>() / n,
            heavy.iter().map(|a| a.coords[1]).sum::<f64>() / n,
            heavy.iter().map(|a| a.coords[2]).sum::<f64>() / n,
        ];
        if !near_pocket(center) {
            continue;
        }
        let mb = heavy.iter().map(|a| a.b_factor).sum::<f64>() / n;
        if is_alphafold && mb < 70.0 {
            high_b_pocket.push(format!("{ch}:{rn}{seq} (pLDDT {mb:.1})"));
        } else if !is_alphafold && mb > 50.0 {
            high_b_pocket.push(format!("{ch}:{rn}{seq} (B {mb:.1})"));
        }
    }

    // 7. Severe steric clash scan on non-bonded protein/cofactor heavy atoms (< 1.90 Å; excludes bulk waters)
    let mut steric_clashes = Vec::new();
    let rec_len = rec.atoms.len();
    for i in 0..rec_len {
        let ai = &rec.atoms[i];
        if !ai.is_heavy() || is_water_res(&ai.res_name) {
            continue;
        }
        for aj in &rec.atoms[(i + 1)..rec_len] {
            if !aj.is_heavy() || is_water_res(&aj.res_name) {
                continue;
            }
            if ai.chain_id == aj.chain_id && (ai.res_seq - aj.res_seq).abs() <= 1 {
                continue;
            }
            // Skip disulfide pairs and metal coordination bonds
            if (ai.element == "S" && aj.element == "S")
                || ai.ad4_type.is_metal()
                || aj.ad4_type.is_metal()
            {
                continue;
            }
            let dx = ai.coords[0] - aj.coords[0];
            if dx.abs() > 1.90 {
                continue;
            }
            let dy = ai.coords[1] - aj.coords[1];
            if dy.abs() > 1.90 {
                continue;
            }
            let dz = ai.coords[2] - aj.coords[2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 < 1.90 * 1.90 && steric_clashes.len() < 25 {
                steric_clashes.push(StericClash {
                    atom_a: format!("{}:{}{}:{}", ai.chain_id, ai.res_name, ai.res_seq, ai.name),
                    atom_b: format!("{}:{}{}:{}", aj.chain_id, aj.res_name, aj.res_seq, aj.name),
                    distance_a: (d2.sqrt() * 100.0).round() / 100.0,
                });
            }
        }
    }

    // 8. Synthesize actionable issues & Structural Quality Grade (A/B/C/D)
    let mut issues = Vec::new();
    let mut penalty = 0i32;

    if is_alphafold {
        issues.push(AuditIssue {
            severity: AuditSeverity::Info,
            category: "StructureSource".to_string(),
            message: format!(
                "Predicted model (AlphaFold/ESMFold) detected: mean pLDDT = {mean_b_or_plddt:.1}"
            ),
        });
        if mean_b_or_plddt < 70.0 {
            penalty += 25;
            issues.push(AuditIssue {
                severity: AuditSeverity::Critical,
                category: "AlphaFoldConfidence".to_string(),
                message: format!("Low global pLDDT ({mean_b_or_plddt:.1} < 70)"),
            });
        }
    } else if let Some(res) = raw.resolution_a {
        if res > 3.0 {
            penalty += 25;
            issues.push(AuditIssue {
                severity: AuditSeverity::Critical,
                category: "Resolution".to_string(),
                message: format!("Low crystallographic resolution ({res:.2} Å > 3.0 Å)"),
            });
        } else if res > 2.5 {
            penalty += 10;
            issues.push(AuditIssue {
                severity: AuditSeverity::Warning,
                category: "Resolution".to_string(),
                message: format!(
                    "Moderate resolution ({res:.2} Å); sidechain rotamers may be uncertain"
                ),
            });
        }
    }
    if let Some(rf) = raw.r_free
        && rf > 0.28
    {
        penalty += 10;
        issues.push(AuditIssue {
            severity: AuditSeverity::Warning,
            category: "R-Free".to_string(),
            message: format!("Elevated R-free ({rf:.3} > 0.28)"),
        });
    }

    let missing_in_selected: Vec<MissingResidueRecord> = raw
        .missing_residues
        .iter()
        .filter(|m| chain_set.contains(&m.chain_id))
        .cloned()
        .collect();
    if !missing_in_selected.is_empty() {
        let pock_breaks = chain_breaks.iter().filter(|b| b.near_pocket).count();
        let sev = if pock_breaks > 0 {
            penalty += 20;
            AuditSeverity::Critical
        } else {
            penalty += 5;
            AuditSeverity::Warning
        };
        issues.push(AuditIssue {
            severity: sev,
            category: "MissingLoops".to_string(),
            message: format!(
                "{} missing residue(s) in REMARK 465 ({} chain break(s), {} near active site)",
                missing_in_selected.len(),
                chain_breaks.len(),
                pock_breaks
            ),
        });
    }

    let pock_missing_atoms = missing_heavy_atoms.iter().filter(|m| m.near_pocket).count();
    if !missing_heavy_atoms.is_empty() {
        let sev = if pock_missing_atoms > 0 {
            penalty += 15;
            AuditSeverity::Critical
        } else {
            penalty += 5;
            AuditSeverity::Warning
        };
        issues.push(AuditIssue {
            severity: sev,
            category: "MissingAtoms".to_string(),
            message: format!(
                "{} residue(s) have incomplete sidechain/backbone atoms ({} within pocket radius)",
                missing_heavy_atoms.len(),
                pock_missing_atoms
            ),
        });
    }

    if !altlocs_resolved.is_empty() {
        issues.push(AuditIssue {
            severity: AuditSeverity::Info,
            category: "Altlocs".to_string(),
            message: format!(
                "Resolved {} residue(s) with alternate conformations by highest occupancy",
                altlocs_resolved.len()
            ),
        });
    }
    if !high_b_pocket.is_empty() {
        penalty += 8;
        issues.push(AuditIssue {
            severity: AuditSeverity::Warning,
            category: if is_alphafold {
                "PocketPLDDT".to_string()
            } else {
                "PocketBFactor".to_string()
            },
            message: format!(
                "{} pocket residue(s) show high flexibility / low confidence: {}",
                high_b_pocket.len(),
                high_b_pocket
                    .iter()
                    .take(5)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        });
    }
    if !steric_clashes.is_empty() {
        penalty += 12;
        issues.push(AuditIssue {
            severity: AuditSeverity::Warning,
            category: "StericClashes".to_string(),
            message: format!(
                "Detected {} severe heavy-atom clash(es) (< 1.90 Å)",
                steric_clashes.len()
            ),
        });
    }

    let quality_grade = if penalty <= 10 {
        "A (Docking-Ready)".to_string()
    } else if penalty <= 25 {
        "B (Good — Review Warnings)".to_string()
    } else if penalty <= 45 {
        "C (Caution — Pocket Loops/Sidechains Incomplete)".to_string()
    } else {
        "D (Poor — Consider Alternative Crystal / Ensemble)".to_string()
    };

    let pocket_mean_b = if b_pock_cnt > 0 {
        Some((b_pock_sum / (b_pock_cnt as f64) * 100.0).round() / 100.0)
    } else {
        None
    };

    let audit = StructuralAudit {
        structure_id: raw.structure_id.clone(),
        title: raw.title.clone(),
        experimental_method: if raw.experimental_method.is_empty() {
            if is_alphafold {
                "COMPUTATIONAL MODEL (AlphaFold/ESMFold)".to_string()
            } else {
                "UNKNOWN".to_string()
            }
        } else {
            raw.experimental_method.clone()
        },
        resolution_a: raw.resolution_a,
        r_work: raw.r_work,
        r_free: raw.r_free,
        is_alphafold,
        quality_grade,
        chains: chain_summaries,
        selected_chains,
        total_protein_residues,
        mean_b_factor_or_plddt: mean_b_or_plddt,
        pocket_mean_b_factor_or_plddt: pocket_mean_b,
        high_b_or_low_plddt_pocket_residues: high_b_pocket,
        missing_residues_header: missing_in_selected,
        missing_heavy_atoms,
        chain_breaks,
        altlocs_resolved,
        disulfide_bonds,
        het_groups,
        steric_clashes,
        issues,
    };

    (rec, extracted_ligands, audit)
}

#[inline]
pub(crate) fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}
