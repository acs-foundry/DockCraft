use std::collections::HashMap;
use std::fmt::Write;

use crate::chemistry::{perceive_all, prepare_receptor};
use crate::error::StructError;
use crate::types::{
    Ad4Type, Atom3D, BondOrder, Molecule3D, element_atomic_number, normalize_element,
};

/// Parse a PDB file (`ATOM`, `HETATM`, `CONECT`, first `MODEL`) cleanly and run receptor/ligand preparation.
pub fn parse_pdb(input: &str, strip_waters: bool) -> Result<Molecule3D, StructError> {
    if input.trim().is_empty() {
        return Err(StructError::EmptyInput);
    }
    let mut mol = Molecule3D::new("PDB");
    let mut serial_to_idx: HashMap<i32, usize> = HashMap::new();
    let mut first_model_done = false;

    for (ln, raw) in input.lines().enumerate() {
        if raw.starts_with("ENDMDL") {
            first_model_done = true;
            continue;
        }
        if first_model_done && (raw.starts_with("ATOM") || raw.starts_with("HETATM")) {
            break;
        }
        if let Some(rest) = raw.strip_prefix("HEADER") {
            let hdr = rest.trim();
            if !hdr.is_empty() {
                mol.name = hdr.split_whitespace().last().unwrap_or("PDB").to_string();
            }
            continue;
        }
        if raw.starts_with("ATOM  ") || raw.starts_with("HETATM") {
            let is_het = raw.starts_with("HETATM");
            let alt_loc = raw.get(16..17).unwrap_or(" ");
            if !matches!(alt_loc, " " | "A" | "1") {
                continue;
            }
            let serial: i32 = raw
                .get(6..11)
                .unwrap_or("")
                .trim()
                .parse()
                .unwrap_or((mol.atoms.len() + 1) as i32);
            let name = raw.get(12..16).unwrap_or("C").trim().to_string();
            let res_name = raw.get(17..20).unwrap_or("UNK").trim().to_string();
            let chain_id = raw
                .get(21..22)
                .unwrap_or("A")
                .trim()
                .chars()
                .next()
                .unwrap_or('A');
            let res_seq: i32 = raw.get(22..26).unwrap_or("1").trim().parse().unwrap_or(1);
            let x: f64 = raw.get(30..38).unwrap_or("").trim().parse().map_err(|_| {
                StructError::InvalidPdb {
                    line: ln + 1,
                    msg: "invalid X coordinate".to_string(),
                }
            })?;
            let y: f64 = raw.get(38..46).unwrap_or("").trim().parse().map_err(|_| {
                StructError::InvalidPdb {
                    line: ln + 1,
                    msg: "invalid Y coordinate".to_string(),
                }
            })?;
            let z: f64 = raw.get(46..54).unwrap_or("").trim().parse().map_err(|_| {
                StructError::InvalidPdb {
                    line: ln + 1,
                    msg: "invalid Z coordinate".to_string(),
                }
            })?;
            if !x.is_finite() || !y.is_finite() || !z.is_finite() {
                return Err(StructError::InvalidPdb {
                    line: ln + 1,
                    msg: "non-finite coordinates".to_string(),
                });
            }
            let b_factor: f64 = raw
                .get(60..66)
                .unwrap_or("20.0")
                .trim()
                .parse()
                .unwrap_or(20.0);
            let raw_el = raw.get(76..78).unwrap_or("").trim();
            let el = if raw_el.is_empty() {
                infer_element_from_pdb_name(&name)
            } else {
                normalize_element(raw_el)
            };

            let mut atom = Atom3D::new(mol.atoms.len(), &el, [x, y, z]);
            atom.is_backbone = !is_het && matches!(name.as_str(), "N" | "CA" | "C" | "O");
            atom.name = name;
            atom.res_name = res_name;
            atom.chain_id = chain_id;
            atom.res_seq = res_seq;
            atom.is_hetatm = is_het;
            atom.b_factor = b_factor;
            if let Some(chg_str) = raw.get(78..80).map(str::trim)
                && chg_str.len() == 2
            {
                let bs = chg_str.as_bytes();
                if bs[0].is_ascii_digit() && matches!(bs[1], b'+' | b'-') {
                    let mag = (bs[0] - b'0') as i8;
                    atom.formal_charge = if bs[1] == b'+' { mag } else { -mag };
                }
            }
            let idx = mol.add_atom(atom);
            serial_to_idx.insert(serial, idx);
        } else if let Some(rest) = raw.strip_prefix("CONECT") {
            let mut nums = Vec::new();
            let mut p = 0;
            while p + 5 <= rest.len() {
                if let Ok(v) = rest[p..p + 5].trim().parse::<i32>() {
                    nums.push(v);
                }
                p += 5;
            }
            if let Some(&src) = nums.first()
                && let Some(&u) = serial_to_idx.get(&src)
            {
                for &dst in &nums[1..] {
                    if let Some(&v) = serial_to_idx.get(&dst) {
                        let _ = mol.add_bond(u, v, BondOrder::Single);
                    }
                }
            }
        }
    }

    if mol.atoms.is_empty() {
        return Err(StructError::InvalidPdb {
            line: 1,
            msg: "no ATOM/HETATM records found".to_string(),
        });
    }
    prepare_receptor(&mut mol, strip_waters);
    Ok(mol)
}

fn infer_element_from_pdb_name(name: &str) -> String {
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
    } else {
        t.chars()
            .next()
            .filter(|c| c.is_ascii_alphabetic())
            .map_or_else(|| "C".to_string(), |c| c.to_ascii_uppercase().to_string())
    }
}

pub fn write_pdb(mol: &Molecule3D) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "HEADER    DOCKCRAFT STRUCTURE {}", mol.name);
    for (idx, a) in mol.atoms.iter().enumerate() {
        let rec = if a.is_hetatm { "HETATM" } else { "ATOM  " };
        let aname = if a.name.len() >= 4 {
            a.name[..4].to_string()
        } else {
            format!(" {:<3}", a.name)
        };
        let rname = format!("{:<3}", &a.res_name[..a.res_name.len().min(3)]);
        let _ = writeln!(
            out,
            "{rec}{:>5} {aname:<4} {rname} {}{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00{:>6.2}          {:>2}",
            idx + 1,
            a.chain_id,
            a.res_seq,
            a.coords[0],
            a.coords[1],
            a.coords[2],
            a.b_factor.clamp(0.0, 999.99),
            a.element
        );
    }
    for (idx, _) in mol.atoms.iter().enumerate() {
        if !mol.adj[idx].is_empty() && mol.atoms[idx].is_hetatm {
            let _ = write!(out, "CONECT{:>5}", idx + 1);
            for &(nb, _) in mol.adj[idx].iter().take(4) {
                let _ = write!(out, "{:>5}", nb + 1);
            }
            let _ = writeln!(out);
        }
    }
    let _ = writeln!(out, "END");
    out
}

/// Parse an mmCIF `_atom_site.` table into a `Molecule3D`.
pub fn parse_mmcif(input: &str, strip_waters: bool) -> Result<Molecule3D, StructError> {
    if input.trim().is_empty() {
        return Err(StructError::EmptyInput);
    }
    let mut mol = Molecule3D::new("mmCIF");
    let mut in_atom_site = false;
    let mut cols: Vec<String> = Vec::new();
    let mut col_idx: HashMap<String, usize> = HashMap::new();

    for (ln, raw) in input.lines().enumerate() {
        let t = raw.trim();
        if t.is_empty() || t.starts_with('#') {
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
        if t.starts_with('_') || t.starts_with("data_") {
            if in_atom_site {
                break;
            }
            if let Some(dn) = t.strip_prefix("data_") {
                mol.name = dn.trim().to_string();
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
        let get = |k: &str| -> Option<&str> {
            col_idx
                .get(k)
                .and_then(|&i| toks.get(i))
                .map(String::as_str)
        };
        let x: f64 = get("_atom_site.Cartn_x")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| StructError::InvalidCif {
                line: ln + 1,
                msg: "missing _atom_site.Cartn_x".to_string(),
            })?;
        let y: f64 = get("_atom_site.Cartn_y")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| StructError::InvalidCif {
                line: ln + 1,
                msg: "missing _atom_site.Cartn_y".to_string(),
            })?;
        let z: f64 = get("_atom_site.Cartn_z")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| StructError::InvalidCif {
                line: ln + 1,
                msg: "missing _atom_site.Cartn_z".to_string(),
            })?;
        let name = get("_atom_site.label_atom_id")
            .or_else(|| get("_atom_site.auth_atom_id"))
            .unwrap_or("C")
            .to_string();
        let el = get("_atom_site.type_symbol")
            .map(normalize_element)
            .unwrap_or_else(|| infer_element_from_pdb_name(&name));
        let res_name = get("_atom_site.label_comp_id")
            .or_else(|| get("_atom_site.auth_comp_id"))
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
        let is_het = get("_atom_site.group_PDB").is_some_and(|g| g.eq_ignore_ascii_case("HETATM"));
        let b_fac: f64 = get("_atom_site.B_iso_or_equiv")
            .and_then(|s| s.parse().ok())
            .unwrap_or(20.0);

        let mut atom = Atom3D::new(mol.atoms.len(), &el, [x, y, z]);
        atom.is_backbone = !is_het && matches!(name.as_str(), "N" | "CA" | "C" | "O");
        atom.name = name;
        atom.res_name = res_name;
        atom.chain_id = chain_id;
        atom.res_seq = res_seq;
        atom.is_hetatm = is_het;
        atom.b_factor = b_fac;
        mol.add_atom(atom);
    }

    if mol.atoms.is_empty() {
        return Err(StructError::InvalidCif {
            line: 1,
            msg: "no _atom_site records parsed".to_string(),
        });
    }
    prepare_receptor(&mut mol, strip_waters);
    Ok(mol)
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

pub fn write_mmcif(mol: &Molecule3D) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "data_{}", mol.name.replace(' ', "_"));
    let _ = writeln!(out, "#");
    let _ = writeln!(out, "loop_");
    for field in [
        "group_PDB",
        "id",
        "type_symbol",
        "label_atom_id",
        "label_comp_id",
        "auth_asym_id",
        "auth_seq_id",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
        "B_iso_or_equiv",
    ] {
        let _ = writeln!(out, "_atom_site.{field}");
    }
    for (idx, a) in mol.atoms.iter().enumerate() {
        let grp = if a.is_hetatm { "HETATM" } else { "ATOM" };
        let _ = writeln!(
            out,
            "{grp} {} {} {} {} {} {} {:.3} {:.3} {:.3} {:.2}",
            idx + 1,
            a.element,
            a.name,
            a.res_name,
            a.chain_id,
            a.res_seq,
            a.coords[0],
            a.coords[1],
            a.coords[2],
            a.b_factor
        );
    }
    out
}

/// Parse Tripos MOL2 file (`@<TRIPOS>MOLECULE`, `ATOM`, `BOND`).
pub fn parse_mol2(input: &str) -> Result<Molecule3D, StructError> {
    if input.trim().is_empty() {
        return Err(StructError::EmptyInput);
    }
    #[derive(Copy, Clone, PartialEq, Eq)]
    enum Sec {
        None,
        Mol,
        Atom,
        Bond,
    }
    let mut sec = Sec::None;
    let mut mol_line = 0usize;
    let mut mol = Molecule3D::new("MOL2");
    let mut id_map: HashMap<usize, usize> = HashMap::new();

    for (ln, raw) in input.lines().enumerate() {
        let t = raw.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if t.starts_with("@<TRIPOS>") {
            sec = match t {
                "@<TRIPOS>MOLECULE" => {
                    mol_line = 0;
                    Sec::Mol
                }
                "@<TRIPOS>ATOM" => Sec::Atom,
                "@<TRIPOS>BOND" => Sec::Bond,
                _ => Sec::None,
            };
            continue;
        }
        match sec {
            Sec::Mol => {
                if mol_line == 0 {
                    mol.name = t.to_string();
                }
                mol_line += 1;
            }
            Sec::Atom => {
                let toks: Vec<&str> = t.split_whitespace().collect();
                if toks.len() < 6 {
                    return Err(StructError::InvalidMol2 {
                        line: ln + 1,
                        msg: "fewer than 6 ATOM fields".to_string(),
                    });
                }
                let ext_id: usize = toks[0].parse().map_err(|_| StructError::InvalidMol2 {
                    line: ln + 1,
                    msg: "invalid atom id".to_string(),
                })?;
                let x: f64 = toks[2].parse().map_err(|_| StructError::InvalidMol2 {
                    line: ln + 1,
                    msg: "invalid X".to_string(),
                })?;
                let y: f64 = toks[3].parse().map_err(|_| StructError::InvalidMol2 {
                    line: ln + 1,
                    msg: "invalid Y".to_string(),
                })?;
                let z: f64 = toks[4].parse().map_err(|_| StructError::InvalidMol2 {
                    line: ln + 1,
                    msg: "invalid Z".to_string(),
                })?;
                let sybyl = toks[5];
                let el = normalize_element(sybyl.split('.').next().unwrap_or(sybyl));
                let mut atom = Atom3D::new(mol.atoms.len(), &el, [x, y, z]);
                atom.name = toks[1].to_string();
                atom.aromatic = sybyl.ends_with(".ar");
                if let Some(q_s) = toks.get(8)
                    && let Ok(q) = q_s.parse::<f64>()
                    && q.is_finite()
                {
                    atom.charge = q;
                }
                let idx = mol.add_atom(atom);
                id_map.insert(ext_id, idx);
            }
            Sec::Bond => {
                let toks: Vec<&str> = t.split_whitespace().collect();
                if toks.len() >= 4 {
                    let u_ext: usize = toks[1].parse().unwrap_or(0);
                    let v_ext: usize = toks[2].parse().unwrap_or(0);
                    let bo = match toks[3] {
                        "2" => BondOrder::Double,
                        "3" => BondOrder::Triple,
                        "ar" | "am" => BondOrder::Aromatic,
                        _ => BondOrder::Single,
                    };
                    if let (Some(&u), Some(&v)) = (id_map.get(&u_ext), id_map.get(&v_ext)) {
                        let _ = mol.add_bond(u, v, bo);
                    }
                }
            }
            Sec::None => {}
        }
    }

    if mol.atoms.is_empty() {
        return Err(StructError::InvalidMol2 {
            line: 1,
            msg: "no atoms parsed".to_string(),
        });
    }
    perceive_all(&mut mol);
    Ok(mol)
}

pub fn write_mol2(mol: &Molecule3D) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "@<TRIPOS>MOLECULE");
    let _ = writeln!(out, "{}", mol.name);
    let _ = writeln!(out, "{} {} 0 0 0", mol.atoms.len(), mol.bonds.len());
    let _ = writeln!(out, "SMALL");
    let _ = writeln!(out, "GASTEIGER");
    let _ = writeln!(out, "@<TRIPOS>ATOM");
    for (i, a) in mol.atoms.iter().enumerate() {
        let sybyl = if a.aromatic && a.element == "C" {
            "C.ar".to_string()
        } else if a.aromatic && a.element == "N" {
            "N.ar".to_string()
        } else {
            a.element.clone()
        };
        let _ = writeln!(
            out,
            "{:>7} {:<6} {:>10.4} {:>10.4} {:>10.4} {:<6} {:>3} {:<6} {:>8.4}",
            i + 1,
            a.name,
            a.coords[0],
            a.coords[1],
            a.coords[2],
            sybyl,
            a.res_seq,
            a.res_name,
            a.charge
        );
    }
    let _ = writeln!(out, "@<TRIPOS>BOND");
    for (bi, b) in mol.bonds.iter().enumerate() {
        let bt = match b.order {
            BondOrder::Single => "1",
            BondOrder::Double => "2",
            BondOrder::Triple => "3",
            BondOrder::Aromatic => "ar",
        };
        let _ = writeln!(out, "{:>6} {:>5} {:>5} {}", bi + 1, b.u + 1, b.v + 1, bt);
    }
    out
}

/// Lazy multi-record 3D SDF V2000/V3000 parser bounded to `max_records`.
pub fn parse_sdf_3d_with_limit(
    input: &str,
    max_records: usize,
) -> Vec<Result<Molecule3D, StructError>> {
    if input.trim().is_empty() {
        return vec![Err(StructError::EmptyInput)];
    }
    let norm = input.replace("\r\n", "\n");
    norm.split("$$$$")
        .filter_map(|rec| {
            let block = rec.strip_prefix('\n').unwrap_or(rec);
            if block.trim().is_empty() {
                None
            } else {
                Some(parse_sdf_mol_block(block))
            }
        })
        .take(max_records)
        .collect()
}

pub fn parse_sdf_mol_block(block: &str) -> Result<Molecule3D, StructError> {
    let lines: Vec<&str> = block.lines().collect();
    if lines.len() < 4 {
        return Err(StructError::InvalidSdf {
            line: 1,
            msg: "MOL block shorter than 4 lines".to_string(),
        });
    }
    let title = lines[0].trim();
    let counts = lines[3];
    if counts.contains("V3000") {
        return parse_sdf_v3000(&lines, title);
    }
    let ac: usize =
        counts
            .get(0..3)
            .unwrap_or("")
            .trim()
            .parse()
            .map_err(|_| StructError::InvalidSdf {
                line: 4,
                msg: "invalid V2000 atom count".to_string(),
            })?;
    let bc: usize =
        counts
            .get(3..6)
            .unwrap_or("")
            .trim()
            .parse()
            .map_err(|_| StructError::InvalidSdf {
                line: 4,
                msg: "invalid V2000 bond count".to_string(),
            })?;
    if lines.len() < 4 + ac + bc {
        return Err(StructError::InvalidSdf {
            line: 4,
            msg: "truncated V2000 block".to_string(),
        });
    }

    let mut mol = Molecule3D::new(if title.is_empty() { "SDF" } else { title });
    for (idx, line) in lines[4..4 + ac].iter().enumerate() {
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() < 4 {
            return Err(StructError::InvalidSdf {
                line: 5 + idx,
                msg: "short atom line".to_string(),
            });
        }
        let x: f64 = toks[0].parse().map_err(|_| StructError::InvalidSdf {
            line: 5 + idx,
            msg: "bad X".to_string(),
        })?;
        let y: f64 = toks[1].parse().map_err(|_| StructError::InvalidSdf {
            line: 5 + idx,
            msg: "bad Y".to_string(),
        })?;
        let z: f64 = toks[2].parse().map_err(|_| StructError::InvalidSdf {
            line: 5 + idx,
            msg: "bad Z".to_string(),
        })?;
        let mut atom = Atom3D::new(idx, toks[3], [x, y, z]);
        if let Some(cs) = toks.get(5)
            && let Ok(code) = cs.parse::<u8>()
        {
            atom.formal_charge = match code {
                1 => 3,
                2 => 2,
                3 => 1,
                5 => -1,
                6 => -2,
                7 => -3,
                _ => 0,
            };
        }
        mol.add_atom(atom);
    }

    for (idx, line) in lines[4 + ac..4 + ac + bc].iter().enumerate() {
        let u_raw = line
            .get(0..3)
            .and_then(|s| s.trim().parse::<usize>().ok())
            .or_else(|| line.split_whitespace().next().and_then(|s| s.parse().ok()))
            .ok_or_else(|| StructError::InvalidSdf {
                line: 5 + ac + idx,
                msg: "bad bond u".to_string(),
            })?;
        let v_raw = line
            .get(3..6)
            .and_then(|s| s.trim().parse::<usize>().ok())
            .or_else(|| line.split_whitespace().nth(1).and_then(|s| s.parse().ok()))
            .ok_or_else(|| StructError::InvalidSdf {
                line: 5 + ac + idx,
                msg: "bad bond v".to_string(),
            })?;
        let t_raw: u8 = line
            .get(6..9)
            .and_then(|s| s.trim().parse().ok())
            .or_else(|| line.split_whitespace().nth(2).and_then(|s| s.parse().ok()))
            .unwrap_or(1);
        if u_raw == 0 || v_raw == 0 || u_raw > ac || v_raw > ac {
            return Err(StructError::InvalidSdf {
                line: 5 + ac + idx,
                msg: "bond atom index out of bounds".to_string(),
            });
        }
        let bo = match t_raw {
            2 => BondOrder::Double,
            3 => BondOrder::Triple,
            4 => BondOrder::Aromatic,
            _ => BondOrder::Single,
        };
        let _ = mol.add_bond(u_raw - 1, v_raw - 1, bo);
    }

    parse_sdf_props_and_tags(&mut mol, &lines, 4 + ac + bc);
    perceive_all(&mut mol);
    Ok(mol)
}

fn parse_sdf_v3000(lines: &[&str], title: &str) -> Result<Molecule3D, StructError> {
    let mut mol = Molecule3D::new(if title.is_empty() { "SDF_V3000" } else { title });
    let mut id_map: HashMap<usize, usize> = HashMap::new();
    let mut in_atom = false;
    let mut in_bond = false;
    let mut end_idx = lines.len();

    for (ln, raw) in lines.iter().enumerate().skip(4) {
        let t = raw.trim();
        if t == "M  END" {
            end_idx = ln;
            break;
        }
        let Some(rest) = t.strip_prefix("M  V30 ") else {
            continue;
        };
        let rest = rest.trim();
        match rest {
            "BEGIN ATOM" => {
                in_atom = true;
                in_bond = false;
                continue;
            }
            "END ATOM" => {
                in_atom = false;
                continue;
            }
            "BEGIN BOND" => {
                in_bond = true;
                in_atom = false;
                continue;
            }
            "END BOND" => {
                in_bond = false;
                continue;
            }
            _ => {}
        }
        if in_atom {
            let toks: Vec<&str> = rest.split_whitespace().collect();
            if toks.len() < 5 {
                return Err(StructError::InvalidSdf {
                    line: ln + 1,
                    msg: "malformed V3000 ATOM".to_string(),
                });
            }
            let ext_id: usize = toks[0].parse().unwrap_or(0);
            let x: f64 = toks[2].parse().unwrap_or(0.0);
            let y: f64 = toks[3].parse().unwrap_or(0.0);
            let z: f64 = toks[4].parse().unwrap_or(0.0);
            let mut atom = Atom3D::new(mol.atoms.len(), toks[1], [x, y, z]);
            for kv in &toks[5..] {
                if let Some(c) = kv.strip_prefix("CHG=").and_then(|v| v.parse::<i32>().ok()) {
                    atom.formal_charge = c.clamp(-15, 15) as i8;
                }
            }
            let nid = mol.add_atom(atom);
            id_map.insert(ext_id, nid);
        } else if in_bond {
            let toks: Vec<&str> = rest.split_whitespace().collect();
            if toks.len() >= 4 {
                let bo = match toks[1] {
                    "2" => BondOrder::Double,
                    "3" => BondOrder::Triple,
                    "4" => BondOrder::Aromatic,
                    _ => BondOrder::Single,
                };
                if let (Ok(u_e), Ok(v_e)) = (toks[2].parse::<usize>(), toks[3].parse::<usize>())
                    && let (Some(&u), Some(&v)) = (id_map.get(&u_e), id_map.get(&v_e))
                {
                    let _ = mol.add_bond(u, v, bo);
                }
            }
        }
    }

    parse_sdf_props_and_tags(&mut mol, lines, end_idx);
    perceive_all(&mut mol);
    Ok(mol)
}

fn parse_sdf_props_and_tags(mol: &mut Molecule3D, lines: &[&str], start: usize) {
    let mut i = start;
    while i < lines.len() {
        let line = lines[i];
        if let Some(rest) = line.strip_prefix("M  CHG") {
            let nums: Vec<i32> = rest
                .split_whitespace()
                .filter_map(|t| t.parse::<i32>().ok())
                .collect();
            if let Some(&cnt) = nums.first()
                && cnt > 0
            {
                for &[idx_raw, chg_val] in nums[1..].as_chunks::<2>().0.iter().take(cnt as usize) {
                    if let Some(one_based) = idx_raw.checked_sub(1)
                        && let Ok(a_idx) = usize::try_from(one_based)
                        && let Some(atom) = mol.atoms.get_mut(a_idx)
                    {
                        atom.formal_charge = chg_val.clamp(-15, 15) as i8;
                    }
                }
            }
        } else if line.trim_start().starts_with("> ")
            && let Some(s0) = line.find('<')
            && let Some(s1) = line[s0 + 1..].find('>')
        {
            let tag = line[s0 + 1..s0 + 1 + s1].trim();
            if let Some(val) = lines.get(i + 1).map(|v| v.trim()).filter(|v| !v.is_empty()) {
                mol.properties.insert(tag.to_string(), val.to_string());
                if tag.eq_ignore_ascii_case("SMILES") && mol.source_smiles.is_none() {
                    mol.source_smiles = Some(val.to_string());
                }
                i += 1;
            }
        }
        i += 1;
    }
}

pub fn write_sdf_3d(mols: &[&Molecule3D]) -> String {
    let mut out = String::new();
    for mol in mols {
        let _ = writeln!(out, "{}", mol.name);
        let _ = writeln!(out, "  DockCraft-3D-V2000");
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "{:>3}{:>3}  0  0  0  0  0  0  0  0999 V2000",
            mol.atoms.len(),
            mol.bonds.len()
        );
        for a in &mol.atoms {
            let _ = writeln!(
                out,
                "{:>10.4}{:>10.4}{:>10.4} {:<3} 0  0  0  0  0  0  0  0  0  0  0  0",
                a.coords[0], a.coords[1], a.coords[2], a.element
            );
        }
        for b in &mol.bonds {
            let t = match b.order {
                BondOrder::Single => 1,
                BondOrder::Double => 2,
                BondOrder::Triple => 3,
                BondOrder::Aromatic => 4,
            };
            let _ = writeln!(out, "{:>3}{:>3}{:>3}  0  0  0  0", b.u + 1, b.v + 1, t);
        }
        let charged: Vec<(usize, i8)> = mol
            .atoms
            .iter()
            .filter(|a| a.formal_charge != 0)
            .map(|a| (a.id + 1, a.formal_charge))
            .collect();
        for chunk in charged.chunks(8) {
            let _ = write!(out, "M  CHG{:>3}", chunk.len());
            for (aid, chg) in chunk {
                let _ = write!(out, "{aid:>4}{chg:>4}");
            }
            let _ = writeln!(out);
        }
        let _ = writeln!(out, "M  END");
        for (k, v) in &mol.properties {
            let _ = writeln!(out, "> <{k}>\n{v}\n");
        }
        let _ = writeln!(out, "$$$$");
    }
    out
}

/// Parse AutoDock PDBQT (receptor or ligand with `ROOT`/`BRANCH`/`TORSDOF`).
pub fn parse_pdbqt(input: &str) -> Result<Molecule3D, StructError> {
    if input.trim().is_empty() {
        return Err(StructError::EmptyInput);
    }
    let mut mol = Molecule3D::new("PDBQT");
    for (ln, raw) in input.lines().enumerate() {
        if raw.starts_with("ENDMDL") {
            break;
        }
        if raw.starts_with("ATOM  ") || raw.starts_with("HETATM") {
            let is_het = raw.starts_with("HETATM");
            let name = raw.get(12..16).unwrap_or("C").trim().to_string();
            let res_name = raw.get(17..20).unwrap_or("LIG").trim().to_string();
            let chain_id = raw
                .get(21..22)
                .unwrap_or("A")
                .trim()
                .chars()
                .next()
                .unwrap_or('A');
            let res_seq: i32 = raw.get(22..26).unwrap_or("1").trim().parse().unwrap_or(1);
            let x: f64 = raw.get(30..38).unwrap_or("").trim().parse().map_err(|_| {
                StructError::InvalidPdbqt {
                    line: ln + 1,
                    msg: "invalid X".to_string(),
                }
            })?;
            let y: f64 = raw.get(38..46).unwrap_or("").trim().parse().map_err(|_| {
                StructError::InvalidPdbqt {
                    line: ln + 1,
                    msg: "invalid Y".to_string(),
                }
            })?;
            let z: f64 = raw.get(46..54).unwrap_or("").trim().parse().map_err(|_| {
                StructError::InvalidPdbqt {
                    line: ln + 1,
                    msg: "invalid Z".to_string(),
                }
            })?;
            let tail: Vec<&str> = raw.get(66..).unwrap_or("").split_whitespace().collect();
            let q: f64 = tail.first().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let ad4_s = tail.last().copied().unwrap_or("C");
            let ad4 = Ad4Type::from_ad4_str(ad4_s);
            let el = match ad4 {
                Ad4Type::A => "C".to_string(),
                Ad4Type::NA | Ad4Type::NS => "N".to_string(),
                Ad4Type::OA | Ad4Type::OS => "O".to_string(),
                Ad4Type::SA => "S".to_string(),
                Ad4Type::HD => "H".to_string(),
                Ad4Type::X => infer_element_from_pdb_name(&name),
                _ => ad4.as_str().to_string(),
            };
            let mut atom = Atom3D::new(mol.atoms.len(), &el, [x, y, z]);
            atom.name = name;
            atom.res_name = res_name;
            atom.chain_id = chain_id;
            atom.res_seq = res_seq;
            atom.is_hetatm = is_het;
            atom.charge = q;
            atom.ad4_type = ad4;
            atom.aromatic = ad4 == Ad4Type::A;
            atom.h_acceptor = ad4.is_acceptor();
            mol.add_atom(atom);
        }
    }
    if mol.atoms.is_empty() {
        return Err(StructError::InvalidPdbqt {
            line: 1,
            msg: "no ATOM/HETATM lines".to_string(),
        });
    }
    mol.infer_bonds_by_distance();
    perceive_all(&mut mol);
    Ok(mol)
}

pub fn write_pdbqt_receptor(mol: &Molecule3D) -> String {
    let mut out = String::new();
    for (idx, a) in mol.atoms.iter().enumerate() {
        out.push_str(&format_pdbqt_atom_line(idx + 1, a));
        out.push('\n');
    }
    out
}

pub fn write_pdbqt_ligand(mol: &Molecule3D) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "REMARK  Name = {}", mol.name);
    let _ = writeln!(out, "ROOT");
    let mut moving_all = std::collections::BTreeSet::new();
    for rb in &mol.rotatable_bonds {
        for &m in &rb.moving_atoms {
            moving_all.insert(m);
        }
    }
    for (idx, a) in mol.atoms.iter().enumerate() {
        if !moving_all.contains(&idx) {
            out.push_str(&format_pdbqt_atom_line(idx + 1, a));
            out.push('\n');
        }
    }
    let _ = writeln!(out, "ENDROOT");
    for rb in &mol.rotatable_bonds {
        let _ = writeln!(out, "BRANCH {:>3} {:>3}", rb.u + 1, rb.v + 1);
        for &m in &rb.moving_atoms {
            if let Some(a) = mol.atoms.get(m) {
                out.push_str(&format_pdbqt_atom_line(m + 1, a));
                out.push('\n');
            }
        }
        let _ = writeln!(out, "ENDBRANCH {:>3} {:>3}", rb.u + 1, rb.v + 1);
    }
    let _ = writeln!(out, "TORSDOF {}", mol.rotatable_bonds.len());
    out
}

fn format_pdbqt_atom_line(serial: usize, a: &Atom3D) -> String {
    let rec = if a.is_hetatm { "HETATM" } else { "ATOM  " };
    let aname = if a.name.len() >= 4 {
        a.name[..4].to_string()
    } else {
        format!(" {:<3}", a.name)
    };
    let rname = format!("{:<3}", &a.res_name[..a.res_name.len().min(3)]);
    let _z = element_atomic_number(&a.element);
    format!(
        "{rec}{serial:>5} {aname:<4} {rname} {}{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00{:>6.2}    {:>+6.3} {:<2}",
        a.chain_id,
        a.res_seq,
        a.coords[0],
        a.coords[1],
        a.coords[2],
        a.b_factor.clamp(0.0, 999.99),
        a.charge,
        a.ad4_type.as_str()
    )
}
