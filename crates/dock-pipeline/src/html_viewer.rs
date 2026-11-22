#![forbid(unsafe_code)]

//! Self-Contained Interactive 3D HTML Scientific Dossier Generator (`export-html`).
//!
//! Produces a single zero-server `.html` file uniting:
//! - Interactive 3D molecular viewport powered by **Mol\*** (inline offline bundle from
//!   `CRAFT_DATA_DIR/assets/molstar/` when `--offline` / cached, or CDN fallback) **plus** a
//!   built-in zero-dependency HTML5 Canvas 3D interactive orbit/zoom fallback that works 100%
//!   air-gapped with zero network access!
//! - Multi-pose / multi-analog switcher (`lead_opt_series.sdf`) with live HYDE v2 per-atom ΔG
//!   coronas, PoseBusters & SeeSAR torsion traffic-lights, directional H-bond/metal interactions,
//!   and optional Ola 1 ΔMIF selectivity hotspots (`SEL_*`), Ola 2 GPU MD / MM-GBSA / RMSF,
//!   Ola 2 RBFE / FEP MBAR diagnostics, and Ola 3 AdmetCraft Human PK / PBPK Dosing & SVG curve.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use dock_engine::{ScoringFunction, score_complex};
use hyde_scorer::{
    analyze_torsions, consensus_affinity_kcal, score_pose_hyde, validate_pose_geometry,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use struct_core::{Molecule3D, write_pdb, write_sdf_3d};
use thiserror::Error;

use crate::md_validate::craft_data_dir;

#[derive(Debug, Error)]
pub enum HtmlExportError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("HTML export error: {0}")]
    Build(String),
}

/// Configuration for generating a self-contained interactive 3D HTML dossier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlDossierConfig {
    pub title: String,
    pub selectivity_json: Option<PathBuf>,
    pub md_metrics_json: Option<PathBuf>,
    pub fep_metrics_json: Option<PathBuf>,
    pub pk_dose_json: Option<PathBuf>,
    pub pk_svg_path: Option<PathBuf>,
    /// When `true`, embeds local `CRAFT_DATA_DIR/assets/molstar/{molstar.js,molstar.css}` if cached.
    pub offline_embed_molstar: bool,
}

impl Default for HtmlDossierConfig {
    fn default() -> Self {
        Self {
            title: "DockCraft Interactive 3D Scientific Dossier".to_string(),
            selectivity_json: None,
            md_metrics_json: None,
            fep_metrics_json: None,
            pk_dose_json: None,
            pk_svg_path: None,
            offline_embed_molstar: true,
        }
    }
}

/// Summary returned after writing an interactive 3D HTML dossier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlDossierSummary {
    pub out_path: String,
    pub bytes: usize,
    pub poses_embedded: usize,
    pub has_selectivity: bool,
    pub has_md_mmgbsa: bool,
    pub has_fep: bool,
    pub has_pk_dose: bool,
    pub molstar_inline_offline: bool,
}

/// Escapes HTML text content safely against XSS injection.
#[must_use]
pub fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_control() && c != '\n' && c != '\r' && c != '\t' {
            continue;
        }
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// Safely serializes a JSON value for embedding inside an HTML `<script>` block
/// (prevents `</script>`, `<!--`, U+2028, U+2029 breakouts).
fn safe_json_for_script(val: &Value) -> Result<String, HtmlExportError> {
    let raw = serde_json::to_string(val)?;
    Ok(raw
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('&', "\\u0026")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029"))
}

fn load_opt_json(path: Option<&Path>) -> Option<Value> {
    let p = path?;
    let raw = std::fs::read_to_string(p).ok()?;
    serde_json::from_str(&raw).ok()
}

/// Extracts pocket-lining receptor heavy atoms within `5.5 Å` of the first ligand for compact Canvas 3D rendering.
fn extract_pocket_atoms_for_canvas(rec: &Molecule3D, lig: &Molecule3D) -> Vec<Value> {
    let lig_heavy: Vec<[f64; 3]> = lig
        .atoms
        .iter()
        .filter(|a| a.is_heavy())
        .map(|a| a.coords)
        .collect();
    if lig_heavy.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for a in &rec.atoms {
        if !a.is_heavy() {
            continue;
        }
        let near = lig_heavy.iter().any(|c| {
            let dx = a.coords[0] - c[0];
            let dy = a.coords[1] - c[1];
            let dz = a.coords[2] - c[2];
            dx * dx + dy * dy + dz * dz <= 5.2 * 5.2
        });
        if near {
            out.push(json!({
                "el": a.element,
                "res": format!("{}:{}{}", a.chain_id, a.res_name, a.res_seq),
                "x": (a.coords[0] * 1000.0).round() / 1000.0,
                "y": (a.coords[1] * 1000.0).round() / 1000.0,
                "z": (a.coords[2] * 1000.0).round() / 1000.0,
            }));
        }
    }
    out
}

/// Builds the complete self-contained interactive 3D HTML string in memory.
pub fn build_interactive_html_dossier(
    receptor: &Molecule3D,
    ligands: &[Molecule3D],
    cfg: &HtmlDossierConfig,
) -> Result<(String, HtmlDossierSummary), HtmlExportError> {
    if ligands.is_empty() {
        return Err(HtmlExportError::Build(
            "at least one 3D ligand pose is required for HTML dossier export".to_string(),
        ));
    }

    let sel_val = load_opt_json(cfg.selectivity_json.as_deref());
    let md_val = load_opt_json(cfg.md_metrics_json.as_deref());
    let fep_val = load_opt_json(cfg.fep_metrics_json.as_deref());
    let pk_val = load_opt_json(cfg.pk_dose_json.as_deref());
    let pk_svg_raw = cfg
        .pk_svg_path
        .as_deref()
        .and_then(|p| std::fs::read_to_string(p).ok());

    let rec_pdb = write_pdb(receptor);
    let pocket_atoms = extract_pocket_atoms_for_canvas(receptor, &ligands[0]);

    // Score & annotate up to 30 poses/analogs in the series
    let mut pose_entries = Vec::new();
    for (idx, lig) in ligands.iter().take(30).enumerate() {
        let vina = score_complex(receptor, lig, ScoringFunction::Vina).total_affinity_kcal;
        let vinardo = score_complex(receptor, lig, ScoringFunction::Vinardo).total_affinity_kcal;
        let hyde_opt = score_pose_hyde(receptor, lig).ok();
        let hyde_dg = hyde_opt.as_ref().map_or(vina, |h| h.total_delta_g_kcal);
        let cons_dg = consensus_affinity_kcal(vina, vinardo, hyde_dg);
        let pb = validate_pose_geometry(receptor, lig);
        let tors = analyze_torsions(lig);
        let sdf_single = write_sdf_3d(&[lig]);

        let atoms_3d: Vec<Value> = lig
            .atoms
            .iter()
            .enumerate()
            .filter(|(_, a)| a.is_heavy())
            .map(|(ai, a)| {
                let dg_a = hyde_opt
                    .as_ref()
                    .and_then(|h| h.atom_scores.iter().find(|s| s.atom_index == ai))
                    .map_or(0.0, |s| s.delta_g_kcal);
                json!({
                    "id": ai,
                    "el": a.element,
                    "x": (a.coords[0] * 1000.0).round() / 1000.0,
                    "y": (a.coords[1] * 1000.0).round() / 1000.0,
                    "z": (a.coords[2] * 1000.0).round() / 1000.0,
                    "dg": (dg_a * 100.0).round() / 100.0,
                })
            })
            .collect();
        let bonds_3d: Vec<[usize; 2]> = lig
            .bonds
            .iter()
            .filter(|b| {
                lig.atoms
                    .get(b.u)
                    .is_some_and(struct_core::Atom3D::is_heavy)
                    && lig
                        .atoms
                        .get(b.v)
                        .is_some_and(struct_core::Atom3D::is_heavy)
            })
            .map(|b| [b.u, b.v])
            .collect();
        let interactions: Vec<Value> = hyde_opt
            .as_ref()
            .map(|h| {
                h.interactions
                    .iter()
                    .map(|it| {
                        let lig_pt = it
                            .lig_atom_indices
                            .first()
                            .and_then(|&i| lig.atoms.get(i))
                            .map(|a| a.coords);
                        let rec_pt = it
                            .rec_atom_indices
                            .first()
                            .and_then(|&i| receptor.atoms.get(i))
                            .map(|a| a.coords);
                        json!({
                            "kind": format!("{:?}", it.kind),
                            "residue": it.rec_residue,
                            "distance_a": (it.distance_a * 100.0).round() / 100.0,
                            "energy_kcal": (it.energy_kcal * 100.0).round() / 100.0,
                            "lig_pt": lig_pt,
                            "rec_pt": rec_pt,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let lname = if lig.name.trim().is_empty() {
            format!("Pose_{}", idx + 1)
        } else {
            lig.name.trim().to_string()
        };
        pose_entries.push(json!({
            "index": idx,
            "name": lname,
            "sdf": sdf_single,
            "vina_kcal": (vina * 100.0).round() / 100.0,
            "vinardo_kcal": (vinardo * 100.0).round() / 100.0,
            "hyde_kcal": (hyde_dg * 100.0).round() / 100.0,
            "consensus_kcal": (cons_dg * 100.0).round() / 100.0,
            "ki_nm": hyde_opt.as_ref().map(|h| (h.estimated_ki_nm * 10.0).round() / 10.0),
            "posebusters_valid": pb.valid,
            "pb_bond_ok": pb.bond_lengths_ok,
            "pb_angle_ok": pb.bond_angles_ok,
            "pb_flat_ok": pb.aromatic_flatness_ok,
            "pb_no_clash": pb.no_internal_clash && pb.no_protein_clash,
            "torsion_red": tors.red_count,
            "torsion_yellow": tors.yellow_count,
            "atoms": atoms_3d,
            "bonds": bonds_3d,
            "interactions": interactions,
        }));
    }

    // Check for locally cached offline Mol* static bundle in CRAFT_DATA_DIR/assets/molstar/
    let molstar_dir = craft_data_dir().join("assets").join("molstar");
    let local_js = molstar_dir.join("molstar.js");
    let local_css = molstar_dir.join("molstar.css");
    let (molstar_inline_css, molstar_inline_js, molstar_offline) =
        if cfg.offline_embed_molstar && local_js.is_file() && local_css.is_file() {
            let css = std::fs::read_to_string(&local_css).unwrap_or_default();
            let js = std::fs::read_to_string(&local_js).unwrap_or_default();
            if !css.is_empty() && !js.is_empty() {
                (Some(css), Some(js), true)
            } else {
                (None, None, false)
            }
        } else {
            (None, None, false)
        };

    let payload = json!({
        "title": cfg.title,
        "receptor_name": receptor.name,
        "receptor_pdb": rec_pdb,
        "pocket_atoms": pocket_atoms,
        "poses": pose_entries,
        "selectivity": sel_val,
        "md": md_val,
        "fep": fep_val,
        "pk": pk_val,
    });
    let safe_payload = safe_json_for_script(&payload)?;
    let safe_title = escape_html(&cfg.title);

    let mut html = String::with_capacity(64 * 1024);
    let _ = writeln!(
        html,
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\"/>\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>\n\
         <title>{safe_title}</title>"
    );

    if let Some(css) = &molstar_inline_css {
        html.push_str("<style>\n");
        html.push_str(&css.replace("</style", "<\\/style"));
        html.push_str("\n</style>\n");
    } else {
        html.push_str(
            "<link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/molstar@4.11.0/build/viewer/molstar.css\"/>\n",
        );
    }

    html.push_str(HTML_DOSSIER_CSS);
    html.push_str("</head>\n<body>\n");
    let _ = writeln!(
        html,
        "<header class=\"topbar\"><div><h1>{safe_title}</h1>\
         <span class=\"sub\">DockCraft v0.8.0 · CraftBio Self-Contained 3D Dossier · {}</span></div>\
         <div class=\"controls\"><label>Pose / Analog: <select id=\"poseSelect\"></select></label>\
         <button id=\"toggleEngineBtn\" title=\"Switch between Mol* WebGL and offline Canvas 3D\">Mode: Auto</button>\
         <button id=\"toggleHydeBtn\">Color: HYDE ΔG</button></div></header>",
        escape_html(&receptor.name)
    );

    html.push_str(
        "<main class=\"layout\">\n\
         <section class=\"viewer-pane\">\n\
           <div id=\"molstar-container\"></div>\n\
           <canvas id=\"canvas3d\"></canvas>\n\
           <div class=\"viewer-hud\" id=\"viewerHud\">Drag to orbit · Scroll to zoom</div>\n\
         </section>\n\
         <aside class=\"dashboard\">\n\
           <nav class=\"tabs\">\n\
             <button class=\"tab active\" data-tab=\"tab-pose\">1. Pose &amp; HYDE</button>\n\
             <button class=\"tab\" data-tab=\"tab-series\">2. Series &amp; ΔMIF</button>\n\
             <button class=\"tab\" data-tab=\"tab-physics\">3. MD / MM-GBSA / FEP</button>\n\
             <button class=\"tab\" data-tab=\"tab-pk\">4. Human PK / PBPK</button>\n\
           </nav>\n\
           <div id=\"tab-pose\" class=\"tab-panel active\"></div>\n\
           <div id=\"tab-series\" class=\"tab-panel\"></div>\n\
           <div id=\"tab-physics\" class=\"tab-panel\"></div>\n\
           <div id=\"tab-pk\" class=\"tab-panel\">",
    );
    if let Some(svg) = &pk_svg_raw {
        // Only embed well-formed SVG element without script tags
        let cleaned = svg
            .replace("<script", "&lt;script")
            .replace("</script", "&lt;/script");
        let _ = writeln!(
            html,
            "<div class=\"svg-box\" id=\"pkSvgSlot\">{cleaned}</div>"
        );
    }
    html.push_str("<div id=\"pkTableSlot\"></div></div>\n</aside>\n</main>\n");

    let _ = writeln!(
        html,
        "<script>window.CRAFTBIO_DATA = {safe_payload};</script>"
    );
    if let Some(js) = &molstar_inline_js {
        html.push_str("<script>\n");
        html.push_str(&js.replace("</script", "<\\/script"));
        html.push_str("\n</script>\n");
    } else {
        html.push_str(
            "<script src=\"https://cdn.jsdelivr.net/npm/molstar@4.11.0/build/viewer/molstar.js\"></script>\n",
        );
    }
    html.push_str("<script>\n");
    html.push_str(HTML_DOSSIER_JS);
    html.push_str("\n</script>\n</body>\n</html>\n");

    let summary = HtmlDossierSummary {
        out_path: String::new(),
        bytes: html.len(),
        poses_embedded: pose_entries.len(),
        has_selectivity: sel_val.is_some(),
        has_md_mmgbsa: md_val.is_some(),
        has_fep: fep_val.is_some(),
        has_pk_dose: pk_val.is_some(),
        molstar_inline_offline: molstar_offline,
    };
    Ok((html, summary))
}

/// Writes the self-contained interactive 3D HTML dossier to `out_path`.
pub fn write_interactive_html_dossier(
    receptor: &Molecule3D,
    ligands: &[Molecule3D],
    cfg: &HtmlDossierConfig,
    out_path: &Path,
) -> Result<HtmlDossierSummary, HtmlExportError> {
    let (html, mut summary) = build_interactive_html_dossier(receptor, ligands, cfg)?;
    if let Some(par) = out_path.parent()
        && !par.as_os_str().is_empty()
    {
        std::fs::create_dir_all(par)?;
    }
    std::fs::write(out_path, &html)?;
    summary.out_path = out_path.display().to_string();
    Ok(summary)
}

const HTML_DOSSIER_CSS: &str = r#"<style>
* { box-sizing: border-box; }
body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: #0f172a; color: #e2e8f0; height: 100vh; display: flex; flex-direction: column; }
.topbar { display: flex; justify-content: space-between; align-items: center; padding: 10px 16px; background: #1e293b; border-bottom: 1px solid #334155; gap: 12px; flex-wrap: wrap; }
.topbar h1 { margin: 0; font-size: 16px; color: #f8fafc; }
.topbar .sub { font-size: 12px; color: #94a3b8; }
.controls { display: flex; align-items: center; gap: 8px; font-size: 13px; }
.controls select, .controls button { background: #0f172a; color: #e2e8f0; border: 1px solid #475569; border-radius: 6px; padding: 5px 10px; font-size: 12px; cursor: pointer; }
.controls button:hover { border-color: #38bdf8; }
.layout { flex: 1; display: grid; grid-template-columns: 1.25fr 1fr; min-height: 0; }
@media (max-width: 980px) { .layout { grid-template-columns: 1fr; grid-template-rows: 50vh 1fr; } }
.viewer-pane { position: relative; background: #0b1120; border-right: 1px solid #334155; overflow: hidden; }
#molstar-container, #canvas3d { position: absolute; inset: 0; width: 100%; height: 100%; }
#canvas3d { display: none; cursor: grab; }
.viewer-hud { position: absolute; left: 10px; bottom: 10px; background: rgba(15,23,42,0.82); color: #cbd5e1; padding: 5px 10px; border-radius: 6px; font-size: 11px; pointer-events: none; z-index: 20; border: 1px solid #334155; }
.dashboard { display: flex; flex-direction: column; min-height: 0; background: #0f172a; }
.tabs { display: flex; background: #1e293b; border-bottom: 1px solid #334155; }
.tab { flex: 1; padding: 9px 6px; background: transparent; color: #94a3b8; border: none; border-bottom: 2px solid transparent; cursor: pointer; font-size: 12px; font-weight: 600; }
.tab.active { color: #38bdf8; border-bottom-color: #38bdf8; background: #0f172a; }
.tab-panel { display: none; flex: 1; overflow-y: auto; padding: 14px; font-size: 13px; }
.tab-panel.active { display: block; }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(125px, 1fr)); gap: 8px; margin-bottom: 12px; }
.kpi { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 8px 10px; }
.kpi .lbl { font-size: 11px; color: #94a3b8; }
.kpi .val { font-size: 16px; font-weight: 700; color: #f8fafc; margin-top: 2px; }
.badge-ok { display: inline-block; padding: 2px 7px; border-radius: 99px; background: #14532d; color: #86efac; font-size: 11px; font-weight: 700; }
.badge-warn { display: inline-block; padding: 2px 7px; border-radius: 99px; background: #713f12; color: #fde047; font-size: 11px; font-weight: 700; }
.badge-bad { display: inline-block; padding: 2px 7px; border-radius: 99px; background: #7f1d1d; color: #fca5a5; font-size: 11px; font-weight: 700; }
table { width: 100%; border-collapse: collapse; margin: 8px 0 14px; font-size: 12px; }
th, td { padding: 6px 8px; border-bottom: 1px solid #1e293b; text-align: left; }
th { background: #1e293b; color: #cbd5e1; position: sticky; top: 0; }
tr.clickable { cursor: pointer; }
tr.clickable:hover, tr.selected-row { background: #1e3a5f; }
h2 { font-size: 14px; margin: 12px 0 6px; color: #7dd3fc; border-bottom: 1px solid #1e293b; padding-bottom: 4px; }
.svg-box svg { max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #334155; background: #fff; }
.bar-wrap { background: #1e293b; border-radius: 4px; height: 10px; overflow: hidden; }
.bar-fill { height: 100%; background: #38bdf8; }
</style>
"#;

const HTML_DOSSIER_JS: &str = r#"
(function() {
  const D = window.CRAFTBIO_DATA;
  let activeIdx = 0;
  let colorByHyde = true;
  let useMolstar = false;
  let molstarViewer = null;
  let rotX = 0.35, rotY = 0.55, zoom = 26.0, dragging = false, lastX = 0, lastY = 0;

  const poseSelect = document.getElementById('poseSelect');
  const toggleEngineBtn = document.getElementById('toggleEngineBtn');
  const toggleHydeBtn = document.getElementById('toggleHydeBtn');
  const molstarEl = document.getElementById('molstar-container');
  const canvasEl = document.getElementById('canvas3d');
  const hudEl = document.getElementById('viewerHud');

  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }
  function fmt(v, n=2) { return (typeof v === 'number' && isFinite(v)) ? v.toFixed(n) : '—'; }

  // Populate pose selector
  D.poses.forEach((p, idx) => {
    const opt = document.createElement('option');
    opt.value = String(idx);
    opt.textContent = `#${idx+1} ${p.name} (${fmt(p.consensus_kcal)} kcal/mol)`;
    poseSelect.appendChild(opt);
  });

  // Tab switching
  document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.tab).classList.add('active');
    });
  });

  // --- Zero-dependency offline HTML5 Canvas 3D interactive fallback ---
  const EL_COLOR = { C:'#94a3b8', N:'#3b82f6', O:'#ef4444', S:'#eab308', F:'#22c55e', Cl:'#16a34a', Br:'#a16207', P:'#f97316', Zn:'#a855f7' };
  function hydeColor(dg) {
    if (dg <= -0.8) return '#22c55e';
    if (dg <= -0.25) return '#86efac';
    if (dg >= 0.45) return '#ef4444';
    if (dg >= 0.15) return '#f59e0b';
    return '#cbd5e1';
  }
  function centerOf(pose) {
    let cx=0, cy=0, cz=0, n=Math.max(pose.atoms.length, 1);
    pose.atoms.forEach(a => { cx+=a.x; cy+=a.y; cz+=a.z; });
    return [cx/n, cy/n, cz/n];
  }
  function project(x, y, z, c, W, H) {
    const dx = x - c[0], dy = y - c[1], dz = z - c[2];
    const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
    const x1 = dx * cosY + dz * sinY;
    const z1 = -dx * sinY + dz * cosY;
    const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
    const y2 = dy * cosX - z1 * sinX;
    const z2 = dy * sinX + z1 * cosX;
    const persp = 500.0 / Math.max(20.0, 260.0 - z2 * zoom * 0.45);
    return [W * 0.5 + x1 * zoom * persp * 0.5, H * 0.5 - y2 * zoom * persp * 0.5, z2];
  }
  function drawCanvas3D() {
    if (useMolstar) return;
    const rect = canvasEl.parentElement.getBoundingClientRect();
    if (canvasEl.width !== Math.floor(rect.width) || canvasEl.height !== Math.floor(rect.height)) {
      canvasEl.width = Math.max(300, Math.floor(rect.width));
      canvasEl.height = Math.max(240, Math.floor(rect.height));
    }
    const ctx = canvasEl.getContext('2d');
    const W = canvasEl.width, H = canvasEl.height;
    ctx.fillStyle = '#0b1120';
    ctx.fillRect(0, 0, W, H);
    const pose = D.poses[activeIdx];
    const ctr = centerOf(pose);

    // Draw pocket receptor context atoms
    const drawItems = [];
    (D.pocket_atoms || []).forEach(a => {
      const p = project(a.x, a.y, a.z, ctr, W, H);
      drawItems.push({ z: p[2] - 0.1, kind: 'rec', x: p[0], y: p[1], el: a.el });
    });
    // Optional ΔMIF selectivity hotspots
    if (D.selectivity && Array.isArray(D.selectivity.selectivity_hotspots)) {
      D.selectivity.selectivity_hotspots.slice(0, 8).forEach(hs => {
        if (hs.coords) {
          const p = project(hs.coords[0], hs.coords[1], hs.coords[2], ctr, W, H);
          drawItems.push({ z: p[2], kind: 'sel', x: p[0], y: p[1], label: hs.id || 'SEL' });
        }
      });
    }
    // Map ligand atom coords
    const atomMap = {};
    pose.atoms.forEach(a => {
      const p = project(a.x, a.y, a.z, ctr, W, H);
      atomMap[a.id] = p;
      drawItems.push({ z: p[2], kind: 'lig', x: p[0], y: p[1], el: a.el, dg: a.dg });
    });
    // Ligand bonds
    pose.bonds.forEach(([u, v]) => {
      if (atomMap[u] && atomMap[v]) {
        const pu = atomMap[u], pv = atomMap[v];
        drawItems.push({ z: 0.5*(pu[2]+pv[2]) - 0.05, kind: 'bond', x1: pu[0], y1: pu[1], x2: pv[0], y2: pv[1] });
      }
    });
    // Directional H-bonds / interactions
    (pose.interactions || []).forEach(it => {
      if (it.lig_pt && it.rec_pt) {
        const p1 = project(it.lig_pt[0], it.lig_pt[1], it.lig_pt[2], ctr, W, H);
        const p2 = project(it.rec_pt[0], it.rec_pt[1], it.rec_pt[2], ctr, W, H);
        drawItems.push({ z: 0.5*(p1[2]+p2[2]), kind: 'inter', x1: p1[0], y1: p1[1], x2: p2[0], y2: p2[1], label: it.residue });
      }
    });

    drawItems.sort((a, b) => a.z - b.z);
    drawItems.forEach(it => {
      if (it.kind === 'rec') {
        ctx.fillStyle = 'rgba(148,163,184,0.28)';
        ctx.beginPath(); ctx.arc(it.x, it.y, 4.2, 0, Math.PI*2); ctx.fill();
      } else if (it.kind === 'sel') {
        ctx.fillStyle = 'rgba(250,204,21,0.22)';
        ctx.strokeStyle = '#facc15'; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.arc(it.x, it.y, 12, 0, Math.PI*2); ctx.fill(); ctx.stroke();
        ctx.fillStyle = '#fef08a'; ctx.font = '10px sans-serif'; ctx.fillText(it.label, it.x+14, it.y+3);
      } else if (it.kind === 'bond') {
        ctx.strokeStyle = '#e2e8f0'; ctx.lineWidth = 2.6;
        ctx.beginPath(); ctx.moveTo(it.x1, it.y1); ctx.lineTo(it.x2, it.y2); ctx.stroke();
      } else if (it.kind === 'inter') {
        ctx.save(); ctx.setLineDash([4, 3]); ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.moveTo(it.x1, it.y1); ctx.lineTo(it.x2, it.y2); ctx.stroke();
        ctx.restore();
        ctx.fillStyle = '#7dd3fc'; ctx.font = '10px sans-serif';
        ctx.fillText(it.label, 0.5*(it.x1+it.x2)+4, 0.5*(it.y1+it.y2));
      } else if (it.kind === 'lig') {
        ctx.fillStyle = colorByHyde ? hydeColor(it.dg) : (EL_COLOR[it.el] || '#cbd5e1');
        ctx.strokeStyle = '#0f172a'; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.arc(it.x, it.y, 7.2, 0, Math.PI*2); ctx.fill(); ctx.stroke();
      }
    });
  }

  canvasEl.addEventListener('mousedown', e => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
  window.addEventListener('mouseup', () => { dragging = false; });
  window.addEventListener('mousemove', e => {
    if (!dragging || useMolstar) return;
    rotY += (e.clientX - lastX) * 0.012;
    rotX += (e.clientY - lastY) * 0.012;
    lastX = e.clientX; lastY = e.clientY;
    drawCanvas3D();
  });
  canvasEl.addEventListener('wheel', e => {
    if (useMolstar) return;
    e.preventDefault();
    zoom = Math.max(8, Math.min(80, zoom * (e.deltaY < 0 ? 1.12 : 0.89)));
    drawCanvas3D();
  }, { passive: false });

  // --- Optional Mol* WebGL Viewer ---
  async function loadMolstarPose(idx) {
    if (!useMolstar || !molstarViewer) return;
    try {
      await molstarViewer.plugin.clear();
      await molstarViewer.loadStructureFromData(D.receptor_pdb, 'pdb', { dataLabel: D.receptor_name });
      await molstarViewer.loadStructureFromData(D.poses[idx].sdf, 'sdf', { dataLabel: D.poses[idx].name });
    } catch (err) {
      console.warn('Mol* load fallback to Canvas 3D:', err);
      switchToCanvas();
    }
  }
  function switchToCanvas() {
    useMolstar = false;
    molstarEl.style.display = 'none';
    canvasEl.style.display = 'block';
    toggleEngineBtn.textContent = 'Viewer: Canvas 3D';
    hudEl.textContent = 'Canvas 3D (Offline) · Drag to orbit · Scroll to zoom · Green=favorable HYDE, Red=clash/desolv';
    drawCanvas3D();
  }
  async function initMolstar() {
    if (typeof window.molstar === 'undefined' || !window.molstar.Viewer) {
      switchToCanvas();
      return;
    }
    try {
      molstarViewer = await window.molstar.Viewer.create('molstar-container', {
        layoutIsExpanded: false,
        layoutShowControls: false,
        layoutShowRemoteState: false,
        layoutShowSequence: false,
        viewportShowExpand: true,
      });
      useMolstar = true;
      molstarEl.style.display = 'block';
      canvasEl.style.display = 'none';
      toggleEngineBtn.textContent = 'Viewer: Mol* WebGL';
      hudEl.textContent = 'Mol* WebGL Viewer active · Switch poses from top selector';
      await loadMolstarPose(activeIdx);
    } catch (err) {
      console.warn('Mol* init failed, using Canvas 3D fallback:', err);
      switchToCanvas();
    }
  }

  toggleEngineBtn.addEventListener('click', () => {
    if (useMolstar) { switchToCanvas(); } else { initMolstar(); }
  });
  toggleHydeBtn.addEventListener('click', () => {
    colorByHyde = !colorByHyde;
    toggleHydeBtn.textContent = colorByHyde ? 'Color: HYDE ΔG' : 'Color: Element CPK';
    drawCanvas3D();
  });

  function renderPoseTab() {
    const p = D.poses[activeIdx];
    const pbBadge = p.posebusters_valid ? '<span class="badge-ok">PoseBusters PASS</span>' : '<span class="badge-warn">PoseBusters WARN</span>';
    const torBadge = p.torsion_red === 0 ? '<span class="badge-ok">0 Red Torsions</span>' : `<span class="badge-bad">${p.torsion_red} Red Torsions</span>`;
    let html = `<div class="kpi-grid">
      <div class="kpi"><div class="lbl">Consensus ΔG</div><div class="val">${fmt(p.consensus_kcal)} kcal/mol</div></div>
      <div class="kpi"><div class="lbl">Vina / Vinardo</div><div class="val">${fmt(p.vina_kcal)} / ${fmt(p.vinardo_kcal)}</div></div>
      <div class="kpi"><div class="lbl">HYDE v2 ΔG</div><div class="val">${fmt(p.hyde_kcal)} kcal/mol</div></div>
      <div class="kpi"><div class="lbl">Geometry Audit</div><div class="val" style="font-size:12px">${pbBadge} ${torBadge}</div></div>
    </div>`;
    html += `<h2>Detected Protein–Ligand Interactions (${(p.interactions||[]).length})</h2>`;
    if (p.interactions && p.interactions.length) {
      html += '<table><thead><tr><th>Type</th><th>Residue</th><th>Dist (Å)</th><th>Energy (kcal/mol)</th></tr></thead><tbody>';
      p.interactions.forEach(it => {
        html += `<tr><td>${esc(it.kind)}</td><td><code>${esc(it.residue)}</code></td><td>${fmt(it.distance_a)}</td><td>${fmt(it.energy_kcal)}</td></tr>`;
      });
      html += '</tbody></table>';
    } else {
      html += '<p>No directional polar/π/metal interactions detected for this pose.</p>';
    }
    document.getElementById('tab-pose').innerHTML = html;
  }

  function renderSeriesTab() {
    let html = `<h2>Embedded Congeneric / Docked Series (${D.poses.length} pose(s))</h2>`;
    html += '<table><thead><tr><th>#</th><th>Compound</th><th>Cons. ΔG</th><th>Vina</th><th>HYDE</th><th>PoseBusters</th></tr></thead><tbody>';
    D.poses.forEach((p, idx) => {
      const cls = idx === activeIdx ? 'clickable selected-row' : 'clickable';
      html += `<tr class="${cls}" data-pose-idx="${idx}"><td>${idx+1}</td><td><code>${esc(p.name)}</code></td><td><b>${fmt(p.consensus_kcal)}</b></td><td>${fmt(p.vina_kcal)}</td><td>${fmt(p.hyde_kcal)}</td><td>${p.posebusters_valid ? 'OK' : 'WARN'}</td></tr>`;
    });
    html += '</tbody></table>';
    if (D.selectivity) {
      const s = D.selectivity;
      html += `<h2>Ola 1 Structural Pocket Selectivity (ΔMIF Feasibility: ${fmt(s.selectivity_feasibility_score, 3)})</h2>`;
      if (Array.isArray(s.selectivity_hotspots) && s.selectivity_hotspots.length) {
        html += '<table><thead><tr><th>Hotspot</th><th>Probe / Role</th><th>Score</th><th>Nearest Differential Residue</th></tr></thead><tbody>';
        s.selectivity_hotspots.slice(0, 12).forEach(h => {
          html += `<tr><td><code>${esc(h.id)}</code></td><td>${esc(h.probe_name || h.kind || '')}</td><td>${fmt(h.delta_score || h.score)}</td><td><code>${esc(h.nearest_diff_residue || h.rationale || '')}</code></td></tr>`;
        });
        html += '</tbody></table>';
      }
    }
    const container = document.getElementById('tab-series');
    container.innerHTML = html;
    container.querySelectorAll('tr[data-pose-idx]').forEach(tr => {
      tr.addEventListener('click', () => selectPose(parseInt(tr.dataset.poseIdx, 10)));
    });
  }

  function renderPhysicsTab() {
    let html = '';
    if (D.md) {
      const m = D.md;
      html += `<h2>Ola 2 OpenMM GPU MD &amp; Physical MM/GBSA (${esc(m.platform_used)} · ${fmt(m.throughput_ns_per_day, 1)} ns/day)</h2>`;
      html += `<div class="kpi-grid">
        <div class="kpi"><div class="lbl">Verdict (${m.replicates_run || 1} rep)</div><div class="val" style="font-size:13px">${esc(m.stability_verdict)}</div></div>
        <div class="kpi"><div class="lbl">Ligand RMSD</div><div class="val">${fmt(m.ligand_rmsd_mean_a)} ± ${fmt(m.ligand_rmsd_std_a)} Å</div></div>
        <div class="kpi"><div class="lbl">MM/GBSA ΔG</div><div class="val">${fmt(m.mmgbsa_dg_kcal)} ± ${fmt(m.mmgbsa_sem_kcal)}</div></div>
      </div>`;
      if (m.mmgbsa && Array.isArray(m.mmgbsa.per_residue_top)) {
        html += '<table><thead><tr><th>Pocket Residue</th><th>ΔG_total</th><th>ΔE_vdW</th><th>ΔE_elec + ΔG_GB</th></tr></thead><tbody>';
        m.mmgbsa.per_residue_top.slice(0, 10).forEach(r => {
          html += `<tr><td><code>${esc(r.residue)}</code></td><td><b>${fmt(r.total_kcal)}</b></td><td>${fmt(r.vdw_kcal)}</td><td>${fmt(r.elec_gb_kcal)}</td></tr>`;
        });
        html += '</tbody></table>';
      }
    }
    if (D.fep && Array.isArray(D.fep.edge_results)) {
      html += `<h2>Ola 2 Alchemical RBFE / FEP (${esc(D.fep.platform_used)} · ${D.fep.n_windows} λ-windows)</h2>`;
      html += '<table><thead><tr><th>Edge (A → B)</th><th>MCS</th><th>ΔΔG MBAR (kcal/mol)</th><th>ΔΔG TI</th><th>K_i Fold</th><th>Min Overlap</th><th>Status</th></tr></thead><tbody>';
      D.fep.edge_results.forEach(e => {
        const minOv = Math.min(e.complex_leg.min_adjacent_overlap, e.solvent_leg.min_adjacent_overlap);
        html += `<tr><td><code>${esc(e.from_name)} → ${esc(e.to_name)}</code></td><td>${e.mcs_heavy_atoms}</td><td><b>${fmt(e.ddg_bind_kcal)} ± ${fmt(e.ddg_sem_kcal)}</b></td><td>${fmt(e.ddg_ti_kcal)}</td><td>${fmt(e.fold_change_ki)}×</td><td>${fmt(minOv, 3)}</td><td>${esc(e.convergence_status)}</td></tr>`;
      });
      html += '</tbody></table>';
    }
    if (!html) {
      html = '<p>No MD/MM-GBSA or RBFE/FEP metrics JSON attached to this dossier. Pass <code>--md-metrics</code> or <code>--fep-metrics</code> to embed them.</p>';
    }
    document.getElementById('tab-physics').innerHTML = html;
  }

  function renderPkTab() {
    const slot = document.getElementById('pkTableSlot');
    if (!D.pk) {
      slot.innerHTML = '<p>No Human PK/PBPK JSON attached. Generate one with <code>admetcraft-cli pk-dose --out-json ... --out-svg ...</code> and pass <code>--pk-json</code>.</p>';
      return;
    }
    const p = D.pk;
    let html = `<h2>Ola 3 Human Minimal-PBPK Dosing (${esc(p.molecule_name)})</h2>`;
    html += `<div class="kpi-grid">
      <div class="kpi"><div class="lbl">Recommended</div><div class="val" style="font-size:13px">${esc(p.recommended_regimen)}</div></div>
      <div class="kpi"><div class="lbl">Oral F / f_u,p</div><div class="val">${fmt(p.oral_bioavailability_f*100,1)}% / ${fmt(p.fup,3)}</div></div>
      <div class="kpi"><div class="lbl">t1/2 (IVIVE)</div><div class="val">${fmt(p.t_half_ivive_h,1)} h</div></div>
      <div class="kpi"><div class="lbl">BCS / CL_tot</div><div class="val" style="font-size:13px">${esc(p.bcs_class)} · ${fmt(p.cl_total_ml_min_kg)}</div></div>
    </div>`;
    html += '<table><thead><tr><th>Regimen</th><th>Daily Dose (90% CI)</th><th>Cmax,u</th><th>Cmin,u</th><th>Do</th><th>hERG Margin</th><th>Feasibility</th></tr></thead><tbody>';
    [p.qd, p.bid].forEach(r => {
      if (!r) return;
      html += `<tr><td><b>${esc(r.regimen_label)}</b></td><td>${fmt(r.daily_dose_mg,1)} mg (${fmt(r.daily_dose_ci90_mg[0],0)}-${fmt(r.daily_dose_ci90_mg[1],0)})</td><td>${fmt(r.cmax_u_nm,1)} nM</td><td>${fmt(r.cmin_u_nm,1)} nM</td><td>${fmt(r.dose_number_do,2)}</td><td>${fmt(r.herg_safety_margin,0)}×</td><td>${esc(r.feasibility)}</td></tr>`;
    });
    html += '</tbody></table>';
    if (Array.isArray(p.warnings) && p.warnings.length) {
      html += '<h2>Translational DMPK Alerts</h2><ul>' + p.warnings.map(w => `<li>${esc(w)}</li>`).join('') + '</ul>';
    }
    slot.innerHTML = html;
  }

  function selectPose(idx) {
    activeIdx = Math.max(0, Math.min(D.poses.length - 1, idx));
    poseSelect.value = String(activeIdx);
    renderPoseTab();
    renderSeriesTab();
    if (useMolstar) { loadMolstarPose(activeIdx); } else { drawCanvas3D(); }
  }

  poseSelect.addEventListener('change', () => selectPose(parseInt(poseSelect.value, 10)));
  window.addEventListener('resize', drawCanvas3D);

  renderPoseTab();
  renderSeriesTab();
  renderPhysicsTab();
  renderPkTab();
  initMolstar();
})();
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use struct_core::{Atom3D, parse_smiles_to_3d, prepare_receptor};

    #[test]
    fn html_dossier_escapes_xss_embeds_series_and_supports_offline_canvas_fallback()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut rec = Molecule3D::new("Target</script><img src=x>");
        let mut a1 = Atom3D::new(1, "N", [3.0, 0.0, 0.0]);
        a1.is_hetatm = false;
        a1.res_name = "LYS".to_string();
        let mut a2 = Atom3D::new(2, "O", [-3.0, 0.0, 0.0]);
        a2.is_hetatm = false;
        a2.res_name = "ASP".to_string();
        rec.add_atom(a1);
        rec.add_atom(a2);
        prepare_receptor(&mut rec, true);

        let l1 = parse_smiles_to_3d("c1ccccc1O", Some("Lead</script>A"))?;
        let l2 = parse_smiles_to_3d("Fc1ccccc1O", Some("Lead_B"))?;
        let cfg = HtmlDossierConfig {
            title: "CDK2 <script>alert(1)</script>".to_string(),
            offline_embed_molstar: false,
            ..Default::default()
        };
        let (html, summary) = build_interactive_html_dossier(&rec, &[l1, l2], &cfg)?;
        assert_eq!(summary.poses_embedded, 2);
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("molstar-container") && html.contains("canvas3d"));
        assert!(html.contains("&lt;script&gt;alert(1)&lt;/script&gt;"));
        assert!(
            !html.contains("Lead</script>A"),
            "raw </script> must never appear unescaped inside embedded JSON"
        );
        Ok(())
    }
}
