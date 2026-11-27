//! Floating, movable, resizable & expandable Mini-Applications for DockCraft.
//!
//! Organizes DockCraft's SBDD capabilities (Binding Pockets & GridBox, Receptor Prep & Bridging
//! Waters, Active-Site MIF Blueprint & Pharmacophores, SeeSAR HYDE v2 / PoseBusters / Torsions,
//! 32-Group In-Pocket Inspirator, Virtual Screening MPO Table & CraftBio Handoffs, and
//! Interactive Tutorials) into self-contained floating windows with plain-Spanish educational
//! banners (*Qué hace / Cuándo usarla / Flujo I/O*) and 1-click maximize/restore.

use std::collections::BTreeSet;

use eframe::egui;

/// Identifier of each floating Mini-Application inside DockCraft.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MiniAppId {
    Tutorials,
    DockingGridBox,
    ReceptorPrepWaters,
    PharmacophoreSite,
    HydeAudit,
    InspiratorAdmet,
    CampaignSuite,
}

/// Descriptive metadata + plain-Spanish explanation for each DockCraft Mini-App.
#[derive(Debug, Clone, Copy)]
pub struct MiniAppMeta {
    pub id: MiniAppId,
    pub short_label: &'static str,
    pub window_title: &'static str,
    pub what_it_does: &'static str,
    pub when_to_use: &'static str,
    pub io_summary: &'static str,
    pub default_size: [f32; 2],
    pub default_offset: [f32; 2],
    pub accent_rgb: [u8; 3],
}

pub const ALL_MINI_APPS: &[MiniAppMeta] = &[
    MiniAppMeta {
        id: MiniAppId::Tutorials,
        short_label: "📚 Tutoriales",
        window_title: "📚 Tutoriales Interactivos de DockCraft",
        what_it_does: "Recorridos prácticos guiados paso a paso con botones de ejecución directa sobre el complejo 3D activo.",
        when_to_use: "Ideal al abrir DockCraft o al estrenar ProtPrep, MIFs farmacofóricos, SeeSAR HYDE v2, Inspirator 32 o cribado virtual.",
        io_summary: "Entrada: complejo 3D activo → Salida: práctica guiada en vivo y seguimiento de progreso.",
        default_size: [540.0, 560.0],
        default_offset: [36.0, 108.0],
        accent_rgb: [16, 185, 129],
    },
    MiniAppMeta {
        id: MiniAppId::DockingGridBox,
        short_label: "📦 Bolsillos & Docking",
        window_title: "📦 Bolsillos 3D, Caja GridBox y Docking",
        what_it_does: "Detecta cavidades druggables (LIGSITE-csc 26 rayos), ajusta la Caja de Docking 3D (CX/CY/CZ, SX/SY/SZ), carga ligandos SMILES/SDF y ejecuta Vina/Vinardo/Hybrid + consenso HYDE.",
        when_to_use: "Para definir el sitio de unión exacto (o sincronizar la caja desde BiomolCraft) y generar poses 3D rankeadas.",
        io_summary: "Entrada: receptor PDB/mmCIF + ligando/SMILES → Salida: poses 3D (.sdf) y energías kcal/mol.",
        default_size: [470.0, 620.0],
        default_offset: [64.0, 118.0],
        accent_rgb: [6, 182, 212],
    },
    MiniAppMeta {
        id: MiniAppId::ReceptorPrepWaters,
        short_label: "🧪 ProtPrep & Aguas",
        window_title: "🧪 Preparación Receptor (ProtPrep pKa) y Aguas Puente",
        what_it_does: "Audita calidad cristalográfica/AlphaFold, calcula estados de protonación a pH fisiológico, optimiza flips Asn/Gln/His y selecciona aguas puente estructurales o coordinación metálica AutoDock4Zn.",
        when_to_use: "Antes de lanzar un cribado serio sobre un PDB/mmCIF crudo o metaloproteínas (Zn/Mg/Fe).",
        io_summary: "Entrada: receptor crudo PDB/mmCIF → Salida: receptor preparado, informe de auditoría (.md) y aguas retenidas.",
        default_size: [480.0, 560.0],
        default_offset: [110.0, 130.0],
        accent_rgb: [59, 130, 246],
    },
    MiniAppMeta {
        id: MiniAppId::PharmacophoreSite,
        short_label: "🎯 MIFs & Farmacóforo",
        window_title: "🎯 Plano del Sitio (MIFs), Farmacóforos y Flexibles",
        what_it_does: "Mapea campos de interacción molecular (7 sondas MIF), calcula volumen y matriz de distancias óptimas del sitio activo, define esferas farmacofóricas suaves y cadenas laterales flexibles.",
        when_to_use: "Cuando quieres guiar el docking hacia interacciones clave (bisagra de quinasa, quelación metálica) o exportar un plano `.pharm.json`.",
        io_summary: "Entrada: receptor + GridBox (Apo o Holo) → Salida: restricciones 3D activas y Site Blueprint (.pharm.json / .md).",
        default_size: [520.0, 600.0],
        default_offset: [155.0, 142.0],
        accent_rgb: [245, 158, 11],
    },
    MiniAppMeta {
        id: MiniAppId::HydeAudit,
        short_label: "🔬 HYDE & PoseBusters",
        window_title: "🔬 SeeSAR HYDE v2, PoseBusters, Torsiones e Interacciones",
        what_it_does: "Descompone la afinidad átomo por átomo (coronas verdes/rojas HYDE v2), verifica validez física PoseBusters (5/5), analiza tensión torsional (semáforo) y enumera interacciones 3D direccionales.",
        when_to_use: "Para detectar por qué un ligando pierde afinidad (desolvatación polar, choques, torsiones tensas) y elegir qué átomo optimizar.",
        io_summary: "Entrada: complejo receptor–ligando → Salida: ΔG consenso, Ki estimada, LE, PoseBusters y diagnóstico por átomo.",
        default_size: [520.0, 640.0],
        default_offset: [210.0, 122.0],
        accent_rgb: [168, 85, 247],
    },
    MiniAppMeta {
        id: MiniAppId::InspiratorAdmet,
        short_label: "🧬 Inspirator & ADMET",
        window_title: "🧬 Inspirator In-Pocket (32 Grupos) y ADMET Rápido",
        what_it_does: "Aplica mutaciones de sustituyentes directamente sobre el átomo seleccionado en 3D con minimización in-pocket instantánea (ΔΔG HYDE), historial Deshacer y vista previa ADMET/QED.",
        when_to_use: "Durante optimización hit-to-lead dentro del bolsillo: prueba halógenos, grupos polares o bioisósteros en segundos antes de enviar a ChemCraft o AdmetCraft.",
        io_summary: "Entrada: átomo seleccionado en el visor 3D → Salida: análogo minimizado, ΔΔG y paso directo a AdmetCraft/ChemCraft.",
        default_size: [500.0, 520.0],
        default_offset: [260.0, 148.0],
        accent_rgb: [236, 72, 153],
    },
    MiniAppMeta {
        id: MiniAppId::CampaignSuite,
        short_label: "📊 Cribado & Suite",
        window_title: "📊 Cribado Virtual (Tabla MPO) y Puentes CraftBio",
        what_it_does: "Muestra el ranking completo de bibliotecas cribadas (Vina, HYDE, Ki, LE, MPO y métricas ROC/BEDROC/EF), permite inspeccionar cada hit con 1 clic y conecta con BiomolCraft, ChemCraft y AdmetCraft.",
        when_to_use: "Al cribar bibliotecas SDF/SMILES/ChemCraft y al transferir los mejores hits al resto del ecosistema CraftBio.",
        io_summary: "Entrada: biblioteca (.sdf/.smi/.csv/.chemcraft) → Salida: tabla rankeada (.csv), poses (.sdf) y envío directo a otras apps.",
        default_size: [720.0, 480.0],
        default_offset: [120.0, 175.0],
        accent_rgb: [14, 165, 233],
    },
];

/// Look up static metadata for a given DockCraft Mini-App.
#[must_use]
pub fn mini_app_meta(id: MiniAppId) -> &'static MiniAppMeta {
    for m in ALL_MINI_APPS {
        if m.id == id {
            return m;
        }
    }
    &ALL_MINI_APPS[0]
}

/// Manages open, expanded, and classic docked panel states in DockCraft.
#[derive(Debug, Clone)]
pub struct WorkspaceWindows {
    pub open_apps: BTreeSet<MiniAppId>,
    pub expanded_apps: BTreeSet<MiniAppId>,
    pub show_classic_side_panels: bool,
}

impl Default for WorkspaceWindows {
    fn default() -> Self {
        let mut open_apps = BTreeSet::new();
        open_apps.insert(MiniAppId::Tutorials);
        Self {
            open_apps,
            expanded_apps: BTreeSet::new(),
            show_classic_side_panels: false,
        }
    }
}

impl WorkspaceWindows {
    pub fn open(&mut self, id: MiniAppId) {
        self.open_apps.insert(id);
    }

    pub fn close(&mut self, id: MiniAppId) {
        self.open_apps.remove(&id);
        self.expanded_apps.remove(&id);
    }

    pub fn toggle(&mut self, id: MiniAppId) {
        if self.open_apps.contains(&id) {
            self.close(id);
        } else {
            self.open(id);
        }
    }

    pub fn toggle_expand(&mut self, id: MiniAppId) {
        if self.expanded_apps.contains(&id) {
            self.expanded_apps.remove(&id);
        } else {
            self.expanded_apps.insert(id);
        }
    }
}

/// Paint a crisp, plain-Spanish educational header inside every floating mini-application.
pub fn render_mini_app_header(
    ui: &mut egui::Ui,
    meta: &MiniAppMeta,
    is_expanded: bool,
) -> bool {
    let mut toggle_expand_clicked = false;
    let [r, g, b] = meta.accent_rgb;
    let accent = egui::Color32::from_rgb(r, g, b);
    let bg = egui::Color32::from_rgba_unmultiplied(r, g, b, 24);

    egui::Frame::group(ui.style())
        .fill(bg)
        .stroke(egui::Stroke::new(1.2_f32, accent))
        .corner_radius(egui::CornerRadius::same(8))
        .inner_margin(egui::Margin::symmetric(10, 7))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.colored_label(
                    accent,
                    egui::RichText::new(meta.short_label).strong().size(13.5),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let lbl = if is_expanded {
                        "🗗 Tamaño normal"
                    } else {
                        "🗖 Ampliar ventana"
                    };
                    if ui
                        .small_button(lbl)
                        .on_hover_text(
                            "Alterna entre tamaño normal y ventana ampliada (también puedes arrastrar y estirar cualquier borde)",
                        )
                        .clicked()
                    {
                        toggle_expand_clicked = true;
                    }
                });
            });
            ui.label(egui::RichText::new(meta.what_it_does).size(12.0));
            ui.small(format!("Cuándo usarla: {}", meta.when_to_use));
            ui.small(format!("Flujo I/O: {}", meta.io_summary));
        });
    ui.add_space(4.0);
    toggle_expand_clicked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_windows_tracks_open_expand_and_classic_mode() {
        let mut ws = WorkspaceWindows::default();
        assert!(ws.open_apps.contains(&MiniAppId::Tutorials));
        assert!(!ws.show_classic_side_panels);
        ws.open(MiniAppId::HydeAudit);
        assert!(ws.open_apps.contains(&MiniAppId::HydeAudit));
        ws.toggle_expand(MiniAppId::HydeAudit);
        assert!(ws.expanded_apps.contains(&MiniAppId::HydeAudit));
        ws.toggle(MiniAppId::HydeAudit);
        assert!(!ws.open_apps.contains(&MiniAppId::HydeAudit));
        assert!(!ws.expanded_apps.contains(&MiniAppId::HydeAudit));
    }

    #[test]
    fn all_mini_apps_have_distinct_ids_and_spanish_docs() {
        let mut ids = BTreeSet::new();
        for m in ALL_MINI_APPS {
            assert!(ids.insert(m.id), "duplicate mini-app id {:?}", m.id);
            assert!(!m.what_it_does.is_empty());
            assert!(!m.when_to_use.is_empty());
            assert!(!m.io_summary.is_empty());
        }
        assert_eq!(ids.len(), 7);
    }
}
