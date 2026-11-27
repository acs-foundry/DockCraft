//! Interactive In-App Guided Tutorials for DockCraft.
//!
//! Walks the user step-by-step through structure-based drug design inside DockCraft's live 3D
//! viewport: pocket detection & GridBox sizing, receptor preparation (ProtPrep pKa & H-bond
//! flips), 7-probe MIF active-site blueprints, SeeSAR HYDE v2 & PoseBusters auditing, 32-group
//! in-pocket lead optimization, and virtual screening handoffs.

use std::collections::BTreeSet;

use crate::miniapps::MiniAppId;

/// Live action triggered when the user clicks "▶ Ejecutar este paso ahora" in a tutorial step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TutorialAction {
    DetectPocketsAndShowBox,
    RunLiveDocking,
    InspectHydeAndTorsions,
    RunProtPrepAudit,
    ScanBridgingWaters,
    AnalyzeMifBlueprint,
    ApplyBlueprintRestraints,
    RunInPocketMutationDemo,
    PreviewAdmetAndLinks,
    RunScreeningCampaignDemo,
    OpenMiniApp(MiniAppId),
}

/// One guided step inside a DockCraft interactive tutorial track.
#[derive(Debug, Clone)]
pub struct TutorialStep {
    pub title: &'static str,
    pub objective: &'static str,
    pub what_to_look_at: &'static str,
    pub action_label: &'static str,
    pub action: TutorialAction,
    pub related_app: Option<MiniAppId>,
}

/// Complete multi-step interactive tutorial track.
#[derive(Debug, Clone)]
pub struct TutorialTrack {
    pub id: &'static str,
    pub title: &'static str,
    pub duration_min: u8,
    pub summary: &'static str,
    pub steps: Vec<TutorialStep>,
}

/// Runtime state of DockCraft's Interactive Tutorials mini-app.
#[derive(Debug, Clone)]
pub struct TutorialState {
    pub active_track_idx: usize,
    pub active_step_idx: usize,
    pub completed_steps: BTreeSet<(usize, usize)>,
    pub last_feedback: String,
}

impl Default for TutorialState {
    fn default() -> Self {
        Self {
            active_track_idx: 0,
            active_step_idx: 0,
            completed_steps: BTreeSet::new(),
            last_feedback: "Elige un tutorial y pulsa '▶ Ejecutar este paso ahora' para practicar sobre el complejo 3D en vivo.".to_string(),
        }
    }
}

impl TutorialState {
    pub fn mark_completed(&mut self, track_idx: usize, step_idx: usize, feedback: String) {
        self.completed_steps.insert((track_idx, step_idx));
        self.last_feedback = feedback;
    }

    #[must_use]
    pub fn completed_in_track(&self, track_idx: usize, total_steps: usize) -> usize {
        (0..total_steps)
            .filter(|&s| self.completed_steps.contains(&(track_idx, s)))
            .count()
    }
}

/// Build the 5 curated interactive tutorials covering DockCraft's complete SBDD workflow.
#[must_use]
pub fn built_in_tutorials() -> Vec<TutorialTrack> {
    vec![
        TutorialTrack {
            id: "docking_gridbox",
            title: "1. Bolsillos 3D, Caja GridBox y Docking Consenso",
            duration_min: 3,
            summary: "Localiza cavidades druggables, visualiza y ajusta la Caja 3D de búsqueda (GridBox), ejecuta docking Vina + consenso HYDE v2 e inspecciona poses.",
            steps: vec![
                TutorialStep {
                    title: "Detectar bolsillos y encuadrar la Caja 3D (GridBox)",
                    objective: "Ejecuta el detector geométrico de 26 rayos sobre el receptor activo, enciende la Caja de Docking 3D y abre la mini-app flotante 'Bolsillos & Docking'.",
                    what_to_look_at: "En el visor 3D verás la caja de selección envolviendo el bolsillo #1. Puedes arrastrar la mini-app flotante y editar CX/CY/CZ y SX/SY/SZ en vivo.",
                    action_label: "▶ Detectar bolsillos y mostrar Caja 3D",
                    action: TutorialAction::DetectPocketsAndShowBox,
                    related_app: Some(MiniAppId::DockingGridBox),
                },
                TutorialStep {
                    title: "Lanzar docking Vina + re-ranking consenso HYDE v2",
                    objective: "Acopla el ligando activo dentro de la Caja 3D usando búsqueda Monte Carlo + BFGS y reordena las poses con afinidad consenso HYDE.",
                    what_to_look_at: "Revisa la lista 'Docked Modes' en la mini-app: haz clic entre Mode 1, Mode 2... para comparar RMSD y energía (kcal/mol) en el 3D.",
                    action_label: "▶ Ejecutar Docking ahora (Ctrl+D)",
                    action: TutorialAction::RunLiveDocking,
                    related_app: Some(MiniAppId::DockingGridBox),
                },
                TutorialStep {
                    title: "Auditar validez física PoseBusters, coronas HYDE y torsiones",
                    objective: "Abre la mini-app 'HYDE & PoseBusters' y activa las coronas átomo por átomo y el semáforo de torsiones en el visor 3D.",
                    what_to_look_at: "Las esferas verdes indican átomos que ganan afinidad; las rojas delatan desolvatación o choques. Comprueba el sello 'PASS PoseBusters (5/5)'.",
                    action_label: "▶ Activar Coronas HYDE + Semáforo Torsiones",
                    action: TutorialAction::InspectHydeAndTorsions,
                    related_app: Some(MiniAppId::HydeAudit),
                },
            ],
        },
        TutorialTrack {
            id: "protprep_waters",
            title: "2. Preparación de Receptor (ProtPrep pKa) y Aguas Puente",
            duration_min: 3,
            summary: "Prepara receptores cristalográficos/AlphaFold a pH fisiológico, optimiza flips de Asn/Gln/His y decide qué aguas estructurales retener.",
            steps: vec![
                TutorialStep {
                    title: "Auditar estructura y protonar a pH 7.4 (ProtPrep)",
                    objective: "Ejecuta la preparación automática: grado de calidad, carga neta, resolución de altlocs, flips de puentes de H y optimización de rotores polares.",
                    what_to_look_at: "En la mini-app 'ProtPrep & Aguas' verás el grado (A/B/C), el resumen de incidencias y el botón para exportar la auditoría en Markdown.",
                    action_label: "▶ Ejecutar ProtPrep y ver auditoría",
                    action: TutorialAction::RunProtPrepAudit,
                    related_app: Some(MiniAppId::ReceptorPrepWaters),
                },
                TutorialStep {
                    title: "Escanear aguas puente estructurales y metales (AutoDock4Zn)",
                    objective: "Detecta moléculas de agua dentro de la GridBox que forman ≥ 2 puentes de hidrógeno con el receptor y activa la coordinación metálica.",
                    what_to_look_at: "Marca o desmarca cualquier agua puente detectada para evaluar su efecto inmediato sobre la afinidad HYDE.",
                    action_label: "▶ Escanear aguas puente en la GridBox",
                    action: TutorialAction::ScanBridgingWaters,
                    related_app: Some(MiniAppId::ReceptorPrepWaters),
                },
            ],
        },
        TutorialTrack {
            id: "mif_pharmacophore",
            title: "3. Plano del Sitio Activo (MIFs) y Restricciones Farmacofóricas",
            duration_min: 4,
            summary: "Calcula campos de interacción molecular (7 sondas), descubre la geometría del bolsillo y convierte hotspots en restricciones 3D para docking guiado.",
            steps: vec![
                TutorialStep {
                    title: "Mapear el sitio activo con 7 sondas MIF (Apo/Holo)",
                    objective: "Calcula volumen accesible (Å³), proporción hidrofóbica/polar, peso molecular óptimo y hotspots farmacofóricos con matriz de distancias.",
                    what_to_look_at: "En el visor 3D aparecerán las esferas y vectores direccionales del farmacóforo; en la mini-app podrás activar/desactivar hotspots individuales.",
                    action_label: "▶ Analizar Sitio Activo (MIFs)",
                    action: TutorialAction::AnalyzeMifBlueprint,
                    related_app: Some(MiniAppId::PharmacophoreSite),
                },
                TutorialStep {
                    title: "Convertir el Site Blueprint en restricciones de docking",
                    objective: "Transfiere los hotspots habilitados como esferas farmacofóricas suaves que guían al motor de docking hacia las interacciones críticas.",
                    what_to_look_at: "Comprueba el contador de restricciones activas en la mini-app y observa cómo se sincronizan en el 3D (puedes exportar `.pharm.json`).",
                    action_label: "▶ Aplicar Blueprint → Restricciones 3D",
                    action: TutorialAction::ApplyBlueprintRestraints,
                    related_app: Some(MiniAppId::PharmacophoreSite),
                },
            ],
        },
        TutorialTrack {
            id: "inspirator_sar",
            title: "4. Optimización In-Pocket (Inspirator 32 Grupos) y ADMET",
            duration_min: 4,
            summary: "Crece o sustituye grupos funcionales directamente dentro del bolsillo 3D estilo SeeSAR con minimización instantánea y chequeo ADMET.",
            steps: vec![
                TutorialStep {
                    title: "Aplicar una mutación in-pocket con minimización instantánea",
                    objective: "Selecciona un átomo del ligando y prueba un sustituyente real (p. ej. Flúor) midiendo al instante el cambio de energía ΔΔG en kcal/mol.",
                    what_to_look_at: "Observa en la mini-app 'Inspirator & ADMET' el ΔΔG resultante (verde = mejora) y usa 'Undo (Ctrl+Z)' cuando quieras revertir el cambio.",
                    action_label: "▶ Mutar átomo activo → -F y medir ΔΔG",
                    action: TutorialAction::RunInPocketMutationDemo,
                    related_app: Some(MiniAppId::InspiratorAdmet),
                },
                TutorialStep {
                    title: "Evaluar perfil ADMET rápido y enlazar con AdmetCraft/ChemCraft",
                    objective: "Calcula MW, cLogP, TPSA, QED y reglas de Lipinski sobre el ligando modificado antes de pasarlo al panel completo de toxicología/PK.",
                    what_to_look_at: "Verás el resumen ADMET bajo la cabecera del Inspirator junto a los botones directos para abrir el candidato en AdmetCraft o ChemCraft.",
                    action_label: "▶ Calcular vista previa ADMET",
                    action: TutorialAction::PreviewAdmetAndLinks,
                    related_app: Some(MiniAppId::InspiratorAdmet),
                },
            ],
        },
        TutorialTrack {
            id: "campaign_handoff",
            title: "5. Cribado Virtual (Tabla MPO) y Ecosistema CraftBio",
            duration_min: 3,
            summary: "Lanza una campaña de cribado multiparamétrico (Vina + HYDE + Ki + LE + MPO) y envía resultados a BiomolCraft ViewDockX, ChemCraft y AdmetCraft.",
            steps: vec![
                TutorialStep {
                    title: "Ejecutar campaña de cribado y abrir la Tabla MPO ampliable",
                    objective: "Ejecuta el pipeline de cribado y muestra la tabla rankeada en la mini-app flotante 'Cribado & Suite' (puedes pulsar '🗖 Ampliar ventana' para leerla a pantalla ancha).",
                    what_to_look_at: "Haz clic en cualquier fila (#1, #2...) para cargar su pose en el 3D y consulta métricas ROC/BEDROC/EF cuando incluyes activos/señuelos.",
                    action_label: "▶ Ejecutar campaña demo y abrir Tabla MPO",
                    action: TutorialAction::RunScreeningCampaignDemo,
                    related_app: Some(MiniAppId::CampaignSuite),
                },
                TutorialStep {
                    title: "Revisar puentes a BiomolCraft (ViewDockX), ChemCraft y AdmetCraft",
                    objective: "Conoce las salidas directas desde la mini-app de Suite: publicación 3D en BiomolCraft, SAR 2D/3D en ChemCraft y ADMET/PBPK en AdmetCraft.",
                    what_to_look_at: "En la zona inferior de 'Cribado & Suite' dispones de botones de 1 clic para enviar poses o tomar una nueva caja desde BiomolCraft.",
                    action_label: "▶ Abrir Mini-App Cribado & Puentes CraftBio",
                    action: TutorialAction::OpenMiniApp(MiniAppId::CampaignSuite),
                    related_app: Some(MiniAppId::CampaignSuite),
                },
            ],
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_tutorials_have_five_complete_tracks() {
        let tracks = built_in_tutorials();
        assert_eq!(tracks.len(), 5);
        for t in &tracks {
            assert!(!t.id.is_empty());
            assert!(!t.steps.is_empty(), "track {} has no steps", t.id);
        }
        let mut st = TutorialState::default();
        st.mark_completed(0, 0, "hecho".to_string());
        assert_eq!(st.completed_in_track(0, tracks[0].steps.len()), 1);
    }
}
