use crate::classification::{ClassificationSummary, ExtremumKind};
use crate::csv_analyzer::{self, DataPoint, IdentificationResult};
use crate::gpu::pipeline::ExtremaClassifierPipeline;
use crate::gpu::GpuContext;
use crate::model::TrigFunction;
use crate::patterns::{self, ExtremaPatternReport};
use crate::pdf_exporter;
use eframe::egui::{self, Color32, RichText, Stroke};
use egui_plot::{Line, MarkerShape, Plot, Points};
use std::f64::consts::TAU;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveTab {
    Classifier,
    CsvAnalysis,
    Help,
}

/// Fargepalett designet spesifikt for universell utforming og høy kontrast i mørk modus
pub struct ThemeColors {
    pub bg_app: Color32,
    pub bg_card: Color32,
    pub bg_card_alt: Color32,
    pub border_stroke: Stroke,
    pub border_accent: Stroke,
    pub text_primary: Color32,
    pub text_secondary: Color32,
    pub text_accent: Color32,
    pub text_warning: Color32,
    pub text_error: Color32,
    pub text_success: Color32,
    pub peak_color: Color32,
    pub trough_color: Color32,
    pub threshold_colors: [Color32; 4],
    pub table_stripe: Color32,
}

impl ThemeColors {
    pub fn dark() -> Self {
        Self {
            bg_app: Color32::from_rgb(15, 23, 42),
            bg_card: Color32::from_rgb(30, 41, 59),
            bg_card_alt: Color32::from_rgb(24, 33, 47),
            border_stroke: Stroke::new(1.2, Color32::from_rgb(71, 85, 105)),
            border_accent: Stroke::new(1.5, Color32::from_rgb(59, 130, 246)),
            text_primary: Color32::from_rgb(248, 250, 252),
            text_secondary: Color32::from_rgb(226, 232, 240),
            text_accent: Color32::from_rgb(56, 189, 248),
            text_warning: Color32::from_rgb(250, 204, 21),
            text_error: Color32::from_rgb(248, 113, 113),
            text_success: Color32::from_rgb(52, 211, 153),
            peak_color: Color32::from_rgb(34, 197, 94),
            trough_color: Color32::from_rgb(239, 68, 68),
            threshold_colors: [
                Color32::from_rgba_premultiplied(148, 163, 184, 180),
                Color32::from_rgba_premultiplied(250, 204, 21, 180),
                Color32::from_rgba_premultiplied(251, 146, 60, 180),
                Color32::from_rgba_premultiplied(248, 113, 113, 180),
            ],
            table_stripe: Color32::from_rgb(22, 30, 44),
        }
    }
}

pub struct ExtremaApp {
    tab: ActiveTab,
    enable_plot_mouse_zoom: bool, // Når false: mus-hjul scroller siden jevnt!

    gpu_ctx: Option<GpuContext>,
    pipeline: Option<ExtremaClassifierPipeline>,
    gpu_adapter_name: String,
    gpu_backend_name: String,

    // Fane 1: Klassifisering
    pub formula_input: String,
    pub x_min: f64,
    pub x_max: f64,
    pub num_samples: u32,
    pub thresholds: [f32; 4],
    pub max_extrema: u32,

    pub current_func: Option<TrigFunction>,
    pub summary: Option<ClassificationSummary>,
    pub patterns: Option<ExtremaPatternReport>,
    pub gpu_time_us: u128,

    pub error_msg: Option<String>,
    pub status_msg: Option<String>,
    pub last_pdf_path: Option<PathBuf>,
    pub last_csv_path: Option<PathBuf>,

    // Visningsvalg for graf
    pub show_peaks: bool,
    pub show_troughs: bool,
    pub show_thresholds: bool,

    // Fane 2: CSV
    pub csv_file_path: Option<PathBuf>,
    pub csv_data: Vec<DataPoint>,
    pub csv_max_n: u32,
    pub csv_terms_per_line: usize,
    pub csv_result: Option<IdentificationResult>,
    pub csv_status: Option<String>,
    pub csv_error: Option<String>,

    // Syntetisk datasett-generering for CSV
    pub gen_formula: String,
    pub gen_points: usize,
    pub gen_noise: f64,
}

impl Default for ExtremaApp {
    fn default() -> Self {
        let (gpu_ctx, pipeline, gpu_adapter_name, gpu_backend_name) = match pollster::block_on(GpuContext::init()) {
            Ok(ctx) => {
                let adapter_name = ctx.adapter_info.name.clone();
                let backend_name = format!("{:?}", ctx.adapter_info.backend);
                let pipe = ExtremaClassifierPipeline::new(&ctx);
                (Some(ctx), Some(pipe), adapter_name, backend_name)
            }
            Err(_) => (
                None,
                None,
                "GPU ikke tilgjengelig (bruker CPU fallback)".to_string(),
                "CPU".to_string(),
            ),
        };

        let mut app = Self {
            tab: ActiveTab::Classifier,
            enable_plot_mouse_zoom: false,

            gpu_ctx,
            pipeline,
            gpu_adapter_name,
            gpu_backend_name,

            formula_input: "[[1, sin(x), 3]] + [[2, cos(x), 4]]".to_string(),
            x_min: 0.0,
            x_max: TAU,
            num_samples: 65_536,
            thresholds: [0.2, 0.5, 0.8, 1.2],
            max_extrema: 256,

            current_func: None,
            summary: None,
            patterns: None,
            gpu_time_us: 0,

            error_msg: None,
            status_msg: None,
            last_pdf_path: None,
            last_csv_path: None,

            show_peaks: true,
            show_troughs: true,
            show_thresholds: false,

            csv_file_path: None,
            csv_data: Vec::new(),
            csv_max_n: 128,
            csv_terms_per_line: 4,
            csv_result: None,
            csv_status: None,
            csv_error: None,

            gen_formula: "sin(x) + cos(2x)".to_string(),
            gen_points: 2000,
            gen_noise: 0.0,
        };

        app.run_gpu_analysis();
        app
    }
}

impl ExtremaApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self::default()
    }

    /// Kjører GPU-ekstremalanalyse for den gjeldende formelstrengen
    pub fn run_gpu_analysis(&mut self) {
        self.error_msg = None;
        self.status_msg = None;

        let parsed_func = match TrigFunction::parse(&self.formula_input) {
            Ok(f) => f,
            Err(e) => {
                self.error_msg = Some(format!("Ugyldig formel: {}", e));
                return;
            }
        };

        let ctx = match self.gpu_ctx.as_ref() {
            Some(c) => c,
            None => {
                self.error_msg = Some("GPU-kontekst er ikke tilgjengelig".to_string());
                return;
            }
        };

        let pipeline = match self.pipeline.as_ref() {
            Some(p) => p,
            None => {
                self.error_msg = Some("GPU-pipeline er ikke tilgjengelig".to_string());
                return;
            }
        };

        let t0 = Instant::now();
        match pipeline.run(
            ctx,
            &parsed_func,
            self.x_min as f32,
            self.x_max as f32,
            self.num_samples,
            self.thresholds,
        ) {
            Ok(summary) => {
                self.gpu_time_us = t0.elapsed().as_micros();
                let pattern_report = patterns::analyze_extrema_patterns(&summary.extrema, 0.05);

                self.status_msg = None;
                self.error_msg = None;
                self.summary = Some(summary);
                self.patterns = Some(pattern_report);
                self.current_func = Some(parsed_func);
            }
            Err(e) => {
                self.error_msg = Some(format!("GPU-beregning feilet: {}", e));
            }
        }
    }

    /// Zoom-funksjoner for grafen som oppdaterer GPU-beregningen direkte
    pub fn zoom_plot_in(&mut self) {
        let mid = (self.x_min + self.x_max) / 2.0;
        let span = (self.x_max - self.x_min) * 0.75;
        self.x_min = mid - span / 2.0;
        self.x_max = mid + span / 2.0;
        self.run_gpu_analysis();
    }

    pub fn zoom_plot_out(&mut self) {
        let mid = (self.x_min + self.x_max) / 2.0;
        let span = (self.x_max - self.x_min) * 1.3333;
        self.x_min = mid - span / 2.0;
        self.x_max = mid + span / 2.0;
        self.run_gpu_analysis();
    }

    pub fn reset_plot_range(&mut self) {
        self.x_min = 0.0;
        self.x_max = TAU;
        self.run_gpu_analysis();
    }

    /// Eksporterer gjeldende analyse til en PDF-rapport
    pub fn export_pdf(&mut self) {
        let func = match self.current_func.as_ref() {
            Some(f) => f,
            None => {
                self.error_msg = Some("Ingen analysert funksjon å eksportere".to_string());
                return;
            }
        };

        let summary = match self.summary.as_ref() {
            Some(s) => s,
            None => {
                self.error_msg = Some("Ingen analyseresultater å eksportere".to_string());
                return;
            }
        };

        let pattern_report = match self.patterns.as_ref() {
            Some(p) => p,
            None => return,
        };

        let target_path = if let Some(path) = rfd::FileDialog::new()
            .set_file_name("ekstrema_analyse.pdf")
            .add_filter("PDF-dokument", &["pdf"])
            .save_file()
        {
            path
        } else {
            PathBuf::from("ekstrema_analyse.pdf")
        };

        let gpu_info = (self.gpu_adapter_name.as_str(), self.gpu_backend_name.as_str());
        match pdf_exporter::export_to_pdf(
            &target_path,
            "Trigonometrisk Ekstremanalyse & Rekonstruksjon - Rapport",
            func,
            gpu_info,
            summary,
            pattern_report,
            None,
            None,
        ) {
            Ok(pdf_path) => {
                self.status_msg = Some(format!("PDF lagret til: {}", pdf_path.display()));
                self.last_pdf_path = Some(pdf_path.clone());
                let _ = Command::new("explorer.exe").arg(&pdf_path).spawn();
            }
            Err(e) => {
                self.error_msg = Some(format!("Kunne ikke generere PDF: {}", e));
            }
        }
    }

    /// Eksporterer detekterte ekstrema og analyse til en ren .csv-fil
    pub fn export_csv(&mut self) {
        let func = match self.current_func.as_ref() {
            Some(f) => f,
            None => {
                self.error_msg = Some("Ingen analysert funksjon å eksportere".to_string());
                return;
            }
        };

        let summary = match self.summary.as_ref() {
            Some(s) => s,
            None => {
                self.error_msg = Some("Ingen analyseresultater å eksportere".to_string());
                return;
            }
        };

        let patterns = match self.patterns.as_ref() {
            Some(p) => p,
            None => return,
        };

        let target_path = if let Some(path) = rfd::FileDialog::new()
            .set_file_name("ekstrema_analyse.csv")
            .add_filter("CSV-regneark", &["csv"])
            .save_file()
        {
            path
        } else {
            PathBuf::from("ekstrema_analyse.csv")
        };

        match export_extrema_to_csv(&target_path, func, summary, patterns) {
            Ok(()) => {
                self.status_msg = Some(format!("CSV-fil lagret til: {}", target_path.display()));
                self.last_csv_path = Some(target_path.clone());
                let _ = Command::new("explorer.exe").arg(&target_path).spawn();
            }
            Err(e) => {
                self.error_msg = Some(format!("Kunne ikke lagre CSV: {}", e));
            }
        }
    }

    /// Genererer syntetisk (x, y) datasett fra en gitt formel og lagrer til CSV for senere analyse
    pub fn export_dataset_csv(&mut self, formula: &str, points: usize, noise: f64) {
        let func = match TrigFunction::parse(formula) {
            Ok(f) => f,
            Err(e) => {
                self.error_msg = Some(format!("Ugyldig formel for datasett: {}", e));
                self.csv_error = Some(format!("Ugyldig formel: {}", e));
                return;
            }
        };

        let target_path = if let Some(path) = rfd::FileDialog::new()
            .set_file_name("datasett.csv")
            .add_filter("CSV-datasett", &["csv"])
            .save_file()
        {
            path
        } else {
            PathBuf::from("datasett.csv")
        };

        match csv_analyzer::generate_test_csv(&target_path, &func, self.x_min, self.x_max, points, noise) {
            Ok(()) => {
                self.status_msg = Some(format!("Datasett lagret til: {}", target_path.display()));
                self.csv_status = Some(format!(
                    "Genererte {} punkter og lagret til: {}",
                    points,
                    target_path.file_name().unwrap_or_default().to_string_lossy()
                ));
                self.last_csv_path = Some(target_path.clone());
                let _ = Command::new("explorer.exe").arg(&target_path).spawn();
            }
            Err(e) => {
                self.error_msg = Some(format!("Kunne ikke lagre datasett: {}", e));
                self.csv_error = Some(format!("Kunne ikke lagre datasett: {}", e));
            }
        }
    }

    /// Genererer syntetiske datapunkter og laster dem direkte inn i analysatoren
    pub fn generate_and_load_dataset(&mut self) {
        self.csv_error = None;
        self.csv_status = None;

        let func = match TrigFunction::parse(&self.gen_formula) {
            Ok(f) => f,
            Err(e) => {
                self.csv_error = Some(format!("Ugyldig formel: {}", e));
                return;
            }
        };

        let pts = csv_analyzer::generate_synthetic_data(&func, self.x_min, self.x_max, self.gen_points, self.gen_noise);
        self.csv_status = Some(format!(
            "Genererte {} syntetiske datapunkter for '{}' (støy: {:.2})",
            pts.len(),
            self.gen_formula,
            self.gen_noise
        ));
        self.csv_file_path = None;
        self.csv_data = pts;
        self.csv_result = None;

        self.analyze_loaded_csv();
    }

    /// Åpner generert PDF-fil i standardleser
    pub fn open_last_pdf(&self) {
        if let Some(path) = &self.last_pdf_path {
            let _ = Command::new("explorer.exe").arg(path).spawn();
        }
    }

    /// Åpner generert CSV-fil i standard regnearkleser (f.eks. Excel)
    pub fn open_last_csv(&self) {
        if let Some(path) = &self.last_csv_path {
            let _ = Command::new("explorer.exe").arg(path).spawn();
        }
    }

    /// Velger og laster CSV-fil
    pub fn pick_and_load_csv(&mut self) {
        self.csv_error = None;
        self.csv_status = None;

        if let Some(path) = rfd::FileDialog::new()
            .add_filter("CSV-filer", &["csv", "txt"])
            .pick_file()
        {
            match csv_analyzer::read_csv_data(&path) {
                Ok(data) => {
                    self.csv_status = Some(format!(
                        "Lastet {} datapunkter fra: {}",
                        data.len(),
                        path.file_name().unwrap_or_default().to_string_lossy()
                    ));
                    self.csv_file_path = Some(path);
                    self.csv_data = data;
                    self.analyze_loaded_csv();
                }
                Err(e) => {
                    self.csv_error = Some(format!("Kunne ikke lese CSV: {}", e));
                }
            }
        }
    }

    /// Kjører formel-identifikasjon på lastet CSV-data
    pub fn analyze_loaded_csv(&mut self) {
        self.csv_error = None;
        if self.csv_data.is_empty() {
            self.csv_error = Some("Ingen CSV-data er lastet inn".to_string());
            return;
        }

        match csv_analyzer::identify_function_from_data(&self.csv_data, self.csv_max_n) {
            Ok(res) => {
                self.csv_status = Some(format!(
                    "Identifiserte funksjon med R² = {:.4} ({:.1}%)",
                    res.best_match.r_squared,
                    res.best_match.r_squared * 100.0
                ));
                self.csv_result = Some(res);
            }
            Err(e) => {
                self.csv_error = Some(format!("Analyse feilet: {}", e));
            }
        }
    }
}

impl eframe::App for ExtremaApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Fast standard zoomfaktor 1.0 (ingen endring av tekststørrelse)
        ui.ctx().set_zoom_factor(1.0);

        let theme = ThemeColors::dark();
        let mut visuals = egui::Visuals::dark();

        visuals.override_text_color = Some(theme.text_primary);
        visuals.panel_fill = theme.bg_app;
        visuals.window_fill = theme.bg_app;
        visuals.extreme_bg_color = theme.bg_app;

        visuals.widgets.noninteractive.bg_fill = theme.bg_card;
        visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, theme.text_primary);
        visuals.widgets.noninteractive.bg_stroke = theme.border_stroke;

        visuals.widgets.inactive.bg_fill = theme.bg_card;
        visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, theme.text_primary);
        visuals.widgets.inactive.bg_stroke = theme.border_stroke;

        visuals.widgets.hovered.bg_fill = theme.bg_card_alt;
        visuals.widgets.hovered.fg_stroke = Stroke::new(1.5, theme.text_accent);
        visuals.widgets.hovered.bg_stroke = theme.border_accent;

        visuals.widgets.active.bg_fill = theme.bg_card_alt;
        visuals.widgets.active.fg_stroke = Stroke::new(2.0, theme.text_accent);
        visuals.widgets.active.bg_stroke = theme.border_accent;

        ui.ctx().set_visuals(visuals);
        ui.spacing_mut().item_spacing = egui::vec2(6.0, 4.0);

        // --- 1. TOPP-PANEL: SLANK HEADER & NAVIGASJON ---
        ui.horizontal(|ui| {
            ui.heading(
                RichText::new("⚡ Trigonometrisk Ekstremanalyse & Rekonstruksjon")
                    .color(theme.text_accent)
                    .strong()
                    .size(16.5),
            );

            ui.separator();

            let tab_classifier = ui.selectable_label(
                self.tab == ActiveTab::Classifier,
                RichText::new("📈 Funksjonsanalyse & Graf").strong().size(12.5),
            );
            if tab_classifier.clicked() {
                self.tab = ActiveTab::Classifier;
            }

            let tab_csv = ui.selectable_label(
                self.tab == ActiveTab::CsvAnalysis,
                RichText::new("📂 CSV-Datasett & Formelgjenkjenning").strong().size(12.5),
            );
            if tab_csv.clicked() {
                self.tab = ActiveTab::CsvAnalysis;
            }

            let tab_help = ui.selectable_label(
                self.tab == ActiveTab::Help,
                RichText::new("❓ Hjelp").strong().size(12.5),
            );
            if tab_help.clicked() {
                self.tab = ActiveTab::Help;
            }

            // Minimalistisk topplinje: kun GPU-info til høyre
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    RichText::new(format!("GPU: {} ({})", self.gpu_adapter_name, self.gpu_backend_name))
                        .color(theme.text_secondary)
                        .monospace()
                        .size(10.5),
                );
            });
        });

        ui.horizontal(|ui| {
            ui.label(
                RichText::new("ℹ MERK: KUN sinus (sin) og cosinus (cos) støttes!")
                    .color(theme.text_accent)
                    .strong()
                    .size(11.5),
            );
            ui.label(
                RichText::new("• Støtter konstanter (f.eks. 2*sin(x), 0.5*cos(2x)) og intervall [[start, func, slutt, steg?]] der func MÅ være et sin- eller cos-uttrykk (f.eks. [[4, 3*cos(x), 100]] eller [[1, sin(x), 10, 2]]).")
                    .color(theme.text_secondary)
                    .size(11.0),
            );
        });

        ui.separator();

        // --- 2. HOVEDINNHOLD - RENDER UTEN FORCERET SCROLLBAR SLIK AT MAN SLIPPER Å SCROLLE ---
        match self.tab {
            ActiveTab::Classifier => self.render_classifier_tab(ui, &theme),
            ActiveTab::CsvAnalysis => self.render_csv_tab(ui, &theme),
            ActiveTab::Help => {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.render_help_tab(ui, &theme);
                });
            }
        }
    }
}

impl ExtremaApp {
    fn render_classifier_tab(&mut self, ui: &mut egui::Ui, theme: &ThemeColors) {
        // --- KONTROLLPANEL FOR INNDATA ---
        egui::Frame::new()
            .fill(theme.bg_card)
            .stroke(theme.border_stroke)
            .inner_margin(8)
            .corner_radius(6)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("f(x) = ").strong().size(15.0).color(theme.text_primary));
                    let input_response = ui.add(
                        egui::TextEdit::singleline(&mut self.formula_input)
                            .desired_width(f32::INFINITY)
                            .font(egui::TextStyle::Monospace),
                    );

                    if input_response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        self.run_gpu_analysis();
                    }

                    if ui
                        .button(
                            RichText::new("⚡ Kjør analyse (Enter)")
                                .strong()
                                .color(theme.text_primary),
                        )
                        .on_hover_text("Kjør GPU-evaluering over intervallet")
                        .clicked()
                    {
                        self.run_gpu_analysis();
                    }
                });

                if let Some(err) = &self.error_msg {
                    ui.add_space(2.0);
                    ui.label(RichText::new(format!("❌ {}", err)).color(theme.text_error).strong().size(11.5));
                }

                ui.add_space(3.0);
                ui.horizontal(|ui| {
                    if ui
                        .button(RichText::new("💾 Lagre CSV (x, y)").color(theme.text_success))
                        .on_hover_text("Generer et komplett datasett med kun x og y verdier til CSV (kan analyseres i CSV-fanen)")
                        .clicked()
                    {
                        self.export_dataset_csv(&self.formula_input.clone(), 2000, 0.0);
                    }

                    if ui
                        .button(RichText::new("📄 Lagre PDF").color(theme.text_accent))
                        .on_hover_text("Generer og lagre fullstendig analyserapport som PDF (åpnes automatisk)")
                        .clicked()
                    {
                        self.export_pdf();
                    }
                });

                ui.add_space(3.0);
                ui.separator();
                ui.add_space(2.0);

                // Parametere (Intervall, samplingspunkter, terskler)
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Intervall:").size(11.5).strong());
                    ui.add(egui::DragValue::new(&mut self.x_min).speed(0.1));
                    ui.label("til");
                    ui.add(egui::DragValue::new(&mut self.x_max).speed(0.1));
                    if ui.button(RichText::new("Sett [0, 2π]").size(11.0)).clicked() {
                        self.reset_plot_range();
                    }

                    ui.separator();
                    ui.label(RichText::new("Oppløsning:").size(11.5).strong());
                    egui::ComboBox::from_id_salt("samples_combo")
                        .selected_text(format!("{} punkter", self.num_samples))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.num_samples, 16_384, "16 384 punkter");
                            ui.selectable_value(&mut self.num_samples, 65_536, "65 536 punkter (Standard)");
                            ui.selectable_value(&mut self.num_samples, 131_072, "131 072 punkter");
                            ui.selectable_value(&mut self.num_samples, 262_144, "262 144 punkter");
                            ui.selectable_value(&mut self.num_samples, 1_048_576, "1 048 576 punkter (Ultra)");
                        });

                    ui.separator();
                    ui.label(RichText::new("Terskler (T1..T4):").size(11.5).strong());
                    ui.add(egui::DragValue::new(&mut self.thresholds[0]).speed(0.05).range(0.01..=5.0));
                    ui.add(egui::DragValue::new(&mut self.thresholds[1]).speed(0.05).range(0.01..=5.0));
                    ui.add(egui::DragValue::new(&mut self.thresholds[2]).speed(0.05).range(0.01..=5.0));
                    ui.add(egui::DragValue::new(&mut self.thresholds[3]).speed(0.05).range(0.01..=5.0));
                });
            });

        ui.add_space(3.0);

        // --- GRAF I FULL BREDDE MED VERKTØYLINJE ---
        self.render_plot_toolbar(ui, theme);
        ui.add_space(2.0);
        let available_height = (ui.available_height() - 4.0).max(340.0);
        self.render_plot_widget(ui, theme, available_height);
    }

    /// Verktøylinje over grafen med eksplisitte zoom-knapper og visningsvalg
    fn render_plot_toolbar(&mut self, ui: &mut egui::Ui, theme: &ThemeColors) {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Kurvevisning:").strong().size(13.5).color(theme.text_primary));

            if ui.button("🔍+ Zoom inn").on_hover_text("Zoom inn på midten av intervallet").clicked() {
                self.zoom_plot_in();
            }
            if ui.button("🔍- Zoom ut").on_hover_text("Zoom ut fra midten av intervallet").clicked() {
                self.zoom_plot_out();
            }
            if ui.button("↺ [0, 2π]").on_hover_text("Tilbakestill visningsområde til standard [0, 2π]").clicked() {
                self.reset_plot_range();
            }

            ui.separator();

            ui.checkbox(&mut self.show_peaks, RichText::new("▲ Topper").color(theme.peak_color).strong());
            ui.checkbox(&mut self.show_troughs, RichText::new("▼ Bunner").color(theme.trough_color).strong());

            ui.separator();

            ui.checkbox(&mut self.enable_plot_mouse_zoom, "Mus-zoom")
                .on_hover_text("Når av (anbefalt): Mushjulet ruller siden jevnt uten å fanges i grafen.");
        });
    }

    /// Selve graf-widgeten
    fn render_plot_widget(&mut self, ui: &mut egui::Ui, theme: &ThemeColors, plot_height: f32) {
        let mut plot = Plot::new("extrema_plot")
            .height(plot_height)
            .allow_zoom(false)
            .allow_drag(true)
            .allow_scroll(self.enable_plot_mouse_zoom)
            .show_grid(true);

        plot = plot.x_axis_label("x (radianer)").y_axis_label("f(x)");

        plot.show(ui, |plot_ui| {
            if let Some(func) = &self.current_func {
                let num_curve_pts = 1000;
                let mut curve_pts = Vec::with_capacity(num_curve_pts);
                let step = (self.x_max - self.x_min) / (num_curve_pts - 1) as f64;
                for i in 0..num_curve_pts {
                    let x = self.x_min + i as f64 * step;
                    let y = func.eval(x);
                    curve_pts.push([x, y]);
                }

                let line = Line::new("f(x)", curve_pts)
                    .color(theme.text_accent)
                    .width(2.5);
                plot_ui.line(line);

                if let Some(summary) = &self.summary {
                    if self.show_peaks {
                        let peak_coords: Vec<[f64; 2]> = summary
                            .extrema
                            .iter()
                            .filter(|e| e.kind == ExtremumKind::Toppunkt)
                            .map(|e| [e.x as f64, e.y as f64])
                            .collect();

                        plot_ui.points(
                            Points::new("Toppunkter", peak_coords)
                                .color(theme.peak_color)
                                .radius(6.5)
                                .shape(MarkerShape::Diamond),
                        );
                    }

                    if self.show_troughs {
                        let trough_coords: Vec<[f64; 2]> = summary
                            .extrema
                            .iter()
                            .filter(|e| e.kind == ExtremumKind::Bunnpunkt)
                            .map(|e| [e.x as f64, e.y as f64])
                            .collect();

                        plot_ui.points(
                            Points::new("Bunnpunkter", trough_coords)
                                .color(theme.trough_color)
                                .radius(6.5)
                                .shape(MarkerShape::Down),
                        );
                    }
                }
            }
        });
    }

    fn render_csv_tab(&mut self, ui: &mut egui::Ui, theme: &ThemeColors) {
        egui::Frame::new()
            .fill(theme.bg_card)
            .stroke(theme.border_stroke)
            .inner_margin(8)
            .corner_radius(6)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if ui
                        .button(RichText::new("📂 Bla gjennom etter CSV-fil...").strong().size(12.5).color(theme.text_primary))
                        .on_hover_text("Velg en CSV-fil med (x, y) eller måledata for å gjenkjenne den underliggende formelen")
                        .clicked()
                    {
                        self.pick_and_load_csv();
                    }

                    if let Some(path) = &self.csv_file_path {
                        ui.label(
                            RichText::new(format!("{}", path.file_name().unwrap_or_default().to_string_lossy()))
                                .monospace()
                                .color(theme.text_accent)
                                .strong()
                                .size(12.0),
                        );
                        ui.label(format!("({} rader innlest)", self.csv_data.len()));
                    } else if !self.csv_data.is_empty() {
                        ui.label(RichText::new(format!("(Aktivt datasett: {} punkter)", self.csv_data.len())).color(theme.text_success).size(12.0));
                    } else {
                        ui.label(RichText::new("Ingen fil valgt").color(theme.text_secondary).size(12.0));
                    }

                    ui.separator();

                    ui.label(RichText::new("Maks k:").strong().size(12.0));
                    ui.add(egui::DragValue::new(&mut self.csv_max_n).range(4..=256));

                    ui.label(RichText::new("Maks ledd/linje:").strong().size(12.0));
                    ui.add(egui::DragValue::new(&mut self.csv_terms_per_line).range(1..=12));

                    if ui
                        .button(RichText::new("🔍 Analyser & Finn Formel").color(theme.text_success).strong().size(12.5))
                        .on_hover_text("Kjør Fourier-analyse og identifiser hvilke sinus- og cosinusledd dataene består av")
                        .clicked()
                    {
                        self.analyze_loaded_csv();
                    }
                });

                if let Some(err) = &self.csv_error {
                    ui.label(RichText::new(format!("❌ {}", err)).color(theme.text_error).strong().size(11.5));
                }

                if let Some(status) = &self.csv_status {
                    ui.label(RichText::new(format!("✔ {}", status)).color(theme.text_success).strong().size(11.5));
                }
            });

        ui.add_space(3.0);

        let mut transfer_to_classifier = None;

        if let Some(res) = &self.csv_result {
            egui::Frame::new()
                .fill(theme.bg_card)
                .stroke(theme.border_accent)
                .inner_margin(8)
                .corner_radius(6)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new("Beste tilpassede modell:")
                                .color(theme.text_secondary)
                                .strong()
                                .size(13.0),
                        );
                        ui.separator();
                        ui.label(RichText::new(format!("R²: {:.4} ({:.1}%)", res.best_match.r_squared, res.best_match.r_squared * 100.0)).strong().size(12.0));
                        ui.separator();
                        ui.label(format!("RMSE: {:.5}", res.best_match.rmse));
                        ui.separator();

                        if ui.button(RichText::new("⚡ Overfør formel til funksjonsanalyse").strong().size(11.5)).clicked() {
                            transfer_to_classifier = Some(res.best_match.function.format_expr());
                        }
                    });

                    ui.add_space(3.0);

                    // Skriv ut funksjonen med definert maks antall ledd på hver linje
                    let multiline_fn = res.best_match.function.format_multiline(self.csv_terms_per_line);
                    ui.label(
                        RichText::new(multiline_fn)
                            .color(theme.text_success)
                            .monospace()
                            .strong()
                            .size(13.5),
                    );
                });

            ui.add_space(3.0);

            // Kompakt plot som passer uten scrolling
            Plot::new("csv_fit_plot")
                .height(230.0)
                .show_grid(true)
                .allow_scroll(false)
                .x_axis_label("x")
                .y_axis_label("y")
                .show(ui, |plot_ui| {
                    let pts: Vec<[f64; 2]> = self.csv_data.iter().map(|p| [p.x, p.y]).collect();
                    plot_ui.points(
                        Points::new("CSV Datapunkter", pts)
                            .color(theme.text_secondary)
                            .radius(2.5),
                    );

                    let func = &res.best_match.function;
                    let num_curve_pts = 400;
                    let mut curve = Vec::with_capacity(num_curve_pts);
                    for i in 0..num_curve_pts {
                        let x = (i as f64 / (num_curve_pts - 1) as f64) * TAU;
                        curve.push([x, func.eval(x)]);
                    }
                    plot_ui.line(
                        Line::new("Tilpasset modell", curve)
                            .color(theme.text_success)
                            .width(2.0),
                    );
                });

            ui.add_space(3.0);

            egui::Frame::new()
                .fill(theme.bg_card)
                .stroke(theme.border_stroke)
                .inner_margin(6)
                .corner_radius(6)
                .show(ui, |ui| {
                    egui::Grid::new("csv_candidates_grid").striped(true).show(ui, |ui| {
                        ui.label(RichText::new("#").strong());
                        ui.label(RichText::new("Kandidat-funksjon").strong());
                        ui.label(RichText::new("R²").strong());
                        ui.label(RichText::new("RMSE").strong());
                        ui.label(RichText::new("Maks feil").strong());
                        ui.end_row();

                        for (i, cand) in res.top_candidates.iter().take(4).enumerate() {
                            ui.label(format!("{}", i + 1));
                            ui.label(
                                RichText::new(cand.function.format_multiline(self.csv_terms_per_line))
                                    .strong()
                                    .monospace()
                                    .color(theme.text_primary),
                            );
                            ui.label(format!("{:.4}", cand.r_squared));
                            ui.label(format!("{:.4}", cand.rmse));
                            ui.label(format!("{:.4}", cand.max_error));
                            ui.end_row();
                        }
                    });
                });
        }

        if let Some(formula) = transfer_to_classifier {
            self.formula_input = formula;
            self.tab = ActiveTab::Classifier;
            self.run_gpu_analysis();
        }
    }

    fn render_help_tab(&mut self, ui: &mut egui::Ui, theme: &ThemeColors) {
        egui::Frame::new()
            .fill(theme.bg_card)
            .stroke(theme.border_stroke)
            .inner_margin(16)
            .corner_radius(6)
            .show(ui, |ui| {
                ui.heading(RichText::new("📖 Hjelp, Syntaks & Rapporter").size(19.0).color(theme.text_primary));
                ui.add_space(6.0);
                ui.label(
                    RichText::new("MERK: Applikasjonen støtter EKSKLUSIVT sinus- (sin) og cosinus- (cos) funksjoner!")
                        .color(theme.text_error)
                        .strong()
                        .size(14.0),
                );

                ui.add_space(10.0);
                ui.heading(RichText::new("1. Konstanter ganget med ledd").size(15.0).color(theme.text_accent));
                ui.label("Du kan multiplisere ledd med vilkårlige heltall eller desimaltall:");
                ui.label("  • 2*sin(x)  eller  2sin(x)");
                ui.label("  • 3.5*cos(2x)  eller  3,5*cos(2x) (norsk desimalkomma støttes)");
                ui.label("  • -4 * sin(3x) (negative fortegn)");
                ui.label("  • sin(x) * 2.5 (multiplikator på høyre side)");
                ui.label("  • 2*sin(x) + 3*cos(2x) - 1.5*sin(4x) (blandede funksjoner)");

                ui.add_space(10.0);
                ui.heading(RichText::new("2. Flere intervallnotasjoner [[start, func, slutt, steg]]").size(15.0).color(theme.text_accent));
                ui.label("Du kan kombinere flere intervaller fritt i samme formel:");
                ui.label("  • [[1, sin(x), 3]] + [[2, cos(x), 4]]");
                ui.label("  • 2 * [[1, sin(x), 2]] - 3 * [[3, cos(x), 4]]");
                ui.label("  • sin(x) + [[2, sin(x), 3]] + cos(4x) + [[5, cos(x), 6]]");
                ui.label("  • [[1, sin(x), 9, 2]] (med valgfri stegverdi)");

                ui.add_space(10.0);
                ui.heading(RichText::new("3. Eksport av rapporter (PDF og CSV)").size(15.0).color(theme.text_accent));
                ui.label("All detaljert informasjon vises i rapportene for å holde skjermbildet ryddig:");
                ui.label("  • PDF-rapport: Inneholder fargekodede tabeller, krumning, amplitudeklasser, mønsteranalyse og vektorgraf.");
                ui.label("  • CSV-eksport: Laster ned alle detekterte ekstrema direkte til regneark (.csv) med x (rad/deg), f(x) og klassifisering.");

                ui.add_space(10.0);
                ui.heading(RichText::new("4. Tilgjengelighet & Mørk Modus").size(15.0).color(theme.text_accent));
                ui.label("• Mørk modus: Spesialtilpasset mørk fargepalett med høy kontrast (WCAG AAA).");
                ui.label("• Tekstskalering: Juster tekststørrelsen (100%, 115%, 130%) i toppmenyen etter behov.");
                ui.label("• Skjermtilpasning: Grensesnittet er utformet slik at alt innhold passer direkte på én skjermflate uten behov for vertikal scrolling.");
            });
    }
}

/// Eksporterer detekterte ekstrema som ren (x, y) CSV-fil klar for analyse
pub fn export_extrema_to_csv(
    path: &Path,
    _func: &TrigFunction,
    summary: &ClassificationSummary,
    _patterns: &ExtremaPatternReport,
) -> Result<(), String> {
    let mut file = File::create(path).map_err(|e| format!("Kunne ikke opprette CSV-fil: {}", e))?;
    writeln!(file, "x,y").map_err(|e| e.to_string())?;

    for ext in &summary.extrema {
        writeln!(file, "{:.8},{:.8}", ext.x, ext.y).map_err(|e| e.to_string())?;
    }

    Ok(())
}

pub fn run_gui() -> Result<(), eframe::Error> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([900.0, 600.0])
            .with_title("Trigonometrisk Ekstremanalyse & Rekonstruksjon - GPU Sanntid (sin/cos)"),
        ..Default::default()
    };

    eframe::run_native(
        "Trigonometrisk Ekstremanalyse & Rekonstruksjon",
        native_options,
        Box::new(|cc| Ok(Box::new(ExtremaApp::new(cc)))),
    )
}
