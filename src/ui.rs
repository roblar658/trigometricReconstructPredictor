use crate::classification::{ClassificationSummary, ExtremumKind};
use crate::csv_analyzer::{self, DataPoint, IdentificationResult};
use crate::gpu::pipeline::ExtremaClassifierPipeline;
use crate::gpu::GpuContext;
use crate::model::TrigFunction;
use crate::patterns::{self, ExtremaPatternReport};
use crate::pdf_exporter;
use eframe::egui::{self, Color32, RichText, Stroke};
use egui_plot::{Line, MarkerShape, Plot, Points, PlotPoints, Polygon, VLine, PlotBounds};
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
    pub csv_fx_expanded: bool,
    pub csv_fx_max_lines: usize,
    pub csv_result: Option<IdentificationResult>,
    pub csv_status: Option<String>,
    pub csv_error: Option<String>,
    pub flac_meta: Option<crate::flac_loader::FlacMetadata>,
    pub flac_options: crate::flac_loader::FlacExtractionOptions,

    // Sub-område markering (minX til maxX) med gul farge
    pub sub_range_enabled: bool,
    pub sub_range_min_x: f64,
    pub sub_range_max_x: f64,

    // Zoom og pan for CSV/FLAC-grafen
    pub csv_enable_mouse_zoom: bool,
    pub csv_plot_bounds_request: Option<[f64; 4]>,
    pub csv_current_bounds: Option<[f64; 4]>,

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
            csv_fx_expanded: true,
            csv_fx_max_lines: 6,
            csv_result: None,
            csv_status: None,
            csv_error: None,
            flac_meta: None,
            flac_options: crate::flac_loader::FlacExtractionOptions::default(),

            sub_range_enabled: false,
            sub_range_min_x: 0.0,
            sub_range_max_x: 2.0 * std::f64::consts::PI,

            csv_enable_mouse_zoom: true,
            csv_plot_bounds_request: None,
            csv_current_bounds: None,

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

    /// Velger og laster CSV- eller FLAC-lydfil
    pub fn pick_and_load_csv(&mut self) {
        self.csv_error = None;
        self.csv_status = None;

        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Måledata & Lydfiler (*.csv, *.flac)", &["csv", "flac", "txt"])
            .add_filter("CSV-filer (*.csv)", &["csv", "txt"])
            .add_filter("FLAC-lydfiler (*.flac)", &["flac"])
            .pick_file()
        {
            let is_flac = path
                .extension()
                .map(|ext| ext.to_string_lossy().to_lowercase() == "flac")
                .unwrap_or(false);

            if is_flac {
                if self.flac_options.start_seconds == 0.0 {
                    if let Some(act) = crate::flac_loader::find_first_active_second(&path, 0.025) {
                        if act > 0.15 {
                            self.flac_options.start_seconds = (act * 100.0).floor() / 100.0;
                        }
                    }
                }
                match crate::flac_loader::read_flac_data(&path, &self.flac_options) {
                    Ok((meta, pts)) => {
                        let hz_info = if self.flac_options.time_as_hz_radians {
                            " (k = Hz)"
                        } else {
                            " (x = t sek)"
                        };
                        self.csv_status = Some(format!(
                            "Lastet FLAC lydfil ({} Hz, {}-bit, {} ch, {} pkt): {}{}",
                            meta.sample_rate,
                            meta.bits_per_sample,
                            meta.channels,
                            pts.len(),
                            path.file_name().unwrap_or_default().to_string_lossy(),
                            hz_info
                        ));
                        let full_min_x = pts.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
                        let full_max_x = pts.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
                        if full_max_x > full_min_x {
                            self.sub_range_min_x = full_min_x;
                            self.sub_range_max_x = full_max_x;
                            self.sub_range_enabled = false;
                            let y_min = pts.iter().map(|p| p.y).fold(0.0f64, f64::min);
                            let y_max = pts.iter().map(|p| p.y).fold(0.0f64, f64::max);
                            let pad = (y_max - y_min).abs().max(0.05) * 0.15;
                            self.csv_plot_bounds_request = Some([full_min_x, y_min - pad, full_max_x, y_max + pad]);
                        }
                        self.csv_file_path = Some(path);
                        let target_k = (meta.sample_rate / 2).max(8000);
                        self.flac_meta = Some(meta);
                        self.csv_data = pts;
                        if self.flac_options.time_as_hz_radians && self.csv_max_n < target_k {
                            self.csv_max_n = target_k;
                        }
                        self.analyze_loaded_csv();
                    }
                    Err(e) => {
                        self.csv_error = Some(format!("Kunne ikke lese FLAC-fil: {}", e));
                    }
                }
            } else {
                match csv_analyzer::read_csv_data(&path) {
                    Ok(data) => {
                        self.csv_status = Some(format!(
                            "Lastet {} datapunkter fra: {}",
                            data.len(),
                            path.file_name().unwrap_or_default().to_string_lossy()
                        ));
                        let full_min_x = data.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
                        let full_max_x = data.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
                        if full_max_x > full_min_x {
                            self.sub_range_min_x = full_min_x;
                            self.sub_range_max_x = full_max_x;
                            self.sub_range_enabled = false;
                            let y_min = data.iter().map(|p| p.y).fold(0.0f64, f64::min);
                            let y_max = data.iter().map(|p| p.y).fold(0.0f64, f64::max);
                            let pad = (y_max - y_min).abs().max(0.05) * 0.15;
                            self.csv_plot_bounds_request = Some([full_min_x, y_min - pad, full_max_x, y_max + pad]);
                        }
                        self.csv_file_path = Some(path);
                        self.flac_meta = None;
                        self.csv_data = data;
                        self.analyze_loaded_csv();
                    }
                    Err(e) => {
                        self.csv_error = Some(format!("Kunne ikke lese CSV: {}", e));
                    }
                }
            }
        }
    }

    /// Laster inn gjeldende FLAC-fil på nytt med oppdaterte innstillinger (tidsskala / punkter)
    pub fn reload_current_flac(&mut self) {
        if let Some(ref path) = self.csv_file_path.clone() {
            if path.extension().map(|e| e.to_string_lossy().to_lowercase() == "flac").unwrap_or(false) {
                match crate::flac_loader::read_flac_data(path, &self.flac_options) {
                    Ok((meta, pts)) => {
                        let target_k = (meta.sample_rate / 2).max(8000);
                        self.flac_meta = Some(meta);
                        self.csv_data = pts;
                        if self.flac_options.time_as_hz_radians && self.csv_max_n < target_k {
                            self.csv_max_n = target_k;
                        }
                        self.analyze_loaded_csv();
                    }
                    Err(e) => {
                        self.csv_error = Some(format!("Feil ved gjeninnlesing av FLAC: {}", e));
                    }
                }
            }
        }
    }

    /// Eksporterer det aktive datasettet (f.eks. fra innlest FLAC-lydfil) til en ren (x, y) CSV-fil
    pub fn export_current_data_to_csv(&mut self) {
        if self.csv_data.is_empty() {
            self.csv_error = Some("Ingen data å eksportere".into());
            return;
        }

        let default_name = if let Some(ref path) = self.csv_file_path {
            let stem = path.file_stem().unwrap_or_default().to_string_lossy();
            format!("{}_eksportert.csv", stem)
        } else {
            "datasett_eksportert.csv".to_string()
        };

        if let Some(target_path) = rfd::FileDialog::new()
            .set_file_name(&default_name)
            .add_filter("CSV-datasett (*.csv)", &["csv"])
            .save_file()
        {
            let res = if let Some(meta) = &self.flac_meta {
                crate::flac_loader::export_flac_to_csv(
                    &self.csv_data,
                    &target_path,
                    meta.sample_rate,
                    self.flac_options.time_as_hz_radians,
                )
            } else {
                let f = File::create(&target_path).map_err(|e| e.to_string());
                match f {
                    Ok(mut file) => {
                        let _ = writeln!(file, "x,y");
                        for p in &self.csv_data {
                            let _ = writeln!(file, "{:.8},{:.8}", p.x, p.y);
                        }
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            };

            match res {
                Ok(()) => {
                    self.csv_status = Some(format!(
                        "Eksporterte {} datapunkter til CSV: {}",
                        self.csv_data.len(),
                        target_path.file_name().unwrap_or_default().to_string_lossy()
                    ));
                }
                Err(e) => {
                    self.csv_error = Some(format!("Feil ved lagring av CSV: {}", e));
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

        let (points_to_analyze, is_sub) = if self.sub_range_enabled && self.sub_range_max_x > self.sub_range_min_x {
            let filtered: Vec<DataPoint> = self.csv_data
                .iter()
                .filter(|p| p.x >= self.sub_range_min_x && p.x <= self.sub_range_max_x)
                .cloned()
                .collect();
            (filtered, true)
        } else {
            (self.csv_data.clone(), false)
        };

        if points_to_analyze.is_empty() {
            self.csv_error = Some("Ingen datapunkter i det valgte sub-området [minX, maxX]".to_string());
            return;
        }

        if points_to_analyze.len() < 5 {
            self.csv_error = Some(format!("For få datapunkter i sub-området ({} stk, krever minst 5)", points_to_analyze.len()));
            return;
        }

        match csv_analyzer::identify_function_from_data(&points_to_analyze, self.csv_max_n) {
            Ok(res) => {
                if is_sub {
                    self.csv_status = Some(format!(
                        "Sub-område x ∈ [{:.3}, {:.3}]: Identifiserte funksjon med R² = {:.4} ({:.1}%, {} av {} pkt)",
                        self.sub_range_min_x,
                        self.sub_range_max_x,
                        res.best_match.r_squared,
                        res.best_match.r_squared * 100.0,
                        points_to_analyze.len(),
                        self.csv_data.len()
                    ));
                } else {
                    self.csv_status = Some(format!(
                        "Identifiserte funksjon med R² = {:.4} ({:.1}%)",
                        res.best_match.r_squared,
                        res.best_match.r_squared * 100.0
                    ));
                }
                self.csv_result = Some(res);
            }
            Err(e) => {
                self.csv_error = Some(format!("Analyse feilet: {}", e));
            }
        }
    }

    /// Eksporterer kun datapunktene i det markerte sub-området til en ny CSV-fil
    pub fn export_sub_range_to_csv(&mut self) {
        if self.csv_data.is_empty() {
            return;
        }
        let sub_pts: Vec<DataPoint> = if self.sub_range_enabled && self.sub_range_max_x > self.sub_range_min_x {
            self.csv_data.iter().filter(|p| p.x >= self.sub_range_min_x && p.x <= self.sub_range_max_x).cloned().collect()
        } else {
            self.csv_data.clone()
        };

        if let Some(target_path) = rfd::FileDialog::new()
            .set_file_name("sub_omraade.csv")
            .add_filter("CSV-data (*.csv)", &["csv"])
            .save_file()
        {
            let res = if let Some(meta) = &self.flac_meta {
                crate::flac_loader::export_flac_to_csv(
                    &sub_pts,
                    &target_path,
                    meta.sample_rate,
                    self.flac_options.time_as_hz_radians,
                )
            } else {
                let mut f = match File::create(&target_path) {
                    Ok(file) => file,
                    Err(e) => {
                        self.csv_error = Some(format!("Kunne ikke opprette fil: {}", e));
                        return;
                    }
                };
                let mut ok = writeln!(f, "x,y").is_ok();
                for p in &sub_pts {
                    if writeln!(f, "{:.8},{:.8}", p.x, p.y).is_err() {
                        ok = false;
                        break;
                    }
                }
                if ok { Ok(()) } else { Err("Feil under skriving av CSV".to_string()) }
            };

            match res {
                Ok(()) => {
                    self.csv_status = Some(format!("Eksporterte {} sub-datapunkter til: {}", sub_pts.len(), target_path.display()));
                }
                Err(e) => {
                    self.csv_error = Some(format!("Eksport feilet: {}", e));
                }
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
                        .button(RichText::new("📂 Åpne CSV / FLAC-lydfil...").strong().size(12.5).color(theme.text_primary))
                        .on_hover_text("Velg en CSV-fil eller FLAC-lydfil for direkte Fourier- og OMP-ekstremanalyse")
                        .clicked()
                    {
                        self.pick_and_load_csv();
                    }

                    if !self.csv_data.is_empty() {
                        if ui
                            .button(RichText::new("💾 Lagre som CSV...").strong().size(12.0))
                            .on_hover_text("Eksporter det aktive datasettet (f.eks. utlest fra FLAC-lydfilen) til en ren (x, y) CSV-fil")
                            .clicked()
                        {
                            self.export_current_data_to_csv();
                        }
                    }

                    if let Some(path) = &self.csv_file_path {
                        ui.label(
                            RichText::new(format!("{}", path.file_name().unwrap_or_default().to_string_lossy()))
                                .monospace()
                                .color(theme.text_accent)
                                .strong()
                                .size(12.0),
                        );
                        ui.label(format!("({} punkter)", self.csv_data.len()));
                    } else if !self.csv_data.is_empty() {
                        ui.label(RichText::new(format!("(Aktivt datasett: {} punkter)", self.csv_data.len())).color(theme.text_success).size(12.0));
                    } else {
                        ui.label(RichText::new("Ingen fil valgt").color(theme.text_secondary).size(12.0));
                    }

                    ui.separator();

                    ui.label(RichText::new("Maks k:").strong().size(12.0));
                    ui.add(
                        egui::DragValue::new(&mut self.csv_max_n)
                            .range(1..=u32::MAX)
                            .speed(1.0),
                    )
                    .on_hover_text("Maksimal frekvens k som testes (ingen øvre grense)");

                    ui.label(RichText::new("Maks ledd/linje:").strong().size(12.0));
                    ui.add(egui::DragValue::new(&mut self.csv_terms_per_line).range(1..=12));

                    ui.label(RichText::new("Rullegardin maks linjer:").strong().size(12.0));
                    ui.add(egui::DragValue::new(&mut self.csv_fx_max_lines).range(1..=20))
                        .on_hover_text("Maksimalt antall synlige linjer i rullegardinen for f(x) (standard: 6 linjer)");

                    if ui
                        .button(RichText::new("🔍 Analyser & Finn Formel").color(theme.text_success).strong().size(12.5))
                        .on_hover_text("Kjør Fourier-analyse og identifiser hvilke sinus- og cosinusledd dataene består av")
                        .clicked()
                    {
                        self.analyze_loaded_csv();
                    }
                });

                let mut need_reload_flac = false;
                if let Some(meta) = self.flac_meta.clone() {
                    ui.add_space(2.0);
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new(format!(
                                "🎵 FLAC-lyd: {} Hz | {}-bit | {} ch | Totalt: {} prøver ({:.2}s)",
                                meta.sample_rate, meta.bits_per_sample, meta.channels, meta.total_samples, meta.duration_seconds
                            ))
                            .color(theme.text_accent)
                            .strong()
                            .size(11.5),
                        );
                        ui.separator();
                        let changed_hz = ui
                            .checkbox(&mut self.flac_options.time_as_hz_radians, "Frekvens k = Hz (x = 2π·t)")
                            .on_hover_text("Når aktivert, tilsvarer funksjonens frekvens k tonehøyden i Hertz direkte")
                            .changed();

                        ui.label(RichText::new("Start (sek):").size(11.5));
                        let changed_start = ui
                            .add(egui::DragValue::new(&mut self.flac_options.start_seconds).range(0.0..=meta.duration_seconds).speed(0.05).max_decimals(2))
                            .on_hover_text("Starttidspunkt i sekunder i lydfilen")
                            .changed();

                        if ui.button("⚡ Hopp til aktiv lyd").on_hover_text("Søk fram til første aktive lydsegment (hopper over innledende stillhet)").clicked() {
                            if let Some(ref path) = self.csv_file_path {
                                if let Some(act_sec) = crate::flac_loader::find_first_active_second(path, 0.025) {
                                    self.flac_options.start_seconds = (act_sec * 100.0).floor() / 100.0;
                                    need_reload_flac = true;
                                }
                            }
                        }

                        ui.label(RichText::new("Maks FLAC-pkt:").size(11.5));
                        let changed_pts = ui
                            .add(egui::DragValue::new(&mut self.flac_options.max_points).range(200..=50_000).speed(100.0))
                            .on_hover_text("Antall prøvepunkter som leses ut fra FLAC-lydfilen")
                            .changed();

                        if changed_hz || changed_start || changed_pts {
                            need_reload_flac = true;
                        }
                    });
                }
                if need_reload_flac {
                    self.reload_current_flac();
                }

                if let Some(err) = &self.csv_error {
                    ui.label(RichText::new(format!("❌ {}", err)).color(theme.text_error).strong().size(11.5));
                }

                if let Some(status) = &self.csv_status {
                    ui.label(RichText::new(format!("✔ {}", status)).color(theme.text_success).strong().size(11.5));
                }
            });

        ui.add_space(3.0);

        let mut transfer_to_classifier = None;
        let mut need_reanalyze = false;
        let mut need_export_sub = false;

        if let Some(res) = self.csv_result.clone() {
            egui::Frame::new()
                .fill(theme.bg_card)
                .stroke(theme.border_accent)
                .inner_margin(8)
                .corner_radius(6)
                .show(ui, |ui| {
                    let multiline_fn = res.best_match.function.format_multiline(self.csv_terms_per_line);
                    let line_count = multiline_fn.lines().count();
                    let max_lines = self.csv_fx_max_lines.max(1);

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

                        if ui.button(RichText::new("⚡ Overfør formel").strong().size(11.5))
                            .on_hover_text("Overfør denne formelen til funksjonsanalyse og ekstrema-klassifisering")
                            .clicked()
                        {
                            transfer_to_classifier = Some(res.best_match.function.format_expr());
                        }

                        if ui.button(RichText::new("📋 Kopier").size(11.5))
                            .on_hover_text("Kopier hele f(x)-uttrykket til utklippstavlen")
                            .clicked()
                        {
                            ui.ctx().copy_text(multiline_fn.clone());
                        }

                        let fold_label = if self.csv_fx_expanded { "▲ Rull opp" } else { "▼ Rull ned" };
                        if ui.button(RichText::new(fold_label).strong().size(11.5))
                            .on_hover_text("Rull opp eller ned visningen av funksjonsuttrykket f(x)")
                            .clicked()
                        {
                            self.csv_fx_expanded = !self.csv_fx_expanded;
                        }

                        if self.csv_fx_expanded && line_count > max_lines {
                            ui.label(
                                RichText::new(format!("(Rullegardin: Viser {} av {} linjer - rull i feltet)", max_lines, line_count))
                                    .size(11.0)
                                    .color(Color32::from_rgb(255, 215, 0))
                                    .italics(),
                            );
                        }
                    });

                    ui.add_space(3.0);

                    if self.csv_fx_expanded {
                        // Rullegardin / ScrollArea for f(x) utprint, begrenset til maks 6 linjer
                        let line_height = ui.text_style_height(&egui::TextStyle::Monospace).max(18.0);
                        let scroll_max_h = (max_lines as f32 * line_height) + 6.0;

                        egui::Frame::new()
                            .fill(theme.bg_card_alt)
                            .stroke(theme.border_stroke)
                            .inner_margin(6)
                            .corner_radius(4)
                            .show(ui, |ui| {
                                egui::ScrollArea::vertical()
                                    .id_salt("csv_fx_rullegardin_scroll")
                                    .max_height(scroll_max_h)
                                    .auto_shrink([false, true])
                                    .show(ui, |ui| {
                                        ui.label(
                                            RichText::new(&multiline_fn)
                                                .color(theme.text_success)
                                                .monospace()
                                                .strong()
                                                .size(13.5),
                                        );
                                    });
                            });
                    } else {
                        let single_line_preview = res.best_match.function.format_expr();
                        let truncated = if single_line_preview.len() > 80 {
                            format!("{}...", &single_line_preview[..77])
                        } else {
                            single_line_preview
                        };
                        ui.label(
                            RichText::new(format!("f(x) = {} ({} ledd)", truncated, res.best_match.function.terms.len()))
                                .color(theme.text_success)
                                .monospace()
                                .size(12.5),
                        );
                    }
                });

            ui.add_space(3.0);

            // --- SUB-OMRÅDE MARKERINGSKONTROLLER (GUL FARGE minX TIL maxX) ---
            let full_min_x = self.csv_data.iter().map(|p| p.x).fold(0.0f64, f64::min);
            let full_max_x = self.csv_data.iter().map(|p| p.x).fold(1.0f64, f64::max);
            let x_span = (full_max_x - full_min_x).max(1e-4);
            let speed = x_span / 200.0;

            ui.horizontal(|ui| {
                ui.label(RichText::new("📐 Sub-område:").strong().size(12.5).color(theme.text_secondary));

                let _ = ui
                    .checkbox(&mut self.sub_range_enabled, RichText::new("Aktiver sub-område").strong().color(Color32::from_rgb(255, 215, 0)))
                    .on_hover_text("Når aktivert, beregnes og finnes formelen KUN for det markerte området mellom minX og maxX");

                ui.separator();
                ui.label(RichText::new("minX:").size(11.5).strong().color(Color32::from_rgb(255, 215, 0)));
                let changed_min = ui
                    .add(egui::DragValue::new(&mut self.sub_range_min_x).range(full_min_x..=self.sub_range_max_x).speed(speed).max_decimals(4))
                    .on_hover_text("Startverdi for sub-område (kan også settes ved venstreklikk i grafen)")
                    .changed();

                ui.label(RichText::new("maxX:").size(11.5).strong().color(Color32::from_rgb(255, 215, 0)));
                let changed_max = ui
                    .add(egui::DragValue::new(&mut self.sub_range_max_x).range(self.sub_range_min_x..=full_max_x).speed(speed).max_decimals(4))
                    .on_hover_text("Sluttverdi for sub-område (kan også settes ved høyreklikk i grafen)")
                    .changed();

                if let Some(ref _meta) = self.flac_meta {
                    let t_min = if self.flac_options.time_as_hz_radians { self.sub_range_min_x / TAU } else { self.sub_range_min_x };
                    let t_max = if self.flac_options.time_as_hz_radians { self.sub_range_max_x / TAU } else { self.sub_range_max_x };
                    ui.label(RichText::new(format!("(tid: {:.3}s - {:.3}s)", t_min, t_max)).size(11.0).color(theme.text_secondary));
                }

                if changed_min || changed_max {
                    self.sub_range_enabled = true;
                }

                if ui
                    .button(RichText::new("⚡ Finn funksjon for sub-område").strong().color(Color32::from_rgb(255, 215, 0)))
                    .on_hover_text("Kjør Fourier- og OMP-analyse utelukkende på det markerte gule sub-området")
                    .clicked()
                {
                    self.sub_range_enabled = true;
                    need_reanalyze = true;
                }

                if ui
                    .button("↺ Hele datasettet")
                    .on_hover_text("Nullstill sub-område og analyser hele det innlastede datasettet på nytt")
                    .clicked()
                {
                    self.sub_range_min_x = full_min_x;
                    self.sub_range_max_x = full_max_x;
                    self.sub_range_enabled = false;
                    need_reanalyze = true;
                }

                if ui
                    .button("💾 Lagre sub-område...")
                    .on_hover_text("Eksporter kun punktene i det markerte sub-området til en ny CSV-fil")
                    .clicked()
                {
                    need_export_sub = true;
                }
            });

            ui.label(
                RichText::new("💡 Tips: Du kan markere direkte i grafen: Venstreklikk = minX, Høyreklikk = maxX. Markert område lyser opp i gul farge.")
                    .size(10.5)
                    .color(theme.text_secondary)
            );

            ui.add_space(2.0);

            // --- ZOOM-VERKTØYLINJE FOR CSV/FLAC-GRAF ---
            let mut zoom_to_sub = false;
            let mut zoom_in_req = false;
            let mut zoom_out_req = false;
            let mut reset_zoom_req = false;

            ui.horizontal(|ui| {
                ui.label(RichText::new("🔍 Zoom & Visning:").strong().size(12.5).color(theme.text_secondary));

                if ui
                    .button(RichText::new("🔍 Zoom til sub-område").strong().color(Color32::from_rgb(255, 215, 0)))
                    .on_hover_text("Zoom direkte inn på det markerte gule sub-området [minX, maxX]")
                    .clicked()
                {
                    zoom_to_sub = true;
                }

                if ui.button("🔍+ Zoom inn").on_hover_text("Zoom 30% inn på midten av grafen").clicked() {
                    zoom_in_req = true;
                }

                if ui.button("🔍- Zoom ut").on_hover_text("Zoom 40% ut fra midten av grafen").clicked() {
                    zoom_out_req = true;
                }

                if ui.button("↺ Vis hele signalet").on_hover_text("Tilbakestill visningen til hele datasettet").clicked() {
                    reset_zoom_req = true;
                }

                ui.separator();

                ui.checkbox(&mut self.csv_enable_mouse_zoom, "Mus-zoom")
                    .on_hover_text("Når aktivert: Rull på musehjulet for å zoome inn/ut i grafen. Du kan også dra med musen for å panorere, eller holde inne for boks-zoom.");
            });

            ui.add_space(2.0);

            let sub_min = if self.sub_range_enabled { self.sub_range_min_x } else { full_min_x };
            let sub_max = if self.sub_range_enabled { self.sub_range_max_x } else { full_max_x };

            let y_min = self.csv_data.iter().map(|p| p.y).fold(0.0f64, f64::min);
            let y_max = self.csv_data.iter().map(|p| p.y).fold(0.0f64, f64::max);
            let y_pad = (y_max - y_min).abs().max(0.05) * 0.15;
            let y_bot = y_min - y_pad;
            let y_top = y_max + y_pad;

            // Håndter zoomforespørsler før plottet tegnes
            if zoom_to_sub {
                let sub_pts: Vec<_> = self.csv_data.iter().filter(|p| p.x >= sub_min && p.x <= sub_max).collect();
                let sub_y_min = sub_pts.iter().map(|p| p.y).fold(y_min, f64::min);
                let sub_y_max = sub_pts.iter().map(|p| p.y).fold(y_max, f64::max);
                let pad = (sub_y_max - sub_y_min).abs().max(0.05) * 0.15;
                self.csv_plot_bounds_request = Some([sub_min, sub_y_min - pad, sub_max, sub_y_max + pad]);
            } else if zoom_in_req {
                if let Some([bx0, by0, bx1, by1]) = self.csv_current_bounds {
                    let mid_x = (bx0 + bx1) / 2.0;
                    let span_x = (bx1 - bx0) * 0.70;
                    let mid_y = (by0 + by1) / 2.0;
                    let span_y = (by1 - by0) * 0.70;
                    self.csv_plot_bounds_request = Some([mid_x - span_x / 2.0, mid_y - span_y / 2.0, mid_x + span_x / 2.0, mid_y + span_y / 2.0]);
                }
            } else if zoom_out_req {
                if let Some([bx0, by0, bx1, by1]) = self.csv_current_bounds {
                    let mid_x = (bx0 + bx1) / 2.0;
                    let span_x = (bx1 - bx0) * 1.40;
                    let mid_y = (by0 + by1) / 2.0;
                    let span_y = (by1 - by0) * 1.40;
                    self.csv_plot_bounds_request = Some([mid_x - span_x / 2.0, mid_y - span_y / 2.0, mid_x + span_x / 2.0, mid_y + span_y / 2.0]);
                }
            } else if reset_zoom_req {
                let pad = (y_max - y_min).abs().max(0.05) * 0.15;
                self.csv_plot_bounds_request = Some([full_min_x, y_min - pad, full_max_x, y_max + pad]);
            }

            let yellow_fill = Color32::from_rgba_unmultiplied(255, 235, 59, 45); // myk lysende gul bakgrunn
            let yellow_solid = Color32::from_rgb(255, 215, 0); // gull/gul

            let bounds_req = self.csv_plot_bounds_request.take();
            let mut updated_bounds = None;

            // Plot med full støtte for zoom, pan, boks-zoom og valgfri mus-hjul zoom
            let plot_resp = Plot::new("csv_fit_plot")
                .height(240.0)
                .show_grid(true)
                .allow_drag(true)
                .allow_zoom(true)
                .allow_boxed_zoom(true)
                .allow_scroll(self.csv_enable_mouse_zoom)
                .x_axis_label(if self.flac_options.time_as_hz_radians { "x = 2π·t (k = Hz)" } else { "x" })
                .y_axis_label("y (amplitude)")
                .show(ui, |plot_ui| {
                    if let Some([bx0, by0, bx1, by1]) = bounds_req {
                        plot_ui.set_plot_bounds(PlotBounds::from_min_max([bx0, by0], [bx1, by1]));
                    }
                    let b = plot_ui.plot_bounds();
                    updated_bounds = Some([b.min()[0], b.min()[1], b.max()[0], b.max()[1]]);
                    // GUL BAKGRUNNSMARKERING MELLOM minX OG maxX
                    if self.sub_range_enabled || sub_max > sub_min {
                        let poly = Polygon::new(
                            "Markert sub-område (gul)",
                            PlotPoints::new(vec![
                                [sub_min, y_bot],
                                [sub_max, y_bot],
                                [sub_max, y_top],
                                [sub_min, y_top],
                            ]),
                        )
                        .fill_color(yellow_fill)
                        .stroke(Stroke::new(1.0, yellow_solid));

                        plot_ui.polygon(poly);

                        // Gule vertikale avgrensningslinjer
                        plot_ui.vline(VLine::new("minX", sub_min).color(yellow_solid).width(2.0));
                        plot_ui.vline(VLine::new("maxX", sub_max).color(yellow_solid).width(2.0));
                    }

                    // Del opp punkter: Innenfor (gul) og utenfor (dempet grå)
                    let mut pts_inside = Vec::new();
                    let mut pts_outside = Vec::new();
                    for p in &self.csv_data {
                        if self.sub_range_enabled && (p.x < sub_min || p.x > sub_max) {
                            pts_outside.push([p.x, p.y]);
                        } else {
                            pts_inside.push([p.x, p.y]);
                        }
                    }

                    if !pts_outside.is_empty() {
                        plot_ui.points(
                            Points::new("Utenfor sub-område", pts_outside)
                                .color(Color32::from_gray(100))
                                .radius(2.0),
                        );
                    }

                    let inside_color = if self.sub_range_enabled { yellow_solid } else { theme.text_secondary };
                    plot_ui.points(
                        Points::new("Markert sub-område", pts_inside)
                            .color(inside_color)
                            .radius(2.8),
                    );

                    let func = &res.best_match.function;
                    let num_curve_pts = 600;
                    let mut curve = Vec::with_capacity(num_curve_pts);
                    let (curve_start, curve_end) = if self.sub_range_enabled && sub_max > sub_min {
                        (sub_min, sub_max)
                    } else {
                        res.x_range
                    };
                    let span = if curve_end > curve_start { curve_end - curve_start } else { TAU };
                    for i in 0..num_curve_pts {
                        let x = curve_start + (i as f64 / (num_curve_pts - 1) as f64) * span;
                        curve.push([x, func.eval(x)]);
                    }
                    plot_ui.line(
                        Line::new("Tilpasset modell", curve)
                            .color(theme.text_success)
                            .width(2.5),
                    );
                });

            if let Some(b) = updated_bounds {
                self.csv_current_bounds = Some(b);
            }

            // Håndter klikk på grafen for å sette minX og maxX interaktivt
            if plot_resp.response.clicked() {
                if let Some(pos) = plot_resp.response.interact_pointer_pos() {
                    let plot_pt = plot_resp.transform.value_from_position(pos);
                    let x_clamped = plot_pt.x.clamp(full_min_x, full_max_x);
                    if x_clamped < self.sub_range_max_x {
                        self.sub_range_min_x = x_clamped;
                    } else {
                        self.sub_range_min_x = x_clamped;
                        self.sub_range_max_x = full_max_x;
                    }
                    self.sub_range_enabled = true;
                }
            }
            if plot_resp.response.secondary_clicked() {
                if let Some(pos) = plot_resp.response.interact_pointer_pos() {
                    let plot_pt = plot_resp.transform.value_from_position(pos);
                    let x_clamped = plot_pt.x.clamp(full_min_x, full_max_x);
                    if x_clamped > self.sub_range_min_x {
                        self.sub_range_max_x = x_clamped;
                    } else {
                        self.sub_range_max_x = x_clamped;
                        self.sub_range_min_x = full_min_x;
                    }
                    self.sub_range_enabled = true;
                }
            }

            ui.add_space(3.0);

            egui::Frame::new()
                .fill(theme.bg_card)
                .stroke(theme.border_stroke)
                .inner_margin(6)
                .corner_radius(6)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new("📋 Kandidat-funksjoner:")
                                .strong()
                                .size(12.5)
                                .color(theme.text_secondary),
                        );
                        ui.label(
                            RichText::new("(R² og RMSE er alltid synlige uansett formelstørrelse)")
                                .size(11.0)
                                .color(theme.text_secondary),
                        );
                    });
                    ui.add_space(3.0);

                    egui::ScrollArea::both()
                        .max_height(170.0)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            egui::Grid::new("csv_candidates_grid")
                                .striped(true)
                                .spacing([10.0, 6.0])
                                .show(ui, |ui| {
                                    ui.label(RichText::new("#").strong().color(theme.text_secondary));
                                    ui.label(RichText::new("R²").strong().color(theme.text_success));
                                    ui.label(RichText::new("RMSE").strong().color(theme.text_accent));
                                    ui.label(RichText::new("Maks feil").strong().color(theme.text_secondary));
                                    ui.label(RichText::new("Kandidat-funksjon").strong().color(theme.text_primary));
                                    ui.label(RichText::new("Valg").strong().color(theme.text_secondary));
                                    ui.end_row();

                                    for (i, cand) in res.top_candidates.iter().enumerate() {
                                        ui.label(RichText::new(format!("#{}", i + 1)).strong());
                                        ui.label(
                                            RichText::new(format!("{:.4}", cand.r_squared))
                                                .strong()
                                                .color(if i == 0 { theme.text_success } else { theme.text_primary }),
                                        );
                                        ui.label(RichText::new(format!("{:.5}", cand.rmse)).monospace());
                                        ui.label(RichText::new(format!("{:.5}", cand.max_error)).monospace());

                                        let formula_text = cand.function.format_multiline(self.csv_terms_per_line);
                                        let cand_line_count = formula_text.lines().count();
                                        let max_cand_lines = self.csv_fx_max_lines.max(1);

                                        if cand_line_count > max_cand_lines {
                                            let line_height = ui.text_style_height(&egui::TextStyle::Monospace).max(18.0);
                                            let scroll_max_h = (max_cand_lines as f32 * line_height) + 4.0;
                                            egui::ScrollArea::vertical()
                                                .id_salt(format!("cand_scroll_{}", i))
                                                .max_height(scroll_max_h)
                                                .auto_shrink([false, true])
                                                .show(ui, |ui| {
                                                    ui.label(
                                                        RichText::new(&formula_text)
                                                            .strong()
                                                            .monospace()
                                                            .color(theme.text_primary),
                                                    );
                                                });
                                        } else {
                                            ui.label(
                                                RichText::new(formula_text)
                                                    .strong()
                                                    .monospace()
                                                    .color(theme.text_primary),
                                            );
                                        }

                                        if ui
                                            .button(RichText::new("⚡ Bruk").size(11.0))
                                            .on_hover_text("Bruk denne kandidaten i funksjonsanalysen")
                                            .clicked()
                                        {
                                            transfer_to_classifier = Some(cand.function.format_expr());
                                        }

                                        ui.end_row();
                                    }
                                });
                        });
                });
        }

        if need_reanalyze {
            self.analyze_loaded_csv();
        }
        if need_export_sub {
            self.export_sub_range_to_csv();
        }

        if let Some(formula) = transfer_to_classifier {
            self.formula_input = formula;
            if let Some(res) = &self.csv_result {
                self.x_min = res.x_range.0;
                self.x_max = res.x_range.1;
            }
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
