mod classification;
mod csv_analyzer;
pub mod flac_loader;
mod gpu;
pub mod realtime_service;
mod model;
mod patterns;
mod pdf_exporter;
pub mod ui;
mod visualizer;

use clap::{Parser, Subcommand};
use colored::Colorize;
use model::SineFunction;
use std::path::PathBuf;

#[cfg(windows)]
unsafe extern "system" {
    fn SetConsoleOutputCP(wCodePageID: u32) -> i32;
    fn SetConsoleCP(wCodePageID: u32) -> i32;
}

/// Sørger for at Windows-konsollen alltid kjører med UTF-8 kodeside (65001) og ANSI farger
fn init_console_utf8() {
    #[cfg(windows)]
    {
        unsafe {
            SetConsoleOutputCP(65001);
            SetConsoleCP(65001);
        }
        let _ = colored::control::set_virtual_terminal(true);
    }
}

#[derive(Parser)]
#[command(
    name = "gpu_extrema_classifier",
    author = "Antigravity Pair Programmer",
    version = "1.1.0",
    about = "GPU-akselerert klassifisering av topp- og bunnpunkter for sinus- og cosinusfunksjoner. MERK: KUN sin og cos støttes!"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Klassifiser toppunkter og bunnpunkter for en gitt funksjon på GPU (KUN sin og cos støttes)
    Classify {
        #[arg(
            help = "Funksjonsformel, f.eks. 'sin(x) + cos(2x)' eller 'cos(x) + cos(3x) + sin(9x)'. MERK: KUN sin og cos støttes!",
            num_args = 1..,
            default_values_t = vec!["sin(x) + cos(2x) + sin(9x)".to_string()]
        )]
        formula: Vec<String>,

        #[arg(long, default_value_t = 65536, help = "Antall GPU-evalueringspunkter")]
        samples: u32,

        #[arg(
            long,
            value_delimiter = ',',
            default_value = "0.5,1.0,1.5,2.0",
            help = "Kommaseparerte amplitudeterskler"
        )]
        thresholds: Vec<f32>,

        #[arg(long, default_value_t = 0.02, help = "Toleranse for å anse to verdier som like")]
        tolerance: f32,

        #[arg(
            long,
            default_value_t = true,
            num_args = 0..=1,
            default_missing_value = "true",
            action = clap::ArgAction::Set,
            help = "Tegn ASCII-kurve i terminal"
        )]
        plot: bool,

        #[arg(long, help = "Eksporter rapport til PDF-fil (f.eks. --pdf rapport.pdf)")]
        pdf: Option<PathBuf>,
    },

    /// Analyser en CSV-fil, identifiser hvilken funksjon f(x) det er (sin/cos), og klassifiser ekstrema og mønstre på GPU
    AnalyzeCsv {
        #[arg(help = "Filsti til CSV-filen som skal analyseres")]
        file: PathBuf,

        #[arg(long, default_value_t = 30, help = "Maksimal N-frekvens som testes under søket")]
        max_n: u32,

        #[arg(long, default_value_t = 0.02, help = "Toleranse for like verdier")]
        tolerance: f32,

        #[arg(
            long,
            default_value_t = true,
            num_args = 0..=1,
            default_missing_value = "true",
            action = clap::ArgAction::Set,
            help = "Tegn ASCII-kurve etter identifisering"
        )]
        plot: bool,

        #[arg(long, help = "Eksporter rapport til PDF-fil (f.eks. --pdf rapport.pdf)")]
        pdf: Option<PathBuf>,

        #[arg(long, help = "Begrens analysen til et markert sub-område med x >= sub_min_x")]
        sub_min_x: Option<f64>,

        #[arg(long, help = "Begrens analysen til et markert sub-område med x <= sub_max_x")]
        sub_max_x: Option<f64>,
    },

    /// Analyser en FLAC-lydfil, konverter til CSV og/eller identifiser underliggende trigonometrisk funksjon
    AnalyzeFlac {
        #[arg(help = "Filsti til FLAC-lydfilen som skal analyseres")]
        file: PathBuf,

        #[arg(long, default_value_t = 8000, help = "Maksimal frekvens k som testes under søket (k = Hz)")]
        max_k: u32,

        #[arg(long, default_value_t = 2000, help = "Maksimalt antall prøvepunkter som leses ut fra FLAC-filen")]
        max_points: usize,

        #[arg(long, default_value_t = 0.0, help = "Startforsinkelse i sekunder")]
        start_seconds: f64,

        #[arg(long, default_value_t = true, action = clap::ArgAction::Set, help = "Skaler x som 2*pi*t slik at k representerer frekvens i Hz")]
        hz_scale: bool,

        #[arg(long, help = "Eksporter de utleste datapunktene til en CSV-fil (f.eks. --export-csv lyd.csv)")]
        export_csv: Option<PathBuf>,

        #[arg(long, default_value_t = 0.02, help = "Toleranse for like verdier")]
        tolerance: f32,

        #[arg(long, default_value_t = true, action = clap::ArgAction::Set, help = "Tegn ASCII-kurve")]
        plot: bool,

        #[arg(long, help = "Eksporter rapport til PDF-fil (f.eks. --pdf rapport.pdf)")]
        pdf: Option<PathBuf>,

        #[arg(long, help = "Begrens analysen til et markert sub-område med x >= sub_min_x")]
        sub_min_x: Option<f64>,

        #[arg(long, help = "Begrens analysen til et markert sub-område med x <= sub_max_x")]
        sub_max_x: Option<f64>,
    },

    /// Generer en syntetisk CSV-fil med en gitt formel og valgfri støy for testing (KUN sin og cos støttes)
    GenerateCsv {
        #[arg(short = 'o', long, help = "Filsti til ut-filen (.csv)")]
        output: PathBuf,

        #[arg(
            short = 'f',
            long,
            default_value = "sin(x) + cos(2x) + sin(9x)",
            help = "Formel som skal genereres (KUN sin og cos støttes)"
        )]
        formula: String,

        #[arg(long, default_value_t = 1000, help = "Antall datapunkter")]
        samples: usize,

        #[arg(long, default_value_t = 0.0, help = "Standardavvik for støy")]
        noise: f64,
    },

    /// Utfør en GPU-sveip over harmoniske frekvenser
    Sweep {
        #[arg(long, default_value_t = 3, help = "Startverdi for N")]
        min_n: u32,

        #[arg(long, default_value_t = 15, help = "Sluttverdi for N")]
        max_n: u32,
    },

    /// Kjør en full demonstrasjon av GPU-klassifisering, mønsteranalyse og CSV-identifikasjon (med både sin og cos)
    Demo,

    /// Åpne det grafiske brukergrensesnittet (Standard når ingen kommando angis)
    Gui,

    /// Start sanntids sensor-lyttetjeneste over TCP (for IoT og sanntidssensorer)
    Service {
        #[arg(long, default_value_t = 8765, help = "TCP-port som tjenesten skal lytte på")]
        port: u16,

        #[arg(long, default_value = "127.0.0.1", help = "Vertsadresse (standard: 127.0.0.1)")]
        bind: String,

        #[arg(long, default_value_t = 32, help = "Maksimal frekvens k i den inkrementelle modellen")]
        max_k: usize,
    },

    /// Koble til en kjørende sensor-tjeneste via terminal (interaktiv TCP-klient)
    Client {
        #[arg(long, default_value_t = 8765, help = "TCP-port å koble til")]
        port: u16,

        #[arg(long, default_value = "127.0.0.1", help = "Vertsadresse å koble til")]
        host: String,
    },
}

fn main() {
    init_console_utf8();
    let cli = Cli::parse();

    match cli.command {
        None | Some(Commands::Gui) => {
            if let Err(e) = ui::run_gui() {
                eprintln!("Feil ved oppstart av grafisk brukergrensesnitt: {}", e);
            }
        }
        Some(cmd) => {
            pollster::block_on(async {
                run_cli_command(cmd).await;
            });
        }
    }
}

async fn run_cli_command(command: Commands) {
    match command {
        Commands::Gui => unreachable!(),
        Commands::Service { port, bind, max_k } => {
            println!("{}", "========================================================================".magenta());
            println!("{}", " ⚡ STARTER SANNTIDS SENSOR-TJENESTE (TCP)...".white().bold());
            println!(" Lytter på: {}:{}", bind, port);
            println!(" Maks frekvens k: {}", max_k);
            println!(" Sikkerhet: Streng numerisk validering, ingen skallinjeksjon");
            println!(" Støtter: 'x,y', 'BATCH ... ENDBATCH', 'STATUS', 'RESET', 'QUIT'");
            println!("{}", "========================================================================".magenta());

            let mut controller = realtime_service::RealtimeServiceController::new(port, max_k);
            controller.bind_host = bind;
            if let Err(e) = controller.start() {
                eprintln!("Feil ved oppstart av tjeneste: {}", e);
                std::process::exit(1);
            }

            println!("Tjeneste kjører. Trykk Ctrl+C for å avslutte.");
            while controller.is_running.load(std::sync::atomic::Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        }
        Commands::Client { port, host } => {
            run_interactive_client(host, port);
        }
        Commands::Classify {
            formula,
            samples,
            thresholds,
            tolerance,
            plot,
            pdf,
        } => {
                let formula_str = formula.join(" ");
                let func = match SineFunction::parse(&formula_str) {
                    Ok(f) => f,
                    Err(e) => {
                        eprintln!("{}: {}", "Ugyldig funksjon".red().bold(), e);
                        eprintln!("{}", "MERK: Programmet støtter KUN sinus (sin) og cosinus (cos) funksjoner!".yellow());
                        return;
                    }
                };

                let mut thresh_arr = [0.5, 1.0, 1.5, 2.0];
                for (i, &t) in thresholds.iter().take(4).enumerate() {
                    thresh_arr[i] = t;
                }

                let ctx = match gpu::GpuContext::init().await {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("Feil ved GPU-init: {}", e);
                        return;
                    }
                };

                println!("Bruker GPU: {} ({:?})", ctx.adapter_info.name.cyan(), ctx.adapter_info.backend);

                let pipeline = gpu::pipeline::ExtremaClassifierPipeline::new(&ctx);
                match pipeline.run(
                    &ctx,
                    &func,
                    0.0,
                    std::f32::consts::TAU,
                    samples,
                    thresh_arr,
                ) {
                    Ok(summary) => {
                        visualizer::print_classification_table(&summary, &func);
                        let pattern_report = patterns::analyze_extrema_patterns(&summary.extrema, tolerance);
                        visualizer::print_pattern_report(&pattern_report);
                        let ascii_plot_str = visualizer::render_ascii_plot(&func, &summary, 70, 18);
                        if plot {
                            println!("\n{}\n", "--- ASCII PLOT AV FUNKSJON OG EKSTREMA ---".bold());
                            print!("{}", ascii_plot_str);
                        }

                        if let Some(pdf_file) = pdf {
                            println!("\n{}", "Genererer PDF-rapport...".cyan());
                            let backend_str = format!("{:?}", ctx.adapter_info.backend);
                            match pdf_exporter::export_to_pdf(
                                &pdf_file,
                                "GPU Ekstrema-Klassifisering & Mønsteranalyse",
                                &func,
                                (&ctx.adapter_info.name, &backend_str),
                                &summary,
                                &pattern_report,
                                None,
                                Some(&ascii_plot_str),
                            ) {
                                Ok(p) => println!("{} PDF-rapport lagret: {}", "Vellykket!".green().bold(), p.display()),
                                Err(e) => eprintln!("{} {}", "Feil ved PDF-generering:".red().bold(), e),
                            }
                        }
                    }
                    Err(e) => eprintln!("GPU-kjørefeil: {}", e),
                }
            }

            Commands::AnalyzeCsv { file, max_n, tolerance, plot, pdf, sub_min_x, sub_max_x } => {
                let is_flac = file.extension().map(|e| e.to_string_lossy().to_lowercase() == "flac").unwrap_or(false);
                let max_n = if is_flac && max_n <= 30 { 1000 } else { max_n };
                let points = if is_flac {
                    println!("{}", "Oppdaget FLAC-lydfil, dekoder lydstrøm...".cyan());
                    let mut opts = flac_loader::FlacExtractionOptions::default();
                    if let Some(act) = flac_loader::find_first_active_second(&file, 0.025) {
                        if act > 0.15 {
                            opts.start_seconds = (act * 100.0).floor() / 100.0;
                            println!("⚡ {}", format!("Innledende stillhet oppdaget. Starter ved t = {:.2}s der lyd/tale begynner.", opts.start_seconds).yellow());
                        }
                    }
                    let (meta, pts) = match flac_loader::read_flac_data(&file, &opts) {
                        Ok(res) => res,
                        Err(e) => {
                            eprintln!("Feil ved lesing av FLAC: {}", e);
                            return;
                        }
                    };
                    println!(
                        "FLAC Metadata: {} Hz | {}-bit | {} ch | Totalt: {} prøver ({:.2}s)",
                        meta.sample_rate, meta.bits_per_sample, meta.channels, meta.total_samples, meta.duration_seconds
                    );
                    pts
                } else {
                    println!("{}", "Leser og analyserer CSV-fil...".cyan());
                    match csv_analyzer::read_csv_data(&file) {
                        Ok(pts) => pts,
                        Err(e) => {
                            eprintln!("Feil ved lesing av CSV: {}", e);
                            return;
                        }
                    }
                };

                let points = if sub_min_x.is_some() || sub_max_x.is_some() {
                    let lo = sub_min_x.unwrap_or(f64::NEG_INFINITY);
                    let hi = sub_max_x.unwrap_or(f64::INFINITY);
                    let filtered: Vec<_> = points.into_iter().filter(|p| p.x >= lo && p.x <= hi).collect();
                    println!("🔍 {}", format!("Analyserer markert sub-område x ∈ [{:.3}, {:.3}] ({} punkter)", lo, hi, filtered.len()).yellow());
                    filtered
                } else {
                    points
                };

                println!("Leste {} datapunkter.", points.len().to_string().yellow());

                let id_result = match csv_analyzer::identify_function_from_data(&points, max_n) {
                    Ok(res) => res,
                    Err(e) => {
                        eprintln!("Kunne ikke identifisere funksjonen: {}", e);
                        return;
                    }
                };

                visualizer::print_identification_report(&id_result);

                // Kjør deretter GPU-klassifisering og mønsteranalyse på den identifiserte funksjonen
                let ctx = match gpu::GpuContext::init().await {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("Feil ved GPU-init: {}", e);
                        return;
                    }
                };

                println!("\nKjører GPU-klassifisering og mønsteranalyse på den identifiserte funksjonen...");
                let pipeline = gpu::pipeline::ExtremaClassifierPipeline::new(&ctx);
                match pipeline.run(
                    &ctx,
                    &id_result.best_match.function,
                    0.0,
                    std::f32::consts::TAU,
                    65536,
                    [0.5, 1.0, 1.5, 2.0],
                ) {
                    Ok(summary) => {
                        visualizer::print_classification_table(&summary, &id_result.best_match.function);
                        let pattern_report = patterns::analyze_extrema_patterns(&summary.extrema, tolerance);
                        visualizer::print_pattern_report(&pattern_report);
                        let ascii_plot_str = visualizer::render_ascii_plot(&id_result.best_match.function, &summary, 70, 18);
                        if plot {
                            println!("\n{}\n", "--- ASCII PLOT AV FUNKSJON OG EKSTREMA ---".bold());
                            print!("{}", ascii_plot_str);
                        }

                        if let Some(pdf_file) = pdf {
                            println!("\n{}", "Genererer PDF-rapport...".cyan());
                            let backend_str = format!("{:?}", ctx.adapter_info.backend);
                            match pdf_exporter::export_to_pdf(
                                &pdf_file,
                                "CSV Funksjonsidentifisering & GPU Ekstrema-Klassifisering",
                                &id_result.best_match.function,
                                (&ctx.adapter_info.name, &backend_str),
                                &summary,
                                &pattern_report,
                                Some(&id_result),
                                Some(&ascii_plot_str),
                            ) {
                                Ok(p) => println!("{} PDF-rapport lagret: {}", "Vellykket!".green().bold(), p.display()),
                                Err(e) => eprintln!("{} {}", "Feil ved PDF-generering:".red().bold(), e),
                            }
                        }
                    }
                    Err(e) => eprintln!("GPU-kjørefeil: {}", e),
                }
            }

            Commands::AnalyzeFlac {
                file,
                max_k,
                max_points,
                start_seconds,
                hz_scale,
                export_csv,
                tolerance,
                plot,
                pdf,
                sub_min_x,
                sub_max_x,
            } => {
                println!("{}", "Leser og dekoder FLAC-lydfil...".cyan());

                let (effective_start, auto_seeked) = if start_seconds == 0.0 {
                    if let Some(act) = flac_loader::find_first_active_second(&file, 0.025) {
                        if act > 0.15 {
                            ((act * 100.0).floor() / 100.0, true)
                        } else {
                            (0.0, false)
                        }
                    } else {
                        (0.0, false)
                    }
                } else {
                    (start_seconds, false)
                };

                if auto_seeked {
                    println!(
                        "⚡ {}",
                        format!("Innledende stillhet oppdaget. Starter automatisk ved t = {:.2}s der tale/lyd begynner (bruk --start-seconds 0.001 for å tvinge start fra 0).", effective_start).yellow()
                    );
                }

                let opts = flac_loader::FlacExtractionOptions {
                    max_points,
                    start_seconds: effective_start,
                    duration_seconds: None,
                    time_as_hz_radians: hz_scale,
                };

                let (meta, points) = match flac_loader::read_flac_data(&file, &opts) {
                    Ok(res) => res,
                    Err(e) => {
                        eprintln!("Feil ved dekoding av FLAC: {}", e);
                        return;
                    }
                };

                println!(
                    "🎵 FLAC Lydmetadata: {} Hz | {}-bit | {} kanaler | Varighet: {:.2}s",
                    meta.sample_rate.to_string().yellow(),
                    meta.bits_per_sample.to_string().yellow(),
                    meta.channels.to_string().yellow(),
                    meta.duration_seconds
                );
                println!("Utlest {} analysepunkter (x-skala: {}).", points.len().to_string().green(), if hz_scale { "k = Hz" } else { "x = sekunder" });

                let points = if sub_min_x.is_some() || sub_max_x.is_some() {
                    let lo = sub_min_x.unwrap_or(f64::NEG_INFINITY);
                    let hi = sub_max_x.unwrap_or(f64::INFINITY);
                    let filtered: Vec<_> = points.into_iter().filter(|p| p.x >= lo && p.x <= hi).collect();
                    println!("🔍 {}", format!("Analyserer markert sub-område x ∈ [{:.3}, {:.3}] ({} punkter)", lo, hi, filtered.len()).yellow());
                    filtered
                } else {
                    points
                };

                if let Some(ref csv_out) = export_csv {
                    match flac_loader::export_flac_to_csv(&points, csv_out, meta.sample_rate, hz_scale) {
                        Ok(()) => println!("{} Eksporterte lyddata til CSV: {}", "Vellykket!".green().bold(), csv_out.display()),
                        Err(e) => eprintln!("Kunne ikke eksportere til CSV: {}", e),
                    }
                }

                println!("
Identifiserer trigonometriske frekvenser (Fourier + OMP, maks k = {})...", max_k);
                let id_result = match csv_analyzer::identify_function_from_data(&points, max_k) {
                    Ok(res) => res,
                    Err(e) => {
                        eprintln!("Kunne ikke identifisere funksjonen: {}", e);
                        return;
                    }
                };

                visualizer::print_identification_report(&id_result);

                // Kjør GPU-klassifisering og mønsteranalyse
                let ctx = match gpu::GpuContext::init().await {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("Feil ved GPU-init: {}", e);
                        return;
                    }
                };

                println!("
Kjører GPU-klassifisering og mønsteranalyse på den identifiserte funksjonen...");
                let pipeline = gpu::pipeline::ExtremaClassifierPipeline::new(&ctx);
                let span = id_result.x_range.1 - id_result.x_range.0;
                let x_max = if span > 0.0 { span as f32 } else { std::f32::consts::TAU };

                match pipeline.run(
                    &ctx,
                    &id_result.best_match.function,
                    0.0,
                    x_max,
                    65536,
                    [0.2, 0.4, 0.6, 0.8],
                ) {
                    Ok(summary) => {
                        visualizer::print_classification_table(&summary, &id_result.best_match.function);
                        let pattern_report = patterns::analyze_extrema_patterns(&summary.extrema, tolerance);
                        visualizer::print_pattern_report(&pattern_report);
                        let ascii_plot_str = visualizer::render_ascii_plot(&id_result.best_match.function, &summary, 70, 18);
                        if plot {
                            println!("
{}
", "--- ASCII PLOT AV LYDBØLGE OG EKSTREMA ---".bold());
                            print!("{}", ascii_plot_str);
                        }

                        if let Some(pdf_file) = pdf {
                            println!("
{}", "Genererer PDF-rapport for lydanalysen...".cyan());
                            let backend_str = format!("{:?}", ctx.adapter_info.backend);
                            match pdf_exporter::export_to_pdf(
                                &pdf_file,
                                "FLAC Lydspekter-identifisering & GPU Ekstrema-Klassifisering",
                                &id_result.best_match.function,
                                (&ctx.adapter_info.name, &backend_str),
                                &summary,
                                &pattern_report,
                                Some(&id_result),
                                Some(&ascii_plot_str),
                            ) {
                                Ok(p) => println!("{} PDF-rapport lagret: {}", "Vellykket!".green().bold(), p.display()),
                                Err(e) => eprintln!("{} {}", "Feil ved PDF-generering:".red().bold(), e),
                            }
                        }
                    }
                    Err(e) => eprintln!("GPU-kjørefeil: {}", e),
                }
            }

            Commands::GenerateCsv {
                output,
                formula,
                samples,
                noise,
            } => {
                let func = match SineFunction::parse(&formula) {
                    Ok(f) => f,
                    Err(e) => {
                        eprintln!("{}: {}", "Ugyldig formel".red().bold(), e);
                        eprintln!("{}", "MERK: Programmet støtter KUN sinus (sin) og cosinus (cos) funksjoner!".yellow());
                        return;
                    }
                };

                println!("Genererer CSV for {} med {} punkter...", func.to_string().yellow(), samples);
                if noise > 0.0 {
                    println!("Legger til støy (std = {})...", noise);
                }

                match csv_analyzer::generate_test_csv(
                    &output,
                    &func,
                    0.0,
                    std::f64::consts::TAU,
                    samples,
                    noise,
                ) {
                    Ok(()) => println!("{} Lagret fil: {}", "Vellykket!".green().bold(), output.display()),
                    Err(e) => eprintln!("Feil ved skriving av CSV: {}", e),
                }
            }

            Commands::Sweep { min_n, max_n } => {
                let ctx = match gpu::GpuContext::init().await {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("Feil ved GPU-init: {}", e);
                        return;
                    }
                };

                println!("=== GPU SVEIP OVER f(x) = sin(x) + cos(2x) + sin(Nx) for N ∈ [{}, {}] ===", min_n, max_n);
                println!("GPU: {} ({:?})\n", ctx.adapter_info.name.cyan(), ctx.adapter_info.backend);

                let pipeline = gpu::pipeline::ExtremaClassifierPipeline::new(&ctx);

                let mut table = comfy_table::Table::new();
                table.set_header(vec![
                    comfy_table::Cell::new("N").fg(comfy_table::Color::Cyan),
                    comfy_table::Cell::new("Funksjon").fg(comfy_table::Color::Yellow),
                    comfy_table::Cell::new("Topper").fg(comfy_table::Color::Green),
                    comfy_table::Cell::new("Bunner").fg(comfy_table::Color::Red),
                    comfy_table::Cell::new("Maks y").fg(comfy_table::Color::Green),
                    comfy_table::Cell::new("Min y").fg(comfy_table::Color::Red),
                    comfy_table::Cell::new("Peak-to-Peak").fg(comfy_table::Color::Magenta),
                ]);

                for n in min_n..=max_n {
                    let formula_str = format!("sin(x) + cos(2x) + sin({}x)", n);
                    let func = SineFunction::parse(&formula_str).unwrap();
                    if let Ok(summary) = pipeline.run(
                        &ctx,
                        &func,
                        0.0,
                        std::f32::consts::TAU,
                        32768,
                        [0.5, 1.0, 1.5, 2.0],
                    ) {
                        let max_y = summary.global_max.map(|(_, y)| format!("{:.3}", y)).unwrap_or("-".to_string());
                        let min_y = summary.global_min.map(|(_, y)| format!("{:.3}", y)).unwrap_or("-".to_string());

                        table.add_row(vec![
                            comfy_table::Cell::new(n.to_string()),
                            comfy_table::Cell::new(func.to_string()),
                            comfy_table::Cell::new(summary.total_peaks.to_string()),
                            comfy_table::Cell::new(summary.total_troughs.to_string()),
                            comfy_table::Cell::new(max_y),
                            comfy_table::Cell::new(min_y),
                            comfy_table::Cell::new(format!("{:.3}", summary.peak_to_peak())),
                        ]);
                    }
                }

                println!("{}", table);
            }

            Commands::Demo => {
                run_full_demonstration().await;
            }
        }
}

async fn run_full_demonstration() {
    println!("{}", "=========================================================================".bright_magenta().bold());
    println!("  {}", "GPU EKSTREMA-KLASSIFISERING, MØNSTERANALYSE & CSV-IDENTIFIKASJON".bold().white());
    println!("  {}", "Støtter vilkårlige summer av sin(kx) og cos(kx) (KUN sin og cos støttes)".italic().cyan());
    println!("{}\n", "=========================================================================".bright_magenta().bold());

    let ctx = match gpu::GpuContext::init().await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Kunne ikke initialisere GPU: {}", e);
            return;
        }
    };

    println!("Hardware: {} ({:?})", ctx.adapter_info.name.green().bold(), ctx.adapter_info.backend);

    // DEL 1: Evaluere blandet funksjon: sin(x) + cos(2x) + sin(9x)
    let formula = "sin(x) + cos(2x) + sin(9x)";
    let func = SineFunction::parse(formula).expect("Kunne ikke parse formel");
    println!("\n{}", "--- DEL 1: GPU-KLASSIFISERING FOR BLANDET SIN/COS FUNKSJON ---".yellow().bold());
    println!("Innskrevet funksjon (maks 4 ledd/linje):");
    for line in func.format_multiline(4).lines() {
        println!("  {}", line.cyan().bold());
    }

    let pipeline = gpu::pipeline::ExtremaClassifierPipeline::new(&ctx);
    match pipeline.run(
        &ctx,
        &func,
        0.0,
        std::f32::consts::TAU,
        65536,
        [0.5, 1.0, 1.5, 2.0],
    ) {
        Ok(summary) => {
            visualizer::print_classification_table(&summary, &func);
            let pattern_report = patterns::analyze_extrema_patterns(&summary.extrema, 0.02);
            visualizer::print_pattern_report(&pattern_report);
            visualizer::draw_ascii_plot(&func, &summary, 70, 18);
        }
        Err(e) => eprintln!("Feil under GPU-kjøring: {}", e),
    }

    // DEL 2: CSV-analyse og funksjonsidentifisering for skjult sin/cos-funksjon
    println!("\n{}", "--- DEL 2: GENERERE OG ANALYSERE CSV FOR SKJULT SIN/COS FUNKSJON ---".yellow().bold());
    let test_csv_path = std::env::temp_dir().join("test_trig_dataset.csv");

    let hidden_func = SineFunction::parse("sin(x) + cos(2x)").unwrap();
    println!("Genererer test-CSV for skjult funksjon {} med 500 punkter og 2% støy...", hidden_func.to_string().cyan());

    if let Err(e) = csv_analyzer::generate_test_csv(
        &test_csv_path,
        &hidden_func,
        0.0,
        std::f64::consts::TAU,
        500,
        0.02,
    ) {
        eprintln!("Feil ved opprettelse av test-CSV: {}", e);
        return;
    }

    println!("Leser CSV fra: {}", test_csv_path.display());
    let points = csv_analyzer::read_csv_data(&test_csv_path).expect("Kunne ikke lese CSV");
    let id_result = csv_analyzer::identify_function_from_data(&points, 10).expect("Identifikasjon feilet");

    visualizer::print_identification_report(&id_result);

    // Verifiser mot GPU
    println!("\nKjører GPU-klassifisering og mønsteranalyse på den identifiserte funksjonen...");
    if let Ok(summary) = pipeline.run(
        &ctx,
        &id_result.best_match.function,
        0.0,
        std::f32::consts::TAU,
        65536,
        [0.5, 1.0, 1.5, 2.0],
    ) {
        let pattern_report = patterns::analyze_extrema_patterns(&summary.extrema, 0.02);
        visualizer::print_pattern_report(&pattern_report);
        visualizer::draw_ascii_plot(&id_result.best_match.function, &summary, 70, 16);
    }

    println!("\n{}", "Demonstrasjon fullført uten feil!".green().bold());
}


fn run_interactive_client(host: String, port: u16) {
    use std::io::{stdin, stdout, BufRead, BufReader, Write};
    use std::net::TcpStream;

    println!("{}", "========================================================================".cyan());
    println!("{}", format!(" ⚡ INTERAKTIV SENSOR-KLIENT (TCP: {}:{})", host, port).white().bold());
    println!(" Kommandoer du kan skrive:");
    println!("   • 1, 2 eller 3       : Send 1, 2 eller 3 sensorverdier automatisk");
    println!("   • x, y (f.eks 0.5, 1): Send et spesifikt målepunkt");
    println!("   • batch              : Send en testbatch på 10 verdier");
    println!("   • status             : Hent nåværende f(x) og N fra tjenesten");
    println!("   • reset              : Nullstill modellen");
    println!("   • quit eller exit    : Avslutt forbindelsen rent");
    println!("{}", "========================================================================".cyan());

    let addr = format!("{}:{}", host, port);
    let mut stream = match TcpStream::connect(&addr) {
        Ok(s) => {
            println!(" Tilkoblet tjenesten på {}!", addr);
            s
        }
        Err(e) => {
            eprintln!("Kunne ikke koble til {}: {}", addr, e);
            eprintln!("Tips: Start tjenesten først, enten fra GUI (Fane '⚡ Sanntids Sensor-Service' -> '▶ Start Tjeneste')");
            eprintln!("eller fra en annen terminal med: gpu_extrema_classifier.exe service --port {}", port);
            return;
        }
    };

    let mut reader = BufReader::new(match stream.try_clone() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Feil ved kloning av strøm: {}", e);
            return;
        }
    });

    let mut input = String::new();
    let mut sim_x = 0.0f64;

    loop {
        print!("sensor> ");
        let _ = stdout().flush();
        input.clear();
        if stdin().read_line(&mut input).is_err() || input.trim().is_empty() {
            continue;
        }

        let cmd = input.trim();
        if cmd.eq_ignore_ascii_case("quit") || cmd.eq_ignore_ascii_case("exit") {
            let _ = stream.write_all(b"QUIT\n");
            let _ = stream.flush();
            let mut reply = String::new();
            let _ = reader.read_line(&mut reply);
            println!("< {}", reply.trim());
            println!("Tilkobling avsluttet.");
            break;
        } else if cmd == "1" {
            let (x, y) = realtime_service::generate_random_test_point(&mut sim_x, false);
            let line = format!("{:.3}, {:.3}\n", x, y);
            let _ = stream.write_all(line.as_bytes());
            let _ = stream.flush();
            let mut reply = String::new();
            let _ = reader.read_line(&mut reply);
            println!("< Sendte tilfeldig punkt ({:.3}, {:.3}) -> {}", x, y, reply.trim());
        } else if cmd == "2" {
            for _ in 0..2 {
                let (x, y) = realtime_service::generate_random_test_point(&mut sim_x, false);
                let line = format!("{:.3}, {:.3}\n", x, y);
                let _ = stream.write_all(line.as_bytes());
                let _ = stream.flush();
                let mut reply = String::new();
                let _ = reader.read_line(&mut reply);
                println!("< Tilfeldig punkt ({:.3}, {:.3}) -> {}", x, y, reply.trim());
            }
        } else if cmd == "3" {
            for _ in 0..3 {
                let (x, y) = realtime_service::generate_random_test_point(&mut sim_x, false);
                let line = format!("{:.3}, {:.3}\n", x, y);
                let _ = stream.write_all(line.as_bytes());
                let _ = stream.flush();
                let mut reply = String::new();
                let _ = reader.read_line(&mut reply);
                println!("< Tilfeldig punkt ({:.3}, {:.3}) -> {}", x, y, reply.trim());
            }
        } else if cmd.eq_ignore_ascii_case("batch") || cmd == "b" {
            let mut batch_msg = String::from("BATCH\n");
            for _ in 0..10 {
                let (x, y) = realtime_service::generate_random_test_point(&mut sim_x, false);
                batch_msg.push_str(&format!("{:.3}, {:.3}\n", x, y));
            }
            batch_msg.push_str("ENDBATCH\n");
            let _ = stream.write_all(batch_msg.as_bytes());
            let _ = stream.flush();
            let mut reply = String::new();
            let _ = reader.read_line(&mut reply);
            println!("< {}", reply.trim());
        } else {
            let line = format!("{}\n", cmd);
            let _ = stream.write_all(line.as_bytes());
            let _ = stream.flush();
            let mut reply = String::new();
            if let Ok(n) = reader.read_line(&mut reply) {
                if n > 0 {
                    println!("< {}", reply.trim());
                } else {
                    println!("(Server lukket tilkoblingen)");
                    break;
                }
            } else {
                break;
            }
        }
    }
}
