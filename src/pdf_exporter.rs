use crate::classification::{ClassificationSummary, ExtremumKind};
use crate::csv_analyzer::IdentificationResult;
use crate::model::SineFunction;
use crate::patterns::ExtremaPatternReport;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Genererer en komplett HTML-rapport og konverterer den til en PDF-fil
pub fn export_to_pdf(
    pdf_path: &Path,
    title: &str,
    func: &SineFunction,
    gpu_info: (&str, &str),
    summary: &ClassificationSummary,
    patterns: &ExtremaPatternReport,
    identification: Option<&IdentificationResult>,
    ascii_plot: Option<&str>,
) -> Result<PathBuf, String> {
    let html_content = generate_html_report(
        title,
        func,
        gpu_info,
        summary,
        patterns,
        identification,
        ascii_plot,
    );

    // Lagre HTML-filen (enten ved siden av PDF eller som temp)
    let abs_pdf_path = if pdf_path.is_absolute() {
        pdf_path.to_path_buf()
    } else {
        std::env::current_dir().map_err(|e| e.to_string())?.join(pdf_path)
    };

    let html_path = abs_pdf_path.with_extension("html");
    if let Err(e) = fs::write(&html_path, &html_content) {
        return Err(format!("Kunne ikke lagre HTML-rapport: {}", e));
    }

    // Finn nettleser (Google Chrome eller Microsoft Edge)
    let browser_exe = find_chromium_browser()
        .ok_or_else(|| "Fant verken Google Chrome eller Microsoft Edge for PDF-konvertering.".to_string())?;

    let file_url = format!("file:///{}", html_path.to_string_lossy().replace('\\', "/"));

    // Opprett en isolert profil-mappe for headless-kjøring for å unngå konflikter med åpne nettlesere
    let temp_profile = std::env::temp_dir().join("gpu_extrema_pdf_profile");
    let _ = fs::create_dir_all(&temp_profile);

    // Konverter HTML til PDF ved hjelp av Chromium headless
    let mut cmd = Command::new(&browser_exe);
    cmd.arg("--headless=new")
        .arg("--disable-gpu")
        .arg("--no-pdf-header-footer")
        .arg(format!("--user-data-dir={}", temp_profile.display()))
        .arg(format!("--print-to-pdf={}", abs_pdf_path.display()))
        .arg(&file_url);

    let status = cmd.status().map_err(|e| format!("Feil ved kjøring av {}: {}", browser_exe.display(), e))?;

    // Vent inntil 5 sekunder på at filen er ferdig skrevet til disk (Chromium flusher asynkront)
    let mut success = false;
    for _ in 0..50 {
        if abs_pdf_path.exists() {
            if let Ok(meta) = fs::metadata(&abs_pdf_path) {
                if meta.len() > 0 {
                    success = true;
                    break;
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    if !success && !status.success() {
        return Err(format!("PDF-generering feilet med statuskode: {:?}", status.code()));
    }

    if !success {
        return Err(format!("PDF-filen ble ikke opprettet på forventet sti: {}", abs_pdf_path.display()));
    }

    Ok(abs_pdf_path)
}

/// Søker etter Google Chrome eller Microsoft Edge på vanlige Windows-stier
fn find_chromium_browser() -> Option<PathBuf> {
    let mut candidate_paths = Vec::new();

    if let Ok(prog_files) = std::env::var("ProgramFiles") {
        candidate_paths.push(PathBuf::from(&prog_files).join(r"Google\Chrome\Application\chrome.exe"));
        candidate_paths.push(PathBuf::from(&prog_files).join(r"Microsoft\Edge\Application\msedge.exe"));
    }

    if let Ok(prog_files_x86) = std::env::var("ProgramFiles(x86)") {
        candidate_paths.push(PathBuf::from(&prog_files_x86).join(r"Google\Chrome\Application\chrome.exe"));
        candidate_paths.push(PathBuf::from(&prog_files_x86).join(r"Microsoft\Edge\Application\msedge.exe"));
    }

    if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
        candidate_paths.push(PathBuf::from(&local_app_data).join(r"Google\Chrome\Application\chrome.exe"));
        candidate_paths.push(PathBuf::from(&local_app_data).join(r"Microsoft\Edge\Application\msedge.exe"));
    }

    for path in candidate_paths {
        if path.exists() {
            return Some(path);
        }
    }

    // Forsøk å finne via PATH
    for name in &["chrome.exe", "msedge.exe"] {
        if let Ok(output) = Command::new("where.exe").arg(name).output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Some(first_line) = stdout.lines().next() {
                    let p = PathBuf::from(first_line.trim());
                    if p.exists() {
                        return Some(p);
                    }
                }
            }
        }
    }

    None
}

/// Genererer en SVG-kurve for funksjonen og markerte topp/bunnpunkter
fn generate_svg_plot(func: &SineFunction, summary: &ClassificationSummary) -> String {
    let width = 760.0;
    let height = 280.0;
    let pad_left = 65.0;
    let pad_right = 35.0;
    let pad_top = 25.0;
    let pad_bottom = 35.0;

    let plot_w = width - pad_left - pad_right;
    let plot_h = height - pad_top - pad_bottom;

    let (min_y, max_y) = match (summary.global_min, summary.global_max) {
        (Some((_, y1)), Some((_, y2))) => {
            let p = (y2 - y1) * 0.15;
            (y1 - p, y2 + p)
        }
        _ => (-3.0, 3.0),
    };

    let y_to_pixel = |y: f32| -> f32 {
        pad_top + (max_y - y) / (max_y - min_y) * plot_h
    };

    let x_to_pixel = |x: f32| -> f32 {
        pad_left + (x / std::f32::consts::TAU) * plot_w
    };

    let mut points_str = String::new();
    let num_steps = 400;
    for i in 0..=num_steps {
        let t = i as f32 / num_steps as f32;
        let x = t * std::f32::consts::TAU;
        let y = func.eval(x as f64) as f32;
        let px = x_to_pixel(x);
        let py = y_to_pixel(y);
        if i == 0 {
            points_str.push_str(&format!("{:.1},{:.1}", px, py));
        } else {
            points_str.push_str(&format!(" {:.1},{:.1}", px, py));
        }
    }

    let zero_y_px = y_to_pixel(0.0);

    let mut markers_svg = String::new();
    for ext in &summary.extrema {
        let px = x_to_pixel(ext.x);
        let py = y_to_pixel(ext.y);
        match ext.kind {
            ExtremumKind::Toppunkt => {
                markers_svg.push_str(&format!(
                    r##"<polygon points="{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}" fill="#16a34a" stroke="#ffffff" stroke-width="1.5" />
                       <text x="{:.1}" y="{:.1}" font-size="10" fill="#15803d" font-weight="bold" text-anchor="middle">{:.2}</text>"##,
                    px, py - 7.0, px - 6.0, py + 3.0, px + 6.0, py + 3.0,
                    px, py - 9.0, ext.y
                ));
            }
            ExtremumKind::Bunnpunkt => {
                markers_svg.push_str(&format!(
                    r##"<polygon points="{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}" fill="#dc2626" stroke="#ffffff" stroke-width="1.5" />
                       <text x="{:.1}" y="{:.1}" font-size="10" fill="#b91c1c" font-weight="bold" text-anchor="middle">{:.2}</text>"##,
                    px, py + 7.0, px - 6.0, py - 3.0, px + 6.0, py - 3.0,
                    px, py + 18.0, ext.y
                ));
            }
        }
    }

    format!(
        r##"<svg width="100%" height="280" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="background:#ffffff; border:1px solid #e2e8f0; border-radius:8px;">
            <!-- Rutenett og Akser -->
            <line x1="{pl}" y1="{pt}" x2="{pl}" y2="{pb}" stroke="#94a3b8" stroke-width="1.5" />
            <line x1="{pl}" y1="{pb}" x2="{pr}" y2="{pb}" stroke="#94a3b8" stroke-width="1.5" />
            <line x1="{pl}" y1="{zero_y}" x2="{pr}" y2="{zero_y}" stroke="#cbd5e1" stroke-dasharray="4,4" stroke-width="1" />

            <!-- Y-akse merker -->
            <text x="{pl_text}" y="{pt}" font-size="11" fill="#64748b" text-anchor="end">{max_y:.1}</text>
            <text x="{pl_text}" y="{zero_y_text}" font-size="11" fill="#64748b" text-anchor="end">0.0</text>
            <text x="{pl_text}" y="{pb}" font-size="11" fill="#64748b" text-anchor="end">{min_y:.1}</text>

            <!-- X-akse merker -->
            <text x="{pl}" y="{pb_text}" font-size="11" fill="#64748b" text-anchor="middle">0</text>
            <text x="{x_pi}" y="{pb_text}" font-size="11" fill="#64748b" text-anchor="middle">π</text>
            <text x="{pr}" y="{pb_text}" font-size="11" fill="#64748b" text-anchor="middle">2π</text>

            <!-- Funksjonskurve -->
            <polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="{pts}" />

            <!-- Ekstremapunkter -->
            {markers}
        </svg>"##,
        w = width,
        h = height,
        pl = pad_left,
        pr = width - pad_right,
        pt = pad_top,
        pb = height - pad_bottom,
        zero_y = zero_y_px,
        pl_text = pad_left - 8.0,
        zero_y_text = zero_y_px + 4.0,
        pb_text = height - pad_bottom + 18.0,
        x_pi = pad_left + plot_w * 0.5,
        max_y = max_y,
        min_y = min_y,
        pts = points_str,
        markers = markers_svg,
    )
}

/// Bygger den komplette HTML-rapporten med identisk innhold som konsollen
fn generate_html_report(
    title: &str,
    func: &SineFunction,
    gpu_info: (&str, &str),
    summary: &ClassificationSummary,
    patterns: &ExtremaPatternReport,
    identification: Option<&IdentificationResult>,
    ascii_plot: Option<&str>,
) -> String {
    let now = chrono_free_timestamp();
    let svg_chart = generate_svg_plot(func, summary);

    // CSV Identifikasjonsseksjon (om relevant)
    let id_section = if let Some(res) = identification {
        let best = &res.best_match;
        let freqs: Vec<String> = best.function.terms.iter().map(|t| t.freq.to_string()).collect();

        let mut cand_rows = String::new();
        for (rank, cand) in res.top_candidates.iter().enumerate() {
            let row_class = if rank == 0 { "font-weight: bold; background-color: #f0fdf4;" } else { "" };
            cand_rows.push_str(&format!(
                r#"<tr style="{}">
                    <td style="text-align:center;">#{}</td>
                    <td>{}</td>
                    <td style="text-align:right;">{:.4}%</td>
                    <td style="text-align:right;">{:.5}</td>
                    <td style="text-align:right;">{:.5}</td>
                </tr>"#,
                row_class,
                rank + 1,
                cand.function,
                cand.r_squared * 100.0,
                cand.rmse,
                cand.max_error
            ));
        }

        format!(
            r#"<section class="card">
                <h2>1. CSV-Data Identifisering</h2>
                <div class="metrics-grid">
                    <div class="metric"><span class="label">Beste tilpasning</span><span class="value val-blue">{}</span></div>
                    <div class="metric"><span class="label">Identifiserte frekvenser</span><span class="value">[{}]</span></div>
                    <div class="metric"><span class="label">Forklaringsgrad (R²)</span><span class="value val-green">{:.5} ({:.2}%)</span></div>
                    <div class="metric"><span class="label">Avvik (RMSE)</span><span class="value">{:.6}</span></div>
                    <div class="metric"><span class="label">Maksimalfeil</span><span class="value">{:.6}</span></div>
                    <div class="metric"><span class="label">Datagrunnlag</span><span class="value">{} punkter</span></div>
                </div>
                <h3>Topp 5 kandidatfunksjoner</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Rang</th>
                            <th>Kandidatfunksjon</th>
                            <th>R² Fit</th>
                            <th>RMSE</th>
                            <th>Maks Feil</th>
                        </tr>
                    </thead>
                    <tbody>
                        {}
                    </tbody>
                </table>
            </section>"#,
            best.function,
            freqs.join(", "),
            best.r_squared,
            best.r_squared * 100.0,
            best.rmse,
            best.max_error,
            res.data_points_count,
            cand_rows
        )
    } else {
        String::new()
    };

    // Oversiktstabell
    let (max_str, min_str) = (
        summary.global_max.map_or("N/A".to_string(), |(x, y)| format!("y = {:.4} ved x = {:.4} rad", y, x)),
        summary.global_min.map_or("N/A".to_string(), |(x, y)| format!("y = {:.4} ved x = {:.4} rad", y, x)),
    );

    // Klassefordeling
    let peak_labels = [
        "Minimal/Negativ Topp (y < 0.2)",
        "Lav Topp (0.2 <= y < 1.0)",
        "Moderat Topp (1.0 <= y < 1.8)",
        "Høy Topp (1.8 <= y < 2.5)",
        "Ekstrem Topp (y >= 2.5)",
    ];
    let trough_labels = [
        "Minimal/Positiv Bunn (y > -0.2)",
        "Grunn Bunn (-1.0 < y <= -0.2)",
        "Moderat Bunn (-1.8 < y <= -1.0)",
        "Dyp Bunn (-2.5 < y <= -1.8)",
        "Ekstrem Bunn (y <= -2.5)",
    ];

    let mut class_rows = String::new();
    for idx in (0..5).rev() {
        class_rows.push_str(&format!(
            r#"<tr>
                <td style="font-weight:bold;">Klasse {}</td>
                <td>{}</td>
                <td style="text-align:right; font-weight:bold; color:#16a34a;">{}</td>
                <td>{}</td>
                <td style="text-align:right; font-weight:bold; color:#dc2626;">{}</td>
            </tr>"#,
            idx,
            peak_labels[idx],
            summary.peak_counts[idx],
            trough_labels[idx],
            summary.trough_counts[idx]
        ));
    }

    // Terskeltabell
    let mut thresh_rows = String::new();
    for (i, &t) in summary.thresholds.iter().enumerate() {
        let count = summary.peak_threshold_exceeded[i];
        let pct = if summary.total_peaks > 0 {
            (count as f32 / summary.total_peaks as f32) * 100.0
        } else {
            0.0
        };
        thresh_rows.push_str(&format!(
            r#"<tr>
                <td style="font-weight:bold;">A &ge; {:.2}</td>
                <td style="text-align:right; font-weight:bold;">{}</td>
                <td style="text-align:right;">{:.1}%</td>
            </tr>"#,
            t, count, pct
        ));
    }

    // Detaljerte ekstremapunkter
    let mut ext_rows = String::new();
    for (i, ext) in summary.extrema.iter().enumerate() {
        let deg = ext.x * (180.0 / std::f32::consts::PI);
        let (badge, color) = match ext.kind {
            ExtremumKind::Toppunkt => (r#"<span class="badge badge-green">&#9650; Toppunkt</span>"#, "#16a34a"),
            ExtremumKind::Bunnpunkt => (r#"<span class="badge badge-red">&#9660; Bunnpunkt</span>"#, "#dc2626"),
        };

        ext_rows.push_str(&format!(
            r#"<tr>
                <td style="text-align:center;">{}</td>
                <td>{:.4} rad ({:.1}&deg;)</td>
                <td style="text-align:right; font-weight:bold; color:{};">{:+.4}</td>
                <td>{}</td>
                <td>{}</td>
            </tr>"#,
            i + 1,
            ext.x,
            deg,
            color,
            ext.y,
            badge,
            ext.class_name
        ));
    }

    // Mønsteranalyse - Klynger
    let mut peak_clusters_html = String::new();
    if patterns.peak_clusters.is_empty() {
        peak_clusters_html.push_str("<p>Ingen identiske toppunkter funnet.</p>");
    } else {
        for c in &patterns.peak_clusters {
            let x_str: Vec<String> = c.x_positions.iter().map(|x| format!("{:.3}", x)).collect();
            peak_clusters_html.push_str(&format!(
                r#"<li>Nivå <b>y &approx; {:+.4}</b>: {} stk ved x = [{}] rad</li>"#,
                c.average_y,
                c.count,
                x_str.join(", ")
            ));
        }
    }

    let mut trough_clusters_html = String::new();
    if patterns.trough_clusters.is_empty() {
        trough_clusters_html.push_str("<p>Ingen identiske bunnpunkter funnet.</p>");
    } else {
        for c in &patterns.trough_clusters {
            let x_str: Vec<String> = c.x_positions.iter().map(|x| format!("{:.3}", x)).collect();
            trough_clusters_html.push_str(&format!(
                r#"<li>Nivå <b>y &approx; {:+.4}</b>: {} stk ved x = [{}] rad</li>"#,
                c.average_y,
                c.count,
                x_str.join(", ")
            ));
        }
    }

    // Mønsteranalyse - Like på rad
    let mut consecutive_html = String::new();
    if patterns.consecutive_peaks.is_empty() && patterns.consecutive_troughs.is_empty() {
        consecutive_html.push_str("<p>Ingen like verdier direkte etter hverandre på rad (amplitudene modulerer kontinuerlig).</p>");
    } else {
        for run in &patterns.consecutive_peaks {
            consecutive_html.push_str(&format!(
                r#"<li><b>Toppunkter på rad:</b> {} topper på rad med verdi y &approx; {:.4} (fra x={:.3} til x={:.3})</li>"#,
                run.count, run.value, run.start_x, run.end_x
            ));
        }
        for run in &patterns.consecutive_troughs {
            consecutive_html.push_str(&format!(
                r#"<li><b>Bunnpunkter på rad:</b> {} bunner på rad med verdi y &approx; {:.4} (fra x={:.3} til x={:.3})</li>"#,
                run.count, run.value, run.start_x, run.end_x
            ));
        }
    }

    // Ekstrema rekkefølge
    let seq_items: Vec<String> = patterns.sequence_pattern.iter().map(|(kind, y, _)| {
        match kind {
            ExtremumKind::Toppunkt => format!(r#"<span class="badge badge-green">&#9650; {:+.2}</span>"#, y),
            ExtremumKind::Bunnpunkt => format!(r#"<span class="badge badge-red">&#9660; {:+.2}</span>"#, y),
        }
    }).collect();

    let ascii_plot_html = if let Some(plot) = ascii_plot {
        format!(
            r#"<section class="card page-break">
                <h2>ASCII-Kurve (fra terminalen)</h2>
                <pre class="ascii-plot">{}</pre>
            </section>"#,
            html_escape(plot)
        )
    } else {
        String::new()
    };

    format!(
        r#"<!DOCTYPE html>
<html lang="no">
<head>
    <meta charset="UTF-8">
    <title>{title} - GPU Rapport</title>
    <style>
        @page {{
            size: A4;
            margin: 15mm;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #1e293b;
            background-color: #f8fafc;
            margin: 0;
            padding: 20px;
            font-size: 13px;
            line-height: 1.5;
        }}
        .header {{
            background: linear-gradient(135deg, #1e3a8a, #3b82f6);
            color: #ffffff;
            padding: 24px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0 0 8px 0;
            font-size: 22px;
            letter-spacing: -0.5px;
        }}
        .header .subtitle {{
            font-size: 14px;
            opacity: 0.9;
            margin-bottom: 12px;
        }}
        .meta-tags {{
            display: flex;
            gap: 15px;
            font-size: 12px;
            opacity: 0.85;
        }}
        .card {{
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 18px;
            margin-bottom: 20px;
            page-break-inside: avoid;
        }}
        .card h2 {{
            margin-top: 0;
            font-size: 16px;
            color: #0f172a;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 8px;
            margin-bottom: 14px;
        }}
        .card h3 {{
            font-size: 14px;
            color: #334155;
            margin: 14px 0 8px 0;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin-bottom: 15px;
        }}
        .metric {{
            background: #f1f5f9;
            border: 1px solid #cbd5e1;
            padding: 10px 14px;
            border-radius: 6px;
        }}
        .metric .label {{
            display: block;
            font-size: 11px;
            text-transform: uppercase;
            color: #64748b;
            font-weight: 600;
        }}
        .metric .value {{
            display: block;
            font-size: 16px;
            font-weight: 700;
            color: #0f172a;
            margin-top: 4px;
        }}
        .val-green {{ color: #16a34a !important; }}
        .val-red {{ color: #dc2626 !important; }}
        .val-blue {{ color: #2563eb !important; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            margin-bottom: 10px;
        }}
        th, td {{
            padding: 7px 10px;
            border: 1px solid #e2e8f0;
        }}
        th {{
            background-color: #f8fafc;
            color: #475569;
            font-weight: 600;
            text-align: left;
        }}
        tr:nth-child(even) td {{
            background-color: #fafaf9;
        }}
        .badge {{
            display: inline-block;
            padding: 2px 7px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }}
        .badge-green {{ background: #dcfce7; color: #166534; }}
        .badge-red {{ background: #fee2e2; color: #991b1b; }}
        .ascii-plot {{
            background: #0f172a;
            color: #38bdf8;
            padding: 14px;
            border-radius: 6px;
            font-family: "Cascadia Code", Consolas, "Courier New", monospace;
            font-size: 11px;
            line-height: 1.25;
            overflow-x: auto;
            white-space: pre;
        }}
        .seq-flow {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            align-items: center;
            margin: 10px 0;
        }}
        .page-break {{
            page-break-before: always;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <div class="subtitle">Funksjon: <b>{func}</b></div>
        <div class="meta-tags">
            <div><b>GPU:</b> {gpu_name} ({gpu_backend})</div>
            <div><b>Dato:</b> {now}</div>
            <div><b>Intervall:</b> x &isin; [0, 2&pi;]</div>
        </div>
    </div>

    {id_section}

    <section class="card">
        <h2>Vektor-Visualisering av Funksjon og Ekstrema</h2>
        {svg_chart}
    </section>

    <section class="card">
        <h2>2. Ekstrema-Oversikt & Amplitude-Klassifisering</h2>
        <div class="metrics-grid">
            <div class="metric"><span class="label">Totalt Toppunkter</span><span class="value val-green">{total_peaks}</span></div>
            <div class="metric"><span class="label">Totalt Bunnpunkter</span><span class="value val-red">{total_troughs}</span></div>
            <div class="metric"><span class="label">Globalt Maksimum</span><span class="value val-green">{max_val}</span></div>
            <div class="metric"><span class="label">Globalt Minimum</span><span class="value val-red">{min_val}</span></div>
            <div class="metric"><span class="label">Topp-til-Tå Amplitude</span><span class="value val-blue">{p2p:.4}</span></div>
        </div>

        <h3>Fordeling per Amplitude-Klasse</h3>
        <table>
            <thead>
                <tr>
                    <th>Nivå</th>
                    <th>Toppunkt Kategori</th>
                    <th style="text-align:right;">Antall Topper</th>
                    <th>Bunnpunkt Kategori</th>
                    <th style="text-align:right;">Antall Bunner</th>
                </tr>
            </thead>
            <tbody>
                {class_rows}
            </tbody>
        </table>

        <h3>Test mot Brukerdefinerte Terskler</h3>
        <table>
            <thead>
                <tr>
                    <th>Amplitudeterskel</th>
                    <th style="text-align:right;">Topper som overstiger (+A)</th>
                    <th style="text-align:right;">Prosent av topper</th>
                </tr>
            </thead>
            <tbody>
                {thresh_rows}
            </tbody>
        </table>
    </section>

    <section class="card">
        <h2>3. Mønsteranalyse (Like Verdier, På Rad & Symmetri)</h2>
        <h3>Identiske / Like Verdier for Toppunkter</h3>
        <ul>
            {peak_clusters_html}
        </ul>

        <h3>Identiske / Like Verdier for Bunnpunkter</h3>
        <ul>
            {trough_clusters_html}
        </ul>

        <h3>Like Verdier Etter Hverandre (På Rad)</h3>
        <ul>
            {consecutive_html}
        </ul>

        <h3>Symmetri og Avstander</h3>
        <p>
            &bull; <b>Oddesymmetri:</b> {sym_text}<br>
            &bull; <b>Gjennomsnittlig avstand mellom topper:</b> &Delta;x = {avg_spacing:.4} rad ({avg_deg:.1}&deg;)<br>
            &bull; <b>Toppunkt-fordeling:</b> {spacing_text}
        </p>

        <h3>Ekstrema-rekkefølge langs perioden</h3>
        <div class="seq-flow">
            {seq_html}
        </div>
    </section>

    <section class="card page-break">
        <h2>4. Detaljerte GPU-Klassifiserte Ekstremapunkter</h2>
        <table>
            <thead>
                <tr>
                    <th style="text-align:center;">#</th>
                    <th>Posisjon (x)</th>
                    <th style="text-align:right;">Amplitude (y)</th>
                    <th>Type</th>
                    <th>Klassifisering</th>
                </tr>
            </thead>
            <tbody>
                {ext_rows}
            </tbody>
        </table>
    </section>

    {ascii_plot_html}
</body>
</html>"#,
        title = title,
        func = func,
        gpu_name = gpu_info.0,
        gpu_backend = gpu_info.1,
        now = now,
        id_section = id_section,
        svg_chart = svg_chart,
        total_peaks = summary.total_peaks,
        total_troughs = summary.total_troughs,
        max_val = max_str,
        min_val = min_str,
        p2p = summary.peak_to_peak(),
        class_rows = class_rows,
        thresh_rows = thresh_rows,
        peak_clusters_html = peak_clusters_html,
        trough_clusters_html = trough_clusters_html,
        consecutive_html = consecutive_html,
        sym_text = if patterns.has_odd_symmetry {
            "JA - 100% av toppunktene har et nøyaktig motstykke i et bunnpunkt (-y) ved 2&pi; - x! (Oddesymmetri / sinus)"
        } else if patterns.has_even_symmetry {
            "JA - funksjonen har ren cosinus-symmetri speilsymmetrisk om &pi;! (Likesymmetri / cosinus)"
        } else {
            "Delvis / asymmetrisk (blandet sinus og cosinus)"
        },
        avg_spacing = patterns.average_peak_spacing,
        avg_deg = patterns.average_peak_spacing * (180.0 / std::f32::consts::PI),
        spacing_text = if patterns.is_equidistant_peaks {
            "Jevnt fordelt (ekvidistante topper)"
        } else {
            "Modulert / ujevn avstand"
        },
        seq_html = seq_items.join(" &rarr; "),
        ext_rows = ext_rows,
        ascii_plot_html = ascii_plot_html,
    )
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn chrono_free_timestamp() -> String {
    // Enkel tidsstempel uten ekstern dependency
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = duration.as_secs();
    let days = secs / 86400;
    let day_secs = secs % 86400;
    let hours = day_secs / 3600;
    let mins = (day_secs % 3600) / 60;
    let s = day_secs % 60;
    format!("Dag {} siden Unix Epoch, {:02}:{:02}:{:02} UTC", days, hours, mins, s)
}
