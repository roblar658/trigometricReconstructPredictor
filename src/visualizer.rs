use crate::classification::{ClassificationSummary, ExtremumKind};
use crate::csv_analyzer::IdentificationResult;
use crate::model::SineFunction;
use crate::patterns::ExtremaPatternReport;
use colored::Colorize;
use comfy_table::{Cell, Color, Table};

/// Skriver ut en tabell over amplitude-klassifisering
pub fn print_classification_table(summary: &ClassificationSummary, func: &SineFunction) {
    println!("\n{}", "=========================================================================".bright_cyan());
    println!("  {} {}", "GPU AMPLITUDE-KLASSIFISERING FOR:".bold().white(), func.to_string().bright_yellow().bold());
    println!("{}\n", "=========================================================================".bright_cyan());

    // 1. Sammendragstabell
    let mut summary_table = Table::new();
    summary_table.set_header(vec![
        Cell::new("Egenskap").fg(Color::Cyan),
        Cell::new("Verdi").fg(Color::Green),
        Cell::new("Kommentar").fg(Color::DarkGrey),
    ]);

    summary_table.add_row(vec![
        Cell::new("Totalt Toppunkter (Peaks)"),
        Cell::new(summary.total_peaks.to_string()).fg(Color::Yellow),
        Cell::new("Lokale maksima f'(x)=0, f''(x)<0"),
    ]);

    summary_table.add_row(vec![
        Cell::new("Totalt Bunnpunkter (Troughs)"),
        Cell::new(summary.total_troughs.to_string()).fg(Color::Yellow),
        Cell::new("Lokale minima f'(x)=0, f''(x)>0"),
    ]);

    if let Some((max_x, max_y)) = summary.global_max {
        summary_table.add_row(vec![
            Cell::new("Globalt Maksimum"),
            Cell::new(format!("y = {:.4} ved x = {:.4}", max_y, max_x)).fg(Color::Green),
            Cell::new("Høyeste punkt i intervallet"),
        ]);
    }

    if let Some((min_x, min_y)) = summary.global_min {
        summary_table.add_row(vec![
            Cell::new("Globalt Minimum"),
            Cell::new(format!("y = {:.4} ved x = {:.4}", min_y, min_x)).fg(Color::Red),
            Cell::new("Laveste punkt i intervallet"),
        ]);
    }

    summary_table.add_row(vec![
        Cell::new("Topp-til-Tå Amplitude (Peak-to-Peak)"),
        Cell::new(format!("{:.4}", summary.peak_to_peak())).fg(Color::Magenta),
        Cell::new("Maks y minus Min y"),
    ]);

    println!("{}\n{}", "--- OVERSIKT ---".bold(), summary_table);

    // 2. Amplitude-fordelingstabell (Klasser)
    let mut class_table = Table::new();
    class_table.set_header(vec![
        Cell::new("Nivå").fg(Color::Cyan),
        Cell::new("Toppunkt Kategori").fg(Color::Green),
        Cell::new("Antall Topper").fg(Color::Green),
        Cell::new("Bunnpunkt Kategori").fg(Color::Red),
        Cell::new("Antall Bunner").fg(Color::Red),
    ]);

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

    for idx in (0..5).rev() {
        class_table.add_row(vec![
            Cell::new(format!("Klasse {}", idx)),
            Cell::new(peak_labels[idx]),
            Cell::new(summary.peak_counts[idx].to_string()).fg(if summary.peak_counts[idx] > 0 { Color::Green } else { Color::DarkGrey }),
            Cell::new(trough_labels[idx]),
            Cell::new(summary.trough_counts[idx].to_string()).fg(if summary.trough_counts[idx] > 0 { Color::Red } else { Color::DarkGrey }),
        ]);
    }

    println!("\n{}\n{}", "--- AMPLITUDE-FORDELING PER KLASSE ---".bold(), class_table);

    // 3. Terskelsjekk-tabell
    let mut thresh_table = Table::new();
    thresh_table.set_header(vec![
        Cell::new("Amplitudeterskel").fg(Color::Cyan),
        Cell::new("Topper som overstiger (+A)").fg(Color::Green),
        Cell::new("Prosent av topper").fg(Color::Green),
    ]);

    for (i, &th) in summary.thresholds.iter().enumerate() {
        let cnt = summary.peak_threshold_exceeded[i];
        let pct = if summary.total_peaks > 0 {
            (cnt as f64 / summary.total_peaks as f64) * 100.0
        } else {
            0.0
        };

        thresh_table.add_row(vec![
            Cell::new(format!("A >= {:.2}", th)),
            Cell::new(cnt.to_string()),
            Cell::new(format!("{:.1}%", pct)),
        ]);
    }

    println!("\n{}\n{}", "--- TEST MOT BRUKERDEFINERTE AMPLITUDETERSKLER ---".bold(), thresh_table);

    // 4. Detaljert liste over ekstrema
    let mut ext_table = Table::new();
    ext_table.set_header(vec![
        Cell::new("#").fg(Color::DarkGrey),
        Cell::new("Posisjon (x)").fg(Color::Cyan),
        Cell::new("Amplitude (y)").fg(Color::Yellow),
        Cell::new("Type").fg(Color::White),
        Cell::new("Klassifisering").fg(Color::Magenta),
    ]);

    for (i, ext) in summary.extrema.iter().enumerate() {
        let (kind_str, kind_color) = match ext.kind {
            ExtremumKind::Toppunkt => ("▲ Toppunkt", Color::Green),
            ExtremumKind::Bunnpunkt => ("▼ Bunnpunkt", Color::Red),
        };

        ext_table.add_row(vec![
            Cell::new((i + 1).to_string()),
            Cell::new(format!("{:.4} rad ({:.1}°)", ext.x, ext.x.to_degrees())),
            Cell::new(format!("{:+.4}", ext.y)).fg(kind_color),
            Cell::new(kind_str).fg(kind_color),
            Cell::new(ext.class_name),
        ]);
    }

    println!("\n{}\n{}", "--- DETALJERTE DETEKTERTE EKSTREMAPUNKTER PÅ GPU ---".bold(), ext_table);
}

/// Skriver ut en mønsterrapport for topp- og bunnpunkter
pub fn print_pattern_report(report: &ExtremaPatternReport) {
    println!("\n{}", "=========================================================================".bright_magenta());
    println!("  {}", "MØNSTERANALYSE AV TOPP- OG BUNNPUNKTER (REPETISJONER & STRUKTUR):".bold().white());
    println!("{}\n", "=========================================================================".bright_magenta());

    // 1. Like verdier / klynger for toppunkter
    let mut peak_val_table = Table::new();
    peak_val_table.set_header(vec![
        Cell::new("Topp-nivå (y)").fg(Color::Green),
        Cell::new("Antall med denne verdien").fg(Color::Yellow),
        Cell::new("Posisjoner x (rad)").fg(Color::Cyan),
    ]);

    for c in &report.peak_clusters {
        let x_str = c
            .x_positions
            .iter()
            .map(|x| format!("{:.3}", x))
            .collect::<Vec<_>>()
            .join(", ");

        peak_val_table.add_row(vec![
            Cell::new(format!("{:+.4}", c.average_y)).fg(Color::Green),
            Cell::new(format!("{} stk", c.count)).fg(if c.count > 1 { Color::Yellow } else { Color::White }),
            Cell::new(x_str),
        ]);
    }

    println!("{}\n{}", "--- IDENTISKE / LIKE VERDIER FOR TOPPUNKTER ---".bold(), peak_val_table);

    // 2. Like verdier / klynger for bunnpunkter
    let mut trough_val_table = Table::new();
    trough_val_table.set_header(vec![
        Cell::new("Bunn-nivå (y)").fg(Color::Red),
        Cell::new("Antall med denne verdien").fg(Color::Yellow),
        Cell::new("Posisjoner x (rad)").fg(Color::Cyan),
    ]);

    for c in &report.trough_clusters {
        let x_str = c
            .x_positions
            .iter()
            .map(|x| format!("{:.3}", x))
            .collect::<Vec<_>>()
            .join(", ");

        trough_val_table.add_row(vec![
            Cell::new(format!("{:+.4}", c.average_y)).fg(Color::Red),
            Cell::new(format!("{} stk", c.count)).fg(if c.count > 1 { Color::Yellow } else { Color::White }),
            Cell::new(x_str),
        ]);
    }

    println!("\n{}\n{}", "--- IDENTISKE / LIKE VERDIER FOR BUNNPUNKTER ---".bold(), trough_val_table);

    // 3. Sekvenser av like verdier etter hverandre (på rad)
    println!("\n{}", "--- LIKE VERDIER ETTER HVERANDRE (PÅ RAD) ---".bold());
    let mut has_consecutive = false;

    if !report.consecutive_peaks.is_empty() {
        has_consecutive = true;
        println!("  {}:", "Toppunkter på rad med samme verdi".green().bold());
        for run in &report.consecutive_peaks {
            println!(
                "    • {} toppunkter på rad med verdi y ≈ {:.4} (fra x={:.3} til x={:.3})",
                run.count.to_string().yellow().bold(),
                run.value,
                run.start_x,
                run.end_x
            );
        }
    }

    if !report.consecutive_troughs.is_empty() {
        has_consecutive = true;
        println!("  {}:", "Bunnpunkter på rad med samme verdi".red().bold());
        for run in &report.consecutive_troughs {
            println!(
                "    • {} bunnpunkter på rad med verdi y ≈ {:.4} (fra x={:.3} til x={:.3})",
                run.count.to_string().yellow().bold(),
                run.value,
                run.start_x,
                run.end_x
            );
        }
    }

    if !has_consecutive {
        println!("  • Ingen like verdier direkte etter hverandre på rad (amplitudene modulerer kontinuerlig).");
    }

    // 4. Symmetri og avstand
    println!("\n{}", "--- SYMMETRI & AVSTANDSMØNSTER ---".bold());
    if report.has_odd_symmetry {
        println!("  • {}: {}", "Oddesymmetri (f(x) = -f(2π-x))".bold(), "JA - 100% av toppunktene har et nøyaktig motstykke i et bunnpunkt (-y) ved 2π - x!".bright_green());
    } else if report.has_even_symmetry {
        println!("  • {}: {}", "Likesymmetri (f(x) = f(2π-x))".bold(), "JA - funksjonen har ren cosinus-symmetri (speilsymmetrisk om π)!".bright_green());
    } else {
        println!("  • {}: Delvis / asymmetrisk (blandet sin/cos eller faseforskyvning)", "Symmetri".bold());
    }

    if !report.peak_spacings.is_empty() {
        println!(
            "  • Gjennomsnittlig avstand mellom påfølgende topper: Δx = {:.4} rad ({:.1}°)",
            report.average_peak_spacing,
            report.average_peak_spacing.to_degrees()
        );
        if report.is_equidistant_peaks {
            println!("  • Toppunkt-fordeling: {}", "Ekvidistant (jevnt fordelt)".cyan());
        } else {
            let min_sp = report.peak_spacings.iter().copied().fold(f32::INFINITY, f32::min);
            let max_sp = report.peak_spacings.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            println!(
                "  • Toppunkt-fordeling: {} (avstand varierer mellom {:.3} og {:.3} rad)",
                "Modulert / ujevn".yellow(),
                min_sp,
                max_sp
            );
        }
    }

    // 5. Sekvensflyt
    println!("\n{}", "--- EKSTREMA-REKKEFØLGE LANGS PERIODEN ---".bold());
    let mut seq_summary = Vec::new();
    for (kind, y, _cls) in &report.sequence_pattern {
        let sym = match kind {
            ExtremumKind::Toppunkt => format!("▲({:+.2})", y).green(),
            ExtremumKind::Bunnpunkt => format!("▼({:+.2})", y).red(),
        };
        seq_summary.push(sym.to_string());
    }
    println!("  {}\n", seq_summary.join(" -> "));
}

/// Skriver ut resultatet av CSV-analyse og funksjonsidentifisering
pub fn print_identification_report(res: &IdentificationResult) {
    println!("\n{}", "=========================================================================".bright_green());
    println!("  {}", "CSV-DATA ANALYSERT - FUNKSJON IDENTIFISERT:".bold().white());
    println!("{}\n", "=========================================================================".bright_green());

    let best = &res.best_match;
    println!("  {}: {}", "Beste tilpasning".bold(), best.function.to_string().bright_yellow().bold());
    let freqs: Vec<String> = best.function.terms.iter().map(|t| t.freq.to_string()).collect();
    println!("  {}: [{}]", "Identifiserte frekvenser".bold(), freqs.join(", ").cyan().bold());
    println!("  {}: {:.5} ({:.2}% match)", "Forklaringsgrad (R²)".bold(), best.r_squared, best.r_squared * 100.0);
    println!("  {}: {:.6}", "Gjennomsnittlig avvik (RMSE)".bold(), best.rmse);
    println!("  {}: {:.6}", "Maksimalt avvik".bold(), best.max_error);
    println!("  {}: {} punkter i x ∈ [{:.3}, {:.3}], y ∈ [{:.3}, {:.3}]", "Datagrunnlag".bold(), res.data_points_count, res.x_range.0, res.x_range.1, res.y_range.0, res.y_range.1);

    // Tabell med topp kandidater
    let mut cand_table = Table::new();
    cand_table.set_header(vec![
        Cell::new("Rang").fg(Color::DarkGrey),
        Cell::new("Kandidatfunksjon").fg(Color::Yellow),
        Cell::new("R² Fit").fg(Color::Green),
        Cell::new("RMSE").fg(Color::White),
        Cell::new("Maks Feil").fg(Color::White),
    ]);

    for (rank, cand) in res.top_candidates.iter().enumerate() {
        cand_table.add_row(vec![
            Cell::new(format!("#{}", rank + 1)),
            Cell::new(cand.function.to_string()),
            Cell::new(format!("{:.4}%", cand.r_squared * 100.0)).fg(if rank == 0 { Color::Green } else { Color::White }),
            Cell::new(format!("{:.5}", cand.rmse)),
            Cell::new(format!("{:.5}", cand.max_error)),
        ]);
    }

    println!("\n{}\n{}", "--- TOPP 5 KANDIDAT-FUNKSJONER ETTER EVALUERING ---".bold(), cand_table);
}

/// Genererer en ASCII-graf av funksjonen som streng
pub fn render_ascii_plot(func: &SineFunction, summary: &ClassificationSummary, width: usize, height: usize) -> String {
    let mut out = String::new();

    let (min_y, max_y) = match (summary.global_min, summary.global_max) {
        (Some((_, y1)), Some((_, y2))) => {
            let pad = (y2 - y1) * 0.1;
            (y1 - pad, y2 + pad)
        }
        _ => (-3.0, 3.0),
    };

    let mut grid = vec![vec![' '; width]; height];

    let zero_row = if min_y < 0.0 && max_y > 0.0 {
        let r = ((max_y - 0.0) / (max_y - min_y) * (height - 1) as f32).round() as usize;
        r.min(height - 1)
    } else {
        height / 2
    };

    for col in 0..width {
        grid[zero_row][col] = '-';
    }

    for col in 0..width {
        let x = (col as f32 / (width - 1) as f32) * std::f32::consts::TAU;
        let y = func.eval(x as f64) as f32;
        let row = ((max_y - y) / (max_y - min_y) * (height - 1) as f32).clamp(0.0, (height - 1) as f32).round() as usize;
        grid[row][col] = '·';
    }

    for ext in &summary.extrema {
        let col = ((ext.x / std::f32::consts::TAU) * (width - 1) as f32).clamp(0.0, (width - 1) as f32).round() as usize;
        let row = ((max_y - ext.y) / (max_y - min_y) * (height - 1) as f32).clamp(0.0, (height - 1) as f32).round() as usize;
        grid[row][col] = match ext.kind {
            ExtremumKind::Toppunkt => '^',
            ExtremumKind::Bunnpunkt => 'v',
        };
    }

    for (row_idx, row_chars) in grid.iter().enumerate() {
        let cur_y = max_y - (row_idx as f32 / (height - 1) as f32) * (max_y - min_y);
        let line: String = row_chars.iter().collect();
        out.push_str(&format!("{:>6.2} | {}\n", cur_y, line));
    }

    out.push_str(&format!("       +{:-<width$}\n", "", width = width));
    out.push_str(&format!("       0.00{:>width$}\n", "2π", width = width - 4));
    out.push_str("       Tegnforklaring: '^' = Toppunkt, 'v' = Bunnpunkt, '·' = f(x)\n");

    out
}

/// Tegner en ASCII-graf av funksjonen i terminalen med markerte ekstrema
pub fn draw_ascii_plot(func: &SineFunction, summary: &ClassificationSummary, width: usize, height: usize) {
    println!("\n{}\n", "--- ASCII PLOT AV FUNKSJON OG EKSTREMA ---".bold());
    print!("{}", render_ascii_plot(func, summary, width, height));
}
