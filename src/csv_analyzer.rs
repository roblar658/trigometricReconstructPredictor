use crate::model::{TrigFunction, TrigKind, TrigTerm};
use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct DataPoint {
    pub x: f64,
    pub y: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct IdentificationCandidate {
    pub function: TrigFunction,
    pub r_squared: f64,
    pub rmse: f64,
    pub max_error: f64,
    pub detected_harmonics: Vec<(TrigKind, u32, f64)>,
}

#[derive(Debug, Clone)]
pub struct IdentificationResult {
    pub best_match: IdentificationCandidate,
    pub top_candidates: Vec<IdentificationCandidate>,
    pub data_points_count: usize,
    pub x_range: (f64, f64),
    pub y_range: (f64, f64),
}

/// Leser CSV-fil med enten (x, y) eller bare y, eller ekstremal-tabell (x_rad, f_x).
/// Støtter kommentarer (#), ulike skilletegn (komma, semikolon, tab) og norsk desimalkomma.
pub fn read_csv_data<P: AsRef<Path>>(path: P) -> Result<Vec<DataPoint>, Box<dyn Error>> {
    let content = std::fs::read_to_string(path)?;
    parse_csv_string(&content)
}

/// Robust parser for CSV-tekst
pub fn parse_csv_string(content: &str) -> Result<Vec<DataPoint>, Box<dyn Error>> {
    let trimmed = content.trim_start_matches('\u{feff}'); // Fjern ev. UTF-8 BOM

    // 1. Filtrer ut kommentarer og tomme linjer
    let non_comment_lines: Vec<&str> = trimmed
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with("//"))
        .collect();

    if non_comment_lines.is_empty() {
        return Err("Ingen numeriske datarekker funnet i CSV (filen inneholder kun kommentarer eller er tom)".into());
    }

    // 2. Auto-detekter skilletegn. Siden komma ofte brukes som desimalskilletegn i Norge/Europa,
    // har semikolon og tab prioritet dersom linjene konsekvent inneholder disse tegnene.
    let sample_lines = &non_comment_lines[..non_comment_lines.len().min(10)];
    let n_samples = sample_lines.len().max(1);
    let semicolon_count: usize = sample_lines.iter().map(|l| l.matches(';').count()).sum();
    let tab_count: usize = sample_lines.iter().map(|l| l.matches('\t').count()).sum();

    let delimiter = if tab_count >= n_samples.saturating_sub(1) && tab_count > 0 {
        '\t'
    } else if semicolon_count >= n_samples.saturating_sub(1) && semicolon_count > 0 {
        ';'
    } else {
        ','
    };

    // Hjelpefunksjon for å parse tall, inkludert støtte for norsk desimalkomma
    let parse_num = |val: &str| -> Option<f64> {
        let v = val.trim().trim_matches(|c| c == '"' || c == '\'').trim();
        if v.is_empty() {
            return None;
        }
        if let Ok(num) = v.parse::<f64>() {
            return Some(num);
        }
        if v.contains(',') && (delimiter == ';' || delimiter == '\t' || !v.contains('.')) {
            let normalized = v.replace(',', ".");
            if let Ok(num) = normalized.parse::<f64>() {
                return Some(num);
            }
        }
        None
    };

    // 3. Del opp linjer med hensyn til eventuelle anførselstegn
    let mut rows: Vec<Vec<String>> = Vec::new();
    for line in &non_comment_lines {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .delimiter(delimiter as u8)
            .flexible(true)
            .from_reader(line.as_bytes());
        if let Some(Ok(record)) = rdr.records().next() {
            rows.push(record.iter().map(|s| s.trim().to_string()).collect());
        } else {
            rows.push(line.split(delimiter).map(|s| s.trim().to_string()).collect());
        }
    }

    if rows.is_empty() {
        return Err("Ingen datalinjer funnet i CSV".into());
    }

    // 4. Detekter eventuell overskriftsrad og sjekk spesifikke kolonnenavn
    let first_row = &rows[0];
    let mut is_header = false;
    let mut col_x_idx: Option<usize> = None;
    let mut col_y_idx: Option<usize> = None;

    for (idx, field) in first_row.iter().enumerate() {
        let lower = field.to_lowercase().trim_matches(|c| c == '"' || c == '\'').to_string();
        if lower == "x" || lower == "x_rad" || lower == "rad" || lower == "t" || lower == "tid" || lower == "vinkel" || lower == "angle" {
            col_x_idx = Some(idx);
            is_header = true;
        } else if lower == "y" || lower == "f_x" || lower == "f(x)" || lower == "fx" || lower == "verdi" || lower == "val" || lower == "value" || lower == "amplitude" {
            col_y_idx = Some(idx);
            is_header = true;
        } else if parse_num(field).is_none() && !field.is_empty() {
            is_header = true;
        }
    }

    let data_rows = if is_header { &rows[1..] } else { &rows[..] };
    if data_rows.is_empty() {
        return Err("Ingen numeriske datarekker funnet i CSV (kun overskriftsrad)".into());
    }

    let mut points = Vec::new();
    let mut single_col_y = Vec::new();

    // Tilfelle A: Funnet eksplisitte kolonner fra overskrift (f.eks. x_rad og f_x, eller x og y)
    if let (Some(xi), Some(yi)) = (col_x_idx, col_y_idx) {
        for row in data_rows {
            if row.len() > xi.max(yi) {
                if let (Some(x), Some(y)) = (parse_num(&row[xi]), parse_num(&row[yi])) {
                    points.push(DataPoint { x, y });
                }
            }
        }
    }

    // Tilfelle B: Ingen spesifikk overskrift, eller overskrift manglet f(x)-match
    if points.is_empty() {
        for row in data_rows {
            if row.is_empty() {
                continue;
            }

            let numeric_cols: Vec<(usize, f64)> = row
                .iter()
                .enumerate()
                .filter_map(|(i, f)| parse_num(f).map(|val| (i, val)))
                .collect();

            if numeric_cols.len() >= 2 {
                // Sjekk om første numeriske kolonne er en indeks (0, 1, 2...) etterfulgt av x og y
                if numeric_cols.len() >= 3 && numeric_cols[0].0 == 0 && numeric_cols[0].1.fract() == 0.0 && numeric_cols[0].1 >= 0.0 {
                    points.push(DataPoint {
                        x: numeric_cols[1].1,
                        y: numeric_cols[2].1,
                    });
                } else {
                    // Standard x, y
                    points.push(DataPoint {
                        x: numeric_cols[0].1,
                        y: numeric_cols[1].1,
                    });
                }
            } else if numeric_cols.len() == 1 {
                single_col_y.push(numeric_cols[0].1);
            }
        }
    }

    // Tilfelle C: Kun y-verdier oppgitt i en enkelt kolonne (generer x jevnt fordelt over [0, 2π])
    if points.is_empty() && !single_col_y.is_empty() {
        let n = single_col_y.len();
        for (i, y) in single_col_y.into_iter().enumerate() {
            let x = (i as f64 / (n - 1).max(1) as f64) * std::f64::consts::TAU;
            points.push(DataPoint { x, y });
        }
    }

    if points.is_empty() {
        return Err("Ingen numeriske datarekker funnet i CSV (verken x,y eller y-verdier)".into());
    }

    Ok(points)
}

fn snap_amplitude(amp: f64) -> f64 {
    if amp.abs() < 1e-9 {
        return 0.0;
    }
    // For små amplituder (f.eks. mikro-ledd og lydspekter med amp < 0.25):
    if amp.abs() < 0.25 {
        let r_tenth = (amp * 10.0).round() / 10.0;
        if (amp - r_tenth).abs() < 0.008 && r_tenth.abs() > 1e-6 {
            return r_tenth;
        }
        let r_hundredth = (amp * 100.0).round() / 100.0;
        if (amp - r_hundredth).abs() < 0.003 && r_hundredth.abs() > 1e-6 {
            return r_hundredth;
        }
        return (amp * 10_000.0).round() / 10_000.0;
    }

    // For større amplituder (stramme toleranser for å bevare R² og presise topper):
    let r_int = amp.round();
    if (amp - r_int).abs() < 0.03 {
        return r_int;
    }
    let r_half = (amp * 2.0).round() / 2.0;
    if (amp - r_half).abs() < 0.015 {
        return r_half;
    }
    let r_quarter = (amp * 4.0).round() / 4.0;
    if (amp - r_quarter).abs() < 0.008 {
        return r_quarter;
    }
    (amp * 10_000.0).round() / 10_000.0
}

/// Løser lineært minste kvadraters problem (A^T W A) c = A^T W y ved hjelp av
/// Gauss-eliminasjon med delvis pivotering og Tikhonov-regularisering for stabilitet.
fn solve_least_squares(a_cols: &[Vec<f64>], y: &[f64], weights: Option<&[f64]>) -> Option<Vec<f64>> {
    let m = a_cols.len();
    if m == 0 {
        return Some(Vec::new());
    }
    let n = y.len();

    let mut ata = vec![vec![0.0; m]; m];
    let mut aty = vec![0.0; m];

    if let Some(w) = weights {
        for i in 0..m {
            for j in i..m {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += a_cols[i][k] * a_cols[j][k] * w[k];
                }
                ata[i][j] = sum;
                ata[j][i] = sum;
            }
            let mut sum_y = 0.0;
            for k in 0..n {
                sum_y += a_cols[i][k] * y[k] * w[k];
            }
            aty[i] = sum_y;
        }
    } else {
        for i in 0..m {
            for j in i..m {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += a_cols[i][k] * a_cols[j][k];
                }
                ata[i][j] = sum;
                ata[j][i] = sum;
            }
            let mut sum_y = 0.0;
            for k in 0..n {
                sum_y += a_cols[i][k] * y[k];
            }
            aty[i] = sum_y;
        }
    }

    // Tikhonov-regularisering for å hindre singularitet ved multikolinearitet
    let avg_diag: f64 = (0..m).map(|i| ata[i][i]).sum::<f64>() / m as f64;
    let lambda = 1e-10 * avg_diag.max(1e-6);
    for i in 0..m {
        ata[i][i] += lambda;
    }

    // Utvidet matrise [A | b]
    let mut mat = vec![vec![0.0; m + 1]; m];
    for i in 0..m {
        for j in 0..m {
            mat[i][j] = ata[i][j];
        }
        mat[i][m] = aty[i];
    }

    // Gauss-Jordan eliminering med pivotering
    for i in 0..m {
        let mut max_val = mat[i][i].abs();
        let mut max_row = i;
        for k in (i + 1)..m {
            let val = mat[k][i].abs();
            if val > max_val {
                max_val = val;
                max_row = k;
            }
        }
        if max_val < 1e-12 {
            return None;
        }
        if max_row != i {
            mat.swap(i, max_row);
        }

        let pivot = mat[i][i];
        for j in i..=m {
            mat[i][j] /= pivot;
        }

        for k in 0..m {
            if k != i {
                let factor = mat[k][i];
                if factor.abs() > 1e-14 {
                    for j in i..=m {
                        mat[k][j] -= factor * mat[i][j];
                    }
                }
            }
        }
    }

    let coeffs: Vec<f64> = (0..m).map(|i| mat[i][m]).collect();
    Some(coeffs)
}

/// Analyserer data og identifiserer hvilken trigonometrisk funksjon (sin/cos) det er
pub fn identify_function_from_data(
    data: &[DataPoint],
    max_n: u32,
) -> Result<IdentificationResult, String> {
    if data.len() < 4 {
        return Err("For få datapunkter til å identifisere funksjonen (minimum 4)".into());
    }

    let n_pts = data.len() as f64;
    let mean_y = data.iter().map(|p| p.y).sum::<f64>() / n_pts;
    let tss: f64 = data.iter().map(|p| (p.y - mean_y).powi(2)).sum();

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for p in data {
        if p.x < min_x { min_x = p.x; }
        if p.x > max_x { max_x = p.x; }
        if p.y < min_y { min_y = p.y; }
        if p.y > max_y { max_y = p.y; }
    }

    // Sorter data etter x og beregn trapesvekter for spektral nøyaktighet uten endepunktslekkasje
    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted_data.len();
    let mut weights = Vec::with_capacity(n);
    if n >= 2 {
        for i in 0..n {
            let w = if i == 0 {
                (sorted_data[1].x - sorted_data[0].x) / 2.0
            } else if i == n - 1 {
                (sorted_data[n - 1].x - sorted_data[n - 2].x) / 2.0
            } else {
                (sorted_data[i + 1].x - sorted_data[i - 1].x) / 2.0
            };
            weights.push(w);
        }
    } else {
        weights.push(1.0);
    }

    let x_vec: Vec<f64> = sorted_data.iter().map(|p| p.x).collect();
    let y_vec: Vec<f64> = sorted_data.iter().map(|p| p.y).collect();

    // Beregn fysisk Nyquist-frekvensgrense: omega_nyq = pi / dx
    let avg_dx = if n > 1 {
        (max_x - min_x) / (n - 1) as f64
    } else {
        1.0
    };
    let nyquist_k = if avg_dx > 1e-7 {
        (std::f64::consts::PI / avg_dx).floor() as u32
    } else {
        max_n
    };
    let effective_max_n = max_n.min(nyquist_k.max(10));

    // Beregn Fourier-projeksjoner og fase-uavhengig spektral effekt for sinus og cosinus
    struct FreqEnergy {
        freq: u32,
        amp_sin: f64,
        amp_cos: f64,
        power: f64,
    }

    let mut spectrum: Vec<FreqEnergy> = Vec::with_capacity(effective_max_n as usize);
    let mut total_spectral_power = 0.0f64;
    let mut max_spectral_power = 0.0f64;

    for k in 1..=effective_max_n {
        let mut sum_proj_sin = 0.0;
        let mut sum_denom_sin = 0.0;
        let mut sum_proj_cos = 0.0;
        let mut sum_denom_cos = 0.0;

        for (p, &w) in sorted_data.iter().zip(&weights) {
            let angle = k as f64 * p.x;
            let s = angle.sin();
            let c = angle.cos();

            sum_proj_sin += w * p.y * s;
            sum_denom_sin += w * s * s;

            sum_proj_cos += w * p.y * c;
            sum_denom_cos += w * c * c;
        }

        let amp_sin = if sum_denom_sin > 1e-9 { sum_proj_sin / sum_denom_sin } else { 0.0 };
        let amp_cos = if sum_denom_cos > 1e-9 { sum_proj_cos / sum_denom_cos } else { 0.0 };
        let power = amp_sin * amp_sin * sum_denom_sin + amp_cos * amp_cos * sum_denom_cos;

        if power > max_spectral_power {
            max_spectral_power = power;
        }
        total_spectral_power += power;

        spectrum.push(FreqEnergy {
            freq: k,
            amp_sin,
            amp_cos,
            power,
        });
    }

    // Identifiser dype spisser og skarpe ekstrema i signalet for spissvekting (Weighted Least Squares)
    let max_dev = sorted_data.iter().map(|p| (p.y - mean_y).abs()).fold(0.0f64, f64::max);
    let std_dev = (tss / n_pts).sqrt();
    let spike_weights: Option<Vec<f64>> = if max_dev > 1.25 * std_dev && max_dev > 1e-5 {
        let mut sw = Vec::with_capacity(n);
        for (p, &w) in sorted_data.iter().zip(&weights) {
            let dev = (p.y - mean_y).abs();
            let ratio = dev / max_dev;
            // Eksponentiell/kubisk vekting på de dypeste spissene/toppene slik at de fanges med ekstrem presisjon
            let boost = 1.0 + 8.0 * ratio.powi(3);
            sw.push(w * boost);
        }
        Some(sw)
    } else {
        None
    };

    let mut candidate_functions = Vec::new();

    // -------------------------------------------------------------
    // METODE 1: Harmonisk klangidentifisering (Tonedeteksjon for tale, vokaler & periodiske funksjoner)
    // Finner den underliggende grunnfrekvensen F0 og tilpasser alle signifikante harmoniske overtoner
    // -------------------------------------------------------------
    if effective_max_n >= 2 && max_spectral_power > 1e-9 {
        let min_search_f0 = if effective_max_n >= 200 { 15u32 } else { 1u32 };
        let max_search_f0 = 800u32.min(effective_max_n / 2).max(min_search_f0);
        let mut best_f0 = min_search_f0;
        let mut best_f0_score = -1.0f64;

        for f0 in min_search_f0..=max_search_f0 {
            let mut tot_energy = 0.0;
            let mut h = 1;
            while h * f0 <= effective_max_n {
                let idx = (h * f0 - 1) as usize;
                tot_energy += spectrum[idx].power;
                h += 1;
            }
            let h_count = (h - 1) as f64;
            if h_count < 2.0 {
                continue;
            }
            // Normaliser mot antall harmoniske ledd slik at lave frekvenser med tusenvis av ledd ikke stjeler seieren
            let norm_score = tot_energy / h_count.powf(0.35);
            if norm_score > best_f0_score {
                best_f0_score = norm_score;
                best_f0 = f0;
            }
        }

        // Subharmonisk oppløsning (f.eks. hvis F0 ble funnet ved en sterk formant som 520 Hz,
        // undersøk om f0 / 4 = 130 Hz, f0 / 3, eller f0 / 2 har signifikant energi i overtonene)
        for div in [4, 3, 2] {
            if best_f0 % div == 0 {
                let sub_f0 = best_f0 / div;
                if sub_f0 >= min_search_f0 {
                    let mut tot_sub = 0.0;
                    let mut h = 1;
                    while h * sub_f0 <= effective_max_n {
                        let idx = (h * sub_f0 - 1) as usize;
                        tot_sub += spectrum[idx].power;
                        h += 1;
                    }
                    let mut tot_orig = 0.0;
                    let mut h_orig = 1;
                    while h_orig * best_f0 <= effective_max_n {
                        let idx = (h_orig * best_f0 - 1) as usize;
                        tot_orig += spectrum[idx].power;
                        h_orig += 1;
                    }
                    let odd_energy = tot_sub - tot_orig;
                    if odd_energy > 0.08 * tot_orig {
                        best_f0 = sub_f0;
                        break;
                    }
                }
            }
        }

        // Beregn den totale harmoniske energien for best_f0
        let mut best_harm_energy = 0.0;
        let mut h_check = 1;
        while h_check * best_f0 <= effective_max_n {
            let idx = (h_check * best_f0 - 1) as usize;
            best_harm_energy += spectrum[idx].power;
            h_check += 1;
        }

        // Hvis den harmoniske strukturen forklarer en vesentlig del av spekteret:
        if best_harm_energy > 0.15 * total_spectral_power {
            let p_thresh = (max_spectral_power * 1e-5).max(1e-10);
            let mut harm_terms: Vec<(TrigKind, u32)> = Vec::new();
            let mut h = 1;
            // Utvid harmoniske frekvenser opp til fysisk Nyquist (inntil 8000 Hz) for å fange dype spisser helt opp til R² ~ 1
            let harmonic_upper_limit = nyquist_k.min(8000).max(effective_max_n);
            while h * best_f0 <= harmonic_upper_limit && harm_terms.len() < 180 {
                let f = h * best_f0;
                let idx = (f - 1) as usize;
                let has_energy = if idx < spectrum.len() {
                    spectrum[idx].power >= p_thresh
                } else {
                    true
                };
                if has_energy {
                    harm_terms.push((TrigKind::Sin, f));
                    harm_terms.push((TrigKind::Cos, f));
                }
                h += 1;
            }

            if !harm_terms.is_empty() {
                let harm_cols: Vec<Vec<f64>> = harm_terms.iter().map(|&(kind, f)| {
                    x_vec.iter().map(|&x| match kind {
                        TrigKind::Sin => (f as f64 * x).sin(),
                        TrigKind::Cos => (f as f64 * x).cos(),
                    }).collect()
                }).collect();

                // 1A. Standard regresjon
                let solved_coeffs = solve_least_squares(&harm_cols, &y_vec, Some(&weights));
                if let Some(ref coeffs) = solved_coeffs {
                    // Eksakt tilpasning (maksimal R²)
                    let exact_terms: Vec<TrigTerm> = harm_terms.iter().zip(coeffs)
                        .filter_map(|(&(kind, f), &c)| {
                            let rounded = (c * 10_000.0).round() / 10_000.0;
                            if rounded.abs() > 1e-6 { Some(TrigTerm::new(kind, f, rounded)) } else { None }
                        })
                        .collect();
                    if !exact_terms.is_empty() {
                        candidate_functions.push(TrigFunction::new(exact_terms));
                    }

                    // Snappet tilpasning
                    let snapped_terms: Vec<TrigTerm> = harm_terms.iter().zip(coeffs)
                        .filter_map(|(&(kind, f), &c)| {
                            let s = snap_amplitude(c);
                            if s.abs() > 1e-6 { Some(TrigTerm::new(kind, f, s)) } else { None }
                        })
                        .collect();
                    if !snapped_terms.is_empty() {
                        candidate_functions.push(TrigFunction::new(snapped_terms));
                    }

                    // 1C. Iterativ spiss-tilpasning (IRLS for å fange de aller dypeste spissene)
                    let mut max_abs_err = 0.0f64;
                    let mut point_errors = Vec::with_capacity(n);
                    for i in 0..n {
                        let mut pred = 0.0;
                        for j in 0..harm_terms.len() {
                            pred += coeffs[j] * harm_cols[j][i];
                        }
                        let err = (y_vec[i] - pred).abs();
                        if err > max_abs_err {
                            max_abs_err = err;
                        }
                        point_errors.push(err);
                    }

                    if max_abs_err > 1e-4 {
                        let mut irls_weights = Vec::with_capacity(n);
                        for i in 0..n {
                            let err_ratio = point_errors[i] / max_abs_err;
                            let boost = 1.0 + 3.0 * err_ratio.powi(2);
                            irls_weights.push(weights[i] * boost);
                        }

                        if let Some(irls_c) = solve_least_squares(&harm_cols, &y_vec, Some(&irls_weights)) {
                            let irls_terms: Vec<TrigTerm> = harm_terms.iter().zip(&irls_c)
                                .filter_map(|(&(kind, f), &c)| {
                                    let rounded = (c * 10_000.0).round() / 10_000.0;
                                    if rounded.abs() > 1e-6 { Some(TrigTerm::new(kind, f, rounded)) } else { None }
                                })
                                .collect();
                            if !irls_terms.is_empty() {
                                candidate_functions.push(TrigFunction::new(irls_terms));
                            }
                        }
                    }
                }

                // 1B. Spiss-vektet regresjon (for dype spisser i periodiske signaler)
                if let Some(ref sw) = spike_weights {
                    if let Some(spk_coeffs) = solve_least_squares(&harm_cols, &y_vec, Some(sw)) {
                        let spk_terms: Vec<TrigTerm> = harm_terms.iter().zip(&spk_coeffs)
                            .filter_map(|(&(kind, f), &c)| {
                                let rounded = (c * 10_000.0).round() / 10_000.0;
                                if rounded.abs() > 1e-6 { Some(TrigTerm::new(kind, f, rounded)) } else { None }
                            })
                            .collect();
                        if !spk_terms.is_empty() {
                            candidate_functions.push(TrigFunction::new(spk_terms));
                        }
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------
    // METODE 2: Spektral Topp-Deteksjon med Non-Maximum Suppression (NMS)
    // Finner reelle frekvenstopper uten å la seg lure av vinduslekkasje/sidelober
    // -------------------------------------------------------------
    let rayleigh_width = ((std::f64::consts::TAU / (max_x - min_x).max(1e-4)).round() as u32).max(1);
    let mut local_peaks: Vec<(u32, f64)> = Vec::new();

    for i in 0..spectrum.len() {
        let p_curr = spectrum[i].power;
        if p_curr < 1e-5 * max_spectral_power {
            continue;
        }
        let p_prev = if i > 0 { spectrum[i - 1].power } else { 0.0 };
        let p_next = if i + 1 < spectrum.len() { spectrum[i + 1].power } else { 0.0 };

        if p_curr >= p_prev && p_curr >= p_next {
            local_peaks.push((spectrum[i].freq, p_curr));
        }
    }

    // Sorter etter effekt og filtrer ut sidelober (NMS)
    local_peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut nms_peaks: Vec<u32> = Vec::new();
    for &(f, _) in &local_peaks {
        if !nms_peaks.iter().any(|&pf| (pf as i64 - f as i64).abs() <= rayleigh_width as i64) {
            nms_peaks.push(f);
            if nms_peaks.len() >= 100 {
                break;
            }
        }
    }

    // Bygg kandidater fra de sterkeste NMS-toppene
    if !nms_peaks.is_empty() {
        let mut peak_terms: Vec<(TrigKind, u32)> = Vec::new();
        for &f in &nms_peaks {
            let item = &spectrum[(f - 1) as usize];
            if item.amp_sin.abs() > 1e-5 {
                peak_terms.push((TrigKind::Sin, f));
            }
            if item.amp_cos.abs() > 1e-5 {
                peak_terms.push((TrigKind::Cos, f));
            }
        }

        if !peak_terms.is_empty() {
            let peak_cols: Vec<Vec<f64>> = peak_terms.iter().map(|&(kind, f)| {
                x_vec.iter().map(|&x| match kind {
                    TrigKind::Sin => (f as f64 * x).sin(),
                    TrigKind::Cos => (f as f64 * x).cos(),
                }).collect()
            }).collect();

            // 2A. Standard regresjon på NMS-topper
            let solved_c_fit = solve_least_squares(&peak_cols, &y_vec, Some(&weights));
            if let Some(ref c_fit) = solved_c_fit {
                let exact_peaks: Vec<TrigTerm> = peak_terms.iter().zip(c_fit)
                    .filter_map(|(&(kind, f), &c)| {
                        let rounded = (c * 10_000.0).round() / 10_000.0;
                        if rounded.abs() > 1e-6 { Some(TrigTerm::new(kind, f, rounded)) } else { None }
                    })
                    .collect();
                if !exact_peaks.is_empty() {
                    candidate_functions.push(TrigFunction::new(exact_peaks));
                }

                let snapped_peaks: Vec<TrigTerm> = peak_terms.iter().zip(c_fit)
                    .filter_map(|(&(kind, f), &c)| {
                        let s = snap_amplitude(c);
                        if s.abs() > 1e-6 { Some(TrigTerm::new(kind, f, s)) } else { None }
                    })
                    .collect();
                if !snapped_peaks.is_empty() {
                    candidate_functions.push(TrigFunction::new(snapped_peaks));
                }

                // 2D. IRLS Refinement for dype spisser
                let mut max_abs_err = 0.0f64;
                let mut pt_errors = Vec::with_capacity(n);
                for i in 0..n {
                    let mut pred = 0.0;
                    for j in 0..peak_terms.len() {
                        pred += c_fit[j] * peak_cols[j][i];
                    }
                    let err = (y_vec[i] - pred).abs();
                    if err > max_abs_err {
                        max_abs_err = err;
                    }
                    pt_errors.push(err);
                }
                if max_abs_err > 1e-4 {
                    let mut irls_w = Vec::with_capacity(n);
                    for i in 0..n {
                        let boost = 1.0 + 3.0 * (pt_errors[i] / max_abs_err).powi(2);
                        irls_w.push(weights[i] * boost);
                    }
                    if let Some(irls_c) = solve_least_squares(&peak_cols, &y_vec, Some(&irls_w)) {
                        let irls_peaks: Vec<TrigTerm> = peak_terms.iter().zip(&irls_c)
                            .filter_map(|(&(kind, f), &c)| {
                                let rounded = (c * 10_000.0).round() / 10_000.0;
                                if rounded.abs() > 1e-6 { Some(TrigTerm::new(kind, f, rounded)) } else { None }
                            })
                            .collect();
                        if !irls_peaks.is_empty() {
                            candidate_functions.push(TrigFunction::new(irls_peaks));
                        }
                    }
                }
            }

            // 2B. Spiss-vektet regresjon på NMS-topper
            if let Some(ref sw) = spike_weights {
                if let Some(spk_c) = solve_least_squares(&peak_cols, &y_vec, Some(sw)) {
                    let spk_peaks: Vec<TrigTerm> = peak_terms.iter().zip(&spk_c)
                        .filter_map(|(&(kind, f), &c)| {
                            let rounded = (c * 10_000.0).round() / 10_000.0;
                            if rounded.abs() > 1e-6 { Some(TrigTerm::new(kind, f, rounded)) } else { None }
                        })
                        .collect();
                    if !spk_peaks.is_empty() {
                        candidate_functions.push(TrigFunction::new(spk_peaks));
                    }
                }
            }

            // 2C. Subsetter for færre ledd (f.eks. topp 1..=5)
            let max_sub = peak_terms.len().min(10);
            for sub_len in 1..=max_sub {
                let sub_terms = &peak_terms[..sub_len];
                let sub_cols = &peak_cols[..sub_len];
                if let Some(sub_c) = solve_least_squares(sub_cols, &y_vec, Some(&weights)) {
                    let sub_fit: Vec<TrigTerm> = sub_terms.iter().zip(&sub_c)
                        .filter_map(|(&(kind, f), &c)| {
                            let s = snap_amplitude(c);
                            if s.abs() > 1e-6 { Some(TrigTerm::new(kind, f, s)) } else { None }
                        })
                        .collect();
                    if !sub_fit.is_empty() {
                        candidate_functions.push(TrigFunction::new(sub_fit));
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------
    // METODE 3: Fullspektret Fourier-projeksjon (for tette spektre, intervaller [[a..b]])
    // -------------------------------------------------------------
    let mut harmonic_spectrum: Vec<(TrigKind, u32, f64)> = Vec::new();
    for item in &spectrum {
        harmonic_spectrum.push((TrigKind::Sin, item.freq, item.amp_sin));
        harmonic_spectrum.push((TrigKind::Cos, item.freq, item.amp_cos));
    }

    let max_harmonic_amp = harmonic_spectrum.iter().map(|(_, _, a)| a.abs()).fold(0.0f64, f64::max);
    let fourier_threshold = (max_harmonic_amp * 0.05).min(0.10).max(1e-5);

    let mut significant_components: Vec<(TrigKind, u32, f64)> = harmonic_spectrum
        .iter()
        .filter(|(_, _, amp)| amp.abs() >= fourier_threshold)
        .copied()
        .collect();

    significant_components.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());

    if !significant_components.is_empty() {
        let top_term = TrigTerm::new(significant_components[0].0, significant_components[0].1, significant_components[0].2);
        candidate_functions.push(TrigFunction::new(vec![top_term]));

        let fourier_all: Vec<TrigTerm> = significant_components
            .iter()
            .map(|&(kind, f, amp)| TrigTerm::new(kind, f, snap_amplitude(amp)))
            .filter(|t| t.amp.abs() > 1e-6)
            .collect();
        if !fourier_all.is_empty() {
            candidate_functions.push(TrigFunction::new(fourier_all));
        }

        if significant_components.len() >= 2 && significant_components.len() <= 120 {
            let comp_cols: Vec<Vec<f64>> = significant_components.iter().map(|&(kind, f, _)| {
                x_vec.iter().map(|&x| match kind {
                    TrigKind::Sin => (f as f64 * x).sin(),
                    TrigKind::Cos => (f as f64 * x).cos(),
                }).collect()
            }).collect();

            if let Some(c_fit) = solve_least_squares(&comp_cols, &y_vec, Some(&weights)) {
                let re_fitted: Vec<TrigTerm> = significant_components.iter().zip(&c_fit)
                    .filter_map(|(&(kind, f, _), &c)| {
                        let rounded = (c * 10_000.0).round() / 10_000.0;
                        if rounded.abs() > 1e-6 { Some(TrigTerm::new(kind, f, rounded)) } else { None }
                    })
                    .collect();
                if !re_fitted.is_empty() {
                    candidate_functions.push(TrigFunction::new(re_fitted));
                }

                let re_snapped: Vec<TrigTerm> = significant_components.iter().zip(&c_fit)
                    .filter_map(|(&(kind, f, _), &c)| {
                        let s = snap_amplitude(c);
                        if s.abs() > 1e-6 { Some(TrigTerm::new(kind, f, s)) } else { None }
                    })
                    .collect();
                if !re_snapped.is_empty() {
                    candidate_functions.push(TrigFunction::new(re_snapped));
                }
            }
        }
    }

    // -------------------------------------------------------------
    // METODE 4: Trinnvis residual-regresjon (OMP) med felles sinus/cosinus-utvalg
    // -------------------------------------------------------------
    let mut residual = y_vec.clone();
    let mut selected_terms: Vec<(TrigKind, u32)> = Vec::new();
    let mut selected_cols: Vec<Vec<f64>> = Vec::new();
    let max_omp_steps = 60.min(n / 3);

    for _step in 0..max_omp_steps {
        let mut r_norm_sq = 0.0;
        for (&r, &w) in residual.iter().zip(&weights) {
            r_norm_sq += w * r * r;
        }
        let r_norm = r_norm_sq.sqrt();
        if r_norm < 1e-7 {
            break;
        }

        let mut best_score = -1.0;
        let mut best_freq = 1u32;
        let mut best_sin_dot = 0.0;
        let mut best_cos_dot = 0.0;

        for k in 1..=effective_max_n {
            let sin_selected = selected_terms.contains(&(TrigKind::Sin, k));
            let cos_selected = selected_terms.contains(&(TrigKind::Cos, k));
            if sin_selected && cos_selected {
                continue;
            }

            let mut dot_s = 0.0;
            let mut norm_s_sq = 0.0;
            let mut dot_c = 0.0;
            let mut norm_c_sq = 0.0;

            for (&x, (&r, &w)) in x_vec.iter().zip(residual.iter().zip(&weights)) {
                let angle = k as f64 * x;
                let s = angle.sin();
                let c = angle.cos();

                if !sin_selected {
                    dot_s += w * r * s;
                    norm_s_sq += w * s * s;
                }
                if !cos_selected {
                    dot_c += w * r * c;
                    norm_c_sq += w * c * c;
                }
            }

            let score_s = if norm_s_sq > 1e-9 { dot_s * dot_s / norm_s_sq } else { 0.0 };
            let score_c = if norm_c_sq > 1e-9 { dot_c * dot_c / norm_c_sq } else { 0.0 };
            let total_score = (score_s + score_c).sqrt() / r_norm;

            if total_score > best_score {
                best_score = total_score;
                best_freq = k;
                best_sin_dot = dot_s;
                best_cos_dot = dot_c;
            }
        }

        if best_score < 0.01 {
            break;
        }

        // Legg til den/de komponentene for best_freq som bidrar
        if !selected_terms.contains(&(TrigKind::Sin, best_freq)) && best_sin_dot.abs() > 1e-5 {
            let col: Vec<f64> = x_vec.iter().map(|&x| (best_freq as f64 * x).sin()).collect();
            selected_terms.push((TrigKind::Sin, best_freq));
            selected_cols.push(col);
        }
        if !selected_terms.contains(&(TrigKind::Cos, best_freq)) && best_cos_dot.abs() > 1e-5 {
            let col: Vec<f64> = x_vec.iter().map(|&x| (best_freq as f64 * x).cos()).collect();
            selected_terms.push((TrigKind::Cos, best_freq));
            selected_cols.push(col);
        }

        if let Some(coeffs) = solve_least_squares(&selected_cols, &y_vec, Some(&weights)) {
            let mut cur_rss = 0.0;
            for i in 0..n {
                let mut pred = 0.0;
                for j in 0..selected_terms.len() {
                    pred += coeffs[j] * selected_cols[j][i];
                }
                let err = y_vec[i] - pred;
                residual[i] = err;
                cur_rss += err * err;
            }

            let cur_rmse = (cur_rss / n_pts).sqrt();

            let snapped_terms: Vec<TrigTerm> = selected_terms.iter().zip(&coeffs)
                .filter_map(|(&(kind, freq), &c)| {
                    let s = snap_amplitude(c);
                    if s.abs() > 1e-6 { Some(TrigTerm::new(kind, freq, s)) } else { None }
                })
                .collect();
            if !snapped_terms.is_empty() {
                candidate_functions.push(TrigFunction::new(snapped_terms));
            }

            let fitted_terms: Vec<TrigTerm> = selected_terms.iter().zip(&coeffs)
                .filter_map(|(&(kind, freq), &c)| {
                    let rounded = (c * 10_000.0).round() / 10_000.0;
                    if rounded.abs() > 1e-6 { Some(TrigTerm::new(kind, freq, rounded)) } else { None }
                })
                .collect();
            if !fitted_terms.is_empty() {
                candidate_functions.push(TrigFunction::new(fitted_terms));
            }

            if cur_rmse < 1e-5 {
                break;
            }
        } else {
            break;
        }
    }

    // Fjern tomme eller duplikate kandidatfunksjoner
    candidate_functions.retain(|f| !f.terms.is_empty());
    candidate_functions.sort_by_key(|f| format!("{}", f));
    candidate_functions.dedup();

    // ULTIMAT SIKRING: Dersom candidate_functions mot formodning er tom,
    // ta det sterkeste spektralpunktet direkte slik at det ALLTID finnes en kandidat
    if candidate_functions.is_empty() {
        let mut sorted_spec = harmonic_spectrum.clone();
        sorted_spec.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());
        if let Some(&(kind, k, amp)) = sorted_spec.first() {
            let safe_amp = if amp.abs() > 1e-6 { amp } else { 1.0 };
            candidate_functions.push(TrigFunction::new(vec![TrigTerm::new(kind, k, safe_amp)]));
        } else {
            candidate_functions.push(TrigFunction::new(vec![TrigTerm::sin(1, 1.0)]));
        }
    }

    let mut evaluated_candidates = Vec::new();

    for func in candidate_functions {
        let mut rss = 0.0;
        let mut max_err: f64 = 0.0;

        for p in data {
            let y_pred = func.eval(p.x);
            let err = (p.y - y_pred).abs();
            rss += err.powi(2);
            if err > max_err {
                max_err = err;
            }
        }

        let mse = rss / n_pts;
        let rmse = mse.sqrt();
        let r_squared = if tss > 1e-12 {
            (1.0 - (rss / tss)).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        evaluated_candidates.push(IdentificationCandidate {
            function: func,
            r_squared,
            rmse,
            max_error: max_err,
            detected_harmonics: significant_components.clone(),
        });
    }

    // Sorter kandidater:
    // 1. Primært: Høyeste R²
    // 2. Sekundært: Ved lik R² (forskjell <= 0.0001), velg færrest ledd (Occam's razor mot overtilpasning)
    // 3. Tertiært: Lavest RMSE og maks feil
    evaluated_candidates.sort_by(|a, b| {
        // Diskrete bøtter for R² (trinn på 0.000001) for å prioritere maksimal R² helt opp til 1.0
        let bucket_a = (a.r_squared * 1_000_000.0).round() as i64;
        let bucket_b = (b.r_squared * 1_000_000.0).round() as i64;

        bucket_b.cmp(&bucket_a)
            .then_with(|| a.max_error.partial_cmp(&b.max_error).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| a.rmse.partial_cmp(&b.rmse).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| a.function.terms.len().cmp(&b.function.terms.len()))
            .then_with(|| a.function.format_expr().cmp(&b.function.format_expr()))
    });

    let best_match = evaluated_candidates
        .first()
        .cloned()
        .ok_or_else(|| "Ingen kandidater evaluert".to_string())?;

    // Unike toppkandidater (dedupliser etter streng-representasjon)
    let mut top_candidates: Vec<IdentificationCandidate> = Vec::new();
    let mut seen_exprs = std::collections::HashSet::new();

    for cand in evaluated_candidates {
        let expr = format!("{}", cand.function);
        if seen_exprs.insert(expr) {
            top_candidates.push(cand);
            if top_candidates.len() >= 8 {
                break;
            }
        }
    }

    Ok(IdentificationResult {
        best_match,
        top_candidates,
        data_points_count: data.len(),
        x_range: (min_x, max_x),
        y_range: (min_y, max_y),
    })
}

pub fn generate_synthetic_data(
    func: &TrigFunction,
    x_min: f64,
    x_max: f64,
    num_points: usize,
    noise_std: f64,
) -> Vec<DataPoint> {
    let mut data = Vec::with_capacity(num_points);
    let mut seed: u64 = 123456789;

    for i in 0..num_points {
        let t = if num_points > 1 {
            i as f64 / (num_points - 1) as f64
        } else {
            0.0
        };
        let x = x_min + t * (x_max - x_min);
        let mut y = func.eval(x);

        if noise_std > 0.0 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = ((seed >> 32) as f64 + 1.0) / 4294967296.0;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = ((seed >> 32) as f64 + 1.0) / 4294967296.0;

            let z0 = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            y += z0 * noise_std;
        }

        data.push(DataPoint { x, y });
    }

    data
}

pub fn generate_test_csv<P: AsRef<Path>>(
    path: P,
    func: &TrigFunction,
    x_min: f64,
    x_max: f64,
    num_points: usize,
    noise_std: f64,
) -> Result<(), Box<dyn Error>> {
    let data = generate_synthetic_data(func, x_min, x_max, num_points, noise_std);
    let mut file = File::create(path)?;
    writeln!(file, "x,y")?;

    for p in data {
        writeln!(file, "{:.8},{:.8}", p.x, p.y)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_csv_robustness() {
        // 1. Test standard x,y med overskrift
        let csv_standard = "x,y\n0.0,1.5\n1.0,2.5\n";
        let pts = parse_csv_string(csv_standard).unwrap();
        assert_eq!(pts.len(), 2);
        assert_eq!(pts[0].x, 0.0);
        assert_eq!(pts[0].y, 1.5);

        // 2. Test kommentarer og ekstremal-tabell format (slik som rean.csv)
        let csv_extrema = r#"# GPU Extrema Classifier
# Formel: sin(x) + cos(2x)
Indeks,Type,x_rad,x_deg,f_x,Klasse,Klasse_Beskrivelse
1,Toppunkt,0.198489,11.37,3.595866,4,"Ekstrem Topp"
2,Bunnpunkt,1.184571,67.87,-0.382907,1,"Grunn Bunn"
"#;
        let pts_extrema = parse_csv_string(csv_extrema).unwrap();
        assert_eq!(pts_extrema.len(), 2);
        assert!((pts_extrema[0].x - 0.198489).abs() < 1e-5);
        assert!((pts_extrema[0].y - 3.595866).abs() < 1e-5);
        assert!((pts_extrema[1].x - 1.184571).abs() < 1e-5);
        assert!((pts_extrema[1].y - -0.382907).abs() < 1e-5);

        // 3. Test europeisk semikolon og norsk desimalkomma
        let csv_norwegian = "x;y\n0,5;1,25\n1,5;2,75\n";
        let pts_nor = parse_csv_string(csv_norwegian).unwrap();
        assert_eq!(pts_nor.len(), 2);
        assert_eq!(pts_nor[0].x, 0.5);
        assert_eq!(pts_nor[0].y, 1.25);

        // 4. Test ren enkeltkolonne (kun y-verdier)
        let csv_single = "1.0\n2.0\n3.0\n";
        let pts_single = parse_csv_string(csv_single).unwrap();
        assert_eq!(pts_single.len(), 3);
        assert_eq!(pts_single[0].y, 1.0);
        assert_eq!(pts_single[2].y, 3.0);
    }

    #[test]
    fn test_identification_sin1_sin3_sin9() {
        let func = TrigFunction::parse("sin(x) + sin(3x) + sin(9x)").unwrap();
        let mut data = Vec::new();
        let num_points = 500;
        for i in 0..num_points {
            let x = (i as f64 / (num_points - 1) as f64) * std::f64::consts::TAU;
            let y = func.eval(x);
            data.push(DataPoint { x, y });
        }

        let res = identify_function_from_data(&data, 20).unwrap();
        assert_eq!(res.best_match.function.terms.len(), 3);
        assert!(res.best_match.r_squared > 0.999);
    }

    #[test]
    fn test_identification_mixed_sin_cos() {
        let func = TrigFunction::parse("sin(x) + cos(2x)").unwrap();
        let mut data = Vec::new();
        let num_points = 500;
        for i in 0..num_points {
            let x = (i as f64 / (num_points - 1) as f64) * std::f64::consts::TAU;
            let y = func.eval(x);
            data.push(DataPoint { x, y });
        }

        let res = identify_function_from_data(&data, 10).unwrap();
        assert_eq!(res.best_match.function.terms.len(), 2);
        assert!(res.best_match.r_squared > 0.999);
    }

    #[test]
    fn test_identification_with_constants() {
        let func = TrigFunction::parse("2*sin(x) + 3*cos(2x)").unwrap();
        let mut data = Vec::new();
        let num_points = 500;
        for i in 0..num_points {
            let x = (i as f64 / (num_points - 1) as f64) * std::f64::consts::TAU;
            let y = func.eval(x);
            data.push(DataPoint { x, y });
        }

        let res = identify_function_from_data(&data, 10).unwrap();
        assert_eq!(res.best_match.function.terms.len(), 2);
        assert!(res.best_match.r_squared > 0.999);
        assert_eq!(res.best_match.function.terms[0].amp, 2.0);
        assert_eq!(res.best_match.function.terms[1].amp, 3.0);
    }

    #[test]
    fn test_identification_bracket_cos_plus_sin() {
        let func = TrigFunction::parse("[[4,3cos(x),100]] + 5sin(x)").unwrap();
        assert_eq!(func.terms.len(), 98); // sin(1x)*5 + cos(4x..100x)*3

        let mut data = Vec::new();
        let num_points = 2000;
        for i in 0..num_points {
            let x = (i as f64 / (num_points - 1) as f64) * std::f64::consts::TAU;
            let y = func.eval(x);
            data.push(DataPoint { x, y });
        }

        let res = identify_function_from_data(&data, 105).unwrap();
        println!("Best match: {}, R2: {}, RMSE: {}, terms: {}", res.best_match.function, res.best_match.r_squared, res.best_match.rmse, res.best_match.function.terms.len());
        assert!(res.best_match.r_squared > 0.95);
    }
    #[test]
    fn test_identification_tektket_dataset() {
        let pts = read_csv_data("tektket.csv").expect("Kunne ikke lese tektket.csv");
        let res = identify_function_from_data(&pts, 256).expect("Kunne ikke identifisere tektket");
        println!("Tektket funnet: {}, R2: {}, RMSE: {}", res.best_match.function, res.best_match.r_squared, res.best_match.rmse);
        assert!(res.best_match.r_squared > 0.9999);
        assert!(res.best_match.rmse < 1e-4);
        assert_eq!(res.best_match.function.terms.len(), 4);
    }

    #[test]
    fn test_identification_deep_spikes() {
        // Syntetiser et signal med skarpe, dype spisser (topp på +4.0 og dyp bunn på -5.0)
        let num_points = 1000;
        let mut data = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let x = (i as f64 / (num_points - 1) as f64) * std::f64::consts::TAU;
            let peak = 4.0 * (-((x - 1.4) / 0.16).powi(2)).exp();
            let trough = -5.0 * (-((x - 4.1) / 0.14).powi(2)).exp();
            let y = peak + trough;
            data.push(DataPoint { x, y });
        }

        let res = identify_function_from_data(&data, 35).expect("Identifikasjon feilet for dype spisser");
        println!(
            "Dype spisser funnet: R2={:.5}, RMSE={:.4}, maks feil={:.4}, ledd={}",
            res.best_match.r_squared,
            res.best_match.rmse,
            res.best_match.max_error,
            res.best_match.function.terms.len()
        );

        // Bekreft at modellen oppnår R² > 0.985 og fanger opp de dype spissene med minimal feil
        assert!(res.best_match.r_squared > 0.985, "R² for lav for dype spisser: {}", res.best_match.r_squared);
        assert!(res.best_match.max_error < 0.35, "Maksfeil ved spiss for høy: {}", res.best_match.max_error);
    }

    #[test]
    fn test_identification_ultra_deep_spikes() {
        // Ekstremt spisst signal med høy amplitude (topp på +10.0 og dyp bunn på -12.0)
        let num_points = 1200;
        let mut data = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let x = (i as f64 / (num_points - 1) as f64) * std::f64::consts::TAU;
            let peak = 10.0 * (-((x - 1.2) / 0.10).powi(2)).exp();
            let trough = -12.0 * (-((x - 4.5) / 0.08).powi(2)).exp();
            let y = peak + trough;
            data.push(DataPoint { x, y });
        }

        let res = identify_function_from_data(&data, 50).expect("Identifikasjon feilet for ultra dype spisser");
        println!(
            "Ultra dype spisser funnet: R2={:.6}, RMSE={:.4}, maks feil={:.4}, ledd={}",
            res.best_match.r_squared,
            res.best_match.rmse,
            res.best_match.max_error,
            res.best_match.function.terms.len()
        );

        // Bekreft at modellen fanger opp de dypeste spissene helt opp mot 1 i R²
        assert!(res.best_match.r_squared > 0.990, "R² for lav for ultra dype spisser: {}", res.best_match.r_squared);
        assert!(res.best_match.max_error < 0.65, "Maksfeil ved ultra dyp spiss for høy: {}", res.best_match.max_error);
    }
}
