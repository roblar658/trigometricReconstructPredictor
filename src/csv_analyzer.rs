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
    let r_int = amp.round();
    if (amp - r_int).abs() < 0.12 {
        return r_int;
    }
    let r_half = (amp * 2.0).round() / 2.0;
    if (amp - r_half).abs() < 0.08 {
        return r_half;
    }
    let r_quarter = (amp * 4.0).round() / 4.0;
    if (amp - r_quarter).abs() < 0.05 {
        return r_quarter;
    }
    (amp * 100.0).round() / 100.0
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

    let mut candidate_functions = Vec::new();

    // -------------------------------------------------------------
    // METODE 1: Orthogonal Matching Pursuit (OMP) / Trinnvis residual-regresjon
    // Fjerner kryss-korrelasjon og sidelober for vilkårlige ikke-periodiske intervaller.
    // -------------------------------------------------------------
    let mut residual = y_vec.clone();
    let mut selected_terms: Vec<(TrigKind, u32)> = Vec::new();
    let mut selected_cols: Vec<Vec<f64>> = Vec::new();
    let max_omp_steps = 20.min(n / 3);

    for _step in 0..max_omp_steps {
        let mut best_score = -1.0;
        let mut best_term = (TrigKind::Sin, 1);

        for k in 1..=effective_max_n {
            for &kind in &[TrigKind::Sin, TrigKind::Cos] {
                if selected_terms.contains(&(kind, k)) {
                    continue;
                }

                let mut dot = 0.0;
                let mut norm_sq = 0.0;
                for (&x, (&r, &w)) in x_vec.iter().zip(residual.iter().zip(&weights)) {
                    let angle = k as f64 * x;
                    let val = match kind {
                        TrigKind::Sin => angle.sin(),
                        TrigKind::Cos => angle.cos(),
                    };
                    dot += w * r * val;
                    norm_sq += w * val * val;
                }

                let norm = norm_sq.sqrt();
                let score = if norm > 1e-9 { dot.abs() / norm } else { 0.0 };
                if score > best_score {
                    best_score = score;
                    best_term = (kind, k);
                }
            }
        }

        // Hvis ingen korrelasjon av betydning gjenstår i residualet, stopp
        if best_score < 0.04 {
            break;
        }

        let winning_col: Vec<f64> = x_vec.iter().map(|&x| {
            let angle = best_term.1 as f64 * x;
            match best_term.0 {
                TrigKind::Sin => angle.sin(),
                TrigKind::Cos => angle.cos(),
            }
        }).collect();

        selected_terms.push(best_term);
        selected_cols.push(winning_col);

        // Løs felles minste kvadraters tilpasning for alle hittil valgte ledd
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

            // Kandidat 1: Avrundede / snappede heltall- og brøkamplituder
            let snapped_terms: Vec<TrigTerm> = selected_terms
                .iter()
                .zip(&coeffs)
                .filter_map(|(&(kind, freq), &c)| {
                    let s = snap_amplitude(c);
                    if s.abs() > 1e-4 {
                        Some(TrigTerm::new(kind, freq, s))
                    } else {
                        None
                    }
                })
                .collect();
            if !snapped_terms.is_empty() {
                candidate_functions.push(TrigFunction::new(snapped_terms));
            }

            // Kandidat 2: Nøyaktige tilpassede regresjonskoeffisienter
            let fitted_terms: Vec<TrigTerm> = selected_terms
                .iter()
                .zip(&coeffs)
                .filter_map(|(&(kind, freq), &c)| {
                    let rounded = (c * 1000.0).round() / 1000.0;
                    if rounded.abs() > 1e-4 {
                        Some(TrigTerm::new(kind, freq, rounded))
                    } else {
                        None
                    }
                })
                .collect();
            if !fitted_terms.is_empty() {
                candidate_functions.push(TrigFunction::new(fitted_terms));
            }

            // Kandidat 3: Enhetsamplituder (±1.0)
            let unit_terms: Vec<TrigTerm> = selected_terms
                .iter()
                .zip(&coeffs)
                .map(|(&(kind, freq), &c)| {
                    TrigTerm::new(kind, freq, if c < 0.0 { -1.0 } else { 1.0 })
                })
                .collect();
            candidate_functions.push(TrigFunction::new(unit_terms));

            // Hvis feilen allerede er tilnærmet maskinpresisjon, er funksjonen funnet
            if cur_rmse < 1e-5 {
                break;
            }
        } else {
            break;
        }
    }

    // -------------------------------------------------------------
    // METODE 2: Fullspektret Fourier-projeksjon (for tette spektre, intervaller [[a..b]])
    // -------------------------------------------------------------
    let mut harmonic_spectrum: Vec<(TrigKind, u32, f64)> = Vec::new();
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

        let amp_sin = if sum_denom_sin > 1e-6 { sum_proj_sin / sum_denom_sin } else { 0.0 };
        let amp_cos = if sum_denom_cos > 1e-6 { sum_proj_cos / sum_denom_cos } else { 0.0 };

        harmonic_spectrum.push((TrigKind::Sin, k, amp_sin));
        harmonic_spectrum.push((TrigKind::Cos, k, amp_cos));
    }

    let mut significant_components: Vec<(TrigKind, u32, f64)> = harmonic_spectrum
        .iter()
        .filter(|(_, _, amp)| amp.abs() >= 0.15)
        .copied()
        .collect();

    significant_components.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());

    if !significant_components.is_empty() {
        // Alle signifikante ledd (essensielt for intervalluttrykk som [[4, 3cos(x), 100]])
        let fourier_all: Vec<TrigTerm> = significant_components
            .iter()
            .map(|&(kind, f, amp)| TrigTerm::new(kind, f, snap_amplitude(amp)))
            .collect();
        candidate_functions.push(TrigFunction::new(fourier_all));

        // Subsett 1..=5
        let top_n = significant_components.len().min(5);
        for count in 1..=top_n {
            let subset_fitted: Vec<TrigTerm> = significant_components[..count]
                .iter()
                .map(|&(kind, f, amp)| TrigTerm::new(kind, f, snap_amplitude(amp)))
                .collect();
            candidate_functions.push(TrigFunction::new(subset_fitted));
        }

        // Felles regresjon på komponenter dersom antallet er håndterbart
        if significant_components.len() >= 2 && significant_components.len() <= 40 {
            let comp_cols: Vec<Vec<f64>> = significant_components.iter().map(|&(kind, f, _)| {
                x_vec.iter().map(|&x| match kind {
                    TrigKind::Sin => (f as f64 * x).sin(),
                    TrigKind::Cos => (f as f64 * x).cos(),
                }).collect()
            }).collect();

            if let Some(c_fit) = solve_least_squares(&comp_cols, &y_vec, Some(&weights)) {
                let re_fitted: Vec<TrigTerm> = significant_components.iter().zip(&c_fit)
                    .filter_map(|(&(kind, f, _), &c)| {
                        let s = snap_amplitude(c);
                        if s.abs() > 1e-4 { Some(TrigTerm::new(kind, f, s)) } else { None }
                    })
                    .collect();
                if !re_fitted.is_empty() {
                    candidate_functions.push(TrigFunction::new(re_fitted));
                }
            }
        }
    }

    // Fjern tomme eller duplikate kandidatfunksjoner
    candidate_functions.retain(|f| !f.terms.is_empty());
    candidate_functions.sort_by_key(|f| format!("{}", f));
    candidate_functions.dedup();

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
        let r2_diff = b.r_squared - a.r_squared;
        if r2_diff.abs() > 1e-4 {
            b.r_squared.partial_cmp(&a.r_squared).unwrap_or(std::cmp::Ordering::Equal)
        } else {
            a.function.terms.len().cmp(&b.function.terms.len())
                .then_with(|| a.rmse.partial_cmp(&b.rmse).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.max_error.partial_cmp(&b.max_error).unwrap_or(std::cmp::Ordering::Equal))
        }
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
}
