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
    (amp * 100.0).round() / 100.0
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

    // 1. Beregn Fourier-projeksjoner for både sinus og cosinus for k = 1..=max_n
    let mut harmonic_spectrum: Vec<(TrigKind, u32, f64)> = Vec::new();
    for k in 1..=max_n {
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

    // 2. Finn de mest signifikante komponentene (amplitude >= 0.15)
    let mut significant_components: Vec<(TrigKind, u32, f64)> = harmonic_spectrum
        .iter()
        .filter(|(_, _, amp)| amp.abs() >= 0.15)
        .copied()
        .collect();

    // Sorter etter amplitude synkende
    significant_components.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());

    if significant_components.is_empty() {
        let mut all_freqs = harmonic_spectrum.clone();
        all_freqs.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());
        for item in all_freqs.into_iter().take(3) {
            if item.2.abs() > 1e-4 {
                significant_components.push(item);
            }
        }
    }

    // 3. Generer kandidat-funksjoner fra de signifikante komponentene
    let mut candidate_functions = Vec::new();

    // Kandidat A: Komponenter med tilpassede konstanter
    if !significant_components.is_empty() {
        let fitted_terms: Vec<TrigTerm> = significant_components
            .iter()
            .map(|&(kind, f, amp)| TrigTerm::new(kind, f, snap_amplitude(amp)))
            .collect();
        candidate_functions.push(TrigFunction::new(fitted_terms));

        // Enhet-amplituder (±1.0)
        let unit_terms: Vec<TrigTerm> = significant_components
            .iter()
            .map(|&(kind, f, amp)| {
                let sign = if amp < 0.0 { -1.0 } else { 1.0 };
                TrigTerm::new(kind, f, sign)
            })
            .collect();
        candidate_functions.push(TrigFunction::new(unit_terms));
    }

    // Kandidat B: Subsett av de 1, 2, 3 eller 4 mest signifikante med tilpasset amplitude
    let top_n = significant_components.len().min(5);
    for count in 1..=top_n {
        let subset_fitted: Vec<TrigTerm> = significant_components[..count]
            .iter()
            .map(|&(kind, f, amp)| TrigTerm::new(kind, f, snap_amplitude(amp)))
            .collect();
        candidate_functions.push(TrigFunction::new(subset_fitted));

        let subset_unit: Vec<TrigTerm> = significant_components[..count]
            .iter()
            .map(|&(kind, f, amp)| {
                let sign = if amp < 0.0 { -1.0 } else { 1.0 };
                TrigTerm::new(kind, f, sign)
            })
            .collect();
        candidate_functions.push(TrigFunction::new(subset_unit));
    }

    // Kandidat C: Par-kombinasjoner
    let pool: Vec<(TrigKind, u32, f64)> = significant_components.iter().take(6).copied().collect();
    for i in 0..pool.len() {
        for j in (i + 1)..pool.len() {
            let t1_fit = TrigTerm::new(pool[i].0, pool[i].1, snap_amplitude(pool[i].2));
            let t2_fit = TrigTerm::new(pool[j].0, pool[j].1, snap_amplitude(pool[j].2));
            candidate_functions.push(TrigFunction::new(vec![t1_fit, t2_fit]));

            let t1_unit = TrigTerm::new(pool[i].0, pool[i].1, if pool[i].2 < 0.0 { -1.0 } else { 1.0 });
            let t2_unit = TrigTerm::new(pool[j].0, pool[j].1, if pool[j].2 < 0.0 { -1.0 } else { 1.0 });
            candidate_functions.push(TrigFunction::new(vec![t1_unit, t2_unit]));
        }
    }

    // Fjern duplikate kandidatfunksjoner
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

    // Sorter kandidater etter høyest R² og lavest RMSE
    evaluated_candidates.sort_by(|a, b| {
        b.r_squared
            .partial_cmp(&a.r_squared)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.rmse.partial_cmp(&b.rmse).unwrap_or(std::cmp::Ordering::Equal))
    });

    let best_match = evaluated_candidates
        .first()
        .cloned()
        .ok_or_else(|| "Ingen kandidater evaluert".to_string())?;

    let top_candidates = evaluated_candidates.into_iter().take(5).collect();

    Ok(IdentificationResult {
        best_match,
        top_candidates,
        data_points_count: data.len(),
        x_range: (min_x, max_x),
        y_range: (min_y, max_y),
    })
}

/// Genererer syntetiske datapunkter (x, y) for en gitt trigonometrisk funksjon
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
}
