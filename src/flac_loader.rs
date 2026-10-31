use crate::csv_analyzer::DataPoint;
use claxon::FlacReader;
use std::f64::consts::TAU;
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct FlacMetadata {
    pub sample_rate: u32,
    pub channels: u32,
    pub bits_per_sample: u32,
    pub total_samples: u64,
    pub duration_seconds: f64,
}

#[derive(Debug, Clone)]
pub struct FlacExtractionOptions {
    /// Maksimalt antall datapunkter som leses ut (f.eks. 2000 eller 4000).
    /// 0 betyr ingen grense (alle tilgjengelige punkter).
    pub max_points: usize,
    /// Starttidspunkt i sekunder (standard: 0.0)
    pub start_seconds: f64,
    /// Valgfri maksimal varighet i sekunder (None = utvalg styrt av max_points eller filslutt)
    pub duration_seconds: Option<f64>,
    /// Skaler x slik at frekvens k i f(x) tilsvarer svingninger i Hertz (Hz):
    /// x = 2*pi*t. Hvis false: x = t (sekunder).
    pub time_as_hz_radians: bool,
}

impl Default for FlacExtractionOptions {
    fn default() -> Self {
        Self {
            max_points: 2000,
            start_seconds: 0.0,
            duration_seconds: None,
            time_as_hz_radians: true,
        }
    }
}

/// Leser metadata og konverterer en FLAC-lydfil til en liste med (x, y) datapunkter.
pub fn read_flac_data<P: AsRef<Path>>(
    path: P,
    options: &FlacExtractionOptions,
) -> Result<(FlacMetadata, Vec<DataPoint>), String> {
    let mut reader = FlacReader::open(path.as_ref())
        .map_err(|e| format!("Kunne ikke åpne FLAC-fil: {}", e))?;

    let streaminfo = reader.streaminfo();
    let sample_rate = streaminfo.sample_rate;
    let channels = streaminfo.channels;
    let bits_per_sample = streaminfo.bits_per_sample;
    let total_samples = streaminfo.samples.unwrap_or(0);
    let duration_seconds = if sample_rate > 0 {
        total_samples as f64 / sample_rate as f64
    } else {
        0.0
    };

    let metadata = FlacMetadata {
        sample_rate,
        channels,
        bits_per_sample,
        total_samples,
        duration_seconds,
    };

    if sample_rate == 0 || channels == 0 || bits_per_sample == 0 {
        return Err("Ugyldig FLAC lydstrøm: mangler gyldig sample-rate eller kanaler".into());
    }

    // Normaliseringsfaktor til [-1.0, 1.0]
    let max_val = (1i64 << (bits_per_sample - 1)) as f64;
    let norm_factor = if max_val > 0.0 { max_val } else { 1.0 };

    let start_frame = (options.start_seconds.max(0.0) * sample_rate as f64).floor() as u64;
    let max_frames = if let Some(dur) = options.duration_seconds {
        (dur * sample_rate as f64).ceil() as usize
    } else if options.max_points > 0 {
        options.max_points
    } else {
        usize::MAX
    };

    let target_count = if options.max_points > 0 {
        max_frames.min(options.max_points)
    } else {
        max_frames
    };

    let mut points = Vec::with_capacity(target_count.min(100_000));
    let mut current_frame = 0u64;

    let mut samples_iter = reader.samples();
    loop {
        // Les ett frame (alle kanaler)
        let mut sum_ch = 0.0;
        let mut got_frame = false;

        for _ in 0..channels {
            if let Some(sample_res) = samples_iter.next() {
                let sample = sample_res.map_err(|e| format!("Feil under dekoding av FLAC-prøve: {}", e))?;
                sum_ch += sample as f64;
                got_frame = true;
            } else {
                got_frame = false;
                break;
            }
        }

        if !got_frame {
            break;
        }

        if current_frame >= start_frame {
            let frame_in_window = current_frame - start_frame;
            if frame_in_window >= target_count as u64 {
                break;
            }

            let t = current_frame as f64 / sample_rate as f64;
            let x = if options.time_as_hz_radians {
                t * TAU
            } else {
                t
            };

            // Gjennomsnitt over alle kanaler (mono-miks), skalert til [-1.0, 1.0]
            let y = (sum_ch / channels as f64) / norm_factor;

            points.push(DataPoint { x, y });
        }

        current_frame += 1;
    }

    if points.is_empty() {
        return Err("Ingen datapunkter funnet i FLAC-filen i det angitte tidsintervallet".into());
    }

    Ok((metadata, points))
}

/// Eksporterer utlest FLAC-lyddata til en ren CSV-fil
pub fn export_flac_to_csv<P: AsRef<Path>>(
    points: &[DataPoint],
    path: P,
    sample_rate: u32,
    time_as_hz_radians: bool,
) -> Result<(), String> {
    let mut file = File::create(path.as_ref())
        .map_err(|e| format!("Kunne ikke opprette CSV-fil: {}", e))?;

    let mode_str = if time_as_hz_radians {
        "x = 2*pi*t (frekvens k = Hz)"
    } else {
        "x = t (sekunder)"
    };

    writeln!(file, "# FLAC Lyddata Eksport").map_err(|e| e.to_string())?;
    writeln!(file, "# Samplerate: {} Hz, Skalering: {}", sample_rate, mode_str).map_err(|e| e.to_string())?;
    writeln!(file, "t_sec,x,y").map_err(|e| e.to_string())?;

    for p in points {
        let t = if time_as_hz_radians {
            p.x / TAU
        } else {
            p.x
        };
        writeln!(file, "{:.8},{:.8},{:.8}", t, p.x, p.y).map_err(|e| e.to_string())?;
    }

    Ok(())
}

/// Søker gjennom FLAC-filen og finner første tidspunkt (i sekunder) der lydamplituden overstiger en gitt terskel (f.eks. 0.03)
pub fn find_first_active_second<P: AsRef<Path>>(path: P, threshold: f64) -> Option<f64> {
    let mut reader = FlacReader::open(path.as_ref()).ok()?;
    let streaminfo = reader.streaminfo();
    let sample_rate = streaminfo.sample_rate;
    let channels = streaminfo.channels;
    let bits_per_sample = streaminfo.bits_per_sample;
    if sample_rate == 0 || channels == 0 || bits_per_sample == 0 {
        return None;
    }
    let norm_factor = (1i64 << (bits_per_sample - 1)) as f64;
    let mut samples_iter = reader.samples();
    let mut frame_idx = 0u64;

    loop {
        let mut sum_ch = 0.0;
        let mut got_frame = false;
        for _ in 0..channels {
            if let Some(Ok(sample)) = samples_iter.next() {
                sum_ch += sample as f64;
                got_frame = true;
            } else {
                return None;
            }
        }
        if !got_frame {
            return None;
        }
        let y = (sum_ch / channels as f64) / norm_factor;
        if y.abs() >= threshold {
            return Some(frame_idx as f64 / sample_rate as f64);
        }
        frame_idx += 1;
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_test_tone_flac() {
        // test_tone.flac ble opprettet med 100 Hz sinus (amp 0.6) og 200 Hz cosinus (amp 0.3)
        if Path::new("test_tone.flac").exists() {
            let opts = FlacExtractionOptions {
                max_points: 400,
                start_seconds: 0.0,
                duration_seconds: None,
                time_as_hz_radians: true,
            };

            let (meta, pts) = read_flac_data("test_tone.flac", &opts).unwrap();
            assert_eq!(meta.sample_rate, 8000);
            assert_eq!(meta.channels, 1);
            assert!(pts.len() >= 350);

            // Test eksport til CSV
            let csv_path = "test_tone_export.csv";
            export_flac_to_csv(&pts, csv_path, meta.sample_rate, true).unwrap();
            assert!(Path::new(csv_path).exists());

            // Test analyse direkte med vår identifikator
            let res = crate::csv_analyzer::identify_function_from_data(&pts, 250).unwrap();
            println!("FLAC tone gjenkjent: {}, R2: {}, RMSE: {}", res.best_match.function, res.best_match.r_squared, res.best_match.rmse);
            assert!(res.best_match.r_squared > 0.98);

            // Sjekk at 100 Hz og 200 Hz ble funnet
            let freqs: Vec<u32> = res.best_match.function.terms.iter().map(|t| t.freq).collect();
            assert!(freqs.contains(&100), "Fant ikke 100 Hz komponent");
            assert!(freqs.contains(&200), "Fant ikke 200 Hz komponent");

            let _ = std::fs::remove_file(csv_path);
        }
    }

    #[test]
    fn test_read_synthesized_vocal_flac() {
        let path = Path::new("test_audio/vocals/vocal_a.flac");
        if path.exists() {
            let opts = FlacExtractionOptions {
                max_points: 2000,
                start_seconds: 0.1,
                duration_seconds: None,
                time_as_hz_radians: true,
            };

            let (meta, pts) = read_flac_data(path, &opts).unwrap();
            assert_eq!(meta.sample_rate, 16000);
            assert_eq!(meta.channels, 1);
            assert!(pts.len() >= 1000);

            let res = crate::csv_analyzer::identify_function_from_data(&pts, 1000).unwrap();
            assert!(res.best_match.r_squared > 0.60);

            // Sjekk at minst en av de fundamentale harmoniske (130, 260, 390, 520, 650, 780 Hz) ble funnet
            let freqs: Vec<u32> = res.best_match.function.terms.iter().map(|t| t.freq).collect();
            assert!(
                freqs.iter().any(|&f| f % 130 == 0 || (f as i32 - 130).abs() <= 5),
                "Fant ikke forventede harmoniske komponenter for vokal A: {:?}",
                freqs
            );
        }
    }
}

