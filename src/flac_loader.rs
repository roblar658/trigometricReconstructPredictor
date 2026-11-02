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

/// Skriver en sekvens med normaliserte lydprøver (-1.0 til 1.0) til en tapsfri FLAC-fil.
pub fn write_flac_samples<P: AsRef<Path>>(
    samples: &[f64],
    path: P,
    sample_rate: u32,
) -> Result<(), String> {
    use flacenc::component::BitRepr;
    use flacenc::error::Verify;
    use flacenc::source::MemSource;

    if samples.is_empty() {
        return Err("Ingen lydprøver å skrive til FLAC-fil".into());
    }

    let pcm_samples: Vec<i32> = samples
        .iter()
        .map(|&s| {
            let clamped = s.clamp(-1.0, 1.0);
            (clamped * 32767.0).round() as i32
        })
        .collect();

    let channels = 1;
    let bits_per_sample = 16;
    let block_size = 4096.min(pcm_samples.len().max(16));

    let mut enc_config = flacenc::config::Encoder::default();
    enc_config.block_size = block_size;

    let config = enc_config
        .into_verified()
        .map_err(|e| format!("Ugyldig FLAC-konfigurasjon: {:?}", e))?;

    let src = MemSource::from_samples(&pcm_samples, channels, bits_per_sample, sample_rate as usize);

    let flac_stream = flacenc::encode_with_fixed_block_size(
        &config,
        src,
        block_size,
    )
    .map_err(|e| format!("Feil under FLAC-enkoding: {:?}", e))?;

    let mut sink = flacenc::bitsink::ByteSink::new();
    flac_stream
        .write(&mut sink)
        .map_err(|e| format!("Feil under skriving av FLAC-data: {:?}", e))?;

    std::fs::write(path.as_ref(), sink.as_slice())
        .map_err(|e| format!("Kunne ikke lagre FLAC-fil: {}", e))?;

    Ok(())
}

/// Eksporterer en sekvens med datapunkter (f.eks. måledata) til en tapsfri FLAC-fil.
pub fn export_points_to_flac<P: AsRef<Path>>(
    points: &[DataPoint],
    path: P,
    sample_rate: u32,
) -> Result<(), String> {
    if points.is_empty() {
        return Err("Ingen datapunkter å eksportere til FLAC".into());
    }
    let samples: Vec<f64> = points.iter().map(|p| p.y).collect();
    write_flac_samples(&samples, path, sample_rate)
}

/// Eksporterer det tilpassede resultatet fra formelgjenkjenningen f(x) evaluert over datapunktene til en tapsfri FLAC-fil.
pub fn export_reconstructed_to_flac<P: AsRef<Path>>(
    points: &[DataPoint],
    func: &crate::model::TrigFunction,
    path: P,
    sample_rate: u32,
) -> Result<(), String> {
    if points.is_empty() {
        return Err("Ingen datapunkter å rekonstruere til FLAC".into());
    }
    let samples: Vec<f64> = points.iter().map(|p| func.eval(p.x)).collect();
    write_flac_samples(&samples, path, sample_rate)
}

/// Syntetiserer en trigonometrisk funksjon f(x) for en gitt varighet (sekunder) og skriver til en tapsfri FLAC-fil.
pub fn export_function_to_flac<P: AsRef<Path>>(
    func: &crate::model::TrigFunction,
    path: P,
    sample_rate: u32,
    duration_seconds: f64,
    time_as_hz_radians: bool,
) -> Result<(), String> {
    let dur = duration_seconds.max(0.01);
    let n_samples = (dur * sample_rate as f64).round() as usize;
    if n_samples == 0 {
        return Err("Ugyldig varighet for FLAC-syntese".into());
    }

    let mut samples = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let t = i as f64 / sample_rate as f64;
        let x = if time_as_hz_radians { t * TAU } else { t };
        samples.push(func.eval(x));
    }

    write_flac_samples(&samples, path, sample_rate)
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
                start_seconds: 0.0,
                duration_seconds: None,
                time_as_hz_radians: true,
            };

            let (meta, pts) = read_flac_data(path, &opts).unwrap();
            assert_eq!(meta.sample_rate, 16000);
            assert_eq!(meta.channels, 1);
            assert!(pts.len() >= 1000);

            let res = crate::csv_analyzer::identify_function_from_data(&pts, 8000).unwrap();
            println!("Vokal A mann match: R2 = {:.5}, ledd = {}", res.best_match.r_squared, res.best_match.function.terms.len());
            assert!(res.best_match.r_squared > 0.999, "R² for lav for vokal A: {}", res.best_match.r_squared);

            // Sjekk at minst en av de fundamentale harmoniske (130, 260, 390, 520, 650, 780 Hz) ble funnet
            let freqs: Vec<u32> = res.best_match.function.terms.iter().map(|t| t.freq).collect();
            assert!(
                freqs.iter().any(|&f| f % 130 == 0 || (f as i32 - 130).abs() <= 5),
                "Fant ikke forventede harmoniske komponenter for mann vokal A: {:?}",
                freqs
            );
        }
    }

    #[test]
    fn test_read_synthesized_female_vocal_flac() {
        let path = Path::new("test_audio/vocals/vocal_female_a.flac");
        if path.exists() {
            let opts = FlacExtractionOptions {
                max_points: 2000,
                start_seconds: 0.0,
                duration_seconds: None,
                time_as_hz_radians: true,
            };

            let (meta, pts) = read_flac_data(path, &opts).unwrap();
            assert_eq!(meta.sample_rate, 16000);
            assert_eq!(meta.channels, 1);
            assert!(pts.len() >= 1000);

            let res = crate::csv_analyzer::identify_function_from_data(&pts, 8000).unwrap();
            println!("Vokal A kvinne match: R2 = {:.5}, ledd = {}", res.best_match.r_squared, res.best_match.function.terms.len());
            assert!(res.best_match.r_squared > 0.999, "R² for lav for kvinne vokal A: {}", res.best_match.r_squared);

            // For kvinnestemme med F0=220 Hz og F1=850 Hz, skal vi finne harmoniske (f.eks. 660, 880 eller 220 Hz)
            let freqs: Vec<u32> = res.best_match.function.terms.iter().map(|t| t.freq).collect();
            assert!(
                freqs.iter().any(|&f| f % 220 == 0 || (f as i32 - 220).abs() <= 5 || (f as i32 - 660).abs() <= 5 || (f as i32 - 880).abs() <= 5),
                "Fant ikke forventede harmoniske komponenter for kvinne vokal A: {:?}",
                freqs
            );
        }
    }

    #[test]
    fn test_write_and_read_flac_roundtrip() {
        let sample_rate = 16000;
        let n_samples = 800; // 0.05 sekunder
        let mut original_samples = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let t = i as f64 / sample_rate as f64;
            // 440 Hz tone med amplitude 0.75
            let y = 0.75 * (2.0 * std::f64::consts::PI * 440.0 * t).sin();
            original_samples.push(y);
        }

        let tmp_path = "test_roundtrip.flac";
        write_flac_samples(&original_samples, tmp_path, sample_rate).expect("write_flac_samples feilet");
        assert!(Path::new(tmp_path).exists());

        // Les tilbake med claxon
        let opts = FlacExtractionOptions {
            max_points: n_samples,
            start_seconds: 0.0,
            duration_seconds: None,
            time_as_hz_radians: false,
        };

        let (meta, pts) = read_flac_data(tmp_path, &opts).expect("read_flac_data feilet");
        assert_eq!(meta.sample_rate, sample_rate);
        assert_eq!(meta.channels, 1);
        assert_eq!(meta.bits_per_sample, 16);
        assert_eq!(pts.len(), n_samples);

        // Verifiser at prøvene er nøyaktige innenfor 16-bit kvantiseringsfeil (< 0.0001)
        for (i, p) in pts.iter().enumerate() {
            let orig = original_samples[i];
            let diff = (p.y - orig).abs();
            assert!(diff < 0.0001, "Sample {} avviker: orig={}, read={}, diff={}", i, orig, p.y, diff);
        }

        let _ = std::fs::remove_file(tmp_path);
    }
}

