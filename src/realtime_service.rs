//! realtime_service.rs
//!
//! Sanntids Sensor-Service for inkrementell trigonometrisk tilpasning.
//!
//! Egenskaper:
//! - Inkrementell Online Gram-matrise: Oppdaterer f(x) i O(1) tid (mikrosekunder) per punkt
//!   uten å måtte iterere over eller lagre alle historiske punkter på nytt.
//! - Støtte for mottak av enkeltverdier (1, 2, 3...) og batcher av måledata (x, y).
//! - TCP-lytter som kan startes og stoppes fra egen fane i GUI eller fra CLI.
//! - Høy sikkerhet: Binder kun til localhost (127.0.0.1) som standard, avviser overdimensjonerte
//!   pakker (DoS-beskyttelse), streng numerisk parsing uten skall/injeksjonsflater.
//! - Støtter QUIT-kommando, RESET, STATUS, BATCH og PING.

use crate::model::{TrigFunction, TrigTerm};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Resultatstatus fra sanntidstjenesten for nåværende f(x) og statistikk
#[derive(Debug, Clone, PartialEq)]
pub struct RealtimeModelStatus {
    pub count: usize,
    pub formula: String,
    pub terms_count: usize,
    pub r_squared: f64,
    pub rmse: f64,
    pub last_update_us: u128,
    pub current_func: Option<TrigFunction>,
}

impl Default for RealtimeModelStatus {
    fn default() -> Self {
        Self {
            count: 0,
            formula: "f(x) = 0".to_string(),
            terms_count: 0,
            r_squared: 0.0,
            rmse: 0.0,
            last_update_us: 0,
            current_func: None,
        }
    }
}

/// Inkrementell online Fourier-fitter basert på løpende Gram-akkumulatorer
pub struct OnlineTrigFitter {
    pub max_k: usize,
    pub dim: usize,
    pub gram: Vec<f64>, // Flatt 2D array: dim x dim (brukes for max_k <= 128)
    pub diag: Vec<f64>, // Dimensjon: dim (alltid allokert for skalering opp til 16000)
    pub rhs_b: Vec<f64>, // Dimensjon: dim
    pub count: usize,
    pub sum_y: f64,
    pub sum_y_sq: f64,
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
    pub recent_points: VecDeque<(f64, f64)>, // Ringbuffer for grafvisning (maks 2000 pkt)
    pub max_recent_points: usize,
}

impl OnlineTrigFitter {
    pub fn new(max_k: usize) -> Self {
        let max_k = max_k.clamp(4, 16000);
        let dim = 2 * max_k;
        let gram = if max_k <= 128 {
            vec![0.0; dim * dim]
        } else {
            Vec::new() // Skånsom minnebruk for k opp til 16000 (ikke 2 GB!)
        };
        Self {
            max_k,
            dim,
            gram,
            diag: vec![0.0; dim],
            rhs_b: vec![0.0; dim],
            count: 0,
            sum_y: 0.0,
            sum_y_sq: 0.0,
            min_x: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            min_y: f64::INFINITY,
            max_y: f64::NEG_INFINITY,
            recent_points: VecDeque::with_capacity(2000),
            max_recent_points: 2000,
        }
    }

    /// Nullstiller alle akkumulatorer
    pub fn reset(&mut self) {
        if !self.gram.is_empty() {
            self.gram.fill(0.0);
        }
        self.diag.fill(0.0);
        self.rhs_b.fill(0.0);
        self.count = 0;
        self.sum_y = 0.0;
        self.sum_y_sq = 0.0;
        self.min_x = f64::INFINITY;
        self.max_x = f64::NEG_INFINITY;
        self.min_y = f64::INFINITY;
        self.max_y = f64::NEG_INFINITY;
        self.recent_points.clear();
    }

    /// Inkrementell oppdatering med 1 enkeltverdi (x, y) i O(dim²) tid
    pub fn add_point(&mut self, x: f64, y: f64) -> bool {
        if !x.is_finite() || !y.is_finite() {
            return false;
        }

        self.count += 1;
        self.sum_y += y;
        self.sum_y_sq += y * y;

        if x < self.min_x { self.min_x = x; }
        if x > self.max_x { self.max_x = x; }
        if y < self.min_y { self.min_y = y; }
        if y > self.max_y { self.max_y = y; }

        if self.recent_points.len() >= self.max_recent_points {
            self.recent_points.pop_front();
        }
        self.recent_points.push_back((x, y));

        let dim = self.dim;
        if !self.gram.is_empty() {
            let mut phi = vec![0.0f64; dim];
            for k in 1..=self.max_k {
                let angle = (k as f64) * x;
                phi[2 * k - 2] = angle.sin();
                phi[2 * k - 1] = angle.cos();
            }

            // Oppdater b og Gram-matrise G
            for i in 0..dim {
                let pi = phi[i];
                self.rhs_b[i] += y * pi;
                self.diag[i] += pi * pi;
                let offset_i = i * dim;
                for j in i..dim {
                    let val = pi * phi[j];
                    self.gram[offset_i + j] += val;
                    if i != j {
                        self.gram[j * dim + i] += val;
                    }
                }
            }
        } else {
            // Skalerbar O(k) oppdatering for stor k (inntil 16000)
            for k in 1..=self.max_k {
                let angle = (k as f64) * x;
                let s = angle.sin();
                let c = angle.cos();
                let idx_s = 2 * k - 2;
                let idx_c = 2 * k - 1;
                self.rhs_b[idx_s] += y * s;
                self.rhs_b[idx_c] += y * c;
                self.diag[idx_s] += s * s;
                self.diag[idx_c] += c * c;
            }
        }

        true
    }

    /// Inkrementell oppdatering for en hel batch av måledata
    pub fn add_batch(&mut self, points: &[(f64, f64)]) -> usize {
        let mut added = 0;
        for &(x, y) in points {
            if self.add_point(x, y) {
                added += 1;
            }
        }
        added
    }

    /// Løser og returnerer gjeldende f(x) og nøyaktighetsmål (R², RMSE)
    pub fn solve_model(&self, max_terms: usize) -> RealtimeModelStatus {
        let t0 = Instant::now();
        if self.count == 0 {
            return RealtimeModelStatus::default();
        }

        let dim = self.dim;
        let mut scores = Vec::with_capacity(dim);
        if !self.gram.is_empty() {
            for i in 0..dim {
                let diag = self.gram[i * dim + i];
                if diag > 1e-9 {
                    let p = (self.rhs_b[i] * self.rhs_b[i]) / diag;
                    scores.push((p, i));
                }
            }
        } else {
            for i in 0..dim {
                let diag = self.diag[i];
                if diag > 1e-9 {
                    let p = (self.rhs_b[i] * self.rhs_b[i]) / diag;
                    scores.push((p, i));
                }
            }
        }

        // Sorter etter høyeste spektrale energi
        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let max_power = scores.first().map(|s| s.0).unwrap_or(0.0);
        if max_power < 1e-9 {
            return RealtimeModelStatus {
                count: self.count,
                formula: "f(x) = 0".to_string(),
                terms_count: 0,
                r_squared: 0.0,
                rmse: 0.0,
                last_update_us: t0.elapsed().as_micros(),
                current_func: None,
            };
        }

        // Automatisk terskel for signifikante ledd, akkurat som i trigonometrisk analyse / csv_analyzer
        // 0.25% av toppeffekt (~5% amplitude) skiller reelle ledd fra spektral lekkasje/støy
        let energy_threshold = (max_power * 0.0025).max(1e-7);

        // Øvre grense for antall ledd (inntil 100, begrenset av antall datapunkter N)
        let upper_cap = if max_terms == 0 { 100 } else { max_terms.clamp(1, 100) };
        let active_count = self.count.min(upper_cap).min(scores.len());
        if active_count == 0 {
            return RealtimeModelStatus {
                count: self.count,
                formula: "f(x) = 0".to_string(),
                terms_count: 0,
                r_squared: 0.0,
                rmse: 0.0,
                last_update_us: t0.elapsed().as_micros(),
                current_func: None,
            };
        }

        // Beregn vindulengde T fra recent_points for Rayleigh-kriterium (NMS / frekvensseparasjon)
        let rayleigh = if self.max_k > 128 && self.recent_points.len() >= 2 {
            let min_x = self.recent_points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
            let max_x = self.recent_points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
            let span = max_x - min_x;
            if span > 1e-5 {
                ((2.0 * std::f64::consts::PI / span).ceil() as usize).clamp(1, 200)
            } else {
                1
            }
        } else {
            1
        };

        // Velg ledd adaptivt: kun signifikante ledd over terskelen inkluderes
        let mut idx: Vec<usize> = Vec::with_capacity(active_count);
        for &(p, cand_idx) in &scores {
            // Avbryt når leddets energi faller under terskelen for signifikans (støy)
            if !idx.is_empty() && p < energy_threshold {
                break;
            }
            let cand_k = (cand_idx / 2) + 1;
            let cand_is_cos = cand_idx % 2 == 1;
            let mut collides = false;
            for &s_idx in &idx {
                let sk = (s_idx / 2) + 1;
                let s_is_cos = s_idx % 2 == 1;
                if sk == cand_k && s_is_cos == cand_is_cos {
                    collides = true;
                    break;
                }
                if sk != cand_k && (sk as isize - cand_k as isize).abs() < rayleigh as isize {
                    collides = true;
                    break;
                }
            }
            if !collides {
                idx.push(cand_idx);
                if idx.len() >= active_count {
                    break;
                }
            }
        }
        let m = idx.len();

        // Bygg delsystem A * c = rhs
        let mut a_mat = vec![vec![0.0f64; m]; m];
        let mut rhs = vec![0.0f64; m];

        for r in 0..m {
            let ir = idx[r];
            rhs[r] = self.rhs_b[ir];
            if !self.gram.is_empty() {
                for c in 0..m {
                    let ic = idx[c];
                    a_mat[r][c] = self.gram[ir * dim + ic];
                }
            }
        }

        if self.gram.is_empty() {
            // For stor k (inntil 16000): Beregn delsystemets Gram-matrise on-demand fra nylige punkter
            let pts_count = self.recent_points.len();
            if pts_count > 0 {
                let scale = (self.count as f64) / (pts_count as f64);
                let mut phi_active = vec![0.0f64; m];
                for &(px, _) in self.recent_points.iter() {
                    for (sub_i, &term_idx) in idx.iter().enumerate() {
                        let k = ((term_idx / 2) + 1) as f64;
                        let angle = k * px;
                        phi_active[sub_i] = if term_idx % 2 == 0 { angle.sin() } else { angle.cos() };
                    }
                    for r in 0..m {
                        let pr = phi_active[r];
                        for c in r..m {
                            let v = pr * phi_active[c];
                            a_mat[r][c] += v;
                        }
                    }
                }
                for r in 0..m {
                    for c in r..m {
                        a_mat[r][c] *= scale;
                        if r != c {
                            a_mat[c][r] = a_mat[r][c];
                        }
                    }
                }
            }
        }

        // Tikhonov regularisering for numerisk stabilitet ved faa punkter
        for r in 0..m {
            a_mat[r][r] += 1e-6;
        }

        // Gauss-Jordan eliminasjon
        for r in 0..m {
            let mut pivot = a_mat[r][r];
            if pivot.abs() < 1e-12 {
                pivot = 1e-6;
                a_mat[r][r] = pivot;
            }
            for c in r..m {
                a_mat[r][c] /= pivot;
            }
            rhs[r] /= pivot;

            for r2 in 0..m {
                if r2 != r {
                    let factor = a_mat[r2][r];
                    for c in r..m {
                        a_mat[r2][c] -= factor * a_mat[r][c];
                    }
                    rhs[r2] -= factor * rhs[r];
                }
            }
        }

        let coeffs = rhs;
        let max_coeff = coeffs.iter().map(|c| c.abs()).fold(0.0f64, f64::max);
        let coeff_thresh = (max_coeff * 0.02).max(1e-4);

        let mut c_dot_b = 0.0f64;
        let mut trig_terms = Vec::new();

        for r in 0..m {
            let c_val = coeffs[r];
            let term_idx = idx[r];

            // Akkurat som i csv_analyzer: Forkast ledd med neglisjerbar amplitude (< 2% av største ledd)
            if c_val.abs() < coeff_thresh {
                continue;
            }

            c_dot_b += c_val * self.rhs_b[term_idx];

            let rounded = (c_val * 10_000.0).round() / 10_000.0;
            if rounded.abs() > 1e-4 {
                let k = ((term_idx / 2) + 1) as u32;
                let kind = if term_idx % 2 == 0 {
                    crate::model::TrigKind::Sin
                } else {
                    crate::model::TrigKind::Cos
                };
                trig_terms.push(TrigTerm::new(kind, k, rounded));
            }
        }

        // Beregn RSS, TSS og R² direkte fra de akkumulerte summene uten å røre historiske punkter!
        let rss = (self.sum_y_sq - c_dot_b).max(0.0);
        let mean_y = self.sum_y / self.count as f64;
        let tss = (self.sum_y_sq - (self.count as f64) * mean_y * mean_y).max(1e-12);
        let r_squared = if tss > 1e-12 {
            (1.0 - (rss / tss)).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let rmse = (rss / self.count as f64).sqrt();

        let (formula_str, func_opt) = if trig_terms.is_empty() {
            ("f(x) = 0".to_string(), None)
        } else {
            let func = TrigFunction::new(trig_terms.clone());
            (format!("{}", func), Some(func))
        };

        RealtimeModelStatus {
            count: self.count,
            formula: formula_str,
            terms_count: trig_terms.len(),
            r_squared,
            rmse,
            last_update_us: t0.elapsed().as_micros(),
            current_func: func_opt,
        }
    }
}

/// Felles tilstand delt mellom TCP-serveren og brukergrensesnittet
pub struct RealtimeServiceState {
    pub fitter: OnlineTrigFitter,
    pub max_terms: usize,
    pub latest_status: RealtimeModelStatus,
    pub log_messages: VecDeque<String>,
    pub max_log_messages: usize,
}

impl RealtimeServiceState {
    pub fn new(max_k: usize) -> Self {
        Self {
            fitter: OnlineTrigFitter::new(max_k),
            max_terms: 100,
            latest_status: RealtimeModelStatus::default(),
            log_messages: VecDeque::with_capacity(100),
            max_log_messages: 100,
        }
    }

    pub fn log(&mut self, msg: String) {
        if self.log_messages.len() >= self.max_log_messages {
            self.log_messages.pop_front();
        }
        self.log_messages.push_back(msg);
    }
}

/// Kontroller for å starte, stoppe og inspisere Sanntids Sensor-Service
pub struct RealtimeServiceController {
    pub is_running: Arc<AtomicBool>,
    pub port: u16,
    pub bind_host: String,
    pub active_clients: Arc<AtomicUsize>,
    pub state: Arc<RwLock<RealtimeServiceState>>,
    listener_thread: Option<JoinHandle<()>>,
}

impl RealtimeServiceController {
    pub fn new(port: u16, max_k: usize) -> Self {
        Self {
            is_running: Arc::new(AtomicBool::new(false)),
            port,
            bind_host: "127.0.0.1".to_string(),
            active_clients: Arc::new(AtomicUsize::new(0)),
            state: Arc::new(RwLock::new(RealtimeServiceState::new(max_k))),
            listener_thread: None,
        }
    }

    /// Starter TCP-lyttetjenesten i bakgrunnen
    pub fn start(&mut self) -> Result<(), String> {
        if self.is_running.load(Ordering::SeqCst) {
            return Ok(());
        }

        let bind_addr = format!("{}:{}", self.bind_host, self.port);
        let listener = TcpListener::bind(&bind_addr)
            .map_err(|e| format!("Kunne ikke binde til {}: {}", bind_addr, e))?;

        // Sett non-blocking slik at tråden kan avbrytes rent ved stopp
        listener.set_nonblocking(true)
            .map_err(|e| format!("Kunne ikke sette non-blocking på socket: {}", e))?;

        self.is_running.store(true, Ordering::SeqCst);
        let running_flag = self.is_running.clone();
        let active_clients = self.active_clients.clone();
        let state_arc = self.state.clone();

        {
            let mut st = self.state.write().unwrap();
            st.log(format!("🟢 Tjeneste startet og lytter på {}", bind_addr));
        }

        let handle = thread::spawn(move || {
            while running_flag.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, peer_addr)) => {
                        active_clients.fetch_add(1, Ordering::SeqCst);
                        let client_running = running_flag.clone();
                        let client_active = active_clients.clone();
                        let client_state = state_arc.clone();

                        {
                            if let Ok(mut st) = client_state.write() {
                                st.log(format!("Klient tilkoblet: {}", peer_addr));
                            }
                        }

                        thread::spawn(move || {
                            handle_client(stream, peer_addr, client_running, client_state);
                            client_active.fetch_sub(1, Ordering::SeqCst);
                        });
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(20));
                    }
                    Err(_) => {
                        thread::sleep(Duration::from_millis(50));
                    }
                }
            }

            if let Ok(mut st) = state_arc.write() {
                st.log("⏹ Tjeneste stoppet".to_string());
            }
        });

        self.listener_thread = Some(handle);
        Ok(())
    }

    /// Stopper TCP-lyttetjenesten
    pub fn stop(&mut self) {
        if !self.is_running.load(Ordering::SeqCst) {
            return;
        }

        self.is_running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.listener_thread.take() {
            let _ = handle.join();
        }
    }

    /// Sender 1 eller flere testpunkter direkte (f.eks. fra GUI-knapper)
    pub fn push_point(&self, x: f64, y: f64) -> RealtimeModelStatus {
        let mut st = self.state.write().unwrap();
        st.fitter.add_point(x, y);
        let max_terms = st.max_terms;
        let status = st.fitter.solve_model(max_terms);
        st.latest_status = status.clone();
        st.log(format!("Mottok punkt ({:.3}, {:.3}) -> N={}, terms={}, R²={:.6}, RMSE={:.6}, fx=\"{}\"", x, y, status.count, status.terms_count, status.r_squared, status.rmse, status.formula));
        status
    }

    /// Sender en batch av punkter direkte
    pub fn push_batch(&self, points: &[(f64, f64)]) -> RealtimeModelStatus {
        let mut st = self.state.write().unwrap();
        let added = st.fitter.add_batch(points);
        let max_terms = st.max_terms;
        let status = st.fitter.solve_model(max_terms);
        st.latest_status = status.clone();
        st.log(format!("Mottok batch på {} punkter -> N={}, added={}, terms={}, R²={:.6}, RMSE={:.6}, fx=\"{}\"", added, status.count, added, status.terms_count, status.r_squared, status.rmse, status.formula));
        status
    }

    /// Nullstiller modellen
    pub fn reset_model(&self) {
        let mut st = self.state.write().unwrap();
        st.fitter.reset();
        st.latest_status = RealtimeModelStatus::default();
        st.log("Modell og data nullstilt".to_string());
    }

    /// Reinitialiserer fitter med ny maksimal frekvens max_k
    pub fn reinit_fitter(&self, max_k: usize) {
        let mut st = self.state.write().unwrap();
        st.fitter = OnlineTrigFitter::new(max_k);
        st.latest_status = RealtimeModelStatus::default();
        st.log(format!("Modell reinitialisert med max_k={}", max_k));
    }
}

impl Drop for RealtimeServiceController {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Håndterer HTTP GET-forespørsler og returnerer (JSON-streng, should_close_connection)
/// Trådsikker pseudo-tilfeldig tallgenerator (XorShift64)
pub fn next_random_f64() -> f64 {
    use std::cell::Cell;
    use std::time::SystemTime;

    thread_local! {
        static RNG: Cell<u64> = Cell::new({
            let nanos = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x853c49e6748fea9b);
            if nanos == 0 { 0x853c49e6748fea9b } else { nanos }
        });
    }

    RNG.with(|r| {
        let mut x = r.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        if x == 0 {
            x = 0x853c49e6748fea9b;
        }
        r.set(x);
        (x as f64) / (u64::MAX as f64)
    })
}

/// Genererer et tilfeldig testpunkt (x, y) for sanntidssimulering
pub fn generate_random_test_point(sim_x: &mut f64, uniform: bool) -> (f64, f64) {
    if uniform {
        let x = ((next_random_f64() * std::f64::consts::TAU) * 1000.0).round() / 1000.0;
        let y = (((next_random_f64() * 5.0) - 2.5) * 1000.0).round() / 1000.0;
        *sim_x = x;
        (x, y)
    } else {
        // Skritt fremover med tilfeldig tidssteg (0.05 til 0.30)
        let dx = 0.05 + 0.25 * next_random_f64();
        *sim_x += dx;
        let x = (*sim_x * 1000.0).round() / 1000.0;
        // Dynamiske tilfeldige amplituder og støy
        let amp1 = 0.7 + 1.5 * next_random_f64();
        let amp2 = 0.3 + 1.0 * next_random_f64();
        let noise = (next_random_f64() - 0.5) * 0.25;
        let y = ((amp1 * x.sin() + amp2 * (2.0 * x).cos() + noise) * 1000.0).round() / 1000.0;
        (x, y)
    }
}

pub fn handle_http_get(
    raw_path_and_query: &str,
    state: &Arc<RwLock<RealtimeServiceState>>,
) -> (String, bool) {
    let clean = raw_path_and_query.trim();
    let (path, query) = match clean.split_once('?') {
        Some((p, q)) => (p, q),
        None => (clean, ""),
    };

    let p = path.trim_matches('/');

    if p.is_empty() || p.eq_ignore_ascii_case("status") || p.eq_ignore_ascii_case("api/status") {
        let status = if let Ok(st) = state.read() {
            st.latest_status.clone()
        } else {
            RealtimeModelStatus::default()
        };
        let json = format!(
            "{{\"status\":\"ok\",\"count\":{},\"fx\":\"{}\",\"terms\":{},\"r_squared\":{:.6},\"rmse\":{:.6}}}",
            status.count, status.formula, status.terms_count, status.r_squared, status.rmse
        );
        (json, true)
    } else if p == "1" || p.eq_ignore_ascii_case("send1") {
        let mut st = state.write().unwrap();
        let mut sim_x = (st.fitter.count as f64) * 0.2;
        let (x, y) = generate_random_test_point(&mut sim_x, false);
        st.fitter.add_point(x, y);
        let max_terms = st.max_terms;
        let s = st.fitter.solve_model(max_terms);
        st.latest_status = s.clone();
        st.log(format!("HTTP GET /1 -> Tilfeldig punkt ({:.3}, {:.3}), N={}", x, y, s.count));
        let json = format!(
            "{{\"status\":\"ok\",\"count\":{},\"added\":1,\"point\":[{:.4},{:.4}],\"fx\":\"{}\",\"terms\":{},\"r_squared\":{:.6},\"rmse\":{:.6}}}",
            s.count, x, y, s.formula, s.terms_count, s.r_squared, s.rmse
        );
        (json, true)
    } else if p == "2" || p.eq_ignore_ascii_case("send2") {
        let mut st = state.write().unwrap();
        let mut sim_x = (st.fitter.count as f64) * 0.2;
        for _ in 0..2 {
            let (x, y) = generate_random_test_point(&mut sim_x, false);
            st.fitter.add_point(x, y);
        }
        let max_terms = st.max_terms;
        let s = st.fitter.solve_model(max_terms);
        st.latest_status = s.clone();
        st.log(format!("HTTP GET /2 -> 2 tilfeldige punkter lagt til, N={}", s.count));
        let json = format!(
            "{{\"status\":\"ok\",\"count\":{},\"added\":2,\"fx\":\"{}\",\"terms\":{},\"r_squared\":{:.6},\"rmse\":{:.6}}}",
            s.count, s.formula, s.terms_count, s.r_squared, s.rmse
        );
        (json, true)
    } else if p == "3" || p.eq_ignore_ascii_case("send3") {
        let mut st = state.write().unwrap();
        let mut sim_x = (st.fitter.count as f64) * 0.2;
        for _ in 0..3 {
            let (x, y) = generate_random_test_point(&mut sim_x, false);
            st.fitter.add_point(x, y);
        }
        let max_terms = st.max_terms;
        let s = st.fitter.solve_model(max_terms);
        st.latest_status = s.clone();
        st.log(format!("HTTP GET /3 -> 3 tilfeldige punkter lagt til, N={}", s.count));
        let json = format!(
            "{{\"status\":\"ok\",\"count\":{},\"added\":3,\"fx\":\"{}\",\"terms\":{},\"r_squared\":{:.6},\"rmse\":{:.6}}}",
            s.count, s.formula, s.terms_count, s.r_squared, s.rmse
        );
        (json, true)
    } else if p.eq_ignore_ascii_case("add") || p.eq_ignore_ascii_case("point") {
        let mut opt_x = None;
        let mut opt_y = None;
        for param in query.split('&') {
            if let Some((k, v)) = param.split_once('=') {
                if k.eq_ignore_ascii_case("x") {
                    opt_x = v.parse::<f64>().ok();
                } else if k.eq_ignore_ascii_case("y") {
                    opt_y = v.parse::<f64>().ok();
                }
            }
        }

        if let (Some(x), Some(y)) = (opt_x, opt_y) {
            if x.is_finite() && y.is_finite() {
                let mut st = state.write().unwrap();
                st.fitter.add_point(x, y);
                let max_terms = st.max_terms;
                let s = st.fitter.solve_model(max_terms);
                st.latest_status = s.clone();
                st.log(format!("HTTP GET /add ({:.3}, {:.3}) -> N={}", x, y, s.count));
                let json = format!(
                    "{{\"status\":\"ok\",\"count\":{},\"point\":[{:.4},{:.4}],\"fx\":\"{}\",\"terms\":{},\"r_squared\":{:.6},\"rmse\":{:.6}}}",
                    s.count, x, y, s.formula, s.terms_count, s.r_squared, s.rmse
                );
                (json, true)
            } else {
                ("{\"status\":\"error\",\"message\":\"x og y må være gyldige endelige tall (ikke NaN/Inf)\"}".to_string(), true)
            }
        } else {
            ("{\"status\":\"error\",\"message\":\"Mangler x og/eller y i spørrestrengen (f.eks. /add?x=0.5&y=1.2)\"}".to_string(), true)
        }
    } else if p.eq_ignore_ascii_case("batch") {
        let mut st = state.write().unwrap();
        let mut sim_x = (st.fitter.count as f64) * 0.2;
        for _ in 0..10 {
            let (x, y) = generate_random_test_point(&mut sim_x, false);
            st.fitter.add_point(x, y);
        }
        let max_terms = st.max_terms;
        let s = st.fitter.solve_model(max_terms);
        st.latest_status = s.clone();
        st.log(format!("HTTP GET /batch -> 10 punkter lagt til, N={}", s.count));
        let json = format!(
            "{{\"status\":\"ok\",\"count\":{},\"added\":10,\"fx\":\"{}\",\"terms\":{},\"r_squared\":{:.6},\"rmse\":{:.6}}}",
            s.count, s.formula, s.terms_count, s.r_squared, s.rmse
        );
        (json, true)
    } else if p.eq_ignore_ascii_case("terms") || p.eq_ignore_ascii_case("max_terms") {
        let mut new_terms = None;
        for param in query.split('&') {
            if let Some((k, v)) = param.split_once('=') {
                if k.eq_ignore_ascii_case("n") || k.eq_ignore_ascii_case("terms") || k.eq_ignore_ascii_case("count") {
                    new_terms = v.parse::<usize>().ok();
                }
            }
        }
        if let Some(t) = new_terms {
            let clamped = t.clamp(1, 100);
            let mut st = state.write().unwrap();
            st.max_terms = clamped;
            let s = st.fitter.solve_model(clamped);
            st.latest_status = s.clone();
            st.log(format!("HTTP GET /terms -> Maks ledd satt til {}", clamped));
            let json = format!(
                "{{\"status\":\"ok\",\"max_terms\":{},\"count\":{},\"terms\":{},\"fx\":\"{}\",\"r_squared\":{:.6},\"rmse\":{:.6}}}",
                clamped, s.count, s.terms_count, s.formula, s.r_squared, s.rmse
            );
            (json, true)
        } else {
            let cur_terms = if let Ok(st) = state.read() { st.max_terms } else { 4 };
            let json = format!("{{\"status\":\"ok\",\"max_terms\":{}}}", cur_terms);
            (json, true)
        }
    } else if p.eq_ignore_ascii_case("reset") {
        let mut st = state.write().unwrap();
        st.fitter.reset();
        st.latest_status = RealtimeModelStatus::default();
        st.log("HTTP GET /reset -> Modell nullstilt".to_string());
        ("{\"status\":\"ok\",\"count\":0,\"message\":\"Modell nullstilt\"}".to_string(), true)
    } else if p.eq_ignore_ascii_case("quit") || p.eq_ignore_ascii_case("exit") {
        let cur_count = if let Ok(st) = state.read() { st.latest_status.count } else { 0 };
        if let Ok(mut st) = state.write() {
            st.log("HTTP GET /quit mottatt".to_string());
        }
        let json = format!(
            "{{\"status\":\"bye\",\"count\":{},\"message\":\"Sesjon avsluttet\"}}",
            cur_count
        );
        (json, true)
    } else if let Some((x, y)) = parse_point(p) {
        let mut st = state.write().unwrap();
        st.fitter.add_point(x, y);
        let max_terms = st.max_terms;
        let s = st.fitter.solve_model(max_terms);
        st.latest_status = s.clone();
        st.log(format!("HTTP GET /{} -> N={}", p, s.count));
        let json = format!(
            "{{\"status\":\"ok\",\"count\":{},\"point\":[{:.4},{:.4}],\"fx\":\"{}\",\"terms\":{},\"r_squared\":{:.6},\"rmse\":{:.6}}}",
            s.count, x, y, s.formula, s.terms_count, s.r_squared, s.rmse
        );
        (json, true)
    } else {
        let json = "{\"status\":\"error\",\"message\":\"Ukjent HTTP GET endepunkt\",\"valid_endpoints\":[\"/status\",\"/1\",\"/2\",\"/3\",\"/add?x=0.5&y=1.2\",\"/batch\",\"/reset\",\"/quit\"]}".to_string();
        (json, true)
    }
}

/// Sikker og lynrask klienthåndterer for innkommende sensordata
fn handle_client(
    mut stream: TcpStream,
    peer_addr: std::net::SocketAddr,
    running_flag: Arc<AtomicBool>,
    state: Arc<RwLock<RealtimeServiceState>>,
) {
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(60)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));

    let mut reader = BufReader::new(match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    });

    let mut in_batch = false;
    let mut batch_points = Vec::new();
    let mut line_buffer = String::new();

    loop {
        if !running_flag.load(Ordering::SeqCst) {
            break;
        }

        line_buffer.clear();
        match reader.read_line(&mut line_buffer) {
            Ok(0) => {
                println!("[SERVER] Klient {} sendte Ok(0) (EOF) -> avslutter", peer_addr);
                break;
            }
            Ok(bytes_read) => {
                // Sikkerhet: Avvis overdimensjonerte linjer (beskyttelse mot buffer-overløp / minne-DoS)
                if bytes_read > 2048 {
                    let err = "{\"status\":\"error\",\"message\":\"Linje for lang (maks 2048 bytes)\"}\n";
                    let _ = stream.write_all(err.as_bytes());
                    continue;
                }

                let trimmed = line_buffer.trim();
                if trimmed.is_empty() {
                    continue;
                }

                // 0. HTTP GET-forespørsler (f.eks. fra curl, nettleser eller sensor) -> returnerer JSON
                if trimmed.starts_with("GET ") {
                    // Les eventuelle gjenværende HTTP request-headers frem til tom linje
                    let mut header_buf = String::new();
                    while let Ok(n) = reader.read_line(&mut header_buf) {
                        if n == 0 || header_buf.trim().is_empty() {
                            break;
                        }
                        header_buf.clear();
                    }

                    let after_get = trimmed[4..].trim_start();
                    let path_and_query = match after_get.find(' ') {
                        Some(idx) => &after_get[..idx],
                        None => after_get,
                    };

                    let (json_resp, _should_close) = handle_http_get(path_and_query, &state);
                    let body = format!("{}\n", json_resp);
                    let http_resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = stream.write_all(http_resp.as_bytes());
                    let _ = stream.flush();
                    break;
                }

                // 1. Støtte for QUIT og EXIT
                if trimmed.eq_ignore_ascii_case("QUIT") || trimmed.eq_ignore_ascii_case("EXIT") {
                    let cur_count = if let Ok(st) = state.read() { st.latest_status.count } else { 0 };
                    let bye_msg = format!(
                        "{{\"status\":\"bye\",\"count\":{},\"message\":\"Sesjon avsluttet\"}}\n",
                        cur_count
                    );
                    let _ = stream.write_all(bye_msg.as_bytes());
                    let _ = stream.flush();

                    if let Ok(mut st) = state.write() {
                        st.log(format!("QUIT mottatt fra {}", peer_addr));
                    }
                    break;
                }

                // Støtte for å sette eller lese maks antall ledd (inntil 100)
                if trimmed.to_ascii_uppercase().starts_with("TERMS") {
                    let parts: Vec<&str> = trimmed.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(n) = parts[1].parse::<usize>() {
                            let clamped = n.clamp(1, 100);
                            let mut st = state.write().unwrap();
                            st.max_terms = clamped;
                            let s = st.fitter.solve_model(clamped);
                            st.latest_status = s.clone();
                            st.log(format!("TCP TERMS {} mottatt -> Maks ledd satt til {}", n, clamped));
                            let resp = format!(
                                "{{\"status\":\"ok\",\"max_terms\":{},\"count\":{},\"terms\":{},\"fx\":\"{}\",\"r_squared\":{:.6},\"rmse\":{:.6}}}\n",
                                clamped, s.count, s.terms_count, s.formula, s.r_squared, s.rmse
                            );
                            let _ = stream.write_all(resp.as_bytes());
                            let _ = stream.flush();
                            continue;
                        }
                    }
                    let cur_terms = if let Ok(st) = state.read() { st.max_terms } else { 4 };
                    let resp = format!("{{\"status\":\"ok\",\"max_terms\":{}}}\n", cur_terms);
                    let _ = stream.write_all(resp.as_bytes());
                    let _ = stream.flush();
                    continue;
                }

                // 2. Støtte for STATUS
                if trimmed.eq_ignore_ascii_case("STATUS") {
                    let status = if let Ok(st) = state.read() {
                        st.latest_status.clone()
                    } else {
                        RealtimeModelStatus::default()
                    };
                    let resp = format!(
                        "{{\"status\":\"ok\",\"count\":{},\"fx\":\"{}\",\"terms\":{},\"r_squared\":{:.6},\"rmse\":{:.6}}}\n",
                        status.count, status.formula, status.terms_count, status.r_squared, status.rmse
                    );
                    let _ = stream.write_all(resp.as_bytes());
                    continue;
                }

                // Støtte for hurtigsending av 1, 2 eller 3 tilfeldige verdier over ren TCP
                if trimmed == "1" || trimmed.eq_ignore_ascii_case("SEND1") {
                    let (s, x, y) = {
                        let mut st = state.write().unwrap();
                        let mut sim_x = (st.fitter.count as f64) * 0.2;
                        let (x, y) = generate_random_test_point(&mut sim_x, false);
                        st.fitter.add_point(x, y);
                        let max_terms = st.max_terms;
                        let s = st.fitter.solve_model(max_terms);
                        st.latest_status = s.clone();
                        st.log(format!("Mottok 1 tilfeldig verdi ({:.3}, {:.3}) -> N={}, terms={}, R²={:.6}, RMSE={:.6}, fx=\"{}\"", x, y, s.count, s.terms_count, s.r_squared, s.rmse, s.formula));
                        (s, x, y)
                    };
                    let resp = format!(
                        "{{\"status\":\"ok\",\"count\":{},\"point\":[{:.4},{:.4}],\"fx\":\"{}\",\"terms\":{},\"r_squared\":{:.6},\"rmse\":{:.6}}}\n",
                        s.count, x, y, s.formula, s.terms_count, s.r_squared, s.rmse
                    );
                    let _ = stream.write_all(resp.as_bytes());
                    let _ = stream.flush();
                    continue;
                }

                if trimmed == "2" || trimmed.eq_ignore_ascii_case("SEND2") {
                    let s = {
                        let mut st = state.write().unwrap();
                        let mut sim_x = (st.fitter.count as f64) * 0.2;
                        for _ in 0..2 {
                            let (x, y) = generate_random_test_point(&mut sim_x, false);
                            st.fitter.add_point(x, y);
                        }
                        let max_terms = st.max_terms;
                        let s = st.fitter.solve_model(max_terms);
                        st.latest_status = s.clone();
                        st.log(format!("Mottok 2 tilfeldige verdier -> N={}, added=2, terms={}, R²={:.6}, RMSE={:.6}, fx=\"{}\"", s.count, s.terms_count, s.r_squared, s.rmse, s.formula));
                        s
                    };
                    let resp = format!(
                        "{{\"status\":\"ok\",\"count\":{},\"added\":2,\"fx\":\"{}\",\"terms\":{},\"r_squared\":{:.6},\"rmse\":{:.6}}}\n",
                        s.count, s.formula, s.terms_count, s.r_squared, s.rmse
                    );
                    let _ = stream.write_all(resp.as_bytes());
                    let _ = stream.flush();
                    continue;
                }

                if trimmed == "3" || trimmed.eq_ignore_ascii_case("SEND3") {
                    let s = {
                        let mut st = state.write().unwrap();
                        let mut sim_x = (st.fitter.count as f64) * 0.2;
                        for _ in 0..3 {
                            let (x, y) = generate_random_test_point(&mut sim_x, false);
                            st.fitter.add_point(x, y);
                        }
                        let max_terms = st.max_terms;
                        let s = st.fitter.solve_model(max_terms);
                        st.latest_status = s.clone();
                        st.log(format!("Mottok 3 tilfeldige verdier -> N={}, added=3, terms={}, R²={:.6}, RMSE={:.6}, fx=\"{}\"", s.count, s.terms_count, s.r_squared, s.rmse, s.formula));
                        s
                    };
                    let resp = format!(
                        "{{\"status\":\"ok\",\"count\":{},\"added\":3,\"fx\":\"{}\",\"terms\":{},\"r_squared\":{:.6},\"rmse\":{:.6}}}\n",
                        s.count, s.formula, s.terms_count, s.r_squared, s.rmse
                    );
                    let _ = stream.write_all(resp.as_bytes());
                    let _ = stream.flush();
                    continue;
                }

                // 3. Støtte for RESET
                if trimmed.eq_ignore_ascii_case("RESET") {
                    if let Ok(mut st) = state.write() {
                        st.fitter.reset();
                        st.latest_status = RealtimeModelStatus::default();
                        st.log(format!("RESET utført av {}", peer_addr));
                    }
                    let resp = "{\"status\":\"ok\",\"count\":0,\"message\":\"Modell nullstilt\"}\n";
                    let _ = stream.write_all(resp.as_bytes());
                    continue;
                }

                // 4. Støtte for PING
                if trimmed.eq_ignore_ascii_case("PING") {
                    let _ = stream.write_all(b"{\"status\":\"pong\"}\n");
                    continue;
                }

                // 5. Støtte for BATCH-blokk (BATCH ... ENDBATCH)
                if trimmed.to_uppercase().starts_with("BATCH") {
                    in_batch = true;
                    batch_points.clear();
                    continue;
                }

                if in_batch {
                    if trimmed.eq_ignore_ascii_case("ENDBATCH") {
                        in_batch = false;
                        if !batch_points.is_empty() {
                            let status = if let Ok(mut st) = state.write() {
                                st.fitter.add_batch(&batch_points);
                                let max_terms = st.max_terms;
                                let s = st.fitter.solve_model(max_terms);
                                st.latest_status = s.clone();
                                st.log(format!(
                                    "Mottok batch på {} punkter fra {} -> N={}, added={}, terms={}, R²={:.6}, RMSE={:.6}, fx=\"{}\"",
                                    batch_points.len(), peer_addr, s.count, batch_points.len(), s.terms_count, s.r_squared, s.rmse, s.formula
                                ));
                                s
                            } else {
                                RealtimeModelStatus::default()
                            };

                            let resp = format!(
                                "{{\"status\":\"ok\",\"count\":{},\"added\":{},\"fx\":\"{}\",\"terms\":{},\"r_squared\":{:.6},\"rmse\":{:.6}}}\n",
                                status.count, batch_points.len(), status.formula, status.terms_count, status.r_squared, status.rmse
                            );
                            let _ = stream.write_all(resp.as_bytes());
                        } else {
                            let _ = stream.write_all(b"{\"status\":\"ok\",\"added\":0}\n");
                        }
                        batch_points.clear();
                        continue;
                    } else if let Some((x, y)) = parse_point(trimmed) {
                        if batch_points.len() < 1000 {
                            batch_points.push((x, y));
                        }
                        continue;
                    }
                }

                // 6. JSON Batch støtte: {"batch": [[x1, y1], [x2, y2]]} eller {"points": ...}
                if trimmed.starts_with('{') {
                    if let Some(points) = parse_json_batch(trimmed) {
                        let status = if let Ok(mut st) = state.write() {
                            st.fitter.add_batch(&points);
                            let max_terms = st.max_terms;
                            let s = st.fitter.solve_model(max_terms);
                            st.latest_status = s.clone();
                            st.log(format!(
                                "Mottok batch på {} punkter fra {} -> N={}, added={}, terms={}, R²={:.6}, RMSE={:.6}, fx=\"{}\"",
                                points.len(), peer_addr, s.count, points.len(), s.terms_count, s.r_squared, s.rmse, s.formula
                            ));
                            s
                        } else {
                            RealtimeModelStatus::default()
                        };

                        let resp = format!(
                            "{{\"status\":\"ok\",\"count\":{},\"added\":{},\"fx\":\"{}\",\"terms\":{},\"r_squared\":{:.6},\"rmse\":{:.6}}}\n",
                            status.count, points.len(), status.formula, status.terms_count, status.r_squared, status.rmse
                        );
                        let _ = stream.write_all(resp.as_bytes());
                        continue;
                    }
                }

                // 7. Enkeltverdi-mottak: x,y eller JSON {"x": ..., "y": ...}
                if let Some((x, y)) = parse_point(trimmed) {
                    let status = if let Ok(mut st) = state.write() {
                        st.fitter.add_point(x, y);
                        let max_terms = st.max_terms;
                        let s = st.fitter.solve_model(max_terms);
                        st.latest_status = s.clone();
                        st.log(format!(
                            "Mottok punkt ({:.3}, {:.3}) fra {} -> N={}, terms={}, R²={:.6}, RMSE={:.6}, fx=\"{}\"",
                            x, y, peer_addr, s.count, s.terms_count, s.r_squared, s.rmse, s.formula
                        ));
                        s
                    } else {
                        RealtimeModelStatus::default()
                    };

                    let resp = format!(
                        "{{\"status\":\"ok\",\"count\":{},\"fx\":\"{}\",\"terms\":{},\"r_squared\":{:.6},\"rmse\":{:.6}}}\n",
                        status.count, status.formula, status.terms_count, status.r_squared, status.rmse
                    );
                    let _ = stream.write_all(resp.as_bytes());
                    let _ = stream.flush();
                } else {
                    let err = "{\"status\":\"error\",\"message\":\"Ugyldig format. Forventet 'x,y', 'QUIT', 'RESET' eller JSON\"}\n";
                    let _ = stream.write_all(err.as_bytes());
                    let _ = stream.flush();
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock || e.kind() == std::io::ErrorKind::TimedOut => {
                if running_flag.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(10));
                    continue;
                } else {
                    break;
                }
            }
            Err(e) => {
                if let Ok(mut st) = state.write() {
                    st.log(format!("Lesefeil fra {}: {:?}", peer_addr, e));
                }
                break;
            }
        }
    }

    if let Ok(mut st) = state.write() {
        st.log(format!("Klient frakoblet: {}", peer_addr));
    }
}

/// Sikker parsing av (x, y) koordinater fra tekst eller JSON uten eksterne injeksjonsflater
fn parse_point(input: &str) -> Option<(f64, f64)> {
    let s = input.trim();

    // 1. Enkel JSON-format: {"x": 1.23, "y": 4.56}
    if s.starts_with('{') && s.ends_with('}') {
        let mut x_val = None;
        let mut y_val = None;
        let clean = s.trim_matches(|c| c == '{' || c == '}');
        for part in clean.split(',') {
            let mut kv = part.split(':');
            let k = kv.next()?.trim().trim_matches('"');
            let v = kv.next()?.trim().trim_matches('"');
            let num: f64 = v.parse().ok()?;
            if !num.is_finite() {
                return None;
            }
            if k.eq_ignore_ascii_case("x") {
                x_val = Some(num);
            } else if k.eq_ignore_ascii_case("y") {
                y_val = Some(num);
            }
        }
        if let (Some(x), Some(y)) = (x_val, y_val) {
            return Some((x, y));
        }
    }

    // 2. CSV / semikolon / mellomrom: "x, y" eller "x; y" eller "x y"
    let delimiter = if s.contains(',') {
        ','
    } else if s.contains(';') {
        ';'
    } else {
        ' '
    };

    let mut parts = s.split(delimiter).filter(|p| !p.trim().is_empty());
    let x_str = parts.next()?.trim();
    let y_str = parts.next()?.trim();

    let x: f64 = x_str.parse().ok()?;
    let y: f64 = y_str.parse().ok()?;

    if x.is_finite() && y.is_finite() {
        Some((x, y))
    } else {
        None
    }
}

/// Sikker parsing av JSON-batch: [[x1, y1], [x2, y2]]
fn parse_json_batch(input: &str) -> Option<Vec<(f64, f64)>> {
    let s = input.trim();
    let open_idx = s.find('[')?;
    let close_idx = s.rfind(']')?;
    if close_idx <= open_idx {
        return None;
    }

    let arr_str = &s[open_idx + 1..close_idx];
    let mut points = Vec::new();

    // Splitt på lukkende klamme ']'
    for item in arr_str.split(']') {
        let clean = item.trim().trim_start_matches(|c| c == ',' || c == '[' || c == ' ');
        if clean.is_empty() {
            continue;
        }
        if let Some((x, y)) = parse_point(clean) {
            points.push((x, y));
            if points.len() >= 1000 {
                break;
            }
        }
    }

    if points.is_empty() {
        None
    } else {
        Some(points)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incremental_fitter_1_2_3_points() {
        let mut fitter = OnlineTrigFitter::new(16);

        // 1 verdi inn
        fitter.add_point(0.5, 2.0 * (3.0 * 0.5f64).sin());
        let res1 = fitter.solve_model(4);
        assert_eq!(res1.count, 1);
        println!("1 verdi: N={}, f(x)={}", res1.count, res1.formula);

        // 2 verdier inn
        fitter.add_point(1.2, 2.0 * (3.0 * 1.2f64).sin());
        let res2 = fitter.solve_model(4);
        assert_eq!(res2.count, 2);
        println!("2 verdier: N={}, f(x)={}", res2.count, res2.formula);

        // 3 verdier inn
        fitter.add_point(2.1, 2.0 * (3.0 * 2.1f64).sin());
        let res3 = fitter.solve_model(4);
        assert_eq!(res3.count, 3);
        println!("3 verdier: N={}, f(x)={}, R2={:.6}", res3.count, res3.formula, res3.r_squared);

        // Bekreft at 3x sinus komponenten er gjenkjent med høy nøyaktighet
        assert!(res3.formula.contains("sin(3x)"));
        assert!(res3.r_squared > 0.99);
    }

    #[test]
    fn test_incremental_batch_update() {
        let mut fitter = OnlineTrigFitter::new(16);

        let mut batch = Vec::new();
        for i in 0..20 {
            let x = (i as f64 / 19.0) * std::f64::consts::TAU;
            let y = 3.0 * (2.0 * x).sin() + 1.5 * (5.0 * x).cos();
            batch.push((x, y));
        }

        let added = fitter.add_batch(&batch);
        assert_eq!(added, 20);
        let res = fitter.solve_model(4);
        println!("Batch 20 pkt: f(x)={}, R2={:.6}", res.formula, res.r_squared);

        assert_eq!(res.count, 20);
        assert!(res.formula.contains("sin(2x)"));
        assert!(res.formula.contains("cos(5x)"));
        assert!(res.r_squared > 0.999);
    }

    #[test]
    fn test_point_parsing_safety() {
        assert_eq!(parse_point("0.5, 1.25"), Some((0.5, 1.25)));
        assert_eq!(parse_point("  -2.4 ; 8.9  "), Some((-2.4, 8.9)));
        assert_eq!(parse_point("{\"x\": 3.14, \"y\": -0.5}"), Some((3.14, -0.5)));

        // Avvis NaN og uendelig (injeksjonsbeskyttelse)
        assert_eq!(parse_point("NaN, 1.0"), None);
        assert_eq!(parse_point("1.0, inf"), None);
        assert_eq!(parse_point("DROP TABLE users;"), None);
        assert_eq!(parse_point("cat /etc/passwd"), None);
    }

    #[test]
    fn test_tcp_service_quit_and_single_values() {
        let port = 8799;
        let mut controller = RealtimeServiceController::new(port, 16);
        controller.start().expect("Kunne ikke starte test-tjeneste");

        // Koble til med standard TcpStream
        let mut stream = TcpStream::connect(format!("127.0.0.1:{}", port))
            .expect("Kunne ikke koble til test-server");

        // Test TERMS-kommando
        stream.write_all(b"TERMS 42\n").unwrap();
        stream.flush().unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        assert!(line.contains("\"max_terms\":42"));

        // Send 1. verdi
        stream.write_all(b"0.5, 1.0\n").unwrap();
        stream.flush().unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        assert!(line.contains("\"count\":1"));

        // Send 2. verdi
        stream.write_all(b"1.0, 2.0\n").unwrap();
        stream.flush().unwrap();
        line.clear();
        reader.read_line(&mut line).unwrap();
        assert!(line.contains("\"count\":2"));

        // Send 3. verdi
        stream.write_all(b"1.5, 1.5\n").unwrap();
        stream.flush().unwrap();
        line.clear();
        reader.read_line(&mut line).unwrap();
        assert!(line.contains("\"count\":3"));

        // Send QUIT
        stream.write_all(b"QUIT\n").unwrap();
        stream.flush().unwrap();
        line.clear();
        reader.read_line(&mut line).unwrap();
        assert!(line.contains("\"status\":\"bye\""));
        assert!(line.contains("\"count\":3"));

        controller.stop();
    }

    #[test]
    fn test_http_get_json_endpoints() {
        use std::io::Read;
        let port = 8798;
        let mut controller = RealtimeServiceController::new(port, 16);
        controller.start().expect("Kunne ikke starte test-tjeneste");

        // 0. Test GET /terms?n=35
        {
            let mut stream = TcpStream::connect(format!("127.0.0.1:{}", port)).unwrap();
            stream.write_all(b"GET /terms?n=35 HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
            stream.flush().unwrap();
            let mut reply = String::new();
            let mut reader = BufReader::new(stream);
            reader.read_to_string(&mut reply).unwrap();
            assert!(reply.contains("\"max_terms\":35"));
        }

        // 1. Test GET /status
        {
            let mut stream = TcpStream::connect(format!("127.0.0.1:{}", port)).unwrap();
            stream.write_all(b"GET /status HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
            stream.flush().unwrap();
            let mut reply = String::new();
            let mut reader = BufReader::new(stream);
            reader.read_to_string(&mut reply).unwrap();
            assert!(reply.starts_with("HTTP/1.1 200 OK"));
            assert!(reply.contains("application/json"));
            assert!(reply.contains("\"status\":\"ok\""));
        }

        // 2. Test GET /add?x=0.5&y=1.2
        {
            let mut stream = TcpStream::connect(format!("127.0.0.1:{}", port)).unwrap();
            stream.write_all(b"GET /add?x=0.5&y=1.2 HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
            stream.flush().unwrap();
            let mut reply = String::new();
            let mut reader = BufReader::new(stream);
            reader.read_to_string(&mut reply).unwrap();
            assert!(reply.contains("\"count\":1"));
            assert!(reply.contains("0.5"));
            assert!(reply.contains("1.2"));
        }

        // 3. Test GET /3
        {
            let mut stream = TcpStream::connect(format!("127.0.0.1:{}", port)).unwrap();
            stream.write_all(b"GET /3 HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
            stream.flush().unwrap();
            let mut reply = String::new();
            let mut reader = BufReader::new(stream);
            reader.read_to_string(&mut reply).unwrap();
            assert!(reply.contains("\"added\":3"));
            assert!(reply.contains("\"count\":4"));
        }

        // 4. Test GET /quit
        {
            let mut stream = TcpStream::connect(format!("127.0.0.1:{}", port)).unwrap();
            stream.write_all(b"GET /quit HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
            stream.flush().unwrap();
            let mut reply = String::new();
            let mut reader = BufReader::new(stream);
            reader.read_to_string(&mut reply).unwrap();
            assert!(reply.contains("\"status\":\"bye\""));
        }

        controller.stop();
    }
#[test]
    fn test_online_trig_fitter_k8000() {
        // Test med k helt opp til 8000 (hoyfrekvent sanntidsstrom)
        let mut fitter = OnlineTrigFitter::new(8000);
        assert_eq!(fitter.max_k, 8000);
        assert_eq!(fitter.dim, 16000);
        assert!(fitter.gram.is_empty(), "For k > 128 skal ikke 2 GB Gram-matrise allokeres");
        assert_eq!(fitter.diag.len(), 16000);
        assert_eq!(fitter.rhs_b.len(), 16000);

        // Generer punkter fra et signal med hoyfrekvente ledd: sin(1500x) og cos(4500x)
        // Nyquist-kriteriet for k=8000 krever sampling med dt < pi / 8000 ~= 0.000392.
        for i in 0..200 {
            let x = (i as f64) * 0.0002;
            let y = 3.0 * (1500.0 * x).sin() + 2.0 * (4500.0 * x).cos();
            fitter.add_point(x, y);
        }

        let res = fitter.solve_model(4);
        assert_eq!(res.count, 200);
        assert!(res.formula.contains("1500"), "Mangler sin(1500x): {}", res.formula);
        assert!(res.formula.contains("4500"), "Mangler cos(4500x): {}", res.formula);
        assert!(res.r_squared > 0.98, "R^2 var {:.4}", res.r_squared);
    }
#[test]
    fn test_max_terms_up_to_100() {
        let mut fitter = OnlineTrigFitter::new(64);
        // Generer signal med 20 ulike sinus- og cosinusledd
        for i in 0..150 {
            let x = (i as f64) * 0.05;
            let mut y = 0.0;
            for k in 1..=10 {
                y += ((k as f64) * 0.1) * (k as f64 * x).sin();
                y += ((k as f64) * 0.08) * (k as f64 * x).cos();
            }
            fitter.add_point(x, y);
        }

        // Med max_terms = 50 skal den kunne returnere inntil 20 signifikante ledd (mer enn 16)
        let res = fitter.solve_model(50);
        assert!(res.terms_count >= 18, "Forventet minst 18 ledd, fikk {}", res.terms_count);
        assert!(res.r_squared > 0.99, "R^2 var {:.4}", res.r_squared);
    }

    #[test]
    fn test_online_trig_fitter_k16000() {
        // Test med k helt opp til 16 000 (svært høyfrekvent sanntidsstrøm)
        let mut fitter = OnlineTrigFitter::new(16000);
        assert_eq!(fitter.max_k, 16000);
        assert_eq!(fitter.dim, 32000);
        assert!(fitter.gram.is_empty(), "For k > 128 skal ikke stor Gram-matrise allokeres");
        assert_eq!(fitter.diag.len(), 32000);
        assert_eq!(fitter.rhs_b.len(), 32000);

        // Generer punkter fra signal med frekvenser opp mot 16 000:
        // dt < pi / 16000 ~= 0.000196
        for i in 0..200 {
            let x = (i as f64) * 0.0001;
            let y = 3.0 * (3000.0 * x).sin() + 2.0 * (9000.0 * x).cos();
            fitter.add_point(x, y);
        }

        let res = fitter.solve_model(4);
        assert_eq!(res.count, 200);
        assert!(res.formula.contains("3000") || res.formula.contains("3001"), "Mangler sin(3000x): {}", res.formula);
        assert!(res.formula.contains("9000") || res.formula.contains("8999") || res.formula.contains("9001"), "Mangler cos(9000x): {}", res.formula);
        assert!(res.r_squared > 0.98, "R^2 var {:.4}", res.r_squared);
    }
}
