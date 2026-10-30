use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtremumKind {
    Toppunkt,  // Peak / Local Maximum
    Bunnpunkt, // Trough / Local Minimum
}

impl fmt::Display for ExtremumKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExtremumKind::Toppunkt => write!(f, "Toppunkt (Maksimum)"),
            ExtremumKind::Bunnpunkt => write!(f, "Bunnpunkt (Minimum)"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassifiedExtremum {
    pub x: f32,
    pub y: f32,
    pub kind: ExtremumKind,
    pub class_index: u32,
    pub class_name: &'static str,
}

impl ClassifiedExtremum {
    pub fn new(x: f32, y: f32, kind_raw: u32, class_idx: u32) -> Self {
        let kind = if kind_raw == 1 {
            ExtremumKind::Toppunkt
        } else {
            ExtremumKind::Bunnpunkt
        };

        let class_name = match (kind, class_idx) {
            (ExtremumKind::Toppunkt, 4) => "Ekstrem Topp (y >= 2.5)",
            (ExtremumKind::Toppunkt, 3) => "Høy Topp (1.8 <= y < 2.5)",
            (ExtremumKind::Toppunkt, 2) => "Moderat Topp (1.0 <= y < 1.8)",
            (ExtremumKind::Toppunkt, 1) => "Lav Topp (0.2 <= y < 1.0)",
            (ExtremumKind::Toppunkt, 0) => "Minimal/Negativ Topp (y < 0.2)",

            (ExtremumKind::Bunnpunkt, 4) => "Ekstrem Bunn (y <= -2.5)",
            (ExtremumKind::Bunnpunkt, 3) => "Dyp Bunn (-2.5 < y <= -1.8)",
            (ExtremumKind::Bunnpunkt, 2) => "Moderat Bunn (-1.8 < y <= -1.0)",
            (ExtremumKind::Bunnpunkt, 1) => "Grunn Bunn (-1.0 < y <= -0.2)",
            (ExtremumKind::Bunnpunkt, 0) => "Minimal/Positiv Bunn (y > -0.2)",
            _ => "Ukjent",
        };

        Self {
            x,
            y,
            kind,
            class_index: class_idx,
            class_name,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct ClassificationSummary {
    pub total_peaks: usize,
    pub total_troughs: usize,

    // Peak bins (0..4)
    pub peak_counts: [usize; 5],
    // Trough bins (0..4)
    pub trough_counts: [usize; 5],

    // Antall topper som overstiger definerte terskler
    pub peak_threshold_exceeded: [usize; 4],
    // Antall bunner som går under -terskler
    pub trough_threshold_exceeded: [usize; 4],
    pub thresholds: [f32; 4],

    pub global_max: Option<(f32, f32)>,
    pub global_min: Option<(f32, f32)>,
    pub extrema: Vec<ClassifiedExtremum>,
}

impl ClassificationSummary {
    pub fn peak_to_peak(&self) -> f32 {
        match (self.global_max, self.global_min) {
            (Some((_, max_y)), Some((_, min_y))) => max_y - min_y,
            _ => 0.0,
        }
    }
}
