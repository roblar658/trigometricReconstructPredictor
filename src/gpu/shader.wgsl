struct TrigTerm {
    freq: f32,
    amp: f32,
    is_cos: u32, // 0 = sin, 1 = cos
    _pad: u32,
};

struct Params {
    num_terms: u32,
    num_samples: u32,
    max_extrema: u32,
    _pad0: u32,
    x_min: f32,
    x_max: f32,
    _pad1: vec2<f32>,
    thresholds: vec4<f32>,
    terms: array<TrigTerm, 64>,
};

struct Extremum {
    x: f32,
    y: f32,
    kind: u32,      // 1 = Toppunkt (Peak), 2 = Bunnpunkt (Trough)
    class_idx: u32, // 0..4
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> atomic_counters: array<atomic<u32>, 16>;
@group(0) @binding(2) var<storage, read_write> extremum_count: atomic<u32>;
@group(0) @binding(3) var<storage, read_write> extrema_buffer: array<Extremum>;

fn eval_f(x: f32) -> f32 {
    var val: f32 = 0.0;
    for (var i = 0u; i < params.num_terms; i = i + 1u) {
        let t = params.terms[i];
        let kx = t.freq * x;
        if (t.is_cos == 0u) {
            val += t.amp * sin(kx);
        } else {
            val += t.amp * cos(kx);
        }
    }
    return val;
}

// Analytisk førstederivert f'(x)
fn eval_df(x: f32) -> f32 {
    var dy: f32 = 0.0;
    for (var i = 0u; i < params.num_terms; i = i + 1u) {
        let t = params.terms[i];
        let f = t.freq;
        let kx = f * x;
        if (t.is_cos == 0u) {
            dy += t.amp * f * cos(kx);
        } else {
            dy -= t.amp * f * sin(kx);
        }
    }
    return dy;
}

// Analytisk andrederivert f''(x)
fn eval_d2f(x: f32) -> f32 {
    var d2y: f32 = 0.0;
    for (var i = 0u; i < params.num_terms; i = i + 1u) {
        let t = params.terms[i];
        let f = t.freq;
        let f2 = f * f;
        let kx = f * x;
        if (t.is_cos == 0u) {
            d2y -= t.amp * f2 * sin(kx);
        } else {
            d2y -= t.amp * f2 * cos(kx);
        }
    }
    return d2y;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Vi undersøker intervallet [x_idx, x_{idx+1}]
    if (idx >= params.num_samples - 1u) {
        return;
    }

    let h = (params.x_max - params.x_min) / f32(params.num_samples - 1u);
    let x_a = params.x_min + f32(idx) * h;
    let x_b = x_a + h;

    let df_a = eval_df(x_a);
    let df_b = eval_df(x_b);

    // Sjekk om førstederivert skifter fortegn
    let is_peak_crossing = (df_a > 0.0 && df_b <= 0.0);
    let is_trough_crossing = (df_a < 0.0 && df_b >= 0.0);

    if (!is_peak_crossing && !is_trough_crossing) {
        return;
    }

    // Finn nøyaktig rot med Newton-Raphson rotfinner på f'(x) = 0
    var x_root = 0.5 * (x_a + x_b);
    for (var iter = 0; iter < 4; iter = iter + 1) {
        let f_prime = eval_df(x_root);
        let f_double_prime = eval_d2f(x_root);
        if (abs(f_double_prime) > 1e-6) {
            let step = f_prime / f_double_prime;
            x_root = clamp(x_root - step, x_a, x_b);
        }
    }

    let y_ext = eval_f(x_root);
    let d2y = eval_d2f(x_root);

    // Bekreft fortegn på andrederivert
    if (is_peak_crossing && d2y < 0.0) {
        // Toppunkt (Maksimum)
        var class_idx: u32 = 0u;
        if (y_ext >= 2.5) {
            class_idx = 4u; // Ekstrem topp
        } else if (y_ext >= 1.8) {
            class_idx = 3u; // Høy topp
        } else if (y_ext >= 1.0) {
            class_idx = 2u; // Moderat topp
        } else if (y_ext >= 0.2) {
            class_idx = 1u; // Lav topp
        } else {
            class_idx = 0u; // Minimal/negativ topp
        }

        atomicAdd(&atomic_counters[class_idx], 1u);
        atomicAdd(&atomic_counters[10u], 1u); // Totalt toppunkter

        // Terskelsjekk
        if (y_ext >= params.thresholds.x) { atomicAdd(&atomic_counters[12u], 1u); }
        if (y_ext >= params.thresholds.y) { atomicAdd(&atomic_counters[13u], 1u); }
        if (y_ext >= params.thresholds.z) { atomicAdd(&atomic_counters[14u], 1u); }
        if (y_ext >= params.thresholds.w) { atomicAdd(&atomic_counters[15u], 1u); }

        let slot = atomicAdd(&extremum_count, 1u);
        if (slot < params.max_extrema) {
            extrema_buffer[slot] = Extremum(x_root, y_ext, 1u, class_idx);
        }
    } else if (is_trough_crossing && d2y > 0.0) {
        // Bunnpunkt (Minimum)
        var class_idx: u32 = 0u;
        if (y_ext <= -2.5) {
            class_idx = 4u; // Ekstrem bunn
        } else if (y_ext <= -1.8) {
            class_idx = 3u; // Dyp bunn
        } else if (y_ext <= -1.0) {
            class_idx = 2u; // Moderat bunn
        } else if (y_ext <= -0.2) {
            class_idx = 1u; // Grunn bunn
        } else {
            class_idx = 0u; // Minimal/positiv bunn
        }

        atomicAdd(&atomic_counters[5u + class_idx], 1u);
        atomicAdd(&atomic_counters[11u], 1u); // Totalt bunnpunkter

        let slot = atomicAdd(&extremum_count, 1u);
        if (slot < params.max_extrema) {
            extrema_buffer[slot] = Extremum(x_root, y_ext, 2u, class_idx);
        }
    }
}
