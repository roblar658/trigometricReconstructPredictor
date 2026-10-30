use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrigKind {
    Sin,
    Cos,
}

impl fmt::Display for TrigKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrigKind::Sin => write!(f, "sin"),
            TrigKind::Cos => write!(f, "cos"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrigTerm {
    pub kind: TrigKind,
    pub freq: u32,
    pub amp: f64,
}

#[allow(dead_code)]
impl TrigTerm {
    pub fn new(kind: TrigKind, freq: u32, amp: f64) -> Self {
        Self { kind, freq, amp }
    }

    pub fn sin(freq: u32, amp: f64) -> Self {
        Self { kind: TrigKind::Sin, freq, amp }
    }

    pub fn cos(freq: u32, amp: f64) -> Self {
        Self { kind: TrigKind::Cos, freq, amp }
    }
}

// Bakoverkompatibel alias
#[allow(dead_code)]
pub type SineTerm = TrigTerm;

/// Representerer en vilkårlig sum av sinus- og cosinusledd:
/// f(x) = a1*sin(n1*x) + b1*cos(m1*x) + ...
/// f.eks. f(x) = sin(x) + cos(2x) + sin(9x)
/// MERK: KUN sin og cos støttes!
#[derive(Debug, Clone, PartialEq)]
pub struct TrigFunction {
    pub terms: Vec<TrigTerm>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FormattedSegment {
    pub text: String,
    pub is_negative: bool,
}

// Bakoverkompatibel alias
pub type SineFunction = TrigFunction;

impl TrigFunction {
    pub fn new(terms: Vec<TrigTerm>) -> Self {
        let mut s = Self { terms };
        s.normalize();
        s
    }

    /// Oppretter en funksjon fra en liste med sinus-frekvenser med amplitude 1.0
    #[allow(dead_code)]
    pub fn from_frequencies(freqs: &[u32]) -> Self {
        let terms = freqs.iter().map(|&f| TrigTerm::sin(f, 1.0)).collect();
        Self::new(terms)
    }

    /// Oppretter en funksjon fra en liste med (TrigKind, frekvens)-par
    #[allow(dead_code)]
    pub fn from_trig_terms(items: &[(TrigKind, u32)]) -> Self {
        let terms = items.iter().map(|&(kind, f)| TrigTerm::new(kind, f, 1.0)).collect();
        Self::new(terms)
    }

    /// Parser en formel-streng som:
    /// "sin(x) + cos(2x) + sin(9x)"
    /// "[[5,sin(x),10]]"  (ekspanderer til sin(5x) + ... + sin(10x))
    /// "[[1,cos(x),5]]"   (ekspanderer til cos(x) + ... + cos(5x))
    /// "sin(x) + [[2,cos(x),6,2]]"
    /// "f(x) = cos(x) - sin(3x)"
    ///
    /// MERK: KUN sinus (sin) og cosinus (cos) støttes.
    pub fn parse(input: &str) -> Result<Self, String> {
        let cleaned = input.trim().trim_matches('"').trim_matches('\'').trim();
        if cleaned.is_empty() {
            return Err("Tom funksjonsstreng".into());
        }

        // Ekspander intervaller av typen [[start, sin(x)/cos(x), slutt]]
        let expanded_expr = Self::expand_bracket_ranges(cleaned)?;

        // Fjern eventuell "f(x) =" eller "y =" prefiks
        let expr = if let Some(idx) = expanded_expr.find('=') {
            &expanded_expr[idx + 1..]
        } else {
            &expanded_expr
        }
        .trim();

        if expr.is_empty() {
            return Err("Ingen uttrykk etter likhetstegnet".into());
        }

        // Kontroller for eksplisitt ikke-støttede matematiske funksjoner
        let lower = expr.to_lowercase();
        for unsupported in &["tan", "cot", "sec", "csc", "exp", "log", "ln", "sqrt", "asin", "acos", "atan"] {
            if lower.contains(unsupported) {
                return Err(format!(
                    "Ugyldig funksjon '{}': KUN sinus (sin) og cosinus (cos) støttes!",
                    unsupported
                ));
            }
        }

        // Splitt på ledd. For å håndtere både + og - erstatter vi '-' med '+-'
        let mut normalized_expr = String::new();
        for (i, ch) in expr.chars().enumerate() {
            if ch == '-' && i > 0 && expr.as_bytes()[i - 1] != b'+' && expr.as_bytes()[i - 1] != b'(' {
                normalized_expr.push('+');
                normalized_expr.push('-');
            } else {
                normalized_expr.push(ch);
            }
        }

        let mut terms = Vec::new();

        for part in normalized_expr.split('+') {
            let token = part.trim();
            if token.is_empty() {
                continue;
            }

            let term = Self::parse_single_term(token)
                .map_err(|e| format!("Feil i ledd '{}': {}", token, e))?;
            terms.push(term);
        }

        if terms.is_empty() {
            return Err("Fant ingen gyldige sinus- eller cosinus-ledd i formelen. KUN sin og cos støttes!".into());
        }

        Ok(Self::new(terms))
    }

    /// Ekspanderer intervallsyntaks av typen [[start, sin(x), slutt]] eller [[start, cos(x), slutt]]
    /// f.eks. "[[5,sin(x),10]]" -> "sin(5x) + sin(6x) + ... + sin(10x)"
    /// Ekspanderer intervallsyntaks av typen [[start, sin(x), slutt]] eller [[start, cos(x), slutt]]
    /// f.eks. "[[5,sin(x),10]]" -> "sin(5x) + sin(6x) + ... + sin(10x)"
    /// f.eks. "[[1,cos(x),4]]"  -> "cos(1x) + cos(2x) + cos(3x) + cos(4x)"
    /// Støtter også konstanter ganget med ledd:
    /// f.eks. "2 * [[1, sin(x), 4]]" eller "[[1, 2*sin(x), 4]]"
    pub fn expand_bracket_ranges(input: &str) -> Result<String, String> {
        let mut result = String::new();
        let mut text = input;

        while let Some(start_pos) = text.find("[[") {
            let before = &text[..start_pos];
            let after_start = &text[start_pos + 2..];
            let end_pos = after_start
                .find("]]")
                .ok_or_else(|| "Mangler avsluttende ']]' i intervall-uttrykket".to_string())?;

            let inner = &after_start[..end_pos].trim();
            let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
            if parts.len() < 3 {
                return Err(format!(
                    "Ugyldig intervallformat '[[{}]]'. Forventet '[[start, sin(x)/cos(x), slutt]]' (KUN sin og cos støttes)",
                    inner
                ));
            }

            let start_n: u32 = parts[0]
                .parse()
                .map_err(|_| format!("Ugyldig startfrekvens '{}' i [[{}]]", parts[0], inner))?;

            // Sjekk om inner funksjonsledd (parts[1]) har konstant eller fortegn
            let (trig_kind, inner_amp) = Self::parse_bracket_fn_part(parts[1], inner)?;

            let end_n: u32 = parts[2]
                .parse()
                .map_err(|_| format!("Ugyldig sluttfrekvens '{}' i [[{}]]", parts[2], inner))?;

            if start_n > end_n {
                return Err(format!(
                    "Startfrekvens ({}) kan ikke være større enn sluttfrekvens ({}) i [[{}]]",
                    start_n, end_n, inner
                ));
            }

            let step: u32 = if parts.len() >= 4 {
                parts[3]
                    .parse()
                    .map_err(|_| format!("Ugyldig stegverdi '{}' i [[{}]]", parts[3], inner))?
            } else {
                1
            };

            if step == 0 {
                return Err("Stegverdi i intervall kan ikke være 0".into());
            }

            // Sjekk om det er en ytre multiplikator foran "[[" i `before`
            let (prefix_keep, outer_amp) = Self::extract_outer_multiplier(before);
            let total_amp = outer_amp * inner_amp;

            let mut expanded_terms = Vec::new();
            let mut curr = start_n;
            while curr <= end_n {
                let func_name = match trig_kind {
                    TrigKind::Sin => "sin",
                    TrigKind::Cos => "cos",
                };
                let freq_str = if curr == 1 {
                    "x".to_string()
                } else {
                    format!("{}x", curr)
                };

                let term_str = if (total_amp - 1.0).abs() < 1e-6 {
                    format!("{}({})", func_name, freq_str)
                } else if (total_amp + 1.0).abs() < 1e-6 {
                    format!("-{}({})", func_name, freq_str)
                } else {
                    format!("{}*{}({})", total_amp, func_name, freq_str)
                };
                expanded_terms.push(term_str);
                curr += step;
            }

            result.push_str(&prefix_keep);
            if !prefix_keep.is_empty()
                && !prefix_keep.trim_end().ends_with('+')
                && !prefix_keep.trim_end().ends_with('-')
                && !prefix_keep.trim_end().ends_with('=')
            {
                result.push_str(" + ");
            }

            let res_trimmed = result.trim_end();
            if !res_trimmed.is_empty()
                && !res_trimmed.ends_with('+')
                && !res_trimmed.ends_with('-')
                && !res_trimmed.ends_with('=')
            {
                result.push_str(" + ");
            }

            result.push_str(&expanded_terms.join(" + "));

            text = &after_start[end_pos + 2..];
        }

        result.push_str(text);
        Ok(result)
    }

    fn parse_bracket_fn_part(fn_raw: &str, inner: &str) -> Result<(TrigKind, f64), String> {
        let lower = fn_raw.to_lowercase();
        let is_sin = lower.contains("sin");
        let is_cos = lower.contains("cos");

        if !is_sin && !is_cos {
            return Err(format!(
                "Ugyldig funksjon '{}' i intervall [[{}]]. KUN 'sin(x)' eller 'cos(x)' støttes!",
                fn_raw, inner
            ));
        }
        if is_sin && is_cos {
            return Err(format!(
                "Ledd '{}' i intervall [[{}]] kan ikke inneholde både 'sin' og 'cos'.",
                fn_raw, inner
            ));
        }

        let (kind, pos) = if is_sin {
            (TrigKind::Sin, lower.find("sin").unwrap())
        } else {
            (TrigKind::Cos, lower.find("cos").unwrap())
        };

        let before = lower[..pos]
            .trim()
            .trim_end_matches('*')
            .trim()
            .replace(',', ".")
            .replace(' ', "");

        let mut amp = if before.is_empty() || before == "+" {
            1.0
        } else if before == "-" {
            -1.0
        } else {
            before.parse::<f64>().map_err(|_| {
                format!("Ugyldig konstant '{}' i ledd '{}' i [[{}]]", before, fn_raw, inner)
            })?
        };

        let rest = lower[pos + 3..].trim();
        if let Some(close_idx) = rest.find(')') {
            let after = rest[close_idx + 1..]
                .trim()
                .trim_start_matches('*')
                .trim()
                .replace(',', ".")
                .replace(' ', "");
            if !after.is_empty() {
                let right_factor = after.parse::<f64>().map_err(|_| {
                    format!("Ugyldig faktor '{}' etter ')' i [[{}]]", after, inner)
                })?;
                amp *= right_factor;
            }
        }

        Ok((kind, amp))
    }

    fn extract_outer_multiplier(before: &str) -> (String, f64) {
        let mut trimmed = before.trim_end();
        if trimmed.is_empty() {
            return (String::new(), 1.0);
        }

        // Håndter kommaseparerte intervaller, f.eks. [[1,sin(x),3]], [[4,cos(x),6]]
        if trimmed.ends_with(',') {
            trimmed = trimmed[..trimmed.len() - 1].trim_end();
        }

        if trimmed.is_empty() {
            return (String::new(), 1.0);
        }

        let without_star = if trimmed.ends_with('*') {
            trimmed[..trimmed.len() - 1].trim_end()
        } else {
            trimmed
        };

        let mut idx = without_star.len();
        let bytes = without_star.as_bytes();

        while idx > 0 {
            let b = bytes[idx - 1];
            if b.is_ascii_digit() || b == b'.' || b == b',' {
                idx -= 1;
            } else {
                break;
            }
        }

        if idx < without_star.len() {
            let num_part = &without_star[idx..];
            let num_val = num_part.replace(',', ".").parse::<f64>();

            if let Ok(mut val) = num_val {
                let mut prefix_end = idx;
                let before_num = without_star[..idx].trim_end();
                if before_num.ends_with('-') {
                    val = -val;
                    prefix_end = before_num.len() - 1;
                } else if before_num.ends_with('+') {
                    prefix_end = before_num.len() - 1;
                }

                let prefix = without_star[..prefix_end].trim_end().to_string();
                return (prefix, val);
            }
        }

        if without_star.ends_with('-') {
            let prefix = without_star[..without_star.len() - 1].trim_end().to_string();
            return (prefix, -1.0);
        } else if without_star.ends_with('+') {
            let prefix = without_star[..without_star.len() - 1].trim_end().to_string();
            return (prefix, 1.0);
        }

        (before.to_string(), 1.0)
    }

    fn parse_single_term(token: &str) -> Result<TrigTerm, String> {
        let s = token.to_lowercase();
        let is_sin = s.contains("sin");
        let is_cos = s.contains("cos");

        if !is_sin && !is_cos {
            return Err(format!(
                "Ukjent funksjonstype. KUN sinus (sin) og cosinus (cos) støttes! Fant: '{}'",
                token
            ));
        }

        if is_sin && is_cos {
            return Err(format!(
                "Ledd '{}' kan ikke inneholde både 'sin' og 'cos' samtidig.",
                token
            ));
        }

        let (kind, pos) = if is_sin {
            (TrigKind::Sin, s.find("sin").unwrap())
        } else {
            (TrigKind::Cos, s.find("cos").unwrap())
        };

        // 1. Amplitude før 'sin' eller 'cos'
        let before_raw = s[..pos].trim();
        let before_clean = if before_raw.ends_with('*') {
            before_raw.trim_end_matches('*').trim()
        } else {
            before_raw
        }
        .replace(',', ".")
        .replace(' ', "");

        let mut amp = if before_clean.is_empty() || before_clean == "+" {
            1.0
        } else if before_clean == "-" {
            -1.0
        } else {
            let mut prod = 1.0;
            for part in before_clean.split('*') {
                if !part.is_empty() {
                    let factor = part.parse::<f64>().map_err(|_| {
                        format!("Ugyldig konstant/amplitude '{}' foran {}", part, token)
                    })?;
                    prod *= factor;
                }
            }
            prod
        };

        // 2. Innhold inne i og etter sin(...) / cos(...)
        let rest = s[pos + 3..].trim();
        let (inside_content, after_close) = if rest.starts_with('(') {
            let close = rest.find(')').ok_or_else(|| "Mangler avsluttende ')'".to_string())?;
            (rest[1..close].trim(), rest[close + 1..].trim())
        } else {
            (rest, "")
        };

        // Multiplikator etter funksjonen, f.eks. "sin(x) * 2.5"
        if !after_close.is_empty() {
            let clean_after = after_close
                .trim_start_matches('*')
                .trim()
                .replace(',', ".")
                .replace(' ', "");
            if !clean_after.is_empty() {
                for part in clean_after.split('*') {
                    if !part.is_empty() {
                        let factor = part.parse::<f64>().map_err(|_| {
                            format!("Ugyldig konstant/faktor '{}' etter {}", part, token)
                        })?;
                        amp *= factor;
                    }
                }
            }
        }

        // 3. Frekvens inne i parentesen
        let clean_inside = inside_content.replace(' ', "");
        let without_x = clean_inside.replace('x', "");
        let freq_str = without_x.trim_matches('*').trim();

        let freq = if freq_str.is_empty() {
            1
        } else {
            freq_str
                .parse::<u32>()
                .map_err(|_| format!("Ugyldig frekvens '{}' for x i {}", inside_content, token))?
        };

        if freq == 0 {
            return Err("Frekvens kan ikke være 0".into());
        }

        Ok(TrigTerm::new(kind, freq, amp))
    }

    /// Normaliserer: kombinerer ledd med samme funksjon og frekvens, og sorterer
    pub fn normalize(&mut self) {
        self.terms.sort_by(|a, b| a.freq.cmp(&b.freq).then_with(|| a.kind.cmp(&b.kind)));
        let mut combined: Vec<TrigTerm> = Vec::new();
        for t in &self.terms {
            if let Some(last) = combined.last_mut() {
                if last.freq == t.freq && last.kind == t.kind {
                    last.amp += t.amp;
                    continue;
                }
            }
            if t.amp.abs() > 1e-6 {
                combined.push(*t);
            }
        }
        self.terms = combined;
    }

    /// Evaluerer f(x)
    #[inline]
    pub fn eval(&self, x: f64) -> f64 {
        let mut y = 0.0;
        for t in &self.terms {
            let angle = t.freq as f64 * x;
            match t.kind {
                TrigKind::Sin => y += t.amp * angle.sin(),
                TrigKind::Cos => y += t.amp * angle.cos(),
            }
        }
        y
    }

    /// Førstederivert f'(x)
    #[allow(dead_code)]
    #[inline]
    pub fn derivative1(&self, x: f64) -> f64 {
        let mut dy = 0.0;
        for t in &self.terms {
            let f = t.freq as f64;
            let angle = f * x;
            match t.kind {
                TrigKind::Sin => dy += t.amp * f * angle.cos(),
                TrigKind::Cos => dy -= t.amp * f * angle.sin(),
            }
        }
        dy
    }

    /// Andrederivert f''(x)
    #[allow(dead_code)]
    #[inline]
    pub fn derivative2(&self, x: f64) -> f64 {
        let mut d2y = 0.0;
        for t in &self.terms {
            let f = t.freq as f64;
            let angle = f * x;
            let f2 = f * f;
            match t.kind {
                TrigKind::Sin => d2y -= t.amp * f2 * angle.sin(),
                TrigKind::Cos => d2y -= t.amp * f2 * angle.cos(),
            }
        }
        d2y
    }

    #[allow(dead_code)]
    pub fn is_valid(&self) -> bool {
        !self.terms.is_empty()
    }

    /// Henter en liste over formaterte segmenter (enkeltledd eller komprimerte intervaller).
    pub fn to_formatted_segments(&self) -> Vec<FormattedSegment> {
        if self.terms.is_empty() {
            return Vec::new();
        }

        let mut sin_terms: Vec<&TrigTerm> = self.terms.iter().filter(|t| t.kind == TrigKind::Sin).collect();
        let mut cos_terms: Vec<&TrigTerm> = self.terms.iter().filter(|t| t.kind == TrigKind::Cos).collect();

        sin_terms.sort_by_key(|t| t.freq);
        cos_terms.sort_by_key(|t| t.freq);

        let compress_kind = |terms: Vec<&TrigTerm>, kind_str: &str| -> Vec<FormattedSegment> {
            let mut segments = Vec::new();
            let mut i = 0;
            while i < terms.len() {
                let mut j = i + 1;
                let target_amp = terms[i].amp;
                while j < terms.len()
                    && terms[j].freq == terms[j - 1].freq + 1
                    && (terms[j].amp - target_amp).abs() < 1e-4
                {
                    j += 1;
                }

                let run_len = j - i;
                if run_len >= 3 {
                    let start_f = terms[i].freq;
                    let end_f = terms[j - 1].freq;
                    let is_neg = target_amp < 0.0;
                    let abs_amp = target_amp.abs();

                    let format_amp = |a: f64| -> String {
                        if (a - 1.0).abs() < 1e-4 {
                            String::new()
                        } else if (a - a.round()).abs() < 1e-4 {
                            format!("{}*", a.round() as i64)
                        } else if a >= 0.095 {
                            let s = format!("{:.2}", a);
                            let t = s.trim_end_matches('0').trim_end_matches('.');
                            format!("{}*", t)
                        } else if a >= 0.00095 {
                            let s = format!("{:.4}", a);
                            let t = s.trim_end_matches('0').trim_end_matches('.');
                            format!("{}*", t)
                        } else {
                            let s = format!("{:.6}", a);
                            let t = s.trim_end_matches('0').trim_end_matches('.');
                            format!("{}*", t)
                        }
                    };

                    let fn_part = format!("{}{}(x)", format_amp(abs_amp), kind_str);

                    segments.push(FormattedSegment {
                        text: format!("[[{}, {}, {}]]", start_f, fn_part, end_f),
                        is_negative: is_neg,
                    });
                    i = j;
                } else {
                    let t = terms[i];
                    let is_neg = t.amp < 0.0;
                    let abs_amp = t.amp.abs();
                    let fn_part = if t.freq == 1 {
                        format!("{}(x)", kind_str)
                    } else {
                        format!("{}({}x)", kind_str, t.freq)
                    };

                    let format_amp = |a: f64| -> String {
                        if (a - 1.0).abs() < 1e-4 {
                            String::new()
                        } else if (a - a.round()).abs() < 1e-4 {
                            format!("{}*", a.round() as i64)
                        } else if a >= 0.095 {
                            let s = format!("{:.2}", a);
                            let t = s.trim_end_matches('0').trim_end_matches('.');
                            format!("{}*", t)
                        } else if a >= 0.00095 {
                            let s = format!("{:.4}", a);
                            let t = s.trim_end_matches('0').trim_end_matches('.');
                            format!("{}*", t)
                        } else {
                            let s = format!("{:.6}", a);
                            let t = s.trim_end_matches('0').trim_end_matches('.');
                            format!("{}*", t)
                        }
                    };

                    let txt = format!("{}{}", format_amp(abs_amp), fn_part);

                    segments.push(FormattedSegment {
                        text: txt,
                        is_negative: is_neg,
                    });
                    i += 1;
                }
            }
            segments
        };

        let mut all_segs = Vec::new();
        all_segs.extend(compress_kind(sin_terms, "sin"));
        all_segs.extend(compress_kind(cos_terms, "cos"));
        all_segs
    }

    /// Formaterer funksjonsuttrykket på én enkelt linje.
    pub fn format_expr(&self) -> String {
        let all_segs = self.to_formatted_segments();
        if all_segs.is_empty() {
            return "0".to_string();
        }

        let mut res = String::new();
        for (idx, seg) in all_segs.iter().enumerate() {
            if idx == 0 {
                if seg.is_negative {
                    res.push_str("- ");
                }
                res.push_str(&seg.text);
            } else {
                if seg.is_negative {
                    res.push_str(" - ");
                } else {
                    res.push_str(" + ");
                }
                res.push_str(&seg.text);
            }
        }

        res
    }

    /// Formaterer funksjonsuttrykket med et maksimalt antall ledd/intervaller på hver linje.
    /// Prefiks som standard "f(x) = ".
    pub fn format_multiline(&self, max_per_line: usize) -> String {
        self.format_multiline_with_prefix(max_per_line, "f(x) = ")
    }

    /// Formaterer funksjonsuttrykket over flere linjer med et gitt prefiks og maksimalt antall ledd per linje.
    pub fn format_multiline_with_prefix(&self, max_per_line: usize, prefix: &str) -> String {
        let all_segs = self.to_formatted_segments();
        if all_segs.is_empty() {
            return format!("{}0", prefix);
        }

        let max_items = if max_per_line == 0 { 4 } else { max_per_line };
        let indent = " ".repeat(prefix.len());

        let mut lines = Vec::new();
        let mut current_line = String::new();

        for (idx, seg) in all_segs.iter().enumerate() {
            let is_new_line = idx > 0 && idx % max_items == 0;
            if is_new_line {
                lines.push(current_line);
                current_line = format!("{}+ ", indent);
                if seg.is_negative {
                    current_line = format!("{}- ", indent);
                }
                current_line.push_str(&seg.text);
            } else if idx == 0 {
                current_line.push_str(prefix);
                if seg.is_negative {
                    current_line.push_str("- ");
                }
                current_line.push_str(&seg.text);
            } else {
                if seg.is_negative {
                    current_line.push_str(" - ");
                } else {
                    current_line.push_str(" + ");
                }
                current_line.push_str(&seg.text);
            }
        }

        if !current_line.is_empty() {
            lines.push(current_line);
        }

        lines.join("\n")
    }
}

impl fmt::Display for TrigFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "f(x) = {}", self.format_expr())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple() {
        let func = TrigFunction::parse("sin(x)").unwrap();
        assert_eq!(func.terms.len(), 1);
        assert_eq!(func.terms[0].kind, TrigKind::Sin);
        assert_eq!(func.terms[0].freq, 1);
        assert_eq!(func.terms[0].amp, 1.0);
    }

    #[test]
    fn test_parse_cos() {
        let func = TrigFunction::parse("cos(x) + cos(3x)").unwrap();
        assert_eq!(func.terms.len(), 2);
        assert_eq!(func.terms[0].kind, TrigKind::Cos);
        assert_eq!(func.terms[0].freq, 1);
        assert_eq!(func.terms[1].kind, TrigKind::Cos);
        assert_eq!(func.terms[1].freq, 3);
    }

    #[test]
    fn test_parse_mixed_sin_cos() {
        let func = TrigFunction::parse("sin(x) + cos(2x) - sin(5x)").unwrap();
        assert_eq!(func.terms.len(), 3);
        assert_eq!(func.terms[0].kind, TrigKind::Sin);
        assert_eq!(func.terms[0].freq, 1);
        assert_eq!(func.terms[1].kind, TrigKind::Cos);
        assert_eq!(func.terms[1].freq, 2);
        assert_eq!(func.terms[2].kind, TrigKind::Sin);
        assert_eq!(func.terms[2].freq, 5);
        assert_eq!(func.terms[2].amp, -1.0);
    }

    #[test]
    fn test_bracket_interval_cos() {
        let expanded = TrigFunction::expand_bracket_ranges("[[1, cos(x), 4]]").unwrap();
        assert_eq!(expanded, "cos(x) + cos(2x) + cos(3x) + cos(4x)");
    }

    #[test]
    fn test_bracket_interval_mixed() {
        let expanded = TrigFunction::expand_bracket_ranges("sin(x) + [[2, cos(x), 4]]").unwrap();
        assert_eq!(expanded, "sin(x) + cos(2x) + cos(3x) + cos(4x)");
    }

    #[test]
    fn test_constants_multiplied_with_terms() {
        // Standard multiplikasjon
        let f1 = TrigFunction::parse("2*sin(x) + 3*cos(2x)").unwrap();
        assert_eq!(f1.terms.len(), 2);
        assert_eq!(f1.terms[0].amp, 2.0);
        assert_eq!(f1.terms[0].freq, 1);
        assert_eq!(f1.terms[1].amp, 3.0);
        assert_eq!(f1.terms[1].freq, 2);

        // Implisitt multiplikasjon
        let f2 = TrigFunction::parse("4sin(3x) - 5cos(x)").unwrap();
        assert_eq!(f2.terms.len(), 2);
        assert_eq!(f2.terms[0].kind, TrigKind::Cos);
        assert_eq!(f2.terms[0].freq, 1);
        assert_eq!(f2.terms[0].amp, -5.0);
        assert_eq!(f2.terms[1].kind, TrigKind::Sin);
        assert_eq!(f2.terms[1].freq, 3);
        assert_eq!(f2.terms[1].amp, 4.0);

        // Desimaltall med komma og punktum
        let f3 = TrigFunction::parse("2.5*sin(x) + 1,75*cos(2x)").unwrap();
        assert_eq!(f3.terms[0].amp, 2.5);
        assert_eq!(f3.terms[1].amp, 1.75);

        // Multiplikasjon på høyre side av leddet
        let f4 = TrigFunction::parse("sin(x) * 4 - cos(2x) * 1.5").unwrap();
        assert_eq!(f4.terms[0].amp, 4.0);
        assert_eq!(f4.terms[1].amp, -1.5);
    }

    #[test]
    fn test_constants_with_bracket_intervals() {
        // Ytre konstant foran intervall
        let f1 = TrigFunction::parse("2 * [[1, sin(x), 3]]").unwrap();
        assert_eq!(f1.terms.len(), 3);
        for t in &f1.terms {
            assert_eq!(t.amp, 2.0);
        }

        // Konstant inne i intervall
        let f2 = TrigFunction::parse("[[1, 3*cos(x), 3]]").unwrap();
        assert_eq!(f2.terms.len(), 3);
        for t in &f2.terms {
            assert_eq!(t.amp, 3.0);
            assert_eq!(t.kind, TrigKind::Cos);
        }

        // Både ytre og indre konstant multiplisert sammen (2 * 2.5 = 5.0)
        let f3 = TrigFunction::parse("2 * [[1, 2.5*sin(x), 3]]").unwrap();
        assert_eq!(f3.terms.len(), 3);
        for t in &f3.terms {
            assert_eq!(t.amp, 5.0);
        }
    }

    #[test]
    fn test_multiple_bracket_intervals() {
        // Flere intervaller lagt sammen med +
        let f1 = TrigFunction::parse("[[1, sin(x), 3]] + [[2, cos(x), 4]]").unwrap();
        // sin(1x) + sin(2x) + sin(3x) + cos(2x) + cos(3x) + cos(4x) -> 6 ledd
        assert_eq!(f1.terms.len(), 6);

        // Flere intervaller med subtraksjon
        let f2 = TrigFunction::parse("[[1, sin(x), 2]] - [[3, cos(x), 4]]").unwrap();
        assert_eq!(f2.terms.len(), 4);
        assert_eq!(f2.terms[2].kind, TrigKind::Cos);
        assert_eq!(f2.terms[2].amp, -1.0);
        assert_eq!(f2.terms[3].kind, TrigKind::Cos);
        assert_eq!(f2.terms[3].amp, -1.0);

        // Flere intervaller med multiplikatorer
        let f3 = TrigFunction::parse("2 * [[1, sin(x), 2]] + 3 * [[3, cos(x), 4]]").unwrap();
        assert_eq!(f3.terms.len(), 4);
        assert_eq!(f3.terms[0].amp, 2.0);
        assert_eq!(f3.terms[1].amp, 2.0);
        assert_eq!(f3.terms[2].amp, 3.0);
        assert_eq!(f3.terms[3].amp, 3.0);

        // Blanding av enkle ledd og flere intervaller
        let f4 = TrigFunction::parse("sin(x) + [[2, sin(x), 3]] + cos(4x) + [[5, cos(x), 6]]").unwrap();
        // 1 + 2 + 1 + 2 = 6 ledd
        assert_eq!(f4.terms.len(), 6);

        // Kommaseparerte intervaller
        let f5 = TrigFunction::parse("[[1, sin(x), 2]], [[3, cos(x), 4]]").unwrap();
        assert_eq!(f5.terms.len(), 4);
    }

    #[test]
    fn test_reject_unsupported_function() {
        let err = TrigFunction::parse("tan(x) + sin(x)").unwrap_err();
        assert!(err.contains("KUN sinus (sin) og cosinus (cos) støttes"));
    }

    #[test]
    fn test_format_multiline_max_terms_per_line() {
        let f = TrigFunction::parse("sin(x) + cos(2x) + sin(3x) + cos(4x) + sin(5x) + cos(6x) + sin(7x) + cos(8x) + sin(9x)").unwrap();
        // Med maks 4 ledd per linje skal 9 ledd fordeles på 3 linjer (4 + 4 + 1)
        let formatted = f.format_multiline(4);
        let lines: Vec<&str> = formatted.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].starts_with("f(x) = "));
        assert!(lines[1].starts_with("       + "));
        assert!(lines[2].starts_with("       + "));
    }
}
