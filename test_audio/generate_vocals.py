#!/usr/bin/env python3
"""
generate_vocals.py - Syntetiserer FLAC-lydfiler for alle vokaler (A, E, I, O, U, Y, Æ, Ø, Å).
Hver vokal syntetiseres med eksakte akustiske formanter (F1, F2, F3) og harmonisk struktur (F0 = 130 Hz).
Disse filene er skreddersydd for testing av Trigonometrisk Ekstremanalyse & Spektral Rekonstruksjon.
"""

import os
import sys
import json
import numpy as np
import soundfile as sf

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocals")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE = 16000  # 16 kHz (standard taledatasett og identisk med te.flac)
DURATION = 2.0       # 2 sekunder per vokal
F0 = 130.0           # Grunnfrekvens i Hz (typisk mannsstemme)

# Formant-frekvenser (F1, F2, F3), båndbredder (B1, B2, B3) og relative vekter
# Basert på akustisk fonetikk (Peterson & Barney / Fant kilderesonansmodell for skandinaviske vokaler)
VOWELS = {
    "a": {
        "navn": "Vokal A",
        "ipa": "ɑ",
        "eksempel": "som i 'bra' / 'far'",
        "formants": [(730, 130, 1.0), (1090, 70, 0.75), (2440, 160, 0.35)],
        "fil": "vocal_a.flac",
    },
    "e": {
        "navn": "Vokal E",
        "ipa": "e",
        "eksempel": "som i 'se' / 'bed'",
        "formants": [(530, 70, 1.0), (1840, 100, 0.70), (2480, 120, 0.35)],
        "fil": "vocal_e.flac",
    },
    "i": {
        "navn": "Vokal I",
        "ipa": "i",
        "eksempel": "som i 'si' / 'bee'",
        "formants": [(270, 50, 1.0), (2290, 90, 0.70), (3010, 100, 0.40)],
        "fil": "vocal_i.flac",
    },
    "o": {
        "navn": "Vokal O",
        "ipa": "u",
        "eksempel": "som i 'mor' / 'boat'",
        "formants": [(570, 80, 1.0), (840, 80, 0.75), (2410, 100, 0.30)],
        "fil": "vocal_o.flac",
    },
    "u": {
        "navn": "Vokal U",
        "ipa": "ʉ",
        "eksempel": "som i 'sol' / 'boot'",
        "formants": [(300, 65, 1.0), (870, 110, 0.65), (2240, 140, 0.30)],
        "fil": "vocal_u.flac",
    },
    "y": {
        "navn": "Vokal Y",
        "ipa": "y",
        "eksempel": "som i 'lys' / 'über'",
        "formants": [(270, 50, 1.0), (1900, 100, 0.70), (2800, 120, 0.35)],
        "fil": "vocal_y.flac",
    },
    "ae": {
        "navn": "Vokal Æ",
        "ipa": "æ",
        "eksempel": "som i 'bær' / 'cat'",
        "formants": [(660, 100, 1.0), (1720, 100, 0.70), (2410, 130, 0.35)],
        "fil": "vocal_ae.flac",
    },
    "oe": {
        "navn": "Vokal Ø",
        "ipa": "ø",
        "eksempel": "som i 'dør' / 'schön'",
        "formants": [(500, 80, 1.0), (1500, 100, 0.70), (2300, 120, 0.35)],
        "fil": "vocal_oe.flac",
    },
    "aa": {
        "navn": "Vokal Å",
        "ipa": "ɔ",
        "eksempel": "som i 'båt' / 'thought'",
        "formants": [(510, 80, 1.0), (900, 80, 0.75), (2500, 100, 0.30)],
        "fil": "vocal_aa.flac",
    },
}

def synthesize_vowel(formants, duration=DURATION, sr=SAMPLE_RATE, f0=F0):
    """
    Syntetiserer en ren trigonometrisk klangfarge for en vokal via harmonisk summering
    og vokalresonans-båndpassfiltre.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t)

    # Beregn maksimal harmonisk under Nyquist-frekvens
    nyquist = sr / 2.0
    max_h = int(nyquist / f0)

    for h in range(1, max_h + 1):
        freq = h * f0
        amp = 0.0

        for f_c, bw, weight in formants:
            # Resonansfilterkurve (andreordens resonanspol)
            q = f_c / bw
            ratio = freq / f_c
            res = 1.0 / np.sqrt(1.0 + q**2 * (ratio - 1.0 / ratio)**2)
            amp += weight * res

        # Glottal spektral-avfall (-6 dB per oktav ~ 1 / sqrt(h))
        amp *= (1.0 / np.sqrt(h))
        signal += amp * np.sin(2.0 * np.pi * freq * t)

    # Fade-in og fade-out for å eliminere klikk og gi naturlig tale-inngang
    fade_len = int(0.04 * sr)
    envelope = np.ones_like(t)
    envelope[:fade_len] = np.sin(np.linspace(0, np.pi / 2, fade_len))**2
    envelope[-fade_len:] = np.cos(np.linspace(0, np.pi / 2, fade_len))**2
    signal *= envelope

    # Normaliser amplitude til 0.75 for ren 16-bit FLAC PCM
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = 0.75 * signal / max_val

    return signal

def main():
    catalog = {}
    sequence_parts = []
    silence_gap = np.zeros(int(SAMPLE_RATE * 0.35))  # 350 ms pause mellom vokaler i sekvensen

    print("=========================================================================")
    print("  SYNTETISERER FLAC-LYDFILER FOR VOKALER (A, E, I, O, U, Y, Æ, Ø, Å)")
    print("=========================================================================")

    for key, info in VOWELS.items():
        file_path = os.path.join(OUTPUT_DIR, info["fil"])
        audio_data = synthesize_vowel(info["formants"], duration=DURATION, sr=SAMPLE_RATE, f0=F0)

        # Lagre som 16-bit FLAC PCM
        sf.write(file_path, audio_data, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
        size_bytes = os.path.getsize(file_path)

        f1, f2, f3 = info["formants"][0][0], info["formants"][1][0], info["formants"][2][0]
        print(f"  [OK] {info['navn']:<10} ({info['ipa']}) -> {info['fil']:<16} | F1={f1}Hz, F2={f2}Hz, F3={f3}Hz | {size_bytes} bytes")

        catalog[key] = {
            "name": info["navn"],
            "ipa": info["ipa"],
            "example": info["eksempel"],
            "filename": info["fil"],
            "path": file_path,
            "sample_rate": SAMPLE_RATE,
            "duration_sec": DURATION,
            "f0_hz": F0,
            "formants_hz": [f1, f2, f3],
        }

        # Bygg sekvens for felles testfil
        sequence_parts.append(audio_data)
        sequence_parts.append(silence_gap)

    # Lagre den samlede sekvensfilen (for testing av sub-område markering i GUI)
    seq_signal = np.concatenate(sequence_parts)
    seq_filename = "all_vocals_sequence.flac"
    seq_path = os.path.join(OUTPUT_DIR, seq_filename)
    sf.write(seq_path, seq_signal, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    seq_size = os.path.getsize(seq_path)

    print("-------------------------------------------------------------------------")
    print(f"  [OK] Sammensatt sekvensfil -> {seq_filename:<16} | Varighet: {len(seq_signal)/SAMPLE_RATE:.2f}s | {seq_size} bytes")
    print("       (Perfekt for å teste sub-område markering mellom minX og maxX!)")
    print("=========================================================================")

    catalog["all_sequence"] = {
        "name": "Alle vokaler i sekvens",
        "filename": seq_filename,
        "path": seq_path,
        "sample_rate": SAMPLE_RATE,
        "total_duration_sec": len(seq_signal) / SAMPLE_RATE,
        "vowels_order": list(VOWELS.keys()),
    }

    # Lagre katalog-metadata som JSON
    catalog_path = os.path.join(OUTPUT_DIR, "vocals_catalog.json")
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    print(f"Katalog lagret til: {catalog_path}")

if __name__ == "__main__":
    main()
