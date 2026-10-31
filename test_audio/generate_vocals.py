#!/usr/bin/env python3
"""
generate_vocals.py - Syntetiserer FLAC-lydfiler for alle vokaler (A, E, I, O, U, Y, Æ, Ø, Å).
Støtter både mannsstemme (F0 = 130 Hz) og kvinnestemme (F0 = 220 Hz).
Hver vokal syntetiseres med eksakte akustiske formanter (F1, F2, F3) og harmonisk struktur.
Disse filene er skreddersydd for testing av Trigonometrisk Ekstremanalyse & Spektral Rekonstruksjon.
"""

import os
import sys
import shutil
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

# -----------------------------------------------------------------------------
# 1. MANNSSTEMME (F0 = 130 Hz)
# -----------------------------------------------------------------------------
MALE_F0 = 130.0
MALE_VOWELS = {
    "a": {
        "navn": "Mann Vokal A",
        "vokal": "A",
        "ipa": "ɑ",
        "eksempel": "som i 'bra' / 'far'",
        "formants": [(730, 130, 1.0), (1090, 70, 0.75), (2440, 160, 0.35)],
        "primary_file": "vocal_a.flac",
        "alias_files": ["vocal_male_a.flac"],
    },
    "e": {
        "navn": "Mann Vokal E",
        "vokal": "E",
        "ipa": "e",
        "eksempel": "som i 'se' / 'bed'",
        "formants": [(530, 70, 1.0), (1840, 100, 0.70), (2480, 120, 0.35)],
        "primary_file": "vocal_e.flac",
        "alias_files": ["vocal_male_e.flac"],
    },
    "i": {
        "navn": "Mann Vokal I",
        "vokal": "I",
        "ipa": "i",
        "eksempel": "som i 'si' / 'bee'",
        "formants": [(270, 50, 1.0), (2290, 90, 0.70), (3010, 100, 0.40)],
        "primary_file": "vocal_i.flac",
        "alias_files": ["vocal_male_i.flac"],
    },
    "o": {
        "navn": "Mann Vokal O",
        "vokal": "O",
        "ipa": "u",
        "eksempel": "som i 'mor' / 'boat'",
        "formants": [(570, 80, 1.0), (840, 80, 0.75), (2410, 100, 0.30)],
        "primary_file": "vocal_o.flac",
        "alias_files": ["vocal_male_o.flac"],
    },
    "u": {
        "navn": "Mann Vokal U",
        "vokal": "U",
        "ipa": "ʉ",
        "eksempel": "som i 'sol' / 'boot'",
        "formants": [(300, 65, 1.0), (870, 110, 0.65), (2240, 140, 0.30)],
        "primary_file": "vocal_u.flac",
        "alias_files": ["vocal_male_u.flac"],
    },
    "y": {
        "navn": "Mann Vokal Y",
        "vokal": "Y",
        "ipa": "y",
        "eksempel": "som i 'lys' / 'über'",
        "formants": [(270, 50, 1.0), (1900, 100, 0.70), (2800, 120, 0.35)],
        "primary_file": "vocal_y.flac",
        "alias_files": ["vocal_male_y.flac"],
    },
    "ae": {
        "navn": "Mann Vokal Æ",
        "vokal": "Æ",
        "ipa": "æ",
        "eksempel": "som i 'bær' / 'cat'",
        "formants": [(660, 100, 1.0), (1720, 100, 0.70), (2410, 130, 0.35)],
        "primary_file": "vocal_ae.flac",
        "alias_files": ["vocal_male_ae.flac", "vocal_æ.flac", "vocal_male_æ.flac"],
    },
    "oe": {
        "navn": "Mann Vokal Ø",
        "vokal": "Ø",
        "ipa": "ø",
        "eksempel": "som i 'dør' / 'schön'",
        "formants": [(500, 80, 1.0), (1500, 100, 0.70), (2300, 120, 0.35)],
        "primary_file": "vocal_oe.flac",
        "alias_files": ["vocal_male_oe.flac", "vocal_ø.flac", "vocal_male_ø.flac"],
    },
    "aa": {
        "navn": "Mann Vokal Å",
        "vokal": "Å",
        "ipa": "ɔ",
        "eksempel": "som i 'båt' / 'thought'",
        "formants": [(510, 80, 1.0), (900, 80, 0.75), (2500, 100, 0.30)],
        "primary_file": "vocal_aa.flac",
        "alias_files": ["vocal_male_aa.flac", "vocal_å.flac", "vocal_male_å.flac"],
    },
}

# -----------------------------------------------------------------------------
# 2. KVINNESTEMME (F0 = 220 Hz)
# -----------------------------------------------------------------------------
# Kvinner har kortere vokaltrakt (~14.5 cm mot menns ~17 cm), noe som forskyver
# formant-resonansene oppover med ca. 15-20 %, kombinert med en høyere grunnfrekvens F0 = 220 Hz.
FEMALE_F0 = 220.0
FEMALE_VOWELS = {
    "a": {
        "navn": "Kvinne Vokal A",
        "vokal": "A",
        "ipa": "ɑ",
        "eksempel": "som i 'bra' / 'far'",
        "formants": [(850, 130, 1.0), (1220, 90, 0.75), (2810, 160, 0.35)],
        "primary_file": "vocal_female_a.flac",
        "alias_files": [],
    },
    "e": {
        "navn": "Kvinne Vokal E",
        "vokal": "E",
        "ipa": "e",
        "eksempel": "som i 'se' / 'bed'",
        "formants": [(590, 80, 1.0), (2310, 110, 0.70), (2990, 130, 0.35)],
        "primary_file": "vocal_female_e.flac",
        "alias_files": [],
    },
    "i": {
        "navn": "Kvinne Vokal I",
        "vokal": "I",
        "ipa": "i",
        "eksempel": "som i 'si' / 'bee'",
        "formants": [(310, 60, 1.0), (2790, 100, 0.70), (3310, 120, 0.40)],
        "primary_file": "vocal_female_i.flac",
        "alias_files": [],
    },
    "o": {
        "navn": "Kvinne Vokal O",
        "vokal": "O",
        "ipa": "u",
        "eksempel": "som i 'mor' / 'boat'",
        "formants": [(600, 90, 1.0), (950, 90, 0.75), (2800, 120, 0.30)],
        "primary_file": "vocal_female_o.flac",
        "alias_files": [],
    },
    "u": {
        "navn": "Kvinne Vokal U",
        "vokal": "U",
        "ipa": "ʉ",
        "eksempel": "som i 'sol' / 'boot'",
        "formants": [(370, 75, 1.0), (950, 120, 0.65), (2600, 140, 0.30)],
        "primary_file": "vocal_female_u.flac",
        "alias_files": [],
    },
    "y": {
        "navn": "Kvinne Vokal Y",
        "vokal": "Y",
        "ipa": "y",
        "eksempel": "som i 'lys' / 'über'",
        "formants": [(310, 60, 1.0), (2300, 110, 0.70), (3200, 130, 0.35)],
        "primary_file": "vocal_female_y.flac",
        "alias_files": [],
    },
    "ae": {
        "navn": "Kvinne Vokal Æ",
        "vokal": "Æ",
        "ipa": "æ",
        "eksempel": "som i 'bær' / 'cat'",
        "formants": [(860, 110, 1.0), (2050, 110, 0.70), (2850, 140, 0.35)],
        "primary_file": "vocal_female_ae.flac",
        "alias_files": ["vocal_female_æ.flac"],
    },
    "oe": {
        "navn": "Kvinne Vokal Ø",
        "vokal": "Ø",
        "ipa": "ø",
        "eksempel": "som i 'dør' / 'schön'",
        "formants": [(550, 90, 1.0), (1800, 110, 0.70), (2700, 130, 0.35)],
        "primary_file": "vocal_female_oe.flac",
        "alias_files": ["vocal_female_ø.flac"],
    },
    "aa": {
        "navn": "Kvinne Vokal Å",
        "vokal": "Å",
        "ipa": "ɔ",
        "eksempel": "som i 'båt' / 'thought'",
        "formants": [(610, 90, 1.0), (1050, 90, 0.75), (2850, 120, 0.30)],
        "primary_file": "vocal_female_aa.flac",
        "alias_files": ["vocal_female_å.flac"],
    },
}

def synthesize_vowel(formants, duration=DURATION, sr=SAMPLE_RATE, f0=MALE_F0):
    """
    Syntetiserer en ren trigonometrisk klangfarge for en vokal via harmonisk summering
    og vokalresonans-båndpassfiltre (Fant/Peterson-Barney kilderesonans).
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

    # Fade-out kun ved filslutt (siste 40 ms) for å eliminere klikk ved avslutning.
    # Starten (t=0) er ren og harmonisk stasjonær da alle ledd har sin(0) = 0.
    fade_len = int(0.04 * sr)
    envelope = np.ones_like(t)
    envelope[-fade_len:] = np.cos(np.linspace(0, np.pi / 2, fade_len))**2
    signal *= envelope

    # Normaliser amplitude til 0.75 for ren 16-bit FLAC PCM
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = 0.75 * signal / max_val

    return signal

def process_voice_set(voice_label, f0, vowels_dict, default_seq_filename):
    catalog_entries = {}
    sequence_parts = []
    silence_gap = np.zeros(int(SAMPLE_RATE * 0.35))  # 350 ms pause mellom vokaler

    print(f"\n--- SYNTETISERER {voice_label.upper()} (F0 = {f0:.0f} Hz) ---")

    for key, info in vowels_dict.items():
        prim_path = os.path.join(OUTPUT_DIR, info["primary_file"])
        audio_data = synthesize_vowel(info["formants"], duration=DURATION, sr=SAMPLE_RATE, f0=f0)

        # Lagre primærfil
        sf.write(prim_path, audio_data, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
        size_bytes = os.path.getsize(prim_path)

        # Opprett eventuelle alias-filer
        for alias in info.get("alias_files", []):
            alias_path = os.path.join(OUTPUT_DIR, alias)
            shutil.copyfile(prim_path, alias_path)

        f1, f2, f3 = info["formants"][0][0], info["formants"][1][0], info["formants"][2][0]
        print(f"  [OK] {info['navn']:<18} ({info['ipa']}) -> {info['primary_file']:<22} | F1={f1:>4}Hz, F2={f2:>4}Hz, F3={f3:>4}Hz | {size_bytes} B")

        catalog_entries[key] = {
            "name": info["navn"],
            "vowel": info["vokal"],
            "voice": voice_label,
            "ipa": info["ipa"],
            "example": info["eksempel"],
            "primary_file": info["primary_file"],
            "alias_files": info.get("alias_files", []),
            "path": prim_path,
            "sample_rate": SAMPLE_RATE,
            "duration_sec": DURATION,
            "f0_hz": f0,
            "formants_hz": [f1, f2, f3],
        }

        # Bygg sekvens for felles testfil
        sequence_parts.append(audio_data)
        sequence_parts.append(silence_gap)

    # Lagre sekvensfil
    seq_signal = np.concatenate(sequence_parts)
    seq_path = os.path.join(OUTPUT_DIR, default_seq_filename)
    sf.write(seq_path, seq_signal, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    seq_size = os.path.getsize(seq_path)

    print(f"  [OK] Sekvensfil: {default_seq_filename:<22} | Varighet: {len(seq_signal)/SAMPLE_RATE:.2f}s | {seq_size} B")

    seq_entry = {
        "name": f"Alle vokaler i sekvens ({voice_label})",
        "voice": voice_label,
        "filename": default_seq_filename,
        "path": seq_path,
        "sample_rate": SAMPLE_RATE,
        "f0_hz": f0,
        "total_duration_sec": len(seq_signal) / SAMPLE_RATE,
        "vowels_order": [v["vokal"] for v in vowels_dict.values()],
        "timeline_seconds": {
            v["vokal"]: [
                round(i * (DURATION + 0.35), 2),
                round(i * (DURATION + 0.35) + DURATION, 2)
            ]
            for i, v in enumerate(vowels_dict.values())
        }
    }

    return catalog_entries, seq_entry

def main():
    print("=========================================================================")
    print("  SYNTETISERING AV VOKALER: MANN (F0=130 Hz) OG KVINNE (F0=220 Hz)")
    print("=========================================================================")

    male_entries, male_seq = process_voice_set(
        "male", MALE_F0, MALE_VOWELS, "all_vocals_sequence.flac"
    )
    # Lag også en eksplisitt kopi all_vocals_sequence_male.flac
    male_seq_copy = os.path.join(OUTPUT_DIR, "all_vocals_sequence_male.flac")
    shutil.copyfile(os.path.join(OUTPUT_DIR, "all_vocals_sequence.flac"), male_seq_copy)

    female_entries, female_seq = process_voice_set(
        "female", FEMALE_F0, FEMALE_VOWELS, "all_vocals_sequence_female.flac"
    )

    catalog = {
        "male": {
            "f0_hz": MALE_F0,
            "vowels": male_entries,
            "sequence": male_seq,
        },
        "female": {
            "f0_hz": FEMALE_F0,
            "vowels": female_entries,
            "sequence": female_seq,
        }
    }

    catalog_path = os.path.join(OUTPUT_DIR, "vocals_catalog.json")
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print("\n=========================================================================")
    print(f"Fullstendig katalog lagret til: {catalog_path}")
    print("Alle FLAC-filer for både manns- og kvinnestemme er generert!")
    print("=========================================================================")

if __name__ == "__main__":
    main()
