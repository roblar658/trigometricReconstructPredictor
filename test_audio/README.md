# Test-Lydfiler for Vokaler (FLAC)

Denne mappen inneholder tapsfrie FLAC-lydfiler (16 kHz, 16-bit PCM mono) syntetisert spesifikt for testing av **Trigonometrisk Ekstremanalyse & Rekonstruksjon**.

Hver vokal er generert basert på etablert akustisk fonetikk (Peterson & Barney / Gunnar Fant kilderesonansmodell) med en stabil grunnfrekvens $F_0 = 130$ Hz (mannsstemme i taleleiet) og de tre karakteristiske formant-resonansene ($F_1, F_2, F_3$).

---

## 1. Individuelle Vokalfiler (`test_audio/vocals/`)

Hver fil har en varighet på 2,0 sekunder med 40 ms jevn fade-in og fade-out for å unngå transienter og klikkelyder:

| Filnavn | Norsk Vokal | IPA | Formanter ($F_1, F_2, F_3$) | Beskrivelse / Fonetisk ekvivalent |
| :--- | :---: | :---: | :---: | :--- |
| [`vocal_a.flac`](vocals/vocal_a.flac) | **A** | [ɑ] | 730 Hz, 1090 Hz, 2440 Hz | Åpen bakre vokal (som i *bra*, *far*) |
| [`vocal_e.flac`](vocals/vocal_e.flac) | **E** | [e] | 530 Hz, 1840 Hz, 2480 Hz | Halvtrang fremre vokal (som i *se*, *bed*) |
| [`vocal_i.flac`](vocals/vocal_i.flac) | **I** | [i] | 270 Hz, 2290 Hz, 3010 Hz | Trang fremre vokal (som i *si*, *see*) |
| [`vocal_o.flac`](vocals/vocal_o.flac) | **O** | [u] | 570 Hz, 840 Hz, 2410 Hz | Halvtrang bakre rundet vokal (som i *mor*, *boat*) |
| [`vocal_u.flac`](vocals/vocal_u.flac) | **U** | [ʉ] | 300 Hz, 870 Hz, 2240 Hz | Trang sentral/bakre vokal (som i *sol*, *boot*) |
| [`vocal_y.flac`](vocals/vocal_y.flac) | **Y** | [y] | 270 Hz, 1900 Hz, 2800 Hz | Trang fremre rundet vokal (som i *lys*, *über*) |
| [`vocal_ae.flac`](vocals/vocal_ae.flac) / `vocal_æ.flac` | **Æ** | [æ] | 660 Hz, 1720 Hz, 2410 Hz | Åpen fremre vokal (som i *bær*, *cat*) |
| [`vocal_oe.flac`](vocals/vocal_oe.flac) / `vocal_ø.flac` | **Ø** | [ø] | 500 Hz, 1500 Hz, 2300 Hz | Halvtrang fremre rundet vokal (som i *dør*, *schön*) |
| [`vocal_aa.flac`](vocals/vocal_aa.flac) / `vocal_å.flac` | **Å** | [ɔ] | 510 Hz, 900 Hz, 2500 Hz | Åpen-halvtrang bakre vokal (som i *båt*, *thought*) |

*Fullstendig maskinlesbar metadata finnes i [`vocals_catalog.json`](vocals/vocals_catalog.json).*

---

## 2. Sammensatt Testsekvens (`all_vocals_sequence.flac`)

Filen [`all_vocals_sequence.flac`](vocals/all_vocals_sequence.flac) inneholder alle 9 vokalene i rekkefølge med 350 ms stillhet mellom hver vokal (total varighet: 21,15 sekunder).

Denne filen er spesielt designet for å teste **sub-område markering** i brukergrensesnittet:
- Last inn `all_vocals_sequence.flac` i Fane 2 (CSV & FLAC).
- Venstreklikk og høyreklikk i kurven for å markere et gult sub-område rundt en bestemt vokal.
- Klikk **«🔍 Zoom til sub-område»** for å isolere kurven.
- Klikk **«🔍 Analyser & Finn Formel»** for å se at systemet identifiserer formantene for akkurat den utvalgte vokalen.

### Tidslinje for vokalsekvensen:
1. **A**: $t \in [0.00, 2.00]\text{s}$
2. **E**: $t \in [2.35, 4.35]\text{s}$
3. **I**: $t \in [4.70, 6.70]\text{s}$
4. **O**: $t \in [7.05, 9.05]\text{s}$
5. **U**: $t \in [9.40, 11.40]\text{s}$
6. **Y**: $t \in [11.75, 13.75]\text{s}$
7. **Æ**: $t \in [14.10, 16.10]\text{s}$
8. **Ø**: $t \in [16.45, 18.45]\text{s}$
9. **Å**: $t \in [18.80, 20.80]\text{s}$

---

## 3. Eksempel på Testing via Kommandolinje (CLI)

```bash
# Test vokal A (identifiserer F0=130 Hz og harmoniske rundt F1=730 Hz og F2=1090 Hz):
cargo run -- analyze-flac "test_audio/vocals/vocal_a.flac" --plot true

# Test vokal E isolert fra sekvensfilen ved bruk av startforsinkelse:
cargo run -- analyze-flac "test_audio/vocals/all_vocals_sequence.flac" --start-seconds 2.35 --max-points 2000

# Regenerer eller tilpass alle vokalfilene med Python:
python test_audio/generate_vocals.py
```

