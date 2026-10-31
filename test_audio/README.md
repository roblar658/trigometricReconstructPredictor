# Test-Lydfiler for Vokaler (FLAC)

Denne mappen inneholder tapsfrie FLAC-lydfiler (16 kHz, 16-bit PCM mono) syntetisert spesifikt for testing av **Trigonometrisk Ekstremanalyse & Rekonstruksjon**.

Lydfilene dekker samtlige 9 norske vokaler (**A, E, I, O, U, Y, Æ, Ø, Å**) for både **mannsstemme** ($F_0 = 130\text{ Hz}$) og **kvinnestemme** ($F_0 = 220\text{ Hz}$), basert på etablert akustisk fonetikk (Peterson & Barney / Gunnar Fant kilderesonansmodell).

---

## 1. Mannsstemme ($F_0 = 130\text{ Hz}$)

Mannlige vokalfiler har en dypere grunntone ($F_0 = 130\text{ Hz}$) og lavere formanter grunnet lengre vokaltrakt (~17 cm):

| Filnavn | Vokal | IPA | Formanter ($F_1, F_2, F_3$) | Beskrivelse / Fonetisk ekvivalent |
| :--- | :---: | :---: | :---: | :--- |
| [`vocal_a.flac`](vocals/vocal_a.flac) / `vocal_male_a.flac` | **A** | [ɑ] | 730 Hz, 1090 Hz, 2440 Hz | Åpen bakre vokal (som i *bra*, *far*) |
| [`vocal_e.flac`](vocals/vocal_e.flac) / `vocal_male_e.flac` | **E** | [e] | 530 Hz, 1840 Hz, 2480 Hz | Halvtrang fremre vokal (som i *se*, *bed*) |
| [`vocal_i.flac`](vocals/vocal_i.flac) / `vocal_male_i.flac` | **I** | [i] | 270 Hz, 2290 Hz, 3010 Hz | Trang fremre vokal (som i *si*, *see*) |
| [`vocal_o.flac`](vocals/vocal_o.flac) / `vocal_male_o.flac` | **O** | [u] | 570 Hz, 840 Hz, 2410 Hz | Halvtrang bakre rundet vokal (som i *mor*, *boat*) |
| [`vocal_u.flac`](vocals/vocal_u.flac) / `vocal_male_u.flac` | **U** | [ʉ] | 300 Hz, 870 Hz, 2240 Hz | Trang sentral/bakre vokal (som i *sol*, *boot*) |
| [`vocal_y.flac`](vocals/vocal_y.flac) / `vocal_male_y.flac` | **Y** | [y] | 270 Hz, 1900 Hz, 2800 Hz | Trang fremre rundet vokal (som i *lys*, *über*) |
| [`vocal_ae.flac`](vocals/vocal_ae.flac) / `vocal_æ.flac` | **Æ** | [æ] | 660 Hz, 1720 Hz, 2410 Hz | Åpen fremre vokal (som i *bær*, *cat*) |
| [`vocal_oe.flac`](vocals/vocal_oe.flac) / `vocal_ø.flac` | **Ø** | [ø] | 500 Hz, 1500 Hz, 2300 Hz | Halvtrang fremre rundet vokal (som i *dør*, *schön*) |
| [`vocal_aa.flac`](vocals/vocal_aa.flac) / `vocal_å.flac` | **Å** | [ɔ] | 510 Hz, 900 Hz, 2500 Hz | Åpen-halvtrang bakre vokal (som i *båt*, *thought*) |

* **Sammensatt sekvens**: [`all_vocals_sequence.flac`](vocals/all_vocals_sequence.flac) (og `all_vocals_sequence_male.flac`), varighet 21,15s.

---

## 2. Kvinnestemme ($F_0 = 220\text{ Hz}$)

Kvindelige vokalfiler har en høyere grunntone ($F_0 = 220\text{ Hz}$, A3) og formanter forskjøvet oppover med ca. 15–20 % grunnet kortere vokaltrakt (~14,5 cm):

| Filnavn | Vokal | IPA | Formanter ($F_1, F_2, F_3$) | Beskrivelse / Fonetisk ekvivalent |
| :--- | :---: | :---: | :---: | :--- |
| [`vocal_female_a.flac`](vocals/vocal_female_a.flac) | **A** | [ɑ] | 850 Hz, 1220 Hz, 2810 Hz | Åpen bakre vokal (som i *bra*, *far*) |
| [`vocal_female_e.flac`](vocals/vocal_female_e.flac) | **E** | [e] | 590 Hz, 2310 Hz, 2990 Hz | Halvtrang fremre vokal (som i *se*, *bed*) |
| [`vocal_female_i.flac`](vocals/vocal_female_i.flac) | **I** | [i] | 310 Hz, 2790 Hz, 3310 Hz | Trang fremre vokal (som i *si*, *see*) |
| [`vocal_female_o.flac`](vocals/vocal_female_o.flac) | **O** | [u] | 600 Hz, 950 Hz, 2800 Hz | Halvtrang bakre rundet vokal (som i *mor*, *boat*) |
| [`vocal_female_u.flac`](vocals/vocal_female_u.flac) | **U** | [ʉ] | 370 Hz, 950 Hz, 2600 Hz | Trang sentral/bakre vokal (som i *sol*, *boot*) |
| [`vocal_female_y.flac`](vocals/vocal_female_y.flac) | **Y** | [y] | 310 Hz, 2300 Hz, 3200 Hz | Trang fremre rundet vokal (som i *lys*, *über*) |
| [`vocal_female_ae.flac`](vocals/vocal_female_ae.flac) / `vocal_female_æ.flac` | **Æ** | [æ] | 860 Hz, 2050 Hz, 2850 Hz | Åpen fremre vokal (som i *bær*, *cat*) |
| [`vocal_female_oe.flac`](vocals/vocal_female_oe.flac) / `vocal_female_ø.flac` | **Ø** | [ø] | 550 Hz, 1800 Hz, 2700 Hz | Halvtrang fremre rundet vokal (som i *dør*, *schön*) |
| [`vocal_female_aa.flac`](vocals/vocal_female_aa.flac) / `vocal_female_å.flac` | **Å** | [ɔ] | 610 Hz, 1050 Hz, 2850 Hz | Åpen-halvtrang bakre vokal (som i *båt*, *thought*) |

* **Sammensatt sekvens**: [`all_vocals_sequence_female.flac`](vocals/all_vocals_sequence_female.flac), varighet 21,15s.

*Fullstendig maskinlesbar metadata for begge stemmesett finnes i [`vocals_catalog.json`](vocals/vocals_catalog.json).*

---

## 3. Sekvensfiler for testing av sub-område markering & zoom

Både [`all_vocals_sequence.flac`](vocals/all_vocals_sequence.flac) (mann) og [`all_vocals_sequence_female.flac`](vocals/all_vocals_sequence_female.flac) (kvinne) inneholder alle 9 vokalene i rekkefølge med 350 ms stillhet mellom hver:

```
[ A (2.0s) ] --350ms-- [ E (2.0s) ] --350ms-- [ I (2.0s) ] -- ... -- [ Å (2.0s) ]
```

### Tidslinje (felles for begge sekvensfilene):
1. **A**: $t \in [0.00, 2.00]\text{s}$
2. **E**: $t \in [2.35, 4.35]\text{s}$
3. **I**: $t \in [4.70, 6.70]\text{s}$
4. **O**: $t \in [7.05, 9.05]\text{s}$
5. **U**: $t \in [9.40, 11.40]\text{s}$
6. **Y**: $t \in [11.75, 13.75]\text{s}$
7. **Æ**: $t \in [14.10, 16.10]\text{s}$
8. **Ø**: $t \in [16.45, 18.45]\text{s}$
9. **Å**: $t \in [18.80, 20.80]\text{s}$

### Slik tester du i brukergrensesnittet (GUI):
1. Last inn `all_vocals_sequence.flac` eller `all_vocals_sequence_female.flac` i **Fane 2 (CSV & FLAC)**.
2. Venstreklikk i kurven der en vokal starter, og høyreklikk der den slutter (området markeres i **gul farge**).
3. Klikk **«🔍 Zoom til sub-område»** for å isolere kurven til den valgte vokalen.
4. Klikk **«🔍 Analyser & Finn Formel»** for å se formelen $f(x)$ for akkurat den utvalgte vokalen, presentert med **4 ledd per linje** i en **6-linjers rullegardin**.

---

## 4. Eksempler på Kommandolinjetesting (CLI)

```bash
# Analyser mann vokal A (F0 = 130 Hz):
cargo run -- analyze-flac "test_audio/vocals/vocal_a.flac" --plot true

# Analyser kvinne vokal A (F0 = 220 Hz):
cargo run -- analyze-flac "test_audio/vocals/vocal_female_a.flac" --plot true

# Analyser kvinne vokal E fra sekvensfilen ved starttidspunkt 2.35 sekunder:
cargo run -- analyze-flac "test_audio/vocals/all_vocals_sequence_female.flac" --start-seconds 2.35 --max-points 2000

# Regenerer alle vokalfiler for mann og kvinne:
python test_audio/generate_vocals.py
```
