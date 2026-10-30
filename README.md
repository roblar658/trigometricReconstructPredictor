# Trigonometrisk Ekstremanalyse & Rekonstruksjon (GPU Sanntid)

Et høytytende beregningssystem utviklet i **Rust** som utnytter GPU (via `wgpu` med Vulkan/DirectX 12) til massivt parallell evaluering, derivasjon, ekstremalklassifisering og formelrekonstruksjon av trigonometriske funksjoner.

Systemet består av to integrerte moduler:
1. **Sanntids GPU-Klassifisering**: Analytisk ekstremalsøk, amplitude-kategorisering (Klasse 0–4), terskeltester og mønsteranalyse for summer av trigonometriske ledd.
2. **Spektral Formelrekonstruksjon**: Invers Fourier-identifisering fra vilkårlige CSV-måledata med trapesintegrasjon, som gjenvinner eksakte formler og komprimerer tette harmoniske rekker tilbake til intervallnotasjon.

> [!IMPORTANT]
> **Støttede funksjoner:** Programmet støtter vilkårlige lineærkombinasjoner av **KUN sinus (`sin`) og cosinus (`cos`)**. Andre matematiske funksjoner (som `tan`, `cot`, `sec`, `exp`, `log`, etc.) støttes ikke og avvises deterministisk i parsingsleddet.

## Bilder

<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/4c602882-fde9-49e4-a64a-8ee0a507c736" />
<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/814f714a-9dfb-41f6-8e72-2f0735640ba5" />
<img width="959" height="563" alt="image" src="https://github.com/user-attachments/assets/1fb4653d-b19f-42b2-8b38-e5ab66b0d6cc" />
<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/63f93a56-95b8-4984-a266-1e5b035a9a41" />
<img width="949" height="500" alt="image" src="https://github.com/user-attachments/assets/2f178b50-9a25-4102-b325-07df9ca1b4c0" />
<img width="957" height="500" alt="image" src="https://github.com/user-attachments/assets/82e1663a-1618-44d4-be63-7b8a77c043c3" />
<img width="956" height="495" alt="image" src="https://github.com/user-attachments/assets/4132b3f8-933d-4580-ae79-d03d80f8ca8c" />



---

## 1. Funksjonsdefinisjon og Intervallsyntaks

Programmet opererer på den generelle familien av trigonometriske polynomer over det fundamentale intervallet $x \in [0, 2\pi]$:

$$f(x) = \sum_{k=1}^{K} a_k \sin(\omega_k x) + \sum_{m=1}^{M} b_m \cos(\lambda_m x)$$

hvor $a_k, b_m \in \mathbb{R}$ er reelle amplituder og $\omega_k, \lambda_m \in \mathbb{N}^+$ er positive heltallsfrekvenser.

### Syntaksregler:
1. **Enkeltledd med multiplikatorer**:
   - `sin(x) + cos(2x) - 1.5*sin(4x)`
   - Støtter konstanter både før og etter leddet: `2*sin(x)`, `2sin(x)`, `sin(x)*2.5`.
   - Norsk og internasjonal desimalnotasjon støttes: `3,5*cos(2x)` eller `3.5*cos(2x)`.
2. **Kompakt intervallnotasjon med valgfri steglengde**:
   $$\left[\left[\text{start},\, \text{func},\, \text{slutt},\, \text{steg?}\right]\right]$$
   - `func` **MÅ være et sinus- eller cosinus-uttrykk** (med eller uten amplitude/konstant).
   - `steg` er valgfritt (standardverdi er $1$).
   - **Eksempler**:
     - `[[1, sin(x), 5]]` ekspanderer til:
       $$\sin(x) + \sin(2x) + \sin(3x) + \sin(4x) + \sin(5x)$$
     - `[[4, 3*cos(x), 100]]` ekspanderer til 97 ledd:
       $$3\cos(4x) + 3\cos(5x) + \dots + 3\cos(100x)$$
     - `[[1, sin(x), 9, 2]]` ekspanderer med steglengde 2 til odde harmoniske:
       $$\sin(x) + \sin(3x) + \sin(5x) + \sin(7x) + \sin(9x)$$
     - Flere intervaller kan kombineres fritt:
       `[[1, sin(x), 3]] + [[2, cos(x), 4]]`

---

## 2. Teoretisk Fundament og Hvorfor Teoremene er Viktige

Systemet bygger på tre fundamentale matematiske prinsipper. I stedet for abstrakte bevis, belyser denne seksjonen hvorfor hvert teorem er kritisk for beregningsnøyaktighet, GPU-ytelse og robust formelgjenvinning:

---

### Teorem 1: Analytiske Deriverte og Kvadratisk Konvergens (Newton-Raphson)

For den generelle trigonometriske funksjonen:
$$f(x) = \sum_{k} a_k \sin(\omega_k x) + \sum_{m} b_m \cos(\lambda_m x)$$

beregnes de eksakte deriverte analytisk i stedet for numerisk:
$$f'(x) = \sum_{k} \omega_k a_k \cos(\omega_k x) - \sum_{m} \lambda_m b_m \sin(\lambda_m x)$$
$$f''(x) = -\sum_{k} \omega_k^2 a_k \sin(\omega_k x) - \sum_{m} \lambda_m^2 b_m \cos(\lambda_m x)$$

Nullpunkter for førstederiverte ($f'(x^*) = 0$) klassifiseres direkte via andrederiverttesten:
* **Toppunkt (Maksimum)**: $f''(x^*) < 0$
* **Bunnpunkt (Minimum)**: $f''(x^*) > 0$

Sub-grid ekstremalpunkter lokaliseres med Newton-Raphson iterasjon på GPU:
$$x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)}$$

#### Hvorfor dette teoremet er viktig:
1. **Eliminerer numerisk differensiasjonsstøy på GPU**: Numerisk derivasjon (som finite differences) taper maskinpresisjon, er svært følsom for avrundingsfeil ved små steglengder ($\Delta x$), og krever flere kostbare minneaksesser. Analytisk derivasjon gir eksakt matematisk presisjon uten tap av signifikante siffer.
2. **Kvadratisk konvergens ($\mathcal{O}(e_n^2)$) gir sanntidsytelse**: Ved standard diskret grid-søk ville sub-grid nøyaktighet på $10^{-7}$ krevd milliarder av evalueringspunkter, noe som ville overbelastet GPU-minnet og båndbredden. Med Newton-Raphsons kvadratiske feildobling dobles antallet korrekte desimaler for hver iterasjon. Hver GPU-tråd oppnår dermed full maskinpresisjon på kun 2–3 iterasjoner, uavhengig av frekvens.
3. **Analytisk sikkerhet mot feilklassifisering**: Den analytiske andrederiverte $f''(x^*)$ garanterer en ufeilbarlig distinksjon mellom lokale maksima, minima og vendepunkter, selv i kaotiske interferensmønstre der flere bølgetopper ligger ekstremt tett.

---

### Teorem 2: Fourier-Ortogonalitet i Hilbert-rommet $L^2([0, 2\pi])$

Mellom vilkårlige harmoniske svingninger over $[0, 2\pi]$ gjelder de fundamentale ortogonalitetsrelasjonene:
$$\langle \sin(kx), \sin(mx) \rangle = \pi \delta_{km}, \quad \langle \cos(kx), \cos(mx) \rangle = \pi \delta_{km}, \quad \langle \sin(kx), \cos(mx) \rangle = 0$$

hvor $\delta_{km}$ er Kronecker-deltaet ($\delta_{km} = 1$ hvis $k=m$, ellers $0$). 

Dette tillater isolert projeksjon av hver enkelt amplitude:
$$a_k = \frac{1}{\pi} \int_0^{2\pi} f(x) \sin(kx) \, dx, \quad b_k = \frac{1}{\pi} \int_0^{2\pi} f(x) \cos(kx) \, dx$$

#### Hvorfor dette teoremet er viktig:
1. **Null kryss-interferens mellom frekvenser**: Ortogonaliteten garanterer at beregningen av én harmonisk komponent (f.eks. $\sin(5x)$) er fullstendig uavhengig av om signalet også inneholder $\sin(4x)$, $\cos(100x)$ eller 90 andre ledd. Ingen komponent "lekker" energi over til andre frekvenser i det kontinuerlige domenet.
2. **Unngår ustabile matrise-inversjoner**: I stedet for å måtte løse store, dårlig kondisjonerte lineære ligningssystemer ($N \times N$) eller risikere overtilpasning (overfitting) med tradisjonell polynomisk regresjon, kan hver koeffisient bestemmes direkte og deterministisk via en uavhengig integralprojeksjon i $\mathcal{O}(N)$ tid.
3. **Optimal støyfiltrering**: Tilfeldig målestøy (hvit støy) er jevnt fordelt over alle frekvenser og har tilnærmet null projeksjon mot de ortogonale basisfunksjonene. Ortogonal integrasjon fungerer derfor som et optimalt støyfilter som trekker frem de sanne signalfrekvensene selv fra ufullstendige eller støyfylte målinger.

---

### Teorem 3: Trapesregelen og Spektral Nøyaktighet (Euler-Maclaurin)

Ved diskret Fourier-analyse av måledata med $N$ punkter på et lukket intervall $[0, 2\pi]$:
$$x_i = i \cdot \Delta x, \quad \Delta x = \frac{2\pi}{N - 1}, \quad i = 0, \dots, N-1$$
benytter systemet korrigerte trapesvekter:
$$w_0 = \frac{\Delta x}{2}, \quad w_{N-1} = \frac{\Delta x}{2}, \quad w_i = \Delta x \quad (0 < i < N-1)$$

i stedet for en naiv likevektet Riemann-sum.

#### Hvorfor dette teoremet er viktig:
1. **Eliminering av kunstig randlekkasje (Endepunktsdobling)**: I en lukket måleserie $[0, 2\pi]$ representerer $x=0$ og $x=2\pi$ nøyaktig samme punkt på enhetssirkelen. En naiv likevektet oppsummering summerer dette punktet to ganger. For en tett funksjon som $f(x) = [[4, 3\cos(x), 100]] + 5\sin(x)$ er randverdien $f(0) = 291$. Ved likevekting forårsaker dette en kunstig additiv forskyvning på $\epsilon \approx \frac{291}{1000} \approx +0.291$ på *samtlige* cosinus-amplituder, noe som forvrenger modellen og genererer en mengde falske «spøkelses-ledd».
2. **Eksponensiell (spektral) konvergens for glatte periodiske funksjoner**: I henhold til Euler-Maclaurins formel bestemmes feilen i trapesintegrasjon av differansen mellom de ulike ordens deriverte i endepunktene:
   $$\text{Feil} \propto \sum_{j} c_j \Delta x^{2j} \left[ g^{(2j-1)}(2\pi) - g^{(2j-1)}(0) \right]$$
   Siden trigonometriske funksjoner er $2\pi$-periodiske og $C^\infty$ (uendelig glatte), er alle deriverte identiske i endepunktene: $g^{(p)}(2\pi) = g^{(p)}(0)$ for *alle* $p \ge 0$. Dermed **kansellerer alle polynomielle feilledd identisk**.
3. **Eksakt rekonstruksjon av tette intervaller**: Resultatet av denne kanselleringen er at trapesregelen oppnår spektral nøyaktighet. Dette gjør at systemet klarer å gjenvinne selv tette intervaller med 98 ledd (`[[4, 3cos(x), 100]] + 5sin(x)`) med feilfri nøyaktighet:
   $$R^2 = 1.0000 \quad \text{og} \quad \text{RMSE} = 0.0000$$
   uten noen form for falske frekvenser eller spøkelsesledd.

---

## 3. GPU Compute-Arkitektur (`shader.wgsl`)

Beregningene akselereres parallelt på GPU via `wgpu` (Vulkan/DirectX 12).

```
+-------------------------------------------------------------------------------+
|                             GPU COMPUTE PIPELINE                              |
+-------------------------------------------------------------------------------+
|  1. Diskrét Evaluering:                                                       |
|     N tråder evaluerer delintervaller [x_i, x_{i+1}] parallelt (opptil 1M pkt)|
|                                                                               |
|  2. Analytisk Zero-Crossing Sjekk:                                            |
|     f'(x_i) * f'(x_{i+1}) <= 0  (Fortegnsskifte detektert i delintervallet)   |
|                                                                               |
|  3. Newton-Raphson Rotfinner på GPU:                                          |
|     x* = x_mid - f'(x_mid) / f''(x_mid) -> Sub-grid posisjonsnøyaktighet       |
|                                                                               |
|  4. Klassifisering & Atomic Lagring:                                          |
|     Toppunkt (f'' < 0) / Bunnpunkt (f'' > 0) -> Amplitude-klasse (Klasse 0..4) |
|     Terskelsjekk (+A / -A) -> atomicAdd på globale GPU-tellere                |
|     Skriv ekstrema (x, y, type, klasse) til GPU Storage Buffer                |
+-------------------------------------------------------------------------------+
```

### Amplitudeklasser:
Ekstrema kategoriseres automatisk etter amplitude $y$:

| Klasse | Toppunkt-kriterium | Bunnpunkt-kriterium |
| :--- | :--- | :--- |
| **Klasse 4: Ekstrem** | $y \ge 2.5$ | $y \le -2.5$ |
| **Klasse 3: Høy / Dyp** | $1.8 \le y < 2.5$ | $-2.5 < y \le -1.8$ |
| **Klasse 2: Moderat** | $1.0 \le y < 1.8$ | $-1.8 < y \le -1.0$ |
| **Klasse 1: Lav / Grunn** | $0.2 \le y < 1.0$ | $-1.0 < y \le -0.2$ |
| **Klasse 0: Minimal** | $y < 0.2$ | $y > -0.2$ |

---

## 4. Brukerveiledning og Start

### A. Interaktiv Grafisk Applikasjon (Anbefalt)
Start applikasjonen direkte via batch-skriptet:
```powershell
.\start.bat
```
Eller via PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

### B. Funksjonalitet i Brukergrensesnittet:
1. **Fane 1: Funksjonsanalyse & Graf**:
   - Skriv inn formel i `f(x) = `, f.eks. `[[4, 3*cos(x), 100]] + 5*sin(x)` eller `2*sin(x) + 3*cos(2x)`.
   - Trykk **`Enter`** eller **`⚡ Kjør analyse`** for øyeblikkelig GPU-evaluering over intervallet.
   - **`💾 Lagre CSV (x, y)`**: Eksporterer 2000 rene $(x, y)$-datapunkter uten kommentarer.
   - **`📄 Lagre PDF`**: Genererer og åpner fullstendig analyserapport som PDF via isolert Chromium headless.
2. **Fane 2: CSV-Datasett & Formelgjenkjenning**:
   - Klikk **`📂 Bla gjennom etter CSV-fil...`** og velg en fil med måledata.
   - Filen leses inn og **analyseres automatisk med én gang**.
   - Viser beste tilpassede modell med $R^2$ og RMSE, kurveplot og kandidattabell.
   - Klikk **`⚡ Overfør formel til funksjonsanalyse`** for å sende formelen direkte over til analysefanen.

### C. Kommandolinje (CLI):
```bash
# Klassifiser en funksjon og lagre PDF-rapport
cargo run -- classify "[[4, 3*cos(x), 100]] + 5sin(x)" --pdf analyse.pdf

# Analyser en CSV-fil fra terminalen
cargo run -- analyze-csv min_data.csv --pdf csv_rapport.pdf

# Kjør alle automatiserte enhetstester
cargo test
```
