# Trigonometrisk Ekstremanalyse & Rekonstruksjon (GPU Sanntid)

Et høytytende beregningssystem utviklet i **Rust** som utnytter GPU (via `wgpu` med Vulkan/DirectX 12) til massivt parallell evaluering, derivasjon, ekstremalklassifisering og formelrekonstruksjon av trigonometriske funksjoner.

Systemet består av to integrerte moduler:
1. **Sanntids GPU-Klassifisering**: Analytisk ekstremalsøk, amplitude-kategorisering (Klasse 0–4), terskeltester og mønsteranalyse for summer av trigonometriske ledd.
2. **Spektral Formelrekonstruksjon**: Invers Fourier-identifisering fra vilkårlige CSV-måledata med trapesintegrasjon, som gjenvinner eksakte formler og komprimerer tette harmoniske rekker tilbake til intervallnotasjon.

> [!IMPORTANT]
> **Støttede funksjoner:** Programmet støtter vilkårlige lineærkombinasjoner av **KUN sinus (`sin`) og cosinus (`cos`)**. Andre matematiske funksjoner (som `tan`, `cot`, `sec`, `exp`, `log`, etc.) støttes ikke og avvises deterministisk i parsingsleddet.

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

## 2. Matematiske Bevis og Teoretisk Fundament

### Teorem 1: Analytiske Deriverte og Ekstremumkriterier
La $f(x) \in C^\infty(\mathbb{R})$ være definert ved:
$$f(x) = \sum_{k} a_k \sin(\omega_k x) + \sum_{m} b_m \cos(\lambda_m x)$$

#### Bevis for deriverte:
Ved lineariteten til den differensielle operatoren $\frac{d}{dx}$ og kjerneregelen:
$$\frac{d}{dx}\sin(\omega x) = \omega \cos(\omega x), \quad \frac{d}{dx}\cos(\lambda x) = -\lambda \sin(\lambda x)$$

Følgelig er førstederiverte $f'(x)$ eksakt gitt ved:
$$f'(x) = \sum_{k} \omega_k a_k \cos(\omega_k x) - \sum_{m} \lambda_m b_m \sin(\lambda_m x)$$

Tilsvarende, ved å derivere $f'(x)$:
$$\frac{d}{dx}\cos(\omega x) = -\omega \sin(\omega x), \quad \frac{d}{dx}\sin(\lambda x) = \lambda \cos(\lambda x)$$

Dette gir den eksakte andrederiverte $f''(x)$:
$$f''(x) = -\sum_{k} \omega_k^2 a_k \sin(\omega_k x) - \sum_{m} \lambda_m^2 b_m \cos(\lambda_m x)$$

#### Klassifisering av ekstrema:
Et punkt $x^* \in [0, 2\pi]$ kalles et stasjonært punkt hvis og bare hvis:
$$f'(x^*) = 0$$

Ved Taylors teorem om $x^*$ har vi for små $\delta x$:
$$f(x^* + \delta x) = f(x^*) + f'(x^*)\delta x + \frac{1}{2}f''(x^*)\delta x^2 + \mathcal{O}(\delta x^3)$$

Siden $f'(x^*) = 0$:
$$f(x^* + \delta x) - f(x^*) = \frac{1}{2}f''(x^*)\delta x^2 + \mathcal{O}(\delta x^3)$$
1. **Lokalt Maksimum (Toppunkt)**: Hvis $f''(x^*) < 0$, er $\frac{1}{2}f''(x^*)\delta x^2 < 0$ for alle $\delta x \neq 0$ i en omegn om $x^*$. Følgelig er $f(x^*) > f(x^* + \delta x)$.
2. **Lokalt Minimum (Bunnpunkt)**: Hvis $f''(x^*) > 0$, er $\frac{1}{2}f''(x^*)\delta x^2 > 0$, og $f(x^*) < f(x^* + \delta x)$.

#### Kvadratisk Konvergens med Newton-Raphson:
For å bestemme nullpunktet til $f'(x)$ med maskinpresisjon på GPU, anvendes Newton-Raphson iterasjon:
$$x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)}$$

Ved Taylor-utvikling av $f'(x^*)$ om $x_n$, der $f'(x^*) = 0$:
$$0 = f'(x_n) + f''(x_n)(x^* - x_n) + \frac{1}{2}f'''(\xi_n)(x^* - x_n)^2$$

Divider med $f''(x_n)$ og la feilen ved iterasjon $n$ være $e_n = x_n - x^*$:
$$e_{n+1} = \frac{f'''(\xi_n)}{2 f''(x_n)} e_n^2 \implies |e_{n+1}| \le C |e_n|^2$$

Dette beviser **kvadratisk konvergens** ($\mathcal{O}(e_n^2)$), som sikrer sub-grid ekstremal-lokalisering med nøyaktighet $< 10^{-7}$ på kun 2–3 iterasjoner per GPU-tråd.

---

### Teorem 2: Fourier-Ortogonalitet og Spektral Projeksjon
La funksjonsrommet $L^2([0, 2\pi])$ være utstyrt med det indreproduktet:
$$\langle u, v \rangle = \int_0^{2\pi} u(x) v(x) \, dx$$

#### Bevis for ortogonalitet:
For vilkårlige heltall $k, m \in \mathbb{N}^+$ gjelder produkt-til-sum identitetene:
$$\sin(kx)\sin(mx) = \frac{1}{2}\left[\cos((k-m)x) - \cos((k+m)x)\right]$$
$$\cos(kx)\cos(mx) = \frac{1}{2}\left[\cos((k-m)x) + \cos((k+m)x)\right]$$
$$\sin(kx)\cos(mx) = \frac{1}{2}\left[\sin((k+m)x) + \sin((k-m)x)\right]$$

Integrasjon over $[0, 2\pi]$ for $k \neq m$:
$$\int_0^{2\pi} \cos((k \pm m)x) dx = \left[ \frac{\sin((k \pm m)x)}{k \pm m} \right]_0^{2\pi} = 0$$

For $k = m \ge 1$:
$$\int_0^{2\pi} \sin^2(kx) dx = \int_0^{2\pi} \frac{1 - \cos(2kx)}{2} dx = \pi$$
$$\int_0^{2\pi} \cos^2(kx) dx = \int_0^{2\pi} \frac{1 + \cos(2kx)}{2} dx = \pi$$

Og for kryssledd:
$$\int_0^{2\pi} \sin(kx)\cos(mx) dx = 0 \quad \forall k, m$$

Dette etablerer de fundamentale ortogonalitetsrelasjonene:
$$\langle \sin(kx), \sin(mx) \rangle = \pi \delta_{km}, \quad \langle \cos(kx), \cos(mx) \rangle = \pi \delta_{km}, \quad \langle \sin(kx), \cos(mx) \rangle = 0$$

Følgelig er Fourier-projeksjonene for et signal $f(x)$:
$$a_k = \frac{\langle f(x), \sin(kx) \rangle}{\langle \sin(kx), \sin(kx) \rangle} = \frac{1}{\pi} \int_0^{2\pi} f(x) \sin(kx) \, dx$$
$$b_k = \frac{\langle f(x), \cos(kx) \rangle}{\langle \cos(kx), \cos(kx) \rangle} = \frac{1}{\pi} \int_0^{2\pi} f(x) \cos(kx) \, dx$$

Dette beviser at enhver harmonisk komponent kan projiseres ut og isoleres med null kryss-interferens fra øvrige harmoniske ledd.

---

### Teorem 3: Trapesregelen og Eliminering av Spektral Lekkasje

Når kontinuerlige funksjoner samples diskret over intervallet $[0, 2\pi]$ med $N$ punkter:
$$x_i = i \cdot \Delta x, \quad \Delta x = \frac{2\pi}{N - 1}, \quad i = 0, 1, \dots, N-1$$
vil både startpunktet $x_0 = 0$ og endepunktet $x_{N-1} = 2\pi$ representere den samme tilstanden på sirkelen $\mathbb{R} / 2\pi\mathbb{Z}$.

#### Feilen ved naiv likevekting (Riemann-sum):
I en standard likevektet diskret sum:
$$S_k = \sum_{i=0}^{N-1} y_i \cos(kx_i)$$
bidrar punktet $x=0$ med $y(0)\cos(0) = y(0)$, og punktet $x=2\pi$ bidrar med $y(2\pi)\cos(2k\pi) = y(2\pi) = y(0)$.
Dette fører til at randverdien telles **to ganger**, og introduserer en systematisk additiv feil:
$$\epsilon_k \approx \frac{y(0)}{N}$$

For en tett funksjon som $f(x) = 5\sin(x) + [[4, 3\cos(x), 100]]$:
$$y(0) = 5\sin(0) + \sum_{k=4}^{100} 3\cos(0) = 0 + 97 \times 3 = 291$$

Med $N = 1000$ punkter gir denne naive oppsummeringen:
$$\epsilon_k \approx \frac{291}{1000} = +0.291$$
Denne feilen forskyver samtlige cosinus-amplituder fra $3.00$ til $3.29$, og genererer i tillegg falske spøkelses-harmoniske med amplitude $0.29 > 0.15$ på frekvensene $k = 1, 2, 3$ og $k > 100$.

#### Løsning via Trapesregelen og Euler-Maclaurin:
Euler-Maclaurin formel for numerisk kvadratur av en funksjon $g(x) = y(x)\cos(kx)$:
$$\sum_{i=0}^{N-1} w_i g(x_i) - \int_0^{2\pi} g(x) dx = \sum_{j=1}^{m} \frac{B_{2j}}{(2j)!} \Delta x^{2j} \left[ g^{(2j-1)}(2\pi) - g^{(2j-1)}(0) \right] + R_{2m}$$
hvor trapesvektene er definert ved:
$$w_0 = \frac{\Delta x}{2}, \quad w_{N-1} = \frac{\Delta x}{2}, \quad w_i = \Delta x \quad (0 < i < N-1)$$
og $B_{2j}$ er Bernoullitallene.

Siden $g(x)$ er en lineærkombinasjon av trigonometriske funksjoner med heltallsfrekvenser, er $g(x)$ periodisk og uendelig glatt ($g \in C^\infty(\mathbb{T})$). Følgelig er alle rand-deriverte identiske ved endepunktene:
$$g^{(p)}(2\pi) - g^{(p)}(0) = 0 \quad \forall p \ge 0$$

Dette medfører at **samtlige polynomielle feilledd i Euler-Maclaurin-summen forsvinner identisk**:
$$\sum_{i=0}^{N-1} w_i g(x_i) - \int_0^{2\pi} g(x) dx = \mathcal{O}\left(e^{-c/\Delta x}\right)$$

Trapesregelen oppnår dermed **spektral nøyaktighet** (eksponensiell konvergens).
Dette beviser matematisk hvorfor Fourier-projeksjon med trapesvekter eliminerer lekkasjen fullstendig og gjenvinner eksakt:
$$A_k = 3.00000000 \quad (k \in [4, 100]), \quad B_1 = 5.00000000, \quad A_{1..3} = 0.00000000$$
med $R^2 = 1.0000$ og $\text{RMSE} = 0.0000$.

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
