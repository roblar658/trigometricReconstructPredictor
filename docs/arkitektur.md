# Teknisk Arkitektur & Systemdiagrammer

Dette dokumentet presenterer den formelle arkitekturen, modulære oppbygningen, databehandlingspipelinen og maskinvareinteraksjonen i **Trigonometrisk Ekstremanalyse & Rekonstruksjon (GPU Sanntid)**.

---

## 1. Overordnet Systemarkitektur

Systemet er bygget i **Rust** og inndelt i seks kjernekomponenter: brukergrensesnitt (`eframe`/`egui`), spektralanalyse og lyddekoding (`claxon`), matematisk funksjonsmodell og parser, mønstergjenkjenning, en massivt parallell GPU-beregningskjerne via **WebGPU (`wgpu`) med Vulkan / DirectX 12**, samt rapportering og WiX-distribusjon.

```mermaid
flowchart TD
    subgraph UI ["1. Brukergrensesnitt & Presentasjonslag (egui / eframe)"]
        App["ExtremaApp (App State)"]
        Tab1["Fane 1: Funksjonsanalyse (Sanntidssimulering)"]
        Tab2["Fane 2: CSV & FLAC Lydspekteranalyse"]
        Tab3["Fane 3: Teori, Teoremer & Hjelp"]
        PlotWidget["egui_plot Widget (Zoom, Pan, Sub-omraade, Markører)"]
        ScrollFx["Rullegardin f(x) (4 ledd/linje, maks 6 linjer)"]
    end

    subgraph AudioCSV ["2. Lyddekoding & Spektralanalyse (CPU)"]
        FlacLoader["flac_loader.rs (claxon FLAC Dekoder)"]
        SilenceDet["Stillhetsdeteksjon (find_first_active_second)"]
        SubRange["Sub-omraade Utvelgelse (minX..maxX, Gul farge)"]
        CSVAnalyzer["csv_analyzer.rs (Spektral Rekonstruksjon)"]
        Trapz["Trapesintegrasjon (Euler-Maclaurin Spektral Kvadratur)"]
        OMP["Orthogonal Matching Pursuit (OMP Residual-regresjon)"]
        LeastSquares["Multivariat Minste Kvadraters Regresjon (Gauss-Jordan)"]
        Occam["Occam's Razor Modelltilpasning (R², RMSE, Leddtelling)"]
    end

    subgraph CoreModel ["3. Matematisk Kjerne & Modellering (src/model.rs)"]
        Parser["TrigFunction Parser (Kun sin & cos støttes)"]
        IntervalExp["Intervall-ekspander [[start, func, slutt, steg]]"]
        IntervalComp["Intervall-komprimerer (to_formatted_segments)"]
        MultilineFmt["format_multiline (Strengt 4 ledd per linje)"]
    end

    subgraph PatternEngine ["4. Mønsteranalyse (src/patterns.rs)"]
        Cluster["Verdiklynger (Identiske topper og bunner)"]
        Consecutive["Verdier etter hverandre (Sekvenser pa rad)"]
        Symmetry["Symmetritesting (Oddesymmetri / Likesymmetri)"]
        Spacing["Toppunkt-fordeling & Ekvidistanse (Delta x)"]
    end

    subgraph GPULayer ["5. GPU Beregningskjerne (wgpu / WGSL Compute Shader)"]
        GpuCtx["GpuContext (Adapter, Device, Queue - Vulkan / DX12)"]
        GpuPipeline["ExtremaClassifierPipeline (Bind Groups, Buffere)"]
        WGSL["shader.wgsl (Compute Shader @workgroup_size 256)"]
        AnalyticDeriv["Analytisk f'(x) og f''(x) Evaluering"]
        NewtonRaphson["Newton-Raphson Sub-grid Rotfinner (3 iterasjoner)"]
        Classifier["Klassifisering (Klasse 0-4, Terskler T1-T4, Topp/Bunn)"]
        Staging["Staging Buffer & Atomic Counters Readback"]
    end

    subgraph ExportDist ["6. Rapportering & Distribusjon"]
        PdfExp["pdf_exporter.rs (Chromium / Edge Headless PDF)"]
        HtmlRep["HTML Rapport med SVG Vektorgrafikk & Tabeller"]
        CsvExp["CSV-Eksport (Sub-omraade data & Ekstremalpunkter)"]
        MsiPkg["WiX Toolset v4 MSI Installasjonspakke (Setup-v1.1.0.msi)"]
    end

    %% Relasjoner
    App --> Tab1
    App --> Tab2
    App --> Tab3
    Tab1 --> PlotWidget
    Tab2 --> PlotWidget
    Tab2 --> ScrollFx

    Tab2 --> FlacLoader
    FlacLoader --> SilenceDet
    SilenceDet --> SubRange
    SubRange --> CSVAnalyzer

    CSVAnalyzer --> Trapz
    Trapz --> OMP
    OMP --> LeastSquares
    LeastSquares --> Occam
    Occam --> IntervalComp
    IntervalComp --> Parser

    Parser --> MultilineFmt
    MultilineFmt --> ScrollFx

    Tab1 --> GpuPipeline
    Tab2 --> GpuPipeline
    GpuPipeline --> GpuCtx
    GpuCtx --> WGSL
    WGSL --> AnalyticDeriv
    AnalyticDeriv --> NewtonRaphson
    NewtonRaphson --> Classifier
    Classifier --> Staging
    Staging --> App

    App --> PatternEngine
    PatternEngine --> Cluster
    PatternEngine --> Consecutive
    PatternEngine --> Symmetry
    PatternEngine --> Spacing

    App --> PdfExp
    PdfExp --> HtmlRep
    Tab2 --> CsvExp
    App --> MsiPkg
```

---

## 2. End-to-End Databehandlings- og Rekonstruksjonspipeline

Denne flyten viser hvordan en rå FLAC-lydfil (som [`te.flac`](file:///c:/Users/rober/OneDrive/Bilder/te.flac)) eller CSV-måleserie transformeres til en ferdig matematisk formel, klassifiseres på GPU og visualiseres.

```mermaid
flowchart TD
    Start(["Bruker velger fil: CSV eller FLAC"]) --> InputCheck{"Filtype?"}

    %% FLAC gren
    InputCheck -- ".flac" --> Decode["claxon: Dekoder 16-bit PCM mono/stereo"]
    Decode --> Silence["find_first_active_second: Detekterer og skipper romstoy (t < 0.20s)"]
    Silence --> Downsample["Ekvidistant nedsampling til N maelepunkter (Hz-skala)"]

    %% CSV gren
    InputCheck -- ".csv" --> ParseCSV["csv_analyzer: Leser kolonner x, y (Robust float-parsing)"]

    %% Felles forbehandling
    Downsample --> PreMerge["DataPoint-vektor klargjort"]
    ParseCSV --> PreMerge

    PreMerge --> SubSelect{"Sub-omraade aktivert?"}
    SubSelect -- "Ja" --> Slicing["Filtrerer punkter: x in [minX, maxX] (Gul farge i GUI)"]
    SubSelect -- "Nei" --> FullData["Bruker hele datasettet"]

    Slicing --> NyquistCalc["Beregner fysikalsk Nyquist-frekvens w_nyq = pi / delta_x"]
    FullData --> NyquistCalc

    %% Spektralanalyse
    NyquistCalc --> OMPLoop["1. Orthogonal Matching Pursuit (OMP Iterasjon)"]
    OMPLoop --> Trapezoid["2. Trapesintegrasjon (Euler-Maclaurin kvadratur mot residual)"]
    Trapezoid --> BestFreq["3. Velg basisfunksjon med maks korrelasjon: sin(kx) eller cos(kx)"]
    BestFreq --> GramMatrix["4. Multivariat regresjon over valgte baser (A^T W A) c = A^T W y"]
    GramMatrix --> GaussJordan["5. Gauss-Jordan eliminasjon med Tikhonov-regularisering"]
    GaussJordan --> UpdateResidual["6. Oppdater residual: r_{s+1}(x) = y(x) - y_hat(x)"]
    UpdateResidual --> OMPTermCheck{"Ferdig? (R² > 0.9999 eller residual konvergert)"}
    OMPTermCheck -- "Nei" --> OMPLoop
    OMPTermCheck -- "Ja" --> OccamFilter["Occam's Razor: Velg enkleste modell blant topp-kandidater"]

    %% Formatering
    OccamFilter --> Compress["Intervall-komprimering: 3+ paafolgende harmoniske -> [[start, func, slutt]]"]
    Compress --> MultilineFormat["Formatering: Maks 4 ledd per linje (format_multiline)"]
    MultilineFormat --> UIRullegardin["GUI Visning: Rullegardin begrenset til maks 6 linjer (med scroll & rull opp/ned)"]

    %% GPU evaluering
    UIRullegardin --> GpuSubmit["Klargjor GPU uniform buffer (Params + 64 terms)"]
    GpuSubmit --> DispatchGPU["Kjor wgpu Compute Pipeline (65 536 traader)"]
    DispatchGPU --> ReadbackGPU["Readback: Ekstremalpunkter, amplitude-klasser, terskeltelling"]
    ReadbackGPU --> RunPattern["Mønsteranalyse: Like verdier, klynger, sekvenser, symmetri"]
    RunPattern --> RenderAll["Oppdater egui_plot, zoom-utsnitt og resultatpaneler"]
    RenderAll --> ExportOption{"Bruker velger eksport?"}
    ExportOption -- "PDF Rapport" --> ChromiumPDF["Chromium/Edge Headless PDF eksport (A4 med SVG og 4 ledd/linje)"]
    ExportOption -- "CSV Data" --> SubCSV["Eksporter filtrert sub-omraade til ny CSV"]
    ExportOption -- "Ingen" --> Done(["Interaktiv utforsking fullfort"])
```

---

## 3. GPU Compute Shader & Newton-Raphson Pipeline

Detaljert arbeidsflyt for hver enkelt GPU-tråd inne i [`src/gpu/shader.wgsl`](file:///c:/dev/gpu_extrema_classifier/src/gpu/shader.wgsl) over intervallet $[x_{\min}, x_{\max}]$ fordelt på 65 536 tråder med arbeidsgruppestørrelse 256 (`@workgroup_size(256)`).

```mermaid
flowchart TD
    InitWorkgroup["wgpu CommandEncoder: DispatchWorkgroups(256, 1, 1)"] --> ThreadEntry["Traad global_id.x: Undersøker delintervall [x_a, x_b]"]
    ThreadEntry --> CalcEndpoints["x_a = x_min + idx * h,  x_b = x_a + h"]

    CalcEndpoints --> EvalDfA["Evaluer analytisk f'(x_a)"]
    EvalDfA --> EvalDfB["Evaluer analytisk f'(x_b)"]

    EvalDfB --> SignCheck{"Fortegnsskifte i f'(x)?"}
    SignCheck -- "df_a > 0 og df_b <= 0" --> PeakCandidate["Kandidat for TOPPUNKT (Maksimum)"]
    SignCheck -- "df_a < 0 og df_b >= 0" --> TroughCandidate["Kandidat for BUNNPUNKT (Minimum)"]
    SignCheck -- "Samme fortegn" --> EarlyExit["Ingen ekstrema i dette delintervallet (Avslutt traad)"]

    PeakCandidate --> NRInit["Start Newton-Raphson: x_0 = (x_a + x_b) / 2"]
    TroughCandidate --> NRInit

    subgraph NewtonLoop ["Newton-Raphson Sub-Grid Rotfinner (3 iterasjoner)"]
        NRInit --> NRIter["Iterasjon n = 0, 1, 2"]
        NRIter --> EvalDF["Evaluer analytisk f'(x_n)"]
        EvalDF --> EvalD2F["Evaluer analytisk f''(x_n)"]
        EvalD2F --> CheckDenom{"|f''(x_n)| > 1e-6?"}
        CheckDenom -- "Ja" --> StepNR["x_{n+1} = x_n - f'(x_n) / f''(x_n)"]
        CheckDenom -- "Nei (Singulaer)" --> KeepX["Avbryt iterasjon, behold beste x"]
        StepNR --> NRIter
        KeepX --> NRDone["x* funnet med sub-grid presisjon (10^-7 rad)"]
        StepNR -- "Etter 3 iterasjoner" --> NRDone
    end

    NRDone --> EvalFinalY["Evaluer y* = f(x*) og f''(x*)"]
    EvalFinalY --> SecondDerivTest{"Andrederiverttest:"}

    SecondDerivTest -- "f''(x*) < 0" --> FinalPeak["Bekreftet TOPPUNKT (kind = 1)"]
    SecondDerivTest -- "f''(x*) > 0" --> FinalTrough["Bekreftet BUNNPUNKT (kind = 2)"]
    SecondDerivTest -- "f''(x*) == 0" --> DiscardPoint["Vendepunkt - forkastes"]

    FinalPeak --> ClassifyPeak["Klassifiser Toppunkt-amplitude:\n- Klasse 4: y >= 2.5 (Ekstrem)\n- Klasse 3: 1.8 <= y < 2.5 (Høy)\n- Klasse 2: 1.0 <= y < 1.8 (Moderat)\n- Klasse 1: 0.2 <= y < 1.0 (Lav)\n- Klasse 0: y < 0.2 (Minimal/Negativ)"]
    FinalTrough --> ClassifyTrough["Klassifiser Bunnpunkt-amplitude:\n- Klasse 4: y <= -2.5 (Ekstrem)\n- Klasse 3: -2.5 < y <= -1.8 (Dyp)\n- Klasse 2: -1.8 < y <= -1.0 (Moderat)\n- Klasse 1: -1.0 < y <= -0.2 (Grunn)\n- Klasse 0: y > -0.2 (Minimal/Positiv)"]

    ClassifyPeak --> ThresholdTest["Test mot brukers terskler T1, T2, T3, T4 (+A / -A)"]
    ClassifyTrough --> ThresholdTest

    ThresholdTest --> AtomicInc["AtomicAdd(atomic_counters[class_idx], 1)\nAtomicAdd(extremum_count, 1)"]
    AtomicInc --> BufferWrite["Skriv Extremum(x*, y*, kind, class_idx) til extrema_buffer"]
    BufferWrite --> ThreadDone(["Traad fullfort"])
```

---

## 4. Sekvensdiagram: Komplett Analyseforløp

Dette sekvensdiagrammet illustrerer tidslinjen og ansvarsfordelingen mellom brukeren, brukergrensesnittet, lyd/CSV-motoren, OMP-rekonstruksjonen, GPU-kjernen og PDF-generatoren.

```mermaid
sequenceDiagram
    autonumber
    actor Bruker as Bruker
    participant UI as ExtremaApp (egui)
    participant FLAC as flac_loader / claxon
    participant OMP as csv_analyzer (OMP + LeastSquares)
    participant GPU as GpuPipeline (wgpu)
    participant Shader as shader.wgsl (GPU Compute)
    participant Pat as patterns.rs
    participant PDF as pdf_exporter (Chromium)

    Bruker->>UI: Velger lydfil "te.flac"
    UI->>FLAC: read_flac_audio(path, options)
    FLAC->>FLAC: Sjekker stillhet: find_first_active_second()
    Note over FLAC: Hopper automatisk over romtone (t < 0.20s)
    FLAC-->>UI: 2000 DataPoints (x in sekunder, y in normalisert amplitude)
    UI->>UI: Tegner kurve med gult markeringsfelt for sub-omraade

    Bruker->>UI: Endrer sub-omraade [minX=1.60s, maxX=1.90s]
    Bruker->>UI: Klikker "🔍 Analyser & Finn Formel"
    UI->>OMP: identify_function_from_data(punkter, max_k=1000)

    loop OMP Residual Iterasjoner
        OMP->>OMP: Euler-Maclaurin trapesintegrasjon mot restfeil
        OMP->>OMP: Finn dominant basisfrekvens (sin eller cos)
        OMP->>OMP: Inverter Gram-matrise (A^T W A) via Gauss-Jordan
        OMP->>OMP: Oppdater modell og trekk fra residual
    end

    OMP->>OMP: Komprimer tette ledd til intervaller [[start, func, slutt]]
    OMP-->>UI: IdentificationResult (Beste match f(x), R²=76.4%, RMSE, topp 5)

    UI->>UI: Formaterer f(x) strengt med 4 ledd per linje
    UI->>UI: Viser f(x) i rullegardin (ScrollArea maks 6 linjer høy)

    UI->>GPU: pipeline.run(ctx, func, x_min=0, x_max=span, samples=65536)
    GPU->>Shader: Dispatch 256 workgroups (65 536 traader)
    Shader->>Shader: Analytisk f'(x), f''(x) & Newton-Raphson rotfinner
    Shader->>Shader: Klassifiser topp/bunn, amplitude Klasse 0-4
    Shader-->>GPU: Skriv til extrema_buffer & atomiske tellere
    GPU-->>UI: ClassificationSummary (global_max, global_min, ekstrema-liste)

    UI->>Pat: analyze_extrema_patterns(ekstrema, tolerance)
    Pat-->>UI: ExtremaPatternReport (klynger, serier pa rad, oddesymmetri)
    UI->>UI: Render oppdatert egui_plot med topp/bunn-markører

    Bruker->>UI: Klikker "📄 Lagre PDF"
    UI->>PDF: export_to_pdf(rapport.pdf, ...)
    PDF->>PDF: Genererer HTML med SVG og 4 ledd/linje f(x)
    PDF->>PDF: Kjorer Chrome/Edge headless med --print-to-pdf
    PDF-->>UI: Rapport lagret og apnet automatisk
    UI-->>Bruker: "Vellykket! PDF lagret"
```

---

## 5. Datamodell & Klasserelasjoner

Klassediagrammet viser sammenhengen mellom datastrukturene i Rust-kjernen:

```mermaid
classDiagram
    class TrigKind {
        <<enumeration>>
        Sin
        Cos
    }

    class TrigTerm {
        +TrigKind kind
        +u32 freq
        +f64 amp
        +new(kind, freq, amp) TrigTerm
        +sin(freq, amp) TrigTerm
        +cos(freq, amp) TrigTerm
    }

    class TrigFunction {
        +Vec~TrigTerm~ terms
        +parse(input) Result~TrigFunction, String~
        +eval(x) f64
        +eval_df(x) f64
        +eval_d2f(x) f64
        +format_expr() String
        +format_multiline(max_per_line) String
        +to_formatted_segments() Vec~FormattedSegment~
    }

    class DataPoint {
        +f64 x
        +f64 y
    }

    class CandidateMatch {
        +TrigFunction function
        +f64 r_squared
        +f64 rmse
        +f64 max_error
        +Vec~f64~ predicted_y
    }

    class IdentificationResult {
        +CandidateMatch best_match
        +Vec~CandidateMatch~ top_candidates
        +usize data_points_count
        +(f64, f64) x_range
        +(f64, f64) y_range
    }

    class ExtremumKind {
        <<enumeration>>
        Toppunkt
        Bunnpunkt
    }

    class ClassifiedExtremum {
        +f32 x
        +f32 y
        +ExtremumKind kind
        +usize class_idx
        +String class_name
    }

    class ClassificationSummary {
        +u32 total_peaks
        +u32 total_troughs
        +(f32, f32) global_max
        +(f32, f32) global_min
        +[u32; 5] peak_counts
        +[u32; 5] trough_counts
        +[f32; 4] thresholds
        +[u32; 4] peak_threshold_exceeded
        +Vec~ClassifiedExtremum~ extrema
        +peak_to_peak() f32
    }

    class ExtremaPatternReport {
        +Vec~ExtremumCluster~ peak_clusters
        +Vec~ExtremumCluster~ trough_clusters
        +Vec~ConsecutiveExtremaRun~ consecutive_peaks
        +Vec~ConsecutiveExtremaRun~ consecutive_troughs
        +bool has_odd_symmetry
        +bool has_even_symmetry
        +f32 average_peak_spacing
        +bool is_equidistant_peaks
    }

    class ExtremaApp {
        +String formula_input
        +u32 num_samples
        +Vec~DataPoint~ csv_data
        +usize csv_terms_per_line
        +bool csv_fx_expanded
        +usize csv_fx_max_lines
        +bool sub_range_enabled
        +f64 sub_range_min_x
        +f64 sub_range_max_x
        +Option~IdentificationResult~ csv_result
        +Option~ClassificationSummary~ summary
        +Option~ExtremaPatternReport~ patterns
        +analyze_loaded_csv()
        +run_gpu_analysis()
        +export_pdf()
    }

    TrigTerm --> TrigKind
    TrigFunction *-- TrigTerm
    CandidateMatch *-- TrigFunction
    IdentificationResult *-- CandidateMatch
    ClassifiedExtremum --> ExtremumKind
    ClassificationSummary *-- ClassifiedExtremum
    ExtremaApp *-- TrigFunction
    ExtremaApp *-- DataPoint
    ExtremaApp *-- IdentificationResult
    ExtremaApp *-- ClassificationSummary
    ExtremaApp *-- ExtremaPatternReport
```

---

## 6. Tilstandsmaskin for GUI (User Interaction State Machine)

Diagrammet illustrerer tilstandene og overgangene i applikasjonens levetid under analyse av signaler og ekstrema:

```mermaid
stateDiagram-v2
    [*] --> Idle: Applikasjon starter (Standard f(x) ferdig innlest)

    state Fane1_Funksjonsanalyse {
        Idle --> VenterPaaFormel: Bruker redigerer formel
        VenterPaaFormel --> GPU_Beregning: Klikker "⚡ Kjoer analyse" / Enter
        GPU_Beregning --> ResultatVises: GPU returnerer ekstrema & klasser
        ResultatVises --> InteraktivtPlott: Pan/zoom, klikk for inspeksjon
        ResultatVises --> EksportererPDF: Klikker "📄 Lagre PDF"
        EksportererPDF --> ResultatVises: PDF generert via Chromium
    }

    state Fane2_CSV_FLAC_Analyse {
        Idle --> LasterFil: Velger CSV eller FLAC fil
        LasterFil --> StillhetsSjekk: flac_loader detekterer start
        StillhetsSjekk --> DataKlar: Automatisk satt til aktiv lyd (t >= 0.20s)

        DataKlar --> SubOmraadeValg: Venstreklikk (minX) / Hoyreklikk (maxX)
        SubOmraadeValg --> SubOmraadeValg: Oppdaterer gult rektangel & linjer

        SubOmraadeValg --> OMP_Beregning: Klikker "🔍 Analyser & Finn Formel"
        OMP_Beregning --> ModellIdentifisert: OMP ferdig (Best match f(x) funnet)

        ModellIdentifisert --> RullegardinVisning: f(x) formateres (4 ledd/linje)
        
        state RullegardinVisning {
            [*] --> RulletNed: Standard visning (Scrollbar aktiv hvis > 6 linjer)
            RulletNed --> RulletOpp: Bruker klikker "▲ Rull opp"
            RulletOpp --> RulletNed: Bruker klikker "▼ Rull ned"
        }

        ModellIdentifisert --> OverfoerFormel: Klikker "⚡ Overfoer formel"
        OverfoerFormel --> GPU_Beregning: Bytt til Fane 1 med identifisert formel

        ModellIdentifisert --> LagreSubCSV: Klikker "💾 Lagre sub-omraade..."
        LagreSubCSV --> ModellIdentifisert: Skrevet til CSV
    }
```

---

## 7. Oppsummering av Nøkkelprinsipper

| Egenskap | Teknisk Realisering | Teoretisk Begrunnelse |
| :--- | :--- | :--- |
| **Matematiske funksjoner** | Kun `sin` og `cos` støttes | Hilbert-rom ortogonalitet og unngår poler/asymptoter (`tan`, `cot`). |
| **GPU Sub-grid presisjon** | Analytisk Newton-Raphson på GPU | Kvadratisk konvergens $\mathcal{O}(e^2)$ gir $10^{-7}$ rad nøyaktighet på 3 iterasjoner. |
| **Trapesintegrasjon** | Euler-Maclaurin korrigerte vekter | Fjerner endepunktsdobling $x=0$ og $x=2\pi$; gir spektral konvergens for glatte funksjoner. |
| **Krysstale-eliminering** | Orthogonal Matching Pursuit (OMP) | Fjerner lekkasje fra dominerende ledd på ikke-periodiske vinduer via trinnvis residual. |
| **Lyddeteksjon** | `find_first_active_second` | Hopper over innledende romtone i taleopptak ([`te.flac`](file:///c:/Users/rober/OneDrive/Bilder/te.flac)) automatisk. |
| **Sub-område visualisering** | Gult rektangel & grenselinjer | Tydelig fargekodet avgrensning av analyseutsnittet med ett-klikks zoom. |
| **Formelvisning** | 4 ledd per linje & 6-linjers rullegardin | Kompakt og ryddig monospace-presentasjon som aldri fortrenger grafen. |
| **Rapportering** | Chromium/Edge Headless PDF | Vektorgrafikk (SVG) og identisk formattering som i programvaren. |
