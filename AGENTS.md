# AGENTS.md#
## Kontrak Operasional Engineering — Biosignal Stress Detection & Android Clinical Companion

> Dokumen ini adalah aturan kerja utama untuk AI coding agent dan developer pada proyek ini.
> `PRD.md` adalah source of truth produk; `AGENTS.md` mengatur **cara implementasi, verifikasi, dan perubahan kode**.
> Data pipeline, classical ML, export, testing, dan build berjalan **lokal**. Deep learning hanya berjalan secara conditional di **Google Colab T4** setelah gate ML kanal terkait gagal.

---

## 1. Misi

Bangun ulang pipeline end-to-end yang bersih, dapat direproduksi, bebas data leakage, dan siap digunakan pada aplikasi Android nyata untuk membantu psikolog/konselor.

Alur utama wajib:

```text
Raw ECG + PPG (32 partisipan)
→ manifest + audit raw
→ koreksi timestamp/clock ECG
→ cleaning + filtering
→ MADC + SQA/SQI
→ peak-to-peak detection ECG–PPG
→ validasi IBI/HR/HRV PPG terhadap ECG
→ EDA
→ feature/waveform engineering
→ participant-safe classical ML lokal
→ conditional DL training di Google Colab T4
→ evaluasi dengan minimum research gate dan preferred gate
→ ONNX export + numerical parity
→ Kotlin preprocessing parity
→ Android inference 5 menit (Home & History)
→ Verbatim+ (Audio + Live PPG) & AI+ (RAG Chatbot) & Profile
```

Aplikasi adalah **decision-support tool**, bukan alat diagnosis. Output fisiologis tidak boleh diklaim sebagai diagnosis, kepastian medis, atau pendeteksi kebohongan.

---

## 2. Konfigurasi Proyek yang Dikunci

```text
participants            : 32 total, ID-01 .. ID-32 ONLY
new participant data    : tidak ada; akuisisi manual telah selesai
model phases             : relax (0), task/stress (1) ONLY; raw/audit may preserve unknown
routine                  : tidak dikumpulkan; dilarang menjadi label
PPG device               : Samsung Galaxy Watch 5
PPG channels             : ppg_ir, ppg_red, ppg_green
ECG device               : Polar H10
ECG role                 : referensi/cross-validation biosignal, bukan input model Android
local CPU                : AMD Ryzen 7 4800H
local GPU                : NVIDIA RTX 2060 Laptop GPU (classical ML/support)
DL runtime               : Google Colab T4, conditional per channel
random seed              : 42
window size              : 180 detik
window stride            : 10 detik
source phase duration    : 300 detik per relax/task phase
analysis core window     : 240 detik, anchor start +30 detik sampai +270 detik
ECG–PPG validation       : 240 detik core, tepat 192 participant × phase × PPG channel
session acquisition     : 300 detik penuh; prediction hanya core elapsed [30s,270s)
session decision         : agregasi maksimal 7 window valid dari core 240 detik
model development        : ID-01 .. ID-28
reused internal test     : ID-29 .. ID-32; tidak dipakai tuning/selection
minimum research gate    : balanced_accuracy/F1/ROC-AUC >= 0.70
preferred gate           : balanced_accuracy >= 0.80, F1 >= 0.78, ROC-AUC >= 0.83
accuracy                 : reported, bukan gate atau escalation trigger
ONNX parity              : max_abs_diff < 1e-5
Android parity           : max_abs_diff < 1e-5 pada test vectors yang disetujui
```

### Aturan penting

- Tidak boleh ada partisipan selain `ID-01` sampai `ID-32` pada manifest, split, laporan, training, evaluasi, fixture, maupun dokumentasi.
- Jangan hardcode peserta yang dianggap ECG-invalid. Validitas ECG harus ditentukan ulang dari raw ECG setelah koreksi timestamp, cleaning, dan SQA.
- Bila ECG suatu sesi tetap gagal kualitas setelah koreksi, sesi tersebut dikeluarkan **hanya dari validasi ECG–PPG**. PPG-nya masih boleh dipakai untuk Gold/training bila SQA PPG lolos.
- Raw/audit `phase` boleh `relax`, `task`, atau `unknown` untuk traceability; Gold/training/model canonical hanya `relax` dan `task`. `routine` tetap dilarang. `label` adalah `0/1` dan `label_name` adalah `Normal/Stres`.
- Source canonical: ECG dari `DataRenamedPartisipan`, PPG dari `DataTrimmedPartisipan`, boundary dari `TimeStampAnchor`.
- Raw/source phase tetap 300 detik dan immutable. Seluruh analisis canonical memakai core window anchor `[+30s, +270s)` berdurasi 240 detik; crop tidak boleh memakai padding atau data sintetis.
- Gold memakai window 180 detik dengan stride 10 detik, maksimal 7 window per phase pada core 240 detik.
- Training PPG mengevaluasi `ppg_green`, `ppg_ir`, dan `ppg_red` sebagai tiga predictor independen. Setiap kanal memiliki preprocessing, feature order, threshold, model, ONNX, dan readiness sendiri; fusion/averaging antar-kanal dilarang.
- Raw timestamp immutable; timestamp canonical dipakai untuk sinkronisasi, sedangkan `elapsed_seconds` dipakai untuk pemrosesan/windowing.
- Readiness wajib memisahkan `ppg_gold_status`, `training_status`, `ecg_validation_status`, dan `model_release_status`; ECG failure tidak otomatis membuang PPG dari Gold/training.
- `ppg_gold_status` dan readiness training mengukur 64 group per kanal serta 192 total. Kegagalan Red tidak boleh memblokir Green/IR, dan status model awal kanal yang belum dievaluasi adalah `not_evaluated`.
- `ID-29..ID-32` adalah `reused_internal_test_cohort`, bukan external validation. Cohort ini tidak boleh dipakai untuk window/quality/feature/model selection, tuning threshold, HPO, calibration, augmentation tuning, scaler fitting, atau early stopping.
- Laporan holdout lama tanpa `model_contract_fingerprint` atau `holdout_contract_fingerprint` yang eksplisit wajib dipertahankan sebagai `historical_unverifiable`; laporan tersebut tidak reusable dan tidak memblokir satu evaluasi canonical pertama. Evaluasi canonical itu wajib dicatat atomik dengan fingerprint kontrak, lalu semua pembukaan ulang berikutnya diblokir kecuali evidence fingerprint yang sama dipakai ulang.
- Minimum research gate 0,70/0,70/0,70 adalah batas bawah yang dikunci dan tidak boleh diturunkan secara ad hoc. Preferred gate tetap 0,80/0,78/0,83.

---

## 3. Batasan Eksekusi

Proyek ini berjalan lokal, kecuali deep learning conditional yang dieksekusi secara manual di Google Colab T4.

Dilarang menjadikan layanan cloud/notebook remote selain Google Colab T4 untuk DL conditional, automation build/release remote, atau experiment-tracking platform eksternal sebagai bagian dari jalur implementasi dan release proyek.

Yang diperbolehkan:

- Python CLI lokal.
- VS Code / terminal lokal.
- virtual environment lokal (`uv`/`venv`).
- CUDA lokal untuk preprocessing support dan classical-model benchmark bila relevan; DL tidak dijalankan lokal.
- Google Colab T4 hanya untuk DL per channel yang gagal gate ML lokal; hanya Gold PPG + manifest/checksum yang boleh diunggah, bukan raw data.
- JSON/YAML/CSV/Parquet lokal untuk konfigurasi, manifest, audit, dan laporan.
- Git lokal/remote hanya sebagai version control source code jika digunakan; bukan automation pipeline.
- Android Studio/Gradle lokal untuk build, lint, test, benchmark, dan APK.

---

## 4. Source of Truth dan BMAD / Spec-Driven Development

Urutan kerja setiap perubahan:

```text
Requirement
→ PRD acceptance criteria
→ kontrak data/model/API
→ rencana minimal
→ implementasi
→ local verification
→ evidence/report
→ update dokumentasi bila kontrak berubah
```

### Aturan agent

1. Baca `PRD.md` dan `AGENTS.md` sebelum mengubah kode.
2. Jangan menebak schema, participant ID, label, sampling rate, atau artifact path.
3. Jika kode bertentangan dengan PRD, perbaiki kode atau ajukan perubahan spec secara eksplisit.
4. Gunakan minimal diff; jangan rewrite massal jika patch lokal cukup.
5. Setiap perubahan harus menyebutkan input, output, failure mode, dan test yang relevan.
6. Jangan menyimpan keputusan penting hanya di chat; simpan di spec/config/ADR lokal bila mengubah kontrak.

BMAD digunakan untuk memperjelas perencanaan dan agent orchestration, bukan untuk menambah birokrasi atau tooling cloud.

---

## 5. Prinsip Engineering Wajib

### 5.1 Clean Code

- Satu fungsi/satu class memiliki satu tanggung jawab utama.
- Gunakan nama domain yang jelas: `EcgClockCorrector`, `PpgQualityAssessor`, `PeakMatcher`, `StressInferenceEngine`.
- Hindari fungsi raksasa, global mutable state, hidden side effect, magic number, dan parameter tersembunyi.
- Threshold signal/model harus berasal dari config tervalidasi.
- I/O berada di boundary; core signal/model logic dibuat semurni mungkin.

### 5.2 DRY

- Satu implementasi canonical untuk preprocessing Python.
- Satu kontrak preprocessing yang direplikasi secara teruji ke Kotlin.
- Jangan copy-paste filter, windowing, normalization, feature list, route, prompt, atau model manifest.
- Shared behavior ditempatkan di module/interface reusable.

### 5.3 SOLID

- **S**: module sesuai satu domain responsibility.
- **O**: sensor/model baru ditambah lewat interface/config, bukan rewrite besar.
- **L**: implementasi interface wajib mempertahankan contract.
- **I**: interface kecil dan spesifik.
- **D**: domain layer bergantung pada abstraction, bukan ONNX/Room/MediaPipe concrete class.

### 5.4 Anti-Spaghetti

Tidak diperbolehkan:

- satu script yang ingest → clean → train → export sekaligus;
- business logic di Composable;
- model loading langsung dari UI;
- duplicated preprocessing di banyak feature;
- notebook sebagai source of truth;
- silent fallback ke dummy response/data;
- catch-all exception tanpa error context.

---

## 6. Linting dan Quality Tools Lokal

### Python

Wajib tersedia:

```bash
ruff check .
ruff format --check .
pytest -q
```

Direkomendasikan:

```bash
pyright
```

### Kotlin / Android

Jalankan lokal:

```bash
./gradlew ktlintCheck
./gradlew detekt
./gradlew lint
./gradlew test
```

Untuk perubahan inference/UI penting, tambahkan instrumented test pada physical device/emulator yang relevan.

---

## 7. Repository Target

```text
project/
├── PRD.md
├── AGENTS.md
├── pyproject.toml
├── requirements.lock / uv.lock
├── PipelinePython/configs/
│   ├── data.yaml
│   ├── cleaning.yaml
│   ├── sqa.yaml
│   ├── madc.yaml
│   ├── alignment.yaml
│   ├── features.yaml
│   ├── training.yaml
│   ├── onnx.yaml
│   └── android.yaml
├── data/
│   ├── raw/
│   │   ├── DataRenamedPartisipan/
│   │   ├── DataTrimmedPartisipan/
│   │   └── TimeStampAnchor/
│   └── artifacts/
│       ├── manifests/
│       ├── bronze/
│       ├── silver/
│       ├── gold/
│       ├── sqa/
│       ├── validation/
│       ├── eda/
│       ├── models/
│       └── export/
├── PipelinePython/
│   ├── 01_data_preprocess/
│   ├── 02_peak_to_peak/
│   ├── 03_stress_onnx/
│   ├── common/
│   ├── configs/
│   └── tests/
├── reports/
│   ├── data_quality/
│   ├── alignment/
│   ├── validation/
│   ├── eda/
│   └── model_eval/
├── configs/
│   └── rag.yaml              # Android AI+ configuration only
└── android/
    └── app/
```

Jangan memindahkan atau mengubah raw data tanpa alasan terdokumentasi. Bila folder aktual berbeda, buat adapter path di config; jangan merusak sumber asli.

---

## 8. Data Contract dan Lineage

Semua artifact harus mempertahankan:

```text
participant_id
session_id
phase
sensor_channel
source_file
source_checksum
sampling_rate_hz
start_timestamp_utc
end_timestamp_utc
window_id
quality_score
quality_tier
quality_flags
exclusion_reason
preprocessing_version
```

Hierarki wajib:

```text
participant_id
└── phase
    └── sensor_channel
        ├── ecg
        ├── ppg_ir
        ├── ppg_red
        └── ppg_green
```

Gabungan tabel diperbolehkan hanya jika lineage tetap lengkap.

---

## 9. Pipeline End-to-End yang Wajib Diulang dari Awal

### Tahap A — Raw Audit dan Manifest

Sebelum cleaning:

1. Enumerasi semua file untuk `ID-01..ID-32`.
2. Validasi tidak ada ID di luar rentang.
3. Hitung checksum file.
4. Parse participant, phase, sensor, channel, dan source path.
5. Hitung sampling rate aktual dari delta timestamp; jangan percaya nominal rate.
6. Audit durasi, row count, duplicate timestamp, non-monotonic timestamp, gaps, NaN, infinite value, clipping, dan malformed rows.
7. Buat `manifest.csv` dan `manifest_issues.csv`.
8. Raw source tidak boleh ditimpa.

### Tahap B — Koreksi Timestamp ECG

ECG diketahui memiliki masalah shifting/time drift, sehingga koreksi wajib dilakukan sebelum perbandingan peak.

Urutan:

1. Normalize timestamp ECG dan PPG ke satu timebase UTC.
2. Gunakan anchor/session start sebagai coarse alignment.
3. Bentuk kandidat timeline ECG dari urutan sample asli serta metadata
   `samplescount`, `duration`, dan `stoptime`. Effective sampling rate adalah
   `(sample_count - 1) / duration` dan hanya diterima bila selisih terhadap
   nominal tidak melebihi 10%.
4. Detect preliminary ECG R-peaks dan PPG beats pada segmen kualitas memadai.
5. Estimasikan offset dan, bila diperlukan, linear clock drift pada ECG.
6. Koreksi **timestamp ECG**, bukan memindahkan PPG untuk memaksa score tinggi.
7. Simpan parameter koreksi per participant/session:

```text
participant_id
initial_offset_ms
drift_slope
correction_method
pre_alignment_score
post_alignment_score
```

8. Jangan memaksa ECG R-peak dan PPG systolic peak berada pada timestamp identik; ada delay fisiologis antara aktivitas listrik jantung dan pulse perifer.
9. Alignment hanya diterima bila meningkatkan konsistensi IBI/beat matching tanpa menghasilkan time warp tidak fisiologis.
10. Lead-in/lead-out ECG ditangani melalui crop anchor. Dilarang mengompresi
    durasi full ECG agar endpoint-nya sama dengan PPG tertrim, dan PPG/F1 kanal
    tidak boleh dipakai untuk memilih kandidat timebase ECG.

### Tahap C — Cleaning

Cleaning dilakukan terpisah untuk tiap sensor/channel.

Wajib:

- sort timestamp;
- remove exact duplicates secara deterministik;
- standardize units;
- handle invalid/NaN secara tercatat;
- reject long dropout;
- detrend/baseline handling bila dibutuhkan;
- zero-phase filter offline untuk research preprocessing bila digunakan;
- semua filter disimpan di `cleaning.yaml`.

Baseline filter candidate yang harus diverifikasi terhadap data aktual:

```text
ECG       : bandpass kira-kira 0.5–40 Hz
PPG IR/Red: bandpass kira-kira 0.5–8 Hz
PPG Green : bandpass kira-kira 0.5–5 Hz
```

Nilai tersebut bukan alasan untuk melewati tuning berbasis sampling rate nyata dan inspection hasil filter.

### Tahap D — MADC: Motion Artifact Detection and Correction

MADC wajib menggunakan accelerometer bila tersedia.

Minimal features/checks:

- accelerometer magnitude;
- rolling variance / energy;
- abrupt PPG morphology change;
- amplitude outlier;
- baseline shift;
- spectral contamination;
- disagreement antar beat.

Policy:

1. **Detection first**: tandai window/region terdampak motion.
2. Correction ringan hanya bila bentuk fisiologis dapat dipertahankan.
3. Untuk artifact berat, lebih baik reject window daripada merekonstruksi sinyal palsu.
4. Jangan gunakan smoothing agresif yang mengubah peak timing.
5. Simpan `motion_score`, `madc_action`, dan sebelum/sesudah metric.

### Tahap E — SQA / SQI

SQA dijalankan dua lapis:

**Pre-peak SQA** mencegah peak palsu pada window buruk.

**Post-peak SQA** memeriksa apakah peak/IBI hasil deteksi masih masuk akal.

Minimal checks:

- missing ratio;
- gap ratio;
- sampling jitter;
- flatline;
- clipping/saturation;
- low amplitude;
- motion score;
- morphology consistency/template similarity;
- plausible beat count;
- IBI outlier ratio;
- filter edge artifacts;
- transition overlap.

Quality tier:

```text
excellent | good | acceptable | poor | invalid
```

`poor` dan `invalid` tidak boleh masuk training/evaluation normal.

### Tahap F — Peak-to-Peak Detection

#### ECG

- Detect R-peaks menggunakan NeuroKit2 `ecg_peaks/ecg_process` sebagai primary implementation.
- Jika raw file mempunyai peak flag, gunakan sebagai secondary reference, bukan kebenaran tanpa audit.
- Simpan R-peak timestamp, sample index, RR interval, dan quality context.

#### PPG

- Process tiap wavelength secara independen.
- Primary: NeuroKit2 PPG Elgendi pipeline.
- Secondary/fallback: `scipy.signal.find_peaks` dengan distance/prominence/width yang dikonfigurasi dari sampling rate dan heart-rate plausibility.
- Jangan memakai satu threshold peak untuk semua participant/channel tanpa validasi.

#### ECG–PPG beat matching

Per participant × phase × PPG channel dalam core window validasi tepat 240 detik:

1. Detect ECG R-peaks dan PPG systolic peaks secara independen.
2. Koreksi ECG clock offset/drift terlebih dahulu.
3. Estimasikan median delay ECG→PPG dan pertahankan delay fisiologis; jangan paksa zero-lag.
4. Match beats one-to-one dalam tolerance yang ditentukan dari distribusi delay bersih.
5. Bandingkan terutama **RR/IBI ECG vs PPI/IBI PPG**, karena interval lebih relevan untuk validasi HR/HRV daripada timestamp peak absolut. Pasangan interval hanya sah bila kedua pasangan peak berurutan langsung (`ECG i→i+1` dan `PPG j→j+1`); jangan menghubungkan blok match yang terpisah oleh gap atau unmatched beat.
6. Bila PPG foot detection stabil, pulse-arrival-time boleh disimpan sebagai metric sekunder.

Metric wajib:

```text
beat precision
beat recall
beat F1
matched beat ratio
mean absolute IBI error (ms)
IBI RMSE (ms)
median absolute IBI error (ms)
IBI Pearson/Spearman correlation
HR MAE
Bland–Altman bias + limits of agreement
```

Hasil disimpan per participant, phase, dan channel; jangan hanya satu angka global. Selalu emit 192 baris, termasuk baris exclusion dengan alasan eksplisit.

---

## 10. Interpolasi yang Diizinkan

Interpolasi tidak boleh dipakai untuk “menciptakan” rekaman yang hilang.

### Diizinkan

- Linear/PCHIP untuk gap waveform yang pendek dan jelas terisolasi, jika gap policy di config terpenuhi.
- Cubic spline/PCHIP pada **IBI tachogram** setelah beat artifact/missed beat ditandai, hanya bila titik valid di sekitar gap cukup.
- Resampling untuk feature-domain HRV bila menggunakan metode yang memang mensyaratkannya.

### Dilarang

- Mengisi long dropout atau hampir seluruh window.
- Interpolasi melewati batas `relax ↔ task`.
- Menggunakan interpolation untuk memperbaiki reused internal test sampai metric model naik.
- Dynamic warping raw signal untuk memaksa ECG dan PPG tampak cocok.
- Menginterpolasi data tanpa menyimpan mask/provenance.

Setiap interpolated sample/beat wajib memiliki flag.

### 10.1 Kebijakan ECG untuk validasi primer

Pada primary ECG–PPG peak-to-peak validation, **waveform ECG tidak boleh
diinterpolasi**. ECG hanya diproses dari sample asli/clean setelah koreksi
timestamp ECG. Gap ECG wajib disimpan sebagai `gap_mask`; peak yang berada di
area gap atau wilayah terdampak gap dikeluarkan dari ground truth, dan long
dropout menghasilkan exclusion eksplisit. `ecg_interpolated_sample_count`
untuk jalur primer harus tetap nol.

Interpolasi RR/IBI hanya analisis sekunder setelah R-peak asli terdeteksi,
default-nya nonaktif, dan hasil `observed` harus dipisahkan dari hasil
`interpolated_secondary`. Interpolasi PPG hanya boleh terjadi pada canonical
preprocessing Silver; validator tidak boleh memfilter atau menginterpolasi PPG
untuk kedua kalinya. `data_foundation.py` dipertahankan sebagai adapter/reference
legacy dan bukan jalur canonical ECG–PPG validation.

---

## 11. Exploratory Data Analysis

EDA dilakukan setelah raw audit dan diulang setelah cleaning/SQA.

### Automated EDA

Gunakan:

- `ydata-profiling` untuk manifest, summary table, Gold features, missingness, correlation, dan time-series overview.
- NeuroKit2 report/plot untuk inspection ECG/PPG process.
- `tsfresh` boleh dipakai sebagai **candidate automated feature generator**, tetapi feature final harus melalui relevance/stability/leakage review.

### EDA wajib

Per participant/phase/channel:

- duration dan sampling-rate distribution;
- missing/gap/jitter distribution;
- SQA distribution;
- motion artifact distribution;
- raw vs cleaned waveform sample;
- peak count dan heart-rate distribution;
- IBI/HRV distribution;
- ECG–PPG agreement;
- class balance relax/task;
- participant-level separability;
- channel comparison IR/Red/Green;
- outlier windows;
- feature correlation dan redundancy.

EDA tidak boleh memakai reused internal test untuk memilih threshold atau features.

---

## 12. Feature Engineering

Stress model Android adalah **PPG-only**. Green, IR, dan Red dievaluasi sebagai model kanal independen; satu model tidak boleh menerima fitur kanal lain. ECG/PAT hanya reference validation.

### Predictor default

```text
median_hr_bpm
ppi_median_ms
ppi_iqr_ms
ppi_cv
rmssd_ms
sdnn_ms
pnn50
pulse_amplitude_cv
normalized_pulse_amplitude_median
pulse_width_half_height_median_ms
rise_time_median_ms
decay_time_median_ms
normalized_pulse_area_median
normalized_rise_slope_median
normalized_decay_slope_median
```

Morphology memakai trough-to-peak amplitude dan robust scale per window. `ibi_std` dilarang karena duplikat `sdnn`; `pulse_width` lama yang sebenarnya inter-peak interval, raw mean/median/std/IQR, rolling extrema, raw area/slope, pseudo-clipping berbasis percentile, serta hardcoded quality values tidak boleh menjadi predictor.

### Experimental-only

```text
sd1_ms | sd2_ms | sample_entropy
waveform_skewness | waveform_kurtosis
log_lf_power | log_hf_power | lf_hf_ratio
```

Experimental feature hanya boleh dipromosikan melalui development ablation yang participant-safe. Spectral PRV bukan default; kestabilannya terhadap durasi window dan agreement ECG–PPG wajib dibuktikan. `LF/HF` tidak boleh ditafsirkan sebagai ukuran langsung sympathovagal balance.

### Quality gate-only dan metadata terlarang

Peak count, physiological/valid IBI ratio, SQI, missing/interpolated/artifact/clipping/flatline ratio, motion, sampling jitter, quality tier, dan quality flags hanya untuk gate/audit. Participant/session ID, phase, label name, path/checksum, timestamp/window index, sampling rate, channel name, ECG, dan PAT dilarang menjadi predictor. Non-finite feature menyebabkan window ditolak; jangan menggantinya dengan nol.

Feature selection, correlation pruning absolut `>0,95`, scaling, dan imputation hanya dilakukan di training fold.

### Waveform input untuk DL

- simpan filtered PPG waveform tensor per channel;
- fixed shape contract;
- normalization parameters versioned;
- mask invalid/interpolated samples bila model menggunakan mask;
- jangan mencampur channel sebelum tiap channel lolos quality analysis.

---

## 13. Participant-Safe Split dan Anti-Leakage

Tidak pernah split berdasarkan row/window secara random.

### Fixed split

```text
Development          : ID-01 .. ID-28
Reused internal test : ID-29 .. ID-32
```

Development memakai tujuh predefined outer folds berisi empat participant dan inner `GroupKFold(5)`. Window overlap bukan sample independen: setiap participant-phase memiliki total weight sama dan primary metrics dihitung setelah median aggregation per participant-phase.

Fit berikut hanya pada training fold:

- scaler;
- imputer;
- feature selector;
- class balancing;
- augmentation;
- calibration;
- threshold tuning;
- hyperparameter optimization.

Reused internal test dipanggil satu kali setelah window, quality cohort, feature set, model, HPO, calibration, dan threshold dibekukan. Status klinis selalu `external_validation_required`.

---

## 14. Augmentasi yang Diizinkan

Augmentation hanya remediation untuk train set dan **selalu setelah participant split**.

### Candidate yang diizinkan dengan batas kecil dan tervalidasi

- low-amplitude Gaussian jitter/noise;
- small multiplicative amplitude scaling;
- mild smooth magnitude warping;
- conservative crop/window shift yang tidak melewati phase boundary;
- same-class pattern mixing hanya bila waveform plausibility check lulus dan tidak merusak peak/IBI physiology.

### Dilarang default

- time reversal;
- amplitude inversion;
- arbitrary permutation yang merusak urutan beat;
- aggressive time warp yang mengubah heart-rate physiology;
- synthetic waveform generative model sebagai pengganti peserta baru;
- augmentation pada validation/reused internal test;
- augmentation sebelum split.

Setiap augmentation harus dibandingkan dengan no-augmentation baseline. Pertahankan hanya jika meningkatkan participant-level generalization tanpa merusak ECG-consistency/morphology checks.

---

## 15. Training Strategy — Classical ML Lokal dan DL Conditional Colab

### 15.1 Classical baseline

Mulai dari model yang interpretable dan kecil:

```text
Logistic Regression
RBF SVM
Extra Trees
LightGBM
XGBoost
```

Classical ML berjalan lokal per kanal untuk Green, IR, dan Red. Lima fixed baseline dibandingkan pada session-level. Dua kandidat terbaik dituning dengan Optuna TPE, 50 trial, seed 42, dan persistent SQLite. Jika BA berbeda `<=0,01`, tie-break adalah F1, ROC-AUC, participant stability, lalu model paling sederhana/exportable.

Quality cohort yang wajib dibandingkan:

```text
strict   : excellent + good
expanded : excellent + good + acceptable
```

Jika selisih BA `<=0,01`, pilih strict. Quality tier tidak menjadi predictor.

Window ablation per kanal:

```text
60/30 | 120/30 | 180/10 | 180/30 | 240/240
```

`180/10` tetap baseline. Kandidat lain hanya mengganti baseline bila session BA naik minimal `0,02`, seluruh gate lulus, coverage tidak berkurang, dan participant variability tidak memburuk. Filter diterapkan sekali sebelum core crop, window dibentuk dari elapsed timestamp, dan padding/cross-phase/long-gap interpolation dilarang.

### 15.2 Deep-learning escalation

Jika baseline tidak mencapai gate secara stabil, coba bertahap:

```text
1D-CNN
TCN
CNN-BiLSTM / lightweight GRU
```

DL hanya dijalankan untuk kanal terkait bila salah satu development metric `balanced_accuracy`, `F1`, atau `ROC-AUC` berada di bawah minimum `0,70`. Red mengikuti alur yang sama dan tidak boleh digabung dengan Green/IR. Runtime wajib Google Colab T4 dan notebook gagal eksplisit bila T4 tidak tersedia. Input hanya Gold PPG kanal terkait; ECG, PAT, motion, dan sibling channel dilarang. Checkpoint Google Drive wajib atomik per epoch, setiap lima fold, serta pada interrupt/exception; resume menolak checksum Gold/config/fold contract yang berbeda.

### 15.3 Optimization discipline

- seed tetap;
- deterministic config sebisa mungkin;
- early stopping hanya memakai validation fold;
- checkpoint best metric participant-level;
- jangan tune terhadap reused internal test;
- simpan config, package versions, seed, git commit jika repository menggunakan Git, dan model checksum.

---

## 16. Performance Gate dan Remediation Loop

Model tidak boleh diekspor ke Android hanya karena training accuracy tinggi.

Gate bertingkat:

```text
research_usable : balanced_accuracy >= 0.70, F1 >= 0.70, ROC-AUC >= 0.70
preferred       : balanced_accuracy >= 0.80, F1 >= 0.78, ROC-AUC >= 0.83
```

Status wajib: `performance_tier=below_minimum|research_usable|preferred` dan `deployment_scope=none|research|standard`. Accuracy tetap report-only. Tier research wajib menyimpan dan menampilkan disclaimer keterbatasan; semua tier tetap `external_validation_required`.

Laporkan juga:

- confusion matrix;
- sensitivity/recall stress;
- specificity/recall relax;
- precision;
- calibration/Brier score bila probabilistic output digunakan;
- per-participant metrics;
- mean ± std antar folds;
- bootstrap CI bila feasible.

### Jika gate gagal

Lakukan berurutan:

1. audit label/phase dan anchor;
2. audit ECG timestamp correction;
3. audit sampling rate dan dropout;
4. review MADC/SQA false rejection/false acceptance;
5. review peak detector per channel;
6. review ECG–PPG IBI agreement;
7. perbaiki allowed interpolation untuk short gaps bila relevan;
8. review normalization/features;
9. lakukan feature selection ulang **di training fold**;
10. coba allowed augmentation **di training fold**;
11. tune classical model;
12. escalate ke DL compact;
13. ulang grouped validation;
14. evaluasi reused internal test hanya setelah keputusan dibekukan.

Jangan:

- menurunkan minimum research gate di bawah 0,70 secara ad hoc;
- menghapus participant sulit hanya agar metric naik;
- memilih preprocessing berdasarkan reused internal test;
- membuat fake participant;
- menambah data manual baru.

---

## 17. ONNX Export

Hanya model yang lolos gate yang menjadi release candidate.

### Export requirements

- export model dengan input/output names eksplisit;
- fixed/versioned preprocessing contract;
- ONNX checker lulus;
- ONNX Runtime local inference lulus;
- native vs ONNX output parity `max_abs_diff < 1e-5` untuk FP32 candidate;
- quantization hanya diterima bila penurunan metric tetap dalam batas yang ditetapkan dan semua performance gate tetap terpenuhi.

Untuk PyTorch, gunakan exporter `torch.export`/`torch.onnx.export(..., dynamo=True)` sesuai dokumentasi current PyTorch.

### Model manifest wajib

```json
{
  "model_name": "stress_classifier",
  "model_version": "...",
  "input_channel": "ppg_green|ppg_ir|ppg_red",
  "window_seconds": 180,
  "stride_seconds": 10,
  "preprocessing_version": "...",
  "feature_version": "...",
  "decision_threshold": 0.5,
  "development_participants": "ID-01..ID-28",
  "holdout_participants": "ID-29..ID-32",
  "test_cohort_status": "reused_internal_test_cohort",
  "clinical_validation_status": "external_validation_required",
  "performance_policy_version": "ml_dl_tiered_v1",
  "performance_tier": "research_usable|preferred",
  "deployment_scope": "research|standard",
  "balanced_accuracy": 0.0,
  "accuracy": 0.0,
  "f1": 0.0,
  "roc_auc": 0.0,
  "onnx_sha256": "..."
}
```

Jangan hardcode contoh metric `0.0` sebagai release evidence; nilai manifest harus diisi otomatis dari evaluator nyata.

---

## 18. Android Stress Detection

### 18.1 Architecture

Gunakan Kotlin + Jetpack Compose dengan layered/Clean Architecture sederhana:

```text
presentation
↓
domain
↑
data / infrastructure
```

UI mengikuti unidirectional data flow (UDF): state turun, event naik.

### 18.2 Runtime pipeline

```text
PPG input
→ stream sanity check
→ SQA
→ channel preprocessing sesuai manifest
→ finalize 300s → core [30s,270s) → 180s window / 10s stride (maksimal 7)
→ normalization
→ ONNX tensor
→ ONNX Runtime Mobile
→ calibrated probability
→ window result
→ 5-minute aggregation
→ Stress / Relax result + confidence + signal quality
```

Jika SQA buruk, **jangan tampilkan prediksi confident**; expose explicit UI state
`INSUFFICIENT_SIGNAL`.

### 18.2.1 Wear transport dan grafik realtime

- Watch memakai tracker Samsung individual `PPG_GREEN`, `PPG_IR`, dan
  `PPG_RED`, mengikuti pola PDSatu. Green dan IR adalah kanal inference, Red
  diagnostik. `PPG_CONTINUOUS` tetap hanya legacy/reference dan tidak boleh
  masuk active dependency graph.
- Watch mengirim immutable batch langsung melalui `MessageClient`. Active path
  tidak memakai `SESSION_READY`, durable outbox, batch ACK, atau terminal
  receipt sebagai gate. `sessionId` dan nonce tetap wajib untuk menolak batch
  sesi lama; batch hilang dicatat sebagai `transport_gap`.
- Timer sesi dimulai saat tracker individual didaftarkan dan berjalan maksimal
  300 detik. Gangguan phone sementara tidak boleh menghentikan tracker. Phone
  memproses hanya sample yang benar-benar diterima; gap panjang menjadi alasan
  abstain, bukan alasan membuat data sintetis.
- Grafik preview hanya merender sampel aktual yang telah diurutkan berdasarkan
  `elapsedMillis`, memakai tail 10 detik dan reduksi ringan hanya bila perlu
  untuk rendering. Bentuk data aktual tidak boleh dihaluskan dengan sample
  buatan; long gap tetap terputus. Setelah finalisasi, grafik memakai
  `CLEANED_CORE` yang sama dengan input inference, bukan filter kedua atau
  normalisasi min--maks per batch.
- State koneksi wajib eksplisit: `NODE_DISCOVERED`, `APP_REACHABLE`,
  `STREAMING`, `DRAINING`, `OFFLINE`, dan `PROTOCOL_ERROR`. Discovery bukan
  bukti aplikasi reachable. Home stress detection wajib menunggu fresh
  correlated pong pada fase `APP_REACHABLE` sebelum mengirim `SESSION_START`;
  node-only tidak boleh memulai sesi. Command/batch/ping terakhir menjadi
  evidence reachability setelah sesi aktif. Intentional disconnect ketika
  stopping/draining tidak boleh mengubah hasil normal menjadi error, dan
  stop/flush/disconnect harus idempotent.
- Laporan Home dan live Verbatim menyimpan `BpmSummary` observed-only per Green
  dan IR: mean, median, minimum, maksimum, jumlah observasi, serta peak valid.
  Summary hanya berasal dari PPG clean core/window valid; nilai hilang tetap
  nullable dan dilarang diganti default atau interpolasi sintetis.
- Performance gate `0,70` mengukur BA/F1/ROC-AUC model, bukan batas minimum
  probability setiap sesi. Probability ONNX terkalibrasi, confidence, signal
  quality, dan performance tier wajib dipisahkan di UI dan laporan. Dilarang
  menaikkan probability secara artifisial.

### 18.3 Keputusan 5 menit

Untuk sesi 5 menit:

- kumpulkan seluruh valid overlapping-window probability;
- tolak sesi bila valid-window coverage di bawah threshold config;
- agregasi dengan median/trimmed mean atau metode yang sudah divalidasi;
- simpan jumlah valid/invalid windows;
- hasil UI: `Relax`, `Stress`, atau `INSUFFICIENT_SIGNAL` dengan alasan quality.

### 18.4 ONNX Runtime

Mulai benchmark dengan CPU/XNNPACK; benchmark NNAPI pada device target. Jangan mengasumsikan NNAPI selalu lebih cepat.

Inference harus off-main-thread dan resource native harus ditutup deterministik.

---

## 19. Android Parity

Python adalah preprocessing reference implementation sampai Kotlin parity disahkan.

Generate test vectors dari **data nyata yang telah dide-identifikasi dan disetujui**, bukan dummy biosignal.

Setiap vector menyimpan:

```text
source participant/window reference
raw input excerpt/reference
expected cleaned tensor
expected feature tensor jika ada
expected native model probability
expected ONNX probability
manifest/checksum
```

Kotlin harus menghasilkan output dalam tolerance yang disetujui.

Invalid SQA vector harus menghasilkan `no prediction`.

---

## 20. Android Clinical Companion — Tanpa Authentication dan Tanpa RBAC

Aplikasi adalah single-device, single-experience tool. Tidak ada login, authentication flow, role model, atau role-based access control.

Privacy tetap wajib melalui local-only storage, consent, encryption, dan explicit delete/export controls.

Navigasi 5-Tab Bottom Bar (Wajib):

```text
Home (Legacy -- Stress Detection)
History (Legacy -- Stress Detection Report)
Verbatim+ (Verbatim Analysis + Live PPG Monitoring)
AI+ (AI Counselor Chatbot RAG Gemma 4 E2B)
Profile (Settings, Model Download, Knowledge Base, Storage Folder)
```

Semua menu nyata; tidak boleh placeholder/dummy screen.

---

## 21. Verbatim Analysis / AI+

Pipeline wajib nyata:

```text
Consent
→ Audio Record / Import
→ VAD
→ Speech-to-Text
→ Speaker Diarization
→ Timestamp Alignment
→ Non-destructive Transcript Cleaning
→ Manual Annotation
→ Psychological/Counseling Coding Suggestion
→ AI Analysis dengan RAG citation
→ Export / Delete
```

### Requirement Khusus Verbatim+ (Tugas Agen Android)

- **Diarization**: Agen Wajib mengimplementasikan pemisahan multi-pembicara (Psikolog, Pasien) pada UI dan STT.
- **STT**: Agen Wajib mendukung stream STT Bahasa Indonesia & Inggris secara Real-time.
- **Timestamp**: Tiap baris percakapan wajib diberi timestamp presisi.
- **Audio review seek**: Playback wajib mempublikasikan posisi dan durasi audio yang tervalidasi; setiap baris transcript dan marker review harus menyediakan aksi seek ke `startOffsetMs`/`episodeStartElapsedMillis` yang sama, tanpa offset sintetis atau fallback audio palsu.
- Seluruh aksi marker/transcript memakai satu operasi atomik `seekAndPlay` yang
  prepare bila perlu, seek, lalu play di bawah action gate yang sama.
- **Unlimited Duration**: Manajemen memori OOM harus dihindari dengan men-stream rekaman ke local storage.
- **Folder Management**: Saat di-*save*, sistem harus membuat folder bernama [Nama Sesi Kustom]_[Tanggal: DD-MM-YYYY]_[Waktu: HH:MM:SS] dan menyimpan 4 file: Audio, Live Transcript, Divergency Events log, dan AI Summary.
- **AI Summary**: Prompt ke Gemma 4 E2B dijalankan otomatis saat sesi berakhir.

`sherpa-onnx` dapat digunakan sebagai kandidat karena menyediakan Android STT/VAD/TTS serta speaker diarization APIs; benchmark kualitas Bahasa Indonesia dan resource usage sebelum dikunci.

---

## 22. Live Monitor + Verbatim Integration

Stress inference boleh berjalan bersamaan dengan recording/verbatim bila device benchmark lulus.

Event model:

```text
session_timestamp
stress_probability
stress_class
signal_quality
speaker_segment_id (opsional)
transcript_segment_id (opsional)
annotation
```

### Aturan istilah

Jangan implementasikan atau tampilkan:

```text
lie detected
deception detected
truth score
patient is lying
```

PPG/HRV mencerminkan respons fisiologis/arousal yang tidak spesifik terhadap kebohongan.

Istilah yang boleh:

```text
physiological arousal increase
stress-response divergence
hesitation/arousal marker
perlu ditinjau oleh psikolog
```

Fitur ini hanya membantu psikolog menandai timestamp yang layak ditinjau bersama konteks verbatim; bukan menentukan kebenaran pernyataan pasien.

### 22.1 Acoustic evidence contract

Implementasi Android canonical memakai `AcousticFrameAnalyzer` pada frame PCM nyata dari `AudioRecord`.
Analyzer wajib menghitung RMS dBFS, zero-crossing rate, clipping ratio, finite-sample ratio, quality score,
dan deviasi terhadap baseline sesi tanpa membuat waveform atau data audio sintetis. Live waveform dibatasi
hingga 64 titik envelope per frame; audio penuh tetap berada pada storage/session artifact.

Marker taxonomy yang disimpan dan ditampilkan adalah:

```text
STRESS_ELEVATED
ACOUSTIC_DIVERGENCE
AUXILIARY_EVENT
INCONSISTENCY_CUE
LOW_AUDIO_QUALITY
MANUAL
```

Low audio quality adalah evidence kualitas, bukan stress/arousal. Marker
fisiologis BPM trajectory, model arousal, acoustic divergence, dan low-audio
quality wajib memakai source berbeda. Acoustic divergence membutuhkan warm-up,
persistence, hysteresis, cooldown, serta episode deduplication; satu frame buruk
tidak boleh menghasilkan marker baru.

Finalisasi Verbatim+ wajib idempotent dan memisahkan kegagalan audio, Watch,
snapshot, dan persistence. State canonical adalah `STOP_REQUESTED`,
`AUDIO_DRAINING`, `PPG_DRAINING`, `SNAPSHOT_CAPTURED`, `PERSISTING`, lalu
`COMPLETED` atau `FAILED_WITH_EVIDENCE`. Exception subsistem diubah menjadi
evidence/UI state dan tidak boleh dibiarkan menutup proses sebelum data lain
disimpan.

`ACOUSTIC_DIVERGENCE` berarti deviasi energi akustik dari baseline speaker/sesi dan `INCONSISTENCY_CUE`
hanya berarti cue untuk review manusia. Keduanya bukan deception detector. `evidenceConfidence` acoustic
diturunkan dari kualitas frame dan tidak boleh dipresentasikan sebagai probabilitas klinis atau confidence
model stres. Marker transition dipersistenkan satu kali per perubahan status agar long silence/clipping tidak
menghasilkan ribuan row identik.

Status verifikasi saat ini: source-isolated Kotlin compile dan Android unit suite lulus; acceptance mikrofon
fisik, benchmark resource, dan validasi end-to-end pada perangkat target tetap wajib sebelum fitur dinyatakan
siap rilis.

---

## 23. AI Counselor + RAG

### Model on-device

Nama resmi yang terverifikasi untuk varian `E2B` adalah **Gemma 4 E2B**.

Jika source code/spec lama memakai nama model E2B yang tidak ada pada dokumentasi resmi, agent harus memperlakukannya sebagai nama yang tidak terverifikasi dan menggunakan Gemma 4 E2B sampai PRD diperbarui berdasarkan dokumentasi resmi yang baru.

Runtime kandidat:

- Google AI Edge / MediaPipe LLM Inference atau LiteRT-LM yang mendukung model target.
- Selalu benchmark memory, first-token latency, token throughput, dan thermal behavior pada device target.

### RAG

Knowledge base lokal dapat memuat dokumen yang secara legal dimiliki/diberikan pengguna, misalnya:

```text
PNPPK
PPDGJ
ICD references yang diizinkan
materi konseling/terapi yang memiliki izin penggunaan
SOP internal psikolog
```

Pipeline:

```text
Document import
→ text extraction
→ chunking
→ metadata + source/page
→ local embedding
→ local vector retrieval
→ relevance threshold
→ grounded prompt
→ Gemma generation
→ citation rendering
```

Aturan:

- jangan menjawab clinical-adjacent recommendation tanpa sumber bila pertanyaan memerlukan grounding;
- bila retrieval rendah, katakan sumber lokal tidak cukup;
- jangan membuat citation palsu;
- jangan diagnosis;
- jangan menggantikan judgement psikolog;
- crisis/safety flow harus deterministic dan terjadi sebelum generative response bila aplikasi memiliki fitur tersebut.

### Kontrak canonical RAG

`configs/rag.yaml` adalah source of truth untuk format, chunk policy, embedding, retrieval, persona, safety, storage, dan release gate AI+.

- Input yang didukung hanya PDF, DOCX, strict UTF-8 TXT, dan Markdown. Binary `.doc`, format ambigu, malformed, atau password-protected harus gagal eksplisit.
- Ekstraksi mempertahankan page/section/paragraph/table provenance serta status `COMPLETE|OCR_COMPLETE|PARTIAL|FAILED`. Dokumen `PARTIAL` tidak retrieval-ready secara default.
- Chunk memakai tokenizer E5 yang sama dengan embedding: target 224, hard maximum 256, overlap 32; heading masuk budget dan chunk tidak melintasi halaman PDF atau top-level section DOCX.
- Embedding canonical: `intfloat/multilingual-e5-small`, `query:`/`passage:`, attention-mask mean pooling, L2 normalization, 384 dimensi, maksimum input 512 token, manifest dan checksum wajib cocok. Silent truncation dilarang.
- Hybrid retrieval: 30 semantic + 30 lexical, RRF `k=60`, adjacent-overlap deduplication, MMR `lambda=0.75`, maksimal empat evidence, tiga per dokumen, dan dua per halaman. SQLite FTS5/BM25 adalah release requirement; adapter in-memory hanya development evidence dan tidak boleh disamarkan sebagai FTS5.
- Threshold no-answer wajib dikalibrasi benchmark lokal terversi; profil 60-query boleh dipakai sebagai evidence lebih ketat. State uncalibrated selalu abstain; jangan menghidupkan jawaban dengan magic threshold.
- Lifecycle UI/worker dokumen wajib dibedakan menjadi `PENDING|INDEXING|PARTIAL|READY_LEXICAL|READY_SEMANTIC|REINDEX_REQUIRED|FAILED`; recovery stale memakai `processingUpdatedAtMillis`. Reindex discope oleh `documentId + contentChecksum + revision`, dan error extraction harus memakai kode terstruktur, bukan satu pesan generik.
- PDF diterima hingga tepat 100 MiB (`104_857_600` byte) tanpa batas halaman statis, lalu diproses per halaman dengan audit progres serta batas kumulatif teks/chunk. OCR hanya untuk halaman tanpa text layer; batas kumulatif harus menjadi `PARTIAL_LIMIT_REACHED`, bukan mengindeks dokumen parsial. Reindex hanya boleh menerbitkan generation atomik setelah seluruh chunk valid; checkpoint/resume per halaman yang durable tidak boleh diklaim sebelum staging chunk transactional tersedia. Generation kosong tidak boleh menggantikan generation `READY` yang masih valid. E5 tokenizer tetap mandatory untuk semantic retrieval; lexical-ready tidak berarti production retrieval boleh melampaui policy benchmark.
- Benchmark minimal memiliki 30 query dengan calibration/test split terpisah, Recall@5 `>=0,85`, citation hit@3 `>=0,80`, dan irrelevant-query abstention `>=0,90`. Profil benchmark 60-query yang lebih ketat diperbolehkan. Tanpa artifact benchmark yang terversi dan tervalidasi, policy tetap `UNCALIBRATED` dan retriever wajib abstain.
- AI+ wajib menyimpan `CounselorContextReceipt` dengan persona, retrieval-policy, citation document/page, source ID riwayat stress valid, alasan history yang dikecualikan, dan versi runtime LLM. Tidak boleh membuat citation atau context palsu.
- Citation registry hanya berisi evidence yang benar-benar masuk prompt. Label `[S#]`, `chunkId`, `documentId`, dan locator harus match tepat; grounded answer tanpa citation valid wajib abstain.
- Imported document adalah untrusted evidence. Role marker, tool instruction, prompt override, atau indirect prompt injection dari dokumen tidak boleh menjadi instruction model. LLM tidak memperoleh tool/action permission.
- Persona adalah profil terstruktur (`language`, `tone`, `responseLength`, `reflectiveQuestions`, optional `focusNote` maksimum 1.500 kata Unicode setelah normalisasi whitespace). Persona hanya mengubah gaya dan selalu di bawah crisis/safety/evidence policy.
- LiteRT generation memakai lifecycle terserialisasi. Pembatalan meminta cancel kooperatif sekali; engine/conversation tidak ditutup bersamaan dengan native stream. Runtime wedged hanya di-reset setelah generation berakhir dan tanpa menyimpan isi prompt ke log.
- Token output tidak boleh dirender selama streaming. UI hanya menampilkan typing indicator sampai final safety, diagnosis, groundedness, dan citation guard menerima jawaban.
- Web Search tidak pernah menjadi fallback diam-diam; perlu consent eksplisit dan citation `[W#]` terpisah.
- Source document wajib AES-GCM/Android Keystore dan knowledge database wajib SQLCipher dengan wrapped passphrase sebelum release. Generation index harus atomic dan policy/model lama harus `REINDEX_REQUIRED`.

Agent wajib melaporkan gate yang belum terbukti. Parser, retrieval class, atau unit test terisolasi tidak cukup untuk mengklaim RAG production-ready tanpa FTS5, benchmark, signed E5 parity, encrypted storage, Gemma production path, dan physical phone test.

---

## 24. Local Storage dan Privacy

Walaupun tanpa login/RBAC:

- audio, transcript, annotation, stress history, dan chat history disimpan lokal;
- gunakan Android Keystore-backed key untuk encryption key material;
- jangan log raw clinical content ke Logcat/release logs;
- sediakan clear/delete per session dan delete-all;
- consent recording/monitoring disimpan per session;
- export hanya atas tindakan eksplisit pengguna.

Tidak ada kebutuhan cloud untuk core workflow.

---

## 25. Testing Minimum

### Data/Signal

- manifest ID whitelist test (`ID-01..ID-32` only);
- timestamp monotonicity;
- sampling-rate estimation;
- ECG shift/drift correction;
- filter response;
- MADC detection;
- SQA tiering;
- peak detection;
- beat one-to-one matching;
- IBI interpolation mask;
- phase boundary leakage;
- participant split leakage.

### ML/DL

- deterministic split;
- preprocessing fitted train-only;
- augmentation train-only;
- grouped CV;
- performance evaluator;
- reused-internal-test single-use guard;
- model manifest generation.

### Export

- ONNX checker;
- native vs ONNX parity;
- quantized vs FP32 metric regression;
- Android vector parity.
- `android-prep` may generate pending raw vectors after ML/ONNX gates pass;
  this diagnostic artifact must retain `android_instrumentation_status=pending`
  and must not promote the model index.
- `PipelinePython/03_stress_onnx/tools/validate_model_index.py` lulus sebelum model index dianggap
  accepted atau asset Android dipaketkan; index yang tidak kosong tetapi
  belum `android_ready` tetap blocked.
- Index promotion additionally requires the referenced vector artifact to
  report `python_vector_status=passed`, `python_vectors_passed=100`,
  `android_instrumentation_status=passed`, and `android_ready=true`.
- Release-evidence snapshot wajib memakai validator yang sama dan tidak boleh
  mengubah missing/failed evidence menjadi pass.

### Android

- SQA failure blocks prediction;
- five-minute aggregation;
- model/manifest compatibility;
- no blocking inference on UI thread;
- recording lifecycle;
- STT failure state;
- diarization relabeling;
- transcript immutability;
- RAG citation provenance;
- low-retrieval fallback;
- local deletion/export;
- combined Verbatim + Live Monitor resource benchmark.

### 25.1 Current verification boundary

Snapshot `20260808T120000Z` remains blocked: `training_status=ready`, while
`ppg_gold_status=blocked` with 191/192 groups (missing `ID-12 / relax /
ppg_red` because all seven windows are invalid), `ecg_validation_status=blocked`,
and `model_release_status=blocked`. The checked-in canonical validator now
emits `ibi_rmse_ms`. Versioned validation evidence was regenerated at
`20260809T120000Z` from the prior canonical Bronze/Sync/Silver boundary:
192/192 combinations, 117 valid, and 75 explicitly excluded. Android
unit/lint gates pass, but instrumentation, Samsung phone–Watch acceptance,
microphone acceptance, and resource benchmarks remain pending. No fake data or
model is a valid substitute for those gates.

---

## 26. Definition of Done

Sebuah perubahan hanya selesai jika:

- memenuhi PRD acceptance criteria;
- tidak menambah ID di luar 32 partisipan;
- menjaga lineage participant/phase/channel;
- tidak menimbulkan participant leakage;
- local lint/test lulus;
- failure states eksplisit;
- tidak ada dummy integration;
- performance/evidence berasal dari data nyata;
- dokumentasi/config diperbarui bila kontrak berubah.

Model stress siap Android hanya jika:

- ECG timestamp correction dan ECG–PPG validation terdokumentasi;
- SQA/MADC gate lulus;
- held-out metrics memenuhi semua gate;
- ONNX parity lulus;
- Android parity lulus;
- five-minute aggregation tervalidasi;
- invalid-quality signal tidak menghasilkan confident prediction.

Clinical companion siap dipakai hanya jika Verbatim, diarization, STT, local RAG/LLM, storage, consent, dan error handling benar-benar berjalan.

---

## 27. Protocol Respons Agent

Setiap kali agent menyelesaikan task, jawab ringkas dengan format:

```text
Scope:
Files changed:
What changed:
Local commands run:
Tests/lint:
Data/model evidence:
Remaining risk:
```

Jangan melaporkan “done” hanya karena kode berhasil ditulis. “Done” harus berdasarkan test/evidence.

---

## 28. Sumber Implementasi Teknis

Gunakan dokumentasi resmi/paper berikut sebagai rujukan implementasi, bukan copy-paste tanpa validasi:

### Biosignal / SQA / Peak

- NeuroKit2 ECG: https://neuropsychology.github.io/NeuroKit/functions/ecg.html
- NeuroKit2 PPG: https://neuropsychology.github.io/NeuroKit/functions/ppg.html
- SciPy `find_peaks`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
- SciPy `CubicSpline`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
- Wearable PPG review: https://pmc.ncbi.nlm.nih.gov/articles/PMC7612541/
- PPG signal quality assessment: https://pmc.ncbi.nlm.nih.gov/articles/PMC9478959/

### EDA / Feature Engineering

- YData Profiling time-series: https://docs.profiling.ydata.ai/latest/features/time_series_datasets/
- tsfresh: https://tsfresh.readthedocs.io/en/stable/text/introduction.html

### Leakage-safe validation

- scikit-learn GroupKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
- scikit-learn grouped CV overview: https://scikit-learn.org/stable/modules/cross_validation.html
- scikit-learn nested CV: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
- Optuna persistent RDB storage: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.RDBStorage.html
- TRIPOD+AI reporting: https://www.tripod-statement.org/

### HRV / PRV feature boundaries

- ESC/NASPE HRV standard: https://www.escardio.org/static-file/Escardio/Guidelines/Scientific-Statements/guidelines-Heart-Rate-Variability-FT-1996.pdf
- Wearable PRV review: https://pubmed.ncbi.nlm.nih.gov/29668452/
- LF/HF limitation: https://pubmed.ncbi.nlm.nih.gov/23431279/

### Time-Series Augmentation

- Time-series augmentation survey: https://pmc.ncbi.nlm.nih.gov/articles/PMC8282049/
- Medical wearable time-series augmentation survey: https://pmc.ncbi.nlm.nih.gov/articles/PMC11957733/

### ONNX / Android

- PyTorch ONNX exporter: https://docs.pytorch.org/docs/stable/onnx.html
- ONNX Runtime Mobile: https://onnxruntime.ai/docs/tutorials/mobile/
- ONNX Runtime quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- ONNX Runtime NNAPI: https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html
- Android architecture recommendations: https://developer.android.com/topic/architecture/recommendations
- Compose architecture/UDF: https://developer.android.com/develop/ui/compose/architecture
- Android Keystore: https://developer.android.com/privacy-and-security/keystore

### Local AI / Verbatim

- Gemma 4: https://ai.google.dev/gemma/docs/core/model_card_4
- Gemma mobile deployment: https://ai.google.dev/gemma/docs/integrations/mobile
- sherpa-onnx Android: https://k2-fsa.github.io/sherpa/onnx/android/build-sherpa-onnx.html
- sherpa-onnx speaker diarization: https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html

### Engineering

- BMAD Method: https://github.com/bmad-code-org/BMAD-METHOD
- Ruff: https://docs.astral.sh/ruff/
- detekt: https://detekt.dev/

### Batas klaim deception

- National Academies — The Polygraph and Lie Detection: https://nap.nationalacademies.org/catalog/10420/the-polygraph-and-lie-detection

---

## 29. Ringkasan Aturan Keras

```text
ONLY 32 PARTICIPANTS: ID-01 .. ID-32.
NO NEW PARTICIPANTS.
NO ROUTINE PHASE.
LOCAL EXECUTION EXCEPT CONDITIONAL DL IN GOOGLE COLAB T4.
ECG = VALIDATION REFERENCE, PPG = ANDROID MODEL INPUT.
CORRECT ECG TIMESTAMP/DRIFT BEFORE ECG–PPG COMPARISON.
SQA + MADC BEFORE TRUSTING PEAKS.
VALIDATE PPG AGAINST ECG WITH BEAT/IBI AGREEMENT.
SPLIT BY PARTICIPANT BEFORE FITTING OR AUGMENTATION.
AUGMENT TRAINING DATA ONLY.
INTERPOLATE SHORT, TRACEABLE GAPS ONLY.
MINIMUM RESEARCH GATE = BALANCED_ACCURACY/F1/ROC-AUC >= 70%; PREFERRED = 80%/78%/83%.
CLASSICAL ML LOCAL; DL ONLY IN CONDITIONAL GOOGLE COLAB T4 RUNS.
EXPORT ONLY GATE-PASSING MODEL TO ONNX.
VERIFY PYTHON ↔ ONNX ↔ KOTLIN PARITY.
5-MINUTE ANDROID RESULT = AGGREGATION OF VALID WINDOWS.
NO AUTHENTICATION.
NO RBAC.
NO DUMMY FEATURES OR FAKE CLINICAL INTEGRATIONS.
USE GEMMA 4 E2B AS THE VERIFIED E2B MODEL NAME.
VERBATIM/STT/DIARIZATION/RAG MUST BE REAL AND LOCAL-FIRST.
NEVER CLAIM PPG STRESS/AROUSAL AS LIE OR DECEPTION DETECTION.
CLEAN CODE + DRY + SOLID + LINTER + TESTS + BMAD/SDD.
```

---

## 30. Pipeline Data End-to-End: Raw → ML/DL-Ready (Addendum v4)

### 30.1 Sumber Data dan ECG Sync

- **DataRenamedPartisipan**: data aktual hasil perekaman, ID-01 .. ID-32.
- **DataTrimmedPartisipan**: data dipotong berdasarkan time anchor (5 menit Relax + 5 menit Task). ECG pada folder ini **rusak** (hanya header tanpa data).
- **Solusi ECG**: Ambil ECG dari DataRenamedPartisipan (sebelum pemotongan), trim ke anchor window, lalu lakukan time-shifting/regularisasi timestamp.
- PPG diambil dari DataTrimmedPartisipan karena sudah dipotong sesuai fase.
- Sumber phase tetap 300 detik per phase. Validasi ECG–PPG memakai core 240 detik (`anchor_start +30s` sampai `+270s`); Gold memakai window 180 detik/stride 10 detik dengan maksimal 7 window per phase.

### 30.2 Layer Pipeline

```text
Bronze  : Ingest raw CSV → normalize timestamps → dedup → lineage → Parquet
Silver  : Bandpass filter → MADC → SQA (7 checks) → quality_tier per channel
Gold    : HRV features + morphology per 180s window (stride 10s) → labels → split
```

### 30.3 Channel Training

Model ML/DL stress detection mengevaluasi `ppg_green`, `ppg_ir`, dan `ppg_red` secara independen tanpa fusion. Red dimulai sebagai `not_evaluated` dan hanya dipromosikan bila data, development, reused internal test, ONNX parity, dan runtime parity kanal Red lulus. Kegagalan Red tidak memblokir Green/IR. ECG dan PAT tidak masuk model—hanya untuk peak-to-peak comparison dan validasi fisiologis.

### 30.4 Bandpass Filter (cleaning.yaml)

```text
ECG        : 0.5–40 Hz, Butterworth order 4, zero-phase
PPG IR/Red : 0.5–8 Hz, Butterworth order 4, zero-phase
PPG Green  : 0.5–5 Hz, Butterworth order 4, zero-phase
```

### 30.5 Interpolasi & Augmentasi — Tanpa Fabrikasi

**Interpolasi diizinkan:**
- Linear/PCHIP untuk gap waveform ≤ 0.5 detik, window SQA `good`/`acceptable`.
- Cubic spline/PCHIP untuk IBI tachogram (missed beat terisolasi).
- Semua hasil interpolasi wajib memiliki mask/provenance.

**Interpolasi dilarang:**
- Long dropout, hampir seluruh window, cross-phase, warping ECG–PPG.

**Augmentasi diizinkan (training fold only, setelah split):**
- Gaussian noise rendah (SNR ~20 dB), amplitude scaling 0.8–1.2, crop/shift ringan.

**Augmentasi dilarang:**
- Time reversal, amplitude inversion, arbitrary permutation, aggressive time warp.
- Augmentasi di validation/reused internal test.

### 30.6 Gate Data Aktual

- Tidak ada ID di luar 01–32 dan tidak ada participant fiktif.
- Semua artifact mempertahankan lineage lengkap sampai ke raw source_file dan checksum.
- ECG-invalid participants (setelah audit) hanya dikeluarkan dari validasi ECG–PPG; PPG tetap boleh dipakai jika SQA PPG lolos.
- Reused internal test ID-29..ID-32 tidak boleh disentuh saat feature selection/tuning.

### 30.7 Pipeline CLI

```bash
python -m PipelinePython data --run-id <run_id> --resume
python -m PipelinePython peaks --run-id <run_id> --resume
python -m PipelinePython stress-onnx --run-id <run_id> --resume
python -m PipelinePython all --run-id <run_id> --resume
```

### 30.8 Android Wear, Live BPM, dan History

- Wear reachability wajib membedakan `NODE_DISCOVERED`, `APP_REACHABLE`, `STREAMING`, `DRAINING`, `OFFLINE`, dan `PROTOCOL_ERROR`; discovery tidak sama dengan aplikasi reachable atau streaming. Home hanya dapat dimulai dari fresh correlated pong pada `APP_REACHABLE`.
- Green, IR, dan Red memakai individual tracker serta batch aktual. Green/IR adalah predictor independen; Red diagnostik. Tidak ada sample, peak, BPM, marker, fusion, atau fallback sintetis.
- Verbatim+ menghitung latest/mean/median BPM per kanal dari peak PPG aktual. Marker fisiologis dan akustik memerlukan warm-up, quality gate, persistence, hysteresis, cooldown, serta episode deduplication. RMS tunggal atau frame audio buruk bukan bukti stress.
- Finalisasi wajib idempotent dan setiap kegagalan subsystem menjadi evidence yang dapat dirender, bukan crash.
- Listener tracker memanggil `HealthTracker.flush()` sebelum `unsetEventListener`; Green mengirim callback aktual segera setelah callback diterima (maksimum satu detik bila callback tersedia), tanpa upsampling atau sample buatan.
- History menyimpan cleaned-core chart `[30,270)` sebagai artefak app-private yang terkait `sessionId` dan `processingFingerprint`. Jika artefak tidak ada, UI menampilkan alasan; dilarang membuat grafik placeholder dari data palsu.
- `transportIncomplete` adalah warning audit sekunder. Hanya kegagalan core/SQA/model/minimum valid window yang membuat outcome `Invalid`; label kanal memakai decision threshold dan calibration version dari manifest kanal, bukan threshold global.
- Home dan Verbatim+ wajib memegang `PpgSessionLease` eksklusif dengan identity/mode/generation. Owner pertama mempertahankan sesi; start kedua gagal eksplisit tanpa menghentikan owner atau mencampur buffer/transcript.
- Semua day/week query memakai `Clock` dan `ZoneId`, dari `LocalDate.atStartOfDay(zone)` hingga awal hari berikutnya. `+24h` fixed dan string-date sebagai primary timestamp dilarang.
