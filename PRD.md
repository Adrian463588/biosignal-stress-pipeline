# PRD.md
# Biosignal Stress Detection & Android Clinical Companion

> **Bahasa:** Indonesia  
> **Metodologi:** BMAD + Spec-Driven Development + Clean Architecture + SOLID + DRY  
> **Eksekusi:** data pipeline, classical ML, export, testing, dan build dilakukan lokal; deep learning conditional berjalan di Google Colab T4
> **Hardware ML lokal:** AMD Ryzen 7 4800H + NVIDIA RTX 2060 Laptop GPU
> **Target aplikasi:** Android, Kotlin + Jetpack Compose

---

## 1. Ringkasan Produk

Produk ini membangun pipeline end-to-end untuk mengolah sinyal **PPG Samsung Galaxy Watch 5** dan **ECG Polar H10** dari **32 partisipan (ID-01 sampai ID-32)**, melatih model klasifikasi stres, mengekspornya ke **ONNX**, lalu menjalankannya secara lokal di aplikasi Android.

ECG digunakan sebagai **referensi fisiologis/cross-validation** untuk mengevaluasi apakah peak, IBI, HR, dan HRV yang diperoleh dari PPG cukup konsisten dengan ECG. ECG **bukan input model Android**.

Aplikasi Android berfungsi sebagai **alat bantu psikolog/konselor**, bukan alat diagnosis atau pengganti keputusan profesional.

---

## 2. Requirement yang Dikunci

```text
participants           : ID-01 .. ID-32 ONLY
new data acquisition   : tidak ada; pengambilan data telah selesai
classes                : Relax = 0, Task/Stress = 1
routine phase          : tidak ada / dilarang menjadi label
PPG                     : IR, Red, Green — Samsung Galaxy Watch 5
PPG training channels   : Green, IR, Red — setiap kanal dilatih dan dievaluasi independen
PPG validation/Gold     : IR, Red, Green — tepat 64 participant-phase group per kanal
ECG                     : Polar H10 — reference/cross-validation
ECG sync               : ECG dari DataRenamedPartisipan, timestamp dikoreksi via time-shifting
PPG source              : DataTrimmedPartisipan, fase dipotong berdasarkan TimeStampAnchor
source phase duration   : 300 detik per relax/task phase; raw tetap immutable
analysis core window    : 240 detik, anchor start +30 detik sampai +270 detik
session acquisition     : 300 detik penuh; data 0–300 detik disimpan untuk audit
prediction/validation   : 240 detik core `[+30s, +270s)` dari sesi 300 detik
model window            : 180 detik, stride 10 detik untuk Gold/training
Gold windows            : maksimal 7 per phase pada core 240 detik
label_name              : Normal untuk relax, Stres untuk task
classical ML           : lokal
deep learning          : Google Colab T4, hanya jika ML kanal terkait gagal gate
CPU                     : AMD Ryzen 7 4800H
GPU                     : NVIDIA RTX 2060 Laptop GPU / CUDA
random seed             : 42
model input production  : satu kanal PPG per model (Green, IR, atau Red), tanpa ECG/PAT/fusion
window                  : 180 detik
stride                  : 10 detik
session result          : agregasi prediksi selama 5 menit
model development       : ID-01 .. ID-28
reused internal test    : ID-29 .. ID-32; bukan external clinical validation
minimum research gate   : BA >= 0.70, F1 >= 0.70, ROC-AUC >= 0.70
preferred gate          : BA >= 0.80, F1 >= 0.78, ROC-AUC >= 0.83
accuracy                : dilaporkan, bukan gate atau trigger eskalasi
ONNX parity             : max_abs_diff < 1e-5 untuk kandidat FP32
```

### Aturan data

- Tidak boleh ada partisipan selain `ID-01..ID-32`.
- Tidak boleh membuat partisipan baru, fake participant, atau dummy physiological data.
- Augmentasi hanya transformasi **training data yang sudah ada**, bukan penambahan partisipan.
- Validitas ECG ditentukan ulang dari raw data; jangan hardcode daftar ECG-invalid.
- ECG yang tetap buruk hanya dikeluarkan dari validasi ECG–PPG. PPG masih dapat digunakan bila kualitas PPG memenuhi gate.
- `ppg_gold_status`, `training_status`, `ecg_validation_status`, dan `model_release_status` dilaporkan terpisah; kegagalan ECG tidak otomatis membuang PPG dari Gold/training.
- `ppg_gold_status` dan aggregate `training_status` mengukur 192 group (`32 × 2 × 3`); readiness juga wajib disimpan per kanal dengan target 64 group. Kegagalan satu kanal tidak memblokir kanal lain.
- Semua analisis memakai core window `[anchor_start +30s, anchor_start +270s)` tanpa padding, synthetic recovery, atau penghapusan raw 300 detik.
- Raw/audit `phase` boleh `relax`, `task`, atau `unknown` untuk menjaga traceability; Gold/training/model canonical hanya `relax` dan `task`. `routine` tetap dilarang. `label` adalah `0/1` dan `label_name` adalah `Normal/Stres`.
- Raw timestamp immutable; timestamp canonical dipakai untuk sinkronisasi dan `elapsed_seconds` dipakai untuk pemrosesan/windowing.
- Semua artifact mempertahankan lineage `participant_id → phase → sensor_channel → window_id`.

---

## 3. Scope

### In Scope

1. Audit dan cleaning raw ECG/PPG.
2. Koreksi timestamp/clock shifting ECG.
3. MADC — Motion Artifact Detection and Correction.
4. Signal Quality Assessment / SQI.
5. Peak-to-peak detection ECG dan PPG.
6. Validasi PPG terhadap ECG menggunakan beat/IBI agreement.
7. Exploratory Data Analysis otomatis dan manual.
8. Feature engineering dan waveform preparation.
9. Training classical ML lokal dan deep learning conditional di Google Colab T4.
10. Remediation preprocessing/augmentation/interpolation bila gate belum tercapai.
11. ONNX export, quantization bila aman, dan parity test.
12. Android 5-Tab Bottom Bar UI (Home, History, Verbatim+, AI+, Profile).
13. Home & History: Inference stres 5 menit dan laporan riwayat (Legacy).
14. Verbatim+: Penyatuan perekaman audio (Diarization/STT) dengan Live PPG Monitoring.
15. AI+: AI Counselor Chatbot berbasis RAG menggunakan **Gemma 4 E2B**.
16. Local storage, consent, encryption, export, dan delete.

### Out of Scope

- Penambahan partisipan baru.
- Cloud training selain Google Colab T4 untuk eskalasi deep learning conditional.
- Automation build/release remote.
- Platform experiment tracking/deployment eksternal.
- Authentication, login, atau RBAC.
- Clinical diagnosis atau medical-device certification.
- Mengklaim PPG sebagai alat pendeteksi kebohongan.
- Fake/dummy physiological dataset untuk menyatakan performa model.

---

## 4. End-to-End Pipeline

```text
Raw ECG + PPG — 32 partisipan, 300 detik per phase
        ↓
Raw audit + manifest + checksum
        ↓
Timestamp normalization
        ↓
ECG clock offset/drift correction
        ↓
Cleaning + filtering per sensor
        ↓
MADC + pre-peak SQA
        ↓
Core crop anchor +30s … +270s (240 detik)
        ↓
ECG R-peak + PPG systolic-peak detection
        ↓
ECG–PPG beat matching 240 detik + IBI/HR agreement
        ↓
Post-peak SQA
        ↓
EDA raw/clean/gold
        ↓
PPG-only feature engineering + waveform tensor (180 detik/10 detik, maksimal 7 window/phase)
        ↓
Participant-level split
        ↓
Classical ML baseline
        ↓
Deep learning di Google Colab T4 jika ML kanal terkait gagal gate
        ↓
Model gate bertingkat: minimum research atau preferred
        ↓
ONNX export + parity + optimization
        ↓
Kotlin preprocessing parity
        ↓
Android 5-minute stress inference (Home & History)
        ↓
Verbatim+ (Audio + Live PPG) & AI+ (RAG Chatbot) & Profile
```

---

## 5. Data Cleaning, Alignment, MADC, dan SQA

### 5.1 Raw Audit

Untuk setiap participant × phase × sensor:

- validasi filename dan participant ID;
- hitung checksum;
- hitung sampling rate aktual dari delta timestamp;
- audit duration, missing values, duplicate timestamps, non-monotonic timestamp, gaps, clipping, flatline, malformed row, dan dropout;
- raw file bersifat immutable.

### 5.2 ECG Timestamp Correction

Masalah timestamp ECG harus diperbaiki **sebelum validasi ECG–PPG**.

Urutan minimum:

1. Normalisasi ECG dan PPG ke satu timebase.
2. Gunakan anchor/session start untuk coarse alignment.
3. Bentuk timeline ECG dari urutan sample asli menggunakan metadata
   `samplescount`, `duration`, dan `stoptime`; effective rate
   `(sample_count - 1) / duration` wajib berada maksimal 10% dari sampling rate
   nominal.
4. Estimasikan offset ECG dan evaluasi drift sepanjang recording tanpa
   menyamakan endpoint full ECG dengan PPG tertrim.
5. Gunakan preliminary R-peak dan PPG peak untuk memeriksa alignment.
6. Bila diperlukan, terapkan koreksi linear/piecewise clock drift pada **timestamp ECG**.
7. Simpan `offset_ms`, `drift`, metode, dan before/after alignment metrics.
8. Jangan memaksa ECG dan PPG menjadi zero-lag karena terdapat delay fisiologis antara aktivasi listrik jantung dan pulse perifer.
9. Lead-in/lead-out ECG dibuang hanya oleh anchor crop. Pemilihan timeline ECG
   tidak boleh menggunakan F1 atau kanal PPG tertentu.

### 5.3 Cleaning dan Filtering

Filtering dilakukan per channel dan disimpan pada configuration file.

Baseline candidate:

```text
ECG        : bandpass sekitar 0.5–40 Hz
PPG IR/Red : bandpass sekitar 0.5–8 Hz
PPG Green  : bandpass sekitar 0.5–5 Hz
```

Cutoff akhir harus diverifikasi terhadap sampling rate aktual dan inspection hasil filter.

### 5.4 MADC

Gunakan accelerometer bila tersedia untuk mendeteksi motion artifact.

Minimal checks:

- acceleration magnitude/variance;
- abrupt morphology change;
- baseline shift;
- amplitude/spectral outlier;
- peak inconsistency.

Artifact ringan boleh dikoreksi jika tidak mengubah peak timing. Artifact berat harus menyebabkan window ditolak, bukan direkonstruksi secara agresif.

### 5.5 SQA / SQI

SQA dijalankan **sebelum peak matching** dan diperiksa kembali setelah peak detection.

Quality tier:

```text
excellent | good | acceptable | poor | invalid
```

`poor` dan `invalid` tidak masuk training/evaluation normal.

---

## 6. Peak-to-Peak dan ECG–PPG Validation

### ECG

Gunakan NeuroKit2 `ecg_process/ecg_peaks` untuk R-peak detection. Peak flag dari sumber data, bila ada, hanya menjadi secondary reference yang tetap diaudit.

### PPG

Process IR, Red, dan Green secara independen untuk Gold, validasi peak-to-peak, dan evaluasi model. Setiap model hanya menerima satu kanal; eligibility dan promotion gate dinilai terpisah tanpa fusion.

Primary candidate:

- NeuroKit2 PPG processing / Elgendi peak detection.

Fallback/tuning:

- `scipy.signal.find_peaks` menggunakan `distance`, `prominence`, dan `width` yang dihitung dari sampling rate dan heart-rate plausibility.

### Metric validasi per participant × phase × channel

```text
beat precision / recall / F1
matched-beat ratio
IBI MAE / RMSE / median absolute error
IBI correlation
HR MAE
Bland–Altman bias + limits of agreement
```

IBI/RR–PPI primer hanya dihitung dari dua matched beat yang berurutan langsung
pada ECG dan PPG. Interval tidak boleh melintasi gap, unmatched beat, atau dua
blok matching yang terpisah.

PPG dianggap cukup layak untuk downstream model hanya ketika kualitas sinyal dan agreement terhadap ECG memenuhi threshold yang ditentukan dari data serta terdokumentasi. Validasi ini adalah **validasi fisiologis terhadap ECG**, bukan bukti bahwa aplikasi menjadi alat diagnosis medis.

---

## 7. Interpolasi dan Augmentasi

### Interpolasi yang Diizinkan

- Linear/PCHIP untuk waveform gap pendek yang terisolasi.
- Cubic spline/PCHIP untuk IBI tachogram setelah missed/ectopic/artifact beat ditandai.
- Semua hasil interpolasi memiliki mask/provenance.

### Interpolasi yang Dilarang

- Long dropout atau hampir seluruh window.
- Melewati batas Relax ↔ Task.
- Mengubah reused internal test untuk menaikkan metric.
- Warping raw signal untuk memaksa ECG dan PPG cocok.

### Augmentasi yang Diizinkan

Hanya pada **training fold setelah participant split**:

- low-amplitude Gaussian noise;
- small amplitude scaling;
- mild smooth magnitude warping;
- conservative crop/window shift tanpa melewati phase boundary.

Dilarang default:

- time reversal;
- amplitude inversion;
- arbitrary beat permutation;
- aggressive time warp;
- augmentasi validation/reused internal test.

Augmentasi hanya dipertahankan jika participant-level generalization membaik tanpa merusak morphology/peak physiology.

### Kebijakan ECG–PPG primer

Validasi ECG–PPG menggunakan core window 240 detik yang sama untuk ECG dan PPG, dengan timestamp ECG yang telah dikoreksi, tetapi tidak
merekonstruksi waveform ECG. Gap ECG diberi mask dan peak yang berada pada
wilayah terdampak tidak digunakan sebagai reference beat; long dropout menjadi
alasan exclusion. R-peak yang digunakan harus berasal dari sample ECG
asli/clean, sehingga interpolasi waveform ECG tidak dapat menaikkan F1 atau
agreement secara artifisial.

Interpolasi RR/IBI hanya opsional untuk analisis HRV sekunder setelah peak asli
tersedia, default nonaktif, dan wajib dilaporkan terpisah dari interval
`observed`. PPG validation mengonsumsi clean Silver dari canonical
preprocessing tanpa filter/interpolasi kedua. `data_foundation.py` adalah
adapter/reference legacy, bukan jalur preprocessing atau validasi canonical.

---

## 8. Exploratory Data Analysis

EDA dilakukan setelah raw audit dan diulang setelah cleaning/SQA.

Tools:

- `ydata-profiling` untuk automated profiling dan time-series gaps;
- NeuroKit2 report/plots untuk ECG/PPG;
- pandas, NumPy, SciPy, Matplotlib;
- `tsfresh` opsional untuk candidate feature extraction, bukan feature final otomatis.

Minimal output:

- sampling-rate dan duration distribution;
- missing/gap/jitter;
- quality-tier dan motion distribution;
- raw-vs-cleaned examples;
- peak/HR/IBI distribution;
- ECG–PPG agreement;
- Relax/Task balance;
- IR/Red/Green comparison;
- feature correlation/redundancy;
- participant-level outlier report.

Reused internal test tidak digunakan untuk memilih feature, threshold, atau preprocessing.

---

## 9. Training dan Evaluation

### Windowing dan segmentation

Akuisisi tetap 300 detik dan seluruh kanal memakai core `[30,270)` sepanjang 240 detik. Baseline adalah window 180 detik/stride 10 detik. Ablation development per kanal membandingkan `60/30`, `120/30`, `180/10`, `180/30`, dan `240/240`. Baseline hanya boleh diganti bila session-level balanced accuracy naik minimal `0,02`, seluruh gate lulus, coverage tidak berkurang, dan variasi antar-partisipan tidak memburuk. Overlapping window tidak dihitung sebagai observasi independen; setiap participant-phase mendapat total weight yang sama dan primary metric dihitung setelah median session aggregation.

### Split dan reused internal test

```text
Development               : ID-01 .. ID-28
Reused internal test      : ID-29 .. ID-32
Clinical validation status: external_validation_required
```

Development menggunakan tujuh outer fold yang masing-masing berisi empat participant dan inner `GroupKFold(5)`. ID-29..ID-32 tidak boleh memilih window, quality cohort, fitur, model, hyperparameter, calibration, atau threshold. Random split per row/window dilarang.

### Feature policy

Predictor default adalah `median_hr_bpm`, `ppi_median_ms`, `ppi_iqr_ms`, `ppi_cv`, `rmssd_ms`, `sdnn_ms`, `pnn50`, `pulse_amplitude_cv`, `normalized_pulse_amplitude_median`, `pulse_width_half_height_median_ms`, `rise_time_median_ms`, `decay_time_median_ms`, `normalized_pulse_area_median`, `normalized_rise_slope_median`, dan `normalized_decay_slope_median`.

`SD1/SD2`, sample entropy, waveform skewness/kurtosis, log LF/HF power, dan LF/HF bersifat experimental-only. Promosi harus melalui ablation development dan stabil terhadap durasi window. LF/HF tidak boleh disebut pengukuran langsung sympathovagal balance.

Peak count, physiological/valid IBI ratio, SQI, missing/interpolation/artifact/clipping/flatline/motion/jitter, quality tier, dan quality flags adalah quality gate, bukan predictor. ID/session/phase/label name/path/checksum/timestamp/window/rate/channel/ECG/PAT dilarang menjadi predictor. `ibi_std` duplikat `sdnn`; inter-peak `pulse_width` lama, raw amplitude summary/area/slope, hardcoded quality value, percentile pseudo-clipping, dan non-finite-to-zero dilarang.

### Classical ML

Fixed baseline per kanal:

- Logistic Regression;
- RBF SVM;
- Extra Trees;
- LightGBM;
- XGBoost.

Dua fixed baseline terbaik dituning dengan Optuna TPE, 50 trial, seed 42, dan persistent SQLite. Objective adalah session-level balanced accuracy. Sigmoid calibration dan threshold memakai inner OOF development prediction saja. Scaler, imputer, correlation pruning absolut `>0,95`, feature selection, calibration, dan HPO selalu fit di training fold.

### Deep Learning

Jika classical model belum memenuhi gate secara stabil:

1. 1D-CNN;
2. TCN;
3. CNN-BiLSTM atau lightweight GRU;
4. compact waveform model lain hanya bila kontrak input dan participant-safe ablation disetujui.

Classical ML berjalan lokal untuk Green, IR, dan Red secara independen tanpa fusion. Bila salah satu dari BA, F1, atau ROC-AUC development berada di bawah `0,70`, hanya kanal tersebut dibuatkan handoff DL ke Google Colab T4. Jika Red gagal DL, tidak ada release Red dan Green/IR tidak ikut terblokir. Notebook hanya menerima Gold kanal terkait, manifest lineage, dan handoff ML; raw ECG/PPG tidak diunggah. Checkpoint Drive harus atomik, dapat di-resume, dan diverifikasi terhadap checksum Gold/config/fold contract.

### Release Gate

Model hanya dapat menjadi kandidat ONNX bila setidaknya memenuhi seluruh minimum research gate. Tier preferred tetap menjadi target yang lebih kuat:

```text
research_usable : balanced_accuracy >= 0.70, F1 >= 0.70, ROC-AUC >= 0.70
preferred       : balanced_accuracy >= 0.80, F1 >= 0.78, ROC-AUC >= 0.83
```

`research_usable` memiliki `deployment_scope=research` dan wajib menampilkan disclaimer "Model penelitian — validasi internal terbatas. Bukan diagnosis klinis." Tier `preferred` memiliki `deployment_scope=standard`. Model di bawah minimum memiliki `deployment_scope=none` dan wajib dieskalasi ke DL. Semua tier tetap `external_validation_required`.

Primary metric dilaporkan pada session level: balanced accuracy, F1, ROC-AUC, stress sensitivity, relax specificity, precision, Brier score, participant-bootstrap 95% CI, per-participant metrics, serta mean ± std antar-fold. Window-level metric hanya sekunder. Quality cohort `excellent+good` dibandingkan dengan expanded `excellent+good+acceptable`; bila selisih BA `<=0,01`, strict dipilih.

Status per kanal: `not_evaluated`, `development_failed`, `development_passed`, `internal_test_failed`, `model_candidate`, `onnx_accepted`, dan `android_ready`. Setiap report juga wajib menyimpan `performance_tier=below_minimum|research_usable|preferred`, `deployment_scope=none|research|standard`, dan fingerprint policy yang terpisah dari fingerprint model. Perubahan policy tidak boleh membuka ulang reused internal test. Hasil tetap berstatus `external_validation_required` secara klinis dan dilaporkan mengikuti TRIPOD+AI.

Laporan holdout historis tanpa `model_contract_fingerprint` atau `holdout_contract_fingerprint` eksplisit disimpan sebagai `historical_unverifiable`: tidak boleh dipakai ulang sebagai evidence dan tidak dianggap sebagai akses canonical. Satu evaluasi canonical pertama yang modelnya sudah dibekukan harus dicatat atomik dengan fingerprint kontrak; setelah itu evaluasi ulang hanya boleh memakai ulang evidence dengan fingerprint identik.

### Current verification boundary

Snapshot `20260808T120000Z` is not a release: `training_status=ready`, but
`ppg_gold_status=blocked` at 191/192 groups, `ecg_validation_status=blocked`,
and `model_release_status=blocked` pending Android evidence. The canonical
validator source now emits `ibi_rmse_ms`. Versioned validation evidence was
regenerated at `20260809T120000Z` from the prior canonical Bronze/Sync/Silver
boundary: 192/192 combinations, 117 valid, and 75 explicitly excluded.
Physical Samsung phone–Watch, microphone, and resource acceptance remain
separate gates.

### Jika Gate Gagal

Jangan menurunkan minimum research gate di bawah 0,70 secara ad hoc. Lakukan remediation loop:

```text
audit label/anchor
→ ECG timestamp correction
→ cleaning/MADC/SQA
→ peak detector + ECG agreement
→ allowed interpolation
→ normalization/features
→ train-fold feature selection
→ allowed augmentation
→ hyperparameter tuning
→ compact DL escalation
→ grouped validation ulang
```

Reused internal test hanya digunakan setelah keputusan pipeline/model dibekukan.

---

## 10. ONNX dan Android Stress Detection

### ONNX

- Export hanya model yang lolos gate.
- PyTorch model menggunakan `torch.onnx.export(..., dynamo=True)`/exporter current yang didukung.
- Jalankan ONNX checker dan ONNX Runtime lokal.
- Native vs ONNX FP32 parity: `max_abs_diff < 1e-5`.
- Quantization hanya diterima bila performa tetap memenuhi seluruh gate.
- Simpan `model_manifest.json` berisi preprocessing version, feature version, input shape, threshold, metric, dan checksum.
- `android-prep` boleh menghasilkan 100 vector nyata berstatus pending setelah
  gate ML dan ONNX lokal lulus; artefak pending tidak boleh mempromosikan index.
- `model_index.json` hanya memuat model kanal yang accepted; kanal gagal tidak memblokir sibling channel. Setiap entry accepted wajib memiliki manifest
  `android_ready=true`, positive source/model rate, reused internal test cohort ID-29..ID-32,
  checksum artifact, ONNX checker/parity, dan 100 deterministic vectors yang
  lolos validator lokal. Vector artifact yang direferensikan juga wajib
  menyatakan `python_vector_status=passed`, `python_vectors_passed=100`,
  `android_instrumentation_status=passed`, dan `android_ready=true`. Index yang
  gagal validator harus membuat packaging fail-closed dan release evidence
  berstatus blocked. Urutan runtime adalah Green → IR → Red tanpa averaging/fusion. Red dapat berstatus `model_candidate`/`onnx_accepted` tetapi `android_ready=false` sampai akuisisi Red nyata dan parity Kotlin tersedia.

### Android Runtime

Stack:

- Kotlin;
- Jetpack Compose;
- ViewModel + UDF;
- coroutines/Flow;
- ONNX Runtime Android/Mobile;
- NNAPI hanya setelah physical-device benchmark membuktikan manfaat.

Runtime stress pipeline:

```text
PPG
→ sanity/SQA
→ preprocessing identik dengan Python
→ selesai sesi 300s → crop core 30–270s → maksimal 7 window 180s / stride 10s
→ ONNX inference
→ valid-window probability
→ agregasi 5 menit
→ Relax / Stress / `INSUFFICIENT_SIGNAL`
```

Inference tidak boleh berjalan di main UI thread.

#### Wear transport dan visualisasi Home

Home memakai tracker Samsung individual `PPG_GREEN`, `PPG_IR`, dan `PPG_RED`,
mengikuti pola akuisisi PDSatu yang sudah terbukti pada perangkat target.
Green adalah kanal utama inference, IR fallback independen, dan Red hanya
diagnostik sampai model serta parity Android Red diterima. Watch mengirim batch
langsung melalui `MessageClient`; `SESSION_READY`, durable outbox, batch ACK,
dan terminal receipt bukan syarat jalur aktif. `sessionId` dan nonce tetap
dipakai untuk membuang batch sesi lama.

Preview grafik memakai sampel aktual terurut berdasarkan elapsed session time.
Preview memakai tail 10 detik seperti PDSatu, tanpa envelope agresif yang
mengubah bentuk sinyal. Long gap tetap diberi jeda visual. Setelah sesi 300
detik selesai, grafik beralih ke PPG clean core 30--270 detik yang sama dengan
pipeline inferensi. Tidak ada padding, interpolasi long gap, atau filter kedua
khusus grafik. Batch yang hilang dilaporkan sebagai `transport_gap`; data
sintetis dilarang.

Status koneksi Wear wajib dibedakan menjadi `NODE_DISCOVERED`,
`APP_REACHABLE`, `STREAMING`, `DRAINING`, `OFFLINE`, dan
`PROTOCOL_ERROR`. Discovery bukan bukti aplikasi reachable atau streaming.
Ping bersifat best-effort pada level transport, tetapi Home stress detection
wajib menunggu fresh correlated pong dan hanya boleh mengirim `SESSION_START`
ketika fase `APP_REACHABLE`; node yang hanya ditemukan tidak boleh memulai
sesi. Timestamp command, batch, dan ping terakhir menjadi evidence setelah
sesi aktif. Disconnect
Samsung yang disengaja selama stop/drain tidak boleh menimpa hasil sesi menjadi
error; stop listener, flush, dan disconnect wajib idempotent.

Setiap laporan Home menyimpan ringkasan BPM observed-only secara independen
untuk Green dan IR: mean, median, minimum, maksimum, jumlah observasi, dan
jumlah peak valid. Ringkasan memakai PPG clean core `[30s,270s)` yang sama
dengan inference. Nilai tidak tersedia harus tetap `null` dengan alasan, bukan
diisi default atau BPM sintetis.

Probability sesi adalah keluaran ONNX terkalibrasi dan berbeda dari performance
gate model. Gate 70% menilai BA/F1/ROC-AUC model pada cohort evaluasi; ia tidak
mewajibkan setiap probability sesi mencapai 70%. Probability dan confidence
harus ditampilkan apa adanya, bersama signal quality dan performance tier.

---

## 11. Android Clinical Companion

Aplikasi sederhana, single experience, **tanpa authentication dan tanpa RBAC**.

Navigasi minimum:

```text
Home / Stress
History
Verbatim Analysis (AI+)
Live Monitor
AI Counselor
Settings
```

Tidak boleh ada dummy screen atau fake integration.

### 11.1 Verbatim Analysis / AI+

```text
Consent
→ Record/Import audio
→ VAD
→ Speech-to-Text
→ Speaker Diarization
→ Timestamp Alignment
→ Non-destructive transcript cleaning
→ Manual annotation
→ Psychological/counseling coding suggestion
→ RAG-based AI analysis
→ Export/Delete
```

Requirement:

- record, pause, resume, stop, playback, dan save bekerja nyata;
- playback menampilkan posisi/durasi audio dan transcript maupun marker memakai
  satu operasi atomik `seekAndPlay` ke offset elapsed yang sama;
- Bahasa Indonesia diprioritaskan pada STT;
- raw audio dan raw transcript tidak ditimpa;
- diarization MVP minimal dua speaker dan label dapat diedit;
- low-confidence segment ditandai;
- hasil AI adalah suggestion, bukan diagnosis;
- sherpa-onnx dapat dibenchmark sebagai kandidat STT/VAD/diarization Android.

### 11.2 Live Monitor + Verbatim

Stress event dapat ditautkan ke timestamp percakapan:

```text
session_timestamp
stress_probability
stress_class
signal_quality
speaker_segment_id
transcript_segment_id
annotation
```

Fitur tidak boleh mengeluarkan klaim `lie detected`, `deception detected`, atau `truth score`.

PPG/stress signal hanya digunakan sebagai **physiological arousal/stress-response marker** yang membantu psikolog menentukan bagian percakapan yang perlu ditinjau. Respons fisiologis tidak memiliki hubungan satu-ke-satu dengan kebohongan.

### 11.2.1 Acoustic evidence contract

Android menggunakan PCM nyata dari `AudioRecord` untuk menghitung RMS dBFS, zero-crossing rate, clipping,
finite-sample ratio, kualitas audio, dan deviasi terhadap baseline akustik sesi. Live waveform hanya membawa
envelope maksimal 64 titik per frame; raw audio tetap disimpan sebagai artifact sesi dan tidak direkonstruksi.

Marker canonical:

```text
STRESS_ELEVATED
ACOUSTIC_DIVERGENCE
AUXILIARY_EVENT
INCONSISTENCY_CUE
LOW_AUDIO_QUALITY
MANUAL
```

Marker fisiologis, model, akustik, dan kualitas audio harus dibedakan. Low audio
quality bukan marker stress/arousal. Baseline akustik memerlukan warm-up;
divergence memerlukan persistence, hysteresis, cooldown, dan deduplikasi per
episode agar satu frame buruk tidak menghasilkan marker baru. BPM rise/recovery
hanya boleh ditandai bila baseline, kenaikan, durasi, dan pemulihan memenuhi
konfigurasi.

Finalisasi Verbatim+ wajib berupa state machine idempotent:
`STOP_REQUESTED → AUDIO_DRAINING → PPG_DRAINING → SNAPSHOT_CAPTURED →
PERSISTING → COMPLETED|FAILED_WITH_EVIDENCE`. Kegagalan Watch, audio, atau Room
diisolasi; kegagalan salah satu subsistem tidak boleh membuang evidence lain atau
menyebabkan process crash sebelum state gagal dapat dirender.

`ACOUSTIC_DIVERGENCE` adalah deviasi dari baseline akustik, bukan bukti deception. `INCONSISTENCY_CUE`
adalah sinyal untuk review psikolog, bukan kesimpulan faktual. Nilai confidence acoustic adalah quality-derived
evidence confidence, bukan probabilitas klinis. Marker transition dipersistenkan per perubahan status untuk
menjaga bounded storage.

Source-isolated compile dan Android unit suite sudah menjadi gate lokal; microphone/device acceptance,
resource benchmark, dan validasi Samsung phone–Watch masih merupakan gate terpisah.

### 11.3 AI Counselor + RAG

LLM utama: **Gemma 4 E2B**, karena dokumentasi Google saat ini mencantumkannya sebagai model Gemma 4 yang ditujukan untuk perangkat mobile.

RAG lokal:

```text
Document import
→ text extraction
→ chunking
→ metadata source/page
→ local embedding
→ local retrieval
→ relevance threshold
→ grounded prompt
→ Gemma 4 E2B
→ answer + citation
```

Knowledge base hanya menggunakan dokumen yang legal/diberikan untuk aplikasi, misalnya PNPPK, PPDGJ, referensi ICD yang diizinkan, materi konseling/terapi, dan SOP psikolog.

Aturan:

- tidak membuat citation palsu;
- jika retrieval tidak cukup, tampilkan fallback yang jelas;
- tidak memberi diagnosis;
- tidak menggantikan judgement psikolog.

Kontrak canonical AI+ disimpan di `configs/rag.yaml` dan bersifat fail-closed:

- format yang didukung adalah PDF, DOCX, TXT UTF-8, dan Markdown; binary `.doc` ditolak eksplisit;
- PDF diterima hingga tepat 100 MiB (`104_857_600` byte), tanpa batas halaman statis. PDF diproses per halaman dengan audit progres, batas kumulatif teks/chunk, dan OCR lokal hanya untuk halaman tanpa text layer; gangguan pemrosesan dipulihkan lewat reindex yang aman terhadap sumber/checksum. Checkpoint/resume per halaman yang durable tidak boleh diklaim tersedia sebelum staging chunk transactional telah diimplementasikan; DOCX mempertahankan heading, list, paragraf, dan tabel serta tidak mengindeks hidden/deleted text atau field instruction;
- dokumen `PARTIAL` tidak ikut retrieval sebelum pengguna menyetujui warning halaman yang gagal;
- chunk dihitung dengan tokenizer embedding yang sama: target 224 token, hard maximum 256 token, overlap 32 token, dan tidak melintasi halaman PDF atau top-level section DOCX;
- embedding canonical adalah `intfloat/multilingual-e5-small` dengan prefix `query:`/`passage:`, attention-mask mean pooling, L2 normalization, maksimum 512 token, dan dimensi 384;
- retrieval mengambil 30 kandidat semantic dan 30 lexical, memakai RRF `k=60`, adjacent-overlap deduplication, MMR `lambda=0.75`, lalu mengekspos maksimal empat evidence chunk;
- threshold no-answer harus berasal dari benchmark lokal terversi; profil canonical 60-query diperbolehkan sebagai evidence lebih ketat, dan konfigurasi tanpa threshold terkalibrasi wajib abstain;
- lifecycle dokumen harus eksplisit: `PENDING`, `INDEXING`, `PARTIAL`, `READY_LEXICAL`, `READY_SEMANTIC`, `REINDEX_REQUIRED`, atau `FAILED`. Waktu stale dihitung dari `processingUpdatedAtMillis`, bukan waktu unggah;
- setiap reindex diberi identitas `documentId + contentChecksum + revision`; OCR lokal hanya dipanggil pada halaman tanpa text layer. Batas kumulatif menghasilkan `PARTIAL_LIMIT_REACHED`; error seperti `PDF_ENCRYPTED`, `OCR_FAILED`, `TOKENIZER_UNAVAILABLE`, dan `MALFORMED_DOCUMENT` harus dapat dirender tanpa menyamarkan penyebabnya. Penerbitan chunk tetap atomik sehingga dokumen parsial tidak pernah menjadi sumber retrieval;
- generation pengganti tanpa dokumen atau chunk tidak boleh menggantikan generation `READY` yang masih valid. `READY_LEXICAL` diizinkan bila E5 embedding belum tersedia, tetapi retrieval tetap fail-closed sampai policy benchmark tervalidasi diinjeksikan;
- benchmark retrieval memerlukan set kalibrasi dan fixed test set terpisah, sekurang-kurangnya 30 query. Kontrak 60 query yang ada adalah profil lebih ketat; minimum evidence yang harus tercatat adalah Recall@5 `>=0,85`, citation hit@3 `>=0,80`, dan irrelevant-query abstention `>=0,90`;
- setiap respons AI+ menyimpan `CounselorContextReceipt` berisi versi persona/policy, citation dokumen dan halaman, source history stress valid, alasan history yang dikeluarkan, serta versi runtime LLM;
- setiap citation `[S#]` harus terikat tepat ke chunk yang benar-benar masuk prompt; citation hilang, tidak cocok, atau fabricated menghasilkan abstain;
- dokumen diperlakukan sebagai untrusted evidence, bukan instruksi; output ditahan sampai crisis, safety, diagnosis, groundedness, dan citation guard selesai;
- persona berbentuk profil terstruktur dan hanya mengubah gaya; persona tidak dapat mengubah safety/evidence dan `focusNote` dibatasi maksimal 1.500 kata Unicode setelah normalisasi whitespace;
- inference dokumen tetap lokal. Web Search hanya boleh berjalan setelah persetujuan eksplisit dan citation web dipisahkan dari citation dokumen lokal;
- source document wajib AES-GCM dengan key material Android Keystore dan knowledge database wajib SQLCipher dengan passphrase yang dibungkus Keystore sebelum release.

Release RAG memerlukan bukti FTS5/BM25, benchmark Recall@5/MRR@5/no-answer, signed E5 tokenizer/ONNX parity, encrypted storage, serta test production end-to-end pada phone. Status implementation gate dicatat secara eksplisit di `configs/rag.yaml`; keberadaan parser atau unit test saja bukan bukti fitur siap rilis.

---

## 12. Privacy dan Local Storage

Walaupun tanpa login/RBAC:

- Data sesi disimpan secara terstruktur dengan penamaan folder: [Nama Sesi Kustom]_[Tanggal: DD-MM-YYYY]_[Waktu: HH:MM:SS]
- Isi Folder Sesi wajib memuat 4 file:
  1. File Rekaman Audio (.wav atau .m4a)
  2. File Live Transcript (memuat diarization dan timestamp)
  3. File Divergency Events (log data PPG, label stres, dan timestamp)
  4. File Ringkasan AI (hasil output Gemma 4 E2B)
- gunakan Android Keystore-backed key untuk data sensitif;
- jangan log raw content ke release Logcat;
- recording dan live monitoring memerlukan consent per session;
- sediakan export/delete per session dan delete-all;
- core workflow tidak bergantung pada cloud.

---

## 13. Engineering Quality

Implementasi mengikuti:

- Clean Code;
- SOLID;
- DRY;
- anti-spaghetti code;
- interface-driven boundaries;
- BMAD + Spec-Driven Development;
- local lint/test sebelum perubahan diterima.

Minimum local checks:

```bash
ruff check .
ruff format --check .
pytest -q

./gradlew ktlintCheck
./gradlew detekt
./gradlew lint
./gradlew test
```

---

## 14. Acceptance Criteria Utama

Produk dianggap memenuhi PRD hanya bila:

1. Hanya `ID-01..ID-32` ditemukan di seluruh pipeline.
2. Raw data tetap utuh dan traceable.
3. ECG timestamp correction terdokumentasi per participant/session.
4. MADC dan SQA mencegah poor/invalid signal masuk training normal.
5. Peak-to-peak/IBI validation ECG–PPG menghasilkan laporan per channel dan participant.
6. Validasi ECG–PPG menggunakan core 240 detik yang sama untuk ECG dan PPG serta selalu menghasilkan 192 baris kombinasi atau exclusion reason.
7. Tidak ada participant/window leakage.
8. Augmentasi hanya pada training fold dan interpolasi mengikuti short-gap policy.
9. Reused internal test memenuhi seluruh performance gate, dengan status klinis tetap `external_validation_required`.
10. ONNX parity lulus dan model load/inference berhasil lokal.
11. Kotlin preprocessing menghasilkan parity terhadap Python reference.
12. Android menghasilkan keputusan 5 menit hanya dari window berkualitas memadai.
13. Semua fitur Android bersifat nyata, bukan dummy.
14. Verbatim dapat record → STT → diarization → save → review/seek transcript-marker → export/delete.
15. RAG menampilkan sumber/fallback yang benar.
16. Live Monitor tidak mengklaim kebohongan; hanya menandai arousal/stress divergence untuk review psikolog.
17. Seluruh jalur utama dijalankan lokal, kecuali deep learning conditional yang dijalankan secara manual di Google Colab T4 dengan checkpoint Google Drive; tidak ada automation build/release remote atau platform deployment/tracking eksternal.

---

## 15. Referensi Implementasi

- NeuroKit2 ECG: https://neuropsychology.github.io/NeuroKit/functions/ecg.html
- NeuroKit2 PPG: https://neuropsychology.github.io/NeuroKit/functions/ppg.html
- SciPy `find_peaks`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
- SciPy `CubicSpline`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
- YData Profiling Time-Series: https://docs.profiling.ydata.ai/latest/features/time_series_datasets/
- scikit-learn GroupKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
- scikit-learn cross-validation / StratifiedGroupKFold: https://scikit-learn.org/stable/modules/cross_validation.html
- scikit-learn nested cross-validation: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
- Optuna persistent RDB storage: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.RDBStorage.html
- ESC/NASPE HRV standard: https://www.escardio.org/static-file/Escardio/Guidelines/Scientific-Statements/guidelines-Heart-Rate-Variability-FT-1996.pdf
- Wearable PRV review: https://pubmed.ncbi.nlm.nih.gov/29668452/
- LF/HF limitation: https://pubmed.ncbi.nlm.nih.gov/23431279/
- TRIPOD+AI reporting: https://www.tripod-statement.org/
- PyTorch ONNX Export: https://docs.pytorch.org/docs/stable/onnx.html
- ONNX Runtime Mobile: https://onnxruntime.ai/docs/tutorials/mobile/
- ONNX Runtime NNAPI: https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html
- Android Architecture Recommendations: https://developer.android.com/topic/architecture/recommendations
- Jetpack Compose Architecture/UDF: https://developer.android.com/develop/ui/compose/architecture
- Google Gemma models: https://ai.google.dev/gemma/docs/get_started
- sherpa-onnx Android: https://k2-fsa.github.io/sherpa/onnx/android/build-sherpa-onnx.html
- sherpa-onnx Speaker Diarization: https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/
- BMAD Method: https://github.com/bmad-code-org/BMAD-METHOD
- National Academies — The Polygraph and Lie Detection: https://www.nationalacademies.org/read/10420/chapter/2

---

## 16. Source of Truth

`PRD.md` mendefinisikan **apa yang harus dibangun dan acceptance criteria**.  
`AGENTS.md` mendefinisikan **bagaimana agent/developer mengimplementasikan dan memverifikasinya**.

Jika ada konflik dengan PRD lama, ketentuan di dokumen ini yang berlaku, khususnya:

```text
32 participant ONLY
local execution except conditional DL in Google Colab T4
no new participant acquisition
ECG = physiological validation reference
PPG = production stress-model input
minimum research gate: balanced_accuracy/F1/ROC-AUC >= 70%; preferred: 80%/78%/83%
real application, no dummy integration
no authentication/RBAC
```

### Kontrak Wear, BPM, Marker, dan History

- Status koneksi Wear dipisahkan menjadi `NODE_DISCOVERED`, `APP_REACHABLE`, `STREAMING`, `DRAINING`, `OFFLINE`, dan `PROTOCOL_ERROR`. Node yang ditemukan belum membuktikan aplikasi reachable atau aliran data; `SESSION_START` hanya sah setelah fresh correlated pong pada fase `APP_REACHABLE`.
- Green, IR, dan Red diperoleh melalui individual tracker. Green dan IR dievaluasi independen untuk stress; Red hanya diagnostik sampai memiliki model dan parity Android yang disetujui.
- Laporan sesi wajib memuat mean/median BPM Green dan IR dari core `[30,270)` dan peak PPG aktual. Nilai default, interpolasi BPM, serta penggabungan kanal dilarang.
- Marker Verbatim+ dibedakan antara `PHYSIOLOGICAL_BPM_TRAJECTORY`, `TEMPORAL_BASELINE_AROUSAL`, `CHANNEL_AGREEMENT`, `LOW_AUDIO_QUALITY`, `ACOUSTIC_ENERGY_CHANGE`, dan `ACOUSTIC_DIVERGENCE`. Marker episode memerlukan quality, warm-up, persistence, hysteresis, cooldown, dan deduplication; satu perubahan RMS bukan bukti stress.
- History wajib membedakan Stress (merah), Normal (hijau), dan Invalid (abu-abu), serta menampilkan cleaned-core chart aktual per kanal dengan gap dan mask interpolasi. Bila chart artifact tidak tersedia, statusnya harus eksplisit.
- Finalisasi Wear, audio, live monitor, dan Room harus idempotent; kegagalan ditulis sebagai evidence yang dapat dirender dan tidak boleh menyebabkan force-close.
- Lifecycle LiteRT harus terserialisasi: pembatalan generasi meminta cancel kooperatif sekali, tidak menutup engine/conversation saat native stream masih aktif, dan runtime wedged di-reset terkontrol setelah stream berakhir.
- `transportIncomplete` adalah warning sekunder, bukan pengganti label model. Sesi hanya `Invalid` bila core, SQA, model, atau jumlah window valid tidak memenuhi kontrak; threshold probability harus selalu berasal dari manifest kanal yang dipakai.
- Home dan Verbatim+ menggunakan lease PPG eksklusif. Satu Watch hanya dapat memiliki satu sesi aktif; permintaan kedua gagal eksplisit `BUSY_BY_HOME`/`BUSY_BY_VERBATIM` tanpa menghentikan sesi pemilik.
- Range hari memakai `LocalDate.atStartOfDay(zone)` sampai awal hari berikutnya dengan rollover ticker. Session timestamp tetap epoch millis dan tidak boleh dihitung melalui penambahan 24 jam tetap.
