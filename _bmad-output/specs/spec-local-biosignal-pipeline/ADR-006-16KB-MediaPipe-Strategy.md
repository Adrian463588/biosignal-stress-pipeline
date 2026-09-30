# ADR-006: Android 16 KB Page Size Compatibility - MediaPipe Strategy

## Status
Accepted (28 July 2026)

## Context
Starting November 1st, 2025, Google Play requires all new apps and updates targeting Android 15+ to support 16 KB page sizes.
The application currently uses `com.google.mediapipe:tasks-genai:0.10.14` for on-device LLM (Gemma 4B) inference.
During conformance auditing, `libllm_inference_engine_jni.so` (bundled within MediaPipe) was found to be misaligned (compiled with legacy 4 KB ELF LOAD segment alignment).
Because this library is pre-compiled by Google and distributed via Maven, we cannot rebuild it locally with the required `-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON` linker flags.

## Decision
We will retain `com.google.mediapipe:tasks-genai:0.10.14` for now and ensure that `packagingOptions { jniLibs { useLegacyPackaging = false } }` remains enabled in `build.gradle` to prevent re-compression of native libraries, which can sometimes exacerbate alignment issues.
To fully resolve the alignment issue for MediaPipe, we will wait for Google to release an updated version of `tasks-genai` that is compiled with 16 KB page size support, and upgrade to that version prior to the November 2025 mandate.

## Consequences
- **Positive**: Prevents complex and brittle local patching/repackaging of third-party libraries.
- **Negative**: The application may still fail rigorous 16 KB page size conformance checks (e.g., `zipalign -c -P 16 -v 4`) specifically due to `libllm_inference_engine_jni.so` until an upstream fix is available.
