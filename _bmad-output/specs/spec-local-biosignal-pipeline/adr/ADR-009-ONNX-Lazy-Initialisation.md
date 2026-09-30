# ADR-009: ONNX Runtime & LiteRT Lazy Initialisation Strategy

## Context
The Android Clinical Companion app uses Heavy ML dependencies:
- LiteRT-LM (Gemma 4 E2B) for generative text processing.
- ONNX Runtime for local text embeddings (RAG capabilities).

Loading native libraries (like `onnxruntime` or LiteRT `.so` files) during class instantiation or dependency injection (e.g., at app startup via Hilt) is highly dangerous. If the native binaries crash during load (due to architecture mismatch, memory constraints, or unexpected OS conditions like 16KB page sizes on Android 15), it will result in a fatal `UnsatisfiedLinkError` or `SIGSEGV` that crashes the entire application at boot, offering zero recourse to the user.

## Decision
All heavy ML dependencies and native AI models **must use lazy initialization**. 
- Never eagerly load native AI libraries in a `ViewModel` constructor or global DI module provider without lazy wrappers.
- The `ensureInitialized()` pattern must be implemented to encapsulate the native loading wrapped in a `try-catch` block.
- Fallback mechanics must gracefully handle initialization failures and notify the UI instead of crashing the app.

## Consequences
- Better startup performance.
- Resiliency against native binary load errors, specifically `UnsatisfiedLinkError`.
- Improved maintainability and deterministic fallback capabilities.
- Added complexity: Callers must invoke and await initialization methods before executing inference methods.
