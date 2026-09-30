# ADR-006: Engine Selection for Verbatim Analysis

## Status
Accepted

## Context
The Verbatim Analysis feature requires offline capabilities for Voice Activity Detection (VAD), Speech-to-Text (STT), Speaker Diarization, and Large Language Model (LLM) inference. The application must perform these tasks locally on the Dimensity 8400 Ultra to ensure data privacy (no cloud dependency) and meet clinical constraints.

## Decision

We have evaluated several options for each subsystem and decided on the following stack:

1. **Speech-to-Text & Speaker Diarization:** `sherpa-onnx`
   - **Reasoning:** `sherpa-onnx` offers a unified ONNX-based pipeline that can handle both STT and two-speaker diarization within the ~200MB model budget limit. It offers robust Android deployment and supports Indonesian + English.
   - **Alternative Considered:** `Whisper.cpp` (did not natively include robust unified diarization at similar performance).
2. **Voice Activity Detection (VAD):** `Silero VAD` (via `android-vad` library)
   - **Reasoning:** DNN-based approach offering higher precision than WebRTC, while remaining lightweight enough to prevent cutting off clinical speech prematurely.
3. **On-Device LLM:** `Gemma 4 E2B` via `ML Kit GenAI Prompt API`
   - **Reasoning:** Simplified integration through ML Kit ensures we can utilize the Dimensity 8400 Ultra GPU/NPU efficiently without building custom LiteRT JNI bindings. 
   - **Usage:** Will be used for filler removal (cleaning), psychological coding via RAG, and analysis generation.
4. **Text-to-Speech (TTS):** `sherpa-onnx`
   - **Reasoning:** Stack consistency with the STT component.

## Consequences
- **Positive:** No cloud dependencies; complete privacy; deterministic behavior.
- **Negative:** Increased APK size (~150-200MB additional assets); higher battery utilization during analysis phases. 
- **Mitigation:** We will execute transcription in batch offline rather than real-time streaming to optimize thermal overhead.
