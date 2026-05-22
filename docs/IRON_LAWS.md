# Quantumn Iron Laws

These laws are the non-negotiable constraints of the Quantumn engine. They are designed to prevent architectural drift and ensure the engine remains a "CPU-first" masterpiece.

## 🚫 The Unbreakables

### 1. Memory Discipline
- **NO** runtime `malloc` or `Box` during the inference loop. Everything must come from the Unified Arena.
- **NO** expanding quantized tensors to FP32 globally. Dequantization happens ONLY in registers.
- **NO** hidden allocations in tensor reshaping or slicing.

### 2. Hardware Constraints
- **NO** AVX-512 intrinsics. Target is strictly AVX2/FMA3 for maximum compatibility.
- **NO** GPU dependencies. No CUDA, No ROCm, No Metal.
- **NO** oversubscription. Exactly 2 physical worker threads, pinned to core 0 and core 1.

### 3. Performance Mandates
- **ALL** major buffers must be 64-byte aligned.
- **EVERY** kernel change must be accompanied by a correctness test and a benchmark.
- **NO** tensor copies unless absolutely required by the hardware layout (e.g., initial transpose).

---

## ⚡ The Pressure Log

In extreme cases, a law may need to be bent for a specific, documented reason (e.g., a critical bug in a specific CPU micro-architecture). 

**Any violation of an Iron Law must be recorded here BEFORE the code is merged.**

| Date | Law # | Violation | Circumstances | Justification | Approval |
| :--- | :--- | :--- | :--- | :--- | :--- |
| | | | | | |

**Pressure Log Entry Template:**
`[YYYY-MM-DD] | [Law #] | [Description of deviation] | [Technical context/constraint] | [Why this is the only way] | [Sign-off]`
