# Quantumn AI Workflow

This document defines the "Agent-First" development strategy for Quantumn, balancing maximum AI acceleration with absolute technical safety.

## 🤖 Agent-First Philosophy
We treat AI agents as the primary "implementers" and humans as the "Architects/Verifiers". The goal is to offload boilerplate and complex SIMD arithmetic to the AI while the human ensures the memory model and alignment remain intact.

## 🛠️ The Development Loop

### 1. Architectural Blueprinting (Human $\to$ Agent)
- The human defines the goal and constraints in `CLAUDE.md` or a specific task.
- The AI proposes a detailed implementation plan (Step-by-Step).
- **Gate**: The human must approve the plan's memory and alignment strategy.

### 2. Safe Implementation (Agent $\to$ Code)
- AI implements a small, atomic unit (e.g., a single kernel or a tensor method).
- AI must use **Explicit Memory Layouts** and **No Hidden Allocations**.
- AI must include unit tests for correctness and a basic benchmark for performance.

### 3. Rigorous Verification (Agent $\to$ Human)
- AI runs the correctness tests and benchmarks.
- AI provides:
    - `Correctness: PASS` (bit-perfect match)
    - `Performance: [X] tokens/sec`
    - `Alignment: Verified 64-byte`
- **Gate**: The human reviews the `perf` output and the `IRON_LAWS.md` compliance before merging.

## 🛡️ Safety Protocols

### The "Pressure Check"
If an agent suggests a change that violates an `IRON_LAW`, it must:
1. Explicitly call out the violation.
2. Provide a detailed technical justification.
3. Propose the entry for the **Pressure Log** in `docs/IRON_LAWS.md`.

### The Benchmark Mandate
No performance-critical code is merged without:
- A before/after comparison.
- L1/L2 cache miss analysis.
- Validation on the target hardware (Kaby Lake class).

## 🚀 AI Tooling Strategy
- Use specialized skills (like `open-rx:database-ops` or `arxitect`) only when they align with the "CPU-First" philosophy.
- Use the `Plan` agent for high-level orchestrations.
- Use `code-reviewer` agents to hunt for "hidden allocations" or "AVX-512 leakage".
