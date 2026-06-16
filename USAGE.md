# Usage and Operational Guide

Quantumn is a CPU-first LLM inference engine.

## Initialization
Build for maximum performance:
```bash
cargo build --release
```

## Running Inference
Run with a GGUF model:
```bash
cargo run -p aether-cli -- /path/to/model.gguf
```

## Features for Reliability
- **Thematic RAG Memory**: When the KV cache fills, the engine automatically archives context into `context_archive.html` for persistent, long-term recall.
- **Zero-Allocation**: No dynamic memory allocations occur during the inference loop.
- **Dynamic Mapping**: The loader automatically scans the GGUF tensor map for correct weight keys (supports `blk.` and `layers.` naming conventions).
