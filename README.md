# Rusty LLMs

This repo is the result of a conference to learn LLM core techniques in Rust: REPL, RAG, and agents.

The implementations are kept at a foundational level.

**Quick overview:**
- `main.rs` — entry point for each module
- `Cargo.toml` — project manifest, where dependencies are declared
- `cargo run` — command to build and run any module


> Base code from [jdortiz/rs-ai](https://github.com/jdortiz/rs-ai)

**Models:** Ollama with `nomic-embed-text` + `mistral` (default)

---

### `repl/` — Chat REPL
Uses the `llm` library

### `rag/` — RAG Pipeline (In-Memory Vector Search)
Uses the `rig` library

### `agents/` — Agentic Loop with Tool Calling
Uses the `rig` library

Run `cargo run` inside any module folder to execute it.

---

## Prerequisites

- [Rust](https://rustup.rs/) (edition 2021)
- [Ollama](https://ollama.com/) for local models, **or** an OpenAI API key for the `api-llm-key` feature
  - run `ollama run mistral` to pull the model (`mistral` is the default and the smallest option)