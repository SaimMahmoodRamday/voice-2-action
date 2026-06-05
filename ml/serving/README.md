# Serving the fine-tuned model via Ollama

The backend talks to Ollama over HTTP and uses the model named `voice2action`.
Ollama cannot load a raw LoRA adapter, so we merge the adapter into the base,
export to GGUF, and register it as an Ollama model.

## Artifacts in this directory

- `Modelfile` — Ollama model definition (Qwen2.5 chat template + system prompt). **Committed.**
- `voice2action-q8_0.gguf` — quantized model weights, ~3.3 GB. **Gitignored** (build artifact). Regenerate with the steps below.

## Regenerate the GGUF (from the trained LoRA adapter)

Requires the training env (torch + transformers + peft) and llama.cpp checked out at `tools/llama.cpp`.

```bash
# 1. Merge LoRA adapter into the base model (fp16, runs on CPU)
python ml/scripts/merge_lora.py

# 2. Convert merged model straight to Q8_0 GGUF
python tools/llama.cpp/convert_hf_to_gguf.py \
    ml/outputs/qwen2.5-3b-voice2action-merged \
    --outtype q8_0 \
    --outfile ml/serving/voice2action-q8_0.gguf
```

To get llama.cpp:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp tools/llama.cpp
pip install gguf sentencepiece
```

## Register with Ollama (local / dev)

```bash
cd ml/serving
ollama create voice2action -f Modelfile
ollama run voice2action "Ahmed ko call karna hai kal tak"
```

The backend then picks it up via `OLLAMA_MODEL=voice2action` in `backend/.env`.

## Docker

`docker-compose.yml` runs Ollama in its own container with a private volume.
The `model-init` one-shot service mounts this directory and registers the model
automatically (idempotently) on `docker compose up` — no manual step needed:

```bash
docker compose up
```

For a distributed deploy, push the model to a registry once and replace the
`ollama create` in the `model-init` service with `ollama pull <registry>/voice2action`.
