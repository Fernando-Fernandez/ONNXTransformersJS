# In‑Browser ONNX Model Demo (Qwen / Llama)

This project demonstrates running ONNX language models directly in the browser using WebGPU and a bundled Transformers IIFE. It's a lightweight, local demo that downloads models at runtime and runs inference inside a Web Worker.

ONNX is the [Open Neural Network Exchange format](https://onnx.ai/) that enables interoperability between AI frameworks (PyTorch, TensorFlow, Caffe2) and across platforms. This demo uses the [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) backend via the [Transformers.js](https://github.com/huggingface/transformers.js) library.

# In‑Browser ONNX Model Demo (Qwen / Llama + more)

This project demonstrates running ONNX language models directly in the browser using WebGPU and a bundled Transformers IIFE. It's a lightweight, local demo that downloads models at runtime and runs inference inside a Web Worker.

The demo uses the [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) backend via the [Transformers.js](https://github.com/huggingface/transformers.js) library.

What's in this repo

- A small frontend (`index.html`, `styles.css`, `app.js`) that talks to a worker for inference.
- A blob-based inlined worker (created from `app.js`) and a standalone worker (`public/worker.js`) that both use the bundled Transformers IIFE (`public/transformers_lib.js`).
- A centralized `MODEL_REGISTRY` at `public/models.js` that contains model ids, friendly names, default dtypes, and whether the model exposes internal "thoughts".

What changed recently

- Model registry centralized: `public/models.js` is the source-of-truth for available models and metadata (`friendly`, `dtype`, `thinking`). The UI populates the model dropdown from this registry.
- Workers receive the registry (the blob worker gets it via `postMessage` and the standalone worker can be configured to `importScripts('public/models.js')` or receive it the same way). Workers prefer the registry's dtype when loading a model.
- Special/control tokens (ASCII `<|...|>` and fullwidth variants like `<｜...｜>`) and explicit end-of-sentence tokens such as `<｜end▁of▁sentence｜>` are now logged to the console but stripped from UI output. This avoids showing control tokens in the chat while keeping them available for debugging via console logs and `token_debug` messages.

Quick usage

1. Open the project root and serve or open `index.html` in a modern Chromium-based browser with WebGPU support. For a quick local server you can run:

```bash
# from the project root
python3 -m http.server 8000
# then open http://localhost:8000 in your browser
```

2. The UI performs a WebGPU check. If WebGPU is available the model loader will proceed.
3. Select a model from the `Model:` dropdown (populated from `public/models.js`). The UI shows loading progress and a friendly model name.
4. When the model is ready you can send messages in the chat input. Responses stream incrementally; `<think>...</think>` segments (if produced) appear in the Thought Panel.

Developer notes

- Centralized model registry: edit `public/models.js` to add/remove models. Each entry should look like:

```js
	'owner/model-name-ONNX': { friendly: 'Friendly Name', dtype: 'q4f16'|'q4'|'fp32', thinking: false }
```

- Worker behavior:
	- The blob worker (created from `app.js`) receives the registry via `worker.postMessage({ type: 'model_registry', data: MODEL_REGISTRY })` on startup.
	- The standalone worker (`public/worker.js`) currently accepts the `model_registry` message as well. Optionally you can have the standalone worker call `importScripts('public/models.js')` to read the registry directly instead of receiving it by postMessage.
	- When the worker loads a model it prefers the registry-defined `dtype` for that model; fallbacks exist for models not present in the registry.

- Token handling:
	- The workers detect both ASCII special tokens (`<|...|>`) and fullwidth variants (`<｜...｜>`), as well as an explicit fullwidth end-of-sentence token pattern like `<｜end▁of▁sentence｜>`. These are logged to the console (and emitted as `token_debug` messages) but removed from the UI output so users don't see control tokens.

Testing tips

- Open DevTools → Console to inspect worker logs. Look for:
	- `registry_received` (worker acknowledged the registry)
	- `Model ready` or `Model load failed: ...` messages
	- `Special token (...)` or `End-of-turn token (...)` logs when token-debugging

- If a model fails to load due to memory or WebGPU errors, the worker falls back to a safer configuration where possible (e.g., `wasm`/`fp32`) and reports status messages to the UI.

Contributing

- To add a model, update `public/models.js` and include a `dtype` suitable for the model (for quantized models use `q4`/`q4f16`, for small FP models use `fp32`).
- If you prefer the standalone worker to read the registry directly, replace the `model_registry` message handler in `public/worker.js` with a call to `importScripts('public/models.js')` and remove the `postMessage` from `app.js` that sends the registry.

License / Disclaimer

This is an experimental demo. Models referenced in the registry are loaded at runtime from Hugging Face (or other remotes) and may have their own licenses and terms. Use with models you have permission to load.

Enjoy running ONNX models in the browser!
