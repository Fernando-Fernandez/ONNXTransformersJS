# In‑Browser ONNX Model Demo (Qwen / Llama)

This project demonstrates running ONNX language models directly in the browser using WebGPU and a bundled Transformers IIFE. It's a lightweight, local demo that downloads models at runtime and runs inference inside a Web Worker.

**What's New:**

- **Model Selection UI:** A dropdown lets you choose which ONNX model to load (`onnx-community/Qwen3-0.6B-ONNX` or `onnx-community/Llama-3.2-1B-Instruct-ONNX`).
- **Persistent Model Label:** The currently loaded model is shown under the selector (`Loaded model:`) with a friendly name.
- **Dynamic Worker Loading:** The main thread sends the selected model id to the worker, which clears cached instances and loads the requested model.

**Key Points:**

- **Vanilla HTML/JS/CSS**: Open `index.html` (file:// or served) — no build step required.
- **Worker-based inference**: `public/transformers.iife.js` is inlined into a Blob worker so the demo works under `file://`.
- **ONNX + WebGPU**: Models run using the ONNX runtime via the Transformers JS bindings, targeting `webgpu` and a quantized dtype for performance.
- **Thought panel & streaming**: The worker streams generated text and separates `<think>...</think>` segments into the thought panel.

**Files changed (recent):**

- `index.html` — added the model selector and a persistent `Loaded model` label.
- `app.js` — wired the selector to send `{ type: 'set_model', data: '<model-id>' }` to the worker, added friendly-name mapping, and updated UI handlers. The worker now includes the selected model id in readiness messages.

**How to run (quick):**

1. Open the project root and launch the demo in your browser (file protocol works):

```bash
open index.html
```

2. Wait for the WebGPU check to complete (the status message appears at the top).
3. Use the `Model:` dropdown to pick a model — the UI will show `Loaded model: <name> (switching...)` while downloading, then update when ready.
4. When ready, use the chat input to send messages and the model will stream responses into the chat and thought panel.

**Troubleshooting:**

- **WebGPU not supported**: The demo requires a browser with WebGPU and a working adapter. If you see `WebGPU not supported` in the UI, try a Chromium-based browser with an enabled WebGPU flag or update GPU drivers.
- **OOM / Memory errors**: Loading large models may fail if the device lacks enough GPU memory. Try a smaller model or close other GPU-intensive applications.
- **Model load fails**: Open DevTools → Console to inspect worker error messages; the worker posts helpful status/errors to the main thread UI as well.