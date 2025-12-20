/*
 * worker.js – Web Worker that runs the Qwen3‑0.6B model in the browser.
 * ---------------------------------------------------------------
 * This script is loaded as a classic Web Worker (via importScripts) to
 * work around the file:// security restrictions. It imports the
 * HuggingFace Transformers library, sets up the model pipeline, and
 * communicates with the main thread via postMessage.
 */
// Use importScripts for classic worker support (required for file:// protocol)
importScripts('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0/dist/transformers.min.js');

// Destructure from the global 'transformers' object
const {
  AutoTokenizer,
  AutoModelForCausalLM,
  TextStreamer,
  InterruptableStoppingCriteria,
} = transformers;

console.log('Imported dependencies via importScripts');

/*
 * Helper: verify that the browser supports WebGPU.
 * This function attempts to request a GPU adapter and reports any
 * failure back to the main thread. It is called during the model
 * loading phase to ensure the environment can run the model.
 */
async function check() {
  console.log('Running WebGPU check');
  try {
    const adapter = await navigator.gpu.requestAdapter();
    console.log('Got adapter:', adapter);
    if (!adapter) {
      throw new Error("WebGPU is not supported (no adapter found)");
    }
  } catch (e) {
    console.error('WebGPU check failed:', e);
    self.postMessage({
      status: "error",
      data: e.toString(),
    });
  }
}

/*
 * TextGenerationPipeline – lazily loads the tokenizer and model.
 * The static `getInstance` method caches the objects so they are only
 * loaded once. It also handles progress callbacks and provides friendly
 * error messages for common failure modes.
 */
class TextGenerationPipeline {
  static model_id = "onnx-community/Qwen3-0.6B-ONNX";

  static async getInstance(progress_callback = null) {
    console.log('Getting pipeline instance');
    try {
      this.tokenizer ??= await AutoTokenizer.from_pretrained(this.model_id, {
        progress_callback,
      });
      console.log('Tokenizer loaded successfully');

      // Choose device/dtype dynamically: prefer an explicit _preferred_device if set,
      // otherwise use WebGPU. Dtype is selected from _preferred_dtype or the centralized registry.
      const preferredDevice = this._preferred_device ?? 'webgpu';
      const preferredDtype = this._preferred_dtype || (this._model_registry && this._model_registry[this.model_id] && this._model_registry[this.model_id].dtype) || (/gemma/i.test(this.model_id) ? 'fp32' : (/nanochat/i.test(this.model_id) ? 'q4' : 'q4f16'));
      this.model ??= await AutoModelForCausalLM.from_pretrained(this.model_id, {
        dtype: preferredDtype,
        device: preferredDevice,
        progress_callback,
      });
      console.log('Model loaded successfully');

      return [this.tokenizer, this.model];
    } catch (error) {
      console.error('Failed to load model:', error);
      let errorMessage = error?.message || error?.toString() || `Unknown error (${typeof error}): ${JSON.stringify(error)}`;

      // Handle specific ONNX/WebGPU errors
      if (errorMessage.includes('3944596720') || errorMessage.includes('WebGPU')) {
        errorMessage = 'WebGPU device creation failed. Try refreshing the page or check your GPU drivers.';
      } else if (errorMessage.includes('onnxruntime') || errorMessage.includes('session')) {
        errorMessage = 'Model initialization failed. The model may be corrupted or incompatible.';
      } else if (errorMessage.includes('memory') || errorMessage.includes('OOM')) {
        errorMessage = 'Insufficient GPU memory. Try closing other tabs or use a device with more VRAM.';
      }

      self.postMessage({
        status: "error",
        data: `Model loading failed: ${errorMessage}`
      });
      throw error;
    }
  }
}

/*
 * Stopping criteria – allows the generation to be interrupted by the
 * user. An instance of `InterruptableStoppingCriteria` is shared across
 * generation calls so that a single interrupt command can stop the
 * current inference.
 */
const stopping_criteria = new InterruptableStoppingCriteria();
let past_key_values_cache = null;

/*
 * generate(messages) – core generation loop.
 * Takes the chat history, builds the model input, streams token output
 * back to the UI, and separates any `<think>` tags into a separate
 * thought payload.
 */
async function generate(messages) {
  console.log('Starting generation with messages:', messages);
  const [tokenizer, model] = await TextGenerationPipeline.getInstance();
  console.log('Got tokenizer and model instances');

  const inputs = tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    return_dict: true,
  });
  console.log('Applied chat template:', inputs);

  let state = "thinking";
  let startTime;
  let numTokens = 0;
  let tps;
  let rawBuffer = "";

  // Regex for special tokens of the form <|...|> (ASCII) and fullwidth variants like <｜...｜>.
  // Also detect explicit end-of-turn and end-of-sentence tokens including fullwidth and U+2581 underscores.
  const SPECIAL_TOKEN_RE = /<\|[^|]*\|>|<｜[^｜]*｜>/g;
  const END_OF_TURN_RE = /<end_of_turn>|<｜end(?:_|▁)of(?:_|▁)sentence｜>/g;

  function logAndStripTokens(str, ctx) {
    if (!str) return str;
    const matches = str.match(SPECIAL_TOKEN_RE);
    if (matches && matches.length) {
      matches.forEach(m => console.log('Special token (' + ctx + '):', m));
    }
    const endMatches = str.match(END_OF_TURN_RE);
    if (endMatches && endMatches.length) {
      endMatches.forEach(m => console.log('End-of-turn token (' + ctx + '):', m));
    }
    return str.replace(SPECIAL_TOKEN_RE, '').replace(END_OF_TURN_RE, '');
  }

  const token_callback_function = (tokens) => {
    // tokens may be BigInt values or numeric ids; normalize for decoding
    startTime ??= performance.now();
    try {
      const tokenIds = Array.isArray(tokens) ? tokens.map(t => (typeof t === 'bigint' ? Number(t) : t)) : [tokens];
      let decoded = null;
      if (tokenizer && typeof tokenizer.decode === 'function') {
        decoded = tokenizer.decode(tokenIds, { skip_special_tokens: false });
      } else if (tokenizer && typeof tokenizer.batch_decode === 'function') {
        decoded = tokenizer.batch_decode([tokenIds], { skip_special_tokens: false })[0];
      }
      if (decoded !== null) {
        console.log('Decoded token text:', decoded);
        const tokenDebugMatches = (decoded || '').match(SPECIAL_TOKEN_RE);
        if (tokenDebugMatches) tokenDebugMatches.forEach(m => console.log('Special token (token_debug):', m));
        const tokenDebugEndMatches = (decoded || '').match(END_OF_TURN_RE);
        if (tokenDebugEndMatches) tokenDebugEndMatches.forEach(m => console.log('End-of-turn (token_debug):', m));
        const tokenDebugSafe = (decoded || '').replace(SPECIAL_TOKEN_RE, '').replace(END_OF_TURN_RE, '');
        self.postMessage({ status: 'token_debug', tokens: tokenIds, text: tokenDebugSafe });
      }
    } catch (e) {
      console.warn('Token decode failed:', e);
    }

    if (numTokens++ > 0 && numTokens % 5 === 0) {
      tps = (numTokens / (performance.now() - startTime)) * 1000;
      console.log('Current TPS:', tps);
    }
  };

  const callback_function = (output) => {
    console.log('Output callback:', output);
    rawBuffer += output;

    // Split thinking vs answer based on <think> ... </think>
    let thought = '';
    let answer = rawBuffer;
    const start = rawBuffer.indexOf('<think>');
    const end = rawBuffer.indexOf('</think>');

    if (start !== -1) {
      if (end !== -1 && end > start) {
        thought = rawBuffer.slice(start + 7, end).trim();
        answer = rawBuffer.slice(end + 8);
        state = "answering";
      } else {
        thought = rawBuffer.slice(start + 7);
        answer = rawBuffer.slice(0, start);
        state = "thinking";
      }
    } else {
      state = "answering";
    }

    // Strip special tokens before sending to the UI, but keep a log for debugging
    thought = logAndStripTokens(thought, 'thought');
    answer = logAndStripTokens(answer, 'answer');

    self.postMessage({
      status: "update",
      output: answer,
      thought,
      tps,
      numTokens,
      state,
    });
  };

  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: false,
    callback_function,
    token_callback_function,
  });
  console.log('Created streamer');

  self.postMessage({ status: "start" });

  const { past_key_values, sequences } = await model.generate({
    ...inputs,
    do_sample: false,
    max_new_tokens: 2048,
    streamer,
    stopping_criteria,
    return_dict_in_generate: true,
  });
  console.log('Generation complete:', sequences);

  past_key_values_cache = past_key_values;

  let decoded = tokenizer.batch_decode(sequences, { skip_special_tokens: true });
  // decoded may be an array of strings; log and strip any special tokens
  if (Array.isArray(decoded)) {
    decoded = decoded.map(d => {
      const matches = (d || '').match(SPECIAL_TOKEN_RE);
      if (matches) matches.forEach(m => console.log('Special token (final):', m));
      const endMatches = (d || '').match(END_OF_TURN_RE);
      if (endMatches) endMatches.forEach(m => console.log('End-of-turn (final):', m));
      return (d || '').replace(SPECIAL_TOKEN_RE, '').replace(END_OF_TURN_RE, '');
    });
  } else if (typeof decoded === 'string') {
    const matches = decoded.match(SPECIAL_TOKEN_RE);
    if (matches) matches.forEach(m => console.log('Special token (final):', m));
    const endMatches = decoded.match(END_OF_TURN_RE);
    if (endMatches) endMatches.forEach(m => console.log('End-of-turn (final):', m));
    decoded = decoded.replace(SPECIAL_TOKEN_RE, '').replace(END_OF_TURN_RE, '');
  }
  console.log('Decoded output:', decoded);
  self.postMessage({ status: "complete", output: decoded });
}

/*
 * handleProgress(event) – reports file download progress to the UI.
 * It distinguishes between initiation, incremental progress, and
 * completion, sending appropriate status messages.
 */
function handleProgress(event) {
  console.log('Progress event:', event);
  if (!event.total) return;

  const friendlyName = TextGenerationPipeline?.model_id || "onnx-community/Qwen3-0.6B-ONNX";
  const fileLabel = event.url || friendlyName;

  if (event.loaded === 0) {
    console.log('Starting file load:', event.url);
    self.postMessage({
      status: "initiate",
      file: fileLabel,
      progress: 0,
      total: event.total,
    });
  } else if (event.loaded < event.total) {
    const percent = Math.round((event.loaded / event.total) * 100);
    console.log(`Loading progress: ${percent}%`);
    self.postMessage({
      status: "progress",
      file: fileLabel,
      progress: percent,
      total: 100,
    });
  } else {
    console.log('File load complete:', event.url);
    self.postMessage({
      status: "done",
      file: fileLabel,
    });
  }
}

/*
 * load() – orchestrates model loading and warm‑up.
 * Checks WebGPU support, loads the tokenizer and model (with progress
 * callbacks), runs a tiny warm‑up generation to compile shaders, and
 * notifies the main thread when the model is ready.
 */
async function load() {
  console.log('Starting model load');
  self.postMessage({ status: "loading", data: "Checking WebGPU support..." });

  try {
    // First check for WebGPU support
    console.log('Running WebGPU check');
    const adapter = await navigator.gpu.requestAdapter();
    console.log('Got adapter:', adapter);
    if (!adapter) {
      throw new Error("WebGPU is not supported (no adapter found)");
    }

    // If we get here, WebGPU is supported, so proceed with loading the model
    const modelId = TextGenerationPipeline?.model_id || "onnx-community/Qwen3-0.6B-ONNX";
    self.postMessage({ status: "loading", data: `Loading ${modelId}...` });

    const [tokenizer, model] = await TextGenerationPipeline.getInstance(handleProgress);
    console.log('Model loaded successfully');

    self.postMessage({ status: "loading", data: "Compiling shaders and warming up model..." });
    const inputs = tokenizer("a");
    console.log('Warmup inputs:', inputs);
    await model.generate({ ...inputs, max_new_tokens: 1 });
    console.log('Warmup complete');
    self.postMessage({ status: "ready", model: modelId });
  } catch (error) {
    console.error('Model load failed:', error);
    const errorMessage = error?.message || error?.toString() || `Unknown error (${typeof error}): ${JSON.stringify(error)}`;
    self.postMessage({
      status: "error",
      data: `Model load failed: ${errorMessage}`
    });
  }
}

async function unloadModel() {
  console.log('Unloading model resources');
  try {
    if (TextGenerationPipeline?.model && typeof TextGenerationPipeline.model.dispose === 'function') {
      TextGenerationPipeline.model.dispose();
    }
  } catch (disposeError) {
    console.warn('Model dispose failed:', disposeError);
  }
  TextGenerationPipeline.model = null;
  TextGenerationPipeline.tokenizer = null;
  TextGenerationPipeline._cpuFallbackTried = false;
  past_key_values_cache = null;
  stopping_criteria.reset();
}

/*
 * Message dispatcher – reacts to commands from the main thread.
 * Supported message types: "check", "load", "generate", "interrupt",
 * and "reset". Each case forwards to the appropriate helper function.
 */
self.addEventListener("message", async (e) => {
  const { type, data } = e.data;
  console.log('Received message:', type, data);

  switch (type) {
    case "check":
      check();
      break;
    case "set_model":
      // Support either a plain string (modelId) or an object { model_id, dtype }
      console.log('Setting model id to', data);
      if (typeof data === 'string') {
        TextGenerationPipeline.model_id = data;
        TextGenerationPipeline._preferred_dtype = null;
      } else if (data && typeof data === 'object') {
        TextGenerationPipeline.model_id = data.model_id || TextGenerationPipeline.model_id;
        TextGenerationPipeline._preferred_dtype = data.dtype || null;
      }
      TextGenerationPipeline.tokenizer = null;
      TextGenerationPipeline.model = null;
      self.postMessage({ status: 'model_changed', data });
      break;
    case "model_registry":
      // Receive centralized registry from main thread
      TextGenerationPipeline._model_registry = data;
      self.postMessage({ status: 'registry_received' });
      break;
    case "load":
      load();
      break;
    case "generate":
      stopping_criteria.reset();
      generate(data);
      break;
    case "interrupt":
      console.log('Interrupting generation');
      stopping_criteria.interrupt();
      break;
    case "reset":
      console.log('Resetting state');
      past_key_values_cache = null;
      stopping_criteria.reset();
      break;
    case "unload":
      console.log('Received unload request');
      stopping_criteria.interrupt();
      self.postMessage({ status: "unloading" });
      await unloadModel();
      self.postMessage({ status: "unloaded" });
      break;
  }
});
