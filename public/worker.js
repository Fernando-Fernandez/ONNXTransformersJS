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

      this.model ??= await AutoModelForCausalLM.from_pretrained(this.model_id, {
        dtype: "q4f16",
        device: "webgpu",
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

  const token_callback_function = (tokens) => {
    console.log('Token callback:', tokens);
    startTime ??= performance.now();
    if (numTokens++ > 0) {
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
    skip_special_tokens: true,
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

  const decoded = tokenizer.batch_decode(sequences, { skip_special_tokens: true });
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

  const friendlyName = "Qwen3-0.6B-ONNX";
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
    self.postMessage({ status: "loading", data: "Loading Qwen3-0.6B-ONNX..." });

    const [tokenizer, model] = await TextGenerationPipeline.getInstance(handleProgress);
    console.log('Model loaded successfully');

    self.postMessage({ status: "loading", data: "Compiling shaders and warming up model..." });
    const inputs = tokenizer("a");
    console.log('Warmup inputs:', inputs);
    await model.generate({ ...inputs, max_new_tokens: 1 });
    console.log('Warmup complete');
    self.postMessage({ status: "ready" });
  } catch (error) {
    console.error('Model load failed:', error);
    const errorMessage = error?.message || error?.toString() || `Unknown error (${typeof error}): ${JSON.stringify(error)}`;
    self.postMessage({
      status: "error",
      data: `Model load failed: ${errorMessage}`
    });
  }
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
  }
});
