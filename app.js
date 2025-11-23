// WebGPU Detection Logic
// Checks if the browser supports WebGPU and if an adapter is available.
async function checkWebGPU() {
    const statusEl = document.getElementById('browser-status');

    // Check if navigator.gpu exists (browser support)
    if (!navigator.gpu) {
        statusEl.innerHTML = '❌ WebGPU not supported in this browser.';
        return false;
    }

    try {
        // Request a GPU adapter
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            statusEl.innerHTML = '❌ No WebGPU adapter found.';
            return false;
        }

        // Request a device to ensure we can actually use the GPU
        const device = await adapter.requestDevice();
        device.destroy(); // Clean up the test device

        statusEl.innerHTML = '✅ WebGPU is supported and ready.';
        return true;
    } catch (e) {
        statusEl.innerHTML = `❌ WebGPU error: ${e.message}`;
        return false;
    }
}

// Worker Code as String (to bypass file:// security restrictions)
// This function returns the entire worker script as a string.
// We do this to create a Blob worker, which avoids "SecurityError" when running from file://
const getWorkerCode = (baseUrl) => `
// Define base URL for the library to resolve relative paths correctly
// This is critical for file:// protocol support where relative paths fail in Blob workers
self.transformersBaseUrl = '${baseUrl}';

// Inlined transformers library
// This variable is injected by the build script and contains the bundled library
${TRANSFORMERS_LIB}

// Destructure from the global 'transformers' object provided by the library
const {
  AutoTokenizer,
  AutoModelForCausalLM,
  TextStreamer,
  InterruptableStoppingCriteria,
} = self.transformers;

// Configure WASM paths to use CDN since we are in a blob worker
// Blob workers have an opaque origin, so relative paths to WASM files fail.
// We explicitly point to the CDN versions of the ONNX Runtime WASM files.
if (self.transformers.env && self.transformers.env.wasm) {
    self.transformers.env.wasm.wasmPaths = {
        'ort-wasm-simd-threaded.wasm': 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.0/dist/ort-wasm-simd-threaded.wasm',
        'ort-wasm-simd.wasm': 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.0/dist/ort-wasm-simd.wasm',
        'ort-wasm-threaded.wasm': 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.0/dist/ort-wasm-threaded.wasm',
        'ort-wasm.wasm': 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.0/dist/ort-wasm.wasm',
    };
}

console.log('Imported dependencies via importScripts');

// Check for WebGPU support within the worker context
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

// Singleton class to manage the model pipeline
class TextGenerationPipeline {
  static model_id = "onnx-community/Qwen3-0.6B-ONNX";

  // Lazy-load the tokenizer and model instance
  static async getInstance(progress_callback = null) {
    console.log('Getting pipeline instance');
    try {
      // Load tokenizer if not already loaded
      this.tokenizer ??= await AutoTokenizer.from_pretrained(this.model_id, {
        progress_callback,
      });
      console.log('Tokenizer loaded successfully');
      
      // Load model if not already loaded
      // We explicitly specify 'webgpu' device and 'q4f16' dtype for performance
      this.model ??= await AutoModelForCausalLM.from_pretrained(this.model_id, {
        dtype: "q4f16",
        device: "webgpu", 
        progress_callback,
      });
      console.log('Model loaded successfully');

      return [this.tokenizer, this.model];
    } catch (error) {
      console.error('Failed to load model:', error);
      let errorMessage = error?.message || error?.toString() || \`Unknown error (\${typeof error}): \${JSON.stringify(error)}\`;
      
      // Handle specific ONNX/WebGPU errors with user-friendly messages
      if (errorMessage.includes('3944596720') || errorMessage.includes('WebGPU')) {
        errorMessage = 'WebGPU device creation failed. Try refreshing the page or check your GPU drivers.';
      } else if (errorMessage.includes('onnxruntime') || errorMessage.includes('session')) {
        errorMessage = 'Model initialization failed. The model may be corrupted or incompatible.';
      } else if (errorMessage.includes('memory') || errorMessage.includes('OOM')) {
        errorMessage = 'Insufficient GPU memory. Try closing other tabs or use a device with more VRAM.';
      }
      
      self.postMessage({
        status: "error",
        data: \`Model loading failed: \${errorMessage}\`
      });
      throw error;
    }
  }
}

// Stopping criteria allows us to interrupt generation
const stopping_criteria = new InterruptableStoppingCriteria();
// Cache for past key values to speed up multi-turn generation
let past_key_values_cache = null;

// Main generation function
async function generate(messages) {
  console.log('Starting generation with messages:', messages);
  const [tokenizer, model] = await TextGenerationPipeline.getInstance();
  console.log('Got tokenizer and model instances');

  // Apply chat template to format messages for the model
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

  // Callback for tracking tokens per second (TPS)
  const token_callback_function = (tokens) => {
    console.log('Token callback:', tokens);
    startTime ??= performance.now();
    if (numTokens++ > 0) {
      tps = (numTokens / (performance.now() - startTime)) * 1000;
      console.log('Current TPS:', tps);
    }
  };

  // Callback for handling generated text output
  const callback_function = (output) => {
    console.log('Output callback:', output);
    rawBuffer += output;

    // Logic to separate "thinking" content (<think>...</think>) from the final answer
    let thought = '';
    let answer = rawBuffer;
    const start = rawBuffer.indexOf('<think>');
    const end = rawBuffer.indexOf('</think>');

    if (start !== -1) {
      if (end !== -1 && end > start) {
        // Thought process is complete
        thought = rawBuffer.slice(start + 7, end).trim();
        answer = rawBuffer.slice(end + 8);
        state = "answering";
      } else {
        // Still thinking
        thought = rawBuffer.slice(start + 7);
        answer = rawBuffer.slice(0, start);
        state = "thinking";
      }
    } else {
      state = "answering";
    }

    // Send update to main thread
    self.postMessage({
      status: "update",
      output: answer,
      thought,
      tps,
      numTokens,
      state,
    });
  };

  // Streamer handles decoding tokens into text incrementally
  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function,
    token_callback_function,
  });
  console.log('Created streamer');

  self.postMessage({ status: "start" });

  // Run generation
  const { past_key_values, sequences } = await model.generate({
    ...inputs,
    do_sample: false, // Greedy decoding for deterministic results
    max_new_tokens: 2048,
    streamer,
    stopping_criteria,
    return_dict_in_generate: true,
  });
  console.log('Generation complete:', sequences);

  // Cache KV pairs for next turn
  past_key_values_cache = past_key_values;

  const decoded = tokenizer.batch_decode(sequences, { skip_special_tokens: true });
  console.log('Decoded output:', decoded);
  self.postMessage({ status: "complete", output: decoded });
}

// Handles progress events during model downloading
function handleProgress(event) {
  console.log('Progress event:', event);
  if (!event.total) return;

  const friendlyName = TextGenerationPipeline?.model_id || "onnx-community/Qwen3-0.6B-ONNX";
  const fileLabel = event.url || friendlyName;

  if (event.loaded === 0) {
    // Download started
    console.log('Starting file load:', event.url);
    self.postMessage({
      status: "initiate",
      file: fileLabel,
      progress: 0,
      total: event.total,
    });
  } else if (event.loaded < event.total) {
    // Download in progress
    const percent = Math.round((event.loaded / event.total) * 100);
    console.log(\`Loading progress: \${percent}%\`);
    self.postMessage({
      status: "progress", 
      file: fileLabel,
      progress: percent,
      total: 100,
    });
  } else {
    // Download complete
    console.log('File load complete:', event.url);
    self.postMessage({
      status: "done",
      file: fileLabel,
    });
  }
}

// Initial load function triggered by the main thread
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
    self.postMessage({ status: "loading", data: \`Loading \${modelId}...\` });

    const [tokenizer, model] = await TextGenerationPipeline.getInstance(handleProgress);
    console.log('Model loaded successfully');
    
    // Perform a dry run to compile shaders and warm up the model
    self.postMessage({ status: "loading", data: "Compiling shaders and warming up model..." });
    const inputs = tokenizer("a");
    console.log('Warmup inputs:', inputs);
    await model.generate({ ...inputs, max_new_tokens: 1 });
    console.log('Warmup complete');
    self.postMessage({ status: "ready", model: modelId });
  } catch (error) {
    console.error('Model load failed:', error);
    const errorMessage = error?.message || error?.toString() || \`Unknown error (\${typeof error}): \${JSON.stringify(error)}\`;
    self.postMessage({
      status: "error",
      data: \`Model load failed: \${errorMessage}\`
    });
  }
}

// Worker message listener
self.addEventListener("message", async (e) => {
  const { type, data } = e.data;
  console.log('Received message:', type, data);

  switch (type) {
    case "check":
      check();
      break;
    case "set_model":
      // Change the model id used by the pipeline and clear any cached instances
      console.log('Setting model id to', data);
      TextGenerationPipeline.model_id = data;
      TextGenerationPipeline.tokenizer = null;
      TextGenerationPipeline.model = null;
      self.postMessage({ status: 'model_changed', data });
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
`;

// Application Logic
async function initApp() {
    // Early check for WebGPU support
    const isSupported = await checkWebGPU();
    if (!isSupported) return;

    // Calculate absolute path to transformers.min.js for file:// protocol support
    // We need to pass this to the worker so it can resolve the library correctly
    const basePath = window.location.href.substring(0, window.location.href.lastIndexOf('/') + 1);
    const transformersUrl = basePath + 'public/transformers.iife.js'; // Use the IIFE path as base

    // Create Worker from Blob to support file:// protocol
    // This bypasses the browser restriction on loading workers from local files
    const workerCode = getWorkerCode(transformersUrl);
    const blob = new Blob([workerCode], { type: 'application/javascript' });
    const workerUrl = URL.createObjectURL(blob);
    const worker = new Worker(workerUrl);

    // UI Elements
    const modelStatus = document.getElementById('model-status');
    const loadingFile = document.getElementById('loading-file');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    const chatInterface = document.getElementById('chat-interface');
    const messagesContainer = document.getElementById('messages');
    const messageInput = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    const stopBtn = document.getElementById('stop-btn');
    const resetBtn = document.getElementById('reset-btn');
    const tpsStatus = document.getElementById('tps-status');
    const tpsValue = document.getElementById('tps-value');

    // Thought Panel Elements
    const thoughtPanel = document.getElementById('thought-panel');
    const thoughtContent = document.getElementById('thought-content');
    const toggleThoughtBtn = document.getElementById('toggle-thought-btn');
    const closeThoughtBtn = document.getElementById('close-thought-btn');

    let isGenerating = false;
    let currentAssistantMessageDiv = null;

    // Worker Message Handling
    worker.addEventListener('message', (e) => {
      const { status, data, progress, file, output, thought, tps, model } = e.data;

      const currentModelNameEl = document.getElementById('current-model-name');

      const MODEL_FRIENDLY = {
        'onnx-community/Qwen3-0.6B-ONNX': 'Qwen3‑0.6B',
        'onnx-community/Llama-3.2-1B-Instruct-ONNX': 'Llama‑3.2‑1B‑Instruct',
      };

      function friendlyName(id) {
        return MODEL_FRIENDLY[id] || id;
      }

        switch (status) {
            case 'loading':
            case 'initiate':
            case 'progress':
                // Show loading status and progress bar
                modelStatus.classList.remove('hidden');
                if (file) loadingFile.textContent = file;
                if (progress) {
                    progressFill.style.width = `${progress}%`;
                    progressText.textContent = `${Math.round(progress)}%`;
                }
                break;

            case 'done':
                // File download complete
                progressFill.style.width = '100%';
                progressText.textContent = '100%';
                setTimeout(() => {
                    modelStatus.classList.add('hidden');
                }, 1000);
                break;

              case 'model_changed':
                // Worker acknowledged model switch
                if (data) {
                  const friendly = friendlyName(data);
                  if (currentModelNameEl) currentModelNameEl.textContent = `${friendly} (switching...)`;
                  // show loader UI
                  modelStatus.classList.remove('hidden');
                  loadingFile.textContent = friendly;
                  progressFill.style.width = `0%`;
                  progressText.textContent = `0%`;
                }
                break;

            case 'ready':
              // Model is fully loaded and ready
              modelStatus.classList.add('hidden');
              chatInterface.classList.remove('hidden');
              console.log('Model ready');
              if (model && currentModelNameEl) currentModelNameEl.textContent = friendlyName(model);
              break;

            case 'start':
                // Generation started
                isGenerating = true;
                appendMessage('assistant', '');
                thoughtContent.textContent = ''; // Clear previous thoughts
                updateButtons();
                break;

            case 'update':
                // Received partial output from generation
                if (tps) {
                    tpsStatus.classList.remove('hidden');
                    tpsValue.textContent = tps.toFixed(2);
                }
                if (output) {
                    updateCurrentAssistantMessage(output);
                }
                if (thought) {
                    thoughtContent.textContent = thought;
                    thoughtContent.scrollTop = thoughtContent.scrollHeight;
                    // Auto-show thought panel if there is thought content
                    if (thoughtPanel.classList.contains('hidden')) {
                        thoughtPanel.classList.remove('hidden');
                        toggleThoughtBtn.textContent = 'Hide Thoughts';
                    }
                }
                break;

            case 'complete':
                // Generation finished
                isGenerating = false;
                currentAssistantMessageDiv = null;
                updateButtons();
                break;
        }
    });

    // Model selection control
    const modelSelect = document.getElementById('model-select');

    function setModelAndLoad(modelId) {
      // Clear progress UI and notify worker to switch model
      modelStatus.classList.remove('hidden');
      loadingFile.textContent = `Selected model: ${modelId}`;
      progressFill.style.width = `0%`;
      progressText.textContent = `0%`;

      worker.postMessage({ type: 'set_model', data: modelId });
      // Ask worker to load the newly selected model
      worker.postMessage({ type: 'load' });
    }

    modelSelect.addEventListener('change', (e) => {
      const val = e.target.value;
      setModelAndLoad(val);
    });

    // Initialize Model with currently selected model
    setModelAndLoad(document.getElementById('model-select').value);

    // UI Helpers
    function appendMessage(role, content) {
        const div = document.createElement('div');
        div.className = `message ${role}`;
        div.textContent = content;
        messagesContainer.appendChild(div);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        if (role === 'assistant') {
            currentAssistantMessageDiv = div;
        }
    }

    function updateCurrentAssistantMessage(content) {
        if (currentAssistantMessageDiv) {
            currentAssistantMessageDiv.textContent = content;
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
    }

    function updateButtons() {
        sendBtn.disabled = isGenerating;
        stopBtn.disabled = !isGenerating;
        messageInput.disabled = isGenerating;
    }

    function sendMessage() {
        const text = messageInput.value.trim();
        if (!text || isGenerating) return;

        appendMessage('user', text);
        messageInput.value = '';

        // Construct conversation history (simplified for this demo)
        const history = Array.from(messagesContainer.children).map(div => ({
            role: div.classList.contains('user') ? 'user' : 'assistant',
            content: div.textContent
        }));

        worker.postMessage({
            type: 'generate',
            data: history
        });
    }

    // Toggle Thought Panel Logic
    function toggleThoughtPanel() {
        const isHidden = thoughtPanel.classList.contains('hidden');
        if (isHidden) {
            thoughtPanel.classList.remove('hidden');
            toggleThoughtBtn.textContent = 'Hide Thoughts';
        } else {
            thoughtPanel.classList.add('hidden');
            toggleThoughtBtn.textContent = 'Show Thoughts';
        }
    }

    // Event Listeners
    sendBtn.addEventListener('click', sendMessage);

    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    stopBtn.addEventListener('click', () => {
        worker.postMessage({ type: 'interrupt' });
    });

    resetBtn.addEventListener('click', () => {
        worker.postMessage({ type: 'reset' });
        messagesContainer.innerHTML = '';
        thoughtContent.textContent = '';
        tpsStatus.classList.add('hidden');
    });

    toggleThoughtBtn.addEventListener('click', toggleThoughtPanel);
    closeThoughtBtn.addEventListener('click', toggleThoughtPanel);
}

initApp();
