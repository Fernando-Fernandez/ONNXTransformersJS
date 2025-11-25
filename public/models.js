(function(global){
  const MODEL_REGISTRY = {
    'onnx-community/Llama-3.2-1B-Instruct-ONNX': { friendly: 'Llama‑3.2‑1B‑Instruct', dtype: 'q4f16', thinking: false },
    'onnx-community/Qwen3-0.6B-ONNX': { friendly: 'Qwen3‑0.6B', dtype: 'q4f16', thinking: true },
    'onnx-community/NanoChat-d32-ONNX': { friendly: 'NanoChat‑d32', dtype: 'q4', thinking: false },
    'onnx-community/gemma-3-270m-it-ONNX': { friendly: 'Gemma‑3‑270m‑IT', dtype: 'fp32', thinking: false },
    'onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX': { friendly: 'DeepSeek R1 (Qwen‑1.5B)', dtype: 'q4f16', thinking: true },
    'onnx-community/LFM2-1.2B-ONNX': { friendly: 'LFM2‑1.2B', dtype: 'q4', thinking: false }
  };

  try {
    if (typeof window !== 'undefined') window.MODEL_REGISTRY = MODEL_REGISTRY;
    if (typeof self !== 'undefined') self.MODEL_REGISTRY = MODEL_REGISTRY;
  } catch (e) {
    // ignore
  }

})(typeof globalThis !== 'undefined' ? globalThis : (typeof self !== 'undefined' ? self : this));
