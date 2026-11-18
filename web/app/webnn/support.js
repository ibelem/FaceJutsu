// WebNN (Web Neural Network API) feature detection, usable from the page and from workers.
//
// Chrome ships WebNN behind a flag / origin trial: chrome://flags/#web-machine-learning-neural-network
// (or --enable-features=WebMachineLearningNeuralNetwork).

export const WEBNN_FLAG = "chrome://flags/#web-machine-learning-neural-network";

export function hasWebNN() {
  return typeof navigator !== "undefined" && !!navigator.ml && typeof MLGraphBuilder !== "undefined";
}

// device: a GPUDevice to share with WebGPU (tensors can then be lent to it, see gpuInterop), or
// null for a context of its own. kind: "gpu" or "npu" (an NPU can't share a GPUDevice)
export async function createWebNNContext(device = null, kind = "gpu") {
  if (!hasWebNN()) throw new Error(`WebNN is not enabled in this browser (turn on ${WEBNN_FLAG})`);
  if (kind === "npu") return navigator.ml.createContext({ powerPreference: "default", deviceType: "npu" });
  if (device) {
    try {
      return await navigator.ml.createContext(device);
    } catch (e) {
      console.warn("WebNN can't share the WebGPU device, using a context of its own:", e.message);
    }
  }
  // deviceType is ignored by current Chrome builds (powerPreference picks the GPU) but older ones need it
  return navigator.ml.createContext({ powerPreference: "high-performance", deviceType: "gpu" });
}

// Whether tensors of a context made from a GPUDevice can be lent to WebGPU: createExportableTensor
// makes a tensor that exportToGPU turns into a GPUBuffer of that device; WebNN can use the tensor
// again once the buffer is destroyed. Lets WebGPU write a graph's inputs and read its outputs
// without a copy through the CPU.
export const gpuInterop = (context) => typeof context.createExportableTensor === "function" && typeof context.exportToGPU === "function";

const types = (limits, op, arg) => limits?.[op]?.[arg]?.dataTypes || [];

// Runs fn, logging before and after it (and how long it took): `what` of `model` on m's device.
export async function timed(m, model, what, fn) {
  const tag = `[WebNN ${m.nnDevice}] ${model}: ${what}`;
  console.log(`${tag} start`);
  const t0 = performance.now();
  try {
    const r = await fn();
    console.log(`${tag} done in ${(performance.now() - t0).toFixed(0)} ms`);
    return r;
  } catch (e) {
    console.log(`${tag} failed after ${(performance.now() - t0).toFixed(0)} ms: ${e.message || e}`);
    throw e;
  }
}

// Runs fn and records how long it took in m.perf[key]: { last, sum, n } (ms). Not for failures.
export async function track(m, key, fn) {
  const t0 = performance.now();
  const r = await fn();
  const ms = performance.now() - t0;
  const s = ((m.perf ||= {})[key] ||= { last: 0, sum: 0, n: 0 });
  s.last = ms;
  s.sum += ms;
  s.n++;
  return r;
}

// A graph's dispatch, through the read back of its outputs (what it takes to have them), recorded
// in m.perf.dispatch; the graph's first run is also logged (g.ran), so frames don't each log.
export function once(g, m, model, what, fn) {
  const run = () => track(m, "dispatch", fn);
  if (g.ran) return run();
  g.ran = true;
  return timed(m, model, what, run);
}

// The compute type for the convolutions: "float16" when supported (and wanted), else "float32".
export function computeType(context, prefer = "float16") {
  const L = context.opSupportLimits?.() || {};
  const f16 = types(L, "conv2d", "input").includes("float16") && types(L, "conv2d", "filter").includes("float16");
  return prefer === "float16" && f16 ? "float16" : "float32";
}

