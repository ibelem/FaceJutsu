# Inference backends

The web app does not use ONNX Runtime Web or Transformers.js. It runs the models with its own code, written directly against the browser's WebGPU and WebNN APIs.

## WebGPU (default)

- Hand-written WebGPU compute shaders: [web/app/gpu/](../web/app/gpu/) (`kernels.js`, `gemm.js`, `ops.js`, `device.js`).
- Model implementations: [web/app/models/](../web/app/models/)
  - `scrfd.js`: face detector
  - `arcface.js`: face recognizer
  - `inswapper.js`: face swapper
  - `xseg.js`: occlusion mask
  - `enhancer.js`: face enhancer
- Weights are `.safetensors` files, read by [web/app/weights.js](../web/app/weights.js). There are no `.onnx` files.
- Frames stay on the GPU between steps. Only the detector's head outputs and the identity embedding come back to the CPU ([web/app/pipeline.js](../web/app/pipeline.js)).

## WebNN (optional)

- Graphs are built with the browser's native `navigator.ml` and `MLGraphBuilder`: [web/app/webnn/](../web/app/webnn/). Feature detection is in [support.js](../web/app/webnn/support.js).
- Chrome needs the flag `chrome://flags/#web-machine-learning-neural-network`.
- Only three parts can run on WebNN:
  - face swapper
  - face detector
  - occlusion mask
- Each part falls back to WebGPU if its WebNN graph fails to build. The error is kept in `pipeline.nnError`.
- The face recognizer and the face enhancer always run on WebGPU.
- `selection.webnnInterop: true` lets the WebNN swapper share the WebGPU device. This is experimental.
- The parts that actually run on WebNN are reported in `gpuInfo.nnParts` ([web/app/session.js](../web/app/session.js)).

### Context options

Every WebNN context is created in `createWebNNContext()` in [support.js](../web/app/webnn/support.js):

```js
navigator.ml.createContext({ powerPreference: "high-performance", deviceType: "gpu" })
```

- `deviceType: "gpu"`: WebNN is always asked for the GPU, never `"cpu"` or `"npu"`. A code comment says current Chrome builds ignore `deviceType` and use `powerPreference` to pick the GPU, so `deviceType` is only there for older builds.
- `powerPreference: "high-performance"`: picks the discrete GPU on laptops that have one. The WebGPU adapter request uses the same preference ([device.js](../web/app/gpu/device.js)).
- WebGPU device sharing: only the swapper can do this. With `webnnInterop` on, [swapper.js](../web/app/webnn/swapper.js) calls `createContext(gpu.device)` with the WebGPU `GPUDevice`. If that fails, it logs a warning and falls back to the options above.
- The detector ([scrfd.js](../web/app/webnn/scrfd.js)) and the occlusion mask ([xseg.js](../web/app/webnn/xseg.js)) call `createWebNNContext()` with no device, so they use the options above.
- The NPU is used only by the "WebNN NPU" engine choice (Settings → Engine), which creates the context with `{ powerPreference: "default", deviceType: "npu" }`. "WebNN GPU" uses the options above. The NPU context never shares the WebGPU device, and an NPU may fail to build a graph, in which case that part falls back to WebGPU.

## Video

Decoding, encoding and muxing use [Mediabunny](../web/app/vendor/mediabunny.min.mjs), which is built on WebCodecs.

## Speed

WebNN may be faster on hardware where it reaches an NPU or a vendor-tuned GPU path. The speed difference has not been measured here.
