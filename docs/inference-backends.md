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

## Performance step

Settings has a **1 Performance** step above Swap. It appears once models are loaded and lists each timed model with up to three numbers. Each title and number has a hover tooltip. Models are named by role and model, e.g. "face detector (scrfd)".

| Column | Meaning |
| --- | --- |
| Compile | WebNN only: building the model's graph and handing it to the platform's ML runtime. For the swapper this includes decoding its weights on the GPU. WebGPU has no such step (it compiles its shaders on first use, which is part of First run), so the column is left out while no model has one. |
| First run | Time to first inference: the first run after the model loaded, until its result is read back. Besides the inference it holds one-time costs: shader and pipeline compiling (WebGPU), buffer allocation and driver warm-up. A shader cache can make it much shorter on a later visit. |
| Steady | Inference time once warmed up: the average of every run after the first, until the result is read back. This is what each frame costs. It shows "–" until there are two runs. |

Hovering a model's name shows the backend it runs on (WebGPU, WebNN GPU or WebNN NPU).

### Which models are timed

- WebNN: the swapper, the detector and the occlusion mask, when they run through WebNN.
- WebGPU: the detector and the recognizer, which are read back after every run, so every run is timed.
- WebGPU swapper, enhancer and occlusion mask: they stay queued on the GPU with no read-back, so a wall-clock time would measure recording the work, not running it. Instead `gpuRun` in [pipeline.js](../web/app/pipeline.js) waits for the work before a run, then for the run itself. That stalls the queue, so only the first three runs and every 8th after them are timed.

### How it is measured

- `track()` in [support.js](../web/app/webnn/support.js) times a function and records `{ first, last, sum, n }` in milliseconds. First run is `first`, and Steady is `(sum - first) / (n - 1)`.
- WebNN: graph builds are timed in `ensureGraph()` of each model, and each dispatch is timed from `dispatch` to the read-back of its result (`once()` in `support.js`). The first dispatch and each build are also logged to the console as `[WebNN npu] inswapper: build graph (batch 1) start` / `done in N ms`, per model and device.
- WebGPU: the detector's and the recognizer's run, through the read-back, are timed in `detect()` and `embed()` of [pipeline.js](../web/app/pipeline.js). The detector's and recognizer's first run is the warm-up on a blank frame made after loading ([session.js](../web/app/session.js)).
- The timings are collected in `SwapPipeline.perf`, sent to the page with the engine's GPU info (`gpuInfo.perf`) and drawn by `renderPerf()` in [main.js](../web/app/main.js). The page refreshes them when the engine replies to a call, so during a long export they update when that call finishes.
- Reloading a model, or switching engine, resets its timings.

### Caveats

- Time to first inference is a proxy for compile time, not a pure measure of it. It also includes allocation and warm-up, and it depends on whether the shader cache is cold or warm.
- Compare WebNN's Compile plus First run with WebGPU's First run, not Compile alone with First run.
- The WebGPU numbers include any queued work the read-back had to wait for, such as the frame upload and the resample before the model.
- The WebNN/WebGPU interop path (`webnnInterop`, off by default) is not timed.
