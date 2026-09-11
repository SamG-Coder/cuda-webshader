# Research and design references

Reviewed for this project on 12 September 2026. Primary sources only; the implementation is original project code. These notes explain design choices, not successful hardware validation.

## CUDA execution model and useful tests

NVIDIA's CUDA Programming Guide, “Writing SIMT Kernels,” defines block/thread/grid indexing and describes shared-memory coordination. Its matrix and transpose examples make useful structural tests for translation: a translator must preserve per-thread indices, bounds and barrier participation, not just substitute names.

https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html

The project therefore exercises vector arithmetic, shared/register-tiled matrix multiplication, cooperative reduction, convolution halos, a shared atomic histogram, padded transpose and particle updates. The tests intentionally include irregular sizes to expose code that appears correct only on exact tile boundaries.

## WGSL language and memory model

The W3C WGSL specification defines the available scalar/vector types, storage/workgroup address spaces, layouts, integer atomic operations and uniformity requirements. CUDA types and memory layout cannot all be mechanically retained. A notable ABI hazard is float3 arrays: the project rejects that pointer ABI rather than assume packed CUDA records match WGSL storage-array stride.

https://www.w3.org/TR/WGSL/

WGSL `select` is a value-selection builtin, not a lazy C conditional expression. The compiler uses real conditional control flow for ternary expressions, preserving the selected branch rather than evaluating potentially invalid loads or side effects unconditionally. WGSL's own shader validator remains the final gate for type/behavior rules and uniformity analysis.

## WebGPU execution, limits and timestamps

The WebGPU specification defines GPUBuffer usage, device limits, command encoding/submission, buffer mapping and optional timestamp-query features. The runtime queries limits, uses explicit storage/uniform bindings, checks launch shapes and labels timestamp versus wall-clock measurements separately.

https://gpuweb.github.io/gpuweb/

All normal-frame compute/render work uses one device queue. Data crosses back to CPU only through explicit test/readback calls. Benchmark output distinguishes GPU-pass intervals from encode/submit-to-completion wall time, and retains repeated-dispatch/caching caveats.

## Three.js rendering and shared storage

Three's WebGPURenderer documentation states that the renderer uses WebGPU and can fall back to WebGL 2. The project refuses that fallback: WebGL rendering is not a substitute for the WGSL compute path.

https://threejs.org/docs/pages/WebGPURenderer.html
https://threejs.org/docs/pages/StorageBufferAttribute.html
https://threejs.org/docs/pages/StorageInstancedBufferAttribute.html

The following exact r186 sources were inspected through the GitHub connector:

- `WebGPUBackend.js` accepts an externally supplied `GPUDevice` and exposes its storage-attribute allocation path.
- `WebGPUAttributeUtils.js` retains a buffer in backend attribute data and pads certain storage layouts.
- `StorageInstancedBufferAttribute.js` supports typed-array initialization.
- `examples/webgpu_compute_particles.html` demonstrates a storage-driven SpriteNodeMaterial and instanced sprites.
- The common renderer API exposes asynchronous render-target pixel readback, used by the interop test.

https://github.com/mrdoob/three.js/blob/r186/src/renderers/webgpu/WebGPUBackend.js
https://github.com/mrdoob/three.js/blob/r186/src/renderers/webgpu/utils/WebGPUAttributeUtils.js
https://github.com/mrdoob/three.js/blob/r186/src/renderers/common/StorageInstancedBufferAttribute.js
https://github.com/mrdoob/three.js/blob/r186/examples/webgpu_compute_particles.html
https://github.com/mrdoob/three.js/blob/r186/src/renderers/common/Renderer.js

The bridge's backend-buffer access is an internal dependency, not a claim of a permanently stable public Three.js API. It is isolated to one module with a strict revision guard and a supplied pixel test. Upgrade the pin only after running that test on the target browser/GPU.

## Claims deliberately not made

No evidence in this project demonstrates arbitrary CUDA compatibility, native-CUDA-equivalent performance, cross-vendor numerical identity, tensor-core support or production readiness. GPU benchmarks must actually run on a target device before speed claims are made. CPU correctness checks and generated source are useful evidence, but are not substitutes for real shader compilation, execution and rendering.
