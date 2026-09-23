# Experimental WebAssembly CPU backend

`src/wasm/compile.mjs` adds a build-time WASM target alongside WGSL. It uses the
existing WebShader frontend to validate CUDA and generate typed kernel ABI
manifests, then Emscripten/LLVM compiles generated C++ wrappers and the CUDA bodies.
It is a bounded experimental backend, not an implementation of all CUDA.

```js
import {compileThreaded} from './src/wasm/compile.mjs';
await compileThreaded(cudaSource, [
  {entry: 'myKernel', workgroupSize: [128, 1, 1]},
], {outDir: './build/wasm', name: 'compute'});
```

Install/activate Emscripten (tested 6.0.10). The driver defaults to `em++`, or set
`EMXX` to its path. On Windows explicit `.bat` paths invoke the sibling Python
driver without shell interpolation; `EMSDK_PYTHON` selects that interpreter.
Outputs are `compute.mjs`, `compute.wasm`, `compute.cpp`, and `compute.abi.json`.
`emitThreaded(source, specs)` returns `{cpp, kernels}` without invoking a toolchain.

Load the generated module inside a dedicated browser worker, then construct
`ThreadedProgram` from `src/wasm/runtime.js` with the module, ABI and 1–8 threads.
Allocate persistent buffers with `alloc(bytes)`, upload with `write(buffer, view)`,
and call `dispatch(entry, [blocksX, blocksY, blocksZ], namedArguments)`.
Pointer arguments are buffer handles; scalar arguments are numbers. `read(buffer)`
copies bytes back. Call `dispose()` when finished. Concurrent host dispatches on
one module are unsupported. The host worker counts as one participating thread.

The browser must be cross-origin isolated (COOP `same-origin`, COEP `require-corp`)
for shared WASM memory. Serve `.wasm` as `application/wasm`. The module reserves
256 MiB initially (512 MiB maximum) and preloads seven helper workers. Node can
run the same generated module for testing. The UI communicates asynchronously
with the host worker; no Asyncify/JSPI transformation is required.

Pthreads share persistent buffers and process workgroups concurrently. Fixed-size
`__shared__` arrays are hoisted per workgroup; C++ coroutines preserve each lane's
state at `__syncthreads()`. A barrier resumes only after all lanes reach the same
site. Divergent exits/barriers fail dispatch. Lane frames reuse per-thread arenas
to avoid shared allocator contention. Logical GPU lanes are not OS threads.

Currently supported: scalar and float-vector helpers used by WaterCuda, typed
buffer/scalar parameters, 3D dispatch indices, fixed literal shared arrays and
barriers in entry functions. Atomics, warp intrinsics, dynamic shared memory,
helper barriers and arbitrary record/pointer ABIs need more lowering. Buffer
access bounds remain the trusted kernel author's responsibility. Source rewriting
is deliberately limited to this tested subset.

`npm test` includes emitter contracts. `npm run test:wasm` requires Emscripten and
executes generated shared-array reduction, repeated barriers, isolated workgroups,
real parallel participation and divergence tests. WaterCuda validates all 17
production kernels, a two-axis FFT against an independent native fixture, and
full-renderer WASM/WebGPU image comparisons in its separate experiment:
https://github.com/SamG-Coder/WaterCuda/tree/main/experiments/webshader-wasm

`independent.mjs` and `independent-runtime.js` retain a smaller single-worker
backend for kernels without workgroup cooperation. Its compiler rejects barriers
and shared arrays. Neither backend changes the existing WebGPU default.
