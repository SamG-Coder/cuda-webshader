# CUDA → WebShader

## [Launch the live showcase explorer ↗](https://samg-coder.github.io/cuda-webshader/)

**[NVIDIA sine-wave showcase](https://samg-coder.github.io/cuda-webshader/showcases/simplegl/)** · **[Particle simulation and kernel lab](https://samg-coder.github.io/cuda-webshader/lab.html)** · **[CUDA vs. WebGPU results](https://samg-coder.github.io/cuda-webshader/reports/performance-comparison.html)**

Run the demos directly in a browser with WebGPU support. No local installation or CUDA Toolkit is needed for the browser showcases.

A bounded **CUDA C kernel → typed AST → WGSL → WebGPU** pipeline, with a GPU-resident Three.js particle renderer, ten kernel examples, numerical tests and a device-local performance tuner.

This is source translation. **It does not run CUDA binaries, PTX, the CUDA driver, cuBLAS or arbitrary C++ in a browser.** It deliberately implements a useful kernel-language subset rather than pretending a few string replacements constitute a CUDA compiler.

## Validated on an RTX 5080

The project is running locally with installed, locked dependencies. **144 Node tests, 79 real WebGPU tests (including Three.js rendered-pixel interop), five native CUDA edge-case checks, and all 20 matched CUDA/WebGPU benchmark cases passed.** The static build also succeeds.

Open [the measured comparison](reports/performance-comparison.html) or read [the full methodology and results](reports/performance-comparison.md). All ten kernel sources are compiled by NVCC and translated to WGSL at two workload sizes. Raw GPU timestamps, CUDA event timings, ordinary CUDA launch timings, and verification logs are in `reports/`.

Measured hardware: NVIDIA GeForce RTX 5080, driver 616.64, CUDA Toolkit 13.3, Microsoft Edge 152, Windows WDDM. Results apply to these specific kernels and workloads, not all CUDA code or optimized CUDA libraries. See [VALIDATION.md](VALIDATION.md).
## Run

Try the [NVIDIA simpleGL showcase](showcases/simplegl/): an unchanged, BSD-3-Clause-licensed NVIDIA CUDA vertex kernel translated by this compiler and displayed through WebGPU/Three.js. It passed 12 native CUDA and 12 browser GPU checks. See [provenance and instructions](showcases/simplegl/README.md).

Install Node.js 20 or newer. Extract the project, then:

```sh
npm install
npm start
```

Open **http://localhost:5173** in a browser with working WebGPU. Do not double-click `index.html`; modules, workers and WebGPU need a proper local/secure origin. No CUDA Toolkit is required for the browser application.

The main page is a searchable showcase explorer. Open **Kernel lab** (lab.html) for the original four-area workbench:

- **Live system:** 131,072 particles by default; selectable up to 524,288 in the UI. Positions and velocities stay on the GPU. Orbit, zoom, pause and reset.
- **Kernel lab:** edit/load `.cu`, change the launch shape, compile in a worker, inspect/export WGSL, and validate with the browser's real shader compiler. Compatible particle kernels can replace the live update kernel.
- **Correctness:** run the actual GPU suite, including a render-target pixel test of compute-to-Three.js buffer sharing. Results begin at **NOT RUN**.
- **Performance:** numerically verify each variant, warm it up, measure it, rank variants and export raw JSON. No invented benchmark values are preloaded.

Three.js is pinned to **0.186.0** because the buffer bridge uses a small, isolated backend-internal API. Dependencies are pinned and package-lock.json is included; use npm ci for a reproducible installation.

## The pipeline

```text
CUDA C kernel source (.cu)
    ↓ tokenizer + parser, inside a Web Worker
Typed AST + semantic checks
    ↓ structured WGSL emitter + explicit buffer/scalar ABI
WGSL compute shader
    ↓ browser shader validation + cached WebGPU compute pipeline
GPUBuffer storage / GPU-only dispatch chains
    ↓ same GPUDevice, same GPUBuffer — no CPU round trip
Three.js WebGPURenderer + TSL material
```

The compiler translates your input source; it does not recognize only the example kernel names and swap in handwritten shaders. Pre-generated `.wgsl` and ABI `.json` files are included for all ten examples.

## Kernels and the behavior they exercise

| Kernel | Main test / performance purpose |
|---|---|
| `saxpy` | Fused `y = a*x + y`, contiguous scalar loads, bounds checks |
| `saxpy_vec4` | Four values per invocation; aligned float4 records |
| `matmul_naive` | Row-major baseline and arbitrary M/N/K |
| `matmul_tiled` | 16×16 shared tiles, zero-filled tails, barriers |
| `matmul_register` | 2×2 outputs per lane, 16×16 shared tiles, fewer invocations |
| `reduce_sum` | Two loads per lane; tree reduction; hierarchical GPU-only stages |
| `convolution` | Five-tap convolution; shared halo; zero-padded edges |
| `histogram` | Shared integer atomics; global merge; output clearing |
| `transpose` | Rectangular matrices; 32×33 padded shared tile |
| `particles` | Device helper, f32 math, in-place float4 state, render interop |

Numerical fixtures deliberately include empty/singleton inputs, nonmultiples of workgroup/tile sizes, rectangular matrices, zero K, signed values, and sentinel guards beyond valid output ranges. The 74 shared numerical fixtures also run through the CPU AST oracle. The remaining browser tests cover uniform snapshots, reusable multi-stage reduction, two workgroup specializations and a Three.js pixel-level integration test.

## Performance decisions

**Keep data resident.** The particle buffers are allocated by Three.js, borrowed by the compute runtime, and read directly by its rendering material. There is no per-frame position download and no CPU upload of a newly computed particle array. The CPU seed arrays remain allocated; they are not an authoritative copy after compute starts.

**Do not synchronize the normal frame loop.** Compute and rendering are submitted to the same device queue. Queue ordering provides the producer/consumer sequence. Readback and `onSubmittedWorkDone()` are reserved for explicit tests, timing, reset and teardown.

**Avoid allocation and compilation inside repeated work.** Compute pipelines and bind groups are cached, parameter storage is reused, and reduction plans preallocate every scratch level. Each frame still records command buffers and small JavaScript objects; this is not a claim of zero CPU overhead.

**Batch safely.** A batch snapshots scalar parameters into aligned dynamic-uniform slots. Changing an invocation's values after one dispatch does not overwrite the earlier dispatch's parameters. Stable parameters reuse a slot. The uniform arena is uploaded once immediately before its batch submission; queue ordering keeps reuse safe across successive submissions. An arena overflow throws instead of silently overwriting earlier values.

**Compare optimizations, do not assume them.** The tuner measures scalar/float4 SAXPY at 64/128/256 lanes and baseline/shared/register matrix multiplication. Workgroup limits are checked. A shared tile or float4 path may lose on a particular device or workload; the tuner keeps that result rather than asserting a universal winner.

**Limit rendering overhead.** The demo uses batched sprites, no shadow maps, no bloom, no MSAA, a capped device-pixel ratio and a compact material. At high particle counts, transparency and fill-rate can dominate even when the compute kernel is fast. FPS is frame cadence, not compute time.

## Minimal programmatic use

Run this as a browser module from the served project:

```js
import {GpuRuntime} from './src/runtime/runtime.js';
import {loadKernelSources} from './src/kernels.js';

const runtime = await GpuRuntime.create();
const sources = await loadKernelSources();
const x = runtime.createBuffer(new Float32Array([1, 2, 3, 4]));
const y = runtime.createBuffer(new Float32Array([10, 20, 30, 40]));

try {
  const kernel = await runtime.kernel(sources.saxpy, {
    entry: 'saxpy', workgroupSize: [128, 1, 1]
  });
  const invocation = kernel.bind({x, y}, {a: 2, n: 4});
  runtime.batch().dispatch(invocation, [1]).submit();
  console.log(await runtime.read(y)); // expected [12, 24, 36, 48]
} finally {
  runtime.destroyBuffer(x);
  runtime.destroyBuffer(y);
  runtime.dispose();
}
```

`dispatch` takes **workgroup counts**, corresponding to CUDA blocks, not element counts. Kernel workgroup size is compiled into the shader. Callers must supply sufficient buffer lengths and correct dimensional/scalar arguments. Raw kernels are not automatically memory-safe mathematical operations; use the plans for the additional size checks they provide.

### Reuse measured choices and GPU-only plans

```js
import {prepareSaxpy, prepareMatmul, prepareReduction}
  from './src/runtime/operations.js';

// With x, y, n already allocated, and report returned by runBenchmarks(...):
const saxpyPlan = await prepareSaxpy(runtime, sources, {
  x, y, n, a: 0.5, choice: report.choices.saxpy
});
saxpyPlan.encode(runtime.batch()).submit();

// Reuses scratch and persistent bindings across calls; no intermediate readback.
const reduction = await prepareReduction(runtime, sources, {input: y, n});
reduction.encode(runtime.batch()).submit();
const sum = await runtime.read(reduction.output); // only the final scalar
console.log(sum[0]);
reduction.dispose(); // after submitted work is complete when sharing this resource
```

A vec4 SAXPY choice falls back to scalar when the input length or storage ABI is unsuitable. Choices are workload/device-specific: retune after changing hardware or materially changing dimensions. They are not globally optimal settings. Matrix plans accept the chosen variant by name and retain that variant's required launch contract.

## Supported language subset

Supports `__global__ void` kernels; by-value scalar/vector `__device__` helpers; `float`, `int`, `unsigned int`/`uint`; scalar/float2/float4 buffer pointers; float2/3/4 local values and component access; `const` and `__restrict__`; fixed local/shared arrays; `if`, `else`, `for`, `while`, `break`, `continue`, `return`; numeric object-like `#define`; basic casts and arithmetic; CUDA block/thread/grid indices; `__syncthreads()`; integer `atomicAdd/Min/Max/Exch`; and a documented set of math intrinsics.

Floating constants need an `f` suffix, e.g. `0.5f`. Mixed scalar expressions are explicitly typed. Guarded ternary expressions become real branches, not an eager `select()` that could evaluate an unselected buffer access or atomic operation.

**Not supported:** arbitrary host CUDA code, `<<<...>>>` in source, `cudaMalloc`/streams/events APIs, binary/PTX input, headers/includes, templates, classes, structs, C++ STL, general pointers, pointer arithmetic, pointer helper parameters, recursion, function-like macros, dynamic shared memory, CUDA texture/surface APIs, cooperative grid barriers, warp shuffles/votes, inline PTX, tensor cores/WMMA, half/double/64-bit arithmetic, or floating-point atomics. Float3 and bool pointer-buffer ABIs are rejected rather than guessed. General CUDA vector arithmetic is not supplied; use explicit components as the examples do.

Unsupported syntax/types fail explicitly where recognized. The browser's WGSL compiler remains the final validation gate for emitted code, including uniformity and implementation limits. Finite f32 numerical agreement is tested with tolerances; do not assume NVCC bit-for-bit equivalence, identical FMA contraction, denormal handling, NaN behavior or transcendental precision. This is an experimental compiler/runtime, not a production-hardened general CUDA replacement.

## Test, compile and build commands

```sh
npm test                         # No npm dependencies needed: compiler/oracle/mock-host tests
npm run compile                  # Regenerate all WGSL + ABI JSON files
npm run check:cuda-syntax         # Optional: requires g++ (or CXX); syntax only
node scripts/compile.mjs kernels/saxpy.cu --entry saxpy --block 128,1,1 --out generated/saxpy-custom

npx playwright install chromium  # Requires npm install first
npm run test:gpu                 # Real WebGPU tests, fails on unavailable/failed GPU
npm run build                    # Static dist/; requires installed Three.js
node scripts/serve.mjs dist       # Serve the static build
```

For local hardware testing, the **Correctness** tab in your normal browser is the simplest path. Playwright defaults to headless Chromium and writes `reports/gpu-local.json`. Use `CW_CHROMIUM` for an explicit executable path, `CW_HEADED=1` for a headed browser, and `CW_BENCH=1` to include benchmarks. In PowerShell, for example:

```powershell
$env:CW_HEADED = "1"
$env:CW_CHROMIUM = "C:\Program Files\Google\Chrome\Application\chrome.exe"
npm run test:gpu
```

GitHub Actions runs compiler tests and the static build only. Software WebGPU/SwiftShader tests are disabled by project preference. Run GPU tests locally on real hardware.

An optional native CUDA baseline is included in `tests/native-reference.cu`. With an appropriate NVIDIA CUDA Toolkit and GPU, run:

```sh
nvcc -O3 -std=c++17 -arch=native tests/native-reference.cu -o native-reference
```

Then run the resulting executable. It checks SAXPY, three matrix variants and hierarchical reduction against independent host references. This baseline was built and all five checks passed on the RTX 5080. The separate benchmarks/ harness provides matched performance tests for all ten kernels. Run npm run bench:native:build, then reports/native-benchmark.exe; see the measured report for complete reproduction commands.

## Reading benchmark output

`gpuMedianMs` uses WebGPU timestamp queries when supported and nonzero. Otherwise the report clearly falls back to `wallMedianMs`: CPU encode/submit-to-completion time amortized across repeated dispatches. The ranking uses the same metric for every compared variant. JSON retains each sample, workgroup shape, dimensions, warm-up count, repeats, exposed device information and skipped-limit reasons.

Compilation, allocation and initial data upload are excluded. Repeated dispatches deliberately reuse buffers and can benefit from cache residency. `effectiveGBs` is logical bytes moved divided by time, **not measured DRAM bandwidth**. Matrix GFLOP/s uses the conventional `2*M*N*K` operation count. Very short timestamp intervals may be quantized. Do not compare these numbers to native CUDA without matching the GPU, data, precision, workload, reuse, timing boundary and synchronization policy.

## Layout

```text
src/compiler/     Parser, type checks, WGSL emitter, worker and CPU AST oracle
src/runtime/      WebGPU runtime, plans, benchmarks, isolated Three.js buffer bridge
src/demo/         GPU-resident Three.js particle scene
kernels/          Ten original CUDA C sample kernels
generated/        Emitted WGSL and ABI metadata
scripts/          Dependency-light server, compiler CLI, build and test runners
tests/            CPU fixtures, host tests, GPU suite, pixel interop, native baseline
reports/          Actual authoring logs and explicit validation status
docs/             Architecture and primary-source research notes
```

## License

The original project source, CUDA kernel examples, generated WGSL, tests, benchmark harness and documentation are available under the [MIT License](LICENSE), Copyright (c) 2026 SamG-Coder and CUDA WebShader contributors. Commercial use, modification and redistribution are permitted subject to preserving the license and copyright notice.

Dependencies retain their own licenses: Three.js is MIT, while Playwright and playwright-core are Apache-2.0 development dependencies. The CUDA Toolkit and browser binaries are external prerequisites and are not included in this source repository. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for license scope and attribution, and [docs/research.md](docs/research.md) for design references.

The static build includes the project license, third-party notices and Three.js license. Native executables and large benchmark input binaries are excluded from Git; `npm run bench:native:build` regenerates them locally. Measured reports, raw timing samples and the generated native launcher source are included.
