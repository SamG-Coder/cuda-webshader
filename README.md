# CUDA WebShader

CUDA WebShader translates device kernels from a defined subset of CUDA C into WGSL and runs them with WebGPU. JavaScript supplies the host side: which kernel to launch, the block size, the grid, and the buffers.

The public showcase explorer is at [samg-coder.github.io/cuda-webshader](https://samg-coder.github.io/cuda-webshader/). Use the local server below when you are changing source or running your own kernels. WebGPU requires localhost or HTTPS, so open the pages through that server.

## Requirements

- Node.js 20 or newer
- A current browser with WebGPU enabled (Chrome, Edge, or another Chromium browser on a machine with a working GPU)
- npm, for the locked dependencies in `package-lock.json`

The browser application does not need the CUDA Toolkit. Native CUDA checks and the optional performance comparison do. Three.js is pinned to 0.186.0 because the compute-to-render bridge uses a private buffer API in that release.

## Run the application

```sh
npm ci
npm start
```

Then open [http://localhost:5173](http://localhost:5173). `npm start` serves the repository on `127.0.0.1:5173`. Override the address with `HOST` and `PORT` when you need to.

Stop the server with Ctrl+C.

### Showcase explorer

The home page lists the runnable demos. Search or filter the cards, then open one. Each card launches the sandbox with that sample already loaded. Cards added from the NVIDIA catalog are kernels that have a recorded passing browser check.

Every sample under `showcases/` has its own README for source, launch settings, and what was compared with native CUDA. Start with these:

| Sample | What you see |
|---|---|
| [Orbital particles](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=particles) | A GPU-resident particle field rendered by Three.js from the same buffer the kernel writes |
| [Path tracer](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=pathtracer) | Roger Allen’s glass, metal, and sphere scene, compiled into several compute passes |
| [Recursive quadtree](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=quadtree) | An unchanged NVIDIA kernel that builds the tree with GPU-scheduled child launches |
| [SAXPY](http://localhost:5173/sandbox.html?example=saxpy) | A numeric kernel. Useful as the smallest check that compilation and readback work |

The Chrono dam-break sample is an experimental port of Project Chrono’s SPH kernels. It runs, and it is slower than real time. Its notes are in [reports/chrono-sph-progress.md](reports/chrono-sph-progress.md).

### Sandbox

Open [sandbox.html](http://localhost:5173/sandbox.html) to compile and run a kernel.

1. Choose an example, paste CUDA into the editor, or drop a `.cu` file onto the page. **Open .cu** does the same thing.
2. Set **Entry** when the file contains more than one `__global__` function. A template kernel uses the specialized name, such as `MatrixMulCUDA<16>`.
3. Set **Threads per block**. That triple is the CUDA block size and the WebGPU workgroup size. It is compiled into the shader, so changing it requires another run.
4. Set **Dynamic shared bytes** only for an `extern __shared__` array. Changing it creates a new shader.
5. Open **Launch settings & buffer inputs** for the grid, scalar arguments, buffer sizes, and fill patterns. **Detect from source** fills in a starting configuration. Those values are defaults, so check the counts and shapes against your kernel.
6. Press **Compile & run**, or use Ctrl+Enter (Cmd+Enter on macOS).

The **CUDA .cu**, **Generated WGSL**, and **Compare** tabs show the input and the shader the browser compiled. The pass selector appears when a sample produces more than one shader. The log records compilation, allocation, dispatch, and readback. Float4 results are drawn as 3D points from the GPU buffer. Scalar results appear as a table and a heatmap. **Animate GPU steps** repeats a simulation plan; **Stop** ends that loop.

**Save .cu** and **Save .wgsl** download the current editor contents. Source stays in the browser. The Monaco editor is served with the application.

A showcase with several kernels is driven by a `pipeline.json` plan: buffers, textures, dispatches, and copies. Loading the example runs that plan. You do not re-enter the original host `.cpp`.

### Kernel lab

[lab.html](http://localhost:5173/lab.html) is the original workbench for the ten kernels in `kernels/`.

- **Live system** runs the particle field. The default count is 131,072, and the selector goes up to 524,288. Orbit, pause, and reset from the page. Positions stay on the GPU.
- **Kernel lab** edits CUDA, chooses the block size, compiles in a worker, and shows the generated WGSL. A compatible particle kernel can replace the live update.
- **Correctness** runs the browser GPU suite in the page you already have open. The panel starts at **NOT RUN**.
- **Performance** checks each bundled variant, measures it on your GPU, and exports the raw samples as JSON.

## Compile a kernel

From the repository:

```sh
node scripts/compile.mjs kernels/saxpy.cu --entry saxpy --block 128,1,1 --out generated/saxpy
```

This writes `generated/saxpy.wgsl` and `generated/saxpy.json`. The JSON file is the ABI: entry point, bindings, strides, uniform offsets, and workgroup size. `--entry` is required when the file has more than one kernel. `--block` is `x,y,z` and defaults to `128,1,1`.

Regenerate the ten bundled examples with:

```sh
npm run compile
```

In Node, without writing files:

```js
import { readFile } from 'node:fs/promises';
import { compile } from './src/compiler/compiler.js';

const source = await readFile('kernels/saxpy.cu', 'utf8');
const artifact = compile(source, {
  entry: 'saxpy',
  workgroupSize: [128, 1, 1]
});

console.log(artifact.entryPoint, artifact.metadata.bindings);
```

`compile` returns the WGSL text and the metadata the runtime needs. `serializableArtifact` in the same module strips that result down to the JSON form.

Compilation options that matter:

| Option | Meaning |
|---|---|
| `entry` | `__global__` function name. Use `kernel<16>` or `kernel<float>` for a supported template specialization. |
| `workgroupSize` | CUDA block dimensions, `[x, y, z]`. |
| `sharedMemoryBytes` | Byte size of the single dynamic `__shared__` allocation. |
| `optimize` | `'dependencies'` (the default) removes unused helper functions after a full validation. `'specialize'` also removes proven integer and bool branches, then falls back to dependency trimming if the ABI would change. `false` keeps the untrimmed shader. |

The default optimization changes the generated text. A cache keyed only by CUDA source also needs the compiler revision (`COMPILER_VERSION` in `src/compiler/compiler.js`) and the `optimize` value. Details are in [docs/dependency-optimization.md](docs/dependency-optimization.md).

## Run a kernel in the browser

### Streaming least squares

`src/runtime/least-squares.js` supplies `NormalEquations` for applications such as
texture fitting that generate many observation rows but solve a small dense
system. Each GPU row contains `[features..., targets...]`; include a constant
feature when the model needs a bias. Append batches from a reusable GPU buffer,
then call `solve()` to obtain output-major coefficients. Products and partial
sums use f32 on the GPU, and the final reduction and pivoted dense solve use f64
on the host. This is not a native f64 or QR solver; normal equations can lose
accuracy for ill-conditioned inputs. The default relative ridge is `1e-6`.

```js
import {NormalEquations} from './src/runtime/least-squares.js';

const fit = await NormalEquations.create(runtime, 3, 2);
try {
  // observationBuffer has rows [x0, x1, 1, target0, target1].
  fit.append(observationBuffer, rowCount);
  // More appends may follow, including GPU-generated rows in the same buffer.
  const weights = await fit.solve(); // 2 output rows of 3 coefficients
} finally {
  await runtime.idle();
  fit.dispose();
}
```

Both feature and output counts are bounded to 1..64. `append` supports `stride`
and `offset` in float elements, validates buffer ownership and bounds, and
submits work immediately. Queue order protects a reused observation buffer.
Singular systems without a ridge, empty fits and non-finite solved equations
are rejected. Test the numerical GPU path with
`node scripts/test-least-squares.mjs` (installed Edge on Windows; Playwright
Chromium elsewhere; override with `CW_BROWSER`).

### General kernel launch

Serve the repository and run this as a module on that origin. `GpuRuntime.create()` requests a WebGPU device.

```js
import { GpuRuntime } from './src/runtime/runtime.js';

const source = `
__global__ void saxpy(const float* __restrict__ x, float* __restrict__ y,
                      float a, unsigned int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = fmaf(a, x[i], y[i]);
}`;

const runtime = await GpuRuntime.create();
const x = runtime.createBuffer(new Float32Array([1, 2, 3, 4]));
const y = runtime.createBuffer(new Float32Array([10, 20, 30, 40]));

try {
  const kernel = await runtime.kernel(source, {
    entry: 'saxpy',
    workgroupSize: [128, 1, 1]
  });
  const invocation = kernel.bind({ x, y }, { a: 2, n: 4 });
  runtime.batch().dispatch(invocation, [1, 1, 1]).submit();
  console.log(await runtime.read(y)); // [12, 24, 36, 48]
} finally {
  runtime.destroyBuffer(x);
  runtime.destroyBuffer(y);
  runtime.dispose();
}
```

`runtime.kernel` compiles the source, asks the browser to validate the WGSL, and caches the compute pipeline. `bind` attaches buffers by parameter name and packs scalar arguments into a uniform snapshot. `dispatch` takes **block counts**, the CUDA grid, not thread counts. One block of 128 threads covers the four-element example because the kernel checks `i < n`.

`createBuffer` owns the allocation and `dispose` destroys owned resources. `importBuffer` borrows a `GPUBuffer` that already belongs to the same device; the runtime does not destroy it. Call `await runtime.idle()` before replacing or destroying buffers that queued work still uses.

For the bundled SAXPY, matrix, and reduction kernels, `src/runtime/operations.js` adds shape checks and reusable launch plans:

```js
import { prepareSaxpy, prepareReduction } from './src/runtime/operations.js';
import { loadKernelSources } from './src/kernels.js';

const sources = await loadKernelSources();
const saxpyPlan = await prepareSaxpy(runtime, sources, { x, y, n: 4, a: 2 });
saxpyPlan.encode(runtime.batch()).submit();

const reduction = await prepareReduction(runtime, sources, { input: y, n: 4 });
reduction.encode(runtime.batch()).submit();
const sum = await runtime.read(reduction.output);
reduction.dispose();
```

`loadKernelSources` uses `fetch`, so it belongs in the served page. A measured tuner choice can be passed as `choice`; an unsuitable float4 SAXPY choice falls back to the scalar kernel.

Readback is for tests and inspection. The particle frame does not download positions. It records the dispatch and lets Three.js read the same storage buffer.

## What a kernel can use

A compilation selects one `__global__` kernel and the device helpers it can reach. Pointer parameters become storage buffers. Scalar arguments become a uniform struct. `__shared__` memory becomes workgroup memory. `threadIdx`, `blockIdx`, `blockDim`, and `gridDim` map to the WebGPU invocation and the declared block size. `__syncthreads()` is a workgroup barrier. A reduction across blocks is a sequence of dispatches, each reading the previous dispatch’s buffer.

The frontend rejects programs it cannot lower without changing their meaning. Packed `float3` pointer parameters are rejected because CUDA and WGSL would use different strides. Writable use of the same buffer through two bindings is rejected. A barrier the workgroup cannot execute uniformly is left for the browser validator to reject.

The accepted subset also covers the features used by the showcases: limited kernel and helper templates, textures and surfaces, integer atomics, dynamic shared memory, and GPU-scheduled child launches for the samples written that way. [docs/architecture.md](docs/architecture.md) is the contract for the ABI, barriers, and resource ownership. A showcase README is the contract for that sample.

## Tests and builds

```sh
npm test
npm run compile
npm run build
node scripts/serve.mjs dist
```

`npm test` runs the Node parser, CPU-oracle, and host tests. It does not launch a browser. `npm run build` writes a static site to `dist/`.

Browser tests need Playwright’s Chromium and a WebGPU adapter:

```sh
npx playwright install chromium
npm run test:gpu
```

On Windows, point the runner at an installed browser when you want a visible window:

```powershell
$env:CW_HEADED = "1"
$env:CW_CHROMIUM = "C:\Program Files\Google\Chrome\Application\chrome.exe"
npm run test:gpu
```

`CW_BENCH=1` includes the benchmark cases. Results are written to `reports/gpu-local.json`. The **Correctness** tab in the kernel lab runs the same suite in your normal browser.

Optional native checks need the CUDA Toolkit:

```sh
nvcc -O3 -std=c++17 -arch=native tests/native-reference.cu -o native-reference
npm run bench:native:build
```

`npm run check:cuda-syntax` only checks that sample sources still parse. It needs `g++` or `CXX`. Syntax success does not establish CUDA runtime behavior.

Measured runs, hardware, and methodology are recorded in [VALIDATION.md](VALIDATION.md) and [reports/performance-comparison.md](reports/performance-comparison.md). Those documents describe the runs that produced them. Later showcase checks live with each showcase and in `reports/`.

## Repository layout

```text
src/compiler/     CUDA frontend, WGSL emitter, worker, CPU oracle
src/runtime/      WebGPU runtime, launch plans, Three.js buffer bridge
src/sandbox/      Sandbox editor, pipeline runner, previews
src/demo/         Particle scene used by the kernel lab
kernels/          Ten original sample kernels
showcases/        Sample sources, pipeline plans, inputs, and notes
generated/        WGSL and ABI JSON for the ten sample kernels
scripts/          Static server, compiler CLI, tests, and benchmarks
tests/            Node tests, browser suite, native baseline
docs/             Architecture and design notes
reports/          Recorded checks and timings
```

## License

Original project source, sample kernels written for this project, generated WGSL, tests, and documentation are under the [MIT License](LICENSE), Copyright (c) 2026 SamG-Coder and CUDA WebShader contributors.

NVIDIA sample kernels and their data are BSD-3-Clause. Three.js and Monaco Editor are MIT. Playwright is Apache-2.0. Attributions and the Three.js pin are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Native half arithmetic

Local, shared, and storage-pointer `__half` / `__half2` values compile to WGSL `f16` / `vec2<f16>`. Storage buffers use native binary16 values with two-byte scalar or four-byte paired records; reads and writes require no packed-integer conversion in the shader. Upload raw binary16 bytes, for example through `Uint16Array` or packed `Uint32Array`, and use the same raw representation with the CPU oracle. Kernel half parameters passed by value remain unsupported. Supported conversions are `__float2half_rn`, `__half2float`, `__floats2half2_rn`, `__float22half2_rn`, and `__half22float2`; arithmetic intrinsics are `__hadd`, `__hsub`, `__hmul` and their `2` variants. Artifacts declare `shader-f16`; the runtime requests it when supported and rejects half kernels on devices without it. Ordinary kernels remain usable without that feature.

This is a bounded subset: kernel half parameters passed by value are rejected. The unprefixed names `half` and `half2` are not reserved. Shared allocation accounting uses two bytes per half lane. The CPU oracle rounds to binary16, but portable GPU behavior for denormals and exceptional values follows WebGPU; do not assume full CUDA floating-point equivalence. This support does not expose tensor cores.

CUDA `__clz` accepts a 32-bit signed or unsigned integer, counts leading zero bits (32 for zero), and lowers to WGSL `countLeadingZeros` with unsigned bit interpretation. The CPU oracle uses the same semantics.

Set `useAdapterWorkgroupLimits: true` to request the adapter's full `maxComputeWorkgroupStorageSize`. The default leaves the WebGPU default limit unchanged. Callers must still check the returned device limit and provide fallbacks for smaller adapters or supplied devices.

`#pragma unroll` now expands small counted `for` loops in generated WGSL. The bounded implementation supports literal integer start/end/positive increments, up to 32 iterations, and preserves statement order and iteration scope. An explicit factor equal to the trip count also expands; factor one disables this frontend transformation. Dynamic bounds, partial factors, counter aliases/mutation, loop exits, and excessive expansion retain ordinary loop control. This is a CUDA source hint; no generated-shader rewriting is needed. See the [CUDA language reference](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html) for the broader native compiler directive.

Component-wise `make_float2/3/4` expressions over matching local vector components can lower to vector arithmetic. Supported expression trees use `+`, `-`, `*`, `/`, `truncf`, `floorf`, `ceilf`, `fabsf`, `fmaxf`, and `fminf`. Arithmetic association is preserved. Mixed component order, references, storage reads and expressions with side effects retain scalar construction. This supports ordinary CUDA helper functions without requiring nonstandard operators on CUDA vector types.
