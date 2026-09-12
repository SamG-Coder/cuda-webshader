# CUDA → WebShader

**New: [NVIDIA recursive quadtree](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=quadtree)**. The unchanged CUDA kernel builds all 193 nodes for 1,024 points using GPU-scheduled recursive launches. Every node and output point matches native CUDA. [Source and validation](showcases/quadtree/README.md).

**New: [NVIDIA Bezier curves — GPU allocation and child launches](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=bezier)**. All 256 curves run through the original parent and child kernels, with 3,958 vertices compared against native CUDA. [Source and validation](showcases/bezier/README.md).

**New: [Glass, metal and 488 spheres — CUDA path tracer](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=pathtracer)**. Roger Allen’s final scene runs through the compiler and sandbox at 1200 × 800, 10 samples per pixel. [Source, pipeline and validation](showcases/pathtracer/README.md).


## [Launch the live showcase explorer ↗](https://samg-coder.github.io/cuda-webshader/)

**[NVIDIA sine-wave showcase](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=wave)** · **[Particle sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=particles)** · **[CUDA vs. WebGPU results](https://samg-coder.github.io/cuda-webshader/reports/performance-comparison.html)**

## CUDA sandbox

[NVIDIA N-body is now a 3D sandbox showcase](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=nbody):
512 bodies, original force/integration kernels, and GPU position feedback.
Native CUDA and NVIDIA WebGPU pass three-step independent reference checks.
See [launch requirements and validation](showcases/nbody/README.md).
[NVIDIA FDTD3d is also available as a 3D scalar-volume showcase](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=fdtd),
with the original shared-memory stencil, editable constant coefficients and
GPU-only field feedback. See [volume setup and validation](showcases/fdtd3d/README.md).
[NVIDIA bicubic filtering](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=bicubic)
provides all five original filter modes, with editable zoom and pan. All 15 tested images
match native CUDA and an independent reference within one colour level.
See [filter modes and validation](showcases/bicubic-texture/README.md).
[NVIDIA texture convolution](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=convolution)
runs the original 17-tap row and column filters, with pixel-coordinate texture
sampling and an intermediate GPU image copy. See [setup and native comparisons](showcases/convolution-texture/README.md).
[NVIDIA surface write](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=surface)
runs the original surface-write and texture-rotation kernels as two GPU passes.
See [surface support and validation](showcases/surface-write/README.md).
[NVIDIA texture rotation](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=texture2d)
runs the original teapot-image rotation kernel with a float 2D GPU texture.
See [sampling settings and native comparisons](showcases/texture2d/README.md).
[NVIDIA volume renderer](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=volume)
runs the complete original ray-marching device code with the Bucky volume and colour transfer table.
Three rendered images are compared with native CUDA, including a rotated camera and partial blocks.
See [volume renderer setup and validation](showcases/volume-render/README.md).
[NVIDIA 3D texture slice](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=texture3d)
samples the original Bucky volume using a real 3D texture and sampler.
See [texture setup and validation](showcases/texture3d/README.md).
[NVIDIA recursive Gaussian](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=gaussian)
runs the original four-pass colour-image filter with an RGBA preview.
See [Gaussian setup and validation](showcases/recursive-gaussian/README.md).
[NVIDIA Haar wavelet](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=haar)
runs a complete 4,096-value transform with two kernel launches and GPU coefficient copies.
See [Haar setup and validation](showcases/haar/README.md).
[NVIDIA separable convolution](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=separable)
runs the original row and column filters as a GPU pass sequence, with both
generated shaders available for comparison. See [pass setup and validation](showcases/convolution-separable/README.md).
Device helper integer arguments also support bounded signed arithmetic such as
`filter<i - 1>` and negative terminating specializations. Native CUDA and
NVIDIA WebGPU verify the expansion, including INT_MIN. These features now power the
[verified texture-convolution showcase](showcases/convolution-texture/README.md),
including texture handles in nested helpers and the original IMAD macro.
Device helpers now
support up to four explicit built-in type arguments or one integer template argument, including nested
calls and arguments forwarded from a kernel template. Single type arguments can be deduced;
multiple types currently require explicit arguments. This is a verified prerequisite for
the [bicubic texture showcase](showcases/bicubic-texture/README.md).
Device helpers also accept trailing scalar literal defaults, including defaults
inherited from primary templates; native CUDA and WebGPU verify omitted and explicit
arguments, overloads and nested texture helpers.
Packed `uchar4` buffers now retain CUDA’s four-byte record layout. Byte components
can be read and edited in local records, with integer promotion and narrowing verified
against native CUDA. All four original bicubic render shaders now run in the sandbox; all five sampling
modes pass complete native-image comparisons. Type-trait structs containing typedef members, including explicit type specializations
and dependent types such as `typename vec4<T>::Type`, now resolve to built-in
value types. Helper type arguments can now be deduced from direct scalar/vector
parameters, and explicit device-function specializations are selected when present.
The original N-body `rsqrt_T` float specialization passes the regression fixture.
Scalar `__constant__` globals now become read-only uniforms, with values supplied
through keys such as `constant.softeningSquared` in the sandbox’s scalar settings.
Omitted values use the declaration’s numeric initializer or zero. NVIDIA’s original
`bodyBodyInteraction` helper passes force checks at three softening values.
Device helpers now support thread/block indices, block barriers, static shared
arrays, one explicitly sized dynamic shared array, and by-value `thread_block`
handles. Storage-buffer pointer parameters now support nested calls, offsets and
type deduction. Local vector brace initializers support full, partial and empty
lists, including dependent vector types; omitted components become zero. Components
must match the element type or use explicit casts. Stateless shared-memory
conversion wrappers now work with an explicitly sized dynamic shared allocation.
N-body uses an explicit whole-block contract to make its early-return guard
uniform; the runtime rejects partial-block counts. The unchecked original guard
still fails WebGPU validation. One-dimensional constant arrays of up to 256
scalar elements are supported, with dynamic indexing and per-dispatch uniforms.
The helper support and NVIDIA’s original vector-trait declarations pass native
CUDA and NVIDIA WebGPU tests for 1, 129 and 1,025 records. Coverage includes
unsigned wraparound, reference mutation, vector position updates and output guards.

**[Bitonic key/value merge](https://samg-coder.github.io/cuda-webshader/sandbox.html?nvidia=37)**
runs NVIDIA’s original global merge kernel and reference-parameter comparator.
The compiler now promotes mixed Boolean/numeric scalar operands as C++ does.
The sandbox shows one merge stage. Separate native CUDA and hardware WebGPU
checks also run complete sorting networks using repeated global-kernel dispatches
and separate source/destination buffers, checking both keys and value permutations.
This does not import the original shared-memory optimized sorting pipeline.

**[Driver/runtime vector addition](https://samg-coder.github.io/cuda-webshader/sandbox.html?nvidia=36)**
compiles the complete original NVIDIA kernel file, including its `extern "C"`
declaration. Single-function C linkage is now accepted for kernels and helpers;
linkage blocks, templates with C linkage and other linkage languages are rejected.
Seven input sizes, including empty input and partial blocks, pass exact-output
checks in native CUDA and hardware WebGPU. The preview covers the compute kernel;
native driver/runtime API interoperability is outside the browser’s scope.

**[Naïve matrix transpose](https://samg-coder.github.io/cuda-webshader/sandbox.html?nvidia=35)**
joins the tiled and padded transpose examples. All three original kernels pass
exact-output checks on four square and rectangular matrix sizes in native CUDA
and hardware WebGPU. Dimensions must be multiples of 32. This validates
correctness; these checks do not measure their performance difference.

**[Square roots · MPI compute stage](https://samg-coder.github.io/cuda-webshader/sandbox.html?nvidia=34)**
runs NVIDIA’s unchanged simpleMPI kernel. The compiler now accepts the float
overload of `sqrt`. Four launch sizes passed native CUDA and real NVIDIA WebGPU
checks, including the smallest normal and largest finite float, with a 0.0000002
relative tolerance. The launch must match the input length because the original
kernel has no bounds check. This is the isolated compute stage; MPI communication
is not implemented in the browser. See [square-root results](reports/nvidia-mpi-sqrt.json).

**[Inverse normal distribution](https://samg-coder.github.io/cuda-webshader/sandbox.html?nvidia=33)**
runs NVIDIA's unchanged `inverseCNDKernel` and `MoroInvCNDgpu` helper with a supplied
unsigned input buffer. Mutable by-value helper parameters now use local copies;
scalar `static_cast` and static kernel declarations are supported. `UL`/`LU`
literals are accepted within the unsigned 32-bit range and use `u32` semantics;
64-bit arithmetic remains unsupported. Storage-buffer conditions are true under
the runtime's required non-null binding contract. Missing/null buffers are
rejected, so this sample's null-input uniform-generator mode is unavailable.
Five sizes passed native CUDA and hardware WebGPU checks against independent
normal-tail references, with a 0.00002 absolute tolerance. See
[inverse-normal results](reports/nvidia-inverse-cnd.json).

NVIDIA's **[64-bin histogram merge](https://samg-coder.github.io/cuda-webshader/sandbox.html?nvidia=31)**
and **[256-bin histogram merge](https://samg-coder.github.io/cuda-webshader/sandbox.html?nvidia=32)**
now run unchanged using the compiler's existing shared-memory and block-barrier
support. Each combines supplied partial histograms into one result. Both passed
native CUDA and hardware WebGPU checks for 0, 1, 17, 255, 256 and 513 partial
histograms, including unsigned wraparound and guard values. The original byte-
counting stages still need unsupported byte storage and/or shared-pointer helper
features; these cards demonstrate the merge stages only.
[Histogram merge results](reports/nvidia-histogram-merge.json).

**[Walsh transform — shared memory](https://samg-coder.github.io/cuda-webshader/sandbox.html?nvidia=30)**
runs NVIDIA's unchanged `fwtBatch1Kernel`, including `extern __shared__ float
s_data[]`. Set **Dynamic shared bytes** in the sandbox, or pass
`sharedMemoryBytes` to `compile` / `runtime.kernel`; changing it creates a new
shader specialization. One unsized dynamic shared array is supported, alongside
fixed shared arrays, and their combined allocation is checked against the device
limit. Six sizes from 4 to 2,048 values passed native CUDA and real WebGPU checks
against the direct Walsh matrix, including odd powers of two and guard values.
The preset runs three complete 256-value transforms with 1,024 shared bytes and
64 threads per block. [Shared-memory results](reports/nvidia-fwt-shared.json).

**[Walsh transform — global pass](https://samg-coder.github.io/cuda-webshader/sandbox.html?nvidia=29)**
runs NVIDIA's unchanged `fwtBatch2Kernel` with separate input/output buffers and
multiple batches. The compiler now accepts local aliases of storage buffers,
including `float *p = input + offset` and alias chains. Offsets are captured at
declaration; reads, writes and atomics retain the original buffer's binding and
const rules. Helpers now accept storage-buffer pointers and offsets. Pointer
`+=`/`-=` updates move storage-pointer offsets; general reassignment, pointer casts
and shared/local-array pointers remain unsupported. All dereferences must stay within the
original allocation. This is one radix-4 transform pass; the shared-memory finishing kernel has its
own preset, and full dyadic convolution is not implemented in this preview.
See [native/WebGPU pass checks](reports/nvidia-fwt-pass.json).

NVIDIA's `alignedTypes` copy template now accepts explicit built-in type
specializations: **[int](https://samg-coder.github.io/cuda-webshader/sandbox.html?nvidia=26)**,
**[uint4](https://samg-coder.github.io/cuda-webshader/sandbox.html?nvidia=27)** and
**[float4](https://samg-coder.github.io/cuda-webshader/sandbox.html?nvidia=28)**.
The compiler supports one `template<class T>` or `template<typename T>` parameter
on a kernel, selected through an entry such as `testKernel<uint4>`. Parameters,
local declarations and casts use the selected built-in type. Custom structs,
template defaults, multiple parameters and templated helpers remain unsupported.
These are new built-in specializations of the unchanged copy kernel, not the
upstream custom-struct alignment benchmark. Native CUDA and real WebGPU passed
12 byte-for-byte copy checks, including signed zero and untouched guards.
See [typed copy results](reports/nvidia-aligned-copy.json).

**[Atomic compare-and-swap counters](https://samg-coder.github.io/cuda-webshader/sandbox.html?nvidia=25)**
runs NVIDIA's unchanged `cas_atomic` retry loop. The compiler now accepts
`do…while` and 32-bit signed/unsigned `atomicCAS` on storage elements or shared
scalars. The generated WGSL retries weak compare-exchange when it fails
spuriously, preserving CUDA's strong compare-and-swap behavior and returning the
observed old value. Native CUDA and hardware WebGPU checks include one million
threads contending over ten counters, with guard values checked. The other
CCCL `atomic_ref` versions in this upstream sample remain unsupported.
See [CAS correctness results](reports/nvidia-atomic-cas.json).

**[Scan block-offset update](https://samg-coder.github.io/cuda-webshader/sandbox.html?nvidia=24)**
runs NVIDIA's unchanged `uniformUpdate` kernel. It adds one shared block offset
to four unsigned integers per thread. This is the final update stage, not the
complete scan pipeline: the earlier scan helpers still require unsupported
shared-memory pointer parameters and barriers in helpers. The compiler now
supports `int2/3/4` and `uint2/3/4` local values, their `make_*` constructors,
two/four-component integer buffer layouts, and shared scalars (including integer
atomics). Three-component buffer pointers remain rejected because CUDA and WGSL
layouts differ. Integer components preserve all 32 bits through helpers, stores
and readback. [Native and hardware checks](reports/nvidia-scan-update.json) include
overflow and values beyond float precision.

NVIDIA's original tiled matrix multiplication kernel now runs as two explicit
integer template specializations:
**[16×16 tiles](https://samg-coder.github.io/cuda-webshader/sandbox.html?nvidia=22)**
and **[32×32 tiles](https://samg-coder.github.io/cuda-webshader/sandbox.html?nvidia=23)**.
Enter `MatrixMulCUDA<16>` or `MatrixMulCUDA<32>` in the sandbox entry field.
The compiler accepts one `template <int NAME>` parameter on a kernel, an explicit
nonnegative 32-bit integer argument, comma-separated `for` step updates, and
`#pragma unroll` hints. The hint leaves optimization to the WGSL backend.
Template defaults, multiple template parameters and helper
templates remain unsupported. For this NVIDIA kernel, launch blocks must match
the selected square tile and all matrix dimensions must be positive multiples
of it; the original kernel has no boundary guards. Both specializations passed
native CUDA and hardware WebGPU checks for square and rectangular matrices.
See [matrix test results](reports/nvidia-matrixmul.json).

**[Black–Scholes sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?nvidia=21)**
now compiles NVIDIA's complete `BlackScholes_kernel.cuh` unchanged, including its
two device helpers. The compiler supports scalar output references in helpers,
comma-separated local declarations, `__restrict`, and single-argument
`__launch_bounds__` (enforced as a maximum thread count). Reference arguments must
be distinct mutable named local scalars of the exact type; storage elements,
vector components, reference returns and general C++ references remain unsupported.
`__fdividef`, `__expf` and `__logf` map to WGSL division, `exp` and `log`;
they do not promise CUDA fast-math bit equivalence. Both output buffers passed
native CUDA and real NVIDIA WebGPU checks at six input sizes, including empty,
partial-block and odd counts. The original kernel processes pairs, so an odd final
option stays untouched. See [hardware results](reports/nvidia-blackscholes.json)
and [native results](reports/nvidia-blackscholes-native.txt).

NVIDIA's batched scalar-product kernel also runs unchanged. It uses the new
`__mul24` / `__umul24` intrinsics and a direct forwarding macro such as
`#define IMUL(a,b) __mul24(a,b)`. Forwarding macros must pass every argument
once, in order, to a named function; general macro substitution is unsupported.
The integer intrinsics retain the low 24 input bits (sign-extending for
`__mul24`) and the low 32 product bits, including overflow. See
[NVIDIA's intrinsic definitions](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html)
and [the native/WebGPU scalar-product checks](reports/nvidia-scalar.json).

NVIDIA's coalesced and padded matrix-transpose kernels now run unchanged in the
sandbox. The compiler accepts `cooperative_groups::thread_block`, explicit
namespace aliases such as `namespace cg = cooperative_groups;`, and
`cg::sync(block)` / `block.sync()` for a local `this_thread_block()` handle.
These become WGSL workgroup barriers, with a storage barrier when needed.
Handles cannot be used as numbers, passed through helpers, or substituted with
grid/tiled groups. Divergent barriers remain subject to WebGPU validation.
The original 32×16 thread launch requires an adapter supporting 512 invocations
per workgroup. See [transpose validation](reports/nvidia-transpose.json).

The main showcase grid contains individual runnable samples; every card opens
the sandbox directly. The NVIDIA audit covers 208 upstream sample directories,
350 compiler entry probes, and 38 kernel entries/specializations checked with both native CUDA and
real NVIDIA WebGPU. See [the audit and remaining blockers](reports/nvidia-audit.md)
and [reproduction instructions](showcases/nvidia/README.md). These kernel checks
are separate from full native application execution and from performance tests.

Open `sandbox.html` from the local server, or use the **Sandbox** navigation link.
The left pane uses Monaco (the editor behind VS Code) with C++ syntax highlighting,
bracket matching, find/replace, and compiler error markers. Paste source or drop a
`.cu` file to compile and run; use Ctrl/Cmd+Enter to rerun edits. The right pane
shows GPU output and timestamped compilation, allocation, dispatch and readback logs.

Float4 output is rendered as 3D points directly from its GPU buffer; scalar output
gets a value table and heatmap. Expand **Launch settings & buffer inputs** to edit
the entry point, threads per block, block counts, scalar arguments, record counts,
synthetic input patterns and output buffer. Suggested values are configurable
defaults, not inferred application semantics. The `blockIdx`, `blockDim` and
`threadIdx` CUDA built-ins are mapped to real WebGPU workgroup/invocation indices.

Desktop `.cu` imports extract standalone `__global__` and `__device__` functions
and numeric macros, preserving diagnostic line numbers. Includes, host allocation,
CUDA/OpenGL interop and window code are not executed. The supported CUDA language
subset still applies; this is not arbitrary CUDA/C++ execution. Source stays local
in the browser. Monaco assets and licenses are served with the app, without a CDN.

Run the demos directly in a browser with WebGPU support. No local installation or CUDA Toolkit is needed for the browser showcases.

A bounded **CUDA C kernel → typed AST → WGSL → WebGPU** pipeline, with a GPU-resident Three.js particle renderer, ten kernel examples, numerical tests and a device-local performance tuner.

This is source translation. **It does not run CUDA binaries, PTX, the CUDA driver, cuBLAS or arbitrary C++ in a browser.** It deliberately implements a useful kernel-language subset rather than pretending a few string replacements constitute a CUDA compiler.

## Validated on an RTX 5080

The project is running locally with installed, locked dependencies. **327 Node tests, 93 real WebGPU checks (including Three.js rendered-pixel interop and expected validation rejections), five native CUDA edge-case checks, and all 20 matched CUDA/WebGPU benchmark cases passed.** The static build also succeeds.

Open [the measured comparison](reports/performance-comparison.html) or read [the full methodology and results](reports/performance-comparison.md). All ten kernel sources are compiled by NVCC and translated to WGSL at two workload sizes. Raw GPU timestamps, CUDA event timings, ordinary CUDA launch timings, and verification logs are in `reports/`.

Measured hardware: NVIDIA GeForce RTX 5080, driver 616.64, CUDA Toolkit 13.3, Microsoft Edge 152, Windows WDDM. Results apply to these specific kernels and workloads, not all CUDA code or optimized CUDA libraries. See [VALIDATION.md](VALIDATION.md).
## Run

Try the [NVIDIA simpleGL sandbox](sandbox.html?example=wave): an unchanged, BSD-3-Clause-licensed NVIDIA CUDA vertex kernel translated by this compiler and displayed through WebGPU/Three.js. It passed 12 native CUDA and 12 browser GPU checks. See [provenance and instructions](showcases/simplegl/README.md).

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

Floating constants with an `f` suffix, e.g. `0.5f`, use float32. Unsuffixed floating literals use binary64 expression arithmetic emulated with integer limbs in WGSL: scalar +, -, *, /, comparisons, negation, and conversion back to float are supported. This preserves mixed-precision CUDA expressions without changing source literals; it is not native hardware FP64 and may use substantially more instructions. Explicit by-value double kernel parameters and double constant-record fields preserve both 32-bit words; size_t constant-record fields likewise preserve all 64 bits. Double local variables/buffers, double-to-integer casts and double transcendental functions remain unsupported. Mixed scalar expressions are explicitly typed. Guarded ternary expressions become real branches, not an eager `select()` that could evaluate an unselected buffer access or atomic operation.

**Not supported:** arbitrary host CUDA execution, kernel launch syntax inside compiled device code, cudaMalloc/streams/events host APIs, binary/PTX input, C++ STL, unrestricted C++ classes or templates, nested or storage-buffer structs, general pointers and unrestricted pointer arithmetic, runtime recursion, cooperative grid barriers, warp shuffles/votes, inline PTX, tensor cores/WMMA, half arithmetic, double storage, unrestricted 64-bit integer arithmetic, or floating-point atomics. Read-only records containing bool fields use native byte layouts when their stride is a multiple of four. Writable packed records and standalone float3/bool pointer-buffer ABIs remain unsupported. The feature-specific sections above describe supported templates, local structs, dynamic shared memory, helper pointers and vector operations. Fully parenthesized expression macros such as IMAD can expand into the typed syntax tree; nested expression macros remain unsupported. Texture support includes tex3D<float>, tex3D<float4>, tex2D<float>, tex1D<float4> and tex2DLayered<float4> with bound kernel handles or explicitly typed helper chains. Surface support includes float4 surf1Dwrite with checked global X coordinates, float surf2Dwrite with checked global XY coordinates, float/byte surf3Dwrite with checked global XYZ coordinates, and float4 surf2DLayeredwrite with checked global XY and a uniform layer; arbitrary coordinates, surface reads and surface helper parameters remain unsupported.

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

[Try NVIDIA post-process glow in the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=postprocess): original RGBA texture and shared-tile CUDA kernel, original teapot input, editable highlights and CUDA/WGSL comparison. [Validation and launch settings](showcases/postprocess-gl/README.md).

[Try NVIDIA bilateral filtering in the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=bilateral): the original edge-preserving filter and photograph, with editable colour-distance smoothing. [Validation and launch settings](showcases/bilateral-filter/README.md).

[Try NVIDIA Mandelbrot in the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=mandelbrot), [Julia](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=julia), or [two accumulated frames](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=mandelbrot-accumulated). Original float kernel, editable viewport and colours, and six exact native frame comparisons with multiply-add fusion disabled in the native build. [Validation, arithmetic profile and supported scope](showcases/mandelbrot/README.md).

### NVIDIA 3D volume filtering

[Try the volume filter in the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=volume-filter). Run the unchanged NVIDIA filter on its original 32³ Bucky volume, compare CUDA/WGSL and move through the output with a Z-slice slider. Separate presets expose voxel coordinates, nearest sampling and the original normalized/wrap settings. Sixteen native volume captures match hardware WebGPU exactly for the tested configurations. [Supported scope and why upstream settings produce a flat volume](showcases/volume-filter/README.md).

### NVIDIA marching cubes

[Open the implicit 3D mesh in the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=marching).
Original NVIDIA kernels classify a 16³ field, compact voxels and generate 2,080 triangles with shared memory. Project CUDA scan kernels connect the stages on the GPU. Compare all generated shaders and orbit the resulting position/normal buffers directly. Native CUDA and hardware WebGPU agree within 1e-6 for every mesh component. [Setup, pipeline format and supported scope](showcases/marching-cubes/README.md). [The Bucky sampled-volume branch also runs in the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=marching-volume), producing 11,126 triangles at the default threshold. Three complete volume meshes match native CUDA.

### NVIDIA box filter

[Blur NVIDIA’s original teapot image in the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=box-filter). The original CUDA row and column kernels use a sliding window and keep the packed intermediate image on the GPU. The default 1024² image uses radius 14; edit both radius settings to try another size. Eight complete colour captures match native within one channel level, and all four scalar entries are verified too. [Setup and supported scope](showcases/box-filter/README.md).

### NVIDIA Sobel edges

[Find teapot edges in the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=sobel). The unchanged NVIDIA texture kernel processes its original 1024² grayscale image, with editable edge intensity and CUDA/WGSL comparison. Eight image captures across edge detection and copy kernels match native CUDA exactly. [Setup, validation and remaining shared-memory path](showcases/sobel/README.md).

[Run the shared-memory Sobel version](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=sobel-shared). NVIDIA's unchanged shared-tile kernel now has its own showcase, with four exact native image comparisons, original byte storage and checked packed-output pitch alignment.

### NVIDIA image denoising

Run [KNN](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=denoise-knn), [non-local means](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=denoise-nlm), or [shared non-local means](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=denoise-nlm2) on NVIDIA's original noisy portrait. All seven filter/copy/diagnostic kernels run in the sandbox with their original function bodies. Fourteen image captures match native CUDA exactly or within one colour level. [Settings and verification](showcases/denoising/README.md).

### NVIDIA DCT image showcase

[Run the DCT image in the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=dct)
(or [locally](http://localhost:5173/sandbox.html?example=dct)). The original first
floating-point DCT, quantization and IDCT kernels reconstruct NVIDIA's 512 × 512
teapot image. All intermediate values and final pixels match native CUDA on the
validated NVIDIA Blackwell GPU. Five compiled CUDA passes, including two labelled
project adapters for host pixel conversion, are available in the comparison view.
See [validation and remaining DCT paths](reports/dct-progress.md). The packed-short path is not yet claimed to work.

[Run the optimized DCT showcase](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=dct-optimized)
(or [locally](http://localhost:5173/sandbox.html?example=dct-optimized)). Its original
shared-memory DCT/IDCT helper functions now run unchanged, with all three stages
matching native CUDA exactly for a padded 64 × 32 image and the full 512 × 512
teapot. This establishes correctness; no speedup over the first path is claimed.

### Animated NVIDIA FFT ocean

[Run the ocean in the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=ocean)
(or [locally](http://localhost:5173/sandbox.html?example=ocean)). This runs the
original spectrum, height and slope CUDA kernels around a project GPU inverse
FFT, producing an animated 256 × 256 ocean with 130,050 triangles. Drag to orbit;
use Animate GPU steps or Stop to control the simulation. Compare generated WGSL
for the original kernels, runtime FFT and labelled CUDA mesh adapter.

All numerical stages were compared with native CUDA/cuFFT at three animation
times; maximum absolute error was 3.17e-7. The preview uses project rendering,
and no cuFFT performance parity is claimed. See [validation details](reports/ocean-progress.md).

### NVIDIA particle collisions

[Run the particle simulation](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=particle-collision)
(or [locally](http://localhost:5173/sandbox.html?example=particle-collision)).
Watch 1,024 spheres fall and collide using NVIDIA's original integration functor,
spatial hashing and collision kernels. Sorting and cell clearing also run on the
GPU, with positions shared directly with the renderer. Pause/resume and orbit
controls are available in the sandbox.

The complete simulation is compared with native Thrust/CUDA after 1, 8, 32 and
64 steps, with maximum position error below 8.35e-7. The preview uses project
initial conditions and rendering. See [validation and support boundaries](reports/particle-collision-progress.md).

### NVIDIA pre-integrated volume

[Open the coloured 3D volume in the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=volume-preintegrated). The original NVIDIA integration, layered colour/opacity tables and ray marcher run with unchanged device function bodies. Both original 1024² transfer layers stay on the GPU; the final 256² image matches native CUDA within one 8-bit channel level. The two differently coloured halves are selected by the original kernel. Camera, density and rendering mode are editable in the launch settings. See [source, controls and validation](showcases/volume-preintegrated/README.md).

### NVIDIA smoke particles

[Run the smoke showcase in the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=smoke): 16,384 particles, original CUDA integration/depth functions, GPU depth sorting and a WebGPU adaptation of NVIDIA's 32-slice shadow renderer. See [the smoke notes](showcases/smoke/README.md) for validation and presentation differences.

[Try NVIDIA fluid flow in the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=fluids): drag to stir 262,144 GPU particles, inspect the original CUDA and generated WGSL, and run a 512 × 512 FFT fluid solver. [Implementation and native comparison details](showcases/fluids/README.md).

[Try NVIDIA stereo disparity in the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=stereo): compare the original CUDA and generated WGSL while recovering a disparity image from NVIDIA's stereo camera pair. [Details and native verification](showcases/stereo/README.md).

[Try NVIDIA optical flow in the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=optical): estimate motion between two images using the original six kernels, five-level pyramid and 7,500 solver iterations. [Configuration and native verification](showcases/optical-flow/README.md).

### NVIDIA FFT convolution

[Run the original-size FFT convolution in the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=fft-convolution). The unchanged padding and modulation kernels process the original 2000 × 2000 random input through 2048 × 2048 real FFTs. Complete output matches native CUDA with relative L2 error 2.0e-7. [Pipeline, preview and validation](showcases/fft-convolution/README.md).

### NVIDIA custom and fused FFT convolution

[Custom FFT convolution](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=fft-custom) and [fused FFT convolution](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=fft-fused) run the other two original convolutionFFT2D paths at 2000 × 2000. Both use 1024 × 2048 complex transforms, unchanged NVIDIA device kernels and GPU-only texture transfers. Full results match native CUDA at approximately 2.02e-7 relative L2 error. [Pipeline details and validation](showcases/fft-convolution-custom/README.md).
