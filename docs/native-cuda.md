# Automatic native CUDA selection

`GpuRuntime.create()` defaults to `backend: 'auto'`. It detects
`navigator.cuda.SupportsNativeCuda()` from
[ChromiumRTXCuda](https://github.com/SamG-Coder/ChromiumRTXCuda) and uses native
CUDA when available and permitted. Other browsers retain WebGPU.

```js
import {GpuRuntime} from './src/runtime/runtime.js';

startButton.onclick = async () => {
  const runtime = await GpuRuntime.create();
  console.log(runtime.backend); // 'native-cuda' or 'webgpu'
  console.log(runtime.describe().native); // capability, permission, fallback reason
  const x = runtime.createBuffer(new Float32Array([1, 2, 3, 4]));
  const kernel = await runtime.kernel(`
    __global__ void scale(float* x, float factor, unsigned n) {
      unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) x[i] *= factor;
    }`, {entry: 'scale', workgroupSize: [128, 1, 1]});
  const invocation = kernel.bind({x}, {factor: 3, n: 4});
  runtime.batch().dispatch(invocation, [1]).submit();
  console.log(await runtime.read(x)); // [3, 6, 9, 12]
  await runtime.dispose();
};
```

The core API remains `createBuffer`, `kernel`, named `bind`, `setScalars`,
`batch().dispatch().submit()`, `write`, `read`, `destroyBuffer`, `idle`, and
`dispose`. Uploads are snapshotted at the call and scalars at each dispatch.
Native resources remain in the native CUDA session between calls.

## Permission and selection

| Situation | `backend: 'auto'` (default) | `backend: 'native'` |
| --- | --- | --- |
| Supported and already granted | Native CUDA | Native CUDA |
| Permission is Ask, called from a user click | Request the normal site permission; use CUDA if granted | Request; reject if denied |
| Ask without user activation | WebGPU, reason `permission-required` | Reject with `NotAllowedError` |
| Denied | WebGPU, reason `permission-denied` | Reject with `NotAllowedError` |
| Native unavailable | WebGPU, reason `unavailable` | Reject with `NotSupportedError` |
| Native initialization fails | WebGPU, with a reported reason/message | Reject the original error |

`backend: 'webgpu'` does not probe native or request permission. Supplying a
WebGPU `device` or `adapter` also selects WebGPU in auto mode. Supplying either
with strict native is an error. `requestPermission: false` prevents an automatic
prompt, even inside a click handler; an existing grant still permits native.
Fallback happens at creation, before application buffers exist. A live runtime
never silently migrates resources when permission is revoked or a dispatch fails.

Permission can be obtained separately from a button:

```js
enableNative.onclick = async () => {
  const state = await GpuRuntime.requestPermission();
  if (state === 'granted') {
    // Dispose any previous runtime, then recreate its kernels and resources.
    const runtime = await GpuRuntime.create({backend: 'native'});
  }
};
```

`SupportsNativeCuda` and `requestPermission` are also named exports from
`src/runtime/runtime.js`. Detection does not request permission. The browser
enforces HTTPS/trusted localhost, an active top-level document and the
`native-gpu` Permissions Policy. A new permission prompt additionally requires
visibility and user activation. ChromiumRTXCuda alpha.4 and later preserve
grants and resources across tab switches, occlusion and minimising. Revoking
permission or closing/deactivating the document terminates its native session;
recreate its resources afterwards. Earlier browser releases also terminated
sessions when hidden and should be updated for applications that resume.
One native runtime owns a document at a time. Await its `dispose()` before opening
a replacement so a delayed close cannot terminate the new session.

## Precompiled dual-backend artifacts

Raw CUDA source works with the common API on either backend, within each
backend's supported language/API scope. A WGSL-only artifact contains no CUDA
source and cannot be executed by NVRTC. Opt in to a dual-backend artifact:

```js
import {compile, serializableArtifact} from './src/compiler/compiler.js';
const artifact = serializableArtifact(compile(cudaSource, {
  entry: 'scale', workgroupSize: [128, 1, 1], includeNativeSource: true,
}));
// Save/serve JSON, then on either backend:
const kernel = await runtime.kernel(artifact);
```

Or use `node scripts/compile.mjs kernel.cu --entry scale --native --out generated/scale`.
Source embedding is opt-in to avoid increasing existing artifact downloads.
The first dual-artifact format supports float/int/uint scalars and pointers,
including scalar defaults and dynamic shared-memory size. It rejects layouts
or compiler-managed features that do not share the native ABI. Original CUDA
source still needs to be self-contained and accepted by NVRTC; this option is
not a promise that all WebCuda language extensions are native CUDA syntax.

## Native CUDA with WebGPU resources

Renderers can retain their real WebGPU device, buffers, textures, and canvas
while selecting native CUDA for individual kernels:

```js
const runtime = await GpuRuntime.create({
  nativeInterop: {requirements: {
    sharedBuffers: true, gpuBufferToTexture: true,
    maxResourceBytes: 32 * 1024 * 1024, sharedBytes: 128 * 1024 * 1024,
    resources: 5, blocksPerLaunch: 32768,
  }},
});
// No implicit prompt here. Request permission from a button separately.
console.log(runtime.device instanceof GPUDevice); // true
console.log(runtime.nativeInteropStatus); // actual-device capabilities / reason
if (runtime.native) {
  const output = await runtime.createSharedBuffer(1024 * 512 * 4);
  const kernel = await runtime.native.kernel(cudaSource, {
    entry: 'render', workgroupSize: [8, 8, 1],
  });
  await runtime.native.batch().dispatch(kernel.bind({output}), [128,64]).submit();
  // output.gpuBuffer is a real GPUBuffer. Submit normal WebGPU use after await.
  const encoder = runtime.device.createCommandEncoder();
  encoder.copyBufferToTexture({buffer:output.gpuBuffer,bytesPerRow:4096},
    {texture:canvasContext.getCurrentTexture()},[1024,512]);
  runtime.device.queue.submit([encoder.finish()]);
}
```

Alternatively, call `await runtime.enableNativeInterop({requirements})` on an
existing WebGPU runtime. `getNativeInteropCapabilities(device)` and
`supportsInteropRequirements(capabilities, requirements)` are named exports.
Detection checks permission, the actual physical adapter, GPU-side fence
synchronization, formats, usage flags, resource bytes/counts and launch limits.
If requirements are unmet, `runtime.native` is null and the WebGPU backend
remains available. An existing grant is required unless the application
explicitly sets `nativeInterop.requestPermission: true` from a user gesture.

Create a shared texture with
`await runtime.createSharedTexture({width,height,format,usage})`. Its
`gpuTexture` and `view` are real WebGPU objects. Native kernels bind the wrapper
to `cudaSurfaceObject_t`; conventional surface signatures are inferred, or
provide an explicit ordered `{name,type:'surface'}` parameter descriptor.
The returned resource is distinct from an arbitrary normal WebGPU texture.
There is no OS-handle or arbitrary-texture import API in this library.

The initial browser backend supports 2D, single-layer, single-mip, single-sample
`rgba8unorm`, `rgba16float`, `rgba32float`, and `r32float` textures. Use CUDA
`uchar4`, half bits in `ushort4`, `float4`, and `float` surface values respectively.
Texture usage supports WebGPU copy, sampling, storage and render attachment
flags. Shared buffers support copy, storage, vertex, index and indirect usage;
mapping and uniform/query use are excluded. The browser's
[resource contract](https://github.com/SamG-Coder/ChromiumRTXCuda/blob/cuda-rtx/rtx_cuda/README.md#native-cuda-with-real-webgpu-resources)
specifies dimensions, limits, alignment and lifecycle behavior.

`runtime.native.batch().submit()` returns a promise. Submit earlier WebGPU work
first, then await this promise before WebGPU uses the acquired resources again.
It resolves after CUDA's signal and WebGPU's wait have been queued on the GPU.
The library never transfers frame pixels through JavaScript or CPU buffers.
Shared-buffer to texture copy/conversion is a GPU-only fallback, and is reported
separately from direct shared-texture access. ArrayBuffer readback helpers are
explicit diagnostic operations, not a presentation route.

Use `destroyBuffer`, `destroyTexture`, and await `runtime.dispose()` for cleanup.
Disposal, permission revocation and device loss invalidate native resources.
Recreate the WebGPU device and scene after an aborted native session; live
buffers are not silently migrated. The browser owns all native handles and
cross-API fences. ClearWater uses the original `bloom_pass` and `present` CUDA
kernels with five shared buffers while retaining WebGPU simulation/rendering
and a GPU copy into its canvas.

## Standalone native runtime boundaries

This integrates the fork's CUDA **buffer-compute** API. Native runtime objects do
not expose a WebGPU `device`, `GPUBuffer`, command encoder, textures, the Three.js
buffer bridge, object arenas, GPU scalar-buffer parameters, or the WebGPU
FFT/sort/scan helpers. Those calls reject with an explicit WebGPU requirement.
Renderers that access these resources use WebGPU or the `nativeInterop` option.
The repository's renderer, sandbox and Bend lab explicitly keep that backend.
The separate interoperability extension above supports selected ClearWater
kernels. It does not add RTX/DLSS rendering to this library.

The native transport currently limits explicit buffers to 64 MiB total, 256
buffers, 128 modules, 256 dispatches per batch, 65,536 blocks per launch and
48 KiB requested dynamic shared memory, subject to the hardware limits. Native
host transfers are chunked at 512 KiB and CPU-mediated. They are not WebGPU/CUDA
zero-copy sharing. The fork is Windows/NVIDIA-specific and remains experimental;
its [API documentation](https://github.com/SamG-Coder/ChromiumRTXCuda/blob/cuda-rtx/rtx_cuda/README.md)
describes its permission and native-process boundaries.

## Validation

`npm test` includes selection, permission-state, ABI, resource ownership,
transactional scalar update, artifact serialization and error-propagation tests.
These use a mocked transport and do not establish GPU execution.

`npm run test:native:browser` launches a real ChromiumRTXCuda executable plus
stock Edge. Set `RTXCUDA_CHROME` to the fork's `chrome.exe` path when necessary.
It executes raw source and dual artifacts, checks all 262,147 SAXPY results,
tests >512 KiB uploads/readbacks, dispatch snapshots, subrange writes, explicit
WebGPU, permission denial, the no-gesture gate, and revocation. Decisions are
set through DevTools in an isolated browser context; this is real browser
permission enforcement, not evidence of a human accepting the prompt.
The report is written to ignored `test-results/native-browser.json`.

The [2026-10-04 validation report](native-cuda-validation.json) records a passing
static-build run, 800 passing unit tests, kernel compilation and site build.
It is correctness evidence, not a native-versus-WebGPU speed benchmark.
