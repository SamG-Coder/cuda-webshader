# Native OptiX with a real WebGPU device

ChromiumRTXCuda supports CUDA-authored OptiX ray programs through the existing
native GPU permission and resource-sharing APIs. WebCuda exposes the objects
without exposing OS handles or CUDA pointers to JavaScript.

```js
// Run from a user click before creating the runtime:
await GpuRuntime.requestPermission();
const rt = await GpuRuntime.create({nativeInterop: {requirements: {
  optix: true, sharedTextures: true, textureFormats: ['rgba8unorm'],
}}});
if (!rt.native) {
  // Retain the existing WebGPU renderer.
} else {
  const vertices = await rt.createSharedBuffer(vertexCount * 12);
  const scene = await rt.native.createAccelerationStructure({
    vertexCount, vertexStride: 12, allowUpdate: true,
  });
  const output = await rt.createSharedTexture({width, height, format: 'rgba8unorm'});
  const pipeline = await rt.native.rayTracingPipeline(cudaSource, {
    raygen: '__raygen__main', miss: '__miss__main', closestHit: '__closesthit__main',
    maxTraceDepth: 2, numPayloadValues: 3,
    parameters: [{name: 'output', type: 'surface'}],
  });
  // Fill vertices.gpuBuffer using WebGPU or a native CUDA geometry kernel.
  await rt.native.batch()
    .buildAccelerationStructure(scene, vertices)
    .trace(pipeline.bind(scene, {output}), [width, height])
    .submit();
  // Sample/copy output.gpuTexture with WebGPU. No CPU pixel transfer is needed.
}
```

The browser provides OptiX headers and a constant `params` structure. CUDA
programs use `optixTrace(params.scene, ...)`, `params.output` as a CUDA surface,
and typed buffer/scalar fields declared by `parameters`. Buffer descriptors
accept `element: 'float3'` and other supported scalar/vector types; the default
is `void`. Parameters support `buffer`, `surface`, `f32`, `i32`, `u32`.
Source must be self-contained: no filesystem `#include` or preprocessing.

`batch.dispatch()`, `batch.buildAccelerationStructure()` and `batch.trace()`
can be mixed in one batch. They run in order on the same native CUDA stream.
GPU fences transfer ownership between WebGPU and native code. Await the
submission before using those resources in subsequent WebGPU commands.
`buildAccelerationStructure(scene, vertices, {update:true})` refits previously
built, update-enabled geometry. Use a new scene for changed vertex count or
stride. Use `pipeline.destroy()` and `scene.destroy()` to release native objects;
runtime disposal and WebGPU device loss also dispose them.

Check `rt.native.capabilities.optix` before choosing this path. This version
supports non-indexed float3 triangles, one geometry acceleration structure per
scene, raygen/miss/closest-hit and optional any-hit programs, recursive trace
depth up to 8, 32 payloads, and two-dimensional launches. It does not expose
instances, custom primitives, motion blur or callable programs. Indexed meshes
can be expanded in an application GPU kernel. Supported shared texture formats
are `rgba8unorm`, `rgba16float`, `rgba32float`, and `r32float`, subject to the
normal sharing limits. A shared packed buffer plus GPU copy is the fallback
when direct texture sharing is unsuitable. There is no arbitrary texture import.

This adds an OptiX pipeline type; it does not make ordinary `native.kernel()`
launches understand OptiX device intrinsics or translate them into WebGPU.
Applications keep their own WebGPU fallback. ClearWater still needs its scene
geometry and reflection shading connected to this API to change its renderer.

The [browser API reference](https://github.com/SamG-Coder/ChromiumRTXCuda/blob/cuda-rtx/rtx_cuda/OPTIX.md)
contains a complete `.cu` example, supported limits and build instructions.
The browser's actual-GPU suite verifies primary and reflected rays, GPU geometry
updates, shared output, ClearWater CUDA bloom/presentation kernels compared with
their WGSL versions, resizes, permission revocation and device loss.
