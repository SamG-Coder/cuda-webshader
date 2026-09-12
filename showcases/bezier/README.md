# NVIDIA Bezier tessellation

[Run in the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=bezier).

Original `cdpBezierTessellation` device functions from NVIDIA cuda-samples,
revision `5443602d89ed99aede2e4b7bf329daddeadb320e`:
[upstream source](https://github.com/NVIDIA/cuda-samples/blob/5443602d89ed99aede2e4b7bf329daddeadb320e/cpp/3_CUDA_Features/cdpBezierTessellation/BezierLineCDP.cu).
NVIDIA's BSD-3-Clause copyright and redistribution notice is retained in kernel.cu.
The compiler, runtime and preview additions are MIT.

The input contains all 256 original control-point triples from the native run,
packed into WGSL records with zero initial counts and null vertex handles.
No native output or tessellation count is loaded by the sandbox.
The original parent computes curvature, chooses tessellation counts, allocates
vertex storage and queues its original child calls. Bounded indirect GPU dispatch
executes those calls using scalar arguments loaded from the queue.

The preview reads the pointer records and allocated float2 vertices directly on
the GPU. Colours are a display aid assigned per curve; the CUDA source does not
produce colours. Scroll to zoom, drag to pan and double-click to reset the view.
This is a 2D curve sample. The original random control points produce a dense
collection of crossing curves.

Verification on NVIDIA Blackwell: all 256 curves / 3,958 vertices match native
CUDA within 1e-6; maximum coordinate difference 1.1920928955078125e-7.
The render path reads no geometry back to the CPU. The scheduler reads only its
8-byte error header. Queue reuse, zero work, overflow rejection and original
cleanup are also covered by the full GPU regression. Nested child launches remain
unsupported. Device allocations use explicitly bounded typed pools.

- [Sandbox evidence](../../reports/bezier-sandbox-check.json)
- [Complete parent/child native comparison](../../reports/bezier-cdp-scheduled.json)
- [Native execution report](../../reports/bezier-cdp-native.json)
- [Compiler progress and limitations](../../reports/bezier-cdp-progress.md)

Run `node scripts/test-bezier-sandbox.mjs` after `npm run build` to verify the
built sandbox with a real NVIDIA adapter. No software GPU fallback is used.
