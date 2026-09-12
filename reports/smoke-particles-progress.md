# NVIDIA smokeParticles

[Run the smoke showcase](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=smoke).

The source is NVIDIA cuda-samples revision
5443602d89ed99aede2e4b7bf329daddeadb320e,
cpp/5_Domain_Specific/smokeParticles. Original noise3D, integrate_functor and
calcDepth_functor bodies are unchanged.

## Verified compute support

- tex3D<float4> with a 64-cubed rgba32float volume. Four sampling configurations
  match native CUDA at 769 coordinates each.
- Typed zip functors: four float4 integration buffers, scalar/texture state,
  float3 sort-vector state and mixed float4/float depth buffers.
- Integration, ages, lifetimes and depth keys match native CUDA exactly at
  steps 1, 8, 32 and 64 for a 257-particle partial-workgroup fixture.
- Float key/uint value sorting remains entirely on the GPU. Native Thrust
  depth sorting matches exactly at all four captured steps. Separate tests
  cover signed zeros, subnormals, infinities and NaN payload preservation.
- The combined sandbox pipeline includes integration, depth calculation,
  MIT index initialization, sorting and GPU feedback copies. It matches the
  native fixture at every checkpoint without intermediate CPU readback.

Detailed reports: smoke-noise-check.json, smoke-integration-check.json,
float-sort-check.json and smoke-pipeline-check.json in this directory.

## Showcase and rendering

The preset contains 16,384 particles in a deterministic sphere. It retains
the original integration defaults and noise texture, with lifetime 1000 and
preview radius 0.06. It is a finite cloud, not the desktop emitter UI.
The main page has an individual card linking only to the sandbox.

The WebGPU presentation adapts NVIDIA's motion-stretched billboards, shadow
sampling and 32 half-angle slices. Each slice draws to the camera image, then
updates a 256-square light attenuation texture. Both front-to-back and
back-to-front camera blending are exercised. Light position (5,5,-5), colour
(1,1,0.5), sprite alpha 0.1, shadow alpha 0.005 and attenuation (0.1,0.2,0.3)
follow the original defaults. The light colour explains the yellow appearance.

Renderer WGSL is adapted presentation code, not generated from CUDA.
The six compute kernels remain available in the CUDA/WGSL comparison.

## Validation

The exact showcase inputs match native CUDA at step 64 for all 131,072
position/velocity components. The sandbox test also checks paused orbit,
pause/resume, unchanged CUDA source, six comparison entries, mobile layout and
renderer disposal when switching examples. See smoke-sandbox-check.json.

A separate 8,192-particle rendering test finds 89,441 pixels changed by shadows,
checks finite output and exercises both blend directions. See
smoke-renderer-check.json and smoke-renderer-preview.png.

All 537 unit tests, 164 real NVIDIA WebGPU regression checks, compilation,
static build and all 82 sandbox presets passed. No software GPU tests ran.

This is not a pixel-equivalence claim against the desktop OpenGL renderer.
Desktop emitter controls and optional shadow blur are not included.
No native/WebGPU smoke performance comparison has been established.
