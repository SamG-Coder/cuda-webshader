# NVIDIA smokeParticles

Candidate: `cpp/5_Domain_Specific/smokeParticles` at NVIDIA CUDA Samples revision
`5443602d89ed99aede2e4b7bf329daddeadb320e`.

This is an in-progress candidate, not yet a smoke showcase. The original sample
combines procedural particle integration, depth sorting, and OpenGL smoke
rendering with volumetric shadows using half-angle slicing.

## Float4 noise texture support verified

The compiler now supports `tex3D<float4>` through typed helper parameter chains,
with independent float4 texels stored as `rgba32float`. Three-dimensional
coordinate scaling and unnormalized nearest sampling preserve the existing
texture-coordinate contract. Scalar and float4 sampling cannot share a handle.

`tests/smoke-noise-kernel.cuh` preserves the original `noise3D` helper unchanged.
`tests/smoke-noise-host.cuh` preserves the original `frand` and
`createNoiseTexture` functions unchanged. The native harness chooses seed 1,
calls the original generator with 64 × 64 × 64 dimensions, and captures the
four-component texture for identical WebGPU inputs. The MIT probe checks the
original helper's XYZ result and separately checks the fourth texture channel.

All four sampling modes match native CUDA exactly at 769 coordinates each
(3,076 float components per mode):

| Mode | Filtering | Addressing | Coordinates |
| --- | --- | --- | --- |
| 0: original smoke settings | Linear | Repeat | Normalized |
| 1 | Nearest | Repeat | Normalized |
| 2 | Linear | Clamp | Normalized |
| 3 | Nearest | Clamp | Unnormalized |

The inputs include texel centres and fractional coordinates outside the nominal
texture domain. The final launch is partial, and output guards stay unchanged.
See `reports/smoke-noise-check.json` and `reports/smoke-noise-native.txt`.

## Original integration and depth calculation verified

Typed zip lowering now accepts one to four contiguous float or float4 tuple
elements and up to four scalar, float-vector or texture state fields. Simple
constructors must map each distinct value argument to one matching field.
Ambiguous tuple types, gaps, pointer state and generated-name collisions are
rejected. The implementation is independent of NVIDIA function names.

The original SimParams declaration, noise3D, integrate_functor and
calcDepth_functor are preserved in tests/smoke-integration-kernel.cuh. Native
CUDA executes them using actual Thrust zip iterators and for_each. WebGPU
executes the generated bounded kernels, exchanging input/output GPU buffers
after each step.

At steps 1, 8, 32 and 64, all 9,252 captured float components match native CUDA
exactly: positions with ages, velocities with lifetimes, and scalar depth keys.
There are 257 particles, so the final 128-thread workgroup is partial; output
guards remain unchanged. Depth calculations exercise two different sort
directions. See reports/smoke-integration-check.json.

This is a controlled numerical fixture, not the desktop application's emitter
state. It uses the captured original 64³ noise texture, original 0.5 timestep,
0.1 noise frequency and 0.001 amplitude, with nonzero gravity/noise motion and
0.99 damping to exercise those fields. Initial ages include expired particles
and values crossing their lifetime. The original age-update behavior is kept,
including its one-step overshoot before clamping.

Validation: 533 unit tests, 162 real NVIDIA WebGPU checks, compile-all and static
build pass. No software GPU execution was used.

## Float depth sorting verified

The runtime and sandbox pipeline now support sortPairs with keyType: f32.
Depth keys and uint particle indices stay on the GPU. Ordering uses integer
transforms of float bits, preserving subnormals, signed-zero bits and NaN
payloads. Equal numeric keys retain their original order, both zeros compare
equal, and NaNs sort last while retaining their relative order. The existing
default uint sort is unchanged in behavior.

Four native Thrust sort_by_key captures of the 257-particle smoke depth keys
(steps 1, 8, 32 and 64) match WebGPU exactly, for both key bits and particle
indices. Separate IEEE edge-case tests cover 1, 2, 127, 128, 129, 257 and 16,384
elements, including infinities, NaNs, both signed zeros, positive and negative
subnormals, and large uint payloads. Guards remain intact and sorting has zero
CPU readback. The edge-case policy is a documented runtime policy, not a claim
about native Thrust's NaN ordering.

See reports/float-sort-check.json and reports/smoke-sort-native.txt.
Validation: 534 unit tests, 163 real NVIDIA WebGPU checks, compile-all and static
build pass. Original NVIDIA functions have not been edited.

## Combined sandbox pipeline verified

The sandbox pipeline supports read-only volume-float4 binary textures and
bounded copyBuffer steps. Feedback copies retain stable output buffers for the
renderer and update the original integration function's separate input buffers
without CPU readback.

The combined test runs original integration, original depth calculation,
MIT GPU index initialization (the host thrust::sequence equivalent), float
sorting and two feedback copies through executePipeline. At steps 1, 8, 32 and
64, positions, velocities, sorted key bits and indices all match the native
captures exactly. Two sort directions are exercised. All six generated kernels
are exposed through the normal CUDA/WGSL comparison callback. The fixture uses
257 particles; it is a numerical pipeline test, not a smoke rendering test.

Validation: 536 unit tests, 164 real NVIDIA WebGPU checks, compile-all and static
build pass. See reports/smoke-pipeline-check.json.

## Remaining work

The smoke preview renderer and public preset/card remain to be implemented.
Inspection of the original SmokeRenderer.cpp confirms 32 half-angle slices.
Each slice first draws to the camera image using the current light attenuation
texture, then draws from the light view to accumulate attenuation. Camera
blending reverses between front-to-back and back-to-front depending on view/light
orientation. The default light buffer is 256 square; sprite and shadow alpha
defaults are 0.1 and 0.005. These rendering stages must be connected to the
verified particle buffers and sorted indices.

No smoke card has been added yet. No smoke simulation/rendering performance
comparison has been established.
