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
state. It uses the captured original 64� noise texture, original 0.5 timestep,
0.1 noise frequency and 0.001 amplitude, with nonzero gravity/noise motion and
0.99 damping to exercise those fields. Initial ages include expired particles
and values crossing their lifetime. The original age-update behavior is kept,
including its one-step overshoot before clamping.

Validation: 533 unit tests, 162 real NVIDIA WebGPU checks, compile-all and static
build pass. No software GPU execution was used.

## Remaining work

Depth keys still need float sorting; the current GPU pair sorter supports uint
keys. The sandbox then needs integration feedback, depth sorting and the smoke
presentation, including the original half-angle slicing and volumetric shadows.
These are not claimed complete by the integration tests, and no smoke card has
been added yet. No smoke simulation/rendering performance comparison has been
established.
