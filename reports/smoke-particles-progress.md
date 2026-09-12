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

## Remaining work

The original integration functor has two state fields (a float time step and a
texture handle) and four float4 zip elements. The existing iterator lowering
accepts one float state field and two float4 elements. Compilation currently
stops at the texture state field with
`Expected '__host__', found 'cudaTextureObject_t'`.

The depth functor additionally needs float3 state and a mixed float4/float zip.
Its depth keys need float sorting; the current GPU pair sorter supports uint
keys. Then the sandbox must run integration with position/velocity feedback,
depth sorting and the smoke presentation. None of these are claimed complete by
the noise tests, and no smoke card has been added prematurely.

Validation for the texture support: 529 unit tests, 161 real NVIDIA WebGPU
checks, compile-all and static build pass. No software GPU execution was used.
No smoke simulation or rendering performance comparison has been established.
