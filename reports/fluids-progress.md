# NVIDIA fluidsGL progress

Candidate: cpp/5_Domain_Specific/fluidsGL at NVIDIA cuda-samples revision
5443602d89ed99aede2e4b7bf329daddeadb320e.

The original advectVelocity_k and diffuseProject_k function bodies are retained
in tests/fluids-advectVelocity_k.cuh and tests/fluids-diffuseProject_k.cuh.
The host header's cData type is supplied as typedef float2 cData.

## Float2 textures and numerical checks

The compiler accepts tex2D<float2>, including typedef names and texture helper
chains. The runtime uploads two-component rg32float textures. Mixing formats
through one texture handle is rejected. Read-write rg32float surfaces are not
part of this addition.

Native CUDA and real NVIDIA WebGPU execute the original two kernels over a
19 by 13 field with partial workgroups. Advection writes 24-float padded rows;
row padding and end guards remain unchanged. Time steps 0, 0.09 and 1 are tested.
Diffusion/projection results match exactly for all 988 float components.

Zero-time and time-1 advection outputs match exactly on the tested GPU. At
time 0.09, the maximum component differences are 0.002017634 and 0.003631532.
This is explicitly not an exact match.

The native harness captures backtrace coordinates and direct tex2D results.
Sampling those same coordinates through WebGPU reproduces the discrepancy.
The unchanged advection kernel matches the isolated WebGPU samples within 1e-6,
so the observed mismatch is isolated to filtering rather than backtrace math.

CUDA documents 8 fractional bits for its interpolation coefficients:
[CUDA C++ Programming Guide, Texture Fetching / Linear Filtering](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf).
This motivates the fixture's conservative per-backend envelope:
(maximum horizontal texel slope + maximum vertical texel slope) / 256 + 1e-6.
Both native CUDA and WebGPU are independently checked against ideal bilinear
interpolation at the captured coordinates. The cross-backend bound is twice
that envelope. This is a tested numerical criterion, not a universal promise
about WebGPU filtering precision or proof of identical hardware quantization.

The native descriptor follows the sample's unnormalized linear texture
settings. WebGPU uses unnormalized clamp addressing, matching the measured
out-of-domain native behaviour. No CUDA function was modified to fit WebGPU.

See reports/fluids-check.json for the measured errors and bounds.
Validation: 541 unit tests, 165 real NVIDIA GPU regression checks, compilation
and static build passed. No software GPU tests ran.

## Remaining blockers

size_t pitch parameters now parse and use a bounded launch ABI: host values must
be integers from 0 through 0xffffffff. The runtime rejects larger values rather
than truncating them. Expressions retain emulated 64-bit unsigned arithmetic,
including signed-operand conversion, multiplication, addition, comparisons,
explicit 32-bit casts and float conversion. Scalar locals are supported;
size_t storage buffers, arrays and struct fields are explicitly rejected.
General size division is still unsupported.

All 35 probe outputs match native CUDA, including products above 32 bits,
negative signed operands and sizeof(size_t) = 8. See size-values-check.json and
size-values-native.txt. Validation: 543 unit tests, 166 real NVIDIA GPU checks,
compilation and static build pass.

The three remaining original kernels now reach their pointer expressions:
addForces_k and updateVelocity_k fail on local aliases combining a byte-address
cast with a trailing element offset. advectParticles_k fails on an inline
dereference of that cast. These require typed pitched byte-pointer lowering. The host pipeline requires forward real-to-complex and inverse
complex-to-real FFT layouts, float2 buffer-to-texture updates, repeated velocity
feedback and particle rendering/interaction. Current FFT support only provides
a complex inverse transform.

No fluids showcase card has been added. Multi-step error accumulation and a
native/WebGPU performance comparison have not yet been established.
