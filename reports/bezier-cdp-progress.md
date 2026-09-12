# NVIDIA adaptive Bézier tessellation

Target: the complete 256-curve `cdpBezierTessellation` workload at NVIDIA CUDA
Samples revision `5443602d89ed99aede2e4b7bf329daddeadb320e`.

## Native baseline

`tests/bezier-cdp-native.cu` includes the original source without changing any
device function body. It uses the original Windows `rand()` control-point
sequence (`srand(1)`), original parent and child kernel launches, and device
allocation/free. A separate capture kernel copies device-heap vertices into a
host-readable allocation. CUDA runtime copies cannot read device-heap pointers
directly; the initial capture attempt returned `cudaErrorInvalidValue` there.

The completed capture contains 256 curves and 3,958 vertices. Every vertex was
checked against an independent double-precision quadratic Bernstein formula;
maximum component error is 1.36080852142e-7. Control points, per-curve counts,
and padded 32-vertex output records are saved in the corresponding `.bin` files.
Padding is zero and is not counted as output vertices.

Build from the repository root after loading the MSVC x64 environment:

```
nvcc -O3 -std=c++17 -arch=native -rdc=true -I.local/nvidia-audit/Common tests/bezier-cdp-native.cu -o .local/nvidia-checks/bezier-cdp.exe
.local\nvidia-checks\bezier-cdp.exe
node scripts/probe-bezier-cdp.mjs
```

## Compiler integration

The first full-source probe rejected the original free operators on `float2`.
The compiler now accepts free binary operators whose operands include CUDA
vector values and dispatches exact declared overloads. Tests retain NVIDIA's
original arithmetic helper bodies and use a deliberately nonstandard float4
overload to detect accidental substitution with WGSL's built-in addition.

The current full-source probe next rejects `float2 *vertexPos` inside
`BezierLine`. The remaining work includes persistent dynamically allocated
vector buffers, pointer fields in records, GPU child-launch scheduling, and
device free. WebGPU does not directly execute CUDA device launches; the
compiler/runtime must preserve this workload through an explicit equivalent
scheduling mechanism. Do not replace it with CPU curve generation or remove
the original launch/allocation operations merely to make it compile.

No browser result or showcase card is claimed yet. Add the sandbox card only
after all 256 original curves execute through the compiler and match the native
captures, with preview output drawn from the resulting GPU buffers.

NVIDIA source is BSD-3-Clause; the capture/probe code is MIT project code.
