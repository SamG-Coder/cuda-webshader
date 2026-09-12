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

## Typed pointer-field records

The original `BezierLine` declaration now parses. Scalar/float-vector pointer
fields retain their element type and lower to opaque 32-bit identities. Record
copies, null assignments, boolean conversion, and same-type equality are
supported. Cross-type assignment, integer fabrication, pointer arithmetic and
dereference remain rejected. This is a representation milestone, not device
allocation support.

The WebGPU Bezier record stride is explicitly 32 bytes (24 bytes of control
points, a 4-byte identity, and a 4-byte count). It is not the native 64-bit CUDA
pointer ABI and must not be populated by copying native struct bytes. A real
NVIDIA test copies 256 records, checks all 2,048 words, clears each original
pointer independently, and verifies null comparisons including high-bit opaque
identities. The next complete-source rejection is the `void **` output cast in
`cudaMalloc`; allocation and child launches still need implementation.

## Device heap and original child-kernel verification

An explicit `deviceHeap: {maxAllocations: 256, maxElements: 32}` option now
provides typed GPU allocations in a persistent object arena. The original
`cudaMalloc((void **)&field, byteCount)` form allocates a slot, writes its typed
identity, and returns a status. `cudaFree` releases the slot. The selected pool
capacity is a bounded implementation limit, not an unbounded CUDA heap.

Byte counts must be multiples of the pointed-to CUDA value size. Oversized,
non-integral-element, or exhausted requests return allocation failure (2);
zero-byte requests return success with a null identity. Out-of-range/null/freed
accesses are guarded (zero reads and ignored stores). This guard behavior is
an implementation choice for accesses that are invalid in the CUDA program.
Pointer arithmetic and general pointer escape remain unsupported. Buffers that
contain pointer identities, including nested records, retain arena ownership.
The CPU oracle does not implement persistent arenas.

A native harness and real NVIDIA WebGPU test verify 256 allocations and 4,224
live float2 elements. Across two allocate/write/read/free cycles all 32,768
float components, including zero padding, match exactly. Separate checks cover
exhaustion, zero requests, reuse, bounds isolation and cross-arena rejection.

The parser now retains device-launch syntax without executing it as a helper.
The unchanged complete module compiles `computeBezierLinePositions` and
`freeVertexMem`. `computeBezierLinesCDP` is rejected specifically for missing
GPU child-launch scheduling.

The original vertex and free kernels were executed on all 256 curves using
native control points and tessellation counts. All 3,958 vertices agree with
native CUDA within 1.1920928955078125e-7; the original cleanup frees every slot.
This is stage verification: the test harness supplies 256 child dispatches and
the captured counts. It is not a substitute for executing the parent's curvature
calculation, allocation, and dynamic launches. The full sandbox card remains
pending that scheduler. Results are in `bezier-cdp-stages.json`.
