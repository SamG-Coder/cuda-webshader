# CUDA path tracer: native baseline

Target: Roger Allen's CUDA translation of *Ray Tracing in One Weekend*, final
chapter branch `ch12_where_next_cuda`, revision
`ab140b12d4923b75270831baabab5e4d4209f305`:
https://github.com/rogerallen/raytracinginoneweekendincuda/tree/ab140b12d4923b75270831baabab5e4d4209f305

## Verified native result

The original program compiles and runs on the local NVIDIA RTX 5080 with
CUDA 13.3 and MSVC. No source or header was modified. NVCC build flags were
`-O3 -std=c++17 -arch=native`; this changes the old Makefile's GPU target to
the installed hardware without changing the program.

The full original defaults were retained: 1200 x 800 pixels, 10 samples per
pixel, 488 spheres, and a maximum of 50 path bounces. The original timer
reported **0.635 seconds**. That timer includes per-pixel RNG initialization,
rendering and synchronization, but excludes scene construction and PPM file
output. It is one observed run, not a stable benchmark or a WebGPU comparison.

`pathtracer-native.png` is a lossless conversion of all 960,000 pixels from
the native PPM. The low sample count produces visible Monte Carlo noise; the
image was not denoised, recoloured, or rendered by a substitute implementation.
The manifest records source and image hashes. This is a native baseline,
**not a working browser showcase**.

## Current compiler findings

`scripts/probe-pathtracer.mjs` records the first actual errors. Loading
`main.cu` directly stops at its external includes. The compiler now parses plain
public value classes, constructors, array fields and const methods, lowering
them to records and device helpers. The isolated `vec3.h` probe now reaches
writable reference-returning index operator before stopping; the scene header
still stops at its forward class declaration. Const unary signs and indexed
reads now compile. No original function bodies are rewritten by those probes.

The class-stage test compares 512 cases with **7,168 values matching native
CUDA exactly**, covering construction, copying, accessors, squared length,
unary signs, dynamic indexed reads and assignment copy isolation. Native
compilation uses the complete original `vec3.h`; the GPU fixture retains the
unchanged supported method bodies and explicitly omits unsupported methods.
This is a focused language test, not support for the complete header or path
tracer. Const writes and resolved recursive class calls are rejected.

On the real NVIDIA WebGPU adapter, adding dynamic array indexing exposed an
incorrect shared value after copying a class containing an array. The emitter
now constructs independent aggregate fields explicitly for initialization and
assignment. Both copy cases match native CUDA; the precise backend cause has
not been isolated. The full regression run passes 197 GPU checks with no
software adapter requested, alongside 610 unit tests.

Source inspection identifies the following connected work:

1. Class values, constructors, member methods, `this`, operator overloads,
   and the three-float `vec3` storage layout.
2. Scene records containing material pointers and pointer-to-pointer parameters.
3. Virtual `hit` and `scatter` calls for the finite sphere/material hierarchy.
4. Device `new`/`delete` for scene construction and cleanup.
5. cuRAND state, initialization and sampling with the original random sequence.
6. Full scene rendering and image comparison, followed by sandbox integration.

The native CUDA version already uses a bounded loop for path bounces, so
recursive ray traversal is not the first blocker here. Preserve the full
final-chapter scene; do not replace it with an earlier chapter and present that
as support for this target. A future browser renderer must execute generated
WGSL from the upstream source, not display the native reference image.

## Reproduction on this PC

Clone the branch into `.local/raytracing-cuda`, then check out the pinned
revision above. From a Visual Studio x64 developer command prompt in the
repository root:

```
nvcc -O3 -std=c++17 -arch=native .local/raytracing-cuda/main.cu -o .local/nvidia-checks/pathtracer.exe
.local\nvidia-checks\pathtracer.exe > .local/pathtracer-native.ppm 2> reports/pathtracer-native.txt
python scripts/capture-pathtracer.py
node scripts/probe-pathtracer.mjs
```

`capture-pathtracer.py` checks the pinned revision and unmodified tracked
source, validates dimensions and every output colour, and writes a PNG plus
the provenance manifest. Native PPM and executable remain under `.local`.
