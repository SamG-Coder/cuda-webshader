# Mandelbrot float renderer validation

Upstream: NVIDIA CUDA Samples `cpp/5_Domain_Specific/Mandelbrot`, revision `5443602d89ed99aede2e4b7bf329daddeadb320e`.

The original `Mandelbrot0<float>` renderer now compiles and runs in the sandbox, with individual showcase entry and Mandelbrot, Julia and accumulated-frame presets. Original CUDA function bodies remain unchanged.

Generic compiler work covers conditional declaration extraction, packed uchar4 and Boolean launch parameters, reference arithmetic, template scalar casts, expression updates, infinite do loops with internal returns, and packed storage component writes. Packed writes use an atomic compare/exchange on the containing word so independent channel writes preserve neighbouring bytes. Destination addresses are evaluated once, after assignment RHS evaluation under the supported CUDA C++17 profile. Shared-memory byte writes and const references to storage records remain unsupported.

423 unit tests pass. Native probes verify 257 packed records, guards, helper/component stores, wraparound and assignment evaluation order. All 124 NVIDIA hardware checks pass. The suite verifies the same storage behaviour and six original fractal frames. Software WebGPU tests are disabled.

All six frames match native CUDA built with `-O3 --fmad=false -std=c++17 -arch=native` and an independent float32 reference exactly. Native default-fusion builds are retained separately and differ in four larger frames, as recorded in `mandelbrot-floating-point.json`. No default-fusion bitwise equivalence or performance claim is made.

The sandbox has editable launch settings, real compiled GPU output and CUDA/WGSL comparison. The browser check validates initial Mandelbrot and Julia, two accumulated frames, edited RGB multipliers/viewport and mobile layout.

Scope is the primary float renderer. Secondary anti-aliasing, double-single arithmetic and fp64 paths are not validated or exposed. The entire desktop application is not translated.
