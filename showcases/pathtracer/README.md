# CUDA path tracer

[Run in the sandbox](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=pathtracer).

Roger Allen's CUDA translation of Peter Shirley's Ray Tracing in One Weekend,
final chapter (`ch12_where_next_cuda`), revision
`ab140b12d4923b75270831baabab5e4d4209f305`:
https://github.com/rogerallen/raytracinginoneweekendincuda

Upstream identifies the code as public domain. The compiler, JSON host pipeline,
and verification harness are MIT project code.

`kernel.cu` assembles the original device declarations and function bodies from
the headers and main.cu. It removes includes, header guards, redundant forward
declarations and host-only stream output. No device function bodies are changed.
`scripts/probe-pathtracer-integration.mjs` reproduces that extraction from a
checkout at `.local/raytracing-cuda`.

The default scene contains 488 spheres at 1200 × 800 and 10 samples per pixel.
Five original kernels initialize RNG state, create the scene, initialize pixel
RNGs, trace rays, and free the scene. The JSON pipeline replaces host allocation
and launches. Two GPU handle copies connect the original differing parameter
names. Object pools persist across submissions; record layouts are explicit.
The RGB float preview displays the original gamma-corrected framebuffer.

Compiler integration required C++17 indexed assignment sequencing, storage-backed
member references, and local RNG pointer forwarding through virtual calls. This
builds on tagged object dispatch, value classes, and the bounded XORWOW library.
It is support for this CUDA subset, not arbitrary C++ or the complete CUDA API.

Validation: `node scripts/test-pathtracer-sandbox.mjs` uses real NVIDIA WebGPU,
checks all 960,000 displayed pixels against the output buffer, retains source
verbatim, and checks all five generated shader panes. The native harness is
`tests/pathtracer-full-native.cu`. Across 2,880,000 float components the recorded
mean absolute difference is 0.000614175; 97.593% differ by less than 0.0001.
Stochastic path divergence makes some individual differences larger. These are
not bit-exact images. See `reports/pathtracer-sandbox-check.json`.

No software GPU results or unmeasured speedup claims are used.
