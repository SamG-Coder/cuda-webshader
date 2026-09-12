# NVIDIA oceanFFT

Pinned upstream: `5443602d89ed99aede2e4b7bf329daddeadb320e`,
`cpp/4_CUDA_Libraries/oceanFFT`.

The complete 256 × 256 compute chain runs on NVIDIA Blackwell:
original `generateSpectrumKernel`, an MIT runtime inverse FFT replacing the
cuFFT library operation, original real-component `updateHeightmapKernel`, and
original `calculateSlopeKernel`. Original CUDA device-function bodies are
unchanged. Native references use NVCC `-O3 --fmad=false -std=c++17 -arch=native`
and cuFFT. Native host spectrum generation retains the original Phillips,
Gaussian and initialization functions, with `srand(1)` and original parameters.
The captured initial spectrum is shared between native and browser tests.

At times 0, 1.25 and 7.5, every spectrum, complex spatial, height and slope
component is compared. Maximum absolute error is 3.166497e-7 across all stages;
the enforced absolute tolerance is 1e-6. These are floating-point comparisons,
not bit-exact claims. See `ocean-check.json` and the native captures.

`GpuRuntime.inverseFFT2D` implements an unnormalized inverse complex transform
for power-of-two axes 1..1024, subject to device workgroup limits. CUDA row/column
passes share their scratch results on the GPU; input and output may alias.
Independent direct-DFT checks cover 4 × 8 complex input, identity covers 1 × 1,
and a 1024-point impulse checks the largest supported axis and trailing guards.
The implementation prioritizes correctness; no performance parity with cuFFT
has been established.

The sandbox preset `example=ocean` animates the full chain and a labelled MIT
CUDA mesh adapter producing 130,050 triangles. The original kernels remain in
the editable source. The runtime FFT shader is labelled separately in the WGSL
selector. Initial spectrum loading is a host operation; subsequent simulation
frames, heights, slopes, positions and normals remain on the GPU. A one-time
position readback fits the camera. Animation has no compute readbacks.

The Three.js material and camera are project preview choices, not a reproduction
of NVIDIA's OpenGL ocean shader. The mesh adapter is project CUDA code, not an
upstream NVIDIA function. `ocean-sandbox-check.json` records native height
comparison, every mesh height, normal lengths, time evolution, automatic frames,
pause/resume, source preservation and mobile layout checks.
