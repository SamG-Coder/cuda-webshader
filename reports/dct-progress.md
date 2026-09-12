# NVIDIA dct8x8 progress

The next candidate is CUDA Samples `cpp/2_Concepts_and_Techniques/dct8x8`
at commit `5443602d89ed99aede2e4b7bf329daddeadb320e`.

## Verified compiler foundation

- Inferred-size `__constant__` scalar arrays retain the original initializer.
- Short and unsigned-short constants occupy validated integer uniform fields.
- CUDA `roundf` ties round away from zero, including signed zero and large finite
  integers. Native/GPU output bits match for the rounding and short-boundary probe.
- Parenthesized direct-call macros preserve the original `FMUL` → `__mul24` path.
- Original `CUDAkernel1DCT`, `CUDAkernelQuantizationFloat` and `CUDAkernel1IDCT`
  functions and their constant matrices remain unchanged in the test fixture.
- All three stages match native CUDA exactly for the full 32 × 24 test plane,
  including trailing guards. The quantized coefficient transfer uses a GPU
  buffer-to-texture copy. Stage readbacks verify results; none supplies input to
  a later stage. See `dct-candidate.json` and native captures.

## Verified full-image showcase

The sandbox's `example=dct` runs NVIDIA's original 512 × 512 `teapot512.ppm`
through grayscale preparation, DCT, quantization, IDCT and display conversion.
Every intermediate float and final display pixel matches the native capture
exactly on NVIDIA Blackwell. The final canvas matches the GPU output. GPU
buffer-to-texture copies supply subsequent stages; control readback is zero.
See `dct-teapot-stages.json` and `dct-sandbox-check.json`.

The original three NVIDIA kernel functions are unchanged. Two labelled MIT CUDA
adapter kernels implement the sample host's integer grayscale formula, centering,
and rounded/clamped display conversion. All five passes are visible in the
sandbox's CUDA/WGSL comparison. This validates the kernels and adapted pipeline,
not the complete upstream desktop executable.

The accompanying PPM is used deliberately: the bundled BMP has a 138-byte pixel
offset, while the upstream BMP loader begins reading after 54 bytes. The PPM
avoids that loader/header mismatch without changing transform kernels.

Three quantized coefficients initially differed because approximate WGSL division
crossed a rounding boundary. Scalar float `/` and `/=` now correct the quotient
with a fused residual. A native bit comparison covers the three observed cases,
signed zero, signs, and small/large normal values. This is a finite precision
improvement, not a complete software IEEE divider; vector and `__fdividef`
operations retain their existing behavior.

## Remaining candidate work

3. Test the optimized floating-point `CUDAkernel2DCT` / `CUDAkernel2IDCT` path.
4. Implement and test the short-integer transform and quantization path. It uses
   packed short storage and pointer reinterpretations between shorts and words;
   that ABI remains unsupported and the path is not claimed to run.

The compiler/test infrastructure is MIT. NVIDIA kernel functions and matrices
retain their original BSD-3-Clause notices. The native Windows harness uses
`nvcc -O3 --fmad=false -std=c++17 -arch=native -Xcompiler /Zc:preprocessor`.
All validation uses real NVIDIA hardware, with no software adapter requested.
