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

The test input is a deterministic byte-range plane centered around zero. These
results do not yet establish the original full-image showcase or other DCT paths.

## Remaining candidate work

1. Run the original teapot image through the complete first floating-point path,
   including original grayscale/centering and display conversion conventions.
2. Add the GPU pipeline to the sandbox and validate its rendered output against
   native before adding a showcase card.
3. Test the optimized floating-point `CUDAkernel2DCT` / `CUDAkernel2IDCT` path.
4. Implement and test the short-integer transform and quantization path. It uses
   packed short storage and pointer reinterpretations between shorts and words;
   that ABI remains unsupported and the path is not claimed to run.

The compiler/test infrastructure is MIT. NVIDIA kernel functions and matrices
retain their original BSD-3-Clause notices. The native Windows harness uses
`nvcc -O3 --fmad=false -std=c++17 -arch=native -Xcompiler /Zc:preprocessor`.
All validation uses real NVIDIA hardware, with no software adapter requested.
