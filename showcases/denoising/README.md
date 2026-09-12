# NVIDIA image denoising

Run [KNN](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=denoise-knn),
[non-local means](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=denoise-nlm),
or [shared non-local means](https://samg-coder.github.io/cuda-webshader/sandbox.html?example=denoise-nlm2).
All links open the sandbox. Its example selector also offers Copy, KNNdiag,
NLMdiag and NLM2diag, exposing all seven original image kernels.

## Source and launch settings

The kernel functions, device helpers, configuration expressions and noisy portrait
come from CUDA Samples commit `5443602d89ed99aede2e4b7bf329daddeadb320e`,
`cpp/2_Concepts_and_Techniques/imageDenoising`. The seven kernel function bodies
are unchanged. Device code from the original files is assembled into `kernel.cu`;
host wrappers, include guards and unused host macros are omitted. NVIDIA's
BSD-3-Clause notices are retained. The compiler and sandbox infrastructure are MIT.

The portrait is the unchanged 320 × 408 BMP. Its decoding keeps original row order,
as does NVIDIA's native BMP loader. The final canvas flips Y and displays opaque
alpha; the kernel's actual output alpha remains zero. There are no intermediate
image readbacks. Image processing comes from the compiled CUDA functions.

The presets use NVIDIA's default host settings: KNN noise level 0.32, NLM noise
level 1.45, and blend 0.2. The kernel argument `Noise` is the reciprocal square
of the noise level, computed with float precision. Edit that coefficient and
`lerpC` (capitalized `LerpC` in NLM2diag) in the pipeline launch settings. These
are kernel parameters, so increasing `Noise` is not the same as increasing the
host noise level. The configuration macros retain the original radius 3.

All presets use 8 × 8 workgroups and 40 × 51 blocks. NLM2 and NLM2diag require
`imageW` and `imageH` to be multiples of 8 because the original shared barrier
sits inside the image bounds check. Their explicit `fullWorkgroups` contract
validates those dimensions before dispatch and allows the compiler to lower the
per-pixel guard to an equivalent uniform per-block guard. The original source is
unchanged. Arbitrary partial blocks remain unsupported for those shared kernels.

Texture reads use the original unnormalized linear/clamp behavior at texel centers,
with normalized colour values uploaded as float4 texels. CUDA wrap addressing on
unnormalized coordinates is represented by clamp addressing. The browser does
not execute the original OpenGL desktop host program.

## Verification

All seven entries run natively and through hardware WebGPU on NVIDIA Blackwell.
Each has two complete native captures: a 32 × 24 synthetic image at Noise 32,
and the original portrait at NVIDIA's default settings. Every active output pixel
and trailing guard word is compared. Synthetic results match exactly. Portrait
results match exactly for Copy, NLM, NLM2 and all diagnostic entries; KNN differs
by at most one 8-bit channel level. The native build disables multiply-add fusion;
CUDA's `__expf` and WGSL's `exp` do not promise identical floating-point results.
These results measure correctness, not performance.

The static sandbox test visits all seven presets, compares every output against
native, checks every displayed RGB pixel against GPU output, preserves the source,
rejects a partial shared block and checks mobile overflow. The full regression
suite passes 501 unit tests and 149 real GPU checks. No software GPU tests run.

Evidence: `reports/denoising-candidate.json`, `reports/denoising-portrait-check.json`,
`reports/denoising-sandbox-check.json` and the native text/binary captures.
Run `node scripts/test-denoising-sandbox.mjs` after `npm run build`.
The native harness is `tests/denoising-native.cu`; link the original
`imageDenoising/bmploader.cpp` as a separate translation unit and pass
`-Xcompiler /Zc:preprocessor` with this Windows CUDA toolkit.
